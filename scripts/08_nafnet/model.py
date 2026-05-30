"""
NAFNet (Nonlinear Activation Free Network, Chen et al. 2022).

구조: U-Net 형태 인코더-디코더 + skip connection
핵심:
  - SimpleGate: 채널을 절반으로 나눠 곱셈 (ReLU/GELU 대체)
  - Simple Channel Attention (SCA): global avg pool + 1x1 conv
  - LayerNorm2d (채널별 정규화)
  - 활성화 함수 없음

설정:
  width=16  -> ~4M params  (소형)
  width=32  -> ~17M params (표준)
  width=64  -> ~67M params (대형)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> LayerNorm on C dim
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """채널을 절반으로 분리한 뒤 element-wise 곱."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = c * dw_expand
        ffn_ch = c * ffn_expand

        # -- Attention branch --
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)  # depthwise
        self.gate  = SimpleGate()                                       # dw_ch -> dw_ch//2
        # SCA: global avg pool on (dw_ch//2) channels
        self.sca   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        # -- FFN branch --
        self.norm2  = LayerNorm2d(c)
        self.conv4  = nn.Conv2d(c, ffn_ch, 1)
        self.gate2  = SimpleGate()                                      # ffn_ch -> ffn_ch//2
        self.conv5  = nn.Conv2d(ffn_ch // 2, c, 1)

        # learnable residual scaling
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp

        # Attention
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)        # (B, dw_ch//2, H, W)
        x = x * self.sca(x)    # channel attention
        x = self.conv3(x)
        inp = inp + x * self.beta

        # FFN
        x = self.norm2(inp)
        x = self.conv4(x)
        x = self.gate2(x)       # (B, ffn_ch//2, H, W)
        x = self.conv5(x)
        return inp + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet — U-Net 기반 인코더-디코더.

    Args:
        in_ch:    입력 채널 (OCT 그레이스케일 = 1)
        width:    기본 채널 수 (16/32/64)
        enc_blks: 인코더 각 스테이지의 NAFBlock 수 리스트
        mid_blks: 병목 NAFBlock 수
        dec_blks: 디코더 각 스테이지의 NAFBlock 수 리스트
    """

    def __init__(
        self,
        in_ch:    int       = 1,
        width:    int       = 32,
        enc_blks: list[int] = None,
        mid_blks: int       = 1,
        dec_blks: list[int] = None,
    ):
        super().__init__()
        if enc_blks is None:
            enc_blks = [1, 1, 1, 28]
        if dec_blks is None:
            dec_blks = [1, 1, 1, 1]

        assert len(enc_blks) == len(dec_blks), "enc/dec 스테이지 수 불일치"

        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1)
        self.outro = nn.Conv2d(width, in_ch, 3, 1, 1)

        # --- 인코더 ---
        self.encoders   = nn.ModuleList()
        self.downs      = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        # --- 병목 ---
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(mid_blks)])

        # --- 디코더 ---
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        for n in dec_blks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 1),          # 채널 확장
                    nn.PixelShuffle(2),                 # ch*2 -> ch//2, H*2, W*2
                )
            )
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)

        # 인코더 — skip 저장
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)

        x = self.middle(x)

        # 디코더 — skip 더하기
        for dec, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = dec(x)

        return self.outro(x) + inp   # global residual: 출력 = 잔차 + 입력
