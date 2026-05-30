"""
DnCNN-B 모델 (Beyond a Gaussian Denoiser, Zhang et al. 2017).

구조: Conv+ReLU → (Conv+BN+ReLU) × (depth-2) → Conv
출력: 잔차(노이즈 추정값). 복원 이미지 = clamp(noisy - output, 0, 1)
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth: int = 20, channels: int = 64):
        super().__init__()
        layers: list[nn.Module] = []

        # 첫 번째 레이어: BN 없음
        layers += [nn.Conv2d(1, channels, 3, padding=1, bias=True), nn.ReLU(inplace=True)]

        # 중간 레이어: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]

        # 마지막 레이어: 잔차 출력, 활성화 없음
        layers += [nn.Conv2d(channels, 1, 3, padding=1, bias=True)]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """잔차(노이즈)를 반환. 복원: (x - forward(x)).clamp(0,1)"""
        return self.net(x)
