"""
5단계: 합성 지도학습 사전학습 가중치로 초기화 후 SBSDI real N2N 쌍으로 fine-tuning.

사전학습: results/03_supervised/checkpoints/best.pth (합성 6,136쌍, loss 0.021621)
Fine-tune: SBSDI real 39세트 x 12 = 468쌍 (N2N 방식)
Loss: L1 + lambda * (1 - SSIM)

결과:
  results/05_finetune/checkpoints/best.pth
  results/05_finetune/metrics/per_image.csv
  results/05_finetune/metrics/summary.csv
  results/05_finetune/metrics/training_log.csv
  results/05_finetune/images/sample_*.png
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "03_sub2full"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Sub2FullDataset
from model import UNet
from utils import load_all_pairs, compute_metrics

PRETRAIN_CKPT = ROOT / "results" / "03_supervised" / "checkpoints" / "best.pth"
RESULTS_DIR   = ROOT / "results" / "05_finetune"
CKPT_DIR      = RESULTS_DIR / "checkpoints"
METRICS_DIR   = RESULTS_DIR / "metrics"
IMAGES_DIR    = RESULTS_DIR / "images"

BASELINE = {"psnr": 27.50, "ssim": 0.652, "cnr": 1.220}   # SRAD
N2N_REF  = {"psnr": 26.84, "ssim": 0.681, "cnr": 1.148}   # N2N from scratch


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.pad = window_size // 2
        kernel = _gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.kernel
        mu1 = F.conv2d(pred,   k, padding=self.pad)
        mu2 = F.conv2d(target, k, padding=self.pad)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = F.conv2d(pred   * pred,   k, padding=self.pad) - mu1_sq
        s2 = F.conv2d(target * target, k, padding=self.pad) - mu2_sq
        s12 = F.conv2d(pred  * target, k, padding=self.pad) - mu12
        num = (2 * mu12 + self.C1) * (2 * s12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        return 1.0 - (num / den).mean()


# ---------------------------------------------------------------------------
# 추론 유틸리티
# ---------------------------------------------------------------------------

def pad_to_multiple(img: np.ndarray, multiple: int = 8):
    H, W = img.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    return np.pad(img, ((0, pH), (0, pW)), mode="reflect"), (H, W)


def infer(model: nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    padded, (H, W) = pad_to_multiple(img, 8)
    x = torch.from_numpy(padded[None, None]).to(device)
    with torch.no_grad():
        out = model(x)
    return out[0, 0].cpu().numpy()[:H, :W]


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, device: torch.device) -> nn.Module:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Sub2FullDataset(
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(base_ch=32).to(device)

    if not PRETRAIN_CKPT.exists():
        print(f"사전학습 가중치 없음: {PRETRAIN_CKPT}")
        print("scripts/03_supervised/run_supervised.py 를 먼저 실행하세요.")
        sys.exit(1)

    state = torch.load(PRETRAIN_CKPT, map_location=device)
    model.load_state_dict(state)
    print(f"사전학습 가중치 로드: {PRETRAIN_CKPT}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Steps/epoch: {len(loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    l1_criterion   = nn.L1Loss()
    ssim_criterion = SSIMLoss().to(device)

    log_records = []
    best_loss = float("inf")
    start_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = l1_criterion(pred, target) + args.ssim_lambda * ssim_criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.perf_counter() - t0
        log_records.append({"epoch": epoch, "loss": round(avg_loss, 6),
                             "time_sec": round(elapsed, 2)})

        if epoch % 20 == 0 or epoch == 1:
            total_min = (time.perf_counter() - start_total) / 60
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={avg_loss:.6f} | {elapsed:.1f}s | total={total_min:.1f}min",
                  flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), CKPT_DIR / f"epoch_{epoch:04d}.pth")

    pd.DataFrame(log_records).to_csv(METRICS_DIR / "training_log.csv", index=False)
    print(f"\n학습 완료 | best loss: {best_loss:.6f}")
    return model


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, device: torch.device) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()

    print("\nSBSDI D1 평가 중 (18쌍)...")
    pairs = load_all_pairs()
    records = []
    SAVE_INDICES = {1, 5, 10, 15}

    for i, (noisy, clean) in enumerate(tqdm(pairs, desc="evaluate"), start=1):
        t0 = time.perf_counter()
        denoised = infer(model, noisy, device)
        elapsed = time.perf_counter() - t0
        m = compute_metrics(clean, denoised)
        records.append({
            "image_idx": i,
            "psnr": round(m["psnr"], 4),
            "ssim": round(m["ssim"], 4),
            "cnr":  round(m["cnr"],  4),
            "time_sec": round(elapsed, 3),
        })
        if i in SAVE_INDICES:
            _save_sample(i, noisy, clean, denoised, m)

    df = pd.DataFrame(records)
    df.to_csv(METRICS_DIR / "per_image.csv", index=False)

    summary = {k: round(df[k].mean(), 4) for k in ["psnr", "ssim", "cnr"]}
    summary.update({f"{k}_std": round(df[k].std(), 4) for k in ["psnr", "ssim", "cnr"]})
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "summary.csv", index=False)

    print("\n========== Fine-tune 성능 ==========")
    for key, label in [("psnr", "PSNR"), ("ssim", "SSIM"), ("cnr", "CNR")]:
        val, std = summary[key], summary[f"{key}_std"]
        b, n = BASELINE[key], N2N_REF[key]
        status = "초과" if val > b else "미달"
        print(f"{label:4s}: {val:.4f} +- {std:.4f}  "
              f"(SRAD {b:.3f} {status} / N2N {n:.3f})")
    print("=====================================")


def _save_sample(idx, noisy, clean, denoised, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [noisy, clean, denoised],
        ["Noisy (Input)", "Clean (Reference)",
         f"Fine-tune\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / f"sample_{idx:02d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-train -> Fine-tune OCT 디노이징")
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--batch-size",        type=int,   default=16)
    p.add_argument("--lr",                type=float, default=1e-5)
    p.add_argument("--patch-size",        type=int,   default=128)
    p.add_argument("--patches-per-image", type=int,   default=8)
    p.add_argument("--ssim-lambda",       type=float, default=0.1,
                   help="SSIM loss 가중치 (loss = L1 + lambda * (1 - SSIM))")
    p.add_argument("--cpu-threads",       type=int,   default=4)
    p.add_argument("--device",            type=str,   default="auto")
    p.add_argument("--eval-only",         action="store_true",
                   help="학습 없이 저장된 best.pth로 평가만 실행")
    return p.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(args.cpu_threads)
    print(f"CPU 스레드 제한: {args.cpu_threads}")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = UNet(base_ch=32).to(device)

    if args.eval_only:
        ckpt = CKPT_DIR / "best.pth"
        if not ckpt.exists():
            print(f"체크포인트 없음: {ckpt}")
            sys.exit(1)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"체크포인트 로드: {ckpt}")
    else:
        model = train(args, device)
        model.load_state_dict(torch.load(CKPT_DIR / "best.pth", map_location=device))

    evaluate(model, device)


if __name__ == "__main__":
    main()
