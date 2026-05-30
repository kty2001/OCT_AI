"""
3단계 지도학습: 합성 noisy-clean 6,136쌍으로 U-Net 훈련 후 SBSDI D1 평가.

결과:
  results/03_supervised/checkpoints/best.pth
  results/03_supervised/metrics/per_image.csv
  results/03_supervised/metrics/summary.csv
  results/03_supervised/metrics/training_log.csv
  results/03_supervised/images/sample_*.png
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "03_sub2full"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SyntheticPairDataset
from model import UNet
from utils import load_all_pairs, compute_metrics

RESULTS_DIR = ROOT / "results" / "03_supervised"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
IMAGES_DIR  = RESULTS_DIR / "images"

BASELINE = {"psnr": 27.50, "ssim": 0.652, "cnr": 1.220}   # SRAD


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

    dataset = SyntheticPairDataset(
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        cache=args.cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = UNet(base_ch=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Steps/epoch: {len(loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.L1Loss()

    log_records = []
    best_loss = float("inf")
    start_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.perf_counter() - t0
        log_records.append({"epoch": epoch, "loss": round(avg_loss, 6),
                             "time_sec": round(elapsed, 2)})

        if epoch % 10 == 0 or epoch == 1:
            total_min = (time.perf_counter() - start_total) / 60
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={avg_loss:.6f} | {elapsed:.1f}s | total={total_min:.1f}min",
                  flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")

        if epoch % 20 == 0:
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

    print("\n========== 지도학습 성능 ==========")
    print(f"PSNR : {summary['psnr']:.4f} +- {summary['psnr_std']:.4f}  "
          f"(SRAD {BASELINE['psnr']:.2f}, {'초과' if summary['psnr'] > BASELINE['psnr'] else '미달'})")
    print(f"SSIM : {summary['ssim']:.4f} +- {summary['ssim_std']:.4f}  "
          f"(SRAD {BASELINE['ssim']:.3f}, {'초과' if summary['ssim'] > BASELINE['ssim'] else '미달'})")
    print(f"CNR  : {summary['cnr']:.4f} +- {summary['cnr_std']:.4f}  "
          f"(SRAD {BASELINE['cnr']:.3f}, {'초과' if summary['cnr'] > BASELINE['cnr'] else '미달'})")
    print("====================================")


def _save_sample(idx, noisy, clean, denoised, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [noisy, clean, denoised],
        ["Noisy (Input)", "Clean (Reference)",
         f"Supervised\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"],
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
    p = argparse.ArgumentParser(description="지도학습 OCT 디노이징")
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=2e-4)
    p.add_argument("--patch-size",        type=int,   default=128)
    p.add_argument("--patches-per-image", type=int,   default=4)
    p.add_argument("--num-workers",       type=int,   default=2,
                   help="DataLoader 워커 수 (0이면 메인 프로세스에서 로드)")
    p.add_argument("--cpu-threads",       type=int,   default=4,
                   help="PyTorch CPU 스레드 수 제한")
    p.add_argument("--no-cache",          action="store_true",
                   help="이미지 메모리 캐싱 비활성화 (RAM 부족 시 사용)")
    p.add_argument("--device",            type=str,   default="auto")
    p.add_argument("--eval-only",         action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.cache = not args.no_cache

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
        model.load_state_dict(
            torch.load(CKPT_DIR / "best.pth", map_location=device)
        )

    evaluate(model, device)


if __name__ == "__main__":
    main()
