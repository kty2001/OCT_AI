"""
3단계-B Sub2Full 자가지도 학습 및 SBSDI D1 평가.

학습: SBSDI real experiments (39세트, 4프레임/세트) FFT 스펙트럼 분리 -> N2N
평가: SBSDI D1 18쌍 (전통 방법 베이스라인과 동일 데이터)

결과:
  results/03_sub2full/checkpoints/best.pth
  results/03_sub2full/metrics/per_image.csv
  results/03_sub2full/metrics/summary.csv
  results/03_sub2full/metrics/training_log.csv
  results/03_sub2full/images/sample_*.png
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Sub2FullDataset
from model import UNet
from utils import load_all_pairs, compute_metrics

RESULTS_DIR = ROOT / "results" / "03_sub2full"
CKPT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
IMAGES_DIR = RESULTS_DIR / "images"


# ---------------------------------------------------------------------------
# 추론 유틸리티
# ---------------------------------------------------------------------------

def pad_to_multiple(img: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple]:
    """H, W를 multiple의 배수로 패딩. 원본 복원용 슬라이스 반환."""
    H, W = img.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    padded = np.pad(img, ((0, pH), (0, pW)), mode="reflect")
    return padded, (H, W)


def infer(model: nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    """단일 이미지 전체를 모델에 통과시켜 복원 이미지 반환."""
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

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

        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inp), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.perf_counter() - t0
        log_records.append({"epoch": epoch, "loss": round(avg_loss, 6), "time_sec": round(elapsed, 2)})

        if epoch % 50 == 0 or epoch == 1:
            elapsed_total = time.perf_counter() - start_total
            print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.6f} | {elapsed:.1f}s/epoch | total={elapsed_total/60:.1f}min")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), CKPT_DIR / f"epoch_{epoch:04d}.pth")

    pd.DataFrame(log_records).to_csv(METRICS_DIR / "training_log.csv", index=False)
    print(f"\n학습 완료 | best loss: {best_loss:.6f}")
    print(f"모델 저장: {CKPT_DIR / 'best.pth'}")
    return model


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, device: torch.device) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

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
            "cnr": round(m["cnr"], 4),
            "time_sec": round(elapsed, 3),
        })

        if i in SAVE_INDICES:
            _save_sample(i, noisy, clean, denoised, m)

    df = pd.DataFrame(records)
    df.to_csv(METRICS_DIR / "per_image.csv", index=False)

    summary = {
        "psnr_mean": round(df["psnr"].mean(), 4),
        "psnr_std": round(df["psnr"].std(), 4),
        "ssim_mean": round(df["ssim"].mean(), 4),
        "ssim_std": round(df["ssim"].std(), 4),
        "cnr_mean": round(df["cnr"].mean(), 4),
        "cnr_std": round(df["cnr"].std(), 4),
    }
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "summary.csv", index=False)

    print("\n========== Sub2Full 성능 ==========")
    print(f"PSNR : {summary['psnr_mean']:.4f} +- {summary['psnr_std']:.4f}")
    print(f"SSIM : {summary['ssim_mean']:.4f} +- {summary['ssim_std']:.4f}")
    print(f"CNR  : {summary['cnr_mean']:.4f} +- {summary['cnr_std']:.4f}")
    print("------------------------------------")
    print("베이스라인 (SRAD): PSNR 27.50 / SSIM 0.652 / CNR 1.220")
    print("====================================")


def _save_sample(idx: int, noisy: np.ndarray, clean: np.ndarray,
                 denoised: np.ndarray, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Noisy (Input)")
    axes[0].axis("off")
    axes[1].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Clean (Reference)")
    axes[1].axis("off")
    axes[2].imshow(denoised, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Sub2Full\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"
    )
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / f"sample_{idx:02d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sub2Full OCT 자가지도 학습")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--patches-per-image", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=4,
                   help="PyTorch CPU 스레드 수 (다른 작업과 병행 시 낮게 설정)")
    p.add_argument("--device", type=str, default="auto",
                   help="cuda / cpu / auto")
    p.add_argument("--eval-only", action="store_true",
                   help="학습 없이 저장된 best.pth로 평가만 실행")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(args.cpu_threads)
    print(f"CPU 스레드 제한: {args.cpu_threads}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
