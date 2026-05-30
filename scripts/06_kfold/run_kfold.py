"""
6단계: SBSDI D1 18쌍 6-fold cross-validation 지도학습.

기존 U-Net + real clean GT 로 clean GT 활용 효과 단독 측정.
k=6, 각 fold: 15쌍 학습 / 3쌍 평가
사전학습 초기화: results/03_supervised/checkpoints/best.pth
Loss: L1 + 0.1 * (1 - SSIM)

결과:
  results/06_kfold/checkpoints/fold_{k}/best.pth
  results/06_kfold/metrics/fold_{k}/per_image.csv
  results/06_kfold/metrics/summary.csv
  results/06_kfold/images/fold_{k}/sample_*.png
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import UNet
from utils import load_all_pairs, compute_metrics

PRETRAIN_CKPT = ROOT / "results" / "03_supervised" / "checkpoints" / "best.pth"
RESULTS_DIR   = ROOT / "results" / "06_kfold"
CKPT_DIR      = RESULTS_DIR / "checkpoints"
METRICS_DIR   = RESULTS_DIR / "metrics"
IMAGES_DIR    = RESULTS_DIR / "images"

BASELINE = {"psnr": 27.50, "ssim": 0.652, "cnr": 1.220}   # SRAD
N2N_REF  = {"psnr": 26.84, "ssim": 0.681, "cnr": 1.148}   # N2N from scratch
FT_REF   = {"psnr": 27.46, "ssim": 0.6795, "cnr": 1.169}  # Pre-train + Fine-tune

K_FOLDS  = 6
N_PAIRS  = 18


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
        self.register_buffer("kernel", _gaussian_kernel(window_size, sigma))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.kernel
        mu1 = F.conv2d(pred,   k, padding=self.pad)
        mu2 = F.conv2d(target, k, padding=self.pad)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1  = F.conv2d(pred   * pred,   k, padding=self.pad) - mu1_sq
        s2  = F.conv2d(target * target, k, padding=self.pad) - mu2_sq
        s12 = F.conv2d(pred   * target, k, padding=self.pad) - mu12
        num = (2 * mu12 + self.C1) * (2 * s12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        return 1.0 - (num / den).mean()


# ---------------------------------------------------------------------------
# 데이터셋 — 패치 전수 추출 (stride 기반)
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    지정된 noisy-clean 쌍에서 stride 간격으로 패치를 전수 추출.
    학습 시 random H/V flip 증강 적용.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]],
                 patch_size: int = 128, stride: int = 32, augment: bool = True):
        self.patch_size = patch_size
        self.augment = augment
        self.patches: list[tuple[np.ndarray, np.ndarray]] = []

        ps = patch_size
        for noisy, clean in pairs:
            H, W = noisy.shape
            for top in range(0, H - ps + 1, stride):
                for left in range(0, W - ps + 1, stride):
                    n_p = noisy[top:top + ps, left:left + ps].copy()
                    c_p = clean[top:top + ps, left:left + ps].copy()
                    self.patches.append((n_p, c_p))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_p, c_p = self.patches[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                n_p = np.fliplr(n_p).copy()
                c_p = np.fliplr(c_p).copy()
            if np.random.rand() > 0.5:
                n_p = np.flipud(n_p).copy()
                c_p = np.flipud(c_p).copy()
        return torch.from_numpy(n_p[None]), torch.from_numpy(c_p[None])


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
# fold 단위 학습
# ---------------------------------------------------------------------------

def val_metrics(model: nn.Module,
                eval_pairs: list[tuple[np.ndarray, np.ndarray]],
                device: torch.device) -> tuple[float, float]:
    """held-out 이미지 전체에 대해 평균 PSNR/SSIM 반환."""
    model.eval()
    psnrs, ssims = [], []
    for noisy, clean in eval_pairs:
        denoised = infer(model, noisy, device)
        m = compute_metrics(clean, denoised)
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_fold(fold: int, train_pairs: list,
               eval_pairs: list[tuple[np.ndarray, np.ndarray]],
               args: argparse.Namespace, device: torch.device) -> nn.Module:
    ckpt_dir = CKPT_DIR / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = METRICS_DIR / f"fold_{fold}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatchDataset(train_pairs, patch_size=args.patch_size,
                           stride=args.stride, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = UNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    l1_crit   = nn.L1Loss()
    ssim_crit = SSIMLoss().to(device)

    log_records = []
    best_score   = -float("inf")   # psnr + 10*ssim
    patience_cnt = 0
    stopped_at   = args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = l1_crit(pred, clean) + args.ssim_lambda * ssim_crit(pred, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        vp, vs = val_metrics(model, eval_pairs, device)
        score = vp + 10 * vs

        log_records.append({
            "epoch": epoch, "train_loss": round(avg_loss, 6),
            "val_psnr": round(vp, 4), "val_ssim": round(vs, 4),
        })

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Fold {fold} | Ep {epoch:4d}/{args.epochs} | "
                  f"loss={avg_loss:.5f} | val PSNR={vp:.3f} SSIM={vs:.4f} | "
                  f"patience={patience_cnt}/{args.patience}", flush=True)

        if score > best_score:
            best_score = score
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                stopped_at = epoch
                print(f"  Fold {fold} | Early stop @ epoch {epoch} | "
                      f"best score={best_score:.4f}", flush=True)
                break

    pd.DataFrame(log_records).to_csv(metrics_dir / "training_log.csv", index=False)
    best_vp = max(r["val_psnr"] for r in log_records)
    best_vs = max(r["val_ssim"] for r in log_records)
    print(f"  Fold {fold} 완료 | stopped={stopped_at} | "
          f"best val PSNR={best_vp:.4f} SSIM={best_vs:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    return model


# ---------------------------------------------------------------------------
# fold 단위 평가
# ---------------------------------------------------------------------------

def evaluate_fold(fold: int, model: nn.Module,
                  eval_pairs: list[tuple[np.ndarray, np.ndarray]],
                  eval_indices: list[int],
                  device: torch.device) -> list[dict]:
    images_dir = IMAGES_DIR / f"fold_{fold}"
    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = METRICS_DIR / f"fold_{fold}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records = []

    for (noisy, clean), img_idx in zip(eval_pairs, eval_indices):
        denoised = infer(model, noisy, device)
        m = compute_metrics(clean, denoised)
        records.append({
            "fold": fold,
            "image_idx": img_idx,
            "psnr": round(m["psnr"], 4),
            "ssim": round(m["ssim"], 4),
            "cnr":  round(m["cnr"],  4),
        })
        _save_sample(images_dir / f"sample_{img_idx:02d}.png",
                     noisy, clean, denoised, m, img_idx)

    pd.DataFrame(records).to_csv(
        METRICS_DIR / f"fold_{fold}" / "per_image.csv", index=False
    )
    return records


def _save_sample(path, noisy, clean, denoised, metrics, idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [noisy, clean, denoised],
        ["Noisy (Input)", "Clean (Reference)",
         f"K-fold (img {idx})\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not PRETRAIN_CKPT.exists():
        print(f"사전학습 가중치 없음: {PRETRAIN_CKPT}")
        sys.exit(1)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = load_all_pairs()   # 18쌍, 인덱스 0~17 (이미지 번호 1~18)

    # 6-fold 분할: 3쌍씩
    fold_indices = [list(range(i, N_PAIRS, K_FOLDS)) for i in range(K_FOLDS)]
    # fold_indices[k] = fold k+1에서 평가할 이미지 인덱스 (0-based)

    patch_counts = []
    all_records: list[dict] = []
    total_start = time.perf_counter()

    for fold in range(1, K_FOLDS + 1):
        eval_idx   = fold_indices[fold - 1]                          # 0-based
        train_idx  = [i for i in range(N_PAIRS) if i not in eval_idx]

        train_pairs = [all_pairs[i] for i in train_idx]
        eval_pairs  = [all_pairs[i] for i in eval_idx]
        img_numbers = [i + 1 for i in eval_idx]                     # 1-based (표시용)

        # 패치 수 계산 (첫 fold만 출력)
        if fold == 1:
            ps, st = args.patch_size, args.stride
            H, W = all_pairs[0][0].shape
            n_h = (H - ps) // st + 1
            n_w = (W - ps) // st + 1
            patches_per_img = n_h * n_w
            total_patches = len(train_pairs) * patches_per_img
            print(f"\n패치: {n_h}x{n_w}={patches_per_img}/이미지, "
                  f"학습쌍 {len(train_pairs)}장 x {patches_per_img} = {total_patches}패치/fold")
            patch_counts.append(total_patches)

        print(f"\n[Fold {fold}/{K_FOLDS}] "
              f"학습 {[i+1 for i in train_idx]} / 평가 {img_numbers}")

        model = train_fold(fold, train_pairs, eval_pairs, args, device)
        records = evaluate_fold(fold, model, eval_pairs, img_numbers, device)
        all_records.extend(records)

    # 전체 집계
    df = pd.DataFrame(all_records)
    df.to_csv(METRICS_DIR / "all_folds.csv", index=False)

    summary = {k: round(df[k].mean(), 4) for k in ["psnr", "ssim", "cnr"]}
    summary.update({f"{k}_std": round(df[k].std(), 4) for k in ["psnr", "ssim", "cnr"]})
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "summary.csv", index=False)

    elapsed = (time.perf_counter() - total_start) / 60
    print(f"\n총 학습 시간: {elapsed:.1f}분")
    print("\n========== 6-fold CV 최종 성능 ==========")
    for key, label in [("psnr", "PSNR"), ("ssim", "SSIM"), ("cnr", "CNR")]:
        val, std = summary[key], summary[f"{key}_std"]
        b  = BASELINE[key]
        ft = FT_REF[key]
        flag = "초과" if val > b else "미달"
        print(f"{label:4s}: {val:.4f} +- {std:.4f}  "
              f"(SRAD {b:.3f} {flag} / Fine-tune {ft:.4f})")
    print("==========================================")


def parse_args():
    p = argparse.ArgumentParser(description="6-fold CV 지도학습 (U-Net + real clean GT)")
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--patch-size",  type=int,   default=128)
    p.add_argument("--stride",      type=int,   default=32)
    p.add_argument("--ssim-lambda", type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=30,
                   help="val PSNR+10*SSIM 기준 early stopping patience")
    p.add_argument("--cpu-threads", type=int,   default=4)
    p.add_argument("--device",      type=str,   default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main()
