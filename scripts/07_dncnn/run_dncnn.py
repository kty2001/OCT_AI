"""
7단계: DnCNN-B 백본으로 6-fold CV 지도학습.

U-Net 대비 백본 교체 효과 측정.
사전학습 없이 from scratch, 잔차 학습(noisy - residual = clean).
Loss: L1(pred_clean, clean) + 0.1*(1-SSIM)

결과:
  results/07_dncnn/checkpoints/fold_{k}/best.pth
  results/07_dncnn/metrics/fold_{k}/per_image.csv
  results/07_dncnn/metrics/summary.csv
  results/07_dncnn/images/fold_{k}/sample_*.png
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import DnCNN
from utils import load_all_pairs, compute_metrics

RESULTS_DIR = ROOT / "results" / "07_dncnn"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
IMAGES_DIR  = RESULTS_DIR / "images"

BASELINE = {"psnr": 27.50, "ssim": 0.652,  "cnr": 1.220}
UNET_REF = {"psnr": 28.19, "ssim": 0.6814, "cnr": 1.169}

K_FOLDS = 6
N_PAIRS = 18


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
# 데이터셋
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]],
                 patch_size: int = 128, stride: int = 32, augment: bool = True):
        self.augment = augment
        self.patches: list[tuple[np.ndarray, np.ndarray]] = []
        ps = patch_size
        for noisy, clean in pairs:
            H, W = noisy.shape
            for top in range(0, H - ps + 1, stride):
                for left in range(0, W - ps + 1, stride):
                    self.patches.append((
                        noisy[top:top+ps, left:left+ps].copy(),
                        clean[top:top+ps, left:left+ps].copy(),
                    ))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_p, c_p = self.patches[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                n_p, c_p = np.fliplr(n_p).copy(), np.fliplr(c_p).copy()
            if np.random.rand() > 0.5:
                n_p, c_p = np.flipud(n_p).copy(), np.flipud(c_p).copy()
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
        residual = model(x)
        out = (x - residual).clamp(0.0, 1.0)
    return out[0, 0].cpu().numpy()[:H, :W]


# ---------------------------------------------------------------------------
# 검증 지표
# ---------------------------------------------------------------------------

def val_metrics(model: nn.Module,
                eval_pairs: list[tuple[np.ndarray, np.ndarray]],
                device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs, ssims = [], []
    for noisy, clean in eval_pairs:
        m = compute_metrics(clean, infer(model, noisy, device))
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ---------------------------------------------------------------------------
# fold 단위 학습
# ---------------------------------------------------------------------------

def train_fold(fold: int, train_pairs: list,
               eval_pairs: list[tuple[np.ndarray, np.ndarray]],
               args: argparse.Namespace, device: torch.device) -> nn.Module:
    ckpt_dir    = CKPT_DIR    / f"fold_{fold}"
    metrics_dir = METRICS_DIR / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatchDataset(train_pairs, patch_size=args.patch_size,
                           stride=args.stride, augment=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    model     = DnCNN(depth=args.depth, channels=args.channels).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    if fold == 1:
        print(f"DnCNN: depth={args.depth}, ch={args.channels}, {n_params:,} params")
        print(f"Steps/epoch: {len(loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    l1_crit   = nn.L1Loss()
    ssim_crit = SSIMLoss().to(device)

    log_records  = []
    best_score   = -float("inf")
    patience_cnt = 0
    stopped_at   = args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred_clean = (noisy - model(noisy)).clamp(0.0, 1.0)
            loss = l1_crit(pred_clean, clean) + args.ssim_lambda * ssim_crit(pred_clean, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        vp, vs   = val_metrics(model, eval_pairs, device)
        score    = vp + 10 * vs

        log_records.append({
            "epoch": epoch, "train_loss": round(avg_loss, 6),
            "val_psnr": round(vp, 4), "val_ssim": round(vs, 4),
        })

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Fold {fold} | Ep {epoch:4d}/{args.epochs} | "
                  f"loss={avg_loss:.5f} | val PSNR={vp:.3f} SSIM={vs:.4f} | "
                  f"patience={patience_cnt}/{args.patience}", flush=True)

        if score > best_score:
            best_score   = score
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
                  eval_pairs: list, eval_indices: list[int],
                  device: torch.device) -> list[dict]:
    images_dir  = IMAGES_DIR  / f"fold_{fold}"
    metrics_dir = METRICS_DIR / f"fold_{fold}"
    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records = []
    for (noisy, clean), img_idx in zip(eval_pairs, eval_indices):
        denoised = infer(model, noisy, device)
        m = compute_metrics(clean, denoised)
        records.append({
            "fold": fold, "image_idx": img_idx,
            "psnr": round(m["psnr"], 4),
            "ssim": round(m["ssim"], 4),
            "cnr":  round(m["cnr"],  4),
        })
        _save_sample(images_dir / f"sample_{img_idx:02d}.png",
                     noisy, clean, denoised, m, img_idx)

    pd.DataFrame(records).to_csv(metrics_dir / "per_image.csv", index=False)
    return records


def _save_sample(path, noisy, clean, denoised, metrics, idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [noisy, clean, denoised],
        ["Noisy (Input)", "Clean (Reference)",
         f"DnCNN (img {idx})\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"],
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

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs   = load_all_pairs()
    fold_indices = [list(range(i, N_PAIRS, K_FOLDS)) for i in range(K_FOLDS)]

    all_records: list[dict] = []
    total_start = time.perf_counter()

    for fold in range(1, K_FOLDS + 1):
        eval_idx    = fold_indices[fold - 1]
        train_idx   = [i for i in range(N_PAIRS) if i not in eval_idx]
        train_pairs = [all_pairs[i] for i in train_idx]
        eval_pairs  = [all_pairs[i] for i in eval_idx]
        img_numbers = [i + 1 for i in eval_idx]

        if fold == 1:
            ps, st = args.patch_size, args.stride
            H, W   = all_pairs[0][0].shape
            n_h    = (H - ps) // st + 1
            n_w    = (W - ps) // st + 1
            print(f"\n패치: {n_h}x{n_w}={n_h*n_w}/이미지, "
                  f"{len(train_pairs)}장 x {n_h*n_w} = {len(train_pairs)*n_h*n_w}패치/fold")

        print(f"\n[Fold {fold}/{K_FOLDS}] "
              f"학습 {[i+1 for i in train_idx]} / 평가 {img_numbers}")

        model   = train_fold(fold, train_pairs, eval_pairs, args, device)
        records = evaluate_fold(fold, model, eval_pairs, img_numbers, device)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(METRICS_DIR / "all_folds.csv", index=False)

    summary = {k: round(df[k].mean(), 4) for k in ["psnr", "ssim", "cnr"]}
    summary.update({f"{k}_std": round(df[k].std(), 4) for k in ["psnr", "ssim", "cnr"]})
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "summary.csv", index=False)

    elapsed = (time.perf_counter() - total_start) / 60
    print(f"\n총 학습 시간: {elapsed:.1f}분")
    print("\n========== DnCNN 6-fold CV 최종 성능 ==========")
    for key, label in [("psnr", "PSNR"), ("ssim", "SSIM"), ("cnr", "CNR")]:
        val, std = summary[key], summary[f"{key}_std"]
        b = BASELINE[key]
        u = UNET_REF[key]
        flag = "초과" if val > b else "미달"
        print(f"{label:4s}: {val:.4f} +- {std:.4f}  "
              f"(SRAD {b:.3f} {flag} / U-Net {u:.4f})")
    print("================================================")


def parse_args():
    p = argparse.ArgumentParser(description="DnCNN 6-fold CV 지도학습")
    p.add_argument("--depth",       type=int,   default=20)
    p.add_argument("--channels",    type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patch-size",  type=int,   default=128)
    p.add_argument("--stride",      type=int,   default=32)
    p.add_argument("--ssim-lambda", type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=30)
    p.add_argument("--cpu-threads", type=int,   default=4)
    p.add_argument("--device",      type=str,   default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main()
