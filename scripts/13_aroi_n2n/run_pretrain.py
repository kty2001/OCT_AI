"""
13단계 사전학습: AROI 인접 B-scan N2N 사전학습.

24명 × 128장 순차 B-scan에서 인접 슬라이스 쌍 (i, i+1)을 N2N 입력으로 사용.
조직 구조가 유사하고 스페클 패턴이 독립적인 인접 슬라이스를 N2N 조건으로 활용.

Train: patient1~20 (2,540쌍), Val: patient21~24 (508쌍, N2N loss)
SBSDI D1 18쌍 PSNR을 주기적으로 모니터링 (early stopping 기준은 val N2N loss).

결과:
  results/13_aroi_n2n/pretrain/best.pth
  results/13_aroi_n2n/pretrain/training_log.csv

실행:
  uv run python scripts/13_aroi_n2n/run_pretrain.py
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "03_sub2full"))
sys.path.insert(0, str(ROOT / "scripts" / "08_nafnet"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import NAFNet
from utils import compute_metrics

AROI_ROOT  = ROOT / "data" / "AROI" / "24 patient"
SBSDI_ROOT = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments"
OUT_DIR    = ROOT / "results" / "13_aroi_n2n" / "pretrain"

N_TRAIN_PATIENTS = 20
MONITOR_INTERVAL = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_aroi_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").transpose(Image.ROTATE_90)
    return np.array(img, dtype=np.float32) / 255.0


def collect_patient_sequences() -> list[list[Path]]:
    sequences = []
    for patient_dir in sorted(AROI_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue
        all_dir = patient_dir / "raw" / "ALL"
        if all_dir.exists():
            scans = sorted(all_dir.glob("*.png"))
            if len(scans) >= 2:
                sequences.append(scans)
    return sequences


def make_n2n_pairs(sequences: list[list[Path]]) -> list[tuple[Path, Path]]:
    pairs = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            pairs.append((seq[i], seq[i + 1]))
    return pairs


def load_sbsdi_d1() -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for i in range(1, 19):
        d = SBSDI_ROOT / str(i)
        noisy = np.array(Image.open(d / "test.tif").convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(d / "average.tif").convert("L"), dtype=np.float32) / 255.0
        pairs.append((noisy, clean))
    return pairs


class N2NPatchDataset(Dataset):
    """Lazy-loading N2N dataset with random crop.

    이미지를 미리 로드하지 않고 __getitem__ 호출 시 on-demand로 로드.
    exhaustive tiling 대신 random crop을 사용해 메모리와 속도 문제 해결.
    samples_per_epoch으로 epoch당 스텝 수를 제어.
    """
    def __init__(self, pair_paths: list[tuple[Path, Path]],
                 patch_size: int = 128,
                 samples_per_epoch: int = 4096,
                 augment: bool = True):
        self.pair_paths = pair_paths
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.patch_size
        p1, p2 = self.pair_paths[np.random.randint(len(self.pair_paths))]
        a = load_aroi_image(p1)
        b = load_aroi_image(p2)
        H, W = a.shape
        top  = np.random.randint(0, H - ps + 1)
        left = np.random.randint(0, W - ps + 1)
        a = a[top:top+ps, left:left+ps].copy()
        b = b[top:top+ps, left:left+ps].copy()
        if self.augment:
            if np.random.rand() > 0.5:
                a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
            if np.random.rand() > 0.5:
                a, b = np.flipud(a).copy(), np.flipud(b).copy()
            if np.random.rand() > 0.5:
                a, b = np.rot90(a, 1).copy(), np.rot90(b, 1).copy()
        return torch.from_numpy(a[None]), torch.from_numpy(b[None])


def pad_to_multiple(img: np.ndarray, multiple: int = 16):
    H, W = img.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    return np.pad(img, ((0, pH), (0, pW)), mode="reflect"), (H, W)


def infer(model: nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    padded, (H, W) = pad_to_multiple(img, 16)
    x = torch.from_numpy(padded[None, None]).to(device)
    with torch.no_grad():
        out = model(x).clamp(0.0, 1.0)
    return out[0, 0].cpu().numpy()[:H, :W]


def eval_d1(model: nn.Module, d1_pairs: list, device: torch.device) -> float:
    model.eval()
    psnrs = [compute_metrics(c, infer(model, n, device))["psnr"] for n, c in d1_pairs]
    return float(np.mean(psnrs))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sequences = collect_patient_sequences()
    print(f"환자 수: {len(sequences)}, 총 B-scan: {sum(len(s) for s in sequences)}")

    train_seqs = sequences[:N_TRAIN_PATIENTS]
    val_seqs   = sequences[N_TRAIN_PATIENTS:]
    train_pairs = make_n2n_pairs(train_seqs)
    val_pairs   = make_n2n_pairs(val_seqs)
    print(f"Train N2N 쌍: {len(train_pairs)}, Val N2N 쌍: {len(val_pairs)}")

    d1_pairs = load_sbsdi_d1()

    train_dataset = N2NPatchDataset(train_pairs, args.patch_size,
                                    args.samples_per_epoch, augment=True)
    val_dataset   = N2NPatchDataset(val_pairs, args.patch_size,
                                    args.samples_per_epoch // 4, augment=False)
    print(f"Train samples/epoch: {len(train_dataset)}, "
          f"steps/epoch: {len(train_dataset)//args.batch_size}")
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0,
                               pin_memory=(device.type == "cuda"))
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    enc_blks = [int(x) for x in args.enc_blks.split(",")]
    dec_blks = [int(x) for x in args.dec_blks.split(",")]
    model = NAFNet(in_ch=1, width=args.width,
                   enc_blks=enc_blks, mid_blks=args.mid_blks, dec_blks=dec_blks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet: width={args.width}, {n_params:,} params")
    print(f"Steps/epoch: {len(train_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)
    criterion = nn.L1Loss()

    log_records = []
    best_val_loss = float("inf")
    patience_cnt = 0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for a, b in train_loader:
            a, b = a.to(device), b.to(device)
            optimizer.zero_grad()
            criterion(model(a).clamp(0, 1), b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += criterion(model(a).clamp(0, 1), b).item()
        scheduler.step()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a, b in val_loader:
                a, b = a.to(device), b.to(device)
                val_loss += criterion(model(a).clamp(0, 1), b).item()
        avg_val = val_loss / len(val_loader)

        record = {"epoch": epoch, "train_loss": round(avg_train, 6),
                  "val_loss": round(avg_val, 6)}

        if epoch % MONITOR_INTERVAL == 0 or epoch == 1:
            d1_psnr = eval_d1(model, d1_pairs, device)
            elapsed = (time.perf_counter() - start) / 60
            print(f"  Ep {epoch:4d}/{args.epochs} | "
                  f"train={avg_train:.5f} val={avg_val:.5f} | "
                  f"D1 PSNR={d1_psnr:.3f} | "
                  f"patience={patience_cnt}/{args.patience} | "
                  f"{elapsed:.1f}min", flush=True)
            record["d1_psnr"] = round(d1_psnr, 4)

        log_records.append(record)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stop @ epoch {epoch} | best val loss={best_val_loss:.6f}")
                break

    pd.DataFrame(log_records).to_csv(OUT_DIR / "training_log.csv", index=False)
    elapsed = (time.perf_counter() - start) / 60
    print(f"\n사전학습 완료 | best val loss: {best_val_loss:.6f} | {elapsed:.1f}분")
    print(f"가중치 저장: {OUT_DIR / 'best.pth'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AROI N2N 사전학습")
    p.add_argument("--width",      type=int,   default=32)
    p.add_argument("--enc-blks",   type=str,   default="1,1,1,28")
    p.add_argument("--mid-blks",   type=int,   default=1)
    p.add_argument("--dec-blks",   type=str,   default="1,1,1,1")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=48)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patch-size",        type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=4096,
                   help="epoch당 랜덤 샘플 수 (steps = samples/batch)")
    p.add_argument("--patience",          type=int, default=30)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
