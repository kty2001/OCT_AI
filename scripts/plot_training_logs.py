"""
각 실험 단계의 training_log.csv를 읽어 loss 그래프를 생성한다.
결과는 각 실험의 metrics/ 또는 결과 디렉토리에 저장된다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path.relative_to(ROOT)}")


def plot_single_loss(csv_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["loss"], color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_path)


def plot_kfold(metrics_dir: Path, out_dir: Path, title_prefix: str) -> None:
    fold_dirs = sorted(metrics_dir.glob("fold_*"))
    if not fold_dirs:
        print(f"  fold dir not found: {metrics_dir}")
        return

    colors = plt.cm.tab10.colors

    # --- train_loss (all folds) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, fold_dir in enumerate(fold_dirs):
        df = pd.read_csv(fold_dir / "training_log.csv")
        label = fold_dir.name.replace("_", " ").title()
        ax.plot(df["epoch"], df["train_loss"], color=colors[i], linewidth=1.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title(f"{title_prefix} - Train Loss (All Folds)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir / "loss_train_all_folds.png")

    # --- val_psnr (all folds) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, fold_dir in enumerate(fold_dirs):
        df = pd.read_csv(fold_dir / "training_log.csv")
        label = fold_dir.name.replace("_", " ").title()
        ax.plot(df["epoch"], df["val_psnr"], color=colors[i], linewidth=1.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val PSNR (dB)")
    ax.set_title(f"{title_prefix} - Val PSNR (All Folds)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir / "loss_val_psnr_all_folds.png")

    # --- val_ssim (all folds) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, fold_dir in enumerate(fold_dirs):
        df = pd.read_csv(fold_dir / "training_log.csv")
        label = fold_dir.name.replace("_", " ").title()
        ax.plot(df["epoch"], df["val_ssim"], color=colors[i], linewidth=1.2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val SSIM")
    ax.set_title(f"{title_prefix} - Val SSIM (All Folds)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir / "loss_val_ssim_all_folds.png")

    # --- per-fold: train_loss + val_psnr (dual axis) ---
    for i, fold_dir in enumerate(fold_dirs):
        df = pd.read_csv(fold_dir / "training_log.csv")
        fold_name = fold_dir.name.replace("_", " ").title()
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.plot(df["epoch"], df["train_loss"], color="steelblue", linewidth=1.5, label="Train Loss")
        ax2.plot(df["epoch"], df["val_psnr"], color="tomato", linewidth=1.5, linestyle="--", label="Val PSNR")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="steelblue")
        ax2.set_ylabel("Val PSNR (dB)", color="tomato")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax2.tick_params(axis="y", labelcolor="tomato")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")
        ax1.grid(True, alpha=0.3)
        fig.suptitle(f"{title_prefix} - {fold_name}")
        save_fig(fig, fold_dir / "loss_curve.png")


def main() -> None:
    # Step 3-A: N2N Sub2Full
    print("\n[Step 3-A] N2N Sub2Full")
    plot_single_loss(
        RESULTS / "03_sub2full" / "metrics" / "training_log.csv",
        RESULTS / "03_sub2full" / "metrics" / "loss_curve.png",
        "Step 3-A: N2N Sub2Full - Train Loss",
    )

    # Step 3-B: Supervised (synthetic data)
    print("\n[Step 3-B] Supervised (synthetic)")
    plot_single_loss(
        RESULTS / "03_supervised" / "metrics" / "training_log.csv",
        RESULTS / "03_supervised" / "metrics" / "loss_curve.png",
        "Step 3-B: Supervised (6,136 synthetic pairs) - Train Loss",
    )

    # Step 5: Pre-train -> Fine-tune
    print("\n[Step 5] Pre-train -> Fine-tune")
    plot_single_loss(
        RESULTS / "05_finetune" / "metrics" / "training_log.csv",
        RESULTS / "05_finetune" / "metrics" / "loss_curve.png",
        "Step 5: Fine-tune (L1+SSIM) - Train Loss",
    )

    # Step 6: 6-fold CV U-Net
    print("\n[Step 6] 6-fold CV U-Net")
    plot_kfold(
        RESULTS / "06_kfold" / "metrics",
        RESULTS / "06_kfold" / "metrics",
        "Step 6: U-Net 6-fold CV",
    )

    # Step 7: DnCNN
    print("\n[Step 7] DnCNN 6-fold CV")
    plot_kfold(
        RESULTS / "07_dncnn" / "metrics",
        RESULTS / "07_dncnn" / "metrics",
        "Step 7: DnCNN 6-fold CV",
    )

    # Step 8: NAFNet
    print("\n[Step 8] NAFNet 6-fold CV")
    plot_kfold(
        RESULTS / "08_nafnet" / "metrics",
        RESULTS / "08_nafnet" / "metrics",
        "Step 8: NAFNet 6-fold CV",
    )

    # Step 9: NAFNet + Aug
    print("\n[Step 9] NAFNet + Aug 6-fold CV")
    plot_kfold(
        RESULTS / "09_nafnet_aug" / "metrics",
        RESULTS / "09_nafnet_aug" / "metrics",
        "Step 9: NAFNet+Aug 6-fold CV",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
