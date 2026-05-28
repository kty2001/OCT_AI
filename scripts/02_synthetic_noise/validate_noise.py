"""
SBSDI D1 데이터에서 실제 스페클 노이즈 파라미터를 추정하고
합성 노이즈와 실제 노이즈 분포를 비교하여 노이즈 모델 검증.

결과 저장:
  results/02_synthetic_noise/validation/params.csv     -- 쌍별 추정 파라미터
  results/02_synthetic_noise/validation/summary.txt    -- 통계 요약
  results/02_synthetic_noise/validation/dist_*.png     -- 분포 비교 플롯
  results/02_synthetic_noise/validation/samples_*.png  -- 시각적 비교
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from tqdm import tqdm

from noise_model import add_speckle, estimate_noise_params

SBSDI_DIR = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments"
OUT_DIR = ROOT / "results" / "02_synthetic_noise" / "validation"
N_PAIRS = 18


def load_pair(idx: int) -> tuple[np.ndarray, np.ndarray]:
    folder = SBSDI_DIR / str(idx)
    noisy = np.array(Image.open(folder / "test.tif").convert("L"), dtype=np.float32) / 255.0
    clean = np.array(Image.open(folder / "average.tif").convert("L"), dtype=np.float32) / 255.0
    return noisy, clean


def plot_noise_distribution(
    noisy: np.ndarray,
    clean: np.ndarray,
    synth_noisy: np.ndarray,
    idx: int,
    L_est: float,
    sigma_a: float,
) -> None:
    """실제 N_s 분포와 합성 Gamma 분포 비교 히스토그램."""
    fg = clean > 0.05
    Ns_real = noisy[fg] / (clean[fg] + 1e-8)
    Ns_synth = synth_noisy[fg] / (clean[fg] + 1e-8)

    x = np.linspace(0, 4, 300)
    gamma_pdf = stats.gamma.pdf(x, a=L_est, scale=1.0 / L_est)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(Ns_real, bins=80, range=(0, 4), density=True, alpha=0.6,
                 color="steelblue", label="Real N_s")
    axes[0].hist(Ns_synth, bins=80, range=(0, 4), density=True, alpha=0.5,
                 color="darkorange", label="Synth N_s")
    axes[0].plot(x, gamma_pdf, "r-", linewidth=2, label=f"Gamma(L={L_est:.2f})")
    axes[0].set_xlabel("N_s = I / S")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Pair {idx}: Speckle distribution")
    axes[0].legend()
    axes[0].set_xlim(0, 4)

    diff = noisy - synth_noisy
    axes[1].hist(diff.ravel(), bins=80, density=True, alpha=0.7, color="purple")
    axes[1].set_xlabel("Real - Synth pixel difference")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Pair {idx}: Pixel diff (sigma_a={sigma_a:.4f})")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"dist_{idx:02d}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def plot_sample_comparison(
    noisy: np.ndarray,
    clean: np.ndarray,
    synth_noisy: np.ndarray,
    idx: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = clean.min(), clean.max()

    axes[0].imshow(noisy, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Pair {idx}: Real Noisy")
    axes[0].axis("off")

    axes[1].imshow(clean, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Pair {idx}: Clean Reference")
    axes[1].axis("off")

    axes[2].imshow(synth_noisy, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Pair {idx}: Synthetic Noisy")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"sample_{idx:02d}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, float]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    records = []
    ks_stats = []
    PLOT_INDICES = {1, 5, 10, 15, 18}

    print("SBSDI D1 노이즈 파라미터 추정 중...")
    for i in tqdm(range(1, N_PAIRS + 1)):
        noisy, clean = load_pair(i)
        params = estimate_noise_params(noisy, clean)
        L_est, sigma_a = params["L"], params["sigma_a"]

        synth_noisy = add_speckle(clean, L=L_est, sigma_a=sigma_a, rng=rng)

        fg = clean > 0.05
        if fg.sum() > 100:
            Ns_real = (noisy[fg] / (clean[fg] + 1e-8)).clip(0, 4)
            ks_stat, ks_p = stats.ks_2samp(
                Ns_real,
                (synth_noisy[fg] / (clean[fg] + 1e-8)).clip(0, 4),
            )
        else:
            ks_stat, ks_p = float("nan"), float("nan")

        records.append({
            "pair": i,
            "L": L_est,
            "sigma_a": sigma_a,
            "ks_stat": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
            "ks_p": round(ks_p, 4) if not np.isnan(ks_p) else None,
        })
        ks_stats.append(ks_stat)

        if i in PLOT_INDICES:
            plot_noise_distribution(noisy, clean, synth_noisy, i, L_est, sigma_a)
            plot_sample_comparison(noisy, clean, synth_noisy, i)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "params.csv", index=False)

    L_vals = df["L"].values
    sig_vals = df["sigma_a"].values
    ks_vals = [v for v in ks_stats if not np.isnan(v)]

    summary = (
        f"SBSDI D1 스페클 노이즈 파라미터 추정 결과 (N={N_PAIRS}쌍)\n"
        f"{'='*50}\n"
        f"L (looks)\n"
        f"  평균:  {L_vals.mean():.3f}\n"
        f"  표준편차: {L_vals.std():.3f}\n"
        f"  범위:  [{L_vals.min():.3f}, {L_vals.max():.3f}]\n"
        f"\nsigma_a (additive noise std)\n"
        f"  평균:  {sig_vals.mean():.5f}\n"
        f"  표준편차: {sig_vals.std():.5f}\n"
        f"  범위:  [{sig_vals.min():.5f}, {sig_vals.max():.5f}]\n"
        f"\nKS 통계량 (실제 vs 합성 N_s 분포)\n"
        f"  평균:  {np.mean(ks_vals):.4f}\n"
        f"  (0에 가까울수록 분포 일치)\n"
        f"\n권장 합성 파라미터:\n"
        f"  L = {L_vals.mean():.2f}  (mean)\n"
        f"  sigma_a = {sig_vals.mean():.5f}  (mean)\n"
    )

    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    calibrated = {
        "L": round(float(L_vals.mean()), 3),
        "sigma_a": round(float(sig_vals.mean()), 5),
    }
    print(f"파라미터 저장: {OUT_DIR / 'params.csv'}")
    return calibrated


if __name__ == "__main__":
    run()
