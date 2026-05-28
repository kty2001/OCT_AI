"""
1단계 베이스라인: SBSDI D1 데이터셋에 전통적 방법 (NLM, BM3D, SRAD) 적용 및 평가.

결과 저장:
  results/01_baseline/metrics/per_image.csv   -- 이미지별 세부 지표
  results/01_baseline/metrics/summary.csv     -- 방법별 평균/표준편차
  results/01_baseline/images/                 -- 시각적 비교 이미지 (sample)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_all_pairs, compute_metrics
from methods import METHODS

RESULTS_DIR = ROOT / "results" / "01_baseline"
METRICS_DIR = RESULTS_DIR / "metrics"
IMAGES_DIR = RESULTS_DIR / "images"


def save_comparison(idx: int, noisy, clean, outputs: dict[str, np.ndarray]) -> None:
    n_methods = len(outputs)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=(4 * (n_methods + 2), 4))

    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Noisy (Input)")
    axes[0].axis("off")

    axes[1].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Clean (Reference)")
    axes[1].axis("off")

    for col, (name, img) in enumerate(outputs.items(), start=2):
        metrics = compute_metrics(clean, img)
        axes[col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title(
            f"{name.upper()}\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f}"
        )
        axes[col].axis("off")

    plt.tight_layout()
    fig.savefig(IMAGES_DIR / f"sample_{idx:02d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("SBSDI D1 데이터 로드 중...")
    pairs = load_all_pairs()
    print(f"  총 {len(pairs)}쌍 로드 완료. 이미지 크기: {pairs[0][0].shape}")

    records = []
    SAVE_SAMPLE_INDICES = {1, 5, 10, 15}  # 비교 이미지 저장할 인덱스 (1-based)

    # {sample_idx: {method_name: denoised}} 형태로 전체 수집 후 한 번에 저장
    all_outputs: dict[int, dict[str, np.ndarray]] = {i: {} for i in SAVE_SAMPLE_INDICES}

    for method_name, method_fn in METHODS.items():
        print(f"\n[{method_name.upper()}] 실행 중...")

        for i, (noisy, clean) in enumerate(tqdm(pairs, desc=method_name), start=1):
            t0 = time.perf_counter()
            denoised = method_fn(noisy)
            elapsed = time.perf_counter() - t0

            metrics = compute_metrics(clean, denoised)
            records.append({
                "method": method_name,
                "image_idx": i,
                "psnr": round(metrics["psnr"], 4),
                "ssim": round(metrics["ssim"], 4),
                "cnr": round(metrics["cnr"], 4),
                "time_sec": round(elapsed, 3),
            })

            if i in SAVE_SAMPLE_INDICES:
                all_outputs[i][method_name] = denoised

    # 모든 메서드 결과를 한 이미지에 비교 저장
    print("\n비교 이미지 저장 중...")
    for i in SAVE_SAMPLE_INDICES:
        noisy, clean = pairs[i - 1]
        save_comparison(i, noisy, clean, all_outputs[i])

    df = pd.DataFrame(records)
    df.to_csv(METRICS_DIR / "per_image.csv", index=False)
    print(f"\nper_image.csv 저장 완료: {METRICS_DIR / 'per_image.csv'}")

    summary = (
        df.groupby("method")[["psnr", "ssim", "cnr", "time_sec"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(METRICS_DIR / "summary.csv")
    print(f"summary.csv 저장 완료: {METRICS_DIR / 'summary.csv'}")

    # 결과 출력
    print("\n========== 방법별 성능 요약 ==========")
    print(f"{'Method':<8} {'PSNR mean':>10} {'PSNR std':>9} {'SSIM mean':>10} {'SSIM std':>9} {'CNR mean':>9}")
    print("-" * 62)
    for method in METHODS:
        row = summary.loc[method]
        print(
            f"{method.upper():<8} "
            f"{row['psnr_mean']:>10.4f} "
            f"{row['psnr_std']:>9.4f} "
            f"{row['ssim_mean']:>10.4f} "
            f"{row['ssim_std']:>9.4f} "
            f"{row['cnr_mean']:>9.4f}"
        )
    print("========================================")
    print(f"\n비교 이미지: {IMAGES_DIR}")


if __name__ == "__main__":
    run()
