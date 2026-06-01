"""
SBSDI D1 clean GT 18장으로부터 합성 노이즈 이미지를 K=4개씩 생성.

노이즈 모델: I = S * Ns + Na
  Ns ~ Gamma(L, 1/L), L=5.266 (2단계 SBSDI D1 캘리브레이션 전체 평균)
  Na ~ N(0, sigma_a^2), sigma_a=0.010

저장 위치:
  data/Final_Publication_2013_SBSDI/augmented_noisy/set{i:02d}/aug_{k:02d}.tif
  data/Final_Publication_2013_SBSDI/augmented_noisy/metadata.csv

재현성: seed = BASE_SEED + (set_idx-1)*K + (aug_idx-1)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "02_synthetic_noise"))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from noise_model import add_speckle

SYNTH_DIR = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments"
OUT_DIR   = ROOT / "data" / "Final_Publication_2013_SBSDI" / "augmented_noisy"

N_PAIRS   = 18
K         = 4       # 쌍당 생성할 합성 노이즈 수
L         = 5.266   # 2단계 SBSDI D1 캘리브레이션 전체 평균값
SIGMA_A   = 0.010
BASE_SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for i in tqdm(range(1, N_PAIRS + 1), desc="Generating"):
        clean_path = SYNTH_DIR / str(i) / "average.tif"
        clean_arr  = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        set_dir = OUT_DIR / f"set{i:02d}"
        set_dir.mkdir(exist_ok=True)

        for k in range(1, K + 1):
            seed = BASE_SEED + (i - 1) * K + (k - 1)
            rng  = np.random.default_rng(seed)

            noisy_arr = add_speckle(clean_arr, L=L, sigma_a=SIGMA_A, rng=rng)
            out_path  = set_dir / f"aug_{k:02d}.tif"
            Image.fromarray((noisy_arr * 255).astype(np.uint8)).save(out_path)

            records.append({
                "set_idx":    i,
                "aug_idx":    k,
                "clean_path": str(SYNTH_DIR / str(i) / "average.tif"),
                "noisy_path": str(out_path),
                "L":          L,
                "sigma_a":    SIGMA_A,
                "seed":       seed,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "metadata.csv", index=False)

    print(f"\n생성 완료: {N_PAIRS}세트 x {K}개 = {N_PAIRS * K}개 합성 노이즈 이미지")
    print(f"저장 위치: {OUT_DIR}")
    print(f"메타데이터: {OUT_DIR / 'metadata.csv'}")

    # 간단한 검증 — 원본 real noisy 대비 PSNR 분포 확인
    from utils import compute_psnr
    real_psnrs, aug_psnrs = [], []
    for i in range(1, N_PAIRS + 1):
        clean = np.array(Image.open(SYNTH_DIR / str(i) / "average.tif").convert("L"),
                         dtype=np.float32) / 255.0
        real  = np.array(Image.open(SYNTH_DIR / str(i) / "test.tif").convert("L"),
                         dtype=np.float32) / 255.0
        real_psnrs.append(compute_psnr(clean, real))

        for k in range(1, K + 1):
            aug = np.array(Image.open(OUT_DIR / f"set{i:02d}" / f"aug_{k:02d}.tif").convert("L"),
                           dtype=np.float32) / 255.0
            aug_psnrs.append(compute_psnr(clean, aug))

    print(f"\n[검증] real noisy PSNR:  {np.mean(real_psnrs):.2f} +- {np.std(real_psnrs):.2f} dB")
    print(f"[검증] aug  noisy PSNR:  {np.mean(aug_psnrs):.2f} +- {np.std(aug_psnrs):.2f} dB")
    print("(aug PSNR이 real PSNR과 유사하면 노이즈 수준이 적절한 것)")


if __name__ == "__main__":
    main()
