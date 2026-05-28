"""
합성 스페클 노이즈 생성 파이프라인.

AROI 및 Kermany OCT2017 clean 이미지에 Gamma 곱셈성 스페클 노이즈를 합성하여
noisy-clean 학습 쌍을 생성한다.

사용법:
  uv run python scripts/02_synthetic_noise/run_synthetic.py
  uv run python scripts/02_synthetic_noise/run_synthetic.py --calibrate
  uv run python scripts/02_synthetic_noise/run_synthetic.py --L 3.5 --sigma-a 0.008 --max-kermany 5000

결과 저장:
  results/02_synthetic_noise/AROI/clean/      *.png
  results/02_synthetic_noise/AROI/noisy/      *.png
  results/02_synthetic_noise/AROI/samples/    비교 이미지
  results/02_synthetic_noise/Kermany/clean/   *.png
  results/02_synthetic_noise/Kermany/noisy/   *.png
  results/02_synthetic_noise/Kermany/samples/ 비교 이미지
  results/02_synthetic_noise/metadata.csv     경로 및 메타데이터
  results/02_synthetic_noise/noise_params.json 사용된 노이즈 파라미터
"""
from __future__ import annotations

import argparse
import json
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
from tqdm import tqdm

from noise_model import add_speckle, estimate_noise_params

AROI_ROOT = ROOT / "data" / "AROI" / "24 patient"
KERMANY_ROOT = ROOT / "data" / "kaggle_RetinalOCTImages" / "oct2017" / "OCT2017_"
SBSDI_DIR = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments"
OUT_DIR = ROOT / "results" / "02_synthetic_noise"

SAMPLE_COUNT = 5


def collect_aroi_paths() -> list[Path]:
    """AROI labeled B-scan 경로 수집 (주석 완료된 이미지만)."""
    paths = []
    for patient_dir in sorted(AROI_ROOT.iterdir()):
        labeled_dir = patient_dir / "raw" / "labeled"
        if labeled_dir.exists():
            paths.extend(sorted(labeled_dir.glob("*.png")))
    return paths


def collect_kermany_paths(max_images: int | None = None) -> list[Path]:
    """Kermany OCT2017 train 이미지 경로 수집."""
    paths = []
    train_dir = KERMANY_ROOT / "train"
    if not train_dir.exists():
        print(f"  Kermany train 디렉토리 없음: {train_dir}")
        return paths
    for cls_dir in sorted(train_dir.iterdir()):
        if cls_dir.is_dir():
            paths.extend(sorted(cls_dir.glob("*.jpeg")))
    if max_images is not None and len(paths) > max_images:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def load_aroi_image(path: Path) -> np.ndarray:
    """AROI 이미지 로드 및 90도 CCW 회전 (표준 landscape 방향 복원)."""
    img = Image.open(path).convert("L")
    img = img.transpose(Image.ROTATE_90)
    return np.array(img, dtype=np.float32) / 255.0


def load_kermany_image(path: Path) -> np.ndarray:
    """Kermany JPEG 이미지 로드."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def calibrate_from_sbsdi() -> dict[str, float]:
    """SBSDI D1 18쌍에서 노이즈 파라미터 추정."""
    L_list, sig_list = [], []
    for i in range(1, 19):
        folder = SBSDI_DIR / str(i)
        noisy = np.array(Image.open(folder / "test.tif").convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(folder / "average.tif").convert("L"), dtype=np.float32) / 255.0
        p = estimate_noise_params(noisy, clean)
        L_list.append(p["L"])
        sig_list.append(p["sigma_a"])
    return {
        "L": round(float(np.mean(L_list)), 3),
        "sigma_a": round(float(np.mean(sig_list)), 5),
    }


def save_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix} Clean")
    axes[0].axis("off")
    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"{title_prefix} Synthetic Noisy")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def process_dataset(
    name: str,
    image_paths: list[Path],
    load_fn,
    params: dict[str, float],
    rng: np.random.Generator,
) -> list[dict]:
    """단일 데이터셋에 대해 noisy 이미지 생성 및 저장."""
    ds_dir = OUT_DIR / name
    clean_dir = ds_dir / "clean"
    noisy_dir = ds_dir / "noisy"
    sample_dir = ds_dir / "samples"
    for d in (clean_dir, noisy_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    sample_indices = set(
        np.linspace(0, len(image_paths) - 1, min(SAMPLE_COUNT, len(image_paths)), dtype=int)
    )

    records = []
    for i, src_path in enumerate(tqdm(image_paths, desc=name)):
        clean_arr = load_fn(src_path)
        noisy_arr = add_speckle(clean_arr, L=params["L"], sigma_a=params["sigma_a"], rng=rng)

        stem = f"{name}_{i:05d}"
        clean_out = clean_dir / f"{stem}.png"
        noisy_out = noisy_dir / f"{stem}.png"

        Image.fromarray((clean_arr * 255).astype(np.uint8), mode="L").save(clean_out)
        Image.fromarray((noisy_arr * 255).astype(np.uint8), mode="L").save(noisy_out)

        if i in sample_indices:
            save_comparison(
                clean_arr,
                noisy_arr,
                sample_dir / f"sample_{i:05d}.png",
                title_prefix=f"{name} {i}",
            )

        records.append({
            "dataset": name,
            "idx": i,
            "source": str(src_path.relative_to(ROOT)),
            "clean": str(clean_out.relative_to(ROOT)),
            "noisy": str(noisy_out.relative_to(ROOT)),
            "height": clean_arr.shape[0],
            "width": clean_arr.shape[1],
        })

    return records


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 노이즈 파라미터 결정
    if args.calibrate:
        print("SBSDI D1에서 노이즈 파라미터 캘리브레이션 중...")
        params = calibrate_from_sbsdi()
        print(f"  추정 결과: L={params['L']}, sigma_a={params['sigma_a']}")
    else:
        params = {"L": args.L, "sigma_a": args.sigma_a}
        print(f"지정된 파라미터 사용: L={params['L']}, sigma_a={params['sigma_a']}")

    params_path = OUT_DIR / "noise_params.json"
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"파라미터 저장: {params_path}")

    rng = np.random.default_rng(args.seed)
    all_records = []

    # AROI 처리
    if "aroi" in args.datasets:
        print("\nAROI 이미지 경로 수집 중...")
        aroi_paths = collect_aroi_paths()
        print(f"  {len(aroi_paths)}장 발견")
        if aroi_paths:
            records = process_dataset("AROI", aroi_paths, load_aroi_image, params, rng)
            all_records.extend(records)
            print(f"  AROI 처리 완료: {len(records)}쌍")
        else:
            print("  AROI 이미지 없음, 건너뜀")

    # Kermany 처리
    if "kermany" in args.datasets:
        print("\nKermany 이미지 경로 수집 중...")
        kermany_paths = collect_kermany_paths(args.max_kermany)
        print(f"  {len(kermany_paths)}장 발견 (max_kermany={args.max_kermany})")
        if kermany_paths:
            records = process_dataset("Kermany", kermany_paths, load_kermany_image, params, rng)
            all_records.extend(records)
            print(f"  Kermany 처리 완료: {len(records)}쌍")
        else:
            print("  Kermany 이미지 없음, 건너뜀")

    if not all_records:
        print("처리된 이미지 없음. 데이터 경로를 확인하세요.")
        return

    df = pd.DataFrame(all_records)
    meta_path = OUT_DIR / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"\n{'='*50}")
    print(f"합성 완료")
    print(f"  총 쌍: {len(df)}")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"  {ds}: {len(sub)}쌍, "
              f"해상도 {sub['height'].mode()[0]}x{sub['width'].mode()[0]} (최빈)")
    print(f"  메타데이터: {meta_path}")
    print(f"  노이즈 파라미터: L={params['L']}, sigma_a={params['sigma_a']}")
    print(f"  결과 디렉토리: {OUT_DIR}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCT 합성 스페클 노이즈 생성 파이프라인")
    p.add_argument(
        "--datasets", nargs="+", default=["aroi", "kermany"],
        choices=["aroi", "kermany"],
        help="처리할 데이터셋 (기본: aroi kermany)",
    )
    p.add_argument(
        "--calibrate", action="store_true",
        help="SBSDI D1에서 노이즈 파라미터 자동 캘리브레이션",
    )
    p.add_argument("--L", type=float, default=4.0, help="looks 수 (기본: 4.0)")
    p.add_argument("--sigma-a", dest="sigma_a", type=float, default=0.01,
                   help="가산 노이즈 표준편차 (기본: 0.01)")
    p.add_argument(
        "--max-kermany", dest="max_kermany", type=int, default=None,
        help="Kermany 최대 이미지 수 (기본: 전체)",
    )
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본: 42)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
