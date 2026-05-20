"""
MedSegBench npz 파일을 PNG 이미지로 변환하는 스크립트.

출력 구조:
data/MedSegBench_images/
  {dataset}/
    {resolution}/
      {split}/
        images/   원본 이미지
        masks/    클래스별 색상이 입혀진 세그멘테이션 마스크
        overlay/  원본 위에 마스크를 반투명하게 오버레이한 이미지
"""

import numpy as np
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("data/MedSegBench")
OUTPUT_DIR = Path("data/MedSegBench_images")

# 데이터셋별 마스크 클래스 색상 (배경은 항상 검정)
COLORMAP = {
    "wbc":          {0: (0, 0, 0), 1: (255, 50, 50), 2: (50, 255, 50)},
    "yeaz":         {0: (0, 0, 0), 1: (255, 255, 255)},
    "cystoidfluid": {0: (0, 0, 0), 1: (0, 200, 255)},   # 낭성액: 하늘색
}


def colorize_mask(mask: np.ndarray, colormap: dict) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in colormap.items():
        rgb[mask == label] = color
    return rgb


def overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """마스크를 이미지 위에 반투명하게 합성. 배경(검정) 픽셀은 원본 유지."""
    img = image.copy().astype(np.float32)
    is_fg = mask_rgb.sum(axis=2) > 0
    img[is_fg] = img[is_fg] * (1 - alpha) + mask_rgb[is_fg].astype(np.float32) * alpha
    return img.clip(0, 255).astype(np.uint8)


def save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def convert_npz(npz_path: Path) -> None:
    parts = npz_path.stem.split("_")   # e.g. ['wbc', '128']
    dataset, resolution = parts[0], parts[1]
    cmap = COLORMAP[dataset]

    data = np.load(npz_path)
    splits = {k.split("_")[0] for k in data.files if not k.endswith(("C1","C2","C3","C4"))}

    for split in sorted(splits):
        images_key = f"{split}_images"
        labels_key = f"{split}_label"
        if images_key not in data.files or labels_key not in data.files:
            continue

        images = data[images_key]   # (N, H, W) or (N, H, W, 3)
        labels = data[labels_key]   # (N, H, W)
        n = len(images)

        base = OUTPUT_DIR / dataset / resolution / split
        print(f"  {dataset}/{resolution}/{split}: {n}장")

        for i in range(n):
            idx = f"{i:04d}"

            # 이미지 저장 (그레이스케일이면 RGB로 변환)
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            save(img, base / "images" / f"{idx}.png")

            # 마스크 저장
            mask_rgb = colorize_mask(labels[i], cmap)
            save(mask_rgb, base / "masks" / f"{idx}.png")

            # 오버레이 저장
            ov = overlay(img, mask_rgb)
            save(ov, base / "overlay" / f"{idx}.png")


def main() -> None:
    npz_files = sorted(INPUT_DIR.glob("*.npz"))
    if not npz_files:
        print(f"npz 파일을 찾을 수 없습니다: {INPUT_DIR}")
        return

    print(f"변환 대상: {len(npz_files)}개 파일\n")
    for npz_path in npz_files:
        print(f"[{npz_path.name}]")
        convert_npz(npz_path)

    print(f"\n완료. 출력 경로: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
