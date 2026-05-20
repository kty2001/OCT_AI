# 데이터셋 목록 및 구조

현재 `data/` 디렉토리에 저장된 데이터셋 목록입니다.

---

## 1. Kermany Retinal OCT Images (OCT2017)

| 항목 | 내용 |
|------|------|
| 경로 | `data/kaggle_RetinalOCTImages/oct2017/OCT2017_/` |
| 출처 | [Kaggle - paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| 촬영 장비 | Heidelberg Spectralis OCT |
| 이미지 포맷 | JPEG, 그레이스케일(L), 약 512×496 |
| 라이선스 | CC BY 4.0 |

### 클래스 설명

| 클래스 | 설명 |
|--------|------|
| CNV | 맥락막 신생혈관 (Choroidal Neovascularization) |
| DME | 당뇨황반부종 (Diabetic Macular Edema) |
| DRUSEN | 드루젠 (초기 AMD) |
| NORMAL | 정상 망막 |

### 분할별 이미지 수

| split | CNV | DME | DRUSEN | NORMAL | 합계 |
|-------|-----|-----|--------|--------|------|
| train | 37,205 | 11,348 | 8,616 | 26,315 | 83,484 |
| val | 8 | 8 | 8 | 8 | 32 |
| test | 242 | 242 | 242 | 242 | 968 |
| **합계** | **37,455** | **11,598** | **8,866** | **26,565** | **84,484** |

### 디렉토리 구조

```
OCT2017_/
├── train/
│   ├── CNV/       *.jpeg
│   ├── DME/       *.jpeg
│   ├── DRUSEN/    *.jpeg
│   └── NORMAL/    *.jpeg
├── val/
│   └── (동일 구조, 클래스당 8장)
└── test/
    └── (동일 구조, 클래스당 242장)
```

### 특이사항

- 분류(Classification) 태스크 전용 데이터. noisy-clean 쌍 없음
- 이미지마다 해상도가 일정하지 않으며 망막의 위치·기울기가 제각각
- train 클래스 불균형 심함 (CNV 37k vs DRUSEN 8.6k)
- Diffusion 모델 사전학습, 자기지도 디노이징(Noise2Void 등) 백본 사전학습에 활용 가능

---

## 2. AROI (Annotated Retinal OCT Images Database)

| 항목 | 내용 |
|------|------|
| 경로 | `data/AROI/` |
| 출처 | [ipg.fer.hr](https://ipg.fer.hr/ipg/resources/oct_image_database) |
| 논문 | Automatika, Vol.62, 2021 |
| 촬영 장비 | Heidelberg Spectralis OCT |
| 대상 | nAMD (신생혈관성 황반변성) 환자 24명 |
| 이미지 포맷 | PNG, 512×1024 (가로×세로) |
| 라이선스 | 연구 목적 공개 |

### 레이블 구조 (6클래스)

| number 값 | colour 색상 | 의미 |
|-----------|------------|------|
| 0 | 검정 | 배경 (유리체, ILM 위) |
| 1 | 빨강 | 내망막 레이어 (ILM ~ IPL/INL) |
| 2 | 노랑 | 외망막 레이어 (IPL/INL ~ RPE) |
| 3 | 시안 | 망막 유체 (SRF / IRF / PED) |
| 4 | 초록 | RPE ~ BM 레이어 |
| 5 | 파랑 | 맥락막 (BM 아래) |

### 데이터 수

| 구분 | B-scan 수 |
|------|----------|
| 전체 raw (미주석 포함) | 3,072장 (24명 × 128장) |
| 주석 완료 (labeled) | 1,136장 |
| inter_intra 검증용 | 300장 |

### 디렉토리 구조

```
AROI/
├── 24 patient/
│   └── patientN/           (N = 1~24)
│       ├── raw/
│       │   ├── ALL/        전체 128장 B-scan
│       │   └── labeled/    주석된 B-scan만
│       └── mask/
│           ├── colour/     색상 마스크 (RGBA, 시각화용)
│           └── number/     레이블 마스크 (Grayscale L, 0~5, 학습용)
└── inter_intra/
    ├── ground truth/
    ├── interobserver/
    └── intraobserver/
```

### 환자별 주석 B-scan 수

| 환자 | labeled | 환자 | labeled |
|------|---------|------|---------|
| patient1 | 41 | patient13 | 30 |
| patient2 | 68 | patient14 | 31 |
| patient3 | 20 | patient15 | 97 |
| patient4 | 56 | patient16 | 49 |
| patient5 | 49 | patient17 | 21 |
| patient6 | 36 | patient18 | 61 |
| patient7 | 103 | patient19 | 21 |
| patient8 | 71 | patient20 | 66 |
| patient9 | 73 | patient21 | 21 |
| patient10 | 87 | patient22 | 31 |
| patient11 | 21 | patient23 | 31 |
| patient12 | 19 | patient24 | 33 |

### 특이사항

- 이미지가 세로로 긴 형태(512×1024)로 저장 — 일반적인 OCT B-scan(가로 긴 형태)과 **90도 회전**된 상태
- `number/` 마스크 값이 0~5로 매우 작아 이미지 뷰어로 보면 전부 검게 보임
- `inter_intra/`는 동일 B-scan에 대해 복수의 전문의가 주석한 데이터로 알고리즘 재현성 평가용

---

## 3. MedSegBench

| 항목 | 내용 |
|------|------|
| 경로 | `data/MedSegBench/` |
| 출처 | [Nature Scientific Data, 2024](https://www.nature.com/articles/s41597-024-04159-2) |
| 포맷 | `.npz` (NumPy 압축 배열) |
| 변환 이미지 | `data/MedSegBench_images/` (PNG 변환본) |

현재 저장된 데이터셋 3종:

---

### 3-1. cystoidfluid (망막 낭성액 세그멘테이션)

| 항목 | 내용 |
|------|------|
| 파일 | `cystoidfluid_256.npz` |
| 모달리티 | **망막 OCT** |
| 이미지 | RGB, 256×256 |
| 태스크 | Binary segmentation |
| 라이선스 | CC BY-NC-SA 4.0 |

| split | 이미지 수 |
|-------|---------|
| train | 703 |
| val | 101 |
| test | 202 |
| **합계** | **1,006** |

레이블: `0` = 배경, `1` = 낭성액 (intraretinal fluid)

---

### 3-2. wbc (백혈구 세그멘테이션)

| 항목 | 내용 |
|------|------|
| 파일 | `wbc_128.npz`, `wbc_256.npz`, `wbc_512.npz` |
| 모달리티 | 현미경 이미지 |
| 이미지 | RGB |
| 태스크 | Multi-class segmentation (3클래스) |
| 라이선스 | CC BY 4.0 |

| split | 이미지 수 |
|-------|---------|
| train | 280 |
| val | 40 |
| test | 80 |
| **합계** | **400** |

레이블: `0` = 배경, `1` = 세포질(cytoplasm), `2` = 핵(nucleus)

세포 종류별 서브셋(C1~C4): Lymphocyte / Monocyte / Neutrophil / Eosinophil

---

### 3-3. yeaz (효모 세포 세그멘테이션)

| 항목 | 내용 |
|------|------|
| 파일 | `yeaz_128.npz`, `yeaz_256.npz`, `yeaz_512.npz` |
| 모달리티 | 현미경 이미지 |
| 이미지 | Grayscale |
| 태스크 | Binary segmentation |
| 라이선스 | CC BY 4.0 |

| split | 이미지 수 |
|-------|---------|
| train | 360 |
| val | 96 |
| test | 251 |
| **합계** | **707** |

레이블: `0` = 배경, `1` = 효모 세포

---

### npz 파일 로드 방법

```python
import numpy as np

data = np.load("data/MedSegBench/cystoidfluid_256.npz")
# 키: train_images, train_label, val_images, val_label, test_images, test_label
images = data["train_images"]   # shape: (N, H, W, 3) or (N, H, W)
labels = data["train_label"]    # shape: (N, H, W)
```

---

## 데이터셋 요약

| 데이터셋 | 모달리티 | 이미지 수 | 태스크 | 노이즈 쌍 |
|---------|---------|---------|--------|---------|
| Kermany OCT2017 | 망막 OCT | 84,484 | 분류 | 없음 |
| AROI | 망막 OCT | 3,072 (주석 1,136) | 레이어 세그멘테이션 | 없음 |
| MedSegBench / cystoidfluid | 망막 OCT | 1,006 | 유체 세그멘테이션 | 없음 |
| MedSegBench / wbc | 현미경 | 400 | 세포 세그멘테이션 | 없음 |
| MedSegBench / yeaz | 현미경 | 707 | 세포 세그멘테이션 | 없음 |
