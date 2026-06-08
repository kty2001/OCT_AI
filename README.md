# AI 기반 OCT 치주질환 처리

OCT(Optical Coherence Tomography)를 활용한 치주질환 탐지·예측 연구에 AI를 도입하는 프로젝트.
치주 OCT 공개 데이터가 극히 부족하므로, 안과 영역의 대규모 망막 OCT 데이터셋으로 모델을 학습한 뒤 치주 도메인으로 전이하는 전략을 취한다.
핵심 목표는 스페클 노이즈 제거와 Super-Resolution을 단일 AI 파이프라인으로 처리하여 진단 품질을 높이는 것이다.

---

## 진행 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 1단계 | 전통적 방법 베이스라인 측정 | 완료 |
| 2단계 | 합성 스페클 노이즈 생성 파이프라인 | 완료 |
| 3단계-A | 자가지도 학습 (N2N 프레임쌍) | 완료 |
| 3단계-B | 지도학습 (합성 noisy-clean 6,136쌍) | 완료 — 도메인 갭 확인 |
| 4단계 | 사전 학습 SR 모델 적용 (Real-ESRGAN 18쌍 전체 평가) | 완료 |
| 5단계 | Pre-train → Fine-tune (합성 사전학습 + N2N fine-tuning, L1+SSIM Loss) | 완료 |
| 6단계-A | 6-fold CV 지도학습 (U-Net, batch=16, ep=150) | 완료 |
| 6단계-B | 6-fold CV 지도학습 (U-Net, batch=64, early stopping, patience=30) | 완료 — SRAD 초과 |
| 7단계 | DnCNN 백본 6-fold CV (depth=20, from scratch) | 완료 — U-Net과 동등, 데이터 병목 확인 |
| 8단계 | NAFNet 백본 6-fold CV (width=32, from scratch) | 완료 — DnCNN/U-Net과 동등, 데이터 병목 재확인 |
| 9단계 | 다중 노이즈 재실현 증강 (K=4) + NAFNet 6-fold CV | 완료 — PSNR/SSIM 소폭 향상 (+0.24 dB) |
| 10단계 | NAFNet lr=1e-4 재실험 (lr 영향 검증) | 완료 — lr=1e-3 대비 오히려 성능 하락, 데이터 병목 재확인 |
| 11단계 | NAFNet + Edge Loss (Sobel, λ=0.5) 손실 함수 개선 | 완료 — CNR 소폭 향상 (+0.003), 데이터 병목이 한계 |
| 12단계 | NAFNet + Edge Loss + Frequency Loss 조합 | 완료 — CNR 오히려 하락, 손실 함수 개선 한계 확인 |
| 13단계 | AROI N2N 사전학습 → SBSDI D1 k-fold fine-tuning | 완료 — SSIM 0.6838 (전체 최고), PSNR +0.12 dB |

---

## 프로젝트 구조

```
OCT_AI/
├── data/
│   ├── Final_Publication_2013_SBSDI/   SBSDI D1(18쌍 noisy-clean) + D2(real 39세트) + D3(마우스)
│   ├── AROI/                           망막 OCT 레이어 세그멘테이션 (1,136장 주석)
│   ├── kaggle_RetinalOCTImages/        Kermany OCT2017 (84,484장)
│   ├── MedSegBench/                    낭성액 세그멘테이션 등 (npz)
│   └── data_description.md             데이터셋 상세 설명
├── scripts/
│   ├── 01_baseline/                    전통적 방법 베이스라인 (NLM, BM3D, SRAD)
│   ├── 02_synthetic_noise/             합성 스페클 노이즈 생성 파이프라인
│   ├── 03_sub2full/                    자가지도 학습 (N2N 프레임쌍)
│   ├── 03_supervised/                  지도학습 (합성 6,136쌍 → U-Net)
│   ├── 04_sr_test/                     사전 학습 SR 모델 테스트 (Real-ESRGAN)
│   ├── 05_finetune/                    합성 사전학습 → N2N fine-tuning
│   ├── 06_kfold/                       6-fold CV 지도학습 (U-Net + real clean GT)
│   ├── 07_dncnn/                       DnCNN 백본 6-fold CV
│   ├── 08_nafnet/                      NAFNet 백본 6-fold CV
│   ├── 09_augment/                     다중 노이즈 재실현 증강 + NAFNet
│   └── make_ppt.py                     발표 자료 자동 생성
├── weights/
│   ├── RealESRGAN_x2plus.pth           Real-ESRGAN x2 사전 학습 가중치 (64MB)
│   └── RealESRGAN_x4plus.pth           Real-ESRGAN x4 사전 학습 가중치 (64MB)
├── results/
│   ├── 01_baseline/                    베이스라인 지표 및 비교 이미지
│   ├── 02_synthetic_noise/             합성 학습 쌍 (6,136쌍) 및 검증 결과
│   ├── 03_sub2full/                    N2N 모델 체크포인트·지표·비교 이미지
│   ├── 03_supervised/                  지도학습 체크포인트·지표·비교 이미지
│   ├── 04_sr_test/                     SR 테스트 비교 이미지
│   ├── 05_finetune/                    fine-tune 체크포인트·지표·비교 이미지
│   ├── 06_kfold/                       6-fold CV 체크포인트·지표·비교 이미지
│   ├── 07_dncnn/                       DnCNN 체크포인트·지표·비교 이미지
│   ├── 08_nafnet/                      NAFNet 체크포인트·지표·비교 이미지
│   ├── 09_nafnet_aug/                  NAFNet+Aug 체크포인트·지표
│   ├── result_description.md          단계별 결과 정리
│   └── presentation.pptx              발표 자료 (13슬라이드)
├── Dockerfile                          서버 배포용 (ubuntu:22.04 + Python 3.12 + uv)
└── pyproject.toml                      uv 의존성 관리
```

---

## 환경 설정

```bash
# 의존성 설치 (uv 필요)
uv sync
```

주요 의존 패키지: `numpy`, `pillow`, `scikit-image`, `scipy`, `bm3d`, `pandas`, `matplotlib`, `tqdm`, `torch` (CUDA), `python-pptx`

---

## 1단계: 전통적 방법 베이스라인 측정 (완료)

딥러닝 모델 도입 전, 전통적 신호처리 방법으로 스페클 노이즈 제거 성능을 측정하여 비교 기준값을 확보한다.

- **데이터**: SBSDI D1 — `data/Final_Publication_2013_SBSDI/For synthetic experiments/` (18쌍, 450×900)
- **스크립트**: `scripts/01_baseline/`
- **결과**: `results/01_baseline/`

### 적용 방법

| 방법 | 핵심 원리 |
|------|----------|
| NLM | 이미지 전체에서 유사 패치를 찾아 가중 평균 |
| BM3D | 유사 블록을 3D 변환 후 임계화 |
| SRAD | 스페클 곱셈성 모델을 반영한 PDE 기반 이방성 확산 |

### 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR | 처리 속도 |
|------|-----------|------|-----|----------|
| NLM | 26.12 ± 2.01 | 0.492 ± 0.050 | 1.130 | ~0.5s/장 |
| BM3D | 27.00 ± 2.51 | 0.599 ± 0.066 | 1.134 | ~3.2s/장 |
| **SRAD** | **27.50 ± 1.98** | **0.652 ± 0.023** | **1.220** | ~5.7s/장 |

- SRAD가 PSNR·SSIM·CNR 모두 1위. 곱셈성 노이즈에 특화된 확산 방정식이 OCT에 적합함을 확인
- BM3D가 SSIM 기준 NLM 대비 큰 폭 향상 (0.492 → 0.599)
- **딥러닝 목표 기준값**: PSNR > 27.50 dB, SSIM > 0.652, CNR > 1.220

실행:
```bash
uv run python scripts/01_baseline/run_baseline.py
```

---

## 2단계: 합성 스페클 노이즈 생성 파이프라인 (완료)

pixel-aligned noisy-clean 쌍이 없는 대규모 망막 OCT 데이터셋(AROI, Kermany)을 활용한다. 이 데이터셋의 이미지는 real OCT 촬영본으로 고유한 스페클 노이즈를 포함하고 있으나, pixel-aligned clean 기준이 존재하지 않으므로 원본 이미지를 의사(pseudo) clean 기준으로 삼아 물리 기반 합성 스페클을 추가함으로써 지도학습용 noisy-clean 쌍을 생성한다. 이 방식으로 학습된 모델은 추가된 합성 스페클만을 제거 대상으로 학습하게 되어, 실제 OCT 스페클에 대한 일반화가 제한된다(3단계-B 도메인 갭 확인).

- **스크립트**: `scripts/02_synthetic_noise/`
- **결과**: `results/02_synthetic_noise/`

### 노이즈 모델

$$I = S \cdot N_s + N_a, \quad N_s \sim \text{Gamma}(L,\, 1/L), \quad N_a \sim \mathcal{N}(0,\, \sigma_a^2)$$

$N_s$는 곱셈성 스페클 노이즈 (mean=1, var=1/L), $N_a$는 가산성 가우시안 노이즈.

### 파라미터 캘리브레이션 (SBSDI D1 18쌍 기반)

실제 noisy-clean 쌍에서 $N_s = I / S$의 Gamma 모멘트 매칭으로 L을 추정했다.

| 파라미터 | 채택값 | 범위 | 설명 |
|---------|-------|------|------|
| L (looks 수) | **5.266** | [3.0, 6.2] | 높을수록 노이즈 약함 |
| sigma_a | **0.010** | 일정 | 가산 노이즈 표준편차 |
| KS 통계량 | 0.119 | — | 실제/합성 분포 일치도 (낮을수록 좋음) |

### 생성된 학습 데이터

| 데이터셋 | 쌍 수 | 해상도 (H×W) | 저장 경로 | 비고 |
|---------|------|------------|---------|------|
| AROI | 1,136 | 512×1024 | `results/02_synthetic_noise/AROI/clean/`, `…/noisy/` | 주석 완료 B-scan, 90도 회전 보정 |
| Kermany OCT2017 | 5,000 | 496×512 | `results/02_synthetic_noise/Kermany/clean/`, `…/noisy/` | train 서브셋 (랜덤 샘플링, seed=42) |
| **합계** | **6,136** | — | `results/02_synthetic_noise/metadata.csv` | clean/noisy 경로 및 이미지 크기 기록 |

실행:
```bash
# 파라미터 검증 (분포 플롯 생성)
uv run python scripts/02_synthetic_noise/validate_noise.py

# 전체 파이프라인 (캘리브레이션 + 합성)
uv run python scripts/02_synthetic_noise/run_synthetic.py --calibrate --max-kermany 5000
```

---

## 3단계-A: 자가지도 학습 — Noise2Noise 프레임쌍 (완료)

clean Ground Truth 없이 OCT 이미지만으로 학습하는 자가지도 방식.

- **학습 데이터**: SBSDI `For real experiments on Humans` (39세트, 450×450, clean 레퍼런스 없음)
- **평가 데이터**: SBSDI D1 18쌍 (전통 방법과 동일)
- **스크립트**: `scripts/03_sub2full/`
- **결과**: `results/03_sub2full/`

### 방법: N2N 프레임쌍 (Noise2Noise)

각 세트의 1~4.tif는 test.tif의 **인접 공간 슬라이스**(3D 볼륨 내 이웃 B-scan)다. 완전한 동일 위치 반복 스캔은 아니지만, 인접 슬라이스는 조직 구조가 매우 유사하고 스페클 패턴은 독립적이므로 근사 Noise2Noise 조건으로 사용한다. (동일 위치 반복 촬영 평균이 GT로 사용되는 D1과 구별됨 — D2 인접 슬라이스 diff/frame std 비율 1.07 vs D1 동일 위치 0.77)

- 학습 쌍: 39세트 × 12 순서쌍(i≠j) = **468쌍**
- 아키텍처: 경량 U-Net (~1.95M params, 인코더 3단계 + 보틀넥 + 디코더 3단계)
- 손실 함수: L1 Loss / 옵티마이저: Adam (lr=1e-4, CosineAnnealing)
- 에포크: 500 / 배치: 4 / 패치: 128×128
- 학습 시간: 약 221분 (RTX A4000)
- **loss 수렴**: 0.105 → 0.099 (500 에포크, 개선 폭 작음 — 학습 데이터 39세트 절대 부족)

### 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| NLM | 26.12 ± 2.01 | 0.492 ± 0.050 | 1.130 ± 0.119 |
| BM3D | 27.00 ± 2.51 | 0.599 ± 0.066 | 1.134 ± 0.119 |
| SRAD | 27.50 ± 1.98 | 0.652 ± 0.023 | 1.220 ± 0.121 |
| **N2N (Sub2Full 방식)** | **26.84 ± 2.18** | **0.681 ± 0.033** | **1.148 ± 0.133** |

- **SSIM 0.681이 SRAD 0.652를 초과** — clean GT 없이도 구조 보존 품질에서 전통 방법 1위를 넘김
- PSNR(26.84)은 SRAD(27.50)보다 0.66 dB 낮고, BM3D(27.00)에 근접
- CNR(1.148)은 SRAD(1.220)보다 낮음 — 경계 대비 보존에서 아직 열세
- loss 수렴 제한: 0.105 → 0.099, 개선 폭이 작음 (학습 데이터 39세트 절대 부족)

실행:
```bash
# 학습 + 평가 (기본: CPU 4스레드 제한, GPU 자동 선택)
uv run python scripts/03_sub2full/run_sub2full.py

# CPU 스레드 조절 (다른 작업 병행 시)
uv run python scripts/03_sub2full/run_sub2full.py --cpu-threads 2

# 저장된 체크포인트로 평가만 실행
uv run python scripts/03_sub2full/run_sub2full.py --eval-only
```

---

## 3단계-B: 지도학습 — 합성 데이터 (완료, 도메인 갭 확인)

합성 noisy-clean 6,136쌍으로 U-Net을 학습하고 SBSDI D1로 평가.

- **학습 데이터**: `results/02_synthetic_noise/metadata.csv` (AROI 1,136 + Kermany 5,000)
- **스크립트**: `scripts/03_supervised/`
- **결과**: `results/03_supervised/`

### 모델 및 학습 설정

| 항목 | 설정 |
|------|------|
| 아키텍처 | U-Net (~1.95M params, 03_sub2full/model.py 공유) |
| 손실 함수 | L1 Loss |
| 옵티마이저 | Adam (lr=2e-4, CosineAnnealing) |
| 에포크 | 100 / 배치 8 / 패치 128×128 |
| 학습 시간 | 약 305분 (RTX A4000) |
| 최적 loss | 0.021621 |

### 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 | 0.652 | 1.220 |
| **지도학습 (합성)** | **22.43 ± 0.94** | **0.333 ± 0.049** | **0.902 ± 0.100** |

세 지표 모두 SRAD 미달. 원인: **합성 노이즈(Gamma+Gaussian)와 실제 OCT 스페클 간 도메인 갭**. 모델이 시뮬레이션 노이즈 제거는 학습했으나 실제 SBSDI D1 노이즈에는 일반화하지 못함.

| 구분 | 학습 데이터 | 평가 데이터 |
|------|-----------|-----------|
| 노이즈 유형 | 합성 Gamma+Gaussian | 실제 OCT 스페클 |
| 이미지 소스 | AROI / Kermany | SBSDI D1 |
| Clean 기준 | 수학적 모델 생성 | 다중 프레임 평균 |

**loss 수렴**: 에포크 1(0.0449) → 10(0.0235) → 50(0.0220) → 100(0.0217, best 0.02162). 에포크 10 이후 수렴 포화 — 합성 노이즈 분포를 빠르게 학습했으나 실제 OCT 노이즈에 대한 일반화 능력을 획득하지 못함.

실행:
```bash
uv run python scripts/03_supervised/run_supervised.py
uv run python scripts/03_supervised/run_supervised.py --eval-only
```

---

## 4단계: 사전 학습 SR 모델 적용 (완료)

스페클 노이즈 제거와 Super-Resolution을 동시에 처리하는 사전 학습 모델을 적용한다.

Real-ESRGAN은 실제 열화(노이즈 + 저해상도)를 동시에 복원하도록 학습된 blind SR 모델이다.

- **가중치**: `weights/RealESRGAN_x2plus.pth`, `weights/RealESRGAN_x4plus.pth` (각 64MB)
- **스크립트**: `scripts/04_sr_test/test_sr.py` (단일 이미지), `scripts/04_sr_test/eval_sr_full.py` (18쌍)
- **결과**: `results/04_sr_test/`

### 결과 (SBSDI D1, 18쌍 평균, 원본 스케일 다운스케일 후 비교)

| 방법 | PSNR (dB) | SSIM | CNR | 처리 시간 |
|------|-----------|------|-----|---------|
| SRAD (베이스라인) | 27.50 ± 1.98 | 0.652 ± 0.023 | 1.220 ± 0.121 | ~5.7s/장 |
| Real-ESRGAN x2 | 27.30 ± 2.35 | **0.674 ± 0.034** | 1.121 ± 0.116 | 0.62s/장 |
| **Real-ESRGAN x4** | **27.57 ± 2.40** | **0.673 ± 0.034** | 1.166 ± 0.117 | 1.42s/장 |

- **x4**: PSNR·SSIM 모두 SRAD 초과. fine-tuning 없이 노이즈 제거 + SR 동시 달성
- **x2**: SSIM만 초과, PSNR은 SRAD보다 0.20 dB 낮음
- **CNR**: x2/x4 모두 SRAD 미달 — blind SR 모델이 전역 대비보다 고주파 디테일 복원에 특화
- 처리 속도 0.62~1.42s/장으로 SRAD(5.7s) 대비 4~9배 빠름

실행:
```bash
uv run python scripts/04_sr_test/test_sr.py       # 단일 이미지
uv run python scripts/04_sr_test/eval_sr_full.py  # 18쌍 전체 평가
```

### 병목 분석 (4단계 시점 진단 → 6~8단계 실험으로 검증)

| 가설 | 진단 시점 판단 | 실험 후 결론 |
|------|-------------|------------|
| 학습 루프에 clean GT 없음 | 근본 원인 | **확인** — k-fold CV로 해소 후 SRAD 초과 |
| real clean GT 18쌍 미활용 | 핵심 병목 | **확인** — 6단계-B에서 PSNR +0.98 dB |
| 모델 capacity 부족 | 의심 요인 | **기각** — DnCNN 667K = U-Net 1.95M = NAFNet 17M |
| 지도학습 도메인 갭 | 확인된 문제 | **확인** — 합성 지도학습 최하위(PSNR 22.43) |

방향 A(k-fold 지도학습)와 방향 B(백본 교체)는 6~8단계에서 모두 완료. 상세 결과는 종합 결과 비교 및 향후 작업 방향 참조.

---

## 5단계: Pre-train → Fine-tune (완료)

합성 지도학습 사전학습 가중치로 초기화 후 SBSDI real N2N 쌍으로 fine-tuning. 동시에 Loss를 L1 + SSIM 조합으로 개선.

- **사전학습 가중치**: `results/03_supervised/checkpoints/best.pth` (합성 6,136쌍, loss 0.021621)
- **Fine-tune 데이터**: SBSDI real 39세트 × 12 = 468쌍 (N2N 방식)
- **스크립트**: `scripts/05_finetune/run_finetune.py`
- **결과**: `results/05_finetune/`

### 학습 설정

| 항목 | 설정 |
|------|------|
| 초기 가중치 | 합성 지도학습 best.pth |
| 손실 함수 | L1 + 0.1 × (1 − SSIM) |
| LR | 1e-5 (사전학습 2e-4 대비 20배 낮춤) |
| 에포크 | 200 / 배치 16 / 패치 128×128 |
| 학습 시간 | 약 35분 (RTX A4000) |
| Best loss | 0.184206 |

### Loss 수렴

| 에포크 | loss |
|--------|------|
| 1 | 0.205499 |
| 20 | 0.184966 |
| 60 | 0.184679 |
| 100 | 0.184483 |
| 200 | 0.184475 |

에포크 20 이후 loss가 0.1844~0.1850 구간에서 포화. 합성 도메인 편향은 빠르게 교정되었으나 N2N 노이즈 타겟의 고유 분산(noise floor) 때문에 추가 개선이 제한됨.

### 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 ± 1.98 | 0.652 ± 0.023 | 1.220 ± 0.121 |
| N2N from scratch | 26.84 ± 2.18 | **0.681 ± 0.033** | 1.148 ± 0.133 |
| **Pre-train + Fine-tune** | **27.46 ± 2.55** | 0.6795 ± 0.037 | **1.169 ± 0.120** |

- **PSNR**: N2N 대비 +0.62 dB 향상 (26.84 → 27.46), SRAD까지 0.04 dB 차이
- **SSIM**: SRAD(0.652) 초과 유지, N2N(0.681)보다 0.001 낮음
- **CNR**: N2N 대비 +0.021 향상 (1.148 → 1.169)
- **주요 병목**: N2N 노이즈 타겟의 고유 분산 — loss floor가 ~0.184로 형성되어 추가 수렴 불가

실행:
```bash
uv run python scripts/05_finetune/run_finetune.py
uv run python scripts/05_finetune/run_finetune.py --eval-only
```

---

## 6단계: 6-fold CV 지도학습 — U-Net + real clean GT (완료)

SBSDI D1 18쌍을 k=6 fold로 나눠 real clean GT로 지도학습. 방향 A(clean GT 활용) 효과 단독 측정.

- **스크립트**: `scripts/06_kfold/run_kfold.py`
- **결과**: `results/06_kfold/`

| 설정 | 6단계-A | 6단계-B |
|------|---------|---------|
| 초기 가중치 | from_scratch | from_scratch |
| 배치 / 에포크 | 16 / 150 | **64 / 500** |
| Early stopping | 없음 | **patience=30** |
| 학습 시간 | 175분 | **47분** |
| PSNR | 27.21 | **28.19** |
| SSIM | 0.6822 | 0.6814 |

- **6단계-B 핵심**: AI 방법 최초로 SRAD(27.50) 초과. early stopping이 과적합 방지와 수렴 동시 해결
- loss floor 0.048 달성 (N2N 0.184, 합성 지도학습 0.022 대비 real clean GT 효과 확인)

실행:
```bash
uv run python scripts/06_kfold/run_kfold.py
```

---

## 7단계: DnCNN 백본 교체 (완료)

동일한 6-fold CV 프레임워크에서 U-Net → DnCNN-B(depth=20, ch=64) 교체. 방향 B(아키텍처) 효과 측정.

- **스크립트**: `scripts/07_dncnn/run_dncnn.py`
- **결과**: `results/07_dncnn/`

| 항목 | 값 |
|------|---|
| 파라미터 수 | 667,073 (U-Net 1.95M 대비 1/3) |
| 초기 가중치 | from scratch |
| 학습 시간 | 355분 |
| PSNR | 28.17 ± 2.47 |
| SSIM | 0.6732 ± 0.032 |

- **핵심 발견**: 667K DnCNN ≈ 1.95M U-Net — 모델 크기가 병목이 아님

실행:
```bash
uv run python scripts/07_dncnn/run_dncnn.py
```

---

## 8단계: NAFNet 백본 교체 (완료)

동일한 6-fold CV 프레임워크에서 NAFNet-width32(~17M params) 적용. SimpleGate + SCA + LayerNorm 구조.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과**: `results/08_nafnet/`

| 항목 | 값 |
|------|---|
| 파라미터 수 | 17,112,673 |
| 초기 가중치 | from scratch |
| 학습 시간 | 123분 |
| PSNR | 28.11 ± 2.26 |
| SSIM | 0.6743 ± 0.029 |
| CNR | **1.183** (3개 아키텍처 중 최고) |

- **핵심 발견**: 17M NAFNet ≈ 1.95M U-Net ≈ 667K DnCNN — 데이터 병목 3번 연속 확인
- **NAFNet 특이 현상 (loss-PSNR 분리)**: lr=1e-3(U-Net 5e-5 대비 20배)으로 인해 val_psnr이 epoch 1~9에서 최고점 도달 후 정체·하락. train_loss는 계속 감소하나 일반화 성능은 개선되지 않는 과적합 패턴 발생. 17M 파라미터가 fold당 15쌍을 조기 암기하는 것이 원인. best.pth는 초반 epoch 가중치로 저장됨. lr 인하 또는 데이터 확보로 해소 가능

실행:
```bash
uv run python scripts/08_nafnet/run_nafnet.py
# 특정 fold부터 재시작
uv run python scripts/08_nafnet/run_nafnet.py --start-fold 2
```

---

## 9단계: 다중 노이즈 재실현 증강 + NAFNet (완료)

18개 clean GT에 노이즈 모델(2단계 방식, L=5.266)로 K=4개 합성 noisy를 추가 생성. 학습 데이터 5배 증가(15쌍 → 75쌍/fold). 평가는 real noisy로만 수행.

- **스크립트**: `scripts/09_augment/run_nafnet_aug.py`
- **증강 데이터**: `data/Final_Publication_2013_SBSDI/augmented_noisy/`
- **결과**: `results/09_nafnet_aug/`

| 항목 | 값 |
|------|---|
| 학습 쌍/fold | 75 (real 15 + aug 60) |
| 배치 크기 | 48 |
| PSNR | **28.35 ± 2.56** (+0.24 vs NAFNet) |
| SSIM | **0.6832 ± 0.032** (+0.009 vs NAFNet) |
| CNR | 1.168 ± 0.121 (SRAD 미달 지속) |

실행:
```bash
uv run python scripts/09_augment/run_nafnet_aug.py --batch-size 48
# 특정 fold만 재실행
uv run python scripts/09_augment/run_nafnet_aug.py --start-fold 5 --end-fold 5
```

---

## 10단계: NAFNet lr=1e-4 재실험 — lr 영향 검증 (완료)

8단계(lr=1e-3)에서 loss 감소에도 val_psnr이 초반에 정점 후 정체·하락하는 현상이 확인됐다. lr을 1e-4로 낮춰 학습이 안정화되는지 검증.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과**: `results/10_nafnet_lr1e4/`

| 항목 | 8단계 (lr=1e-3) | 10단계 (lr=1e-4) |
|------|---------------|----------------|
| PSNR | **28.11 ± 2.26** | 27.95 ± 2.11 |
| SSIM | **0.6743 ± 0.029** | 0.6666 ± 0.026 |
| CNR | 1.183 ± 0.120 | **1.191 ± 0.116** |
| Early stop epoch | 32~46 | 40~61 |
| Epoch 1 val PSNR | ~28.27 (이미 높음) | ~21~25 (낮음) |

**결과**: lr=1e-4가 오히려 성능 하락 (PSNR -0.16 dB). lr=1e-3은 초반 몇 에포크에서 빠르게 좋은 수렴점을 찾고 early stopping이 그 시점을 포착한다. lr=1e-4는 patience=30 내에 그 수렴점까지 도달하지 못해 최종 성능이 낮다. 하이퍼파라미터 조정으로는 데이터 병목을 해소할 수 없음을 재확인.

실행:
```bash
uv run python scripts/08_nafnet/run_nafnet.py --lr 1e-4 --batch-size 48 --results-dir results/10_nafnet_lr1e4
```

---

## 11단계: NAFNet + Edge Loss (손실 함수 개선) (완료)

CNR SRAD 미달 문제를 해소하기 위해 Sobel 기반 Edge Loss를 추가. 6회 시도를 통해 안정적인 정규화 방식을 확립했다.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과**: `results/11_nafnet_edge/`

### Edge Loss 정규화 시도 과정

| 시도 | 정규화 | λ | 결과 |
|------|-------|---|------|
| 1 | 없음 | 0.1 | Fold 4~6 발산 — Sobel 스케일 과대 |
| 2 | 없음 | 0.01 | 안정, 효과 없음 — 기여 0.8% 미만 |
| 3 | per-batch mean 나눔 | 0.1 | 전 fold 발산 — mean으로 나누면 값 20배 증폭 |
| 4 | SOBEL_MAX(11.314) 나눔 | 0.1 | 안정, 효과 없음 — 기여 여전히 작음 |
| 5 | SOBEL_MAX 나눔 | 1.0 | Fold 2, 4 발산 — fold마다 edge 스케일 달라 불안정 |
| **6** | **SOBEL_MAX + clamp(0,1)** | **0.5** | **전 fold 안정, CNR +0.003** |

### 최종 손실 함수

```
Loss = L1 + 0.1×(1-SSIM) + 0.5×EdgeLoss
EdgeLoss = L1(clamp(Sobel(pred)/11.314, 0,1), clamp(Sobel(clean)/11.314, 0,1))
```

### 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 ± 1.98 | 0.652 ± 0.023 | **1.220 ± 0.121** |
| 8단계 NAFNet (기준) | **28.11 ± 2.26** | **0.6743 ± 0.029** | 1.183 ± 0.120 |
| **11단계 NAFNet + EdgeLoss** | 28.06 ± 2.08 | 0.6699 ± 0.027 | **1.186 ± 0.116** |

- CNR 1.183 → 1.186 (+0.003): 개선 방향은 맞으나 ±0.116 std 대비 통계적 유의성 없음
- PSNR·SSIM은 소폭 하락 — edge 보존 강조가 픽셀 정확도를 미세하게 희생
- **결론**: 손실 함수 개선의 방향은 확인했으나 데이터 18쌍 병목이 CNR 향상의 상한을 결정. 데이터 확보 없이는 loss 튜닝 효과가 noise 수준에 그침

실행:
```bash
uv run python scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --seed 42 --results-dir results/11_nafnet_edge
```

---

## 12단계: NAFNet + Edge Loss + Frequency Loss (완료)

11단계 Edge Loss에 FFT 기반 Frequency Loss를 추가해 고주파 구조 보존 효과를 검증.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과**: `results/12_nafnet_freq/`

```
Loss = L1 + 0.1×(1-SSIM) + 0.5×EdgeLoss + 0.1×FreqLoss
FreqLoss = L1(log1p(|rfft2(pred)|), log1p(|rfft2(clean)|))
```

| 방법 | PSNR | SSIM | CNR |
|------|------|------|-----|
| 8단계 baseline | 28.11 ± 2.26 | 0.6743 ± 0.029 | 1.183 ± 0.120 |
| 11단계 +Edge | 28.06 ± 2.08 | 0.6699 ± 0.027 | **1.186 ± 0.116** |
| **12단계 +Edge+Freq** | 28.05 ± 2.03 | 0.6683 ± 0.028 | 1.182 ± 0.120 |

**결과**: Frequency Loss 추가 시 CNR이 1.186 → 1.182로 오히려 하락. Edge Loss가 경계 spatial gradient를 직접 최적화하는 반면, Frequency Loss는 전체 스펙트럼 차이를 페널티해 CNR과 무관한 성분에 gradient를 분산시켜 두 항이 충돌. **손실 함수 개선의 최선은 Edge Loss 단독(11단계)이며, 추가 조합은 효과 없음.**

실행:
```bash
uv run python scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --lambda-freq 0.1 --seed 42 --results-dir results/12_nafnet_freq
```

---

## 13단계: AROI N2N 사전학습 → SBSDI D1 k-fold fine-tuning (완료)

보유 데이터(AROI 3,072장)를 활용해 인접 B-scan N2N으로 사전학습 후 SBSDI D1 18쌍으로 fine-tuning.

### 사전학습 (AROI N2N)

- **스크립트**: `scripts/13_aroi_n2n/run_pretrain.py`
- **결과**: `results/13_aroi_n2n/pretrain/best.pth`
- Train: patient 1~20, 인접 쌍 2,540개 / Val: patient 21~24, 508쌍
- 데이터셋: lazy loading + random crop (4,096 samples/epoch, 85 steps/epoch)
- Early stop epoch 93, best val N2N loss: 0.04305
- D1 PSNR 모니터링 최고: **27.086** (epoch 50) — 3단계-A SBSDI D2 N2N(26.84) 초과

### fine-tuning (SBSDI D1 k-fold)

- **스크립트**: `scripts/08_nafnet/run_nafnet.py --pretrain-ckpt results/13_aroi_n2n/pretrain/best.pth`
- **결과**: `results/13_aroi_n2n/finetune/`
- lr=1e-4, λ_edge=0.5, seed=42, 모든 fold epoch 31~34 수렴

| 방법 | PSNR | SSIM | CNR |
|------|------|------|-----|
| 8단계 NAFNet (scratch) | 28.11 ± 2.26 | 0.6743 ± 0.029 | **1.183 ± 0.120** |
| 9단계 NAFNet+Aug | **28.35 ± 2.56** | 0.6832 ± 0.032 | 1.168 ± 0.121 |
| **13단계 AROI pretrain→finetune** | 28.23 ± 2.49 | **0.6838 ± 0.029** | 1.177 ± 0.121 |

- **SSIM 0.6838**: 전체 방법 중 최고
- **PSNR +0.12 dB**: scratch(28.11) 대비 향상
- Epoch 1 val PSNR이 28~30 수준으로 시작 — 사전학습 효과 확인
- CNR은 scratch(1.183) 대비 소폭 하락 — AROI N2N이 smooth 출력을 학습해 경계 대비 미세 희생

실행:
```bash
# 1단계: 사전학습
uv run python scripts/13_aroi_n2n/run_pretrain.py

# 2단계: fine-tuning
uv run python scripts/08_nafnet/run_nafnet.py \
  --pretrain-ckpt results/13_aroi_n2n/pretrain/best.pth \
  --lr 1e-4 --lambda-edge 0.5 --seed 42 \
  --results-dir results/13_aroi_n2n/finetune
```

---

## 종합 결과 비교 (SBSDI D1, 18쌍 평균)

| 방법 | 유형 | PSNR (dB) | SSIM | CNR | SRAD 대비 |
|------|------|-----------|------|-----|----------|
| NLM | 전통 | 26.12 ± 2.01 | 0.492 ± 0.050 | 1.130 ± 0.119 | 전 지표 미달 |
| BM3D | 전통 | 27.00 ± 2.51 | 0.599 ± 0.066 | 1.134 ± 0.119 | 전 지표 미달 |
| **SRAD** | 전통 | 27.50 ± 1.98 | 0.652 ± 0.023 | **1.220 ± 0.121** | 기준값 |
| N2N 프레임쌍 | 자가지도 | 26.84 ± 2.18 | **0.681 ± 0.033** | 1.148 ± 0.133 | SSIM만 초과 |
| 지도학습 (합성) | 지도 | 22.43 ± 0.94 | 0.333 ± 0.049 | 0.902 ± 0.100 | 전 지표 미달 (도메인 갭) |
| Pre-train + Fine-tune | 전이학습 | 27.46 ± 2.55 | 0.6795 ± 0.037 | 1.169 ± 0.120 | PSNR·SSIM 초과 |
| **6-fold CV U-Net (batch=16)** | real clean GT | 27.21 ± 2.41 | 0.6822 ± 0.032 | 1.164 ± 0.113 | SSIM 초과 |
| **6-fold CV U-Net (batch=64, ES)** | real clean GT | **28.19 ± 2.56** | **0.6814 ± 0.031** | 1.169 ± 0.121 | PSNR+SSIM 초과 |
| 6-fold CV DnCNN (depth=20, ch=64) | real clean GT | 28.17 ± 2.47 | 0.6732 ± 0.032 | 1.167 ± 0.127 | U-Net과 동등 |
| 6-fold CV NAFNet (width=32) | real clean GT | 28.11 ± 2.26 | 0.6743 ± 0.029 | **1.183 ± 0.120** | CNR 최고, 분산 최소 |
| **NAFNet + Aug (K=4)** | real+aug clean GT | **28.35 ± 2.56** | **0.6832 ± 0.032** | 1.168 ± 0.121 | PSNR/SSIM 현재 최고 |
| NAFNet lr=1e-4 (10단계) | real clean GT | 27.95 ± 2.11 | 0.6666 ± 0.026 | 1.191 ± 0.116 | lr 인하로 오히려 하락 |
| NAFNet + EdgeLoss λ=0.5 (11단계) | real clean GT | 28.06 ± 2.08 | 0.6699 ± 0.027 | **1.186 ± 0.116** | CNR +0.003, 손실함수 최고 |
| NAFNet + Edge+Freq (12단계) | real clean GT | 28.05 ± 2.03 | 0.6683 ± 0.028 | 1.182 ± 0.120 | Freq 추가로 오히려 하락 |
| **AROI pretrain→D1 finetune (13단계)** | N2N pretrain+real GT | 28.23 ± 2.49 | **0.6838 ± 0.029** | 1.177 ± 0.121 | SSIM 전체 최고, PSNR +0.12 |
| Real-ESRGAN x2 | SR(blind) | 27.30 ± 2.35 | 0.674 ± 0.034 | 1.121 ± 0.116 | SSIM만 초과 |
| Real-ESRGAN x4 | SR(blind) | 27.57 ± 2.40 | 0.673 ± 0.034 | 1.166 ± 0.117 | PSNR+SSIM 초과 |

> Real-ESRGAN 지표는 SR 출력을 원본 해상도(450×900)로 다운스케일 후 clean reference와 비교한 값.

**핵심 발견**
- **SSIM 최고 (현재)**: **AROI pretrain→D1 finetune — SSIM 0.6838** (13단계)
- **PSNR 최고 (현재)**: **NAFNet+Aug — PSNR 28.35** (9단계)
- PSNR 최고 (증강 전): 6-fold CV U-Net (28.19 dB) — AI 방법 최초로 SRAD 초과
- CNR 최고: **6-fold CV NAFNet (1.183)** — 어떤 방법도 SRAD(1.220) 미달
- CNR은 전통 방법 SRAD(1.220)를 어떤 AI 방법도 아직 초과하지 못함
- 667K(DnCNN) ≈ 1.95M(U-Net) ≈ 17M(NAFNet) ≈ NAFNet+Aug(K=4 증강) — 데이터 병목이 근본 원인
- **lr 조정 무효 (10단계)**: NAFNet lr=1e-4는 lr=1e-3 대비 PSNR -0.16 dB. lr=1e-3의 빠른 초기 수렴이 데이터 부족 환경에서 오히려 유리하게 작용. 하이퍼파라미터 튜닝의 한계 재확인
- **Edge Loss 효과 미미 (11단계)**: SOBEL_MAX+clamp 정규화로 안정적 학습 확립. CNR 1.183 → 1.186 (+0.003)이나 std(±0.116) 대비 통계적 유의성 없음
- **Frequency Loss 추가 역효과 (12단계)**: Edge+Freq 조합에서 CNR 1.186 → 1.182로 하락. 두 loss 항이 gradient 방향 충돌. **손실 함수 개선 실험 종료**

**결론**

real clean GT k-fold CV + early stopping으로 SRAD 초과 가능. 아키텍처 변경(DnCNN/U-Net/NAFNet), 데이터 증강(K=4), lr 조정, 손실 함수 개선(Edge+Freq) 모두 데이터 18쌍 병목 앞에서 근본 한계를 해소하지 못함. 다음 단계는 데이터 측면(Duke AMD SD-OCT, 외부 clean GT)으로 전환 필요.

---

## 향후 작업 방향

10단계까지의 실험으로 **아키텍처 교체·lr 조정·데이터 증강 모두 근본 한계를 해소하지 못함**이 확인됐다. 미해결 과제는 CNR(SRAD 미달)과 데이터 병목(18쌍)이며, 각각에 대한 접근이 필요하다.

| 우선순위 | 방향 | 비용 | 기대 효과 |
|---------|------|------|---------|
| 1 | **손실 함수 개선** (Edge Loss 완료, Frequency loss 미시도) | 낮음 | Edge Loss CNR +0.003 확인 — Frequency loss 추가 시도 가능하나 데이터 병목이 상한 |
| 2 | **TTA (Test-Time Augmentation)** | 낮음 | 추가 학습 없이 fold 간 분산(std 2.56) 감소 |
| 3 | **Duke AMD SD-OCT + N2N 자기지도** | 중간 | 38,400 B-scans로 다양성 확보, 직접 다운로드 가능 |
| 4 | **외부 clean GT 데이터셋 접근 재시도** | 중간 | PKU37(37쌍), Sub2Full vis-OCT 저자 컨택 |
| 5 | **Diffusion 기반 방법 (GARD 등)** | 높음 | 최고 복원 품질, OCT 스페클 Gamma 분포 직접 모델링 |
| 6 | **치주 OCT 직접 수집** | 매우 높음 | 실제 타겟 도메인 — 교수님과 촬영 프로토콜 협의 필요 |
| 장기 | **Joint Denoising + SR** | 높음 | 스페클 제거 완성 후 Real-ESRGAN 파이프라인 결합 |

> 상세 분석 및 손실 함수 구현 계획은 `results/result_description.md` 향후 작업 방향 섹션 참조.

---

## 환경 및 알려진 이슈

### 실험 환경

- GPU: NVIDIA RTX A4000 × 2 (GPU 0: CUDA 연산 전용, GPU 1: 디스플레이 출력)
- CUDA half precision으로 추론 (Real-ESRGAN tile=400)
- Task Manager에서 CUDA 연산이 "3D" 엔진으로 표시되는 것은 Windows WDDM 정상 동작

### basicsr / torchvision 호환성 패치

최신 torchvision에서 `torchvision.transforms.functional_tensor` 모듈이 제거됨. basicsr 설치 후 아래 파일을 수동 패치해야 한다.

```
.venv/Lib/site-packages/basicsr/data/degradations.py  line 8
```

```python
# 수정 전 (오류)
from torchvision.transforms.functional_tensor import rgb_to_grayscale
# 수정 후
from torchvision.transforms.functional import rgb_to_grayscale
```

### U-Net 모델 공유

3단계-A(N2N)와 3단계-B(지도학습)는 동일한 U-Net 아키텍처(`scripts/03_sub2full/model.py`)를 공유한다. 3단계-B의 `run_supervised.py`는 `sys.path`에 `03_sub2full`을 추가하여 모델을 임포트한다.

---

## 자료 조사

### OCT란

**OCT(광간섭단층촬영, Optical Coherence Tomography)**는 근적외선 파장의 빛을 이용해 생체 조직 내부의 단층 구조를 마이크로미터($\mu m$) 단위의 초고해상도로 획득하는 비침습적 광학 영상 기술이다. 마이켈슨 간섭계 구조를 활용해 빛의 후방 산란 및 간섭 현상으로 단층 영상을 재구성한다.

- **A-scan**: 깊이 방향 단일 데이터(1차원)
- **B-scan**: 연속된 A-scan으로 구성된 단면 영상(2차원)
- **C-scan**: 3D 볼륨을 일정 깊이로 절단한 수평면 영상(2차원)
- **3D OCT volume**: 연속된 B-scan을 쌓아 구성한 공간 데이터(3차원)
- **축 방향 해상도**: 광원 대역폭에 의해 결정 ($\Delta z \propto \lambda_0^2 / \Delta\lambda$), 임상 기준 5~10 µm
- **측 방향 해상도**: 대물렌즈 개구수(NA)에 의해 결정

현대 임상 표준은 FFT 기반 **Fourier-Domain OCT(FD-OCT)**이며, 치과/연조직 검사에는 **SS-OCT(Swept-Source OCT)** 방식이 투과 깊이 면에서 유리하다.

관련 링크: [visionsystem.kr](https://www.visionsystem.kr/technical-info?tpf=board/view&board_code=1&code=639) | [서울아산병원](https://www.amc.seoul.kr/asan/healthinfo/management/managementDetail.do?managementId=417)

---

### 오픈 데이터 탐색

#### 치주·치과 영역

치주 OCT 공개 데이터는 대부분 비공개(In-house)로, 아래 방법론 및 인접 도메인 데이터로 대체한다.

- **Transformer 기반 Dental OCT 탐지** — [MDPI 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10671998/): ViT + Attention Gate로 치아 결손 탐지
- **Open-Set Dental Detector** — [GitHub FD-SOS](https://github.com/xmed-lab/FD-SOS): MICCAI 2024, 치조골 세그멘테이션 프레임워크
- **ENPAT** — [PeerJ 2024](https://peerj.com/articles/cs-3229/): YOLOv8 기반 치주 질환 스크리닝 (구강 내 임상 이미지)

#### 망막 OCT (보유 데이터셋)

현재 보유 중인 데이터셋 구조 및 상세 설명 → [data/data_description.md](data/data_description.md)

| 데이터셋 | 이미지 수 | 태스크 | noisy-clean 쌍 |
|---------|---------|--------|---------------|
| Kermany OCT2017 | 84,484 | 분류 (CNV/DME/DRUSEN/NORMAL) | 없음 |
| AROI | 3,072 (주석 1,136) | 레이어 세그멘테이션 | 없음 |
| MedSegBench / cystoidfluid | 1,006 | 낭성액 세그멘테이션 | 없음 |
| SBSDI (Fang TMI 2013) | D1 synthetic 18쌍 / D2 real 39세트 / D3 마우스 | 디노이징 + SR | D1만 있음 (average.tif) |

SBSDI 패키지는 D1~D3를 모두 포함하나, clean GT(다중 프레임 평균)는 D1(18쌍)에만 존재한다. D2는 인간 망막 실제 노이즈 39세트, D3는 마우스 망막 실제 노이즈로 GT 없음.

**average.tif 생성 방법**: 동일 위치를 N≈40회 반복 스캔 → 픽셀별 평균. `test.tif`는 그 중 단일 프레임, `1~4.tif`는 인접 공간 슬라이스(다른 위치). 프레임 수는 Dictionary 학습용 HH/LL 쌍 10개의 배경 노이즈 std 비율(≈6.3)의 제곱으로 추정.

**보유 데이터 clean GT 추가 생성 가능 여부**: 불가. AROI·Kermany·MedSegBench·SBSDI D2/D3 모두 동일 위치 반복 스캔이 아닌 순차 슬라이스 또는 단일 이미지로, 평균을 통한 GT 생성 요건을 충족하지 않는다 (인접 프레임 간 diff std / frame std 비율: 0.86~1.31, 기준값인 SBSDI D1 동일 위치 반복 0.77 미달).

외부 공개 데이터셋 탐색 결과:

| 데이터셋 | 규모 | clean GT | 상태 |
|---------|------|---------|------|
| PKU37 | 37쌍 | 있음 | 알리바바 클라우드 계정 요구 — 접근 실패 |
| Sub2Full vis-OCT | 미공개 | 있음 | 데이터 비공개 |
| RETOUCH | 112 볼륨 | 없음 | Grand Challenge 계정 요구 — 접근 실패 |
| ODTiD | 242장 | 없음 | OPENICPSR 계정 요구 — 접근 실패 |
| Duke AMD SD-OCT (RPEDC 2013) | 384명, 38,400 B-scans | 없음 | 직접 다운로드 가능 — 합성 노이즈 추가 활용만 가능 |
| Duke Srinivasan BOE 2014 | 45명 | 없음 | 직접 다운로드 가능 — 소규모, 분류용 |

---

### 스페클 노이즈 제거 방식

OCT 스페클 노이즈는 레이저 가간섭성으로 인한 물리적 현상으로, 곱셈성 노이즈 모델을 따른다.

$$I(x,y) = S(x,y) \cdot N_s(x,y) + N_a(x,y)$$

#### 전통적 방법 (AI 미사용)

| 계열 | 방법 | 특징 |
|------|------|------|
| 하드웨어 | 프레임 에버리징, 컴파운딩 | 임상 Gold Standard. 촬영 시간 증가 |
| 공간 도메인 | Lee/Frost 필터, **SRAD** | SRAD는 곱셈성 특성 반영, 경계 보존 우수 |
| 비국소/변환 | **NLM**, Wavelet 임계화 | NLM은 반복 레이어 구조 보존에 효과적 |
| 블록 | **BM3D** | 유사 블록 3D 변환, 성능 우수하나 느림 |

#### AI 기반 방법

| 계열 | 대표 방법 | 특징 |
|------|---------|------|
| 지도학습 | DnCNN, U-Net, Restormer | clean GT 필요. L1/L2 + Perceptual Loss 조합 |
| 자가지도 | Noise2Noise, Noise2Void, **Sub2Full** | clean 데이터 불필요. Sub2Full은 OCT 전용 |
| 비지도 | CycleGAN | 쌍 없는 도메인 변환 |
| 생성형 | **Diffusion PnP** | 최고 복원 품질, 느린 추론 |

#### OCT 스페클 제거 최신 연구 (2022~2025)

**자가지도 / 비지도 계열**

| 논문 | 연도 | 핵심 |
|------|------|------|
| [Sub2Full](https://arxiv.org/abs/2401.10128) | 2024 | 스펙트럼을 절반으로 분리해 noisy-clean 쌍 생성. N2N·N2V보다 우수. *Optics Letters* |
| [Self2Self (S2Snet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10890874/) | 2024 | Dropout 기반 단일 이미지 자가지도 |
| [SSN2V](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10299996/) | 2023 | 스페클 분리 기반 Noise2Void 확장 |

**Diffusion 계열**

| 논문 | 연도 | 핵심 |
|------|------|------|
| [GARD](https://arxiv.org/abs/2509.10341) | 2025 | OCT 스페클 감마 분포를 Diffusion에 직접 적용. *MICCAI 2025* |
| [Content-Preserving Diffusion](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_62) | 2023 | 비지도 Diffusion, 해부학적 구조 보존. *MICCAI 2023* |

**방법론 계층 (2024 systematic review 기준)**

| 계층 | 계열 | 비고 |
|------|------|------|
| 1 | Diffusion (물리 기반) | 최고 복원 품질 |
| 2 | GAN 기반 | 연구의 44% 차지, hallucination 위험 |
| 3 | 자가지도 (Sub2Full 등) | 실용성 1위, clean 데이터 불필요 |
| 4 | 지도학습 CNN/Transformer | 성숙 단계, clean GT 의존이 약점 |

---

### OCT와 결합하는 멀티모달리티

OCT는 투과 깊이가 1.5~2mm로 제한되므로, 아래 모달리티와 융합하여 진단 범위를 확장할 수 있다.

| 모달리티 | 융합 시너지 | 대표 연구 |
|---------|-----------|---------|
| **광음향 영상 (PAI)** | OCT가 형태 구조를 제공하면 PAI는 염증 부위 미세혈관·산소포화도 매핑 | [PMC6226559](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226559/) |
| **고주파 초음파 (HFUS)** | OCT의 표층(1.5mm) + 초음파의 심부 치조골 정보 상호 보완 | [MDPI 센서](https://www.mdpi.com/1424-8220/13/7/8928) |
| **정량 광형광 (QLF)** | 세균 바이오필름·치석의 형광 신호로 병변 위치를 3D OCT에 정합 | [Liverpool Repo](https://livrepository.liverpool.ac.uk/1143/1/Amaechi2002OCTcorrelates.pdf) |

---

### 평가 지표

디노이징 성능을 정량적으로 비교하기 위해 아래 세 지표를 사용한다.

#### PSNR (Peak Signal-to-Noise Ratio, 최대 신호 대 잡음비)

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

- **MSE**: 복원 이미지와 clean 이미지의 픽셀별 제곱 오차 평균
- **MAX**: 픽셀 최댓값 (0~1 정규화 기준으로 1.0)
- 단위: dB. 실용 범위는 보통 20~40 dB
- MSE가 분모에 위치하므로 **오차가 작을수록 PSNR이 높아진다** → 클수록 좋음
- **한계**: 픽셀 단위 수치 오차만 반영하므로 인간이 인지하는 구조적 선명도를 완전히 포착하지 못함

#### SSIM (Structural Similarity Index, 구조적 유사도)

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- $\mu$: 지역 평균 (밝기), $\sigma^2$: 분산 (대비), $\sigma_{xy}$: 공분산 (구조)
- 범위: 0~1. 복원 이미지와 clean 이미지의 **밝기·대비·구조** 유사도를 동시에 측정
- 오차가 아닌 유사도이므로 **클수록 좋음**
- PSNR보다 인간 시각 특성에 더 가까운 평가 지표
- **한계**: 지역 윈도(11×11) 기반이라 전역적 왜곡은 놓칠 수 있음

#### CNR (Contrast-to-Noise Ratio, 대비 대 잡음비)

$$\text{CNR} = \frac{|\mu_\text{signal} - \mu_\text{bg}|}{\sqrt{\sigma_\text{signal}^2 + \sigma_\text{bg}^2}}$$

- signal 영역: clean reference 기준 상위 50% 픽셀 (median 이상)
- background 영역: clean reference 기준 하위 50% 픽셀
- 분모: signal과 background 표준편차의 RSS(제곱합의 제곱근)
- **클수록** 레이어 경계가 선명하고 구조 식별이 쉬움
- PSNR·SSIM이 전체 픽셀을 고려하는 반면, CNR은 **임상적 진단 품질**에 더 직접적으로 대응
- 구현: `scripts/01_baseline/utils.py` `compute_cnr()` 함수 (모든 단계에서 동일 기준 사용)

---

### Super-Resolution

OCT 해상도는 두 축이 독립적으로 제한된다.

| 축 | 제한 원인 | 임상 한계 |
|----|----------|---------|
| 축 방향 (Axial) | 광원 파장·대역폭 | ~5–10 µm |
| 측 방향 (Lateral) | 대물렌즈 NA | 초점 범위 벗어나면 급격히 저하 |

딥러닝 SR은 기존 저해상도 스캔에서 소프트웨어적으로 해상도를 복원하는 대안이다.

#### Joint Denoising + SR

OCT 열화 모델: $y = D(H(x) \cdot N_s + N_a)$

순차(Sequential) 처리는 Step1 오류가 Step2에서 증폭되므로, 단일 네트워크 Joint 방식이 유리하다.

**주요 논문 비교**

| 논문 | 연도 | 아키텍처 | 학습 방식 | 성능 (PKU37 ×4) |
|------|------|---------|---------|----------------|
| [SDSR-OCT](https://pubmed.ncbi.nlm.nih.gov/31052772/) | 2019 | GAN | 지도학습 | — |
| [N2NSR-OCT](https://www.researchgate.net/publication/344907164) | 2020 | U-Net+DBPN | 반지도(N2N) | — |
| [O-PRESS](https://arxiv.org/abs/2401.03150) | 2024 | 반복+등변 | 자가지도 | — |
| [**PSCAT**](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161353/) | 2024 | 경량 Transformer | 지도학습 | **PSNR 31.48 dB, SSIM 0.8712** |
| [PnP-DM](https://arxiv.org/abs/2505.14916) | 2025 | Diffusion PnP | 역문제 | 구조 선명도 최고 |

---

## 발표 자료

```bash
# results/presentation.pptx 생성 (13슬라이드)
uv run python scripts/make_ppt.py
```

---

## 데이터

보유 데이터셋 구조 및 상세 설명 → [data/data_description.md](data/data_description.md)

단계별 실험 결과 정리 → [results/result_description.md](results/result_description.md)
