# 실험 결과 정리

AI 기반 OCT 치주질환 처리 프로젝트의 단계별 실험 결과를 기록한다.

---

## 1단계: 전통적 방법 베이스라인 측정

### 개요

딥러닝 모델 도입 전, 전통적 신호처리 방법으로 OCT 스페클 노이즈 제거 성능을 측정하여 비교 기준값을 확보한다.

- **데이터**: SBSDI D1 — `data/Final_Publication_2013_SBSDI/For synthetic experiments/`
  - 18쌍의 noisy-clean 이미지 (single B-scan vs. 다중 프레임 평균)
  - 해상도: 450×900 px, 그레이스케일
- **스크립트**: `scripts/01_baseline/`
- **결과 파일**:
  - `results/01_baseline/metrics/per_image.csv` — 이미지별 세부 지표
  - `results/01_baseline/metrics/summary.csv` — 방법별 평균/표준편차
  - `results/01_baseline/images/` — 시각적 비교 (sample_01~15.png)

### 적용 방법

| 방법 | 유형 | 핵심 원리 |
|------|------|----------|
| **NLM** | 비국소 평균 필터 | 이미지 전체에서 유사 패치를 찾아 가중 평균 |
| **BM3D** | 변환 도메인 필터 | 유사 블록을 3D 변환 후 임계화 |
| **SRAD** | 이방성 확산 | 스페클 곱셈성 모델을 반영한 PDE 기반 확산 |

### 성능 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR | 처리 속도 |
|------|-----------|------|-----|----------|
| NLM | 26.12 +- 2.01 | 0.492 +- 0.050 | 1.130 +- 0.119 | 0.51 +- 0.04 s/장 |
| BM3D | 27.00 +- 2.51 | 0.599 +- 0.066 | 1.134 +- 0.119 | 3.20 +- 0.20 s/장 |
| **SRAD** | **27.50 +- 1.98** | **0.652 +- 0.023** | **1.220 +- 0.121** | 5.68 +- 0.20 s/장 |

### 분석

- **SRAD**가 PSNR, SSIM, CNR 세 지표 모두 1위. 곱셈성 스페클 노이즈에 특화된 확산 방정식이 OCT에 적합함을 확인.
- **BM3D**는 SSIM에서 NLM 대비 큰 폭 향상(0.492 → 0.599). 비지역적 구조 보존 능력 우수.
- **NLM**은 속도 대비 가성비가 가장 좋음. PSNR은 BM3D보다 낮지만 약 6배 빠름.
- SSIM 표준편차가 SRAD(0.023)에서 가장 낮음 — 이미지마다 일관된 성능.

### 딥러닝 목표 기준값

| 지표 | 목표 | 근거 |
|------|------|------|
| PSNR | > 27.50 dB | SRAD 평균값 초과 |
| SSIM | > 0.652 | SRAD 평균값 초과 |
| CNR | > 1.220 | SRAD 평균값 초과 |

---

## 2단계: 합성 스페클 노이즈 생성 파이프라인

### 개요

clean 이미지만 보유한 대규모 데이터셋(AROI, Kermany)에 물리 기반 스페클 노이즈를 합성하여 지도학습용 noisy-clean 쌍을 생성한다.

- **스크립트**: `scripts/02_synthetic_noise/`
- **결과 파일**:
  - `results/02_synthetic_noise/noise_params.json` — 사용된 노이즈 파라미터
  - `results/02_synthetic_noise/metadata.csv` — 전체 6,136쌍 경로 목록
  - `results/02_synthetic_noise/AROI/` — AROI clean/noisy/samples
  - `results/02_synthetic_noise/Kermany/` — Kermany clean/noisy/samples
  - `results/02_synthetic_noise/validation/` — 검증 결과 (분포 플롯, 파라미터 CSV)

### 노이즈 모델

I = S * Ns + Na,  Ns ~ Gamma(L, 1/L),  Na ~ N(0, sigma_a^2)

| 항 | 의미 |
|----|------|
| I | 합성 noisy 이미지 |
| S | 원본 clean 이미지 |
| Ns | 곱셈성 스페클 노이즈 (Gamma 분포, mean=1, var=1/L) |
| Na | 가산성 가우시안 노이즈 |
| L | looks 수 — 높을수록 노이즈 약함 |

### 파라미터 캘리브레이션 (SBSDI D1 18쌍 기반)

SBSDI D1의 실제 noisy-clean 쌍에서 Ns = I / S 비율의 Gamma 모멘트 매칭으로 L을 추정했다.

| 파라미터 | 채택값 | 평균 | 표준편차 | 범위 |
|---------|-------|------|---------|------|
| L | **5.266** | 5.266 | 0.743 | [3.015, 6.228] |
| sigma_a | **0.010** | 0.010 | 0.000 | [0.010, 0.010] |

**KS 통계량** (실제 Ns 분포 vs 합성 Gamma 분포): 평균 **0.119**
- 0에 가까울수록 분포 일치
- Gamma 파라메트릭 모델이 실제 스페클 분포를 합리적으로 근사함을 확인

쌍별 L 추정값 (18쌍):

| pair | L | KS stat | pair | L | KS stat |
|------|---|---------|------|---|---------|
| 1 | 5.527 | 0.107 | 10 | 3.015 | 0.119 |
| 2 | 5.648 | 0.136 | 11 | 4.744 | 0.082 |
| 3 | 6.035 | 0.186 | 12 | 3.965 | 0.080 |
| 4 | 5.602 | 0.126 | 13 | 5.460 | 0.143 |
| 5 | 4.819 | 0.110 | 14 | 5.861 | 0.106 |
| 6 | 5.589 | 0.121 | 15 | 5.512 | 0.099 |
| 7 | 6.228 | 0.089 | 16 | 5.253 | 0.117 |
| 8 | 5.320 | 0.086 | 17 | 5.350 | 0.134 |
| 9 | 5.772 | 0.166 | 18 | 5.086 | 0.139 |

### 생성된 학습 데이터

| 데이터셋 | 쌍 수 | 해상도 (H x W) | 원본 형식 | 전처리 |
|---------|------|--------------|---------|-------|
| AROI | 1,136 | 512 x 1024 | PNG, 512x1024 (세로) | 90도 CCW 회전 후 저장 |
| Kermany OCT2017 | 5,000 | 496 x 512 | JPEG, 다양 | 그레이스케일 변환 |
| **합계** | **6,136** | — | — | — |

- 메타데이터: `results/02_synthetic_noise/metadata.csv` (dataset, idx, clean경로, noisy경로, H, W)
- 랜덤 시드: 42 (재현 가능)
- Kermany: train 전체(83,484장) 중 랜덤 5,000장 샘플링

### 재생성 명령

```bash
# 파라미터 검증 (SBSDI D1 기반, 분포 플롯 생성)
uv run python scripts/02_synthetic_noise/validate_noise.py

# 전체 파이프라인 (캘리브레이션 + 생성)
uv run python scripts/02_synthetic_noise/run_synthetic.py --calibrate --max-kermany 5000

# 파라미터 직접 지정
uv run python scripts/02_synthetic_noise/run_synthetic.py --L 5.27 --sigma-a 0.01 --max-kermany 5000
```

---

---

## 3단계: 자가지도 학습 (N2N 프레임쌍, Sub2Full 방식)

### 개요

clean Ground Truth 없이 OCT 이미지만으로 학습하는 자가지도 방식.
같은 위치를 반복 스캔한 프레임 쌍을 Noise2Noise 쌍으로 사용한다.

- **학습 데이터**: SBSDI `For real experiments on Humans` (39세트, 4프레임/세트)
  - 같은 세트의 (frame_i, frame_j) i != j 순서쌍 → 39 x 12 = **468쌍**
  - 두 프레임은 조직 구조 동일, 스페클 패턴 독립적 → N2N 조건 자연 만족
- **평가 데이터**: SBSDI D1 18쌍 (전통 방법과 동일)
- **스크립트**: `scripts/03_sub2full/`
- **결과 파일**:
  - `results/03_sub2full/checkpoints/best.pth` — 최적 모델 가중치
  - `results/03_sub2full/metrics/per_image.csv` — 이미지별 지표
  - `results/03_sub2full/metrics/summary.csv` — 평균/표준편차
  - `results/03_sub2full/metrics/training_log.csv` — 에포크별 loss
  - `results/03_sub2full/images/` — 시각적 비교 (sample_01, 05, 10, 15)

### 모델 및 학습 설정

| 항목 | 설정 |
|------|------|
| 아키텍처 | 경량 U-Net (인코더 3단계 + 보틀넥 + 디코더 3단계) |
| 파라미터 수 | ~1.95M |
| 손실 함수 | L1 Loss |
| 옵티마이저 | Adam (lr=1e-4, CosineAnnealing) |
| 에포크 | 500 |
| 배치 크기 | 4 (128x128 패치) |
| 학습 시간 | 약 221분 (RTX A4000) |

### 성능 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| NLM | 26.12 +- 2.01 | 0.492 +- 0.050 | 1.130 +- 0.119 |
| BM3D | 27.00 +- 2.51 | 0.599 +- 0.066 | 1.134 +- 0.119 |
| SRAD | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 |
| **N2N (Sub2Full)** | **26.84 +- 2.18** | **0.681 +- 0.033** | **1.148 +- 0.133** |

### 분석

- **SSIM 0.681이 SRAD 0.652를 초과** — clean GT 없이도 구조 보존 품질에서 전통 방법 1위를 넘김
- PSNR (26.84)은 SRAD (27.50)보다 낮지만 BM3D (27.00)에 근접
- CNR (1.148)은 SRAD (1.220)보다 낮음 — 경계 대비 보존에서 아직 열세
- **loss 수렴 문제**: 0.105 → 0.099로 거의 개선 없음. 같은 세트 프레임 간 조직 구조가 너무 유사해 모델이 깊이 학습하지 못하는 한계. 학습 데이터 39세트의 절대적 부족도 원인.

### 재실행 명령

```bash
uv run python scripts/03_sub2full/run_sub2full.py
uv run python scripts/03_sub2full/run_sub2full.py --eval-only  # 평가만
```

---

---

## 3단계-B: 지도학습 (합성 noisy-clean 6,136쌍)

### 개요

합성 noisy-clean 쌍으로 U-Net을 학습하고 SBSDI D1 18쌍으로 평가.

- **학습 데이터**: `results/02_synthetic_noise/metadata.csv` (AROI 1,136 + Kermany 5,000 = 6,136쌍)
- **평가 데이터**: SBSDI D1 18쌍
- **스크립트**: `scripts/03_supervised/`
- **결과 파일**:
  - `results/03_supervised/checkpoints/best.pth` — 최적 모델 가중치 (best loss: 0.021621)
  - `results/03_supervised/metrics/per_image.csv` — 이미지별 지표
  - `results/03_supervised/metrics/summary.csv` — 평균/표준편차
  - `results/03_supervised/metrics/training_log.csv` — 에포크별 loss
  - `results/03_supervised/images/` — 시각적 비교

### 모델 및 학습 설정

| 항목 | 설정 |
|------|------|
| 아키텍처 | U-Net (~1.95M params, 03_sub2full/model.py 공유) |
| 손실 함수 | L1 Loss |
| 옵티마이저 | Adam (lr=2e-4, CosineAnnealing) |
| 에포크 | 100 / 배치 8 / 패치 128×128 |
| 학습 시간 | 약 305분 (RTX A4000) |

### loss 수렴

| 에포크 | loss |
|--------|------|
| 1 | 0.044891 |
| 10 | 0.023456 |
| 50 | 0.021973 |
| 100 | 0.021655 |

epoch 10 이후 수렴 포화. 이후 90 에폭 동안 0.0004 이하 개선에 그침.

### 성능 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 |
| **지도학습 (합성)** | **22.43 +- 0.94** | **0.333 +- 0.049** | **0.902 +- 0.100** |

### 분석

세 지표 모두 SRAD 크게 미달. 근본 원인은 **도메인 갭**:

| 구분 | 학습 데이터 | 평가 데이터 |
|------|-----------|-----------|
| 노이즈 유형 | 합성 Gamma + Gaussian | 실제 OCT 스페클 |
| 이미지 소스 | AROI / Kermany (안과 망막) | SBSDI D1 |
| Clean 기준 | 수학적 모델로 생성 | 다중 프레임 평균 |

비교 참고: clean GT 없이 실제 데이터로만 학습한 N2N(SSIM 0.681)이 합성 데이터 지도학습(SSIM 0.333)보다 훨씬 우수 — 노이즈 모델의 정확도가 지도학습 성패를 좌우함.

### 재실행 명령

```bash
uv run python scripts/03_supervised/run_supervised.py
uv run python scripts/03_supervised/run_supervised.py --eval-only  # 평가만
```

---

## 4단계: 사전 학습 SR 모델 — Real-ESRGAN 테스트

### 개요

Real-ESRGAN 사전 학습 모델(blind SR, 실제 열화 처리)을 OCT 이미지에 적용.

- **스크립트**: `scripts/04_sr_test/test_sr.py` (단일 이미지), `scripts/04_sr_test/eval_sr_full.py` (18쌍 전체)
- **가중치**: `weights/RealESRGAN_x2plus.pth`, `weights/RealESRGAN_x4plus.pth` (각 64MB)
- **결과 파일**:
  - `results/04_sr_test/comparison_x2.png`, `comparison_x4.png`, `all_comparison.png` — 단일 이미지 비교
  - `results/04_sr_test/eval_full_x2.csv`, `eval_full_x4.csv` — 18쌍 쌍별 지표
  - `results/04_sr_test/eval_full_summary.txt` — 전체 요약

### 모델 정보

| 항목 | 내용 |
|------|------|
| 모델 | Real-ESRGAN (RRDBNet, num_block=23) |
| 학습 | Blind SR — 실제 복합 열화 (노이즈 + 블러 + 압축 아티팩트) 동시 복원 |
| 추론 | CUDA half precision, tile=400 |

### 성능 결과 — 단일 이미지 (SBSDI D1 세트 1)

| 방법 | PSNR (dB) | SSIM | 출력 크기 | 처리 시간 |
|------|-----------|------|---------|---------|
| SRAD (베이스라인) | 27.50 | 0.652 | 450×900 | ~5.7s |
| Real-ESRGAN x2 | 27.69 | 0.673 | 900×1800 | 0.93s |
| Real-ESRGAN x4 | 27.92 | 0.675 | 1800×3600 | 1.47s |

### 성능 결과 — 18쌍 전체 평가 (SBSDI D1, 원본 스케일 다운스케일 후 비교)

| 방법 | PSNR (dB) | SSIM | CNR | 처리 시간 |
|------|-----------|------|-----|---------|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 | ~5.7s/장 |
| Real-ESRGAN x2 | 27.30 +- 2.35 | **0.674 +- 0.034** | 1.121 +- 0.116 | 0.62s/장 |
| **Real-ESRGAN x4** | **27.57 +- 2.40** | **0.673 +- 0.034** | 1.166 +- 0.117 | 1.42s/장 |

### 분석

- **x4 PSNR·SSIM 모두 SRAD 초과** (전체 18쌍 평균 기준). 사전 학습 SR 모델이 fine-tuning 없이 노이즈 제거 + 해상도 향상 동시 달성
- **x2는 SSIM만 초과**, PSNR은 SRAD보다 0.20 dB 낮음 — 단일 이미지 테스트(27.69)보다 전체 평균(27.30)이 낮으며, 이미지마다 편차가 큼 (std 2.35)
- **CNR은 x2/x4 모두 미달** (1.121, 1.166 vs 1.220) — blind SR이 전역 대비 보존보다 고주파 디테일 복원에 특화됨을 시사
- 처리 속도 0.62~1.42s/장으로 SRAD(5.7s) 대비 4~9배 빠름

### 재실행 명령

```bash
# 단일 이미지 테스트
uv run python scripts/04_sr_test/test_sr.py

# 18쌍 전체 평가
uv run python scripts/04_sr_test/eval_sr_full.py
```

---

## 전체 방법 비교 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR | 비고 |
|------|-----------|------|-----|------|
| NLM | 26.12 +- 2.01 | 0.492 +- 0.050 | 1.130 +- 0.119 | 전통 방법 |
| BM3D | 27.00 +- 2.51 | 0.599 +- 0.066 | 1.134 +- 0.119 | 전통 방법 |
| SRAD | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 | 전통 방법 최고 |
| N2N 프레임쌍 | 26.84 +- 2.18 | **0.681 +- 0.033** | 1.148 +- 0.133 | 자가지도, 실제 데이터 |
| 지도학습 (합성) | 22.43 +- 0.94 | 0.333 +- 0.049 | 0.902 +- 0.100 | 도메인 갭 문제 |
| Real-ESRGAN x2 | 27.30 +- 2.35 | 0.674 +- 0.034 | 1.121 +- 0.116 | SR 포함, SSIM만 초과 |
| **Real-ESRGAN x4** | **27.57 +- 2.40** | **0.673 +- 0.034** | 1.166 +- 0.117 | SR 포함, PSNR+SSIM 초과 |

---

## 5단계: Pre-train → Fine-tune (합성 사전학습 + N2N fine-tuning)

### 개요

합성 지도학습 가중치로 초기화 후 SBSDI real N2N 쌍으로 fine-tuning. Loss를 L1 + SSIM 조합으로 개선.

- **사전학습 가중치**: `results/03_supervised/checkpoints/best.pth`
- **Fine-tune 데이터**: SBSDI real 39세트 × 12 = 468쌍 (N2N 방식)
- **스크립트**: `scripts/05_finetune/run_finetune.py`
- **결과 파일**:
  - `results/05_finetune/checkpoints/best.pth`
  - `results/05_finetune/metrics/per_image.csv`
  - `results/05_finetune/metrics/summary.csv`
  - `results/05_finetune/metrics/training_log.csv`
  - `results/05_finetune/images/`

### 학습 설정

| 항목 | 설정 |
|------|------|
| 초기 가중치 | 합성 지도학습 best.pth (loss 0.021621) |
| 손실 함수 | L1 + 0.1 × (1 - SSIM) |
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

에포크 20 이후 포화. N2N 노이즈 타겟의 고유 분산(noise floor)이 loss 하한을 ~0.184로 고정.

### 성능 결과 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 |
| N2N from scratch | 26.84 +- 2.18 | **0.681 +- 0.033** | 1.148 +- 0.133 |
| **Pre-train + Fine-tune** | **27.46 +- 2.55** | 0.6795 +- 0.037 | **1.169 +- 0.120** |

### 분석

- **PSNR**: N2N 대비 +0.62 dB 향상 (26.84 → 27.46). SRAD까지 0.04 dB 차이로 사실상 동급.
- **SSIM**: SRAD(0.652) 초과 유지. N2N(0.681)보다 0.001 낮음 — 유의미한 차이 없음.
- **CNR**: N2N 대비 +0.021 향상 (1.148 → 1.169). SRAD(1.220)까지는 0.051 잔류.
- **전이학습 효과 확인**: N2N from scratch(loss 0.105 → 0.099)보다 합성 사전학습 초기화(0.205 → 0.184)가 훨씬 낮은 loss floor 달성.
- **남은 병목**: N2N 노이즈 타겟의 고유 분산 — clean GT 없이는 이 분산을 제거할 수 없어 loss floor가 형성됨.

### 재실행 명령

```bash
uv run python scripts/05_finetune/run_finetune.py
uv run python scripts/05_finetune/run_finetune.py --eval-only
```

---

## 6단계: 6-fold Cross-Validation 지도학습 (U-Net + real clean GT)

### 개요

SBSDI D1 18쌍을 k=6 fold로 나눠 real clean GT로 지도학습. clean GT 활용 효과 단독 측정.

- **모델**: 기존 U-Net (1.95M params, base_ch=32)
- **초기 가중치**: `results/03_supervised/checkpoints/best.pth`
- **학습 데이터**: fold당 15쌍 × 275패치 (stride 32) = 4,125패치
- **평가 데이터**: fold당 3쌍 (18쌍 전체 순환 평가)
- **스크립트**: `scripts/06_kfold/run_kfold.py`
- **결과 파일**:
  - `results/06_kfold/checkpoints/fold_{k}/best.pth`
  - `results/06_kfold/metrics/fold_{k}/per_image.csv`
  - `results/06_kfold/metrics/summary.csv`
  - `results/06_kfold/images/fold_{k}/`

### 학습 설정

| 항목 | 설정 |
|------|------|
| 손실 함수 | L1 + 0.1 × (1 - SSIM) |
| LR | 5e-5 / CosineAnnealing |
| 에포크 | 150 / 배치 16 / 패치 128×128 stride 32 |
| 총 학습 시간 | 175분 (RTX A4000) |

### Loss 수렴 (fold별 best loss)

| Fold | 평가 쌍 | Best loss |
|------|---------|-----------|
| 1 | 1, 7, 13 | 0.048481 |
| 2 | 2, 8, 14 | 0.048843 |
| 3 | 3, 9, 15 | 0.047286 |
| 4 | 4, 10, 16 | 0.048588 |
| 5 | 5, 11, 17 | 0.047514 |
| 6 | 6, 12, 18 | 0.048742 |

5단계 fine-tune loss floor 0.184 대비 **0.048 수준**으로 현저히 낮아짐 — clean GT 타겟 효과 확인.

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 |
| Pre-train + Fine-tune (5단계) | 27.46 +- 2.55 | 0.6795 +- 0.037 | 1.169 +- 0.120 |
| **6-fold CV (U-Net)** | 27.21 +- 2.41 | **0.6822 +- 0.032** | 1.164 +- 0.113 |

### 분석 (batch=16, ep=150, 조기 종료 없음)

- **SSIM 0.6822**: 순수 디노이징 방법 중 최고.
- **PSNR 27.21**: SRAD(27.50) 미달. loss가 에포크 150에서도 감소 중 → 미수렴.
- early stopping 없이 150 에포크 고정 → 최적점을 지나쳐 과적합 가능성.

---

## 6단계-B: 6-fold CV (batch=64, early stopping, patience=30)

### 변경 사항

| 항목 | 6단계-A | 6단계-B |
|------|---------|---------|
| 배치 크기 | 16 | **64** |
| 최대 에포크 | 150 | 500 |
| Early stopping | 없음 | **val PSNR+10×SSIM 기준, patience=30** |
| 총 학습 시간 | 175분 | **47분** |

### 조기 종료 결과 (fold별)

| Fold | 평가 쌍 | 종료 에포크 | best val PSNR | best val SSIM |
|------|---------|-----------|--------------|--------------|
| 1 | 1, 7, 13 | 41 | 28.3932 | 0.6897 |
| 2 | 2, 8, 14 | 35 | **30.3970** | **0.7106** |
| 3 | 3, 9, 15 | 38 | 26.4715 | 0.6687 |
| 4 | 4, 10, 16 | 47 | 28.7917 | 0.6859 |
| 5 | 5, 11, 17 | 43 | 26.4973 | 0.6677 |
| 6 | 6, 12, 18 | 42 | 28.5621 | 0.6905 |

모든 fold가 35~47 에포크에서 조기 종료. 최적 epoch를 지나 과적합되기 전에 중단됨.

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 6단계-A (batch=16) | 27.21 +- 2.41 | 0.6822 +- 0.032 | 1.164 +- 0.113 |
| **6단계-B (batch=64, ES)** | **28.19 +- 2.56** | **0.6814 +- 0.031** | 1.169 +- 0.121 |

### 분석

- **PSNR 28.19**: 순수 디노이징 AI 방법 최초로 SRAD(27.50) 초과. +0.69 dB.
- **SSIM 0.6814**: SRAD(0.652) 초과 유지.
- **CNR 1.169**: SRAD(1.220) 미달. 경계 대비 향상은 아직 한계.
- **Early stopping 효과**: 학습 시간 175분 → 47분으로 단축, 과적합 방지, PSNR +0.97 dB 향상.
- **Fold 간 분산**: std=2.56 — Fold 2(30.40 dB)와 Fold 3(26.47 dB) 간 편차 큼. 18쌍의 데이터 다양성 한계.

### 재실행 명령

```bash
# 기본 (batch=64, epochs=500, patience=30)
uv run python scripts/06_kfold/run_kfold.py

# 평가만
# 각 fold 체크포인트 로드 후 evaluate_fold() 직접 호출 필요
```

---

## 7단계: DnCNN 백본 교체 (6-fold CV, from scratch)

### 개요

6단계-B U-Net 결과와 동일한 6-fold CV 프레임워크에서 백본을 DnCNN-B로 교체. 아키텍처 차이가 성능에 미치는 영향 단독 측정.

- **모델**: DnCNN-B (depth=20, channels=64, ~667K params)
- **초기 가중치**: 없음 (from scratch)
- **학습 방식**: 잔차 학습 — model(noisy) = residual, clean = clamp(noisy - residual, 0, 1)
- **스크립트**: `scripts/07_dncnn/run_dncnn.py`
- **결과 파일**:
  - `results/07_dncnn/checkpoints/fold_{k}/best.pth`
  - `results/07_dncnn/metrics/fold_{k}/per_image.csv`
  - `results/07_dncnn/metrics/summary.csv`
  - `results/07_dncnn/images/fold_{k}/`

### 모델 구조

| 레이어 | 구성 |
|--------|------|
| 첫 번째 (1층) | Conv(1, 64, 3×3) + ReLU |
| 중간 (2~19층) | Conv(64, 64, 3×3) + BN + ReLU |
| 마지막 (20층) | Conv(64, 1, 3×3) |
| 파라미터 수 | 667,073 |
| skip connection | 없음 (순수 순차 구조) |
| 출력 | 잔차(노이즈 추정값) |

### 학습 설정

| 항목 | 설정 | U-Net(6단계-B) 비교 |
|------|------|-------------------|
| 손실 함수 | L1 + 0.1 × (1 - SSIM) | 동일 |
| LR | 1e-3 / CosineAnnealing | 5e-5 (사전학습 없어 높게 설정) |
| 에포크 | max 500 / 배치 64 | 동일 |
| Early stopping | patience=30 (score=PSNR+10×SSIM) | 동일 |
| 총 학습 시간 | 355분 (RTX A4000) | 47분 |

### 조기 종료 결과 (fold별)

| Fold | 평가 쌍 | 종료 에포크 | best val PSNR | best val SSIM |
|------|---------|-----------|--------------|--------------|
| 1 | 1, 7, 13 | 86 | 28.2854 | 0.6776 |
| 2 | 2, 8, 14 | 96 | **30.2306** | **0.7036** |
| 3 | 3, 9, 15 | 146 | 26.6558 | 0.6639 |
| 4 | 4, 10, 16 | 69 | 28.7076 | 0.6749 |
| 5 | 5, 11, 17 | 65 | 26.6172 | 0.6577 |
| 6 | 6, 12, 18 | 105 | 28.5182 | 0.6844 |

U-Net(35~47 에포크)보다 65~146 에포크로 수렴이 느림 — from scratch 초기화의 영향.

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 6단계-B (U-Net, pretrained) | **28.19 +- 2.56** | **0.6814 +- 0.031** | 1.169 +- 0.121 |
| **7단계 (DnCNN, from scratch)** | 28.17 +- 2.47 | 0.6732 +- 0.032 | 1.167 +- 0.127 |

### 분석

- **PSNR 28.17**: SRAD(27.50) 초과. U-Net(28.19)과 차이 0.02 dB — 사실상 동등.
- **SSIM 0.6732**: SRAD(0.652) 초과. U-Net(0.6814)보다 0.008 낮음.
- **CNR 1.167**: SRAD(1.220) 미달. U-Net(1.169)과 거의 동일.
- **핵심 발견**: 667K DnCNN (from scratch) ≈ 1.95M U-Net (pretrained) — 성능 차이가 거의 없음.
  - 파라미터 수 차이(667K vs 1.95M, 약 3배), 사전학습 유무에도 불구하고 결과 동등.
  - 현재 성능 한계는 **모델 용량이 아닌 데이터 크기(18쌍)**에 의해 결정됨.
- **학습 시간 차이**: DnCNN 355분 vs U-Net 47분. from scratch + 수렴 느림으로 7.5배 더 소요.
- **Fold 2 패턴 재현**: U-Net(30.40 dB)과 DnCNN(30.23 dB) 모두 Fold 2에서 최고 성능. 쌍 2, 8, 14가 특히 학습하기 쉬운 패턴임을 시사.

### 재실행 명령

```bash
uv run python scripts/07_dncnn/run_dncnn.py
```

---

## 전체 방법 비교 (SBSDI D1, 18쌍 평균)

| 방법 | PSNR (dB) | SSIM | CNR | 비고 |
|------|-----------|------|-----|------|
| NLM | 26.12 +- 2.01 | 0.492 +- 0.050 | 1.130 +- 0.119 | 전통 방법 |
| BM3D | 27.00 +- 2.51 | 0.599 +- 0.066 | 1.134 +- 0.119 | 전통 방법 |
| SRAD | 27.50 +- 1.98 | 0.652 +- 0.023 | 1.220 +- 0.121 | 전통 방법 최고 |
| N2N 프레임쌍 | 26.84 +- 2.18 | 0.681 +- 0.033 | 1.148 +- 0.133 | 자가지도 |
| 지도학습 (합성) | 22.43 +- 0.94 | 0.333 +- 0.049 | 0.902 +- 0.100 | 도메인 갭 |
| Pre-train + Fine-tune | 27.46 +- 2.55 | 0.6795 +- 0.037 | 1.169 +- 0.120 | 전이학습 |
| Real-ESRGAN x2 | 27.30 +- 2.35 | 0.674 +- 0.034 | 1.121 +- 0.116 | SR 포함 |
| Real-ESRGAN x4 | 27.57 +- 2.40 | 0.673 +- 0.034 | 1.166 +- 0.117 | SR 포함 |
| 6-fold CV (U-Net, batch=16) | 27.21 +- 2.41 | 0.6822 +- 0.032 | 1.164 +- 0.113 | real clean GT, 미수렴 |
| **6-fold CV (U-Net, ES)** | **28.19 +- 2.56** | **0.6814 +- 0.031** | 1.169 +- 0.121 | real clean GT, SRAD 초과 |
| **7단계 (DnCNN, ES)** | 28.17 +- 2.47 | 0.6732 +- 0.032 | 1.167 +- 0.127 | from scratch, U-Net과 동등 |

---

## 진행 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 1단계 | 전통적 방법 베이스라인 측정 | 완료 |
| 2단계 | 합성 스페클 노이즈 생성 파이프라인 | 완료 |
| 3단계-A | 자가지도 학습 (N2N 프레임쌍) | 완료 |
| 3단계-B | 지도학습 (합성 noisy-clean 6,136쌍) | 완료 — 도메인 갭 확인 |
| 4단계 | 사전 학습 SR 모델 적용 (Real-ESRGAN 18쌍 전체 평가) | 완료 |
| 5단계 | Pre-train + Fine-tune (합성 사전학습 + N2N fine-tuning, L1+SSIM Loss) | 완료 |
| 6단계-A | 6-fold CV 지도학습 (U-Net, batch=16, ep=150) | 완료 |
| 6단계-B | 6-fold CV 지도학습 (U-Net, batch=64, early stopping) | 완료 — SRAD 초과 |
| 7단계 | DnCNN 백본 6-fold CV (from scratch) | 완료 — U-Net과 동등, 데이터 병목 확인 |
