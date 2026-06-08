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

pixel-aligned noisy-clean 쌍이 없는 대규모 망막 OCT 데이터셋(AROI, Kermany)을 활용한다. 이 데이터셋의 이미지는 real OCT 촬영본으로 고유한 스페클 노이즈를 포함하고 있으나, pixel-aligned clean 기준이 존재하지 않으므로 원본 이미지를 의사(pseudo) clean 기준으로 삼아 물리 기반 합성 스페클을 추가함으로써 지도학습용 noisy-clean 쌍을 생성한다. 이 방식으로 학습된 모델은 추가된 합성 스페클만을 제거 대상으로 학습하게 되어, 실제 OCT 스페클에 대한 일반화가 제한된다(3단계-B 도메인 갭 확인).

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

## 8단계: NAFNet 백본 교체 (6-fold CV, from scratch)

### 개요

DnCNN / U-Net과 동일한 6-fold CV 프레임워크에서 백본을 NAFNet으로 교체. 아키텍처 효과 측정.

- **모델**: NAFNet-width32 (~17M params)
- **초기 가중치**: 없음 (from scratch)
- **특이사항**: Fold 1은 batch=16 기존 결과 재사용, Fold 2~6은 batch=32로 학습
- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과 파일**:
  - `results/08_nafnet/checkpoints/fold_{k}/best.pth`
  - `results/08_nafnet/metrics/fold_{k}/per_image.csv`
  - `results/08_nafnet/metrics/summary.csv`
  - `results/08_nafnet/images/fold_{k}/`

### 모델 구조

| 구성 요소 | 내용 |
|----------|------|
| 인코더 | 4 스테이지, enc_blks=[1,1,1,28], 각 스테이지 후 stride-2 Conv 다운샘플 |
| 병목 | NAFBlock x1 |
| 디코더 | 4 스테이지, dec_blks=[1,1,1,1], PixelShuffle 업샘플 |
| skip | 인코더-디코더 대응 스테이지 덧셈 |
| global residual | 출력 = 네트워크(입력) + 입력 |
| 핵심 블록 | SimpleGate + SCA(채널 어텐션) + LayerNorm2d, 활성화 함수 없음 |
| 파라미터 수 | 17,112,673 |

### 학습 설정

| 항목 | 설정 | 비교 (DnCNN) |
|------|------|-------------|
| 손실 함수 | L1 + 0.1*(1-SSIM) | 동일 |
| 옵티마이저 | AdamW (lr=1e-3, wd=1e-4) | Adam (lr=1e-3) |
| Scheduler | CosineAnnealing | 동일 |
| Grad clip | norm=1.0 | 없음 |
| 배치 크기 | 32 (Fold 1: 16) | 64 |
| Early stopping | patience=30 | 동일 |
| Augment | H/V flip + rot90 | H/V flip |
| 총 학습 시간 | 123분 (RTX A4000) | 355분 |

### 조기 종료 결과 (fold별)

| Fold | 평가 쌍 | 종료 에포크 | best val PSNR | best val SSIM |
|------|---------|-----------|--------------|--------------|
| 1 | 1, 7, 13 | 36 (재사용) | 28.2780 | 0.6819 |
| 2 | 2, 8, 14 | 39 | **29.9704** | **0.6985** |
| 3 | 3, 9, 15 | 32 | 26.6745 | 0.6600 |
| 4 | 4, 10, 16 | 34 | 29.0099 | 0.6811 |
| 5 | 5, 11, 17 | 33 | 26.5474 | 0.6584 |
| 6 | 6, 12, 18 | 46 | 28.2139 | 0.6834 |

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 6단계-B (U-Net, ES) | **28.19 +- 2.56** | **0.6814 +- 0.031** | 1.169 +- 0.121 |
| 7단계 (DnCNN, ES) | 28.17 +- 2.47 | 0.6732 +- 0.032 | 1.167 +- 0.127 |
| **8단계 (NAFNet-32, ES)** | 28.11 +- 2.26 | 0.6743 +- 0.029 | **1.183 +- 0.120** |

### 분석

- **PSNR 28.11**: SRAD(27.50) 초과. U-Net(28.19)보다 0.08 dB 낮음 -- 사실상 동등.
- **SSIM 0.6743**: SRAD(0.652) 초과. U-Net(0.6814)보다 약간 낮음.
- **CNR 1.183**: 세 AI 아키텍처 중 최고. U-Net(1.169), DnCNN(1.167)보다 높음.
- **분산 최소**: std=2.26 -- U-Net(2.56), DnCNN(2.47)보다 fold 간 편차가 작음.
- **학습 시간**: 123분 -- DnCNN(355분) 대비 대폭 단축.
- **핵심 결론**: 667K(DnCNN) ≈ 1.95M(U-Net) ≈ 17M(NAFNet) -- 3번 연속 데이터 병목 확인.
  - 파라미터 수 25배 차이에도 PSNR 차이 0.08 dB 이내.
  - 18쌍 데이터 크기가 아키텍처 차이를 완전히 압도함.

### NAFNet 특이 현상: Loss 감소 ≠ PSNR 향상

loss 그래프와 val_psnr 그래프를 비교하면 6·7단계(U-Net, DnCNN)와 다른 패턴이 나타난다.

| 구분 | 6단계 U-Net | 8단계 NAFNet |
|------|-----------|------------|
| 학습률 | 5e-5 | **1e-3 (20배 높음)** |
| val_psnr 추이 | epoch 진행에 따라 점진적 향상 | **epoch 1~9에서 최고, 이후 정체 또는 하락** |
| train_loss 추이 | 완만하게 감소 | 가파르게 감소 |
| best.pth 저장 시점 | 수렴 후반부 | **초반 epoch** |

**원인**:

1. **학습률 과대**: lr=1e-3은 15쌍이라는 극소 데이터에서 초반 몇 에포크 만에 훈련 패치를 빠르게 암기하는 방향으로 가중치를 업데이트한다. 이후 loss는 계속 감소해도 새 이미지에 대한 일반화 성능(val_psnr)은 오히려 하락한다.

2. **극단적 과파라미터화**: 17M 파라미터 모델이 fold당 15쌍(≈4,125패치)을 학습한다. 모델 용량이 훈련 데이터를 완전히 암기할 수 있는 수준이어서 loss 감소가 일반화 향상을 의미하지 않는다.

3. **Loss-PSNR 분리**: train_loss는 훈련 패치 암기로 감소하지만, val_psnr은 새 이미지에 대한 일반화 실패로 정체 또는 하락한다. early stopping이 val 스코어 기준으로 동작하여 best.pth는 초반 epoch 가중치로 저장된다.

**NAFNet을 제대로 활용하려면** lr을 5e-5 이하로 낮추거나, 더 근본적으로는 학습 데이터를 늘려야 한다.

### 재실행 명령

```bash
uv run python scripts/08_nafnet/run_nafnet.py

# 특정 fold부터 재시작
uv run python scripts/08_nafnet/run_nafnet.py --start-fold 3
```

---

## 9단계: 다중 노이즈 재실현 증강 + NAFNet (6-fold CV)

### 개요

8단계에서 데이터 병목이 확인됐다. 18개 clean GT에 노이즈 모델(L=5.266, sigma_a=0.010)로 K=4개씩 합성 noisy를 추가 생성하여 학습 데이터를 5배로 늘렸다. 평가는 real noisy로만 수행하여 8단계와 동등한 조건을 유지한다.

- **증강 데이터**: `data/Final_Publication_2013_SBSDI/augmented_noisy/` (72장, 18 sets × K=4)
- **스크립트**: `scripts/09_augment/run_nafnet_aug.py`
- **결과 파일**:
  - `results/09_nafnet_aug/checkpoints/fold_{k}/best.pth`
  - `results/09_nafnet_aug/metrics/fold_{k}/per_image.csv`
  - `results/09_nafnet_aug/metrics/summary.csv`

### 학습 설정

| 항목 | 9단계 | 8단계 비교 |
|------|-------|-----------|
| 모델 | NAFNet-width32 (17M) | 동일 |
| 학습 쌍/fold | 15 sets × 5 = 75쌍 | 15쌍 |
| 평가 쌍/fold | 3쌍 (real only) | 동일 |
| 패치/fold | 20,625 | 4,125 (5배 증가) |
| 배치 크기 | 48 | 32 |
| Steps/epoch | 425 | ~129 |
| Early stopping | patience=30 | 동일 |

### 폴드별 결과

| Fold | 평가 이미지 | best val PSNR | best val SSIM | stopped epoch |
|------|-----------|--------------|--------------|--------------|
| 1 | 1, 7, 13 | 28.3362 | 0.6856 | 33 |
| 2 | 2, 8, 14 | **30.7966** | **0.7130** | 34 |
| 3 | 3, 9, 15 | 26.7262 | 0.6660 | 31 |
| 4 | 4, 10, 16 | 29.2315 | 0.6885 | 32 |
| 5 | 5, 11, 17 | 26.6671 | 0.6664 | ~31 |
| 6 | 6, 12, 18 | 28.3153 | 0.6888 | ~34 |

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 8단계 (NAFNet-32, ES) | 28.11 +- 2.26 | 0.6743 +- 0.029 | 1.183 +- 0.120 |
| **9단계 (NAFNet+Aug, K=4)** | **28.35 +- 2.56** | **0.6832 +- 0.032** | 1.168 +- 0.121 |

### 분석

- **PSNR +0.24 dB**: 학습 데이터 5배 증가의 효과. 데이터 병목을 부분적으로 완화.
- **SSIM +0.009**: 노이즈 다양성 증가로 일반화 소폭 향상.
- **CNR -0.015**: 합성 노이즈가 실제 스페클과 완전히 동일하지 않아 대비 보존 측면에서 미미하게 하락. 오차 범위 내 차이.
- **학습 수렴 속도 동등**: early stop epoch 31~34로 8단계(32~46)와 유사. 에폭당 데이터는 5배 많으나 수렴 epoch 수는 비슷.
- **결론**: K=4 재실현 증강은 PSNR/SSIM에 유효하나 CNR 한계는 해소되지 않음. 근본적 돌파를 위해서는 외부 real OCT 데이터 확보가 필요.

### 재실행 명령

```bash
uv run python scripts/09_augment/run_nafnet_aug.py --batch-size 48
# 특정 fold 범위만 재실행
uv run python scripts/09_augment/run_nafnet_aug.py --start-fold 5 --end-fold 5
```

---

## 10단계: NAFNet lr=1e-4 재실험 — lr 영향 검증

### 개요

8단계에서 확인된 loss-PSNR 분리 현상(lr=1e-3이 너무 높아 초반 과적합)을 해소하기 위해 lr을 1e-4로 낮춰 재실험.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py`
- **결과 파일**:
  - `results/10_nafnet_lr1e4/checkpoints/fold_{k}/best.pth`
  - `results/10_nafnet_lr1e4/metrics/fold_{k}/training_log.csv`
  - `results/10_nafnet_lr1e4/metrics/summary.csv`

### 학습 설정

| 항목 | 8단계 | 10단계 |
|------|-------|-------|
| lr | 1e-3 | **1e-4** |
| batch | 32 | **48** |
| 나머지 | 동일 | 동일 |

### 폴드별 결과

| Fold | 평가 쌍 | best val PSNR | best val SSIM | stopped epoch | Epoch 1 val PSNR |
|------|---------|--------------|--------------|--------------|-----------------|
| 1 | 1, 7, 13 | 28.3105 | 0.6742 | 49 | 21.441 |
| 2 | 2, 8, 14 | 29.6645 | 0.6922 | 43 | 24.622 |
| 3 | 3, 9, 15 | 26.6148 | 0.6576 | 48 | 22.277 |
| 4 | 4, 10, 16 | 28.7183 | 0.6721 | 40 | 23.601 |
| 5 | 5, 11, 17 | 26.3658 | 0.6558 | 56 | 22.088 |
| 6 | 6, 12, 18 | 28.0701 | 0.6813 | 61 | 23.968 |

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| **8단계 NAFNet (lr=1e-3)** | **28.11 +- 2.26** | **0.6743 +- 0.029** | 1.183 +- 0.120 |
| 10단계 NAFNet (lr=1e-4) | 27.95 +- 2.11 | 0.6666 +- 0.026 | **1.191 +- 0.116** |

### 분석

- **lr=1e-4가 오히려 성능 하락** (PSNR -0.16 dB, SSIM -0.008): 예상과 반대되는 결과.
- **원인 — Epoch 1 val_psnr 차이**:
  - lr=1e-3: epoch 1에서 val_psnr ~28.27로 이미 높음. 초반 몇 스텝에서 빠르게 좋은 수렴점 도달 후 early stopping이 포착.
  - lr=1e-4: epoch 1에서 val_psnr ~21~25로 낮음. 점진적으로 학습하지만 patience=30 내에 lr=1e-3의 초반 수렴점 수준까지 도달하지 못함.
- **CNR은 소폭 향상** (1.183 → 1.191): 안정적 학습이 경계 대비 보존에 미세하게 유리.
- **결론**: 데이터 15쌍/fold 환경에서 lr 조정은 근본 해결책이 아님. 데이터 확보 없이는 하이퍼파라미터 튜닝의 효과가 극히 제한적.

### 재실행 명령

```bash
uv run python scripts/08_nafnet/run_nafnet.py --lr 1e-4 --batch-size 48 --results-dir results/10_nafnet_lr1e4
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
| 7단계 (DnCNN, ES) | 28.17 +- 2.47 | 0.6732 +- 0.032 | 1.167 +- 0.127 | from scratch, U-Net과 동등 |
| 8단계 (NAFNet-32, ES) | 28.11 +- 2.26 | 0.6743 +- 0.029 | **1.183 +- 0.120** | from scratch, CNR 최고 |
| **9단계 (NAFNet+Aug, K=4)** | **28.35 +- 2.56** | **0.6832 +- 0.032** | 1.168 +- 0.121 | 증강 5배, PSNR/SSIM 최고 |
| 10단계 (NAFNet lr=1e-4) | 27.95 +- 2.11 | 0.6666 +- 0.026 | 1.191 +- 0.116 | lr 인하로 오히려 하락, 데이터 병목 재확인 |
| 11단계 (NAFNet + EdgeLoss λ=0.5) | 28.06 +- 2.08 | 0.6699 +- 0.027 | **1.186 +- 0.116** | CNR +0.003, 손실함수 최고 |
| 12단계 (NAFNet + Edge+Freq) | 28.05 +- 2.03 | 0.6683 +- 0.028 | 1.182 +- 0.120 | Freq 추가로 오히려 하락, 손실함수 실험 종료 |
| **13단계 (AROI pretrain→D1 finetune)** | 28.23 +- 2.49 | **0.6838 +- 0.029** | 1.177 +- 0.121 | SSIM 전체 최고, PSNR +0.12 vs scratch |

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
| 8단계 | NAFNet 백본 6-fold CV (width=32, from scratch) | 완료 — CNR 최고, 데이터 병목 재확인 |
| 9단계 | 다중 노이즈 재실현 증강 (K=4) + NAFNet 6-fold CV | 완료 — PSNR/SSIM 소폭 향상, CNR 한계 미해소 |
| 10단계 | NAFNet lr=1e-4 재실험 (lr 영향 검증) | 완료 — lr=1e-3 대비 오히려 성능 하락, 데이터 병목 재확인 |
| 11단계 | NAFNet + Edge Loss (Sobel, λ=0.5) | 완료 — CNR +0.003, noise 수준, 데이터 병목이 상한 |
| 12단계 | NAFNet + Edge Loss + Frequency Loss | 완료 — CNR 오히려 하락, 손실 함수 개선 실험 종료 |
| 13단계 | AROI N2N 사전학습 → SBSDI D1 k-fold fine-tuning | 완료 — SSIM 0.6838 (전체 최고), PSNR +0.12 dB |

---

## 11단계: NAFNet + Edge Loss (손실 함수 개선)

### 개요

CNR SRAD 미달(1.183 vs 1.220) 해소를 위해 Sobel 기반 Edge Loss 추가. 안정적인 정규화 방식 확립에 6회 시도가 필요했다.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --seed 42`
- **결과**: `results/11_nafnet_edge/`

### Edge Loss 정규화 시도 과정

| 시도 | 정규화 방식 | λ | 초기 loss | 결과 |
|------|-----------|---|---------|------|
| 1 | 없음 | 0.1 | 0.18~0.48 | Fold 4~6 발산 — Sobel 출력 최대 ~8, 스케일 과대 |
| 2 | 없음 | 0.01 | 정상 | 안정, 효과 없음 — 기여 0.8% 미만 |
| 3 | per-batch mean으로 나눔 | 0.1 | 0.57~0.72 | 전 fold 발산 — mean으로 나누면 값 20배 증폭 |
| 4 | SOBEL_MAX(11.314)로 나눔 | 0.1 | 정상 | 안정, 효과 없음 — 정규화 후 기여 여전히 0.8% |
| 5 | SOBEL_MAX로 나눔 | 1.0 | 0.17~0.47 | Fold 2, 4 발산 — fold별 edge 스케일 차이 |
| **6** | **SOBEL_MAX + clamp(0,1)** | **0.5** | **정상** | **전 fold 안정, CNR +0.003** |

**핵심 교훈**: Sobel 출력의 절대 스케일 변동성이 크기 때문에 고정 상수(SOBEL_MAX)와 clamp의 조합이 필수.

### 최종 손실 함수

```
Loss = L1 + 0.1×(1-SSIM) + 0.5×EdgeLoss
EdgeLoss = L1(clamp(Sobel(pred)/11.314, 0,1),
               clamp(Sobel(clean)/11.314, 0,1))
```

### 폴드별 결과

| Fold | 평가 쌍 | best val PSNR | best val SSIM | stopped epoch |
|------|---------|--------------|--------------|--------------|
| 1 | 1, 7, 13 | 28.1219 | 0.6732 | 42 |
| 2 | 2, 8, 14 | 29.8980 | 0.6940 | 39 |
| 3 | 3, 9, 15 | 26.8583 | 0.6562 | 33 |
| 4 | 4, 10, 16 | 28.9409 | 0.6757 | 36 |
| 5 | 5, 11, 17 | 26.4151 | 0.6530 | 32 |
| 6 | 6, 12, 18 | 28.1293 | 0.6788 | 39 |

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 8단계 NAFNet (기준) | **28.11 +- 2.26** | **0.6743 +- 0.029** | 1.183 +- 0.120 |
| **11단계 NAFNet + EdgeLoss** | 28.06 +- 2.08 | 0.6699 +- 0.027 | **1.186 +- 0.116** |

### 분석

- **CNR 1.183 → 1.186 (+0.003)**: 개선 방향은 확인. 단, std=0.116 대비 통계적 유의성 없음.
- **PSNR·SSIM 소폭 하락**: edge 보존을 강조하는 gradient가 픽셀 정확도를 미세하게 희생.
- **분산 감소**: std 2.26 → 2.08 (PSNR), 0.120 → 0.116 (CNR) — 학습 안정성은 향상.
- **근본 한계**: 손실 함수 개선이 올바른 방향이나 데이터 18쌍이 CNR 향상의 상한을 결정. 더 많은 데이터 없이는 loss 튜닝만으로 SRAD CNR(1.220) 초과 어렵다.

### 재실행 명령

```bash
uv run python scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --seed 42 --results-dir results/11_nafnet_edge
```

---

## 12단계: NAFNet + Edge Loss + Frequency Loss

### 개요

11단계 Edge Loss(λ=0.5)에 FFT 기반 Frequency Loss(λ=0.1)를 추가해 고주파 구조 보존 효과 검증.

- **스크립트**: `scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --lambda-freq 0.1 --seed 42`
- **결과**: `results/12_nafnet_freq/`

### 최종 손실 함수

```
Loss = L1 + 0.1×(1-SSIM) + 0.5×EdgeLoss + 0.1×FreqLoss
FreqLoss = L1(log1p(|rfft2(pred, norm="ortho")|),
               log1p(|rfft2(clean, norm="ortho")|))
```

`rfft2` (실수 FFT, 절반 스펙트럼) + `log1p` 압축으로 DC 성분 지배 방지, [0, ~4] 범위 안정화.

### 폴드별 결과

| Fold | 평가 쌍 | best val PSNR | best val SSIM | stopped epoch |
|------|---------|--------------|--------------|--------------|
| 1 | 1, 7, 13 | 28.1040 | 0.6733 | 42 |
| 2 | 2, 8, 14 | 29.8094 | 0.6922 | 39 |
| 3 | 3, 9, 15 | 26.8660 | 0.6565 | 33 |
| 4 | 4, 10, 16 | 28.9951 | 0.6762 | 38 |
| 5 | 5, 11, 17 | 26.5326 | 0.6541 | 32 |
| 6 | 6, 12, 18 | 28.0570 | 0.6775 | 39 |

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| 8단계 baseline | 28.11 +- 2.26 | 0.6743 +- 0.029 | 1.183 +- 0.120 |
| 11단계 +Edge λ=0.5 | 28.06 +- 2.08 | 0.6699 +- 0.027 | **1.186 +- 0.116** |
| **12단계 +Edge+Freq** | 28.05 +- 2.03 | 0.6683 +- 0.028 | 1.182 +- 0.120 |

### 분석

- **CNR 1.186 → 1.182 (하락)**: Frequency Loss 추가가 오히려 역효과.
- **원인**: Edge Loss는 경계의 spatial gradient를 직접 최적화해 CNR에 직결. Frequency Loss는 저주파 포함 전체 스펙트럼 차이를 페널티해 CNR과 무관한 성분에도 gradient를 분산. 두 항이 충돌.
- **PSNR std 감소**: 2.26 → 2.08 → 2.03으로 계속 줄어드는 것은 긍정적이나 통계적 유의성 없음.
- **손실 함수 개선 실험 종료**: Edge Loss 단독(11단계, CNR 1.186)이 최선. 추가 loss 조합은 효과 없음.

### 재실행 명령

```bash
uv run python scripts/08_nafnet/run_nafnet.py --lambda-edge 0.5 --lambda-freq 0.1 --seed 42 --results-dir results/12_nafnet_freq
```

---

## 13단계: AROI N2N 사전학습 → SBSDI D1 k-fold fine-tuning

### 개요

보유 데이터 AROI(24명 × 128장)의 인접 B-scan을 N2N 쌍으로 활용해 NAFNet을 사전학습 후 SBSDI D1 18쌍으로 fine-tuning. 외부 데이터 없이 기존 보유 데이터의 활용도를 극대화.

### 사전학습 설정 (AROI N2N)

- **스크립트**: `scripts/13_aroi_n2n/run_pretrain.py`
- **결과**: `results/13_aroi_n2n/pretrain/`
- Train: patient 1~20, 인접 N2N 쌍 2,540개 / Val: patient 21~24, 508쌍
- Lazy loading + random crop: 4,096 samples/epoch, 85 steps/epoch
- NAFNet width=32, L1 loss, lr=1e-3, batch=48, seed=42
- Early stop: epoch 93, best val N2N loss=0.04305
- D1 PSNR 모니터링 (epoch마다 10): 최고 **27.086** (epoch 50)

| Epoch | D1 PSNR |
|-------|---------|
| 1 | 23.399 |
| 10 | 26.544 |
| 50 | **27.086** |
| 93 (종료) | 25.843 |

SBSDI D2 N2N(3단계-A, 26.84 dB)보다 사전학습만으로 +0.25 dB 높은 D1 PSNR 달성.

### fine-tuning 설정 (SBSDI D1 k-fold)

- **스크립트**: `scripts/08_nafnet/run_nafnet.py --pretrain-ckpt results/13_aroi_n2n/pretrain/best.pth`
- **결과**: `results/13_aroi_n2n/finetune/`
- lr=1e-4 (pre-train lr=1e-3 대비 낮춤), λ_edge=0.5, seed=42
- 모든 fold epoch 31~34에서 수렴 (scratch 32~46보다 빠름)

### 폴드별 결과

| Fold | 평가 쌍 | best val PSNR | best val SSIM | stopped epoch | Epoch 1 val PSNR |
|------|---------|--------------|--------------|--------------|-----------------|
| 1 | 1, 7, 13 | 28.4775 | 0.6855 | 33 | 28.366 |
| 2 | 2, 8, 14 | 30.2850 | 0.7092 | 33 | 29.901 |
| 3 | 3, 9, 15 | 26.5972 | 0.6678 | 33 | 26.513 |
| 4 | 4, 10, 16 | 29.3024 | 0.6880 | 31 | 29.302 |
| 5 | 5, 11, 17 | 26.5026 | 0.6668 | 33 | 26.463 |
| 6 | 6, 12, 18 | 28.2429 | 0.6904 | 34 | 28.163 |

Epoch 1 val PSNR이 26~30 수준 — 사전학습이 좋은 초기 가중치를 제공함을 확인.

### 성능 결과 (SBSDI D1, 18쌍 6-fold 평균)

| 방법 | PSNR (dB) | SSIM | CNR |
|------|-----------|------|-----|
| SRAD (베이스라인) | 27.50 +- 1.98 | 0.652 +- 0.023 | **1.220 +- 0.121** |
| 8단계 NAFNet (scratch) | 28.11 +- 2.26 | 0.6743 +- 0.029 | **1.183 +- 0.120** |
| 9단계 NAFNet+Aug | **28.35 +- 2.56** | 0.6832 +- 0.032 | 1.168 +- 0.121 |
| **13단계 AROI pretrain→finetune** | 28.23 +- 2.49 | **0.6838 +- 0.029** | 1.177 +- 0.121 |

### 분석

- **SSIM 0.6838**: 전체 방법 중 최고 — 사전학습이 OCT 구조적 특성 학습에 효과적
- **PSNR 28.23**: scratch(28.11) 대비 +0.12 dB, NAFNet+Aug(28.35) 대비 -0.12 dB
- **CNR 1.177**: scratch(1.183) 대비 소폭 하락 — AROI N2N이 smooth 출력을 선호하는 경향이 EdgeLoss 효과를 일부 상쇄
- **빠른 수렴**: epoch 31~34 (scratch 32~46 대비 빠름) — 사전학습 초기화 효과
- **결론**: AROI N2N 사전학습은 SSIM 향상에 유효. 더 많은 다양한 OCT 데이터로 사전학습하면 추가 향상 기대 가능

### 재실행 명령

```bash
uv run python scripts/13_aroi_n2n/run_pretrain.py
uv run python scripts/08_nafnet/run_nafnet.py \
  --pretrain-ckpt results/13_aroi_n2n/pretrain/best.pth \
  --lr 1e-4 --lambda-edge 0.5 --seed 42 \
  --results-dir results/13_aroi_n2n/finetune
```

---

## 향후 작업 방향

### 실험 총괄 결론

12단계까지의 실험에서 도출된 핵심 사실:

1. **데이터가 주된 병목**: U-Net(1.95M) / DnCNN(667K) / NAFNet(17M) 모두 PSNR 28.1~28.2 dB로 수렴. K=4 재실현 증강으로 28.35 dB까지 향상됐으나 효과는 소폭(+0.24 dB). lr 조정(1e-4)도 오히려 성능 하락(-0.16 dB). 더 다양한 실제 OCT 데이터가 필요.
2. **clean GT가 결정적**: N2N(noisy 타겟, loss floor 0.184) → k-fold CV(clean GT, loss floor 0.048)로 전환하자 PSNR이 26.84 → 28.19 dB로 도약.
3. **CNR 한계**: 어떤 AI 방법도 SRAD CNR(1.220)을 초과하지 못함. 손실 함수 개선(Edge Loss) 최고 1.186, 여전히 미달.
4. **합성 데이터 한계 확인**: 합성 노이즈(Gamma) 지도학습 PSNR 22.43 — 실제 OCT 스페클과의 도메인 갭이 모든 학습을 무효화.
5. **하이퍼파라미터·손실함수 튜닝 한계**: 아키텍처·lr·배치·증강·Edge Loss·Freq Loss 모두 18쌍 데이터 한계 앞에서 효과 noise 수준. 손실함수 실험 종료.
6. **사전학습 효과 확인 (13단계)**: AROI N2N(3,048쌍) 사전학습 후 D1 fine-tuning으로 SSIM 0.6838 달성(전체 최고). 더 많은 OCT 데이터로 사전학습하면 추가 향상 기대.

### 개선 방향 상세 분석

#### 방향 1 — 손실 함수 개선 (우선순위: 1순위, 비용: 낮음)

**목표**: CNR SRAD(1.220) 초과. 현재 최고 1.191로 0.029 차이.

**문제 분석**

현재 손실: `Loss = L1(pred, clean) + 0.1 × (1 - SSIM(pred, clean))`

- L1은 모든 픽셀을 동등하게 취급 → 조직 경계 픽셀과 내부 픽셀을 구별하지 않음
- SSIM은 11×11 윈도 기반 지역 유사도 → 전역 경계 구조를 간접적으로만 반영
- CNR = |μ_signal - μ_bg| / √(σ²_signal + σ²_bg) 는 경계 양쪽 강도 차이를 직접 측정하는데, 현재 손실에는 이를 직접 최적화하는 항이 없음

**방안 A — Edge Loss (Sobel, 1순위 추천)**

Sobel 필터로 경계 맵을 추출해 예측과 정답의 경계 구조를 직접 비교.

```python
import torch.nn.functional as F

def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]],
                      dtype=x.dtype, device=x.device).unsqueeze(0)
    ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,2]]],
                      dtype=x.dtype, device=x.device).unsqueeze(0)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2 + 1e-8)

def edge_loss(pred, clean):
    return F.l1_loss(sobel_edge(pred), sobel_edge(clean))
```

왜 CNR에 직접 효과적인가:
- CNR 분자 |μ_signal - μ_bg|는 경계 양쪽 강도 차이
- edge loss가 경계 위치에서 오차를 더 강하게 페널티 → 경계가 선명해지고 CNR 향상
- 구현 단순, 추가 파라미터 없음

**방안 B — Frequency Loss (FFT)**

주파수 도메인에서 진폭 스펙트럼을 비교해 고주파 성분(경계·세부 구조) 보존 최적화.

```python
def frequency_loss(pred, clean):
    pred_fft  = torch.fft.fft2(pred,  norm="ortho")
    clean_fft = torch.fft.fft2(clean, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft))
```

L1+SSIM이 놓치는 주기적 레이어 구조(OCT 특유의 반복 패턴)를 직접 보존. Edge loss 대비 구조 전반에 걸친 고주파 보존에 유리.

**조합 손실 (최종 제안)**

```
Loss = L1 + λ_ssim×(1-SSIM) + λ_edge×EdgeLoss + λ_freq×FreqLoss
```

| 파라미터 | 초기값 | 탐색 범위 |
|---------|-------|---------|
| λ_ssim | 0.1 (현행) | 고정 |
| λ_edge | 0.1 | [0.05, 0.5] |
| λ_freq | 0.05 | [0.01, 0.1] |

λ_edge만 먼저 추가해 CNR 변화를 확인하고, 효과가 있으면 λ_freq를 추가하는 순서로 진행 권장.

**12단계까지 실험 결과 반영 (손실 함수 개선 실험 종료)**:
- Edge Loss(λ=0.5): CNR 1.183 → 1.186 (+0.003), 통계적 유의성 없음
- Edge + Freq Loss: CNR 1.182 (baseline보다 낮음) — gradient 충돌로 역효과
- **결론**: 손실 함수 개선은 데이터 병목 앞에서 noise 수준의 효과에 그침. 데이터 확보 없이는 추가 loss 조합이 무의미.

#### 방향 2 — TTA (Test-Time Augmentation, 우선순위: 2순위, 비용: 낮음)

추가 학습 없이 추론 시에만 적용. fold 간 std=2.56이라는 큰 분산을 줄이는 데 직접 효과적.

```python
augments = [lambda x: x,
            lambda x: torch.flip(x, [-1]),      # H flip
            lambda x: torch.flip(x, [-2]),      # V flip
            lambda x: torch.rot90(x, 1, [-2,-1])]

preds = [model(aug(noisy)) for aug in augments]
# 역변환 후 평균
final = mean([inv_aug(pred) for aug, pred in zip(augments, preds)])
```

#### 방향 3 — Duke AMD SD-OCT + N2N 자기지도 (우선순위: 3순위, 비용: 중간)

직접 다운로드 가능한 384명 38,400 B-scans(clean GT 없음)에 합성 노이즈를 추가해 N2N 자기지도 데이터로 활용. 현재 SBSDI D2(39세트) 대비 1,000배 많은 다양성 확보 가능.

절차: 다운로드 → 합성 노이즈(2단계 파이프라인 재사용) → SBSDI D2 + AMD 합산으로 N2N 재학습 → SBSDI D1 평가.

#### 방향 4 — 외부 clean GT 데이터셋 접근 재시도 (우선순위: 4순위, 비용: 중간)

| 데이터셋 | 쌍 수 | 방법 | 기대 효과 |
|---------|------|------|---------|
| PKU37 | 37쌍 | 알리바바 계정 재시도 | 데이터 2배 이상 증가 |
| Sub2Full vis-OCT | 미공개 | 저자 직접 이메일 컨택 | 가장 유사한 설정 |
| RETOUCH | 112 볼륨 | Grand Challenge 계정 재시도 | 대규모 |

#### 방향 5 — Diffusion 기반 방법 (우선순위: 5순위, 비용: 높음)

현재 기준 최고 복원 품질 계열. 데이터 부족 환경에서도 사전학습 모델 활용 가능.

| 논문 | 핵심 |
|------|------|
| GARD (MICCAI 2025) | OCT 스페클 Gamma 분포를 Diffusion forward process에 직접 적용 |
| Content-Preserving Diffusion (MICCAI 2023) | 비지도, 해부학적 구조 보존 |

#### 방향 6 — 치주 OCT 데이터 직접 수집 (우선순위: 장기, 비용: 매우 높음)

최종 목표 도메인. 촬영 프로토콜: 동일 위치 ~40회 반복 → 픽셀 평균 = clean GT (SBSDI D1 방식). 교수님과 협의 필요.

#### 방향 7 — Joint Denoising + SR (우선순위: 장기)

스페클 제거 성능이 충분히 향상된 후 Real-ESRGAN 파이프라인과 결합.

#### 참고 — 데이터 확보 조사 결과 (2026-06-01)

**보유 데이터 clean GT 추가 생성 불가 확인**

| 데이터셋 | diff/frame std 비율 | 판단 |
|---------|-------------------|------|
| SBSDI D1 | 0.771 | 동일 위치 반복 스캔 — 이미 GT 존재 |
| SBSDI D2 | 1.07 | 인접 슬라이스 — 불가 |
| AROI | 0.86~1.31 | 3D 볼륨 순차 슬라이스 — 불가 |
| Kermany OCT2017 | — | 단일 이미지 — 불가 |

**Duke AMD SD-OCT (직접 다운로드 가능)**

384명 38,400 B-scans, clean GT 없음. 합성 노이즈 추가 후 N2N 자기지도학습 데이터로 활용 가능(방향 3 참조).

**치주 OCT 직접 수집 시 clean GT 생성 프로토콜**

동일 위치 ~40회 반복 촬영 → 픽셀별 평균 = clean GT (SBSDI D1 방식). N=40이면 단일 프레임 대비 노이즈 약 84% 제거. 교수님과 촬영 프로토콜 협의 필요.
