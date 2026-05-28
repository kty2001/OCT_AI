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

## 진행 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 1단계 | 전통적 방법 베이스라인 측정 | 완료 |
| 2단계 | 합성 스페클 노이즈 생성 파이프라인 | 완료 |
| 3단계 | 자가지도 학습 베이스라인 구현 (N2N 프레임쌍, Sub2Full 방식) | 완료 |
| 4단계 | Joint Denoising + SR 모델 구현 (PSCAT, PnP-DM) | 미착수 |
