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
| 3단계 | 자가지도 학습 베이스라인 구현 (N2N 프레임쌍, Sub2Full 방식) | 완료 |
| 4단계 | Joint Denoising + SR 모델 구현 (PSCAT, PnP-DM) | 미착수 |

---

## 프로젝트 구조

```
OCT_AI/
├── data/
│   ├── Final_Publication_2013_SBSDI/   SBSDI D1 (noisy-clean 18쌍 + real 39세트)
│   ├── AROI/                           망막 OCT 레이어 세그멘테이션 (1,136장 주석)
│   ├── kaggle_RetinalOCTImages/        Kermany OCT2017 (84,484장)
│   ├── MedSegBench/                    낭성액 세그멘테이션 등 (npz)
│   └── data_description.md            데이터셋 상세 설명
├── scripts/
│   ├── 01_baseline/                    전통적 방법 베이스라인 (NLM, BM3D, SRAD)
│   └── 02_synthetic_noise/             합성 스페클 노이즈 생성 파이프라인
├── results/
│   ├── 01_baseline/                    베이스라인 지표 및 비교 이미지
│   ├── 02_synthetic_noise/             합성 학습 쌍 (6,136쌍) 및 검증 결과
│   └── result_description.md          단계별 결과 정리
└── pyproject.toml                      uv 의존성 관리
```

---

## 환경 설정

```bash
# 의존성 설치 (uv 필요)
uv sync
```

주요 의존 패키지: `numpy`, `pillow`, `scikit-image`, `scipy`, `bm3d`, `pandas`, `matplotlib`, `tqdm`

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

clean 이미지만 보유한 대규모 데이터셋(AROI, Kermany)에 물리 기반 스페클 노이즈를 합성하여 지도학습용 noisy-clean 쌍을 생성한다.

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

| 데이터셋 | 쌍 수 | 해상도 (H×W) | 비고 |
|---------|------|------------|------|
| AROI | 1,136 | 512×1024 | 주석 완료 B-scan, 90도 회전 보정 |
| Kermany OCT2017 | 5,000 | 496×512 | train 서브셋 (랜덤 샘플링, seed=42) |
| **합계** | **6,136** | — | `results/02_synthetic_noise/metadata.csv` |

실행:
```bash
# 파라미터 검증 (분포 플롯 생성)
uv run python scripts/02_synthetic_noise/validate_noise.py

# 전체 파이프라인 (캘리브레이션 + 합성)
uv run python scripts/02_synthetic_noise/run_synthetic.py --calibrate --max-kermany 5000
```

---

## 3단계: 딥러닝 디노이징 베이스라인 (미착수)

### 3-A: 사전학습 모델 직접 적용 (학습 없음)

별도 학습 없이 공개 가중치를 그대로 사용해 성능을 먼저 측정한다.
전통 방법 기준값(PSNR 27.50 dB)을 초과하는지 확인하여 fine-tuning 필요성을 판단한다.

**제로샷 Diffusion 기반** (추론만, 학습 불필요)

| 방법 | 논문 | 핵심 | VRAM |
|------|------|------|------|
| **DiffPIR** | [Zhu et al., CVPR 2023](https://arxiv.org/abs/2305.08995) | Diffusion prior를 Plug-and-Play로 주입. 가장 안정적 | 8 GB+ |
| **DDRM** | [Kawar et al., NeurIPS 2022](https://arxiv.org/abs/2201.00021) | 선형 열화 역문제를 Diffusion으로 zero-shot 복원 | 8 GB+ |
| **DPS** | [Chung et al., ICLR 2023](https://arxiv.org/abs/2209.14687) | Diffusion Posterior Sampling. 디노이징·SR 동시 처리 | 8 GB+ |

**일반 이미지 복원 (공개 가중치 직접 사용)**

| 모델 | 학습 도메인 | 태스크 | VRAM | 링크 |
|------|-----------|--------|------|------|
| **Restormer** | SIDD, DND (스마트폰) | 실사 디노이징 | 2 GB+ | [GitHub](https://github.com/swz30/Restormer) |
| **NAFNet** | SIDD (스마트폰) | 실사 디노이징 | 2 GB+ | [GitHub](https://github.com/megvii-research/NAFNet) |
| **SwinIR** | DIV2K | SR (x2/x4) + 디노이징 | 2 GB+ | [GitHub](https://github.com/JingyunLiang/SwinIR) |
| **Real-ESRGAN** | 합성 열화 이미지 | SR + 실사 열화 | 2 GB+ | [GitHub](https://github.com/xinntao/Real-ESRGAN) |

- 추론 속도: GPU 기준 이미지 1장에 0.1~1초 (Diffusion 계열은 수십 초~수 분)
- OCT 전용 학습이 아니므로 성능 편차 있음. SBSDI D1 18쌍으로 즉시 정량 평가 가능

### 3-B: 자가지도 학습 (OCT 전용, 소규모 학습)

clean Ground Truth 없이 OCT 이미지만으로 학습하는 방식.

- **학습 데이터**: SBSDI `For real experiments on Humans` (39세트, 450×450, clean 레퍼런스 없음)
- **평가 데이터**: SBSDI D1 18쌍 (전통 방법과 직접 비교)

| 우선순위 | 방법 | 핵심 |
|---------|------|------|
| 1순위 | **Sub2Full** (2024, *Optics Letters*) | B-scan FFT 스펙트럼을 두 절반으로 분리해 noisy쌍 생성 → N2N 학습. OCT 전용 |
| 2순위 | **SSN2V** (2023) | 스페클 분리 기반 Noise2Void 확장 |

**Sub2Full 학습 비용 예측**

| 항목 | 조건 | 예측 |
|------|------|------|
| 학습 데이터 | 39장 (450×450) | 매우 소규모 |
| 아키텍처 | 경량 U-Net (~7M params) | — |
| 필요 VRAM | 128×128 패치, batch 4 | **2~4 GB** |
| 학습 시간 (GPU) | RTX 3060 기준, 500 epoch | **30~60분** |
| 학습 시간 (CPU) | — | 4~8시간 |
| Google Colab | T4 무료 티어 | **사용 가능** |

---

## 4단계: Joint Denoising + SR 모델 구현 (미착수)

스페클 노이즈 제거와 Super-Resolution을 단일 파이프라인으로 처리하는 통합 모델을 구현한다.

OCT 획득 이미지의 열화 모델:
$$y = D(H(x) \cdot N_s + N_a)$$

($x$: 고해상도 clean, $H$: 측방향 블러, $D$: 다운샘플링)

순차(Sequential) 처리는 중간 오류가 누적되므로, 단일 네트워크로 동시 처리하는 Joint 방식을 채택한다.

| 우선순위 | 방법 | 아키텍처 | 성능 기준 |
|---------|------|---------|---------|
| 1순위 | **PSCAT** (2024, *BOE*) | 경량 Transformer, End-to-End Joint | PSNR 31.48 dB @ PKU37 ×4 |
| 2순위 | **PnP-DM** (2025) | Diffusion PnP, 역문제 | 구조 선명도 최고 |

학습 전략: 안과 데이터(AROI, Kermany) 사전학습 → 치주 OCT 데이터로 fine-tuning

---

## 자료 조사

### OCT란

**OCT(광간섭단층촬영, Optical Coherence Tomography)**는 근적외선 파장의 빛을 이용해 생체 조직 내부의 단층 구조를 마이크로미터($\mu m$) 단위의 초고해상도로 획득하는 비침습적 광학 영상 기술이다. 마이켈슨 간섭계 구조를 활용해 빛의 후방 산란 및 간섭 현상으로 단층 영상을 재구성한다.

- **A-scan**: 깊이 방향 단일 데이터(1차원)
- **B-scan**: 연속된 A-scan으로 구성된 단면 영상(2차원)
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
| SBSDI D1 | synthetic 18쌍 + real 39세트 | 디노이징 + SR | 있음 (synthetic) |

외부 공개 데이터셋 접근 시도 (실패):
- PKU37 (37쌍 noisy-clean) → 알리바바 클라우드 계정 요구
- Sub2Full vis-OCT → 데이터 비공개
- RETOUCH (112 볼륨) → Grand Challenge 계정 요구
- ODTiD (242장) → OPENICPSR 계정 요구

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

$$\text{CNR} = \frac{|\mu_\text{signal} - \mu_\text{background}|}{\sigma_\text{background}}$$

- OCT에서 조직 레이어(신호 영역)와 배경(잡음 영역) 간 대비를 잡음 수준으로 나눈 값
- **클수록** 레이어 경계가 선명하고 구조 식별이 쉬움
- PSNR·SSIM이 전체 픽셀을 고려하는 반면, CNR은 **임상적 진단 품질**에 더 직접적으로 대응

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

## 데이터

보유 데이터셋 구조 및 상세 설명 → [data/data_description.md](data/data_description.md)

단계별 실험 결과 정리 → [results/result_description.md](results/result_description.md)
