# AI 기반 OCT 치주질환 처리

## TODO 리스트
[] 자료 조사

## 자료 조사
- OCT란
- 오픈 데이터 탐색
- 스페클 노이즈 제거 방식
- super-resolution 이용한 해상도 증가
- OCT와 결합하여 성능을 높이는 모달리티 탐색


### OCT란

**OCT(광간섭단층촬영, Optical Coherence Tomography)**는 근적외선 파장의 빛을 이용해 생체 조직 내부의 단층 구조를 마이크로미터($\mu m$) 단위의 초고해상도로 획득하는 비침습적 광학 영상 기술입니다. 흔히 의료 분야에서 초음파 영상이 '음파의 에코 타임 딜레이'를 측정하는 것처럼, OCT는 **'빛의 후방 산란 및 간섭 현상'**을 이용하여 단층 영상을 재구성하므로 **'광학적 생검(Optical Biopsy)'**이라고도 불립니다.

#### 1. OCT의 핵심 원리 및 물리적 메커니즘

① 낮은 가간섭성 간섭계 (Low-Coherence Interferometry)
빛의 속도는 초당 약 30만 km로 너무 빠르기 때문에 일반적인 센서로는 조직 내부에서 반사되어 돌아오는 빛의 시간 차이(Time-of-flight)를 직접 측정할 수 없습니다. 이를 극복하기 위해 OCT는 **마이켈슨 간섭계(Michelson Interferometer)** 구조를 활용합니다.

*   광원에서 나온 광선이 광분할기(Beam Splitter)를 통해 **참조광(Reference Arm)**과 **측정광(Sample Arm)**으로 나뉩니다.
*   참조광은 고정되거나 정밀하게 움직이는 거울에서 반사되고, 측정광은 생체 조직(치주 조직) 내부의 각 레이어 경계면에서 후방 산란되어 돌아옵니다.
*   두 경로를 거쳐 돌아온 빛이 다시 만나 합쳐질 때, 두 빛의 경로 차이가 광원의 **가간섭 길이(Coherence Length)** 이내일 때만 물리적인 간섭 무늬(Interference Pattern)가 발생합니다. 이 무늬의 진폭과 주파수를 분석하여 깊이 정보를 계산합니다.

② 분해능(Resolution)의 물리적 정의
OCT의 가장 큰 장점 중 하나는 **축 방향(Axial) 해상도와 측 방향(Lateral) 해상도가 서로 독립적**이라는 점입니다.

*   **축 방향 해상도 ($\Delta z$):** 광원의 물리적 특성인 중심 파장($\lambda_0$)과 대역폭($\Delta \lambda$)에 의해서만 결정됩니다. 광원의 대역폭이 넓을수록(즉, 가간섭 길이가 짧을수록) 해상도가 수 마이크로미터 수준으로 정밀해집니다.
$$\Delta z = \frac{2\ln 2}{\pi} \cdot \frac{\lambda_0^2}{\Delta \lambda}$$
*   **측 방향 해상도 ($\Delta x$):** 일반적인 광학계와 마찬가지로 대물렌즈의 개구수(Numerical Aperture, NA)와 초점 거리에 의해 결정되며, 빔 스폿의 최소 크기와 직결됩니다.

③ 아키텍처의 진화: TD-OCT에서 FD-OCT로
*   **Time-Domain OCT (TD-OCT):** 초기 가동 방식으로, 참조 거울을 기계적으로 이동시키며 깊이 별 반사율을 측정(A-scan)했습니다. 스캔 속도가 느리다는 단점이 있습니다.
*   **Fourier-Domain OCT (FD-OCT):** 현대의 표준 방식으로, 참조 거울을 고정하는 대신 간섭광의 스펙트럼 자체를 획득하여 분광 정보에 **고속 푸리에 변환(FFT)**을 적용해 깊이 정보를 한 번에 계산합니다. 속도와 신호 대 잡음비(SNR)가 수십~수백 배 향상되었습니다. 
    *   치과 및 연조직 생체 검사에는 대개 파장을 고속으로 가변하는 레이저를 사용하는 **SS-OCT (Swept-Source OCT)** 방식이 투과 깊이와 신호 안정성 면에서 주로 채택됩니다.

A Scan: 깊이 면 단일 데이터(1차원) / B Scan: 연속된 A Scan 데이터(2차원)

관련 링크
- https://www.visionsystem.kr/technical-info?tpf=board/view&board_code=1&code=639
- https://brunch.co.kr/@gazezaet/2
- https://www.amc.seoul.kr/asan/healthinfo/management/managementDetail.do?managementId=417


### 오픈 데이터 탐색

#### 1순위: 치주 및 치과(Dental) 영역 OCT 오픈 데이터 / 소스
치과 AI 분야의 한계점 중 하나는 데이터 빈곤(Data Poverty)입니다. 최신 학술 저널인 *Journal of Dental Research(2024)*의 분석에 따르면 공개된 치과 AI 데이터셋의 60% 이상이 파노라마 X-ray나 CBCT에 편중되어 있으며, 치주 OCT 데이터는 개별 연구팀이 비공개(In-house)로 처리하는 경우가 많습니다. 대안으로 활용할 수 있는 프록시 소스 및 프레임워크 코드는 다음과 같습니다.

*   **Transformer 기반 Dental OCT 병변 탐지 방법론 (Methodology)**
    *   **설명:** 치과용 OCT 스캔에서 발생하는 스페클 노이즈를 극복하고 ViT(Vision Transformer) 및 Attention Gate를 활용해 치아 결손 및 미세 레이어를 탐지하는 모델 구조입니다. 하드웨어 한계를 트랜스포머 아키텍처로 돌파하는 코드 베이스의 참고용으로 적합합니다.
    *   **관련 논문 정보:** [Lesion Detection in OCT with Transformer-Enhanced Detector (MDPI 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10671998/)
*   **치조골 및 구강 구조 탐지용 Open-Set Detector (코드 소스)**
    *   **설명:** MICCAI 2024에 공개된 뼈 및 구강 구조 세그멘테이션을 위한 비전-언어 오픈셋 탐지기 프레임워크입니다. 치과 영역 데이터 전처리와 멀티모달 프롬프팅 구조를 차용할 수 있습니다.
    *   **소스코드 링크:** [GitHub - xmed-lab/FD-SOS](https://github.com/xmed-lab/FD-SOS)
*   **ENPAT: YOLOv8 기반 치주 질환 스크리닝 (인접 도메인 데이터)**
    *   **설명:** OCT 데이터는 아니지만, 구강 내 임상 이미지(Intraoral Photos)를 기반으로 치은염 지수(MGI) 및 치간유두 충전 지수(PFI)를 자동 세그멘테이션하고 등급을 분류하는 최신 치주 질환 오픈 프로젝트입니다. Labelme 양식의 치주 조직 데이터 주석 처리(Annotation) 파이프라인을 벤치마킹하기 좋습니다.
    *   **프로젝트 및 데이터 정보:** [YOLOv8-based Esthetic-zonal Periodontal Assessment Tool](https://peerj.com/articles/cs-3229/)

#### 2순위: 안구(Ophthalmology) 관련 retinal OCT 오픈 데이터셋

안과 영역은 인공지능 기반 OCT 분석이 전 세계적으로 가장 활발한 분야입니다. 해상도 향상(Super-Resolution) 및 디노이징(Denoising) 알고리즘을 테스트하기 위한 **대규모 소스 데이터 공급처**로 활용됩니다.

현재 보유 중인 데이터셋(Kermany OCT2017, AROI, MedSegBench) 구조 및 상세 설명 → [data/data_description.md](data/data_description.md)

#### 3순위: 기타 생체 조직 및 기술적 OCT 신호처리 오픈소스

OCT 하드웨어 원천 신호 처리, 스페클 노이즈 제어 및 3D 그래픽 변환 솔루션을 제공하는 툴킷 위주입니다.

*   **Deep Learning Toolbox for Cortical OCT**
    *   **설명:** 브라운 대학교 연구팀 등이 개발한 툴킷으로, 생체 조직(뇌 피질 등)의 OCT 스캔 이미지에서 스페클 노이즈를 제거(Enhancement)하고 미세 혈관 및 단면 경계를 딥러닝으로 자동 세그멘테이션하여 그래프 구조로 변환해 주는 딥러닝 파이프라인 소스입니다.
    *   **논문 및 기술 링크:** [Biomedical Optics Express - Deep Learning Toolbox for OCT](https://opg.optica.org/abstract.cfm?uri=boe-11-12-7325)


### Speckle 노이즈 제거 방식

OCT에서 스페클 노이즈(Speckle Noise)는 레이저의 가간섭성 때문에 발생하는 물리적인 현상입니다. 조직 내부의 미세 구조물들이 빛을 무작위로 산란시키면서 위상차가 생기고, 이것이 무작위적인 보강/상쇄 간섭을 일으켜 알갱이 형태의 노이즈로 나타납니다. 
수학적으로 스페클 노이즈는 단순 더해지는 노이즈가 아니라 신호의 크기에 비례하여 곱해지는 **곱셈성 노이즈(Multiplicative Noise)** 모델을 따르기 때문에 제거하기가 매우 까다롭습니다.

$$I(x,y) = S(x,y) \cdot N_s(x,y) + N_a(x,y)$$
*(단, $I$는 획득된 영상, $S$는 실제 신호, $N_s$는 곱셈성 스페클 노이즈, $N_a$는 가산성 백색 노이즈)*

#### 1. AI를 사용하지 않는 방식 (Traditional Methods)

AI를 사용하지 않는 방식은 광학 하드웨어를 조절하는 방식(Hardware/Optical Compounding)과 이미지의 국소적 통계치나 픽셀 간 기하학적 유사성을 이용하는 디지털 신호 처리(DSP) 방식으로 분류됩니다.

-  ① 하드웨어 및 데이터 획득 기반 방식 (Compounding)
    *   **프레임 에버리징 (B-scan Frame Averaging):** 동일한 위치를 고속으로 여러 번(예: 20~50회) 촬영한 뒤 평균을 내는 방식입니다. 스페클 노이즈는 무작위적(Stochastic) 특성을 가지므로, 프레임을 평균 내면 노이즈는 상쇄되고 구조적 신호만 남습니다. **현재 임상 장비에서 가장 신뢰하는 기준(Ground Truth)**이지만, 촬영 시간이 늘어나 환자의 움직임 Artifact가 생길 수 있습니다. ([Schmitt et al., JBO 1999](https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-4/issue-1/0000/Speckle-in-optical-coherence-tomography/10.1117/1.429925.full))
    *   **공간/주파수 컴파운딩 (Spatial/Frequency Compounding):** 빛의 입사 각도를 미세하게 다르게 하거나(공간), 서로 다른 파장 대역의 빛을 분할 조사하여(주파수) 얻은 영상을 결합합니다. 각기 다른 스페클 패턴을 오버랩시켜 상쇄하는 원리입니다. ([Pircher et al., JOSAA 2003](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-20-12-2247))

-  ② 공간 도메인 필터링 (Spatial Domain Filtering)
    *   **국소 적응형 필터 (Local Adaptive Filters):** 이미지의 특정 윈도우(예: 3x3, 5x5) 내에서 평균과 분산을 계산하여 노이즈를 줄입니다.
        *   *[Lee](https://ieeexplore.ieee.org/document/4766994), Kuan, [Frost](https://ieeexplore.ieee.org/document/4767350) 필터:* 픽셀 주변의 국소 통계치를 기반으로, 변화가 적은 평탄한 구역은 강하게 부드럽게(Smoothing) 만들고, 경계면(Edge)은 필터링을 약하게 하여 구조를 보존합니다. 단, 연산은 빠르지만 경계면이 다소 흐려지는 블러링(Blurring) 현상이 있습니다.
    *   **이방성 확산 필터 (Anisotropic Diffusion / SRAD):** 픽셀 간 밝기 기울기(Gradient)를 분석하여 경계면과 수평인 방향으로만 확산(Smoothing)을 진행합니다. 특히 **[SRAD(Speckle Reducing Anisotropic Diffusion)](https://ieeexplore.ieee.org/document/1042311)**는 스페클의 곱셈성 특성을 미분 방정식에 반영하여, 치은선이나 망막 레이어의 경계면을 매우 날카롭게 유지하면서 내부 노이즈만 지워내는 탁월한 능력을 보여줍니다.

-  ③ 비국소 및 변환 도메인 필터링 (Advanced DSP)
    *   **[비국소 평균 필터 (Non-Local Means, NLM)](https://epubs.siam.org/doi/10.1137/040616024):** 특정 윈도우 안만 보는 것이 아니라, 이미지 전체(또는 넓은 영역)에서 **기하학적으로 유사한 패턴을 가진 패치(Patch)**들을 찾아내어 이들의 가중 평균을 구합니다. OCT 특유의 반복적인 레이어 구조를 보존하는 데 매우 효과적이지만, 연산량이 극도로 많다는 단점이 있습니다.
    *   **웨이블릿 임계화 (Wavelet Thresholding):** 영상을 주파수 영역(Wavelet Domain)으로 변환하면 고주파 영역에 스페클 노이즈 계수들이 집중됩니다. 특정 임계값(Threshold) 이하의 계수들을 제거하거나 줄인 뒤 역변환(Inverse Transform)하는 방식입니다. ([Donoho & Johnstone, Biometrika 1994](https://academic.oup.com/biomet/article-abstract/81/3/425/256924))

#### 2. AI를 사용하는 방식 (Deep Learning / AI Methods)

딥러닝을 사용하는 방식은 물리적 수식을 계산하는 대신, 대규모 고품질 데이터를 백본 네트워크(U-Net, Transformer 등)에 학습시켜 노이즈 제거 능력을 내재화(Implicit)하는 방식입니다.

- ① 지도 학습 기반 방식 (Supervised Denoising)
    *   **쌍을 이룬 데이터 학습 (Paired Data Learning):**
        *   *데이터셋 구성:* 노이즈가 심한 단일 스캔 이미지(Low SNR Input)와 프레임 에버리징을 통해 얻은 깨끗한 이미지(High SNR Ground Truth)를 쌍으로 매칭합니다.
        *   *모델 아키텍처:* **[DnCNN (Denoising CNN)](https://arxiv.org/abs/1608.03981)**이나 **[U-Net](https://arxiv.org/abs/1505.04597)** 구조를 주로 사용하며, 최근에는 전역적 문맥을 파악하는 트랜스포머 기반의 **[Restormer](https://arxiv.org/abs/2111.09881)**나 **[SwinIR](https://arxiv.org/abs/2108.10257)** 등이 활용됩니다.
        *   *손실 함수:* 단순 L1, L2 Loss만 사용하면 평균적인 값으로 수렴하여 이미지가 흐려지기 때문에, 사람의 시각적 인지 경계를 보존하는 **[VGG 기반 Perceptual Loss(지각적 손실)](https://arxiv.org/abs/1603.08155)**나 **[SSIM(Structural Similarity) Loss](https://ieeexplore.ieee.org/document/1284395)**를 결합하여 치주 연조직의 미세한 텍스처를 살려냅니다.

-  ② 자가 학습 및 비지도 학습 기반 방식 (Self-Supervised / Unsupervised): 임상 환경에서 완벽히 깨끗한 Ground Truth(GT) 이미지를 얻기 어려울 때 사용하는 돌파구입니다.
    *   **[Noise2Noise (N2N)](https://arxiv.org/abs/1803.04189):** 완전히 깨끗한 타깃 영상이 없더라도, **동일한 부위를 촬영한 노이즈 낀 영상 2장**만 있으면 학습이 가능합니다. 두 영상의 스페클 노이즈는 통계적으로 독립적이라는 점을 이용하여, AI가 서로의 노이즈를 예측하는 과정에서 노이즈의 평균값(0)을 찾아내어 신호만 남기게 만듭니다.
    *   **[Noise2Void (N2V)](https://arxiv.org/abs/1811.10980) / Blind Spot Network:** 단 한 장의 노이즈 영상으로도 학습이 가능합니다. 특정 픽셀의 값을 주변 픽셀들을 통해 예측(Blind-spot)하도록 모델을 제어하면, 무작위로 발생하는 스페클 노이즈는 예측할 수 없고 연속적인 조직 구조 신호만 예측할 수 있게 되어 자연스럽게 노이즈가 걸러집니다.
    *   **[CycleGAN](https://arxiv.org/abs/1703.10593) 기반 Domain Translation:** 노이즈가 있는 이미지 그룹(Domain A)과 노이즈가 없는 타깃 그룹(Domain B)이 서로 1:1로 매칭되지 않더라도, 두 도메인 간의 스타일 전환 메커니즘을 통해 노이즈 특성만 지워내는 비지도 방식입니다.

- ③ 생성형 모델 기반 방식 (Generative Priors)
    *   **[Diffusion Plug-and-Play (PnP) Priors](https://arxiv.org/abs/2209.14687):** 최근 가장 각광받는 기법으로, 깨끗한 의료 영상의 분포를 먼저 학습한 확산 모델(Diffusion Model)을 노이즈 제거 과정의 역방향(Reverse Process) 조건으로 주입합니다. 단순 매핑보다 누락된 레이어 구조를 복원하는 능력이 매우 뛰어나며, 초해상도(Super-Resolution) 작업과 동시에 수행하기 적합합니다.

#### 3. OCT 스페클 제거 최신 연구 (2022~2025)

위 범용 방법들을 OCT 스페클 특성에 맞게 특화한 최신 논문들입니다.

**자가지도 / 비지도 계열**

| 논문 | 연도 | 핵심 |
|------|------|------|
| [Sub2Full](https://arxiv.org/abs/2401.10128) | 2024 | 스펙트럼을 절반으로 분리해 noisy-clean 쌍 생성. 별도 clean 데이터 불필요. N2N·N2V보다 우수. *Optics Letters* |
| [Self2Self (S2Snet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10890874/) | 2024 | Dropout 기반 단일 노이즈 이미지 자가지도 학습 |
| [SSN2V](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10299996/) | 2023 | 스페클 분리(Speckle Split) 기반 비지도 제거. Noise2Void 확장 |
| [Slice-to-Slice Registration](https://arxiv.org/abs/2311.15167) | 2023 | OCT A-scan 간 강한 픽셀 상관관계 문제를 슬라이스 정합으로 해결 |

**Diffusion 모델 계열**

| 논문 | 연도 | 핵심 |
|------|------|------|
| [GARD](https://arxiv.org/abs/2509.10341) | 2025 | OCT 스페클의 감마 분포를 Diffusion에 직접 적용. 가우시안 가정 탈피. *MICCAI 2025* |
| [Content-Preserving Diffusion](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_62) | 2023 | 비지도 Diffusion으로 AS-OCT 스페클 제거. 해부학적 구조 보존 강조. *MICCAI 2023* |
| [Unsupervised Diffusion Retinal OCT](https://arxiv.org/abs/2201.11760) | 2022 | 망막 OCT에 Diffusion 확률 모델 최초 적용 |

**Transformer / 기타**

| 논문 | 연도 | 핵심 |
|------|------|------|
| [Self-supervised Transformer NLM](https://www.sciencedirect.com/science/article/abs/pii/S1746809422008023) | 2023 | Transformer 기반 NLM 재설계. 전통 NLM 대비 성능 향상 |
| [Interference Fringe Enhancement](https://www.nature.com/articles/s42003-023-04846-7) | 2023 | OCT 원시 간섭 프린지 신호를 딥러닝으로 직접 처리. *Nature Comm. Biology* |

#### 4. 성능 트렌드 및 방법론 비교

2024년 systematic review([PMC11050869](https://pmc.ncbi.nlm.nih.gov/articles/PMC11050869/)) 기준, 연구 동향 및 성능 계층은 다음과 같습니다.

| 계층 | 계열 | 점유율 / 성능 | 비고 |
|------|------|-------------|------|
| 1 | **Diffusion (물리 기반)** | 최고 복원 품질 | 감마 분포 반영 시 OCT 특성 직접 모델링 |
| 2 | **GAN 기반** | 연구의 44% 차지, PSNR 26~158 dB | 현 주류. hallucination 위험 |
| 3 | **자가지도 (Sub2Full 등)** | 실용성 1위 | clean 데이터 불필요, N2N·N2V 능가 |
| 4 | **지도학습 CNN/Transformer** | 성숙 단계 | DnCNN 13%, clean GT 의존이 약점 |

**연도별 흐름**

```
2018~2021  GAN 지도학습 지배
2022~2023  N2N·N2V 변형 + 비지도 확산 모델 등장
2023~2025  Diffusion × 물리 기반 노이즈 모델(Gamma) + 자가지도 혼합 → 최전선
```

**핵심 한계**: 연구마다 평가 데이터셋·지표(PSNR, SSIM, CNR, ENL)가 달라 공정 비교가 어렵고, 임상 검증이 아직 부족합니다.

**이 프로젝트에서의 시사점**: AROI 데이터에 합성 스페클을 추가해 학습한다면 **자가지도(Sub2Full·SSN2V) + Diffusion PnP** 조합이 clean 데이터 없이 가장 강한 성능을 낼 수 있는 현실적 선택입니다.

#### 스페클 노이즈 제거 데이터셋

학습 방식별 데이터 요구사항 및 보유 데이터 활용 방안 → [data/data_description.md](data/data_description.md)

외부 공개 데이터셋:
- [PKU37](https://tianchi.aliyun.com/dataset/133217) — 37쌍 noisy-clean (지도학습용)
- [D1](https://www.nature.com/articles/s41598-019-51062-7) — 18쌍 noisy-clean, Bioptigen SDOCT (저자 요청)
- [Sub2Full vis-OCT](https://github.com/PittOCT/Sub2Full-OCT-Denoising) — 반복 스캔 쌍
- [RETOUCH](https://retouch.grand-challenge.org/) — 112 볼륨, 3개 장비 (Noise2Noise용)
- [ODTiD](https://doi.org/10.3886/E137701V3) — 시신경 유두 SD-OCT 242장

### OCT와 결합하여 성능을 높이는 모달리티 탐색

OCT는 마이크로미터 단위의 초고해상도를 자랑하지만, 생체 조직 내부로 들어갈수록 빛이 무작위로 흩어지는 **광산란(Optical Scattering) 현상** 때문에 투과 깊이가 1.5~2mm 내외로 제한됩니다. 치주 질환 진단 시 깊은 치주낭(Periodontal pocket)의 바닥이나 심부 치조골(Alveolar bone)까지 관찰하기에는 한계가 있습니다.

이러한 물리적 한계를 극복하기 위해 **구조 정보(OCT)에 생화학적·기능적·생리적 정보를 더해주는 멀티모달리티(Multimodality) 융합 시스템**이 활발히 연구되고 있습니다. AI 알고리즘을 적용할 때도 단일 모달리티보다 다중 모달리티 텐서를 인풋으로 넣었을 때 진단 정확도가 비약적으로 상승합니다. 대표적인 3가지 결합 모달리티와 연구 근거 링크를 나열합니다.

#### 1. OCT + 광음향 영상 (Photoacoustic Imaging, PAI / PAM)
*   **융합 원리:** 광음향은 "빛을 쏘고 소리를 듣는(Light in, Sound out)" 기술입니다. 특정 파장의 레이저를 조직에 조사하면, 염증 부위의 헤모글로빈이나 혈관이 빛을 흡수해 순간적으로 열팽창을 하며 초음파(Acoustic wave) 신호를 발생시킵니다. 
*   **시너지 효과:** OCT가 잇몸 연조직의 형태학적 층(Layer) 구조와 치은 두께를 정밀하게 묘사하면, PAI는 치주염으로 인해 나타나는 **미세 혈관의 확장, 신생 혈관 형성, 혈중 산소 포화도 변화 등 '초기 염증 반응'**을 매핑합니다. 
*   AI 모델 구조 적용 시, 영상의 픽셀 단위 경계선(OCT 변수)과 염증 활성도 데이터(PAI 변수)를 하나의 고차원 맵으로 정합(Co-registration)하여 세그멘테이션 백본의 인풋으로 활용합니다.

학술 근거 및 관련 링크
*   **치주 질환 최초 적용 임상 연구:** 인간을 대상으로 음식물 등급의 잉크를 조영제로 활용해 치주낭 깊이와 치은 구조를 완벽히 시각화한 선도적 연구입니다. 
    *   [Photoacoustic Imaging for Monitoring Periodontal Health - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226559/)
*   **다중 광학 플랫폼 통합 논문:** PAM, OCT, 광도플러(ODT), 공초점 형광 현미경을 하나의 광학 가이드 기반 플랫폼으로 통합해 구조와 기능을 동시 스캔하는 메커니즘을 규명했습니다.
    *   [Integrated Multimodal Photoacoustic Microscopy with OCT - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6363202/)

#### 2. OCT + 고주파 초음파 영상 (High-Frequency Ultrasound, HFUS)
*   **융합 원리:** 초음파는 "소리를 쏘고 소리를 듣는" 방식으로, 매질의 밀도 차이에 따른 에코 신호를 기반으로 작동합니다. 
*   **시너지 효과:** OCT는 표층(치은 내부 1.5mm 이내)을 극도로 선명하게 잡아내지만 치조골 내부 깊숙한 곳은 보지 못합니다. 반면 고주파 초음파(예: 40MHz 이상)는 해상도는 OCT보다 조금 낮지만 수 센티미터 깊이까지 투과합니다. 따라서 **AI가 표층 연조직은 OCT 데이터로 처리하고, 하방의 단단한 치조골 소실(Alveolar Bone Loss) 지점과 깊은 치주낭의 기저부는 초음파 데이터로 상호 보완**하여 하나의 완성된 치주 지형도를 그리게 만듭니다.

학술 근거 및 관련 링크
*   **치과용 OCT 진단 능력 및 메커니즘 뷰:** 치과 영역에서 초음파 개념과 매칭하여 치조골, 치아 경조직, 연조직 변화를 정량화하는 하이브리드 진단 장치로서의 유효성을 총평한 종설 논문입니다.
    *   [Dental Optical Coherence Tomography Overview - MDPI](https://www.mdpi.com/1424-8220/13/7/8928)

#### 3. OCT + 정량 광형광 영상 (Quantitative Light-induced Fluorescence, QLF)
*   **융합 원리:** 특정 청색광을 치아 및 잇몸에 조사하면 생체 조직이나 세균의 대사 산물이 고유한 형광(Fluorescence)을 발산하는 원리입니다.
*   **시너지 효과:** 치주 질환의 근본적 원인은 잇몸 밑에 숨겨진 **설하 치석(Subgingival calculus)과 박테리아 바이오필름(세균막)**입니다. QLF나 공초점 형광 기술은 세균 덩어리가 유발하는 포르피린 성분을 감지해 생화학적 위치를 붉은색 형광으로 짚어냅니다. 이를 3차원 OCT 구조 데이터와 융합하면, AI가 단순 잇몸 부종을 넘어 "어느 깊이에 치석과 세균막이 얼마나 쌓여 염증을 유발했는지"를 부피(Volume) 단위로 자동 추적하고 정량화할 수 있습니다.

학술 근거 및 관련 링크
*   **초기 병변 및 바이오필름 탐지 융합 연구:** 정량 광형광 기술(QLF)의 탈회/세균 탐지 신호와 en-face OCT 시스템의 반사율 손실 지표를 선형 결합하여, 초기 치과 병변의 깊이와 면적을 정밀 측정해 낸 연구 결과입니다.
    *   [Optical Coherence Tomography Correlated with Functional Fluorescence Imaging - University of Liverpool Repository](https://livrepository.liverpool.ac.uk/1143/1/Amaechi2002OCTcorrelates.pdf)


### super-resolution 이용한 해상도 증가
해당 내용 스킵


## 데이터
보유 데이터셋 구조 및 상세 설명 → [data/data_description.md](data/data_description.md)
