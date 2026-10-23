# DCS 2026 논문용 그림 캡션 (한국어)

모든 수치는 `outputs/` 아래의 저장된 실험 결과에서 그대로 가져온 값이며, 새로 생성하거나 재계산한 값은 없습니다.
그림 내부 라벨은 한글 폰트 신뢰성 문제로 영문으로 작성했고, 캡션은 아래 한국어 버전을 사용합니다.
표기: β = Sharp Softmax 역온도, λ = Entmax/PLADIS 혼합 계수(λ_entmax). 본 그림들에서 λ_sharp는 사용하지 않습니다.

---

## 그림 1 — 방법 개요 (fig1_method_overview.png / .pdf)

**캡션(삽입용):**
그림 1. Auffusion Cross-Attention에 적용한 Attention 조절 방법의 개요. 텍스트 인코더와 Diffusion 과정은 그대로 두고, U-Net의 텍스트 Cross-Attention 가중치 계산만을 Dense Softmax, Sharp Softmax(β), Entmax, PLADIS(λ)로 교체하였다. 우측의 토큰별 가중치 막대는 집중도 차이를 나타내는 개념도이며 측정값이 아니다.

**포함 근거(캡션 외):** 2쪽 분량에서 방법 설명 문단을 줄이고 네 조건의 수식 관계를 한 번에 보여 준다.

---

## 그림 2 — 핵심 정량 결과 (fig2_metric_delta.png / .pdf, 1단용 fig2_metric_delta_compact.png)

**캡션(삽입용):**
그림 2. Attention 조절 방법에 따른 Dense 대비 생성 성능 변화. 24개 진단 프롬프트 × 3개 시드에서 프롬프트 단위로 평균한 뒤 Dense(점선, 0)와의 차이를 나타냈으며, 오차 막대는 프롬프트 단위 부트스트랩(10,000회) 95% 신뢰구간이다. (a) CLAP 코사인 유사도, (b) AQAScore, (c) AudioBox Production Quality(PQ).

**포함 근거(캡션 외):** 표 1이 절대값을 제시한다면, 이 그림은 불확실성과 함께 "약한 조절은 변화가 작고, Entmax가 강해질수록 감소하며, PLADIS λ=2에서 감소 폭이 가장 크다"는 핵심 패턴을 한눈에 전달한다.

---

## 그림 3 — Mel-Spectrogram 비교, D01 (fig3_spectrogram_D01.png / .pdf)

**캡션(삽입용):**
그림 3. 동일한 프롬프트("A dog barking while rain falls in the background.")와 초기 노이즈(시드 2026)에서 Attention 방식에 따라 생성된 Mel-Spectrogram 비교. (a) Dense, (b) Sharp Softmax β=1.5, (c) Entmax λ=1, (d) PLADIS λ=2. 네 패널은 동일한 전처리(16 kHz, FFT 1024, hop 160, 128 mel, 0–8 kHz)와 동일한 dB 색상 범위를 사용하며, 각 패널 위의 수치는 해당 샘플의 CLAP·AQAScore·PQ 값이다. Attention 조건에 따라 생성된 시간-주파수 구조의 차이를 확인할 수 있다.

**포함 근거(캡션 외):** 다중 이벤트(전경+배경) 연구 질문에 해당하는 대표 사례를 정성적으로 보여 주되, 시드와 색상 범위를 고정하여 공정한 비교가 되도록 했다.

---

## 그림 3b (선택) — Mel-Spectrogram 비교, A05 (fig3b_spectrogram_A05.png)

**캡션(삽입용):**
그림 3b. 단일 이벤트 프롬프트("A wooden door slamming shut.", 시드 2026)에 대한 그림 3과 동일한 조건의 Mel-Spectrogram 비교. 전처리와 색상 범위는 그림 3과 동일하다.

**포함 근거(캡션 외):** 이전 실험에서 Sharp Softmax에 상대적으로 관대했던 개별 프롬프트의 예시이나, 지면상 선택 사항이며 D01이 우선이다. 유리한 수치를 이유로 D01을 대체해서는 안 된다.

---

## 그림 4 (선택) — 집중도 메커니즘 (fig_optional_concentration.png / .pdf)

**캡션(삽입용):**
그림 4. Sharp Softmax의 β 증가에 따른 (a) 조건부 Cross-Attention의 유효 Support(exp(H), 77개 텍스트 토큰 기준)와 (b) 평균 CLAP(24개 프롬프트)의 변화. 유효 Support는 16개 층 × 100 denoising 스텝의 조건부 CFG 절반에서 측정한 엔트로피로부터 계산하였다.

**포함 근거(캡션 외):** "집중도가 올라갈수록 정렬이 좋아지지 않는다"는 메커니즘 서술을 수치로 뒷받침하지만, 표·그림 2와 일부 중복되므로 지면이 남을 때만 포함한다.
