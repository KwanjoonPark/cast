# 최종 논문 그림·표 캡션 (한국어)

수치는 `paper_assets/final/tables/table1_pladis_guidance_full.csv`(원천: `outputs/cfg_sweep_v1/analysis/delta_table.csv`, `outputs/no_cfg_v1/analysis/delta_table.csv`)와 일치한다.
공통 조건: 24개 프롬프트 × 시드 3개, 프롬프트 단위(시드 평균) 짝지은 차이 Δ = 조건 − Dense, 프롬프트 단위 bootstrap 10,000회 95 % 신뢰구간.
방법 표기: Dense, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2 (λ=0 Dense Softmax, λ=1 순수 Entmax, λ=2 PLADIS 외삽).

## 그림 1 (최종 권장, 1단 폭, 동일 하드웨어) — `fig1_guidance_pladis_a5000.pdf`

출처: `outputs/a5000_guidance_sweep_v1/analysis/delta_table_unified.csv` (CFG 7.5/3.0/1.5는 RTX A5000에서 재생성, no CFG는 기존 RTX A5000 실험).

그림 1. Guidance 강도에 따른 PLADIS(λ=2)의 Dense 대비 성능 변화. (a) ΔCLAP, (b) ΔAQAScore, (c) ΔPQ(AudioBox). 네 조건(CFG 7.5, 3.0, 1.5, no CFG = 조건부 전용 추론)은 모두 동일한 RTX A5000 환경에서 같은 24개 프롬프트 × 시드 3개, 같은 초기 latent로 생성·평가하였다. 점선은 Dense(Δ=0), 오차 막대는 프롬프트 단위 bootstrap(10,000회) 95 % 신뢰구간. 가이던스를 약화할수록 세 지표 모두에서 PLADIS의 성능 격차가 줄었으며(ΔCLAP −0.192 → −0.102 → −0.068 → −0.046, ΔAQAScore −0.100 → −0.066 → −0.027 → −0.011, ΔPQ −0.64 → −0.25 → −0.06 → −0.06), CFG 1.5 이하에서 AQAScore와 PQ는 Dense와 통계적으로 구분되지 않았으나 CLAP의 손실은 가이던스 없는 조건에서도 유지되었다. 어느 조건에서도 PLADIS는 Dense를 넘지 않았다.

짧은 버전: 그림 1. Guidance 강도에 따른 PLADIS(λ=2)의 Dense 대비 성능 변화(동일 RTX A5000, 24개 프롬프트 × 시드 3개, 점선 Dense, 오차 막대 95 % bootstrap CI). 가이던스를 약화할수록 PLADIS의 성능 격차가 감소하였고, CFG 1.5 이하에서 AQAScore와 PQ는 Dense와 구분되지 않았으나 CLAP의 손실은 남았다.

## 그림 1 (이전 버전, RTX 2080 CFG만, 대체됨) — `fig1_cfg_pladis_trend.pdf`

그림 1. CFG 스케일 변화에 따른 PLADIS(λ=2)의 Dense 대비 성능 변화. (a) ΔCLAP, (b) ΔAQAScore, (c) ΔPQ(AudioBox). 점선은 Dense(Δ=0), 오차 막대는 프롬프트 단위 bootstrap(10,000회) 95 % 신뢰구간(24개 프롬프트 × 시드 3개). CFG가 7.5에서 1.5로 감소할수록 PLADIS의 성능 저하는 세 지표 모두에서 단조롭게 줄었으며(ΔCLAP −0.193 → −0.070, ΔAQAScore −0.099 → −0.028, ΔPQ −0.63 → −0.06), CFG 1.5에서 AQAScore와 PQ의 저하는 0과 구분되지 않았으나 CLAP의 저하는 유지되었다.

짧은 버전: 그림 1. CFG 변화에 따른 PLADIS(λ=2)의 Dense 대비 성능 변화(점선 Dense, 오차 막대 95 % bootstrap CI, 24개 프롬프트 × 시드 3개). CFG가 감소할수록 CLAP, AQAScore, PQ에서 PLADIS의 성능 저하가 줄었으나 어느 CFG에서도 Dense를 넘지 않았다.

## 그림 2 (후보, 1단 폭) — `fig2_nocfg_delta.pdf`

그림 2. True no-CFG 조건(guidance scale 1.0, 무조건부 예측 없이 조건부 전용 추론)에서 PLADIS(λ=2)의 Dense 대비 성능 변화. (a) ΔCLAP, (b) ΔAQAScore, (c) ΔPQ(AudioBox). 점선은 Dense(Δ=0), 오차 막대는 95 % bootstrap 신뢰구간(24개 프롬프트 × 시드 3개, 동일 초기 latent로 짝지은 비교). AQAScore(−0.011 [−0.052, +0.032])와 PQ(−0.06 [−0.15, +0.02])는 Dense와 통계적으로 구분되지 않았으나, CLAP(−0.046 [−0.077, −0.019])에서는 성능 저하가 유지되었다. 이 실험은 CFG 실험과 다른 GPU에서 별도로 생성·평가한 결과이다.

## 그림 1A (선택, 2단 폭) — `fig1_cfg_delta_grid.pdf`

그림 1A. CFG 스케일(행: 7.5, 3.0, 1.5)과 지표(열: ΔCLAP, ΔAQAScore, ΔPQ)별 Dense 대비 성능 변화. 각 패널의 조건은 왼쪽부터 Entmax λ=0.5, Entmax λ=1, PLADIS λ=2이며, 점선은 Dense(Δ=0), 오차 막대는 95 % bootstrap 신뢰구간(24개 프롬프트 × 시드 3개). λ가 클수록, 그리고 CFG가 높을수록 Dense 대비 저하가 컸으며 어느 조건도 Dense를 유의하게 넘지 않았다.

## 표 1 (최종 권장 수치 출처) — `table_a5000_guidance_delta.md`

표 1. Guidance 설정별 PLADIS(λ=2)의 Dense 대비 변화량(Δ = PLADIS − Dense, 95 % bootstrap CI, 24개 프롬프트 × 시드 3개, 모두 RTX A5000). No CFG는 무조건부 예측을 계산하지 않는 조건부 전용 추론(guidance scale 1.0)이다. 본문용 축약형을 쓸 경우 CI 대신 ‡(CI가 0을 포함하지 않음)로 표시한다.

## 선택 그림 (DEPRECATED_FOR_PAPER, 하드웨어 혼합) — `fig_optional_cfg_to_nocfg_DEPRECATED_FOR_PAPER.pdf`

그림 S(사용 안 함). PLADIS(λ=2)의 Dense 대비 성능 변화를 CFG 7.5, 3.0, 1.5(채운 사각형, 실선 연결)와 가이던스 없는 조건(빈 원, †)에 대해 함께 나타낸 것. 점선은 Dense(Δ=0), 오차 막대는 95 % bootstrap 신뢰구간. CFG 실험은 RTX 2080, 가이던스 없는 실험은 RTX A5000에서 생성하였으므로 두 묶음은 하드웨어가 통제된 하나의 연속 실험이 아니며, 각 Δ는 자기 실험 안에서 짝지은 값이다. 따라서 이 궤적은 서술적 비교이다.

## 표 1 (후보) — `table1_pladis_guidance_compact.md`

표 1. 가이던스 설정별 PLADIS(λ=2)의 Dense 대비 변화량(Δ = PLADIS − Dense, 24개 프롬프트 × 시드 3개, 시드 평균). ‡는 프롬프트 단위 bootstrap 95 % 신뢰구간이 0을 포함하지 않음을 뜻한다. *No-CFG는 조건부 전용 추론(guidance scale 1.0)으로, CFG 7.5/3.0/1.5(RTX 2080)와 다른 GPU(RTX A5000)에서 별도로 짝지어 평가한 결과이다.

## 표 S (선택) — `table_optional_cfg75_methods.md`

표 S. CFG 7.5에서 조건별 절대 점수(24개 프롬프트 × 시드 3개 평균). 희소화 강도가 커질수록(Entmax λ=0.5 → λ=1 → PLADIS λ=2) CLAP, AQAScore, PQ가 모두 낮아졌다.
