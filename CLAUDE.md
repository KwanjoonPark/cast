# Project: PLADIS for Text-to-Audio

## Goal

Apply PLADIS sparse cross-attention to Auffusion
without additional training.

Main comparison:

1. Vanilla Auffusion
2. Sparse Attention
   - alpha = 1.5
   - lambda = 1.0
3. PLADIS
   - alpha = 1.5
   - lambda = 2.0

## Repository Structure

third_party/PLADIS
- Original PLADIS implementation
- READ ONLY

third_party/Auffusion
- Original Auffusion implementation
- READ ONLY unless absolutely necessary

cast/
- All new implementations should go here (CAST package).

experiments/
- Experiment configurations and prompts.

outputs/
- Generated audio and evaluation results.

## Important Rules

- Do not modify baseline behavior.
- Vanilla output must remain reproducible.
- Use the same random seed across methods.
- Modify cross-attention only.
- Do not modify self-attention.
- Do not train or finetune the model.
- PLADIS is inference-time only.

## PLADIS

Dense attention:

A_dense = softmax(QK^T / sqrt(d)) V

Sparse attention:

A_sparse = entmax_1.5(QK^T / sqrt(d)) V

PLADIS:

A = A_dense + lambda * (A_sparse - A_dense)

lambda = 0: vanilla
lambda = 1: sparse attention
lambda = 2: default PLADIS

## Current Priority

First milestone:

Generate the same prompt and same seed using:

- Vanilla
- Sparse
- PLADIS

Do not implement additional ablations before this works.

---

# Branch `wonseok/advisor-feedback`: main 대비 핵심 변경 사항 (2026-10-25)

이 섹션은 `main`(830bd52)에서 갈라진 이 브랜치의 변경점과, 이어서 작업하는 팀원(AQAScore 채점, 논문 보완)에게 필요한 정보를 정리한다.
위의 기존 규칙(Do not modify baseline behavior 등)은 그대로 유효하다. 위 "Current Priority"의 first milestone(Vanilla/Sparse/PLADIS 동일 프롬프트·시드 생성)은 main에서 이미 완료되었다.
상세 서술 보고서: `docs/CAST_CFG_실험_인수인계_보고서.docx` (그림 포함).

## 0. 이 브랜치가 하는 일

교수님 피드백(2026-10-23): "baseline(CFG 7.5)이 너무 잘 나와서 PLADIS 효과가 안 보일 수 있다. CFG를 낮춘 약한 baseline에서 비교하고, 논문 2페이지 표 1 그래프 위에 같은 모양을 얹어라."
이에 따라 (1) CFG sweep, (2) 새 프롬프트셋(v2) 1차 평가, (3) 층(해상도) 선택 ablation을 수행했다. 모든 결과는 **RTX 2080(8GB) PC에서 새로 생성**했다.

## 1. 한눈에 보는 결과

| 실험 | 결과 |
|---|---|
| CFG sweep (기존 24개 프롬프트 × 시드 3 × 5조건 × CFG 7.5/3.0/1.5, 1,080개) | PLADIS는 어떤 CFG에서도 Dense를 넘지 못함. 손해는 CFG에 비례: ΔCLAP −0.193 / −0.102 / −0.070, ΔPQ −0.630 / −0.257 / −0.064 (CFG 7.5 / 3.0 / 1.5). 약한 baseline에서 이득이 커진다는 가설은 지지되지 않음 |
| 프롬프트 v2 (16개 × 시드 3 × CFG 3.0/1.5) | v2에서도 PLADIS는 CLAP 기준 Dense 미만. 모호 프롬프트는 손해가 작지만(Δ≈0) 모호−명확 차이의 CI가 0을 포함해 가설 지지 불가 |
| 층 선택 (고해상도 5층 / 저해상도 6층만 λ=2, CFG 3.0) | 둘 다 전체 적용보다 손해는 작으나 Dense 미만. 고/저 해상도 차이는 확인되지 않음 |

CFG 7.5 수치는 논문 표 1과 일치한다 (Dense CLAP 0.555, PLADIS 0.363 대 논문 0.364, PQ 5.918/5.288 대 5.92/5.28). 단, **오디오 파일은 A5000 결과와 bitwise로 다르다**(GPU 차이).

## 2. 코드 변경 (main 대비)

### 2.1 기존 파일 수정 (기본값이 기존 동작이므로 baseline 재현성에 영향 없음, pytest 13개 통과)
- `cast/integration/auffusion.py`: `apply_cast(..., only=None)` 추가. `only`가 주어지면 해당 attn2 모듈 이름 목록에만 CAST processor를 설치하고, 알 수 없는 이름은 `ValueError`. `None`이면 기존대로 16개 전부.
- `cast/pipeline.py`: `run_one(..., only_modules=None)` 추가, `apply_cast(only=only_modules)`로 전달.
- `.gitignore`: `참고자료/`(공유 안 하는 로컬 참고 폴더), `.venv-cast/`, `.venv-cast-eval/` 추가.

### 2.2 새 스크립트 (`experiments/`)

| 스크립트 | 역할 |
|---|---|
| `run_cfg_sweep_v1.py` | CFG sweep 생성. 프롬프트 × 시드 × CFG × 5조건(dense, sharp_b1.5, entmax_l0.5, entmax_l1, entmax_l2). 이어서 실행 가능. 출력 `outputs/cfg_sweep_v1/` |
| `make_sweep_index.py` | 생성된 샘플 목록 `audio_index.csv` 작성 (AQAScore/PQ 채점기가 읽는 형식 + cfg, arm 열) |
| `score_sweep_clap.py` | sweep/ablation 샘플 CLAP 채점 (이어서 채점). Windows 심볼릭 링크 권한 문제 패치 포함 |
| `analyze_cfg_sweep_v1.py` | 프롬프트 단위 bootstrap CI(10,000회, rng seed 20261024), CFG별 그래프, 시드 간 일관성 진단. `aqascore/sample_scores.csv`가 있으면 ΔAQAScore 패널 자동 추가 |
| `make_listening_set.py` | 프롬프트 파일에서 청취 세트 생성 (vanilla/dense_l0/sparse_l1/pladis_l2). `--first-per-category`, `--ids`, `--core-only` |
| `score_listening_clap.py` | 청취 세트 CLAP 채점 |
| `make_listening_index.py` | 청취 세트 `audio_index.csv` 작성 |
| `analyze_listening_v2.py` | v2 시드 3개 분석 (명확/모호 그룹, 모호−명확 차이 CI, 시드 간 일관성) |
| `run_layer_ablation_v1.py` | 층 선택 ablation 생성. 사전 검증(sweep과 bitwise 일치, 적용 모듈 수) 후 생성. `--groups high low [mid]` |
| `analyze_layer_ablation_v1.py` | 층 선택 분석, 그림 |

### 2.3 층 선택 설계의 핵심 (실수하기 쉬운 부분)
- 16개 attn2를 해상도로 분할 (latent 32×128): L0 고해상도 = `down_blocks.0` + `up_blocks.3` (5개), L1 중간 = `down_blocks.1` + `up_blocks.2` (5개), L2 저해상도 = `down_blocks.2` + `up_blocks.1` (5개), `mid_block` (1개). "low" = L2 + mid(6개). `up_blocks.0`에는 attention이 없다.
- **부분 적용 시 비선택 모듈은 CAST λ=0(fp32 dense) processor로 둔다.** 순정 fp16 processor로 두면 정밀도 차이가 섞여 sweep의 Dense와 비교 불가. 구현은 드라이버가 비선택 모듈에 λ=0을 먼저 설치하고 `run_one(only_modules=선택)`이 선택 모듈에 λ를 설치하는 방식.
- 검증: 모든 모듈 λ=0 샘플이 sweep Dense와, 모든 모듈 λ=2 샘플이 sweep `entmax_l2`와 audio SHA-256 일치해야 생성이 진행된다.

## 3. 출력물 구조와 git 추적 범위

추적됨(작은 파일): 각 실험 폴더의 `audio_index.csv`, `clap_scores.csv`, `audiobox/*.csv|json`, `analysis/*`(csv, json, 그림 png/pdf), `experiment_manifest.json`, `generation_summary.json`, 청취 세트의 `INDEX.md`, `listening_summary.json`, `clap_summary.csv`.
**추적 안 됨(.gitignore): `*.wav`, `*.pt`, `spectrogram.png`, 샘플별 `manifest.json`, 로그.** 따라서 **이 브랜치만으로는 오디오가 없다.**

| 경로 | 내용 |
|---|---|
| `outputs/cfg_sweep_v1/` | sweep 1,080개 (인덱스 1,080행). `analysis/delta_table.csv`, `fig_cfg_delta_grid`, `fig_cfg_delta_lines` |
| `outputs/listen_prompts_v2/cfg{3,1.5}_seed{42,123,2026}/` | v2 청취 세트 6개. 통합 인덱스 `audio_index_all.csv`(208행), `audiobox_all/`, `analysis/` |
| `outputs/layer_ablation_v1/` | 층 선택 144개. `analysis/fig_layer_ablation` |
| `docs/` | 인수인계 보고서(docx) |

## 4. 환경 (Windows, 이 PC 기준)

- 생성: `.venv-cast` — Python 3.10, torch 2.0.1+cu118, diffusers 0.18.2, transformers 4.31.0, entmax 1.3 (`experiments/env/requirements-cast.txt`와 동일). 서브모듈은 `git submodule update --init third_party/Auffusion third_party/PLADIS`.
- 평가: `.venv-cast-eval` — torch 2.4.1+cu124, transformers 4.57.1, audiobox_aesthetics (`requirements-eval.txt`). AQAScore(Qwen2.5-Omni-7B, 약 17GB VRAM)는 8GB GPU에서 실행 불가.
- 한글 로그가 있으므로 `export PYTHONIOENCODING=utf-8`. Windows에서 HF 캐시 심볼릭 링크 권한 오류(WinError 1314)는 개발자 모드를 켜거나 `score_*_clap.py`의 패치를 참고.
- 샘플당 생성 약 15~23초. 생성 중에는 GPU를 쓰는 다른 프로그램을 피할 것(VRAM 8GB).

## 5. 실험 규칙 (이 브랜치에서 추가로 지킨 것, 이어서 작업할 때도 유지)

- 모든 비교 조건은 같은 시드·같은 초기 latent·같은 설정(100 step, 256×1024, fp16, deterministic self-attention)을 쓰고 바뀌는 것은 의도한 하나(CFG, λ, 적용 층)뿐이다.
- **이 PC(RTX 2080)에서 만든 샘플과 A5000에서 만든 샘플을 한 비교에 섞지 않는다.** 지표 평균은 일치하나 오디오는 bitwise로 다르다.
- 통계 단위는 프롬프트(시드 평균), Δ = 조건 − Dense(같은 CFG, 같은 프롬프트), 프롬프트 단위 bootstrap 95% CI. CI가 0을 포함하면 "차이 확인 안 됨"으로 쓴다.
- 가설은 결과를 보기 전에 기록하고, 선별 규칙은 결과 보기 전에 정한다. `cast_prompts_v2.json`은 사전 고정 파일이므로 수정하지 않는다.
- 쓸 수 있는/피해야 할 문장은 보고서 7장 참조. 요약: "어느 CFG에서도 Dense를 넘지 못했고 손해는 CFG와 함께 커졌다"는 가능. "낮은 CFG에서 이득", "층 선택으로 손해 감소 성과", "특정 해상도가 원인", "모호 프롬프트에서 이득"은 근거 없음.

## 6. 정정 사항 (이전 중간 보고와 다른 내용)

- v2 1차 평가(시드 42)에서 "일관된 이득"으로 보였던 IM01, NA01은 시드 3개에서 유지되지 않음 (NA01은 시드 123에서 크게 손해). 시드 3개 모두 이득이었던 것은 FX01(CFG 3.0만)뿐, EN01·MS01·HL01은 두 CFG에서 계속 손해.
- "낮은 CFG에서 프롬프트별 효과가 노이즈"는 기존 24개(CFG 1.5, 시드 간 r≈0.10)에서만 관찰되었고 v2 16개에서는 성립하지 않음(r≈0.37). 일반화 금지.

## 7. 이어서 할 일

### 7.1 AQAScore 채점 (A5000 보유자)
1. **오디오는 git에 없다.** 채점 대상은 이 PC에서 생성한 1,080개 wav 그대로여야 CLAP/PQ와 같은 샘플이 된다(재생성하면 다른 샘플). 전달 패키지(285MB, GitHub 100MB 제한 초과라 브랜치에 못 올림): `cast_cfg_sweep_aqa_package.zip`, SHA-256 `a46b69fd6a46022be17b3df169fcaaa728c5373f229b2121f21c803b98600bf1`. 김원석에게 별도 경로(클라우드 등)로 받을 것. 압축 안에는 wav, `audio_index.csv`, 안내문이 있고 저장소 루트에서 풀면 된다.
2. 풀린 wav가 맞는지 확인: `outputs/cfg_sweep_v1/audio_index.csv`의 `audio_sha256`과 파일의 SHA-256이 일치해야 한다.
3. `cast-eval` 환경에서 실행 (프롬프트 이벤트는 v1 프롬프트 파일을 쓰며 인덱스의 prompt_id가 v1 ID라 수정 없이 동작):
   `python experiments/eval_aqascore.py --model-dir <Qwen2.5-Omni thinker 경로> --index outputs/cfg_sweep_v1/audio_index.csv --out outputs/cfg_sweep_v1/aqascore`
   (thinker 디렉터리는 `experiments/make_thinker_dir.py`). `validation.json`의 `repeat_max_abs_diff`가 0이어야 한다.
4. `python experiments/analyze_cfg_sweep_v1.py`를 다시 실행하면 그래프에 ΔAQAScore 패널이 추가된다 (`outputs/cfg_sweep_v1/analysis/`).
5. v2(208개)와 층 선택(144개)용 AQAScore는 미실시. v2는 `eval_aqascore.py`가 v1 프롬프트 파일을 고정해서 읽으므로 `--prompt-file` 옵션 추가가 필요하다(미구현). 인덱스는 이미 있음: `outputs/listen_prompts_v2/audio_index_all.csv`, `outputs/layer_ablation_v1/audio_index.csv`.

### 7.2 논문 보완
- 논문 2페이지 표 1 그래프 위에 `outputs/cfg_sweep_v1/analysis/fig_cfg_delta_grid.pdf`(CFG별)를 얹는 구성. 크기와 글꼴은 학회 템플릿에 맞출 것(그림은 영문 레이블, 벡터 PDF 있음).
- 기존 스펙트로그램 그림은 supplementary로 이동(교수님 의견), 발표 때 소리와 함께 사용.
- 결과·결론 문단 수정. 교수님이 제안한 결론 방향(범용 적용 가능성, 가이던스 미학습 모델에서 효과)은 이번 실험이 직접 뒷받침하지 않으므로 표현에 주의. 한계에 단일 모델, AQAScore 상태, Auffusion이 latent diffusion이라 디코더가 영향을 줄 수 있다는 지적을 포함.
- 제출 마감(월요일 10/26 추정, 녹취상 "5월까지요"로 인식되어 미확정), 제출처, 지도교수 정보 필요 여부는 미확인.

### 7.3 추가 실험 후보 (비용은 RTX 2080 기준)
- CFG 1.0~1.2: 교수님 가설(가이던스를 못 쓰는 상태)의 직접 검증. `run_cfg_sweep_v1.py --cfgs 1.2`(예). diffusers에서 CFG=1.0은 uncond 가지를 쓰지 않는 경로일 수 있어 1.1~1.2 권장(미확인). 360개, 약 2시간.
- 층 선택 L1 추가: `run_layer_ablation_v1.py --groups mid` (72개, 약 20분). 다른 CFG: `--cfgs 7.5 1.5`.
- v2 확장/선별: 모호 프롬프트 표본 확대(core 64개에 모호 32개 포함). `cast_prompts_v2.json`(팀원 제작, 사전 고정 파일)은 저장소 루트에 포함되어 있다. 수정하지 말 것. core 64개 선정 기준은 파일에 명시되어 있지 않다(제작자에게 확인).
- Dense를 다른 시드에서 가져오는 재분석: Dense 점수와 Δ 상관의 평균 회귀 착시 제거 (생성 불필요).

## 8. 한계 (이 브랜치의 모든 결과에 적용)

Auffusion 단일 모델, 지표 CLAP과 AudioBox PQ 두 개(AQAScore 미채점), CFG 최저값 1.5(이때도 Dense CLAP이 16%만 하락), 층 선택은 CFG 3.0 한 점, v2는 모호 프롬프트 4개와 시드 3개. "왜 CFG에 비례해 손해가 커지는가"(CFG 증폭 가설 등)는 검증하지 않았다.
