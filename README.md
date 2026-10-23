<h1 align="center">CAST</h1>

<p align="center">
  <b>Cross-Attention Sparsification for Text-to-Audio Generation</b>
</p>

<p align="center">
  Training-free sparse and sharpened cross-attention for text-to-audio diffusion models.
</p>

CAST is a research prototype that studies whether **sparse cross-attention**, originally explored in text-to-image diffusion (PLADIS), transfers to **text-to-audio (T2A)** generation.

The implementation applies inference-time attention interventions to **Auffusion** without retraining the model. Only the 16 text cross-attention (`attn2`) modules are modified; self-attention, the scheduler and classifier-free guidance are untouched.

---

## Overview

Text-to-audio diffusion models rely on cross-attention to connect a text prompt with an evolving audio latent representation.

CAST asks:

> **Does moving text-conditioning cross-attention toward a sparse (or merely sharper) distribution help text-to-audio generation?**

With `Z = QKᵀ / √d` and output `A V`, the processor implements three attention types on the same 16 modules:

| `attention_type` | Attention map | Config field | Modes |
|---|---|---|---|
| `entmax_mix` (PLADIS) | `W = W_dense + λ_entmax · (Entmax₁.₅(Z) − W_dense)` | `lam` | `cast_vanilla`, `cast_sparse`, `cast_pladis`, `cast` |
| `softmax` (Sharp Softmax) | `W = Softmax(β Z)` | `beta` | `sharp_softmax` |
| `sharp_mix` (partial sharp) | `W = W_dense + λ_sharp · (Softmax(β Z) − W_dense)` | `beta`, `lambda_sharp` | `sharp_mix` |

`W_dense = Softmax(Z)` is the Auffusion baseline. Logits, softmax, entmax and mixing run in fp32 inside the fp16 pipeline.

### Interpretation of λ_entmax

| λ_entmax | Interpretation |
|---:|---|
| `0.0` | Dense Softmax baseline (`cast_vanilla`) |
| `0 < λ < 1` | Partial movement toward sparse attention |
| `1.0` | Pure Entmax-1.5 sparse attention (`cast_sparse`) |
| `λ > 1` | PLADIS-style extrapolation (`cast_pladis` uses `λ = 2`) |

`α = 1.5` is fixed for the sparse branch. `β = 1` and `λ_sharp = 0` reproduce the dense baseline bitwise at the module level, which is used as a parity check. The three names `lam` (λ_entmax), `beta` and `lambda_sharp` are kept distinct everywhere in code, configs and reports.

---

## Why Audio?

Sparse cross-attention has shown promising behavior in image diffusion, where stronger token selectivity can improve semantic localization.

Audio is different. Multiple acoustic events can naturally overlap in the same time-frequency region:

- speech + background music
- dog barking + rain
- footsteps + birds
- engine noise + conversation

This makes text-to-audio a testbed for whether stronger concentration of cross-attention improves semantic selectivity or instead removes secondary acoustic information. CAST therefore separates three aspects of generation quality:

- **text-audio alignment**
- **event preservation**
- **audio fidelity**

---

## Backbone

CAST uses **Auffusion** (*Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation*): a Stable-Diffusion-style latent diffusion U-Net on mel-spectrogram latents, CLIP text encoder with a condition adapter, PNDM sampling and a HiFi-GAN vocoder.

In its text-conditioning cross-attention, **Q** comes from the noisy spectrogram latent and **K, V** from the text sequence. The released `auffusion/auffusion` checkpoint contains 16 self-attention (`attn1`) and 16 text cross-attention (`attn2`) modules; CAST is installed only on the 16 `attn2` modules and the original processors are restored after each run.

---

## Repository Structure

```text
CAST/                                   repository root
├── cast/                               CAST package (importable as `cast`)
│   ├── config.py                       CastConfig dataclass and MODES table
│   ├── pipeline.py                     controlled generation: pre-sampled latents, manifests, outputs
│   ├── attention/
│   │   ├── sparse_ops.py               dense softmax, α-entmax, dense–sparse mixing
│   │   └── cast_processor.py           CastCrossAttnProcessor (diffusers 0.18.2 processor API)
│   ├── integration/auffusion.py        import Auffusion, select the 16 attn2 modules, install / restore
│   ├── eval/clap.py                    shared CLAP scorer
│   └── tests/                          pytest suite (13 tests)
├── experiments/
│   ├── configs/*.yaml                  one config per experiment; the arm definitions live here
│   ├── prompts/cast_diagnostic_v1.json frozen 24-prompt 2×2 taxonomy (SHA-256 recorded in the report)
│   ├── env/                            requirements and setup scripts for the two conda envs
│   ├── run_*.py                        generation drivers
│   ├── eval_*.py                       metric scripts (CLAP, AQAScore, AudioBox Aesthetics)
│   ├── analyze_*.py                    bootstrap analysis → csv / json / plots
│   └── make_*.py                       human-eval packages, entropy tables, Qwen thinker directory
├── outputs/                            experiment-level results are versioned (csv / json / md / plots);
│                                       per-run audio, latents, spectrograms and manifests are not
├── paper_assets/                       figure sources, exported figures, Korean captions, FIGURE_SELECTION.md
├── third_party/                        git submodules, read-only: Auffusion (needed at runtime), PLADIS, ELSA
├── CAST_*_REPORT.md                    one report per stage (see Experiments)
├── CLAUDE.md                           project rules for assistant-driven work
└── README.md
```

`third_party/Auffusion` is imported at runtime without modification. `third_party/PLADIS` and `third_party/ELSA` are reference implementations only. CAST does **not** import PLADIS: Auffusion and PLADIS depend on incompatible Diffusers versions, so the PLADIS attention rule is reimplemented against the Diffusers 0.18.2 processor API.

---

## Environment

Two conda environments are used. The generation environment is frozen to the Auffusion dependency stack; the evaluation environment holds the newer stack needed by Qwen2.5-Omni and AudioBox Aesthetics.

**`cast` — generation, parity tests, CLAP**

```text
Python              3.10
PyTorch             2.0.1+cu118
diffusers           0.18.2
transformers        4.31.0
huggingface_hub     0.16.4
tokenizers          0.13.3
entmax              1.3
librosa             0.9.2
soundfile           0.12.1
accelerate          0.21.0
```

**`cast-eval` — AQAScore and AudioBox Aesthetics**

```text
Python              3.10
PyTorch             2.4.1+cu124
transformers        4.57.1
audiobox_aesthetics
```

Exact pins: `experiments/env/requirements-cast.txt` and `experiments/env/requirements-eval.txt`.

### Setup

```bash
git clone --recurse-submodules https://github.com/KwanjoonPark/cast.git CAST
cd CAST
bash experiments/env/setup_env.sh          # creates the `cast` env
bash experiments/env/setup_eval_env.sh     # creates the `cast-eval` env (optional, for AQAScore / AudioBox)
```

Notes:

- The `auffusion/auffusion` checkpoint is downloaded anonymously on first use (`token=False`); no Hugging Face login is required.
- AQAScore uses the Qwen2.5-Omni-7B *thinker* only. `experiments/make_thinker_dir.py` builds a local model directory from the cached snapshot (shards 1–4) so the talker weights never need to be downloaded.
- The GPU driver must support CUDA 11.8 builds for the `cast` env.

---

## Running the Validated Milestone

The first milestone verifies:

1. original Auffusion generation,
2. CAST injection into exactly 16 cross-attention modules,
3. dense-attention parity at `λ_entmax = 0`,
4. sparse attention at `λ_entmax = 1`,
5. PLADIS-style extrapolation at `λ_entmax = 2`,
6. identical initial noise across all three CAST arms.

```bash
bash experiments/run_first_milestone.sh    # pytest + validate_parity.py + run_first_milestone.py
```

Each run directory contains `audio.wav`, `spectrogram.png`, `final_latents.pt` and `manifest.json` (config, initial-latent SHA-256, audio SHA-256, attention statistics). For controlled single-sample comparisons set

```yaml
deterministic_self_attn: true
```

which forces the math SDPA backend for self-attention. The stock fp16 SDPA kernel is not bitwise repeatable at this sequence length, and the resulting run-to-run variation would otherwise be confounded with the intervention.

---

## Experiments

Each stage has a driver script, a config that defines the arms, and a report. The config and driver are the source of truth for what was run; the reports document results and decisions.

| Stage | Generation | Evaluation / analysis | Report |
|---|---|---|---|
| Architecture analysis (no code) | – | – | `CAST_ANALYSIS_REPORT.md` |
| First milestone | `run_first_milestone.sh`, `configs/first_milestone.yaml` | `validate_parity.py` | `CAST_IMPLEMENTATION_REPORT.md` |
| Pilot 0 (5 prompts × λ_entmax {0, 1, 2}) | `run_pilot0.py`, `configs/pilot0.yaml` | listening index | `CAST_PILOT0_REPORT.md` |
| λ_entmax sweep (7 values) | `run_lambda_sweep0.py`, `configs/lambda_sweep0.yaml` | blind listening index | `CAST_LAMBDA_SWEEP_REPORT.md` |
| CLAP evaluation of the sweep | – | `eval_clap.py` | `CAST_CLAP_EVALUATION_REPORT.md` |
| Diagnostic v1 (24 prompts × λ_entmax {0, .5, 1, 2} × 3 seeds) | `run_diagnostic_v1.py`, `configs/diagnostic_v1.yaml` | `eval_clap_diagnostic.py`, `analyze_diagnostic_v1.py`, `make_human_eval_package.py` | `CAST_DIAGNOSTIC_V1_REPORT.md` |
| Sharp Softmax v1 (β {1, 1.25, 1.5, 2}) | `run_sharp_softmax_v1.py`, `configs/sharp_softmax_v1.yaml` | `eval_clap_diagnostic.py --cond-key beta`, `make_entropy_csv.py`, `analyze_sharp_softmax_v1.py`, `make_human_eval_package_sharp.py` | `CAST_SHARP_SOFTMAX_REPORT.md` |
| Multi-metric evaluation (AQAScore, AudioBox) | – | `eval_aqascore.py`, `eval_audiobox.py`, `analyze_multimetric.py` (`cast-eval` env) | `CAST_MULTI_METRIC_EVALUATION_REPORT.md` |
| Partial sharp v1 (β = 1.5, λ_sharp sweep) | `run_sharp_mix_v1.py`, `configs/sharp_mix_v1.yaml` | `eval_sharp_mix_metrics.py`, `analyze_sharp_mix_v1.py` | `CAST_PARTIAL_SHARP_REPORT.md` |

Drivers reuse existing audio when the stored manifest matches the requested condition bitwise, so later stages share the dense baseline and other overlapping arms with earlier ones instead of regenerating them.

---

## Evaluation

Implemented metrics:

- **CLAP score** (`laion/clap-htsat-unfused`, cosine similarity): text-audio alignment. Exploratory event-phrase CLAP (foreground / background phrases) is used for multi-event prompts.
- **AQAScore** (Qwen2.5-Omni-7B thinker, yes/no probability for "does this audio contain the described sound events"): alignment with a different inductive bias from CLAP. An event-level variant scores each requested event separately.
- **AudioBox Aesthetics** (CE, CU, PC, PQ; PQ is the primary fidelity measure): reference-free audio quality.
- **Human evaluation**: blind, label-free listening packages are generated for the diagnostic and sharp-softmax stages (`outputs/*/human_eval`). Ratings have not been collected.

Not run: **ELSA** (event-level alignment) requires the gated `facebook/sam-audio-large-tv` weights and an OpenAI API key; the repository is pinned under `third_party/ELSA` and the analyzer picks up `elsa_scores.csv` automatically if it is ever produced.

Statistics follow one convention across stages: the prompt is the unit of analysis (seeds averaged), differences are paired against the dense baseline, and 95 % confidence intervals come from 10,000 prompt-level bootstrap resamples with a fixed bootstrap seed recorded in each report.

---

## Experimental Principles

To keep comparisons controlled, every compared condition shares:

- checkpoint, prompt, seed and initial latent
- scheduler, number of steps, CFG scale and precision
- the deterministic self-attention backend
- the same CAST processor code path

Only the intended attention hyperparameter differs: `lam` (λ_entmax) in the Entmax stages, `beta` in Sharp Softmax, `lambda_sharp` in the partial-sharp stage. The 24-prompt set is frozen (hash recorded) and seeds are fixed before generation; qualitative examples are chosen by predeclared prompt and seed, not after inspection.

---

## Results

Results, their statistical treatment and the decision taken at each stage are documented in the stage reports listed above. This README intentionally does not restate numbers; read the latest report (`CAST_PARTIAL_SHARP_REPORT.md`) for the current conclusion and the earlier ones for the evidence chain.

---

## Research Questions

1. Does sparse cross-attention generalize from text-to-image to text-to-audio diffusion?
2. Does stronger sparsity improve text-audio semantic alignment?
3. Does sparse attention suppress secondary or background acoustic events?
4. Is the effect driven by hard sparsity (exact zeros) or by concentration of the attention distribution in general?
5. Is interpolation between dense and sparse (or sharp) attention more suitable for audio than extrapolation?

---

## Project Status

```text
Architecture analysis                  DONE
Auffusion environment                  DONE
CAST attention implementation          DONE  (entmax_mix, softmax, sharp_mix)
Dense parity validation                DONE  (λ_entmax = 0, β = 1, λ_sharp = 0)
Pilot 0 and λ_entmax sweep             DONE
CLAP evaluation                        DONE
Diagnostic v1 (24 prompts × 3 seeds)   DONE
Sharp Softmax v1                       DONE
Multi-metric evaluation                DONE  (AQAScore, AudioBox Aesthetics)
Partial sharp v1                       DONE
ELSA                                   BLOCKED (gated weights, API key)
Human evaluation                       PACKAGES READY, ratings pending
Paper figures                          DONE  (paper_assets/)
```

---

## References

### PLADIS

**PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity**  
Kwanyoung Kim, Byeongsu Sim.  
ICCV 2025 / arXiv:2503.07677.

### Auffusion

**Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation**  
Jinlong Xue, Yayue Deng, Yingming Gao, Ya Li.  
arXiv:2401.01044.

### Entmax

Peters et al.  
**Sparse Sequence-to-Sequence Models**  
ACL 2019.

### Metrics

- CLAP: Wu et al., **Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**, ICASSP 2023 (arXiv:2211.06687). Checkpoint `laion/clap-htsat-unfused`.
- AQAScore: arXiv:2601.14728, implemented with Qwen2.5-Omni-7B.
- AudioBox Aesthetics: Tjandra et al., **Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound**, arXiv:2502.05139.
- ELSA: https://github.com/kento2247/ELSA (pinned as a submodule; not run).

---

## Disclaimer

CAST is a **research prototype**. The experiments cover one backbone, one frozen 24-prompt set and automatic metrics plus prepared but uncollected human ratings. Conclusions in the reports are scoped to that setting and should not be read as general statements about sparse cross-attention in text-to-audio generation.
