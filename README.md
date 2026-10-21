# CAST

> **Cross-Attention Sparsification for Text-to-Audio Generation**

CAST is a research prototype that studies whether **sparse cross-attention**, originally explored in text-to-image diffusion, can improve or reshape **text-to-audio (T2A)** generation.

The current implementation applies a PLADIS-style sparse attention intervention to **Auffusion** at inference time, without retraining the model.

---

## Overview

Text-to-audio diffusion models rely on cross-attention to connect a text prompt with an evolving audio latent representation.

CAST asks:

> **Does moving text-conditioning cross-attention toward a sparse representation help text-to-audio generation?**

We start from Auffusion and replace only its **text cross-attention (`attn2`)** modules with a controllable CAST attention processor.

The dense and sparse attention maps are

\[
W_{\text{dense}}
=
\operatorname{Softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right),
\]

\[
W_{\text{sparse}}
=
\operatorname{Entmax}_{1.5}
\left(
\frac{QK^\top}{\sqrt{d}}
\right).
\]

CAST then interpolates or extrapolates between them:

\[
W_{\text{CAST}}
=
W_{\text{dense}}
+
\lambda
\left(
W_{\text{sparse}}
-
W_{\text{dense}}
\right).
\]

The resulting attention output is

\[
\operatorname{Attention}_{\text{CAST}}
=
W_{\text{CAST}}V.
\]

### Interpretation of \(\lambda\)

| \(\lambda\) | Interpretation |
|---:|---|
| `0.0` | Dense Softmax baseline |
| `0 < λ < 1` | Partial movement toward sparse attention |
| `1.0` | Pure Entmax-1.5 sparse attention |
| `λ > 1` | Extrapolation toward the sparse direction |

The current default comparison uses:

- **CAST Vanilla**: `λ = 0`
- **CAST Sparse**: `λ = 1`
- **CAST PLADIS**: `λ = 2`

`α = 1.5` is fixed for the sparse branch.

---

## Why Audio?

Sparse cross-attention has shown promising behavior in image diffusion, where stronger token selectivity can improve semantic localization.

Audio is different.

Multiple acoustic events can naturally overlap in the same time-frequency region:

- speech + background music
- dog barking + rain
- footsteps + birds
- engine noise + conversation

This makes text-to-audio an interesting testbed for studying whether stronger sparsity improves semantic selectivity or instead removes useful secondary acoustic information.

CAST therefore focuses not only on whether sparsity changes generation, but also on the trade-off between:

- **text-audio alignment**
- **event preservation**
- **audio fidelity**

---

## Backbone

CAST currently uses **Auffusion** (*Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation*).

Auffusion is based on a Stable Diffusion-style latent diffusion U-Net operating on mel-spectrogram latents.

In its text-conditioning cross-attention:

- **Q** comes from the current noisy audio / spectrogram latent representation.
- **K, V** come from the text-conditioning sequence.
- CAST modifies only these text-conditioning cross-attention layers.
- Self-attention remains unchanged.

The released `auffusion/auffusion` checkpoint contains 16 self-attention (`attn1`) and 16 text cross-attention (`attn2`) modules. CAST is installed only on the 16 `attn2` modules.

---

## Repository Structure

```text
CAST/cast/
├── cast/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── sparse_ops.py
│   │   └── cast_processor.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── auffusion.py
│   └── tests/
│       ├── test_sparse_ops.py
│       └── test_processor.py
├── experiments/
│   ├── configs/
│   │   ├── first_milestone.yaml
│   │   └── pilot0.yaml
│   ├── env/
│   │   ├── requirements-cast.txt
│   │   └── setup_env.sh
│   ├── run_first_milestone.py
│   ├── run_first_milestone.sh
│   ├── run_pilot0.py
│   └── validate_parity.py
├── third_party/
│   ├── Auffusion/
│   └── PLADIS/
├── outputs/
├── CAST_ANALYSIS_REPORT.md
├── CAST_IMPLEMENTATION_REPORT.md
├── CAST_PILOT0_REPORT.md
└── README.md
```

`third_party/Auffusion` and `third_party/PLADIS` are treated as **read-only reference implementations**.

---

## Environment

The validated environment is based on the dependency stack required by the original Auffusion implementation.

```text
Python              3.10.20
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

Auffusion and the original PLADIS repository depend on substantially different Diffusers versions. CAST therefore **does not import PLADIS at runtime**. Instead, the PLADIS attention formulation is reimplemented against the Diffusers 0.18.2 attention processor API used by Auffusion.

### Setup

```bash
bash experiments/env/setup_env.sh
```

Or install the pinned requirements manually:

```bash
pip install -r experiments/env/requirements-cast.txt
```

---

## Running the Validated Milestone

The first milestone verifies:

1. original Auffusion generation,
2. CAST injection into exactly 16 cross-attention modules,
3. dense-attention parity at `λ = 0`,
4. sparse attention at `λ = 1`,
5. PLADIS-style extrapolation at `λ = 2`,
6. identical initial noise across all three CAST arms.

```bash
bash experiments/run_first_milestone.sh
```

Expected outputs:

```text
outputs/
├── golden/
└── first_milestone_deterministic/
    ├── cast_vanilla/
    ├── cast_sparse/
    └── cast_pladis/
```

For controlled experiments, use:

```yaml
deterministic_self_attn: true
```

This avoids nuisance variation from the stock self-attention SDPA backend during single-sample comparisons.

---

## Pilot 0

The first qualitative pilot uses five prompts:

```text
P01  A dog barking loudly.
P02  Heavy rain falling on a roof.
P03  A dog barking while rain falls in the background.
P04  A man speaking while a car horn sounds nearby.
P05  Footsteps on a wooden floor while birds chirp outside.
```

Each prompt is generated using:

```text
λ = 0.0
λ = 1.0
λ = 2.0
```

for a total of 15 audio samples.

```bash
python experiments/run_pilot0.py
```

Results are written to:

```text
outputs/pilot0/
```

---

## Current Validation Status

The implementation has passed the following checks:

- [x] Original Auffusion inference runs successfully.
- [x] Exactly 16 text cross-attention modules are modified.
- [x] Self-attention is not modified by CAST.
- [x] Original attention processors are restored after each run.
- [x] `λ = 0` reproduces dense attention within the numerical/kernel floor.
- [x] `λ = 1` exactly selects the Entmax sparse branch.
- [x] `λ = 2` follows the expected extrapolation rule.
- [x] No NaN / Inf was observed.
- [x] Identical initial noise is used across compared conditions.
- [x] Deterministic single-sample comparison is supported.

At the module level, the CAST dense path is bitwise-equivalent to the explicit dense Diffusers attention implementation in fp32 and differs from SDPA only at floating-point/kernel precision.

---

## Preliminary Observations

> These observations come from a very small pilot and are **not final experimental conclusions**.

The first pilot showed that the sparse intervention produces substantial changes in the generated latent trajectory.

Conditional Entmax-1.5 attention was also found to be extremely sparse, typically retaining only about **2–3 non-zero text positions out of 77** per query on average.

Informal listening suggested that:

- `λ = 1` can sound similar to Vanilla in some cases, while occasionally reducing perceived fidelity.
- `λ = 2` can be noticeably more aggressive.
- In the compositional prompt  
  *“A dog barking while rain falls in the background,”*  
  the background rain appeared less prominent under strong extrapolation.

These observations motivate the current **lambda-response sweep**, especially the interpolation region:

```text
λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}
```

The working hypothesis is that audio generation may prefer a **partial dense–sparse balance** rather than full sparse replacement or extrapolation.

---

## Evaluation Plan

CAST separates three aspects of generation quality.

### 1. Text-Audio Alignment

Primary automatic metric:

- **CLAP Score**

This measures global semantic similarity between the prompt and generated audio.

### 2. Event Preservation

For multi-event prompts, global CLAP similarity may not reveal whether every requested event is present.

We therefore plan to evaluate **event coverage / event preservation**, for example:

```text
Prompt:
"A dog barking while rain falls in the background."

Requested events:
[dog barking, rain]

Generated audio:
dog barking  -> present
rain         -> absent

Event Coverage = 1 / 2
```

### 3. Audio Fidelity

Human listeners separately judge:

- distortion
- muddiness
- unnatural texture
- overall perceptual quality

This separation is important because stronger sparse guidance may improve semantic selectivity while degrading secondary-event preservation or fidelity.

---

## Experimental Principles

To keep comparisons controlled:

- same checkpoint
- same prompt
- same initial latent
- same random seed
- same scheduler
- same inference steps
- same CFG scale
- same precision
- same deterministic self-attention backend
- same CAST processor

Only the intended attention hyperparameter should differ.

For the current lambda sweep:

```text
α = 1.5 fixed
λ varies
```

---

## Research Questions

CAST currently investigates:

1. **Does sparse cross-attention generalize from text-to-image to text-to-audio diffusion?**
2. **Does stronger sparsity improve text-audio semantic alignment?**
3. **Does sparse attention suppress secondary or background acoustic events?**
4. **Is interpolation between dense and sparse attention more suitable for audio than PLADIS-style extrapolation?**
5. **How does the sparse-attention regime differ between visual and additive acoustic generation?**

---

## Project Status

```text
Architecture analysis              DONE
Auffusion environment              DONE
CAST attention implementation      DONE
Dense parity validation            DONE
λ = 1 / λ = 2 validation           DONE
Pilot 0                            DONE
Lambda response sweep              IN PROGRESS
Quantitative evaluation            PLANNED
Human evaluation                   PLANNED
Full experiment                    PLANNED
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

---

## Disclaimer

CAST is currently a **research prototype**.

The observations above are preliminary and should not be interpreted as evidence that sparse cross-attention universally improves text-to-audio generation. Larger prompt sets, multiple seeds, quantitative metrics, and controlled human evaluation are required before drawing general conclusions.
