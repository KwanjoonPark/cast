# CAST: Cross-Attention Sparsification for Text-to-Audio Generation

CAST is a research project investigating whether sparse cross-attention, originally explored in text-to-image diffusion, can improve **text-to-audio alignment** in diffusion-based audio generation.

The current implementation integrates **Entmax-based sparse cross-attention** into **Auffusion** at inference time only, without retraining or modifying the original model weights.

> **Status:** Research prototype / pilot stage  
> **Backbone:** Auffusion  
> **Main intervention:** Text-conditioning cross-attention (`attn2`)  
> **Training required:** No

---

## Motivation

Text-to-audio generation requires a model to align textual concepts with acoustic events over time and frequency.

CAST asks:

> **Can sparse cross-attention improve text-audio alignment without additional training?**

The project is inspired by PLADIS, which replaces dense Softmax attention with sparse Entmax attention and further extrapolates in the sparse-attention direction.

Audio, however, differs from images in an important way: multiple sound sources can overlap in the same time-frequency region. This makes aggressive sparsification potentially useful for selectivity, but also potentially harmful to secondary or background events.

CAST studies that trade-off.

---

## Method

For a text-conditioned cross-attention layer:

\[
L = \frac{QK^\top}{\sqrt{d}}
\]

Dense attention:

\[
W_{\text{dense}} = \operatorname{Softmax}(L)
\]

Sparse attention:

\[
W_{\text{sparse}} = \operatorname{Entmax}_{1.5}(L)
\]

CAST mixes the two attention weights as:

\[
W_{\lambda}
=
W_{\text{dense}}
+
\lambda
\left(
W_{\text{sparse}}
-
W_{\text{dense}}
\right)
\]

and computes:

\[
\operatorname{Attention}(Q,K,V)
=
W_{\lambda}V
\]

### Interpretation of lambda

| Mode | lambda | Interpretation |
|---|---:|---|
| CAST Vanilla | 0.0 | Dense Softmax attention |
| Partial Sparse | 0 < lambda < 1 | Interpolation toward sparse attention |
| CAST Sparse | 1.0 | Pure Entmax-1.5 attention |
| CAST PLADIS | 2.0 | Sparse-direction extrapolation |

For lambda > 1, the mixed attention weights can contain negative values. They are therefore treated as **extrapolated attention weights**, not probabilities.

---

## Auffusion Integration

CAST modifies only the text-conditioning cross-attention modules in Auffusion.

- `attn1`: self-attention — unchanged
- `attn2`: text cross-attention — replaced by CAST processor
- Total attention modules: 32
- CAST-modified cross-attention modules: 16

The original `third_party/Auffusion` and `third_party/PLADIS` source trees remain unchanged.

The CAST processor:

1. reuses Auffusion's original `Q`, `K`, and `V` projections,
2. computes attention logits explicitly in FP32,
3. computes Softmax and Entmax-1.5 from the same logits,
4. mixes the attention weights using lambda,
5. casts back to the model dtype before multiplying by `V`,
6. preserves the original output projection and residual path.

---

## Repository Structure

```text
CAST/
├── papers/
│   ├── PLADIS*.pdf
│   └── Auffusion*.pdf
│
├── third_party/
│   ├── PLADIS/                 # reference implementation, read-only
│   └── Auffusion/              # reference implementation, read-only
│
├── cast/
│   ├── config.py
│   ├── pipeline.py
│   ├── attention/
│   │   ├── sparse_ops.py
│   │   └── cast_processor.py
│   ├── integration/
│   │   └── auffusion.py
│   └── tests/
│
├── experiments/
│   ├── configs/
│   ├── env/
│   ├── run_first_milestone.py
│   ├── run_pilot0.py
│   └── validate_parity.py
│
├── outputs/
│   ├── golden/
│   ├── first_milestone/
│   ├── first_milestone_deterministic/
│   └── pilot0/
│
├── CAST_ANALYSIS_REPORT.md
├── CAST_IMPLEMENTATION_REPORT.md
├── CAST_PILOT0_REPORT.md
└── README.md
```

---

## Environment

The validated environment is pinned around the original Auffusion stack.

| Package | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.0.1+cu118 |
| diffusers | 0.18.2 |
| transformers | 4.31.0 |
| huggingface_hub | 0.16.4 |
| entmax | 1.3 |
| librosa | 0.9.2 |
| soundfile | 0.12.1 |

The project intentionally does **not** import the PLADIS runtime pipeline because the PLADIS and Auffusion repositories depend on incompatible Diffusers versions.

CAST re-implements only the required attention computation against the Auffusion / Diffusers 0.18.2 API.

### Setup

```bash
bash experiments/env/setup_env.sh
conda activate cast
```

---

## Reproducibility

Controlled experiments use:

- checkpoint: `auffusion/auffusion`
- 100 PNDM denoising steps
- CFG scale: 7.5
- resolution: `256 x 1024`
- FP16 model weights
- CUDA autocast
- fixed seed
- identical pre-sampled initial latent across compared conditions
- deterministic self-attention backend for single-sample comparisons

The three CAST arms use the **same processor implementation** and differ only in lambda.

---

## Validation

Before running pilot experiments, CAST was checked against the original Auffusion implementation.

### Dense parity

CAST with lambda = 0 reproduces dense attention to numerical precision:

- FP32 per-module parity: effectively identical
- FP16 per-module difference: approximately one FP16 ULP
- full U-Net difference: within the stock kernel repeat floor
- full-pipeline CAST lambda = 0 difference remains far smaller than the changes introduced by lambda = 1 or lambda = 2

### Sparse attention validation

The implementation verifies that:

\[
\lambda=0 \Rightarrow W=W_{\text{dense}}
\]

\[
\lambda=1 \Rightarrow W=W_{\text{sparse}}
\]

\[
\lambda=2 \Rightarrow W=2W_{\text{sparse}}-W_{\text{dense}}
\]

No NaN or Inf values were observed in the validated milestone runs.

---

## Pilot Experiment

The initial pilot uses five prompts covering single-event and compositional audio generation.

Examples:

```text
A dog barking loudly.

Heavy rain falling on a roof.

A dog barking while rain falls in the background.

A man speaking while a car horn sounds nearby.

Footsteps on a wooden floor while birds chirp outside.
```

The first pilot compares:

```text
lambda = 0.0
lambda = 1.0
lambda = 2.0
```

with identical initial noise.

A follow-up response sweep evaluates:

```text
lambda in {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}
```

while keeping:

```text
alpha = 1.5
```

fixed.

---

## Preliminary Observation

The initial pilot suggests that aggressive sparsification may behave differently in audio than in image generation.

In particular:

- Entmax-1.5 produces very sparse conditional attention support in Auffusion.
- Strong sparse-direction extrapolation can substantially alter the denoising trajectory.
- Preliminary listening suggests that secondary/background events may become weaker under aggressive sparsification.
- Sparse conditions may also introduce perceptual degradation in audio fidelity.

These observations are **preliminary** and are not treated as final performance claims.

---

## Evaluation Plan

CAST separates text-audio alignment from audio quality.

### Text-Audio Alignment

- CLAP Score
- Event Coverage / Event Preservation
- Human prompt-alignment rating

### Audio Quality

- Human quality rating
- distortion / artifact assessment
- distribution-level metrics such as FAD only when enough samples are available

For compositional prompts, event-level evaluation is important because a global text-audio similarity score may fail to detect missing secondary events.

---

## Research Questions

CAST currently focuses on the following questions:

1. Does sparse cross-attention improve text-audio alignment?
2. Is pure Entmax attention too sparse for overlapping acoustic events?
3. Is interpolation `0 < lambda < 1` more suitable for audio than PLADIS-style extrapolation `lambda > 1`?
4. Does stronger sparsification improve foreground semantic selectivity at the cost of background-event preservation?
5. Can an audio-specific sparsity schedule outperform direct transfer from text-to-image diffusion?

---

## Current Status

- [x] Auffusion environment reproduced
- [x] Original Auffusion golden output verified
- [x] CAST cross-attention processor implemented
- [x] lambda = 0 dense parity validated
- [x] lambda = 1 sparse attention validated
- [x] lambda = 2 PLADIS-style extrapolation validated
- [x] Deterministic comparison pipeline established
- [x] Initial 5-prompt pilot completed
- [ ] Lambda response sweep
- [ ] Larger prompt benchmark
- [ ] CLAP evaluation
- [ ] Event-level evaluation
- [ ] Human evaluation
- [ ] Final paper experiments

---

## References

### PLADIS

**PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity**

Introduces inference-time sparse cross-attention using Entmax and sparse-direction extrapolation.

### Auffusion

**Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation**

Adapts latent diffusion and text cross-attention from text-to-image generation to text-to-audio generation through spectrogram modeling.

---

## Disclaimer

CAST is currently an experimental research prototype.

The present results are based on small pilot studies and should not be interpreted as evidence of general performance improvement until larger quantitative and human evaluations are completed.
