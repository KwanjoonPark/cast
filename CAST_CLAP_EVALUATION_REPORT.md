# CAST CLAP Evaluation Report

Date: 2026-10-22. Evaluation only; no audio was regenerated and `cast/` was not modified. Script: `experiments/eval_clap.py`. Outputs: `outputs/lambda_sweep0/clap_scores.{csv,json}`, plots in `outputs/lambda_sweep0/clap/`, log `outputs/_clap_eval.log`, overshoot/clamp analysis `outputs/lambda_sweep0/clap/clipping_sensitivity.json`.

## 1. Executive Summary

- CLAP model: `laion/clap-htsat-unfused` (HF snapshot `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`), via `transformers 4.31.0` `ClapModel`/`ClapProcessor` in the existing pinned `cast` env (no environment change was needed).
- Evaluated: **35 / 35** canonical samples (`outputs/lambda_sweep0/<PID>/lambda_<λ>/audio.wav`); all scores finite; no preprocessing errors.
- Deterministic: three complete re-evaluations (load → resample → feature extraction → embedding) of the same file/prompt gave identical scores (max difference 0.0).
- Broad numerical trend (n = 5 prompts, 1 seed): mean CLAP cosine is nearly flat for λ ∈ [0, 0.75] (0.649 → 0.639), drops at λ = 1 (0.598) and falls further at λ = 1.5 / 2 (0.511 / 0.446). Per-prompt shapes differ (one prompt rises up to λ = 1, two collapse at λ ≥ 1 or ≥ 1.5). No λ is selected.

## 2. Why CLAP

CLAP (Contrastive Language-Audio Pretraining) embeds text and audio into a shared space; the cosine similarity of the two embeddings is a global text-audio semantic-alignment score. Auffusion reports CLAP as one of its objective metrics and states it uses `laion/clap-htsat-unfused` (Auffusion README; paper Sec. 4.1, Tables 1–2). CLAP measures global similarity only: it is not a measure of audio fidelity, event completeness, or temporal structure.

## 3. Metric Definition

```
text_embedding  = L2_normalize( ClapModel.get_text_features(prompt) )
audio_embedding = L2_normalize( ClapModel.get_audio_features(audio) )
clap_cosine     = dot(audio_embedding, text_embedding)          # == cosine similarity
clap_percent    = 100 * clap_cosine
```

Both values are stored for every sample. Convention check: Auffusion's own CLAP script is not in its repository (the README points to `audioldm_eval` for FD/FAD/KL/IS and names only the HF model for CLAP; no local copy of `audioldm_eval`/`laion_clap` exists on this machine), so the exact formula could not be read from code. The paper reports CLAP as percentages in the 42–60 % range (Table 1), which is consistent with 100 × cosine of normalized embeddings and inconsistent with learned-scale logits (which would be ≈ 10–15). The percent values here are therefore reported as `100 × cosine` and labelled as such. The HF scaled logits were verified and **not** used: `logits_per_audio = 12.009 = cosine (0.6436) × exp(logit_scale_a) (18.66)`.

## 4. Model and Preprocessing

| Item | Value |
|---|---|
| Model / revision | `laion/clap-htsat-unfused`, snapshot `8fa0f1c6…`, weights SHA-256 recorded in `clap_scores.json → model_info` |
| Library | `transformers 4.31.0` (`ClapModel`, `ClapProcessor`, `ClapFeatureExtractor`), `torch 2.0.1+cu118`, `torchaudio 2.0.2` |
| Model dtype / device | float32, CUDA, `model.eval()`, `torch.no_grad()` |
| Input audio | 16 000 Hz, mono, int16, 160 000 samples (10.00 s) for all 35 files (verified) |
| Loading | `scipy.io.wavfile.read` → float32 ÷ 32768 |
| Resampling | `torchaudio.functional.resample(16000 → 48000)` (windowed-sinc, deterministic) → exactly 480 000 samples for every file |
| CLAP frontend | official `ClapFeatureExtractor` (48 kHz, 64 mel, hop 480, `max_length_s = 10`, `nb_max_samples = 480000`, `truncation = rand_trunc`, `padding = repeatpad`); input features shape (1, 1, 1001, 64) for every file |
| Duration handling | 480 000 resampled samples = `nb_max_samples`, so neither random truncation nor padding is triggered (confirmed by the repeatability test) |
| Determinism | `torch.manual_seed(0)`, eval mode, no dropout, no random crop; repeat test max diff 0.0 |
| Text | the exact prompt strings; `ClapProcessor` tokenizer with padding |

Overshoot note (sanity check): no file is clipped by loading, but sinc resampling of waveforms that touch full scale produces overshoot above 1.0 in 20 of the 35 files (peak up to 1.56; affected fraction ≤ 0.12 % of samples per file; 7 files have input peaks ≥ 0.999, all at λ ≥ 1.5 or P03 λ = 2). The primary scores use the unmodified resampled signal (identical preprocessing for every condition). Re-scoring those 20 files with the signal clamped to [−1, 1] changes CLAP by at most 4.7e-3 cosine (`clipping_sensitivity.json`), an order of magnitude below the smallest λ effects discussed below.

## 5. Validation

| Check | Result |
|---|---|
| Repeatability (P03 λ=1.00, 3 full evaluations) | scores 0.643550, 0.643550, 0.643550 → max diff **0.0** |
| All 35 audio paths found | yes |
| All outputs finite | yes |
| Input sample rate / channels / length | 16 000 Hz / mono / 160 000 for all 35 |
| Resampled length | 480 000 for all 35 |
| Preprocessing errors | none |
| Provenance carried from manifests | α = 1.5, seed = 42, checkpoint `auffusion/auffusion`, initial-latent SHA-256 `ea92f6ab…` identical for all 35 |

## 6. Per-Sample Results

Full table in `outputs/lambda_sweep0/clap_scores.csv` (columns: prompt_id, prompt, lambda, audio_path, clap_cosine, clap_percent, alpha, seed, checkpoint, initial_latent_sha256, source, event_* ). Compact view (CLAP cosine):

| Prompt \ λ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P01 "A dog barking loudly." | 0.590 | 0.602 | 0.625 | 0.629 | 0.655 | 0.600 | 0.559 |
| P02 "Heavy rain falling on a roof." | 0.608 | 0.575 | 0.551 | 0.573 | 0.389 | 0.414 | 0.355 |
| P03 "A dog barking while rain falls in the background." | 0.672 | 0.673 | 0.658 | 0.663 | 0.644 | 0.592 | 0.498 |
| P04 "A man speaking while a car horn sounds nearby." | 0.810 | 0.807 | 0.802 | 0.790 | 0.778 | 0.732 | 0.617 |
| P05 "Footsteps on a wooden floor while birds chirp outside." | 0.563 | 0.571 | 0.569 | 0.541 | 0.524 | 0.219 | 0.202 |

## 7. Lambda-Level Results

Across the five prompts (population std; n = 5, no significance testing):

| λ | Mean CLAP (cosine) | Std | Min | Max | Mean CLAP (%) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.649 | 0.088 | 0.563 | 0.810 | 64.9 |
| 0.25 | 0.646 | 0.089 | 0.571 | 0.807 | 64.6 |
| 0.50 | 0.641 | 0.089 | 0.551 | 0.802 | 64.1 |
| 0.75 | 0.639 | 0.086 | 0.541 | 0.790 | 63.9 |
| 1.00 | 0.598 | 0.132 | 0.389 | 0.778 | 59.8 |
| 1.50 | 0.511 | 0.178 | 0.219 | 0.732 | 51.1 |
| 2.00 | 0.446 | 0.150 | 0.202 | 0.617 | 44.6 |

Plot: `outputs/lambda_sweep0/clap/clap_mean_vs_lambda.png`.

## 8. Delta CLAP vs Lambda=0

ΔCLAP(λ) = CLAP(λ) − CLAP(0) per prompt, then averaged:

| λ | Mean ΔCLAP vs λ=0 | Std | Min | Max |
|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 |
| 0.25 | −0.003 | 0.016 | −0.033 (P02) | +0.012 (P01) |
| 0.50 | −0.008 | 0.030 | −0.058 (P02) | +0.035 (P01) |
| 0.75 | −0.010 | 0.029 | −0.036 (P02) | +0.039 (P01) |
| 1.00 | −0.051 | 0.100 | −0.219 (P02) | +0.065 (P01) |
| 1.50 | −0.137 | 0.123 | −0.344 (P05) | +0.010 (P01) |
| 2.00 | −0.203 | 0.126 | −0.361 (P05) | −0.031 (P01) |

Plot: `outputs/lambda_sweep0/clap/clap_delta_vs_lambda.png`.

## 9. Prompt-Level Response

Numerical curve shapes only (no audio-quality statements).

- **P01 (single: dog)** — the only prompt whose CLAP increases with λ: monotone rise 0.590 → 0.655 at λ = 1 (+0.065), then decline to 0.600 (λ = 1.5) and 0.559 (λ = 2, −0.031 vs λ = 0). Peak at λ = 1.
- **P02 (single: rain)** — decreases from λ = 0.25 onward with a non-monotone dip/rebound in [0.25, 0.75] (0.575, 0.551, 0.573), then a large drop at λ = 1 (0.389, −0.219), roughly flat afterwards (0.414, 0.355). Largest early-λ loss of all prompts.
- **P03 (dog + rain)** — flat up to λ = 0.75 (0.672 → 0.663, within ±0.014), then monotone decline: 0.644 (λ = 1), 0.592 (1.5), 0.498 (2; −0.174).
- **P04 (speech + horn)** — highest absolute CLAP; slow monotone decline over the whole range: −0.003, −0.008, −0.020, −0.032, −0.077, −0.193 at λ = 0.25 … 2.
- **P05 (footsteps + birds)** — flat/slightly up to λ = 0.5 (0.563 → 0.571 → 0.569), mild decline to 0.524 at λ = 1, then a collapse to 0.219 (λ = 1.5) and 0.202 (λ = 2; −0.361, the largest loss in the sweep).

Shape summary: in [0, 0.75] every prompt stays within ±0.04 of its λ = 0 value except P02 (−0.03 … −0.06); λ = 1 separates the prompts (P01 up, P02 sharply down, others slightly down); λ ≥ 1.5 lowers every prompt, with two prompts (P02 from λ = 1, P05 from λ = 1.5) falling to the 0.2–0.4 range. Plot: `outputs/lambda_sweep0/clap/clap_by_prompt_vs_lambda.png`.

## 10. Optional Event-Level CLAP (exploratory only)

`event_clap` = cosine between the same audio embedding and a single-event phrase (not event detection; not a main metric; not to be read as event presence). Phrases: P03 A "a dog barking" / B "rain falling"; P04 A "a man speaking" / B "a car horn"; P05 A "footsteps on a wooden floor" / B "birds chirping". Plot: `outputs/lambda_sweep0/clap/event_clap_exploratory.png`.

| Prompt | λ=0.00 A / B | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P03 dog / rain | 0.426 / 0.351 | 0.436 / 0.375 | 0.432 / 0.313 | 0.460 / 0.239 | 0.482 / 0.189 | 0.485 / 0.089 | 0.499 / −0.065 |
| P04 speech / horn | 0.213 / 0.424 | 0.210 / 0.429 | 0.213 / 0.454 | 0.208 / 0.447 | 0.211 / 0.442 | 0.186 / 0.389 | 0.139 / 0.307 |
| P05 footsteps / birds | 0.311 / 0.152 | 0.280 / 0.141 | 0.269 / 0.137 | 0.257 / 0.122 | 0.217 / 0.112 | 0.026 / 0.027 | 0.053 / 0.020 |

Exploratory observations: for P03 the "dog" similarity rises monotonically with λ (0.426 → 0.499) while the "rain" similarity falls monotonically (0.351 → −0.065), i.e. the two event similarities move in opposite directions as λ grows; for P04 both event similarities are roughly flat up to λ = 1 and both decline beyond; for P05 both decline with λ and both collapse at λ ≥ 1.5 together with the full-prompt score. These are embedding similarities to short phrases and are not evidence of event presence or absence.

## 11. Relation to Pilot Listening

Prior qualitative context from the Pilot 0 / sweep listening (manual, not a measurement): P03 λ = 2 was perceived to suppress the background rain; sparse (λ = 1) and PLADIS (λ = 2) outputs were sometimes perceived as less clean. Numerical comparison, without asserting that either source is "correct":

- The full-prompt CLAP for P03 decreases from λ ≥ 1 (−0.028 at λ = 1, −0.174 at λ = 2), and the exploratory rain-phrase similarity decreases over the whole range while the dog-phrase similarity increases. Both numbers move in the same direction as the listening note about rain at large λ; they neither confirm nor quantify event presence.
- The listening impression that λ = 1 is "roughly comparable" to vanilla is matched by the mean CLAP for three prompts (P03/P04/P05 within −0.04 of λ = 0 at λ = 1) but not for P02 (−0.219) and not in the opposite direction for P01 (+0.065). A global-similarity drop can coexist with, or be independent of, the perceived loss of cleanliness; CLAP cannot distinguish semantic drift from fidelity loss.
- Where CLAP stays flat or rises (P01 up to λ = 1; all prompts in [0, 0.75]) while listeners perceive blur or background loss, this should be read as a possible divergence between global semantic similarity and fine-grained event preservation / perceptual fidelity, not as a contradiction.

## 12. Limitations

- Five prompts, one seed, one checkpoint; population std across prompts is 0.09–0.18, comparable to the λ effects for λ ≤ 1.
- CLAP is a global embedding similarity; it is not a fidelity metric and cannot establish audio quality, naturalness, or artifacts.
- CLAP is known to be weak for fine-grained multi-event alignment, temporal relations, and foreground/background balance; the event-level phrase similarities are exploratory only.
- The percent convention (100 × cosine) is inferred from the paper's reported range, not read from Auffusion's evaluation code.
- Several λ ≥ 1.5 files touch full scale at the vocoder output; the evaluation does not alter them, and clamping sensitivity is ≤ 4.7e-3, but amplitude growth with λ (file RMS rises with λ for P01/P03/P04/P05, see `clipping_sensitivity.json`) is a property of the generated audio that CLAP does not separately account for.

## 13. Recommendation for Next Evaluation

Based on these CLAP curves the next evaluation should focus primarily on **fine-grained event preservation** (the P03 opposite-direction event similarities and the P02/P05 collapses at λ ≥ 1–1.5 are exactly the cases where a global metric cannot say which event is retained), and secondarily on **perceptual audio quality** (the interpolation range λ ≤ 0.75 is nearly flat in CLAP, so whether it differs from vanilla at all is a fidelity question, not a semantic one). Candidate region worth further evaluation on this basis: λ ≤ 0.75 (near-vanilla global alignment) and λ ≈ 1 for single-source prompts like P01; no λ is called optimal. No further metric was run.
