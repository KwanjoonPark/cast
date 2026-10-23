# CAST Lambda Response Sweep Report

Date: 2026-10-20. Driver `experiments/run_lambda_sweep0.py`, config `experiments/configs/lambda_sweep0.yaml`. Raw numbers: `outputs/lambda_sweep0/lambda_sweep_summary.json`; per-condition `manifest.json`; console log `outputs/_lambda_sweep0.log`. Only change to `cast/`: a generic `"cast"` entry in `cast/config.py: MODES` (α 1.5, λ given explicitly) so that `run_one` can be called with arbitrary λ; the processor, mixing rule, and pipeline are untouched (10/10 unit tests still pass).

## 1. Executive Summary

- The sweep completed: **35 / 35** conditions (5 prompts × 7 λ) available.
- **20 newly generated** (λ = 0.25, 0.50, 0.75, 1.50) and **15 reused from Pilot 0** (λ = 0, 1, 2) after manifest-equality checks; one reused arm was regenerated and matched Pilot 0 **bitwise** (final-latent and audio SHA-256).
- Invalid runs: **0**; no NaN / Inf; no runtime failures.
- Deterministic execution (`deterministic_self_attn = true`, math SDPA for the untouched self-attention) in all 35 conditions.
- All 35 conditions share one bitwise-identical seed-42 initial latent (SHA-256 `ea92f6abbdf335c0…`).

No semantic-quality conclusions are drawn here.

## 2. Motivation

Pilot-driven hypotheses (from manual listening to Pilot 0; not established facts): pure sparse attention (λ=1) may affect audio fidelity (some outputs perceived as less clean / blurred); λ=2 extrapolation may be too aggressive; background or secondary events (e.g. rain in P03) may be vulnerable. Pilot 0 also measured extremely sparse conditional Entmax support (≈ 2.3–2.8 of 77 keys). The sweep therefore samples the interpolation range 0 < λ < 1 alongside the original λ = 0 / 1 / 2 conditions and adds λ = 1.5, to see how the output responds as λ moves continuously from dense toward and beyond sparse attention.

## 3. Experimental Setup

| Setting | Value |
|---|---|
| Checkpoint | `auffusion/auffusion` (snapshot 211aeb85) |
| Prompts | the five Pilot 0 prompts, verbatim (P01–P05, listed in §6) |
| α | 1.5 (exact `entmax15`), fixed |
| λ | 0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00 |
| Attention rule | `W(λ) = W_dense + λ (W_sparse − W_dense)`, `W_dense = Softmax(QKᵀ/√d)`, `W_sparse = Entmax_1.5(QKᵀ/√d)`, no clamping (`cast/attention/sparse_ops.py: mix_attention_weights`) |
| Seed | 42 |
| Steps / CFG | 100 PNDM / 7.5, `negative_prompt=None` |
| Size / dtype / autocast | 256×1024 / fp16 / on |
| `deterministic_self_attn` | true for every run |
| Initial noise | `cast.pipeline.sample_initial_latents` once; the same tensor passed as `latents=` to every generated condition; SHA-256 equal in all 35 manifests |
| Processor | `CastCrossAttnProcessor` on the 16 `attn2` modules; `attn1` stock `AttnProcessor2_0` |

## 4. Reuse of Pilot 0 Outputs

Path taken: **reuse** for λ = 0 / 1 / 2 (15 conditions), generate the other 20.

Eligibility check (`lambda_sweep_summary.json → reuse_checks`): for each of the 15 Pilot 0 manifests, the fields checkpoint, prompt, seed, num_inference_steps, guidance_scale, height, width, dtype, autocast, apply_to_uncond, deterministic_self_attn, alpha, lam, initial-latent SHA-256, attn2 processor class, and finiteness were compared with the sweep configuration; **all 15 matched with zero mismatches**. Implementation identity was then proven by regenerating `P03_TWO_DOG_RAIN` at λ = 1.00 with the current code and comparing with Pilot 0: final-latent SHA-256 equal = True, audio SHA-256 equal = True (bitwise).

The reused files (`audio.wav`, `spectrogram.png`, `final_latents.pt`) were copied unchanged into `outputs/lambda_sweep0/<PID>/lambda_{0.00,1.00,2.00}/`; each copied `manifest.json` carries a `provenance.reused_from` field pointing at the Pilot 0 directory. `outputs/pilot0/` was not modified.

## 5. Validation

Pre-run validation on P03 (all seven λ) before the other prompts, all PASS (`validation.pre_run_checks`): `deterministic_self_attn` true; 7 conditions present; `attn2` = `['CastCrossAttnProcessor']` (16 modules; `select_cross_attention` asserts 32 / 16); `attn1` = `['AttnProcessor2_0']` (stock, unchanged apart from the math-SDPA kernel control); processors restored after each run; α = 1.5 everywhere; λ resolved exactly to [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; identical initial-latent SHA-256 across the seven arms; no NaN / Inf; Pilot 0 reproduced bitwise. The same properties hold for all 35 conditions (`all_initial_latents_identical = true`, `counts.invalid = 0`).

## 6. Numerical Results

Rel-L2 vs λ=0 = ‖z_λ − z_0‖/‖z_0‖ (final latent, same prompt). Step-wise Rel-L2 = ‖z_λj − z_λi‖/‖z_λi‖ for the preceding λ in the list. Mean Entmax support = mean non-zero `entmax15` entries per conditional query row (of 77), averaged over 16 layers × 100 steps, with the per-layer min–max; N/A at λ=0 (entmax not evaluated). Runtime per 100-step sample with the deterministic kernel; peak GPU memory 3.11 GB for all runs.

| Prompt | λ | Final latent norm | Rel-L2 vs λ=0 | Step-wise Rel-L2 | Mean Entmax support | Runtime | Valid |
|---|---:|---:|---:|---:|---:|---:|---|
| P01 "A dog barking loudly." | 0.00 | 90.83 | 0 | – | N/A | 7.8 s | ✔ (reused) |
| | 0.25 | 92.25 | 0.244 | 0.244 | 2.30 (1.91–2.86) | 9.6 s | ✔ |
| | 0.50 | 94.63 | 0.382 | 0.242 | 2.31 (1.94–2.86) | 9.6 s | ✔ |
| | 0.75 | 97.22 | 0.567 | 0.407 | 2.30 (1.95–2.86) | 9.7 s | ✔ |
| | 1.00 | 100.14 | 0.654 | 0.187 | 2.30 (1.93–2.88) | 9.3 s | ✔ (reused) |
| | 1.50 | 107.60 | 0.756 | 0.391 | 2.30 (1.87–2.86) | 9.7 s | ✔ |
| | 2.00 | 114.91 | 0.893 | 0.395 | 2.33 (1.85–2.88) | 9.4 s | ✔ (reused) |
| P02 "Heavy rain falling on a roof." | 0.00 | 77.24 | 0 | – | N/A | 7.4 s | ✔ (reused) |
| | 0.25 | 76.97 | 0.112 | 0.112 | 2.66 (1.27–3.62) | 9.7 s | ✔ |
| | 0.50 | 76.73 | 0.185 | 0.109 | 2.69 (1.27–3.64) | 9.7 s | ✔ |
| | 0.75 | 76.49 | 0.247 | 0.134 | 2.71 (1.28–3.74) | 9.7 s | ✔ |
| | 1.00 | 76.07 | 0.297 | 0.103 | 2.73 (1.29–3.83) | 9.3 s | ✔ (reused) |
| | 1.50 | 74.58 | 0.396 | 0.209 | 2.78 (1.38–3.98) | 9.7 s | ✔ |
| | 2.00 | 75.49 | 0.567 | 0.307 | 2.82 (1.54–4.05) | 9.5 s | ✔ (reused) |
| P03 "A dog barking while rain falls in the background." | 0.00 | 76.92 | 0 | – | N/A | 7.5 s | ✔ (reused) |
| | 0.25 | 76.84 | 0.083 | 0.083 | 2.37 (1.74–3.00) | 9.4 s | ✔ |
| | 0.50 | 76.95 | 0.217 | 0.175 | 2.37 (1.74–3.03) | 9.4 s | ✔ |
| | 0.75 | 77.14 | 0.322 | 0.224 | 2.40 (1.74–3.03) | 9.5 s | ✔ |
| | 1.00 | 77.76 | 0.399 | 0.193 | 2.40 (1.74–3.05) | 9.5 s | ✔ (reused) |
| | 1.50 | 81.18 | 0.510 | 0.384 | 2.42 (1.75–3.17) | 9.5 s | ✔ |
| | 2.00 | 93.05 | 0.838 | 0.530 | 2.45 (1.86–3.35) | 9.7 s | ✔ (reused) |
| P04 "A man speaking while a car horn sounds nearby." | 0.00 | 88.84 | 0 | – | N/A | 7.6 s | ✔ (reused) |
| | 0.25 | 90.14 | 0.258 | 0.258 | 2.29 (1.81–2.89) | 9.8 s | ✔ |
| | 0.50 | 92.39 | 0.475 | 0.372 | 2.32 (1.81–2.99) | 9.8 s | ✔ |
| | 0.75 | 94.04 | 0.537 | 0.174 | 2.33 (1.81–3.00) | 9.8 s | ✔ |
| | 1.00 | 95.54 | 0.585 | 0.160 | 2.34 (1.80–2.99) | 9.6 s | ✔ (reused) |
| | 1.50 | 98.50 | 0.693 | 0.371 | 2.37 (1.77–2.93) | 9.8 s | ✔ |
| | 2.00 | 102.12 | 0.787 | 0.338 | 2.41 (1.74–2.94) | 9.7 s | ✔ (reused) |
| P05 "Footsteps on a wooden floor while birds chirp outside." | 0.00 | 76.83 | 0 | – | N/A | 7.7 s | ✔ (reused) |
| | 0.25 | 77.23 | 0.120 | 0.120 | 2.54 (1.15–3.60) | 9.8 s | ✔ |
| | 0.50 | 78.06 | 0.199 | 0.106 | 2.57 (1.16–3.68) | 9.8 s | ✔ |
| | 0.75 | 78.91 | 0.266 | 0.095 | 2.60 (1.18–3.76) | 9.8 s | ✔ |
| | 1.00 | 79.86 | 0.333 | 0.120 | 2.64 (1.20–3.87) | 9.6 s | ✔ (reused) |
| | 1.50 | 82.53 | 0.567 | 0.359 | 2.72 (1.32–4.14) | 9.8 s | ✔ |
| | 2.00 | 86.91 | 0.741 | 0.402 | 2.80 (1.33–4.35) | 9.8 s | ✔ (reused) |

## 7. Lambda Response by Prompt

Numerical trends only.

- **P01 (dog)**: Rel-L2 vs λ=0 rises monotonically 0.24 → 0.38 → 0.57 → 0.65 → 0.76 → 0.89; the largest single steps are 0.50→0.75 (0.41) and the two extrapolation steps (0.39, 0.40); the 0.75→1.00 step is the smallest (0.19). Latent norm increases monotonically 90.8 → 114.9. Support ≈ 2.30 at every λ.
- **P02 (rain)**: smallest response of the five: 0.11 → 0.18 → 0.25 → 0.30 → 0.40 → 0.57; steps within [0, 1] are uniform (0.10–0.13) and grow beyond λ=1 (0.21, 0.31). Latent norm *decreases* with λ in [0, 1.5] (77.2 → 74.6) and is 75.5 at λ=2. Support 2.66 → 2.82, the widest per-layer range (1.3–4.1).
- **P03 (dog + rain)**: 0.08 → 0.22 → 0.32 → 0.40 → 0.51 → 0.84; the smallest λ=0.25 response of all prompts (0.08); the λ=1.5→2 step (0.53) is the largest single step in the whole sweep, and λ=2 alone accounts for the jump in latent norm (77.8 → 81.2 → 93.1). Support 2.37 → 2.45.
- **P04 (speech + horn)**: 0.26 → 0.48 → 0.54 → 0.59 → 0.69 → 0.79; a large 0.25→0.50 step (0.37) followed by small steps up to λ=1 (0.17, 0.16), then larger extrapolation steps (0.37, 0.34). Norm increases monotonically 88.8 → 102.1. Support ≈ 2.3–2.4.
- **P05 (footsteps + birds)**: 0.12 → 0.20 → 0.27 → 0.33 → 0.57 → 0.74; steps within [0, 1] are small and shrinking (0.12, 0.11, 0.09, 0.12), then roughly triple beyond λ=1 (0.36, 0.40). Norm increases monotonically 76.8 → 86.9. Support 2.54 → 2.80.

Common pattern across all five prompts: Rel-L2 vs λ=0 is monotonically increasing in λ; the step-wise Rel-L2 for λ > 1 (0.21–0.53) is larger than the typical step inside [0, 1] (0.08–0.24, with two prompts showing one larger step at 0.25→0.50 or 0.50→0.75); and λ = 0.25 already moves the latent by 0.08–0.26, i.e. 10–30× the fp16 kernel floor of ≈ 0.007–0.01 measured in the milestone.

## 8. Cross-Prompt Numerical Summary

| λ | Mean Rel-L2 vs λ=0 | Std (population, n=5) | Mean latent norm | Mean runtime | Mean Entmax support (cond.) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 82.13 | 7.6 s | N/A |
| 0.25 | 0.163 | 0.073 | 82.69 | 9.6 s | 2.43 |
| 0.50 | 0.292 | 0.116 | 83.75 | 9.7 s | 2.45 |
| 0.75 | 0.388 | 0.137 | 84.76 | 9.7 s | 2.47 |
| 1.00 | 0.454 | 0.141 | 85.88 | 9.4 s | 2.48 |
| 1.50 | 0.584 | 0.129 | 88.88 | 9.7 s | 2.52 |
| 2.00 | 0.765 | 0.111 | 94.50 | 9.6 s | 2.56 |

Descriptive remarks (numerical only): mean Rel-L2 grows sub-linearly inside [0, 1] (0.16, 0.29, 0.39, 0.45) and then adds ≈ 0.13 and ≈ 0.18 for the two extrapolation steps; mean latent norm increases slowly up to λ=1 (+4.6 %) and faster beyond (+15 % at λ=2). Mean support rises from 2.43 to 2.56 across the sweep; per §10 of the task, this is a trajectory statistic (λ changes the latents fed into the same entmax), not a direct effect of λ on sparsity. Runtime of λ ≠ 0 is 9.3–9.8 s regardless of λ (the entmax cost is paid once per call; mixing is free).

## 9. Human Listening Materials

- `outputs/lambda_sweep0/LISTENING_INDEX.md` — per prompt, the seven files in ascending λ order plus the blank manual-evaluation table.
- `outputs/lambda_sweep0/BLIND_LISTENING_INDEX.md` — the same 35 files relabelled A–G per prompt (random order, RNG seed 20261020) and served as copies `outputs/lambda_sweep0/blind/<PID>_<label>.wav` so that the label does not reveal λ; blank table per prompt.
- `outputs/lambda_sweep0/blind_key.json` — label → λ mapping (open after listening).
- Canonical λ-named directories are unchanged: `outputs/lambda_sweep0/<PID>/lambda_<λ>/{audio.wav, spectrogram.png, final_latents.pt, manifest.json}`.

## 10. Problems Encountered

1. First launch failed immediately with `KeyError: 'cast'`: the one-line addition of the generic `"cast"` mode to `MODES` had not been applied (whitespace mismatch in the patch). Fixed and relaunched; no GPU time was lost (the error occurred before any generation).
2. The first version of the blind index listed canonical paths containing the λ value, which defeats blinding. Replaced by blind copies in `blind/` (canonical files untouched).
3. No runtime, memory, or numerical problems: 35 / 35 finite, peak 3.11 GB, ~9.6 s per generated sample, total new GPU time ≈ 3.3 min.

## 11. Decision Pending Human Review

No λ is selected here; with five prompts and one seed any value is at most a *candidate* for the next, larger experiment. After listening (preferably blind), human review determines which of the following applies:

- **A.** A weak interpolation value (λ ≈ 0.25–0.75) appears most promising → use that region for the next quantitative pilot.
- **B.** λ ≈ 1 remains most promising → retain sparse attention without PLADIS extrapolation.
- **C.** All λ > 0 degrade fidelity → investigate weaker α / alternative audio-specific sparsity control.
- **D.** λ > 1 unexpectedly appears most promising → retain PLADIS-style extrapolation and validate quantitatively.

Numerical context for the review (not a recommendation): the interpolation arms are numerically distinct from λ=0 at every λ ≥ 0.25 for every prompt, and the extrapolation arms (1.5, 2.0) change the latent by larger per-step amounts than any interpolation step, so the listening set spans a clearly graded range of change.
