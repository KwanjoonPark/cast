# CAST Pilot 0 Report

Date: 2026-10-20. Project root `/home/reallab/parkkwanjoon/CAST`. Driver: `experiments/run_pilot0.py` with `experiments/configs/pilot0.yaml` (reuses the validated `cast.pipeline.run_one`; no method code changed). Raw numbers: `outputs/pilot0/pilot0_summary.json`; per-run `manifest.json` in each arm directory; console log `outputs/_pilot0.log`.

## 1. Executive Summary

- All 15 outputs (5 prompts × λ ∈ {0, 1, 2}) were generated: **15 / 15**.
- All runs were finite (audio, spectrogram, final latents): **0 NaN / Inf**, 0 invalid runs.
- Initial-noise equality was verified: one pre-sampled seed-42 latent (SHA-256 `ea92f6ab…`) is recorded in all 15 manifests; identical across every prompt and mode.
- Deterministic execution was used: `deterministic_self_attn = true` (math SDPA backend for the untouched self-attention) in all 15 runs; the P03 arms reproduce the first-milestone deterministic runs bitwise (identical final-latent and audio SHA-256).
- Runtime failures: **none**. No retries, no memory issues (peak 3.11 GB).

No audio-quality claims are made in this report.

## 2. Experimental Setup

| Setting | Value |
|---|---|
| Checkpoint | `auffusion/auffusion` (HF snapshot 211aeb85) |
| Prompts | P01_SINGLE_DOG "A dog barking loudly." / P02_SINGLE_RAIN "Heavy rain falling on a roof." / P03_TWO_DOG_RAIN "A dog barking while rain falls in the background." / P04_TWO_SPEECH_HORN "A man speaking while a car horn sounds nearby." / P05_TWO_FOOTSTEPS_BIRDS "Footsteps on a wooden floor while birds chirp outside." (verbatim) |
| α | 1.5 (exact `entmax15`) |
| λ | 0.0 (CAST Vanilla) / 1.0 (CAST Sparse) / 2.0 (CAST PLADIS) |
| Seed | 42 for every prompt and mode |
| CFG | 7.5, `negative_prompt=None` (zero null condition), processor applied to the whole `[uncond, cond]` batch |
| Steps | 100 (PNDM) |
| `deterministic_self_attn` | true (math SDPA; bitwise-repeatable) |
| dtype / autocast | fp16 weights / `torch.autocast("cuda")` on (same as the milestone) |
| Size | 256 × 1024 (latent 4 × 32 × 128) |
| Initial-noise control | `cast.pipeline.sample_initial_latents` (bitwise equal to the pipeline's internal seed-42 draw, verified in the milestone) sampled **once** and passed as `latents=` to all 15 runs |
| Processor | one `CastCrossAttnProcessor` class on the 16 `attn2` modules; only `lam` differs between arms |
| Stock golden | not an arm (validation already done in the milestone) |

## 3. Validation

Dry check on P01 (all three arms) before the remaining prompts, all PASS (`pilot0_summary.json → dry_check_P01`):

| Check | Result |
|---|---|
| `deterministic_self_attn == true` in every run | PASS |
| Exactly 16 `attn2` modules carry `CastCrossAttnProcessor` (of 32 processors) | PASS (`attn2_processor == ['CastCrossAttnProcessor']` in all manifests; `select_cross_attention` asserts 32 / 16) |
| `attn1` untouched (`AttnProcessor2_0`, stock class) apart from the math-SDPA kernel control | PASS |
| Original processors restored after each run | PASS (processor map equals the stock map after the runs) |
| Same initial latent for λ = 0 / 1 / 2 (SHA-256) | PASS (identical in all 15 runs) |
| α = 1.5 in all arms | PASS |
| λ exactly 0.0 / 1.0 / 2.0 | PASS |
| No NaN / Inf | PASS (processor finite-guard never fired; audio/spectrogram/latents finite) |
| Reproducibility vs. milestone | P03 λ=0/1/2 final-latent and audio SHA-256 identical to `outputs/first_milestone_deterministic/` |

## 4. Results

Rel-L2 vs λ=0 = ‖z_λ − z_0‖ / ‖z_0‖ on the final latent of the same prompt. Mean Entmax support = mean number of non-zero `entmax15` entries per query row over the conditional half of the CFG batch, averaged over the 16 layers and all 100 steps (min–max across layers in parentheses); not evaluated at λ=0.

| Prompt | Mode | λ | Final latent norm | Rel-L2 vs λ=0 | Mean Entmax support (of 77) | Runtime | Valid |
|---|---|---:|---:|---:|---:|---:|---|
| P01_SINGLE_DOG | CAST Vanilla | 0 | 90.83 | 0 | N/A | 7.8 s | finite ✔ |
| P01_SINGLE_DOG | CAST Sparse | 1 | 100.14 | 0.654 | 2.30 (1.93–2.88) | 9.3 s | finite ✔ |
| P01_SINGLE_DOG | CAST PLADIS | 2 | 114.91 | 0.893 | 2.33 (1.85–2.88) | 9.4 s | finite ✔ |
| P02_SINGLE_RAIN | CAST Vanilla | 0 | 77.24 | 0 | N/A | 7.4 s | finite ✔ |
| P02_SINGLE_RAIN | CAST Sparse | 1 | 76.07 | 0.297 | 2.73 (1.29–3.83) | 9.3 s | finite ✔ |
| P02_SINGLE_RAIN | CAST PLADIS | 2 | 75.49 | 0.567 | 2.82 (1.54–4.05) | 9.5 s | finite ✔ |
| P03_TWO_DOG_RAIN | CAST Vanilla | 0 | 76.92 | 0 | N/A | 7.5 s | finite ✔ |
| P03_TWO_DOG_RAIN | CAST Sparse | 1 | 77.76 | 0.399 | 2.40 (1.74–3.05) | 9.5 s | finite ✔ |
| P03_TWO_DOG_RAIN | CAST PLADIS | 2 | 93.05 | 0.838 | 2.45 (1.86–3.35) | 9.7 s | finite ✔ |
| P04_TWO_SPEECH_HORN | CAST Vanilla | 0 | 88.84 | 0 | N/A | 7.6 s | finite ✔ |
| P04_TWO_SPEECH_HORN | CAST Sparse | 1 | 95.54 | 0.585 | 2.34 (1.80–2.99) | 9.6 s | finite ✔ |
| P04_TWO_SPEECH_HORN | CAST PLADIS | 2 | 102.12 | 0.787 | 2.41 (1.74–2.94) | 9.7 s | finite ✔ |
| P05_TWO_FOOTSTEPS_BIRDS | CAST Vanilla | 0 | 76.83 | 0 | N/A | 7.7 s | finite ✔ |
| P05_TWO_FOOTSTEPS_BIRDS | CAST Sparse | 1 | 79.86 | 0.333 | 2.64 (1.20–3.87) | 9.6 s | finite ✔ |
| P05_TWO_FOOTSTEPS_BIRDS | CAST PLADIS | 2 | 86.91 | 0.741 | 2.80 (1.33–4.35) | 9.8 s | finite ✔ |

Peak GPU memory 3.11 GB for every run. Seed 42 and initial-latent SHA-256 `ea92f6abbdf335c0…` for every run. (The "all rows" support statistic, which includes the unconditional rows where entmax is uniform over all 77 keys, is ≈ 39.7 in every sparse/PLADIS run and is recorded in the manifests.)

## 5. Per-Prompt Outputs

All paths relative to `outputs/pilot0/`. Each arm directory contains `audio.wav` (16 kHz, 10 s, int16), `spectrogram.png`, `final_latents.pt` (final + initial latents + spectrogram tensor), `manifest.json`.

| Prompt | Vanilla (λ=0) | Sparse (λ=1) | PLADIS (λ=2) | δ_sparse | δ_pladis | ‖z₂ − z₁‖/‖z₁‖ |
|---|---|---|---|---:|---:|---:|
| P01 "A dog barking loudly." | `P01_SINGLE_DOG/cast_vanilla/audio.wav` | `P01_SINGLE_DOG/cast_sparse/audio.wav` | `P01_SINGLE_DOG/cast_pladis/audio.wav` | 0.654 | 0.893 | 0.550 |
| P02 "Heavy rain falling on a roof." | `P02_SINGLE_RAIN/cast_vanilla/audio.wav` | `P02_SINGLE_RAIN/cast_sparse/audio.wav` | `P02_SINGLE_RAIN/cast_pladis/audio.wav` | 0.297 | 0.567 | 0.443 |
| P03 "A dog barking while rain falls in the background." | `P03_TWO_DOG_RAIN/cast_vanilla/audio.wav` | `P03_TWO_DOG_RAIN/cast_sparse/audio.wav` | `P03_TWO_DOG_RAIN/cast_pladis/audio.wav` | 0.399 | 0.838 | 0.723 |
| P04 "A man speaking while a car horn sounds nearby." | `P04_TWO_SPEECH_HORN/cast_vanilla/audio.wav` | `P04_TWO_SPEECH_HORN/cast_sparse/audio.wav` | `P04_TWO_SPEECH_HORN/cast_pladis/audio.wav` | 0.585 | 0.787 | 0.546 |
| P05 "Footsteps on a wooden floor while birds chirp outside." | `P05_TWO_FOOTSTEPS_BIRDS/cast_vanilla/audio.wav` | `P05_TWO_FOOTSTEPS_BIRDS/cast_sparse/audio.wav` | `P05_TWO_FOOTSTEPS_BIRDS/cast_pladis/audio.wav` | 0.333 | 0.741 | 0.575 |

Spectrograms: `<prompt_dir>/<mode>/spectrogram.png` for each of the 15 arms.

## 6. Observed Numerical Behavior

Numerical observations only; none of these is a statement about audio quality or text-audio alignment.

- For every prompt, λ=2 moves the final latent further from λ=0 than λ=1 does (δ_pladis 0.57–0.89 vs δ_sparse 0.30–0.65), and λ=2 is also far from λ=1 (0.44–0.72). All of these are 50–100× above the fp16 kernel floor established in the milestone (≈ 0.007–0.01 rel-L2), so the three arms are numerically distinct outputs, not rounding variants.
- The final-latent norm increases with λ for four prompts (P01 91→100→115, P03 77→78→93, P04 89→96→102, P05 77→80→87) and decreases slightly for P02 (77.2→76.1→75.5). The sign of the norm change is therefore prompt-dependent; the direction λ=1 → λ=2 continues the direction λ=0 → λ=1 in all five prompts.
- Entmax support on conditional rows is very small everywhere: 2.3–2.8 non-zero keys of 77 on average, with per-layer means between 1.2 and 4.4. Single-event prompts P01 (2.30) and the compositional P03/P04 (2.40/2.34) are at the low end; P02 and P05 (2.73/2.64) are slightly higher. Support at λ=2 is marginally higher than at λ=1 for the same prompt (the λ=2 trajectory feeds different latents into the same entmax).
- The single-event / compositional distinction does not separate cleanly in these numbers: the largest δ values belong to P01 (single) and P03 (compositional); the smallest to P02 (single) and P05 (compositional).
- Runtime overhead of the explicit fp32 attention: λ=0 7.4–7.8 s, λ=1/2 9.3–9.8 s per 100-step sample (entmax sort), with the deterministic self-attention kernel.

## 7. Problems Encountered

None. No errors, retries, OOM events, or non-finite values. The only addition to the code base is the pilot driver and its config (`experiments/run_pilot0.py`, `experiments/configs/pilot0.yaml`); `cast/` and `third_party/` are unchanged.

## 8. Ready for Human Listening?

**YES.** Index: `outputs/pilot0/LISTENING_INDEX.md` (paths for all 15 files plus blank manual-notes fields per prompt).

## 9. Recommended Decision After Listening

The decision is deferred to manual listening. After listening, select one of:

- **A.** If λ=1 and/or λ=2 sound promising → proceed to the 30-prompt pilot / main experiment.
- **B.** If λ=2 is clearly too aggressive but λ=1 is promising → run a small λ sweep around 0.5–1.5.
- **C.** If both λ=1 and λ=2 suppress secondary events → investigate weaker sparsity / audio-specific sparsity control.
- **D.** If there is almost no audible difference → verify alignment quantitatively before changing the method.

Numerical facts relevant to that choice (not a recommendation): the λ=1 and λ=2 outputs differ substantially from λ=0 and from each other for all five prompts, so option D's premise would have to come from listening rather than from the latent numbers.
