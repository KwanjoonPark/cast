# Hardware reproduction note: RTX 2080 CFG sweep → RTX A5000 (2026-10-26)

Purpose: remove the hardware inconsistency between the teammate's CFG sweep (RTX 2080, `outputs/cfg_sweep_v1/`) and the true no-CFG experiment (RTX A5000, `outputs/no_cfg_v1/`) by reproducing the main Dense-vs-PLADIS CFG conditions on the A5000 and combining them with the existing A5000 no-CFG run. This was a controlled reproduction, not a search for a different result. It was the final generation experiment.

## What was generated

- `outputs/a5000_guidance_sweep_v1/`: 24 frozen prompts × seeds {42, 123, 2026} × CFG {7.5, 3.0, 1.5} × {Dense (λ=0), PLADIS λ=2} = **432 / 432 samples, 0 failures, 64.1 min**, driver `experiments/run_cfg_sweep_v1.py --cfgs 7.5 3.0 1.5 --arms dense entmax_l2 --experiment cast_a5000_guidance_sweep_v1` (unchanged settings: `auffusion/auffusion` snapshot 211aeb85…, 100-step PNDM, 256×1024, fp16 + autocast, α=1.5, deterministic math-SDPA self-attention, 16 attn2 modules, processors restored).
- Hardware/software: NVIDIA RTX A5000 (24,564 MiB, driver 570.133.07), Python 3.10.20, torch 2.0.1+cu118, CUDA 11.8, diffusers 0.18.2, transformers 4.31.0. Evaluation: `cast-eval` (torch 2.4.1+cu124, transformers 4.57.1); AQAScore Qwen2.5-Omni-7B thinker bf16 (validation: 3 repeats identical, matching 0.788 vs unrelated 0.012); CLAP `laion/clap-htsat-unfused`; AudioBox Aesthetics PQ. All 432 scored on each metric, all finite, all joined to the index by path and SHA-256.
- Smoke test before the full run (`experiments/smoke_guidance_pairs.py`, `smoke_verification.json`): CFG 7.5 and CFG 1.5 pairs on A01 seed 42 — CFG code path confirmed by the UNet probe (latent/conditioning batch 2, one null-condition row per call, 101 calls), same initial latent for both arms, finite, 10.0 s @ 16 kHz, Dense rerun bitwise identical.
- Same-latent pairs: **216 / 216** (per-seed latents `ea92f6ab…`, `6e34569f…`, `5ea8d1c0…`, identical to the no-CFG run and to the RTX 2080 sweep). Index `audio_index.csv` carries `prompt_id, prompt, seed, cfg, arm, lam, guidance_scale, audio_path, audio_sha256, latent_sha256, gpu`.
- Statistics: unchanged CAST protocol (prompt unit, seeds averaged, paired Δ = PLADIS − Dense at the same CFG, 10,000 prompt-level bootstrap resamples, rng 20261024), `experiments/analyze_a5000_guidance_v1.py` (imports `load`/`boot_ci` from the sweep analyser).

## Comparison with the RTX 2080 sweep (`analysis/hardware_reproduction_comparison.csv`)

| Metric | CFG | Dense 2080 → A5000 | PLADIS Δ 2080 → A5000 | Δ difference | A5000 Δ inside 2080 CI | Significance 2080 → A5000 |
|---|---|---|---|---|---|---|
| CLAP | 7.5 | 0.555 → 0.555 | −0.193 → −0.192 | +0.001 | yes | neg → neg |
| CLAP | 3.0 | 0.531 → 0.531 | −0.102 → −0.102 | −0.000 | yes | neg → neg |
| CLAP | 1.5 | 0.465 → 0.464 | −0.070 → −0.068 | +0.002 | yes | neg → neg |
| AQAScore | 7.5 | 0.906 → 0.905 | −0.099 → −0.100 | −0.001 | yes | neg → neg |
| AQAScore | 3.0 | 0.869 → 0.871 | −0.065 → −0.066 | −0.001 | yes | neg → neg |
| AQAScore | 1.5 | 0.784 → 0.783 | −0.028 → −0.027 | +0.000 | yes | n.s. → n.s. |
| PQ | 7.5 | 5.918 → 5.922 | −0.630 → −0.641 | −0.011 | yes | neg → neg |
| PQ | 3.0 | 5.886 → 5.884 | −0.257 → −0.252 | +0.005 | yes | neg → neg |
| PQ | 1.5 | 5.791 → 5.789 | −0.064 → −0.061 | +0.002 | yes | n.s. → n.s. |

- **Qualitative CFG trend reproduced: YES.** On every metric the PLADIS penalty shrinks monotonically 7.5 → 3.0 → 1.5 on the A5000 exactly as on the RTX 2080, with the same significance pattern (CLAP negative at all three CFG; AQAScore and PQ negative at 7.5 and 3.0, indistinguishable from zero at 1.5).
- **Paired deltas numerically similar: YES.** Largest absolute difference 0.011 (PQ, CFG 7.5; CI half-width ≈ 0.25); all nine A5000 deltas fall inside the corresponding RTX 2080 95 % CIs; Dense means differ by ≤ 0.004.
- **Any metric where hardware changes the conclusion: NO.** The audio is not bitwise identical across GPUs (as documented in the handoff), but prompt-level means and paired deltas agree to the third decimal.

## Unified same-hardware result (`analysis/delta_table_unified.csv`, RTX A5000 for all four settings)

| Guidance | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] | Dense CLAP / AQA / PQ |
|---|---|---|---|---|
| CFG 7.5 | −0.192 [−0.228, −0.156] | −0.100 [−0.156, −0.046] | −0.641 [−0.902, −0.395] | 0.555 / 0.905 / 5.922 |
| CFG 3.0 | −0.102 [−0.135, −0.068] | −0.066 [−0.131, −0.009] | −0.252 [−0.443, −0.069] | 0.531 / 0.871 / 5.884 |
| CFG 1.5 | −0.068 [−0.096, −0.043] | −0.027 [−0.081, +0.029] | −0.061 [−0.196, +0.056] | 0.464 / 0.783 / 5.789 |
| No CFG | −0.046 [−0.077, −0.019] | −0.011 [−0.052, +0.032] | −0.062 [−0.149, +0.024] | 0.385 / 0.680 / 5.731 |

Fraction of prompts with PLADIS > Dense: CLAP 0/24, 2/24, 3/24, 5/24; AQAScore 4, 5, 8, 9; PQ 4, 4, 13, 10.

Remaining implementation difference between the rows (expected; this is the experimental variable): CFG > 1 runs the conditional and unconditional branches and combines them with the CFG formula; no CFG (guidance_scale = 1.0) runs the conditional branch only (`do_classifier_free_guidance = guidance_scale > 1.0`; runtime-verified 0 unconditional UNet passes). Prompt set, seeds, initial latents, checkpoint, sampler, attention settings and evaluators are identical across all four rows (asserted by the analysis script from the manifests).

## Answers to the core questions

- **Q1 Does the RTX 2080 conclusion reproduce on A5000?** Yes, numerically and qualitatively (table above).
- **Q2 Does PLADIS still fail to beat Dense at CFG 7.5 / 3.0 / 1.5?** Yes; all nine point estimates negative; CLAP significant at all three, AQAScore and PQ significant at 7.5 and 3.0.
- **Q3 Does the penalty still shrink as CFG decreases?** Yes, monotonically on all three metrics (CLAP −0.192 → −0.102 → −0.068; AQAScore −0.100 → −0.066 → −0.027; PQ −0.641 → −0.252 → −0.061).
- **Q4 Adding true no-CFG, does the trajectory continue toward zero?** CLAP and AQAScore continue toward zero (−0.046 and −0.011); PQ flattens at the CFG 1.5 level (−0.062 vs −0.061).
- **Q5 Does any metric become positive?** No; all twelve point estimates are negative.
- **Q6 Does any positive point estimate have a CI fully above zero?** No (there is no positive point estimate).
- **Q7 Is CLAP still significantly negative at no-CFG?** Yes: −0.046 [−0.077, −0.019].

## Interpretation (manuscript-safe)

PLADIS λ=2 does not outperform Dense at any tested guidance setting on the same hardware. The performance gap decreases consistently as guidance is weakened; at CFG 1.5 and without CFG, AQAScore and PQ are statistically indistinguishable from Dense, while CLAP retains a residual, significant loss. This suggests an interaction between classifier-free guidance strength and sparse cross-attention; it does not establish a causal mechanism and does not generalise beyond Auffusion. The RTX 2080 sweep results are kept unchanged as reproducibility evidence (`outputs/cfg_sweep_v1/`).
