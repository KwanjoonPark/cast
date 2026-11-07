# CAST Final No-CFG Experiment

Date 2026-10-25. Branch `kwanjoon/advisor-feedback-continuation`. Outputs: `outputs/no_cfg_v1/` (audio under `cfg1/…`, `audio_index.csv`, `clap_scores.csv`, `aqascore/`, `audiobox/`, `analysis/`, `runtime_verification.json`, `NO_CFG_CODEPATH_VERIFICATION.md`). All numbers below are copied from `outputs/no_cfg_v1/analysis/delta_table.csv`, `analysis_summary.json` and `per_prompt_delta.csv`; the CFG-sweep numbers are quoted from the committed `outputs/cfg_sweep_v1/analysis/delta_table.csv` and were not recomputed.

## 1. Research Question

In a **true no-CFG** Auffusion inference setting (conditional branch only, no unconditional denoiser prediction, no `eps_uncond + s·(eps_cond − eps_uncond)`), does PLADIS λ=2 outperform the corresponding Dense baseline? This is the last open form of the advisor's hypothesis after the CFG sweep (7.5 / 3.0 / 1.5) showed a PLADIS penalty that shrank with guidance strength but never reversed.

## 2. What "No-CFG" Means in This Implementation

`third_party/Auffusion/auffusion_pipeline.py` (submodule f44233d, diffusers 0.18.2), `AuffusionPipeline.__call__` line 931: `do_classifier_free_guidance = guidance_scale > 1.0`. With `guidance_scale = 1.0` the flag is `False`, so (i) `_encode_prompt` never creates the null-condition tensor (`torch.zeros_like(prompt_embeds)`, lines 733–736) and never concatenates `[uncond, cond]` (line 767); (ii) the denoising loop does not duplicate the latent batch (line 969); (iii) the UNet is called once per step on a batch of one conditional sample; (iv) the CFG arithmetic (`chunk(2)` + `uncond + s·(text − uncond)`, lines 982–984) and the guidance-rescale block (986) are skipped. Any `negative_prompt` argument is ignored. Values below 1.0 take the identical code path, so `guidance_scale = 0` would not have been a different setting; 1.0 was used. The CAST processor needs no change: with batch 1 its `half = batch_size // 2` is 0 and all rows are conditional. Full code-level and runtime documentation: `outputs/no_cfg_v1/NO_CFG_CODEPATH_VERIFICATION.md`.

## 3. Runtime Verification

`experiments/verify_no_cfg_runtime.py` (diagnostic script; a forward pre-hook on `pipe.unet`, removed at exit; no production code modified) on prompt A01, seed 42, RTX A5000, full generation settings:

| Run | guidance | UNet calls | latent batch | cond batch | zero (unconditional) rows |
|---|---|---|---|---|---|
| Dense λ=0 | 1.0 | 101 | 1 | 1 | 0 |
| PLADIS λ=2 | 1.0 | 101 | 1 | 1 | 0 |
| Dense rerun | 1.0 | 101 | 1 | 1 | 0 → audio and final latent bitwise identical to the first run |
| contrast Dense, 3 steps | 7.5 | 4 | 2 | 2 | 1 per call |
| Dense, 10 steps, `negative_prompt` None vs "A loud explosion." | 1.0 | 11 / 11 | 1 | 1 | 0 → outputs bitwise identical |

Unconditional forward passes under the no-CFG setting: **0** (conditional: 101 per sample; PNDM with `skip_prk_steps` adds one warm-up evaluation to the 100 steps). Dense and PLADIS start from the same initial latent (`ea92f6ab…` for seed 42). Output finite, 160,000 samples at 16 kHz (10.0 s). All checks `ALL_PASS = true`.

## 4. Experimental Setup

- Prompts: the frozen 24-prompt set `experiments/prompts/cast_diagnostic_v1.json` (4 categories × 6). Seeds 42, 123, 2026. Conditions: No-CFG Dense (CAST path, λ_entmax = 0) and No-CFG PLADIS (α = 1.5, λ_entmax = 2, `W = W_dense + 2·(Entmax₁.₅(Z) − W_dense)`). 24 × 3 × 2 = **144 samples, 144 / 144 generated, 0 failures**, 13.2 min.
- Settings identical to the CFG sweep except `guidance_scale = 1.0`: checkpoint `auffusion/auffusion`, PNDM 100 steps, 256×1024 mel, fp16 + autocast, deterministic self-attention (math SDPA), intervention on the 16 `attn2` modules only (`CastCrossAttnProcessor` on all 16; `attn1` on the stock `AttnProcessor2_0`), processors restored after each run.
- Same initial latent per prompt × seed: **72 / 72 pairs** have identical `initial_latents.sha256`; the three per-seed latents (`ea92f6ab…`, `6e34569f…`, `5ea8d1c0…`) are the same tensors used by the CFG sweep.
- Driver: the teammate's `experiments/run_cfg_sweep_v1.py --cfgs 1.0 --arms dense entmax_l2 --out outputs/no_cfg_v1 --experiment cast_no_cfg_v1` (only an `--experiment` name argument was added). Index: `make_sweep_index.py`.
- **Hardware: NVIDIA RTX A5000** (this machine). The CFG sweep audio was generated on the teammate's RTX 2080. The two runs share initial latents and settings but are not bitwise comparable; therefore the primary inference is the **within-run paired comparison** No-CFG PLADIS − No-CFG Dense, and the no-CFG result is shown in a separate figure. The CFG trend is compared descriptively only.
- Evaluation: the same validated implementations as the paper and the CFG sweep — CLAP `laion/clap-htsat-unfused` cosine (`score_sweep_clap.py`, `cast` env); AQAScore with the Qwen2.5-Omni-7B thinker, bf16, probabilistic Yes/No, same question template (`eval_aqascore.py`, `cast-eval` env; built-in validation: 3 repeats bitwise identical, finite, in range, matching 0.469 vs unrelated 0.012 on the first sample); AudioBox Aesthetics PQ (`eval_audiobox.py`, `cast-eval`). 144 / 144 scored on each metric, all finite, all joined to the index by path and SHA-256.
- Statistics (identical to the sweep): prompt = unit (3 seeds averaged), paired Δ = PLADIS − Dense per prompt, 10,000 prompt-level bootstrap resamples, 95 % percentile CI, RNG seed 20261024 (`boot_ci` imported from `analyze_cfg_sweep_v1.py`; metrics consume the generator in the same order clap, pq, aqa).

## 5. CLAP Results

| | Dense | PLADIS | Δ = PLADIS − Dense | 95 % CI | prompts PLADIS > Dense |
|---|---|---|---|---|---|
| No-CFG CLAP | **0.385** | **0.339** | **−0.046** | [−0.077, −0.019] | 5 / 24 (0.21) |

Classification: **Significant degradation** (CI entirely below 0). Relative to Dense: −11.9 %.

## 6. AQAScore Results

| | Dense | PLADIS | Δ | 95 % CI | prompts PLADIS > Dense |
|---|---|---|---|---|---|
| No-CFG AQAScore | **0.680** | **0.669** | **−0.011** | [−0.052, +0.032] | 9 / 24 (0.38) |

Classification: **No measurable difference** (negative point estimate, CI includes 0). Relative: −1.6 %.

## 7. AudioBox PQ Results

| | Dense | PLADIS | Δ | 95 % CI | prompts PLADIS > Dense |
|---|---|---|---|---|---|
| No-CFG PQ | **5.731** | **5.669** | **−0.062** | [−0.149, +0.024] | 10 / 24 (0.42) |

Classification: **No measurable difference** (negative point estimate, CI includes 0). Relative: −1.1 %.

## 8. Comparison with CFG 7.5 / 3.0 / 1.5

PLADIS λ=2 − Dense (previous rows RTX 2080, within-run paired; last row A5000, within-run paired; descriptive comparison only):

| Setting | ΔCLAP [95 % CI] | ΔAQAScore | ΔPQ | Dense CLAP / AQA / PQ |
|---|---|---|---|---|
| CFG 7.5 | −0.193 [−0.231, −0.155] | −0.099 [−0.157, −0.045] | −0.630 [−0.890, −0.377] | 0.555 / 0.906 / 5.918 |
| CFG 3.0 | −0.102 [−0.135, −0.070] | −0.065 [−0.132, −0.008] | −0.257 [−0.458, −0.078] | 0.531 / 0.869 / 5.886 |
| CFG 1.5 | −0.070 [−0.099, −0.044] | −0.028 [−0.085, +0.032] | −0.064 [−0.193, +0.055] | 0.465 / 0.784 / 5.791 |
| **No CFG** (separate hardware run) | **−0.046 [−0.077, −0.019]** | **−0.011 [−0.052, +0.032]** | **−0.062 [−0.149, +0.024]** | 0.385 / 0.680 / 5.731 |

Trajectory: **CLAP continues toward zero but stays negative and statistically clear** (−0.193 → −0.102 → −0.070 → −0.046; relative to Dense −35 % → −19 % → −15 % → −12 %). **AQAScore continues toward zero and is indistinguishable from Dense** since CFG 1.5 (−0.099 → −0.065 → −0.028 → −0.011). **PQ flattens**: the no-CFG Δ (−0.062) is essentially the CFG 1.5 Δ (−0.064), both indistinguishable from zero. **No metric reverses into a positive effect.** Removing guidance also weakens Dense further on the alignment metrics (CLAP 0.465 → 0.385, AQAScore 0.784 → 0.680 from CFG 1.5 to no CFG; PQ 5.791 → 5.731), so the no-CFG baseline is the weakest baseline tested, and PLADIS still did not beat it. Figures: `analysis/fig_no_cfg_pladis_delta.{png,pdf}` (primary) and `analysis/fig_cfg_to_nocfg_trend.{png,pdf}` (optional; the no-CFG point is drawn as an open red marker and the figure states the hardware difference).

## 9. Prompt-Level Heterogeneity

`analysis/per_prompt_delta.csv` (24 rows; Dense, PLADIS and Δ per metric). Counts of prompts (sign of Δ; and with a descriptive neutrality band |Δ| ≤ 0.01 for CLAP and AQAScore, ≤ 0.05 for PQ):

| Metric | Δ > 0 | Δ < 0 | improved / degraded / ≈neutral (band) | seed consistency of per-prompt Δ (mean pairwise r) |
|---|---|---|---|---|
| CLAP | 5 | 19 | 4 / 17 / 3 | 0.16 |
| AQAScore | 9 | 15 | 8 / 13 / 3 | 0.18 |
| PQ | 10 | 14 | 10 / 12 / 2 | 0.06 |

The per-prompt effects are weakly reproducible across seeds (r ≤ 0.18), so individual prompts should not be read as reliable winners or losers; the largest single-prompt movements (e.g. CLAP −0.27 on A03, −0.20 on D04; AQAScore +0.24 on C01, −0.22 on D01; PQ −0.54 on D04, −0.53 on B05) are listed only to characterise heterogeneity. No prompt subset is selected for the paper.

## 10. Does No-CFG Rescue PLADIS?

**No.** With classifier-free guidance removed entirely, PLADIS λ=2 did not outperform Dense on any of the three metrics. On CLAP it remained significantly below Dense (−0.046 [−0.077, −0.019]); on AQAScore and PQ the differences were small, negative in sign and not distinguishable from zero. The outcome is **case C on CLAP** ("removing CFG further reduced the PLADIS penalty but did not reverse it into a gain") and **"no measurable difference" on AQAScore and PQ**; taken together it is metric-dependent only between "still worse" and "indistinguishable", never "better". This directly rejects the strongest version of the advisor hypothesis — that PLADIS becomes useful when guidance is absent — for this backbone and configuration.

## 11. Claims Supported by the Data

1. In the tested no-CFG Auffusion setting (conditional-only inference, verified at runtime), PLADIS λ=2 did not outperform Dense on CLAP, AQAScore or AudioBox PQ (24 prompts × 3 seeds, paired, same initial latents).
2. The CLAP penalty was still statistically clear without guidance (−0.046 [−0.077, −0.019]); the AQAScore and PQ differences were not distinguishable from zero.
3. Read together with the CFG sweep, the PLADIS alignment penalty (CLAP) shrank monotonically as guidance was reduced and remained at no CFG; the AQAScore penalty shrank to neutrality; the PQ penalty reached neutrality at CFG 1.5 and stayed there.
4. The effect of the intervention depends strongly on the guidance setting; this is consistent with an interaction between sparse cross-attention and classifier-free guidance strength.
5. Lowering or removing guidance weakens the Dense baseline itself on the alignment metrics (CLAP 0.555 → 0.385, AQAScore 0.906 → 0.680 from CFG 7.5 to no CFG), far more than on PQ (5.918 → 5.731).

## 12. Claims NOT Supported by the Data

- "PLADIS is effective for T2A models without CFG" or "PLADIS works when guidance is absent" (no metric exceeds Dense; one backbone, one no-CFG configuration).
- "PLADIS is harmless without CFG" (CLAP CI is entirely negative).
- "AQAScore / PQ improve under PLADIS without CFG" (point estimates are negative; CIs include 0).
- "CFG amplification causes the PLADIS degradation" (only a monotone trend across guidance settings was observed; no causal test).
- Any statement that treats the no-CFG samples as hardware-identical extensions of the RTX 2080 sweep (different GPU; only within-run paired deltas are used).
- Generalisation to other T2A backbones or to other sparse-attention variants (Sharp Softmax, Entmax λ ≤ 1 were not run without CFG).

## 13. Recommended Final Paper Story

"PLADIS-style sparse cross-attention did not outperform dense attention in Auffusion at any guidance setting, including conditional-only inference without classifier-free guidance. Its cost decreased as guidance was reduced — the fidelity (PQ) and question-answering (AQAScore) penalties became indistinguishable from zero at low or no guidance — but an alignment (CLAP) penalty remained even without guidance. The effect of sparse cross-attention therefore depends strongly on the guidance setting, consistent with an interaction between the two, while never turning into an improvement in the tested setting." Report the no-CFG result as a separate paired experiment on the A5000 (Section 8 table or a one-line addition), keep the CFG-sweep 3×3 figure as the main figure, and move the spectrogram figure to supplementary. A CFG ≈ 1.1–1.2 or another-backbone experiment is not needed to support these sentences and is outside this paper's scope.

## 14. Same-Hardware Reproduction (added 2026-10-26)

To remove the hardware inconsistency noted in Sections 4 and 8, the Dense-vs-PLADIS CFG conditions were reproduced on the RTX A5000 (`outputs/a5000_guidance_sweep_v1/`: 24 prompts × 3 seeds × CFG {7.5, 3.0, 1.5} × {Dense, PLADIS λ=2} = 432 / 432, 216 / 216 same-latent pairs, same settings and evaluators) and combined with this no-CFG run. All nine A5000 deltas have the same sign and significance class as the RTX 2080 sweep and lie inside its 95 % CIs (largest difference 0.011, PQ at CFG 7.5; Dense means differ by ≤ 0.004). Unified A5000 result (PLADIS λ=2 − Dense): ΔCLAP −0.192 → −0.102 → −0.068 → −0.046 (all CIs < 0); ΔAQAScore −0.100 → −0.066 → −0.027 → −0.011 (CI includes 0 from CFG 1.5); ΔPQ −0.641 → −0.252 → −0.061 → −0.062 (CI includes 0 from CFG 1.5). The no-CFG point is therefore the continuation of a same-hardware trend, and the hardware caveat in Section 8 no longer applies to the final figure (`paper_assets/final/figures/fig1_guidance_pladis_a5000`). The RTX 2080 results are kept unchanged as reproducibility evidence. Details: `HARDWARE_REPRODUCTION_NOTE.md`, `outputs/a5000_guidance_sweep_v1/analysis/{delta_table_unified.csv, hardware_reproduction_comparison.csv}`.
