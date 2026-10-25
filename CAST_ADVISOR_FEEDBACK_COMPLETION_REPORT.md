# CAST Advisor Feedback Completion Report

Status date: 2026-10-25. Branch `kwanjoon/advisor-feedback-continuation` (from `origin/wonseok/advisor-feedback` @ 1fd7fb3).
All numbers are copied from the committed analysis outputs (`outputs/cfg_sweep_v1/analysis/delta_table.csv`, `analysis_summary.json`,
`outputs/listen_prompts_v2/analysis/*`, `outputs/layer_ablation_v1/analysis/*`); nothing was recomputed or edited by hand.
Sections that depend on AQAScore are marked **[PENDING AQAScore]** and must be filled after the scoring step in `HANDOFF_VERIFICATION.md` → Remaining Required Work.

## 1. Executive Summary

- **Branch inherited:** yes. The teammate branch is a clean linear extension of `main`; the continuation branch was created without any destructive Git operation. Tests: 13 passed.
- **CFG sweep:** generation complete on the teammate's RTX 2080 (1,080 / 1,080, no failures), CLAP and AudioBox PQ scored and analysed. **The 1,080 wav files are not on this machine** (gitignored; handed over as a 285 MB zip that has not arrived). Nothing was regenerated, because A5000 audio must not be mixed with the RTX 2080 audio that CLAP/PQ were scored on.
- **AQAScore:** **BLOCKED** on the missing audio. Evaluator (`experiments/eval_aqascore.py`, Qwen2.5-Omni-7B thinker, bf16, probabilistic Yes/No), environment (`cast-eval`) and model weights are verified ready; the analysis script already consumes its output; the 3-metric figure path was dry-run tested. One command sequence remains once the zip is on disk. Disk space (683 MB free) must be freed first.
- **Prompt-v2:** exploratory experiment complete (16 prompts, 3 seeds, CFG 3.0/1.5, 208 samples) and reviewed; no rerun. Conservative conclusion retained.
- **Layer ablation:** complete at CFG 3.0 (high 5 layers / low 6 layers, 144 new samples, pre-generation bitwise validation recorded) and reviewed; no rerun.

## 2. Advisor Questions

Restated from the handoff (advisor feedback of 2026-10-23), without additions:

1. The CFG 7.5 baseline may be "too good" to show a PLADIS effect. Does PLADIS help when the baseline is weakened by lowering CFG?
2. Performance appears to get worse as CFG increases. Does the PLADIS gap depend on CFG?
3. Present the result in the same form as the Table 1 figure of the 2-page paper, one row per CFG. Move the spectrogram figure to supplementary.
4. (Teammate-added, pre-registered) Is the PLADIS gain larger on ambiguous prompts than on clear prompts? (prompt-v2)
5. (Teammate-added) Does restricting PLADIS to high- or low-resolution cross-attention layers reduce the damage, and does one resolution carry the damage? (layer ablation)

## 3. CFG Sweep Setup

24 prompts (`experiments/prompts/cast_diagnostic_v1.json`, 4 categories × 6) × seeds {42, 123, 2026} × CFG {7.5, 3.0, 1.5} × 5 conditions {Dense, Sharp β=1.5, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2} = 1,080 samples. 100-step PNDM, 256×1024 mel, fp16, α=1.5, deterministic self-attention, intervention on the 16 `attn2` modules only, identical per-seed initial latents across conditions (and identical to the A5000 runs). Generated on RTX 2080 8 GB; the CFG 7.5 means reproduce the paper's A5000 numbers (Dense CLAP 0.555 / PLADIS 0.363 vs paper 0.555 / 0.364; PQ 5.918 / 5.288 vs 5.92 / 5.28) although the audio is not bitwise identical.

Statistics: prompt = unit (3 seeds averaged), Δ = condition − Dense at the same CFG and prompt, 95 % CI from 10,000 prompt-level bootstrap resamples, RNG seed 20261024 (teammate's script, unchanged).

## 4. CFG Sweep — CLAP

Dense CLAP: **0.555 (CFG 7.5) → 0.531 (3.0) → 0.465 (1.5)**.

| Condition | ΔCLAP @7.5 [95 % CI] | ΔCLAP @3.0 | ΔCLAP @1.5 |
|---|---|---|---|
| Sharp β=1.5 | −0.020 [−0.036, −0.004] | −0.013 [−0.025, −0.002] | −0.008 [−0.024, +0.006] |
| Entmax λ=0.5 | −0.010 [−0.025, +0.004] | −0.014 [−0.029, −0.000] | −0.008 [−0.020, +0.002] |
| Entmax λ=1 | −0.053 [−0.080, −0.025] | −0.026 [−0.046, −0.008] | −0.015 [−0.029, −0.002] |
| PLADIS λ=2 | **−0.193 [−0.231, −0.155]** | **−0.102 [−0.135, −0.070]** | **−0.070 [−0.099, −0.044]** |

Fraction of prompts on which PLADIS beats Dense (CLAP): 0/24, 3/24, 3/24. PLADIS absolute CLAP: 0.363, 0.430, 0.395.

## 5. CFG Sweep — AQAScore

**[PENDING AQAScore]** Not scored. The teammate could not run Qwen2.5-Omni-7B on 8 GB; the audio needed for scoring on this machine has not been received. When scored, this section must report Dense AQAScore at the three CFG values and the Δ table in the same layout as Section 4, taken from `delta_table.csv` rows with `metric == aqa`.

## 6. CFG Sweep — AudioBox PQ

Dense PQ: **5.918 → 5.886 → 5.791**.

| Condition | ΔPQ @7.5 [95 % CI] | ΔPQ @3.0 | ΔPQ @1.5 |
|---|---|---|---|
| Sharp β=1.5 | −0.090 [−0.187, +0.007] | +0.021 [−0.068, +0.096] | −0.024 [−0.076, +0.030] |
| Entmax λ=0.5 | −0.087 [−0.173, −0.001] | +0.004 [−0.069, +0.064] | −0.009 [−0.052, +0.038] |
| Entmax λ=1 | −0.219 [−0.356, −0.081] | −0.024 [−0.117, +0.055] | −0.013 [−0.070, +0.041] |
| PLADIS λ=2 | **−0.630 [−0.890, −0.377]** | **−0.257 [−0.458, −0.078]** | −0.064 [−0.193, +0.055] |

Fraction of prompts on which PLADIS beats Dense (PQ): 5/24, 4/24, 13/24. PLADIS absolute PQ: 5.288, 5.629, 5.727.

## 7. Cross-Metric Interpretation

What the two available metrics show together (AQAScore to be added):

- **Direction agrees.** On both CLAP and PQ, every intervention's Δ is ≤ 0 or indistinguishable from 0 at every CFG; PLADIS λ=2 is the worst condition at every CFG on both metrics.
- **Magnitude shrinks with CFG on both metrics, faster on PQ.** PLADIS ΔCLAP goes −0.193 → −0.102 → −0.070 (the CFG 1.5 penalty is ≈36 % of the CFG 7.5 penalty; relative to Dense −35 % → −19 % → −15 %). PLADIS ΔPQ goes −0.630 → −0.257 → −0.064 (≈10 %; relative −10.6 % → −4.4 % → −1.1 %).
- **Statistical status differs at CFG 1.5.** CLAP: the PLADIS CI stays entirely below 0 at all three CFG. PQ: the CI includes 0 at CFG 1.5 (−0.064 [−0.193, +0.055]) — the fidelity penalty is no longer distinguishable from zero there, while the alignment penalty is.
- **Weaker interventions converge to Dense.** Sharp β=1.5 and Entmax λ=0.5 are within CI of 0 on CLAP at CFG 1.5 and on PQ at CFG 3.0 and 1.5; Entmax λ=1 remains below 0 on CLAP at all CFG but with a shrinking penalty (−0.053 → −0.015).
- **Per-prompt consistency across seeds falls as CFG falls** (PLADIS Δ, mean pairwise Pearson r between seeds): CLAP 0.38 / 0.31 / 0.10, PQ 0.55 / 0.38 / 0.20. At CFG 1.5 the per-prompt PLADIS effect is close to seed noise on CLAP for these 24 prompts (the handoff notes that this does not hold for the v2 prompts, r ≈ 0.37, so it is not a general statement).
- **Dense strength vs PLADIS penalty:** Spearman(Dense score, PLADIS Δ) is −0.29 (CLAP) / −0.32 (PQ) at CFG 7.5 and near 0 at 3.0 and 1.5; with n = 24 and Δ containing the Dense score itself, this is not interpretable as evidence either way.
- **[PENDING AQAScore]** Whether AQAScore follows CLAP (penalty persists at CFG 1.5), follows PQ (penalty vanishes), or stays flat as in the earlier A5000 multi-metric study (where AQAScore was ≈flat for β ≤ 2 and λ ≤ 0.5 and dropped only for λ ≥ 1) decides the headline cross-metric sentence. Report whichever occurs.

## 8. Does Lower CFG Rescue PLADIS?

**No, on the two metrics available.** PLADIS λ=2 never exceeds Dense at any CFG: its ΔCLAP CI is entirely negative at CFG 7.5, 3.0 and 1.5, and its ΔPQ CI is negative at 7.5 and 3.0. **Partially, in the restricted sense** that the penalty shrinks monotonically as CFG is lowered and, on PQ at CFG 1.5, becomes statistically indistinguishable from zero (−0.064, CI spanning 0). It is therefore **metric-dependent at the lowest CFG**: alignment (CLAP) still shows a loss, fidelity (PQ) does not show a distinguishable loss. The weaker baseline hypothesis ("gain appears when Dense is weakened") is not supported: Dense CLAP fell 16 % from CFG 7.5 to 1.5 and PLADIS still lost 0.070 CLAP against it. **[PENDING AQAScore]** may add a third metric but cannot turn the CLAP result positive.

## 9. Prompt-v2 Findings (exploratory only)

Design actually run: 16 prompts = the first prompt of each of the 16 categories in `cast_prompts_v2.json` (clear 12: SG01, MU01, AT01, TM01, VO01, EN01, MS01, NA01, MC01, HL01, CU01, FX01; ambiguous 4: AB01, IM01, UN01, LX01), seeds 42/123/2026, CFG 3.0 and 1.5, Dense vs PLADIS λ=2 (plus stock-Auffusion "vanilla" at CFG 3.0 seed 42 only). The 12-vs-4 split is confounded with category/domain: all domain-axis "01" prompts fall in the clear group. This is not a balanced clear-vs-ambiguous benchmark.

PLADIS − Dense (prompt-level bootstrap CI):

| Metric, CFG | All (n=16) | Clear (n=12) | Ambiguous (n=4) | Ambiguous − Clear |
|---|---|---|---|---|
| CLAP 3.0 | −0.044 [−0.086, −0.001] | −0.058 [−0.108, −0.004] | −0.002 [−0.022, +0.018] | +0.056 [−0.003, +0.113] |
| CLAP 1.5 | −0.052 [−0.090, −0.017] | −0.063 [−0.104, −0.024] | −0.019 [−0.108, +0.036] | +0.044 [−0.046, +0.118] |
| PQ 3.0 | −0.151 [−0.358, +0.045] | −0.199 [−0.437, +0.020] | −0.009 [−0.321, +0.428] | +0.189 [−0.217, +0.673] |
| PQ 1.5 | −0.003 [−0.213, +0.194] | −0.080 [−0.321, +0.142] | +0.226 [−0.126, +0.577] | +0.305 [−0.118, +0.726] |

Dense CLAP means: all 0.400 / 0.342, clear 0.445 / 0.381, ambiguous 0.266 / 0.224 (CFG 3.0 / 1.5).

Conclusion retained: ambiguous prompts showed a smaller PLADIS penalty in this exploratory subset, but no statistically distinguishable PLADIS gain was established (all ambiguous−clear CIs include 0; the ambiguous group has 4 prompts and a much weaker Dense baseline, so ambiguity and baseline weakness cannot be separated). Correction already in the handoff: the single-seed "consistent gains" for IM01 and NA01 did not hold over three seeds; only FX01 gained on all three seeds at CFG 3.0.

## 10. Layer Ablation Findings (exploratory only)

CFG 3.0, 24 prompts × 3 seeds, λ=2 applied to the high-resolution group (down_blocks.0 + up_blocks.3, 5 modules) or the low-resolution group (down_blocks.2 + mid_block + up_blocks.1, 6 modules); non-selected modules run the λ=0 fp32 path so that only the selected layers differ from Dense. Pre-generation validation recorded in the manifest: all-λ0 ≡ sweep Dense and all-λ2 ≡ sweep PLADIS by audio SHA-256; censuses 5/6; attn1 untouched; processors restored.

| Applied layers | ΔCLAP [95 % CI] | ΔPQ [95 % CI] |
|---|---|---|
| all 16 (sweep) | −0.102 [−0.135, −0.070] | −0.257 [−0.455, −0.074] |
| high (5) | −0.055 [−0.075, −0.035] | −0.110 [−0.254, +0.030] |
| low (6) | −0.036 [−0.059, −0.015] | −0.132 [−0.249, −0.031] |
| high − low | −0.019 [−0.048, +0.011] | +0.022 [−0.109, +0.150] |

Conclusion retained: restricting the number of intervened layers reduced the damage relative to applying PLADIS everywhere (high−all +0.046 [+0.024, +0.069] CLAP; low−all +0.066 [+0.033, +0.100]), which is the expected consequence of intervening less, but neither subset exceeded Dense (CLAP CIs below 0 for both) and no clear high-vs-low resolution difference was established (both high−low CIs include 0). The middle-resolution group (L1) was not run.

## 11. What the Data Support

Manuscript-safe claims (CLAP and PQ; 24 prompts × 3 seeds; Auffusion only):

1. Lowering CFG weakens the Dense baseline (CLAP 0.555 → 0.531 → 0.465; PQ 5.918 → 5.886 → 5.791).
2. PLADIS λ=2 did not outperform Dense at any CFG tested; on CLAP its loss was statistically clear at all three CFG values.
3. The PLADIS penalty decreased monotonically as CFG was lowered (ΔCLAP −0.193 → −0.102 → −0.070; ΔPQ −0.630 → −0.257 → −0.064), and on PQ at CFG 1.5 it was no longer distinguishable from zero.
4. Milder interventions (Sharp β=1.5, Entmax λ=0.5, Entmax λ=1) moved toward Dense as CFG decreased, without exceeding it.
5. This pattern is consistent with a possible interaction between guidance strength and cross-attention modification.
6. (Exploratory) Applying PLADIS to only 5 high- or 6 low-resolution layers at CFG 3.0 reduced but did not remove the loss; the two groups were not distinguishable.
7. (Exploratory) On 16 v2 prompts, PLADIS again scored below Dense on CLAP; the 4 ambiguous prompts showed a smaller penalty that was not statistically distinguishable from the clear group's.

## 12. What the Data Do NOT Support

- "Lower CFG makes PLADIS beneficial" / "a weaker baseline reveals a PLADIS gain".
- "CFG amplification causes the PLADIS degradation" (mechanism untested; only a monotone trend over three CFG values was observed).
- "PLADIS is harmless at CFG 1.5" (CLAP CI is still entirely negative; only PQ is indistinguishable from 0).
- "Layer selection fixes PLADIS" or "high-resolution layers are the cause" (neither subset beats Dense; high−low not distinguishable; L1 not run).
- "Ambiguous prompts benefit from PLADIS" (n = 4, CI includes 0, confounded with domain and with baseline weakness).
- "Per-prompt effects at low CFG are noise" as a general statement (holds for the 24 v1 prompts, r ≈ 0.10, not for the 16 v2 prompts, r ≈ 0.37).
- Any statement about AQAScore for the CFG sweep until it is scored; any generalisation beyond Auffusion.

## 13. Recommended Manuscript Changes

- **연구 배경:** Add one sentence that the baseline strength itself (CFG) is a confound for inference-time attention interventions and that the present study therefore evaluates the intervention at three guidance scales rather than one. Remove any sentence implying PLADIS is expected to help T2A.
- **연구 방법 및 실험 설정:** State the sweep explicitly (24 prompts × 3 seeds × CFG {7.5, 3.0, 1.5} × 5 conditions = 1,080 samples; identical initial latents per seed; deterministic self-attention; 16 attn2 modules only). Keep the statistical protocol sentence (prompt-level paired Δ, 10,000-resample bootstrap 95 % CI). Mention hardware once (RTX 2080 for the sweep; A5000 for the earlier Table 1 run; metric means agree, audio not bitwise identical) and that all paired comparisons are within one machine.
- **실험 결과 및 분석:** Replace the single-CFG Table 1 figure with the 3-row grid (rows CFG 7.5 / 3.0 / 1.5; columns ΔCLAP, ΔAQAScore, ΔPQ once scored — until then ΔCLAP, ΔPQ). Report Dense absolute values per CFG in the text or a one-line table. State the monotone shrinkage and the CFG 1.5 PQ null; keep the CLAP loss explicit. Move the spectrogram figure to supplementary. Add one sentence each for the layer ablation and v2 as exploratory results with the conservative wording of Sections 9–10.
- **결론:** Rewrite as an analysis conclusion: PLADIS-style sparse cross-attention did not improve Auffusion at any guidance scale tested; its cost scales with guidance strength and nearly vanishes on fidelity at CFG 1.5 while an alignment loss remains. Phrase the mechanism only as "consistent with an interaction between guidance and cross-attention modification". Do not adopt the suggested "generally applicable to T2A / effective on guidance-free models" framing — the data do not test it (CFG ≥ 1.5 only).
- **한계 (if space):** single backbone; 24 prompts; CFG ≥ 1.5; AQAScore status; latent-diffusion decoder may shape the metrics.

## 14. Remaining Limitations

- Auffusion only; latent diffusion with a HiFi-GAN vocoder, so decoder effects are not separated.
- 24 prompts (sweep, ablation), 16 prompts (v2) with 4 ambiguous; 3 seeds.
- CFG minimum 1.5; CFG ≈ 1 (no effective guidance) untested.
- Prompt-v2 12-vs-4 imbalance and category/domain confounding; ambiguity not measured independently of baseline weakness.
- Layer ablation at one CFG, without the middle-resolution group; shares of harm sum to < 100 %, so effects may be non-additive.
- No causal test of the CFG mechanism (e.g., `apply_to_uncond` variants, guidance-free generation).
- Hardware-specific audio: RTX 2080 and A5000 outputs must not be mixed inside a paired comparison; AQAScore must be run on the RTX 2080 files.
- AQAScore for the sweep not yet scored; AQAScore for v2 and layer ablation not planned in this task.

## 15. Final Recommendation

**Not yet ready for the final manuscript revision; ready for a draft revision.** The CLAP/PQ story is complete, verified and stable, and the manuscript can be redrafted now with Sections 11–13 as the boundary of what may be claimed. The paper's results section should not be finalised until AQAScore for the 1,080 sweep samples is scored (one command once the audio package arrives and ~2 GB of disk is freed) and the 3-metric figure is regenerated, because the cross-metric sentence (Section 7, last bullet) and the AQAScore column of the figure depend on it. A CFG ≈ 1 experiment is **not** required for this paper's claims: the present sweep already answers the advisor's question negatively at three guidance scales; CFG ≈ 1 would only test the mechanistic extreme and is listed as future work.
