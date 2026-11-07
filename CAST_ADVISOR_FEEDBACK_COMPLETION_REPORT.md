# CAST Advisor Feedback Completion Report

Status date: 2026-10-25. Branch `kwanjoon/advisor-feedback-continuation` (from `origin/wonseok/advisor-feedback` @ 1fd7fb3).
All numbers are copied from the analysis outputs regenerated in this session (`outputs/cfg_sweep_v1/analysis/delta_table.csv`,
`analysis_summary.json`) and from the committed `outputs/listen_prompts_v2/analysis/*` and `outputs/layer_ablation_v1/analysis/*`;
nothing was recomputed by hand. AQAScore was run in this task **only on the CFG sweep** (not on prompt-v2 or the layer ablation).

## 1. Executive Summary

- **Branch inherited:** yes. The teammate branch is a clean linear extension of `main`; the continuation branch was created without any destructive Git operation. Tests: 13 passed (CAST core code unchanged in this task).
- **CFG sweep:** complete. 1,080 / 1,080 RTX 2080 wavs received in the teammate's package (zip SHA-256 `a46b69fd…b98600bf1` verified), extracted with `experiments/verify_sweep_audio.py` (index-listed members only; no tracked file touched) and verified file-by-file against `audio_index.csv` (1080 SHA-256 matches, 16 kHz mono PCM_16, 160,000 samples each). Nothing regenerated.
- **AQAScore:** **complete.** `experiments/eval_aqascore.py` (Qwen2.5-Omni-7B thinker, bf16, `inference_mode`, probabilistic Yes/No, paper question template) scored 1,080 / 1,080 samples in 7.7 min on the A5000 (peak 16.8 GB). Smoke test passed before the run: the same sample scored three times gave bitwise-identical results (0.798187, max |Δ| = 0), finite log-probabilities, score in [0, 1], matching prompt 0.798 vs unrelated prompt 0.011. Output: no NaN/Inf, all in [0.033, 0.987], no duplicates, every row matches the index on path, SHA-256, prompt, seed and prompt_id.
- **3-metric analysis and figure:** `analyze_cfg_sweep_v1.py` rerun with the unchanged protocol; `fig_cfg_delta_grid_3metrics.{png,pdf}` produced (rows CFG 7.5 / 3.0 / 1.5; columns ΔCLAP, ΔAQAScore, ΔPQ).
- **Prompt-v2:** exploratory experiment reviewed earlier; unchanged, no AQAScore, no rerun.
- **Layer ablation:** reviewed earlier; unchanged, no AQAScore, no rerun.
- **Final no-CFG experiment (added 2026-10-25, Section 16):** true conditional-only inference verified at runtime (0 unconditional UNet passes); 24 prompts × 3 seeds × {Dense, PLADIS λ=2} = 144 / 144 generated on the A5000, 72 / 72 same-latent pairs, all three metrics scored. PLADIS did not beat Dense: ΔCLAP −0.046 [−0.077, −0.019], ΔAQAScore −0.011 [−0.052, +0.032], ΔPQ −0.062 [−0.149, +0.024]. Details in `CAST_NO_CFG_FINAL_REPORT.md`.
- **Same-hardware reproduction (added 2026-10-26, Section 17):** the RTX 2080 CFG sweep's Dense-vs-PLADIS conditions were reproduced on the RTX A5000 (432 / 432, 216 / 216 same-latent pairs) and combined with the A5000 no-CFG run; all nine deltas reproduce inside the original CIs (max difference 0.011). The final figure `paper_assets/final/figures/fig1_guidance_pladis_a5000` is hardware-consistent.

## 2. Advisor Questions

Restated from the handoff (advisor feedback of 2026-10-23), without additions:

1. The CFG 7.5 baseline may be "too good" to show a PLADIS effect. Does PLADIS help when the baseline is weakened by lowering CFG?
2. Performance appears to get worse as CFG increases. Does the PLADIS gap depend on CFG?
3. Present the result in the same form as the Table 1 figure of the 2-page paper, one row per CFG. Move the spectrogram figure to supplementary.
4. (Teammate-added, pre-registered) Is the PLADIS gain larger on ambiguous prompts than on clear prompts? (prompt-v2)
5. (Teammate-added) Does restricting PLADIS to high- or low-resolution cross-attention layers reduce the damage, and does one resolution carry the damage? (layer ablation)

## 3. CFG Sweep Setup

24 prompts (`experiments/prompts/cast_diagnostic_v1.json`, 4 categories × 6) × seeds {42, 123, 2026} × CFG {7.5, 3.0, 1.5} × 5 conditions {Dense, Sharp β=1.5, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2} = 1,080 samples. 100-step PNDM, 256×1024 mel, fp16, α=1.5, deterministic self-attention, intervention on the 16 `attn2` modules only, identical per-seed initial latents across conditions (and identical to the A5000 runs). Generated on RTX 2080 8 GB; the CFG 7.5 means reproduce the paper's A5000 numbers (Dense CLAP 0.555 / PLADIS 0.363 vs paper 0.555 / 0.364; PQ 5.918 / 5.288 vs 5.92 / 5.28; Dense AQAScore 0.906 vs the earlier A5000 multi-metric study ≈0.905, PLADIS ΔAQA −0.099 vs −0.100) although the audio is not bitwise identical. All paired comparisons are within the RTX 2080 set.

Statistics: prompt = unit (3 seeds averaged), Δ = condition − Dense at the same CFG and prompt, 95 % CI from 10,000 prompt-level bootstrap resamples, RNG seed 20261024 (teammate's script, protocol unchanged).

## 4. CFG Sweep — CLAP

Dense CLAP: **0.555 (CFG 7.5) → 0.531 (3.0) → 0.465 (1.5)** (−16.4 %).

| Condition | ΔCLAP @7.5 [95 % CI] | ΔCLAP @3.0 | ΔCLAP @1.5 |
|---|---|---|---|
| Sharp β=1.5 | −0.020 [−0.036, −0.004] | −0.013 [−0.025, −0.002] | −0.008 [−0.024, +0.006] |
| Entmax λ=0.5 | −0.010 [−0.025, +0.004] | −0.014 [−0.029, −0.000] | −0.008 [−0.020, +0.002] |
| Entmax λ=1 | −0.053 [−0.080, −0.025] | −0.026 [−0.046, −0.008] | −0.015 [−0.029, −0.002] |
| PLADIS λ=2 | **−0.193 [−0.231, −0.155]** | **−0.102 [−0.135, −0.070]** | **−0.070 [−0.099, −0.044]** |

Fraction of prompts on which PLADIS beats Dense (CLAP): 0/24, 3/24, 3/24. PLADIS absolute CLAP: 0.363, 0.430, 0.395.

## 5. CFG Sweep — AQAScore

Dense AQAScore: **0.906 (CFG 7.5) → 0.869 (3.0) → 0.784 (1.5)** (−13.5 %).

| Condition | ΔAQA @7.5 [95 % CI] | ΔAQA @3.0 | ΔAQA @1.5 |
|---|---|---|---|
| Sharp β=1.5 | −0.010 [−0.026, +0.007] | +0.019 [−0.006, +0.050] | +0.018 [−0.022, +0.064] |
| Entmax λ=0.5 | +0.001 [−0.011, +0.015] | +0.015 [−0.010, +0.046] | +0.022 [−0.014, +0.065] |
| Entmax λ=1 | −0.031 [−0.063, −0.002] | +0.013 [−0.011, +0.046] | +0.025 [−0.016, +0.076] |
| PLADIS λ=2 | **−0.099 [−0.157, −0.045]** | **−0.065 [−0.132, −0.008]** | −0.028 [−0.085, +0.032] |

Fraction of prompts on which PLADIS beats Dense (AQAScore): 3/24, 6/24, 8/24. PLADIS absolute AQAScore: 0.807, 0.804, 0.756.
Per-prompt PLADIS Δ seed consistency (mean pairwise r): 0.17 / 0.15 / 0.26 — lower than CLAP or PQ at CFG 7.5, i.e. the per-prompt AQAScore effect is noisier than the other two metrics. Spearman(Dense AQA, PLADIS ΔAQA): −0.20 / −0.01 / −0.23 (n = 24; not interpretable as evidence).
Event-level AQAScore (`aqascore/event_scores.csv`, 1,620 rows) was produced as a diagnostic and is not analysed here.

## 6. CFG Sweep — AudioBox PQ

Dense PQ: **5.918 → 5.886 → 5.791** (−2.2 %).

| Condition | ΔPQ @7.5 [95 % CI] | ΔPQ @3.0 | ΔPQ @1.5 |
|---|---|---|---|
| Sharp β=1.5 | −0.090 [−0.187, +0.007] | +0.021 [−0.068, +0.096] | −0.024 [−0.076, +0.030] |
| Entmax λ=0.5 | −0.087 [−0.173, −0.001] | +0.004 [−0.069, +0.064] | −0.009 [−0.052, +0.038] |
| Entmax λ=1 | −0.219 [−0.356, −0.081] | −0.024 [−0.117, +0.055] | −0.013 [−0.070, +0.041] |
| PLADIS λ=2 | **−0.630 [−0.890, −0.377]** | **−0.257 [−0.458, −0.078]** | −0.064 [−0.193, +0.055] |

Fraction of prompts on which PLADIS beats Dense (PQ): 5/24, 4/24, 13/24. PLADIS absolute PQ: 5.288, 5.629, 5.727.

## 7. Cross-Metric Interpretation

**Q1 — Dense baseline vs CFG.** Lowering CFG weakens Dense on all three metrics, but very unequally: CLAP −16.4 % (0.555 → 0.465), AQAScore −13.5 % (0.906 → 0.784), PQ −2.2 % (5.918 → 5.791). The two text-alignment metrics lose a comparable share; perceptual quality barely moves.

**Q2 — PLADIS λ=2 vs Dense.** PLADIS never exceeds Dense on any metric at any CFG (all nine point estimates negative). Statistically distinguishable from Dense (CI excludes 0): CLAP at 7.5, 3.0 and 1.5; AQAScore at 7.5 and 3.0; PQ at 7.5 and 3.0. Approximately neutral (CI includes 0): AQAScore at 1.5 (−0.028 [−0.085, +0.032]) and PQ at 1.5 (−0.064 [−0.193, +0.055]). On no metric does PLADIS win on a majority of prompts except PQ at CFG 1.5 (13/24, with Δ still negative on average).

**Q3 — Does the penalty shrink with lower CFG?** Yes on all three metrics, monotonically from 7.5 → 3.0 → 1.5: ΔCLAP −0.193 → −0.102 → −0.070 (the CFG 1.5 penalty is 36 % of the CFG 7.5 penalty), ΔAQA −0.099 → −0.065 → −0.028 (28 %), ΔPQ −0.630 → −0.257 → −0.064 (10 %). Relative to the Dense value at the same CFG: CLAP −35 % → −19 % → −15 %, AQAScore −11 % → −7 % → −4 %, PQ −11 % → −4 % → −1 %.

**Q4 — Cross-metric agreement: pattern A, with a CLAP residual.** All three metrics show a shrinking PLADIS penalty. The metrics differ in how far it shrinks at CFG 1.5: AQAScore and PQ become indistinguishable from Dense, while CLAP remains clearly below Dense. AQAScore therefore does not sit between the other two as a tie-breaker; it tracks PQ in reaching neutrality at CFG 1.5 and tracks CLAP in the direction and the significance at CFG 7.5 and 3.0. Outcome B (AQAScore flat while others shrink) and outcome C (only PQ recovers) are **not** observed.

**Q5 — Weak interventions.** Sharp β=1.5, Entmax λ=0.5 and Entmax λ=1 approach Dense as CFG decreases on CLAP (all penalties shrink; Sharp and λ=0.5 are within CI of 0 at 1.5, λ=1 remains just below 0) and are within CI of 0 on PQ at 3.0 and 1.5. On AQAScore their point estimates are **positive** at CFG 3.0 (+0.013 to +0.019) and CFG 1.5 (+0.018 to +0.025), but every one of these six CIs includes 0 (e.g. Entmax λ=1 at 1.5: +0.025 [−0.016, +0.076]); at CFG 7.5 Entmax λ=1 is below 0 on AQAScore (−0.031 [−0.063, −0.002]) and the other two are neutral. No weak intervention reliably outperforms Dense on any metric at any CFG. The sign difference between CLAP (slightly negative) and AQAScore (slightly positive) for the weak interventions at low CFG is a divergence worth stating, but it is inside the noise of both metrics.

Other observations: per-prompt PLADIS Δ consistency across seeds falls with CFG on CLAP (0.38 / 0.31 / 0.10) and PQ (0.55 / 0.38 / 0.20) and is low throughout on AQAScore (0.17 / 0.15 / 0.26); the weak interventions' AQAScore gains at low CFG should be read in that light.

## 8. Does Lower CFG Rescue PLADIS?

**No.** PLADIS λ=2 did not exceed Dense on CLAP, AQAScore or PQ at any CFG tested. **Partially, in the restricted sense** that the penalty shrank monotonically as CFG was lowered on all three metrics and, at CFG 1.5, became statistically indistinguishable from zero on AQAScore and PQ while remaining clearly negative on CLAP (−0.070 [−0.099, −0.044]). So the answer is **metric-dependent only at the lowest CFG**, and only between "clearly worse" and "not distinguishable from Dense"; no metric moves to "better". The weakened-baseline hypothesis is not supported: Dense CLAP fell 16 % and Dense AQAScore 13.5 % from CFG 7.5 to 1.5, and PLADIS still lost against both.

## 9. Prompt-v2 Findings (exploratory only; CLAP and PQ, no AQAScore)

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

## 10. Layer Ablation Findings (exploratory only; CLAP and PQ, no AQAScore)

CFG 3.0, 24 prompts × 3 seeds, λ=2 applied to the high-resolution group (down_blocks.0 + up_blocks.3, 5 modules) or the low-resolution group (down_blocks.2 + mid_block + up_blocks.1, 6 modules); non-selected modules run the λ=0 fp32 path so that only the selected layers differ from Dense. Pre-generation validation recorded in the manifest: all-λ0 ≡ sweep Dense and all-λ2 ≡ sweep PLADIS by audio SHA-256; censuses 5/6; attn1 untouched; processors restored.

| Applied layers | ΔCLAP [95 % CI] | ΔPQ [95 % CI] |
|---|---|---|
| all 16 (sweep) | −0.102 [−0.135, −0.070] | −0.257 [−0.455, −0.074] |
| high (5) | −0.055 [−0.075, −0.035] | −0.110 [−0.254, +0.030] |
| low (6) | −0.036 [−0.059, −0.015] | −0.132 [−0.249, −0.031] |
| high − low | −0.019 [−0.048, +0.011] | +0.022 [−0.109, +0.150] |

Conclusion retained: restricting the number of intervened layers reduced the damage relative to applying PLADIS everywhere (high−all +0.046 [+0.024, +0.069] CLAP; low−all +0.066 [+0.033, +0.100]), which is the expected consequence of intervening less, but neither subset exceeded Dense (CLAP CIs below 0 for both) and no clear high-vs-low resolution difference was established (both high−low CIs include 0). The middle-resolution group (L1) was not run.

## 11. What the Data Support

Manuscript-safe claims (CLAP, AQAScore, PQ; 24 prompts × 3 seeds; Auffusion only):

1. Lowering CFG weakens the Dense baseline on all three metrics (CLAP 0.555 → 0.531 → 0.465; AQAScore 0.906 → 0.869 → 0.784; PQ 5.918 → 5.886 → 5.791).
2. PLADIS λ=2 did not outperform Dense on any metric at any CFG tested; its loss was statistically clear on CLAP at all three CFG values and on AQAScore and PQ at CFG 7.5 and 3.0.
3. The PLADIS penalty decreased monotonically as CFG was lowered on all three metrics (ΔCLAP −0.193 → −0.102 → −0.070; ΔAQA −0.099 → −0.065 → −0.028; ΔPQ −0.630 → −0.257 → −0.064); at CFG 1.5 it was no longer distinguishable from zero on AQAScore and PQ, while it remained on CLAP.
4. Milder interventions (Sharp β=1.5, Entmax λ=0.5, Entmax λ=1) moved toward Dense as CFG decreased and did not reliably exceed it on any metric (their positive AQAScore point estimates at CFG ≤ 3.0 all have CIs that include 0).
5. The pattern is consistent with a possible interaction between guidance strength and cross-attention modification; the effect of the intervention depends strongly on the guidance setting.
6. (Exploratory, CLAP/PQ) Applying PLADIS to only 5 high- or 6 low-resolution layers at CFG 3.0 reduced but did not remove the loss; the two groups were not distinguishable.
7. (Exploratory, CLAP/PQ) On 16 v2 prompts, PLADIS again scored below Dense on CLAP; the 4 ambiguous prompts showed a smaller penalty that was not statistically distinguishable from the clear group's.

## 12. What the Data Do NOT Support

- "Lower CFG makes PLADIS beneficial" / "a weaker baseline reveals a PLADIS gain" / "PLADIS works at low CFG" (no metric exceeds Dense anywhere).
- "CFG amplification causes the PLADIS degradation" (mechanism untested; only a monotone trend over three CFG values was observed).
- "PLADIS is harmless at CFG 1.5" (CLAP CI is still entirely negative; AQAScore and PQ are indistinguishable from 0, not shown equal).
- "Mild sharpening improves semantic alignment at low CFG" (AQAScore point estimates are positive but every CI includes 0; CLAP points the other way).
- "AQAScore disagrees with CLAP/PQ" (it shows the same shrinking penalty; the only difference is reaching neutrality at CFG 1.5, as PQ does).
- "Layer selection fixes PLADIS" or "high-resolution layers are the cause" (neither subset beats Dense; high−low not distinguishable; L1 not run; no AQAScore).
- "Ambiguous prompts benefit from PLADIS" (n = 4, CI includes 0, confounded with domain and baseline weakness; no AQAScore).
- "Per-prompt effects at low CFG are noise" as a general statement (holds for the 24 v1 prompts on CLAP, r ≈ 0.10, not for the 16 v2 prompts, r ≈ 0.37).
- Any generalisation beyond Auffusion, or to CFG < 1.5.

## 13. Recommended Manuscript Changes

- **연구 배경:** Add one sentence that the strength of the baseline (guidance scale) is itself a confound for inference-time attention interventions, so the study evaluates the intervention at three guidance scales rather than one. Remove any sentence implying PLADIS is expected to help T2A.
- **연구 방법 및 실험 설정:** State the sweep explicitly (24 prompts × 3 seeds × CFG {7.5, 3.0, 1.5} × 5 conditions = 1,080 samples; identical initial latents per seed; deterministic self-attention; 16 attn2 modules only). Keep the three metrics (CLAP, AQAScore with Qwen2.5-Omni-7B, AudioBox PQ) and the statistical sentence (prompt-level paired Δ, 10,000-resample bootstrap 95 % CI). Mention once that the sweep audio was generated on an RTX 2080 and the earlier Table 1 run on an A5000, that the CFG 7.5 metric means agree, and that all paired comparisons are within one machine.
- **실험 결과 및 분석:** Replace the single-CFG Table 1 figure with `fig_cfg_delta_grid_3metrics.pdf` (rows CFG 7.5 / 3.0 / 1.5; columns ΔCLAP, ΔAQAScore, ΔPQ). Give the Dense absolute values per CFG in one line. State: no condition exceeds Dense; the PLADIS penalty shrinks monotonically on all three metrics; at CFG 1.5 AQAScore and PQ are indistinguishable from Dense while CLAP still shows a loss. Move the spectrogram figure to supplementary. One sentence each for the layer ablation and v2 as exploratory CLAP/PQ results with the wording of Sections 9–10.
- **결론:** Rewrite as an analysis conclusion: PLADIS-style sparse cross-attention did not improve Auffusion at any guidance scale on alignment (CLAP, AQAScore) or fidelity (PQ); its cost scales with guidance strength, nearly vanishing on AQAScore and PQ at CFG 1.5 while an alignment loss on CLAP remains. Phrase the mechanism only as "consistent with an interaction between guidance and cross-attention modification". Do not adopt the "generally applicable to T2A / effective on guidance-free models" framing — untested (CFG ≥ 1.5 only).
- **한계 (if space):** single backbone; 24 prompts; CFG ≥ 1.5; exploratory sub-studies without AQAScore; latent-diffusion decoder may shape the metrics.

## 14. Remaining Limitations

- Auffusion only; latent diffusion with a HiFi-GAN vocoder, so decoder effects are not separated.
- 24 prompts (sweep, ablation), 16 prompts (v2) with 4 ambiguous; 3 seeds.
- CFG minimum 1.5; CFG ≈ 1 (no effective guidance) untested.
- Prompt-v2 12-vs-4 imbalance and category/domain confounding; ambiguity not measured independently of baseline weakness; no AQAScore.
- Layer ablation at one CFG, without the middle-resolution group; no AQAScore; shares of harm sum to < 100 %, so effects may be non-additive.
- No causal test of the CFG mechanism (e.g., `apply_to_uncond` variants, guidance-free generation).
- AQAScore per-prompt effects are noisier across seeds (r ≈ 0.15–0.26) than CLAP/PQ; AQAScore is bounded near 0.9 at CFG 7.5 (ceiling).
- Hardware-specific audio: RTX 2080 and A5000 outputs are never mixed inside a paired comparison; all sweep metrics (CLAP, AQAScore, PQ) refer to the same RTX 2080 files.

## 15. Final Recommendation

**Ready for manuscript revision.** All three metrics are scored on the same 1,080 samples, the analysis is regenerated under the unchanged protocol, and the manuscript figure exists. The evidence supports the story "PLADIS did not outperform Dense at any tested CFG; the penalty decreased as CFG was lowered, indicating that the effect of sparse cross-attention depends strongly on the guidance setting" — with the explicit qualifier that at CFG 1.5 the penalty is indistinguishable from zero on AQAScore and PQ but not on CLAP. AQAScore does not contradict CLAP/PQ. The no-CFG question was subsequently tested directly (Section 16): the answer is also negative, so no further generation experiment is needed for this paper.

## 16. Final No-CFG Experiment (2026-10-25)

Full report: `CAST_NO_CFG_FINAL_REPORT.md`; code-path and runtime proof: `outputs/no_cfg_v1/NO_CFG_CODEPATH_VERIFICATION.md`.

1. **True no-CFG verification.** In `third_party/Auffusion/auffusion_pipeline.py` `do_classifier_free_guidance = guidance_scale > 1.0`; at `guidance_scale = 1.0` no null-condition embedding is built, the latent batch is not duplicated and the CFG arithmetic is skipped. A UNet pre-hook confirmed at runtime: 101 conditional calls per sample with batch 1 and 0 unconditional rows (vs batch 2 with one zero row per call at CFG 7.5); a negative prompt changes nothing at guidance 1.0; reruns are bitwise identical.
2. **Hardware.** NVIDIA RTX A5000 (this machine). The CFG sweep was generated on the teammate's RTX 2080; initial latents are identical across the two machines but audio is not bitwise comparable, so the no-CFG result is a separate within-run paired comparison and is plotted separately.
3. **Design.** 24 frozen prompts × seeds {42, 123, 2026} × {No-CFG Dense, No-CFG PLADIS λ=2} = 144 / 144 samples, 0 failures, 72 / 72 pairs with identical initial latents; all other settings as the sweep (100-step PNDM, 256×1024, fp16, α = 1.5, deterministic self-attention, 16 attn2 modules). Same evaluators and statistics as the sweep (prompt unit, 10,000 bootstrap resamples, rng 20261024).
4. **CLAP.** Dense 0.385, PLADIS 0.339, Δ −0.046 [−0.077, −0.019]; 5 / 24 prompts favour PLADIS → significant degradation.
5. **AQAScore.** Dense 0.680, PLADIS 0.669, Δ −0.011 [−0.052, +0.032]; 9 / 24 → no measurable difference.
6. **PQ.** Dense 5.731, PLADIS 5.669, Δ −0.062 [−0.149, +0.024]; 10 / 24 → no measurable difference.
7. **Relation to the CFG sweep (descriptive).** PLADIS Δ across CFG 7.5 → 3.0 → 1.5 → no CFG: CLAP −0.193 → −0.102 → −0.070 → −0.046 (toward zero, still negative and significant); AQAScore −0.099 → −0.065 → −0.028 → −0.011 (toward zero, neutral since CFG 1.5); PQ −0.630 → −0.257 → −0.064 → −0.062 (flat since CFG 1.5, neutral). No reversal. The Dense baseline itself is weakest without guidance (CLAP 0.385, AQAScore 0.680, PQ 5.731).
8. **Conservative interpretation.** Removing classifier-free guidance did not make PLADIS beneficial in Auffusion: the alignment penalty shrank but persisted on CLAP, while AQAScore and PQ were indistinguishable from Dense. This rejects the strongest form of the hypothesis ("PLADIS helps when guidance is absent") for this backbone and configuration and completes the guidance-dependence story: the cost of sparse cross-attention scales with guidance strength but does not turn into a gain at zero guidance. No causal mechanism is claimed; no generalisation to other backbones.

## 17. Same-Hardware Reproduction of the CFG Sweep (2026-10-26)

The previous RTX 2080 CFG sweep was reproduced on the RTX A5000 for hardware-consistent comparison with the no-CFG experiment (`outputs/a5000_guidance_sweep_v1/`, driver `run_cfg_sweep_v1.py --cfgs 7.5 3.0 1.5 --arms dense entmax_l2`; smoke-tested CFG code path, same latents, determinism; 432 / 432 generated in 64 min, 0 failures; CLAP, AQAScore and PQ scored 432 / 432, all finite). Only Dense and PLADIS λ=2 were reproduced (Sharp Softmax and Entmax λ ≤ 1 were not). The RTX 2080 results (Sections 4–8) are unchanged and remain the evidence for the full five-arm comparison.

| Guidance (all RTX A5000) | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] | Dense CLAP / AQA / PQ |
|---|---|---|---|---|
| CFG 7.5 | −0.192 [−0.228, −0.156] | −0.100 [−0.156, −0.046] | −0.641 [−0.902, −0.395] | 0.555 / 0.905 / 5.922 |
| CFG 3.0 | −0.102 [−0.135, −0.068] | −0.066 [−0.131, −0.009] | −0.252 [−0.443, −0.069] | 0.531 / 0.871 / 5.884 |
| CFG 1.5 | −0.068 [−0.096, −0.043] | −0.027 [−0.081, +0.029] | −0.061 [−0.196, +0.056] | 0.464 / 0.783 / 5.789 |
| No CFG | −0.046 [−0.077, −0.019] | −0.011 [−0.052, +0.032] | −0.062 [−0.149, +0.024] | 0.385 / 0.680 / 5.731 |

Reproduction check (A5000 vs RTX 2080, per metric and CFG): same sign 9 / 9, same significance class 9 / 9, A5000 delta inside the RTX 2080 CI 9 / 9, largest |Δ difference| 0.011 (PQ, CFG 7.5), largest |Dense difference| 0.004. Conclusion-level reproducibility is established; audio is not bitwise identical across GPUs. Answers (Q1–Q7): reproduced — yes; PLADIS beats Dense at any CFG — no; penalty shrinks with CFG — yes, monotonically on all metrics; trajectory with no-CFG — continues toward zero on CLAP and AQAScore, flat on PQ from CFG 1.5; any positive point estimate — none; any CI fully above zero — none; CLAP significant at no-CFG — yes. The manuscript sentence remains: PLADIS does not outperform Dense at any tested guidance setting; the gap decreases consistently as guidance is weakened; at low/no guidance AQAScore and PQ are indistinguishable from Dense while CLAP retains a residual loss; this suggests an interaction between guidance strength and sparse cross-attention (no causal claim, no generalisation beyond Auffusion). Files: `HARDWARE_REPRODUCTION_NOTE.md`, `outputs/a5000_guidance_sweep_v1/analysis/`, `paper_assets/final/{figures/fig1_guidance_pladis_a5000.*, tables/table_a5000_guidance_delta.*, tables/table_a5000_guidance_absolute.*}`.
