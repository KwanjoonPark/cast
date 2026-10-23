# CAST Partial Sharp Guidance Report

Date: 2026-10-23. Scripts: `experiments/run_sharp_mix_v1.py` (generation; config `experiments/configs/sharp_mix_v1.yaml`), `experiments/eval_clap_diagnostic.py --root outputs/sharp_mix_v1 --cond-key lambda_sharp`, `experiments/eval_sharp_mix_metrics.py` (SHA-256 reuse + merge), `experiments/eval_aqascore.py` / `eval_audiobox.py` (new files only), `experiments/analyze_sharp_mix_v1.py`. Outputs under `outputs/sharp_mix_v1/`. Terminology throughout: **lambda_sharp** (this experiment, mixing weight toward the β=1.5 Sharp Softmax map) vs **lambda_entmax** (Diagnostic V1 Entmax/PLADIS mixing weight); **beta** = 1.5 fixed.

## 1. Executive Summary

| Item | Status |
|---|---|
| β | 1.5, fixed (the Sharp Softmax V1 endpoint) |
| lambda_sharp | 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50 (primary interest 0.25–0.75; >1 = signed extrapolation characterization) |
| Conditions | 504 = 24 development prompts × 3 seeds × 7; **360 newly generated** (52 min GPU), **144 reused** (72 dense = Diagnostic lambda_entmax=0 ≡ Sharp β=1; 72 Sharp β=1.5), all finite |
| Endpoint parity | lambda_sharp=0 and =1 are bitwise the dense and β=1.5 paths per module (unit test) and in three full regeneration checks (latents and audio SHA-256 equal) |
| Evaluation | CLAP PASS (504/504, reused endpoints reproduce stored CLAP exactly), AQAScore PASS (360 new + 144 reused by file SHA-256; deterministic), AudioBox PQ PASS (same), Event-AQAScore PASS |
| Development candidate | **NO DEVELOPMENT CANDIDATE IDENTIFIED.** No interpolation point has non-negative point estimates on all of CLAP, AQAScore and PQ: lambda_sharp=0.25 is indistinguishable from Dense (ΔCLAP −0.003, ΔAQA +0.004, ΔPQ −0.020, all CIs spanning 0); 0.50 and 0.75 keep AQAScore flat while CLAP (−0.011 / −0.010) and PQ (−0.060 / −0.078) drift down (CIs touch 0). Extrapolation (>1) degrades gradually, not abruptly. |

## 2. Motivation

Sharp Softmax (β=1.5) was far less destructive than Entmax/PLADIS but did not beat Dense on category means, while a few prompts (A01, A05, B03) improved and AQAScore stayed flat under moderate concentration as CLAP/PQ drifted down. The remaining question was whether the *direction* W_sharp − W_dense is useful at a smaller step than the full β=1.5 replacement, i.e. whether an interpolation regime exists where semantic metrics are preserved or improved with quality near Dense.

## 3. Method

For each cross-attention logits tensor Z = QKᵀ/√d (fp32):

```
W_dense   = Softmax(Z)
W_sharp   = Softmax(beta · Z),  beta = 1.5
W_partial = W_dense + lambda_sharp · (W_sharp − W_dense) = (1 − lambda_sharp)·W_dense + lambda_sharp·W_sharp
output    = W_partial.to(V.dtype) @ V
```

Implemented as `attention_type="sharp_mix"` in the existing `CastCrossAttnProcessor` (same Q/K/V projections, fp32 logits/softmax/mixing, no clipping, no renormalization, same 16 `attn2` modules, same CFG batch handling, scheduler, dtype, initial latents, deterministic self-attention; `attn1` untouched). The mixing reuses `mix_attention_weights`, so lambda_sharp=0 returns W_dense itself and lambda_sharp=1 returns W_sharp itself (bitwise). This is **not** the Entmax mix: there `W_sparse = entmax_1.5(Z)` and the weight is lambda_entmax; here both endpoints are Softmax distributions.

## 4. Validation

- Unit test (`test_sharp_mix_algebra_and_parity`, 13/13 tests pass): lambda_sharp=0 output `torch.equal` to the validated dense path; lambda_sharp=1 output `torch.equal` to the Sharp Softmax β=1.5 path; lambda_sharp=0.5 output equals 0.5·dense + 0.5·sharp (linearity in W, max diff < 1e-4·σ); on the weight level rows sum to 1 (±1e-4) for all seven values; W_partial ≥ −1e-6 for lambda_sharp ≤ 1; negative entries appear for lambda_sharp=1.5 on sharp logits and are not clamped.
- Dataset: SHA-256 `c89ac135…968d38` equals the Diagnostic V1 / Sharp Softmax V1 record; 24 prompts, 6 per category, unmodified. Seeds 42/123/2026 → initial-latent SHA-256 `ea92f6ab…`, `6e34569f…`, `5ea8d1c0…`, asserted equal to both prior manifests; generation settings asserted equal field by field.
- Regeneration checks (`experiment_manifest.json → checks.endpoint_regeneration`): D01/seed 42 at lambda_sharp=0 vs Sharp β=1.00 (≡ Diagnostic lambda_entmax=0), A01/seed 42 and D01/seed 2026 at lambda_sharp=1 vs Sharp β=1.50 — final-latent and audio SHA-256 equal in all three. The 144 endpoint conditions were therefore copied with `provenance.reused_from`.
- Metric reuse: AQAScore/AudioBox were run only on the 360 new file-byte SHA-256s (validation on the first new file: 3 repeats identical, 0.9627; matching vs unrelated prompt 0.963 vs 0.006); CLAP was recomputed for all 504 and the 144 reused files matched their stored values exactly.

## 5. Attention Statistics

Conditional-half means over 16 layers × 100 steps, then over prompts (`attention_stats.csv`; figures `sharp_mix_entropy.png`, `sharp_mix_negative_mass.png`):

| lambda_sharp | Type | H_norm (W_partial) | effective support exp(H) | min weight | negative fraction | negative mass / row | ‖z_T‖ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.00 | Dense | 0.673 | 18.6 | ≥ 0 | 0 | 0 | 86.7 |
| 0.25 | Interpolation | 0.598 | 13.5 | 4e-13 | 0 | 0 | 87.5 |
| 0.50 | Interpolation | 0.516 | 9.4 | 3e-13 | 0 | 0 | 88.2 |
| 0.75 | Interpolation | 0.425 | 6.3 | 3e-13 | 0 | 0 | 88.9 |
| 1.00 | Full Sharp | 0.324 | 4.1 | ≥ 0 | 0 | 0 | 89.7 |
| 1.25 | Extrapolation | N/A (signed weights) | N/A | −0.0014 | 0.263 | 0.0028 | 90.6 |
| 1.50 | Extrapolation | N/A (signed weights) | N/A | −0.0074 | 0.843 | 0.0452 | 91.4 |

Entropy decreases smoothly and almost linearly with lambda_sharp on [0, 1] (effective support 18.6 → 4.1). Beyond 1 the mixed map is signed: at 1.25 about a quarter of the conditional-half entries are negative but their total magnitude is tiny (0.003 per row); at 1.5 most entries are slightly negative (0.045 per row). The endpoint W_sharp(β=1.5) entropy (H_norm ≈ 0.32) is recorded for reference in every condition. These negative masses are far smaller than those of PLADIS lambda_entmax=2 (where W = 2·W_sparse − W_dense sets every key outside the ≈2.5-key Entmax support negative, i.e. roughly the whole dense tail).

## 6. Overall CLAP

| lambda_sharp | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean CLAP (24 prompts) | 0.555 | 0.553 | 0.544 | 0.546 | 0.535 | 0.527 | 0.512 |
| ΔCLAP [95 % CI] | 0 | −0.003 [−0.011, +0.006] | −0.011 [−0.024, +0.001] | −0.010 [−0.022, +0.003] | −0.021 [−0.035, −0.005] | −0.028 [−0.044, −0.011] | −0.043 [−0.067, −0.018] |
| median | 0.588 | 0.581 | 0.561 | 0.557 | 0.538 | 0.531 | 0.526 |

CLAP declines monotonically in the mean (0.75 ≈ 0.50); interpolation deltas are within CI of zero, the full-sharp and extrapolation deltas are not.

## 7. Overall AQAScore

| lambda_sharp | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean AQAScore | 0.905 | 0.909 | 0.902 | 0.908 | 0.892 | 0.888 | 0.893 |
| ΔAQA [CI] | 0 | +0.004 [−0.005, +0.013] | −0.003 [−0.013, +0.006] | +0.004 [−0.008, +0.017] | −0.013 [−0.036, +0.007] | −0.017 [−0.038, +0.002] | −0.012 [−0.043, +0.015] |
| mean Event-AQA (Δ) | 0.873 | 0.874 (+0.001) | 0.870 (−0.003) | 0.878 (+0.004) | 0.868 (−0.005) | 0.857 (−0.016) | 0.866 (−0.007) |

AQAScore is flat across the whole sweep (all CIs include 0); it sits near its ceiling (median 0.93).

## 8. Overall Production Quality

| lambda_sharp | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean PQ | 5.92 | 5.90 | 5.86 | 5.84 | 5.83 | 5.75 | 5.70 |
| ΔPQ [CI] | 0 | −0.020 [−0.063, +0.021] | −0.060 [−0.130, +0.007] | −0.078 [−0.164, +0.007] | −0.094 [−0.191, +0.004] | −0.172 [−0.297, −0.055] | −0.220 [−0.374, −0.074] |
| CE / CU / PC | 3.06 / 5.57 / 2.69 | 3.05 / 5.52 / 2.67 | 3.00 / 5.47 / 2.67 | 2.97 / 5.42 / 2.68 | 2.97 / 5.40 / 2.67 | 2.92 / 5.32 / 2.67 | 2.91 / 5.27 / 2.66 |

PQ declines monotonically with lambda_sharp; the interpolation CIs include 0 but their upper limits are ≤ +0.02, i.e. the data do not support any PQ gain. CE and CU follow PQ; PC is flat.

## 9. Joint Multi-Metric Analysis

Required summary (overall, 24 prompts; `overall_summary.csv`; figure `sharp_mix_multimetric_delta.png`):

| lambda_sharp | Type | CLAP | ΔCLAP | AQAScore | ΔAQA | PQ | ΔPQ | H_norm | Eff. support | Neg. mass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | Dense | 0.555 | 0 | 0.905 | 0 | 5.92 | 0 | 0.673 | 18.6 | 0 |
| 0.25 | Interpolation | 0.553 | −0.003 | 0.909 | +0.004 | 5.90 | −0.020 | 0.598 | 13.5 | 0 |
| 0.50 | Interpolation | 0.544 | −0.011 | 0.902 | −0.003 | 5.86 | −0.060 | 0.516 | 9.4 | 0 |
| 0.75 | Interpolation | 0.546 | −0.010 | 0.908 | +0.004 | 5.84 | −0.078 | 0.425 | 6.3 | 0 |
| 1.00 | Full Sharp | 0.535 | −0.021 | 0.892 | −0.013 | 5.83 | −0.094 | 0.324 | 4.1 | 0 |
| 1.25 | Extrapolation | 0.527 | −0.028 | 0.888 | −0.017 | 5.75 | −0.172 | N/A | N/A | 0.003 |
| 1.50 | Extrapolation | 0.512 | −0.043 | 0.893 | −0.012 | 5.70 | −0.220 | N/A | N/A | 0.045 |

Against the pre-stated cases: lambda_sharp=0.25 is **Case D (all metrics flat)** — every |Δ| ≤ 0.02 with CIs spanning 0; 0.50 and 0.75 are **Case D / mild B** — AQAScore flat, CLAP and PQ slightly lower (point estimates negative, CIs touching 0). No condition shows the Case A pattern (CLAP up, AQAScore stable, PQ stable) or Case C (AQAScore up, others stable). No post-hoc thresholds were used; the screening rule in `candidate_summary.json` (non-negative point estimates on all three metrics, at least one CI excluding 0 on the positive side, none on the negative side) is met by none of the interpolation points.

## 10. Category A — Discrete Single

Δ vs Dense (n = 6): CLAP −0.007 / −0.005 / **+0.017** [−0.005, +0.035] / +0.000 / −0.009 / −0.025; AQAScore −0.002 / +0.000 / **+0.018** [−0.014, +0.050] / +0.012 / −0.005 / −0.012; PQ +0.030 / −0.019 / −0.001 [−0.20, +0.19] / +0.013 / +0.028 / −0.009 (lambda_sharp 0.25 … 1.5). At lambda_sharp=0.75 all three A point estimates are non-negative, but every CI includes 0, the pattern is driven by A01/A05 (§14) against A03, and category A is a secondary mechanistic analysis: it does not qualify a candidate on its own.

## 11. Category B — Diffuse Single

CLAP −0.001 / −0.018 / −0.027 / −0.023 / −0.027 / −0.038 (CIs include 0); AQAScore +0.001 / +0.001 / −0.005* / −0.007* / −0.013* / −0.019* (small but CIs exclude 0 from 0.75); PQ −0.069 / −0.176* / −0.222* / −0.164 / −0.271* / −0.271* (CIs exclude 0 from 0.5). Diffuse textures lose production quality already at lambda_sharp=0.5 with no semantic gain.

## 12. Category C — Multi-event

CLAP −0.001 / −0.013 / −0.013 / −0.033* / −0.041* / −0.065*; AQAScore −0.002 / −0.022* / −0.015* / −0.023* / −0.035* / −0.029*; PQ −0.072 / −0.075 / −0.073 / −0.118 / −0.261* / −0.392*. Two-discrete-event prompts are the only category whose AQAScore drops with CIs excluding 0 inside the interpolation range (from 0.5), and PQ falls most at extrapolation.

## 13. Category D — Foreground / Background

| lambda_sharp | Full CLAP | Full AQAScore | FG Event-AQA | BG Event-AQA | FG phrase CLAP | BG phrase CLAP | PQ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.616 | 0.900 | 0.878 | 0.697 | 0.411 | 0.210 | 5.39 |
| 0.25 | 0.613 | 0.920 (+0.020 [+0.000, +0.045]) | 0.882 | 0.705 | 0.418 | 0.201 | 5.42 (+0.03) |
| 0.50 | 0.608 | 0.907 (+0.008) | 0.884 | 0.706 | 0.412 | 0.190 | 5.42 (+0.03) |
| 0.75 | 0.600 (−0.015*) | 0.917 (+0.017) | 0.893 | 0.731 | 0.414 | 0.182 | 5.38 (−0.02) |
| 1.00 | 0.589 (−0.027*) | 0.865 (−0.035) | 0.858 | 0.699 | 0.408 | 0.171 | 5.28 (−0.11) |
| 1.25 | 0.581 (−0.034*) | 0.884 (−0.016) | 0.860 | 0.690 | 0.411 | 0.147 | 5.21 (−0.18*) |
| 1.50 | 0.571 (−0.044*) | 0.910 (+0.011) | 0.890 | 0.725 | 0.408 | 0.132 | 5.19 (−0.21*) |

As in the previous experiments the two "background" proxies disagree: background-phrase CLAP declines monotonically while the background Event-AQAScore is flat to slightly higher (0.697 → 0.731 at 0.75, CI includes 0) and the foreground Event-AQAScore is flat. D's full AQAScore is +0.020 at 0.25 with a CI whose lower bound is +0.0001, and its PQ is +0.03 at 0.25/0.5 (CIs include 0); CLAP is down from 0.75. Per prompt, D01's background Event-AQA declines from lambda_sharp ≥ 0.75 (0.941 → 0.928 → 0.926 → 0.892 → 0.861) while D06's rises (0.642 → 0.879); event presence is not inferred from these values.

## 14. Previously Improving Prompts (descriptive only)

Seed-averaged trajectories over lambda_sharp 0 → 1.5 (CLAP | AQAScore | PQ):

- **A01** dog: CLAP 0.624 → 0.630 → 0.640 → 0.651 → 0.655 → 0.659 → 0.660 (monotone up); AQA ≈ 0.97 flat; PQ 6.6 flat.
- **A05** door slam: CLAP 0.587 → 0.608 → 0.626 → 0.638 → 0.652 → 0.657 → 0.652; AQA 0.902 → 0.92; PQ 6.60 → 6.77 → 6.83 → 6.97 → 7.05 → 7.08 → 7.09 (up).
- **B03** waves: CLAP 0.467 → 0.524 → 0.496 → 0.487 → 0.534 → 0.536 → 0.575 (non-monotone); AQA ≈ 0.96 flat; PQ 4.76 → 5.01.

Contrasting prompts that previously degraded:

- **A03** car horn: CLAP 0.589 → 0.534 → 0.492 → 0.560 → 0.501 → 0.487 → 0.421; AQA 0.917 → 0.879 → 0.852 → 0.869 → 0.834 → 0.735 → 0.642; PQ 7.15 → 6.67.
- **B05** stream: CLAP 0.451 → 0.431 → 0.379 → 0.359 → 0.394 → 0.410 → 0.378; AQA 0.938 → 0.910; PQ 4.86 → 4.62.
- **C03** speech + bell: CLAP 0.584 → 0.470; AQA 0.927 → 0.905; PQ 4.67 → 4.05.
- **C01** dog + horn, **D01** dog + rain: monotone CLAP/PQ declines (0.748 → 0.628 / 6.38 → 6.08; 0.673 → 0.610 / 5.62 → 5.40).

The gainers keep gaining along the direction and the losers keep losing; partial steps do not turn losers into gainers. These subsets were not used for selection.

## 15. Pareto Analysis

Objectives: maximize CLAP, AQAScore, PQ (overall means; `pareto_summary.json`, `sharp_mix_pareto.png`). Non-dominated: **Dense (0.00)** and **0.25**; dominated: 0.50 and 0.75 (by 0.25), 1.00, 1.25, 1.50. lambda_sharp=0.25 is non-dominated only because its AQAScore mean is +0.004 above Dense while its CLAP (−0.003) and PQ (−0.020) are below; this is a within-noise trade-off, not an advantage over Dense.

## 16. Comparison with Sharp / Entmax / PLADIS

Overall means (24 prompts), same prompts, seeds and initial latents in every row (`outputs/multimetric_eval/combined/cross_metric_summary.csv` for the earlier rows):

| Method | Control | CLAP | AQAScore | mean Event-AQA | PQ |
|---|---|---:|---:|---:|---:|
| Dense | — | 0.555 | 0.905 | 0.873 | 5.92 |
| Partial Sharp (development-set point of interest, **not a candidate**) | lambda_sharp=0.25, β=1.5 | 0.553 | 0.909 | 0.874 | 5.90 |
| Partial Sharp | lambda_sharp=0.75, β=1.5 | 0.546 | 0.908 | 0.878 | 5.84 |
| Sharp Softmax | β=1.5 (≡ lambda_sharp=1) | 0.535 | 0.892 | 0.868 | 5.83 |
| Entmax interpolation | lambda_entmax=0.5 | 0.545 | 0.906 | 0.875 | 5.83 |
| Pure Entmax | lambda_entmax=1 | 0.502 | 0.878 | 0.848 | 5.70 |
| PLADIS extrapolation | lambda_entmax=2 | 0.364 | 0.805 | 0.793 | 5.28 |
| Partial Sharp extrapolation | lambda_sharp=1.5, β=1.5 | 0.512 | 0.893 | 0.866 | 5.70 |

Signed extrapolation along the Softmax direction (lambda_sharp 1.25/1.5, negative mass 0.003/0.045 per row) continues the gradual decline of the interpolation range (CLAP −0.028/−0.043, PQ −0.17/−0.22) and does not reproduce the PLADIS cliff (CLAP −0.19, AQAScore −0.10, PQ −0.64), whose extrapolated map carries orders of magnitude more negative mass. The extrapolation effect here is therefore indistinguishable from continued concentration; the PLADIS failure is specific to extrapolating a hard-sparse map.

## 17. Development Candidate

**NO DEVELOPMENT CANDIDATE IDENTIFIED.** On all three primary metrics jointly, no lambda_sharp in {0.25, 0.50, 0.75} improves on Dense: 0.25 is equivalent to Dense within noise (ΔCLAP −0.003 [−0.011, +0.006], ΔAQA +0.004 [−0.005, +0.013], ΔPQ −0.020 [−0.063, +0.021]); 0.50 and 0.75 keep AQAScore flat (−0.003 / +0.004) while CLAP (−0.011 / −0.010) and PQ (−0.060 / −0.078) decline with CIs reaching only to ≈ 0. Category A at 0.75 and category D at 0.25 show mildly positive point estimates, but these are secondary analyses on 6 prompts each with CIs including 0 and, for D, a conflicting background-phrase CLAP; they do not constitute a joint advantage over Dense on the development set.

## 18. Limitations

- The 24 prompts are a reused development/ablation set that has now informed the taxonomy analysis, the β selection and this lambda_sharp sweep; nothing here is an unbiased final evaluation, and no generalization claim is made.
- Three seeds; one T2A backbone (Auffusion), one CFG scale; one β target (1.5), so the interpolation covers only one direction in attention space.
- Automatic metrics: CLAP is global; AQAScore is near its ceiling and backbone-dependent; Event-AQAScore is a diagnostic adaptation; AudioBox PQ is a predictor, not a human rating; human ratings remain blank.
- Extrapolation conditions produce signed weights with tiny negative mass; they are not probability distributions and their entropy is undefined.
- ELSA remains unavailable.

## 19. Required Next Step

Since no development-set candidate was identified: **stop uniform concentration-based method development** (dense→sharp interpolation included) for Auffusion. The only remaining open thread is prompt-specific behaviour (a minority of prompts gain monotonically along the sharp direction while others lose), which would require a selective or adaptive mechanism and a hold-out prompt set designed before any such method is evaluated; neither is started here.
