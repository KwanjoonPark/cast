# CAST Sharp Softmax Diagnostic Report

Date: 2026-10-23. Scripts: `experiments/run_sharp_softmax_v1.py` (generation; config `experiments/configs/sharp_softmax_v1.yaml`), `experiments/eval_clap_diagnostic.py --root outputs/sharp_softmax_v1 --cond-key beta` (CLAP), `experiments/make_entropy_csv.py`, `experiments/analyze_sharp_softmax_v1.py`, `experiments/make_human_eval_package_sharp.py`. Raw results under `outputs/sharp_softmax_v1/`. Diagnostic V1 Entmax results were reused unchanged.

## 1. Executive Summary

| Item | Status |
|---|---|
| Conditions | 288 = 24 prompts × 3 seeds × β ∈ {1.00, 1.25, 1.50, 2.00} (Diagnostic V1 prompts/seeds/settings reused exactly) |
| Newly generated / reused | **216 new** (β = 1.25, 1.5, 2.0; 29.3 min GPU, mean 9.6 s each) / **72 reused** (β = 1 ≡ Diagnostic V1 λ = 0) |
| β = 1 parity | **PASS** — per-module bitwise equal to the λ = 0 path (unit test, fp32 and fp16); full pipeline: all 72 (prompt, seed) β = 1 re-runs reproduce the Diagnostic V1 λ = 0 final latents **and** audio bitwise; the Entmax path is unchanged by the processor extension (λ = 1 and λ = 2 re-runs bitwise equal) |
| Entropy diagnostics | **PASS** — conditional-half attention entropy recorded for all 288 conditions (16 layers × 100 steps), plus counterfactual probes on the dense trajectory |
| CLAP | **PASS** — 288 / 288, validated `laion/clap-htsat-unfused` convention, deterministic (repeat diff 0.0); β = 1 scores equal the Diagnostic λ = 0 scores exactly |
| Invalid | 0 |
| Broad numerical pattern | Sharpening lowers attention entropy strongly (effective support 18.6 → 8.2 → 4.1 → 1.9 keys of 77) while CLAP changes only mildly: all-prompt mean 0.555 → 0.546 → 0.535 → 0.519 (β = 2: −0.036), versus Entmax λ = 1 (−0.053) and λ = 2 (−0.19). Category A is flat up to β = 1.5; B, C, D decline monotonically. In Category D the exploratory foreground-phrase similarity is flat at every β while the background-phrase similarity declines at every β (CIs excluding 0): the same asymmetry Entmax produced, obtained without any exact zeroing. |

## 2. Motivation

Diagnostic V1 showed that Entmax-1.5-based sparsification (λ = 1) and PLADIS extrapolation (λ = 2) lower global CLAP in every category, that λ = 0.5 is mild, that the conditional Entmax map keeps only ≈ 2.5 of 77 text keys, and that in discrete-foreground + diffuse-background prompts the background-phrase CLAP falls with λ while the foreground-phrase CLAP is stable up to λ = 1. Two mechanisms could produce this: **hard sparsity** (exactly zeroing most text keys removes weak-but-necessary context) or **attention concentration** per se (a sharper distribution over the same keys). Sharp Softmax, `Softmax(β · logits)`, raises concentration continuously with every weight strictly positive, so it separates the two.

## 3. Pre-Registered Hypotheses

- **H1 — Soft concentration.** Moderate Sharp Softmax may change selectivity while preserving global alignment better than Entmax-based sparsification.
- **H2 — Hard sparsity.** If Sharp Softmax maintains foreground/background correspondence better than Entmax at comparable concentration, exact zeroing is a likely contributor to the Entmax degradation.
- **H3 — Concentration.** If Sharp Softmax also degrades alignment and background preservation as entropy decreases, excessive concentration itself is the likely problem.
- **H4 — Acoustic-structure dependence.** Discrete/salient events may tolerate moderate sharpening better than diffuse and multi-event scenes.

## 4. Method

In `cast/attention/cast_processor.py` (extended, not forked: `attention_type="softmax"`, `beta`), for the 16 `attn2` modules:

```
logits = (Q Kᵀ) · d^(−1/2)            fp32 (autocast disabled)
W_β    = Softmax(β · logits)           fp32; β·logits in fp32; T = 1/β
out    = W_β.to(V.dtype) @ V
```

No λ, no Entmax, no mixing in this path. β = 1 is the dense Softmax of the validated λ = 0 path (multiplication by 1.0 is exact, hence bitwise identical). Q/K/V projections, output projection, CFG batch handling (whole `[uncond, cond]` batch), model dtype, scheduler, initial noise, and the deterministic math-SDPA self-attention backend are unchanged; `attn1` is never touched. Entropy statistics (conditional half, per call = mean over heads and query rows): `H = −Σ p log p`, `H_norm = H / log 77`, `effective_support = exp(H)`; probes evaluate `Softmax(b·logits)` for b ∈ {1.25, 1.5, 2} and `entmax15(logits)` (entropy and exact support) on the same logits during the β = 1 pass, without affecting outputs.

## 5. Dataset and Controls

| Control | Value / check |
|---|---|
| Prompt file | `experiments/prompts/cast_diagnostic_v1.json`, SHA-256 `c89ac1351381c16b7d85c84569543a615af52cdb5ba17bef812e350325968d38` = value in `CAST_DIAGNOSTIC_V1_REPORT.md` (asserted by the driver); 24 prompts, 6 per category, unmodified |
| Seeds | 42, 123, 2026; per-seed initial latents SHA-256 `ea92f6ab…`, `6e34569f…`, `5ea8d1c0…`, asserted equal to `outputs/diagnostic_v1/experiment_manifest.json`; same tensor passed as `latents=` to all four β for each (prompt, seed) (`same_initial_latent_per_prompt_seed = true`) |
| Generation settings | checkpoint `auffusion/auffusion`, 100 PNDM steps, CFG 7.5, no negative prompt, 256×1024, fp16 + autocast, `deterministic_self_attn = true`; asserted field-by-field equal to the Diagnostic V1 manifest |
| Processors | `CastCrossAttnProcessor(attention_type="softmax")` on the 16 `attn2`; `attn1` stock `AttnProcessor2_0`; restored after each run |
| Output layout | `outputs/sharp_softmax_v1/<CAT>/<ID>/seed_<s>/beta_<β>/{audio.wav, spectrogram.png, final_latents.pt, manifest.json}`; β = 1.00 directories are copies of the Diagnostic V1 λ = 0 files with `provenance.reused_from` and the entropy statistics of the bitwise-identical parity re-run |

## 6. Beta=1 Parity

| Level | Test | Result |
|---|---|---|
| Per module (random-init SD1.5 U-Net, 2 modules, fp32 and fp16) | `softmax β=1` vs `entmax_mix λ=0` output | **bitwise equal** (`torch.equal`); vs stock `AttnProcessor2_0` within the established floor; β = 2 differs on conditioned rows and is bitwise equal on the zero-conditioned (uncond) row |
| Full pipeline, all 72 (prompt, seed) units | β = 1 re-run vs Diagnostic V1 λ = 0 | **72 / 72 final-latent SHA-256 equal, 72 / 72 audio SHA-256 equal** (11.6 min) |
| Entmax path after the processor extension | D01 seed 42, λ = 1 and λ = 2 re-run vs Diagnostic V1 | bitwise equal (latents and audio) |
| CLAP | β = 1 scores vs Diagnostic λ = 0 scores | identical for all 72 |

Unit tests: 12 / 12 pass (`cast/tests`), including the two new Sharp-Softmax tests.

## 7. Attention Concentration

Conditional-half statistics, mean over 16 layers × 100 steps, then over prompts (seed-averaged). "Own trajectory" = measured in the run that produced the audio.

| β | H (nats) | H_norm = H/log 77 | effective support exp(H) | mean CLAP (24 prompts) | mean ‖z_T‖ |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 2.923 | 0.673 | 18.6 | 0.555 | 86.7 |
| 1.25 | 2.100 | 0.484 | 8.2 | 0.546 | 88.4 |
| 1.50 | 1.408 | 0.324 | 4.1 | 0.535 | 89.7 |
| 2.00 | 0.655 | 0.151 | 1.9 | 0.519 | 91.1 |

Counterfactual probes on the dense (β = 1) trajectory, same logits (`attention_entropy.csv`, `probe_*` columns): Softmax at b = 1.25 / 1.5 / 2.0 → H_norm 0.479 / 0.317 / 0.145, effective support 8.0 / 4.0 / 1.9 (within 3 % of the own-trajectory values above, i.e. the entropy response is set by β, not by trajectory drift); **entmax15 on the same logits → H_norm 0.078, effective support 1.40, exact support 2.51**. Entmax exact support on its own trajectories (Diagnostic V1, W_sparse): 2.54 / 2.57 / 2.64 at λ = 0.5 / 1 / 2. So on the concentration axis the ordering is dense (18.6) ≫ β 1.25 (8.2) ≫ β 1.5 (4.1) > β 2 (1.9) > entmax15 (1.4 effective / 2.5 exact); no β condition is as concentrated as pure Entmax, and λ = 0.5 (a 50/50 mix) has no single concentration value. Entropy of the signed λ = 2 map is undefined and was not computed.

By category, H_norm at β = 1 / 1.25 / 1.5 / 2: A 0.681 / 0.486 / 0.321 / 0.143; B 0.660 / 0.473 / 0.317 / 0.150; C 0.676 / 0.490 / 0.333 / 0.159; D 0.674 / 0.486 / 0.326 / 0.151 — the concentration response is essentially category-independent (figures `sharp_entropy_vs_beta.png`, `sharp_effective_support_vs_beta.png`).

## 8. CLAP Results

Prompt-level (seed-averaged) CLAP cosine; category means; Δ vs β = 1 with 95 % prompt-level bootstrap CIs (10 000 resamples, seed 20261022, n = 6). Full numbers: `category_summary.csv`, `hypothesis_summary.json`; figures `sharp_clap_by_category.png`, `sharp_delta_clap.png`.

| Category | β=1 | β=1.25 | β=1.5 | β=2 | Δ1.25 [CI] | Δ1.5 [CI] | Δ2 [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| A DISCRETE_SINGLE | 0.504 | 0.506 | 0.504 | 0.484 | +0.002 [−0.037, +0.032] | +0.000 [−0.040, +0.036] | −0.020 [−0.087, +0.033] |
| B DIFFUSE_SINGLE | 0.532 | 0.519 | 0.508 | 0.501 | −0.013 [−0.042, +0.018] | −0.023 [−0.057, +0.017] | −0.031 [−0.070, +0.024] |
| C DISCRETE_DISCRETE | 0.570 | 0.559 | 0.538 | 0.513 | −0.011 [−0.027, +0.005] | −0.033 [−0.055, −0.010] | −0.057 [−0.092, −0.020] |
| D DISCRETE_DIFFUSE | 0.616 | 0.600 | 0.589 | 0.580 | −0.015 [−0.024, −0.006] | −0.027 [−0.032, −0.022] | −0.036 [−0.060, −0.002] |

Prompt-level values (CLAP at β = 1 / 1.25 / 1.5 / 2 | Δ vs β = 1 | effective support):

| ID | β=1 | β=1.25 | β=1.5 | β=2 | Δ1.25 | Δ1.5 | Δ2 | eff. support (β=1 → 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A01 | 0.624 | 0.643 | 0.655 | 0.660 | +0.019 | +0.030 | +0.035 | 19.3 → 1.8 |
| A02 | 0.275 | 0.301 | 0.273 | 0.225 | +0.026 | −0.002 | −0.050 | 18.8 → 2.1 |
| A03 | 0.589 | 0.505 | 0.501 | 0.425 | −0.084 | −0.088 | −0.165 | 20.2 → 1.8 |
| A04 | 0.606 | 0.593 | 0.600 | 0.586 | −0.014 | −0.006 | −0.020 | 19.0 → 1.9 |
| A05 | 0.587 | 0.639 | 0.652 | 0.656 | +0.052 | +0.065 | +0.069 | 20.1 → 1.8 |
| A06 | 0.342 | 0.354 | 0.344 | 0.354 | +0.012 | +0.001 | +0.011 | 18.2 → 1.9 |
| B01 | 0.482 | 0.507 | 0.478 | 0.436 | +0.025 | −0.004 | −0.046 | 16.5 → 2.0 |
| B02 | 0.623 | 0.609 | 0.607 | 0.592 | −0.014 | −0.016 | −0.030 | 19.6 → 1.8 |
| B03 | 0.467 | 0.510 | 0.534 | 0.560 | +0.043 | +0.067 | +0.093 | 17.8 → 1.9 |
| B04 | 0.571 | 0.547 | 0.525 | 0.528 | −0.024 | −0.046 | −0.043 | 18.0 → 2.0 |
| B05 | 0.451 | 0.376 | 0.394 | 0.395 | −0.075 | −0.057 | −0.056 | 16.7 → 1.7 |
| B06 | 0.595 | 0.564 | 0.514 | 0.494 | −0.031 | −0.082 | −0.102 | 17.0 → 2.2 |
| C01 | 0.748 | 0.732 | 0.713 | 0.646 | −0.016 | −0.035 | −0.102 | 19.6 → 1.8 |
| C02 | 0.454 | 0.471 | 0.376 | 0.414 | +0.017 | −0.078 | −0.040 | 19.4 → 2.0 |
| C03 | 0.584 | 0.538 | 0.537 | 0.470 | −0.047 | −0.047 | −0.114 | 19.7 → 1.9 |
| C04 | 0.430 | 0.411 | 0.413 | 0.425 | −0.019 | −0.017 | −0.005 | 18.4 → 2.2 |
| C05 | 0.571 | 0.557 | 0.538 | 0.490 | −0.014 | −0.033 | −0.080 | 17.7 → 2.1 |
| C06 | 0.634 | 0.644 | 0.648 | 0.635 | +0.010 | +0.014 | +0.001 | 18.6 → 2.0 |
| D01 | 0.673 | 0.653 | 0.636 | 0.613 | −0.019 | −0.037 | −0.059 | 18.9 → 1.8 |
| D02 | 0.608 | 0.613 | 0.590 | 0.578 | +0.005 | −0.018 | −0.030 | 19.2 → 1.8 |
| D03 | 0.415 | 0.399 | 0.394 | 0.461 | −0.016 | −0.021 | +0.046 | 18.5 → 2.1 |
| D04 | 0.648 | 0.619 | 0.618 | 0.589 | −0.029 | −0.030 | −0.059 | 18.8 → 1.9 |
| D05 | 0.687 | 0.663 | 0.661 | 0.619 | −0.025 | −0.027 | −0.069 | 19.3 → 1.9 |
| D06 | 0.662 | 0.654 | 0.633 | 0.616 | −0.008 | −0.029 | −0.046 | 17.6 → 2.0 |

Seed-level rows for all 288 conditions: `clap_scores.csv`; entropy per condition: `attention_entropy.csv`.

## 9. Category A — Discrete Single

Mean Δ: +0.002 / 0.000 / −0.020 (all CIs include 0); medians +0.015 / −0.000 / −0.004. Three prompts rise monotonically with β (A01 +0.035, A05 +0.069, A06 ≈ 0 at β = 2), A04 is flat, A02 falls only at β = 2, A03 falls at every β (−0.084 → −0.165). Under Entmax the same category gave +0.012 / −0.029 / −0.169 at λ = 0.5 / 1 / 2. Discrete single events are, on average, tolerant of soft sharpening up to a 4-key effective support; the loss at β = 2 (1.9 effective keys) is smaller than Entmax λ = 1's (−0.030) and far smaller than λ = 2's.

## 10. Category B — Diffuse Single

Mean Δ: −0.013 / −0.023 / −0.031, monotone; all CIs include 0 (prompt std 0.04–0.07). B03 (waves) rises with β (+0.093 at β = 2) while B05 (stream) and B06 (crowd) fall from β = 1.25 (−0.075, −0.031) — the same two prompts that fell first under Entmax. Entmax: −0.019 / −0.050 / −0.229. The B − A contrast is negative at every β (−0.015 / −0.023 / −0.011) with CIs including 0.

## 11. Category C — Discrete + Discrete

Mean Δ: −0.011 / −0.033 / −0.057; CIs exclude 0 at β = 1.5 and 2. Five of six prompts are negative at β = 2 (C06 ≈ 0). Entmax: −0.020 / −0.077 / −0.202. C − A: −0.013 / −0.033 / −0.037 (CIs include 0). Two-discrete-event prompts lose more than single discrete events under both families, and C is the most sharpening-sensitive category on full-prompt CLAP.

## 12. Category D — Foreground + Background

Full-prompt Δ: −0.015 / −0.027 / −0.036; CIs exclude 0 at every β (the tightest category, prompt std 0.01–0.04). D − C: −0.004 / +0.006 / +0.021 (CIs include 0): D does not degrade more than C on the global metric, as in Diagnostic V1.

Exploratory event-phrase CLAP (not an event detector; `category_d_event_phrase_sharp.png`):

| Phrase | β=1 | β=1.25 | β=1.5 | β=2 | Δ1.25 [CI] | Δ1.5 [CI] | Δ2 [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| foreground (Sharp Softmax) | 0.411 | 0.408 | 0.408 | 0.409 | −0.003 [−0.017, +0.010] | −0.003 [−0.012, +0.007] | −0.002 [−0.031, +0.034] |
| background (Sharp Softmax) | 0.210 | 0.190 | 0.171 | 0.132 | −0.020 [−0.030, −0.009] | −0.039 [−0.060, −0.023] | −0.079 [−0.111, −0.045] |
| foreground (Entmax, λ=0.5/1/2) | 0.411 | 0.412 | 0.404 | 0.302 | +0.001 | −0.006 | −0.108 |
| background (Entmax, λ=0.5/1/2) | 0.210 | 0.171 | 0.120 | 0.080 | −0.039 | −0.090 | −0.130 |

The foreground-phrase similarity is flat at all β (|Δ| ≤ 0.003); the background-phrase similarity declines monotonically with β and every CI excludes 0. This is the same asymmetry observed under Entmax, with a smaller magnitude at the same control-parameter index and without any attention weight being exactly zero.

## 13. Sharp Softmax vs Entmax

All 24 prompts, prompt-level means; concentration statistics from §7; Entmax rows from Diagnostic V1 (`comparison_with_entmax.csv`, `sharp_vs_entmax_clap.png`). β and λ are different control parameters and are not concentration-matched.

| Family | Condition | Concentration (cond. half) | Mean CLAP | Δ vs dense | ΔA | ΔB | ΔC | ΔD | ΔD-fg | ΔD-bg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | β=1 ≡ λ=0 | eff. 18.6, H_norm 0.67 | 0.555 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sharp Softmax | β=1.25 | eff. 8.2 | 0.546 | −0.009 | +0.002 | −0.013 | −0.011 | −0.015 | −0.003 | −0.020 |
| Sharp Softmax | β=1.5 | eff. 4.1 | 0.535 | −0.021 | +0.000 | −0.023 | −0.033 | −0.027 | −0.003 | −0.039 |
| Sharp Softmax | β=2 | eff. 1.9, H_norm 0.15 | 0.519 | −0.036 | −0.020 | −0.031 | −0.057 | −0.036 | −0.002 | −0.079 |
| Entmax interpolation | λ=0.5 | mix of dense (eff. 18.6) and W_sparse (exact 2.5); no single value | 0.545 | −0.011 | +0.012 | −0.019 | −0.020 | −0.016 | +0.001 | −0.039 |
| Pure Entmax | λ=1 | exact 2.6; eff. ≈ 1.4 on dense logits, H_norm 0.08 | 0.502 | −0.053 | −0.030 | −0.050 | −0.077 | −0.055 | −0.006 | −0.090 |
| PLADIS extrapolation | λ=2 | signed weights (entropy undefined); W_sparse exact 2.6 | 0.364 | −0.191 | −0.169 | −0.229 | −0.202 | −0.166 | −0.108 | −0.130 |

Reading along the concentration axis: dense → β 1.25 → β 1.5 → β 2 → Entmax λ = 1 is a monotone sequence of increasing concentration (18.6 → 8.2 → 4.1 → 1.9 → ≈1.4 effective keys) and of monotonically decreasing mean CLAP (0.555 → 0.546 → 0.535 → 0.519 → 0.502), with the same category ordering (A least affected, C most on the global score) and the same D foreground/background asymmetry at every point. Entmax λ = 1 sits at the concentrated end of this sequence and its degradation (−0.053) continues the trend from β = 2 (−0.036) rather than breaking it. λ = 2 (−0.191) lies far outside this trend: it is not a concentration point but an extrapolation with negative weights, and its harm is 3–4× the most concentrated non-negative condition. λ = 0.5 behaves like a mild condition comparable to β 1.25–1.5 on the global score (−0.011) but with a larger background-phrase drop (−0.039 vs −0.020 at β 1.25).

## 14. Hard Sparsity vs Concentration

Assessment: **evidence favors concentration** as the main driver of the λ ≤ 1 degradation, with insufficient evidence on whether exact zeroing adds harm beyond concentration, and a separate extrapolation effect at λ = 2.

- The D foreground/background asymmetry (foreground flat, background falling with CIs excluding 0) is fully reproduced by Sharp Softmax, where no weight is zero. Hard zeroing is therefore **not necessary** for the phenomenon.
- Global CLAP, category ordering, and background-phrase loss all decrease monotonically as measured entropy decreases across β, and pure Entmax (the most concentrated non-negative map) continues that monotone trend. A concentration-only account predicts exactly this.
- Whether Entmax λ = 1 is worse than a hypothetical Softmax of equal concentration cannot be decided here: λ = 1 is more concentrated (≈1.4 effective keys) than β = 2 (1.9), and no β was run beyond 2 (pre-registered set; no post-hoc β was added). Its slightly larger loss (−0.053 vs −0.036; D background −0.090 vs −0.079) is consistent with either continued concentration or an additional zeroing effect.
- λ = 2's loss (−0.191) is incompatible with the concentration trend and is attributable to extrapolation with negative weights, not to sparsity or concentration per se.

In the task's terms: outcome **B** (Sharp Softmax degrades as entropy decreases) with an element of **D** (no β > 1 improves mean CLAP in any category; category A is merely flat up to β = 1.5), and no support for **A**; **C** (an intermediate-concentration regime that improves some categories) is not supported at the category level (A: +0.002 / 0.000 with CIs spanning ±0.035), although individual prompts (A01, A05, B03) rise with β.

## 15. Human Evaluation Materials

`outputs/sharp_softmax_v1/human_eval/`: `HUMAN_EVAL_PROTOCOL.md` (Diagnostic V1 protocol + package note), `HUMAN_EVENT_COVERAGE_TEMPLATE.csv` (blank `present_yes_no`, with `event_role` foreground/background for D), `HUMAN_QUALITY_TEMPLATE.csv` (blank 1–5), `blind_key.json` (72 units × 4 labels), `blind_audio/` (288 files `<ID>_s<seed>_<A–D>.wav` = Sharp Softmax β 1 / 1.5 / 2 + Entmax λ = 1 for the same prompt and seed, labels randomized per unit with seed 20261023, no method in filenames). Ratings are blank.

## 16. Limitations

- One T2A backbone (Auffusion), one CFG scale (7.5), one α for the Entmax comparison.
- 24 hand-designed diagnostic prompts in a hypothesis-driven taxonomy; prompt-level std (0.01–0.08) is comparable to the β effects, and single prompts (A03, B05, B06, D03) move category means.
- Three seeds; seed-level variation at β = 1 is 0.01–0.08.
- CLAP is a global embedding similarity; it does not measure fidelity, event completeness, or temporal structure.
- Event-phrase CLAP is exploratory and is not an event detector.
- No completed human quality or event-coverage ratings for either experiment.
- β values are not concentration-matched to Entmax; the most concentrated Softmax condition (β = 2, 1.9 effective keys) is less concentrated than pure Entmax (≈1.4 effective / 2.5 exact); conclusions about zeroing beyond concentration are therefore limited.
- Entropy statistics are conditional-half means over heads, query rows, layers and steps; per-layer values are stored but not analysed here.
- ELSA remains blocked (not retried).

## 17. Hypothesis Status

- **H1 (soft concentration preserves global alignment better than Entmax): Supported.** At every β the category Δ are smaller in magnitude than Entmax λ = 1 and λ = 2 (all-prompt −0.009 / −0.021 / −0.036 vs −0.053 / −0.191); Category A is flat through β = 1.5 (+0.002, 0.000; CIs include 0). No β improves mean CLAP.
- **H2 (hard sparsity is a likely contributor): Not supported as a necessary cause; insufficient evidence as an additional cause.** Sharp Softmax reproduces the foreground-stable / background-declining asymmetry and the category ordering without zeroing; the residual gap between β = 2 and λ = 1 is confounded with their concentration difference.
- **H3 (concentration): Supported.** Global CLAP (B, C, D), the D background-phrase similarity (CIs excluding 0 at every β) and the overall mean decline monotonically as measured entropy decreases, and Entmax λ = 1 continues the trend at higher concentration.
- **H4 (acoustic-structure dependence): Partially supported.** A is flat up to β = 1.5 while B, C, D decline, and C declines most; the between-category contrasts (B − A, C − A) have CIs including 0, and D does not degrade more than C on the global metric.

## 18. Recommendation

Based strictly on these results, future CAST development should **not** pursue uniform concentration-based cross-attention modification as a route to better global text-audio alignment in Auffusion (**option D**): neither Entmax mixing nor Softmax sharpening raised mean CLAP in any category, and both lower the background-phrase correspondence in proportion to concentration. Of the remaining options, **A (weak Sharp Softmax)** offers only neutrality for discrete single events (not improvement); **B (weaker Entmax / α → 1)** is predicted by the concentration account to behave like a mild β and the λ = 0.5 data (+0.012 on A, −0.016…−0.020 elsewhere) match that; **C (adaptive concentration)** is the only direction the data leave open, because the harm is concentrated on background/secondary events, but it would need a mechanism that leaves background keys unsharpened, and it should be pursued only if the pending human event-coverage and quality ratings reveal a benefit that CLAP does not capture. Nothing was implemented or tuned beyond the pre-registered conditions.
