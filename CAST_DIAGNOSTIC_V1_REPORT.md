# CAST Diagnostic V1 Report

Date: 2026-10-22. Scripts: `experiments/run_diagnostic_v1.py` (generation), `experiments/eval_clap_diagnostic.py` (CLAP), `experiments/analyze_diagnostic_v1.py` (analysis), `experiments/make_human_eval_package.py` (blind human-eval package); configs `experiments/configs/diagnostic_v1.yaml`, dataset `experiments/prompts/cast_diagnostic_v1.json`. Raw results under `outputs/diagnostic_v1/`. The CAST attention method (`cast/attention/*`, `cast/pipeline.py`) was not modified; the only code additions are the scripts above and a shared CLAP scorer module `cast/eval/clap.py` (same convention as the validated evaluation).

## 1. Executive Summary

| Item | Status |
|---|---|
| Prompts | 24 (4 categories × 6), frozen before generation, SHA-256 `c89ac1351381c16b7d85c84569543a615af52cdb5ba17bef812e350325968d38` |
| λ conditions | 4: 0.0, 0.5, 1.0, 2.0 (α = 1.5 fixed) |
| Seeds | 3: 42, 123, 2026 |
| Expected / completed generations | 288 / **288**, all finite, 0 failures, 44 min GPU (mean 9.2 s per sample, peak 3.11 GB) |
| ELSA (primary metric) | **BLOCKED** — official implementation requires the manually gated `facebook/sam-audio-large-tv` separation model (no valid Hugging Face credential on this machine; anonymous and cached-token requests both return HTTP 401) and an `OPENAI_API_KEY` for its GPT-5.2 event parser (not set). No substitute metric was used. Details in §5 and §7. |
| CLAP (secondary metric) | **PASS** — 288 / 288 scored with the validated `laion/clap-htsat-unfused` cosine convention, deterministic (repeat diff 0.0) |
| Human evaluation | Blind package prepared (72 units, 288 relabelled files, 432 event rows), ratings blank |
| Measured pattern (CLAP only) | Mean CLAP decreases with λ in every category; λ = 0.5 is within ±0.02 of vanilla in all categories (slightly positive only for A); λ = 2 loses 0.17–0.23. Interpretation matrix: **Pattern 4** by the pre-declared rule (no category improves at all λ; A improves only at λ = 0.5). H1 weakly supported, H2 supported, H3 supported, H4 not supported on the primary comparison (details §10–§13, §17). No conclusion beyond these measurements is drawn. |

## 2. Pre-Registered Hypotheses

- **H1** — Discrete / salient acoustic events may tolerate or benefit from sparse cross-attention. Primary: Category A, λ = 0.5 vs 0; secondary λ = 1, 2 vs 0.
- **H2** — Diffuse / texture-like events may be harmed by strong sparse cross-attention. Category B λ = 0.5 / 1 / 2 vs 0, and Δ_B(λ) − Δ_A(λ).
- **H3** — Multi-event generation may be more vulnerable to sparsification than single-event generation. Category C vs A at λ = 0.5 / 1 / 2.
- **H4** — Mixtures of a salient foreground and a diffuse background may be especially vulnerable to background suppression. Category D vs C, particularly at λ = 1 / 2; if official per-event ELSA scores existed, foreground vs background.

Decision rules exactly as specified in the task brief (§18 there) are implemented in `experiments/analyze_diagnostic_v1.py` and applied to the primary metric where available; because ELSA is blocked they were applied to CLAP, which must be read as the *secondary* metric.

## 3. Prompt Dataset

`experiments/prompts/cast_diagnostic_v1.json`, SHA-256 `c89ac135…968d38` (recorded in `outputs/diagnostic_v1/experiment_manifest.json`). 24 prompts, 6 per category, verbatim from the brief; category D entries carry `foreground_event` / `background_event` annotations; every prompt carries an `events` inventory (used only for the human-eval template and for exploratory CLAP phrase similarities).

| Category | IDs | Definition |
|---|---|---|
| A DISCRETE_SINGLE | A01–A06 | one temporally local, salient event (dog bark, bottle shatter, car horn, telephone, door slam, pan crash) |
| B DIFFUSE_SINGLE | B01–B06 | one continuous texture (rain, wind, waves, fire, stream, crowd murmur) |
| C DISCRETE_DISCRETE | C01–C06 | two salient events (e.g. dog + horn, phone + knocking) |
| D DISCRETE_DIFFUSE | D01–D06 | salient foreground + diffuse background (e.g. dog + rain, speech + traffic) |

The file was not modified after generation; no prompt was removed or replaced.

## 4. Experimental Setup

| Setting | Value |
|---|---|
| Checkpoint | `auffusion/auffusion` (HF snapshot 211aeb85) |
| α / λ | 1.5 / {0.0, 0.5, 1.0, 2.0}; `W = W_dense + λ(W_sparse − W_dense)`, no clamping |
| Seeds | 42, 123, 2026 → three distinct initial latents (SHA-256 `ea92f6ab…`, `6e34569f…`, `5ea8d1c0…`) |
| Steps / CFG / negative prompt | 100 PNDM / 7.5 / None |
| Size / dtype / autocast | 256×1024 / fp16 / on |
| `deterministic_self_attn` | true for all 288 runs (math SDPA on the untouched `attn1`) |
| Processor | `CastCrossAttnProcessor` on the 16 `attn2` modules; `attn1` stock `AttnProcessor2_0`; restored after each run |
| Initial-noise control | per (prompt, seed) one latent from `sample_initial_latents(seed)`, passed as `latents=` to all four λ arms |
| Output layout | `outputs/diagnostic_v1/<CAT>/<ID>/seed_<s>/lambda_<λ>/{audio.wav, spectrogram.png, final_latents.pt, manifest.json}` |

## 5. Evaluation Metrics

- **ELSA (primary) — BLOCKED.** Official implementation: `github.com/kento2247/ELSA` (commit `d2eb34a8`, cloned read-only into `third_party/ELSA` for inspection; not executed). Per paper (arXiv 2606.17404) and `src/model.py`, the single reported quantity is the **"ELSA score"** = adaptive fusion `0.4^M · ŷ_c + (1 − 0.4^M) · ŷ_f` where ŷ_c is the normalized global cosine `(cos + 1)/2` between Human-CLAP text and audio embeddings, ŷ_f is the F1 of event-level precision/recall from greedy matching (`max − mean` of pairwise cosines) between GPT-5.2-parsed text events and SAM-Audio-separated audio segments, and M is the number of parsed events. The implementation does **not** expose per-event scores (they are averaged inside precision/recall), so no foreground/background ELSA exists to report and none was invented. Required components: `sarulab-speech/human-clap-wsce-mae` + `laion/clap-htsat-fused` processor (public), `facebook/sam-audio-large-tv` (**gated: manual**; HTTP 401 anonymously and with the machine's cached token, which the Hub reports as an invalid OAuth signature), OpenAI API (`OPENAI_API_KEY` not set; the repo also offers a local `Qwen/Qwen3-4B-Instruct-2507` parser), Python 3.11.1 + torch 2.7 cu118 (`uv sync`; would need a separate `cast-elsa` env). The `cast` environment was not touched.
- **CLAP (secondary, global).** Identical convention to `CAST_CLAP_EVALUATION_REPORT.md`: `laion/clap-htsat-unfused` (snapshot 8fa0f1c6), fp32, cosine of L2-normalized `get_audio_features` / `get_text_features`, 16 kHz → 48 kHz torchaudio sinc resampling (480 000 samples, no truncation), no clamping. Repeatability 0.0.
- **Exploratory event-phrase CLAP** (not a metric of event presence): cosine between the same audio embedding and each event phrase from the dataset (`event1_clap`, `event2_clap`, and for D `fg_event_clap`, `bg_event_clap`). Reported only in §13 as exploratory.
- **Human Event Coverage / Human Quality** — prepared, pending (§15).

## 6. Validation

Pre-generation validation (A01 × seed 42 × four λ, then checks), all PASS (`experiment_manifest.json → pre_generation_validation`): 24 prompts, 6 per category, dataset SHA recorded, α = 1.5, λ = [0, 0.5, 1, 2], seeds = [42, 123, 2026], `deterministic_self_attn` true, 16 `attn2` = `CastCrossAttnProcessor`, `attn1` = `AttnProcessor2_0`, processors restored, same initial latent across the four λ arms, settings equal to the validated milestone, no NaN/Inf. The A01/seed-42 arms reproduce the earlier λ-sweep values exactly (‖z_T‖ 90.83 / 94.63 / 100.14 / 114.91).

Post-generation (`generation_summary.json`): 288 / 288 completed and finite; `same_initial_latent_per_prompt_seed = true` for all 72 (prompt, seed) units; three seed latents distinct; 0 failures.

## 7. ELSA Results

**ELSA STATUS: BLOCKED.** Exact blocker: (1) `facebook/sam-audio-large-tv` is a manually gated Hugging Face repository; this machine has no valid Hugging Face credential (the cached token fails signature verification) and the auth files were deliberately left untouched; (2) `OPENAI_API_KEY` is absent. Both are credential/access issues external to the code; the official repo and its dependency list are otherwise installable (uv, Python 3.11.1, torch 2.7 cu118, 12–20 GB VRAM). No `elsa_scores.csv` was produced and no substitute fine-grained metric was computed. The required ELSA figures (`diagnostic_elsa_by_category.png`, `diagnostic_delta_elsa.png`, `foreground_background_elsa.png`) therefore do not exist. `analyze_diagnostic_v1.py` will include ELSA automatically once `outputs/diagnostic_v1/elsa_scores.csv` (columns incl. `prompt_id, category, seed, lambda, elsa_score`) exists.

## 8. CLAP Results

Prompt-level (seed-averaged) CLAP cosine; category means with 95 % prompt-level bootstrap CIs (10 000 resamples, seed 20261022, n = 6 prompts). Full numbers: `category_summary.csv`, `hypothesis_summary.json`; figures `diagnostic_clap_by_category.png`, `diagnostic_delta_clap.png`.

| Category | λ=0 | λ=0.5 | λ=1 | λ=2 | Δ0.5 | Δ1 | Δ2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A DISCRETE_SINGLE | 0.504 | 0.516 | 0.474 | 0.335 | **+0.012** | −0.029 | −0.169 |
| B DIFFUSE_SINGLE | 0.532 | 0.513 | 0.481 | 0.303 | −0.019 | −0.050 | −0.229 |
| C DISCRETE_DISCRETE | 0.570 | 0.550 | 0.493 | 0.368 | −0.020 | −0.077 | −0.202 |
| D DISCRETE_DIFFUSE | 0.616 | 0.600 | 0.560 | 0.449 | −0.016 | −0.055 | −0.166 |

Per-prompt CLAP (seed-averaged) and paired deltas:

| ID | λ=0 | λ=0.5 | λ=1 | λ=2 | Δ0.5 | Δ1 | Δ2 | seed-std at λ=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A01 | 0.624 | 0.657 | 0.661 | 0.588 | +0.032 | +0.037 | −0.036 | 0.025 |
| A02 | 0.275 | 0.278 | 0.217 | 0.144 | +0.003 | −0.058 | −0.131 | 0.030 |
| A03 | 0.589 | 0.535 | 0.398 | 0.254 | −0.054 | −0.191 | −0.335 | 0.070 |
| A04 | 0.606 | 0.611 | 0.570 | 0.361 | +0.004 | −0.036 | −0.246 | 0.046 |
| A05 | 0.587 | 0.648 | 0.657 | 0.439 | +0.062 | +0.070 | −0.147 | 0.042 |
| A06 | 0.342 | 0.366 | 0.344 | 0.221 | +0.024 | +0.002 | −0.121 | 0.022 |
| B01 | 0.482 | 0.508 | 0.446 | 0.238 | +0.026 | −0.036 | −0.244 | 0.028 |
| B02 | 0.623 | 0.619 | 0.576 | 0.467 | −0.004 | −0.046 | −0.156 | 0.014 |
| B03 | 0.467 | 0.515 | 0.529 | 0.396 | +0.048 | +0.063 | −0.071 | 0.060 |
| B04 | 0.571 | 0.540 | 0.521 | 0.313 | −0.031 | −0.050 | −0.258 | 0.042 |
| B05 | 0.451 | 0.361 | 0.340 | 0.124 | −0.090 | −0.111 | −0.327 | 0.045 |
| B06 | 0.595 | 0.532 | 0.475 | 0.280 | −0.063 | −0.121 | −0.315 | 0.036 |
| C01 | 0.748 | 0.712 | 0.606 | 0.468 | −0.036 | −0.142 | −0.280 | 0.027 |
| C02 | 0.454 | 0.449 | 0.442 | 0.277 | −0.005 | −0.012 | −0.178 | 0.064 |
| C03 | 0.584 | 0.528 | 0.394 | 0.233 | −0.056 | −0.190 | −0.351 | 0.022 |
| C04 | 0.430 | 0.409 | 0.397 | 0.329 | −0.021 | −0.033 | −0.101 | 0.012 |
| C05 | 0.571 | 0.553 | 0.491 | 0.404 | −0.018 | −0.080 | −0.167 | 0.008 |
| C06 | 0.634 | 0.650 | 0.626 | 0.498 | +0.017 | −0.007 | −0.135 | 0.022 |
| D01 | 0.673 | 0.638 | 0.609 | 0.520 | −0.034 | −0.064 | −0.153 | 0.022 |
| D02 | 0.608 | 0.596 | 0.549 | 0.475 | −0.012 | −0.059 | −0.133 | 0.034 |
| D03 | 0.415 | 0.426 | 0.465 | 0.387 | +0.011 | +0.050 | −0.028 | 0.055 |
| D04 | 0.648 | 0.639 | 0.532 | 0.352 | −0.009 | −0.116 | −0.296 | 0.084 |
| D05 | 0.687 | 0.648 | 0.603 | 0.476 | −0.040 | −0.084 | −0.212 | 0.015 |
| D06 | 0.662 | 0.651 | 0.604 | 0.486 | −0.011 | −0.058 | −0.176 | 0.031 |

Seed-level values for all 288 samples: `clap_scores.csv`. Across all 72 samples per λ the mean is 0.555 / 0.545 / 0.502 / 0.364 (λ = 0 / 0.5 / 1 / 2); mean final-latent norm 86.7 / 89.0 / 91.6 / 98.1; mean conditional entmax support 2.54 / 2.57 / 2.64 keys of 77 (trajectory statistic only).

## 9. Paired Delta Analysis

Category-level prompt-averaged ΔCLAP (mean of 6 prompt deltas; std across prompts; 95 % prompt-level bootstrap CI). ELSA columns are blocked.

| Category | λ | Mean ΔELSA | 95 % CI | Mean ΔCLAP | Std | Median | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.5 | BLOCKED | — | +0.012 | 0.039 | +0.014 | [−0.018, +0.039] |
| A | 1.0 | BLOCKED | — | −0.029 | 0.092 | −0.017 | [−0.102, +0.030] |
| A | 2.0 | BLOCKED | — | −0.169 | 0.105 | −0.139 | [−0.253, −0.099] |
| B | 0.5 | BLOCKED | — | −0.019 | 0.053 | −0.018 | [−0.057, +0.019] |
| B | 1.0 | BLOCKED | — | −0.050 | 0.066 | −0.048 | [−0.093, −0.001] |
| B | 2.0 | BLOCKED | — | −0.229 | 0.099 | −0.251 | [−0.296, −0.155] |
| C | 0.5 | BLOCKED | — | −0.020 | 0.025 | −0.020 | [−0.038, −0.002] |
| C | 1.0 | BLOCKED | — | −0.077 | 0.075 | −0.056 | [−0.136, −0.025] |
| C | 2.0 | BLOCKED | — | −0.202 | 0.095 | −0.172 | [−0.273, −0.137] |
| D | 0.5 | BLOCKED | — | −0.016 | 0.019 | −0.011 | [−0.029, −0.003] |
| D | 1.0 | BLOCKED | — | −0.055 | 0.056 | −0.061 | [−0.090, −0.009] |
| D | 2.0 | BLOCKED | — | −0.166 | 0.089 | −0.164 | [−0.229, −0.101] |

Between-category contrasts of mean ΔCLAP (difference of category means; bootstrap over prompts within each category):

| Contrast | λ=0.5 | λ=1 | λ=2 |
|---|---:|---:|---:|
| B − A | −0.031 [−0.079, +0.018] | −0.021 [−0.097, +0.064] | −0.059 [−0.159, +0.046] |
| C − A | −0.032 [−0.065, +0.003] | −0.048 [−0.132, +0.038] | −0.033 [−0.134, +0.070] |
| D − C | +0.004 [−0.018, +0.027] | +0.022 [−0.043, +0.093] | +0.036 [−0.057, +0.134] |

All contrast CIs include 0.

## 10. H1 — Discrete Single

Primary comparison (A, λ = 0.5 vs 0): mean ΔCLAP **+0.012**, CI [−0.018, +0.039] (includes 0); 5 of 6 prompts positive (A01 +0.032, A02 +0.003, A04 +0.004, A05 +0.062, A06 +0.024), A03 −0.054. Secondary: λ = 1 −0.029 [−0.102, +0.030] (A01, A05, A06 ≥ 0; A02, A03, A04 negative); λ = 2 −0.169 [−0.253, −0.099] (all six negative). Evidence: weak interpolation is neutral-to-slightly-positive on the global metric for this category, pure sparse is mixed, extrapolation is uniformly negative. ELSA evidence: none (blocked).

## 11. H2 — Diffuse Single

B: Δ = −0.019 / −0.050 / −0.229 at λ = 0.5 / 1 / 2 (monotonically more negative; CI excludes 0 at λ = 1 and λ = 2, includes 0 at λ = 0.5). B is below A at every λ (B − A = −0.031 / −0.021 / −0.059), but each B − A contrast CI includes 0. Per prompt, B05 (stream) and B06 (crowd) lose the most already at λ = 0.5 (−0.090, −0.063); B03 (waves) gains at λ = 0.5 and 1 (+0.048, +0.063). Evidence: the pre-declared rule (monotone decline AND lower than A) is met on point estimates; the between-category difference is not resolved at n = 6.

## 12. H3 — Multi-Event Effect

C: Δ = −0.020 / −0.077 / −0.202 (CIs exclude 0 at all three λ). C is below A at every λ (C − A = −0.032 / −0.048 / −0.033; CIs include 0, the λ = 0.5 contrast just touching 0 at [−0.065, +0.003]). Every C prompt is negative at λ = 1 and 2; at λ = 0.5 only C06 is positive (+0.017). Evidence: by the pre-declared rule (C consistently loses more than A) H3 is met on point estimates; as with H2 the category difference is within bootstrap uncertainty. Because both events in C are discrete, this loss cannot be attributed to a diffuse background.

## 13. H4 — Background Suppression

Primary comparison (D vs C on the full-prompt metric): D is **not** lower than C at any λ (D − C = +0.004 / +0.022 / +0.036; CIs include 0); D's absolute CLAP is the highest of the four categories at every λ. By the pre-declared rule H4 is not supported on the primary comparison. No official per-event ELSA exists (blocked, and not defined by the implementation).

Exploratory only (event-phrase CLAP, not an event detector; `foreground_background_clap_exploratory.png`): in category D the cosine to the *foreground* phrase is flat up to λ = 1 (0.411 → 0.412 → 0.404) and drops at λ = 2 (0.302; Δ −0.108 [−0.232, +0.004]), whereas the cosine to the *background* phrase declines at every λ: 0.210 → 0.171 → 0.120 → 0.080, Δ = −0.039 [−0.059, −0.020], −0.090 [−0.128, −0.053], −0.130 [−0.212, −0.056] (all CIs exclude 0), and 6/6 prompts decline at λ = 1. D01 (dog + rain) background-phrase cosine goes 0.273 → 0.201 → 0.118 → −0.027 while its foreground-phrase cosine rises 0.409 → 0.459. This is consistent in direction with the pilot listening note about rain suppression, but it is a similarity to a short phrase, not evidence of event absence, and it is not the primary metric.

## 14. Cross-Metric Agreement

ELSA vs CLAP agreement cannot be assessed (ELSA blocked). Within CLAP: the full-prompt score and the exploratory background-phrase score disagree on H4 — the full-prompt score shows D no worse than C, while the background-phrase score declines monotonically in D at every λ. Neither is treated as "correct": the full-prompt score is a global similarity that a preserved foreground can dominate, and the phrase score is an unvalidated proxy. This divergence is the main reason the blocked fine-grained metric and the pending human event-coverage ratings matter.

## 15. Human Evaluation Materials

`outputs/diagnostic_v1/human_eval/`: `HUMAN_EVAL_PROTOCOL.md` (procedure, Event Coverage definition, foreground/background split for D, 1–5 quality scale), `HUMAN_EVENT_COVERAGE_TEMPLATE.csv` (432 rows = one per unit × label × requested event, `present_yes_no` blank, `event_role` ∈ single/event/foreground/background), `HUMAN_QUALITY_TEMPLATE.csv` (288 rows, `quality_1_to_5` blank), `blind_key.json` (label → λ and source path per unit; RNG seed 20261022), `blind_audio/` (288 copies named `<ID>_s<seed>_<A–D>.wav`; labels randomized independently per (prompt, seed); λ not exposed). Ratings have not been filled.

## 16. Limitations

- One T2A backbone (Auffusion, `auffusion/auffusion`), one α (1.5), one CFG scale (7.5).
- 24 hand-designed diagnostic prompts; the category taxonomy is hypothesis-driven, and single prompts (e.g. A03, B05, D04) move category means noticeably at n = 6.
- Three seeds; seed-level std at λ = 0 is 0.01–0.08, comparable to the λ = 0.5 effects.
- English prompts only.
- No completed human evaluation; Event Coverage and Quality are pending.
- The primary fine-grained metric (ELSA) is blocked; all hypothesis statuses rest on CLAP, a global embedding similarity that does not measure fidelity, event completeness, or temporal structure, and may be weak for multi-event alignment.
- CLAP cannot separate semantic drift from fidelity loss; latent norm and file amplitude grow with λ (116 of 288 resampled files overshoot 1.0 after sinc resampling, max 1.61; clamping sensitivity was ≤ 0.005 in the previous evaluation).

## 17. Hypothesis Status

Applied to CLAP (secondary metric) because ELSA is blocked; to be re-assessed when ELSA and human ratings are available.

- **H1: Partially Supported** ("weakly supported" by the rule) — Category A ΔCLAP at λ = 0.5 is +0.012 with a CI including 0 (5/6 prompts positive); λ = 1 is mixed and λ = 2 uniformly negative.
- **H2: Supported (point estimates), unresolved between categories** — Category B declines monotonically (−0.019 / −0.050 / −0.229; CI excludes 0 from λ = 1) and lies below A at every λ, but every B − A contrast CI includes 0.
- **H3: Supported (point estimates), unresolved between categories** — Category C loses more than A at every λ (−0.020 / −0.077 / −0.202, all CIs excluding 0), with C − A contrast CIs including 0.
- **H4: Not Supported on the primary comparison** — D is not lower than C at any λ on full-prompt CLAP; the exploratory background-phrase similarity in D does decline at every λ with CIs excluding 0 while the foreground phrase is flat up to λ = 1, which is consistent with background suppression but is not the primary metric.

Interpretation matrix: **Pattern 4** (categories vary without the strict Pattern-1 structure: A improves only at λ = 0.5; B, C, D degrade at all λ; D does not degrade more than C). Read together with the exploratory phrase scores, the data are also compatible with a reading in which the global metric is dominated by preserved foreground events, but that cannot be decided from CLAP alone.

## 18. Recommendation

Based strictly on these results: (1) obtain the two credentials needed for the official ELSA (access to `facebook/sam-audio-large-tv` via a valid Hugging Face login, and an OpenAI key or the repo's local Qwen parser) and run it on the existing 288 files in a separate `cast-elsa` environment — the hypotheses were pre-registered on ELSA and the H4 question is exactly where the global and phrase-level CLAP readings diverge; (2) complete the blind human Event Coverage / Quality ratings on the prepared package, prioritising categories D and C at λ = 1 and 2; (3) do not run further generation, λ, α or CFG changes until (1)–(2) are in. No λ is selected; λ = 0.5 is the only condition whose category deltas are within ±0.02 of vanilla everywhere, which makes it a candidate worth further evaluation, not an "optimal" setting.
