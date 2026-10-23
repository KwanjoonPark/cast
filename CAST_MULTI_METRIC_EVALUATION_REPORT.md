# CAST Multi-Metric Evaluation Report

Date: 2026-10-23. Evaluation only: no audio generated, CAST untouched, no new conditions. Scripts: `experiments/eval_aqascore.py`, `experiments/eval_audiobox.py`, `experiments/analyze_multimetric.py`, `experiments/make_thinker_dir.py`; environment `experiments/env/requirements-eval.txt` + `setup_eval_env.sh` (conda env `cast-eval`; the frozen `cast` generation env was not modified). Outputs under `outputs/multimetric_eval/` (logs `outputs/_aqascore.log`, `outputs/_audiobox.log`).

## 1. Executive Summary

| Item | Status |
|---|---|
| Unique audios | **504** (576 provenance rows − 72 bitwise-identical dense files shared by Diagnostic λ=0 and Sharp β=1; verified by SHA-256, max 2 provenances per file) |
| AQAScore | **PASS** — 504 / 504 full-prompt scores + 756 Event-AQAScore rows, all finite, deterministic (3 repeats identical), 3.6 min |
| AQAScore backbone | **Qwen2.5-Omni-7B** (thinker only, bf16, 8.93 B parameters loaded, 16.8 GB peak VRAM on the RTX A5000); no 3B fallback was needed |
| AudioBox Aesthetics | **PASS** — official `audiobox_aesthetics 0.0.4` + pretrained all-axes checkpoint (`facebook/audiobox-aesthetics`, snapshot 9b1dd8e5), 504 / 504, deterministic; primary axis **PQ** |
| Event-AQAScore (diagnostic) | **PASS** |
| Broad trends (numerical) | AQAScore is far less sensitive to concentration than CLAP: the all-prompt mean is 0.905 for dense and within ±0.013 for every Sharp β and for Entmax λ=0.5; it drops −0.027 at λ=1 and −0.100 at PLADIS λ=2. AudioBox PQ declines monotonically with concentration in both families (Sharp β=2 −0.20; Entmax λ=1 −0.22, λ=2 −0.64), most strongly for two-event and diffuse prompts and least for discrete-single prompts. In Category D the Event-AQAScore of the *background* event does **not** decline with concentration at the category level (unlike the exploratory background-phrase CLAP), while the *foreground* Event-AQAScore falls only under PLADIS λ=2. |

## 2. Why Additional Metrics

- **CLAP** (already computed): cosine similarity of global audio and text embeddings; measures coarse relevance, cannot tell which requested event is present or whether the audio is clean.
- **AQAScore** (arXiv:2601.14728): probabilistic semantic verification — an audio-language model is asked whether the audio contains the described events and the normalized probability of "Yes" versus "No" is read out; sensitive to missing or incorrect elements that embedding similarity overlooks.
- **AudioBox Aesthetics PQ** (arXiv:2502.05139): automatic perceptual production-quality predictor (1–10) trained on human aesthetic ratings; CE/CU/PC are the other three axes.
- **Human Event Coverage / Quality**: the eventual ground truth for event presence and perceived quality; blind packages exist, ratings are still blank.

## 3. Evaluation Dataset

Source files: Diagnostic V1 (`outputs/diagnostic_v1`, 24 prompts × 3 seeds × λ ∈ {0, 0.5, 1, 2}) and Sharp Softmax V1 (`outputs/sharp_softmax_v1`, same prompts/seeds × β ∈ {1, 1.25, 1.5, 2}), prompt dataset SHA-256 `c89ac135…968d38`. Deduplication by audio SHA-256: 576 rows → 504 unique files; exactly 72 files carry two provenances (`entmax:lambda=0` and `sharp_softmax:beta=1`), as expected from the bitwise reuse verified in the Sharp Softmax report. Every metric was computed once per unique file and joined back to all 576 provenance rows (`combined/audio_index.csv`, `combined/provenance_rows.csv`, `combined/all_metrics.csv`). Family labels: `dense` (λ=0 ≡ β=1), `sharp_softmax` (β>1), `entmax` (λ>0).

## 4. AQAScore Method

Question (paper Sec. 3, default template without quotation marks):

```
Does this audio contain the sound events described by the text: {t}? Please answer yes or no.
```

System prompt: paper Table 7, verbatim (listen attentively; judge strictly on audible sounds; do not assume missing/unclear events exist; respond concisely with "yes"/"no"). Score (paper Sec. 3):

```
s_yes = log p_θ("Yes" | a, q(t)),   s_no = log p_θ("No" | a, q(t))
AQAScore(a, t) = exp(s_yes) / (exp(s_yes) + exp(s_no))      # computed as a numerically stable softmax over [s_yes, s_no]
```

Event-AQAScore (diagnostic adaptation, not an official benchmark): the identical procedure with each pre-registered event phrase from `cast_diagnostic_v1.json` substituted for {t}; for Category D the `foreground_event` / `background_event` phrases give `foreground_aqascore` / `background_aqascore`. Scores are kept continuous; no threshold is applied.

## 5. AQAScore Implementation Validation

| Item | Value |
|---|---|
| Environment | conda `cast-eval`: Python 3.10.22, torch 2.4.1+cu124 (CUDA 12.4), transformers 4.57.1, huggingface_hub 0.36.2, torchvision 0.19.1; RTX A5000 24 GB |
| Backbone / revision | `Qwen/Qwen2.5-Omni-7B`, HF snapshot `ae9e1690`; loaded as `Qwen2_5OmniThinkerForConditionalGeneration` (thinker = LLM + audio/vision encoders; talker and token2wav not loaded) from a local directory whose `model.safetensors.index.json` lists only thinker tensors in shards 1–4 (shard 5 contains talker/token2wav only; weights unmodified; `experiments/make_thinker_dir.py`) — done because only 29 GB of disk were available |
| Precision / mode | bf16 weights, `model.eval()`, `torch.inference_mode()`, no sampling or generation (log-probabilities read directly) |
| Audio | original 16 kHz mono wav via soundfile, no resampling (the Qwen2.5-Omni feature extractor operates at 16 kHz) |
| Yes/No tokenization | `"Yes"` → token 9454, `"No"` → token 2753 (single tokens each; decoded back to "Yes"/"No"), so s_yes/s_no are the next-token log-softmax values after the assistant turn start; the code path for multi-token responses (teacher-forced sum) exists but was not needed. Lowercase `"yes"` (9693) / `"no"` (2152) log-probs are recorded as a secondary robustness column (`aqascore_full_lowercase_variant`), not used as the primary score |
| Determinism | 3 full evaluations of A01/seed 42/λ=0 with its prompt: scores 0.966914, 0.966914, 0.966914 (max diff 0.0); log p(Yes) = −2.477, log p(No) = −5.852, finite, in [0, 1] |
| Sanity | same audio: matching prompt 0.967 vs unrelated prompt ("Steady rain falling on a roof.") 0.008 |
| Calibration note | absolute probabilities are low (mean p(Yes) 0.088, p(No) 0.012 over all files; the system prompt asks for lowercase answers and the lowercase-variant mean is 0.938), which is why the paper normalizes Yes against No; the normalized score is saturated near its ceiling for most files (mean 0.884, median 0.932, 68 % above 0.9) |

## 6. Full-Prompt AQAScore Results

Prompt-level (3 seeds averaged) means; Δ vs dense with 95 % prompt-level bootstrap CIs (10 000 resamples, seed 20261023, n = 6 prompts per category). Full tables: `combined/category_summary.csv`; figures `figures/{sharp,entmax}_aqascore_by_category.png`, `figures/{sharp,entmax}_delta_aqascore.png`.

| Scope | Dense | β=1.25 | β=1.5 | β=2 | λ=0.5 | λ=1 | λ=2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 24 prompts | 0.905 | 0.905 (+0.000 [−.012, +.013]) | 0.892 (−0.013 [−.036, +.007]) | 0.894 (−0.011 [−.029, +.007]) | 0.906 (+0.002 [−.012, +.016]) | 0.878 (−0.027 [−.057, +.000]) | 0.805 (−0.100 [−.157, −.045]) |
| A DISCRETE_SINGLE | 0.858 | 0.871 (+0.013) | 0.870 (+0.012) | 0.862 (+0.004) | 0.870 (+0.012) | 0.850 (−0.007) | 0.847 (−0.011) |
| B DIFFUSE_SINGLE | 0.943 | 0.940 (−0.003*) | 0.936 (−0.007*) | 0.928 (−0.015*) | 0.939 (−0.004) | 0.924 (−0.019*) | 0.791 (−0.152*) |
| C DISCRETE_DISCRETE | 0.919 | 0.902 (−0.017*) | 0.896 (−0.023*) | 0.893 (−0.026*) | 0.897 (−0.022*) | 0.846 (−0.073*) | 0.810 (−0.109*) |
| D DISCRETE_DIFFUSE | 0.900 | 0.908 (+0.008) | 0.865 (−0.035) | 0.894 (−0.006) | 0.920 (+0.020) | 0.892 (−0.008) | 0.770 (−0.129*) |

\* = 95 % CI excludes 0. Category A deltas all have CIs spanning ±0.04–0.15 (A02 and A06 start low, 0.64 and 0.76, and rise under Entmax; A03 falls from 0.92 to 0.60 at λ=2). Per-prompt values: `combined/analysis_summary.json → families.*.metrics.aqascore_full.prompt_level`.

## 7. Event-AQAScore Results (diagnostic adaptation)

Mean over the requested event phrases of each prompt (`mean_event_aqascore`; 1 phrase for A/B, 2 for C/D):

| Scope | Dense | β=1.25 | β=1.5 | β=2 | λ=0.5 | λ=1 | λ=2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 0.873 | 0.871 (−0.002) | 0.868 (−0.005) | 0.861 (−0.013) | 0.875 (+0.001) | 0.848 (−0.025) | 0.793 (−0.080*) |
| A | 0.862 | 0.874 | 0.877 | 0.859 | 0.875 | 0.850 | 0.852 |
| B | 0.958 | 0.956 | 0.951* | 0.944* | 0.952* | 0.941* | 0.816* |
| C | 0.886 | 0.864* | 0.865* | 0.848* | 0.858 | 0.817* | 0.785* |
| D | 0.787 | 0.789 | 0.779 | 0.792 | 0.814 | 0.784 | 0.720 |

Event-level scores track the full-prompt scores: small, mostly within-CI changes for Sharp Softmax and λ ≤ 1 (except C, where both β ≥ 1.25 and λ ≥ 0.5 are below dense with CIs excluding 0), and a clear drop only for PLADIS λ=2. This is the same AQAScore procedure applied to short phrases and is not an event detector.

## 8. Category D Foreground / Background Analysis

Category D (discrete foreground + diffuse background), prompt-level means with Δ vs dense (CIs in `category_summary.csv`; figures `figures/{sharp,entmax}_fg_bg_aqascore.png`):

| Condition | Full AQAScore | FG Event-AQA | BG Event-AQA | FG phrase CLAP | BG phrase CLAP | PQ |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.900 | 0.878 | 0.696 | 0.411 | 0.210 | 5.39 |
| Sharp β=1.25 | 0.908 (+0.008) | 0.897 (+0.019) | 0.681 (−0.016) | 0.408 | 0.190* | 5.40 (+0.01) |
| Sharp β=1.5 | 0.865 (−0.035) | 0.858 (−0.020) | 0.699 (+0.003) | 0.408 | 0.171* | 5.28 (−0.11) |
| Sharp β=2 | 0.894 (−0.006) | 0.871 (−0.007) | 0.712 (+0.016) | 0.409 | 0.132* | 5.23 (−0.16*) |
| Entmax λ=0.5 | 0.920 (+0.020) | 0.899 (+0.021) | 0.728 (+0.032) | 0.412 | 0.171* | 5.37 (−0.02) |
| Entmax λ=1 | 0.892 (−0.008) | 0.868 (−0.010) | 0.700 (+0.004) | 0.404 | 0.120* | 5.20 (−0.19*) |
| PLADIS λ=2 | 0.770 (−0.129*) | 0.749 (−0.129*) | 0.691 (−0.006) | 0.302 | 0.080* | 4.93 (−0.46*) |

Numerical reading: at the category level the **background Event-AQAScore is flat** across all Sharp β and all Entmax λ (every Δ within ±0.03 with CIs of ±0.05–0.11), whereas the **foreground Event-AQAScore is flat up to λ=1 / all β and drops only under PLADIS λ=2** (−0.129, CI excluding 0) together with the full-prompt score. This does not reproduce the exploratory background-phrase CLAP pattern (monotone decline, CIs excluding 0 at every condition). Per prompt the picture is heterogeneous (`analysis_summary.json`): D01 (dog + rain) is the one prompt where the background score does fall with concentration in both families (Entmax 0.94 → 0.93 → 0.87 → 0.81; Sharp 0.94 → 0.94 → 0.93 → 0.86) while its foreground stays at 0.97; D06 (door + crowd) moves the opposite way (0.64 → 0.82 → 0.85 → 0.88 under Entmax); D05's background phrase ("wind blowing through trees") scores ≈ 0.19 under every condition, including dense. The category mean therefore hides prompt-specific behaviour, and the D01 trajectory is the only AQA evidence consistent with the earlier listening note about rain.

## 9. AudioBox Aesthetics Method

Official package `audiobox_aesthetics 0.0.4` (`facebookresearch/audiobox-aesthetics`), `initialize_predictor()` with the default pretrained all-axes checkpoint downloaded from `facebook/audiobox-aesthetics` (HF snapshot 9b1dd8e5, `model.safetensors`); no retraining or finetuning. Output axes (1–10): CE Content Enjoyment, CU Content Usefulness, PC Production Complexity, **PQ Production Quality**. PQ is the primary CAST quality metric because the perceptual hypothesis concerns blurring, distortion and loss of cleanliness, which PQ is defined to capture; CE/CU/PC are reported descriptively and are not averaged into any composite. Preprocessing: the package loads the file path itself (`torchaudio.load`), converts to mono and resamples to its constant 16 kHz (a no-op for our 16 kHz files), and scores fixed-size windows; identical for every file. Device CUDA, fp32. Validation: three repeated scorings of one file identical (max diff 0.0); batch-size sensitivity (batch 1 vs batch 16, 16 files) ≤ 8.3e-4 on PQ (`audiobox/validation.json`).

## 10. Production Quality Results

PQ, prompt-level means, Δ vs dense with 95 % bootstrap CIs (`figures/{sharp,entmax}_pq_by_category.png`):

| Scope | Dense | β=1.25 | β=1.5 | β=2 | λ=0.5 | λ=1 | λ=2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 5.92 | 5.85 (−0.07 [−.15, +.01]) | 5.83 (−0.09 [−.19, +.01]) | 5.73 (−0.20 [−.34, −.06]) | 5.83 (−0.09 [−.18, +.00]) | 5.70 (−0.22 [−.36, −.09]) | 5.28 (−0.64 [−.90, −.39]) |
| A | 6.74 | 6.71 (−0.03) | 6.75 (+0.01) | 6.75 (+0.02) | 6.72 (−0.02) | 6.78 (+0.04) | 6.21 (−0.53) |
| B | 5.12 | 4.95 (−0.17) | 4.95 (−0.16) | 4.87 (−0.25*) | 4.91 (−0.21*) | 4.77 (−0.35*) | 4.74 (−0.38*) |
| C | 6.44 | 6.36 (−0.08) | 6.32 (−0.12) | 6.05 (−0.39*) | 6.33 (−0.11) | 6.04 (−0.40*) | 5.24 (−1.20*) |
| D | 5.39 | 5.40 (+0.01) | 5.28 (−0.11) | 5.23 (−0.16*) | 5.37 (−0.02) | 5.20 (−0.19*) | 4.93 (−0.46*) |

Answers to the four questions, numerically: (1) PQ decreases with concentration in both families, monotonically at the all-prompt level, and the drop at matched control index is larger for Entmax than for Sharp Softmax (λ=1 −0.22 vs β=2 −0.20 despite β=2 being less concentrated than λ=1 by the entropy measures). (2) PLADIS λ=2 is the clear outlier (−0.64 overall, −1.20 in C). (3) Category A is the only category whose PQ is unchanged by every non-extrapolated condition (|Δ| ≤ 0.04); D degrades moderately (−0.16 … −0.46 from β=2 / λ=1 upward); B degrades already at the mildest conditions (λ=0.5 −0.21*, β=1.25 −0.17) and C most at the strong ones. (4) Agreement with the earlier "blurred / smeared" perception cannot be claimed until human quality ratings exist; the automatic trend has the same sign. CE and CU follow PQ (all-prompt CE 3.06 → 2.65, CU 5.57 → 4.81 at λ=2); PC is flat (2.6–2.7) except a rise in D under λ=2 (2.87 → 3.25).

## 11. Combined CLAP / AQAScore / PQ Results

All 24 prompts (prompt-level means; `combined/cross_metric_summary.csv` also holds the per-category versions):

| Method | Control | CLAP | AQAScore | mean Event-AQA | PQ | CE | CU | PC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | λ=0 ≡ β=1 | 0.555 | 0.905 | 0.873 | 5.92 | 3.06 | 5.57 | 2.69 |
| Sharp Softmax | β=1.25 | 0.546 | 0.905 | 0.871 | 5.85 | 3.00 | 5.46 | 2.67 |
| Sharp Softmax | β=1.5 | 0.535 | 0.892 | 0.868 | 5.83 | 2.97 | 5.40 | 2.67 |
| Sharp Softmax | β=2 | 0.519 | 0.894 | 0.861 | 5.73 | 2.93 | 5.30 | 2.67 |
| Entmax interpolation | λ=0.5 | 0.545 | 0.906 | 0.875 | 5.83 | 2.99 | 5.43 | 2.66 |
| Pure Entmax | λ=1 | 0.502 | 0.878 | 0.848 | 5.70 | 2.90 | 5.26 | 2.65 |
| PLADIS extrapolation | λ=2 | 0.364 | 0.805 | 0.793 | 5.28 | 2.65 | 4.81 | 2.70 |

Category-level: A — CLAP 0.504 / AQA 0.858 / PQ 6.74 at dense; AQA and PQ flat for all β and λ ≤ 1 (AQA +0.013/+0.012/+0.004 β; +0.012/−0.007 λ), PQ −0.53 and CLAP −0.17 only at λ=2. B — AQA 0.943 → 0.791 and PQ 5.12 → 4.74 at λ=2, with PQ already down at λ=0.5/β=1.25. C — the most sensitive on every metric (AQA 0.919 → 0.810, PQ 6.44 → 5.24, CLAP 0.570 → 0.368 at λ=2). D — full AQA flat to λ=1 (0.900 → 0.892), −0.129 at λ=2; PQ −0.19 at λ=1, −0.46 at λ=2.

## 12. Attention Concentration vs Metrics

Sharp Softmax prompt-level points (24 prompts × 4 β = 96; entropy from the Sharp Softmax run's conditional-half statistics, no attention regenerated). Pooled correlations between normalized attention entropy and metric are near zero (Spearman: CLAP 0.08, AQAScore 0.02, PQ 0.11, mean Event-AQA 0.00) because between-prompt differences dominate; the within-prompt mean Spearman (entropy vs metric across the four β of the same prompt, averaged over prompts) is positive for CLAP (0.46), AQAScore (0.39) and PQ (0.28) — lower entropy tends to go with lower metric within a prompt — and ≈ 0 for the background Event-AQA (0.07) and foreground Event-AQA (0.23). Correlational only; no causal claim.

## 13. Agreement / Disagreement Across Metrics

- **CLAP down, AQAScore stable** (Sharp β ≤ 2, Entmax λ=0.5, Category A at every condition): global similarity moves while semantic verification does not; AQAScore is near its ceiling, so it may be insensitive to small changes, or CLAP may be reacting to acoustic rather than semantic change.
- **CLAP background-phrase down, background Event-AQA flat** (Category D, all β and λ ≤ 1): the two "background" proxies disagree at the category level; only D01 shows a concentration-dependent background decline in both. Which proxy is right is exactly what the blank human Event Coverage ratings are needed for.
- **Semantic score stable, PQ down** (B at λ=0.5/β=1.25, D at λ=1, C at β=2): quality degrades before semantic verification moves.
- **All three down** (PLADIS λ=2 everywhere; Entmax λ=1 in C): the strong conditions degrade on every axis.
- Nowhere does any metric show a category-level improvement with a CI excluding 0; the largest positive point estimates are AQAScore +0.013 (A, β=1.25) and +0.020 (D, λ=0.5).

## 14. Human Evaluation Readiness

`outputs/multimetric_eval/human_eval_join_template.csv` (576 rows): one row per blind human-eval file of both packages (`outputs/diagnostic_v1/human_eval`, `outputs/sharp_softmax_v1/human_eval`), carrying unit, label, blind file, source audio, family and control value, and the automatic metrics (CLAP, AQAScore, event1/event2/fg/bg Event-AQA, PQ/CE/CU/PC); the human columns (`human_event_coverage`, `human_fg_present`, `human_bg_present`, `human_quality_1_to_5`) are empty and must be filled from the blind rating templates before any join.

## 15. Limitations

- AQAScore depends on one ALLM backbone and its calibration; absolute P(Yes) is low and the normalized score sits near 1 for most files (ceiling effect), limiting sensitivity to mild degradation.
- Event-AQAScore is a diagnostic adaptation of the AQAScore procedure to short phrases; the background scores are prompt-dependent (D05's wind phrase ≈ 0.19 for every condition) and have wide CIs at n = 6.
- AudioBox PQ is an automatic perceptual predictor, not a human rating; its batch sensitivity (≤ 8e-4) is negligible but its validity for synthetic environmental sounds is assumed.
- No completed human Event Coverage or Quality ratings; the agreement of automatic trends with the earlier listening impressions is untested.
- One T2A backbone, one CFG, 24 hand-designed prompts in a hypothesis-driven taxonomy, 3 seeds.
- ELSA remains unavailable (gated SAM-Audio + missing OpenAI key); not retried.
- Only shards 1–4 of Qwen2.5-Omni-7B were downloaded (thinker weights complete); disk space was the reason and the loaded parameter count (8.93 B) matches the thinker.

## 16. Implication for CAST

The data favor **C (both semantic coverage and quality degrade)** for the strong conditions (Entmax λ=1 in multi-event prompts, PLADIS λ=2 everywhere), and **B (concentration mainly harms audio fidelity)** for the mild-to-moderate conditions (Sharp β ≤ 2, Entmax λ=0.5, and Entmax λ=1 in single-event prompts), where AQAScore is flat within CI while PQ already declines (B and D categories). **A (semantic concentration harms event coverage)** is supported at the category level only by the exploratory CLAP background-phrase similarity, not by the background Event-AQAScore, which is flat; the one prompt where both proxies agree is D01 (dog + rain). **D (CLAP underestimated a benefit revealed by AQAScore)** is not supported: AQAScore shows no category-level improvement with a CI excluding 0 under any condition, and the positive point estimates (A +0.013, D +0.020) are within noise. Taken together with the Sharp Softmax report, concentration-based cross-attention modification in Auffusion does not improve semantic verification and lowers predicted production quality in proportion to concentration; the decisive remaining evidence is the blank human event-coverage and quality ratings, for which the join template is prepared.
