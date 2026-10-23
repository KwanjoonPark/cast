# Figure selection for the DCS 2026 2-page paper

Scope of this task: figure preparation only. No new audio, no new experiments, no recomputed statistics. All numbers come
from `outputs/multimetric_eval/combined/analysis_summary.json`, `outputs/multimetric_eval/combined/all_metrics.csv` and
`outputs/sharp_softmax_v1/hypothesis_summary.json`; spectrograms are computed from existing wav files only. Source scripts
for every figure are in `paper_assets/source/`; 300-dpi PNG and vector PDF exports are in `paper_assets/figures/`;
column-width previews used for the readability test are in `paper_assets/previews/`; Korean captions are in
`paper_assets/captions/figure_captions_ko.md`. Figures use English labels (only Noto Serif CJK Bold is installed; mixed
Korean/Latin typography inside figures was judged unreliable); captions are Korean. No DCS template was found locally
(`find` over the home directory); layout assumptions: two-column, ≈8.4 cm per column, ≈17 cm full width.

## Candidates

| # | File | Content | Size (in) | Readability test (column-width preview) | Redundancy with Table 1 |
|---|---|---|---|---|---|
| Fig 1 | `fig1_method_overview.{png,pdf}` | Original schematic: Auffusion pipeline → cross-attention intervention (four formulas) + conceptual concentration bars (marked "not measured") | 6.8 × 3.0 (two-column) | Pass after two layout passes: boxes, formulas and panel labels legible at 1300 px; conceptual bars clearly labelled | None (method, not results) |
| Fig 2 | `fig2_metric_delta.{png,pdf}`; one-column `fig2_metric_delta_compact.{png,pdf}` | ΔCLAP / ΔAQAScore / ΔPQ vs Dense for Sharp β=1.5, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2; 95 % prompt-level bootstrap CIs copied from the reports; Dense = dashed 0 | 6.8 × 2.1 (two-column) or 3.3 × 5.3 (one-column, stacked) | Pass at both widths: tick labels, panel tags, error bars and markers readable; no overlaps | Low: Table 1 gives absolute means; Fig 2 adds direction, magnitude and uncertainty in one glance |
| Fig 3 | `fig3_spectrogram_D01.{png,pdf}` | 2×2 mel-spectrograms, D01 "A dog barking while rain falls in the background.", seed 2026, identical initial latent (`5ea8d1c0…`), same preprocessing, one shared dB scale, exact per-sample CLAP/AQA/PQ in the panel titles | 6.8 × 3.9 (two-column) | Pass at 1300 px; panel tags, axes, colorbar and metric annotations readable | None |
| Fig 3b (optional) | `fig3b_spectrogram_A05.png` | Same layout for A05 "A wooden door slamming shut.", seed 2026 | 6.8 × 3.9 | Pass | None |
| Fig 4 (optional) | `fig_optional_concentration.{png,pdf}` | (a) β vs effective support exp(H) = 18.6 / 8.2 / 4.1 / 1.9; (b) β vs mean CLAP 0.555 / 0.546 / 0.535 / 0.519 (Sharp Softmax report values) | 3.3 × 1.6 (one-column) | Pass at 640 px after relabelling ticks | Medium: CLAP panel repeats a Table-1-like quantity for Sharp β |

## Recommendation (after inspecting the renderings)

- **MUST INCLUDE — Fig 2 (metric deltas).** It carries the paper's central finding with uncertainty: Sharp β=1.5 and Entmax λ=0.5 are small changes whose CIs reach 0 on AQAScore and PQ (CLAP for β=1.5 is slightly below 0), pure Entmax lowers all three, PLADIS λ=2 has the largest drop on every metric. Use the two-column version if the layout allows a full-width figure; otherwise the stacked one-column version is readable but costs ≈13 cm of column height, so the two-column version is preferred.
- **SHOULD INCLUDE — Fig 3 (D01 spectrograms).** The only qualitative evidence in the paper, tied to the multi-event research question, with a fixed seed chosen a priori and a fair shared colour scale. Caption language stays neutral (differences in time-frequency structure; no claim of event disappearance). If space is short, it is the first figure to shrink to one column (2×2 remains legible at ≈8.4 cm width only if the metric annotations are dropped from the titles; keep them in the caption instead).
- **OPTIONAL — Fig 1 (method schematic).** Useful if the text cannot afford a method paragraph with four equations; in a 2-page paper the schematic (≈6.5 cm tall at full width) competes directly with Fig 3 for space. Judgement: Fig 3 is more valuable than Fig 1 here, because the method can be stated in two lines of equations in the text, whereas the qualitative comparison cannot be replaced by text. Include Fig 1 only if the method section would otherwise need more than ~8 lines.
- **OPTIONAL — Fig 4 (concentration mechanism).** Supports the mechanistic sentence "degradation tracks concentration, not hard zeroing" but partly duplicates Table 1 / Fig 2. Include only if space remains after Figs 2 and 3; the effective-support panel alone (18.6 → 1.9 keys) is the non-redundant part.
- **OMIT — Fig 3b (A05).** A second spectrogram does not fit a 2-page paper and A05 is one of the prompts that happened to be tolerant of sharpening; showing it would read as cherry-picking. Kept in `paper_assets/figures/` for the appendix or presentation only.

Suggested set for 2 pages: Table 1 + Fig 2 (full width) + Fig 3 (full width or one column without in-panel metrics). Add Fig 1 or Fig 4 only if a column remains.

## Third-party figures considered (reference only; NOT copied into paper_assets/figures)

- PLADIS: Kim & Sim, "PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity", ICCV 2025 (arXiv:2503.07677). Fig. 2(b) explains the λ-extrapolation between sparse and dense cross-attention; Fig. 11 shows cross-attention maps sharpening with λ. Useful for understanding, but our Fig 1 conveys the same relation with our notation, so no third-party figure is needed.
- Auffusion: Xue et al., "Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation", arXiv:2401.01044. Fig. 1 (architecture) and Fig. 2 (per-token cross-attention maps over spectrograms). Our Fig 1's pipeline row is an original simplified redraw of the same stages; the original is not copied.

## Notes on fairness and provenance

- Fig 2 uses the same statistical convention as the reports (prompt = unit, 3 seeds averaged, 10 000 bootstrap resamples, seed 20261023); values were read from the stored JSON, not recomputed.
- Fig 3/3b: seed 2026 was fixed before rendering; the four panels verified equal prompt, seed and initial-latent SHA-256 from the manifests; vmin/vmax fixed to [−80, 0] dB relative to the global maximum over the four clips (reference amplitude 5.95 for D01, 11.89 for A05); identical mel parameters (16 kHz, FFT 1024, hop 160, 128 mel, 0–8 kHz).
- The conceptual bars in Fig 1 are a stylised illustration (random logits with three temperatures/top-2 support) and are labelled as such.
