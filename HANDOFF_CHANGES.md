# HANDOFF_CHANGES — advisor-feedback continuation (2026-10-25)

Branch `kwanjoon/advisor-feedback-continuation`, based on `origin/wonseok/advisor-feedback` @ `1fd7fb3` (itself linear on `main` @ `830bd52`).
No audio was generated, no new experiment was run, no hyper-parameter was changed, `third_party/` and the CAST attention code were not touched. Nothing was pushed.
Session 1 = inheritance and verification (committed locally as `20ebfcd`). Session 2 = AQAScore on the CFG sweep, analysis, documents (uncommitted; see "Git state at the end").

## Files modified

| File | Change | Why |
|---|---|---|
| `experiments/analyze_cfg_sweep_v1.py` (session 1) | x-tick label font 6.5 → 7.5 pt; when CLAP, AQAScore and PQ are all present the grid figure is also saved as `analysis/fig_cfg_delta_grid_3metrics.{png,pdf}` | readability at two-column width; fixed manuscript file name |
| `experiments/analyze_cfg_sweep_v1.py` (session 2) | figure column order `("clap", "aqa", "pq")` instead of `("clap", "pq", "aqa")` (affects the grid and the line figure only); `delta_table.csv` written with `lineterminator="\n"` | requested layout (ΔCLAP, ΔAQAScore, ΔPQ); the committed table is LF, so CRLF output turned every rerun into a whole-file diff. **Statistics untouched**: the bootstrap draws happen in data-loading order (clap, pq, aqa), not figure order; verified that `delta_table.csv` and `analysis_summary.json` are identical before/after the reorder, and that the git diff of `delta_table.csv` is exactly the 12 new `aqa` rows |
| `.gitignore` (session 2) | added `cast_cfg_sweep_aqa_package.zip` | the 272 MiB handover zip sits at the repo root; prevents an accidental `git add .` |

## Files added

| File | Purpose |
|---|---|
| `experiments/verify_sweep_audio.py` (session 1) | Verifies the handed-over sweep audio against `outputs/cfg_sweep_v1/audio_index.csv` (presence, per-file SHA-256, 16 kHz mono PCM_16); optionally checks the zip SHA-256 and extracts **only** index-listed wav members. Tested on an empty tree (FAIL/exit 1), the 504-file A5000 index (PASS), a synthetic 3-file zip with a prefixed member path and non-wav members (PASS, non-wav skipped), and a wrong zip hash (exit 2). Used for the real extraction in session 2 (1080/1080 PASS). |
| `HANDOFF_VERIFICATION.md`, `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md`, `PAPER_RESULT_SNIPPETS_KO.md`, `HANDOFF_CHANGES.md` | Handoff documents; all `[PENDING AQAScore]` markers resolved in session 2 |

## Commands run (session 2, repo root)

```
git status; git branch --show-current; git log -5 --oneline --decorate
sha256sum cast_cfg_sweep_aqa_package.zip            # a46b69fd… = expected
df -h / ; du -sh cast_cfg_sweep_aqa_package.zip      # 402 MB free -> STOP (below 2 GB)
# user approved deleting the RLinf model cache; freed 7.0 GB (see "Disk")
rm -rf ~/.cache/huggingface/hub/models--RLinf--RLinf-Pi05-ManiSkill-25Main-SFT
python experiments/verify_sweep_audio.py --zip ./cast_cfg_sweep_aqa_package.zip \
    --expected-zip-sha a46b69fd6a46022be17b3df169fcaaa728c5373f229b2121f21c803b98600bf1 --extract   # 1080/1080 PASS
git status                                            # no tracked file modified by the extraction
HF_HUB_OFFLINE=1 ~/miniconda3/envs/cast-eval/bin/python experiments/eval_aqascore.py --model-dir ~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker \
    --index outputs/cfg_sweep_v1/audio_index.csv --out outputs/cfg_sweep_v1/aqascore --validate-only   # smoke test PASS
HF_HUB_OFFLINE=1 ~/miniconda3/envs/cast-eval/bin/python experiments/eval_aqascore.py --model-dir ~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker \
    --index outputs/cfg_sweep_v1/audio_index.csv --out outputs/cfg_sweep_v1/aqascore   # 1080/1080, 7.7 min (log: outputs/_cfg_sweep_aqascore.log)
<pandas one-off> completeness / identity validation of aqascore/sample_scores.csv against the index
~/miniconda3/envs/cast-eval/bin/python experiments/analyze_cfg_sweep_v1.py   # twice: once before, once after the column-order/LF edit; results identical
```

## Generated analysis files (session 2)

- `outputs/cfg_sweep_v1/audio_verification.json` (new)
- `outputs/cfg_sweep_v1/aqascore/{validation.json, sample_scores.csv, event_scores.csv, run_info.json}` (new, 820 KB total)
- `outputs/cfg_sweep_v1/analysis/delta_table.csv`, `analysis_summary.json` (updated: + AQAScore rows)
- `outputs/cfg_sweep_v1/analysis/fig_cfg_delta_grid.{png,pdf}`, `fig_cfg_delta_lines.{png,pdf}` (regenerated with the AQAScore column)
- `outputs/cfg_sweep_v1/analysis/fig_cfg_delta_grid_3metrics.{png,pdf}` (new, manuscript figure)
- `outputs/_cfg_sweep_aqascore.log` (gitignored log)

## Disk

- Before extraction: 402 MB free (the zip itself had consumed most of the previous 674 MB). The user approved deleting the RLinf Pi05 model cache (`~/.cache/huggingface/hub/models--RLinf--RLinf-Pi05-ManiSkill-25Main-SFT`, 7.0 GB, no open files, re-downloadable) → 7.5 GB free. Three other `RLinf`-named cache entries (12 KB each, empty stubs) were left alone. Nothing else was deleted.
- After completion: **7.1 GB free**; extracted wavs 334 MB (`outputs/cfg_sweep_v1/**/audio.wav`, gitignored); zip 272 MiB (`cast_cfg_sweep_aqa_package.zip`, gitignored, kept at the repo root — can be moved outside the repo or deleted later once the extracted wavs are considered safe; not deleted); AQAScore outputs 820 KB.

## Files intentionally not committed

- `cast_cfg_sweep_aqa_package.zip` (now gitignored), all `*.wav`, `*.pt`, per-run `spectrogram.png` / `manifest.json`, logs (gitignored).
- Session scratchpad material (preserved prompt-file copy, extracted handoff text, test zips/reports, fake-data figure dry run which was deleted).

## Session 3 (2026-10-25): final no-CFG experiment

| File | Change | Why |
|---|---|---|
| `experiments/run_cfg_sweep_v1.py` (modified) | new optional `--experiment` argument (default `cast_cfg_sweep_v1`) written into `experiment_manifest.json` | reuse the teammate's driver unchanged for the no-CFG run (`--cfgs 1.0 --arms dense entmax_l2 --out outputs/no_cfg_v1 --experiment cast_no_cfg_v1`) while labelling the manifest correctly |
| `experiments/verify_no_cfg_runtime.py` (new) | diagnostic: UNet forward pre-hook counting latent/conditioning batch sizes and zero (null-condition) rows; Dense/PLADIS smoke generation, determinism rerun, CFG 7.5 contrast, negative-prompt invariance; writes `outputs/no_cfg_v1/runtime_verification.json` | runtime proof that guidance 1.0 is true no-CFG (0 unconditional passes) before generating; the hook is removed at exit, no production code touched |
| `experiments/analyze_no_cfg_v1.py` (new) | paired PLADIS − Dense analysis for the two-arm no-CFG set, importing `load`/`boot_ci`/`B`/`rng_seed` from `analyze_cfg_sweep_v1.py` (identical bootstrap convention); writes `delta_table.csv`, `analysis_summary.json`, `per_prompt_delta.csv`, `fig_no_cfg_pladis_delta.{png,pdf}`, `fig_cfg_to_nocfg_trend.{png,pdf}` (no-CFG point drawn distinctly, hardware note in the figure) | the sweep analyser requires all five arms per group, so a two-arm variant was needed; statistics reused, not re-implemented |
| `CAST_NO_CFG_FINAL_REPORT.md` (new), `outputs/no_cfg_v1/NO_CFG_CODEPATH_VERIFICATION.md` (new), `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md` (§1 and new §16), `PAPER_RESULT_SNIPPETS_KO.md` (new §H–K) | documentation of the experiment and its interpretation | — |

Commands (repo root): `verify_no_cfg_runtime.py --out outputs/no_cfg_v1` (smoke, A01 seed 42, ALL_PASS) → `run_cfg_sweep_v1.py --cfgs 1.0 --arms dense entmax_l2 --out outputs/no_cfg_v1 --experiment cast_no_cfg_v1` (144/144, 13.2 min, log `outputs/_no_cfg_gen.log`) → `make_sweep_index.py --root outputs/no_cfg_v1` → `score_sweep_clap.py --root outputs/no_cfg_v1` (cast env) → `eval_aqascore.py --index outputs/no_cfg_v1/audio_index.csv --out outputs/no_cfg_v1/aqascore` (cast-eval, 1.1 min, log `outputs/_no_cfg_aqascore.log`) → `eval_audiobox.py --index … --out outputs/no_cfg_v1/audiobox` → `analyze_no_cfg_v1.py` (run twice: once before, once after moving the trend-figure legend; `delta_table.csv` identical).
Generated under `outputs/no_cfg_v1/`: 144 wav + per-run `final_latents.pt`/`spectrogram.png`/`manifest.json` (gitignored, 300 MB total), `experiment_manifest.json`, `generation_summary.json`, `audio_index.csv`, `clap_scores.csv`, `aqascore/` (4 files), `audiobox/` (2 files), `runtime_verification.json`, `analysis/` (7 files). Disk after: 6.8 GB free. Tests: `pytest cast/tests` 13 passed (CAST core code unchanged). Nothing committed or pushed in session 3 (not required by the workflow); suggested additions to the session-2 commit below: `experiments/run_cfg_sweep_v1.py experiments/verify_no_cfg_runtime.py experiments/analyze_no_cfg_v1.py CAST_NO_CFG_FINAL_REPORT.md outputs/no_cfg_v1/{NO_CFG_CODEPATH_VERIFICATION.md,runtime_verification.json,experiment_manifest.json,generation_summary.json,audio_index.csv,clap_scores.csv,aqascore,audiobox,analysis}` (all small).

## Session 4 (2026-10-25): final paper assets (no experiments)

- `experiments/make_final_paper_assets.py` (new): reads `outputs/cfg_sweep_v1/analysis/delta_table.csv`, `outputs/no_cfg_v1/analysis/delta_table.csv` and the two manifests; writes `paper_assets/final/{figures,tables,source,previews}` (Fig 1A grid, Fig 1B PLADIS trend, Fig 2 no-CFG, optional CFG→no-CFG; Table 1A/1B, optional CFG 7.5 method table; source notes; 1-col/2-col previews). Deterministic; no audio, no inference. Sharp Softmax excluded by design.
- `paper_assets/final/FIGURE_CAPTIONS_KO.md`, `paper_assets/final/PAPER_ASSET_SELECTION.md` (new): captions and MUST/SHOULD/OPTIONAL/OMIT selection.
- Verification: all twelve PLADIS λ=2 (Δ, CI) values in the tables match `CAST_NO_CFG_FINAL_REPORT.md` §8 (programmatic cell-by-cell check).
- Nothing committed or pushed; `paper_assets/final/` is ≈1 MB and suitable for the suggested commit.

## Session 5 (2026-10-26): same-hardware (A5000) reproduction of the CFG sweep — FINAL generation experiment

- `experiments/run_cfg_sweep_v1.py` reused unchanged (`--cfgs 7.5 3.0 1.5 --arms dense entmax_l2 --out outputs/a5000_guidance_sweep_v1 --experiment cast_a5000_guidance_sweep_v1`): 432/432 in 64 min, 0 failures, 216/216 same-latent pairs.
- `experiments/smoke_guidance_pairs.py` (new): CFG-pair smoke test reusing `UNetProbe` from `verify_no_cfg_runtime.py` (CFG code path, same latent, finite, duration/sr, determinism); writes `smoke_verification.json`.
- `experiments/make_sweep_index.py` (modified): appends `lam, guidance_scale, latent_sha256, gpu` columns read from each sample manifest (existing readers use column names; unaffected).
- `experiments/analyze_a5000_guidance_v1.py` (new): per-CFG paired analysis with the sweep protocol (imports `load`/`boot_ci`), unified table with the copied no-CFG row (`delta_table_unified.csv`), hardware comparison vs the RTX 2080 sweep (`hardware_reproduction_comparison.csv`), diagnostic figure; asserts identical settings/latents/prompts/seeds between the two A5000 runs.
- `experiments/make_final_paper_assets.py` (modified): adds `fig1_guidance_pladis_a5000`, `table_a5000_guidance_delta`, `table_a5000_guidance_absolute` (+ source notes) when the unified table exists; the hardware-mixed optional figure is now written as `fig_optional_cfg_to_nocfg_DEPRECATED_FOR_PAPER`.
- Documents: `HARDWARE_REPRODUCTION_NOTE.md` (new); `CAST_NO_CFG_FINAL_REPORT.md` §14, `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md` §1/§17, `PAPER_RESULT_SNIPPETS_KO.md` §L–M, `paper_assets/final/{PAPER_ASSET_SELECTION.md, FIGURE_CAPTIONS_KO.md}` updated. RTX 2080 results untouched.
- Evaluation: CLAP (`score_sweep_clap.py`, cast env), AQAScore (`eval_aqascore.py`, cast-eval, 3.3 min, validation repeat max|Δ| 0.0), AudioBox (`eval_audiobox.py`); 432/432 each. Logs `outputs/_a5000_guidance_gen.log`, `outputs/_a5000_guidance_aqascore.log` (gitignored).
- Generated under `outputs/a5000_guidance_sweep_v1/`: 432 wav/pt/png/manifest (gitignored, 888 MB) + small tracked-by-policy files (`experiment_manifest.json`, `generation_summary.json`, `smoke_verification.json`, `audio_index.csv`, `clap_scores.csv`, `aqascore/`, `audiobox/`, `analysis/`). Disk after: ≈4.7 GB free. Nothing committed or pushed. Tests: 13 passed (CAST core unchanged).

## Git state at the end of session 2 (no commit made in session 2, per the task instruction)

Modified tracked files: `.gitignore`, `experiments/analyze_cfg_sweep_v1.py`, `HANDOFF_VERIFICATION.md`, `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md`, `PAPER_RESULT_SNIPPETS_KO.md`, `HANDOFF_CHANGES.md`, `outputs/cfg_sweep_v1/analysis/{analysis_summary.json, delta_table.csv, fig_cfg_delta_grid.png/.pdf, fig_cfg_delta_lines.png/.pdf}`.
Untracked small files to commit: `outputs/cfg_sweep_v1/audio_verification.json`, `outputs/cfg_sweep_v1/aqascore/` (4 files), `outputs/cfg_sweep_v1/analysis/fig_cfg_delta_grid_3metrics.{png,pdf}`.
Suggested commit (all small; ≈1.5 MB):
```
git add .gitignore experiments/analyze_cfg_sweep_v1.py HANDOFF_VERIFICATION.md CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md \
    PAPER_RESULT_SNIPPETS_KO.md HANDOFF_CHANGES.md outputs/cfg_sweep_v1/analysis outputs/cfg_sweep_v1/aqascore outputs/cfg_sweep_v1/audio_verification.json
git commit -m "CFG sweep AQAScore (1080/1080), 3-metric analysis and figure, handoff documents completed"
```

## Committed 2026-11-07

Sessions 2–5 were committed on `kwanjoon/advisor-feedback-continuation` as 1802aa5 (CFG sweep AQAScore/analysis), fb99271 (no-CFG experiment), b6b5b26 (A5000 reproduction), 293c9c1 (paper assets) and the following docs commit (README, handoff/completion/snippet documents, this log). Nothing pushed; wav/pt/zip/logs remain gitignored.
