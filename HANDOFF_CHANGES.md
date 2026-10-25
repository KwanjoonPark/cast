# HANDOFF_CHANGES — advisor-feedback continuation (2026-10-25)

Branch `kwanjoon/advisor-feedback-continuation`, based on `origin/wonseok/advisor-feedback` @ `1fd7fb3` (itself linear on `main` @ `830bd52`).
No audio was generated, no experiment was run, no hyper-parameter was changed, `third_party/` and the CAST attention code were not touched. Nothing was pushed.

## Files modified

| File | Change | Why |
|---|---|---|
| `experiments/analyze_cfg_sweep_v1.py` | (1) x-tick label font 6.5 → 7.5 pt in the grid figure; (2) when all three metrics (`clap`, `pq`, `aqa`) are present, the grid figure is additionally saved as `analysis/fig_cfg_delta_grid_3metrics.{png,pdf}` | (1) readability at two-column width ("no tiny text"); (2) the fixed manuscript file name requested for the 3-metric figure. Statistics, bootstrap seed (20261024), resamples (10,000), arm order and colours are unchanged. Regression check: re-running on a scratch copy of the real CLAP/PQ tables reproduces `analysis_summary.json` byte-for-byte and `delta_table.csv` identically in content (only line endings differ: the csv module writes CRLF, while the committed file was LF-normalised by git on the teammate's Windows checkout — expect a whole-file line-ending diff when the analysis is re-run after AQAScore; the numbers are unchanged); the `_3metrics` files are not written while AQAScore is absent. The 3-metric path was exercised once with fabricated AQAScore values in the session scratchpad only (deleted afterwards; never under `outputs/`). |

## Files added

| File | Purpose |
|---|---|
| `experiments/verify_sweep_audio.py` | Verifies the handed-over sweep audio against `outputs/cfg_sweep_v1/audio_index.csv` (presence, per-file SHA-256, 16 kHz mono PCM_16); optionally checks the zip's SHA-256 and extracts **only** the index-listed wav members, so tracked files bundled in the zip (`eval_aqascore.py`, prompt file, index) are never overwritten. Writes `outputs/cfg_sweep_v1/audio_verification.json`. Tested: FAIL/exit 1 on the current empty tree (1080 missing), PASS on the 504-file A5000 multimetric index, PASS on a 3-file synthetic zip with a prefixed member path and non-wav members, exit 2 on a zip SHA mismatch. |
| `HANDOFF_VERIFICATION.md` | Git state, verification of every handoff claim against the repository, remaining work with exact commands. |
| `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md` | Completion report; AQAScore-dependent sections marked `[PENDING AQAScore]`. |
| `PAPER_RESULT_SNIPPETS_KO.md` | Korean manuscript sentences (CFG, v2, layer ablation, conclusion, caption, limitation) from the committed analysis outputs; AQAScore slots marked. |
| `HANDOFF_CHANGES.md` | This file. |

## Commands run (chronological, repo root unless noted)

```
git status; git branch -vv; git remote -v; git fetch --all --prune; git branch -r; git log --oneline --decorate --graph --all -n 30
git diff --name-status main origin/wonseok/advisor-feedback; git diff main origin/wonseok/advisor-feedback -- CLAUDE.md cast/ .gitignore
cp cast_prompts_v2.json <scratchpad>/preserved/ ; mv cast_prompts_v2.json <scratchpad>/preserved/…moved_aside   # byte-identical to the branch copy
git checkout -b kwanjoon/advisor-feedback-continuation origin/wonseok/advisor-feedback
git show origin/wonseok/advisor-feedback:"docs/CAST_CFG_실험_인수인계_보고서.docx" > <scratchpad>/handoff_docx/report.docx   # text extracted with a zipfile/XML one-off script
~/miniconda3/envs/cast/bin/python -m pytest cast/tests -q            # 13 passed
<pandas one-offs> verifying audio_index.csv / clap_scores.csv / audiobox / manifests for cfg_sweep_v1, listen_prompts_v2, layer_ablation_v1
sha256sum experiments/prompts/cast_diagnostic_v1.json ; sed 's/$/\r/' … | sha256sum   # CRLF explains the manifest hash
find /home /tmp /mnt /media /data /srv /opt -iname 'cast_cfg_sweep_aqa_package*'        # not found
ls -la ~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker ; cast-eval python -c "import torch, transformers"   # evaluator ready
~/miniconda3/envs/cast-eval/bin/python experiments/analyze_cfg_sweep_v1.py --root <scratchpad>/figtest   # fake-AQA layout test, then deleted
~/miniconda3/envs/cast-eval/bin/python experiments/analyze_cfg_sweep_v1.py --root <scratchpad>/regress   # byte-identical regression check
~/miniconda3/envs/cast-eval/bin/python experiments/verify_sweep_audio.py …                               # four test runs, reports in scratchpad
```

## Generated analysis files

None under `outputs/` in this session (AQAScore blocked). Expected after the remaining steps: `outputs/cfg_sweep_v1/audio_verification.json`,
`outputs/cfg_sweep_v1/aqascore/{validation.json,sample_scores.csv,event_scores.csv,run_info.json}`, updated `outputs/cfg_sweep_v1/analysis/{delta_table.csv,analysis_summary.json,fig_cfg_delta_grid.*,fig_cfg_delta_lines.*}` and new `fig_cfg_delta_grid_3metrics.{png,pdf}`.

## Files intentionally not committed

- Any `*.wav`, `*.pt`, per-run `spectrogram.png` / `manifest.json`, logs (already gitignored); the audio package zip itself (keep it outside the repo tree).
- Session scratchpad material: preserved copy of `cast_prompts_v2.json`, extracted handoff text, test zips/reports, fake-data figure test (deleted).
- Nothing was deleted from the machine; disk space (683 MB free) is for the user to resolve (candidates listed in HANDOFF_VERIFICATION.md).
