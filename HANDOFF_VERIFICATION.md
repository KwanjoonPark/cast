# CAST Advisor-Feedback Handoff Verification

Date: 2026-10-25. Machine: RTX A5000 24 GB (`reallab`), repo `/home/reallab/parkkwanjoon/CAST`.
Everything below was checked against the repository contents, not inferred from file names or from the handoff text.

## Git State

- **Remote branch used:** `origin/wonseok/advisor-feedback` (the only remote branch besides `origin/main`; name matches the handoff).
- **Teammate branch tip / base commit of this continuation:** `1fd7fb31c4fef131eb1c269c4e6f86bd409af267` ("Add cast_prompts_v2.json and sync the docs with the pushed state").
- **Teammate branch base:** `830bd52` = `origin/main` HEAD. The branch is a linear 4-commit extension of `main` (`0c171b5` code, `ee21acc` result tables/figures, `3b91728` CLAUDE.md + handoff docx, `1fd7fb3` prompt file); no divergence, no rebase needed.
- **Continuation branch:** `kwanjoon/advisor-feedback-continuation`, created locally from `origin/wonseok/advisor-feedback`. Not pushed.
- **Working tree at start:** one untracked file, `cast_prompts_v2.json` (SHA-256 prefix `a0a93135…`), byte-identical to the file committed on the teammate branch. It was copied to the session scratchpad (`…/scratchpad/preserved/`) and moved aside so that `git checkout` could place the identical tracked copy; nothing was stashed, reset or discarded. After checkout the tree is clean (`git status --short` empty), submodules unchanged (Auffusion f44233d, PLADIS 248b9d1, ELSA d2eb34a).
- **Tests:** `pytest cast/tests` → 13 passed (same count as the handoff). The branch's only core-code change is the optional `only=` / `only_modules=` argument in `cast/integration/auffusion.py` and `cast/pipeline.py` (default `None` = previous behaviour).
- **Handoff report:** `docs/CAST_CFG_실험_인수인계_보고서.docx` (committed, 350 KB, 4 embedded figures). Text extracted to the scratchpad for reading; its content agrees with the branch's `CLAUDE.md` section and with the tables below.

## CFG Sweep (`outputs/cfg_sweep_v1/`)

| Item | Expected | Found |
|---|---|---|
| `audio_index.csv` rows | 1080 | **1080**, 24 prompts × 3 seeds × 3 CFG × 5 arms, exactly 24 rows per (cfg, arm, seed) cell; 0 duplicate paths, 0 duplicate SHA-256 |
| prompts | 24 (A01–D06) | 24; prompt texts identical to `experiments/prompts/cast_diagnostic_v1.json` (0 mismatches) |
| seeds | 42, 123, 2026 | 42, 123, 2026 |
| CFG | 7.5, 3.0, 1.5 | 7.5, 3.0, 1.5 |
| conditions | Dense, Sharp β=1.5, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2 | arms `dense`, `sharp_b1.5`, `entmax_l0.5`, `entmax_l1`, `entmax_l2`; manifest arm definitions: dense = entmax_mix λ=0, sharp = `sharp_softmax` β=1.5, entmax λ 0.5 / 1 / 2 |
| generation settings | 100-step PNDM, 256×1024, fp16, α=1.5, deterministic self-attn, attn2 only | manifest: steps 100, 256×1024, float16 + autocast, alpha 1.5, `deterministic_self_attn: true`, `apply_to_uncond: true`, checkpoint `auffusion/auffusion`; versions torch 2.0.1+cu118, diffusers 0.18.2, transformers 4.31.0, entmax 1.3; GPU "NVIDIA GeForce RTX 2080" |
| generation summary | 1080 completed | `completed_slots 1080 / total 1080`, `failures: []` |
| initial latents | per-seed shared | per-seed SHA-256 identical to the A5000 Diagnostic v1 manifests (e.g. seed 2026 `5ea8d1c0…`), so the RTX 2080 sweep starts from the same noise; only the denoising output differs by hardware |
| prompt-file hash | — | manifest records `6fab2188…`, local file is `c89ac135…`; the difference is CRLF line endings on the Windows checkout (CRLF variant of the local file hashes to `6fab2188…`). Content identical. |
| **WAV files on this machine** | 1080 | **0**. The wavs are gitignored and were never on this machine. The handoff names a separate package `cast_cfg_sweep_aqa_package.zip` (285 MB, SHA-256 `a46b69fd6a46022be17b3df169fcaaa728c5373f229b2121f21c803b98600bf1`) to be received from 김원석 outside git. A filesystem search (`/home`, `/tmp`, `/mnt`, `/media`, `/data`, `/opt`) found no copy. |
| duplicate / corrupt files | — | not assessable until the package arrives; `experiments/verify_sweep_audio.py` (new, see HANDOFF_CHANGES.md) checks count, per-file SHA-256 against the index, 16 kHz mono PCM_16 |

**Missing samples: 1080 / 1080 (all).** Per the task rules nothing was regenerated: regenerating on the A5000 would produce different audio than the RTX 2080 files that CLAP and PQ were scored on, and mixing them inside one paired comparison is prohibited.

## Existing Metrics

- **CLAP:** `clap_scores.csv`, 1080 rows (columns `audio_path, audio_sha256, prompt_id, seed, cfg, arm, clap`). Analysis present (`analysis/delta_table.csv`, 24 rows; `analysis_summary.json`). Status **COMPLETE**.
- **AudioBox PQ:** `audiobox/sample_scores.csv`, 1080 rows (CE, CU, PC, PQ), `audiobox/validation.json` (torch 2.4.1+cu124, AesPredictor, 16 kHz input). Status **COMPLETE**.
- **AQAScore:** no `aqascore/` directory, no AQAScore column anywhere in `outputs/cfg_sweep_v1/`. Status **NOT STARTED – BLOCKED** (no audio on this machine). The teammate's `analyze_cfg_sweep_v1.py` already reads `aqascore/sample_scores.csv` column `aqascore_full` keyed by `audio_path`, which is exactly the schema written by the existing CAST evaluator `experiments/eval_aqascore.py` (Qwen2.5-Omni-7B thinker, bf16, `inference_mode`, probabilistic Yes/No, 3-repeat determinism check, matching-vs-unrelated sanity check). The sweep index carries every column that evaluator needs (`audio_sha256, prompt_id, category, prompt, seed, audio_path, provenances`); `cfg` and `arm` are re-attached by the analysis script through `audio_path`. The evaluation environment (`cast-eval`: Python 3.10.22, torch 2.4.1+cu124, transformers 4.57.1) and the thinker-only model directory `~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker` (shards 1–4 resolve, 4 × 4.7 GB) are intact.

## Prompt-v2 (`cast_prompts_v2.json`, `outputs/listen_prompts_v2/`)

- **Prompt file:** repo root `cast_prompts_v2.json`, 128 prompts, 16 categories × 8, `family` clear 64 / ambiguous 64, `core` 64. Pre-declared hypothesis stored in the file ("PLADIS gain larger for ambiguous prompts…"). Not edited.
- **Frozen status:** committed on the teammate branch (1fd7fb3); the local untracked copy was byte-identical. Treated as read-only.
- **Experimental subset actually used:** `audio_index_all.csv` has 208 rows over exactly 16 prompt IDs: clear SG01, MU01, AT01, TM01, VO01, EN01, MS01, NA01, MC01, HL01, CU01, FX01 (12) and ambiguous AB01, IM01, UN01, LX01 (4) — matches the handoff. Arms `vanilla` (CFG 3.0, seed 42 only), `dense_l0`, `pladis_l2`; CFG 3.0 and 1.5; seeds 42/123/2026 (CFG 3.0 seed 42 has 48 rows because it includes the 16 vanilla samples; the other five cells have 32). CLAP per set, AudioBox for all 208 (`audiobox_all/`). Analysis outputs (`analysis/delta_summary.csv`, `per_prompt_delta.csv`, `diagnostics.json`) reproduce the handoff tables (e.g. CLAP CFG 3.0 all Δ −0.044 [−0.086, −0.001]; ambiguous−clear +0.056 [−0.003, +0.113]).
- The 12-vs-4 split is confounded with category/domain (all domain-axis "01" prompts fall in the clear group); not a balanced benchmark. No wavs on this machine (gitignored). Nothing rerun.

## Layer Ablation (`outputs/layer_ablation_v1/`)

- `audio_index.csv` 144 rows: arms `pladis_high` (72) and `pladis_low` (72), CFG 3.0 only, 24 prompts × 3 seeds. Dense and full-16-layer PLADIS are reused from the sweep (same CFG, same seeds).
- Manifest groups: high = `down_blocks.0` (2) + `up_blocks.3` (3) = 5 attn2 modules; low = `down_blocks.2` (2) + `mid_block` (1) + `up_blocks.1` (3) = 6 modules. λ = 2.0. Recorded validation flags all `true`: all-λ0 sample equals sweep Dense (SHA-256), all-λ2 sample equals sweep `entmax_l2`, census high 5 / low 6 with non-selected modules on the λ=0 fp32 path, attn1 untouched, processors restored.
- CLAP 144 rows, AudioBox 144 rows, `analysis/delta_table.csv` and `analysis_summary.json` present and consistent with the handoff (CLAP Δ all −0.102, high −0.055, low −0.036; high−low −0.019 [−0.048, +0.011]). `generation_summary`: 144/144, no failures. Status **COMPLETE (exploratory, CFG 3.0 only, L1 not run)**. Nothing rerun.

## Remaining Required Work

1. **Obtain the audio package** from the teammate and place it on this machine (not in the repo tree, e.g. `~/handoff/cast_cfg_sweep_aqa_package.zip`). Then, from the repo root:
   ```
   python experiments/verify_sweep_audio.py --zip ~/handoff/cast_cfg_sweep_aqa_package.zip \
       --expected-zip-sha a46b69fd6a46022be17b3df169fcaaa728c5373f229b2121f21c803b98600bf1 --extract
   ```
   This extracts only the 1,080 index-listed wavs and must end with `verified 1080/1080 … PASS`. Do not unzip the package manually at the repo root: it also contains copies of `eval_aqascore.py`, the prompt file and `audio_index.csv` that would overwrite tracked files (CRLF versions).
2. **Disk space first:** only **683 MB** is free on `/`. The zip (285 MB) plus the extracted wavs (≈350 MB) does not fit safely, and the Qwen run needs headroom. At least ~2 GB must be freed before step 1. Nothing was deleted in this session; candidates belong to other projects (`~/.cache/uv` 37 GB, `~/.cache/huggingface` 70 GB shared with other models, `~/.cache/datalab` 3.3 GB, `~/miniconda3/pkgs` 1.8 GB cache).
3. **AQAScore** (`cast-eval` env, A5000):
   ```
   ~/miniconda3/envs/cast-eval/bin/python experiments/eval_aqascore.py \
       --model-dir ~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker \
       --index outputs/cfg_sweep_v1/audio_index.csv --out outputs/cfg_sweep_v1/aqascore
   ```
   The script stops by itself unless the 3-repeat score is bitwise identical and finite (`validation.json`). Expected runtime on the A5000 for 1,080 samples ≈ the 504-sample multimetric run scaled ×2.1.
4. **Re-run the analysis**: `python experiments/analyze_cfg_sweep_v1.py` (either env) → `analysis/delta_table.csv` gains the `aqa` rows, `fig_cfg_delta_grid.*` gains the ΔAQAScore column and the fixed-name copy `fig_cfg_delta_grid_3metrics.{png,pdf}` is written (3-metric path dry-run tested on fake data in the scratchpad only).
5. **Fill the AQAScore-dependent sections** of `CAST_ADVISOR_FEEDBACK_COMPLETION_REPORT.md` and `PAPER_RESULT_SNIPPETS_KO.md` (marked `[PENDING AQAScore]`).

Not required / not done on purpose: no regeneration, no CFG 1.0–1.2, no prompt-v2 or layer-ablation extension, no AQAScore for the v2 (208) or layer-ablation (144) sets (that would also need a `--prompt-file` option in `eval_aqascore.py`, not implemented).
