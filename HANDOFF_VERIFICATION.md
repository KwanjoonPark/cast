# CAST Advisor-Feedback Handoff Verification

Date: 2026-10-25 (initial verification and AQAScore completion on the same day). Machine: RTX A5000 24 GB (`reallab`), repo `/home/reallab/parkkwanjoon/CAST`.
Everything below was checked against the repository contents, not inferred from file names or from the handoff text.

## Git State

- **Remote branch used:** `origin/wonseok/advisor-feedback` (the only remote branch besides `origin/main`; name matches the handoff).
- **Teammate branch tip / base commit of this continuation:** `1fd7fb31c4fef131eb1c269c4e6f86bd409af267` ("Add cast_prompts_v2.json and sync the docs with the pushed state").
- **Teammate branch base:** `830bd52` = `origin/main` HEAD. The branch is a linear 4-commit extension of `main` (`0c171b5` code, `ee21acc` result tables/figures, `3b91728` CLAUDE.md + handoff docx, `1fd7fb3` prompt file); no divergence, no rebase needed.
- **Continuation branch:** `kwanjoon/advisor-feedback-continuation`, created locally from `origin/wonseok/advisor-feedback`; one local commit `20ebfcd` (verification docs, audio verifier, figure export) on top. Not pushed.
- **Working tree at start of the first session:** one untracked file, `cast_prompts_v2.json` (SHA-256 prefix `a0a93135…`), byte-identical to the file committed on the teammate branch. It was copied to the session scratchpad and moved aside so that `git checkout` could place the identical tracked copy; nothing was stashed, reset or discarded. Submodules unchanged (Auffusion f44233d, PLADIS 248b9d1, ELSA d2eb34a).
- **Working tree at start of the second session:** clean except the untracked handover zip at the repo root (now listed in `.gitignore`).
- **Tests:** `pytest cast/tests` → 13 passed (same count as the handoff). The branch's only core-code change is the optional `only=` / `only_modules=` argument in `cast/integration/auffusion.py` and `cast/pipeline.py` (default `None` = previous behaviour). CAST core code was not modified in either session.
- **Handoff report:** `docs/CAST_CFG_실험_인수인계_보고서.docx` (committed, 350 KB, 4 embedded figures). Its content agrees with the branch's `CLAUDE.md` section and with the tables below.

## CFG Sweep (`outputs/cfg_sweep_v1/`)

| Item | Expected | Found |
|---|---|---|
| `audio_index.csv` rows | 1080 | **1080**, 24 prompts × 3 seeds × 3 CFG × 5 arms = 45 cells of exactly 24 rows; 0 duplicate rows / paths / SHA-256 / (prompt, seed, cfg, arm) keys |
| prompts | 24 (A01–D06) | 24; prompt texts identical to `experiments/prompts/cast_diagnostic_v1.json` (0 mismatches) |
| seeds | 42, 123, 2026 | 42, 123, 2026 |
| CFG | 7.5, 3.0, 1.5 | 7.5, 3.0, 1.5 |
| conditions | Dense, Sharp β=1.5, Entmax λ=0.5, Entmax λ=1, PLADIS λ=2 | arms `dense`, `sharp_b1.5`, `entmax_l0.5`, `entmax_l1`, `entmax_l2`; manifest arm definitions: dense = entmax_mix λ=0, sharp = `sharp_softmax` β=1.5, entmax λ 0.5 / 1 / 2 |
| generation settings | 100-step PNDM, 256×1024, fp16, α=1.5, deterministic self-attn, attn2 only | manifest: steps 100, 256×1024, float16 + autocast, alpha 1.5, `deterministic_self_attn: true`, `apply_to_uncond: true`, checkpoint `auffusion/auffusion`; versions torch 2.0.1+cu118, diffusers 0.18.2, transformers 4.31.0, entmax 1.3; GPU "NVIDIA GeForce RTX 2080" |
| generation summary | 1080 completed | `completed_slots 1080 / total 1080`, `failures: []` |
| initial latents | per-seed shared | per-seed SHA-256 identical to the A5000 Diagnostic v1 manifests (e.g. seed 2026 `5ea8d1c0…`), so the RTX 2080 sweep starts from the same noise; only the denoising output differs by hardware |
| prompt-file hash | — | manifest records `6fab2188…`, local file is `c89ac135…`; the difference is CRLF line endings on the Windows checkout (CRLF variant of the local file hashes to `6fab2188…`). Content identical. |
| handover package | `cast_cfg_sweep_aqa_package.zip`, SHA-256 `a46b69fd6a46022be17b3df169fcaaa728c5373f229b2121f21c803b98600bf1` | present at the repo root (placed by the user), 284,840,438 bytes, **SHA-256 matches**; 1,084 members = 1,080 wav (345.6 MB uncompressed) + 4 non-wav (README_AQAScore.txt, copies of `audio_index.csv`, `eval_aqascore.py`, the prompt file) |
| **WAV files on this machine** | 1080 | **1080 / 1080**, extracted by `experiments/verify_sweep_audio.py --extract` (index-listed wav members only; the 4 non-wav members were not extracted; `git status` showed no tracked file modified). Every file's SHA-256 equals the index `audio_sha256`; all 16 kHz mono PCM_16, 160,000 samples (10.0 s). Report: `outputs/cfg_sweep_v1/audio_verification.json` (`verdict: PASS`). |
| duplicate / corrupt files | — | none: 0 missing, 0 SHA mismatches, 0 format anomalies |

**Missing samples: 0 / 1080.** Nothing was regenerated; all metrics below refer to the same RTX 2080 files.

## Existing Metrics

- **CLAP:** `clap_scores.csv`, 1080 rows (columns `audio_path, audio_sha256, prompt_id, seed, cfg, arm, clap`). Status **COMPLETE**.
- **AudioBox PQ:** `audiobox/sample_scores.csv`, 1080 rows (CE, CU, PC, PQ), `audiobox/validation.json` (torch 2.4.1+cu124, AesPredictor, 16 kHz input). Status **COMPLETE**.
- **AQAScore:** Status **COMPLETE (this session)**. `experiments/eval_aqascore.py` (unchanged: Qwen2.5-Omni-7B thinker only, bf16, `model.eval()` + `torch.inference_mode()`, paper Table-7 system prompt, question "Does this audio contain the sound events described by the text: {t}? Please answer yes or no.", AQAScore = softmax([log p(Yes), log p(No)])[0] with single tokens `Yes`=9454 / `No`=2753, 16 kHz audio unresampled). Environment: `cast-eval` Python 3.10.22, torch 2.4.1+cu124, transformers 4.57.1, CUDA 12.4, RTX A5000, model dir `~/.cache/cast_eval/Qwen2.5-Omni-7B-thinker` (shards 1–4), peak 16.8 GB VRAM. Smoke test (`--validate-only`) on `cfg1.5/DIFFUSE_SINGLE/B01/seed_123/dense`: three repeats 0.798187 / 0.798187 / 0.798187 (max |Δ| 0.0), logP_yes −2.4754, logP_no −3.8504, finite, in range; matching prompt 0.798 vs unrelated prompt ("A dog barking loudly.") 0.011. Full run 1,080 / 1,080 in 7.7 min; `aqascore/sample_scores.csv` 1080 rows, 0 NaN / Inf, range [0.033, 0.987], 0 duplicates; joined 1080/1080 to the index on `audio_path` with equal `audio_sha256`, `prompt`, `seed`, `prompt_id`. `aqascore/event_scores.csv` 1,620 rows (diagnostic), `run_info.json`, `validation.json`.

## Prompt-v2 (`cast_prompts_v2.json`, `outputs/listen_prompts_v2/`)

- **Prompt file:** repo root `cast_prompts_v2.json`, 128 prompts, 16 categories × 8, `family` clear 64 / ambiguous 64, `core` 64. Pre-declared hypothesis stored in the file. Not edited.
- **Frozen status:** committed on the teammate branch (1fd7fb3); the local untracked copy was byte-identical. Treated as read-only.
- **Experimental subset actually used:** `audio_index_all.csv` has 208 rows over exactly 16 prompt IDs: clear SG01, MU01, AT01, TM01, VO01, EN01, MS01, NA01, MC01, HL01, CU01, FX01 (12) and ambiguous AB01, IM01, UN01, LX01 (4) — matches the handoff. Arms `vanilla` (CFG 3.0, seed 42 only), `dense_l0`, `pladis_l2`; CFG 3.0 and 1.5; seeds 42/123/2026. CLAP per set, AudioBox for all 208. Analysis outputs reproduce the handoff tables. The 12-vs-4 split is confounded with category/domain. No wavs on this machine; not rerun; **no AQAScore** (not in scope).

## Layer Ablation (`outputs/layer_ablation_v1/`)

- `audio_index.csv` 144 rows: `pladis_high` (72) and `pladis_low` (72), CFG 3.0 only, 24 prompts × 3 seeds; Dense and full-16-layer PLADIS reused from the sweep. Manifest validation flags all `true` (all-λ0 ≡ sweep Dense, all-λ2 ≡ sweep `entmax_l2`, census 5/6, attn1 untouched, processors restored). CLAP and AudioBox 144 rows; analysis consistent with the handoff. Status **COMPLETE (exploratory, CFG 3.0 only, L1 not run)**; not rerun; **no AQAScore** (not in scope).

## Analysis and Figure

- `experiments/analyze_cfg_sweep_v1.py` rerun with CLAP + AQAScore + PQ; protocol unchanged (prompt unit, seeds averaged, paired Δ vs Dense at the same CFG, 10,000 bootstrap resamples, rng seed 20261024). `delta_table.csv` gained exactly the 12 `aqa` rows; all previously committed CLAP/PQ rows are unchanged (verified by diff).
- `outputs/cfg_sweep_v1/analysis/fig_cfg_delta_grid_3metrics.{png,pdf}` produced: rows CFG 7.5 / 3.0 / 1.5, columns ΔCLAP, ΔAQAScore, ΔPQ, four methods in fixed order, dashed zero = Dense, 95 % bootstrap CI bars, white background. Rendering inspected: no overlaps or clipping.

## Remaining Required Work

None for the CFG experiment. Open items outside this task's scope (not started on purpose): manuscript revision itself; AQAScore for the v2 (208) and layer-ablation (144) sets (would need a `--prompt-file` option in `eval_aqascore.py`); CFG ≈ 1 experiment (judged unnecessary for the paper's claims, see the completion report §15); pushing the branch (user decision).
