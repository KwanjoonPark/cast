# CAST First Milestone Implementation Report

Date: 2026-10-20. Project root `/home/reallab/parkkwanjoon/CAST`. Companion document: `CAST_ANALYSIS_REPORT.md` (plan). Scope: environment, golden reference, CAST processor, λ=0 parity, λ=1 / λ=2 validation, one controlled three-arm generation. Nothing beyond that was run.

## 1. Executive Summary

| Item | Status |
|---|---|
| Environment built (`cast` conda env, pinned) | **YES** — Python 3.10.20, torch 2.0.1+cu118, diffusers 0.18.2, transformers 4.31.0, huggingface_hub 0.16.4, entmax 1.3; GPU RTX A5000 usable |
| Original Auffusion inference works | **YES** — `outputs/golden/` (via `cast.pipeline`, stock `AttnProcessor2_0`) and an independent replay of the untouched `third_party/Auffusion/inference.py` (`outputs/golden_official/`) |
| CAST injected on exactly the 16 `attn2` modules, `attn1` untouched, originals restored | **YES** — asserted at runtime and in tests |
| λ=0 parity | **PASS** — per-module: bitwise equal to diffusers' `AttnProcessor` in fp32, 1 fp16 ulp (1.2e-4) vs `AttnProcessor2_0`/SDPA in fp16; full U-Net (real weights): rel-L2 2e-5 … 1.8e-4 (fp32), 2e-4 … 5.4e-4 (fp16+autocast) = the stock kernel repeat floor; full 100-step pipeline: 0.66 % rel-L2 on the final latent, which is *smaller* than the 0.95 % produced by changing only the stock self-attention kernel |
| λ=1 (sparse) and λ=2 (PLADIS) outputs valid | **YES** — finite everywhere; final-latent rel-L2 vs λ=0 of 40 % and 84 % (60–130× the floor); mean entmax support on conditional rows ≈ 2.4 of 77 keys |
| Same initial noise across arms | **YES** — pre-sampled latent is bitwise identical to the pipeline's internal seed-42 draw; passed as `latents=` to all three arms |
| Ready for pilot experiments | **YES, PROCEED TO PILOT**, with the caveats in §11 (kernel non-determinism of the stock self-attention; use `deterministic_self_attn=true` for controlled comparisons) |

## 2. Final Environment

Conda env `cast` (`experiments/env/setup_env.sh`, `experiments/env/requirements-cast.txt`). Exact versions after a successful model load + inference:

| Package | Version | Notes |
|---|---|---|
| Python | 3.10.20 | |
| torch | 2.0.1+cu118 (CUDA runtime 11.8, cuDNN 8700) | driver 570.133.07 / CUDA 12.8 accepts cu118 wheels; the base env's torch cu130 cannot run on this driver |
| diffusers | 0.18.2 | required by `auffusion_pipeline.py` (imports `randn_tensor` from `diffusers.utils`, removed later) |
| transformers | **4.31.0** (pinned exactly) | compatible with huggingface_hub 0.16.4 and the checkpoint's CLIP config (written by 4.31.0.dev0) |
| tokenizers | 0.13.3 | as in Auffusion requirements |
| huggingface_hub | 0.16.4 | diffusers 0.18.2 needs `cached_download` |
| entmax | 1.3 | pure Python |
| accelerate | 0.21.0 | |
| numpy / scipy | 1.23.5 / 1.10.1 | librosa 0.9.2 compatible |
| librosa / soundfile | 0.9.2 / 0.12.1 | librosa needs `setuptools==69.5.1` (`pkg_resources`) on py3.10 |
| datasets / pyarrow | 2.14.5 / **14.0.2** | only for replaying `inference.py`; pyarrow had to be pinned (`PyExtensionType` removed in newer pyarrow) |
| matplotlib / pyyaml / pytest | 3.7.2 / 6.0.1 / 7.4.0 | |
| xformers | **not installed** | stock backend is `AttnProcessor2_0` (PyTorch SDPA) |
| GPU | NVIDIA RTX A5000 24 GB | peak usage 2.8 GB (default kernels) / 3.1 GB (math SDPA) |

Checkpoint: `auffusion/auffusion`, HF snapshot `211aeb853fe565a306bb164d1227505e70558529`, 4.32 GB, cached in `~/.cache/huggingface/hub`. Download used `snapshot_download(..., token=False)` so the stale token in `~/.cache/huggingface/token` is never sent; that file was not modified.

## 3. Files Added or Modified

| File | Purpose | New/Modified |
|---|---|---|
| `CLAUDE.md` | `ours/` → `cast/` as the implementation directory | Modified (one line) |
| `cast/__init__.py` | package | New |
| `cast/config.py` | `CastConfig` dataclass; `MODES` = {golden, cast_vanilla, cast_sparse, cast_pladis} → (use_cast, α, λ); `deterministic_self_attn` flag | New |
| `cast/attention/sparse_ops.py` | `dense_softmax`, `alpha_entmax` (1.0→softmax, 1.5→`entmax15`, 2.0→`sparsemax`, else `entmax_bisect`), `mix_attention_weights` | New |
| `cast/attention/cast_processor.py` | `CastCrossAttnProcessor` (diffusers-0.18.2 processor protocol; explicit fp32 attention; no SDPA/xformers) | New |
| `cast/integration/auffusion.py` | `import_auffusion`, `resolve_checkpoint` (token=False), `load_pipeline`, `select_cross_attention`, `apply_cast`, `restore_processors`, `cast_installed` | New |
| `cast/pipeline.py` | `sample_initial_latents`, `run_one` (single entry point for golden + 3 arms; saves audio/spectrogram/latents/manifest) | New |
| `cast/tests/test_sparse_ops.py`, `cast/tests/test_processor.py`, `cast/tests/fixtures/unet_config.json` | unit tests (10) | New |
| `experiments/configs/first_milestone.yaml` | milestone configuration | New |
| `experiments/env/requirements-cast.txt`, `experiments/env/setup_env.sh` | pinned environment | New |
| `experiments/run_first_milestone.py`, `experiments/run_first_milestone.sh` | driver: golden + latent-control checks + 3 arms + summary | New |
| `experiments/validate_parity.py` | checkpoint-level U-Net parity (fp32, fp16+autocast) | New |
| `experiments/run_official_inference.sh` | replays untouched `third_party/Auffusion/inference.py` and compares with the golden | New |
| `outputs/**` | generated artefacts (see §4, §8) and logs `outputs/_*.log` | New |

**`third_party/PLADIS/**` and `third_party/Auffusion/**` were not modified.** Verified with `find third_party -newermt "2026-10-20 16:36"`: the only entry was `third_party/Auffusion/__pycache__/` (bytecode cache created by importing the pipeline), which was removed; no source file changed. PLADIS code is never imported at runtime.

## 4. Original Auffusion Golden Run

Configuration (identical for every run in this report): checkpoint `auffusion/auffusion`; prompt "A dog barking while rain falls in the background."; seed 42 (`torch.Generator("cuda").manual_seed(42)`); 100 PNDM steps; guidance 7.5; 256×1024; fp16 weights (`AuffusionPipeline.from_pretrained` default) + `torch.autocast("cuda")` (as `inference.py`); `negative_prompt=None`; no xformers; stock processors = `AttnProcessor2_0` on all 32 attention modules.

| Output | Path | Notes |
|---|---|---|
| Golden (default SDPA kernel) | `outputs/golden/{audio.wav, spectrogram.png, final_latents.pt, manifest.json}` | runtime 6.2 s, peak 2.79 GB, all finite |
| Golden (math SDPA kernel, bitwise-repeatable) | `outputs/golden_deterministic/…` | runtime 7.4 s, peak 3.11 GB |
| Official script replay | `outputs/golden_official/audios/official.wav`, `compare.json` | `third_party/Auffusion/inference.py` run unmodified on a 1-row json; differs from `outputs/golden/audio.wav` by mean 284 int16 units — the same magnitude as two consecutive runs of the stock pipeline itself (287), see §6.3 |

`manifest.json` records checkpoint, prompt, seed, steps, guidance, size, dtype, autocast, torch/CUDA/diffusers/transformers/hub/entmax versions, processor classes for attn1 and attn2, scheduler, git commit (`null`: the project is not a git repository), runtime, peak memory, finiteness, final-latent norm and SHA-256, audio SHA-256.

## 5. CAST Implementation

**Attention selection** — `cast/integration/auffusion.py: select_cross_attention(unet)`. Requires `len(unet.attn_processors) == 32`, then for each processor key demands agreement of (1) name ends with `attn2.processor` and (2) `to_k.in_features == unet.config.cross_attention_dim (768)` and `!= to_q.in_features`; raises on disagreement; requires exactly 16 selected. Runtime guard inside the processor: `encoder_hidden_states is not None` (raises otherwise), `attention_mask is None`, no `spatial_norm/group_norm/norm_cross`.

**Installation / restoration** — `apply_cast(unet, alpha, lam, …)` stores the original processor object of each selected module and calls `module.set_processor(CastCrossAttnProcessor(...))` (per-module; `attn1` never touched). `restore_processors` puts the saved objects back; `cast_installed` is the context-manager form; `run_one` restores in a `finally`. Tests assert `attn1` processors are the *same objects* before/after and that all 32 originals are restored.

**Q/K/V handling** — `CastCrossAttnProcessor.__call__` copies the diffusers-0.18.2 `AttnProcessor2_0` prologue: `query = attn.to_q(hidden_states)`, `key = attn.to_k(encoder_hidden_states)`, `value = attn.to_v(encoder_hidden_states)` in the model dtype, reshaped to `(B, H, N, d)` / `(B, H, S, d)` with `H = attn.heads = 8`, `d = inner_dim / 8`.

**fp32 normalization** — inside `torch.autocast(device_type, enabled=False)` (autocast would otherwise re-cast the matmul to fp16): `logits = matmul(query.float(), key.float().T) * attn.scale` (`attn.scale = d**-0.5`), `W_dense = softmax(logits)`, `W_sparse = alpha_entmax(logits, α)` from the **same logits**, `W = mix_attention_weights(W_dense, W_sparse, λ) = W_dense + λ (W_sparse − W_dense)` (λ=0 returns `W_dense` itself, λ=1 returns `W_sparse` itself), `torch.isfinite(W).all()` check (raises `FloatingPointError` otherwise), then `out = matmul(W.to(value.dtype), value)`; epilogue `reshape → .to(query.dtype) → attn.to_out[0] → attn.to_out[1]` unchanged.

**Entmax** — `cast/attention/sparse_ops.py: alpha_entmax`: α=1.5 uses the exact sort-based `entmax.entmax15` (what PLADIS uses); α=1.0 is dispatched to `torch.softmax` explicitly (`entmax_bisect(alpha=1.0)` is not softmax, verified); α=2.0 → `sparsemax`; other α → `entmax_bisect`. Only α=1.5 was used in runs.

**λ mixing** — `mix_attention_weights`; terminology "attention weights", no clamping (negative entries at λ=2 are expected and present).

**CFG behavior** — the processor is applied to the whole `[uncond, cond]` CFG batch (`apply_to_uncond=True`, PLADIS-faithful). The uncond embedding is `zeros_like(prompt_embeds)` and `to_k/to_v` have no bias, so on those rows `K=V=0`, all logits are 0, softmax = entmax = uniform, and the output is `to_out.bias` whatever λ is. Verified: per-module uncond row λ=2 vs λ=0 differs by exactly 0.0 (fp32 and fp16); full U-Net with real weights, uncond row λ=2 vs λ=0 differs by ≤ the stock repeat floor (fp32: ≤6.5e-4 with cuDNN TF32 on, 4.2e-6 with TF32 off).

**Lightweight statistic** — the processor accumulates the mean number of non-zero entmax entries per query row (all rows, and the conditional half separately); it is stored per layer in each manifest.

**Modes** — one processor class for all arms: `cast_vanilla` (α 1.5, λ 0), `cast_sparse` (α 1.5, λ 1), `cast_pladis` (α 1.5, λ 2). `golden` = no swap. All arms go through `cast.pipeline.run_one` with the same `CastConfig`, only `mode` differing.

Math ↔ code map (PLADIS Eq. 12/13/15 → `cast_processor.py`): `Softmax(QKᵀ/√d)` → `dense_softmax(logits)`; `1.5-Entmax(QKᵀ/√d)` → `alpha_entmax(logits, 1.5)`; `At + λ(At_α − At)` → `mix_attention_weights(...)` then `@ V` (identical by linearity of `(·)V`, and identical to PLADIS's `λ·S + (1−λ)·D` form).

## 6. λ=0 Parity Results

All numbers are in `outputs/validation/`.

### 6.1 Per-module (random-init SD1.5 U-Net from the checkpoint config, 4 attn2 modules at N = 4096/1024/256/64; `unit_parity_float{32,16}.json`)

| dtype | CAST λ=0 vs `AttnProcessor2_0` (SDPA) max / mean abs | CAST λ=0 vs `AttnProcessor` (baddbmm+softmax) max / mean | SDPA vs baddbmm (stock vs stock) max | output std |
|---|---|---|---|---|
| fp32 | 1e-7 – 2e-7 / ≤1e-8 | **0.0 / 0.0 (bitwise)** | 1e-7 – 2e-7 | 0.031 – 0.042 |
| fp16 | 1.22e-4 (= 1 fp16 ulp) / 5–6e-6 | 1.22e-4 / 5e-6 | 1.22e-4 | 0.031 – 0.042 |

Also per module: uncond row λ=2 vs λ=0 = 0.0 exactly; linearity `out(λ=2) = 2·out(λ=1) − out(λ=0)` holds to 1e-6 (fp32).

### 6.2 Full U-Net, one forward

Random-init U-Net, fp32, cuDNN TF32 off (`unit_full_unet_fp32.json`): stock repeat floor max 4.5e-6; CAST λ=0 vs stock max **4.8e-6**; uncond row λ=2 vs stock max 4.2e-6; cond row λ=2 vs stock max 0.38 (ref std 0.33); each attn2 called exactly once per forward (16/16).

Real checkpoint weights, real prompt embeddings, noised seed-42 latent, timesteps 991 / 501 / 1 (`unet_parity.json`, `experiments/validate_parity.py`; cuDNN TF32 at its default = on):

| Setting | t | stock repeat floor (max) | stock vs CAST λ=0 max / rel-L2 | λ=1 vs λ=0 rel-L2 | λ=2 vs λ=0 rel-L2 | uncond row λ=2 vs λ=0 max |
|---|---|---|---|---|---|---|
| fp32 | 991 | 1.5e-4 | 1.7e-4 / 2.3e-5 | 4.1e-3 | 1.0e-2 | 1.5e-4 |
| fp32 | 501 | 5.9e-4 | 6.2e-4 / 4.0e-5 | 1.5e-2 | 2.9e-2 | 3.2e-4 |
| fp32 | 1 | 6.0e-4 | 6.9e-4 / 1.8e-4 | 2.7e-2 | 5.8e-2 | 6.5e-4 |
| fp16+autocast | 991 | 2.0e-3 | 2.0e-3 / 2.1e-4 | 4.1e-3 | 1.0e-2 | 2.0e-3 |
| fp16+autocast | 501 | 3.9e-3 | 2.4e-3 / 2.4e-4 | 1.5e-2 | 2.9e-2 | 2.0e-3 |
| fp16+autocast | 1 | 3.9e-3 | 3.9e-3 / 5.4e-4 | 2.7e-2 | 5.8e-2 | 2.0e-3 |

CAST λ=0 differs from the stock output by no more than the stock output differs from itself on a repeated call; the λ effects are 10–100× larger per forward.

### 6.3 Full 100-step pipeline (final latents, 1×4×32×128; audio int16)

| Comparison | latent max abs | latent rel-L2 | audio SNR (dB, vs second item) | Source |
|---|---|---|---|---|
| Stock golden run twice (default SDPA kernel) | 0.086 | 0.80 % | 6.9 | `outputs/first_milestone/summary.json` |
| Official `inference.py` vs stock golden | – | – | mean abs 284 int16 (≈ repeat noise 287) | `outputs/golden_official/compare.json` |
| Stock golden, default kernel vs math kernel (**stock vs stock; only the self-attention kernel changed**) | 0.089 | **0.95 %** | 6.1 | `outputs/validation/pipeline_floor_controls.json` |
| **CAST λ=0 vs stock golden** (default kernels) | 0.053 | **0.55 %** | 10.6 | `outputs/first_milestone/summary.json` |
| **CAST λ=0 vs stock golden** (math kernel, both bitwise-repeatable) | 0.070 | **0.66 %** | 7.4 | `outputs/first_milestone_deterministic/summary.json` |
| CAST λ=0 repeated (math kernel) | 0 | 0 (bitwise identical) | ∞ | this report, §10 |
| CAST λ=1 vs CAST λ=0 | 1.66 | **39.9 %** | −3.0 | both summaries |
| CAST λ=2 vs CAST λ=0 | 7.88 | **83.8 %** | −7.5 | both summaries |

Interpretation: after 100 denoising steps, *any* fp16-rounding-level perturbation of the stock pipeline (a different self-attention kernel, or kernel non-determinism) moves the final latent by ≈ 0.8–1.0 % relative L2 and yields audio that differs at ≈ 6–7 dB SNR; the CAST λ=0 path sits at or below that floor (0.55–0.66 %), while λ=1 / λ=2 are 60–130× above it. The λ=0 discrepancy is therefore floating-point/kernel level, not semantic. Spectrograms of golden and `cast_vanilla` are visually indistinguishable.

## 7. Sparse and PLADIS Validation

Unit tests (`cast/tests/test_sparse_ops.py`, all pass): α=1.0 dispatch equals `torch.softmax` bitwise; `alpha_entmax(·,1.5)` equals `entmax15` bitwise and matches `entmax_bisect(α=1.5)` to <1e-5; entmax15 rows sum to 1 within 1e-5, are non-negative, and >50 % of the 77 entries are exactly 0; `mix(λ=0) is W_dense`, `mix(λ=1) is W_sparse`, `mix(λ=2) == 2·W_sparse − W_dense` (atol 1e-6); rows of every mixed map sum to 1 within 1e-4; λ=2 contains negative weights; all finite; zero logits give identical uniform dense/sparse maps.

Processor tests (`test_processor.py`, all pass): λ=1 changes the conditioned row (max 0.35–0.40 vs output std 0.03–0.04), λ=2 output equals `2·out(λ=1) − out(λ=0)` to 1e-6 (fp32) / 6e-4 (fp16), uncond row unchanged exactly; full U-Net λ=2 changes the conditional half (max 0.38) and not the uncond half (≤ floor).

Runtime (`validate_parity.py`, real weights): all λ outputs finite at t = 991 / 501 / 1 in fp32 and fp16+autocast; mean entmax support on the conditional half ≈ 2.4 of 77 keys (min 1.7, max 3.4 across the 16 layers), i.e. the 1.5-entmax map is extremely sparse on this prompt.

Milestone runs: `FloatingPointError` guard never fired; `finite = {audio, spectrogram, final_latents} = True` for every arm.

## 8. First Milestone Outputs

Default kernels (`outputs/first_milestone/`), identical pre-sampled initial latent (SHA-256 recorded in each manifest, equal across arms):

| Mode | α | λ | Audio | Spectrogram | Runtime | Peak mem | ‖z_T‖ | Valid |
|---|---:|---:|---|---|---:|---:|---:|---|
| CAST Vanilla | 1.5 | 0 | `outputs/first_milestone/cast_vanilla/audio.wav` | `…/cast_vanilla/spectrogram.png` | 6.3 s | 2.80 GB | 76.93 | finite ✔ |
| CAST Sparse | 1.5 | 1 | `outputs/first_milestone/cast_sparse/audio.wav` | `…/cast_sparse/spectrogram.png` | 8.1 s | 2.80 GB | 77.74 | finite ✔ |
| CAST PLADIS | 1.5 | 2 | `outputs/first_milestone/cast_pladis/audio.wav` | `…/cast_pladis/spectrogram.png` | 8.3 s | 2.80 GB | 93.05 | finite ✔ |

Bitwise-repeatable variant (`deterministic_self_attn=true`, math SDPA for the untouched self-attention; `outputs/first_milestone_deterministic/`): runtimes 7.5 / 9.5 / 9.6 s, peak 3.11 GB, ‖z_T‖ 76.92 / 77.76 / 93.05, all finite. Each directory also holds `final_latents.pt` (final + initial latents + spectrogram tensor) and `manifest.json`. Golden references: `outputs/golden/`, `outputs/golden_deterministic/`, `outputs/golden_official/`.

Overhead of the explicit fp32 attention: +0.1 s (λ=0) and +2 s (λ=1,2; entmax sort) per 100-step sample, ≈ +30 % over the 6.2 s stock run.

## 9. Controlled-Comparison Verification

| Factor | Same across λ=0/1/2? | Evidence |
|---|---|---|
| Prompt | Yes | config |
| Seed | Yes (42) | manifests |
| Initial noise | **Yes, bitwise** | `sample_initial_latents` equals `pipe.prepare_latents(…, generator seeded 42)` bitwise (`init_noise_sigma=1.0`); the same tensor object is passed as `latents=` to all three arms; SHA-256 equal in the three manifests; under the deterministic kernel, golden-with-presampled-latents == golden-with-internal-seeding bitwise |
| Scheduler | Yes (PNDM, 100 steps, deterministic `step`, no generator) | code (`scheduling_pndm.py`), manifests |
| CFG | Yes (7.5, `[uncond, cond]` batch) | config, manifests |
| Steps | Yes (100) | manifests |
| Checkpoint | Yes (`auffusion/auffusion`, snapshot 211aeb85) | manifests |
| dtype / autocast | Yes (fp16 weights, autocast on) | manifests |
| CAST implementation | Yes — same `CastCrossAttnProcessor` class on the same 16 modules; only `lam` differs | manifests (`attention_processor_class`), tests |
| Self-attention backend | Yes within each set (default kernel in `first_milestone/`, math kernel in `first_milestone_deterministic/`) | manifests (`deterministic_self_attn`) |

Caveat: with the default SDPA kernel the stock self-attention is not bitwise repeatable (single forward max 2e-3 → 0.8 % final-latent rel-L2 after 100 steps). The default-kernel arms therefore contain that nuisance noise; the deterministic set removes it (CAST λ=0 repeats bitwise there). Both sets lead to the same conclusions.

## 10. Problems Encountered

1. **Base environment unusable**: base conda torch is a CUDA 13.0 build; the driver supports ≤ 12.8. A new env with torch 2.0.1+cu118 was built. The torch wheel download took ~45 min (slow mirror).
2. **Stale HF token** (`~/.cache/huggingface/token`) causes 401 on API calls. Solved non-destructively by `snapshot_download(token=False)` in `resolve_checkpoint`; the official `inference.py` replay was given the local snapshot path for the same reason. The token file was not touched.
3. **Two missing pins**: `librosa 0.9.2` needs `pkg_resources` (added `setuptools==69.5.1`); `datasets 2.14.5` breaks on new pyarrow (added `pyarrow==14.0.2`). Both recorded in `requirements-cast.txt`.
4. **cuDNN TF32 non-determinism in fp32**: the first full-U-Net fp32 unit test failed its uncond-row tolerance (max 1e-3) because the *stock* U-Net itself is not repeatable with cuDNN TF32 on Ampere (repeat max 1.05e-3); with TF32 off the repeat floor is 4.5e-6 and CAST λ=0 matches at 4.8e-6. The test now disables cuDNN TF32 and compares to the measured repeat floor.
5. **fp16 SDPA self-attention kernel non-determinism**: cross-attention SDPA (S=77) and cuDNN convs repeat bitwise; the N=4096 self-attention flash/mem-efficient kernel does not (max 2e-3 per forward). This is inside the untouched `attn1` modules. Added `deterministic_self_attn` (math SDPA, +~1.3 s per run) as an optional control; the default remains the stock backend.
6. `AuffusionPipeline` is not a git repo / no `model_index.json` on the Hub — only the custom loader works (as planned); `git_commit` in manifests is `null` because the project is not a git repository.
7. `experiments/validate_parity.py` runs with cuDNN TF32 at its default (on); its fp32 "floor" column therefore reflects TF32 noise, not pure fp32. The TF32-off number is in `unit_full_unet_fp32.json`.

Nothing was hidden: every number above is in `outputs/validation/*.json`, `outputs/*/summary.json`, or `outputs/_*.log`.

## 11. Remaining Risks (before pilot)

| Risk | Note |
|---|---|
| Stock self-attention kernel non-determinism (0.8 % rel-L2 / ≈7 dB audio SNR run-to-run) | Any pilot comparison at the single-sample level must either use `deterministic_self_attn: true` or treat ≈1 % latent rel-L2 / ≈7 dB SNR as the noise floor; aggregate metrics over many prompts/seeds are unaffected in expectation |
| λ=2 pushes ‖z_T‖ from 77 to 93 and strongly changes the spectrogram | Expected direction (extrapolation), but at CFG 7.5 (PLADIS tuned at 3–5) over-sharpening is plausible; no quality claim is made here |
| Extremely sparse maps (≈2.4 of 77 keys on conditional rows) | Consistent with the identical-pad-key structure noted in the analysis; whether this is benign for mixtures ("rain" background) is the scientific question of the pilot, not an implementation issue |
| Project is not under git | Manifests cannot record a commit; recommend `git init` before pilot runs |

## 12. Recommendation for Next Step

**PROCEED TO PILOT.** The environment is pinned and reproducible, the golden path is verified two ways (through `cast.pipeline` and through the untouched `inference.py`), CAST touches exactly the 16 text cross-attention modules and restores them, λ=0 reproduces stock attention to bitwise (fp32, per module) / one-ulp (fp16) / kernel-floor (full pipeline) accuracy, λ=1 and λ=2 behave exactly as the PLADIS algebra prescribes with finite outputs, and the three arms share bitwise-identical initial noise. For pilot runs use `experiments/configs/first_milestone.yaml` as the template with `deterministic_self_attn: true` when single-sample comparisons matter.
