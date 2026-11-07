# No-CFG code-path verification (Auffusion + CAST, diffusers 0.18.2)

Date 2026-10-25. Question: is `guidance_scale = 1.0` a **true no-CFG** inference in the stack this project actually runs
(conditional branch only, no unconditional denoiser prediction, no `eps_uncond + s·(eps_cond − eps_uncond)`)? **Answer: yes (case A).**

## 1. Code inspection

1. **Source file.** `third_party/Auffusion/auffusion_pipeline.py` (submodule pinned at `f44233d`, 2024-03-25), class `AuffusionPipeline`, a
   Stable-Diffusion-style pipeline that the project imports unmodified (`cast/integration/auffusion.py`). Runtime library: diffusers 0.18.2, torch 2.0.1+cu118 (`cast` env).
2. **Relevant function / conditional.** `AuffusionPipeline.__call__`, line 931:
   ```python
   # `guidance_scale = 1` corresponds to doing no classifier free guidance.
   do_classifier_free_guidance = guidance_scale > 1.0
   ```
   The flag is passed to `_encode_prompt(...)` (line 934) and tested again in the denoising loop (lines 969, 982, 986).
3. **Meaning of `guidance_scale <= 1`.** Any value ≤ 1.0 sets the flag to `False`; 1.0 is the value whose documented meaning is "no classifier-free guidance" and is the one used here. (Values below 1 take exactly the same code path — they are not "weaker guidance" — which is why `guidance_scale = 0` was not used as a substitute.) Values > 1.0 enable CFG with the standard formula.
4. **Unconditional embeddings.** `_encode_prompt` lines 733–767: the null-condition tensor (`torch.zeros_like(prompt_embeds)`, Auffusion's unconditional embedding) is created only inside `if do_classifier_free_guidance and negative_prompt_embeds is None:`; the concatenation `torch.cat([negative_prompt_embeds, prompt_embeds])` is inside `if do_classifier_free_guidance:`. With the flag `False` neither executes and `prompt_embeds` is returned with batch size = number of prompts (1). A `negative_prompt` argument is therefore ignored entirely at guidance 1.0.
5. **UNet batch duplication.** Line 969: `latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents` → no duplication; the UNet receives one latent with one conditional embedding.
6. **Unconditional noise prediction.** Not computed: the UNet is called once per step on the batch of one conditional sample (`self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, ...)`, line 973).
7. **CFG arithmetic.** Lines 982–984 (`noise_pred_uncond, noise_pred_text = noise_pred.chunk(2); noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)`) and the `guidance_rescale` block (line 986) are inside `if do_classifier_free_guidance:` and are skipped. (With a batch of one, `chunk(2)` would also fail to unpack, so a successful run is itself evidence that the block did not execute.)

**CAST side.** `cast/pipeline.py::run_one` passes `guidance_scale=cfg.guidance_scale` straight to the pipeline and pre-samples the initial latent with the same generator as `prepare_latents` (shape independent of CFG, so the per-seed initial latent is the same tensor used by the CFG sweep). `cast/attention/cast_processor.py` splits the batch as `half = batch_size // 2` only for statistics and for the optional `apply_to_uncond=False` branch (not used); with batch 1, `half = 0` and all rows are treated as conditional. No code change was needed for conditional-only inference.

## 2. Runtime proof (`experiments/verify_no_cfg_runtime.py`, results in `runtime_verification.json`)

A forward pre-hook on `pipe.unet` (diagnostic script only, removed at exit) recorded for every UNet call the latent batch size, the `encoder_hidden_states` batch size and the number of all-zero conditioning rows (Auffusion's null condition). Prompt A01 "A dog barking loudly.", seed 42, GPU NVIDIA RTX A5000, full settings (100 PNDM steps, 256×1024, fp16, α=1.5, deterministic self-attention, 16 attn2 modules):

| Run | guidance | UNet calls | latent batch | cond batch | zero (unconditional) rows | result |
|---|---|---|---|---|---|---|
| A. Dense (CAST λ=0) | 1.0 | 101 | {1} | {1} | 0 | finite, 160,000 samples @ 16 kHz = 10.0 s |
| A. PLADIS (λ=2) | 1.0 | 101 | {1} | {1} | 0 | finite, 10.0 s; audio differs from Dense |
| B. Dense rerun | 1.0 | 101 | {1} | {1} | 0 | audio and final-latent SHA-256 bitwise equal to A |
| C. contrast: Dense, 3 steps | 7.5 | 4 | {2} | {2} | 1 per call (4 total) | CFG path: duplicated batch, one null-condition row |
| D. Dense, 10 steps, `negative_prompt=None` vs `"A loud explosion."` | 1.0 | 11 / 11 | {1} | {1} | 0 / 0 | outputs bitwise identical → the unconditional branch is never built or used |

(PNDM with `skip_prk_steps=True` evaluates the UNet 101 times for 100 steps: one extra warm-up evaluation.)

**Conclusion for the proposed no-CFG setting:** conditional forward passes per sample = 101, **unconditional forward passes = 0**, UNet input batch = 1, CFG combination arithmetic not executed. Both arms start from the same initial latent (`ea92f6ab…` for seed 42, identical to the CFG-sweep latent for that seed). Generation at guidance 1.0 was therefore allowed to proceed. All checks: `ALL_PASS = true`.
