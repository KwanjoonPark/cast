| Guidance | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] |
|---|---|---|---|
| CFG 7.5 | -0.192 [-0.228, -0.156] | -0.100 [-0.156, -0.046] | -0.64 [-0.90, -0.39] |
| CFG 3.0 | -0.102 [-0.135, -0.068] | -0.066 [-0.131, -0.009] | -0.25 [-0.44, -0.07] |
| CFG 1.5 | -0.068 [-0.096, -0.043] | -0.027 [-0.081, +0.029] | -0.06 [-0.20, +0.06] |
| No CFG | -0.046 [-0.077, -0.019] | -0.011 [-0.052, +0.032] | -0.06 [-0.15, +0.02] |

Δ = PLADIS λ=2 − Dense; prompt = unit of analysis (3 seeds averaged); Δ = condition − Dense at the same guidance setting and prompt; 95 % CI from 10,000 prompt-level bootstrap resamples, rng seed 20261024; 24 prompts (experiments/prompts/cast_diagnostic_v1.json), seeds 42/123/2026, 100-step PNDM, 256×1024 mel, fp16, α=1.5, deterministic self-attention, 16 attn2 modules only. All four settings generated and evaluated on the same GPU (NVIDIA RTX A5000); CFG 7.5 / 3.0 / 1.5 from outputs/a5000_guidance_sweep_v1 (conditional + unconditional branch, standard CFG), No CFG from outputs/no_cfg_v1 (guidance_scale = 1.0, conditional-only inference). Same 24 prompts, seeds, initial latents, checkpoint, sampler and evaluators.
