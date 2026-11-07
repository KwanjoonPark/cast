| Guidance | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] |
|---|---|---|---|
| CFG 7.5 | -0.193 [-0.231, -0.155] | -0.099 [-0.157, -0.045] | -0.63 [-0.89, -0.38] |
| CFG 3.0 | -0.102 [-0.135, -0.070] | -0.065 [-0.132, -0.008] | -0.26 [-0.46, -0.08] |
| CFG 1.5 | -0.070 [-0.099, -0.044] | -0.028 [-0.085, +0.032] | -0.06 [-0.19, +0.05] |
| No-CFG* | -0.046 [-0.077, -0.019] | -0.011 [-0.052, +0.032] | -0.06 [-0.15, +0.02] |

Δ = PLADIS λ=2 − Dense; prompt = unit of analysis (3 seeds averaged); Δ = condition − Dense at the same guidance setting and prompt; 95 % CI from 10,000 prompt-level bootstrap resamples, rng seed 20261024; 24 prompts (experiments/prompts/cast_diagnostic_v1.json), seeds 42/123/2026, 100-step PNDM, 256×1024 mel, fp16, α=1.5, deterministic self-attention, 16 attn2 modules only. * No-CFG (guidance_scale = 1.0, conditional-only inference) was evaluated in a separate paired run on NVIDIA RTX A5000; CFG 7.5 / 3.0 / 1.5 were generated on NVIDIA GeForce RTX 2080. Absolute scores are not hardware-controlled across rows; each Δ is paired within its own run.
