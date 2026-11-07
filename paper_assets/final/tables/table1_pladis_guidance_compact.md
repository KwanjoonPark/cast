| Guidance | ΔCLAP | ΔAQAScore | ΔPQ |
|---|---:|---:|---:|
| CFG 7.5 | -0.193‡ | -0.099‡ | -0.63‡ |
| CFG 3.0 | -0.102‡ | -0.065‡ | -0.26‡ |
| CFG 1.5 | -0.070‡ | -0.028 | -0.06 |
| No-CFG* | -0.046‡ | -0.011 | -0.06 |

Δ = PLADIS λ=2 − Dense (24 prompts × 3 seeds, seeds averaged within prompt). ‡ 95 % prompt-level bootstrap CI excludes 0 (10,000 resamples). * No-CFG (guidance_scale = 1.0, conditional-only inference) was evaluated in a separate paired run on NVIDIA RTX A5000; CFG 7.5 / 3.0 / 1.5 were generated on NVIDIA GeForce RTX 2080. Absolute scores are not hardware-controlled across rows; each Δ is paired within its own run.
