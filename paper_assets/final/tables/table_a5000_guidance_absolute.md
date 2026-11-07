| Setting | Method | CLAP | AQAScore | PQ |
|---|---|---:|---:|---:|
| CFG 7.5 | Dense | 0.555 | 0.905 | 5.92 |
| CFG 7.5 | PLADIS λ=2 | 0.364 | 0.805 | 5.28 |
| CFG 3.0 | Dense | 0.531 | 0.871 | 5.88 |
| CFG 3.0 | PLADIS λ=2 | 0.429 | 0.805 | 5.63 |
| CFG 1.5 | Dense | 0.464 | 0.783 | 5.79 |
| CFG 1.5 | PLADIS λ=2 | 0.396 | 0.756 | 5.73 |
| No CFG | Dense | 0.385 | 0.680 | 5.73 |
| No CFG | PLADIS λ=2 | 0.339 | 0.669 | 5.67 |

Absolute means (24 prompts × 3 seeds). All four settings generated and evaluated on the same GPU (NVIDIA RTX A5000); CFG 7.5 / 3.0 / 1.5 from outputs/a5000_guidance_sweep_v1 (conditional + unconditional branch, standard CFG), No CFG from outputs/no_cfg_v1 (guidance_scale = 1.0, conditional-only inference). Same 24 prompts, seeds, initial latents, checkpoint, sampler and evaluators. Reference table; not recommended for the 2-page paper.
