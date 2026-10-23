# CAST Lambda Sweep 0 Listening Index

All paths relative to `outputs/lambda_sweep0/`. alpha = 1.5; seed 42; one bitwise-identical initial latent for every prompt and lambda; deterministic self-attention; CFG 7.5; 100 steps. lambda 0 / 1 / 2 are the Pilot 0 outputs (copied, see manifests `provenance`).

## P01 — A dog barking loudly.

- λ=0.00: `P01_SINGLE_DOG/lambda_0.00/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_0.00/spectrogram.png`)
- λ=0.25: `P01_SINGLE_DOG/lambda_0.25/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_0.25/spectrogram.png`)
- λ=0.50: `P01_SINGLE_DOG/lambda_0.50/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_0.50/spectrogram.png`)
- λ=0.75: `P01_SINGLE_DOG/lambda_0.75/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_0.75/spectrogram.png`)
- λ=1.00: `P01_SINGLE_DOG/lambda_1.00/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_1.00/spectrogram.png`)
- λ=1.50: `P01_SINGLE_DOG/lambda_1.50/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_1.50/spectrogram.png`)
- λ=2.00: `P01_SINGLE_DOG/lambda_2.00/audio.wav`  (spectrogram: `P01_SINGLE_DOG/lambda_2.00/spectrogram.png`)

Manual evaluation:

| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |
|---:|---|---|---|---|---|---|
| 0.00 | | | | | | |
| 0.25 | | | | | | |
| 0.50 | | | | | | |
| 0.75 | | | | | | |
| 1.00 | | | | | | |
| 1.50 | | | | | | |
| 2.00 | | | | | | |

## P02 — Heavy rain falling on a roof.

- λ=0.00: `P02_SINGLE_RAIN/lambda_0.00/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_0.00/spectrogram.png`)
- λ=0.25: `P02_SINGLE_RAIN/lambda_0.25/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_0.25/spectrogram.png`)
- λ=0.50: `P02_SINGLE_RAIN/lambda_0.50/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_0.50/spectrogram.png`)
- λ=0.75: `P02_SINGLE_RAIN/lambda_0.75/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_0.75/spectrogram.png`)
- λ=1.00: `P02_SINGLE_RAIN/lambda_1.00/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_1.00/spectrogram.png`)
- λ=1.50: `P02_SINGLE_RAIN/lambda_1.50/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_1.50/spectrogram.png`)
- λ=2.00: `P02_SINGLE_RAIN/lambda_2.00/audio.wav`  (spectrogram: `P02_SINGLE_RAIN/lambda_2.00/spectrogram.png`)

Manual evaluation:

| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |
|---:|---|---|---|---|---|---|
| 0.00 | | | | | | |
| 0.25 | | | | | | |
| 0.50 | | | | | | |
| 0.75 | | | | | | |
| 1.00 | | | | | | |
| 1.50 | | | | | | |
| 2.00 | | | | | | |

## P03 — A dog barking while rain falls in the background.

- λ=0.00: `P03_TWO_DOG_RAIN/lambda_0.00/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_0.00/spectrogram.png`)
- λ=0.25: `P03_TWO_DOG_RAIN/lambda_0.25/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_0.25/spectrogram.png`)
- λ=0.50: `P03_TWO_DOG_RAIN/lambda_0.50/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_0.50/spectrogram.png`)
- λ=0.75: `P03_TWO_DOG_RAIN/lambda_0.75/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_0.75/spectrogram.png`)
- λ=1.00: `P03_TWO_DOG_RAIN/lambda_1.00/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_1.00/spectrogram.png`)
- λ=1.50: `P03_TWO_DOG_RAIN/lambda_1.50/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_1.50/spectrogram.png`)
- λ=2.00: `P03_TWO_DOG_RAIN/lambda_2.00/audio.wav`  (spectrogram: `P03_TWO_DOG_RAIN/lambda_2.00/spectrogram.png`)

Manual evaluation:

| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |
|---:|---|---|---|---|---|---|
| 0.00 | | | | | | |
| 0.25 | | | | | | |
| 0.50 | | | | | | |
| 0.75 | | | | | | |
| 1.00 | | | | | | |
| 1.50 | | | | | | |
| 2.00 | | | | | | |

## P04 — A man speaking while a car horn sounds nearby.

- λ=0.00: `P04_TWO_SPEECH_HORN/lambda_0.00/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_0.00/spectrogram.png`)
- λ=0.25: `P04_TWO_SPEECH_HORN/lambda_0.25/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_0.25/spectrogram.png`)
- λ=0.50: `P04_TWO_SPEECH_HORN/lambda_0.50/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_0.50/spectrogram.png`)
- λ=0.75: `P04_TWO_SPEECH_HORN/lambda_0.75/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_0.75/spectrogram.png`)
- λ=1.00: `P04_TWO_SPEECH_HORN/lambda_1.00/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_1.00/spectrogram.png`)
- λ=1.50: `P04_TWO_SPEECH_HORN/lambda_1.50/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_1.50/spectrogram.png`)
- λ=2.00: `P04_TWO_SPEECH_HORN/lambda_2.00/audio.wav`  (spectrogram: `P04_TWO_SPEECH_HORN/lambda_2.00/spectrogram.png`)

Manual evaluation:

| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |
|---:|---|---|---|---|---|---|
| 0.00 | | | | | | |
| 0.25 | | | | | | |
| 0.50 | | | | | | |
| 0.75 | | | | | | |
| 1.00 | | | | | | |
| 1.50 | | | | | | |
| 2.00 | | | | | | |

## P05 — Footsteps on a wooden floor while birds chirp outside.

- λ=0.00: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.00/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.00/spectrogram.png`)
- λ=0.25: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.25/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.25/spectrogram.png`)
- λ=0.50: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.50/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.50/spectrogram.png`)
- λ=0.75: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.75/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_0.75/spectrogram.png`)
- λ=1.00: `P05_TWO_FOOTSTEPS_BIRDS/lambda_1.00/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_1.00/spectrogram.png`)
- λ=1.50: `P05_TWO_FOOTSTEPS_BIRDS/lambda_1.50/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_1.50/spectrogram.png`)
- λ=2.00: `P05_TWO_FOOTSTEPS_BIRDS/lambda_2.00/audio.wav`  (spectrogram: `P05_TWO_FOOTSTEPS_BIRDS/lambda_2.00/spectrogram.png`)

Manual evaluation:

| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |
|---:|---|---|---|---|---|---|
| 0.00 | | | | | | |
| 0.25 | | | | | | |
| 0.50 | | | | | | |
| 0.75 | | | | | | |
| 1.00 | | | | | | |
| 1.50 | | | | | | |
| 2.00 | | | | | | |
