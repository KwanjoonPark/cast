# Project: PLADIS for Text-to-Audio

## Goal

Apply PLADIS sparse cross-attention to Auffusion
without additional training.

Main comparison:

1. Vanilla Auffusion
2. Sparse Attention
   - alpha = 1.5
   - lambda = 1.0
3. PLADIS
   - alpha = 1.5
   - lambda = 2.0

## Repository Structure

third_party/PLADIS
- Original PLADIS implementation
- READ ONLY

third_party/Auffusion
- Original Auffusion implementation
- READ ONLY unless absolutely necessary

cast/
- All new implementations should go here (CAST package).

experiments/
- Experiment configurations and prompts.

outputs/
- Generated audio and evaluation results.

## Important Rules

- Do not modify baseline behavior.
- Vanilla output must remain reproducible.
- Use the same random seed across methods.
- Modify cross-attention only.
- Do not modify self-attention.
- Do not train or finetune the model.
- PLADIS is inference-time only.

## PLADIS

Dense attention:

A_dense = softmax(QK^T / sqrt(d)) V

Sparse attention:

A_sparse = entmax_1.5(QK^T / sqrt(d)) V

PLADIS:

A = A_dense + lambda * (A_sparse - A_dense)

lambda = 0: vanilla
lambda = 1: sparse attention
lambda = 2: default PLADIS

## Current Priority

First milestone:

Generate the same prompt and same seed using:

- Vanilla
- Sparse
- PLADIS

Do not implement additional ablations before this works.