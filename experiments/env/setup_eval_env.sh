#!/usr/bin/env bash
# Creates the `cast-eval` conda env (AQAScore + AudioBox). Never touches the `cast` generation env.
set -euo pipefail
ENV_NAME=cast-eval
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then conda create -y -n "${ENV_NAME}" python=3.10; fi
PIP="$HOME/miniconda3/envs/${ENV_NAME}/bin/pip"
"$PIP" install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
grep -vE "^(torch|torchaudio|torchvision)==" experiments/env/requirements-eval.txt | grep -vE '^#' | grep -v '^$' > /tmp/cast_eval_reqs.txt
"$PIP" install --no-cache-dir -r /tmp/cast_eval_reqs.txt
"$HOME/miniconda3/envs/${ENV_NAME}/bin/python" -c "import torch, transformers, audiobox_aesthetics; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), '| transformers', transformers.__version__); from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor; print('Qwen2.5-Omni classes OK')"
