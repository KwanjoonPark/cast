#!/usr/bin/env bash
# Creates the `cast` conda env. Run from the project root. Idempotent-ish (skips create if env exists).
set -euo pipefail
ENV_NAME=cast
PY=3.10
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -y -n "${ENV_NAME}" python="${PY}"
fi
PIP="$HOME/miniconda3/envs/${ENV_NAME}/bin/pip"
"$PIP" install --upgrade "pip<24.1"
"$PIP" install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
grep -vE '^(torch|torchaudio|torchvision)==' experiments/env/requirements-cast.txt | grep -vE '^#' | grep -v '^$' > /tmp/cast_reqs_rest.txt
"$PIP" install -r /tmp/cast_reqs_rest.txt
"$HOME/miniconda3/envs/${ENV_NAME}/bin/python" -c "import torch, diffusers, transformers, huggingface_hub, entmax; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'hub', huggingface_hub.__version__, 'entmax', entmax.__version__)"
