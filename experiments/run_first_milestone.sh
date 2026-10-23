#!/usr/bin/env bash
# Runs the full first milestone (golden + latent-control checks + 3 CAST arms) in the `cast` env.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/miniconda3/envs/cast/bin/python"
"$PY" -m pytest -q cast/tests
"$PY" experiments/validate_parity.py
"$PY" experiments/run_first_milestone.py "$@"
