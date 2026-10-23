#!/usr/bin/env bash
# Independent golden cross-check: run third_party/Auffusion/inference.py UNMODIFIED on a 1-row dataset with the
# milestone settings (seed 42, 100 steps, CFG 7.5, 256x1024, fp16, autocast, no xformers) and compare its wav
# bit-for-bit with outputs/golden/audio.wav produced through cast.pipeline (mode=golden).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/miniconda3/envs/cast/bin/python"
SNAP=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('auffusion/auffusion', token=False))")
OUT=outputs/golden_official
mkdir -p "$OUT"
echo '{"audio_path": "official.wav", "text": "A dog barking while rain falls in the background."}' > "$OUT/one.json"
rm -f "$OUT/audios/official.wav"
( cd third_party/Auffusion && "$PY" inference.py --pretrained_model_name_or_path="$SNAP" --test_data_dir="../../$OUT/one.json" \
    --output_dir="../../$OUT" --audio_column=audio_path --caption_column=text --sample_rate=16000 --duration=10 \
    --num_inference_steps=100 --guidance_scale=7.5 --mixed_precision=fp16 --seed=42 --width=1024 --height=256 )
"$PY" - <<'PYEOF'
import numpy as np, json
from scipy.io.wavfile import read
sr1, a = read("outputs/golden_official/audios/official.wav"); sr2, b = read("outputs/golden/audio.wav")
d = np.abs(a.astype(np.int64) - b.astype(np.int64))
res = {"sr": [int(sr1), int(sr2)], "len": [int(len(a)), int(len(b))], "identical": bool(np.array_equal(a, b)), "max_abs_int16": int(d.max()), "mean_abs_int16": float(d.mean())}
print("[official inference.py vs cast golden]", res)
json.dump(res, open("outputs/golden_official/compare.json", "w"), indent=2)
PYEOF
