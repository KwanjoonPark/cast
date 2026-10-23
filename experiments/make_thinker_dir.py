"""Build a local Qwen2.5-Omni model directory that contains only the thinker weights (shards 1-4 of the 7B checkpoint;
shard 5 holds talker/token2wav only) by symlinking the cached snapshot files and writing a filtered
model.safetensors.index.json. Weights are untouched; this only lets `from_pretrained` skip the absent shard."""
import json, os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Omni-7B"
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/cast_eval/" + repo.split("/")[-1] + "-thinker")
src = snapshot_download(repo, token=False, allow_patterns=["*.json", "*.txt", "*.py"], local_files_only=True)
os.makedirs(dst, exist_ok=True)
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
present = {f for f in os.listdir(src) if f.endswith(".safetensors")}
wm = {k: v for k, v in idx["weight_map"].items() if k.startswith("thinker.") and v in present}
dropped = {k: v for k, v in idx["weight_map"].items() if k.startswith("thinker.") and v not in present}
assert not dropped, f"thinker weights missing from downloaded shards: {list(dropped)[:5]}"
for f in os.listdir(src):
    if f == "model.safetensors.index.json": continue
    p = os.path.join(dst, f)
    if not os.path.exists(p): os.symlink(os.path.realpath(os.path.join(src, f)), p)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
print("thinker dir:", dst, "| thinker tensors kept:", len(wm), "| shards:", sorted(set(wm.values())), "| non-thinker tensors dropped from index:", sum(1 for k in idx["weight_map"] if not k.startswith("thinker.")))
