"""Index every finished sample of outputs/cfg_sweep_v1 (format read by eval_audiobox.py / eval_aqascore.py, plus cfg and arm columns).
Usage: python experiments/make_sweep_index.py [--root outputs/cfg_sweep_v1] [--prompt-file experiments/prompts/cast_diagnostic_v1.json]"""
import argparse, csv, glob, hashlib, json, os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/cfg_sweep_v1"); ap.add_argument("--prompt-file", default="experiments/prompts/cast_diagnostic_v1.json")
    a = ap.parse_args()
    pr = {p["id"]: p for p in json.load(open(a.prompt_file, encoding="utf-8"))["prompts"]}; rows = []
    for wav in sorted(glob.glob(os.path.join(a.root, "cfg*", "*", "*", "seed_*", "*", "audio.wav"))):
        if not os.path.exists(os.path.join(os.path.dirname(wav), "manifest.json")): continue
        parts = wav.replace("\\", "/").split("/")[-6:-1]; cfg, cat, pid, seed, arm = parts
        rows.append({"audio_sha256": hashlib.sha256(open(wav, "rb").read()).hexdigest(), "prompt_id": pid, "category": cat, "prompt": pr[pid]["prompt"], "seed": int(seed.split("_")[1]),
                     "audio_path": wav.replace("\\", "/"), "n_provenances": 1, "provenances": f"{cfg}:{arm}", "cfg": float(cfg[3:]), "arm": arm})
    out = os.path.join(a.root, "audio_index.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "samples ->", out)


if __name__ == "__main__":
    main()
