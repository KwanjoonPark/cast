"""Build an audio_index.csv (the format eval_audiobox.py / eval_aqascore.py read) from listening sets made by make_listening_set.py.
Usage: python experiments/make_listening_index.py --roots outputs/listen_prompts_v2/cfg3_seed42 outputs/listen_prompts_v2/cfg1.5_seed42 --out outputs/listen_prompts_v2/audio_index.csv"""
import argparse, csv, hashlib, json, os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = []
    for root in a.roots:
        s = json.load(open(os.path.join(root, "listening_summary.json"), encoding="utf-8"))
        for p in s["prompts"]:
            for arm in s["arms"]:
                path = os.path.join(root, "flat", f"{p['id']}_{arm}.wav").replace("\\", "/")
                rows.append({"audio_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"],
                             "seed": s["seed"], "audio_path": path, "n_provenances": 1, "provenances": f"cfg{s['cfg']:g}:{arm}", "cfg": s["cfg"], "arm": arm, "family": p.get("family")})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
