"""CLAP (laion/clap-htsat-unfused, validated convention) for all diagnostic_v1 conditions.
Writes outputs/diagnostic_v1/clap_scores.{csv,json}. Also records exploratory per-event phrase cosines (not a primary metric)."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cast.eval.clap import ClapScorer

ROOT = "outputs/diagnostic_v1"; COND = "lambda"


def main():
    global ROOT, COND
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--cond-key", default=COND, help="condition column name: lambda or beta")
    a = ap.parse_args(); ROOT, COND = a.root, a.cond_key
    gs = json.load(open(f"{ROOT}/generation_summary.json")); runs = gs["runs"]
    ds = {p["id"]: p for p in json.load(open("experiments/prompts/cast_diagnostic_v1.json"))["prompts"]}
    scorer = ClapScorer()
    # repeatability on one file
    k0 = next(iter(runs)); reps = [scorer.score(runs[k0]["audio_path"], [runs[k0]["prompt"]])[0][runs[k0]["prompt"]] for _ in range(3)]
    rows = []
    for k, r in runs.items():
        p = ds[r["prompt_id"]]; texts = [r["prompt"]] + p["events"]
        sc, info = scorer.score(r["audio_path"], texts)
        row = {"prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], COND: r[COND], "alpha": r.get("alpha"),
               "audio_path": r["audio_path"], "initial_latent_sha256": r["initial_latent_sha256"], "final_latent_norm": r["final_latent_norm"], "finite": r["valid"],
               "clap_cosine": sc[r["prompt"]], "clap_percent": 100 * sc[r["prompt"]], "resampled_len": info["resampled_len"], "resampled_peak": info["resampled_peak"]}
        for i, ev in enumerate(p["events"]): row[f"event{i+1}_text"] = ev; row[f"event{i+1}_clap"] = sc[ev]
        if "foreground_event" in p: row["fg_event_clap"] = sc[p["foreground_event"]]; row["bg_event_clap"] = sc[p["background_event"]]
        rows.append(row); print(f"[{k}] clap={row['clap_cosine']:.4f}", flush=True)
    res = {"metric": "clap_cosine = dot(L2norm(audio_emb), L2norm(text_emb)); clap_percent = 100*clap_cosine", "model_info": scorer.info,
           "repeatability": {"sample": runs[k0]["audio_path"], "scores": reps, "max_abs_diff": max(reps) - min(reps)},
           "sanity": {"n": len(rows), "all_finite": all(r["finite"] and r["clap_cosine"] == r["clap_cosine"] for r in rows), "all_resampled_480000": all(r["resampled_len"] == 480000 for r in rows),
                      "max_resampled_peak": max(r["resampled_peak"] for r in rows)}, "samples": rows}
    json.dump(res, open(f"{ROOT}/clap_scores.json", "w"), indent=2)
    cols = ["prompt_id", "category", "prompt", "seed", COND, "alpha", "audio_path", "initial_latent_sha256", "final_latent_norm", "finite", "clap_cosine", "clap_percent",
            "event1_text", "event1_clap", "event2_text", "event2_clap", "fg_event_clap", "bg_event_clap"]
    with open(f"{ROOT}/clap_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rows]
    print("CLAP done:", res["sanity"], "repeat max diff", res["repeatability"]["max_abs_diff"])


if __name__ == "__main__":
    main()
