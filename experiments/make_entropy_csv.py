"""Sample-level attention-entropy table for sharp_softmax_v1 from the per-condition manifests (cond. CFG half).
Also carries the counterfactual probe statistics measured on the beta=1 (dense) trajectory."""
import csv, json, os
ROOT = "outputs/sharp_softmax_v1"
gs = json.load(open(f"{ROOT}/generation_summary.json"))["runs"]; clap = {}
if os.path.exists(f"{ROOT}/clap_scores.csv"):
    for r in csv.DictReader(open(f"{ROOT}/clap_scores.csv")): clap[(r["prompt_id"], int(r["seed"]), float(r["beta"]))] = float(r["clap_cosine"])
rows = []
for k, r in gs.items():
    m = json.load(open(os.path.join(r["dir"], "manifest.json"))); e = m["entropy_stats_cond"]
    row = {"prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], "beta": r["beta"], "audio_path": r["audio_path"],
           "initial_latent_sha256": r["initial_latent_sha256"], "final_latent_norm": r["final_latent_norm"], "clap_cosine": clap.get((r["prompt_id"], r["seed"], r["beta"]), ""),
           "mean_attention_entropy": e["mean_entropy"], "normalized_attention_entropy": e["normalized_entropy"], "effective_support": e["effective_support"], "finite": r["valid"],
           "entropy_by_layer": json.dumps([round(x, 4) for x in e["entropy_by_layer"]]), "source": r["source"]}
    for pk, pv in (e.get("probe") or {}).items():
        row[f"probe_{pk}_normalized_entropy"] = pv["normalized_entropy"]; row[f"probe_{pk}_effective_support"] = pv["effective_support"]
        if "exact_support" in pv: row[f"probe_{pk}_exact_support"] = pv["exact_support"]
    rows.append(row)
cols = ["prompt_id", "category", "prompt", "seed", "beta", "audio_path", "initial_latent_sha256", "final_latent_norm", "clap_cosine", "mean_attention_entropy", "normalized_attention_entropy", "effective_support", "finite", "source",
        "probe_beta_1.25_normalized_entropy", "probe_beta_1.25_effective_support", "probe_beta_1.50_normalized_entropy", "probe_beta_1.50_effective_support", "probe_beta_2.00_normalized_entropy", "probe_beta_2.00_effective_support",
        "probe_entmax15_normalized_entropy", "probe_entmax15_effective_support", "probe_entmax15_exact_support", "entropy_by_layer"]
with open(f"{ROOT}/attention_entropy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rows]
print("attention_entropy.csv rows:", len(rows))
