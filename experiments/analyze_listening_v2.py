"""Analysis of the prompt-v2 screening set (16 prompts, first per category) over seeds {42,123,2026} and CFG {3.0, 1.5}: PLADIS(lambda=2) - Dense.
Inputs: outputs/listen_prompts_v2/cfg*_seed*/clap_scores.csv, outputs/listen_prompts_v2/audio_index_all.csv, outputs/listen_prompts_v2/audiobox_all/sample_scores.csv
Prompt = unit (seeds averaged); 10,000 prompt-level bootstrap resamples (rng seed 20261024). Writes outputs/listen_prompts_v2/analysis/*.
Usage: python experiments/analyze_listening_v2.py"""
import csv, glob, itertools, json, os
import numpy as np

R = "outputs/listen_prompts_v2"; OUT = os.path.join(R, "analysis"); os.makedirs(OUT, exist_ok=True); B = 10000
rng = np.random.default_rng(20261024)
val = {"clap": {}, "pq": {}}; fam, cat = {}, {}
for f in glob.glob(os.path.join(R, "cfg*_seed*", "clap_scores.csv")):
    d = os.path.basename(os.path.dirname(f)); c, s = d[3:].split("_seed"); c, s = float(c), int(s)
    for r in csv.DictReader(open(f, encoding="utf-8")):
        val["clap"][(c, r["id"], s, r["arm"])] = float(r["clap"]); fam[r["id"]] = r["family"]; cat[r["id"]] = r["category"]
idx = {r["audio_path"]: r for r in csv.DictReader(open(os.path.join(R, "audio_index_all.csv"), encoding="utf-8"))}
for r in csv.DictReader(open(os.path.join(R, "audiobox_all", "sample_scores.csv"), encoding="utf-8")):
    i = idx[r["audio_path"].replace("\\", "/")]; val["pq"][(float(i["cfg"]), i["prompt_id"], int(i["seed"]), i["arm"])] = float(r["PQ"])


def ci(x):
    x = np.asarray(x); m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rows, per_prompt, diag = [], [], {}
for met in ("clap", "pq"):
    for cfg in (3.0, 1.5):
        ids = sorted({k[1] for k in val[met] if k[0] == cfg}); seeds = sorted({k[2] for k in val[met] if k[0] == cfg})
        ids = [p for p in ids if all((cfg, p, s, a) in val[met] for s in seeds for a in ("dense_l0", "pladis_l2"))]
        D = {s: np.array([val[met][(cfg, p, s, "pladis_l2")] - val[met][(cfg, p, s, "dense_l0")] for p in ids]) for s in seeds}
        d = np.mean([D[s] for s in seeds], axis=0); dense = np.mean([[val[met][(cfg, p, s, "dense_l0")] for p in ids] for s in seeds], axis=0)
        lo, hi = ci(d); row = {"metric": met, "cfg": cfg, "group": "all", "n_prompts": len(ids), "n_seeds": len(seeds), "dense_mean": float(dense.mean()), "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "frac_pladis_better": float((d > 0).mean())}
        rows.append(row)
        for g in ("clear", "ambiguous"):
            sel = np.array([fam[p] == g for p in ids]); l, h = ci(d[sel]); rows.append({**row, "group": g, "n_prompts": int(sel.sum()), "dense_mean": float(dense[sel].mean()), "delta": float(d[sel].mean()), "ci_lo": l, "ci_hi": h, "frac_pladis_better": float((d[sel] > 0).mean())})
        gd = d[np.array([fam[p] == "ambiguous" for p in ids])]; gc = d[np.array([fam[p] == "clear" for p in ids])]
        diff = rng.choice(gd, (B, len(gd))).mean(axis=1) - rng.choice(gc, (B, len(gc))).mean(axis=1)
        cors = [float(np.corrcoef(D[a], D[b])[0, 1]) for a, b in itertools.combinations(seeds, 2)] if len(seeds) > 1 else []
        diag[f"{met}_cfg{cfg:g}"] = {"ambiguous_minus_clear_delta": float(gd.mean() - gc.mean()), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                     "seed_consistency_mean_pairwise_r": float(np.mean(cors)) if cors else None, "seeds": seeds}
        for p, x, y in zip(ids, d, dense): per_prompt.append({"metric": met, "cfg": cfg, "prompt_id": p, "category": cat[p], "family": fam[p], "dense": float(y), "delta": float(x), **{f"delta_seed{s}": float(D[s][ids.index(p)]) for s in seeds}})
for name, data in (("delta_summary.csv", rows), ("per_prompt_delta.csv", per_prompt)):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r}, key=lambda k: list(data[0]).index(k) if k in data[0] else 99)); w.writeheader(); w.writerows(data)
json.dump(diag, open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1)
print(f"{'metric':5s} {'cfg':>4s} {'group':10s} {'n':>3s} {'seeds':>5s} {'dense':>7s} {'delta':>8s}  95% CI              pladis_better")
for r in rows: print(f"{r['metric']:5s} {r['cfg']:4g} {r['group']:10s} {r['n_prompts']:3d} {r['n_seeds']:5d} {r['dense_mean']:7.3f} {r['delta']:+8.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['frac_pladis_better']:.2f}")
print(json.dumps(diag, indent=1))
