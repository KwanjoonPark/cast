"""Layer ablation analysis (CFG 3.0): PLADIS (lambda=2) applied to resolution subsets vs Dense and vs all 16 layers.
Arms: dense, all (= sweep entmax_l2) from outputs/cfg_sweep_v1; high (pladis_high), low (pladis_low), optionally mid (pladis_mid) from outputs/layer_ablation_v1.
Unit = prompt (seeds averaged); 10,000 prompt-level bootstrap resamples (rng seed 20261024).
Reports: delta vs Dense per arm; paired difference high-low (and each subset minus all); fraction of the all-layer harm retained per subset.
Needs <root>/audio_index.csv, clap_scores.csv, audiobox/sample_scores.csv in both roots (make_sweep_index.py / score_sweep_clap.py / eval_audiobox.py).
Usage: python experiments/analyze_layer_ablation_v1.py [--cfg 3.0]"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

SWEEP, ABL = "outputs/cfg_sweep_v1", "outputs/layer_ablation_v1"
B = 10000


def load(root, cfg):
    idx = {r["audio_path"]: r for r in csv.DictReader(open(os.path.join(root, "audio_index.csv"), encoding="utf-8"))}
    out = {"clap": {}, "pq": {}}
    for met, path, col in (("clap", "clap_scores.csv", "clap"), ("pq", os.path.join("audiobox", "sample_scores.csv"), "PQ")):
        for r in csv.DictReader(open(os.path.join(root, path), encoding="utf-8")):
            i = idx.get(r["audio_path"].replace("\\", "/"))
            if i is None or abs(float(i["cfg"]) - cfg) > 1e-9: continue
            out[met][(i["prompt_id"], int(i["seed"]), i["arm"])] = float(r[col])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", type=float, default=3.0); a = ap.parse_args()
    rng = np.random.default_rng(20261024); S, A = load(SWEEP, a.cfg), load(ABL, a.cfg)
    arms = {"dense": ("S", "dense"), "all": ("S", "entmax_l2"), "high": ("A", "pladis_high"), "low": ("A", "pladis_low"), "mid": ("A", "pladis_mid")}
    out_dir = os.path.join(ABL, "analysis"); os.makedirs(out_dir, exist_ok=True); res, rows = {}, []
    def ci(x): m = np.asarray(x)[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    for met in ("clap", "pq"):
        src = {"S": S[met], "A": A[met]}
        have = [k for k, (s, arm) in arms.items() if any(kk[2] == arm for kk in src[s])]
        ids = sorted({k[0] for k in src["A"]}); seeds = sorted({k[1] for k in src["A"]})
        ids = [p for p in ids if all((p, s, arms[k][1]) in src[arms[k][0]] for s in seeds for k in have)]
        val = {k: np.array([np.mean([src[arms[k][0]][(p, s, arms[k][1])] for s in seeds]) for p in ids]) for k in have}
        res[met] = {"n_prompts": len(ids), "seeds": seeds, "arms": have, "dense_mean": float(val["dense"].mean())}
        for k in have:
            if k == "dense": continue
            d = val[k] - val["dense"]; lo, hi = ci(d); rows.append({"metric": met, "arm": k, "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "frac_better_than_dense": float((d > 0).mean())})
        def diff(x, y):
            d = val[x] - val[y]; lo, hi = ci(d); return {"mean": float(d.mean()), "ci95": [lo, hi]}
        res[met]["high_minus_low"] = diff("high", "low")
        for k in ("high", "low", "mid"):
            if k in have: res[met][f"{k}_minus_all"] = diff(k, "all")
        for k in ("high", "low", "mid"):
            if k in have:
                dall = (val["all"] - val["dense"]).mean(); res[met][f"{k}_share_of_all_harm"] = float((val[k] - val["dense"]).mean() / dall) if abs(dall) > 1e-9 else None
    with open(os.path.join(out_dir, "delta_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(res, open(os.path.join(out_dir, "analysis_summary.json"), "w"), indent=1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
    order = [k for k in ("all", "high", "mid", "low") if k in res["clap"]["arms"]]
    lab = {"all": "All 16\nlayers", "high": "High-res\n(L0, 5)", "mid": "Mid-res\n(L1, 5)", "low": "Low-res\n(L2+mid, 6)"}
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for ax, (met, title) in zip(axes, (("clap", "ΔCLAP"), ("pq", "ΔPQ (AudioBox)"))):
        for i, k in enumerate(order):
            r = next(x for x in rows if x["metric"] == met and x["arm"] == k)
            ax.errorbar(i, r["delta"], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]], fmt="s", color="#1f4e79" if k == "all" else "#c05621", ms=4.5, capsize=3, lw=1.0)
        ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[k] for k in order], fontsize=6.5); ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(f"{title}, CFG {a.cfg:.1f}", fontsize=8.5, loc="left"); ax.grid(axis="y", alpha=0.25, lw=0.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("PLADIS λ=2 on subset, Δ vs Dense")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_layer_ablation.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig_layer_ablation.pdf"), bbox_inches="tight")
    print(f"{'metric':5s} {'arm':5s} {'delta':>8s}  95% CI              better-than-dense")
    for r in rows: print(f"{r['metric']:5s} {r['arm']:5s} {r['delta']:+8.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['frac_better_than_dense']:.2f}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
