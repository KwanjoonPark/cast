"""No-CFG v1 analysis: paired PLADIS (entmax_l2) - Dense per prompt (seeds averaged) for CLAP / AQAScore / AudioBox PQ, using the
CFG-sweep bootstrap convention imported from analyze_cfg_sweep_v1 (same load(), same boot_ci(), 10,000 prompt-level resamples,
rng seed 20261024; metrics consume the rng in the same loading order clap, pq, aqa).
Inputs: <root>/audio_index.csv, clap_scores.csv, audiobox/sample_scores.csv, aqascore/sample_scores.csv, experiment_manifest.json.
Outputs (<root>/analysis): delta_table.csv, analysis_summary.json, per_prompt_delta.csv, fig_no_cfg_pladis_delta.{png,pdf} and, unless
--no-trend, fig_cfg_to_nocfg_trend.{png,pdf} (previous sweep deltas read from <sweep-root>/analysis/delta_table.csv; the no-CFG point is
drawn with a different marker/colour and the figure states that it is a separate hardware run).
Usage: python experiments/analyze_no_cfg_v1.py [--root outputs/no_cfg_v1] [--sweep-root outputs/cfg_sweep_v1] [--no-trend]"""
import argparse, csv, itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_cfg_sweep_v1 import load, boot_ci, B, rng_seed

ARMS = ("dense", "entmax_l2")
ORDER = ("clap", "aqa", "pq")
LABEL = {"clap": "ΔCLAP", "aqa": "ΔAQAScore", "pq": "ΔPQ (AudioBox)"}
NEUTRAL = {"clap": 0.01, "aqa": 0.01, "pq": 0.05}      # descriptive per-prompt neutrality band (|Δ| below it counts as ~neutral)


def classify(delta, lo, hi):
    if lo > 0: return "Significant improvement"
    if hi < 0: return "Significant degradation"
    return "Positive but uncertain" if delta > 0 else "No measurable difference"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/no_cfg_v1"); ap.add_argument("--sweep-root", default="outputs/cfg_sweep_v1"); ap.add_argument("--no-trend", action="store_true")
    a = ap.parse_args(); out_dir = os.path.join(a.root, "analysis"); os.makedirs(out_dir, exist_ok=True)
    m = load(a.root)                                   # metric -> {(cfg, prompt, seed, arm): value}; keys in loading order clap, pq, aqa
    man = json.load(open(os.path.join(a.root, "experiment_manifest.json"), encoding="utf-8"))
    prompt_text = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(os.path.join(a.root, "audio_index.csv"), encoding="utf-8"))}
    rng = np.random.default_rng(rng_seed); rows, per_prompt = [], {}
    summary = {"experiment": man.get("experiment"), "gpu": man.get("versions", {}).get("gpu"), "guidance_scale": man.get("cfgs"), "arms": list(ARMS),
               "bootstrap": {"resamples": B, "rng_seed": rng_seed, "unit": "prompt (seeds averaged)", "rng_consumption_order": list(m)}, "metrics": {}}
    for met, d in m.items():
        g = {}
        for (cfg, pid, seed, arm), v in d.items(): g.setdefault(pid, {}).setdefault(seed, {})[arm] = v
        prompts = sorted(p for p, s in g.items() if all(all(arm in s[sd] for arm in ARMS) for sd in s))
        seeds = sorted({sd for p in prompts for sd in g[p]})
        dense = np.array([np.mean([g[p][sd]["dense"] for sd in g[p]]) for p in prompts]); plad = np.array([np.mean([g[p][sd]["entmax_l2"] for sd in g[p]]) for p in prompts])
        delta = plad - dense; lo, hi = boot_ci(delta, rng)
        dseed = {sd: np.array([g[p][sd]["entmax_l2"] - g[p][sd]["dense"] for p in prompts if sd in g[p]]) for sd in seeds}
        full = [sd for sd in seeds if len(dseed[sd]) == len(prompts)]
        cors = [float(np.corrcoef(dseed[x], dseed[y])[0, 1]) for x, y in itertools.combinations(full, 2)] if len(full) >= 2 else []
        row = {"metric": met, "condition": "no_cfg (guidance_scale=1.0)", "arm": "entmax_l2", "n_prompts": len(prompts), "n_seeds": len(seeds), "dense_mean": float(dense.mean()), "arm_mean": float(plad.mean()),
               "delta": float(delta.mean()), "ci_lo": lo, "ci_hi": hi, "frac_prompts_arm_better": float((delta > 0).mean()), "classification": classify(float(delta.mean()), lo, hi),
               "n_improved": int((delta > NEUTRAL[met]).sum()), "n_degraded": int((delta < -NEUTRAL[met]).sum()), "n_neutral": int((np.abs(delta) <= NEUTRAL[met]).sum()), "neutral_threshold": NEUTRAL[met],
               "n_prompts_delta_positive": int((delta > 0).sum()), "n_prompts_delta_negative": int((delta < 0).sum()),
               "pladis_delta_seed_consistency_mean_pairwise_r": float(np.mean(cors)) if cors else None}
        rows.append(row); summary["metrics"][met] = row
        for i, p in enumerate(prompts): per_prompt.setdefault(p, {})[met] = (float(dense[i]), float(plad[i]), float(delta[i]))
    if not rows: print("no complete data yet"); return
    with open(os.path.join(out_dir, "delta_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(out_dir, "analysis_summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "per_prompt_delta.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["prompt_id", "prompt"] + [f"{k}_{met}" for met in ORDER for k in ("dense", "pladis", "delta")]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for p in sorted(per_prompt):
            r = {"prompt_id": p, "prompt": prompt_text.get(p, "")}
            for met in ORDER:
                if met in per_prompt[p]: r[f"dense_{met}"], r[f"pladis_{met}"], r[f"delta_{met}"] = per_prompt[p][met]
            w.writerow(r)

    mets = [k for k in ORDER if k in summary["metrics"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
    # ---- primary figure: one PLADIS point per metric, 95 % CI, dashed zero = Dense, no title ----
    fig, axes = plt.subplots(1, len(mets), figsize=(1.75 * len(mets) + 0.5, 2.1), squeeze=False)
    for ax, met in zip(axes[0], mets):
        r = summary["metrics"][met]
        ax.errorbar(0, r["delta"], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]], fmt="s", color="#1f4e79", ms=5.5, capsize=4, lw=1.1)
        ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xlim(-0.7, 0.7); ax.set_xticks([0]); ax.set_xticklabels(["PLADIS λ=2\n− Dense"], fontsize=7.5)
        ax.set_title(LABEL[met], fontsize=8.5, loc="left"); ax.grid(axis="y", alpha=0.25, lw=0.5)
        lim = max(abs(r["ci_lo"]), abs(r["ci_hi"])) * 1.25; ax.set_ylim(-lim, lim)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Δ vs Dense (no CFG)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_no_cfg_pladis_delta.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig_no_cfg_pladis_delta.pdf"), bbox_inches="tight"); plt.close(fig)

    # ---- optional trend figure: previous sweep (RTX 2080) PLADIS deltas + this no-CFG point (separate hardware) ----
    sweep_tab = os.path.join(a.sweep_root, "analysis", "delta_table.csv")
    if not a.no_trend and os.path.exists(sweep_tab):
        sw = [r for r in csv.DictReader(open(sweep_tab, encoding="utf-8")) if r["arm"] == "entmax_l2"]
        sw_gpu = json.load(open(os.path.join(a.sweep_root, "experiment_manifest.json"), encoding="utf-8")).get("versions", {}).get("gpu", "previous run")
        cfgs = sorted({float(r["cfg"]) for r in sw}, reverse=True); xs = list(range(len(cfgs))); xn = len(cfgs)
        fig, axes = plt.subplots(1, len(mets), figsize=(2.3 * len(mets) + 0.4, 2.5), squeeze=False)
        for ax, met in zip(axes[0], mets):
            pts = [next((r for r in sw if r["metric"] == met and float(r["cfg"]) == c), None) for c in cfgs]
            if all(p is not None for p in pts):
                y = [float(p["delta"]) for p in pts]; lo = [float(p["ci_lo"]) for p in pts]; hi = [float(p["ci_hi"]) for p in pts]
                ax.errorbar(xs, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="s-", color="#1f4e79", ms=4.5, capsize=3, lw=1.0, label=f"CFG sweep ({sw_gpu})")
            r = summary["metrics"][met]
            ax.errorbar(xn, r["delta"], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]], fmt="o", mfc="white", color="#c0392b", ms=6, capsize=3, lw=1.0, label=f"no CFG ({summary['gpu']}, separate run)")
            ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xticks(xs + [xn]); ax.set_xticklabels([f"CFG {c:g}" for c in cfgs] + ["no CFG*"], fontsize=7.5)
            ax.set_title(LABEL[met], fontsize=8.5, loc="left"); ax.grid(axis="y", alpha=0.25, lw=0.5)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        axes[0][0].set_ylabel("PLADIS λ=2 − Dense", fontsize=8)
        h, l = axes[0][0].get_legend_handles_labels(); fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))   # outside the axes
        fig.text(0.5, -0.13, f"* no-CFG audio generated on {summary['gpu']}; CFG-sweep audio on {sw_gpu}. Each Δ is paired within its own run; the two runs are not hardware-identical.", ha="center", fontsize=6.5)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_cfg_to_nocfg_trend.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig_cfg_to_nocfg_trend.pdf"), bbox_inches="tight"); plt.close(fig)

    print(f"{'metric':5s} {'n':>3s} {'dense':>7s} {'pladis':>7s} {'delta':>8s}  95% CI              better  class                     up/down/neutral(|Δ|<=thr)  seed-r")
    for r in rows:
        print(f"{r['metric']:5s} {r['n_prompts']:3d} {r['dense_mean']:7.3f} {r['arm_mean']:7.3f} {r['delta']:+8.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['frac_prompts_arm_better']:.2f}   {r['classification']:25s} {r['n_improved']}/{r['n_degraded']}/{r['n_neutral']} (thr {r['neutral_threshold']})  {r['pladis_delta_seed_consistency_mean_pairwise_r']}")


if __name__ == "__main__":
    main()
