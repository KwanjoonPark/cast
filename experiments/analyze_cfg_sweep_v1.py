"""CFG sweep analysis: delta vs Dense per CFG level for CLAP / AudioBox PQ / (AQAScore if scored), prompt-level bootstrap CIs, figures.

Unit of analysis = prompt (seeds averaged); only (cfg, prompt, seed) groups with all arms present are used. Delta = arm - dense (same cfg, same prompt).
95% CI: 10,000 prompt-level bootstrap resamples (fixed rng seed 20261024). Extra diagnostics: seed consistency of per-prompt PLADIS delta (mean pairwise
Pearson r between seeds) and the Spearman correlation between a prompt's Dense score and its PLADIS delta.
Inputs (all optional except CLAP): <root>/audio_index.csv, <root>/clap_scores.csv, <root>/audiobox/sample_scores.csv, <root>/aqascore/sample_scores.csv
Usage: python experiments/analyze_cfg_sweep_v1.py [--root outputs/cfg_sweep_v1]"""
import argparse, csv, itertools, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ARMS = ["dense", "sharp_b1.5", "entmax_l0.5", "entmax_l1", "entmax_l2"]
LABEL = {"sharp_b1.5": "Sharp\nβ=1.5", "entmax_l0.5": "Entmax\nλ=0.5", "entmax_l1": "Entmax\nλ=1", "entmax_l2": "PLADIS\nλ=2"}
SHORT = {"sharp_b1.5": "Sharp β=1.5", "entmax_l0.5": "Entmax λ=0.5", "entmax_l1": "Entmax λ=1", "entmax_l2": "PLADIS λ=2"}
COL = {"sharp_b1.5": "#4a4a4a", "entmax_l0.5": "#6a9fcf", "entmax_l1": "#2b6cb0", "entmax_l2": "#1f4e79"}
METRIC_TITLE = {"clap": "ΔCLAP", "pq": "ΔPQ (AudioBox)", "aqa": "ΔAQAScore"}
rng_seed, B = 20261024, 10000


def load(root):
    idx = {r["audio_path"]: r for r in csv.DictReader(open(os.path.join(root, "audio_index.csv"), encoding="utf-8"))}
    m = {}   # metric -> {(cfg, prompt, seed, arm): value}
    def add(name, path, col):
        if not os.path.exists(path): return
        d = {}
        for r in csv.DictReader(open(path, encoding="utf-8")):
            i = idx.get(r["audio_path"].replace("\\", "/"))
            if i is None: continue
            d[(float(i["cfg"]), i["prompt_id"], int(i["seed"]), i["arm"])] = float(r[col])
        m[name] = d
    add("clap", os.path.join(root, "clap_scores.csv"), "clap")
    add("pq", os.path.join(root, "audiobox", "sample_scores.csv"), "PQ")
    add("aqa", os.path.join(root, "aqascore", "sample_scores.csv"), "aqascore_full")
    return m


def prompt_table(d):
    """-> {cfg: {arm: {prompt: [values by seed]}}} using only complete (cfg,prompt,seed) groups."""
    groups = {}
    for (cfg, pid, seed, arm), v in d.items(): groups.setdefault((cfg, pid, seed), {})[arm] = v
    t = {}
    for (cfg, pid, seed), arms in groups.items():
        if not all(a in arms for a in ARMS): continue
        for a in ARMS: t.setdefault(cfg, {}).setdefault(a, {}).setdefault(pid, {})[seed] = arms[a]
    return t


def boot_ci(x, rng):
    x = np.asarray(x, float); n = len(x)
    means = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ranks(x):
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/cfg_sweep_v1"); a = ap.parse_args()
    m = load(a.root); rng = np.random.default_rng(rng_seed); out_dir = os.path.join(a.root, "analysis"); os.makedirs(out_dir, exist_ok=True)
    rows, summary = [], {"bootstrap": {"resamples": B, "rng_seed": rng_seed, "unit": "prompt (seeds averaged)"}, "metrics": {}, "diagnostics": {}}
    tabs = {k: prompt_table(v) for k, v in m.items()}
    for met, t in tabs.items():
        for cfg in sorted(t, reverse=True):
            prompts = sorted(set.intersection(*[set(t[cfg][arm]) for arm in ARMS]))
            seeds_per = [len(t[cfg]["dense"][p]) for p in prompts]
            val = {arm: np.array([np.mean(list(t[cfg][arm][p].values())) for p in prompts]) for arm in ARMS}
            summary["metrics"].setdefault(met, {})[f"{cfg:g}"] = {"n_prompts": len(prompts), "min_seeds": int(min(seeds_per)) if seeds_per else 0, "dense_mean": float(val["dense"].mean())}
            for arm in ARMS[1:]:
                d = val[arm] - val["dense"]; lo, hi = boot_ci(d, rng)
                rows.append({"metric": met, "cfg": cfg, "arm": arm, "n_prompts": len(prompts), "dense_mean": val["dense"].mean(), "arm_mean": val[arm].mean(), "delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "frac_prompts_arm_better": float((d > 0).mean())})
            # diagnostics for PLADIS
            dseed = {s: np.array([t[cfg]["entmax_l2"][p][s] - t[cfg]["dense"][p][s] for p in prompts if s in t[cfg]["dense"][p] and s in t[cfg]["entmax_l2"][p]]) for s in sorted({s for p in prompts for s in t[cfg]["dense"][p]})}
            full = [s for s, v in dseed.items() if len(v) == len(prompts)]
            cors = [float(np.corrcoef(dseed[x], dseed[y])[0, 1]) for x, y in itertools.combinations(full, 2)] if len(full) >= 2 else []
            dl = val["entmax_l2"] - val["dense"]; sp = float(np.corrcoef(ranks(val["dense"]), ranks(dl))[0, 1]) if len(prompts) > 2 else None
            summary["diagnostics"].setdefault(met, {})[f"{cfg:g}"] = {"pladis_delta_seed_consistency_mean_pairwise_r": float(np.mean(cors)) if cors else None, "n_full_seeds": len(full),
                                                                       "spearman_dense_score_vs_pladis_delta": sp}
    if not rows: print("no complete data yet"); return
    with open(os.path.join(out_dir, "delta_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(out_dir, "analysis_summary.json"), "w"), indent=1)

    cfgs = sorted({r["cfg"] for r in rows}, reverse=True); mets = [k for k in ("clap", "pq", "aqa") if k in tabs and any(r["metric"] == k for r in rows)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
    # Figure A: Table-1 style grid, rows = CFG (high -> low), cols = metric, shared y per column
    fig, axes = plt.subplots(len(cfgs), len(mets), figsize=(2.3 * len(mets) + 0.6, 1.9 * len(cfgs) + 0.3), squeeze=False, sharey="col")
    for j, met in enumerate(mets):
        for i, cfg in enumerate(cfgs):
            ax = axes[i][j]
            for k, arm in enumerate(ARMS[1:]):
                r = next((x for x in rows if x["metric"] == met and x["cfg"] == cfg and x["arm"] == arm), None)
                if r is None: continue
                ax.errorbar(k, r["delta"], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]], fmt="s" if arm != "sharp_b1.5" else "o", color=COL[arm], ms=4.5, capsize=3, lw=1.0)
            ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xlim(-0.6, 3.6); ax.grid(axis="y", alpha=0.25, lw=0.5)
            ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[x] for x in ARMS[1:]] if i == len(cfgs) - 1 else [], fontsize=6.5)
            if i == 0: ax.set_title(METRIC_TITLE[met], fontsize=8.5, loc="left")
            if j == 0: ax.set_ylabel(f"CFG {cfg:.1f}\nΔ vs Dense", fontsize=8)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_cfg_delta_grid.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig_cfg_delta_grid.pdf"), bbox_inches="tight"); plt.close(fig)
    # Figure B: delta vs CFG (lines with CI), plus Dense baseline vs CFG
    fig, axes = plt.subplots(2, len(mets), figsize=(2.6 * len(mets) + 0.4, 4.2), squeeze=False)
    xs = sorted(cfgs)
    for j, met in enumerate(mets):
        ax = axes[0][j]
        for arm in ARMS[1:]:
            pts = [(c, next(x for x in rows if x["metric"] == met and x["cfg"] == c and x["arm"] == arm)) for c in xs if any(x["metric"] == met and x["cfg"] == c and x["arm"] == arm for x in rows)]
            if not pts: continue
            ax.plot([p[0] for p in pts], [p[1]["delta"] for p in pts], marker="o", ms=3.5, color=COL[arm], lw=1.1, label=SHORT[arm])
            ax.fill_between([p[0] for p in pts], [p[1]["ci_lo"] for p in pts], [p[1]["ci_hi"] for p in pts], color=COL[arm], alpha=0.10, lw=0)
        ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_title(METRIC_TITLE[met], fontsize=8.5, loc="left"); ax.set_xlabel("CFG scale"); ax.set_xticks(xs); ax.grid(alpha=0.25, lw=0.5)
        if j == 0: ax.set_ylabel("Δ vs Dense")
        ax2 = axes[1][j]; dm = [summary["metrics"][met][f"{c:g}"]["dense_mean"] for c in xs if f"{c:g}" in summary["metrics"][met]]
        ax2.plot([c for c in xs if f"{c:g}" in summary["metrics"][met]], dm, marker="s", ms=3.5, color="black", lw=1.1)
        ax2.set_xlabel("CFG scale"); ax2.set_xticks(xs); ax2.grid(alpha=0.25, lw=0.5); ax2.set_title(f"Dense {METRIC_TITLE[met][1:].split(' ')[0]} (baseline)", fontsize=8.5, loc="left")
        for ax_ in (ax, ax2):
            for s in ("top", "right"): ax_.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=6, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cfg_delta_lines.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig_cfg_delta_lines.pdf"), bbox_inches="tight"); plt.close(fig)

    print(f"{'metric':5s} {'cfg':>4s} {'arm':12s} {'n':>3s} {'dense':>7s} {'arm':>7s} {'delta':>8s}  95% CI              better")
    for r in rows:
        print(f"{r['metric']:5s} {r['cfg']:4g} {r['arm']:12s} {r['n_prompts']:3d} {r['dense_mean']:7.3f} {r['arm_mean']:7.3f} {r['delta']:+8.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['frac_prompts_arm_better']:.2f}")
    print(json.dumps(summary["diagnostics"], indent=1))


if __name__ == "__main__":
    main()
