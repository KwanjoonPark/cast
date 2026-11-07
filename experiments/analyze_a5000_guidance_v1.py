"""Unified same-hardware (RTX A5000) guidance analysis: PLADIS (entmax_l2) − Dense per guidance setting.
 1. A5000 CFG reproduction (<root> = outputs/a5000_guidance_sweep_v1, CFG 7.5 / 3.0 / 1.5): paired per-prompt deltas with the CAST
    protocol imported from analyze_cfg_sweep_v1 (same load(), same boot_ci(), 10,000 prompt-level resamples, rng seed 20261024;
    metrics in loading order clap, pq, aqa; CFG descending) -> <root>/analysis/{delta_table.csv, analysis_summary.json, per_prompt_delta.csv}.
 2. Unified table: the three CFG rows above + the already-analysed true no-CFG row copied from <nocfg>/analysis/delta_table.csv
    (same GPU, same protocol; not recomputed) -> <root>/analysis/delta_table_unified.csv (authoritative source for the paper figure).
 3. Hardware reproduction comparison against the RTX 2080 sweep (<sweep>/analysis/delta_table.csv, entmax_l2 rows + dense means)
    -> <root>/analysis/hardware_reproduction_comparison.csv.
 4. Diagnostic figure <root>/analysis/fig_a5000_guidance_pladis.{png,pdf} (ordered categories 7.5, 3.0, 1.5, no CFG; same GPU).
Usage: python experiments/analyze_a5000_guidance_v1.py [--root outputs/a5000_guidance_sweep_v1] [--nocfg outputs/no_cfg_v1] [--sweep outputs/cfg_sweep_v1]"""
import argparse, csv, itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_cfg_sweep_v1 import load, boot_ci, B, rng_seed

ARMS = ("dense", "entmax_l2"); ORDER = ("clap", "aqa", "pq"); LABEL = {"clap": "ΔCLAP", "aqa": "ΔAQAScore", "pq": "ΔPQ (AudioBox)"}


def wcsv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/a5000_guidance_sweep_v1"); ap.add_argument("--nocfg", default="outputs/no_cfg_v1"); ap.add_argument("--sweep", default="outputs/cfg_sweep_v1")
    a = ap.parse_args(); out = os.path.join(a.root, "analysis"); os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(a.root, "experiment_manifest.json"), encoding="utf-8")); gpu = man["versions"]["gpu"]
    nman = json.load(open(os.path.join(a.nocfg, "experiment_manifest.json"), encoding="utf-8")); sman = json.load(open(os.path.join(a.sweep, "experiment_manifest.json"), encoding="utf-8"))
    assert nman["versions"]["gpu"] == gpu, f"no-CFG run GPU {nman['versions']['gpu']} != {gpu}"
    for k in ("num_inference_steps", "height", "width", "dtype", "alpha", "deterministic_self_attn", "checkpoint", "apply_to_uncond"):
        assert man["generation_settings"][k] == nman["generation_settings"][k], f"setting {k} differs between CFG reproduction and no-CFG run"
    assert man["initial_latent_sha256_by_seed"] == nman["initial_latent_sha256_by_seed"] and man["prompt_file"] == nman["prompt_file"] and man["seeds"] == nman["seeds"]
    prompt_text = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(os.path.join(a.root, "audio_index.csv"), encoding="utf-8"))}

    # ---- 1. per-CFG paired analysis on the A5000 reproduction ----
    m = load(a.root); rng = np.random.default_rng(rng_seed); rows, per_prompt = [], {}
    summary = {"experiment": man["experiment"], "gpu": gpu, "cfgs": sorted(man["cfgs"], reverse=True), "arms": list(ARMS), "bootstrap": {"resamples": B, "rng_seed": rng_seed, "unit": "prompt (seeds averaged)", "rng_consumption_order": [f"{met}:{c}" for met in m for c in sorted(man["cfgs"], reverse=True)]}, "metrics": {}, "diagnostics": {}}
    for met, d in m.items():
        for cfg in sorted(man["cfgs"], reverse=True):
            g = {}
            for (c, pid, seed, arm), v in d.items():
                if c == cfg: g.setdefault(pid, {}).setdefault(seed, {})[arm] = v
            prompts = sorted(p for p, s in g.items() if all(all(arm in s[sd] for arm in ARMS) for sd in s)); seeds = sorted({sd for p in prompts for sd in g[p]})
            dense = np.array([np.mean([g[p][sd]["dense"] for sd in g[p]]) for p in prompts]); plad = np.array([np.mean([g[p][sd]["entmax_l2"] for sd in g[p]]) for p in prompts])
            delta = plad - dense; lo, hi = boot_ci(delta, rng)
            dseed = {sd: np.array([g[p][sd]["entmax_l2"] - g[p][sd]["dense"] for p in prompts if sd in g[p]]) for sd in seeds}; full = [sd for sd in seeds if len(dseed[sd]) == len(prompts)]
            cors = [float(np.corrcoef(dseed[x], dseed[y])[0, 1]) for x, y in itertools.combinations(full, 2)] if len(full) >= 2 else []
            row = {"metric": met, "guidance": f"CFG {cfg:g}", "cfg": cfg, "arm": "entmax_l2", "n_prompts": len(prompts), "n_seeds": len(seeds), "dense_mean": float(dense.mean()), "arm_mean": float(plad.mean()), "delta": float(delta.mean()),
                   "ci_lo": lo, "ci_hi": hi, "frac_prompts_arm_better": float((delta > 0).mean()), "gpu": gpu}
            rows.append(row); summary["metrics"].setdefault(met, {})[f"{cfg:g}"] = row
            summary["diagnostics"].setdefault(met, {})[f"{cfg:g}"] = {"pladis_delta_seed_consistency_mean_pairwise_r": float(np.mean(cors)) if cors else None}
            for i, p in enumerate(prompts): per_prompt.setdefault((p, cfg), {})[met] = (float(dense[i]), float(plad[i]), float(delta[i]))
    if not rows: print("no complete data yet"); return
    wcsv(os.path.join(out, "delta_table.csv"), rows); json.dump(summary, open(os.path.join(out, "analysis_summary.json"), "w"), indent=1)
    pp = []
    for (p, cfg) in sorted(per_prompt, key=lambda k: (-k[1], k[0])):
        r = {"prompt_id": p, "cfg": cfg, "prompt": prompt_text.get(p, "")}
        for met in ORDER:
            if met in per_prompt[(p, cfg)]: r[f"dense_{met}"], r[f"pladis_{met}"], r[f"delta_{met}"] = per_prompt[(p, cfg)][met]
        pp.append(r)
    wcsv(os.path.join(out, "per_prompt_delta.csv"), pp)

    # ---- 2. unified table (A5000 CFG rows + A5000 no-CFG row, copied) ----
    nrows = [r for r in csv.DictReader(open(os.path.join(a.nocfg, "analysis", "delta_table.csv"), encoding="utf-8")) if r["arm"] == "entmax_l2"]
    uni = [{k: r[k] for k in ("metric", "guidance", "cfg", "dense_mean", "arm_mean", "delta", "ci_lo", "ci_hi", "frac_prompts_arm_better", "n_prompts")} | {"source": a.root, "gpu": gpu} for r in rows]
    for r in nrows:
        uni.append({"metric": r["metric"], "guidance": "No CFG", "cfg": 1.0, "dense_mean": r["dense_mean"], "arm_mean": r["arm_mean"], "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "frac_prompts_arm_better": r["frac_prompts_arm_better"], "n_prompts": r["n_prompts"], "source": a.nocfg, "gpu": gpu})
    order = {"CFG 7.5": 0, "CFG 3": 1, "CFG 1.5": 2, "No CFG": 3}; uni.sort(key=lambda r: (ORDER.index(r["metric"]), order[r["guidance"]]))
    wcsv(os.path.join(out, "delta_table_unified.csv"), uni)

    # ---- 3. hardware reproduction comparison (RTX 2080 sweep vs A5000 reproduction) ----
    sw = {(r["metric"], float(r["cfg"])): r for r in csv.DictReader(open(os.path.join(a.sweep, "analysis", "delta_table.csv"), encoding="utf-8")) if r["arm"] == "entmax_l2"}
    comp = []
    for r in rows:
        s = sw[(r["metric"], r["cfg"])]; d2, l2, h2 = float(s["delta"]), float(s["ci_lo"]), float(s["ci_hi"]); dA, lA, hA = r["delta"], r["ci_lo"], r["ci_hi"]
        sig = lambda lo, hi: "neg" if hi < 0 else ("pos" if lo > 0 else "n.s.")
        comp.append({"metric": r["metric"], "cfg": r["cfg"], "dense_mean_rtx2080": float(s["dense_mean"]), "dense_mean_a5000": r["dense_mean"], "dense_diff_a5000_minus_2080": r["dense_mean"] - float(s["dense_mean"]),
                     "pladis_mean_rtx2080": float(s["arm_mean"]), "pladis_mean_a5000": r["arm_mean"], "delta_rtx2080": d2, "ci_rtx2080": f"[{l2:+.3f}, {h2:+.3f}]", "delta_a5000": dA, "ci_a5000": f"[{lA:+.3f}, {hA:+.3f}]",
                     "delta_diff_a5000_minus_2080": dA - d2, "same_sign": (dA < 0) == (d2 < 0), "a5000_delta_inside_2080_ci": l2 <= dA <= h2, "cis_overlap": max(l2, lA) <= min(h2, hA),
                     "significance_rtx2080": sig(l2, h2), "significance_a5000": sig(lA, hA), "gpu_rtx2080": sman["versions"]["gpu"], "gpu_a5000": gpu})
    wcsv(os.path.join(out, "hardware_reproduction_comparison.csv"), comp)

    # ---- 4. diagnostic figure (same GPU, four ordered guidance settings) ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), squeeze=False); cats = ["CFG 7.5", "CFG 3", "CFG 1.5", "No CFG"]
    for j, met in enumerate(ORDER):
        ax = axes[0][j]; rs = sorted([r for r in uni if r["metric"] == met], key=lambda r: order[r["guidance"]]); x = list(range(4))
        y = [float(r["delta"]) for r in rs]; lo = [float(r["ci_lo"]) for r in rs]; hi = [float(r["ci_hi"]) for r in rs]
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="s-", color="#1f4e79", ms=4.5, capsize=3, lw=1.1)
        ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(["7.5", "3.0", "1.5", "no CFG"]); ax.set_xlabel("guidance setting"); ax.set_title(LABEL[met], fontsize=8.5, loc="left"); ax.grid(axis="y", alpha=0.25, lw=0.5)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axes[0][0].set_ylabel(f"PLADIS λ=2 − Dense ({gpu.replace('NVIDIA ', '')})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_a5000_guidance_pladis.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out, "fig_a5000_guidance_pladis.pdf"), bbox_inches="tight"); plt.close(fig)

    print(f"{'metric':5s} {'guidance':8s} {'n':>3s} {'dense':>7s} {'pladis':>7s} {'delta':>8s}  95% CI              better  source")
    for r in uni: print(f"{r['metric']:5s} {r['guidance']:8s} {int(r['n_prompts']):3d} {float(r['dense_mean']):7.3f} {float(r['arm_mean']):7.3f} {float(r['delta']):+8.3f}  [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]  {float(r['frac_prompts_arm_better']):.2f}   {r['source']}")
    print("\nhardware comparison (A5000 − RTX 2080):")
    for c in comp: print(f"  {c['metric']:5s} CFG {c['cfg']:<4g} dense {c['dense_mean_rtx2080']:.3f}->{c['dense_mean_a5000']:.3f} ({c['dense_diff_a5000_minus_2080']:+.3f}) | delta {c['delta_rtx2080']:+.3f}->{c['delta_a5000']:+.3f} ({c['delta_diff_a5000_minus_2080']:+.3f}) same_sign={c['same_sign']} inside2080CI={c['a5000_delta_inside_2080_ci']} sig {c['significance_rtx2080']}->{c['significance_a5000']}")


if __name__ == "__main__":
    main()
