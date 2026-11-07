"""Final paper assets for the 2-page CAST paper (figures, tables, source notes, previews) from EXISTING analysis tables only.
No audio is generated, no model is run, nothing is recomputed: every number is read from
  outputs/cfg_sweep_v1/analysis/delta_table.csv   (CFG 7.5 / 3.0 / 1.5; RTX 2080 run; prompt-level paired bootstrap, 10,000 resamples, rng 20261024)
  outputs/no_cfg_v1/analysis/delta_table.csv      (true no-CFG, guidance_scale=1.0; RTX A5000 run; same protocol)
plus the two experiment_manifest.json files (hardware names). Sharp Softmax is intentionally excluded from every asset here.
Outputs -> paper_assets/final/{figures,tables,source,previews}. Deterministic. Usage: python experiments/make_final_paper_assets.py"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SWEEP = os.path.join(ROOT, "outputs", "cfg_sweep_v1"); NOCFG = os.path.join(ROOT, "outputs", "no_cfg_v1")
OUT = os.path.join(ROOT, "paper_assets", "final"); FIG, TAB, SRC, PRE = [os.path.join(OUT, d) for d in ("figures", "tables", "source", "previews")]
METRICS = ("clap", "aqa", "pq"); MLABEL = {"clap": "ΔCLAP", "aqa": "ΔAQAScore", "pq": "ΔPQ (AudioBox)"}; MNAME = {"clap": "CLAP", "aqa": "AQAScore", "pq": "PQ"}
ARMS = ("entmax_l0.5", "entmax_l1", "entmax_l2"); ANAME = {"dense": "Dense", "entmax_l0.5": "Entmax λ=0.5", "entmax_l1": "Entmax λ=1", "entmax_l2": "PLADIS λ=2"}
ANAME2 = {"entmax_l0.5": "Entmax\nλ=0.5", "entmax_l1": "Entmax\nλ=1", "entmax_l2": "PLADIS\nλ=2"}
COL = {"entmax_l0.5": "#6a9fcf", "entmax_l1": "#2b6cb0", "entmax_l2": "#1f4e79"}; NOCFG_COL = "#c0392b"
CFGS = (7.5, 3.0, 1.5); DEC = {"clap": 3, "aqa": 3, "pq": 2}
SCRIPT = "experiments/make_final_paper_assets.py"


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def f(x, d): return f"{x:+.{d}f}"
def ci(r, d): return f"{f(float(r['delta']), d)} [{f(float(r['ci_lo']), d)}, {f(float(r['ci_hi']), d)}]"
def excl0(r): return float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0
def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42, "savefig.facecolor": "white"})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.03); fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
def eb(ax, x, r, **kw):
    d, lo, hi = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]); ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], **kw)
def ylim_for(ax, rs, pad=0.18):
    lo = min(min(float(r["ci_lo"]) for r in rs), 0.0); hi = max(max(float(r["ci_hi"]) for r in rs), 0.0); span = hi - lo or 1.0; ax.set_ylim(lo - pad * span, hi + pad * span)
def preview(name, widths):
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(os.path.join(FIG, name + ".png"))
    for tag, w in widths.items():
        im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save(os.path.join(PRE, f"{name}_{tag}.png"))


def main():
    for d in (FIG, TAB, SRC, PRE): os.makedirs(d, exist_ok=True)
    sw = rows(os.path.join(SWEEP, "analysis", "delta_table.csv")); nc = rows(os.path.join(NOCFG, "analysis", "delta_table.csv"))
    gpu_sw = json.load(open(os.path.join(SWEEP, "experiment_manifest.json")))["versions"]["gpu"]; gpu_nc = json.load(open(os.path.join(NOCFG, "experiment_manifest.json")))["versions"]["gpu"]
    boot = json.load(open(os.path.join(SWEEP, "analysis", "analysis_summary.json")))["bootstrap"]
    S = {(r["metric"], float(r["cfg"]), r["arm"]): r for r in sw}; N = {r["metric"]: r for r in nc if r["arm"] == "entmax_l2"}
    pl = {m: [S[(m, c, "entmax_l2")] for c in CFGS] for m in METRICS}
    protocol = (f"prompt = unit of analysis (3 seeds averaged); Δ = condition − Dense at the same guidance setting and prompt; 95 % CI from {boot['resamples']:,} prompt-level bootstrap resamples, rng seed {boot['rng_seed']}; "
                "24 prompts (experiments/prompts/cast_diagnostic_v1.json), seeds 42/123/2026, 100-step PNDM, 256×1024 mel, fp16, α=1.5, deterministic self-attention, 16 attn2 modules only.")
    style()

    # ---------------- Figure 1A: 3 x 3 grid (two-column width) ----------------
    fig, axes = plt.subplots(3, 3, figsize=(6.8, 4.5), sharey="col", squeeze=False)
    for j, m in enumerate(METRICS):
        for i, c in enumerate(CFGS):
            ax = axes[i][j]; rs = [S[(m, c, a)] for a in ARMS]
            for k, (a, r) in enumerate(zip(ARMS, rs)): eb(ax, k, r, fmt="s", color=COL[a], ms=4.2, capsize=3, lw=1.0)
            ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xlim(-0.6, 2.6); ax.set_xticks(range(3)); clean(ax)
            ax.set_xticklabels([ANAME2[a] for a in ARMS] if i == 2 else [], fontsize=7.5)
            if i == 0: ax.set_title(MLABEL[m], fontsize=8.5, loc="left")
            if j == 0: ax.set_ylabel(f"CFG {c:.1f}\nΔ vs Dense", fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.8); save(fig, "fig1_cfg_delta_grid"); preview("fig1_cfg_delta_grid", {"1col": 1000, "2col": 2000})

    # ---------------- Figure 1B: compact PLADIS trend over CFG (one-column width) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.75), squeeze=False)
    for j, m in enumerate(METRICS):
        ax = axes[0][j]; rs = pl[m]; x = list(range(3))
        y = [float(r["delta"]) for r in rs]; lo = [float(r["ci_lo"]) for r in rs]; hi = [float(r["ci_hi"]) for r in rs]
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="s-", color=COL["entmax_l2"], ms=3.6, capsize=2.5, lw=1.0, elinewidth=0.9)
        ax.axhline(0, color="black", lw=0.7, ls="--"); ax.set_xlim(-0.45, 2.45); ax.set_xticks(x); ax.set_xticklabels([f"{c:.1f}" for c in CFGS], fontsize=7); ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"({'abc'[j]}) {MLABEL[m].replace(' (AudioBox)', '')}", fontsize=8, loc="left"); clean(ax); ylim_for(ax, rs)
        ax.set_xlabel("CFG scale", fontsize=7.5)
    axes[0][0].set_ylabel("PLADIS λ=2 − Dense", fontsize=7.5)
    fig.tight_layout(w_pad=0.7); save(fig, "fig1_cfg_pladis_trend"); preview("fig1_cfg_pladis_trend", {"1col": 1000})

    # ---------------- Figure 2: true no-CFG paired result (one-column width) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.65), squeeze=False)
    for j, m in enumerate(METRICS):
        ax = axes[0][j]; r = N[m]; eb(ax, 0, r, fmt="s", color=COL["entmax_l2"], ms=4.5, capsize=3.5, lw=1.0)
        ax.axhline(0, color="black", lw=0.7, ls="--"); ax.set_xlim(-0.7, 0.7); ax.set_xticks([0]); ax.set_xticklabels(["PLADIS λ=2"], fontsize=7); ax.tick_params(axis="y", labelsize=7)
        lim = max(abs(float(r["ci_lo"])), abs(float(r["ci_hi"]))) * 1.3; ax.set_ylim(-lim, lim)
        ax.set_title(f"({'abc'[j]}) {MLABEL[m].replace(' (AudioBox)', '')}", fontsize=8, loc="left"); clean(ax)
    axes[0][0].set_ylabel("Δ vs Dense (no CFG)", fontsize=7.5)
    fig.tight_layout(w_pad=0.7); save(fig, "fig2_nocfg_delta"); preview("fig2_nocfg_delta", {"1col": 1000})

    # ---------------- Optional: CFG -> no-CFG descriptive trend (no-CFG visually separated) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.9), squeeze=False)
    for j, m in enumerate(METRICS):
        ax = axes[0][j]; rs = pl[m]; x = list(range(3))
        y = [float(r["delta"]) for r in rs]; lo = [float(r["ci_lo"]) for r in rs]; hi = [float(r["ci_hi"]) for r in rs]
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="s-", color=COL["entmax_l2"], ms=3.4, capsize=2.5, lw=1.0, elinewidth=0.9)
        ax.axvline(2.5, color="#999999", lw=0.7, ls=":"); eb(ax, 3, N[m], fmt="o", mfc="white", color=NOCFG_COL, ms=4.2, capsize=2.5, lw=1.0)
        ax.axhline(0, color="black", lw=0.7, ls="--"); ax.set_xlim(-0.45, 3.45); ax.set_xticks(x + [3]); ax.set_xticklabels([f"{c:g}" for c in CFGS] + ["no\nCFG†"], fontsize=6.8)   # short labels: four ticks in a one-column panel; ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"({'abc'[j]}) {MLABEL[m].replace(' (AudioBox)', '')}", fontsize=8, loc="left"); clean(ax); ylim_for(ax, rs + [N[m]]); ax.set_xlabel("CFG scale", fontsize=7.5)
    axes[0][0].set_ylabel("PLADIS λ=2 − Dense", fontsize=7.5)
    fig.text(0.99, -0.04, f"† no CFG: separate GPU run ({gpu_nc.replace('NVIDIA ', '')}); CFG points: {gpu_sw.replace('NVIDIA GeForce ', '')}", ha="right", fontsize=6.3, color="#333333")
    # DEPRECATED_FOR_PAPER: hardware-mixed trend (RTX 2080 CFG points + A5000 no-CFG point). Kept for provenance only; superseded by fig1_guidance_pladis_a5000.
    for ext in ("png", "pdf"):
        old = os.path.join(FIG, f"fig_optional_cfg_to_nocfg.{ext}")
        if os.path.exists(old): os.remove(old)
    old_prev = os.path.join(PRE, "fig_optional_cfg_to_nocfg_1col.png")
    if os.path.exists(old_prev): os.remove(old_prev)
    fig.tight_layout(w_pad=0.7); save(fig, "fig_optional_cfg_to_nocfg_DEPRECATED_FOR_PAPER"); preview("fig_optional_cfg_to_nocfg_DEPRECATED_FOR_PAPER", {"1col": 1000})

    # ---------------- Same-hardware (A5000) unified guidance figure + tables (only if the reproduction analysis exists) ----------------
    uni_path = os.path.join(ROOT, "outputs", "a5000_guidance_sweep_v1", "analysis", "delta_table_unified.csv")
    if os.path.exists(uni_path):
        uni = rows(uni_path); GORDER = ["CFG 7.5", "CFG 3", "CFG 1.5", "No CFG"]; GLABEL = {"CFG 7.5": "7.5", "CFG 3": "3.0", "CFG 1.5": "1.5", "No CFG": "no\nCFG"}; GNAME = {"CFG 7.5": "CFG 7.5", "CFG 3": "CFG 3.0", "CFG 1.5": "CFG 1.5", "No CFG": "No CFG"}
        U = {(r["metric"], r["guidance"]): r for r in uni}; gpu_u = uni[0]["gpu"]
        fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.85), squeeze=False)
        for j, m in enumerate(METRICS):
            ax = axes[0][j]; rs = [U[(m, g)] for g in GORDER]; x = list(range(4))
            y = [float(r["delta"]) for r in rs]; lo = [float(r["ci_lo"]) for r in rs]; hi = [float(r["ci_hi"]) for r in rs]
            ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="s-", color=COL["entmax_l2"], ms=3.4, capsize=2.5, lw=1.0, elinewidth=0.9)
            ax.axhline(0, color="black", lw=0.7, ls="--"); ax.set_xlim(-0.5, 3.5); ax.set_xticks(x); ax.set_xticklabels([GLABEL[g] for g in GORDER], fontsize=6.4); ax.tick_params(axis="y", labelsize=6.8); ax.tick_params(axis="x", pad=2)
            ax.set_title(f"({'abc'[j]}) {MLABEL[m].replace(' (AudioBox)', '')}", fontsize=8, loc="left"); clean(ax); ylim_for(ax, rs)
        axes[0][0].set_ylabel("PLADIS λ=2 − Dense", fontsize=7.5)
        fig.supxlabel("guidance setting (CFG scale; no CFG = conditional-only)", fontsize=6.8, y=-0.04)   # one shared label: ordered categories, not a numeric CFG axis
        fig.tight_layout(w_pad=0.6); save(fig, "fig1_guidance_pladis_a5000"); preview("fig1_guidance_pladis_a5000", {"1col": 1000})
        ufoot = f"All four settings generated and evaluated on the same GPU ({gpu_u}); CFG 7.5 / 3.0 / 1.5 from outputs/a5000_guidance_sweep_v1 (conditional + unconditional branch, standard CFG), No CFG from outputs/no_cfg_v1 (guidance_scale = 1.0, conditional-only inference). Same 24 prompts, seeds, initial latents, checkpoint, sampler and evaluators."
        with open(os.path.join(TAB, "table_a5000_guidance_delta.md"), "w", encoding="utf-8") as fh:
            fh.write("| Guidance | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] |\n|---|---|---|---|\n")
            for g in GORDER: fh.write(f"| {GNAME[g]} | " + " | ".join(ci(U[(m, g)], DEC[m]) for m in METRICS) + " |\n")
            fh.write(f"\nΔ = PLADIS λ=2 − Dense; {protocol} {ufoot}\n")
        with open(os.path.join(TAB, "table_a5000_guidance_delta.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["guidance", "metric", "dense_mean", "pladis_mean", "delta", "ci_lo", "ci_hi", "ci_excludes_zero", "frac_prompts_pladis_better", "n_prompts", "source", "gpu"])
            for g in GORDER:
                for m in METRICS:
                    r = U[(m, g)]; w.writerow([GNAME[g], MNAME[m], f"{float(r['dense_mean']):.6f}", f"{float(r['arm_mean']):.6f}", f"{float(r['delta']):.6f}", f"{float(r['ci_lo']):.6f}", f"{float(r['ci_hi']):.6f}", int(excl0(r)), f"{float(r['frac_prompts_arm_better']):.4f}", r["n_prompts"], r["source"], r["gpu"]])
        with open(os.path.join(TAB, "table_a5000_guidance_absolute.md"), "w", encoding="utf-8") as fh:
            fh.write("| Setting | Method | CLAP | AQAScore | PQ |\n|---|---|---:|---:|---:|\n")
            for g in GORDER:
                fh.write(f"| {GNAME[g]} | Dense | " + " | ".join(f"{float(U[(m, g)]['dense_mean']):.{DEC[m]}f}" for m in METRICS) + " |\n")
                fh.write(f"| {GNAME[g]} | PLADIS λ=2 | " + " | ".join(f"{float(U[(m, g)]['arm_mean']):.{DEC[m]}f}" for m in METRICS) + " |\n")
            fh.write(f"\nAbsolute means (24 prompts × 3 seeds). {ufoot} Reference table; not recommended for the 2-page paper.\n")
        with open(os.path.join(TAB, "table_a5000_guidance_absolute.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["guidance", "method", "clap", "aqascore", "pq", "gpu"])
            for g in GORDER:
                w.writerow([GNAME[g], "Dense"] + [f"{float(U[(m, g)]['dense_mean']):.6f}" for m in METRICS] + [gpu_u]); w.writerow([GNAME[g], "PLADIS λ=2"] + [f"{float(U[(m, g)]['arm_mean']):.6f}" for m in METRICS] + [gpu_u])
        for k, v in {"fig1_guidance_pladis_a5000": "Final Figure 1 (same hardware). Arm `entmax_l2` − Dense for the four ordered guidance settings CFG 7.5 / 3.0 / 1.5 / No CFG, one panel per metric, 95 % CI, dashed Dense; points connected as an ordered categorical trend (the x-axis is NOT a numeric CFG axis; 'no CFG' is conditional-only inference, not CFG = 0). One-column width (3.4 in).",
                     "table_a5000_guidance_delta": "Final delta table (authoritative source of Figure 1): `delta [ci_lo, ci_hi]` of arm `entmax_l2` per metric and guidance setting from the unified table; CSV adds `dense_mean`, `pladis_mean`, `ci_excludes_zero`, `frac_prompts_pladis_better`, `n_prompts`, `source`, `gpu`.",
                     "table_a5000_guidance_absolute": "Reference table: absolute `dense_mean` / `arm_mean` per guidance setting and metric from the unified table."}.items():
            with open(os.path.join(SRC, f"{k}_source.md"), "w", encoding="utf-8") as fh:
                fh.write(f"# {k}\n\n{v}\n\n- Source CSV: `outputs/a5000_guidance_sweep_v1/analysis/delta_table_unified.csv` (CFG rows computed by `experiments/analyze_a5000_guidance_v1.py` from `outputs/a5000_guidance_sweep_v1/{{clap_scores.csv, aqascore/sample_scores.csv, audiobox/sample_scores.csv}}`; the No CFG row copied from `outputs/no_cfg_v1/analysis/delta_table.csv`, same protocol, not recomputed).\n- Statistical protocol: {protocol}\n- Hardware: {gpu_u} for all four settings (`outputs/*/experiment_manifest.json` → `versions.gpu`).\n- Generated by `{SCRIPT}` (reads tables only; no audio, no inference).\n")
        print("A5000 unified assets written (fig1_guidance_pladis_a5000, table_a5000_guidance_delta, table_a5000_guidance_absolute)")
    else:
        print("A5000 unified table not found; same-hardware assets skipped")

    # ---------------- Tables ----------------
    foot = f"* No-CFG (guidance_scale = 1.0, conditional-only inference) was evaluated in a separate paired run on {gpu_nc}; CFG 7.5 / 3.0 / 1.5 were generated on {gpu_sw}. Absolute scores are not hardware-controlled across rows; each Δ is paired within its own run."
    guid = [("CFG 7.5", 7.5), ("CFG 3.0", 3.0), ("CFG 1.5", 1.5), ("No-CFG*", None)]
    def rec(m, c): return S[(m, c, "entmax_l2")] if c is not None else N[m]
    # 1A compact (point estimates; ‡ marks CI excluding 0)
    with open(os.path.join(TAB, "table1_pladis_guidance_compact.md"), "w", encoding="utf-8") as fh:
        fh.write("| Guidance | ΔCLAP | ΔAQAScore | ΔPQ |\n|---|---:|---:|---:|\n")
        for g, c in guid: fh.write(f"| {g} | " + " | ".join(f(float(rec(m, c)["delta"]), DEC[m]) + ("‡" if excl0(rec(m, c)) else "") for m in METRICS) + " |\n")
        fh.write(f"\nΔ = PLADIS λ=2 − Dense (24 prompts × 3 seeds, seeds averaged within prompt). ‡ 95 % prompt-level bootstrap CI excludes 0 (10,000 resamples). {foot}\n")
    with open(os.path.join(TAB, "table1_pladis_guidance_compact.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["guidance", "delta_clap", "delta_aqascore", "delta_pq", "ci_excludes_zero_clap", "ci_excludes_zero_aqascore", "ci_excludes_zero_pq", "generation_gpu"])
        for g, c in guid: w.writerow([g] + [f"{float(rec(m, c)['delta']):.6f}" for m in METRICS] + [int(excl0(rec(m, c))) for m in METRICS] + [gpu_sw if c is not None else gpu_nc])
    # 1B full (point estimate + CI)
    with open(os.path.join(TAB, "table1_pladis_guidance_full.md"), "w", encoding="utf-8") as fh:
        fh.write("| Guidance | ΔCLAP [95 % CI] | ΔAQAScore [95 % CI] | ΔPQ [95 % CI] |\n|---|---|---|---|\n")
        for g, c in guid: fh.write(f"| {g} | " + " | ".join(ci(rec(m, c), DEC[m]) for m in METRICS) + " |\n")
        fh.write(f"\nΔ = PLADIS λ=2 − Dense; {protocol} {foot}\n")
    with open(os.path.join(TAB, "table1_pladis_guidance_full.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["guidance", "metric", "dense_mean", "pladis_mean", "delta", "ci_lo", "ci_hi", "n_prompts", "generation_gpu"])
        for g, c in guid:
            for m in METRICS:
                r = rec(m, c); w.writerow([g, MNAME[m], f"{float(r['dense_mean']):.6f}", f"{float(r['arm_mean']):.6f}", f"{float(r['delta']):.6f}", f"{float(r['ci_lo']):.6f}", f"{float(r['ci_hi']):.6f}", r["n_prompts"], gpu_sw if c is not None else gpu_nc])
    # optional: CFG 7.5 absolute means by method (Dense, Entmax 0.5 / 1, PLADIS 2)
    with open(os.path.join(TAB, "table_optional_cfg75_methods.md"), "w", encoding="utf-8") as fh:
        fh.write("| Method (CFG 7.5) | CLAP | AQAScore | PQ |\n|---|---:|---:|---:|\n")
        fh.write("| Dense | " + " | ".join(f"{float(S[(m, 7.5, 'entmax_l2')]['dense_mean']):.{DEC[m]}f}" for m in METRICS) + " |\n")
        for a in ARMS: fh.write(f"| {ANAME[a]} | " + " | ".join(f"{float(S[(m, 7.5, a)]['arm_mean']):.{DEC[m]}f}" for m in METRICS) + " |\n")
        fh.write(f"\nAbsolute means, 24 prompts × 3 seeds, CFG 7.5 ({gpu_sw} run; the Dense / PLADIS CLAP and PQ means agree with the paper's original A5000 Table 1 to three decimals).\n")
    with open(os.path.join(TAB, "table_optional_cfg75_methods.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["method", "clap", "aqascore", "pq"])
        w.writerow(["Dense"] + [f"{float(S[(m, 7.5, 'entmax_l2')]['dense_mean']):.6f}" for m in METRICS])
        for a in ARMS: w.writerow([ANAME[a]] + [f"{float(S[(m, 7.5, a)]['arm_mean']):.6f}" for m in METRICS])

    # ---------------- Source / traceability notes ----------------
    common = (f"- Source CSV (CFG sweep): `outputs/cfg_sweep_v1/analysis/delta_table.csv` — columns `metric, cfg, arm, dense_mean, arm_mean, delta, ci_lo, ci_hi`; generated on {gpu_sw}.\n"
              f"- Source CSV (no-CFG): `outputs/no_cfg_v1/analysis/delta_table.csv` — same columns (`condition` = no_cfg, guidance_scale = 1.0); generated on {gpu_nc}.\n"
              f"- Hardware names: `outputs/*/experiment_manifest.json` → `versions.gpu`.\n- Statistical protocol (unchanged, from the analysis scripts): {protocol}\n- Generated by `{SCRIPT}` (reads tables only; no audio, no inference).\n")
    notes = {
        "fig1_cfg_delta_grid": "Figure 1A. Rows CFG 7.5 / 3.0 / 1.5 (sweep table), columns ΔCLAP / ΔAQAScore / ΔPQ, arms `entmax_l0.5`, `entmax_l1`, `entmax_l2` (Dense = dashed 0). Sharp β=1.5 excluded by design. Two-column width (6.8 in).",
        "fig1_cfg_pladis_trend": "Figure 1B. Arm `entmax_l2` only (PLADIS λ=2), x = CFG 7.5 / 3.0 / 1.5 from the sweep table, one panel per metric; `delta` with `ci_lo`/`ci_hi` error bars. One-column width (3.4 in).",
        "fig2_nocfg_delta": "Figure 2. Arm `entmax_l2` from the no-CFG table only (within-run paired PLADIS − Dense, A5000); one point per metric. One-column width (3.4 in). No sweep data used.",
        "fig_optional_cfg_to_nocfg_DEPRECATED_FOR_PAPER": "DEPRECATED_FOR_PAPER (provenance only; superseded by fig1_guidance_pladis_a5000). Sweep `entmax_l2` rows (RTX 2080, CFG 7.5 / 3.0 / 1.5, connected) plus the A5000 no-CFG `entmax_l2` row after a dotted separator as an open marker labelled `no CFG†` (separate GPU run). Descriptive only; the two runs are not hardware-identical.",
        "table1_pladis_guidance_compact": "Table 1A. `delta` of arm `entmax_l2` per metric for CFG 7.5 / 3.0 / 1.5 (sweep) and No-CFG; ‡ = `ci_lo` > 0 or `ci_hi` < 0. 3 decimals for CLAP/AQAScore, 2 for PQ.",
        "table1_pladis_guidance_full": "Table 1B. Same rows with `delta [ci_lo, ci_hi]`; CSV adds `dense_mean`, `pladis_mean` (= `arm_mean`), `n_prompts`, GPU.",
        "table_optional_cfg75_methods": "Optional table. CFG 7.5 rows of the sweep table: `dense_mean` (Dense) and `arm_mean` of `entmax_l0.5`, `entmax_l1`, `entmax_l2`; absolute means, CLAP / AQAScore / PQ."}
    for k, v in notes.items():
        with open(os.path.join(SRC, f"{k}_source.md"), "w", encoding="utf-8") as fh: fh.write(f"# {k}\n\n{v}\n\n{common}")

    # ---------------- Verification printout ----------------
    print("PLADIS λ=2 − Dense (point, 95 % CI) as used in the assets:")
    for g, c in guid: print(f"  {g:8s} " + "  ".join(f"{MNAME[m]} {ci(rec(m, c), 3)}" for m in METRICS))
    print(f"GPU: sweep = {gpu_sw}; no-CFG = {gpu_nc}. Written to {OUT}")


if __name__ == "__main__":
    main()
