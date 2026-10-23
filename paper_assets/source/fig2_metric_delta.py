"""Fig 2: Delta vs Dense (CLAP, AQAScore, AudioBox PQ) for the representative methods, with 95% prompt-level bootstrap CIs
read verbatim from outputs/multimetric_eval/combined/analysis_summary.json (24 prompts x 3 seeds, prompt-level unit)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42})
A = json.load(open("outputs/multimetric_eval/combined/analysis_summary.json"))["families"]
METHODS = [("Sharp\nβ=1.5", "sharp_softmax", "1.5"), ("Entmax\nλ=0.5", "entmax", "0.5"), ("Entmax\nλ=1", "entmax", "1.0"), ("PLADIS\nλ=2", "entmax", "2.0")]
PANELS = [("clap_cosine", "(a) ΔCLAP"), ("aqascore_full", "(b) ΔAQAScore"), ("PQ", "(c) ΔPQ (AudioBox)")]
COL = {"sharp_softmax": "#4a4a4a", "entmax": "#1f4e79"}; MK = {"sharp_softmax": "o", "entmax": "s"}


def draw(fig_w, fig_h, out, ncols=3):
    fig, axes = plt.subplots(1 if ncols == 3 else 3, 3 if ncols == 3 else 1, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()
    for ax, (m, title) in zip(axes, PANELS):
        xs = np.arange(len(METHODS))
        for i, (lab, fam, k) in enumerate(METHODS):
            d = A[fam]["metrics"][m]["all"]["delta"][k]; lo, hi = d["ci95"]
            ax.errorbar(i, d["mean"], yerr=[[d["mean"] - lo], [hi - d["mean"]]], fmt=MK[fam], color=COL[fam], ms=4.5, capsize=3, lw=1.0, mfc=COL[fam] if fam == "entmax" else "white", mew=1.0)
        ax.axhline(0, color="black", lw=0.8, ls="--"); ax.set_xticks(xs); ax.set_xticklabels([l for l, _, _ in METHODS], fontsize=7)
        ax.set_title(title, fontsize=8.5, loc="left"); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_xlim(-0.6, len(METHODS) - 0.4)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ vs Dense (dashed = Dense)", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


draw(6.8, 2.2, "paper_assets/figures/fig2_metric_delta")                 # two-column width (~17 cm)
draw(3.3, 5.4, "paper_assets/figures/fig2_metric_delta_compact", ncols=1)  # one-column stacked variant
print("fig2 done")
