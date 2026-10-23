"""Optional Fig: Sharp Softmax measured concentration (effective support exp(H), conditional half, 24 prompts x 3 seeds)
and mean CLAP vs beta. Values read from outputs/sharp_softmax_v1/hypothesis_summary.json (same numbers as the report)."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42})
H = json.load(open("outputs/sharp_softmax_v1/hypothesis_summary.json")); B = [1.0, 1.25, 1.5, 2.0]
eff = [H["entropy"]["by_beta"][str(b)]["effective_support"] for b in B]; pl = H["sharp"]["prompt_level"]
clap = [sum(pl[p][str(b)] for p in pl) / len(pl) for b in B]
ci = [H["sharp"]["category"]["DISCRETE_SINGLE"]["abs"][str(b)] for b in B]  # not used; overall CI not stored -> plot means only
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6))
axes[0].plot(B, eff, marker="o", color="#1f4e79", lw=1.1, ms=4); axes[0].set_ylabel("exp(H), of 77 keys", fontsize=7); axes[0].set_title("(a) effective support", loc="left", fontsize=7.5)
for b, e in zip(B, eff): axes[0].annotate(f"{e:.1f}", (b, e), textcoords="offset points", xytext=(5, -2 if b == 1.0 else 3), fontsize=6.5)
axes[0].set_ylim(0, 22)
axes[1].plot(B, clap, marker="s", color="#4a4a4a", lw=1.1, ms=4); axes[1].set_ylabel("CLAP (24 prompts)", fontsize=7); axes[1].set_title("(b) mean CLAP", loc="left", fontsize=7.5)
for ax in axes: ax.set_xlabel("β (Sharp Softmax)", fontsize=7.5); ax.set_xticks(B); ax.set_xticklabels(["1", "1.25", "1.5", "2"]); ax.set_xlim(0.9, 2.15); ax.grid(alpha=0.25, lw=0.5); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.tick_params(labelsize=7)
fig.tight_layout(w_pad=1.0); fig.savefig("paper_assets/figures/fig_optional_concentration.png", dpi=300, bbox_inches="tight"); fig.savefig("paper_assets/figures/fig_optional_concentration.pdf", bbox_inches="tight"); plt.close(fig)
print("optional fig done; eff", [round(e, 2) for e in eff], "clap", [round(c, 4) for c in clap])
