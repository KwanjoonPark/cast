"""Fig 1: original schematic of the CAST experiment (Auffusion pipeline + cross-attention intervention) with a CONCEPTUAL
(stylised, not measured) illustration of attention concentration. Drawn from scratch with matplotlib."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(6.8, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
def box(x, y, w, h, text, fc="white", fs=7.5, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="black", lw=lw)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)
def arrow(x0, y0, x1, y1, lw=0.9): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw, color="black"))
# ---- pipeline row ----
y, h = 35.5, 6
for t, x, w in [("Text prompt", 1, 11), ("CLIP text encoder\n(frozen)", 14.5, 15), ("U-Net cross-attention\n(16 attn2 layers)", 32, 20), ("Mel-spectrogram\n(VAE decode)", 68.5, 15), ("Audio\n(HiFi-GAN)", 86.5, 12)]: box(x, y, w, h, t)
arrow(12.2, y + h / 2, 14.3, y + h / 2); arrow(29.7, y + h / 2, 31.8, y + h / 2); arrow(52.2, y + h / 2, 68.3, y + h / 2); arrow(83.7, y + h / 2, 86.3, y + h / 2)
ax.text(60.25, y + h / 2 - 2.6, "100 steps, CFG 7.5", ha="center", fontsize=6.0, color="#333333")
# ---- intervention block (left) ----
bx, by, bw, bh = 1, 1.5, 56, 28
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.25,rounding_size=1.0", fc="#f4f4f4", ec="black", lw=0.9)); arrow(42, y - 0.3, 42, by + bh + 0.5)
ax.text(bx + 1.5, by + bh - 2.2, "Cross-attention intervention  ($Z = QK^{\\top}/\\sqrt{d}$,  output $= A\\,V$)", fontsize=7.3, va="center")
rows = [("Dense (baseline)", r"$A_{\mathrm{dense}} = \mathrm{Softmax}(Z)$"), ("Sharp Softmax", r"$A_{\mathrm{sharp}} = \mathrm{Softmax}(\beta Z)$"),
        ("Entmax (sparse)", r"$A_{\mathrm{sparse}} = \mathrm{Entmax}_{1.5}(Z)$"), ("PLADIS", r"$A = A_{\mathrm{dense}} + \lambda_{\mathrm{entmax}}\,(A_{\mathrm{sparse}} - A_{\mathrm{dense}})$")]
for i, (name, eq) in enumerate(rows):
    yy = by + bh - 7.2 - i * 5.0; ax.text(bx + 3, yy, "•  " + name, fontsize=7.5, va="center"); ax.text(bx + 23.5, yy, eq, fontsize=8, va="center")
ax.text(bx + bw / 2, by + 1.7, "16 text cross-attention layers only; self-attention, scheduler and CFG unchanged", fontsize=6.0, ha="center", va="center", color="#333333")
# ---- conceptual illustration (right) ----
cx, cy, cw, ch = 60, 1.5, 39, 28
ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.25,rounding_size=1.0", fc="white", ec="black", lw=0.9, ls="--"))
ax.text(cx + cw / 2, cy + ch - 2.2, "Conceptual (not measured): attention over text tokens", fontsize=6.8, ha="center", va="center", style="italic")
ax.text(cx + cw / 2, cy + ch - 4.8, "concentration  →", fontsize=7.2, ha="center", va="center")
rng = np.random.default_rng(3); z = np.sort(rng.normal(0, 1.0, 10))[::-1]; z[0] += 0.8
def softmax(v): e = np.exp(v - v.max()); return e / e.sum()
def entmax_like(v): p = softmax(2.0 * v); q = np.where(p >= np.sort(p)[-2], p, 0.0); return q / q.sum()
panels = [("Dense\nmany active", softmax(0.7 * z)), ("Sharp\nnon-zero,\nconcentrated", softmax(1.6 * z)), ("Entmax\nfew non-zero", entmax_like(z))]
pw = (cw - 6) / 3; hmax = 9.5; base = cy + 8.2
for j, (t, p) in enumerate(panels):
    x0 = cx + 2.0 + j * (pw + 1.0); bw_ = pw / len(p) * 0.78; pmax = max(q.max() for _, q in panels)
    for k, v in enumerate(p):
        xx = x0 + k * (pw / len(p)); hh = max(v / pmax * hmax, 0.0)
        ax.add_patch(Rectangle((xx, base), bw_, hh, fc="#1f4e79" if v > 1e-9 else "none", ec="#1f4e79", lw=0.6))
    ax.plot([x0 - 0.2, x0 + pw - 0.4], [base, base], color="black", lw=0.7); ax.text(x0 + pw / 2 - 0.3, cy + 4.3, t, fontsize=6.2, ha="center", va="center", linespacing=1.1)
fig.savefig("paper_assets/figures/fig1_method_overview.png", dpi=300); fig.savefig("paper_assets/figures/fig1_method_overview.pdf"); plt.close(fig); print("fig1 done")
