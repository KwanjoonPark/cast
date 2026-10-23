"""Minimal spectrogram comparison of EXISTING audio (no metric annotations).
usage: fig_spectrogram_simple.py <PID> <CAT_DIR> <seed> <out_prefix>
Produces <out_prefix>_2panel.png (Dense | PLADIS λ=2) and <out_prefix>_4panel.png (Dense | Sharp β=1.5 | Entmax λ=1 | PLADIS λ=2).
Same preprocessing for every panel; one shared dB scale."""
import json, sys, numpy as np, soundfile as sf, librosa, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
pid, cat, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
COND = [("Dense (baseline)", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_0"),
        ("Sharp Softmax (β=1.5)", f"outputs/sharp_softmax_v1/{cat}/{pid}/seed_{seed}/beta_1.50"),
        ("Entmax (λ=1)", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_1"),
        ("PLADIS (λ=2)", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_2")]
mans = [json.load(open(d + "/manifest.json")) for _, d in COND]
assert len({m["initial_latents"]["sha256"] for m in mans}) == 1
prompt = mans[0]["config"]["prompt"]
dbs = []
for _, d in COND:
    y, sr = sf.read(d + "/audio.wav", dtype="float32")
    dbs.append(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=128, fmax=8000, power=1.0))
ref = max(S.max() for S in dbs); dbs = [librosa.amplitude_to_db(S, ref=ref) for S in dbs]
def draw(idx, fname, w):
    fig, axes = plt.subplots(1, len(idx), figsize=(w, 2.3), sharey=True)
    for ax, i in zip(np.atleast_1d(axes), idx):
        D = dbs[i]; im = ax.imshow(D, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=[0, D.shape[1] * 160 / 16000, 0, 8])
        ax.set_title(COND[i][0], fontsize=9); ax.set_xlabel("Time (s)"); ax.set_yticks([0, 2, 4, 6, 8])
    np.atleast_1d(axes)[0].set_ylabel("Frequency (kHz)")
    fig.tight_layout(rect=(0, 0, 0.94, 1)); cax = fig.add_axes([0.955, 0.2, 0.012, 0.65]); cb = fig.colorbar(im, cax=cax); cb.set_label("dB", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(fname + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03); fig.savefig(fname + ".pdf", bbox_inches="tight", pad_inches=0.03); plt.close(fig)
draw([0, 3], out + "_2panel", 5.4); draw([0, 1, 2, 3], out + "_4panel", 9.0)
print("done", pid, "|", prompt)
