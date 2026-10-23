"""Fig 3 / 3b: mel-spectrogram comparison of EXISTING generated audio for one prompt and seed across four attention
conditions. Identical preprocessing (16 kHz, n_fft 1024, hop 160, 128 mel, 0-8 kHz, amplitude->dB) and one shared
colour scale (vmin/vmax fixed from the global maximum over the four clips; 80 dB range). Metric annotations are the
exact stored sample-level values (all_metrics.csv). Manifests are checked for identical prompt/seed/initial latent."""
import csv, json, sys, numpy as np, soundfile as sf, librosa, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
pid, cat, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); out = sys.argv[4]
COND = [("(a) Dense", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_0"), ("(b) Sharp β=1.5", f"outputs/sharp_softmax_v1/{cat}/{pid}/seed_{seed}/beta_1.50"),
        ("(c) Entmax λ=1", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_1"), ("(d) PLADIS λ=2", f"outputs/diagnostic_v1/{cat}/{pid}/seed_{seed}/lambda_2")]
mets = {r["audio_path"]: r for r in csv.DictReader(open("outputs/multimetric_eval/combined/all_metrics.csv"))}
mans = [json.load(open(d + "/manifest.json")) for _, d in COND]
assert len({m["config"]["prompt"] for m in mans}) == 1 and len({m["config"]["seed"] for m in mans}) == 1 and len({m["initial_latents"]["sha256"] for m in mans}) == 1, "panels differ in prompt/seed/latent"
prompt = mans[0]["config"]["prompt"]
specs = []
for _, d in COND:
    y, sr = sf.read(d + "/audio.wav", dtype="float32"); assert sr == 16000
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=128, fmin=0, fmax=8000, power=1.0); specs.append(S)
ref = max(S.max() for S in specs); dbs = [librosa.amplitude_to_db(S, ref=ref) for S in specs]; vmax = 0.0; vmin = -80.0
fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.9), sharex=True, sharey=True)
for ax, (lab, d), D in zip(axes.ravel(), COND, dbs):
    im = ax.imshow(D, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, extent=[0, D.shape[1] * 160 / 16000, 0, 8000])
    r = mets[d + "/audio.wav"]; ax.set_title(f"{lab}   CLAP {float(r['clap_cosine']):.3f} · AQA {float(r['aqascore_full']):.3f} · PQ {float(r['PQ']):.2f}", fontsize=7.5, loc="left")
    ax.set_yticks([0, 2000, 4000, 6000, 8000]); ax.set_yticklabels(["0", "2", "4", "6", "8"])
for ax in axes[1]: ax.set_xlabel("time (s)")
for ax in axes[:, 0]: ax.set_ylabel("frequency (kHz)")
fig.suptitle(f'"{prompt}"  (seed {seed}, identical initial latent)', fontsize=8, y=0.995)
fig.tight_layout(rect=(0, 0, 0.92, 0.97)); cax = fig.add_axes([0.935, 0.12, 0.015, 0.76]); cb = fig.colorbar(im, cax=cax); cb.set_label("dB (shared scale)", fontsize=7); cb.ax.tick_params(labelsize=7)
fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf"); plt.close(fig); print("done", out, "prompt:", prompt, "ref max over 4 clips: %.4f" % ref)
