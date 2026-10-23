"""CLAP evaluation of the 35 canonical lambda-sweep samples (evaluation only; no generation).

Metric: clap_cosine = <L2norm(CLAP_audio(wav)), L2norm(CLAP_text(prompt))>, clap_percent = 100 * clap_cosine.
Model: laion/clap-htsat-unfused via transformers ClapModel/ClapProcessor (fp32, eval, no_grad).
Audio: 16 kHz mono int16 wav -> float32 -> torchaudio sinc resample to 48 kHz (exactly 480000 samples = CLAP's
nb_max_samples, so no random truncation is triggered) -> official ClapFeatureExtractor.
"""
import csv, hashlib, json, os, statistics, sys
import numpy as np, torch, torchaudio
from scipy.io.wavfile import read as wav_read
from huggingface_hub import snapshot_download
from transformers import ClapModel, ClapProcessor

ROOT = "outputs/lambda_sweep0"; OUT = f"{ROOT}/clap"; os.makedirs(OUT, exist_ok=True)
MODEL = "laion/clap-htsat-unfused"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PROMPTS = {"P01_SINGLE_DOG": "A dog barking loudly.", "P02_SINGLE_RAIN": "Heavy rain falling on a roof.",
           "P03_TWO_DOG_RAIN": "A dog barking while rain falls in the background.",
           "P04_TWO_SPEECH_HORN": "A man speaking while a car horn sounds nearby.",
           "P05_TWO_FOOTSTEPS_BIRDS": "Footsteps on a wooden floor while birds chirp outside."}
EVENTS = {"P03_TWO_DOG_RAIN": {"A": "a dog barking", "B": "rain falling"},
          "P04_TWO_SPEECH_HORN": {"A": "a man speaking", "B": "a car horn"},
          "P05_TWO_FOOTSTEPS_BIRDS": {"A": "footsteps on a wooden floor", "B": "birds chirping"}}
dev = "cuda"
torch.manual_seed(0)


def load_audio_48k(path):
    sr, x = wav_read(path)
    info = {"input_sr": int(sr), "input_dtype": str(x.dtype), "input_len": int(len(x)), "channels": 1 if x.ndim == 1 else int(x.shape[1])}
    if x.ndim == 2: x = x.mean(axis=1)                               # not expected (mono files)
    if x.dtype == np.int16: xf = x.astype(np.float32) / 32768.0
    else: xf = x.astype(np.float32)
    info["input_peak"] = float(np.abs(xf).max())
    y = torchaudio.functional.resample(torch.from_numpy(xf), sr, 48000)     # windowed-sinc, deterministic
    info.update({"clap_sr": 48000, "resampled_len": int(y.numel()), "resampled_peak": float(y.abs().max()),
                 "clipped_after_resample": bool(y.abs().max() > 1.0)})
    return y.numpy(), info


def main():
    local = snapshot_download(MODEL, token=False)
    model = ClapModel.from_pretrained(local).to(dev).eval()
    proc = ClapProcessor.from_pretrained(local)
    fe = proc.feature_extractor
    logit_scale_a = float(model.logit_scale_a.exp()); logit_scale_t = float(model.logit_scale_t.exp())
    model_info = {"model": MODEL, "snapshot": os.path.basename(local), "param_dtype": str(next(model.parameters()).dtype), "device": dev,
                  "transformers": __import__("transformers").__version__, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                  "feature_extractor": {k: getattr(fe, k, None) for k in ["sampling_rate", "max_length_s", "nb_max_samples", "truncation", "padding", "feature_size", "hop_length", "fft_window_size"]},
                  "logit_scale_audio_exp": logit_scale_a, "logit_scale_text_exp": logit_scale_t,
                  "weights_sha256": hashlib.sha256(open(os.path.join(local, "pytorch_model.bin"), "rb").read()).hexdigest()}

    @torch.no_grad()
    def text_emb(texts):
        t = proc(text=texts, return_tensors="pt", padding=True).to(dev)
        e = model.get_text_features(**t); return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def audio_emb(wave48):
        a = proc(audios=[wave48], sampling_rate=48000, return_tensors="pt").to(dev)
        e = model.get_audio_features(**a); return torch.nn.functional.normalize(e, dim=-1), a["input_features"].shape

    # ---- repeatability: same wav + prompt, 3 full evaluations (load->resample->extract->embed) ----
    pid0 = "P03_TWO_DOG_RAIN"; p0 = f"{ROOT}/{pid0}/lambda_1.00/audio.wav"
    reps = []
    for _ in range(3):
        w, _i = load_audio_48k(p0); ae, _ = audio_emb(w); te = text_emb([PROMPTS[pid0]]); reps.append(float((ae * te).sum()))
    repeat = {"sample": p0, "scores": reps, "max_abs_diff": max(reps) - min(reps)}
    # cross-check: HF logits_per_audio == logit_scale * cosine (documentation of the scaled-logit convention we do NOT use)
    w, _i = load_audio_48k(p0); a = proc(audios=[w], sampling_rate=48000, return_tensors="pt").to(dev); t = proc(text=[PROMPTS[pid0]], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad(): out = model(**t, **a)
    logit_check = {"logits_per_audio": float(out.logits_per_audio.flatten()[0]), "cosine": reps[0], "cosine_x_logit_scale_t": reps[0] * logit_scale_t, "cosine_x_logit_scale_a": reps[0] * logit_scale_a}

    text_cache = {pid: text_emb([p]) for pid, p in PROMPTS.items()}
    event_cache = {pid: {k: text_emb([v]) for k, v in ev.items()} for pid, ev in EVENTS.items()}
    rows, audio_checks = [], {}
    for pid, prompt in PROMPTS.items():
        for lam in LAMBDAS:
            d = f"{ROOT}/{pid}/lambda_{lam:.2f}"; path = f"{d}/audio.wav"
            assert os.path.exists(path), path
            m = json.load(open(f"{d}/manifest.json"))
            w, info = load_audio_48k(path); ae, feat_shape = audio_emb(w)
            cos = float((ae * text_cache[pid]).sum())
            row = {"prompt_id": pid, "prompt": prompt, "lambda": lam, "audio_path": path, "clap_cosine": cos, "clap_percent": 100 * cos,
                   "alpha": m["alpha"], "seed": m["config"]["seed"], "checkpoint": m["config"]["checkpoint"],
                   "initial_latent_sha256": m["initial_latents"]["sha256"], "source": "reused_pilot0" if "provenance" in m else "generated",
                   "finite": bool(np.isfinite(cos))}
            if pid in EVENTS:
                for k, v in EVENTS[pid].items():
                    row[f"event_clap_{k}"] = float((ae * event_cache[pid][k]).sum()); row[f"event_text_{k}"] = v
            rows.append(row); audio_checks[f"{pid}/lambda_{lam:.2f}"] = {**info, "input_features_shape": list(feat_shape)}
            print(f"[{pid}/lambda_{lam:.2f}] clap_cosine={cos:.4f} ({100*cos:.2f}%)" + (f"  event A={row['event_clap_A']:.4f} B={row['event_clap_B']:.4f}" if pid in EVENTS else ""))

    # ---- summaries ----
    by_lam = {}
    for lam in LAMBDAS:
        v = [r["clap_cosine"] for r in rows if r["lambda"] == lam]
        by_lam[f"{lam:.2f}"] = {"mean": statistics.mean(v), "std": statistics.pstdev(v), "min": min(v), "max": max(v), "n": len(v),
                                "mean_percent": 100 * statistics.mean(v)}
    per_prompt = {pid: {f"{lam:.2f}": next(r["clap_cosine"] for r in rows if r["prompt_id"] == pid and r["lambda"] == lam) for lam in LAMBDAS} for pid in PROMPTS}
    delta = {pid: {k: v - per_prompt[pid]["0.00"] for k, v in per_prompt[pid].items()} for pid in PROMPTS}
    mean_delta = {f"{lam:.2f}": {"mean": statistics.mean(delta[p][f"{lam:.2f}"] for p in PROMPTS), "std": statistics.pstdev([delta[p][f"{lam:.2f}"] for p in PROMPTS]),
                                 "min": min(delta[p][f"{lam:.2f}"] for p in PROMPTS), "max": max(delta[p][f"{lam:.2f}"] for p in PROMPTS)} for lam in LAMBDAS}
    event = {pid: {f"{lam:.2f}": {k: next(r[f"event_clap_{k}"] for r in rows if r["prompt_id"] == pid and r["lambda"] == lam) for k in "AB"} for lam in LAMBDAS} for pid in EVENTS}
    sanity = {"all_found": len(rows) == 35, "all_finite": all(r["finite"] for r in rows), "n": len(rows),
              "all_input_sr_16000": all(c["input_sr"] == 16000 for c in audio_checks.values()), "all_mono": all(c["channels"] == 1 for c in audio_checks.values()),
              "all_resampled_len_480000": all(c["resampled_len"] == 480000 for c in audio_checks.values()),
              "any_clipping_after_resample": any(c["clipped_after_resample"] for c in audio_checks.values()),
              "max_input_peak": max(c["input_peak"] for c in audio_checks.values()), "max_resampled_peak": max(c["resampled_peak"] for c in audio_checks.values()),
              "input_features_shape": sorted({tuple(c["input_features_shape"]) for c in audio_checks.values()})}
    result = {"metric": "clap_cosine = dot(L2norm(audio_emb), L2norm(text_emb)); clap_percent = 100*clap_cosine", "model_info": model_info,
              "repeatability": repeat, "logit_scale_check": logit_check, "sanity": sanity, "audio_checks": audio_checks,
              "samples": rows, "by_lambda": by_lam, "per_prompt": per_prompt, "delta_vs_lambda0": delta, "mean_delta_vs_lambda0": mean_delta,
              "event_clap_exploratory": event, "event_phrases": EVENTS}
    json.dump(result, open(f"{ROOT}/clap_scores.json", "w"), indent=2)
    cols = ["prompt_id", "prompt", "lambda", "audio_path", "clap_cosine", "clap_percent", "alpha", "seed", "checkpoint", "initial_latent_sha256", "source",
            "event_text_A", "event_clap_A", "event_text_B", "event_clap_B"]
    with open(f"{ROOT}/clap_scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); [wr.writerow({c: r.get(c, "") for c in cols}) for r in rows]

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = LAMBDAS; means = [by_lam[f"{l:.2f}"]["mean"] for l in xs]; stds = [by_lam[f"{l:.2f}"]["std"] for l in xs]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3); ax.set_xlabel("lambda"); ax.set_ylabel("mean CLAP cosine (5 prompts, ±pop. std)")
    ax.set_title("CLAP vs lambda (laion/clap-htsat-unfused)"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT}/clap_mean_vs_lambda.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pid in PROMPTS: ax.plot(xs, [per_prompt[pid][f"{l:.2f}"] for l in xs], marker="o", label=pid)
    ax.set_xlabel("lambda"); ax.set_ylabel("CLAP cosine"); ax.set_title("CLAP vs lambda, per prompt"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/clap_by_prompt_vs_lambda.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); md = [mean_delta[f"{l:.2f}"]["mean"] for l in xs]
    for pid in PROMPTS: ax.plot(xs, [delta[pid][f"{l:.2f}"] for l in xs], marker=".", alpha=.5, label=pid)
    ax.plot(xs, md, marker="o", color="k", lw=2, label="mean"); ax.axhline(0, color="gray", lw=.8); ax.set_xlabel("lambda"); ax.set_ylabel("ΔCLAP cosine vs λ=0"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/clap_delta_vs_lambda.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, pid in zip(axes, EVENTS):
        ax.plot(xs, [per_prompt[pid][f"{l:.2f}"] for l in xs], marker="o", color="k", label="full prompt")
        for k in "AB": ax.plot(xs, [event[pid][f"{l:.2f}"][k] for l in xs], marker=".", label=f"{k}: {EVENTS[pid][k]}")
        ax.set_title(pid, fontsize=9); ax.set_xlabel("lambda"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("CLAP cosine (exploratory event-level)"); fig.tight_layout(); fig.savefig(f"{OUT}/event_clap_exploratory.png", dpi=120); plt.close(fig)

    print("\nrepeatability:", repeat); print("logit check:", logit_check); print("sanity:", sanity)
    print("by_lambda:", {k: (round(v["mean"], 4), round(v["std"], 4), round(v["min"], 4), round(v["max"], 4)) for k, v in by_lam.items()})
    print("mean_delta:", {k: round(v["mean"], 4) for k, v in mean_delta.items()})
    ok = sanity["all_found"] and sanity["all_finite"]
    print("\nCAST CLAP Evaluation\n" + "-" * 32 + f"\nStatus: {'PASS' if ok else 'FAIL'}\nCLAP model: {MODEL} ({model_info['snapshot'][:8]})\nSamples: {len(rows)} / 35\nInvalid: {sum(not r['finite'] for r in rows)}\nDeterministic: max repeat diff {repeat['max_abs_diff']:.2e}\nReport: CAST_CLAP_EVALUATION_REPORT.md\nCSV: {ROOT}/clap_scores.csv")


if __name__ == "__main__":
    main()
