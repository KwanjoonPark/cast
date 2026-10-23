"""AudioBox Aesthetics (facebookresearch/audiobox-aesthetics, official package + pretrained all-axes checkpoint) for the
deduplicated CAST audio set. Outputs CE, CU, PC, PQ per file; PQ (Production Quality) is CAST's primary quality axis."""
import argparse, csv, inspect, json, os, sys, time
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", default="outputs/multimetric_eval/combined/audio_index.csv"); ap.add_argument("--out", default="outputs/multimetric_eval/audiobox")
    ap.add_argument("--limit", type=int, default=None); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    import audiobox_aesthetics
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor()
    rows = list(csv.DictReader(open(args.index)))
    # provenance of the evaluator
    src_infer = inspect.getsource(sys.modules["audiobox_aesthetics.infer"])
    info = {"package_version": getattr(audiobox_aesthetics, "__version__", "unknown"), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(getattr(predictor, "device", "unknown")),
            "predictor_class": type(predictor).__name__, "checkpoint": getattr(predictor, "ckpt", None) or getattr(predictor, "checkpoint_pth", None),
            "infer_source_mentions_resample": "resample" in src_infer.lower(), "infer_source_sample_rate_tokens": sorted(set(w for w in src_infer.replace("(", " ").replace(")", " ").replace(",", " ").split() if w in ("16000", "16_000", "sample_rate", "target_sr", "sr"))),
            "input_audio": "original Auffusion 16 kHz mono int16 wav, passed as file path (package handles loading/resampling)"}
    try: info["model_dtype"] = str(next(predictor.model.parameters()).dtype)
    except Exception: info["model_dtype"] = "unknown"
    # determinism: one file, 3 runs
    r0 = rows[0]; reps = [predictor.forward([{"path": r0["audio_path"]}])[0] for _ in range(3)]
    val = {"sample": r0["audio_path"], "repeats": reps, "max_abs_diff": max(abs(reps[i][k] - reps[0][k]) for i in range(3) for k in reps[0])}
    json.dump({"info": info, "validation": val}, open(f"{args.out}/validation.json", "w"), indent=2)
    print("[validation]", val, flush=True); print("[info]", info, flush=True)
    out = []; t0 = time.time(); batch = 16; sel = rows[: args.limit] if args.limit else rows
    for i in range(0, len(sel), batch):
        chunk = sel[i: i + batch]; preds = predictor.forward([{"path": r["audio_path"]} for r in chunk])
        for r, p in zip(chunk, preds):
            out.append({"audio_sha256": r["audio_sha256"], "prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], "audio_path": r["audio_path"], "provenances": r["provenances"],
                        "CE": p["CE"], "CU": p["CU"], "PC": p["PC"], "PQ": p["PQ"]})
        print(f"  {min(i+batch, len(sel))}/{len(sel)} elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    with open(f"{args.out}/sample_scores.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"AudioBox done: {len(out)} files, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
