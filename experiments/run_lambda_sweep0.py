"""CAST lambda response sweep: 5 prompts x lambda in {0,.25,.5,.75,1,1.5,2}, alpha 1.5, deterministic self-attn,
one shared seed-42 initial latent. lambda 0/1/2 are reused from Pilot 0 when their manifests match the sweep
settings exactly (and one reused arm is regenerated and compared bitwise as proof); otherwise everything is
regenerated. The validation prompt (P03) runs first and the script stops if any check fails."""
import argparse, hashlib, json, os, random, shutil, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, yaml

from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype

MATCH_KEYS = ["checkpoint", "prompt", "seed", "num_inference_steps", "guidance_scale", "height", "width", "dtype",
              "autocast", "apply_to_uncond", "deterministic_self_attn"]


def lam_dir(l): return f"lambda_{l:.2f}"
def rel_l2(a, b): a, b = a.float(), b.float(); return float((a - b).norm() / b.norm())
def finite_ok(m): return all(v for v in m["finite"].values() if v is not None)
def support(m):
    st = m["cast_processor_stats"]
    c = [v["mean_sparse_support_cond_half"] for v in st.values() if v.get("mean_sparse_support_cond_half") is not None]
    return None if not c else {"mean_over_layers": sum(c) / len(c), "min": min(c), "max": max(c)}


def pilot0_match(pilot_manifest, cfg, z0_sha):
    pc = pilot_manifest["config"]; cd = cfg.to_dict()
    mism = {k: (pc.get(k), cd[k]) for k in MATCH_KEYS if pc.get(k) != cd[k]}
    if pilot_manifest["alpha"] != cfg.alpha: mism["alpha"] = (pilot_manifest["alpha"], cfg.alpha)
    if pilot_manifest["lam"] != cfg.lam: mism["lam"] = (pilot_manifest["lam"], cfg.lam)
    if pilot_manifest["initial_latents"]["sha256"] != z0_sha: mism["initial_latent_sha256"] = ("differs", z0_sha)
    if pilot_manifest["attention_processor_class"]["attn2"] != ["CastCrossAttnProcessor"]: mism["attn2"] = pilot_manifest["attention_processor_class"]["attn2"]
    if not finite_ok(pilot_manifest): mism["finite"] = pilot_manifest["finite"]
    return mism


def reuse_pilot0(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in ["audio.wav", "spectrogram.png", "final_latents.pt"]:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    m = json.load(open(os.path.join(src_dir, "manifest.json")))
    m["provenance"] = {"reused_from": src_dir, "note": "copied unchanged from Pilot 0 (identical settings verified); not regenerated"}
    json.dump(m, open(os.path.join(dst_dir, "manifest.json"), "w"), indent=2)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="experiments/configs/lambda_sweep0.yaml")
    ap.add_argument("--force-regenerate", action="store_true"); args = ap.parse_args()
    y = yaml.safe_load(open(args.config))
    prompts, lambdas, alpha = y.pop("prompts"), [float(l) for l in y.pop("lambdas")], float(y.pop("alpha"))
    pilot_dir, pilot_modes, vprompt = y.pop("pilot0_dir"), {float(k): v for k, v in y.pop("pilot0_modes").items()}, y.pop("validation_prompt")
    base = CastConfig(**y, prompt="", mode="cast", lam=0.0, alpha=alpha); assert base.deterministic_self_attn is True
    root = base.output_dir; os.makedirs(root, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
    assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0 = sample_initial_latents(base)
    z0_sha = hashlib.sha256(z0.float().cpu().numpy().tobytes()).hexdigest()

    # ---- reuse eligibility: every Pilot 0 manifest for lambda 0/1/2 must match the sweep settings exactly ----
    reuse_checks = {}
    for pid, prompt in prompts.items():
        for l, mode in pilot_modes.items():
            mp = os.path.join(pilot_dir, pid, mode, "manifest.json")
            cfg = CastConfig(**{**base.to_dict(), "prompt": prompt, "lam": l})
            reuse_checks[f"{pid}/{lam_dir(l)}"] = pilot0_match(json.load(open(mp)), cfg.resolved(), z0_sha) if os.path.exists(mp) else {"missing": mp}
    reuse_ok = (not args.force_regenerate) and all(not v for v in reuse_checks.values())
    print(f"[reuse] all 15 Pilot 0 manifests match: {reuse_ok}" + ("" if reuse_ok else f" mismatches={ {k: v for k, v in reuse_checks.items() if v} }"))

    summary = {"config": {**y, "alpha": alpha, "lambdas": lambdas}, "initial_latent_sha256": z0_sha, "reuse_path": "reuse" if reuse_ok else "regenerate_all",
               "reuse_checks": reuse_checks, "runs": {}, "per_prompt": {}, "validation": {}, "failures": []}

    # ---- proof that the current implementation reproduces Pilot 0 bitwise: regenerate one reused arm ----
    if reuse_ok:
        cfg = CastConfig(**{**base.to_dict(), "prompt": prompts[vprompt], "lam": 1.0})
        r = run_one(pipe, cfg, initial_latents=z0, save=False)
        pm = json.load(open(os.path.join(pilot_dir, vprompt, pilot_modes[1.0], "manifest.json")))
        summary["validation"]["pilot0_bitwise_reproduction"] = {
            "arm": f"{vprompt}/lambda_1.00", "final_latent_sha_equal": r["manifest"]["final_latents_sha256"] == pm["final_latents_sha256"],
            "audio_sha_equal": r["manifest"]["audio_sha256"] == pm["audio_sha256"]}
        print("[validation] regenerated", vprompt, "lambda=1.00 vs Pilot 0:", summary["validation"]["pilot0_bitwise_reproduction"])
        if not all(v for k, v in summary["validation"]["pilot0_bitwise_reproduction"].items() if k.endswith("equal")):
            print("Pilot 0 not reproduced bitwise -> regenerating everything"); reuse_ok = False; summary["reuse_path"] = "regenerate_all"

    order = [vprompt] + [p for p in prompts if p != vprompt]
    for pi, pid in enumerate(order):
        prompt = prompts[pid]; summary["runs"][pid] = {}
        for l in lambdas:
            dst = os.path.join(root, pid, lam_dir(l))
            if reuse_ok and l in pilot_modes:
                m = reuse_pilot0(os.path.join(pilot_dir, pid, pilot_modes[l]), dst); src = "reused_pilot0"
            else:
                cfg = CastConfig(**{**base.to_dict(), "prompt": prompt, "lam": l})
                try:
                    m = run_one(pipe, cfg, initial_latents=z0, out_dir=dst)["manifest"]; src = "generated"
                except Exception as e:
                    summary["failures"].append({"prompt_id": pid, "lambda": l, "error": repr(e), "trace": traceback.format_exc()})
                    print(f"[{pid}/{lam_dir(l)}] FAILED {e!r}"); continue
            summary["runs"][pid][f"{l:.2f}"] = {
                "prompt_id": pid, "prompt": prompt, "alpha": m["alpha"], "lambda": m["lam"], "seed": m["config"]["seed"],
                "deterministic_self_attn": m["config"]["deterministic_self_attn"], "initial_latent_sha256": m["initial_latents"]["sha256"],
                "final_latent_norm": m["final_latent_norm"], "final_latent_sha256": m["final_latents_sha256"], "runtime_sec": m["runtime_sec"],
                "peak_gpu_memory_gb": m["peak_cuda_mem_gb"], "finite": m["finite"], "valid": finite_ok(m),
                "attn2_processor": m["attention_processor_class"]["attn2"], "attn1_processor": m["attention_processor_class"]["attn1"],
                "entmax_support_cond": support(m), "source": src, "dir": dst}
            print(f"[{pid}/{lam_dir(l)}] {src:14s} |zT|={m['final_latent_norm']:.2f} rt={m['runtime_sec']:.1f}s finite={finite_ok(m)}")
        # per-prompt deltas from the saved final latents
        z = {l: torch.load(os.path.join(root, pid, lam_dir(l), "final_latents.pt"))["final_latents"] for l in lambdas if f"{l:.2f}" in summary["runs"][pid]}
        pp = {"prompt": prompt, "delta_vs_lambda0": {f"{l:.2f}": rel_l2(z[l], z[0.0]) for l in z},
              "delta_step": {f"{a:.2f}->{b:.2f}": rel_l2(z[b], z[a]) for a, b in zip(lambdas[:-1], lambdas[1:]) if a in z and b in z},
              "final_latent_norms": {f"{l:.2f}": float(z[l].float().norm()) for l in z},
              "initial_latent_identical": len({r["initial_latent_sha256"] for r in summary["runs"][pid].values()}) == 1}
        summary["per_prompt"][pid] = pp
        print(f"[{pid}] delta_vs_0={ {k: round(v, 4) for k, v in pp['delta_vs_lambda0'].items()} }")
        if pi == 0:   # ---- pre-run validation on the validation prompt ----
            runs = summary["runs"][pid]; after = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
            checks = {"deterministic_self_attn_true": all(r["deterministic_self_attn"] for r in runs.values()),
                      "seven_conditions": len(runs) == 7 and not summary["failures"],
                      "attn2_all_cast_16": all(r["attn2_processor"] == ["CastCrossAttnProcessor"] for r in runs.values()),
                      "attn1_untouched_stock": all(r["attn1_processor"] == sorted({stock[n] for n in stock if n.endswith("attn1.processor")}) for r in runs.values()),
                      "processors_restored": after == stock,
                      "alpha_1p5": all(r["alpha"] == 1.5 for r in runs.values()),
                      "lambda_resolved_exactly": [r["lambda"] for r in runs.values()] == lambdas,
                      "same_initial_latent": pp["initial_latent_identical"] and all(r["initial_latent_sha256"] == z0_sha for r in runs.values()),
                      "no_nan_inf": all(r["valid"] for r in runs.values()),
                      "pilot0_reproduced": summary["validation"].get("pilot0_bitwise_reproduction", {}).get("final_latent_sha_equal", "n/a (regenerated)")}
            summary["validation"]["pre_run_checks"] = checks; print("[pre-run validation on", pid, "]", checks)
            if not all(v is True or v == "n/a (regenerated)" for v in checks.values()):
                summary["aborted"] = "pre-run validation failed"
                json.dump(summary, open(os.path.join(root, "lambda_sweep_summary.json"), "w"), indent=2); print("VALIDATION FAILED - STOPPING"); sys.exit(1)

    # ---- cross-prompt summary ----
    import statistics
    cross = {}
    for l in lambdas:
        k = f"{l:.2f}"; ds = [summary["per_prompt"][p]["delta_vs_lambda0"][k] for p in prompts if k in summary["per_prompt"][p]["delta_vs_lambda0"]]
        norms = [summary["runs"][p][k]["final_latent_norm"] for p in prompts if k in summary["runs"][p]]
        rts = [summary["runs"][p][k]["runtime_sec"] for p in prompts if k in summary["runs"][p]]
        sup = [summary["runs"][p][k]["entmax_support_cond"]["mean_over_layers"] for p in prompts if k in summary["runs"][p] and summary["runs"][p][k]["entmax_support_cond"]]
        cross[k] = {"mean_delta": statistics.mean(ds), "std_delta": statistics.pstdev(ds), "mean_latent_norm": statistics.mean(norms),
                    "mean_runtime_sec": statistics.mean(rts), "mean_entmax_support_cond": statistics.mean(sup) if sup else None, "n": len(ds)}
    summary["cross_prompt"] = cross
    allruns = [r for p in summary["runs"].values() for r in p.values()]
    summary["counts"] = {"total": len(allruns), "generated": sum(r["source"] == "generated" for r in allruns),
                         "reused_pilot0": sum(r["source"] == "reused_pilot0" for r in allruns), "invalid": sum(not r["valid"] for r in allruns) + len(summary["failures"])}
    summary["all_initial_latents_identical"] = all(r["initial_latent_sha256"] == z0_sha for r in allruns)
    json.dump(summary, open(os.path.join(root, "lambda_sweep_summary.json"), "w"), indent=2)

    # ---- listening indices ----
    L = ["# CAST Lambda Sweep 0 Listening Index", "", "All paths relative to `outputs/lambda_sweep0/`. alpha = 1.5; seed 42; one bitwise-identical initial latent for every prompt and lambda; deterministic self-attention; CFG 7.5; 100 steps. lambda 0 / 1 / 2 are the Pilot 0 outputs (copied, see manifests `provenance`).", ""]
    B = ["# CAST Lambda Sweep 0 — Blind Listening Index", "", "Same files as LISTENING_INDEX.md, relabelled A–G in a random order per prompt. The mapping is in `blind_key.json` (do not open before listening).", ""]
    key = {}; rng = random.Random(20261020)
    tbl = ["| λ | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |", "|---:|---|---|---|---|---|---|"] + [f"| {l:.2f} | | | | | | |" for l in lambdas]
    for pid, prompt in prompts.items():
        L += [f"## {pid.split('_')[0]} — {prompt}", ""] + [f"- λ={l:.2f}: `{pid}/{lam_dir(l)}/audio.wav`  (spectrogram: `{pid}/{lam_dir(l)}/spectrogram.png`)" for l in lambdas] + ["", "Manual evaluation:", ""] + tbl + [""]
        labels = list("ABCDEFG"); rng.shuffle(labels); key[pid] = {lab: f"{l:.2f}" for lab, l in zip(labels, lambdas)}
        B += [f"## {pid.split('_')[0]} — {prompt}", ""] + [f"- {lab}: `{pid}/{lam_dir(l)}/audio.wav`" for lab, l in sorted(zip(labels, lambdas))] + ["", "| Label | Event presence | Secondary/background preservation | Separation / clarity | Audio fidelity | Artifacts | Overall preference |", "|---|---|---|---|---|---|---|"] + [f"| {lab} | | | | | | |" for lab in "ABCDEFG"] + [""]
    open(os.path.join(root, "LISTENING_INDEX.md"), "w").write("\n".join(L))
    open(os.path.join(root, "BLIND_LISTENING_INDEX.md"), "w").write("\n".join(B))
    json.dump(key, open(os.path.join(root, "blind_key.json"), "w"), indent=2)
    c = summary["counts"]
    print("\nCAST Lambda Sweep\n" + "-" * 32 + f"\nStatus: {'PASS' if c['total'] == len(prompts) * len(lambdas) and c['invalid'] == 0 else 'FAIL'}\nPrompts: {len(prompts)}\nLambda values: {len(lambdas)}\nTotal conditions: {c['total']}\nNew generations: {c['generated']}\nReused Pilot 0 outputs: {c['reused_pilot0']}\nInvalid runs: {c['invalid']}\nReport: CAST_LAMBDA_SWEEP_REPORT.md\nListening index: {root}/LISTENING_INDEX.md")


if __name__ == "__main__":
    main()
