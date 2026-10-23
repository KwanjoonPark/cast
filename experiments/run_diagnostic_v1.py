"""CAST diagnostic v1 generation: frozen prompt set x lambda x seed with one shared initial latent per (prompt, seed).
Pre-generation validation runs on (validation_prompt, validation_seed) across all lambdas; the script stops if any check fails.
Writes outputs/diagnostic_v1/experiment_manifest.json and generation_summary.json (resumable: existing complete conditions are skipped)."""
import argparse, hashlib, json, os, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, yaml
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, _versions

CAT_DIR = {"DISCRETE_SINGLE": "A_DISCRETE_SINGLE", "DIFFUSE_SINGLE": "B_DIFFUSE_SINGLE", "DISCRETE_DISCRETE": "C_DISCRETE_DISCRETE", "DISCRETE_DIFFUSE": "D_DISCRETE_DIFFUSE"}


def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def sha_tensor(t): return hashlib.sha256(t.float().cpu().numpy().tobytes()).hexdigest()
def finite_ok(m): return all(v for v in m["finite"].values() if v is not None)
def cond_dir(root, p, seed, lam): return os.path.join(root, CAT_DIR[p["category"]], p["id"], f"seed_{seed}", f"lambda_{lam:g}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="experiments/configs/diagnostic_v1.yaml"); args = ap.parse_args()
    y = yaml.safe_load(open(args.config)); lambdas = [float(l) for l in y.pop("lambdas")]; seeds = [int(s) for s in y.pop("seeds")]
    alpha = float(y.pop("alpha")); pfile = y.pop("prompt_file"); vp, vs = y.pop("validation_prompt"), int(y.pop("validation_seed"))
    ds = json.load(open(pfile)); prompts = ds["prompts"]; ds_sha = sha_file(pfile)
    assert len(prompts) == 24 and all(sum(p["category"] == c for p in prompts) == 6 for c in CAT_DIR)
    assert lambdas == [0.0, 0.5, 1.0, 2.0] and seeds == [42, 123, 2026] and alpha == 1.5
    base = CastConfig(**y, prompt="", mode="cast", lam=0.0, alpha=alpha, seed=seeds[0]); assert base.deterministic_self_attn is True
    root = base.output_dir; os.makedirs(root, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}; assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0s = {s: sample_initial_latents(CastConfig(**{**base.to_dict(), "seed": s})) for s in seeds}
    z0_sha = {s: sha_tensor(z0s[s]) for s in seeds}; assert len(set(z0_sha.values())) == len(seeds), "seeds must give different latents"
    manifest = {"experiment": "cast_diagnostic_v1", "prompt_file": pfile, "prompt_file_sha256": ds_sha, "n_prompts": len(prompts),
                "categories": {c: [p["id"] for p in prompts if p["category"] == c] for c in CAT_DIR}, "alpha": alpha, "lambdas": lambdas, "seeds": seeds,
                "initial_latent_sha256_by_seed": z0_sha, "generation_settings": {k: v for k, v in base.to_dict().items() if k not in ("prompt", "mode", "lam", "seed")},
                "stock_processor_classes": sorted(set(stock.values())), "versions": _versions(), "expected_conditions": len(prompts) * len(lambdas) * len(seeds),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(manifest, open(os.path.join(root, "experiment_manifest.json"), "w"), indent=2)
    print(f"[dataset] {pfile} sha256={ds_sha} prompts={len(prompts)}  seeds→z0 sha: { {s: v[:12] for s, v in z0_sha.items()} }")

    runs, failures = {}, []
    def do(p, seed, lam):
        d = cond_dir(root, p, seed, lam); mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp) and os.path.exists(os.path.join(d, "audio.wav")) and os.path.exists(os.path.join(d, "final_latents.pt")):
            m = json.load(open(mp)); src = "existing"
        else:
            cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "lam": lam, "seed": seed})
            m = run_one(pipe, cfg, initial_latents=z0s[seed], out_dir=d)["manifest"]; src = "generated"
        st = m["cast_processor_stats"]; sup = [v["mean_sparse_support_cond_half"] for v in st.values() if v.get("mean_sparse_support_cond_half") is not None]
        rec = {"prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "lambda": lam, "alpha": m["alpha"],
               "audio_path": os.path.join(d, "audio.wav"), "dir": d, "initial_latent_sha256": m["initial_latents"]["sha256"], "final_latent_norm": m["final_latent_norm"],
               "final_latent_sha256": m["final_latents_sha256"], "runtime_sec": m["runtime_sec"], "peak_gpu_memory_gb": m["peak_cuda_mem_gb"], "finite": m["finite"], "valid": finite_ok(m),
               "deterministic_self_attn": m["config"]["deterministic_self_attn"], "attn2_processor": m["attention_processor_class"]["attn2"], "attn1_processor": m["attention_processor_class"]["attn1"],
               "mean_entmax_support_cond": (sum(sup) / len(sup)) if sup else None, "source": src}
        runs[f"{p['id']}/seed_{seed}/lambda_{lam:g}"] = rec
        print(f"[{p['id']} seed={seed} lam={lam:g}] {src:9s} |zT|={m['final_latent_norm']:.2f} {m['runtime_sec']:.1f}s finite={rec['valid']}", flush=True)
        return rec

    # ---- pre-generation validation on one prompt/seed across the four lambdas ----
    pv = next(p for p in prompts if p["id"] == vp)
    vrecs = [do(pv, vs, l) for l in lambdas]
    after = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
    checks = {"prompt_file_24": len(prompts) == 24, "six_per_category": all(len(v) == 6 for v in manifest["categories"].values()), "dataset_sha_recorded": bool(ds_sha),
              "alpha_1p5": all(r["alpha"] == 1.5 for r in vrecs), "lambdas_exact": [r["lambda"] for r in vrecs] == [0.0, 0.5, 1.0, 2.0], "seeds_exact": seeds == [42, 123, 2026],
              "deterministic_self_attn": all(r["deterministic_self_attn"] for r in vrecs), "attn2_cast_16": all(r["attn2_processor"] == ["CastCrossAttnProcessor"] for r in vrecs),
              "attn1_stock": all(r["attn1_processor"] == ["AttnProcessor2_0"] for r in vrecs), "processors_restored": after == stock,
              "same_initial_latent_across_lambdas": len({r["initial_latent_sha256"] for r in vrecs}) == 1 and vrecs[0]["initial_latent_sha256"] == z0_sha[vs],
              "settings_match_milestone": all(manifest["generation_settings"][k] == v for k, v in {"checkpoint": "auffusion/auffusion", "num_inference_steps": 100, "guidance_scale": 7.5, "height": 256, "width": 1024, "dtype": "float16", "autocast": True, "deterministic_self_attn": True, "apply_to_uncond": True}.items()),
              "no_nan_inf": all(r["valid"] for r in vrecs)}
    manifest["pre_generation_validation"] = checks; json.dump(manifest, open(os.path.join(root, "experiment_manifest.json"), "w"), indent=2)
    print("[pre-generation validation]", checks, flush=True)
    if not all(checks.values()):
        print("VALIDATION FAILED - STOPPING"); json.dump({"runs": runs, "aborted": "validation failed"}, open(os.path.join(root, "generation_summary.json"), "w"), indent=2); sys.exit(1)

    t0 = time.time(); n_total = len(prompts) * len(seeds) * len(lambdas)
    for p in prompts:
        for seed in seeds:
            for lam in lambdas:
                if f"{p['id']}/seed_{seed}/lambda_{lam:g}" in runs: continue
                try: do(p, seed, lam)
                except Exception as e:
                    failures.append({"prompt_id": p["id"], "seed": seed, "lambda": lam, "error": repr(e), "trace": traceback.format_exc()}); print(f"[{p['id']} seed={seed} lam={lam:g}] FAILED {e!r}", flush=True)
            # per (prompt, seed) latent-equality check
            keys = [f"{p['id']}/seed_{seed}/lambda_{l:g}" for l in lambdas if f"{p['id']}/seed_{seed}/lambda_{l:g}" in runs]
            if keys and len({runs[k]["initial_latent_sha256"] for k in keys}) != 1: failures.append({"prompt_id": p["id"], "seed": seed, "error": "initial latent mismatch across lambdas"})
            done = len(runs); print(f"  progress {done}/{n_total}  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump({"runs": runs, "failures": failures}, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    latent_ok = all(len({runs[f"{p['id']}/seed_{s}/lambda_{l:g}"]["initial_latent_sha256"] for l in lambdas if f"{p['id']}/seed_{s}/lambda_{l:g}" in runs}) == 1 for p in prompts for s in seeds)
    summary = {"experiment_manifest": "experiment_manifest.json", "n_expected": n_total, "n_completed": len(runs), "n_valid": sum(r["valid"] for r in runs.values()),
               "n_generated_now": sum(r["source"] == "generated" for r in runs.values()), "failures": failures, "same_initial_latent_per_prompt_seed": latent_ok,
               "seed_latents_distinct": len(set(z0_sha.values())) == 3, "runs": runs}
    json.dump(summary, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    print(f"\nGENERATION DONE: completed {len(runs)}/{n_total} valid {summary['n_valid']} failures {len(failures)} latent_ok {latent_ok} elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
