"""CAST CFG sweep v1: the paper's 24 frozen prompts (cast_diagnostic_v1.json) x guidance_scale x 5 attention arms x seeds {42,123,2026}.
Arms: dense (CAST path, lambda_entmax=0), sharp_b1.5 (Softmax(1.5*Z)), entmax_l0.5, entmax_l1 (sparse), entmax_l2 (PLADIS). All other settings as Diagnostic V1
(100 steps, 256x1024, fp16, deterministic self-attn, same initial latent per seed for every CFG and arm). Only guidance_scale differs between CFG levels.
Resumable (existing audio.wav + manifest.json skipped). Order: CFG outermost (in --cfgs order), then seed, prompt, arm, so partial runs stay balanced.
Usage: python experiments/run_cfg_sweep_v1.py [--cfgs 7.5 1.5 3.0] [--seeds 42 123 2026] [--limit-prompts N] [--out outputs/cfg_sweep_v1]"""
import argparse, hashlib, json, os, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, _versions

ARMS = {"dense": dict(mode="cast", attention_type="entmax_mix", lam=0.0),
        "sharp_b1.5": dict(mode="sharp_softmax", beta=1.5),
        "entmax_l0.5": dict(mode="cast", attention_type="entmax_mix", lam=0.5),
        "entmax_l1": dict(mode="cast", attention_type="entmax_mix", lam=1.0),
        "entmax_l2": dict(mode="cast", attention_type="entmax_mix", lam=2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", type=float, default=[7.5, 1.5, 3.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    ap.add_argument("--prompt-file", default="experiments/prompts/cast_diagnostic_v1.json")
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="outputs/cfg_sweep_v1")
    a = ap.parse_args()
    ds = json.load(open(a.prompt_file, encoding="utf-8")); prompts = ds["prompts"][: a.limit_prompts] if a.limit_prompts else ds["prompts"]
    base = CastConfig(prompt="", mode="cast", lam=0.0, alpha=1.5, seed=a.seeds[0], deterministic_self_attn=True, collect_stats=False, output_dir=a.out)
    os.makedirs(a.out, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}; assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0 = {s: sample_initial_latents(CastConfig(**{**base.to_dict(), "seed": s})) for s in a.seeds}
    z0_sha = {s: hashlib.sha256(z0[s].float().cpu().numpy().tobytes()).hexdigest() for s in a.seeds}
    json.dump({"experiment": "cast_cfg_sweep_v1", "prompt_file": a.prompt_file, "prompt_file_sha256": hashlib.sha256(open(a.prompt_file, "rb").read()).hexdigest(),
               "n_prompts": len(prompts), "cfgs": a.cfgs, "seeds": a.seeds, "arms": {k: ARMS[k] for k in a.arms}, "initial_latent_sha256_by_seed": z0_sha,
               "generation_settings": {k: v for k, v in base.to_dict().items() if k not in ("prompt", "mode", "lam", "seed")}, "versions": _versions(),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(a.out, "experiment_manifest.json"), "w"), indent=2)
    total = len(a.cfgs) * len(a.seeds) * len(prompts) * len(a.arms); done, fails, t0 = 0, [], time.time()
    for cfg_scale in a.cfgs:
        for seed in a.seeds:
            for p in prompts:
                for arm in a.arms:
                    d = os.path.join(a.out, f"cfg{cfg_scale:g}", p["category"], p["id"], f"seed_{seed}", arm)
                    done += 1
                    if os.path.exists(os.path.join(d, "audio.wav")) and os.path.exists(os.path.join(d, "manifest.json")): continue
                    try:
                        cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "seed": seed, "guidance_scale": cfg_scale, **ARMS[arm]})
                        m = run_one(pipe, cfg, initial_latents=z0[seed], out_dir=d)["manifest"]
                        assert m["initial_latents"]["sha256"] and all(v for v in m["finite"].values() if v is not None), "non-finite output"
                    except Exception as e:
                        fails.append({"cfg": cfg_scale, "seed": seed, "prompt": p["id"], "arm": arm, "error": repr(e), "trace": traceback.format_exc()}); print("FAILED", cfg_scale, seed, p["id"], arm, repr(e), flush=True)
                print(f"cfg {cfg_scale:g} seed {seed} {p['id']}  [{done}/{total}]  {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump({"failures": fails, "completed_slots": done, "total": total}, open(os.path.join(a.out, "generation_summary.json"), "w"), indent=1)
    print(f"DONE {done}/{total} failures {len(fails)} {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
