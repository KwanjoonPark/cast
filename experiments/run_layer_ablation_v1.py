"""CAST layer ablation v1: PLADIS (lambda_entmax=2, alpha 1.5) on a resolution-defined subset of the 16 text cross-attention modules.

Design (so that the only difference to the Dense arm is lambda on the selected modules): every attn2 module runs the CAST processor; the non-selected modules use
lambda=0 (dense, fp32 path, identical to the sweep's Dense arm) and the selected modules use lambda=2. Dense and "all 16 layers" arms are NOT regenerated: they are the
sweep arms `dense` and `entmax_l2` of outputs/cfg_sweep_v1 (same machine, same settings, deterministic).
Resolution groups (latent 32x128): L0 high = down_blocks.0 + up_blocks.3 (5 modules, 4096 queries); L1 mid = down_blocks.1 + up_blocks.2 (5, 1024);
L2 low = down_blocks.2 + up_blocks.1 (5, 256); mid_block (1, 64). "low" = L2 + mid_block (6 modules).
Pre-generation validation (stops on failure): (A) all modules lambda=0 and no selected module == sweep Dense audio bitwise; (B) CAST lambda=2 on all 16 == sweep entmax_l2 bitwise;
(C) processor census per group. Resumable.
Usage: python experiments/run_layer_ablation_v1.py [--cfgs 3.0] [--groups high low] [--seeds 42 123 2026] [--limit-prompts N]"""
import argparse, hashlib, json, os, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention, apply_cast, restore_processors
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, _versions

GROUP_PREFIXES = {"high": ["down_blocks.0.", "up_blocks.3."], "mid": ["down_blocks.1.", "up_blocks.2."], "low": ["down_blocks.2.", "up_blocks.1.", "mid_block."]}
EXPECTED = {"high": 5, "mid": 5, "low": 6}
SWEEP = "outputs/cfg_sweep_v1"


def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", type=float, default=[3.0]); ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    ap.add_argument("--groups", nargs="+", default=["high", "low"]); ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--prompt-file", default="experiments/prompts/cast_diagnostic_v1.json"); ap.add_argument("--out", default="outputs/layer_ablation_v1")
    ap.add_argument("--lam", type=float, default=2.0)
    a = ap.parse_args()
    ds = json.load(open(a.prompt_file, encoding="utf-8")); prompts = ds["prompts"][: a.limit_prompts] if a.limit_prompts else ds["prompts"]
    base = CastConfig(prompt="", mode="cast", lam=a.lam, alpha=1.5, seed=a.seeds[0], deterministic_self_attn=True, collect_stats=False, output_dir=a.out)
    os.makedirs(a.out, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device); unet = pipe.unet
    stock = {n: type(p).__name__ for n, p in unet.attn_processors.items()}; allnames = sorted(select_cross_attention(unet)); assert len(stock) == 32 and len(allnames) == 16
    groups = {g: [n for n in allnames if any(n.startswith(p) for p in GROUP_PREFIXES[g])] for g in a.groups}
    for g, names in groups.items(): assert len(names) == EXPECTED[g], (g, len(names), names)
    if len(a.groups) > 1: assert not (set(groups[a.groups[0]]) & set(groups[a.groups[1]]))
    z0 = {s: sample_initial_latents(CastConfig(**{**base.to_dict(), "seed": s})) for s in a.seeds}
    z0_sha = {s: hashlib.sha256(z0[s].float().cpu().numpy().tobytes()).hexdigest() for s in a.seeds}

    def run(cfg_scale, seed, p, sel, d):
        """lambda=0 CAST on all non-selected modules, lambda=a.lam on `sel` (list of names; [] = none)."""
        rest = [n for n in allnames if n not in set(sel)]
        orig0 = apply_cast(unet, 1.5, 0.0, base.apply_to_uncond, False, attention_type="entmax_mix", only=rest)
        try:
            cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "seed": seed, "guidance_scale": cfg_scale, "lam": a.lam})
            return run_one(pipe, cfg, initial_latents=z0[seed], out_dir=d, only_modules=sel)["manifest"]
        finally:
            restore_processors(unet, orig0)

    # ---- pre-generation validation on prompt 0, seed 42 (or first seed), first cfg ----
    p0, s0, c0 = prompts[0], a.seeds[0], a.cfgs[0]
    ref = lambda arm: os.path.join(SWEEP, f"cfg{c0:g}", p0["category"], p0["id"], f"seed_{s0}", arm, "audio.wav")
    vdir = os.path.join(a.out, "_validation"); checks = {}
    if os.path.exists(ref("dense")) and os.path.exists(ref("entmax_l2")) and abs(a.lam - 2.0) < 1e-9:
        run(c0, s0, p0, [], os.path.join(vdir, "A_all_lambda0")); checks["A_all_lambda0_equals_sweep_dense"] = sha_file(os.path.join(vdir, "A_all_lambda0", "audio.wav")) == sha_file(ref("dense"))
        run(c0, s0, p0, allnames, os.path.join(vdir, "B_all_lambda2")); checks["B_all_lambda2_equals_sweep_entmax_l2"] = sha_file(os.path.join(vdir, "B_all_lambda2", "audio.wav")) == sha_file(ref("entmax_l2"))
    else:
        checks["skipped_sweep_reference_missing_or_lam_not_2"] = True
    census = {}
    for g, names in groups.items():
        rest = [n for n in allnames if n not in set(names)]
        o0 = apply_cast(unet, 1.5, 0.0, base.apply_to_uncond, False, attention_type="entmax_mix", only=rest)
        o1 = apply_cast(unet, 1.5, a.lam, base.apply_to_uncond, False, attention_type="entmax_mix", only=names)
        procs = unet.attn_processors
        lam_by_name = {n: getattr(procs[n], "lam", None) for n in allnames}
        n_sel = sum(1 for n in allnames if lam_by_name[n] == a.lam); n_dense = sum(1 for n in allnames if lam_by_name[n] == 0.0)
        census[g] = {"selected_modules": len(names), "n_lambda_%g" % a.lam: n_sel, "n_lambda_0": n_dense, "selected_are_exactly_group": {n for n in allnames if lam_by_name[n] == a.lam} == set(names),
                     "attn1_untouched": all(type(p).__name__ == "AttnProcessor2_0" for n, p in procs.items() if n.endswith("attn1.processor"))}
        restore_processors(unet, {**o0, **o1})
        checks[f"census_{g}"] = census[g]["selected_are_exactly_group"] and n_sel == EXPECTED[g] and n_dense == 16 - EXPECTED[g] and census[g]["attn1_untouched"]
    after = {n: type(p).__name__ for n, p in unet.attn_processors.items()}
    checks["processors_restored_to_stock"] = after == stock
    json.dump({"experiment": "cast_layer_ablation_v1", "groups": groups, "lam": a.lam, "cfgs": a.cfgs, "seeds": a.seeds, "n_prompts": len(prompts), "initial_latent_sha256_by_seed": z0_sha,
               "validation": checks, "census": census, "versions": _versions(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(a.out, "experiment_manifest.json"), "w"), indent=2)
    print("[validation]", checks, flush=True)
    hard = {k: v for k, v in checks.items() if k.startswith(("A_", "B_", "census_", "processors_restored"))}
    if not all(hard.values()):
        print("VALIDATION FAILED - STOPPING", hard); sys.exit(1)

    total = len(a.cfgs) * len(a.seeds) * len(prompts) * len(groups); done, fails, t0 = 0, [], time.time()
    for cfg_scale in a.cfgs:
        for seed in a.seeds:
            for p in prompts:
                for g, names in groups.items():
                    d = os.path.join(a.out, f"cfg{cfg_scale:g}", p["category"], p["id"], f"seed_{seed}", f"pladis_{g}"); done += 1
                    if os.path.exists(os.path.join(d, "audio.wav")) and os.path.exists(os.path.join(d, "manifest.json")): continue
                    try: run(cfg_scale, seed, p, names, d)
                    except Exception as e:
                        fails.append({"cfg": cfg_scale, "seed": seed, "prompt": p["id"], "group": g, "error": repr(e), "trace": traceback.format_exc()}); print("FAILED", p["id"], g, repr(e), flush=True)
                print(f"cfg {cfg_scale:g} seed {seed} {p['id']}  [{done}/{total}]  {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump({"failures": fails, "completed_slots": done, "total": total}, open(os.path.join(a.out, "generation_summary.json"), "w"), indent=1)
    print(f"DONE {done}/{total} failures {len(fails)} {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
