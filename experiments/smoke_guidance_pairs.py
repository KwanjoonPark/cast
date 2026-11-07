"""Generation smoke test for a Dense / PLADIS(entmax_l2) pair at given guidance scales, with the UNet probe from
verify_no_cfg_runtime.py (diagnostic hook, removed at exit). For each requested CFG it generates the pair for one prompt and
one seed INTO the experiment tree (<out>/cfg<g>/<cat>/<pid>/seed_<s>/<arm>/, so run_cfg_sweep_v1.py later skips them) and
re-runs Dense into a scratch dir to check bitwise determinism. Checks per CFG: GPU name, CFG code path (latent/cond batch
2 and one null-condition row per UNet call for CFG > 1; batch 1 and none for CFG = 1), same initial latent for both arms,
finite output, 10.0 s at 16 kHz, deterministic rerun. Writes <out>/smoke_verification.json.
Usage: python experiments/smoke_guidance_pairs.py --out outputs/a5000_guidance_sweep_v1 --cfgs 7.5 1.5 [--prompt-index 0] [--seed 42]"""
import argparse, hashlib, json, os, shutil, sys, time
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, SAMPLE_RATE
from run_cfg_sweep_v1 import ARMS
from verify_no_cfg_runtime import UNetProbe


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--cfgs", nargs="+", type=float, required=True); ap.add_argument("--scratch", default=None)
    ap.add_argument("--prompt-file", default="experiments/prompts/cast_diagnostic_v1.json"); ap.add_argument("--prompt-index", type=int, default=0); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--expect-gpu", default="NVIDIA RTX A5000")
    a = ap.parse_args(); scratch = a.scratch or os.path.join(a.out, "_smoke_rerun"); os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    if gpu != a.expect_gpu: print(f"WRONG GPU: {gpu} != {a.expect_gpu}"); sys.exit(2)
    p = json.load(open(a.prompt_file, encoding="utf-8"))["prompts"][a.prompt_index]
    base = CastConfig(prompt=p["prompt"], mode="cast", lam=0.0, alpha=1.5, seed=a.seed, deterministic_self_attn=True, collect_stats=False, output_dir=a.out)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device); assert len(select_cross_attention(pipe.unet)) == 16
    z0 = sample_initial_latents(base); z0_sha = hashlib.sha256(z0.float().cpu().numpy().tobytes()).hexdigest()
    probe = UNetProbe(pipe.unet); res = {"gpu": gpu, "prompt_id": p["id"], "prompt": p["prompt"], "seed": a.seed, "initial_latent_sha256": z0_sha, "cfgs": {}}
    all_ok = True
    for g in a.cfgs:
        r = {}; man = {}
        for arm in ["dense", "entmax_l2"]:
            probe.reset(); d = os.path.join(a.out, f"cfg{g:g}", p["category"], p["id"], f"seed_{a.seed}", arm)
            cfg = CastConfig(**{**base.to_dict(), **ARMS[arm], "guidance_scale": g}); t0 = time.time(); o = run_one(pipe, cfg, initial_latents=z0, out_dir=d); m = o["manifest"]; man[arm] = m
            r[arm] = {**probe.summary(), "dir": d, "runtime_s": round(time.time() - t0, 1), "initial_latents_sha256": m["initial_latents"]["sha256"], "audio_sha256": m["audio_sha256"], "final_latents_sha256": m["final_latents_sha256"],
                      "finite": all(v for v in m["finite"].values() if v is not None), "n_audio_samples": int(len(o["audio"])), "sample_rate": SAMPLE_RATE, "attn2": m["attention_processor_class"]["attn2"], "attn1": m["attention_processor_class"]["attn1"]}
        probe.reset(); cfg = CastConfig(**{**base.to_dict(), **ARMS["dense"], "guidance_scale": g}); o2 = run_one(pipe, cfg, initial_latents=z0, out_dir=os.path.join(scratch, f"cfg{g:g}_dense_rerun"))
        rerun_equal = o2["manifest"]["audio_sha256"] == man["dense"]["audio_sha256"] and o2["manifest"]["final_latents_sha256"] == man["dense"]["final_latents_sha256"]
        exp_batch, exp_zero = ([2], [1]) if g > 1.0 else ([1], [0])
        checks = {"cfg_code_path": all(r[x]["latent_batch_values"] == exp_batch and r[x]["cond_batch_values"] == exp_batch and r[x]["zero_cond_rows_per_call_values"] == exp_zero for x in r),
                  "same_initial_latent": man["dense"]["initial_latents"]["sha256"] == man["entmax_l2"]["initial_latents"]["sha256"] == z0_sha, "finite": all(r[x]["finite"] for x in r),
                  "duration_10s_16k": all(r[x]["n_audio_samples"] == SAMPLE_RATE * 10 for x in r), "deterministic_rerun": rerun_equal, "dense_vs_pladis_differ": man["dense"]["audio_sha256"] != man["entmax_l2"]["audio_sha256"],
                  "attn2_cast_attn1_stock": all(r[x]["attn2"] == ["CastCrossAttnProcessor"] and r[x]["attn1"] == ["AttnProcessor2_0"] for x in r)}
        checks["PASS"] = all(checks.values()); all_ok &= checks["PASS"]; res["cfgs"][f"{g:g}"] = {"runs": r, "checks": checks}
        print(f"CFG {g:g}: {checks} | dense {r['dense']['unet_calls']} calls batch {r['dense']['latent_batch_values']} zero-rows/call {r['dense']['zero_cond_rows_per_call_values']} {r['dense']['runtime_s']} s")
    probe.remove(); res["ALL_PASS"] = all_ok
    json.dump(res, open(os.path.join(a.out, "smoke_verification.json"), "w"), indent=2); shutil.rmtree(scratch, ignore_errors=True)
    print("ALL_PASS", all_ok, "->", os.path.join(a.out, "smoke_verification.json")); sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
