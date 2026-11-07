"""Runtime proof that guidance_scale=1.0 is TRUE no-CFG in this Auffusion / diffusers-0.18.2 stack, plus the no_cfg_v1
generation smoke test. Diagnostic only: a forward pre-hook on pipe.unet records, for every UNet call, the latent batch size,
the encoder_hidden_states batch size and the number of all-zero conditioning rows (Auffusion's null condition is zeros).
The hook is removed at the end; no production code is modified.

Runs (one prompt, one seed, full generation settings unless noted):
  A. Dense and PLADIS (entmax_l2) at guidance 1.0 into the experiment tree (<out>/cfg1/...; the sweep driver later skips them)
  B. Dense at guidance 1.0 again into a scratch dir -> bitwise determinism (audio + final latent SHA)
  C. contrast: Dense at guidance 7.5, 3 steps, not saved -> expect latent batch 2 and one zero conditioning row per call
  D. behavioural proof at guidance 1.0 (10 steps, stock processors): negative_prompt=None vs a real negative prompt ->
     identical audio and spectrogram, i.e. the unconditional branch is never built or used.
Usage: python experiments/verify_no_cfg_runtime.py [--out outputs/no_cfg_v1] [--scratch DIR] [--prompt-index 0] [--seed 42]"""
import argparse, contextlib, hashlib, json, os, shutil, sys, time
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, SAMPLE_RATE
from run_cfg_sweep_v1 import ARMS


class UNetProbe:
    def __init__(self, unet):
        self.records = []; self.handle = unet.register_forward_pre_hook(self._hook, with_kwargs=True)
    def _hook(self, module, args, kwargs):
        sample = args[0] if args else kwargs["sample"]; ehs = kwargs.get("encoder_hidden_states")
        zero_rows = int((ehs.float().abs().sum(dim=(1, 2)) == 0).sum().item()) if ehs is not None else None
        self.records.append({"latent_batch": int(sample.shape[0]), "cond_batch": int(ehs.shape[0]) if ehs is not None else None, "zero_cond_rows": zero_rows})
    def summary(self):
        r = self.records
        return {"unet_calls": len(r), "latent_batch_values": sorted({x["latent_batch"] for x in r}), "cond_batch_values": sorted({x["cond_batch"] for x in r}),
                "zero_cond_rows_total": sum(x["zero_cond_rows"] for x in r), "zero_cond_rows_per_call_values": sorted({x["zero_cond_rows"] for x in r})}
    def reset(self): self.records = []
    def remove(self): self.handle.remove()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/no_cfg_v1"); ap.add_argument("--scratch", default=None)
    ap.add_argument("--prompt-file", default="experiments/prompts/cast_diagnostic_v1.json"); ap.add_argument("--prompt-index", type=int, default=0); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); scratch = a.scratch or os.path.join(a.out, "_smoke_rerun"); os.makedirs(a.out, exist_ok=True)
    p = json.load(open(a.prompt_file, encoding="utf-8"))["prompts"][a.prompt_index]
    base = CastConfig(prompt=p["prompt"], mode="cast", lam=0.0, alpha=1.5, seed=a.seed, guidance_scale=1.0, deterministic_self_attn=True, collect_stats=False, output_dir=a.out)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device); assert len(select_cross_attention(pipe.unet)) == 16
    z0 = sample_initial_latents(base); z0_sha = hashlib.sha256(z0.float().cpu().numpy().tobytes()).hexdigest()
    probe = UNetProbe(pipe.unet); res = {"prompt_id": p["id"], "prompt": p["prompt"], "seed": a.seed, "initial_latent_sha256": z0_sha, "gpu": torch.cuda.get_device_name(0),
                                        "code_path": {"file": "third_party/Auffusion/auffusion_pipeline.py", "rule": "do_classifier_free_guidance = guidance_scale > 1.0", "guidance_scale_used": 1.0}}
    # ---- A. Dense and PLADIS at guidance 1.0 (full settings), saved into the experiment tree ----
    man = {}
    for arm in ["dense", "entmax_l2"]:
        probe.reset(); d = os.path.join(a.out, "cfg1", p["category"], p["id"], f"seed_{a.seed}", arm)
        cfg = CastConfig(**{**base.to_dict(), **ARMS[arm]}); t0 = time.time(); r = run_one(pipe, cfg, initial_latents=z0, out_dir=d); m = r["manifest"]; man[arm] = m
        res[f"A_{arm}"] = {**probe.summary(), "dir": d, "runtime_s": round(time.time() - t0, 1), "initial_latents_sha256": m["initial_latents"]["sha256"], "audio_sha256": m["audio_sha256"],
                           "final_latents_sha256": m["final_latents_sha256"], "finite": m["finite"], "n_audio_samples": int(len(r["audio"])), "duration_s": len(r["audio"]) / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
                           "attn2_processor_classes": m.get("attn2_processor_classes"), "attn1_processor_classes": m.get("attn1_processor_classes")}
    # ---- B. determinism: Dense again into scratch ----
    probe.reset(); cfg = CastConfig(**{**base.to_dict(), **ARMS["dense"]}); r2 = run_one(pipe, cfg, initial_latents=z0, out_dir=os.path.join(scratch, "dense_rerun"))
    res["B_dense_rerun"] = {**probe.summary(), "audio_sha256": r2["manifest"]["audio_sha256"], "final_latents_sha256": r2["manifest"]["final_latents_sha256"],
                            "bitwise_equal_to_A": r2["manifest"]["audio_sha256"] == man["dense"]["audio_sha256"] and r2["manifest"]["final_latents_sha256"] == man["dense"]["final_latents_sha256"]}
    # ---- C. contrast: guidance 7.5, 3 steps, not saved ----
    probe.reset(); cfg = CastConfig(**{**base.to_dict(), **ARMS["dense"], "guidance_scale": 7.5, "num_inference_steps": 3}); run_one(pipe, cfg, initial_latents=z0, out_dir=None, save=False)
    res["C_contrast_cfg7.5_3steps"] = probe.summary()
    # ---- D. behavioural: negative_prompt is ignored at guidance 1.0 (stock processors, 10 steps) ----
    outs = {}
    for tag, neg in [("none", None), ("explicit", "A loud explosion.")]:
        probe.reset(); gen = torch.Generator(device=base.device).manual_seed(a.seed)
        with torch.autocast("cuda"), torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            o = pipe(prompt=p["prompt"], negative_prompt=neg, num_inference_steps=10, guidance_scale=1.0, generator=gen, height=base.height, width=base.width, latents=z0)
        outs[tag] = {"audio_sha256": hashlib.sha256(np.asarray(o.audios[0]).tobytes()).hexdigest(), "spec_sha256": hashlib.sha256(o.spectrograms[0].float().cpu().numpy().tobytes()).hexdigest(), **probe.summary()}
    res["D_negative_prompt_invariance"] = {**outs, "identical": outs["none"]["audio_sha256"] == outs["explicit"]["audio_sha256"] and outs["none"]["spec_sha256"] == outs["explicit"]["spec_sha256"]}
    probe.remove()
    # ---- verdicts ----
    A = [res["A_dense"], res["A_entmax_l2"]]
    res["checks"] = {
        "no_cfg_latent_batch_is_1": all(x["latent_batch_values"] == [1] for x in A), "no_cfg_cond_batch_is_1": all(x["cond_batch_values"] == [1] for x in A),
        "no_cfg_unconditional_rows_total": sum(x["zero_cond_rows_total"] for x in A), "no_cfg_unconditional_forward_passes": 0 if all(x["zero_cond_rows_total"] == 0 and x["latent_batch_values"] == [1] for x in A) else "NONZERO",
        "same_initial_latent_dense_vs_pladis": man["dense"]["initial_latents"]["sha256"] == man["entmax_l2"]["initial_latents"]["sha256"] == z0_sha,
        "finite": all(all(v for v in x["finite"].values() if v is not None) for x in A), "duration_10s_16k": all(x["n_audio_samples"] == SAMPLE_RATE * 10 for x in A),
        "deterministic_rerun": res["B_dense_rerun"]["bitwise_equal_to_A"], "dense_vs_pladis_audio_differ": man["dense"]["audio_sha256"] != man["entmax_l2"]["audio_sha256"],
        "contrast_cfg7.5_latent_batch_2_and_one_zero_row": res["C_contrast_cfg7.5_3steps"]["latent_batch_values"] == [2] and res["C_contrast_cfg7.5_3steps"]["zero_cond_rows_per_call_values"] == [1],
        "negative_prompt_ignored_at_guidance_1": res["D_negative_prompt_invariance"]["identical"]}
    res["checks"]["ALL_PASS"] = all(v is True or v == 0 for k, v in res["checks"].items() if k != "no_cfg_unconditional_rows_total")
    json.dump(res, open(os.path.join(a.out, "runtime_verification.json"), "w"), indent=2)
    shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(res["checks"], indent=1)); print("A dense:", {k: res["A_dense"][k] for k in ("unet_calls", "latent_batch_values", "cond_batch_values", "zero_cond_rows_total", "runtime_s")})
    print("C contrast:", res["C_contrast_cfg7.5_3steps"]); print("written", os.path.join(a.out, "runtime_verification.json"))


if __name__ == "__main__":
    main()
