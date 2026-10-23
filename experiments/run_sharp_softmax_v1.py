"""Sharp Softmax diagnostic generation (Softmax(beta*logits) on the 16 attn2 modules), reusing Diagnostic V1 exactly.
Stages: (0) dataset/latent checks vs Diagnostic V1; (1) entmax-path regression: regenerate one lambda=1 and one lambda=2
condition and require bitwise equality with Diagnostic V1; (2) beta=1 parity+entropy pass over all 72 (prompt, seed)
units without saving audio, requiring bitwise equality of final latents and audio with Diagnostic V1 lambda=0, and
collecting entropy (+ counterfactual probes at beta in probe_betas and entmax15 on the same logits); the beta_1.00
condition directories are then filled by copying the Diagnostic V1 lambda=0 files with provenance; (3) generate
beta>1 conditions (resumable). Any failed check stops the script before stage 3."""
import argparse, hashlib, json, os, shutil, statistics, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, yaml
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, _versions

CAT_DIR = {"DISCRETE_SINGLE": "A_DISCRETE_SINGLE", "DIFFUSE_SINGLE": "B_DIFFUSE_SINGLE", "DISCRETE_DISCRETE": "C_DISCRETE_DISCRETE", "DISCRETE_DIFFUSE": "D_DISCRETE_DIFFUSE"}
LOG77 = __import__("math").log(77)


def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def sha_tensor(t): return hashlib.sha256(t.float().cpu().numpy().tobytes()).hexdigest()
def finite_ok(m): return all(v for v in m["finite"].values() if v is not None)
def bdir(root, p, seed, beta): return os.path.join(root, CAT_DIR[p["category"]], p["id"], f"seed_{seed}", f"beta_{beta:.2f}")
def ddir(droot, p, seed, lam): return os.path.join(droot, CAT_DIR[p["category"]], p["id"], f"seed_{seed}", f"lambda_{lam:g}")


def entropy_summary(m, probe_betas):
    """Run-level entropy stats from per-layer per-step records (conditional half)."""
    st = m["cast_processor_stats"]; layers = sorted(st)
    by_layer = [st[n]["mean_entropy_cond"] for n in layers]
    steps = [st[n]["entropy_by_step"] for n in layers]
    by_step = [statistics.mean(s[i] for s in steps) for i in range(len(steps[0]))] if steps and steps[0] else None
    H = statistics.mean(by_layer)
    out = {"mean_entropy": H, "normalized_entropy": H / LOG77, "effective_support": __import__("math").exp(H), "entropy_by_layer": by_layer, "entropy_by_step": by_step}
    if st[layers[0]].get("probe_entropy_by_step"):
        out["probe"] = {}
        for b in probe_betas:
            hb = statistics.mean(statistics.mean(st[n]["probe_entropy_by_step"][str(b)]) for n in layers)
            out["probe"][f"beta_{b:.2f}"] = {"mean_entropy": hb, "normalized_entropy": hb / LOG77, "effective_support": __import__("math").exp(hb)}
        he = statistics.mean(statistics.mean(st[n]["probe_entmax_entropy_by_step"]) for n in layers)
        se = statistics.mean(statistics.mean(st[n]["probe_entmax_support_by_step"]) for n in layers)
        out["probe"]["entmax15"] = {"mean_entropy": he, "normalized_entropy": he / LOG77, "effective_support": __import__("math").exp(he), "exact_support": se}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="experiments/configs/sharp_softmax_v1.yaml"); args = ap.parse_args()
    y = yaml.safe_load(open(args.config)); betas = [float(b) for b in y.pop("betas")]; seeds = [int(s) for s in y.pop("seeds")]
    pfile, psha, droot, probe = y.pop("prompt_file"), y.pop("prompt_file_sha256"), y.pop("diagnostic_dir"), [float(b) for b in y.pop("probe_betas")]
    assert sha_file(pfile) == psha, "prompt dataset SHA-256 mismatch vs Diagnostic V1"
    ds = json.load(open(pfile)); prompts = ds["prompts"]; assert len(prompts) == 24 and betas == [1.0, 1.25, 1.5, 2.0] and seeds == [42, 123, 2026]
    dman = json.load(open(os.path.join(droot, "experiment_manifest.json"))); assert dman["prompt_file_sha256"] == psha
    base = CastConfig(**y, prompt="", mode="sharp_softmax", beta=1.0, seed=seeds[0]); assert base.deterministic_self_attn is True and base.resolved().attention_type == "softmax"
    gs = {k: v for k, v in base.to_dict().items() if k not in ("prompt", "mode", "lam", "seed", "beta", "attention_type", "alpha")}
    dgs = dman["generation_settings"]; mism = {k: (gs[k], dgs.get(k)) for k in gs if k != "output_dir" and dgs.get(k) != gs[k]}
    assert not mism, f"generation settings differ from Diagnostic V1: {mism}"
    root = base.output_dir; os.makedirs(root, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}; assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0s = {s: sample_initial_latents(CastConfig(**{**base.to_dict(), "seed": s})) for s in seeds}; z0_sha = {s: sha_tensor(z0s[s]) for s in seeds}
    assert all(z0_sha[s] == dman["initial_latent_sha256_by_seed"][str(s)] for s in seeds), "initial latents differ from Diagnostic V1"
    manifest = {"experiment": "cast_sharp_softmax_v1", "builds_on": "outputs/diagnostic_v1", "prompt_file": pfile, "prompt_file_sha256": psha, "n_prompts": 24,
                "attention_type": "softmax", "betas": betas, "seeds": seeds, "initial_latent_sha256_by_seed": z0_sha, "generation_settings": gs,
                "stock_processor_classes": sorted(set(stock.values())), "versions": _versions(), "expected_conditions": 24 * 3 * 4, "probe_betas": probe,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    def dump(): json.dump(manifest, open(os.path.join(root, "experiment_manifest.json"), "w"), indent=2)
    dump(); print(f"[stage0] dataset sha OK, settings match Diagnostic V1, seed latents bitwise equal: {z0_sha}", flush=True)

    # ---- stage 1: entmax path regression after the processor refactor -------------------------------------
    pD = next(p for p in prompts if p["id"] == "D01"); reg = {}
    for lam in [1.0, 2.0]:
        cfg = CastConfig(**{**base.to_dict(), "mode": "cast", "attention_type": "entmax_mix", "prompt": pD["prompt"], "lam": lam, "seed": 42})
        r = run_one(pipe, cfg, initial_latents=z0s[42], save=False)["manifest"]; dm = json.load(open(os.path.join(ddir(droot, pD, 42, lam), "manifest.json")))
        reg[f"D01/seed_42/lambda_{lam:g}"] = {"final_latent_sha_equal": r["final_latents_sha256"] == dm["final_latents_sha256"], "audio_sha_equal": r["audio_sha256"] == dm["audio_sha256"]}
    manifest["checks"]["entmax_path_regression"] = reg; dump(); print("[stage1] entmax path regression:", reg, flush=True)
    if not all(all(v.values()) for v in reg.values()): print("ENTMAX PATH CHANGED - STOPPING"); sys.exit(1)

    # ---- stage 2: beta=1 parity + entropy pass over all 72 units (no audio saved) -------------------------
    parity = {}; t0 = time.time()
    for p in prompts:
        for seed in seeds:
            cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "beta": 1.0, "seed": seed})
            r = run_one(pipe, cfg, initial_latents=z0s[seed], save=False, probe_betas=probe, probe_entmax=True)["manifest"]
            dm = json.load(open(os.path.join(ddir(droot, p, seed, 0.0), "manifest.json")))
            ent = entropy_summary(r, probe)
            parity[f"{p['id']}/seed_{seed}"] = {"final_latent_sha_equal": r["final_latents_sha256"] == dm["final_latents_sha256"], "audio_sha_equal": r["audio_sha256"] == dm["audio_sha256"],
                                                "final_latent_norm": r["final_latent_norm"], "diag_final_latent_norm": dm["final_latent_norm"], "entropy": ent, "runtime_sec": r["runtime_sec"]}
            pu = parity[f"{p['id']}/seed_{seed}"]; pr = {k: round(v["effective_support"], 1) for k, v in ent["probe"].items()}
            print(f"[stage2 {p['id']} seed={seed}] beta=1 latent==diag_lambda0: {pu['final_latent_sha_equal']} audio: {pu['audio_sha_equal']} H_norm={ent['normalized_entropy']:.3f} effS={ent['effective_support']:.1f} probes={pr}", flush=True)
            # fill beta_1.00 dir by copying the Diagnostic V1 lambda=0 files (bitwise identical content) with provenance
            src, dst = ddir(droot, p, seed, 0.0), bdir(root, p, seed, 1.0); os.makedirs(dst, exist_ok=True)
            for f in ["audio.wav", "spectrogram.png", "final_latents.pt"]: shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            m2 = dict(dm); m2["attention_type"] = "softmax"; m2["beta"] = 1.0
            m2["provenance"] = {"reused_from": src, "note": "Diagnostic V1 lambda=0 (dense Softmax) output reused as Sharp Softmax beta=1.00; a beta=1 re-run reproduced its final latents and audio bitwise (see experiment_manifest.json checks.beta1_parity)"}
            m2["entropy_stats_cond"] = ent; m2["entropy_stats_note"] = "measured during the bitwise-identical beta=1 parity re-run (conditional CFG half); probes = counterfactual Softmax(b*logits)/entmax15 on the same logits"
            json.dump(m2, open(os.path.join(dst, "manifest.json"), "w"), indent=2)
    ok = all(v["final_latent_sha_equal"] and v["audio_sha_equal"] for v in parity.values())
    manifest["checks"]["beta1_parity"] = {"n_units": len(parity), "all_final_latent_bitwise_equal": all(v["final_latent_sha_equal"] for v in parity.values()),
                                          "all_audio_bitwise_equal": all(v["audio_sha_equal"] for v in parity.values()), "elapsed_min": (time.time() - t0) / 60, "units": parity}; dump()
    print(f"[stage2] beta=1 parity over 72 units: {'PASS' if ok else 'FAIL'} ({(time.time()-t0)/60:.1f} min)", flush=True)
    if not ok: print("BETA=1 PARITY FAILED - STOPPING"); sys.exit(1)

    # ---- stage 3: beta > 1 generation (resumable) -----------------------------------------------------------
    runs, failures = {}, []; t0 = time.time(); n_total = 24 * 3 * 4
    for p in prompts:
        for seed in seeds:
            runs[f"{p['id']}/seed_{seed}/beta_1.00"] = {"prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "beta": 1.0, "attention_type": "softmax",
                "audio_path": os.path.join(bdir(root, p, seed, 1.0), "audio.wav"), "dir": bdir(root, p, seed, 1.0), "initial_latent_sha256": z0_sha[seed],
                "final_latent_norm": parity[f"{p['id']}/seed_{seed}"]["diag_final_latent_norm"], "valid": True, "source": "reused_diagnostic_v1_lambda0",
                "entropy": {k: v for k, v in parity[f"{p['id']}/seed_{seed}"]["entropy"].items() if k not in ("entropy_by_layer", "entropy_by_step")}}
            for beta in betas[1:]:
                key = f"{p['id']}/seed_{seed}/beta_{beta:.2f}"; d = bdir(root, p, seed, beta); mp = os.path.join(d, "manifest.json")
                try:
                    if os.path.exists(mp) and os.path.exists(os.path.join(d, "audio.wav")):
                        m = json.load(open(mp)); src = "existing"
                    else:
                        cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "beta": beta, "seed": seed})
                        m = run_one(pipe, cfg, initial_latents=z0s[seed], out_dir=d)["manifest"]; src = "generated"
                        m["entropy_stats_cond"] = entropy_summary(m, []); json.dump(m, open(mp, "w"), indent=2)
                    ent = m["entropy_stats_cond"]
                    runs[key] = {"prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "beta": beta, "attention_type": "softmax",
                                 "audio_path": os.path.join(d, "audio.wav"), "dir": d, "initial_latent_sha256": m["initial_latents"]["sha256"], "final_latent_norm": m["final_latent_norm"],
                                 "final_latent_sha256": m["final_latents_sha256"], "runtime_sec": m["runtime_sec"], "peak_gpu_memory_gb": m["peak_cuda_mem_gb"], "finite": m["finite"], "valid": finite_ok(m),
                                 "attn2_processor": m["attention_processor_class"]["attn2"], "attn1_processor": m["attention_processor_class"]["attn1"], "source": src,
                                 "entropy": {k: v for k, v in ent.items() if k not in ("entropy_by_layer", "entropy_by_step")}}
                    print(f"[{p['id']} seed={seed} beta={beta:.2f}] {src:9s} |zT|={m['final_latent_norm']:.2f} {m['runtime_sec']:.1f}s H_norm={ent['normalized_entropy']:.3f} effS={ent['effective_support']:.1f} finite={finite_ok(m)}", flush=True)
                except Exception as e:
                    failures.append({"prompt_id": p["id"], "seed": seed, "beta": beta, "error": repr(e), "trace": traceback.format_exc()}); print(f"[{p['id']} seed={seed} beta={beta}] FAILED {e!r}", flush=True)
            done = sum(1 for k in runs if not k.endswith("beta_1.00")); print(f"  progress {done}/216 new  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump({"runs": runs, "failures": failures}, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    latent_ok = all(len({runs[f"{p['id']}/seed_{s}/beta_{b:.2f}"]["initial_latent_sha256"] for b in betas if f"{p['id']}/seed_{s}/beta_{b:.2f}" in runs}) == 1 for p in prompts for s in seeds)
    summary = {"experiment_manifest": "experiment_manifest.json", "n_expected": n_total, "n_completed": len(runs), "n_valid": sum(r["valid"] for r in runs.values()),
               "n_generated_now": sum(r["source"] == "generated" for r in runs.values()), "n_reused_beta1": sum(r["source"] == "reused_diagnostic_v1_lambda0" for r in runs.values()),
               "failures": failures, "same_initial_latent_per_prompt_seed": latent_ok, "runs": runs}
    json.dump(summary, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    print(f"\nGENERATION DONE: completed {len(runs)}/{n_total} (new {summary['n_generated_now']}, reused {summary['n_reused_beta1']}) valid {summary['n_valid']} failures {len(failures)} latent_ok {latent_ok} elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
