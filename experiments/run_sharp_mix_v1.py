"""Partial sharp guidance generation: attention_type=sharp_mix, beta=1.5 fixed, lambda_sharp sweep, Diagnostic V1 prompts/seeds.
Stage 0: dataset SHA / settings / latent checks vs Diagnostic V1 and Sharp Softmax V1. Stage 1: regeneration checks —
lambda_sharp=0 must reproduce the dense output (Diagnostic lambda_entmax=0 == Sharp beta=1) bitwise and lambda_sharp=1 must
reproduce Sharp beta=1.5 bitwise (one unit each, plus a second unit for lambda_sharp=1). Stage 2: fill the two reused
conditions by copying with provenance. Stage 3: generate lambda_sharp in {0.25,0.5,0.75,1.25,1.5} (resumable)."""
import argparse, hashlib, json, math, os, shutil, statistics, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, yaml
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype, _versions

CAT_DIR = {"DISCRETE_SINGLE": "A_DISCRETE_SINGLE", "DIFFUSE_SINGLE": "B_DIFFUSE_SINGLE", "DISCRETE_DISCRETE": "C_DISCRETE_DISCRETE", "DISCRETE_DIFFUSE": "D_DISCRETE_DIFFUSE"}
LOG77 = math.log(77)


def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def sha_tensor(t): return hashlib.sha256(t.float().cpu().numpy().tobytes()).hexdigest()
def finite_ok(m): return all(v for v in m["finite"].values() if v is not None)
def mdir(root, p, seed, ls): return os.path.join(root, CAT_DIR[p["category"]], p["id"], f"seed_{seed}", f"lambda_sharp_{ls:.2f}")
def sdir(sroot, p, seed, beta): return os.path.join(sroot, CAT_DIR[p["category"]], p["id"], f"seed_{seed}", f"beta_{beta:.2f}")


def stats_from_manifest(m, ls):
    """Run-level attention statistics (conditional half) from per-layer per-step records."""
    st = m["cast_processor_stats"]; layers = sorted(st); out = {"lambda_sharp": ls}
    sm = [st[n].get("sharp_mix") for n in layers]
    if all(x for x in sm):
        out["neg_fraction"] = statistics.mean(statistics.mean(x["neg_fraction_by_step"]) for x in sm)
        out["neg_mass"] = statistics.mean(statistics.mean(x["neg_mass_by_step"]) for x in sm)
        out["min_weight"] = min(min(x["min_weight_by_step"]) for x in sm)
        he = statistics.mean(statistics.mean(x["entropy_sharp_endpoint_by_step"]) for x in sm)
        out["endpoint_sharp_entropy"] = he; out["endpoint_sharp_normalized_entropy"] = he / LOG77; out["endpoint_sharp_effective_support"] = math.exp(he)
    ent = [st[n]["mean_entropy_cond"] for n in layers if st[n].get("mean_entropy_cond") is not None]
    if ent and ls <= 1.0:
        H = statistics.mean(ent); out.update({"mean_entropy": H, "normalized_entropy": H / LOG77, "effective_support": math.exp(H), "entropy_by_layer": ent})
    else:
        out.update({"mean_entropy": None, "normalized_entropy": None, "effective_support": None, "entropy_note": "N/A (signed weights)" if ls > 1.0 else "N/A"})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="experiments/configs/sharp_mix_v1.yaml"); args = ap.parse_args()
    y = yaml.safe_load(open(args.config)); lss = [float(x) for x in y.pop("lambda_sharps")]; seeds = [int(s) for s in y.pop("seeds")]
    pfile, psha, droot, sroot = y.pop("prompt_file"), y.pop("prompt_file_sha256"), y.pop("diagnostic_dir"), y.pop("sharp_dir"); beta = float(y.pop("beta"))
    assert sha_file(pfile) == psha and lss == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] and seeds == [42, 123, 2026] and beta == 1.5
    ds = json.load(open(pfile)); prompts = ds["prompts"]; assert len(prompts) == 24
    dman = json.load(open(os.path.join(droot, "experiment_manifest.json"))); sman = json.load(open(os.path.join(sroot, "experiment_manifest.json")))
    assert dman["prompt_file_sha256"] == psha and sman["prompt_file_sha256"] == psha
    base = CastConfig(**y, prompt="", mode="sharp_mix", beta=beta, lambda_sharp=0.0, seed=seeds[0]); assert base.deterministic_self_attn and base.resolved().attention_type == "sharp_mix"
    gs = {k: v for k, v in base.to_dict().items() if k not in ("prompt", "mode", "lam", "seed", "beta", "lambda_sharp", "attention_type", "alpha", "output_dir")}
    for name, man in [("diagnostic_v1", dman), ("sharp_softmax_v1", sman)]:
        mism = {k: (gs[k], man["generation_settings"].get(k)) for k in gs if man["generation_settings"].get(k) != gs[k]}
        assert not mism, f"settings differ from {name}: {mism}"
    root = base.output_dir; os.makedirs(root, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}; assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0s = {s: sample_initial_latents(CastConfig(**{**base.to_dict(), "seed": s})) for s in seeds}; z0_sha = {s: sha_tensor(z0s[s]) for s in seeds}
    assert all(z0_sha[s] == dman["initial_latent_sha256_by_seed"][str(s)] == sman["initial_latent_sha256_by_seed"][str(s)] for s in seeds)
    manifest = {"experiment": "cast_sharp_mix_v1", "builds_on": [droot, sroot], "prompt_file": pfile, "prompt_file_sha256": psha, "n_prompts": 24, "attention_type": "sharp_mix", "beta": beta,
                "lambda_sharps": lss, "seeds": seeds, "initial_latent_sha256_by_seed": z0_sha, "generation_settings": gs, "stock_processor_classes": sorted(set(stock.values())),
                "versions": _versions(), "expected_conditions": 24 * 3 * 7, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {},
                "terminology": "lambda_sharp (this experiment) != lambda_entmax (Diagnostic V1 `lam`); beta=1.5 is the fixed Sharp Softmax endpoint"}
    def dump(): json.dump(manifest, open(os.path.join(root, "experiment_manifest.json"), "w"), indent=2)
    dump(); print("[stage0] dataset/settings/latents match Diagnostic V1 and Sharp Softmax V1", flush=True)

    # ---- stage 1: regeneration checks for the two reused endpoints ----
    reg = {}
    for pid, seed, ls, refdir in [("D01", 42, 0.0, None), ("A01", 42, 1.0, None), ("D01", 2026, 1.0, None)]:
        p = next(x for x in prompts if x["id"] == pid)
        cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "lambda_sharp": ls, "seed": seed})
        r = run_one(pipe, cfg, initial_latents=z0s[seed], save=False)["manifest"]
        ref = json.load(open(os.path.join(sdir(sroot, p, seed, 1.0 if ls == 0.0 else 1.5), "manifest.json")))
        reg[f"{pid}/seed_{seed}/lambda_sharp_{ls:.2f}"] = {"reference": "sharp_softmax_v1 beta=1.00 (== diagnostic lambda_entmax=0)" if ls == 0.0 else "sharp_softmax_v1 beta=1.50",
                                                            "final_latent_sha_equal": r["final_latents_sha256"] == ref["final_latents_sha256"], "audio_sha_equal": r["audio_sha256"] == ref["audio_sha256"]}
        print("[stage1]", list(reg)[-1], reg[list(reg)[-1]], flush=True)
    manifest["checks"]["endpoint_regeneration"] = reg; dump()
    if not all(all(v[k] for k in ("final_latent_sha_equal", "audio_sha_equal")) for v in reg.values()): print("ENDPOINT PARITY FAILED - STOPPING"); sys.exit(1)

    # ---- stage 2: reuse the two endpoints (copy with provenance) ----
    runs, failures = {}, {}
    def record(p, seed, ls, m, d, src):
        st = stats_from_manifest(m, ls)
        runs[f"{p['id']}/seed_{seed}/lambda_sharp_{ls:.2f}"] = {"prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "beta": beta, "lambda_sharp": ls,
            "attention_type": "sharp_mix", "audio_path": os.path.join(d, "audio.wav"), "dir": d, "audio_sha256": m["audio_sha256"], "initial_latent_sha256": m["initial_latents"]["sha256"],
            "final_latent_norm": m["final_latent_norm"], "final_latent_sha256": m["final_latents_sha256"], "runtime_sec": m.get("runtime_sec"), "finite": m["finite"], "valid": finite_ok(m),
            "attn2_processor": m["attention_processor_class"]["attn2"], "attn1_processor": m["attention_processor_class"]["attn1"], "source": src, "stats": {k: v for k, v in st.items() if k != "entropy_by_layer"}}
    for p in prompts:
        for seed in seeds:
            for ls, bsrc in [(0.0, 1.0), (1.0, 1.5)]:
                src, dst = sdir(sroot, p, seed, bsrc), mdir(root, p, seed, ls); os.makedirs(dst, exist_ok=True)
                for f in ["audio.wav", "spectrogram.png", "final_latents.pt"]:
                    if not os.path.exists(os.path.join(dst, f)): shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                m = json.load(open(os.path.join(src, "manifest.json"))); m2 = dict(m); m2["attention_type"] = "sharp_mix"; m2["lambda_sharp"] = ls; m2["beta"] = beta
                m2["provenance"] = {"reused_from": src, "note": ("dense output (Diagnostic V1 lambda_entmax=0 == Sharp Softmax beta=1.00) reused as lambda_sharp=0.00" if ls == 0.0 else "Sharp Softmax beta=1.50 output reused as lambda_sharp=1.00") + "; regeneration check bitwise (experiment_manifest.json checks.endpoint_regeneration)"}
                # entropy stats for the reused endpoints come from the Sharp Softmax run (same bitwise trajectory)
                es = m.get("entropy_stats_cond", {}); m2["cast_processor_stats"] = m.get("cast_processor_stats", {})
                json.dump(m2, open(os.path.join(dst, "manifest.json"), "w"), indent=2)
                record(p, seed, ls, m2, dst, "reused_sharp_softmax_v1_beta_%.2f" % bsrc)
                if es: runs[f"{p['id']}/seed_{seed}/lambda_sharp_{ls:.2f}"]["stats"].update({"mean_entropy": es["mean_entropy"], "normalized_entropy": es["normalized_entropy"], "effective_support": es["effective_support"], "neg_fraction": 0.0, "neg_mass": 0.0, "min_weight": None, "entropy_note": None})
    print(f"[stage2] reused {sum(1 for r in runs.values() if r['source'].startswith('reused'))} endpoint conditions", flush=True)

    # ---- stage 3: generate the five new lambda_sharp values (resumable) ----
    t0 = time.time(); new_ls = [0.25, 0.5, 0.75, 1.25, 1.5]; n_new = 24 * 3 * len(new_ls)
    for p in prompts:
        for seed in seeds:
            for ls in new_ls:
                d = mdir(root, p, seed, ls); mp = os.path.join(d, "manifest.json")
                try:
                    if os.path.exists(mp) and os.path.exists(os.path.join(d, "audio.wav")):
                        m = json.load(open(mp)); src = "existing"
                    else:
                        cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], "lambda_sharp": ls, "seed": seed})
                        m = run_one(pipe, cfg, initial_latents=z0s[seed], out_dir=d)["manifest"]; src = "generated"
                        m["attention_stats_cond"] = stats_from_manifest(m, ls); json.dump(m, open(mp, "w"), indent=2)
                    record(p, seed, ls, m, d, src); r = runs[f"{p['id']}/seed_{seed}/lambda_sharp_{ls:.2f}"]
                    print(f"[{p['id']} seed={seed} ls={ls:.2f}] {src:9s} |zT|={m['final_latent_norm']:.2f} {m['runtime_sec']:.1f}s negfrac={r['stats'].get('neg_fraction', 0):.4f} negmass={r['stats'].get('neg_mass', 0):.4f} Hn={r['stats'].get('normalized_entropy') if r['stats'].get('normalized_entropy') is None else round(r['stats']['normalized_entropy'], 3)} finite={r['valid']}", flush=True)
                except Exception as e:
                    failures[f"{p['id']}/seed_{seed}/lambda_sharp_{ls:.2f}"] = {"error": repr(e), "trace": traceback.format_exc()}; print(f"[{p['id']} seed={seed} ls={ls:.2f}] FAILED {e!r}", flush=True)
            done = sum(1 for k, r in runs.items() if r["source"] in ("generated", "existing")); print(f"  progress {done}/{n_new} new  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump({"runs": runs, "failures": failures}, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    latent_ok = all(len({runs[f"{p['id']}/seed_{s}/lambda_sharp_{l:.2f}"]["initial_latent_sha256"] for l in lss if f"{p['id']}/seed_{s}/lambda_sharp_{l:.2f}" in runs}) == 1 for p in prompts for s in seeds)
    summary = {"experiment_manifest": "experiment_manifest.json", "n_expected": 24 * 3 * 7, "n_completed": len(runs), "n_valid": sum(r["valid"] for r in runs.values()),
               "n_generated_now": sum(r["source"] == "generated" for r in runs.values()), "n_reused": sum(r["source"].startswith("reused") for r in runs.values()), "failures": failures,
               "same_initial_latent_per_prompt_seed": latent_ok, "runs": runs}
    json.dump(summary, open(os.path.join(root, "generation_summary.json"), "w"), indent=2)
    print(f"\nGENERATION DONE: completed {len(runs)}/504 (new {summary['n_generated_now']}, reused {summary['n_reused']}) valid {summary['n_valid']} failures {len(failures)} latent_ok {latent_ok} elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
