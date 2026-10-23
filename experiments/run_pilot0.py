"""CAST Pilot 0: prompts x {lambda 0,1,2} with one shared initial latent, deterministic self-attention.
The first prompt is a dry check: the required validations run on it and the script stops before the
remaining prompts if any fails. Writes outputs/pilot0/pilot0_summary.json and LISTENING_INDEX.md."""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, yaml

from cast.config import CastConfig, MODES
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype
from cast.attention.cast_processor import CastCrossAttnProcessor

EXPECTED = {"cast_vanilla": (1.5, 0.0), "cast_sparse": (1.5, 1.0), "cast_pladis": (1.5, 2.0)}


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def support_stats(manifest):
    st = manifest["cast_processor_stats"]
    cond = [v["mean_sparse_support_cond_half"] for v in st.values() if v.get("mean_sparse_support_cond_half") is not None]
    allr = [v["mean_sparse_support"] for v in st.values() if v.get("mean_sparse_support") is not None]
    f = lambda xs: None if not xs else {"mean_over_layers": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}
    return {"cond_half": f(cond), "all_rows": f(allr)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="experiments/configs/pilot0.yaml"); args = ap.parse_args()
    y = yaml.safe_load(open(args.config)); prompts = y.pop("prompts"); modes = y.pop("modes")
    base = CastConfig(**y, prompt="", mode="cast_vanilla")
    assert base.deterministic_self_attn is True
    root = base.output_dir; os.makedirs(root, exist_ok=True)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
    assert len(stock) == 32 and len(select_cross_attention(pipe.unet)) == 16
    z0 = sample_initial_latents(base)                       # seed 42, shared by every prompt and every mode
    summary = {"config": {**y, "modes": modes}, "stock_processor_classes": sorted(set(stock.values())),
               "initial_latent_sha256": None, "runs": [], "per_prompt": {}, "dry_check_P01": None, "failures": []}
    for pi, (pid, prompt) in enumerate(prompts.items()):
        res = {}
        for mode in modes:
            cfg = CastConfig(**{**base.to_dict(), "prompt": prompt, "mode": mode})
            out_dir = os.path.join(root, pid, mode)
            try:
                r = run_one(pipe, cfg, initial_latents=z0, out_dir=out_dir)
            except Exception as e:
                summary["failures"].append({"prompt_id": pid, "mode": mode, "error": repr(e), "trace": traceback.format_exc()})
                print(f"[{pid}/{mode}] FAILED: {e!r}"); continue
            m = r["manifest"]; res[mode] = r
            rec = {"prompt_id": pid, "prompt": prompt, "mode": mode, "alpha": m["alpha"], "lambda": m["lam"], "seed": cfg.seed,
                   "deterministic_self_attn": cfg.deterministic_self_attn, "initial_latent_sha256": m["initial_latents"]["sha256"],
                   "final_latent_norm": m["final_latent_norm"], "final_latent_sha256": m["final_latents_sha256"],
                   "runtime_sec": m["runtime_sec"], "peak_gpu_memory_gb": m["peak_cuda_mem_gb"], "finite": m["finite"],
                   "attn2_processor": m["attention_processor_class"]["attn2"], "attn1_processor": m["attention_processor_class"]["attn1"],
                   "entmax_support": support_stats(m), "out_dir": out_dir}
            summary["runs"].append(rec)
            print(f"[{pid}/{mode}] lam={m['lam']} |zT|={m['final_latent_norm']:.2f} {m['runtime_sec']:.1f}s {m['peak_cuda_mem_gb']:.2f}GB finite={all(v for v in m['finite'].values() if v is not None)}")
        if "cast_vanilla" in res:
            z = {m: res[m]["final_latents"] for m in res}
            summary["per_prompt"][pid] = {"prompt": prompt,
                "delta_sparse": rel_l2(z["cast_sparse"], z["cast_vanilla"]) if "cast_sparse" in z else None,
                "delta_pladis": rel_l2(z["cast_pladis"], z["cast_vanilla"]) if "cast_pladis" in z else None,
                "delta_pladis_vs_sparse": rel_l2(z["cast_pladis"], z["cast_sparse"]) if "cast_pladis" in z and "cast_sparse" in z else None,
                "final_latent_norms": {m: float(z[m].float().norm()) for m in z},
                "initial_latent_sha256_identical": len({res[m]["manifest"]["initial_latents"]["sha256"] for m in res}) == 1}
            print(f"[{pid}] delta_sparse={summary['per_prompt'][pid]['delta_sparse']:.4f} delta_pladis={summary['per_prompt'][pid]['delta_pladis']:.4f}")
        # ---- dry check on the first prompt: stop before the rest if anything fails ------------------
        if pi == 0:
            runs = [r for r in summary["runs"] if r["prompt_id"] == pid]
            after = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
            checks = {
                "deterministic_self_attn_true": all(r["deterministic_self_attn"] for r in runs),
                "three_runs": len(runs) == 3 and not summary["failures"],
                "attn2_all_cast_16": all(r["attn2_processor"] == ["CastCrossAttnProcessor"] for r in runs),
                "attn1_untouched_stock": all(r["attn1_processor"] == sorted({stock[n] for n in stock if n.endswith("attn1.processor")}) for r in runs),
                "processors_restored_after_runs": after == stock,
                "same_initial_latent": len({r["initial_latent_sha256"] for r in runs}) == 1,
                "alpha_1p5_all": all(r["alpha"] == 1.5 for r in runs),
                "lambda_exact_0_1_2": [r["lambda"] for r in runs] == [EXPECTED[m][1] for m in modes],
                "no_nan_inf": all(all(v for v in r["finite"].values() if v is not None) for r in runs),
            }
            summary["dry_check_P01"] = checks
            print("[dry-check P01]", checks)
            if not all(checks.values()):
                summary["aborted"] = "dry check failed"
                json.dump(summary, open(os.path.join(root, "pilot0_summary.json"), "w"), indent=2)
                print("DRY CHECK FAILED - STOPPING"); sys.exit(1)
    summary["initial_latent_sha256"] = summary["runs"][0]["initial_latent_sha256"] if summary["runs"] else None
    summary["all_initial_latents_identical"] = len({r["initial_latent_sha256"] for r in summary["runs"]}) == 1
    summary["n_runs"] = len(summary["runs"]); summary["n_invalid"] = len(summary["failures"]) + sum(
        not all(v for v in r["finite"].values() if v is not None) for r in summary["runs"])
    json.dump(summary, open(os.path.join(root, "pilot0_summary.json"), "w"), indent=2)

    # ---- listening index -------------------------------------------------------------------------
    lines = ["# CAST Pilot 0 Listening Index", "",
             "All files relative to `outputs/pilot0/`. Same seed (42) and bitwise-identical initial latent for every prompt and mode; "
             "deterministic self-attention; alpha = 1.5; lambda = 0 (Vanilla) / 1 (Sparse) / 2 (PLADIS).", ""]
    for pid, prompt in prompts.items():
        lines += [f"## {pid.split('_')[0]} — {prompt}", "", f"- Vanilla (λ=0): `{pid}/cast_vanilla/audio.wav`",
                  f"- Sparse (λ=1): `{pid}/cast_sparse/audio.wav`", f"- PLADIS (λ=2): `{pid}/cast_pladis/audio.wav`",
                  f"- Spectrograms: `{pid}/cast_vanilla/spectrogram.png`, `{pid}/cast_sparse/spectrogram.png`, `{pid}/cast_pladis/spectrogram.png`", "",
                  "Manual notes:", "", "    Event presence:", "    Foreground clarity:", "    Background preservation:",
                  "    Extraneous sounds:", "    Overall quality:", "    Preferred method:", ""]
    open(os.path.join(root, "LISTENING_INDEX.md"), "w").write("\n".join(lines))
    n_ok = sum(all(v for v in r["finite"].values() if v is not None) for r in summary["runs"])
    print(f"\nPilot 0: {'PASS' if n_ok == len(prompts) * len(modes) and not summary['failures'] else 'FAIL'}")
    print(f"Generated audios: {n_ok} / {len(prompts) * len(modes)}   invalid runs: {summary['n_invalid']}")
    print(f"Output root: {root}\nSummary: {root}/pilot0_summary.json\nListening index: {root}/LISTENING_INDEX.md")


if __name__ == "__main__":
    main()
