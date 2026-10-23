"""Golden reference + latent-control checks + the three CAST arms, then a compact summary.
Usage: python experiments/run_first_milestone.py [--config experiments/configs/first_milestone.yaml] [--steps N]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch

from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, select_cross_attention
from cast.pipeline import run_one, sample_initial_latents, torch_dtype


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float().cpu(), b.float().cpu()
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "rel_l2": float(d.norm() / (b.norm() + 1e-12)),
            "identical": bool(torch.equal(a, b))}


def audio_diff(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(a - b)
    return {"max_abs_int16": float(d.max()), "mean_abs_int16": float(d.mean()), "identical": bool(np.array_equal(a, b)),
            "snr_db": float(10 * np.log10((b ** 2).sum() / max(((a - b) ** 2).sum(), 1e-12)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/first_milestone.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--skip-golden-checks", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="force math SDPA for self-attention (bitwise-repeatable)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--golden-dir", default="outputs/golden")
    args = ap.parse_args()
    over = {} if args.steps is None else {"num_inference_steps": args.steps}
    if args.deterministic: over["deterministic_self_attn"] = True
    if args.output_dir: over["output_dir"] = args.output_dir
    base = CastConfig.from_yaml(args.config, **over)
    summary = {"checks": {}, "runs": {}, "diffs": {}}

    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    sel = select_cross_attention(pipe.unet)
    stock = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
    summary["checks"]["processors_total"] = len(stock)
    summary["checks"]["attn2_selected"] = len(sel)
    summary["checks"]["stock_processor_classes"] = sorted(set(stock.values()))
    print(f"[census] processors={len(stock)} attn2 selected={len(sel)} stock classes={sorted(set(stock.values()))}")

    # ---- 1. golden: original Auffusion, stock processors, internal seeding (exactly inference.py) ----
    gcfg = CastConfig(**{**base.to_dict(), "mode": "golden", "output_dir": "outputs"})
    golden = run_one(pipe, gcfg, initial_latents=None, out_dir=args.golden_dir)
    summary["runs"]["golden"] = golden["manifest"]
    print(f"[golden] runtime {golden['manifest']['runtime_sec']:.1f}s  finite={golden['manifest']['finite']}  -> {args.golden_dir}")

    # ---- 2. latent control: pre-sampled latents reproduce the internal draw; determinism -------------
    z0 = sample_initial_latents(base)
    if not args.skip_golden_checks:
        g_lat = run_one(pipe, gcfg, initial_latents=z0, save=False)
        summary["diffs"]["golden_vs_golden_with_presampled_latents"] = {
            "final_latents": diff_stats(g_lat["final_latents"], golden["final_latents"]),
            "audio": audio_diff(g_lat["audio"], golden["audio"])}
        g_rep = run_one(pipe, gcfg, initial_latents=None, save=False)
        summary["diffs"]["golden_repeat_determinism"] = {
            "final_latents": diff_stats(g_rep["final_latents"], golden["final_latents"]),
            "audio": audio_diff(g_rep["audio"], golden["audio"])}
        print("[latent-control] presampled vs internal:", summary["diffs"]["golden_vs_golden_with_presampled_latents"]["final_latents"])
        print("[determinism]    repeat vs golden:      ", summary["diffs"]["golden_repeat_determinism"]["final_latents"])

    # ---- 3. three CAST arms, identical z0 ------------------------------------------------------------
    results = {}
    for mode in ["cast_vanilla", "cast_sparse", "cast_pladis"]:
        cfg = CastConfig(**{**base.to_dict(), "mode": mode})
        r = run_one(pipe, cfg, initial_latents=z0)
        results[mode] = r
        m = r["manifest"]; summary["runs"][mode] = m
        print(f"[{mode}] alpha={m['alpha']} lam={m['lam']} runtime {m['runtime_sec']:.1f}s peak {m['peak_cuda_mem_gb']:.2f}GB "
              f"finite={m['finite']} |z_T|={m['final_latent_norm']:.2f} attn2={m['attention_processor_class']['attn2']} attn1={m['attention_processor_class']['attn1']}")
    # processors restored?
    after = {n: type(p).__name__ for n, p in pipe.unet.attn_processors.items()}
    summary["checks"]["processors_restored"] = after == stock
    summary["checks"]["same_initial_latents"] = len({results[m]["manifest"]["initial_latents"]["sha256"] for m in results}) == 1

    v = results["cast_vanilla"]
    summary["diffs"]["golden_vs_cast_vanilla"] = {"final_latents": diff_stats(v["final_latents"], golden["final_latents"]),
                                                  "spectrogram": diff_stats(v["spectrogram"], golden["spectrogram"]),
                                                  "audio": audio_diff(v["audio"], golden["audio"])}
    for mode in ["cast_sparse", "cast_pladis"]:
        r = results[mode]
        summary["diffs"][f"{mode}_vs_cast_vanilla"] = {"final_latents": diff_stats(r["final_latents"], v["final_latents"]),
                                                       "spectrogram": diff_stats(r["spectrogram"], v["spectrogram"]),
                                                       "audio": audio_diff(r["audio"], v["audio"])}
    os.makedirs(base.output_dir, exist_ok=True)
    with open(os.path.join(base.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- summary -------------------------------------------------------------------------------------
    allfinite = all(all(x for x in summary["runs"][m]["finite"].values() if x is not None) for m in summary["runs"])
    print("\nCAST First Milestone\n" + "-" * 50)
    print(f"Golden Auffusion: {'PASS' if golden['manifest']['finite']['audio'] else 'FAIL'}")
    print(f"Cross-attn selection: {len(sel)}/16 {'PASS' if len(sel) == 16 else 'FAIL'}")
    print(f"Self-attn untouched: {'PASS' if all(results[m]['manifest']['attention_processor_class']['attn1'] == stock_cls for m in results for stock_cls in [sorted({stock[n] for n in stock if n.endswith('attn1.processor')})]) else 'FAIL'}")
    print(f"Processors restored: {'PASS' if summary['checks']['processors_restored'] else 'FAIL'}")
    print(f"Same initial noise (lambda 0/1/2): {'PASS' if summary['checks']['same_initial_latents'] else 'FAIL'}")
    d = summary["diffs"]["golden_vs_cast_vanilla"]
    print(f"lambda=0 dense parity vs golden: max|dz|={d['final_latents']['max_abs']:.3e} rel_l2={d['final_latents']['rel_l2']:.3e} audio SNR={d['audio']['snr_db']:.1f} dB")
    for mode in ["cast_sparse", "cast_pladis"]:
        d = summary["diffs"][f"{mode}_vs_cast_vanilla"]
        print(f"{mode} vs lambda=0: max|dz|={d['final_latents']['max_abs']:.3e} rel_l2={d['final_latents']['rel_l2']:.3e} audio SNR={d['audio']['snr_db']:.1f} dB  changed={'YES' if not d['final_latents']['identical'] else 'NO'}")
    print(f"NaN/Inf: {'NONE' if allfinite else 'FOUND'}")
    print(f"Outputs:\n  {args.golden_dir}/audio.wav")
    for m in results: print(f"  {results[m]['manifest']['out_dir']}/audio.wav")


if __name__ == "__main__":
    main()
