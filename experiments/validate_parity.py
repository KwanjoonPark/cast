"""Checkpoint-level parity: stock Auffusion U-Net vs CAST lambda=0 (and lambda=1/2 for scale) on real
prompt embeddings and real noisy latents at several timesteps, in fp32 and in the fp16+autocast
inference setting. Writes outputs/validation/unet_parity.json."""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline, cast_installed, cast_processors
from cast.pipeline import sample_initial_latents, torch_dtype


def stats(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "rel_l2": float(d.norm() / b.float().norm()), "ref_std": float(b.float().std())}


def main():
    cfg = CastConfig.from_yaml("experiments/configs/first_milestone.yaml")
    pipe = load_pipeline(cfg.checkpoint, torch_dtype(cfg.dtype), cfg.device)
    unet, dev = pipe.unet, cfg.device
    with torch.no_grad():
        emb = pipe._encode_prompt(cfg.prompt, dev, 1, True, None)            # [2, 77, 768]: [uncond(zeros), cond]
    assert emb.shape == (2, 77, 768) and float(emb[0].abs().max()) == 0.0
    z0 = sample_initial_latents(cfg)
    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=dev)
    ts = pipe.scheduler.timesteps
    picks = [ts[0], ts[len(ts) // 2], ts[-1]]
    noise = torch.randn_like(z0)
    out = {"timesteps": [int(t) for t in picks], "prompt": cfg.prompt}
    for setting in ["fp32", "fp16_autocast"]:
        if setting == "fp32":
            unet.float(); e = emb.float(); ac = torch.autocast("cuda", enabled=False)
        else:
            unet.half(); e = emb.half(); ac = torch.autocast("cuda")
        res = {}
        for t in picks:
            zt = pipe.scheduler.add_noise(z0.float(), noise.float(), t.reshape(1)).to(e.dtype)
            x = torch.cat([zt] * 2)
            with torch.no_grad(), ac:
                ref = unet(x, t, encoder_hidden_states=e).sample
                ref2 = unet(x, t, encoder_hidden_states=e).sample            # stock repeat -> kernel non-determinism floor
                outs = {}
                for lam in [0.0, 1.0, 2.0]:
                    with cast_installed(unet, 1.5, lam, collect_stats=True):
                        outs[lam] = unet(x, t, encoder_hidden_states=e).sample
                        supp = [p.mean_sparse_support for p in cast_processors(unet).values()]
            r = {"stock_repeat_floor": stats(ref2, ref),
                 "stock_vs_cast_lam0": stats(outs[0.0], ref),
                 "stock_vs_cast_lam0_uncond_row": stats(outs[0.0][0], ref[0]),
                 "cast_lam1_vs_lam0": stats(outs[1.0], outs[0.0]),
                 "cast_lam2_vs_lam0": stats(outs[2.0], outs[0.0]),
                 "cast_lam2_uncond_row_vs_lam0": stats(outs[2.0][0], outs[0.0][0]),
                 "mean_sparse_support_per_layer_lam2": supp,
                 "finite": {str(l): bool(torch.isfinite(o).all()) for l, o in outs.items()}}
            res[int(t)] = r
            print(f"[{setting} t={int(t)}] floor max={r['stock_repeat_floor']['max_abs']:.1e} stock-vs-cast0 max={r['stock_vs_cast_lam0']['max_abs']:.3e} rel_l2={r['stock_vs_cast_lam0']['rel_l2']:.3e} | "
                  f"lam1-vs-0 rel_l2={r['cast_lam1_vs_lam0']['rel_l2']:.3e} | lam2-vs-0 rel_l2={r['cast_lam2_vs_lam0']['rel_l2']:.3e} | uncond lam2-vs-0 max={r['cast_lam2_uncond_row_vs_lam0']['max_abs']:.1e}")
        out[setting] = res
    unet.half()
    os.makedirs("outputs/validation", exist_ok=True)
    json.dump(out, open("outputs/validation/unet_parity.json", "w"), indent=2)
    print("wrote outputs/validation/unet_parity.json")


if __name__ == "__main__":
    main()
