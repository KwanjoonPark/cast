"""Single entry point for all runs (golden reference and the three CAST arms).

The ONLY difference between arms is which processor sits on the 16 attn2 modules (none for `golden`,
CastCrossAttnProcessor with the mode's lambda otherwise). AuffusionPipeline.__call__ is used verbatim.
"""
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Optional

import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

from cast.config import CastConfig
from cast.integration.auffusion import apply_cast, restore_processors, cast_processors, select_cross_attention

SAMPLE_RATE = 16000
LATENT_CHANNELS = 4
VAE_SCALE = 8


def torch_dtype(name: str):
    return {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[name]


def sample_initial_latents(cfg: CastConfig, batch_size: int = 1) -> torch.Tensor:
    """Reproduces AuffusionPipeline.prepare_latents' random draw exactly (same shape/dtype/device/generator
    path through diffusers.utils.randn_tensor). The pipeline multiplies by scheduler.init_noise_sigma
    afterwards (1.0 for PNDM), so passing this tensor as `latents=` is equivalent to seeding internally."""
    from diffusers.utils import randn_tensor
    shape = (batch_size, LATENT_CHANNELS, cfg.height // VAE_SCALE, cfg.width // VAE_SCALE)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    return randn_tensor(shape, generator=gen, device=torch.device(cfg.device), dtype=torch_dtype(cfg.dtype))


def _versions():
    import diffusers, transformers, huggingface_hub, entmax, scipy, numpy
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "entmax": entmax.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def save_spectrogram_png(spec: torch.Tensor, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = spec.detach().float().cpu().numpy()          # (256 mel, 1024 frames), log-magnitude
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(s, origin="lower", aspect="auto", cmap="viridis", extent=[0, s.shape[1] / 100.0, 0, s.shape[0]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log-mel")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run_one(pipe, cfg: CastConfig, initial_latents: Optional[torch.Tensor] = None, out_dir: Optional[str] = None,
            use_generator: bool = True, save: bool = True, probe_betas=(), probe_entmax: bool = False, only_modules=None) -> dict:
    """Run one generation. Returns a result dict (audio, spectrogram, final latents, manifest).
    `initial_latents` (if given) is passed verbatim as `latents=`; a seeded generator is still passed
    (harmless: PNDM.step ignores it, prepare_latents skips the draw when latents are provided)."""
    cfg = cfg.resolved()
    unet = pipe.unet
    originals = {}
    if cfg.use_cast:
        originals = apply_cast(unet, cfg.alpha, cfg.lam, cfg.apply_to_uncond, cfg.collect_stats,
                               attention_type=cfg.attention_type, beta=cfg.beta, probe_betas=probe_betas, probe_entmax=probe_entmax, lambda_sharp=cfg.lambda_sharp, only=only_modules)
    procs = unet.attn_processors
    attn2_classes = sorted({type(p).__name__ for n, p in procs.items() if n.endswith("attn2.processor")})
    attn1_classes = sorted({type(p).__name__ for n, p in procs.items() if n.endswith("attn1.processor")})

    captured = {}
    n_steps = cfg.num_inference_steps

    def _cb(i, t, latents):
        if i == n_steps - 1:
            captured["final_latents"] = latents.detach().clone()

    try:
        gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed) if use_generator else None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ac = torch.autocast("cuda") if cfg.autocast else torch.autocast("cuda", enabled=False)
        # The default fp16 flash/mem-efficient SDPA kernel used by the untouched self-attention (attn1) is not
        # bitwise repeatable at N=4096; the math backend is. This flag only changes the self-attention kernel.
        sdp = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
               if cfg.deterministic_self_attn else contextlib.nullcontext())
        with ac, sdp:
            out = pipe(prompt=cfg.prompt, num_inference_steps=n_steps, guidance_scale=cfg.guidance_scale,
                       generator=gen, height=cfg.height, width=cfg.width,
                       latents=initial_latents, callback=_cb, callback_steps=1)
        torch.cuda.synchronize(); runtime = time.perf_counter() - t0
        peak_mem_gb = torch.cuda.max_memory_allocated() / 2**30
        cast_stats = {n: {"calls": p.calls, "mean_sparse_support": p.mean_sparse_support,
                          "mean_sparse_support_cond_half": p.mean_sparse_support_cond,
                          "mean_entropy_cond": p.mean_entropy, "mean_entropy_sparse_cond": p.mean_entropy_sparse,
                          "entropy_by_step": p.entropy_calls if cfg.collect_stats else None,
                          "probe_entropy_by_step": {str(b): v for b, v in p.probe_entropy_calls.items()} if p.probe_betas else None,
                          "probe_entmax_entropy_by_step": p.probe_entmax_entropy_calls if p.probe_entmax else None,
                          "probe_entmax_support_by_step": p.probe_entmax_support_calls if p.probe_entmax else None,
                          "sharp_mix": ({"entropy_sharp_endpoint_by_step": p.entropy_sharp_endpoint_calls, "neg_fraction_by_step": p.neg_fraction_calls,
                                         "neg_mass_by_step": p.neg_mass_calls, "min_weight_by_step": p.min_weight_calls} if p.attention_type == "sharp_mix" else None)}
                      for n, p in cast_processors(unet).items()}
    finally:
        if originals:
            restore_processors(unet, originals)

    audio = out.audios[0][: SAMPLE_RATE * 10]           # int16 numpy, exactly as inference.py
    spec = out.spectrograms[0]                           # (256, 1024) tensor
    final_latents = captured.get("final_latents")
    finite = {
        "audio": bool(np.isfinite(audio).all()),
        "spectrogram": bool(torch.isfinite(spec).all().item()),
        "final_latents": bool(torch.isfinite(final_latents).all().item()) if final_latents is not None else None,
    }
    manifest = {
        "config": cfg.to_dict(),
        "alpha": cfg.alpha, "lam": cfg.lam, "attention_type": cfg.attention_type, "beta": cfg.beta, "lambda_sharp": cfg.lambda_sharp,
        "initial_latents": None if initial_latents is None else {
            "sha256": _tensor_digest(initial_latents), "shape": list(initial_latents.shape),
            "dtype": str(initial_latents.dtype), "mean": float(initial_latents.float().mean()), "std": float(initial_latents.float().std())},
        "attention_processor_class": {"attn2": attn2_classes, "attn1": attn1_classes},
        "scheduler": type(pipe.scheduler).__name__,
        "versions": _versions(),
        "git_commit": _git_commit(),
        "runtime_sec": runtime,
        "peak_cuda_mem_gb": peak_mem_gb,
        "finite": finite,
        "final_latent_norm": float(final_latents.float().norm()) if final_latents is not None else None,
        "final_latents_sha256": _tensor_digest(final_latents) if final_latents is not None else None,
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "cast_processor_stats": cast_stats,
        "sample_rate": SAMPLE_RATE,
    }
    if save:
        out_dir = out_dir or os.path.join(cfg.output_dir, cfg.mode)
        os.makedirs(out_dir, exist_ok=True)
        wav_write(os.path.join(out_dir, "audio.wav"), SAMPLE_RATE, audio)
        save_spectrogram_png(spec, os.path.join(out_dir, "spectrogram.png"),
                             title=f"{cfg.mode}  alpha={cfg.alpha} lambda={cfg.lam}  '{cfg.prompt}' seed={cfg.seed}")
        torch.save({"final_latents": final_latents.cpu() if final_latents is not None else None,
                    "initial_latents": initial_latents.cpu() if initial_latents is not None else None,
                    "spectrogram": spec.cpu()}, os.path.join(out_dir, "final_latents.pt"))
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        manifest["out_dir"] = out_dir
    return {"audio": audio, "spectrogram": spec, "final_latents": final_latents, "manifest": manifest}
