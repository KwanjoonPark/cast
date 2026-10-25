"""Glue between CAST and the unmodified third_party/Auffusion pipeline (diffusers 0.18.2)."""
import contextlib
import os
import sys
from typing import Dict

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUFFUSION_DIR = os.path.join(PROJECT_ROOT, "third_party", "Auffusion")


def import_auffusion():
    """Import `auffusion_pipeline` from third_party without modifying it."""
    if AUFFUSION_DIR not in sys.path:
        sys.path.insert(0, AUFFUSION_DIR)
    import auffusion_pipeline  # noqa: E402
    return auffusion_pipeline


def resolve_checkpoint(name_or_path: str) -> str:
    """Return a local directory for the checkpoint. Downloads anonymously (token=False) so that a stale
    cached HF token is never sent; nothing in the user's HF auth files is touched."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, token=False)


def load_pipeline(checkpoint: str, dtype=torch.float16, device: str = "cuda"):
    ap = import_auffusion()
    local = resolve_checkpoint(checkpoint)
    pipe = ap.AuffusionPipeline.from_pretrained(local, dtype=dtype, device=device)   # stock processors
    pipe = pipe.to(device, dtype)                                                     # as in inference.py
    pipe.set_progress_bar_config(disable=True)
    return pipe


def select_cross_attention(unet) -> Dict[str, torch.nn.Module]:
    """Exactly the 16 text cross-attention `attn2` Attention modules, keyed by processor name.
    Selection requires agreement of (1) name ends with 'attn2.processor' and
    (2) to_k.in_features == unet.config.cross_attention_dim != to_q.in_features."""
    procs = unet.attn_processors
    if len(procs) != 32:
        raise RuntimeError(f"expected 32 attention processors in the SD1.5 U-Net, found {len(procs)}")
    xdim = unet.config.cross_attention_dim
    selected = {}
    for name in procs:
        mod = unet.get_submodule(name[: -len(".processor")])
        by_name = name.endswith("attn2.processor")
        by_shape = (mod.to_k.in_features == xdim) and (mod.to_k.in_features != mod.to_q.in_features)
        if by_name != by_shape:
            raise RuntimeError(f"self/cross disagreement for {name}: by_name={by_name} by_shape={by_shape}")
        if by_name:
            selected[name] = mod
    if len(selected) != 16:
        raise RuntimeError(f"expected 16 cross-attention modules, selected {len(selected)}")
    return selected


def apply_cast(unet, alpha: float, lam: float, apply_to_uncond: bool = True, collect_stats: bool = False,
               attention_type: str = "entmax_mix", beta: float = 1.0, probe_betas=(), probe_entmax: bool = False, lambda_sharp: float = 0.0, only=None):
    """Install CastCrossAttnProcessor on the 16 attn2 modules (or, if `only` is given, just on the named subset). Returns {name: original_processor}."""
    from cast.attention.cast_processor import CastCrossAttnProcessor
    originals = {}
    selected = select_cross_attention(unet)
    if only is not None:
        only = set(only)
        unknown = only - set(selected)
        if unknown:
            raise ValueError(f"`only` names are not attn2 processors: {sorted(unknown)}")
    for name, mod in selected.items():
        if only is not None and name not in only:
            continue
        originals[name] = mod.processor
        mod.set_processor(CastCrossAttnProcessor(alpha=alpha, lam=lam, apply_to_uncond=apply_to_uncond,
                                                 collect_stats=collect_stats, name=name, attention_type=attention_type,
                                                 beta=beta, probe_betas=probe_betas, probe_entmax=probe_entmax, lambda_sharp=lambda_sharp))
    return originals


def restore_processors(unet, originals: Dict[str, object]):
    for name, proc in originals.items():
        unet.get_submodule(name[: -len(".processor")]).set_processor(proc)


@contextlib.contextmanager
def cast_installed(unet, alpha: float, lam: float, apply_to_uncond: bool = True, collect_stats: bool = False, **kw):
    originals = apply_cast(unet, alpha, lam, apply_to_uncond, collect_stats, **kw)
    try:
        yield originals
    finally:
        restore_processors(unet, originals)


def cast_processors(unet):
    from cast.attention.cast_processor import CastCrossAttnProcessor
    return {n: p for n, p in unet.attn_processors.items() if isinstance(p, CastCrossAttnProcessor)}
