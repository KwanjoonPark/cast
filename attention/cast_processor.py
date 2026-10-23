"""CAST cross-attention processor for diffusers 0.18.2 `Attention` modules (text cross-attention only).

The prologue (projections, head split) and epilogue (merge heads, to_out, residual) are copied from
diffusers 0.18.2 `AttnProcessor2_0`. The single call to `F.scaled_dot_product_attention` is replaced by an
explicit, fp32 computation of the attention weights:

  attention_type == "entmax_mix" (validated CAST / PLADIS path):
    logits   = (Q K^T) * attn.scale                      # attn.scale = head_dim ** -0.5
    W_dense  = Softmax(logits)
    W_sparse = alpha-Entmax(logits)                      # same logits, same Q/K
    W        = W_dense + lam * (W_sparse - W_dense)      # lam=0 dense, lam=1 sparse, lam=2 PLADIS
  attention_type == "softmax" (Sharp Softmax diagnostic):
    W        = Softmax(beta * logits)                    # beta=1 is bitwise the lam=0 path above; T = 1/beta
  attention_type == "sharp_mix" (partial sharp guidance):
    W_dense  = Softmax(logits);  W_sharp = Softmax(beta * logits)
    W        = W_dense + lambda_sharp * (W_sharp - W_dense)   # lambda_sharp=0 -> W_dense (bitwise), 1 -> W_sharp (bitwise), >1 signed extrapolation; no clamping

  out = W @ V

Neither `F.scaled_dot_product_attention` nor xformers is used here. Self-attention (`attn1`) modules are
never given this processor (see cast.integration.auffusion.select_cross_attention).

Statistics (collect_stats=True), conditional CFG half only (rows [B//2:]), one scalar per call (mean over
heads and query rows): entmax exact support (entmax path, lam != 0), Shannon entropy H = -sum p log p of the
attention map actually used when it is a probability distribution (softmax path: W_beta; entmax path:
W_dense and W_sparse separately; never the signed W for lam > 1). Optional probes evaluate, on the same
logits, the entropy of Softmax(b * logits) for b in probe_betas and the entropy/support of entmax15(logits),
without affecting the output.
"""
import math
from typing import Optional, Sequence

import torch

from .sparse_ops import dense_softmax, alpha_entmax, mix_attention_weights


def _row_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy along dim=-1 for a (non-negative, row-normalized) map; zeros contribute 0."""
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


class CastCrossAttnProcessor:
    def __init__(
        self,
        alpha: float = 1.5,
        lam: float = 0.0,
        apply_to_uncond: bool = True,
        check_finite: bool = True,
        collect_stats: bool = False,
        name: str = "",
        attention_type: str = "entmax_mix",
        beta: float = 1.0,
        lambda_sharp: float = 0.0,
        probe_betas: Sequence[float] = (),
        probe_entmax: bool = False,
    ):
        if attention_type not in ("entmax_mix", "softmax", "sharp_mix"):
            raise ValueError(f"unknown attention_type {attention_type!r}")
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.apply_to_uncond = apply_to_uncond
        self.check_finite = check_finite
        self.collect_stats = collect_stats
        self.name = name
        self.attention_type = attention_type
        self.beta = float(beta)
        self.lambda_sharp = float(lambda_sharp)
        self.probe_betas = tuple(float(b) for b in probe_betas)
        self.probe_entmax = probe_entmax
        self.reset_counters()

    def reset_counters(self):
        self.calls = 0
        self.stat_support_sum = 0.0        # running sum of mean nnz per row of W_sparse, all batch rows (lam != 0 only)
        self.stat_support_sum_cond = 0.0   # same, rows [B//2:] (= conditional half under Auffusion's [uncond, cond] CFG layout)
        self.stat_support_n = 0
        self.entropy_calls = []            # per call: mean entropy (cond half) of the map used (W_beta or W_dense)
        self.entropy_sparse_calls = []     # per call: mean entropy (cond half) of W_sparse (entmax path, lam != 0)
        self.probe_entropy_calls = {b: [] for b in self.probe_betas}   # per call: entropy of Softmax(b*logits), cond half
        self.probe_entmax_entropy_calls = []
        self.probe_entmax_support_calls = []
        self.entropy_sharp_endpoint_calls = []   # sharp_mix: entropy of the endpoint W_sharp(beta) (cond half), reference only
        self.neg_fraction_calls = []             # sharp_mix: fraction of negative entries of W_partial (cond half)
        self.neg_mass_calls = []                 # sharp_mix: mean per-row sum(|w| for w<0) (cond half)
        self.min_weight_calls = []               # sharp_mix: min entry of W_partial (cond half)

    @property
    def mean_sparse_support(self) -> Optional[float]:
        return self.stat_support_sum / self.stat_support_n if self.stat_support_n else None

    @property
    def mean_sparse_support_cond(self) -> Optional[float]:
        return self.stat_support_sum_cond / self.stat_support_n if self.stat_support_n else None

    @property
    def mean_entropy(self) -> Optional[float]:
        return sum(self.entropy_calls) / len(self.entropy_calls) if self.entropy_calls else None

    @property
    def mean_entropy_sparse(self) -> Optional[float]:
        return sum(self.entropy_sparse_calls) / len(self.entropy_sparse_calls) if self.entropy_sparse_calls else None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # ---- runtime guards: this processor is only valid on text cross-attention without masks --------
        if encoder_hidden_states is None:
            raise RuntimeError(f"CastCrossAttnProcessor[{self.name}] called as self-attention (encoder_hidden_states is None)")
        if attention_mask is not None:
            raise RuntimeError(f"CastCrossAttnProcessor[{self.name}] does not support attention masks (got one)")
        if attn.spatial_norm is not None or attn.group_norm is not None or attn.norm_cross is not None:
            raise RuntimeError("CastCrossAttnProcessor assumes no spatial_norm/group_norm/norm_cross (SD1.5 attn2)")

        # ---- prologue (== diffusers 0.18.2 AttnProcessor2_0) -------------------------------------------
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)                     # (B, N, C_inner), model dtype
        key = attn.to_k(encoder_hidden_states)               # (B, S, C_inner)
        value = attn.to_v(encoder_hidden_states)             # (B, S, C_inner)
        inner_dim = query.shape[-1]
        heads = attn.heads
        head_dim = inner_dim // heads
        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)   # (B, H, N, d)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)       # (B, H, S, d)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)   # (B, H, S, d)

        # ---- CAST core: explicit fp32 attention weights (autocast disabled so matmul stays fp32) --------
        half = batch_size // 2
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scale   # (B, H, N, S) fp32
            if self.attention_type == "softmax":
                weights = dense_softmax(logits * self.beta)
                if self.collect_stats:
                    self.entropy_calls.append(_row_entropy(weights[half:]).mean().item())
            elif self.attention_type == "sharp_mix":
                w_dense = dense_softmax(logits)
                w_sharp = dense_softmax(logits * self.beta)
                weights = mix_attention_weights(w_dense, w_sharp, self.lambda_sharp)   # 0 -> w_dense, 1 -> w_sharp (same objects), else affine mix
                if self.collect_stats:
                    wc = weights[half:]
                    self.min_weight_calls.append(wc.min().item())
                    neg = wc < 0
                    self.neg_fraction_calls.append(neg.float().mean().item())
                    self.neg_mass_calls.append((wc.abs() * neg).sum(-1).mean().item())
                    self.entropy_sharp_endpoint_calls.append(_row_entropy(w_sharp[half:]).mean().item())
                    if self.lambda_sharp <= 1.0:
                        self.entropy_calls.append(_row_entropy(wc.clamp_min(0)).mean().item())   # non-negative up to fp tolerance for lambda_sharp<=1
            else:
                w_dense = dense_softmax(logits)
                if self.lam == 0.0:
                    weights = w_dense
                else:
                    w_sparse = alpha_entmax(logits, self.alpha)
                    weights = mix_attention_weights(w_dense, w_sparse, self.lam)
                    if not self.apply_to_uncond:
                        # Auffusion CFG batch layout is [uncond, cond]; keep dense weights on the uncond half.
                        weights = torch.cat([w_dense[:half], weights[half:]], dim=0)
                    if self.collect_stats:
                        nnz = (w_sparse > 0).sum(-1).float()
                        self.stat_support_sum += nnz.mean().item()
                        self.stat_support_sum_cond += nnz[half:].mean().item()
                        self.stat_support_n += 1
                        self.entropy_sparse_calls.append(_row_entropy(w_sparse[half:]).mean().item())
                if self.collect_stats:
                    self.entropy_calls.append(_row_entropy(w_dense[half:]).mean().item())
            if self.collect_stats and (self.probe_betas or self.probe_entmax):
                lc = logits[half:]
                for b in self.probe_betas:
                    self.probe_entropy_calls[b].append(_row_entropy(dense_softmax(lc * b)).mean().item())
                if self.probe_entmax:
                    ws = alpha_entmax(lc, 1.5)
                    self.probe_entmax_entropy_calls.append(_row_entropy(ws).mean().item())
                    self.probe_entmax_support_calls.append((ws > 0).sum(-1).float().mean().item())
            if self.check_finite and not torch.isfinite(weights).all():
                raise FloatingPointError(f"CastCrossAttnProcessor[{self.name}]: non-finite attention weights ({self.attention_type}, lam={self.lam}, beta={self.beta})")

        hidden_states = torch.matmul(weights.to(value.dtype), value)          # (B, H, N, d)

        # ---- epilogue (== AttnProcessor2_0) ------------------------------------------------------------
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)        # Linear
        hidden_states = attn.to_out[1](hidden_states)        # Dropout(p=0)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        self.calls += 1
        return hidden_states
