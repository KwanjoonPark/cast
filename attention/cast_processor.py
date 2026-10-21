"""CAST cross-attention processor for diffusers 0.18.2 `Attention` modules (text cross-attention only).

The prologue (projections, head split) and epilogue (merge heads, to_out, residual) are copied from
diffusers 0.18.2 `AttnProcessor2_0`. The single call to `F.scaled_dot_product_attention` is replaced by an
explicit, fp32 computation of

    logits   = (Q K^T) * attn.scale                      # attn.scale = head_dim ** -0.5
    W_dense  = Softmax(logits)
    W_sparse = alpha-Entmax(logits)                      # same logits, same Q/K
    W        = W_dense + lam * (W_sparse - W_dense)      # lam=0 dense, lam=1 sparse, lam=2 PLADIS
    out      = W @ V

Neither `F.scaled_dot_product_attention` nor xformers is used here. Self-attention (`attn1`) modules are
never given this processor (see cast.integration.auffusion.select_cross_attention).
"""
from typing import Optional

import torch

from .sparse_ops import dense_softmax, alpha_entmax, mix_attention_weights


class CastCrossAttnProcessor:
    def __init__(
        self,
        alpha: float = 1.5,
        lam: float = 0.0,
        apply_to_uncond: bool = True,
        check_finite: bool = True,
        collect_stats: bool = False,
        name: str = "",
    ):
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.apply_to_uncond = apply_to_uncond
        self.check_finite = check_finite
        self.collect_stats = collect_stats
        self.name = name
        self.reset_counters()

    def reset_counters(self):
        self.calls = 0
        self.stat_support_sum = 0.0        # running sum of mean nnz per row of W_sparse, all batch rows (lam != 0 only)
        self.stat_support_sum_cond = 0.0   # same, rows [B//2:] (= conditional half under Auffusion's [uncond, cond] CFG layout)
        self.stat_support_n = 0

    @property
    def mean_sparse_support(self) -> Optional[float]:
        return self.stat_support_sum / self.stat_support_n if self.stat_support_n else None

    @property
    def mean_sparse_support_cond(self) -> Optional[float]:
        return self.stat_support_sum_cond / self.stat_support_n if self.stat_support_n else None

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
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scale   # (B, H, N, S) fp32
            w_dense = dense_softmax(logits)
            if self.lam == 0.0:
                weights = w_dense
            else:
                w_sparse = alpha_entmax(logits, self.alpha)
                weights = mix_attention_weights(w_dense, w_sparse, self.lam)
                if not self.apply_to_uncond:
                    # Auffusion CFG batch layout is [uncond, cond]; keep dense weights on the uncond half.
                    half = batch_size // 2
                    weights = torch.cat([w_dense[:half], weights[half:]], dim=0)
                if self.collect_stats:
                    nnz = (w_sparse > 0).sum(-1).float()
                    self.stat_support_sum += nnz.mean().item()
                    self.stat_support_sum_cond += nnz[batch_size // 2:].mean().item()
                    self.stat_support_n += 1
            if self.check_finite and not torch.isfinite(weights).all():
                raise FloatingPointError(f"CastCrossAttnProcessor[{self.name}]: non-finite attention weights (lam={self.lam})")

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
