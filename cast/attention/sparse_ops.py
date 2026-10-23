"""Pure tensor ops for CAST attention normalization and lambda-extrapolation.

All functions expect float32 logits/weights of shape (..., S) and normalize/mix along dim=-1.

    dense_softmax(logits)              = Softmax(logits)                      (PLADIS Eq. 12 map)
    alpha_entmax(logits, alpha)        = alpha-Entmax(logits)                 (PLADIS Eq. 13 map)
    mix_attention_weights(D, S, lam)   = D + lam * (S - D)                    (PLADIS Eq. 15, on the weight maps)

For lam > 1 the mixed tensor is an *extrapolated attention weight* matrix, not a probability
distribution: rows still sum to 1 but entries may be negative. It is deliberately not clamped.
"""
import torch
from entmax import entmax15, sparsemax, entmax_bisect


def dense_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def alpha_entmax(logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha-Entmax along dim=-1. alpha=1 -> softmax, 1.5 -> exact entmax15 (PLADIS), 2 -> sparsemax."""
    if alpha == 1.0:
        # entmax_bisect(alpha=1.0) is NOT a valid softmax (verified numerically); dispatch explicitly.
        return torch.softmax(logits, dim=-1)
    if alpha == 1.5:
        return entmax15(logits, dim=-1)
    if alpha == 2.0:
        return sparsemax(logits, dim=-1)
    return entmax_bisect(logits, alpha=alpha, dim=-1)


def mix_attention_weights(w_dense: torch.Tensor, w_sparse: torch.Tensor, lam: float) -> torch.Tensor:
    """W = D + lam * (S - D). lam=0 -> D (returned as-is), lam=1 -> S (returned as-is), lam=2 -> 2S - D."""
    if lam == 0.0:
        return w_dense
    if lam == 1.0:
        return w_sparse
    return w_dense + lam * (w_sparse - w_dense)
