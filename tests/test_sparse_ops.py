import torch
import pytest
from entmax import entmax15, entmax_bisect

from cast.attention.sparse_ops import dense_softmax, alpha_entmax, mix_attention_weights


@pytest.fixture
def logits():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 8, 64, 77, generator=g) * 2.5          # fp32, realistic std


def test_alpha_dispatch(logits):
    assert torch.equal(alpha_entmax(logits, 1.0), torch.softmax(logits, -1))
    s15 = alpha_entmax(logits, 1.5)
    assert torch.equal(s15, entmax15(logits, dim=-1))
    assert (s15 - entmax_bisect(logits, alpha=1.5, dim=-1)).abs().max() < 1e-5
    s2 = alpha_entmax(logits, 2.0)
    assert (s2.sum(-1) - 1).abs().max() < 1e-5
    assert (s2 > 0).sum(-1).float().mean() <= (s15 > 0).sum(-1).float().mean()   # sparsemax sparser than 1.5-entmax


def test_entmax15_rows_sum_to_one_and_are_sparse(logits):
    s = alpha_entmax(logits, 1.5)
    assert (s.sum(-1) - 1).abs().max() < 1e-5
    assert (s >= 0).all()
    assert (s == 0).float().mean() > 0.5                            # majority of the 77 keys receive exactly zero


def test_mix_identities(logits):
    d, s = dense_softmax(logits), alpha_entmax(logits, 1.5)
    assert mix_attention_weights(d, s, 0.0) is d
    assert mix_attention_weights(d, s, 1.0) is s
    w2 = mix_attention_weights(d, s, 2.0)
    assert torch.allclose(w2, 2 * s - d, atol=1e-6)
    for w in (d, s, w2):
        assert (w.sum(-1) - 1).abs().max() < 1e-4                   # rows still sum to 1
    assert w2.min() < 0                                            # negative extrapolated weights are expected
    assert torch.isfinite(w2).all()


def test_zero_logits_make_sparse_equal_dense():
    # Auffusion's unconditional branch: K = 0 -> all-zero logits -> both maps uniform -> lambda has no effect.
    z = torch.zeros(1, 2, 5, 77)
    d, s = dense_softmax(z), alpha_entmax(z, 1.5)
    assert torch.allclose(d, torch.full_like(z, 1 / 77))
    assert torch.allclose(s, torch.full_like(z, 1 / 77), atol=1e-6)
    assert torch.allclose(mix_attention_weights(d, s, 2.0), d, atol=1e-6)
