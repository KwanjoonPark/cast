"""Processor-level tests on a randomly initialised SD1.5 U-Net built from the Auffusion unet config
(no checkpoint needed). Parity numbers are written to outputs/validation/unit_parity.json."""
import json
import os

import pytest
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

from cast.attention.cast_processor import CastCrossAttnProcessor
from cast.integration.auffusion import select_cross_attention, apply_cast, restore_processors, cast_installed

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def unet():
    cfg = json.load(open(os.path.join(HERE, "fixtures", "unet_config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    torch.manual_seed(0)
    return UNet2DConditionModel.from_config(cfg).to(DEV).eval()


def _stats(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_std": float(b.float().std())}


def test_census_and_selection(unet):
    procs = unet.attn_processors
    assert len(procs) == 32
    assert sum(n.endswith("attn1.processor") for n in procs) == 16
    sel = select_cross_attention(unet)
    assert len(sel) == 16 and all(n.endswith("attn2.processor") for n in sel)
    for n, m in sel.items():
        assert m.to_k.in_features == 768 and m.to_k.bias is None and m.to_v.bias is None


def test_apply_and_restore_touch_only_attn2(unet):
    before = dict(unet.attn_processors)
    originals = apply_cast(unet, alpha=1.5, lam=2.0)
    after = unet.attn_processors
    for n in before:
        if n.endswith("attn1.processor"):
            assert after[n] is before[n]                               # self-attention untouched (same object)
        else:
            assert isinstance(after[n], CastCrossAttnProcessor) and after[n].lam == 2.0
    restore_processors(unet, originals)
    final = unet.attn_processors
    assert all(final[n] is before[n] for n in before)                 # exact objects restored


def test_runtime_guards(unet):
    mod = select_cross_attention(unet)["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"]
    p = CastCrossAttnProcessor(lam=0.0)
    with pytest.raises(RuntimeError):
        p(mod, torch.randn(1, 8, 320, device=DEV))                     # self-attention call must fail
    with pytest.raises(RuntimeError):
        p(mod, torch.randn(1, 8, 320, device=DEV), torch.randn(1, 77, 768, device=DEV), attention_mask=torch.zeros(1, 1, 77, device=DEV))


@pytest.mark.parametrize("dtype", [torch.float32] + ([torch.float16] if DEV == "cuda" else []))
def test_lambda0_parity_per_module(unet, dtype):
    sel = select_cross_attention(unet)
    names = ["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",      # N=4096, C=320
             "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",      # N=1024, C=640
             "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",      # N=256,  C=1280
             "mid_block.attentions.0.transformer_blocks.0.attn2.processor"]          # N=64,   C=1280
    tokens = {320: 4096, 640: 1024, 1280: 256}
    results = {}
    g = torch.Generator(device=DEV).manual_seed(1)
    for n in names:
        mod = sel[n]
        C = mod.to_q.in_features
        N = 64 if n.startswith("mid") else tokens[C]
        orig_dtype = next(mod.parameters()).dtype
        mod.to(dtype)
        h = torch.randn(2, N, C, generator=g, device=DEV, dtype=dtype)
        e = torch.randn(2, 77, 768, generator=g, device=DEV, dtype=dtype)
        e[0] = 0                                                       # row 0 mimics the zero null condition
        with torch.no_grad():
            ref_sdpa = AttnProcessor2_0()(mod, h, e)
            ref_bmm = AttnProcessor(mod)(mod, h, e) if False else AttnProcessor()(mod, h, e)
            cast0 = CastCrossAttnProcessor(alpha=1.5, lam=0.0)(mod, h, e)
            cast1 = CastCrossAttnProcessor(alpha=1.5, lam=1.0)(mod, h, e)
            cast2 = CastCrossAttnProcessor(alpha=1.5, lam=2.0)(mod, h, e)
        mod.to(orig_dtype)
        r = {"vs_AttnProcessor2_0": _stats(cast0, ref_sdpa), "vs_AttnProcessor": _stats(cast0, ref_bmm),
             "sdpa_vs_bmm": _stats(ref_sdpa, ref_bmm),
             "linearity_2x1_minus_0": _stats(cast2, 2 * cast1.float() - cast0.float()),
             "uncond_row_lam2_vs_lam0": _stats(cast2[0], cast0[0]),
             "cond_row_lam1_vs_lam0": _stats(cast1[1], cast0[1])}
        results[n] = r
        tol = 1e-4 if dtype == torch.float32 else 2e-2                 # relative to the output std
        assert r["vs_AttnProcessor2_0"]["max_abs"] < tol * max(r["vs_AttnProcessor2_0"]["ref_std"], 1e-3) * 10
        assert r["vs_AttnProcessor"]["max_abs"] < tol * max(r["vs_AttnProcessor"]["ref_std"], 1e-3) * 10
        assert r["uncond_row_lam2_vs_lam0"]["max_abs"] == 0.0           # exact no-op on the zero-conditioned row
        assert r["cond_row_lam1_vs_lam0"]["max_abs"] > 0.0              # sparse changes the conditioned row
        if dtype == torch.float32:
            assert r["linearity_2x1_minus_0"]["max_abs"] < 1e-4 * max(r["linearity_2x1_minus_0"]["ref_std"], 1e-3) * 10
    os.makedirs(os.path.join(ROOT, "outputs", "validation"), exist_ok=True)
    path = os.path.join(ROOT, "outputs", "validation", f"unit_parity_{str(dtype).split('.')[-1]}.json")
    json.dump(results, open(path, "w"), indent=2)


def test_full_unet_forward_counts_and_parity(unet):
    # cuDNN TF32 convolutions (default on Ampere) make even stock-vs-stock repeats differ at ~1e-3 max;
    # disable them here so the comparison is against a ~5e-6 repeat floor (measured on RTX A5000).
    tf32_prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        dtype = torch.float32
        g = torch.Generator(device=DEV).manual_seed(2)
        H, W = (32, 128) if DEV == "cuda" else (8, 32)
        x = torch.randn(2, 4, H, W, generator=g, device=DEV, dtype=dtype)
        e = torch.randn(2, 77, 768, generator=g, device=DEV, dtype=dtype); e[0] = 0
        t = torch.tensor(500, device=DEV)
        with torch.no_grad():
            ref = unet(x, t, encoder_hidden_states=e).sample
            ref2 = unet(x, t, encoder_hidden_states=e).sample
            with cast_installed(unet, alpha=1.5, lam=0.0):
                out0 = unet(x, t, encoder_hidden_states=e).sample
                calls = [p.calls for n, p in unet.attn_processors.items() if n.endswith("attn2.processor")]
            with cast_installed(unet, alpha=1.5, lam=2.0):
                out2 = unet(x, t, encoder_hidden_states=e).sample
        floor = max(float((ref - ref2).abs().max()), 1e-7)              # stock repeat non-determinism floor
        res = {"stock_repeat_floor": _stats(ref2, ref), "stock_vs_cast0": _stats(out0, ref),
               "uncond_row_stock_vs_cast2": _stats(out2[0], ref[0]), "cond_row_stock_vs_cast2": _stats(out2[1], ref[1]), "calls": calls}
        os.makedirs(os.path.join(ROOT, "outputs", "validation"), exist_ok=True)
        json.dump(res, open(os.path.join(ROOT, "outputs", "validation", "unit_full_unet_fp32.json"), "w"), indent=2)
        assert calls == [1] * 16                                           # every attn2 called exactly once per forward
        assert res["stock_vs_cast0"]["max_abs"] <= 10 * floor               # lambda=0 == stock within the kernel floor
        assert res["cond_row_stock_vs_cast2"]["max_abs"] > 100 * floor      # PLADIS changes the conditional output
        assert res["uncond_row_stock_vs_cast2"]["max_abs"] <= 10 * floor    # ...but not the uncond one (K=V=0)
        assert all(not isinstance(p, CastCrossAttnProcessor) for p in unet.attn_processors.values())
    finally:
        torch.backends.cudnn.allow_tf32 = tf32_prev


def test_sharp_softmax_beta1_bitwise_equals_lambda0_and_beta2_differs(unet):
    """Sharp Softmax beta=1 must be bitwise identical to the validated entmax_mix lambda=0 path (same code up to a
    multiplication by 1.0); beta=2 must differ on conditioned rows and be a no-op on the zero-conditioned row."""
    sel = select_cross_attention(unet)
    g = torch.Generator(device=DEV).manual_seed(3)
    for n in ["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor", "mid_block.attentions.0.transformer_blocks.0.attn2.processor"]:
        mod = sel[n]; C = mod.to_q.in_features; N = 64 if n.startswith("mid") else 4096
        for dtype in [torch.float32] + ([torch.float16] if DEV == "cuda" else []):
            od = next(mod.parameters()).dtype; mod.to(dtype)
            h = torch.randn(2, N, C, generator=g, device=DEV, dtype=dtype); e = torch.randn(2, 77, 768, generator=g, device=DEV, dtype=dtype); e[0] = 0
            with torch.no_grad():
                lam0 = CastCrossAttnProcessor(alpha=1.5, lam=0.0)(mod, h, e)
                b1 = CastCrossAttnProcessor(attention_type="softmax", beta=1.0)(mod, h, e)
                b2 = CastCrossAttnProcessor(attention_type="softmax", beta=2.0)(mod, h, e)
                ref = AttnProcessor2_0()(mod, h, e)
            mod.to(od)
            assert torch.equal(b1, lam0), f"beta=1 != lambda=0 ({n}, {dtype})"
            assert (b1.float() - ref.float()).abs().max() < (1e-4 if dtype == torch.float32 else 2e-2) * max(float(ref.float().std()), 1e-3) * 10
            assert (b2[1].float() - b1[1].float()).abs().max() > 0
            assert torch.equal(b2[0], b1[0])                               # K=V=0 row: softmax of zeros is uniform for any beta


def test_entropy_stats_and_probes(unet):
    sel = select_cross_attention(unet); mod = sel["down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor"]
    g = torch.Generator(device=DEV).manual_seed(4)
    h = torch.randn(2, 1024, 640, generator=g, device=DEV); e = torch.randn(2, 77, 768, generator=g, device=DEV); e[0] = 0
    import math
    with torch.no_grad():
        p1 = CastCrossAttnProcessor(attention_type="softmax", beta=1.0, collect_stats=True, probe_betas=(1.0, 2.0), probe_entmax=True); p1(mod, h, e)
        p2 = CastCrossAttnProcessor(attention_type="softmax", beta=2.0, collect_stats=True); p2(mod, h, e)
        pe = CastCrossAttnProcessor(alpha=1.5, lam=1.0, collect_stats=True); pe(mod, h, e)
    H1, H2 = p1.mean_entropy, p2.mean_entropy
    assert 0 < H2 < H1 <= math.log(77) + 1e-6                             # sharpening lowers entropy; bounded by log(S)
    assert abs(p1.probe_entropy_calls[1.0][0] - H1) < 1e-6                 # probe at beta=1 equals the run statistic
    assert abs(p1.probe_entropy_calls[2.0][0] - H2) < 1e-5                 # probe at beta=2 equals the beta=2 run statistic (same logits)
    assert abs(p1.probe_entmax_entropy_calls[0] - pe.mean_entropy_sparse) < 1e-5 and abs(p1.probe_entmax_support_calls[0] - pe.mean_sparse_support_cond) < 1e-5
    assert pe.mean_entropy_sparse < pe.mean_entropy                        # entmax map is more concentrated than the dense map on the same logits
    assert math.exp(pe.mean_entropy_sparse) <= pe.mean_sparse_support_cond + 1e-6   # effective support <= exact support (per row, Jensen-safe on means only approx) 


def test_sharp_mix_algebra_and_parity(unet):
    """sharp_mix: lambda_sharp=0 bitwise == dense (lam=0 path), =1 bitwise == softmax(beta=1.5); 0.5 == 0.5*dense+0.5*sharp
    (via outputs: linear in W); rows sum to 1; non-negative for lambda_sharp<=1; signed allowed for >1 (not clamped)."""
    from cast.attention.sparse_ops import dense_softmax, mix_attention_weights
    sel = select_cross_attention(unet); mod = sel["down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"]
    g = torch.Generator(device=DEV).manual_seed(5)
    h = torch.randn(2, 4096, 320, generator=g, device=DEV); e = torch.randn(2, 77, 768, generator=g, device=DEV); e[0] = 0
    with torch.no_grad():
        dense = CastCrossAttnProcessor(alpha=1.5, lam=0.0)(mod, h, e)
        sharp = CastCrossAttnProcessor(attention_type="softmax", beta=1.5)(mod, h, e)
        m0 = CastCrossAttnProcessor(attention_type="sharp_mix", beta=1.5, lambda_sharp=0.0)(mod, h, e)
        m1 = CastCrossAttnProcessor(attention_type="sharp_mix", beta=1.5, lambda_sharp=1.0)(mod, h, e)
        p05 = CastCrossAttnProcessor(attention_type="sharp_mix", beta=1.5, lambda_sharp=0.5, collect_stats=True); m05 = p05(mod, h, e)
        p15 = CastCrossAttnProcessor(attention_type="sharp_mix", beta=1.5, lambda_sharp=1.5, collect_stats=True); m15 = p15(mod, h, e)
    assert torch.equal(m0, dense) and torch.equal(m1, sharp)
    assert (m05 - (0.5 * dense + 0.5 * sharp)).abs().max() < 1e-4 * max(float(dense.std()), 1e-3) * 10     # output linear in W
    # weight-level algebra on the same logits
    q = mod.to_q(h); k = mod.to_k(e); H, d = mod.heads, q.shape[-1] // mod.heads
    q = q.view(2, -1, H, d).transpose(1, 2).float(); k = k.view(2, -1, H, d).transpose(1, 2).float()
    Z = torch.matmul(q, k.transpose(-1, -2)) * mod.scale; Wd, Ws = dense_softmax(Z), dense_softmax(1.5 * Z)
    for ls in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        W = mix_attention_weights(Wd, Ws, ls)
        assert (W.sum(-1) - 1).abs().max() < 1e-4
        if ls <= 1.0: assert W.min() >= -1e-6
    # negative entries at lambda_sharp>1 are possible, not guaranteed (random-init logits are nearly flat); check on sharper logits
    Zs = Z * 8; Wd8, Ws8 = dense_softmax(Zs), dense_softmax(1.5 * Zs); W15 = mix_attention_weights(Wd8, Ws8, 1.5)
    assert W15.min() < 0 and (W15.sum(-1) - 1).abs().max() < 1e-4                           # signed extrapolation, rows still sum to 1
    assert mix_attention_weights(Wd8, Ws8, 1.0).min() >= 0 and mix_attention_weights(Wd8, Ws8, 0.75).min() >= -1e-6
    assert p05.min_weight_calls[0] >= -1e-6 and p05.neg_mass_calls[0] < 1e-6 and p05.entropy_calls
    assert len(p15.neg_fraction_calls) == 1 and len(p15.neg_mass_calls) == 1 and not p15.entropy_calls   # diagnostics recorded; no entropy on possibly-signed weights
    assert abs(p15.entropy_sharp_endpoint_calls[0] - p05.entropy_sharp_endpoint_calls[0]) < 1e-6          # endpoint reference identical
