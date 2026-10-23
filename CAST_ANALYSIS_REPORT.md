# CAST Integration Analysis Report

**CAST — Cross-Attention Sparsification for Text-to-Audio Generation**
Static analysis of PLADIS (ICCV'25) → Auffusion (arXiv 2401.01044) transfer. Analysis only; no implementation performed.

Date: 2026-10-20. Repository root: `/home/reallab/parkkwanjoon/CAST`.

Evidence tags used throughout:

| Tag | Meaning |
|---|---|
| **[PAPER]** | Fact read from the PLADIS or Auffusion PDF (section / equation / table cited) |
| **[CODE]** | Fact read from `third_party/PLADIS`, `third_party/Auffusion`, diffusers 0.18.2 / 0.33.1 source, or entmax 1.3 source |
| **[MEASURED]** | Result of a lightweight numeric test run during this analysis (scripts in the session scratchpad; no checkpoints downloaded) |
| **[INFERENCE]** | My own reasoning or recommendation |

---

## 1. Executive Summary

**Is PLADIS technically transferable to Auffusion?** Yes, with high confidence. **[CODE]** Auffusion's U-Net is a stock diffusers 0.18.2 `UNet2DConditionModel` in the Stable Diffusion v1.5 configuration (16 `BasicTransformerBlock`s, each with a self-attention `attn1` and a text cross-attention `attn2`, 8 heads, `cross_attention_dim=768`). PLADIS is defined on exactly this `Attention` abstraction, and its reference implementation is a drop-in `AttentionProcessor` whose only functional change is replacing `softmax` with a mix of `softmax` and `entmax15` on the attention logits. Nothing in the method depends on images, SDXL, or diffusers 0.33.x.

**Cleanest intervention point.** Replace the `.processor` object of the 16 `*.transformer_blocks.0.attn2` modules of `pipeline.unet` with a CAST processor that recomputes `Q, K, V` exactly as the stock processor does, forms the logits `QKᵀ/√d` once in fp32, computes `P_dense = softmax(logits)` and `P_sparse = entmax_1.5(logits)`, and returns `(P_dense + λ·(P_sparse − P_dense)) @ V` through the unchanged `to_out` projection. `attn1` processors are never touched. No change to `third_party/*` is required: the swap is done from outside via `module.set_processor(...)` (diffusers public API) and can be reverted.

**Largest risks.**
1. **Environment / dependency pinning (HIGH, operational).** Auffusion requires `diffusers==0.18.2` (its pipeline imports symbols removed in later releases; verified to fail on 0.30.2). PLADIS targets 0.33.1. The two cannot share one environment; CAST must re-implement the ~40-line processor against the 0.18.2 API (trivial) rather than import PLADIS's pipeline. Additionally, this machine's base Python ships a torch built for CUDA 13.0 which the installed driver (570, CUDA 12.8) cannot run; a dedicated env is mandatory.
2. **Baseline backend ambiguity (MEDIUM, scientific).** Auffusion's official `inference.sh` enables xformers; our processor must materialize attention probabilities, so the dense branch cannot use xformers/SDPA kernels. Vanilla must therefore be defined as "stock processor, untouched" and the CAST-dense path validated against it at fp16 rounding level (not bit-exactness).
3. **Audio-specific semantics (MEDIUM, scientific, not a blocker).** Auffusion zero-pads text embeddings to 77 *before* the adapter, so all padding positions share one identical key vector; and the unconditional CFG branch uses an all-zero embedding, which makes PLADIS a provable no-op on that branch. Neither breaks the integration, but both change what "sparsity" does compared to SDXL and must be measured, not assumed.

**Recommended strategy.** One pipeline, one processor class, one config with `mode ∈ {vanilla, sparse, pladis}` mapping to `λ ∈ {0 (and no swap), 1, 2}` with `α = 1.5`. Build in four verifiable stages: (i) environment + golden vanilla output, (ii) module census and processor swap with λ=0 parity test, (iii) entmax ops with unit tests, (iv) λ wiring and the three-mode sanity run. Details in §14.

---

## 2. Repository Overview

```
CAST/
├── CLAUDE.md                      # project rules (notes ours/ as implementation dir; this task uses cast/ — see §16)
├── papers/
│   ├── PLADIS ... .pdf            # arXiv 2503.07677v3 (19 Jul 2025), 30 pages incl. supplement
│   └── Auffusion ... .pdf         # arXiv 2401.01044v1 (2 Jan 2024), 17 pages incl. supplement
├── third_party/
│   ├── PLADIS/                    # READ ONLY. 4 pipeline files, no package, no tests
│   │   ├── README.md
│   │   └── pipeline/{pipeline_sdxl.py, pipeline_sdxl_seg.py, pipeline_controlnet.py, pipeline_flux.py}
│   └── Auffusion/                 # READ ONLY. diffusers-0.18.2-era pipeline
│       ├── auffusion_pipeline.py  # AuffusionPipeline + ConditionAdapter + HiFi-GAN Generator (1039 lines)
│       ├── inference.py / inference.sh / requirements.txt / config/clip_condition_config.json
│       ├── converter.py, utils.py # mel/vocoder helpers, dataset utils
│       └── prompt2prompt/         # P2P attention-control (precedent for processor injection)
├── cast/            (empty)       # future implementation
├── experiments/     (empty)
└── outputs/         (empty)
```

**Versions [CODE].**

| Component | Version / pin | Source |
|---|---|---|
| Auffusion required stack | `torch==2.0.1`, `diffusers==0.18.2`, `huggingface_hub==0.16.4`, `transformers` (unpinned), `xformers==0.0.20`, `tokenizers==0.13.3`, `librosa==0.9.2` | `third_party/Auffusion/requirements.txt`, README "Prerequisites" |
| Auffusion checkpoint | `auffusion/auffusion` (HF), `_diffusers_version: 0.18.2` in `unet/config.json`, `vae/config.json`, `scheduler/scheduler_config.json` | HF repo listing (fetched JSON configs only) |
| PLADIS required stack | `diffusers 0.33.1`, `pip install entmax`, torch ≥ 2.0 | `third_party/PLADIS/README.md`; `pipeline_sdxl.py` imports |
| entmax | 1.3 (pure Python, `torch>=1.3`) | wheel metadata |
| This machine | RTX A5000 24 GB, driver 570.133.07 (CUDA 12.8). Base conda: Python 3.13, torch 2.12.1+cu130 (**GPU unusable**: driver too old), no diffusers. Env `acg`: Python 3.10, torch 2.7.1+cu128 (GPU OK), diffusers 0.30.2, entmax 1.3, no xformers. No env has diffusers 0.18.2. | `nvidia-smi`, per-env import scan |
| HF cache | No Auffusion checkpoint cached | `~/.cache/huggingface/hub` |

The HF repo `auffusion/auffusion` contains: `condition_adapter_0/{config.json,condition_adapter.pt}`, `condition_config.json`, `feature_extractor/`, `scheduler/`, `text_encoder_0/{config.json,pytorch_model.bin,tokenizer files}`, `unet/`, `vae/`, `vocoder/{config.json,vocoder.pt}`. There is **no `model_index.json`**, so it can only be loaded by `AuffusionPipeline.from_pretrained` (custom loader), not by generic `DiffusionPipeline.from_pretrained`. **[CODE]**

---

## 3. PLADIS Paper Analysis

### 3.1 Formulation

**[PAPER]** PLADIS paper, Sec. 2.3 and Sec. 3 (Eqs. 12, 13, 15):

```
At(Q,K,V)        = Softmax(QKᵀ/√d) V                              (Eq. 12, dense)
At_α(Q,K,V)      = α-Entmax(QKᵀ/√d) V                             (Eq. 13, sparse)
At_ours(α,λ,Q,K,V) = At(Q,K,V) + λ ( At_α(Q,K,V) − At(Q,K,V) )     (Eq. 15, PLADIS)
```

α-Entmax (Eq. 10–11, supplement Eq. 22–24): `α-Entmax(z) = argmax_{p∈Δ} ⟨p,z⟩ − Ψ_α(p)` with Tsallis entropy Ψ_α; closed form `[(α−1)z − τ(z)1]_+^{1/(α−1)}`. α=1 → softmax, α=2 → sparsemax, α=1.5 has an exact sort-based algorithm (Peters et al. 2019).

**Where sparsity is introduced.** On the attention *probability map* (the output of the normalization applied to logits `QKᵀ/√d`). Logits are unchanged; `Q, K, V` are unchanged; entries of the probability map below the threshold τ become exactly 0. Because `(·)V` is linear, Eq. 15 written on outputs equals mixing the probability maps first: `(P_dense + λ(P_sparse − P_dense)) V`. The code does the latter (§4).

**Role of α.** Sparsity degree, 1 < α ≤ 2. Sec. 3.2 / Fig. 4: increasing α improves FID / CLIPScore / ImageReward on MS-COCO. Sec. 6 "The Effect of α" and Table 5: α=2 is best on FID/CLIP but worse on ImageReward; α=1.5 chosen as default ("balanced", and ~3× faster than non-closed-form α values). Supplement H, Fig. 21: α=1.5 "more stable and natural". Theory (Thm. 2, Cor. 2.1, supplement Thm. 4): for 1<α≤2 the retrieval error under query noise is polynomial of order 1/(α−1) instead of exponential (softmax).

**Role of λ.** Extrapolation scale, Sec. 3.4: λ=0 baseline, λ=1 pure sparse attention (Sec. 3.2 setting), λ>1 PLADIS. Sec. 6 "The Scale λ" and Fig. 8: FID/CLIPScore peak at λ=2.0, PickScore at 1.5; default λ=2.0. Fig. 5 / Fig. 20: too-large λ (3–4) gives smoother textures and artifacts "similar to those seen in CFG". Supplement G.5 Table 10: λ=2 > λ=1 (only sparse) > baseline on GenEval. Supplement Thm. 5: error bound of the mixture is `|λ|·(polynomial) + |1−λ|·(exponential)`.

### 3.2 Which modules are modified

| Question | Answer | Source |
|---|---|---|
| Self-attention modified? | **No.** Sparse self-attention concentrates on the diagonal and "ultimately leads to failure in image generation" | **[PAPER]** Sec. 3.1, Fig. 3(a) |
| Cross-attention modified? | **Yes**, text→latent cross-attention | **[PAPER]** Sec. 3.1–3.4, Fig. 2(b) |
| Every cross-attention layer? | **Yes**: "applying Eq. 15 to all cross-attention layers is sufficient"; Algorithm 1 lines 1–2 loop over all L layers; Table 9: All > any subset (Up alone is the strongest single group) | **[PAPER]** Sec. 3.4, Alg. 1, Supp. G.3 / Table 9 |
| Cond / uncond CFG branches? | The paper does not single out the unconditional branch. Alg. 1 replaces the modules first (line 2) and then evaluates `ε_θ(x_t, c)` and, for CFG, Eq. 4 — i.e. the replaced module is used for whatever the sampler evaluates, including `ε_θ(x_t, ∅)` | **[PAPER]** Alg. 1; **[CODE]** confirms both branches share one batched forward (§4.5) |
| Number of forward passes / NFEs | **Unchanged.** "without requiring additional training or extra inference steps"; Table 1 ("Need extra inference": ✗); Fig. 2 (≈1× memory/time) | **[PAPER]** Abstract, Sec. 1, Table 1 |
| Temperature | Supplement G.1 (Eq. 36, Fig. 10): lowering τ sharpens softmax but does not reach 1.5-Entmax performance; 1.5-Entmax is robust to τ. Default τ=1 (β=1/√d) | **[PAPER]** Supp. G.1 |
| Extrapolation type | "In-model" extrapolation inside attention. Output-level extrapolation (mixing two model outputs, like guidance) is inferior and costs an extra NFE. Identity (PAG) or blurred (SEG) attention cannot be extrapolated | **[PAPER]** Supp. G.4, Fig. 12, Table 8 |

### 3.3 Implementation details not obvious from Eq. 15 [PAPER]

- Experiments: SDXL, 25 steps, CFG scale drawn uniformly in (3, 5), PAG/SEG scale 3.0, H100 (Sec. 4, Supp. C). Auffusion uses CFG 7.5 and 100 steps (§5), so the λ–CFG operating point will differ.
- Cost: Table 5, 25-step SDXL, per prompt: 2.521 s (α=1) vs 3.085 s (α=1.5), memory 16.44 vs 16.45 GB. So ≈ +22 % wall-clock, negligible memory.
- Qualitative mechanism: Fig. 11 — cross-attention maps become sharper and better localized per word; λ=2 sharper than λ=1.
- The entmax library used is deep-spin/entmax (footnote in Sec. 4).

### 3.4 What may NOT transfer from T2I to T2A [INFERENCE]

1. **Query semantics.** In SDXL a query token is a spatial patch usually "owned" by one object (occlusion is exclusive). In Auffusion a query token is a time–frequency cell of a mel-spectrogram (one 8×8 latent cell = 8 mel bins × 80 ms, see §5.2); audio is *additive* — "dog barking while rain falls" means the same cell legitimately mixes two sources. Forcing each cell to attend to ≤ a few tokens could suppress background/ambient events. This is the central scientific question of CAST and cannot be answered statically; the pad-mass / support-size probe in §13 is designed to observe it.
2. **Text token structure.** Auffusion zero-pads token states before a trainable adapter (§6.2), producing ~60–70 *identical* pad keys per prompt; SDXL feeds distinct CLIP hidden states for every position. Sparse normalization treats identical keys as one group (all in or all out of the support), so the effective number of keys is `n_tokens + 1`, not 77.
3. **Unconditional branch.** SDXL uncond uses CLIP("") embeddings (non-zero); Auffusion uncond is an all-zero tensor, which makes sparse = dense exactly on that branch (§10). PLADIS's gains reported "without CFG" (Table 2 top) therefore have no analogue here; all effect flows through the conditional branch and is amplified by `w=7.5`.
4. **Hyper-parameters.** α=1.5, λ=2 were tuned on SDXL at CFG 3–5 with 25 steps. Fig. 5/8 show λ is sensitive; the same values on a different backbone and CFG scale are a starting point, not a guarantee.
5. **Spatial-resolution mix.** SDXL (1024²) cross-attention operates at 32² and 64² tokens (2048 → 1024 tokens); Auffusion's largest level has 4096 tokens of a very anisotropic 32×128 grid. Entmax is applied per query row and is independent of the grid shape, so this is a cost, not a correctness, difference.

---

## 4. PLADIS Code Analysis

All paths relative to `third_party/PLADIS/`. The SDXL pipeline is the reference; `pipeline_sdxl_seg.py` and `pipeline_controlnet.py` contain byte-identical copies of the two PLADIS functions (verified by diff; only the unused `method` default differs), and `pipeline_flux.py` has an MMDiT variant.

### 4.1 Component map [CODE]

| Component | Location | What it does | Paper correspondence |
|---|---|---|---|
| Dense attention | `pipeline/pipeline_sdxl.py`, `sparse_scaled_dot_product_attention(...)`, branch `use_sparse=False` | `attn_weight = q @ kᵀ * scale (+bias)`; `softmax(dim=-1)`; `torch.dropout(p=0.0)`; `@ v` | Eq. 12 |
| Sparse attention | same function, branch `use_sparse=True`, `method=="ent15max"` → `entmax15(attn_weight/temper, dim=-1)`; `method=="sparsemax"` → `sparsemax(...)` | α-Entmax on the same logits | Eq. 13, α∈{1.5, 2} only |
| PLADIS extrapolation | same function: `dense_attn = softmax(attn_weight)`; `attn_weight = pladis_scale*sparse_attn + (1-pladis_scale)*dense_attn`; `return attn_weight @ value` | Mix of probability maps, single matmul with V | Eq. 15 (algebraically identical: `λS + (1−λ)D = D + λ(S−D)`) |
| α | not a parameter; selected by `method` string (`"ent15max"`=1.5 default, `"sparsemax"`=2). No general-α path (`entmax_bisect` not imported) | — | Sec. 6: only α∈{1.5, 2} considered |
| λ | `pladis_scale` (pipeline `__call__` kwarg, default 0.0) → `self._pladis_scale` → `SparseAttnProcessor.pladis_scale` | `do_sparse_guidance = pladis_scale > 0.0` | λ |
| Temperature | `temper=1.0` default argument, never passed from the processor → fixed to 1 | — | Supp. G.1 ablation not exposed in released code |
| Attention processor | `class SparseAttnProcessor` (`__init__(pladis_scale, method)`, `__call__(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs)`) | Verbatim copy of diffusers 0.33.1 `AttnProcessor2_0.__call__` with `F.scaled_dot_product_attention(...)` replaced by `sparse_scaled_dot_product_attention(..., use_sparse=True, method=self.method, pladis_scale=self.pladis_scale)` (the only functional diff; confirmed by `diff`) | "simply replacing the cross-attention module" |
| Target-layer selection | `PLADISStableDiffusionXLPipeline.__call__`, block "if self.do_sparse_guidance:" before the denoising loop: `for name, module in self.unet.named_modules(): if "attn2" in name and "to" not in name:` → group by `name.split(".")[0].split("_")[0]` ∈ {down, mid, up} | String heuristic on module names; `"to" not in name` excludes `attn2.to_q/to_k/to_v/to_out` children | Sec. 3.4 / Table 9 layer groups |
| Processor swap | inside the denoising loop, every step: `replace_processor = SparseAttnProcessor(self.pladis_scale, self.method)`; for each selected group: `layer.processor = replace_processor` (direct attribute assignment, same object shared by all layers) | Re-assigned each step (idempotent). After the loop: `layer.processor = AttnProcessor2_0()` — unconditionally resets to the torch-2 default, **not** to whatever processor was there before | Alg. 1 line 2 |
| Default layer groups | `pladis_applied_layers: List[str] = ['up','down']` in `pipeline_sdxl.py` and `pipeline_sdxl_seg.py` (**mid excluded**); `['down','mid','up']` in `pipeline_controlnet.py`; Flux replaces every `transformer_blocks[i].attn` | — | **Discrepancy with paper default "All" (Sec. 3.4, Table 9)** |
| CFG handling | `latent_model_input = torch.cat([latents]*2)`; `prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])`; one `self.unet(...)` call; `noise_pred.chunk(2)`; standard CFG | Processor acts on the whole batch, i.e. on both uncond and cond rows; no branch-specific logic | Alg. 1 |
| Model-specific handling | SDXL only (`UNet2DConditionModel` naming, `added_cond_kwargs`); PAG/SEG self-attention processors live in the same file but are separate classes applied to `attn1` only when `pag_scale>0`; Flux variant upcasts to fp32, slices the image→text block of joint attention and renormalizes it to the dense mass | — | — |

### 4.2 Verified execution path (SDXL, CFG only) [CODE]

```
PLADISStableDiffusionXLPipeline.__call__(prompt, guidance_scale, pladis_scale=2.0, ...)
 ├─ encode_prompt → prompt_embeds [2B, 77, 2048] = cat(negative, positive)
 ├─ collect ot_down_layers / ot_mid_layers / ot_up_layers (attn2 modules by name)
 └─ for t in timesteps:
      latent_model_input = cat([latents]*2)
      if do_sparse_guidance: for layer in selected groups: layer.processor = SparseAttnProcessor(λ, method)
      noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, ...)
          └─ ... BasicTransformerBlock.attn2.forward → Attention.forward → self.processor(self, hidden_states, encoder_hidden_states, ...)
                └─ SparseAttnProcessor.__call__ → sparse_scaled_dot_product_attention(use_sparse=True)
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2); CFG
 └─ after loop: selected layers ← AttnProcessor2_0()
```

### 4.3 Mechanism classification [CODE]

| Property | Finding |
|---|---|
| Replaces a diffusers `AttentionProcessor`? | **Yes** (object assigned to `Attention.processor`) |
| Monkey-patches forward functions? | No |
| Modifies attention weights (probabilities)? | **Yes** — mixes `softmax` and `entmax15` probability maps before `@ V` |
| Modifies attention output directly? | No (only through the modified probabilities) |
| Custom kernels? | No. Pure PyTorch + `entmax` (pure Python: sort, cumsum, sqrt, clamp) |
| Depends on xformers? | No. `XFormersAttnProcessor` is imported only for the inherited SDXL `upcast_vae` check |
| Depends on diffusers version? | Written against 0.33.1 (`Attention.norm_q/norm_k`, `deprecate("scale", ...)` signature, 4-D SDPA layout). The processor body is version-sensitive; the two core functions are not |
| Assumes SDXL? | The pipeline class, yes; the processor assumes only the diffusers `Attention` attributes (`to_q/to_k/to_v/to_out`, `heads`, `spatial_norm`, `group_norm`, `norm_cross`, `residual_connection`, `rescale_output_factor`) |
| SDPA/flash kernels used in the sparse path? | No — materialized `q @ kᵀ`, explicit normalization, explicit `@ v` |
| Dropout | `dropout_p=0.0`; sparse branch returns before the dropout call |
| Precision | SDXL variant computes everything in the model dtype (fp16 in README examples). Flux variant explicitly `.float()`s before `softmax`/`entmax15` and casts back |

### 4.4 Tensor shapes and head convention [CODE]

In `SparseAttnProcessor.__call__` (B = batch incl. CFG rows, N = query tokens, S = key tokens = 77, H = heads, d = head_dim):

| Tensor | Shape | Produced by |
|---|---|---|
| `hidden_states` (input) | `(B, N, C)` (3-D; 4-D conv maps are flattened first) | caller (`BasicTransformerBlock`) |
| `query` | `(B, N, C)` → `.view(B, -1, H, d).transpose(1, 2)` → **`(B, H, N, d)`** | `attn.to_q` |
| `key`, `value` | `(B, S, C)` → **`(B, H, S, d)`** | `attn.to_k`, `attn.to_v` on `encoder_hidden_states` |
| `attn_bias` | `(N, S)` zeros (+ mask) | function-local |
| `attn_weight` (logits) | **`(B, H, N, S)`** = `query @ key.transpose(-2,-1) * d^-0.5` | |
| `sparse_attn`, `dense_attn`, mixed | `(B, H, N, S)`, normalized along `dim=-1` (keys) | `entmax15`, `softmax` |
| output | `(B, H, N, d)` → `.transpose(1, 2).reshape(B, -1, H*d)` → `(B, N, C)` → `to_out[0]` (Linear) → `to_out[1]` (Dropout) | |

Head convention: **4-D "batch, heads, tokens, dim"** (the `AttnProcessor2_0` / SDPA convention). Note diffusers' older `AttnProcessor` uses the 3-D "(B·H, tokens, dim)" convention via `head_to_batch_dim`; both normalize over the last axis, so entmax is convention-agnostic as long as `dim=-1`.

### 4.5 Faithfulness of code to paper [INFERENCE]

- Eq. 15 ⇔ code: exact (probability-level mixing then one matmul equals output-level mixing by linearity). **[MEASURED]** `λS+(1−λ)D` vs `D+λ(S−D)` differ by 1.2e-7 in fp32 and ≤ 9.8e-4 in fp16 (same inputs).
- "All cross-attention layers" (paper) vs `['up','down']` default (SDXL code): the code default omits the mid block. For CAST, follow the paper (all 16 `attn2`) and keep the group list configurable.
- The code applies sparse attention to the unconditional CFG row as well; the paper is silent. See §10.
- Restoring `AttnProcessor2_0()` rather than the original processor is a latent bug (it would silently disable xformers / LoRA processors set by the user). CAST must save and restore originals.

---

## 5. Auffusion Paper Analysis

**[PAPER]** Auffusion paper, Sec. 3 (Fig. 1 overview), Sec. 3.5–3.6, Sec. 4.1 "Training Setup", Supplement B.

### 5.1 Architecture relevant to CAST

| Component | Paper statement | Detail |
|---|---|---|
| Feature spaces | audio → mel-spectrogram → "pixel" image → VAE latent (Sec. 3.1–3.2) | 16 kHz mono, 10.24 s, 256 mel bins, 1024-pt window, 2048 FFT, hop 160 → (256 × 1024) mel image, replicated to 3 channels, normalized by dataset-wide μ/σ (Sec. 4.1; Supp. B.1) |
| Latent | VAE (SD v1.5 VAE, frozen) 8× downsampling → **(4, 32, 128)** latent (Supp. B.1) | height = mel axis, width = time axis |
| Text encoder | CLIP ViT-L/14 text encoder as the default for the released models (Sec. 4.2 "Both Auffusion and Auffusion-Full use CLIP as default text encoder"); frozen (Sec. 4.1) | Sec. 3.5 describes CLAP+FlanT5 ensembles used in the *ablation*; those are not the released default |
| LDM / U-Net | Pretrained Stable Diffusion v1.5 U-Net (860 M), fine-tuned; VAE frozen (Sec. 4.1, Supp. B.2) | Diffusers SD1.5 architecture |
| Cross-attention conditioning | Sec. 3.4, Eqs. 6–8: `Q = W_q^{(i)} ϑ_i(z_t)`, `K = W_k^{(i)} τ`, `V = W_v^{(i)} τ`, `Attention = softmax(QKᵀ/√d) V`; "cross-attention mechanism between conditional embedding sequence and latent vectors in the U-Net backbone" | Exactly PLADIS Eq. 12. Fig. 2 visualizes per-token cross-attention maps over the spectrogram |
| CFG | Sec. 3.6, Eq. 9: `ε̂ = (1+w) ε(z_t,t,τ) − w ε(z_t,t)`; 10 % condition dropout at training; "when all embeddings are dropped, corresponds to unconditional training" (Sec. 3.5) | Eq. 9 is the same affine form as `ε_u + w'(ε_c − ε_u)` with `w' = 1+w`; diffusers implements the latter with `guidance_scale = w'` |
| Sampling defaults | 100 steps, guidance 7.5 (Supp. C, Table 5: 100 steps / 7.5 chosen as balanced) | README/`inference.sh` agree |
| Vocoder | Own HiFi-GAN trained on 256-mel spectrograms (Supp. B.1) | `vocoder/` in checkpoint |

### 5.2 What Q, K, V are in Auffusion's text cross-attention [PAPER + CODE]

- **Q**: a linear projection of the U-Net's intermediate feature map at layer *i*, flattened over the (mel-frequency × time) grid. At the top level each query token is one latent cell = an 8 × 8 patch of the 256 × 1024 mel image = **8 mel bins × 8 frames (80 ms)**. Query lengths per level: 4096, 1024, 256, 64 (32×128, 16×64, 8×32, 4×16).
- **K, V**: linear projections of the 77 × 768 conditioning sequence τ (CLIP token states after Auffusion's adapter), shared by all 16 cross-attention layers in the same forward pass (`encoder_hidden_states` is passed once to the U-Net and broadcast to every block — **[CODE]** diffusers `UNet2DConditionModel.forward` → blocks → `Transformer2DModel.forward` → `BasicTransformerBlock.forward(encoder_hidden_states=...)`).
- The paper's Fig. 2 confirms the token axis: columns `<startoftext>, dogs, are, barking, ..., <endoftext>` — i.e. only real tokens carry distinct maps (consistent with the zero-padding found in code, §6.2).

### 5.3 Self vs cross modules, sequence dimensions, canonical checkpoint

- Self-attention: `attn1` in each of the 16 transformer blocks (queries, keys, values all from the latent feature map). Cross-attention: `attn2` (keys/values from τ). **[CODE]** (§6.4).
- All cross-attention layers receive the **same** τ (same 77 × 768 tensor); they differ only in `W_q^{(i)}, W_k^{(i)}, W_v^{(i)}` and in query resolution.
- Different text encoders change `condition_dim` but the adapter maps everything to `cross_attention_dim = 768`; the sequence length is forced to `tokenizer_model_max_length = 77` by pad/truncate; with several encoders their sequences are **concatenated along the token axis** (`torch.cat(encoder_hidden_states_list, dim=1)`), so S would be 77·n_encoders. The released `auffusion/auffusion` has one encoder → **S = 77**. **[CODE]** `AuffusionPipeline._encode_prompt`, `condition_config.json`.
- **Canonical baseline [CODE/INFERENCE]**: `auffusion/auffusion` (CLIP-L/14 + adapter, trained on AudioCaps) — it is the default of `inference.py` (`--pretrained_model_name_or_path` default and `inference.sh`), the README quickstart, and `notebooks/text_to_audio.ipynb`. `auffusion/auffusion-full` (same architecture, trained on the full 2K-h set; better numbers in Table 1) is a drop-in alternative via the same loader. Recommendation: start with `auffusion/auffusion`; make the checkpoint a config field.

---

## 6. Auffusion Code Analysis

All paths relative to `third_party/Auffusion/`. Diffusers internals are from the 0.18.2 wheel (`diffusers/models/attention_processor.py`, `attention.py`, `transformer_2d.py`, `unet_2d_condition.py`, `schedulers/scheduling_pndm.py`).

### 6.1 Inference path, prompt → waveform [CODE]

```
inference.py main()
 ├─ AuffusionPipeline.from_pretrained("auffusion/auffusion")           # auffusion_pipeline.py
 │    ├─ snapshot_download → local dir
 │    ├─ AutoencoderKL.from_pretrained(dir, subfolder="vae")             # diffusers
 │    ├─ UNet2DConditionModel.from_pretrained(dir, subfolder="unet")    # diffusers, SD1.5 config
 │    ├─ PNDMScheduler.from_pretrained(dir, subfolder="scheduler")      # diffusers
 │    ├─ Generator.from_pretrained(dir, subfolder="vocoder")            # Auffusion HiFi-GAN (in auffusion_pipeline.py)
 │    ├─ for each item in condition_config.json (one item):
 │    │     AutoTokenizer.from_pretrained(dir/text_encoder_0)           # CLIPTokenizer
 │    │     CLIPTextModel.from_pretrained(dir/text_encoder_0)           # transformers
 │    │     ConditionAdapter.from_pretrained(dir/condition_adapter_0)   # Linear(768→768)+LayerNorm(768)
 │    └─ pipeline.to(device, dtype=torch.float16)                       # default dtype fp16 for ALL modules
 ├─ pipeline.to(device, weight_dtype); optional enable_xformers_memory_efficient_attention()
 ├─ generator = torch.Generator(device).manual_seed(args.seed)          # seed 42 default
 └─ with torch.autocast("cuda"):
      output = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5, generator, width=1024, height=256)

AuffusionPipeline.__call__                                              # auffusion_pipeline.py
 ├─ check_inputs; batch_size; do_classifier_free_guidance = guidance_scale > 1.0
 ├─ prompt_embeds = _encode_prompt(prompt, ..., do_classifier_free_guidance, negative_prompt=None)
 │    ├─ input_ids = tokenizer(prompt, return_tensors="pt").input_ids   # NO padding, NO truncation → [1, n_tok]
 │    ├─ cond_embs = text_encoder(input_ids).last_hidden_state          # [1, n_tok, 768]
 │    ├─ zero-pad token axis to 77 (or slice to 77)                     # F.pad(..., value=0)
 │    ├─ cond_embs = adapter(cond_embs)                                  # Linear + LayerNorm, applied to padded rows too
 │    ├─ prompt_embeds [1, 77, 768] → .to(unet.dtype)
 │    ├─ negative_prompt_embeds = torch.zeros_like(prompt_embeds)        # null condition = all zeros (negative_prompt None)
 │    └─ prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])   # [2, 77, 768]: row 0 = uncond, row 1 = cond
 ├─ scheduler.set_timesteps(100); timesteps
 ├─ latents = randn_tensor((1, 4, 32, 128), generator) * init_noise_sigma    # only use of the generator
 ├─ for t in timesteps:
 │    latent_model_input = torch.cat([latents]*2)                        # [2, 4, 32, 128]
 │    latent_model_input = scheduler.scale_model_input(...)              # PNDM: identity
 │    noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=None)[0]
 │    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
 │    noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text − noise_pred_uncond)
 │    latents = scheduler.step(noise_pred, t, latents)[0]               # PNDM.step: deterministic, no generator
 ├─ image = vae.decode(latents / 0.18215)[0]                            # [1, 3, 256, 1024]
 ├─ image = image_processor.postprocess(image, output_type="pt")        # (x/2+0.5).clamp(0,1)
 ├─ spectrogram = denormalize_spectrogram(img)                          # flip mel axis, take channel 0, log-scale → [256, 1024]
 ├─ audio = vocoder.inference(spectrogram, lengths=160000)[0]           # int16 waveform, 10 s @ 16 kHz
 └─ PipelineOutput(images=PIL(viridis), spectrograms, audios)
```

Inside the U-Net (diffusers 0.18.2):

```
UNet2DConditionModel.forward(sample, t, encoder_hidden_states, cross_attention_kwargs, encoder_attention_mask=None)
 └─ CrossAttnDownBlock2D / UNetMidBlock2DCrossAttn / CrossAttnUpBlock2D
      └─ Transformer2DModel.forward: GroupNorm → proj_in (conv 1×1) → permute/reshape to (B, H·W, C)
           └─ BasicTransformerBlock.forward(hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs)
                ├─ attn1(norm1(h), encoder_hidden_states=None)            # SELF-attention  (only_cross_attention=False)
                ├─ attn2(norm2(h), encoder_hidden_states=τ, attention_mask=encoder_attention_mask)   # CROSS-attention
                └─ ff(norm3(h))
                     Attention.forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
                        → self.processor(self, hidden_states, encoder_hidden_states=..., attention_mask=..., **kwargs)
```

### 6.2 Facts with direct consequences for CAST [CODE]

1. **Null condition is literally zeros.** `negative_prompt_embeds = torch.zeros_like(prompt_embeds)` (post-adapter). Since `to_k`/`to_v` have `bias=False` (verified by instantiating the UNet from the checkpoint's `unet/config.json`: `to_k.bias is None` for all 32 attention modules), the unconditional row has `K = V = 0`.
2. **Padding is applied before the adapter.** Rows `n_tok … 76` of `cond_embs` are zero *input* to `adapter = LayerNorm(Linear(·))`, so every pad row becomes the same constant vector `LayerNorm(b_proj)`; all pad keys are identical.
3. **No cross-attention mask** is ever passed (`encoder_attention_mask` is not in the call) → `attention_mask is None` inside `attn2` processors.
4. **No attention-probability dropout**: `UNet` dropout 0.0; `Attention.to_out[1] = Dropout(p=0.0)` is applied *after* the output projection; SDPA is called with `dropout_p=0.0`. The pipeline runs under `@torch.no_grad()` and modules are in eval mode.
5. **`cross_attention_kwargs` reach every processor, including `attn1`.** Default 0.18.2 processors have the fixed signature `(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None)` with no `**kwargs`, so passing custom keys (e.g. `{"lam": 2.0}`) through the pipeline would raise `TypeError` in the untouched self-attention modules. CAST must configure its processors by attribute, not by `cross_attention_kwargs`.
6. **Determinism.** The only stochastic operation is `randn_tensor` for the initial latent (seeded `torch.Generator`). PNDM `step` takes no generator; `scale_model_input` returns `sample` unchanged. Hence same seed + same processors ⇒ same latents, up to kernel-level non-determinism.
7. **Precision.** `AuffusionPipeline.from_pretrained` casts *everything* (U-Net, VAE, CLIP, adapter, vocoder) to fp16 by default (`dtype=torch.float16` argument, `DiffusionPipeline.to(device, dtype)`), and `inference.py`/the notebook wrap the call in `torch.autocast("cuda")`; `upcast_attention=False` in the U-Net config.
8. **Processor type at runtime.** `Attention.__init__` picks `AttnProcessor2_0()` (torch SDPA, 4-D layout) whenever `torch.nn.functional.scaled_dot_product_attention` exists (torch ≥ 2.0), else `AttnProcessor` (3-D `baddbmm` + `softmax`). `inference.sh` additionally calls `enable_xformers_memory_efficient_attention()` → `Attention.set_use_memory_efficient_attention_xformers(True)` → `XFormersAttnProcessor` (3-D layout, `xformers.ops.memory_efficient_attention`). All three expose no probabilities.
9. **Long prompts.** The tokenizer is called without truncation; prompts > 77 tokens would overflow CLIP's 77 position embeddings before the slicing to 77 happens. Keep sanity prompts short (irrelevant to CAST but a baseline fragility).
10. **Batch layout.** Row order in every tensor the processor sees is `[uncond, cond]` (`torch.cat([negative, positive])`). Auffusion's own prompt2prompt code relies on this (`AttentionControl.__call__`: `attn[h // 2:]` = conditional half).

### 6.3 Attention backend summary [CODE]

| Backend | When active | Convention | Probabilities exposed? |
|---|---|---|---|
| `AttnProcessor2_0` (`torch.nn.functional.scaled_dot_product_attention`) | default with torch ≥ 2.0 (the pinned 2.0.1 qualifies) | 4-D `(B, H, N, d)` | No |
| `XFormersAttnProcessor` | `inference.py --enable_xformers_memory_efficient_attention` (set in `inference.sh`) | 3-D `(B·H, N, d)` | No |
| `AttnProcessor` (`baddbmm` + `softmax`) | torch < 2.0, or `unet.set_default_attn_processor()` | 3-D | Yes (`attn.get_attention_scores`) |
| `P2PCrossAttnProcessor` (Auffusion's own `prompt2prompt/pipeline_prompt2prompt.py`) | word-swap / reweight notebooks | 3-D via `attn.head_to_batch_dim` + `attn.get_attention_scores` | Yes — **precedent that processor injection works on this exact U-Net** |

### 6.4 Attention module census (instantiated from the checkpoint's `unet/config.json` with diffusers 0.33.1; the 0.18.2 wheel cannot import against the base env's huggingface_hub, but the SD1.5 topology is identical) [CODE/MEASURED]

| Group | Modules (name prefix) | attn2 count | query tokens (256×1024 input) | C | heads | d |
|---|---|---|---|---|---|---|
| down_blocks.0 | `down_blocks.0.attentions.{0,1}.transformer_blocks.0` | 2 | 4096 | 320 | 8 | 40 |
| down_blocks.1 | `down_blocks.1.attentions.{0,1}...` | 2 | 1024 | 640 | 8 | 80 |
| down_blocks.2 | `down_blocks.2.attentions.{0,1}...` | 2 | 256 | 1280 | 8 | 160 |
| mid_block | `mid_block.attentions.0.transformer_blocks.0` | 1 | 64 | 1280 | 8 | 160 |
| up_blocks.1 | `up_blocks.1.attentions.{0,1,2}...` | 3 | 256 | 1280 | 8 | 160 |
| up_blocks.2 | `up_blocks.2.attentions.{0,1,2}...` | 3 | 1024 | 640 | 8 | 80 |
| up_blocks.3 | `up_blocks.3.attentions.{0,1,2}...` | 3 | 4096 | 320 | 8 | 40 |
| **Total** | | **16 attn2 + 16 attn1 = 32 processors** | | | | |

For every `attn2`: `to_q: C→C`, `to_k, to_v: 768→C`, all `bias=False`; `to_out[0]: C→C` with bias; `norm_cross=None`, `group_norm=None`, `spatial_norm=None`, `residual_connection=False`, `rescale_output_factor=1.0`, `upcast_attention=False`, `scale = d^-0.5`. For every `attn1`: `to_k.in_features = C ≠ 768`. (diffusers 0.18.2 `Attention` has **no** `is_cross_attention` attribute; 0.33.1 does.)

### 6.5 Can PLADIS be injected without touching Auffusion source? — **Yes.** [CODE/INFERENCE]

- `UNet2DConditionModel.attn_processors` (dict `"<module path>.processor" → processor`) and `Attention.set_processor(processor)` are public in 0.18.2. Per-module `set_processor` is the right primitive: `unet.set_attn_processor(dict)` demands a dict covering **all 32** modules (it raises otherwise) and would force us to also "set" `attn1`, which we avoid by addressing only the 16 `attn2` modules.
- The pipeline object exposes `pipeline.unet`; the swap is performed after `from_pretrained` and before `__call__`. No subclassing of `AuffusionPipeline` is necessary; `AuffusionPipeline.__call__` is used verbatim, which is exactly what keeps the vanilla path reproducible.
- `attention_mask` will be `None`, `encoder_hidden_states` will be the `[2, 77, 768]` τ, `hidden_states` 3-D. The processor must mirror `AttnProcessor2_0` (0.18.2 signature: no `*args/**kwargs`, `temb` present) and must not rely on 0.33.1-only attributes (`norm_q`, `norm_k`, `is_cross_attention`).

---

## 7. PLADIS → Auffusion Mapping

```
PLADIS / SDXL (T2I)                                   Auffusion (T2A)
─────────────────────────────────────────────         ──────────────────────────────────────────────────────────
noisy image latent  z_t ∈ R^{4×128×128}        →      noisy mel-latent  z_t ∈ R^{4×32×128}  (mel bins/8 × frames/8)
U-Net feature map, flattened → Q (N tokens)    →      same (N = 4096 / 1024 / 256 / 64)
CLIP-L + OpenCLIP-G text states [77, 2048]     →      CLIP-L/14 states, zero-padded, adapter → τ [77, 768]
K = W_k τ,  V = W_v τ                          →      K = W_k τ,  V = W_v τ   (bias-free Linear 768→C)
Softmax(QKᵀ/√d) V in each attn2                →      identical module (diffusers Attention, attn2)
1.5-Entmax(QKᵀ/√d) V                           →      identical operator, same axis (keys, dim=-1)
At + λ(At_α − At), all cross-attn layers       →      same, 16 attn2 modules
```

**Validity [INFERENCE].** Auffusion Eqs. 6–8 are PLADIS Eq. 12 verbatim; the implementation is the same `diffusers.models.attention_processor.Attention` class the PLADIS processor already targets. The row-wise operator (softmax/entmax over the 77 keys of one query) is agnostic to what the query represents. The mapping is therefore mathematically and structurally valid. What differs is only (a) the distribution of logits (unknown until measured), (b) the identical-pad-key structure, (c) the zero null condition. None affects correctness of the operator; all affect interpretation (§3.4, §10).

**Exact intervention point.** The line in the processor that would call `F.scaled_dot_product_attention(query, key, value, ...)` (0.18.2 `AttnProcessor2_0`) or `attn.get_attention_scores(query, key, attention_mask)` (0.18.2 `AttnProcessor`). CAST replaces it with: `logits = (q.float() @ k.float().transpose(-1,-2)) * attn.scale` → `P = mix(softmax(logits), entmax15(logits), λ)` → `out = P.to(v.dtype) @ v`. Everything before (`to_q/to_k/to_v`, head reshape) and after (`reshape`, `to_out`, residual in the block) is kept identical.

**Answers to the ten questions.**

| # | Question | Answer |
|---|---|---|
| 1 | Which module to intercept? | The `processor` of each of the 16 `*.transformer_blocks.0.attn2` `Attention` modules of `pipeline.unet`. Not `attn1`, not `Transformer2DModel`, not `BasicTransformerBlock`. |
| 2 | How to distinguish self from cross reliably? | Three independent signals; use all: (a) name: `attn_processors` key ends with `"attn2.processor"` (same heuristic Auffusion's P2P code uses with `attn1.processor`); (b) structure: `module.to_k.in_features == unet.config.cross_attention_dim (768)` **and** `!= module.to_q.in_features` — true for all 16 attn2, false for all 16 attn1 (C ∈ {320, 640, 1280} ≠ 768); (c) runtime: `encoder_hidden_states is not None` inside the processor, with an `assert` (the sparse path must never run on self-attention). Do **not** use PLADIS's `"attn2" in name and "to" not in name` substring test; (a)+(b) are exact. |
| 3 | Before or after attention dropout? | There is no probability dropout in this model/pipeline (§6.2 item 4). The intervention happens on probabilities, before `@ V`; the only dropout (`to_out[1]`, p=0, eval) is after the projection and is untouched. |
| 4 | Same Q, K, V for both branches? | **Yes** (PLADIS computes both maps from one `attn_weight`). Compute logits once; derive `P_dense` and `P_sparse` from the same tensor. This also guarantees `λ=0` reproduces dense attention exactly. |
| 5 | Attention masks? | None are passed in Auffusion. v1: `assert attention_mask is None` in the CAST processor (fail loudly rather than silently mis-mask). If ever needed: add the additive bias to the fp32 logits *before both* normalizations, using a finite large-negative value (e.g. −1e4). **[MEASURED]** entmax15 returned no NaN with `-inf`, `-1e4` (fp32) or `-65504` (fp16) masked entries and placed zero mass on masked keys, but finite bias is the safer contract. |
| 6 | Mixed precision? | Weights/activations are fp16 (pipeline default) possibly under autocast. Upcast `q, k` to fp32 for the logits, run softmax and entmax in fp32, cast `P` to `v.dtype` for the matmul, return in the input dtype (same pattern as `Attention.get_attention_scores` with `upcast_attention=True` and as PLADIS's Flux processor). |
| 7 | Does Entmax need fp32? | Strongly recommended. **[MEASURED]** (random logits, std 2.5, S=77) fp16-native `entmax15` vs fp32: max abs error 6.6e-3 and row sums in [0.990, 1.014]; fp32: row-sum error 1.6e-6. Cost of fp32 at the largest layer 2.65 ms vs 1.63 ms — immaterial. Under `torch.autocast`, `softmax`, `cumsum`, `pow` are auto-promoted to fp32 but `sort`/`sqrt` are not, so explicit `.float()` is also what makes results independent of whether the caller used autocast. |
| 8 | Disables SDPA/xformers? | For the 16 cross-attention modules only — yes, by necessity (those kernels never materialize `P`). Self-attention (the expensive 4096×4096 maps) keeps its stock kernel. Cross-attention has S=77, so the materialized `(2, 8, 4096, 77)` map is 5 M entries (20 MB fp32). |
| 9 | Overhead? | **[MEASURED, RTX A5000]** per layer, `(2, 8, N, 77)`: entmax15 fp32 2.65 / 0.69 / 0.35 / 0.34 ms for N = 4096 / 1024 / 256 / 64 (softmax 0.04 / 0.01 / 0.007 / 0.007 ms; SDPA full attention at N=4096 0.044 ms vs manual matmul+softmax 0.16 ms). Summed over 16 layers ≈ 19 ms per U-Net forward (CFG batch 2) → ≈ +1.9 s per 100-step sample, plus ~2 ms for the manual dense path. Expect roughly +20–35 % wall-clock relative to the SDPA vanilla (to be measured in §13 T9). Transient memory < 0.5 GB. PLADIS paper Table 5 reports +22 % on SDXL. The entmax cost is dominated by the full sort over 77 keys per row; entmax's `k=` partial-sort option is available if needed. |
| 10 | Compatible with projection/residual path? | Yes. Output is `P_mix @ V` with the same shape; rows of `P_mix` sum to 1 (λ·1 + (1−λ)·1) so the output stays an affine combination of value vectors, just with possibly negative coefficients for λ>1 (**[MEASURED]** min weight −0.12 at λ=2). `to_out`, the residual add in `BasicTransformerBlock`, and `rescale_output_factor=1` are unchanged. Nothing downstream assumes non-negativity. |

---

## 8. Proposed CAST Architecture

Minimal, single-pipeline, mode-switched. Nothing in `third_party/` is modified; `third_party/Auffusion` is put on `sys.path` by the integration module.

```
cast/
├── __init__.py
├── config.py                  # CastConfig dataclass + mode→(alpha, lambda) resolution + validation
├── attention/
│   ├── __init__.py
│   ├── sparse_ops.py          # alpha_entmax(logits, alpha) [1.0→softmax, 1.5→entmax15, 2.0→sparsemax, else entmax_bisect],
│   │                          # dense_softmax(logits), mix_probs(P_dense, P_sparse, lam)  — all fp32, pure functions
│   └── cast_processor.py      # CastCrossAttnProcessor (diffusers-0.18.2 processor protocol) + optional stats hook
├── integration/
│   ├── __init__.py
│   └── auffusion.py           # import_auffusion() (sys.path), load_pipeline(ckpt, dtype, device),
│                              # select_cross_attention(unet) -> {name: Attention} with structural asserts,
│                              # apply_cast(unet, cfg) -> originals, restore(unet, originals), context manager
├── pipeline.py                # run_one(prompt, seed, cfg) -> wav/spec/latents; the ONLY entry point for all modes
└── tests/
    ├── test_sparse_ops.py     # §13 T4, T5
    ├── test_processor.py      # §13 T1–T3, T7 (random tensors + a from_config UNet; no checkpoint needed)
    └── test_pipeline_smoke.py # §13 T6, T8–T10 (needs checkpoint; marked slow)
experiments/
├── configs/first_milestone.yaml   # prompt, seed=42, steps=100, cfg=7.5, size 256×1024, dtype fp16, modes [vanilla, sparse, pladis]
└── run_first_milestone.sh         # loops modes with the same config; writes outputs/first_milestone/<mode>/...
outputs/                            # generated wav/png/pt + run manifest (git hash, versions, config, timing)
```

Design rules [INFERENCE]:

- **One processor class** for all three modes. `mode="vanilla"` performs **no processor swap at all** (stock `AttnProcessor2_0`/xformers remain), guaranteeing bit-identical reproduction of the original pipeline. A separate `mode="vanilla_manual"` (CAST processor with λ=0) exists only for parity testing, never as an experimental arm.
- `mode="sparse"` ⇒ α=1.5, λ=1.0; `mode="pladis"` ⇒ α=1.5, λ=2.0. Explicit `alpha`/`lam` overrides are allowed for later ablations but the three names are fixed.
- The processor is stateless except for its config and an optional stats accumulator; parameters are set on the object, never via `cross_attention_kwargs` (§6.2 item 5).
- `apply_cast` returns the original processor objects and `restore` puts them back; a context manager wraps a run so that a crash cannot leave the U-Net modified.
- No new abstraction beyond these; no registry, no plugin system, no subclassed pipeline.

The evaluated layout in the task brief (`attention/entmax_attention.py`, `attention/pladis_processor.py`, `integration/auffusion.py`, `config.py`) is appropriate; the names above merely merge "entmax" and "mixing" into one ops module and name the processor by project rather than by method, because the same class serves all three modes.

---

## 9. Proposed Attention Implementation (pseudocode, not production code)

```python
# cast/attention/sparse_ops.py
def dense_softmax(logits_f32):                      # logits: (B, H, N, S) float32
    return torch.softmax(logits_f32, dim=-1)

def alpha_entmax(logits_f32, alpha):
    if alpha == 1.0:  return torch.softmax(logits_f32, dim=-1)        # entmax_bisect(alpha=1.0) is NOT softmax (measured); dispatch explicitly
    if alpha == 1.5:  return entmax.entmax15(logits_f32, dim=-1)       # exact sort-based algorithm (what PLADIS uses)
    if alpha == 2.0:  return entmax.sparsemax(logits_f32, dim=-1)
    return entmax.entmax_bisect(logits_f32, alpha=alpha, dim=-1)       # general alpha, iterative (n_iter=50)

def mix_probs(p_dense, p_sparse, lam):
    if lam == 0.0: return p_dense                                       # exact vanilla (no arithmetic)
    if lam == 1.0: return p_sparse                                      # exact sparse
    return p_dense + lam * (p_sparse - p_dense)                         # == lam*p_sparse + (1-lam)*p_dense (PLADIS code form)


# cast/attention/cast_processor.py
class CastCrossAttnProcessor:
    """diffusers-0.18.2 AttentionProcessor for text cross-attention (attn2) only."""
    def __init__(self, alpha=1.5, lam=2.0, apply_to_uncond=True, stats=None):
        self.alpha, self.lam, self.apply_to_uncond, self.stats = alpha, lam, apply_to_uncond, stats
        self.calls = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # --- identical prologue to diffusers 0.18.2 AttnProcessor2_0 ---------------------------------
        assert encoder_hidden_states is not None, "CAST processor installed on a self-attention module"
        assert attention_mask is None,            "Auffusion passes no cross-attention mask; masking unsupported in v1"
        assert attn.spatial_norm is None and attn.group_norm is None and attn.norm_cross is None   # true for SD1.5 attn2
        B, N, C = hidden_states.shape                     # hidden_states is 3-D here (Transformer2DModel flattens)
        S = encoder_hidden_states.shape[1]                # 77
        q = attn.to_q(hidden_states)                      # (B, N, C)
        k = attn.to_k(encoder_hidden_states)              # (B, S, C)
        v = attn.to_v(encoder_hidden_states)              # (B, S, C)
        H, d = attn.heads, C // attn.heads
        q = q.view(B, N, H, d).transpose(1, 2)            # (B, H, N, d)
        k = k.view(B, S, H, d).transpose(1, 2)            # (B, H, S, d)
        v = v.view(B, S, H, d).transpose(1, 2)            # (B, H, S, d)

        # --- CAST core: replaces F.scaled_dot_product_attention --------------------------------------
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * attn.scale     # (B, H, N, S) fp32, attn.scale = d**-0.5
        p_dense = dense_softmax(logits)
        if self.lam == 0.0:
            p = p_dense
        else:
            p_sparse = alpha_entmax(logits, self.alpha)
            p = mix_probs(p_dense, p_sparse, self.lam)
            if not self.apply_to_uncond:                  # optional: rows [0:B//2] are the CFG null-condition rows
                p = torch.cat([p_dense[: B // 2], p[B // 2 :]], dim=0)
        if self.stats is not None: self.stats.record(attn, logits, p_dense, p_sparse if self.lam else None)
        out = torch.matmul(p.to(v.dtype), v)              # (B, H, N, d), back to model dtype for the matmul

        # --- identical epilogue to AttnProcessor2_0 -------------------------------------------------
        out = out.transpose(1, 2).reshape(B, N, H * d).to(q.dtype)
        out = attn.to_out[0](out)                         # Linear C→C (bias)
        out = attn.to_out[1](out)                         # Dropout(p=0.0)
        self.calls += 1
        return out                                        # residual add happens in BasicTransformerBlock


# cast/integration/auffusion.py
def select_cross_attention(unet):
    sel = {}
    for name, proc in unet.attn_processors.items():               # keys like 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor'
        mod = unet.get_submodule(name[: -len(".processor")])
        is_cross_by_name  = name.endswith("attn2.processor")
        is_cross_by_shape = (mod.to_k.in_features == unet.config.cross_attention_dim) and (mod.to_k.in_features != mod.to_q.in_features)
        assert is_cross_by_name == is_cross_by_shape, name           # both signals must agree
        if is_cross_by_name: sel[name] = mod
    assert len(sel) == 16 and len(unet.attn_processors) == 32
    return sel

def apply_cast(unet, cfg):
    if cfg.mode == "vanilla": return {}                           # no swap: stock processors stay
    originals = {}
    for name, mod in select_cross_attention(unet).items():
        originals[name] = mod.processor
        mod.set_processor(CastCrossAttnProcessor(alpha=cfg.alpha, lam=cfg.lam, apply_to_uncond=cfg.apply_to_uncond))
    return originals

def restore(unet, originals):
    for name, proc in originals.items():
        unet.get_submodule(name[: -len(".processor")]).set_processor(proc)


# cast/pipeline.py
def run_one(prompt, seed, cfg):                                   # cfg: checkpoint, mode, steps=100, guidance=7.5, H=256, W=1024, dtype=fp16, autocast=True
    pipe = load_pipeline(cfg.checkpoint, cfg.dtype, cfg.device)   # AuffusionPipeline.from_pretrained; NO xformers unless cfg says so
    originals = apply_cast(pipe.unet, cfg)
    try:
        g = torch.Generator(device=cfg.device).manual_seed(seed)
        with maybe_autocast(cfg):
            out = pipe(prompt=prompt, num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                       generator=g, height=cfg.height, width=cfg.width)   # AuffusionPipeline.__call__ untouched
    finally:
        restore(pipe.unet, originals)
    save(out.audios[0], out.spectrograms[0], manifest(cfg, seed, versions, timing))
```

Notes on fidelity to PLADIS [INFERENCE]: the core three lines are exactly `sparse_scaled_dot_product_attention(use_sparse=True)` with (i) fp32 upcast added (as in PLADIS's Flux variant), (ii) `temper` dropped (fixed to 1 in PLADIS anyway), (iii) α dispatch instead of a `method` string, (iv) the `λ=0`/`λ=1` fast paths added so that those modes are *exactly* dense/sparse rather than `0·S + 1·D` in floating point (though **[MEASURED]** even the arithmetic form is exact for λ∈{0,1} in fp16 and fp32).

---

## 10. CFG Handling

**How CFG is implemented [CODE].** `AuffusionPipeline` does a **single batched U-Net forward** per step: `latent_model_input = torch.cat([latents]*2)`, `encoder_hidden_states = cat([zeros, τ])`, then `noise_pred.chunk(2)` and `ε = ε_u + w(ε_c − ε_u)` with `w = guidance_scale = 7.5`. There is no separate unconditional forward and no `guidance_rescale` by default. Consequently **every cross-attention processor call sees a batch whose first half is the unconditional branch and second half the conditional branch**, and PLADIS (which replaces the module, not the call) is executed on both — exactly as in the PLADIS SDXL code.

**What happens on the unconditional rows in Auffusion [CODE + MEASURED].**

- τ_uncond = 0 ⇒ `K_u = to_k(0) = 0` and `V_u = to_v(0) = 0` (bias-free projections, verified for all 16 attn2).
- Logits `Q K_uᵀ/√d = 0` for every query and key ⇒ `softmax(0) = 1/77` uniform; `entmax_1.5(0) = 1/77` uniform (constant vectors are a fixed point of every α-entmax). **[MEASURED]** max |entmax15 − softmax| on all-zero rows = 9.3e-10 (fp32), 0.0 (fp16).
- Therefore `P_sparse − P_dense = 0` and `P_mix = P_dense` for any λ — and, independently, `P @ V_u = 0` for *any* `P`, so the uncond cross-attention output is the constant `to_out.bias` regardless of the processor.

**Conclusion.** Applying CAST to the unconditional branch is a **mathematically exact no-op** for the default null condition (negative_prompt=None). The PLADIS-faithful choice ("modify the module wherever cross-attention executes") and the "conditional-only" choice coincide. The difference `ε_c^{CAST} − ε_u` is therefore the entire effect, scaled by `w = 7.5` in the CFG combination — a larger multiplier than PLADIS's SDXL experiments used (3–5), which is a reason to expect λ=2 to be *stronger* here, not weaker (§16).

**When the two choices would diverge [INFERENCE].** Only if a textual `negative_prompt` is used (then τ_uncond ≠ 0 and sparse ≠ dense on that branch), or if `guidance_scale ≤ 1` (no CFG batch at all; the processor then sees only conditional rows). Neither is part of the planned experiment.

**Alternatives and recommendation.**

| Option | Behaviour | Pros | Cons |
|---|---|---|---|
| A. Whole batch (PLADIS-faithful) — **recommended default** | sparse mix on all rows | Zero branch logic; identical to reference code; provably equal to B for the planned runs | With negative prompts, sharpens the "away-from" direction too (semantics unclear) |
| B. Conditional rows only (`apply_to_uncond=False`, slice `[B//2:]` like Auffusion's P2P code) | dense on rows `[0:B//2]` | Keeps ε_u exactly vanilla even with negative prompts | Needs the batch-layout assumption `[uncond, cond]`; fragile if `num_images_per_prompt>1` or no CFG |
| C. Runtime detection (`K.abs().max()==0` ⇒ skip) | automatic | Cheap and exact for zero null condition | Hides the semantics behind a numerical test; unnecessary given A ≡ B |

Implement A with a config flag for B (one `torch.cat` line) and, in the validation suite, a hook that asserts the uncond-half cross-attention outputs are identical across modes (§13 T7). Do not implement C.

---

## 11. Dependency and Compatibility Analysis

| Item | Auffusion needs | PLADIS needs | CAST decision [INFERENCE] |
|---|---|---|---|
| diffusers | **0.18.2** (`randn_tensor` from `diffusers.utils`, `LoraLoaderMixin`, `is_accelerate_version` — **[MEASURED]** import fails on 0.30.2 with `ImportError: cannot import name 'randn_tensor'`); checkpoint configs carry `_diffusers_version 0.18.2` | 0.33.1 (`Attention.norm_q/norm_k`, `deprecate("scale")`, SDXL/Flux pipelines) | Pin **0.18.2**. Do **not** import anything from `third_party/PLADIS` at runtime; re-implement the ~40-line processor against the 0.18.2 API (§9). PLADIS code remains the reference for review/diff only. |
| huggingface_hub | 0.16.4 pinned (0.18.2 imports `cached_download`, removed in hub ≥ 0.26) | ≥ 0.27 | 0.16.4 (as pinned). Note: this also constrains `transformers` to a release compatible with hub 0.16.x (≈ ≤ 4.3x; the checkpoint's CLIP config was written by `transformers 4.31.0.dev0`). |
| torch | 2.0.1 (+ matching CUDA); SDPA present ⇒ `AttnProcessor2_0` default | ≥ 2.0 | 2.0.1 with cu118 wheels (driver 570 supports CUDA ≤ 12.8, so 11.8 runtime works). A newer torch 2.x would also run diffusers 0.18.2 but is untested with it; keep the pin for the baseline. |
| xformers | 0.0.20 (optional flag; enabled in `inference.sh`) | not needed | **Do not install** (0.0.20 is bound to torch 2.0.1 and not needed by SDPA). Define vanilla with the stock `AttnProcessor2_0`. If xformers parity is ever required, it is an A/B on the vanilla arm, not a CAST change. |
| entmax | — | `pip install entmax` (1.3) | 1.3; pure Python, `torch>=1.3`; no build. |
| Python | 3.8–3.11 for torch 2.0.1 wheels | — | 3.10 (new conda env `cast`). |
| Other Auffusion deps | `datasets` (inference.py), `soundfile 0.12.1`, `librosa 0.9.2`, `scipy`, `matplotlib`, `accelerate` | — | Install per `requirements.txt` minus xformers. |
| GPU / driver | — | — | RTX A5000 24 GB is ample (SD1.5 U-Net fp16, 32×128 latent). Base-env torch (cu130) **cannot** use this driver; the `acg` env (torch 2.7.1+cu128) works but has diffusers 0.30.2 — unusable for Auffusion. ⇒ a fresh env is unavoidable. |
| Checkpoint access | HF Hub public repo `auffusion/auffusion` | — | Anonymous HTTP works; the `huggingface_hub` client on this machine currently fails with a stale/invalid cached OAuth token (`~/.cache/huggingface/token`) — set `HF_TOKEN` unset / remove the token file or log in before `snapshot_download`. |

Conflict summary: PLADIS and Auffusion **cannot coexist in one environment** (diffusers 0.33.1 vs 0.18.2, hub ≥0.27 vs 0.16.4). This is harmless because CAST only needs the two 30-line functions from PLADIS, re-typed against 0.18.2.

---

## 12. Risks and Mitigations

| Risk | Severity | Cause | Mitigation |
|---|---|---|---|
| Environment cannot be built / GPU unusable | **HIGH** | Base torch cu130 vs driver CUDA 12.8; no env has diffusers 0.18.2; hub pin 0.16.4 constrains transformers | Create `cast` conda env (py3.10, torch 2.0.1 cu118, diffusers 0.18.2, hub 0.16.4, transformers ≤4.3x, entmax 1.3). Gate everything on an import + `cuda.is_available()` + one vanilla run (§14 step 0). |
| PLADIS code assumes diffusers 0.33.1 / SDXL classes | **MEDIUM** | `SparseAttnProcessor` uses `norm_q/norm_k`, `deprecate`, SDXL pipeline | Never import PLADIS; reimplement against the 0.18.2 `AttnProcessor2_0` prologue/epilogue (§9), diff-review against PLADIS's function. |
| AttentionProcessor API mismatch (signature, `**kwargs`) | **MEDIUM** | 0.18.2 processors have fixed signatures; custom `cross_attention_kwargs` would break untouched `attn1` | Configure by attributes; processor signature `(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None)`; test T2 counts calls. |
| xformers / SDPA silently bypassing the CAST processor | **MEDIUM** | `enable_xformers_memory_efficient_attention()` after `apply_cast` would overwrite processors with `XFormersAttnProcessor`; PLADIS's own reset-to-`AttnProcessor2_0` pattern shows how easily originals are lost | Apply CAST last, immediately before `__call__`; T1 asserts the processor *object identity* after loading; `restore()` uses saved originals; never call `enable_xformers` in CAST runs. |
| SDPA hides probabilities | LOW | `F.scaled_dot_product_attention` has no probability output | Materialize attention manually in the CAST processor (S=77 ⇒ cheap); self-attention keeps SDPA. |
| Entmax numerical instability in fp16 | **MEDIUM** | sort/cumsum/sqrt in fp16; measured row-sum drift ±1.4 % | fp32 logits + fp32 entmax/softmax, cast probs to fp16 only for `@V`; NaN/Inf guard (T8). |
| Masks incompatible with entmax | LOW | `-inf` bias could NaN inside sort/cumsum (measured: did not, but undocumented) | `assert attention_mask is None` in v1; if needed, finite −1e4 additive bias before both normalizations. |
| Head-reshape errors | LOW | 3-D vs 4-D conventions coexist in 0.18.2 | Use the 4-D convention copied from `AttnProcessor2_0`; T3 parity (λ=0) against the stock processor at the single-module level catches any transpose error (an error would give O(1) differences, not 1e-3). |
| CFG behaviour differs from expectation | LOW (now understood) | Batched `[uncond, cond]`; null condition = zeros | §10: provably no-op on uncond; T7 asserts uncond outputs identical across modes; `apply_to_uncond` flag retained. |
| Accidentally modifying self-attention | LOW | wrong name filter | Double selection test (name ∧ shape) with `assert len==16`; runtime `assert encoder_hidden_states is not None`; T1 checks all 16 `attn1` processors are the original objects. |
| Checkpoint text encoder differs from expectation | LOW | `condition_config.json` could specify CLAP/T5 or multiple encoders (S = 77·n) | Verified for `auffusion/auffusion`: one CLIP-L/14 encoder, S=77, dim 768. Assert `encoder_hidden_states.shape[1:] == (77, 768)` at runtime; log `condition_config.json` in the run manifest. |
| Output difference caused by scheduler/config rather than CAST | **MEDIUM** | Any deviation in steps, CFG, size, dtype, autocast, seed, processor backend between arms | One `run_one` entry point; all arms share one `CastConfig` with only `mode` varying; manifest records everything; vanilla arm = stock processors; T6 vanilla vs `vanilla_manual` quantifies the backend-only difference once. |
| Seed reproducibility | LOW | Only the initial latent is random; PNDM deterministic; GPU kernels may be non-bitwise-deterministic | `torch.Generator(device).manual_seed(seed)` per run; T6 repeat-run check; store initial latents for reuse across modes (`latents=` argument of `__call__`) — the strongest guarantee that all three arms start from the same noise. |
| Dependency conflicts PLADIS ↔ Auffusion | LOW (once accepted) | diffusers 0.33.1 vs 0.18.2 | Separate env; no runtime import of PLADIS. |
| `enable_gqa`/`repeat_interleave` or `temper` subtleties from PLADIS code | LOW | both are no-ops in PLADIS (ratio 1, temper 1) | Omit; document. |
| Hyper-parameters tuned for SDXL at CFG 3–5 | **MEDIUM** (scientific) | λ=2 with w=7.5 may over-sharpen (PLADIS Fig. 5/20: artifacts at large λ) | Keep w=7.5 fixed per project rules; inspect spectrograms for over-smoothing; λ sweep is the natural *next* experiment after the milestone, not before. |
| Identical pad keys / additive-audio semantics | **MEDIUM** (scientific) | §3.4 items 1–2 | Stats hook (T11): per-layer attention mass on pad positions and entmax support size under softmax vs entmax; this converts the risk into a measurement. |
| HF Hub auth failure | LOW | stale cached OAuth token on this machine | Unset `HF_TOKEN`/remove `~/.cache/huggingface/token` or re-login; anonymous download verified working via HTTP. |

---

## 13. Minimal Validation Plan

Run in this order; each test gates the next. All tests except T6/T9/T10/T11 run on random tensors or a `from_config` U-Net and need no checkpoint.

| ID | Test | Assertion | Needs |
|---|---|---|---|
| T0 | Environment | `import auffusion_pipeline` succeeds; `diffusers.__version__ == "0.18.2"`; `torch.cuda.is_available()`; `import entmax`; one unmodified `inference.py`-equivalent run writes a wav; save it as `outputs/golden/vanilla_seed42.wav` with manifest | env + checkpoint |
| T1 | Module census & isolation | `len(unet.attn_processors) == 32`; `select_cross_attention` returns exactly the 16 `attn2`; name-rule == shape-rule for all 32; after `apply_cast`, all 16 `attn1.processor` objects are `is`-identical to before and all 16 `attn2.processor` are `CastCrossAttnProcessor`; after `restore`, all 32 are the original objects | from_config U-Net |
| T2 | Processor actually called | `calls` counter: 16 per U-Net forward; 1600 per 100-step run; `encoder_hidden_states.shape == (2, 77, 768)` inside | from_config U-Net / checkpoint |
| T3 | Dense parity | Single `attn2`, random fp16 inputs: CAST(λ=0) vs stock `AttnProcessor2_0` and vs `AttnProcessor` (baddbmm): max abs diff ≲ 1e-3 (fp16) and ≲ 1e-5 (fp32). Whole U-Net: ε from `vanilla_manual` vs stock ≲ fp16 rounding level, reported as a number | from_config U-Net |
| T4 | α dispatch | `alpha_entmax(·,1.0) == softmax` exactly; `alpha_entmax(·,1.5)` vs `entmax_bisect(alpha=1.5)` ≤ 1e-5 (**[MEASURED]** 8.3e-7); `alpha_entmax(·,2.0)` rows sum to 1 and are sparser than α=1.5; `entmax_bisect(alpha=1.0)` is **not** used (measured: not softmax) | none |
| T5 | λ algebra | `mix(λ=1) is p_sparse`; `mix(λ=0) is p_dense`; `mix(λ=2) == 2·p_sparse − p_dense` elementwise; row sums of `mix` = 1 ± 1e-5 in fp32 for λ∈{0,1,2}; `mix` has negative entries at λ=2 (expected) | none |
| T6 | Determinism & baseline identity | (a) `vanilla` twice with seed 42 → identical waveform (bitwise, or report max |Δ|); (b) `vanilla` equals golden T0 output; (c) `vanilla` vs `vanilla_manual`: report max |Δlatent|, max |Δspectrogram|, waveform SNR — this number is the backend floor below which mode differences are meaningless | checkpoint |
| T7 | Uncond invariance | Forward hook on each `attn2` output: rows `[0:1]` identical (max |Δ| = 0 or ≤ fp16 eps) between `vanilla_manual`, `sparse`, `pladis` for the same inputs; `K_uncond.abs().max() == 0` | checkpoint |
| T8 | Finite outputs | No NaN/Inf in `noise_pred` at every step (callback), final latents, spectrogram, waveform, for all three modes | checkpoint |
| T9 | Cost | Wall-clock and peak memory per mode (100 steps); expected ≈ +20–35 % for sparse/pladis | checkpoint |
| T10 | Modes are not no-ops and are ordered | `‖z_sparse − z_vanilla‖ > 0`, `‖z_pladis − z_vanilla‖ > 0`, both ≫ the T6(c) floor; typically `‖z_pladis − z_vanilla‖ > ‖z_sparse − z_vanilla‖` (expected, not asserted) | checkpoint |
| T11 | Audio-specific diagnostics (stats hook, no assertion) | Per layer/timestep: entmax support size (mean nnz of 77), softmax vs entmax attention mass on pad positions (`j ≥ n_tok`), mass on BOS/EOS, logit std; dump per-token cross-attention maps (as Auffusion Fig. 2 / PLADIS Fig. 11) for the sanity prompt | checkpoint |

**Sanity experiment (after T0–T10 pass).**

| Setting | Value |
|---|---|
| Prompt | "A dog barking while rain falls in the background." |
| Seed | 42 (`torch.Generator("cuda").manual_seed(42)`); additionally pass the *same* pre-sampled `latents` tensor to all three arms |
| Checkpoint | `auffusion/auffusion` |
| Steps / CFG / size / dtype / autocast | 100 / 7.5 / 256×1024 / fp16 / on (as `inference.py`) |
| Arms | `vanilla` (stock processors), `sparse` (α 1.5, λ 1), `pladis` (α 1.5, λ 2) |
| Outputs | `outputs/first_milestone/<mode>/{audio.wav, spectrogram.png, latents.pt, manifest.json}` + T11 attention dumps |
| Read-out | Listen; compare spectrograms (does "rain" background survive under sparsity? is the bark sharper?); T11 numbers; optional CLAP score (laion/clap-htsat-unfused, as in Auffusion README) |

---

## 14. Recommended Implementation Sequence

1. **Environment and golden baseline.** Create conda env `cast` (py3.10; torch 2.0.1+cu118; diffusers 0.18.2; huggingface_hub 0.16.4; transformers ≤4.3x; entmax 1.3; datasets, soundfile, librosa 0.9.2, scipy, matplotlib, accelerate; **no xformers**). Fix the HF token issue. Run the unmodified `third_party/Auffusion/inference.py`-equivalent on the sanity prompt (seed 42) → `outputs/golden/`. (T0) *Checkpoint download and GPU runs are the user's call per project practice; everything up to the run command can be prepared.*
2. **Integration skeleton.** `cast/integration/auffusion.py`: `import_auffusion()`, `load_pipeline()`, `select_cross_attention()`, `apply_cast()/restore()`. `cast/config.py`. Tests T1 with a `from_config` U-Net (no weights). `cast/pipeline.py` with `mode="vanilla"` only; verify it reproduces `outputs/golden/` (T6a/b).
3. **Dense-only CAST processor.** `CastCrossAttnProcessor` with λ fixed to 0 (manual fp32 softmax path). Tests T2, T3, T6(c) — this isolates all reshape/dtype/backend issues *before* any sparsity exists.
4. **Sparse ops.** `cast/attention/sparse_ops.py` + T4, T5 unit tests (pure tensor tests).
5. **Wire λ/α/mode.** Enable `sparse` and `pladis`; T7, T8; stats hook for T11.
6. **First milestone run.** The three arms with shared `latents`; T9, T10, T11; archive outputs and manifest. Stop here and review results before any ablation (per CLAUDE.md "Current Priority").
7. **Only afterwards:** λ sweep (0.5–3), layer-group ablation (down/mid/up), `apply_to_uncond=False` with a negative prompt, `auffusion-full`, AudioCaps-test evaluation with audioldm_eval/CLAP.

Each stage has a mechanical pass/fail and touches one new concern, so a failure localizes to that stage.

---

## 15. Files That Will Need Modification

| File | Purpose | New/Modified |
|---|---|---|
| `cast/__init__.py` | package marker, version | New |
| `cast/config.py` | `CastConfig` (checkpoint, mode, alpha, lam, apply_to_uncond, steps, guidance, height, width, dtype, autocast, seed, device, applied_layers=all) + mode resolution/validation | New |
| `cast/attention/__init__.py` | — | New |
| `cast/attention/sparse_ops.py` | `dense_softmax`, `alpha_entmax`, `mix_probs` (fp32, pure) | New |
| `cast/attention/cast_processor.py` | `CastCrossAttnProcessor` (0.18.2 protocol) + `AttnStats` hook | New |
| `cast/integration/__init__.py` | — | New |
| `cast/integration/auffusion.py` | sys.path import of `third_party/Auffusion`, `load_pipeline`, `select_cross_attention`, `apply_cast`, `restore`, context manager | New |
| `cast/pipeline.py` | `run_one()` single entry point; saving + manifest | New |
| `cast/tests/test_sparse_ops.py` | T4, T5 | New |
| `cast/tests/test_processor.py` | T1–T3, T7 (from_config U-Net, random tensors) | New |
| `cast/tests/test_pipeline_smoke.py` | T6, T8–T10 (slow, needs checkpoint) | New |
| `experiments/configs/first_milestone.yaml` | the sanity configuration | New |
| `experiments/run_first_milestone.sh` | loops the three modes; one command | New |
| `experiments/env/cast_environment.yml` (or `requirements-cast.txt`) | pinned environment | New |
| `third_party/PLADIS/**`, `third_party/Auffusion/**` | — | **Unmodified** |
| `CLAUDE.md` | optionally reconcile `ours/` vs `cast/` (see §16) | Possibly modified (user decision) |

---

## 16. Open Questions

1. **Directory name.** `CLAUDE.md` says new code goes in `ours/`; this task specifies `cast/`. Both directories do not exist yet (`cast/` is empty). Decision needed; the analysis assumes `cast/`.
2. **Checkpoint.** `auffusion/auffusion` (repo default) vs `auffusion/auffusion-full` (stronger, same architecture). Assumed: `auffusion/auffusion` first.
3. **Baseline backend.** `inference.sh` enables xformers; a torch-2 environment without xformers uses SDPA. Both are "vanilla Auffusion" up to fp16 kernel rounding. Assumed: SDPA (`AttnProcessor2_0`), documented in the manifest; T6(c) quantifies the floor.
4. **Autocast.** `inference.py` and the notebook use `torch.autocast("cuda")`; the README quickstart does not. Assumed: autocast on (matches the evaluation script). The CAST processor is autocast-independent by construction (explicit fp32).
5. **λ at CFG 7.5.** PLADIS tuned λ=2 at CFG 3–5 on SDXL. Whether λ=2 over-sharpens at w=7.5 (CFG scale is fixed by project rules) is empirical.
6. **Pad-token behaviour and additive-mixture semantics.** Cannot be determined statically; T11 measures it. If entmax zeroes the pad group entirely, the effective key set is `n_tok + 1`; this may be benign or may remove a "sink" that the fine-tuned model relies on.
7. **Logit scale of Auffusion's cross-attention.** Entmax support size depends strongly on it (**[MEASURED]** on Gaussian logits with S=77: mean support 24 / 10.6 / 4.8 / 3.3 / 2.2 keys at logit std 0.5 / 1 / 2 / 3 / 5). Unknown until T11.
8. **Evaluation protocol** for the real experiment (AudioCaps test set, FD/FAD/KL/IS/CLAP via audioldm_eval) is out of scope here and should be fixed before the full run.
9. **transformers version** compatible with `huggingface_hub 0.16.4` and the checkpoint's CLIP config (written by 4.31.0.dev0) must be confirmed at env-build time (step 1).

---

## 17. Final Recommendation

**Proceed.** The transfer is architecturally exact: Auffusion's text cross-attention is the same diffusers `Attention` module PLADIS already patches, with the same equations, the same 77-token key axis, and no masks or dropout to complicate the swap. Implement CAST as a single diffusers-0.18.2 `AttentionProcessor` installed on the 16 `attn2` modules from outside the Auffusion source, with one pipeline entry point and `mode ∈ {vanilla, sparse, pladis}` ⇒ `λ ∈ {0 (no swap), 1, 2}`, α=1.5, fp32 logits/normalization, applied to the whole CFG batch (provably a no-op on the zero null condition). Build it in the staged order of §14 so that environment, processor-swap isolation, dense parity, and sparse algebra are each verified before the first three-arm sanity run. The dominant engineering risk is the pinned, mutually-exclusive dependency stacks (diffusers 0.18.2 for Auffusion vs 0.33.1 for PLADIS, plus a GPU driver that rejects the base environment's torch); the dominant scientific risk is that audio's additive, zero-padded conditioning may respond to sparsity differently than images — which is precisely what the planned measurements (T6, T10, T11) are designed to reveal rather than assume.
