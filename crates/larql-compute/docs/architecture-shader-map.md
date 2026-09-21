# Architecture → Metal shader map

**Class: ARCHIVE.** This family-routing map records an earlier backend design. Current [Metal implementation](../../larql-compute-metal/README.md) and [V3 lowering contracts](../../../docs/lowering-plane-inventory.md) own present dispatch support.

**Date**: 2026-05-09
**Purpose**: Make obvious which Metal shaders each model architecture dispatches to. Bridges `crates/larql-models/src/architectures/{family}.rs` (architecture trait implementations) to `crates/larql-compute/src/metal/shaders/*.rs` (the kernels themselves), via the dispatch logic in `metal/decode/`, `metal/stages/`, `metal/ops/`, `metal/prefill.rs`, `metal/decode_hybrid.rs`.

## How dispatch happens today

The compute crate doesn't have explicit per-architecture dispatch tables. Instead, the kernel selected at each stage depends on **format predicates** that effectively encode "is this model X?" implicitly:

| Predicate | True for | Implicit arch grouping |
|---|---|---|
| `wq.format == Q4_K && wk.format == Q4_K && wv.format == Q6_K` | Models extracted with the ollama Q4_K_M convention | Gemma 3 / 4 (mostly), Llama 2/3 / Mistral with Q6_K down (some) |
| `wq.format == wk.format && wk.format == wv.format && wq.format != Q6_K` (uniform Q4_K) | Llama 2/3, Mistral (Q4_K_M without Q6_K) | "Uniform Q4_K" path |
| `wq.format == Q4_KF` | GGUF-direct extracts (144-byte block layout) | llama.cpp-saved GGUFs of any family |
| `norm_type == RmsNorm && input_norm_bias.is_none()` | Gemma 3/4 / Llama / Mistral / Qwen | RMS-norm models (almost all modern transformers) |
| `norm_type == LayerNorm` | StarCoder2, GPT-2, BERT | LayerNorm models |
| `has_v_norm` | Gemma 4 only | parameter-free V-norm |
| `attn_q_norm_key().is_some()` | Gemma 3 / 4 | per-head QK-norm |
| `has_post_norms` | Gemma 3 / 4 (4 norms per layer) | post-norm-on-attention pipeline |
| `is_global_layer(layer)` | Gemma 4 (every Nth layer is global) | global attention vs sliding window |

This works (and is genuinely model-agnostic at dispatch time), but a future-reader has to grep the predicate logic to figure out which kernels a given architecture uses. This doc lays it out explicitly.

## Per-architecture shader map

### Gemma 3 (4B, etc.) — `larql-models/architectures/gemma3.rs`

Hidden=2560 (4B), 34 layers, 8 Q heads × head_dim=256, 4 KV heads, vocab=262K.

| Stage | Shader(s) | Notes |
|---|---|---|
| Embedding scale | `residual_inject::scale_vector` | Gemma scales embed by `sqrt(hidden_size)`. |
| Input RMS-norm | `rms_norm` (in `residual_inject`) | Standard RMS-norm, weight offset baked into stored weight (HF convention). |
| QKV projection | **`q4k_q6k_qkv_proj`** (mixed Q4_K Q/K + Q6_K V; non-fused since 2026-05-09 ADR-016) — also has `NORMED_SHADER` opt-in via `LARQL_QKV_FUSED=1` | Production. Uses `encode_q4k_input_norm` + `encode_q4k_qkv` in the defused path. |
| QK-norm | `qk_norm` (or `qk_norm_rope_fused` when `LARQL_FUSED_QK_NORM_ROPE=1` default) | Per-head RMS-norm on Q and K; required to prevent softmax NaN on Gemma 3 weights. |
| RoPE | `rope` (or fused with QK-norm in `qk_norm_rope_fused`) | Standard Llama-style RoPE; full 256-d rotation. |
| KV-cache append | `kv_append_attend_fused` (Gemma 3 has-post-norms path) | Saves 1 dispatch/layer. |
| Attention (decode) | `kv_attention` (T ≤ 1024) | Sliding-window every layer except every 6th. |
| Attention (prefill) | `fused_attention` | Handles RoPE, QK-norm, GQA, causal mask in one kernel. |
| O-projection | `q4k_matvec_8sg` (per-position) | Production default since 2026-04-28 (8sg dispatch). |
| Post-attn norm + residual | `post_attn_residual_norm_store` | Triple-fused: post-attn RMS + residual + ffn-norm + store (one dispatch). |
| FFN gate+up | `q4k_ffn_gate_up_8sg` (default, 8sg) — with `LARQL_GATE_UP_8SG=0` opt-out to `q4k_ffn_gate_up`, `LARQL_F16_ACC=1` to `q4k_ffn_gate_up_f16acc`, `LARQL_GATE_UP_COOP=1` to `q4k_ffn_gate_up_coop` | All production fired per-position; matmul wiring twice-falsified (D-PREFILL-MM closed). |
| GEGLU activation | `geglu` (GELU-tanh variant) | Element-wise gate × up activation. |
| FFN down | `q6k_matvec` (Q6_K weights, **4sg — the default**; `LARQL_Q6K_8SG=1` opts into the 8sg arm, which did not translate end-to-end) | Q6_K convention from ollama extracts. |
| Post-FFN norm + residual | `post_ffn_norm_residual_add` | Fused norm + residual into next-layer input. |
| Final norm | `residual_inject::rms_norm` | Standalone one-TG dispatch. |
| lm_head | `q4k_matvec` (production since 2026-05-02 dispatch fix) | Falls back to `q4k_matvec_stride32` if `LARQL_LM_HEAD_SKIP_Q4K=1`, then `f16_gemv` (tied embed), then `f32_gemv`. |

### Gemma 4 (E2B, 31B dense) — `larql-models/architectures/gemma4.rs`

Hidden=1536 (E2B) / 5376 (31B), 35/60 layers, **alternating sliding (head_dim=256) and global (head_dim=512) layers**.

Largely the same shaders as Gemma 3, with these differences:

| Stage | Difference | Shader(s) used |
|---|---|---|
| V-norm | Parameter-free RMS-norm on V before attention (Gemma 4 only) | `v_norm` |
| RoPE rotary fraction | Global layers use 25% rotation (192/256 rotated dims), sliding layers use full | `rope` with `rotary_dim` per layer |
| Attention head dims | Global vs sliding layers have different head_dim | `kv_attention` (long variant for global layers, T ≤ 4096) |
| QKV input norm offset | `0.0` (Gemma 4 vs Gemma 3's `1.0` for HF-saved weights) | Same `q4k_q6k_qkv_proj` kernel, different config |
| FFN intermediate size | E2B: 6144, 31B: 21504 | Same `q4k_ffn_gate_up_8sg` + `q6k_matvec` |

**Diagnosed anomaly (2026-05-09)**: gemma4-e2b decode runs at **~1670 ms/tok on CPU**, not Metal. Root cause: **Per-Layer Embeddings (PLE, `hidden_size_per_layer_input: 256`) are not implemented in the Metal pipeline.** `larql-inference/src/layer_graph/generate/gpu/` explicitly checks `weights.arch.has_per_layer_embeddings()` and routes the entire generate path to `generate_via_cpu_q4k`. The Metal `decode_token_with_moe_split_fn` is never called — `[gpu-timing]` lines never fire for E2B, while they do for Gemma 3 4B and Gemma 4 31B (which don't have PLE). The CPU fallback is documented in the source comment as deliberate: "Without this routing the model produces multilingual gibberish."

**To restore E2B to Metal**: implement Per-Layer Embeddings in the Metal pipeline (ROADMAP **D-METAL-PLE**). The PLE math is in `larql-inference/src/forward/ple.rs`:
- Precompute (once at prefill): `projected = main_embeds @ per_layer_model_projection.T * 1/sqrt(hidden)`, then per-layer RMSNorm + add `embed_tokens_per_layer[token_ids] * sqrt(ple_dim)`, scaled by `1/sqrt(2)`.
- Per layer: `gate = gelu_tanh(h × W_input_gate.T)` → `gated = gate * per_layer_input` → `contribution = gated × W_projection.T` → `RMSNorm(contribution)` → `h += normed`.

Most kernels needed already exist (matvec, geglu element-wise, rms_norm, residual_inject::add). Plumbing + per-layer dispatch + caching the precomputed per-layer-input streams in Metal buffers is the actual work. Estimated 1-2 days; brings E2B from ~1670 ms/tok CPU to ~10-20 ms/tok Metal (80-150× speedup at E2B's compute scale).

### Gemma 4 26B-A4B (MoE) — `larql-models/architectures/gemma4.rs` with MoE config

5B activated, 26B total, expert routing.

| Stage | Shader(s) | Notes |
|---|---|---|
| MoE gate scoring | `f32_gemv` (production, Metal gate scoring landed 2026-04-19) | Picks top-K experts per token. |
| Expert FFN | `q4k_ffn_gate_up` + `q4k_geglu_down` (per expert, dispatched via `moe_dispatch/`) | Geometry fix landed 2026-05-02; pre-fix was 5.1 tok/s (broken dispatch), post-fix 19.4 tok/s. |
| Expert combine | Custom Metal/CPU outer-combine helper (`outer_combine.rs`) | Resolved 4 silent CPU/Metal divergences 2026-04-26. |

### Llama 1/2/3 — `larql-models/architectures/llama.rs`

Hidden=4096 (7B), 32 layers, 32 Q heads, 32 (Llama 1) or 8 (Llama 2 GQA) KV heads. **Largely uses Llama defaults** in the architecture trait (which are themselves Llama-shaped).

| Stage | Shader(s) | Notes |
|---|---|---|
| Input RMS-norm | `rms_norm` | Standard. |
| QKV projection | **uniform Q4_K path**: `q4k_qkv_proj` (single fused kernel) — when `wq.format == wk.format == wv.format == Q4_K`. Or `q4k_q6k_qkv_proj` if Q6_K V is used. | Different from Gemma 3 (which uses the mixed Q/K Q4_K + V Q6_K convention). |
| RoPE | `rope` | NeoX-style RoPE convention: ⚠️ **gap**: we don't have `rope_neox` variant for the few Llama variants that use it; current `rope` is interleaved-style. Not a problem for Llama 2/3 standard; flagged in `rope.rs` TODO. |
| Attention (decode) | `kv_attention` | Default GQA path. |
| Attention (prefill) | `fused_attention` | Same as Gemma. |
| FFN (gated SiLU) | `q4k_ffn_gate_up_8sg` + `geglu` (SiLU) + `q4k_geglu_down` (Q4_K down) **OR** `q6k_matvec` (Q6_K down convention) | Activation is **SiLU** for Llama (vs GELU-tanh for Gemma). |
| lm_head | Same as Gemma: `q4k_matvec` (Q4_K), `f16_gemv` (untied embed), `f32_gemv` fallback. | Llama typically has untied embed + lm_head. |

### Mistral 7B — `larql-models/architectures/mistral.rs`

Same structure as Llama (Mistral inherits most defaults). Hidden=4096, 32 layers, GQA.

| Difference from Llama | Notes |
|---|---|
| Sliding window attention | Mistral uses 4096-token sliding window; engaged through `kv_attention` long variant when `is_sliding_window_layer(layer)`. |
| Norm offset, eps | `1e-5` vs Llama 2's `1e-6`. |

Otherwise identical shader path to Llama.

### DeepSeek (V2 / V3) — `larql-models/architectures/deepseek.rs`

**Architecture supported, compute path partially exercised.** DeepSeek uses MLA (Multi-Latent-Attention) and a different MoE expert pattern than Gemma 4 26B-A4B. The current MoE dispatch (`moe_dispatch/`) handles Gemma 4 a4b's pattern but DeepSeek-V3's 256-expert + 8-shared-expert pattern needs verification.

⚠️ **gap**: no DeepSeek vindex tested in this audit. Architecture trait exists (137 LOC in `deepseek.rs`); compute path not validated.

### Qwen — `larql-models/architectures/qwen.rs`

Hidden=4096+, GQA, Llama-shaped. **Compute path equivalent to Llama** for Qwen 2/2.5 standard variants. Qwen 3 MoE variants would route through MoE dispatch (untested).

⚠️ **gap**: no Qwen vindex tested in this audit.

### Mixtral / GPT-OSS — MoE archs

- `mixtral.rs`: 8-expert MoE, top-2 routing. Should route through Gemma 4 a4b's MoE path with the right expert count config.
- `gpt_oss.rs`: GPT-OSS architecture.

⚠️ **gap**: not validated in current cross-arch bench.

### StarCoder2 — `larql-models/architectures/starcoder2.rs`

Hidden=3072+, **LayerNorm (not RMS-norm)**, **standard FFN (not gated)**.

| Stage | Shader(s) | Notes |
|---|---|---|
| Input norm | `layer_norm` (mean-subtraction + variance) | Different from RMS-norm path. |
| QKV | `q4k_qkv_proj` (uniform Q4_K, no QK-norm) | Same kernel as Llama. |
| Attention | `kv_attention` | Same. |
| FFN | `activation` (GeluTanh / GeluErf) + standard `q4k_matvec` for up + `q4k_matvec` for down | **Non-gated FFN**. Goes through `encode_standard` in `stages/ffn.rs`, not `encode_gated`. |
| lm_head | Same lm_head dispatch chain. | |

### GPT-2 — `larql-models/architectures/gpt2.rs`

LayerNorm + non-gated FFN + learned position embedding. Same path as StarCoder2 with these differences:

| Difference | Shader(s) |
|---|---|
| Position embeddings | Loaded from `wpe.weight` instead of RoPE; no rotation applied. | `residual_inject::add` for the embedding. |
| Bias terms on QKV/lm_head | All linear layers carry bias. | Format-aware kernels handle bias when present in vindex. |

### Granite, TinyModel — `larql-models/architectures/{granite,tinymodel}.rs`

Granite is Llama-derived with attention scale modifications. TinyModel is the LARQL custom v10c/v11 architecture for interpretability research. Both use Llama-shaped compute paths.

## Cross-architecture bench data

**Captured 2026-05-09 under heavy system contention** (3 concurrent claude sessions + cargo rustc compile). Absolute numbers are throttled ~15× from quiet-state baseline; relative ratios across families on the same machine state are still informative.

| Family | Vindex | Hidden | Layers | GPU fwd ms (contended) | tok/s (contended) | Status |
|---|---|---|---|---|---|---|
| gemma3 | gemma3-4b-q4k-v2 | 2560 | 34 | 139.9 | 6.1 | ✓ — baseline |
| gemma4 | **gemma4-e2b-q4k** | **1536** | **35** | **4205** | **0.2** | **30× slower than expected — bug** |
| gemma4 | gemma4-31b-q4k | 5376 | 60 | 591 | 1.7 | ✓ — scales as expected |
| llama | llama2-7b-q4k | 4096 | 32 | 202.6 | 3.8 | ✓ — 1.45× Gemma 3 (matches hidden² scaling) |
| mistral | mistral-7b-v0.1-q4k | 4096 | 32 | 217.2 | 3.7 | ✓ — within 7% of Llama 2 7B (expected — same shape) |
| mistral | mistral-7b-instruct-v0.3-q4k | 4096 | 32 | 219.0 | 4.1 | ✓ — same as v0.1 base |

**Three things this confirms:**

1. **Cross-arch dispatch works.** All non-PLE families (Gemma 3, Gemma 4 31B dense, Llama, Mistral) run end-to-end on Metal without crashes.
2. **Llama / Mistral scaling is correct.** 4096 hidden × 32 layers vs Gemma 3 4B's 2560 × 34 ≈ 1.5× compute scaling — measured 1.45-1.55× GPU fwd. Matches expectation.
3. **Gemma 4 E2B is on CPU, not Metal** — diagnosed (see anomaly section above). PLE-using models fall back to CPU until D-METAL-PLE lands. The 30× number is CPU-vs-Metal, not a Metal kernel bug.

**To re-bench cleanly** (when system is idle):

```bash
for v in gemma3-4b-q4k-v2 gemma4-e2b-q4k gemma4-31b-q4k llama2-7b-q4k mistral-7b-v0.1-q4k; do
  echo "=== $v ==="
  ./target/release/larql bench ~/.cache/larql/local/${v}.vindex --tokens 30 --warmup 5 \
    --prompt "The capital of France is" 2>&1 | grep -E "larql-metal|GPU fwd|lm_head"
done
```

## Coverage gaps

| Gap | Impact |
|---|---|
| Gemma 4 E2B 30× slowdown | Real bug; investigation needed. |
| No DeepSeek vindex tested | Compute path not validated for MLA + 256-expert MoE pattern. |
| No Qwen vindex tested | Compute path equivalent to Llama in theory; untested. |
| No Mixtral / GPT-OSS vindex tested | MoE pattern variations untested. |
| No GPT-2 / StarCoder2 vindex tested | LayerNorm + non-gated FFN path; should work via existing kernels but unverified. |
| No `rope_neox` variant | Affects models using NeoX-style interleaved RoPE (some Falcon, GPT-NeoX, Pythia variants). |
| No `rope_multi` variant | Affects models with multiple RoPE frequency bands. |
| Per-shader cross-model parity tests | Tests in `crates/larql-compute/tests/` are mostly Gemma-only; integration-level tests in `crates/larql-inference/tests/test_logits_goldens.rs` cover 4 families end-to-end. |

## How to add a new architecture

1. Add the architecture trait impl: `crates/larql-models/src/architectures/{family}.rs` overriding `ModelArchitecture` methods that differ from defaults.
2. Add detection logic in `crates/larql-models/src/detect/`.
3. Add at least one entry in `crates/larql-inference/tests/test_logits_goldens.rs` with the model's golden tokens.
4. Bench with `./target/release/larql bench <vindex>` to verify dispatch works end-to-end.
5. **If a new shader is needed**: add it under `crates/larql-compute-metal/src/shaders/`, document its applicability in `shader-inventory.md`, register in `metal/mod.rs::all_shaders`, and add a row to this doc's per-architecture table.

## Related

- `crates/larql-compute/docs/shader-inventory.md` — per-shader retention rationale + applicability.
- `crates/larql-compute/docs/llama-cpp-comparison.md` — kernel-architecture comparison vs llama.cpp.
- `crates/larql-models/docs/architecture-trait.md` — the `ModelArchitecture` trait reference.
- `crates/larql-compute/PERFORMANCE.md` — current state, perf history.
- `crates/larql-compute/ROADMAP.md` — open tracks including D-GEMMA4-E2B (the 30× anomaly).
