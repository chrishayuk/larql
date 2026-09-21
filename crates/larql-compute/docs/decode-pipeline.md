# Decode Pipeline — larql-compute

**Class: ARCHIVE.** This dated decode/dispatch walkthrough is retained for provenance. Current GPU execution belongs to [larql-compute-metal](../../larql-compute-metal/README.md), and V3 executes its own [component program](../../../docs/vindex3/execution.md). Historical fusion defaults below are not current capability claims.

How `decode_token` processes one token through all layers with KV cache.

## Overview

```
Input: x[hidden] (embedded token)
Output: h[hidden] (final hidden state for logit projection)

Per layer (Gemma 3 4B, post-2026-05-02 — 9 dispatches with 5 fusions
default-on; all in a SINGLE Metal encoder):
  1. Fused input_norm + QKV projection
       (q4k_q6k_qkv_proj_normed — 1 dispatch)
       OR: rms_norm (1) + q4k_q6k_qkv_proj (1) = 2 dispatches
  2. Fused QK-norm + RoPE
       (qk_norm_rope_fused — 1 dispatch; was qk_norm_qk + rope = 2)
  3. Batched V-norm (Gemma 4 only — Gemma 3 skips)
  4. Fused KV append + KV attend
       (kv_append_attend_fused — 1 dispatch; was 2)
  5. O projection (q4k_matvec / q4kf_proj)
  6. Fused post-attn norm + residual + ffn-norm + h_post_attn store
       (post_attn_residual_norm_store — 1 dispatch; was 3 on the
       has_post_norms path, was 2 on the residual_norm_store path)
  7. Fused FFN gate + up (q4k_ffn_gate_up_8sg — 1 dispatch)
  8. Fused GEGLU + down (q4k_geglu_gelu_tanh_down — 1 dispatch when
     down format is Q4_K; falls back to GEGLU + matvec when not)
  9. Fused post-FFN norm + residual_add
       (post_ffn_norm_residual_add — 1 dispatch; was 2)
```

All layers run in a **single Metal command buffer with a single global encoder**.
No per-layer encoder create/end overhead. Apple Silicon serialises compute
dispatches within an encoder so no explicit barriers are needed.

## Dispatch fusion history

Starting from ~14 dispatches/layer (~476/token):

**2026-04-25 wave** (4 fusions, ~136 dispatches/token saved):

| Fusion | Dispatches saved | Technique |
|---|---|---|
| `qk_norm_qk` | 34/token | One dispatch for Q+K heads instead of two |
| `rope_at_pos_batched_qk` | 34/token | One dispatch for Q+K heads |
| `residual_norm_store` | 34/token | Writes normed + raw sum simultaneously |
| `q4k_q6k_qkv_proj_normed` | 34/token | Norm computed inline in QKV TGs |

**2026-05-01 / 2026-05-02 wave** (5 fusions, ~136 dispatches/token saved):

| Fusion | Dispatches saved | Technique |
|---|---|---|
| `qk_norm_rope_fused` | 34/token | One TG/head: RMS-norm + RoPE in one pass; supersedes the qk_norm_qk + rope chain |
| `kv_append_attend_fused` | 34/token | Per-Q-head TG cooperatively writes new K/V row at pos, then attends; absorbs the kv_cache_append dispatch |
| `post_attn_residual_norm_store` | ~68/token | Triple fusion on the `has_post_norms` path: post-attn RMS + residual + ffn-norm + store |
| `post_ffn_norm_residual_add` | 34/token | Single 1-TG kernel: RMS over down_out + per-element norm + residual sum into next-layer input |
| (`attn_fused` — opt-in only) | — | Attempted further merge of qk_norm_rope + kv_append_attend; regressed -1.45 ms (parallelism loss). Kept registered as `LARQL_FUSED_ATTN=1`. |

Current: ~306 dispatches/token (9 dispatches/layer × 34 layers).
At measured ~6 µs/saved-dispatch this is ~1.84 ms of dispatch overhead;
the remainder of the ~11.5 ms GPU forward is genuine compute.

Each 2026-05 fusion has an `LARQL_FUSED_*=0` opt-out for diagnostic A/B.

## Dual-Path Architecture

`decode_token` auto-detects the weight format from `FullPipelineLayer.wq.format`.

### Q4_K + Q6_K Path (production — Gemma 3 / 4 Ollama extracts, 2026-05-02)

```
h_buf [f32]
  → q4k_q6k_qkv_proj_normed (RMS norm inline + fused Q4_K Q/K + Q6_K V)
  → qk_norm_rope_fused (Q+K norm + RoPE in one kernel)
  → v_norm_batched (Gemma 4 only)
  → kv_append_attend_fused (writes new K/V row + attends in one kernel)
  → q4k_matvec / q4kf_proj (O projection)
  → post_attn_residual_norm_store
        → ffn_norm_out [f32] + h_post_attn [f32]
  → q4k_ffn_gate_up_8sg (fused gate+up) → q4k_geglu_gelu_tanh_down (fused GEGLU+down)
  → post_ffn_norm_residual_add → h_buf [f32] (next-layer input)
```

### Q4_KF Path (fastest for Q4_KF vindexes)

```
h_buf [f32]
  → rms_norm → norm_f32 [f32]
  → q4kf_qkv_proj → Q, K, V [f32]
  → rope_at_pos_batched_qk + kv_attach
  → q4kf_proj (O) → residual_norm_store → FFN via q4kf_proj
```

### Q8 Path (legacy)

```
h_buf [f32]
  → rms_norm_q8 (fused) → q8_buf + q8s_buf
  → q8_qkv_proj → Q, K, V → kv_attend
  → quantize_q8 → q8_matvec (O)
  → residual_norm_q8 → FFN (same as Q4_K)
```

## KV Cache

```rust
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

pub struct LayerKVCache {
    pub k_cache: Buffer,    // [max_seq, num_kv_heads, head_dim] f32
    pub v_cache: Buffer,    // same
    pub current_len: usize,
    pub max_seq: usize,     // default 4096
}
```

Populated during prefill; extended by `kv_cache_append` each decode step.
`kv_attention` attends Q against all cached K/V (positions 0..current_len).

## Hybrid MoE — Batched Prefill Path (2026-04-26)

For hybrid MoE models (e.g. Gemma 4 26B A4B), each decoder layer has both
a dense FFN block (GPU) and a sparse expert block (CPU). `dispatch_full_pipeline`
accepts an optional `moe_fn` callback that fires after each MoE layer's dense FFN.

**Before (token-by-token loop):**
```
for pos in 0..seq_len:
    decode_token(layers, h[pos])   // ALL layers per token
```
O(seq_len × num_layers) GPU command buffer commits.

**After (batched per layer):**
```
for l in 0..num_layers:
    GPU: dispatch all seq_len positions through layer l's attention + dense FFN
    commit + wait
    if layer l has MoE:
        CPU: moe_fn(l, h_post_attn[0..seq_len], new_h[0..seq_len])
             ↳ experts for all positions + outer_norm + layer_scalar
```
O(num_layers) commits. For a 5-token prefill on 26 MoE layers: **26 commits vs 130**.

**Key invariant:** The GPU `layer_scalar` step (step 11) is skipped for MoE layers
when `moe_fn` is provided. The callback applies `layer_scalar` itself after
combining dense + MoE output — matching HF's `hidden_states *= layer_scalar`
placement at the end of `Gemma4TextDecoderLayer.forward`.

**Measured gain (Gemma 4 26B A4B, M3 Max, 15 warmup / 30 tokens):**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Prefill (5-token) | 1889ms | 1297ms | **−31%** |
| Decode GPU fwd | 334ms/tok | 246ms/tok | **−26%** |
| Decode tok/s | 2.9 | **3.9** | **+35%** |

**KV cache:** Per-layer variant `populate_kv_one_layer` (in `kv_copy.rs`)
copies one layer's K/V scratch immediately after each per-layer commit,
so the cache is current before the MoE callback reads `h_post_attn`.

## Performance (M3 Max, 2026-05-02)

### Gemma 3 4B (dense, 34 layers, all five 2026-05 fusions default-on)

| Path | GPU fwd | tok/s | vs Ollama |
|---|---|---|---|
| **Q4_K+Q6_K decode (34L)** | **11.5–12.0ms** | **72–75** | **1.30–1.45×** slower |
| Ollama gemma3:4b | ~10ms | 96–104 | 1.0× |

Per-stage: GPU fwd 79%, lm_head 20%.

The 2026-05 wave landed -0.99 ms cumulative GPU savings vs. unfused baseline
(10.45 → 9.46 ms isolated kernel time). End-to-end gain is smaller than the
isolated saving (cold/warm GPU thermal variance dominates at this scale on
M3 Max). The further `attn_fused` merger was attempted and regressed —
parallelism loss is the reason it's kept opt-in. Path-to-80 lever search is
documented in `crates/larql-inference/ROADMAP.md` (G-3, G-5 still open).

### Gemma 4 26B A4B (hybrid MoE, 26 layers, batched prefill)

| Metric | tok/s | GPU fwd/tok |
|---|---|---|
| **LARQL Metal** | **3.9** | **246ms** |

Effective bandwidth: LARQL ~329 GB/s, Ollama ~348 GB/s (Gemma 3).
Total weight data per token: 3029 MB (34 layers × 89.1 MB/layer).
See `PERFORMANCE.md` for the full bandwidth budget and gap analysis.
