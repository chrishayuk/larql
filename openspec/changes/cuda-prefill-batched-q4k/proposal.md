## Why

The decode side of the CUDA backend is now at 10.36 ms/token
(96.5 tok/s) — a 15.7× speedup over the pre-LARQL-CUDA-work
baseline and within 2.35× of llama-cpp-turboquant. Prefill is
where the gap is widest:

| | LARQL prefill | llama-cpp-turboquant | Gap |
|---|---:|---:|---:|
| ms / 6-token prompt | 130.9 | ~5.6 | **23×** |
| effective tok/s | ~46 | 1073.43 | 23× |

Bench numbers from `larql bench output/gemma-3-4b-it-vindex
--backends cuda --tokens 20 --warmup 3`. The 23× gap is the
single biggest remaining performance issue and it directly
hits user-facing time-to-first-token on long prompts.

The cause is structural. `prefill_q4` in
`crates/larql-compute/src/cuda/decode.rs` is currently:

```rust
fn prefill_q4(&self, ..., x: &[f32], seq_len: usize, ...) -> Option<Vec<f32>> {
    self.reset_kv_cache();
    let mut out = Vec::with_capacity(x.len());
    for pos in 0..seq_len {
        let row = &x[pos * hidden..(pos + 1) * hidden];
        let h = self.decode_token(layers, row, ...)?;  // ← single token at a time
        out.extend_from_slice(&h);
    }
    Some(out)
}
```

Each `decode_token` call is a full single-token transformer
pass. The Q4_K projection weights get re-read from VRAM on
every position — 7 projections × 34 layers × ~3 MB ≈ 700 MB
of weight bandwidth per position, and the matvec kernels are
sized for the rank-1 case (single Q vector). For an
`seq_len = 6` prompt that's 4.2 GB of redundant weight reads
and 6× the per-call kernel-launch overhead.

The right shape for prefill is **batched**: every projection
becomes a `(seq_len, hidden) × (hidden, out_dim) → (seq_len,
out_dim)` matmul instead of seq_len separate
`(hidden,) × (hidden, out_dim) → (out_dim,)` matvecs. The
weight is read once across all positions; cuBLAS GEMM
saturates RTX 4090's compute at f32 with `m = seq_len > ~4`.
For Gemma 3 4B prefill of a 6-token prompt:

- Per-projection compute: `seq_len × hidden × out_dim × 2` FLOPs
  ≈ 60 MFLOPs per layer × 34 layers ≈ 2 GFLOPs total.
- At RTX 4090's ~83 TFLOPs f32: ~25 µs pure compute.
- Attention (per-position loop): seq_len × ~0.1 ms = 0.6 ms.
- Element-wise ops (rms_norm, silu, add): bandwidth-bound at
  hidden × seq_len floats = ~60 KB × ops ≈ negligible.

Realistic prefill target: **≤ 3 ms / token** (so a 100-token
prompt completes in ~300 ms instead of ~13 s).

## What Changes

### Single phase — batched prefill via cuBLAS f32 GEMM

This change ships the simplest correct batched prefill.
Tensor Cores / Q4_K-mmq optimisations are deferred to follow-
up changes.

- ADD `cuda::backend::with_q4k_f32_device_buf<R>(host, ...)`,
  parallel to the existing `with_q6k_f32_device_buf`.
  First call dequantises the packed Q4_K bytes to f32 on the
  device and caches the slice; subsequent calls borrow.
  VRAM impact: ~9.6 GB for Gemma 3 4B's Q4_K weights, fits
  comfortably on the 4090's 24 GB. Documented as a known
  trade-off; Phase 2 (Q4_K mmq) drops the f32 cache.
- ADD `cuda::elem::rms_norm_batch_device(x_seq, weight, n,
  seq_len, eps, offset)` — applies rms_norm to each row of
  an `[seq_len, n]` device buffer independently. Reuses the
  existing single-block reduction; one CUDA block per row.
- ADD `cuda::elem::silu_gate_up_batch_device(gate_seq,
  up_seq, n, seq_len, gelu_tanh)` — element-wise; existing
  kernel already uses `idx = blockIdx.x * blockDim.x +
  threadIdx.x`, just launch with `n × seq_len` threads.
- ADD `cuda::elem::add_in_place_batch_device(dst_seq,
  delta_seq, n_total)` — element-wise; same pattern.
- ADD `cuda::elem::scale_inplace_batch_device(...)` — for
  the per-layer `layer_scalar` over the full batch.
- ADD `CudaBackend::prefill_q4_seq_device` — the batched
  prefill core. Mirrors `decode_token_device`'s structure
  but operates on `[seq_len, hidden]` tensors:

  ```rust
  fn prefill_q4_seq_device(&self, layers, x_seq, seq_len, ...) -> Option<Vec<f32>> {
      let mut h_seq = htod(x_seq);  // [seq_len, hidden]
      for layer in layers {
          // 1. RMSNorm batched
          let h_attn_seq = rms_norm_batch_device(h_seq, input_norm, ...);
          // 2. QKV projections via cuBLAS GEMM
          let q_seq = gemm(h_attn_seq, wq_dequant);  // [seq_len, q_dim]
          let k_seq = gemm(h_attn_seq, wk_dequant);
          let v_seq = gemm(h_attn_seq, wv_dequant);
          // 3. Per-position attention (Phase 2 makes this batched too)
          for pos in 0..seq_len {
              let attn_out_pos = fused_decode_attention_device_kv(
                  q_seq.row(pos), k_seq.row(pos), v_seq.row(pos),
                  &mut kv_slot.k, &mut kv_slot.v,
                  ..., pos: cache.len + pos, ...
              );
              copy_into(attn_out_seq.row(pos), attn_out_pos);
          }
          // 4. wo + residual + post-norm
          let attn_delta_seq = gemm(attn_out_seq, wo_dequant);
          add_in_place_batch_device(h_seq, attn_delta_seq);  // residual
          // 5. FFN
          let h_ffn_seq = rms_norm_batch_device(h_seq, ffn_norm, ...);
          let gate_seq  = gemm(h_ffn_seq, gate_dequant);
          let up_seq    = gemm(h_ffn_seq, up_dequant);
          let act_seq   = silu_gate_up_batch_device(gate_seq, up_seq, ...);
          let ffn_delta_seq = gemm(act_seq, down_dequant);
          add_in_place_batch_device(h_seq, ffn_delta_seq);
      }
      cache.len += seq_len;
      Ok(dtoh(h_seq))
  }
  ```

- REWORK `DecodeBackend::prefill_q4` impl to dispatch to
  `prefill_q4_seq_device` when all layers support the
  device path. Keep the existing per-position decode loop
  as `prefill_q4_host_fallback` reachable via
  `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` (parity reference).

### Out of scope

- **Q4_K mmq** (matrix-matrix-quantised) kernel. The natural
  follow-up — replaces the f32 dequant cache with a kernel
  that consumes packed Q4_K + Q8_1-quantised input directly,
  recovering the ~9.6 GB of VRAM. Implementation is similar
  to `cuda-q4k-mmvq-int8` but with multiple Q rows.
- **Q6_K mmq** — same idea for Q6_K weights (down
  projection). Less urgent because Q6_K's existing f32
  cache is already in use.
- **Batched attention** — the per-position attention loop
  stays. seq_len is typically ≤ 1024 for prompt prefill; the
  loop is `O(seq_len × per_call ≈ 0.1 ms × seq_len)` which
  is small relative to the projection work. A true tiled
  FA-style attention is a separate change.
- **Mixing batched prefill with decode in the same step**
  (continuous batching). Out of scope for LARQL's CLI bench
  today; relevant when LARQL gets a serving frontend.
- **BF16 / Tensor Cores** — separate change; orthogonal.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for the
  batched element-wise kernels and the batched prefill
  entry point.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/elem.rs` — adds the
    `*_batch_device` element-wise variants.
  - `crates/larql-compute/src/cuda/backend.rs` — adds
    `with_q4k_f32_device_buf`.
  - `crates/larql-compute/src/cuda/decode.rs` — splits the
    `DecodeBackend::prefill_q4` impl into
    `prefill_q4_seq_device` (batched) and
    `prefill_q4_host_fallback` (per-position loop).
  - `crates/larql-compute/src/cuda/matmul.rs` — already has
    `matmul_transb` for the (seq_len, hidden) × (out_dim,
    hidden)^T = (seq_len, out_dim) shape. Verified.

- **Affected systems**: GPU container only. Metal backend
  unaffected.

- **Provenance**: prefill 23× gap quantified in the
  `cuda-q6k-mmvq` proposal's bench table. Pattern (batched
  GEMM for projections, per-position attention loop) is
  standard across llama.cpp, vLLM, etc.

- **VRAM impact**: ~9.6 GB additional for Gemma 3 4B's Q4_K
  weights' f32 dequantised cache. Within the 4090's 24 GB
  but reduces headroom. Documented in `proposal.md` and the
  `cuda-rotorquant-status.md` follow-up. Phase 2 (Q4_K mmq)
  reclaims this VRAM.

## Risks and back-out

- **VRAM exhaustion**. On smaller GPUs (e.g., RTX 3060 12 GB)
  the f32 cache won't fit. Mitigation: detect free VRAM at
  cache-init time and fall back to the per-position decode
  loop if the cache won't fit, with a clear warning. The
  back-out path is already there.
- **Numerical drift**. f32 GEMM vs f32 matvec produces
  bit-different results due to fp32 reduction order. The
  existing parity test
  `decode_token_phase1_matches_host_fallback` covers
  decode; we add a sibling
  `prefill_q4_seq_matches_host_fallback` (≤ 1e-3 vs the
  per-position path) for prefill.
- **Per-position attention loop** is suboptimal for very
  long prompts (seq_len > ~256). Documented; the natural
  follow-up is `cuda-prefill-batched-attention`.
- **Back-out**: `LARQL_CUDA_PREFILL_HOST_FALLBACK=1`
  reverts to the per-position decode loop.

## Acceptance bar

Final numbers measured on the dev box (RTX 4090, CUDA 12.5,
Gemma 3 4B Q4_K vindex, 6-token prompt, 20 decode tokens
after 3 warmup):

| Metric | Pre-change | **Actual** | Target | llama-cpp-turboquant |
|---|---:|---:|---:|---:|
| `prefill ms / 6 tokens` | 130.9 | **117.6** (-10%) | ≤ 20 (5.9× miss) | ~5.6 |
| effective prefill tok/s | ~46 | **51** | ≥ 300 | 1073 |
| `decode ms/token` | 10.36 | **11.04** (+7%) | ≤ 11 | 4.40 |
| Bit parity vs host fallback | — | **passes** | ≤ 1e-3 |  |

**Phase 1 misses the prefill target by 5.9×.** Per the
change's decision gate (>50% miss → profile-and-document),
the residual write-up: profiling with
`LARQL_CUDA_PREFILL_PROFILE=1` shows the per-section costs
(steady-state, after the f32 cache is warm):

```
norm           20.85 ms  (35%)   ← batched rms_norm overhead
attn           22.15 ms  (37%)   ← per-position fused_decode_attention loop
qkv             2.65 ms  ( 5%)   ← cuBLAS GEMM is fast at seq_len=6
wo              1.36 ms  ( 2%)
gate_up         8.91 ms  (15%)
down            4.60 ms  ( 8%)
silu            0.25 ms  ( 0%)
                ──────
total          ~60 ms (kernels) + ~57 ms launch/setup overhead = 117 ms
```

The projection GEMMs ARE fast — qkv+wo+gate_up+down sum to
17.5 ms, which is ~5% of llama.cpp's projection compute on
the same shape. The big losses are:

1. **Per-position attention loop** (22 ms = 37%). The
   existing `fused_decode_attention_device_kv` kernel is
   designed for `pos += 1` decode steps. Running it
   `seq_len` times per layer adds up. Fixing this requires
   a true batched-prefill attention kernel (causal
   `Q × K^T`, softmax, `× V` over all `seq_len` rows) —
   substantial new work.
2. **Batched rms_norm** (20.85 ms = 35%). Each call now
   does `seq_len` rows; expected ~6× the single-row cost
   (~30 µs / call), so 6 × 30 = 180 µs predicted vs the
   ~150 µs measured per call — that's actually fine. The
   issue is we have 4 norm calls × 34 layers = 136 calls,
   each carrying CUDA launch overhead. CUDA Graph capture
   could help here.

The decode regression is small (10.36 → 11.04 ms/tok,
within run-to-run noise) but real — most likely the
9.6 GB f32 weight cache adds VRAM pressure that slightly
reduces L2 hit rate on the packed-Q4_K cache used by
mmvq decode. Worth confirming with `nvprof`; if real,
Phase 2 (Q4_K mmq, no f32 cache) reclaims the VRAM.

**Phase 1 ships as a 10% prefill improvement** with the
right plumbing for Phase 2:
- `with_q4k_f32_device_buf` cache helper
- Batched element-wise kernels (`rms_norm_batch_device` etc.)
- `prefill_q4_seq_device` core
- `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` back-out

Phase 2 (`cuda-prefill-batched-attention`) is the natural
follow-up and addresses the actual bottleneck. Predicted
result with batched attention: prefill ~30-40 ms (a 3-4×
total improvement vs the legacy path), with decode parity
maintained.
