## Context

`cuda-f32-baseline` proved out:

- driver init via `cudarc::driver::CudaContext::new(0)`,
- cuBLAS handle on the default stream,
- the row-major-via-column-major transposed-product identity,
- host↔device blocking copies,
- a parity-test harness that runs on a CPU host (no-op) and on the
  GPU host (real cuBLAS).

Gemma 3 4B's hot path is dominated by Q4_K projections. A workable
"the GPU does work" milestone requires Q4_0 / Q4_K / Q6_K matvec to
return real values rather than `None`.

The CPU path already has correct dequantizers in
`larql_models::quant::ggml::{legacy::dequantize_q4_0,
q4_k::dequantize_q4_k, q6_k::dequantize_q6_k}`. They take `&[u8] +
n_elements` and emit `Vec<f32>`.

## Goals / Non-Goals

**Goals:**

- `CudaBackend::q4_matvec` / `q4k_matvec` / `q6k_matvec` return
  `Some(_)` and match CPU within 5e-4 absolute on Gemma 4B-shaped
  weights.
- The dispatch is simple: dequant → upload → `gemv` → return.
- Correctness gated by the same env-var pattern as `cuda-f32-baseline`.

**Non-Goals:**

- Custom CUDA dequant kernels (next-next sub-change).
- Hot-path optimisation. Dequant on host is the bottleneck; we
  accept it.
- Prefill batching (`q4k_matmul`). The default trait impl falls
  back to repeated `q4k_matvec`, which is correct.
- Q4_KF special handling. Q4_KF is the same on-disk layout as Q4_K
  with a different scaling convention; the dispatch in
  `quant_matvec`'s default impl already routes both to `q4k_matvec`,
  so this sub-change inherits Q4_KF support.

## Decisions

### D1 — Host-side dequant via `larql-models`

Three options:

1. **Host dequant via existing CPU dequantizers** → upload f32 →
   cuBLAS gemv. ~50 lines of glue. Correctness oracle is the CPU
   path itself.
2. **Custom CUDA dequant kernel** via cudarc NVRTC. ~200 lines per
   format, plus parity scaffolding.
3. **Vendor llama.cpp's ggml-cuda Q4_K kernels.** ~1500 lines of `.cu`
   code, plus a `build.rs` that runs `nvcc`.

Chose option 1. It's the smallest change that closes the dispatch
hole; the optimisation work has its own change ID. Trade-off: the
CPU dequant for a Gemma 4B FFN gate (10240 × 2560 Q4_K) takes
~6–8 ms — a meaningful fraction of decode latency but a fraction
that (a) is independent of the GPU work, (b) parallelises across
layers via existing rayon usage in the CPU dequantizers.

### D2 — Reuse `cuda::matmul::gemv` rather than introduce a new entry point

We already have `gemv_f32` and `matmul_transb`. The only thing the
quantised path needs is f32 weights pointing at host memory. So:

```rust
fn q4k_matvec(&self, q4k: &[u8], x: &[f32], n: usize, k: usize)
    -> Option<Vec<f32>> {
    let w = larql_models::quant::ggml::q4_k::dequantize_q4_k(q4k, n*k).ok()?;
    cuda::matmul::gemv(&self.drv, &w, x, n, k).ok()
}
```

No new public surface. No new dependency.

### D3 — `quant_matvec` default impl stays in charge

The `QuantMatVec` trait's default `quant_matvec` method already
matches on `QuantFormat` and dispatches to `q4k_matvec` /
`q6k_matvec` / etc. By implementing the per-format methods we
inherit `quant_matvec` for free.

For `Q4_0` and `Q8_0`, the default `quant_matvec` calls
`quantize_x_to_q8` then `q4_matvec` / `q8_matvec`. Implementing
`q4_matvec` (with the same dequant-then-gemv pattern, ignoring the
Q8 input quantisation since we go through f32 anyway) closes Q4_0.

### D4 — Capability bits

After this change:

| Capability | CudaBackend.supports |
|---|---|
| `Cuda` | true |
| `F32Gemv` | true (from baseline) |
| `QuantMatVec` | **true (new)** |
| `Q4VecMat` | **true (new)** — semantically "we can do Q4_0 matvec" |
| `FlashAttentionV2` | false (next change) |
| `KvCompressionRotorQuant` | false (later phase) |

## Risks / Trade-offs

- **Risk: CPU dequant is the bottleneck on this path.** Gemma 4B
  FFN gate dequant: ~6 ms × 30 layers × 2 (gate + up) ≈ 360 ms /
  token if repeated naïvely. → Mitigation: this is a correctness
  milestone. The follow-up `cuda-q4-matvec-fused` sub-change
  replaces dequant with a CUDA kernel; weights live as Q4 on
  device, dequant happens inline.
- **Risk: VRAM allocator churn.** Each call allocates a fresh f32
  buffer the size of the dequantised weight matrix. For Gemma 4B
  FFN gate that's ~25 MB allocated and freed per layer. → Cudarc's
  caching allocator should keep this cheap. We can measure and
  add a per-backend scratch arena if it shows up in profiling.
- **Risk: numeric divergence on close-call tokens.** Float-precision
  noise from one extra round-trip to f32 may flip top-1 on borderline
  tokens. → Tolerance bumped to 1e-3 absolute (vs 5e-4 in
  cuda-f32-baseline) to accommodate the dequant noise floor.
  Cosine ≥ 0.9999 still required.

## Migration Plan

Land. The library's existing trait methods accept the new `Some(_)`
returns transparently. No call-site changes needed in the rest of
the workspace.

Rollback: revert. `cuda-f32-baseline` still works.

## Open Questions

- **Q1: Should we also implement `q4_vecmat` (the down-projection
  scatter)?** It's `out[K] = activation[N] @ Q4[N, K]` — a transposed
  shape vs `q4_matvec`. Cheap to add via the same dequant-then-gemm
  pattern. **Recommendation:** yes, add it; it's another `None`
  fallback site that the inference path may take.
