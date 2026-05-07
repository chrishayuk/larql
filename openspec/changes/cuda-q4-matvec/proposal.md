## Why

Second sub-change of [`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md).
With f32 GEMM/GEMV running on the RTX 4090 (`cuda-f32-baseline`),
inference still falls off the GPU the moment a quantised weight is
touched — Gemma's gate / up / down projections and the LM head are
all Q4_K / Q4_KF / Q6_K. Until the CUDA backend can dispatch those
formats, `default_backend()` on Linux returns CUDA but every layer's
expert path silently degrades to `None` and the upper layer falls
back to repeated CPU dequant. Wiring at least a correct (if not yet
fastest) CUDA path for Q4_0 / Q4_K / Q6_K closes the dispatch hole
and unblocks `cuda-fused-attention`.

## What Changes

- ADD `larql_compute::cuda::dequant` — host-side dispatch over the
  existing CPU dequantizers in `larql_models::quant::ggml`:
  `dequant_q4_0`, `dequant_q4_k`, `dequant_q6_k`. Returns `Vec<f32>`
  ready to upload to the device.
- ADD `cuda::matmul::gemv_dequant` — convenience helper that runs
  dequant on host, uploads, calls cuBLAS gemv, syncs, and returns
  the host vector.
- MODIFY `CudaBackend::q4_matvec`, `q4k_matvec`, `q6k_matvec` to
  return `Some(_)` via the dequant-then-gemv path. Override
  `quant_matvec` only if the default dispatch isn't sufficient
  (initially it is; the sub-trait methods are all that
  `default_backend()` needs).
- MODIFY `CudaBackend::supports` to advertise `Capability::Q4VecMat`
  and `Capability::QuantMatVec`. `Capability::F32Gemv` is already on.
- ADD parity tests in `crates/larql-compute/tests/test_cuda_q4.rs`
  comparing CUDA outputs against the CPU `q4k_matvec` /
  `q6k_matvec` results on Gemma 4B-shaped inputs.

This is non-breaking. CPU + Metal paths untouched. The dequant
dispatch is correctness-first; production-grade fused matvec
kernels (one cuBLAS call avoided per dispatch) are a future
optimisation tracked in tasks.md as Phase 2c.

## Capabilities

### New Capabilities

(none — implements scenarios that already exist on the
`compute-cuda-kernels` capability via the parent change.)

### Modified Capabilities

- `compute-cuda-kernels`: scenarios for Q4_K / Q4_KF / Q6_K matvec
  (already declared in the parent delta with `<!-- test: unbacked -->`)
  are now backed by real test annotations. Two new requirements
  cover the dequant-then-gemv contract and the capability bits.

## Impact

- **Affected files**: `crates/larql-compute/src/cuda/{matmul,
  backend,dequant}.rs`; new
  `crates/larql-compute/tests/test_cuda_q4.rs`. No Cargo.toml
  changes — the dequantizers come from `larql-models`, which is
  already a dep.
- **Affected systems**: CUDA-only. CPU build identical. Metal
  build identical.
- **Performance**: dequant runs on CPU once per matmul, paying ~30%
  of total dispatch time on Gemma 4B (10240 features × 2560 hidden,
  256-element blocks). For correctness-first this is fine. A
  follow-up sub-change (`cuda-q4-matvec-fused`) will land custom
  CUDA dequant kernels and cut this to single-digit percentage.
- **VRAM cost**: matmul allocates a temporary device buffer for the
  dequantised weights (`num_rows * hidden * 4 bytes`), which is
  freed at end of dispatch. Gemma 4B FFN is ~100 MB peak; well
  within the 24 GB budget.
- **Out of scope**: prefill batching (`q4k_matmul` returning
  Some), Q5 family, custom CUDA dequant kernels, Q4 top-1 / top-K
  arg-max for the LM head — all in Phase 2c follow-ups.
