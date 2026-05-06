## Why

First sub-change of the parent
[`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md): the
CUDA backend Phase-1 stub is wired in but every dispatch path is
`unimplemented!()`. To call anything else "running on CUDA" the f32
GEMM/GEMV path needs to be real — every higher kernel (Q4 matvec,
fused attention, RotorQuant) builds on the host↔device plumbing,
cudarc driver context, and cuBLAS handle that this change introduces.
Landing it as a self-contained sub-change makes it bisectable, lets us
declare an early "first kernel running" milestone, and avoids the
sprawl risk of a single 2-week PR.

## What Changes

- ADD a real `larql_compute::cuda::Driver` that lazily creates a
  `cudarc::driver::CudaContext`, owns a `cublas::CudaBlas` handle, and
  exposes a small set of device-buffer helpers (`device_buf_from`,
  `to_host`).
- ADD device matmul helpers `gemm_f32` and `gemv_f32` in
  `cuda::matmul` that reconcile ndarray's row-major slices with
  cuBLAS's column-major API via the standard transposed-GEMM identity
  (`A B = (B^T A^T)^T`).
- MODIFY `CudaBackend::matmul` and `CudaBackend::matmul_transb` to
  call those helpers instead of `unimplemented!()`. Override
  `MatMul::f32_gemv` to use the cuBLAS gemv path and advertise
  `Capability::F32Gemv`.
- MODIFY `CudaBackend::new()` to actually probe for a CUDA driver via
  `CudaContext::new(0)`; on success record the device name + arch.
  Failure modes map to `CudaInitError::DriverMissing` /
  `::NoDevices` / `::ToolkitMismatch` per the parent design.
- ADD a kernel-cache directory under `$XDG_CACHE_HOME/larql/cudarc/`
  keyed by `(driver_version, gpu_arch, code_hash)` for future PTX
  modules. This change ships the cache plumbing but doesn't yet load
  any custom kernels — Q4 matvec is where it matters.
- ADD parity tests under `crates/larql-compute/tests/test_cuda_f32.rs`
  that compare CUDA outputs against the existing CPU `matmul` /
  `matmul_transb` / `f32_gemv` implementations on Gemma 4B-shaped
  matrices, gated by `cfg(feature = "cuda")` and `LARQL_CUDA_AVAILABLE=1`
  so non-GPU CI keeps passing.
- MODIFY the doc table in `lib.rs` to drop the "(Phase-1 stub)" caveat
  on f32 paths.

This is non-breaking. No existing test changes meaning. The CPU and
Metal backends are untouched.

## Capabilities

### New Capabilities

(none — this sub-change implements `compute-cuda-kernels`, which
already exists as a delta in the parent change.)

### Modified Capabilities

- `compute-cuda-kernels`: the existing parent-change delta declares
  cuBLAS f32 GEMM/GEMV requirements with `<!-- test: unbacked -->`
  annotations. This sub-change attaches real test references to those
  scenarios and adds two new requirements covering the driver/handle
  lifecycle and the kernel-cache directory layout.

## Impact

- **Affected files**: `crates/larql-compute/src/cuda/` (new modules:
  `driver.rs`, `matmul.rs`; existing: `backend.rs`, `error.rs`,
  `mod.rs`); `crates/larql-compute/tests/test_cuda_f32.rs` (new);
  `crates/larql-compute/Cargo.toml` (no dep changes — cudarc 0.19
  already pinned).
- **Affected systems**: only the CUDA path. CPU + Metal builds
  byte-identical to pre-change. The Phase-1 stub's existing inline
  tests (`name_is_cuda`, `supports_cuda_capability`) keep passing.
- **CI**: a new `cargo test -p larql-compute --features cuda` line in
  the GPU-runner job; existing CPU-runner jobs unchanged.
- **Risk**: `f32_gemv` advertised as a capability changes
  `default_backend()` selection consequences when callers branch on
  `supports(Capability::F32Gemv)`. Mitigation — the CPU backend
  already advertises `F32Gemv`, so all branches that exist today
  remain on the same path.
- **Out of scope**: Q4 / Q6 matvec (next sub-change
  `cuda-q4-matvec`); fused attention (`cuda-fused-attention`);
  RotorQuant integration (later phase).
