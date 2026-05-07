## Why

`rotorquant-kernels` (shipped) ships a CPU reference for
quantize/dequantize across all four formats (Iso3 / Planar3 /
Iso4 / Planar4). The CPU path is correct (cosine ≥ 0.95
round-trip on Gemma 4B-shaped inputs) but slow — ~600 µs per row
on a 4090's CPU thread. For real serving, especially behind the
`engine-rotorquant-auto-compress` decorator that compresses on
every decode step, this is the bottleneck.

This sub-change ships PTX kernels for the hot quantize +
dequantize loops, dispatched via cudarc NVRTC (same plumbing the
softmax kernel uses — see `cuda-fused-attention`). The expected
speedup based on the rotation arithmetic alone is ~30× on RTX
4090 vs single-threaded CPU.

## What Changes

- ADD `crates/larql-rotorquant/src/cuda/kernels.rs` with two
  kernel sources:
  - `planar_quantize_kernel`: per-block 2D Givens rotation +
    scalar quantize. One thread per row, fanned out across rows.
  - `iso_quantize_kernel`: per-block 4D quaternion rotation +
    scalar quantize.
- ADD `cuda::dequant_kernels.rs` with two more:
  - `planar_dequantize_kernel`: inverse Givens + codebook lookup.
  - `iso_dequantize_kernel`: inverse quaternion + codebook lookup.
- ADD `cuda::launcher` module that compiles each kernel via
  cudarc NVRTC at first use and caches the resulting cubin under
  `$XDG_CACHE_HOME/larql/cudarc/<arch>/rotorquant-*.cubin`.
- MODIFY `crates/larql-rotorquant/src/lib.rs` to dispatch through
  the CUDA path when the `cuda` feature is on AND the underlying
  CUDA driver is reachable; otherwise transparently fall back to
  the CPU reference.
- ADD parity tests in `tests/cuda_parity.rs` (env-gated by
  `LARQL_CUDA_AVAILABLE=1`): every (format, K|V) round-trip
  matches the CPU reference within 1e-3 absolute on synthetic
  Gemma-shaped inputs.
- MODIFY `CudaBackend::supports` (in `larql-compute`) to flip
  `Capability::KvCompressionRotorQuant` to `true`.

This is non-breaking. The CPU path stays available; the CUDA
path is opt-in via feature flag.

## Capabilities

### New Capabilities

(none — implements scenarios already on
`kv-cache-rotorquant` via the parent change.)

### Modified Capabilities

- `kv-cache-rotorquant`: scenarios for production CUDA kernel
  performance (parent declared `<!-- test: unbacked -->`) get real
  test annotations.
- `compute-backend-traits`: `Capability::KvCompressionRotorQuant`
  scenario flips on for `CudaBackend`.

## Impact

- **Affected files**: new `crates/larql-rotorquant/src/cuda/`
  module (~400 lines incl. PTX strings); 4 PTX kernel sources;
  one test file; small change to `larql-compute::cuda::backend`
  to flip the capability bit.
- **Affected systems**: rotorquant + compute backend. Inference
  + server + router untouched.
- **Performance target**: ≥ 10× faster than CPU reference per row
  on RTX 4090 / Iso3 (the most-used format).
- **Out of scope**: f16-coded kernels; multi-stream dispatch;
  kernel fusion across (rotation + quantize + scaling).
