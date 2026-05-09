## Why

The direct CUDA Q4_K kernel cut real decode from 9.25 s/token to 5.10 s/token,
but it still uploads packed Q4_K weights on every matvec. The LM head alone
is roughly 295 MB of packed weights per token, so repeated host-to-device
copies remain a dominant artificial cost.

## What Changes

- Cache immutable packed Q4_K byte buffers in device memory inside
  `CudaBackend`.
- Key cached buffers by host slice identity and a small fingerprint so repeated
  matvecs for the same mmap-backed weights reuse the same `CudaSlice<u8>`.
- Route the direct Q4_K matvec launcher through the cache by default.
- Reduce Q4_K launch overhead by computing multiple output rows per CUDA block.
- Load `lm_head_q4.bin` in the `larql bench --backends cuda` Q4K path so
  LM-head actually uses the Q4_K backend route being benchmarked.
- Cache dequantized Q6_K matrices as f32 device buffers so Q6_K down
  projections stop paying CPU dequant and full f32 upload every token.
- Keep the existing `LARQL_CUDA_Q4K_HOST_DEQUANT=1` debug fallback unchanged.
- Re-run the same real CUDA benchmark and record whether LM-head and decode
  timing improve.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `compute-cuda-kernels`: CUDA Q4_K matvec should keep immutable packed weight
  buffers resident on device across repeated calls from the same backend.

## Impact

- **Affected code**: `crates/larql-compute/src/cuda/{backend,matmul,q4k_direct,quant_matvec}.rs`,
  `crates/larql-cli/src/commands/primary/bench_cmd.rs`, and focused CUDA Q4/Q6 tests.
- **APIs**: no public API changes.
- **Memory**: CUDA backends may retain packed Q4_K weights and dequantized Q6_K
  f32 matrices for the lifetime of the backend. This is expected for inference
  because vindex weights are immutable and mmap-backed.
- **Risk**: caching by host slice identity is only correct for immutable
  weights. Tests include a fingerprint in the key to avoid stale reuse when
  allocators recycle a pointer.
