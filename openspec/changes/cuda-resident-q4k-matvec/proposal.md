## Why

`larql bench --backends cuda` now runs end to end on the RTX host, but the
measured decode rate is not useful yet:

- prefill: 43.5 s
- decode: 9.25 s/token
- throughput: 0.1 tok/s

The current CUDA Q4/Q6 path is correctness-first. It dequantizes quantized
weights on the CPU, uploads a temporary f32 matrix, then calls cuBLAS for each
matvec. That means the GPU backend pays CPU dequant, PCIe upload, and device
allocation cost repeatedly for hot matrices such as gate/up/down/wo and the
LM head. The result is functionally correct but predictably slower than the
CPU path in real inference.

## What Changes

- Add a direct CUDA Q4_K matvec path that consumes packed Q4_K blocks and an
  f32 input vector on the device, accumulating one f32 output per row without
  materializing the full f32 weight matrix.
- Route `CudaBackend::q4k_matvec` through the direct kernel by default, keeping
  the existing host-dequant + cuBLAS implementation as an explicit fallback
  for debugging and parity checks.
- Use the direct path from decode-time quantized projections where the weight
  format is Q4_K.
- Add focused correctness tests comparing direct CUDA Q4_K matvec against the
  existing CPU dequant reference at production-like shapes.
- Add a manual benchmark acceptance scenario documenting the expected
  improvement from eliminating repeated host dequant and upload.

This change is intentionally scoped to Q4_K. Q4_KF, Q6_K, resident KV cache
storage, and fully persistent per-layer device weight residency remain follow-up
work unless needed to complete the first benchmark pass.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `compute-cuda-kernels`: Q4_K matvec must use a direct CUDA kernel by default
  instead of the earlier host-dequant bridge.
- `kv-cache-benchmark-strategies`: CUDA benchmark output must include enough
  timing detail to confirm whether the resident/direct Q4_K path is active and
  whether it materially improves the real `larql bench` run.

## Impact

- **Affected files**: `crates/larql-compute/src/cuda/*`,
  `crates/larql-compute/tests/test_cuda_q4.rs`, decode integration in
  `crates/larql-compute/src/cuda/decode.rs`, and benchmark docs/spec metadata.
- **Performance**: removes the largest known artificial cost in CUDA Q4_K
  matvec by avoiding full f32 matrix materialization and upload per call.
- **Compatibility**: CPU and Metal behavior are unchanged. CUDA keeps the old
  host-dequant implementation as a fallback path.
- **Risk**: a first direct kernel may be bandwidth-bound or less optimized than
  llama.cpp/ggml CUDA kernels. That is acceptable if it proves the architecture
  and improves the bench enough to guide the next optimization.
