## 1. Kernel Contract

- [x] 1.1 Confirm the canonical Q4_K block layout from `larql-models` CPU dequantizers and document the CUDA indexing assumptions in code comments.
- [x] 1.2 Add a direct CUDA Q4_K matvec kernel that consumes packed Q4_K bytes and an f32 input vector on device.
- [x] 1.3 Add a small Rust wrapper for launching the Q4_K kernel and returning f32 row outputs.

## 2. Backend Integration

- [x] 2.1 Route `CudaBackend::q4k_matvec` through the direct kernel by default.
- [x] 2.2 Preserve the previous host-dequant + cuBLAS path behind an explicit debug fallback.
- [x] 2.3 Update CUDA decode projection helpers to dispatch Q4_K-compatible matvecs through `QuantMatVec::q4k_matvec`.
- [x] 2.4 Keep Q4_KF/Q6_K behavior on the existing fallback path unless this slice requires them for bench completion.

## 3. Tests

- [x] 3.1 Add direct Q4_K parity tests for small deterministic matrices.
- [x] 3.2 Add direct Q4_K parity tests for FFN gate dimensions.
- [x] 3.3 Add direct Q4_K parity tests for LM-head dimensions or a production-like sampled LM-head shape if full shape is too slow for CI.
- [x] 3.4 Add a fallback parity test proving the host-dequant debug path still works.
- [x] 3.5 Add decode dispatch coverage proving Q4_K projections choose the quant matvec path.

## 4. Validation

- [x] 4.1 `openspec validate cuda-resident-q4k-matvec --strict` passes.
- [x] 4.2 `cargo test -p larql-compute --features cuda --test test_cuda_q4 -- --test-threads=1` passes.
- [x] 4.3 `cargo test -p larql-compute --features cuda --test test_cuda_decode -- --test-threads=1` passes.
- [x] 4.4 `cargo build --release -p larql-cli --features cuda` passes.
- [x] 4.5 A real CUDA `larql bench` pass completes and the result is recorded against the previous 9.25 s/token baseline.

Benchmark result (`LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose`):

- Previous baseline: prefill 43546.1ms, decode 9249.36ms/token, 0.1 tok/s, GPU fwd 7758.175ms, LM-head 1527.103ms.
- Direct Q4_K result: prefill 20192.6ms, decode 5102.07ms/token, 0.2 tok/s, GPU fwd 3600.307ms, LM-head 1510.047ms.
