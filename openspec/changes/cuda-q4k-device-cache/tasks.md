## 1. Device Cache

- [x] 1.1 Add a backend-local packed Q4_K device buffer cache keyed by immutable host slice identity and fingerprint.
- [x] 1.2 Route direct Q4_K matvec launches through the cached device buffer.
- [x] 1.3 Preserve `LARQL_CUDA_Q4K_HOST_DEQUANT=1` as a cache-bypassing fallback.
- [x] 1.4 Update the direct Q4_K kernel to compute multiple rows per CUDA block.
- [x] 1.5 Load `lm_head_q4.bin` in the Q4K `larql bench` path so CUDA LM-head uses Q4_K matvec.
- [x] 1.6 Cache dequantized Q6_K matrices as f32 device buffers and route Q6_K matvec through cached device GEMV.

## 2. Tests

- [x] 2.1 Add CUDA test coverage proving repeated Q4_K matvec calls reuse the cache and remain numerically correct.
- [x] 2.2 Run the existing Q4_K direct and fallback parity tests.
- [x] 2.3 Add CUDA test coverage proving repeated Q6_K matvec calls reuse the cache and remain numerically correct.

## 3. Validation

- [x] 3.1 `openspec validate cuda-q4k-device-cache --strict` passes.
- [x] 3.2 `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --test test_cuda_q4 -- --test-threads=1 --nocapture` passes.
- [x] 3.3 `cargo build --release -p larql-cli --features cuda` passes.
- [x] 3.4 A real CUDA `larql bench` pass completes and the result is recorded against the 5.10 s/token baseline.

Benchmark result (`LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose`):

- Previous direct-Q4K baseline: prefill 20192.6ms, decode 5102.07ms/token, GPU fwd 3600.307ms, LM-head 1510.047ms.
- After loading `lm_head_q4.bin`: decode 3640.62ms/token, GPU fwd 3638.832ms, LM-head 3.392ms.
- After Q6_K device f32 cache: prefill 1155.1ms, decode 162.72ms/token, 6.1 tok/s, GPU fwd 160.820ms, LM-head 1.888ms.
