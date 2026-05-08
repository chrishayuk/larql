# cuda-decode-backend tasks

## 1. CUDA DecodeBackend

- [x] 1.1 Add CUDA KV-cache state to `CudaBackend` with reset, truncate,
      preallocation, population, and length reporting.
- [x] 1.2 Implement correctness-first `decode_token` using existing CUDA
      QKV projection / fused-attention helpers and Q4/Q6 matvec dispatch.
- [x] 1.3 Implement correctness-first `prefill_q4` by repeatedly feeding
      sequence positions through the CUDA decode path while populating KV.
- [x] 1.4 Flip `Capability::DecodeToken` and `Capability::PrefillQ4` only
      after CUDA parity tests pass.
- [x] 1.5 Add CUDA decode parity tests against the CPU/Metal-independent
      scalar reference and run them with `LARQL_CUDA_AVAILABLE=1`.

## 2. CLI benchmark support

- [x] 2.1 Add `cuda` to `larql bench --backends`.
- [x] 2.2 Benchmark a real Gemma 3 4B vindex with `--backends cuda` and
      report prefill, decode `ms/tok`, and `tok/s`.

## 3. Attention service CUDA routing

- [x] 3.1 Teach attention-service prefill/decode handlers to select CUDA
      when `LARQL_BACKEND=cuda` or backend auto-selection returns CUDA.
- [x] 3.2 Add an attention-service smoke test that proves CUDA decode is
      selected inside the GPU container.

## 4. RotorQuant CUDA KV compression

- [x] 4.1 Implement CUDA RotorQuant quantize/dequantize wrappers for the
      active KV formats.
      - [x] FP16 -> packed RotorQuant device copy/quantize wrapper for
            Planar3/4 and Iso3/4.
      - [x] Matching CUDA dequantize wrapper.
- [x] 4.2 Flip `Capability::KvCompressionRotorQuant` on CUDA only after
      round-trip and throughput tests pass.
- [x] 4.3 Benchmark FP16 KV vs RotorQuant KV on the same prompt and report
      memory plus throughput.

      RTX 4090 benchmark, 16,384 x 320 values, 20 iterations:
      Planar3/Iso3 use 19.53% of FP16 KV bytes and run
      quantize+dequantize in ~0.393 ms/iter at ~0.985 cosine.
      Planar4/Iso4 use 26.56% of FP16 KV bytes and run in
      ~0.301 ms/iter at ~0.994 cosine. See
      `docs/cuda-rotorquant-status.md`.

## 5. Validation

- [x] 5.1 `openspec validate cuda-decode-backend --strict`.
- [x] 5.2 `cargo check -p larql-compute -p larql-cli -p larql-server --features cuda`.
- [x] 5.3 Focused CUDA tests pass with `LARQL_CUDA_AVAILABLE=1`.
- [x] 5.4 `make traceability-check` and `make openspec-validate` pass.
