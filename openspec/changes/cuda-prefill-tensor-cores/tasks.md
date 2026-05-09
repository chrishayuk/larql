# cuda-prefill-tensor-cores — tasks

## 1. Conversion kernels

- [x] 1.1 `F32_F16_CONVERT_SRC` PTX with two `extern "C"` kernels:
      `f32_to_f16` (cvt.rn.f16.f32) and `f16_to_f32` (cvt.f32.f16).
- [x] 1.2 `elem::f32_to_f16_device` /
      `elem::f16_to_f32_device` wrappers — allocate the destination
      buffer, launch with `block_dim = 256`, return owned buffer.

## 2. f16 weight cache

- [x] 2.1 `q4k_f16_device_cache:
      Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<half::f16>>>>`
      on `CudaBackend`.
- [x] 2.2 Same for Q6_K (`q6k_f16_device_cache`).
- [x] 2.3 `with_q4k_f16_device_buf(host, n_elements, |w_dev| ...)` —
      first call dequant Q4_K → f32, downcast f32 → f16 on host,
      htod the f16 buffer; subsequent calls `Arc::clone` the
      cached entry.
- [x] 2.4 Same for `with_q6k_f16_device_buf`.

## 3. f16 cuBLAS GEMM

- [x] 3.1 `matmul::matmul_transb_device_inout_f16` —
      `cublasGemmEx`/`Gemm<half::f16>` with `CUDA_R_16F` inputs,
      `CUBLAS_COMPUTE_32F` accumulator, `CUBLAS_GEMM_DEFAULT`
      algo (Tensor Core dispatch on Ada/Ampere/Hopper).
- [x] 3.2 Cargo deps: `cudarc` `f16` feature + direct
      `half = "2"` dep gated on the `cuda` feature.

## 4. Prefill dispatch

- [x] 4.1 `decode::gemm_proj_seq` env-var gate: when
      `LARQL_CUDA_PREFILL_TENSOR_CORES=1`, route through the f16
      path (downcast x_seq, hgemm, upcast result); otherwise use
      the existing f32 path.
- [x] 4.2 Add `prefill_tensor_cores_enabled()` helper.

## 5. Tests + bench

- [x] 5.1 Existing parity suite passes with the f16 path enabled
      (139 lib tests + 56 integration tests, including
      `decode_token_phase1_matches_host_fallback`).
- [x] 5.2 Generated-text parity: `larql run` with and without
      `LARQL_CUDA_PREFILL_TENSOR_CORES=1` produces identical
      Gemma 3 4B output.
- [x] 5.3 Bench gate (RTX 4090, Gemma 3 4B Q4_K, 6-token prompt):
      prefill 18.0 ms → 10.7 ms (-40%, 5-run average).
      Decode unchanged within run-to-run noise.

## 6. Documentation + archive

- [x] 6.1 `LARQL_CUDA_PREFILL_TENSOR_CORES=1` documented in
      `proposal.md`.
- [ ] 6.2 Archive when reviewed.
