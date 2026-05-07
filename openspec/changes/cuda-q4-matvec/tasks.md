## 1. CUDA dequant dispatch

- [ ] 1.1 Add `crates/larql-compute/src/cuda/dequant.rs` with three
      shim fns: `dequantize_q4_0`, `dequantize_q4_k`, `dequantize_q6_k`,
      each calling into `larql_models::quant::ggml::*`.
- [ ] 1.2 Re-export from `cuda::mod` (private to the crate).

## 2. matvec dispatch path

- [ ] 2.1 In `cuda::matmul`, add `pub(crate) gemv_dequant_q4k` etc.
      that call dequant + `gemv` with the right shapes. Or implement
      the full dispatch in `backend.rs`.
- [ ] 2.2 Override `QuantMatVec::q4_matvec` on `CudaBackend`.
- [ ] 2.3 Override `QuantMatVec::q4k_matvec`.
- [ ] 2.4 Override `QuantMatVec::q6k_matvec`.
- [ ] 2.5 Update `CudaBackend::supports` to add `QuantMatVec` and
      `Q4VecMat` to the supported set.

## 3. Parity tests

- [ ] 3.1 Create `crates/larql-compute/tests/test_cuda_q4.rs` with
      `LARQL_CUDA_AVAILABLE` gating identical to test_cuda_f32.
- [ ] 3.2 `q4_0_matvec_parity` — synthesise a Q4_0 weight buffer
      via the existing CPU quantiser, run on CPU + CUDA, compare.
- [ ] 3.3 `q4k_matvec_ffn_gate_parity` — Gemma 4B-shaped 10240×2560
      Q4_K matvec.
- [ ] 3.4 `q4k_matvec_lm_head_parity` — Llama-class 128256×4096.
- [ ] 3.5 `q6k_matvec_lm_head_parity` — same shape, Q6_K weights.
- [ ] 3.6 `quant_matvec_dispatches_to_q4k` — dispatch via the umbrella
      method and via the direct method, assert byte-equal output.

## 4. Inline backend test

- [ ] 4.1 Add `cuda::backend::tests::supports_q4_matvec_after_q4_baseline`
      that runs without GPU and asserts the static capability set.

## 5. Validation

- [ ] 5.1 `openspec validate cuda-q4-matvec --strict` passes.
- [ ] 5.2 `cargo check --workspace --features 'larql-cli/cuda'` passes.
- [ ] 5.3 `cargo test -p larql-compute --features cuda --lib` passes.
- [ ] 5.4 `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute
      --features cuda --test test_cuda_q4` passes (5 tests).
- [ ] 5.5 `make traceability-check` and `make openspec-validate` pass.
- [ ] 5.6 Commit references the parent change in subject.
