## 1. Softmax PTX kernel

- [ ] 1.1 `crates/larql-compute/src/cuda/ptx_softmax.rs` with a
      `SOFTMAX_SRC: &str` constant containing the CUDA C source.
- [ ] 1.2 `cuda::attn::softmax::compile_or_load(drv) -> Module` that
      runs cudarc NVRTC, caches to disk under
      `cache_dir()/<arch>/softmax.cubin`, and reuses on warm start.
- [ ] 1.3 `cuda::attn::softmax::launch_inplace(drv, mod, x, rows,
      cols, scale, softcap, causal)` that issues the dispatch.

## 2. decode_attention helper

- [ ] 2.1 `cuda::attn::AttentionOpts` struct (causal, softcap).
- [ ] 2.2 `cuda::attn::decode_attention(drv, q, k, v, ...) ->
      Result<Vec<f32>, CudaInitError>` chaining cuBLAS gemm →
      softmax → cuBLAS gemm with one terminal sync.

## 3. Capability bit + tests

- [ ] 3.1 Update `CudaBackend::supports` to add `FlashAttentionV2`.
- [ ] 3.2 New inline test
      `cuda::backend::tests::supports_fa2_after_fused_attention`.
- [ ] 3.3 New `crates/larql-compute/tests/test_cuda_attn.rs` with
      gpu-or-skip + naive scalar reference + 5 parity tests:
      - `softmax_small_parity`
      - `softmax_long_row_parity`
      - `softmax_causal_mask`
      - `softmax_softcap_50`
      - `decode_attention_small_parity`
      - `decode_attention_gemma4b_head_parity`

## 4. Validation

- [ ] 4.1 `openspec validate cuda-fused-attention --strict` passes.
- [ ] 4.2 `cargo check -p larql-compute --features cuda` clean.
- [ ] 4.3 `cargo test -p larql-compute --features cuda --lib` passes.
- [ ] 4.4 `LARQL_CUDA_AVAILABLE=1 cargo test --features cuda
      --test test_cuda_attn` 6 tests pass.
- [ ] 4.5 traceability + openspec-validate green.
