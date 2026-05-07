## 1. PTX kernel sources

- [ ] 1.1 `crates/larql-rotorquant/src/cuda/kernels.rs` with
      `IsoN_QUANTIZE_PTX`, `PLANAR_QUANTIZE_PTX` static strings.
- [ ] 1.2 Same for `..._DEQUANTIZE_PTX`.
- [ ] 1.3 Codebook + rotation tables loaded into `__constant__`
      memory at module init.

## 2. Launcher infrastructure

- [ ] 2.1 `cuda::launcher` module with cached PTX module loaders,
      mirroring the pattern from `larql-compute::cuda::attn::softmax_function`.
- [ ] 2.2 `quantize_iso_cuda(driver, data, n_rows, head_dim) -> QuantizedKv`.
- [ ] 2.3 Same for planar + dequant.

## 3. Dispatch wiring

- [ ] 3.1 `lib.rs::quantize_k` / `quantize_v` route through
      cudarc-driven path when feature `cuda` enabled and backend
      successfully compiles kernels.
- [ ] 3.2 Fallback to CPU reference on any kernel error.

## 4. Capability flip

- [ ] 4.1 `larql-compute::cuda::backend::supports` returns true
      for `Capability::KvCompressionRotorQuant`.
- [ ] 4.2 Update `CudaBackend::tests::supports_*` to assert.

## 5. Tests

- [ ] 5.1 `tests/cuda_parity.rs` env-gated by `LARQL_CUDA_AVAILABLE=1`.
- [ ] 5.2 Iso3 K + V parity vs CPU.
- [ ] 5.3 Planar3 K + V parity vs CPU.
- [ ] 5.4 Iso4 + Planar4 parity (smaller config to keep test fast).
- [ ] 5.5 Throughput test asserts ≥ 10× speedup at Gemma 4B
      head shape.

## 6. Validation

- [ ] 6.1 `openspec validate rotorquant-cuda-kernels --strict` passes.
- [ ] 6.2 `cargo check --features cuda` clean.
- [ ] 6.3 `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-rotorquant --features cuda` passes.
- [ ] 6.4 `make traceability-check` and `make openspec-validate` pass.
