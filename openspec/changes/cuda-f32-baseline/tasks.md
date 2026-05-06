## 1. Driver and cuBLAS handle plumbing

- [ ] 1.1 Create `crates/larql-compute/src/cuda/driver.rs` with a
      `Driver` struct holding `Arc<CudaContext>` + `CudaBlas`.
- [ ] 1.2 `Driver::new()` calls `CudaContext::new(0)` and creates the
      cuBLAS handle; on failure maps `cudarc::DriverError` to the
      appropriate `CudaInitError` variant.
- [ ] 1.3 `Driver::device_buf_from(&[f32]) -> CudaSlice<f32>` and
      `Driver::to_host(CudaSlice<f32>) -> Vec<f32>` helpers.
- [ ] 1.4 `Driver::sync()` thin wrapper around the default stream's
      synchronize.
- [ ] 1.5 `CudaBackend::new()` uses `Driver::new()`; backend stores the
      driver as `Arc<Driver>`.

## 2. cuBLAS f32 GEMM and GEMV

- [ ] 2.1 Create `crates/larql-compute/src/cuda/matmul.rs`.
- [ ] 2.2 `gemm_f32(driver, op_a, op_b, a, b, m, n, k) -> Vec<f32>`
      using cuBLAS `Sgemm` with the row-major-via-column-major
      transposed-product identity. Calls `Driver::sync()` before
      copying the result back.
- [ ] 2.3 `gemv_f32(driver, weights, x, n, k) -> Vec<f32>` using
      cuBLAS `Sgemv` (or `Sgemm` with M=1; either works).
- [ ] 2.4 Replace `unimplemented!()` in `CudaBackend::matmul` and
      `matmul_transb` with calls to `gemm_f32`.
- [ ] 2.5 Override `MatMul::f32_gemv` on `CudaBackend` to return
      `Some(gemv_f32(...))`.
- [ ] 2.6 `CudaBackend::supports(Capability::F32Gemv)` returns `true`.

## 3. Kernel cache directory

- [ ] 3.1 `cuda::cache::cache_dir() -> PathBuf` honours
      `XDG_CACHE_HOME`; defaults to `$HOME/.cache/larql/cudarc/`.
- [ ] 3.2 `cuda::cache::ensure_initialised(driver)` creates the dir
      and writes a `.version` file with cudarc version + CUDA
      toolkit version + compute capability.
- [ ] 3.3 `Driver::new()` calls `ensure_initialised` once.

## 4. Parity tests

- [ ] 4.1 Create `crates/larql-compute/tests/test_cuda_f32.rs` with a
      `gpu_only!` macro that skips when `LARQL_CUDA_AVAILABLE` is
      unset.
- [ ] 4.2 `matmul_square_parity` — 256×256 by 256×256 random matmul.
- [ ] 4.3 `matmul_gemma4b_shape_parity` — 64×2560 by 2560×10240.
- [ ] 4.4 `matmul_transb_parity` — 32×4096 by 4096×4096 transposed.
- [ ] 4.5 `gemv_lm_head_parity` — 1×4096 by 4096×128256.
- [ ] 4.6 `gemv_returns_some` — `f32_gemv` returns `Some(_)`.
- [ ] 4.7 `sequential_matmul_no_contamination` — two RNG-distinct
      matmuls dispatched back-to-back match independent CPU runs.
- [ ] 4.8 `driver_init_succeeds_when_cuda_available` — explicit init
      check with the `LARQL_CUDA_AVAILABLE` gate.
- [ ] 4.9 Inline `cuda::backend::tests` get `driver_missing_returns_typed_error`
      and `supports_f32_gemv_after_baseline` covering paths that
      don't need a GPU.
- [ ] 4.10 `kernel_cache_dir_created` and
      `kernel_cache_respects_xdg_cache_home` (both run anywhere; just
      check the filesystem effect).

## 5. Doc + CI

- [ ] 5.1 Update `crates/larql-compute/src/lib.rs` doc table to drop
      "(Phase-1 stub)" caveat on f32 paths.
- [ ] 5.2 Add `make ci-cuda` target that runs `LARQL_CUDA_AVAILABLE=1
      cargo test -p larql-compute --features cuda` (skip in main
      `make ci`).
- [ ] 5.3 Update `.github/workflows/coverage.yml` (if needed) to set
      `LARQL_CUDA_AVAILABLE=1` on GPU runners.

## 6. Validation

- [ ] 6.1 `openspec validate cuda-f32-baseline --strict` passes.
- [ ] 6.2 `cargo check -p larql-compute --features cuda` passes.
- [ ] 6.3 `cargo test -p larql-compute --features cuda` passes (the
      gated tests no-op on a CPU box, run for real on a GPU box).
- [ ] 6.4 Existing CPU tests (`cargo test -p larql-compute`) untouched.
- [ ] 6.5 `make traceability-check` passes.
- [ ] 6.6 Commit references the parent change in the subject; archive
      after merge.
