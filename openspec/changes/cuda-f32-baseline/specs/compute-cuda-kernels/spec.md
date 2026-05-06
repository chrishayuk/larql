## ADDED Requirements

### Requirement: Driver and cuBLAS handle lifecycle

The `larql_compute::cuda::Driver` struct SHALL own a
`cudarc::driver::CudaContext` for device 0 and a
`cudarc::cublas::CudaBlas` handle, both created lazily on the first
backend init and reused across all subsequent kernel launches. The
driver MUST be `Send + Sync` so a single backend instance can be
shared across worker threads. `Drop` SHALL release the cuBLAS handle
before the context.

#### Scenario: Driver init succeeds on a CUDA host
- **WHEN** `CudaBackend::new()` is called on a host with a working CUDA driver
- **THEN** the call SHALL return `Ok(backend)` and the underlying `Driver` SHALL hold a non-null context and cuBLAS handle
<!-- test: larql_compute::test_cuda_f32::driver_init_succeeds_when_cuda_available -->

#### Scenario: Driver init returns a typed error when CUDA is missing
- **WHEN** `CudaBackend::new()` runs on a host with no CUDA driver loaded
- **THEN** the call SHALL return `Err(CudaInitError::DriverMissing(_))` rather than panic
<!-- test: larql_compute::cuda::backend::tests::driver_missing_returns_typed_error -->

#### Scenario: Capability set reflects what's actually wired
- **WHEN** the backend successfully initialises and `supports(Capability::F32Gemv)` is queried
- **THEN** the value SHALL be `true` (the f32 GEMV path is now real)
<!-- test: larql_compute::cuda::backend::tests::supports_f32_gemv_after_baseline -->

### Requirement: cuBLAS-backed f32 matmul matches CPU within 5e-4

`CudaBackend::matmul` and `CudaBackend::matmul_transb` SHALL produce
results matching the CPU backend within a maximum absolute element
difference of `5e-4` on Gemma 4B-shaped inputs (M, N, K up to
`{2560, 10240, 2560}` and `{1, 256000, 2560}` for LM-head shapes).

#### Scenario: Square matmul matches CPU
- **WHEN** a 256×256 by 256×256 random f32 matmul is computed on both backends
- **THEN** the maximum absolute difference SHALL be ≤ 5e-4
<!-- test: larql_compute::test_cuda_f32::matmul_square_parity -->

#### Scenario: Production-shape matmul matches CPU
- **WHEN** a 64×2560 by 2560×10240 matmul (Gemma 4B FFN gate-projection prefill batch) runs on both backends
- **THEN** the maximum absolute difference SHALL be ≤ 5e-4
<!-- test: larql_compute::test_cuda_f32::matmul_gemma4b_shape_parity -->

#### Scenario: matmul_transb matches CPU
- **WHEN** a 32×4096 input is multiplied by a 4096×4096 weight matrix in row-major-transposed form on both backends
- **THEN** the maximum absolute difference SHALL be ≤ 5e-4
<!-- test: larql_compute::test_cuda_f32::matmul_transb_parity -->

### Requirement: cuBLAS-backed f32 gemv at LM-head dimensions

`CudaBackend::f32_gemv` SHALL be implemented (not return `None`) and
produce outputs matching the CPU backend within a max absolute
difference of `5e-4` on LM-head shapes (`1×4096` by `4096×128256`
and similar). The backend SHALL advertise `Capability::F32Gemv`.

#### Scenario: LM-head gemv matches CPU
- **WHEN** a 1×4096 by 4096×128256 LM-head gemv is computed on both backends
- **THEN** the cosine similarity between outputs SHALL be ≥ 0.9999 and the max absolute difference SHALL be ≤ 5e-4
<!-- test: larql_compute::test_cuda_f32::gemv_lm_head_parity -->

#### Scenario: f32_gemv path returns Some
- **WHEN** `f32_gemv(w, x)` is called on the CUDA backend
- **THEN** the return value SHALL be `Some(_)` rather than `None`
<!-- test: larql_compute::test_cuda_f32::gemv_returns_some -->

### Requirement: Kernel cache directory is created and XDG-compliant

The CUDA backend SHALL ensure a kernel cache directory exists at
`$XDG_CACHE_HOME/larql/cudarc/` (defaulting to
`$HOME/.cache/larql/cudarc/`) on first init, and SHALL write a
`.version` marker file recording the cudarc version, the CUDA
toolkit version, and the GPU compute capability. Custom kernels in
later sub-changes will populate this directory with cached PTX.

#### Scenario: Cache directory exists after first init
- **WHEN** `CudaBackend::new()` is called for the first time
- **THEN** `$XDG_CACHE_HOME/larql/cudarc/` SHALL exist and contain a `.version` file
<!-- test: larql_compute::test_cuda_f32::kernel_cache_dir_created -->

#### Scenario: XDG_CACHE_HOME is honoured
- **WHEN** `XDG_CACHE_HOME=/tmp/test-cache` is set in the environment and the backend is constructed
- **THEN** the cache directory SHALL be `/tmp/test-cache/larql/cudarc/`
<!-- test: larql_compute::test_cuda_f32::kernel_cache_respects_xdg_cache_home -->

### Requirement: Kernel launches MUST synchronise before returning

The wrapper SHALL call `CudaContext::default_stream().synchronize()` (or equivalent) after every cuBLAS dispatch in `gemm_f32` / `gemv_f32` before copying the result back to the host. This guarantees the host buffer is populated when the call returns, matching the CPU backend's synchronous semantics.

#### Scenario: Sequential matmul calls produce independent results
- **WHEN** two different matmuls are dispatched back-to-back without manual sync
- **THEN** the second result SHALL not be contaminated by the first (verified by random-input parity)
<!-- test: larql_compute::test_cuda_f32::sequential_matmul_no_contamination -->
