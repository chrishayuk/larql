## ADDED Requirements

### Requirement: CUDA backend implements ComputeBackend trait family

A `larql_compute::cuda::CudaBackend` SHALL implement every supertrait of
`ComputeBackend` (`MatMul`, `QuantMatVec`, `DecodeBackend`,
`Capability`). The backend MUST be constructible only when the `cuda`
feature is enabled, the host kernel exposes a CUDA driver, and at
least one CUDA-capable GPU is reachable. Construction failures SHALL
return a typed error (`CudaInitError::DriverMissing`,
`CudaInitError::NoDevices`, `CudaInitError::ToolkitMismatch`) rather
than panic.

#### Scenario: Backend construction succeeds on a working CUDA host
- **WHEN** `CudaBackend::new()` is called on a Linux box with a healthy CUDA runtime
- **THEN** the call SHALL return `Ok(backend)` and `backend.name()` SHALL contain "cuda"
<!-- test: unbacked -->

#### Scenario: Backend construction fails clearly when CUDA is missing
- **WHEN** `CudaBackend::new()` is called on a host with no CUDA runtime
- **THEN** the call SHALL return `Err(CudaInitError::DriverMissing)`
<!-- test: unbacked -->

#### Scenario: Capability bits reflect compiled feature set
- **WHEN** `backend.supports(Capability::Cuda)` is read
- **THEN** the result SHALL be `true`
<!-- test: unbacked -->

### Requirement: cuBLAS-backed f32 matmul and gemv

The CUDA backend SHALL implement `MatMul::matmul_f32`,
`MatMul::matmul_transb_f32`, and `MatMul::gemv_f32` via cuBLAS. Inputs
SHALL be transferred to GPU memory via the standard cudarc
`HostToDevice` / `DeviceToHost` paths and outputs SHALL be returned
in the original ndarray layout.

#### Scenario: f32 matmul matches the CPU reference
- **WHEN** a 256×512×384 random f32 matmul is computed on both CPU and CUDA backends
- **THEN** the maximum absolute element difference SHALL be below 5e-4
<!-- test: unbacked -->

#### Scenario: f32 gemv matches the CPU reference for LM-head shape
- **WHEN** a 1×4096 by 4096×128256 LM-head gemv is computed on both backends
- **THEN** the cosine similarity between outputs SHALL be ≥ 0.9999
<!-- test: unbacked -->

### Requirement: Quantised matvec for Q4_0, Q4_K, Q4_KF, Q6_K

The CUDA backend SHALL implement `QuantMatVec::quant_matvec` for
formats `Q4_0`, `Q4_K`, `Q4_KF`, and `Q6_K` with kernels at parity
to the CPU implementations. Each format SHALL pass the existing
parity tests in `larql-compute::test_q4k_parity` when the feature
is enabled.

#### Scenario: Q4_K matvec parity with CPU at production dimensions
- **WHEN** a Gemma 4B-shaped Q4_K matvec (`hidden=2560`, `intermediate=10240`) is computed on both backends
- **THEN** the maximum cosine deviation per output element SHALL be ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: Q4_KF FFN matvec selected when manifest declares it
- **WHEN** a vindex with Q4_KF FFN weights is loaded and `quant_matvec` is invoked
- **THEN** the CUDA backend SHALL dispatch the Q4_KF kernel (not Q4_K) and the result SHALL match a CPU Q4_KF run
<!-- test: unbacked -->

### Requirement: Fused decode-time attention kernel

The CUDA backend SHALL implement
`DecodeBackend::decode_attention` with a fused kernel that combines
QK norm, RoPE rotation, KV cache append, scaled dot-product, softmax,
and value aggregation in a single launch. The kernel MUST handle
GQA, partial RoPE, and softcap variants per the existing CPU and
Metal contracts.

#### Scenario: GQA fused attention matches CPU reference
- **WHEN** a 32-token Gemma 3 4B prefill is run through `decode_attention` on both CPU and CUDA backends with the same RNG-seeded input
- **THEN** the max absolute hidden-state difference at the final layer SHALL be below 5e-3
<!-- test: unbacked -->

#### Scenario: Softcap is honoured when the architecture declares it
- **WHEN** a Gemma 2 attention block with softcap = 50 is run through CUDA decode
- **THEN** logit values SHALL be bounded by `tanh(x/50) * 50` per the published Gemma 2 spec
<!-- test: unbacked -->

### Requirement: cudarc-driven NVRTC compilation for non-vendored kernels

Custom kernels not vendored from upstream SHALL be compiled at backend
init via NVRTC, cached in `~/.cache/larql/cudarc/<toolkit>-<arch>/`,
and re-used on subsequent runs. A cache miss SHALL incur ≤ 200 ms of
extra startup; a cache hit SHALL incur ≤ 5 ms.

#### Scenario: Kernel cache miss falls within the latency budget
- **WHEN** the cudarc cache is empty and the CUDA backend boots
- **THEN** total backend-init time SHALL be ≤ 500 ms on an RTX 4090 / CUDA 13
<!-- test: unbacked -->

#### Scenario: Kernel cache hit accelerates subsequent runs
- **WHEN** the backend boots a second time with the cache populated
- **THEN** total backend-init time SHALL be ≤ 100 ms
<!-- test: unbacked -->

### Requirement: GPU architecture targeting

The build MUST target sm_70 minimum and tune for sm_89 (Ada
Lovelace). The `LARQL_CUDA_ARCH` environment variable SHALL override
the runtime PTX-JIT target. The vendored kernel build (`build.rs`)
SHALL emit cubin for the architectures listed in
`crates/larql-rotorquant/cuda/ARCHS.txt` and embed PTX as a fallback.

#### Scenario: Backend boots on sm_86 hardware
- **WHEN** `CudaBackend::new()` runs on an RTX 3090
- **THEN** init SHALL succeed and `device_info()` SHALL contain "sm_86"
<!-- test: unbacked -->

### Requirement: Default-backend precedence honours CUDA on Linux

`larql_compute::default_backend()` SHALL return a `CudaBackend` when
all of the following hold: the build was compiled with `--features cuda`,
the runtime detects a healthy CUDA driver, and `LARQL_BACKEND` is not
set to a different value. Otherwise it SHALL fall back to Metal (on
macOS, with `--features metal`) or CPU.

#### Scenario: CUDA wins on Linux + cuda feature
- **WHEN** the binary is built with `--features cuda` and run on the dev box
- **THEN** `default_backend().name()` SHALL contain "cuda"
<!-- test: unbacked -->

#### Scenario: LARQL_BACKEND override forces CPU
- **WHEN** `LARQL_BACKEND=cpu` is set in the environment
- **THEN** `default_backend().name()` SHALL contain "cpu" regardless of feature flags
<!-- test: unbacked -->
