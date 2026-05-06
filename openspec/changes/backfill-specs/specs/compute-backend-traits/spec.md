## ADDED Requirements

### Requirement: Trait split into focused sub-traits

The `larql_compute::ComputeBackend` umbrella trait SHALL be expressed as
the sum of four narrower sub-traits — `MatMul`, `QuantMatVec`,
`DecodeBackend`, plus the umbrella's own metadata methods (`name`,
`device_info`, `as_any`, `supports`) — so that adding a new backend or
extending an existing one only requires touching the relevant
sub-trait. Every sub-trait MUST live in its own module under
`crates/larql-compute/src/backend/` and MUST be re-exported from
`backend::mod` so callers continue to bind only to
`&dyn ComputeBackend`.

#### Scenario: ComputeBackend supertraits the four sub-traits
- **WHEN** a backend type implements `ComputeBackend`
- **THEN** the type SHALL also satisfy `MatMul + QuantMatVec + DecodeBackend + Send + Sync` and SHALL provide `name`, `device_info`, `supports`, and `as_any`
<!-- test: larql_compute::test_pipeline_and_moe::cpu_backend_is_dyn_compatible -->
<!-- test: larql_compute::test_pipeline_and_moe::cpu_backend_name_is_nonempty -->
<!-- test: larql_compute::test_pipeline_and_moe::cpu_backend_device_info_is_nonempty -->

#### Scenario: Default backend factory exposes a usable name
- **WHEN** `larql_compute::default_backend()` is invoked on any host platform
- **THEN** the returned `Box<dyn ComputeBackend>` SHALL report a non-empty `name()` string
<!-- test: larql_compute::test_correctness::default_backend_has_name -->
<!-- test: larql_compute::test_pipeline_and_moe::default_backend_name_is_nonempty -->

### Requirement: Capability probe replaces `Option<…>` polling

Backends SHALL declare what they accelerate via the
`Capability` enum and the `ComputeBackend::supports(cap)` method.
Callers MUST be able to branch on `supports(cap)` before calling the
underlying method instead of pattern-matching on `None` returns. The
default `supports` implementation MUST return `false` for every
capability so that adding a new capability variant does not silently
flip behaviour for existing backends.

#### Scenario: CPU backend's truth table matches its real surface
- **WHEN** `supports(cap)` is queried on `CpuBackend` for every `Capability` variant
- **THEN** the returned booleans SHALL exactly match the set of methods CpuBackend implements (Q4 fast paths true; GPU-only capabilities such as `F32Gemv`, `F16Gemv`, `FullPipelineQ4`, `DecodeToken`, `DecodeMoe`, `DecodeProfile`, `PrefillQ4` false)
<!-- test: larql_compute::test_correctness::cpu_backend_capability_truth_table -->

#### Scenario: Default `supports` impl returns false everywhere
- **WHEN** a custom backend that does not override `supports` is queried for any `Capability`
- **THEN** the call SHALL return `false`
<!-- test: larql_compute::test_backend_matmul_quant::default_supports_returns_false -->

#### Scenario: Metal backend reports the GPU truth table
- **WHEN** `supports(cap)` is queried on `MetalBackend`
- **THEN** the GPU-only capabilities SHALL report `true` and the truth table SHALL agree with the trait method coverage advertised by the Metal trait_impl modules
<!-- test: larql_compute::test_kernel_handle_contract::metal_backend_capability_truth_table -->

### Requirement: dyn-compat preserved across all sub-traits

Every method on `ComputeBackend` and its four sub-traits SHALL be
object-safe so the trait can be used as `&dyn ComputeBackend`. Methods
that take or return arrays MUST use `ArrayView2` / `&[u8]` / `&[f32]`
slice types rather than generic parameters, and methods returning
multiple buffers MUST use concrete `Vec<…>` / tuple types so the
v-table compiles unchanged.

#### Scenario: CpuBackend can be wrapped as a trait object
- **WHEN** `&CpuBackend as &dyn ComputeBackend` is constructed and a downstream caller dispatches `matmul_transb`, `q4k_matvec`, and `quant_matvec` through the trait object
- **THEN** every dispatch SHALL succeed without monomorphisation errors and SHALL return values matching the direct-call results
<!-- test: larql_compute::test_pipeline_and_moe::cpu_backend_is_dyn_compatible -->
<!-- test: larql_compute::backend::helpers::tests::dot_proj_gpu_some_backend_matches_fallback -->
<!-- test: larql_compute::backend::helpers::tests::matmul_gpu_some_backend_matches_fallback -->

#### Scenario: Helper passes `None` backend through to ndarray BLAS
- **WHEN** `dot_proj_gpu(a, b, None)` or `matmul_gpu(a, b, None)` is called with a `None` backend
- **THEN** the helper SHALL execute the matmul on CPU via `ndarray` and SHALL return a result equal (within 1e-6) to the direct `ndarray::dot` reference
<!-- test: larql_compute::backend::helpers::tests::dot_proj_gpu_none_backend_uses_ndarray -->
<!-- test: larql_compute::backend::helpers::tests::matmul_gpu_none_backend_uses_ndarray -->

### Requirement: Format-dispatched and pre-quantised matvec entry points

`QuantMatVec::quant_matvec` SHALL be the convenience entry point for
callers with f32 input — it MUST dispatch on the `QuantFormat` variant
to the right per-format helper (`q4_matvec`, `q4k_matvec`, `q6k_matvec`)
and MUST internally quantise to Q8 only when the format requires it
(Q4_0 / Q8_0). `quant_matvec_q8_input` SHALL accept already-quantised
Q8 input on the hot decode path, dispatch directly for Q4_0 / Q8_0
formats, and dequantise back to f32 only when the underlying shader
takes f32 input (Q4_K / Q4_KF / Q6_K). Both methods MUST return `None`
for formats the backend does not implement and for non-quantised
formats (BF16, F16, F32).

#### Scenario: q8-input fast path matches direct q4_matvec for Q4_0
- **WHEN** `quant_matvec_q8_input(QuantFormat::Q4_0, …)` is called with the same Q8 buffers a caller would feed to `q4_matvec`
- **THEN** the returned vector SHALL be identical to the direct `q4_matvec` call
<!-- test: larql_compute::test_correctness::cpu_quant_matvec_q8_input_q4_0_matches_q4_matvec -->

#### Scenario: quant_matvec dispatches to the format-specific helper
- **WHEN** `quant_matvec(QuantFormat::Q4_K, …)` and `quant_matvec(QuantFormat::Q6_K, …)` are called against the same weights and input the per-format helpers consume
- **THEN** results SHALL match the direct `q4k_matvec` / `q6k_matvec` outputs element-wise
<!-- test: larql_compute::test_correctness::cpu_quant_matvec_matches_per_format_helpers -->
<!-- test: larql_compute::test_backend_matmul_quant::quant_matvec_q4k_dispatches_to_q4k_kernel -->
<!-- test: larql_compute::test_backend_matmul_quant::quant_matvec_q4kf_dispatches_same_as_q4k -->
<!-- test: larql_compute::test_backend_matmul_quant::quant_matvec_q6k_dispatches_to_q6k_kernel -->

#### Scenario: Q8 input path dequantises before f32-shader dispatch
- **WHEN** `quant_matvec_q8_input(QuantFormat::Q4_K, …)` or `quant_matvec_q8_input(QuantFormat::Q6_K, …)` is invoked on a backend whose Q4_K/Q6_K shader expects f32 input
- **THEN** the helper SHALL internally dequantise Q8 → f32 and then call the format-specific helper, returning a result within Q8 quant noise of the direct f32 path
<!-- test: larql_compute::test_backend_matmul_quant::quant_matvec_q8_input_q4k_dequantises_then_dispatches -->
<!-- test: larql_compute::test_backend_matmul_quant::quant_matvec_q8_input_q6k_dequantises_then_dispatches -->

#### Scenario: Default per-format stubs return None
- **WHEN** a backend that does not override the per-format helpers is asked for `q4_matvec`, `q4k_matvec`, `q6k_matvec`, `q4_vecmat`, `q4_matvec_topk1`, or `q4_matvec_pair_batch`
- **THEN** every method SHALL return `None`
<!-- test: larql_compute::test_backend_matmul_quant::default_quant_matvec_stubs_return_none -->

### Requirement: MatMul gemv specialisations and batch dispatch

`MatMul` SHALL define a default `matmul_batch(ops)` impl that runs the
ops serially through `matmul` / `matmul_transb`, and SHALL provide
optional gemv specialisations (`f32_gemv`, `f32_gemv_topk1`,
`f32_gemv_force`, `f16_gemv`, `f16_gemv_topk1`, `f16_gemv_topk`,
`f16_gemv_force`) that backends MAY override. Default
implementations of the gemv methods MUST return `None` so callers can
detect non-specialised backends and fall back to `matmul_transb`. The
`*_force` variants MUST skip the internal flop threshold and call the
non-forced gemv directly.

#### Scenario: Default matmul_batch fans out to matmul / matmul_transb
- **WHEN** `matmul_batch` is called on a backend that does not override it, with a mix of transposed and non-transposed `MatMulOp` entries
- **THEN** the returned `Vec<Array2<f32>>` SHALL contain results equal to the per-op direct calls
<!-- test: larql_compute::test_backend_matmul_quant::matmul_batch_no_transpose_serial_dispatch -->
<!-- test: larql_compute::test_backend_matmul_quant::matmul_batch_with_transpose_serial_dispatch -->

#### Scenario: Default gemv stubs return None on CPU backend
- **WHEN** `f32_gemv`, `f32_gemv_force`, `f16_gemv`, or `f16_gemv_force` is called on `CpuBackend`
- **THEN** each call SHALL return `None`
<!-- test: larql_compute::test_backend_matmul_quant::f32_gemv_returns_none_on_cpu -->
<!-- test: larql_compute::test_backend_matmul_quant::f32_gemv_force_returns_none_on_cpu -->
<!-- test: larql_compute::test_backend_matmul_quant::f16_gemv_returns_none_on_cpu -->
<!-- test: larql_compute::test_backend_matmul_quant::f16_gemv_force_returns_none_on_cpu -->

#### Scenario: Q4 vecmat reachable through the trait surface
- **WHEN** `q4_vecmat(activation, q4_data, …)` is invoked through `&dyn ComputeBackend`
- **THEN** the call SHALL return `Some(Vec<f32>)` containing a non-zero output for non-zero inputs
<!-- test: larql_compute::test_backend_matmul_quant::q4_vecmat_via_trait_nonzero -->

### Requirement: DecodeBackend defaults are no-ops returning None

Every method on `DecodeBackend` SHALL have a default implementation
that either returns `None` (for fallible methods) or is a no-op (for
side-effecting methods such as `populate_kv_layer`, `reset_kv_cache`,
`truncate_kv_cache`, `preallocate_kv_cache_per_layer`). Methods MUST
NOT panic on backends that do not implement decode, and `kv_cache_len`
MUST default to `0`. The `decode_token_with_moe_split` default impl
MUST synthesise a single synchronous closure from the fire/collect pair
and forward to `decode_token_with_moe`.

#### Scenario: Default decode stubs return None / 0 on CPU
- **WHEN** `decode_token`, `prefill_q4`, `full_pipeline_q4`, and `multi_layer_q4_ffn` are called on `CpuBackend` and `kv_cache_len` is queried, and `populate_kv_layer` / `reset_kv_cache` / `truncate_kv_cache` are invoked
- **THEN** the four fallible methods SHALL return `None`, `kv_cache_len` SHALL return `0`, and the side-effecting methods SHALL return without panicking
<!-- test: larql_compute::test_backend_matmul_quant::default_decode_stubs -->

#### Scenario: Default device_info delegates to name
- **WHEN** a backend that does not override `device_info` is queried
- **THEN** the returned string SHALL equal `name().to_string()`
<!-- test: larql_compute::test_backend_matmul_quant::default_device_info_delegates_to_name -->
