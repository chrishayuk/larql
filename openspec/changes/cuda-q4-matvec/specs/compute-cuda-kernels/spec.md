## ADDED Requirements

### Requirement: Q4_0 matvec returns real values via dequant + cuBLAS

`CudaBackend::q4_matvec` SHALL return `Some(_)` after this change
(was `None` in `cuda-f32-baseline`). The implementation MAY dequantise
the weights on host before uploading, but the user-visible result
MUST match the CPU `q4_matvec` within 1e-3 absolute and cosine
similarity ≥ 0.9999 on synthetic Q4_0 weights of any shape with
`num_rows * hidden % 32 == 0`.

#### Scenario: Q4_0 matvec parity at 1024×1024
- **WHEN** a Q4_0 matvec is run on both CPU and CUDA backends with the same Q4-quantised input and weights
- **THEN** the maximum absolute element difference SHALL be ≤ 1e-3 and cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_q4::q4_0_matvec_parity -->

### Requirement: Q4_K matvec returns real values via dequant + cuBLAS

`CudaBackend::q4k_matvec` SHALL return `Some(_)` for any Q4_K weight
buffer whose byte length is a multiple of 144 (Q4_K block size) and
whose row × hidden equals a multiple of 256. The result MUST match
the CPU `q4k_matvec` within 1e-3 absolute on Gemma 4B-shaped inputs.

#### Scenario: Q4_K matvec at FFN-gate dimensions
- **WHEN** a 10240×2560 Q4_K weight matrix is multiplied by a 2560-element f32 vector on both backends
- **THEN** the result vectors SHALL agree within 1e-3 absolute, cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_ffn_gate_parity -->

#### Scenario: Q4_K matvec at LM-head dimensions
- **WHEN** a 128256×4096 Q4_K LM-head matrix is multiplied by a 4096-element residual on both backends
- **THEN** the result SHALL agree within 1e-3 absolute, cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_lm_head_parity -->

### Requirement: Q6_K matvec returns real values via dequant + cuBLAS

`CudaBackend::q6k_matvec` SHALL return `Some(_)` for any Q6_K weight
buffer with the canonical 210-byte block layout (`Q6_K_BLOCK_BYTES`),
matching CPU within 1e-3 absolute on Gemma 4B-shaped inputs.

#### Scenario: Q6_K matvec parity at lm-head shape
- **WHEN** a 128256×4096 Q6_K LM-head matrix is multiplied by a 4096-element residual on both backends
- **THEN** the result SHALL agree within 1e-3 absolute, cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_q4::q6k_matvec_lm_head_parity -->

### Requirement: Capability bits reflect Q4 path being live

After this change, `CudaBackend::supports` MUST return `true` for
`Capability::QuantMatVec` and `Capability::Q4VecMat` (in addition to
the previously-set `Cuda` and `F32Gemv`). It MUST still return
`false` for `Capability::FlashAttentionV2` and
`Capability::KvCompressionRotorQuant` (those land in later changes).

#### Scenario: capability set tracks the kernel surface
- **WHEN** `CudaBackend::supports` is queried after the Q4 baseline lands
- **THEN** `QuantMatVec`, `Q4VecMat` SHALL be `true`; `FlashAttentionV2`, `KvCompressionRotorQuant` SHALL be `false`
<!-- test: larql_compute::cuda::backend::tests::supports_q4_matvec_after_q4_baseline -->

### Requirement: `quant_matvec` dispatch routes through the new methods

The default `QuantMatVec::quant_matvec` impl SHALL transparently use
the new `q4k_matvec` / `q6k_matvec` paths. Callers that go through
the umbrella `quant_matvec` method MUST get the same result as
direct calls.

#### Scenario: quant_matvec dispatch matches direct call
- **WHEN** the same Q4_K weights and input are dispatched via `quant_matvec(QuantFormat::Q4_K, ...)` and via direct `q4k_matvec(...)`
- **THEN** both paths SHALL produce byte-identical Vec<f32> outputs on the CUDA backend
<!-- test: larql_compute::test_cuda_q4::quant_matvec_dispatches_to_q4k -->
