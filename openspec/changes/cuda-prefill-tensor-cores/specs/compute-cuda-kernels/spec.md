## ADDED Requirements

### Requirement: Prefill projection GEMM SHALL have an f16 / Tensor Core path

`gemm_proj_seq` SHALL provide an alternate path that runs the
`(seq_len, hidden) × (out_dim, hidden)^T → (seq_len, out_dim)`
projection via `matmul_transb_device_inout_f16`
(`cublasGemmEx`/`Gemm<half::f16>` with `CUDA_R_16F` inputs and
`CUBLAS_COMPUTE_32F` accumulator). The path SHALL be gated on
`LARQL_CUDA_PREFILL_TENSOR_CORES=1`. The default (env var unset
or any other value) SHALL retain the existing f32 SGEMM path.
The f16 cache SHALL only populate when the gate is on.

#### Scenario: bench shows non-trivial prefill speedup

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20 --warmup 3` is run twice — once with
  `LARQL_CUDA_PREFILL_TENSOR_CORES=1` and once without
- **THEN** the prefill column with the gate on SHALL be ≤ 70%
  of the prefill column with the gate off, averaged over 5
  trials each
<!-- test: unbacked -->

### Requirement: f16 prefill SHALL produce parity-equivalent output

`LARQL_CUDA_PREFILL_TENSOR_CORES=1` and the default f32 path
SHALL produce identical generated text on the same prompt for
the same model. f16 input + f32 accumulator keeps per-element
error ≤ 1e-3 against the existing f32 reference.

#### Scenario: generated text matches across the gate

- **WHEN** `larql run output/gemma-3-4b-it-vindex "The capital
  of France is" --max-tokens 20` is invoked twice — once with
  `LARQL_CUDA_PREFILL_TENSOR_CORES=1` and once without
- **THEN** the produced token sequences SHALL be identical
<!-- test: unbacked -->
