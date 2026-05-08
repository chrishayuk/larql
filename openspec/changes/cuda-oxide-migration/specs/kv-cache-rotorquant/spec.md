## ADDED Requirements

### Requirement: cuda-oxide Iso3 dequantize MUST match the CPU reference

The cuda-oxide pilot MUST reconstruct `KvFormat::Iso3`
`QuantizedKv` buffers compatibly with the CPU reference
(`larql_rotorquant::dequantize_k`). The input SHALL be generated
with the existing CPU quantize path so the cuda-oxide test isolates
dequantize behavior.

#### Scenario: cuda-oxide Iso3 dequantize matches CPU dequantize

- **WHEN** a synthetic 64 × 320 random input is quantized with
  `quantize_k(KvFormat::Iso3, ...)`, then dequantized by both
  `dequantize_k` and `cuda_oxide::dequantize_iso3`
- **THEN** the cuda-oxide reconstruction SHALL match the CPU
  reconstruction to max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: cuda-oxide Iso3 round-trip cosine ≥ 0.99

- **WHEN** the cuda-oxide dequantize pilot reconstructs the
  CPU-quantized synthetic input
- **THEN** the per-row cosine similarity vs the original input
  SHALL be ≥ 0.99
<!-- test: unbacked -->

### Requirement: cuda-oxide pilot MUST coexist with the cudarc-NVRTC variant

The pilot SHALL allow both backends to be exercised against the
same input. If the parent `rotorquant-cuda-kernels` change has
shipped its cudarc-NVRTC Iso3 variant, both implementations MUST
pass a three-way parity test against the CPU reference within
1e-3 max-element absolute difference.

#### Scenario: three-way Iso3 parity (CPU / cudarc / cuda-oxide)

- **WHEN** the same CPU-quantized 64 × 320 input is processed
  through CPU dequantize, cudarc dequantize, and cuda-oxide
  dequantize
- **THEN** every pair of reconstructions SHALL agree to
  max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

### Requirement: cuda-oxide tests SHALL be GPU-gated, not workspace-default

The cuda-oxide round-trip test SHALL only run when both
`LARQL_CUDA_AVAILABLE=1` is set and the build was compiled with
`--features cuda-oxide`. The default `make ci` target SHALL NOT
require a GPU or LLVM 21.

#### Scenario: CPU-only host does not require LLVM 21

- **WHEN** `make ci` runs on a host with no GPU and no LLVM 21
- **THEN** every CI step SHALL succeed and SHALL NOT attempt to
  invoke `cargo oxide` or load any cuda-oxide artifact
<!-- test: unbacked -->
