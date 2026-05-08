## ADDED Requirements

### Requirement: cuda-oxide Iso3 quantize MUST round-trip against the CPU reference

The pilot kernel `larql_rotorquant::cuda_oxide::quantize_iso3` MUST
produce a `QuantizedKv` byte-compatible with the CPU reference
(`larql_rotorquant::cpu_ref::quantize` for `KvFormat::Iso3`). A
round-trip through `cpu_ref::dequantize` SHALL match the input to
cosine ≥ 0.99 per row on synthetic 64 × 320 input.

#### Scenario: cuda-oxide Iso3 round-trip cosine ≥ 0.99

- **WHEN** the pilot kernel quantises a synthetic 64 × 320 random
  input on the GPU and the result is dequantised on the CPU
- **THEN** the per-row cosine similarity vs the original input
  SHALL be ≥ 0.99
<!-- test: unbacked -->

#### Scenario: cuda-oxide Iso3 ↔ CPU byte parity

- **WHEN** the same synthetic input is quantised through both
  `cpu_ref::quantize(KvFormat::Iso3, ...)` and
  `cuda_oxide::quantize_iso3(...)`
- **THEN** the resulting `QuantizedKv.codes`, `.norms`, and
  `.rotation_indices` SHALL be bit-identical (the kernels share
  the same Lloyd-Max codebook + quaternion rotation table; any
  divergence is a kernel bug, not a numerical artifact)
<!-- test: unbacked -->

### Requirement: cuda-oxide pilot MUST coexist with the cudarc-NVRTC variant

The pilot SHALL allow both backends to be exercised against the
same input. If the parent `rotorquant-cuda-kernels` change has
shipped its cudarc-NVRTC Iso3 variant, both implementations MUST
pass a three-way parity test against the CPU reference within
1e-3 max-element absolute difference.

#### Scenario: three-way Iso3 parity (CPU / cudarc / cuda-oxide)

- **WHEN** the same 64 × 320 input is processed through
  `cpu_ref::quantize → cpu_ref::dequantize`,
  `cudarc::quantize_iso3 → cpu_ref::dequantize`, and
  `cuda_oxide::quantize_iso3 → cpu_ref::dequantize`
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
