# compute-backend-traits Specification

## Purpose
TBD - created by archiving change cuda-and-rotorquant-kv. Update Purpose after archive.
## Requirements
### Requirement: New capability bits for CUDA, FlashAttentionV2, KvCompressionRotorQuant

The `Capability` enum SHALL gain three new variants:

- `Capability::Cuda` — backend dispatches via CUDA / cuBLAS / cudarc.
- `Capability::FlashAttentionV2` — backend implements a Flash-Attention
  v2-style fused attention kernel.
- `Capability::KvCompressionRotorQuant` — backend supports K/V cache
  compression via RotorQuant `iso3` / `planar3` / `iso4` / `planar4`
  formats.

Existing backends MUST NOT spontaneously advertise these bits. The
CPU backend reports none of them; the Metal backend reports none of
them in this change (a future Metal RotorQuant port may add the
KV-compression bit). The CUDA backend reports all three when the
respective kernels are available.

#### Scenario: CPU backend does not claim CUDA capability
- **WHEN** `cpu_backend.supports(Capability::Cuda)` is read
- **THEN** the value SHALL be `false`
<!-- test: unbacked -->

#### Scenario: CUDA backend claims its full capability set
- **WHEN** `cuda_backend.supports(Capability::Cuda)`, `_::FlashAttentionV2`, and `_::KvCompressionRotorQuant` are read
- **THEN** all three SHALL be `true` once the corresponding kernels are wired in
<!-- test: unbacked -->

### Requirement: Default backend honours CUDA → Metal → CPU precedence

`larql_compute::default_backend()` SHALL pick the most capable
available backend in the order CUDA, Metal, CPU, with the
`LARQL_BACKEND` environment variable as an override. Backends not
compiled in (feature-gated off) SHALL be skipped without a warning.

#### Scenario: CUDA wins on Linux + cuda feature
- **WHEN** the binary is built with `--features cuda` on Linux and a healthy CUDA driver is present
- **THEN** `default_backend().name()` SHALL contain "cuda"
<!-- test: unbacked -->

#### Scenario: Metal still wins on macOS without CUDA available
- **WHEN** the binary is built with `--features metal` on macOS
- **THEN** `default_backend().name()` SHALL contain "metal"
<!-- test: unbacked -->

#### Scenario: LARQL_BACKEND override is authoritative
- **WHEN** `LARQL_BACKEND=cpu` is set even on a CUDA-capable host
- **THEN** `default_backend().name()` SHALL contain "cpu"
<!-- test: unbacked -->

