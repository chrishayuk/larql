## ADDED Requirements

### Requirement: cuda-oxide MUST be opt-in via a dedicated feature flag

`larql-compute` and `larql-rotorquant` SHALL expose a `cuda-oxide`
feature that's mutually exclusive with the existing `cuda` feature.
Building both simultaneously SHALL fail at compile time via a
top-level `compile_error!`. The default workspace build SHALL NOT
require the cuda-oxide toolchain — it stays on stable Rust.

#### Scenario: default build uses no cuda-oxide

- **WHEN** `cargo build --workspace` runs on a clean Linux host
  with no LLVM 21 and no nightly Rust
- **THEN** the build SHALL succeed without referencing cuda-oxide
<!-- test: unbacked -->

#### Scenario: enabling both cuda and cuda-oxide is rejected

- **WHEN** `cargo build -p larql-rotorquant --features
  cuda,cuda-oxide` runs
- **THEN** the build SHALL fail with a clear `compile_error!`
  message naming both features
<!-- test: unbacked -->

#### Scenario: cuda-oxide build requires the documented toolchain

- **WHEN** `make cuda-oxide-pilot` runs on a host missing LLVM 21
  or `nightly-2026-04-03`
- **THEN** the failure message SHALL surface from `cargo oxide
  doctor`, listing the missing components and how to install them
<!-- test: unbacked -->

### Requirement: cuda-oxide and cudarc MUST coexist for cuBLAS

The pilot SHALL keep cuBLAS calls (`f32_gemv`, `matmul`,
`matmul_transb`) on cudarc even under the `cuda-oxide` feature.
No Rust-native cuBLAS replacement exists; reimplementing GEMM in
cuda-oxide is explicitly out of scope for this change.

#### Scenario: cuBLAS gemv path is unchanged under cuda-oxide

- **WHEN** the LM-head gemv parity test runs against the
  `cuda-oxide` build
- **THEN** the test SHALL invoke the same cudarc-backed
  `gemv_lm_head_parity` path as the `cuda` build, and the
  numerics SHALL match
<!-- test: unbacked -->

### Requirement: Phase 2 evaluation gates the rollout

The change document MUST record an explicit yes/no/abort decision
before any kernel beyond the Iso3 pilot is migrated to cuda-oxide.
The decision SHALL be backed by measurements:

- Build cost ≤ 90 s on the dev box
- PTX size ≤ 1.5× the cudarc-NVRTC reference
- Throughput ≥ 0.75× CPU reference on Gemma 4B head shape
- Zero hard CI failures over a 2-week burn-in

#### Scenario: Phase 3 work blocked without a written decision

- **WHEN** a contributor opens a PR that ports a non-pilot kernel
  (e.g. Iso4) to cuda-oxide
- **THEN** the review SHALL block until
  `docs/cuda-oxide-pilot-report.md` exists and records a "go"
  decision
<!-- test: unbacked -->

#### Scenario: Phase 1 abort is a clean revert

- **WHEN** Phase 2 evaluation lands "no" or "abort"
- **THEN** removing the `cuda-oxide` feature SHALL leave no
  orphan modules outside `crates/larql-rotorquant/src/cuda_oxide/`,
  and `cargo build --features cuda` SHALL produce a binary
  byte-identical to the pre-pilot baseline
<!-- test: unbacked -->
