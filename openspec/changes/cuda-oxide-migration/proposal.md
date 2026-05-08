## Why

LARQL's CUDA backend currently relies on
[`cudarc`](https://crates.io/crates/cudarc) (driver bindings + cuBLAS
+ NVRTC runtime PTX compile). Custom kernels — the fused softmax in
`cuda-fused-attention`, and the planned RotorQuant Iso/Planar
quantize / dequantize PTX kernels — are written in CUDA C strings,
compiled at startup via NVRTC, cached on disk. That works, but:

1. **CUDA C in `format!("...")` is fragile.** Compile errors surface
   at server boot, not `cargo build`. There's no autocomplete, no
   borrow checker on shared-memory buffers, and the host ↔ device
   ABI is hand-marshalled. Every refactor of a CUDA struct is a
   manual sync against the kernel string.
2. **No first-class Rust device-side abstractions.** LARQL has
   careful Rust types for `KvFormat`, `QuantizedKv`, `SharedKV` —
   none of that crosses the device boundary. Inside a kernel, K
   and V are raw `f32*` pointers and the rotation index is a
   bare `u16`.
3. **PTX-vs-cubin tradeoff is opaque.** NVRTC produces PTX that
   the driver JITs at first launch — extra ~200 ms cold-start
   that we paper over with a cudarc cache directory. Pre-compiled
   cubin would be faster but requires a per-arch build pipeline
   we don't have.

[`NVlabs/cuda-oxide`](https://github.com/NVlabs/cuda-oxide) is
NVIDIA's experimental rustc backend that compiles `#[kernel]`
functions to PTX as part of `cargo build`. Single-source: device
code lives in the same `.rs` file as host code, behind a
`#[kernel]` attribute. The host crate (`cuda-core` /
`cuda-async` / `cuda-host`) replaces the driver-API parts of
cudarc with type-safe abstractions for memory + launches.

This change proposes a **phased pilot** of cuda-oxide for
LARQL's custom kernels. cuBLAS continues to flow through cudarc
(no Rust replacement for hand-tuned vendor BLAS). The
RotorQuant + custom-attention kernels — the workstream that
otherwise lands as hand-written CUDA C — gets to be Rust.

The change is a **plan**, not a commitment to migrate. It
documents the evaluation criteria, the kernels we'd port first,
and the back-out plan if cuda-oxide's alpha-quality bites us.

## What Changes

### Phase 1 — Pilot: one RotorQuant Iso3 kernel in cuda-oxide

- ADD `crates/larql-rotorquant/src/cuda_oxide/` module behind a
  new `cuda-oxide` cargo feature (off by default, mutually
  exclusive with the existing `cuda` feature flagged for cudarc).
- ADD `cargo-oxide` to the dev toolchain (`rust-toolchain.toml`
  pins a nightly that supports the rustc-codegen-cuda backend
  alongside the existing stable workspace; the GPU container
  builds with the nightly toolchain).
- ADD a single `#[kernel]` Rust function for **Iso3 quantize**
  (the simplest of the four RotorQuant formats — 3-bit per
  coordinate, 4D quaternion rotation per block).
- ADD a parity test that round-trips against the existing CPU
  reference (`larql_rotorquant::cpu_ref::quantize`) with cosine
  ≥ 0.99 — same bound as the parent `rotorquant-cuda-kernels`
  proposal targets for cudarc-NVRTC.

### Phase 2 — Evaluation

After Phase 1 lands, the team measures:

- **Build cost**: how much does adding cuda-oxide add to a clean
  `cargo build --features cuda-oxide`? Target: ≤ 90 s on the dev
  box (CUDA 12.5, LLVM 21). Beyond that, contributors will skip
  the feature.
- **PTX size**: cuda-oxide vs hand-written PTX for the Iso3
  kernel. Target: ≤ 1.5×.
- **Throughput**: tokens/sec on a synthetic Gemma-3-4B-shaped
  workload. Target: within 90% of the cudarc-NVRTC variant; if
  worse, file an upstream issue and decide.
- **Stability**: number of cuda-oxide-side panics or build
  breakages over a 2-week burn-in. Target: zero hard failures
  in CI; known-flaky tests must be quarantined upstream first.

If any target misses by > 25%, abort the migration and document
what we learned. The cudarc + NVRTC path stays the canonical
CUDA build.

### Phase 3 — Conditional rollout (only if Phase 2 passes)

- MIGRATE the remaining three RotorQuant formats (Iso4, Planar3,
  Planar4) into cuda-oxide.
- MIGRATE the fused-softmax / decode-attention kernel from
  NVRTC to cuda-oxide.
- KEEP cudarc on the `cuda` feature for cuBLAS (`f32_gemv`,
  `matmul`, `matmul_transb`) — there's no Rust replacement for
  cuBLAS and writing one isn't in scope.
- DOCUMENT the dual-feature (`cuda` + `cuda-oxide`) build matrix
  in `deploy/docker/README.md` and the GPU Dockerfile.

## Capabilities

### New Capabilities

(none — extends the planned and existing CUDA capabilities below.)

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for the new feature
  flag + dual-toolchain build, and an evaluation matrix for the
  pilot.
- `kv-cache-rotorquant` — adds a parity scenario for the
  cuda-oxide-built Iso3 kernel against the CPU reference (the
  existing `rotorquant-cuda-kernels` change covers the cudarc
  variant; this proposal is symmetric for cuda-oxide).

## Impact

- **Affected files (Phase 1)**:
  - `crates/larql-rotorquant/Cargo.toml` — add `cuda-oxide`
    feature.
  - `crates/larql-rotorquant/src/cuda_oxide/mod.rs` — new
    module, only built under the feature flag.
  - `rust-toolchain.toml` — document nightly companion toolchain
    in a comment; do NOT switch the workspace default.
  - `deploy/docker/Dockerfile.gpu` — add LLVM 21 + clang-21 +
    `cargo install cargo-oxide`. Conditional on the feature.
  - `Makefile` — `make cuda-oxide-pilot` target that builds and
    runs the parity test.

- **Affected systems**: GPU container only. CPU FFN container,
  router, and tests are unaffected.

- **Provenance**: NVlabs/cuda-oxide is NVIDIA Research's
  Rust-CUDA codegen backend. Apache-2.0 + MIT licensed.
  README rev as of 2026-05-08 declares "early stage (alpha) and
  under active development". This proposal explicitly accepts
  that risk for the pilot phase only.

- **Out of scope**:
  - Migrating cuBLAS — there is no Rust-native cuBLAS, and the
    hand-tuned vendor library is faster than anything cuda-oxide
    could compile.
  - Migrating the host-side `cudarc::driver` calls in
    `larql-compute/src/cuda/driver.rs` — the cuda-oxide host
    runtime (`cuda-core`) is a candidate replacement, but
    swapping it would touch every cuBLAS call too. Defer.
  - Writing host-side attention coordination in cuda-oxide — the
    fused-softmax kernel is the candidate, not the GEMM
    composition that wraps it.
  - Cross-platform kernels — cuda-oxide is Linux-only today.
    Metal kernels stay in the existing Objective-C++ pipeline.

## Risks and back-out

- **Alpha API breakage.** cuda-oxide's README says: "expect
  bugs, incomplete features, and API breakage". Mitigation: pin
  to a specific commit hash in `Cargo.toml`'s git dep, not a
  semver range. If upstream breaks the pinned commit, freeze
  the pin and let cuda-oxide-side updates be opt-in.
- **Nightly toolchain drift.** cuda-oxide pins a specific
  `nightly-YYYY-MM-DD`. If that nightly conflicts with a
  workspace dep, the GPU build will diverge from the CPU build.
  Mitigation: the workspace stays on stable; the cuda-oxide
  feature uses `+nightly-YYYY-MM-DD` per the upstream
  `rust-toolchain.toml`. Document this in
  `docs/cuda-rotorquant-status.md`.
- **LLVM 21 / clang-21 build dependency.** Ubuntu 24.04 ships
  LLVM 18 by default; LLVM 21 needs `apt.llvm.org`. Mitigation:
  the Dockerfile pins a tag of an `nvidia/cuda` image with LLVM
  21 already installed, OR adds the LLVM apt repo at build time.
- **Back-out:** the `cuda-oxide` feature is off by default and
  mutually exclusive with `cuda`. Removing it is `cargo
  build --features cuda` exactly as today; no orphaned files
  beyond the `crates/larql-rotorquant/src/cuda_oxide/` module
  which is gated.
