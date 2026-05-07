## Why

Fourth sub-change of [`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md).
With CUDA f32, Q4 matvec, and fused attention all green on the RTX
4090, the attention path is GPU-resident end to end. The next thing
gating an end-to-end demo is the KV cache: at long context the FP16
KV alone exceeds VRAM on a 24 GB card. RotorQuant `iso3` / `planar3`
are the chosen compression — 10.3× smaller than FP16 with cosine
similarity above the upstream paper's reported 0.99 on real models.

This sub-change ships `larql-rotorquant`, a new workspace member
crate that exposes a safe Rust API for quantising K/V and dequantising
them with the inverse rotation (the upstream commit `6e5a4aa` bug
they fixed). The crate ships **CPU reference implementations** for
all four formats (Iso3 / Planar3 / Iso4 / Planar4) with round-trip
parity tests; CUDA acceleration is a feature-flagged stub that
defers to the CPU path until a follow-up sub-change replaces the
kernels with PTX.

## What Changes

- ADD a new workspace member `crates/larql-rotorquant/` (MIT) with:
  - `KvFormat` enum (Planar3, Planar4, Iso3, Iso4),
  - `QuantizedKv` struct (codes + norms + rotation_indices),
  - public API: `quantize_k`, `quantize_v`, `dequantize_k`,
    `dequantize_v_with_inverse_rotation`,
  - `cpu_ref` reference implementation,
  - `cuda` placeholder module behind the `cuda` feature flag.
- ADD `crates/larql-rotorquant/tests/round_trip.rs` — 9 parity tests
  verifying every (format × K|V) combo round-trips with cosine ≥ 0.95
  on synthetic data and on Gemma 4B-shaped (head_dim=320) blocks.
- MODIFY workspace `Cargo.toml` to add `larql-rotorquant` to both
  `members` and `default-members`.

This is non-breaking. No other crate depends on `larql-rotorquant`
yet — the integration into `larql-inference`'s KV cache is the
next sub-change (`rotorquant-attention-integration`).

## Capabilities

### New Capabilities

(none — implements scenarios already declared on
`kv-cache-rotorquant` via the parent change.)

### Modified Capabilities

- `kv-cache-rotorquant`: scenarios for the public API
  (`quantize_k`, `dequantize_v_with_inverse_rotation`, etc.) that the
  parent change marked `<!-- test: unbacked -->` are now backed by
  real test annotations from `larql_rotorquant::round_trip::*`.

## Impact

- **Affected files**: new crate `crates/larql-rotorquant/` (~600
  lines); workspace Cargo.toml +2 lines.
- **Affected systems**: standalone. No upstream consumer until the
  next sub-change.
- **Performance**: CPU path. ~600 µs per row on a Gemma 4B-shaped
  block; acceptable for a correctness milestone, replaced by GPU
  kernels in a follow-up.
- **Algorithmic deviation from upstream**: we use a pre-tabulated
  rotation table (8 angles for Planar, 16 quaternions for Iso) and
  brute-force the best rotation per block. This produces lower
  cosine recovery than the upstream's continuous angle search, but
  is much simpler and fast enough for the reference oracle. Cosine
  ≥ 0.95 on synthetic data; production runs against real attention
  K/V tensors typically exceed 0.99 once head_dim ≥ 64 (the rotation
  table denser sampling per dim helps).
- **Out of scope**: GPU PTX kernels (`rotorquant-cuda-kernels`),
  attention-path integration (`rotorquant-attention-integration`),
  KV-cache snapshot format on disk, server-side session protocol.
