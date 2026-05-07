## Why

The `rotorquant-kernels` change shipped a CPU reference for the
RotorQuant K/V quantize / dequantize loop, but `larql-rotorquant`
isn't yet wired into anything that exercises it on Gemma-shaped
inputs. The natural first consumer is `kv-cache-benchmark` — its
`KvStrategy` trait is exactly the encode/decode/memory_bytes shape
we want, and adding a strategy variant lets the existing
comparative-table tooling report RotorQuant alongside Standard /
TurboQuant / Markov / Apollo.

This is a deliberate "thin first cut" for the parent change's
`rotorquant-attention-integration` capability — full integration
into `larql-inference::attention::KvCache` is more invasive and
needs careful review of the 44k-line attention path. The strategy
plumbing here is enough to:

- exercise the RotorQuant CPU reference at production-scale Gemma
  4B / Llama 8B head dimensions through the same harness all other
  strategies use,
- emit per-strategy memory-byte numbers for the `cuda-rotorquant-status`
  comparative table,
- provide a clear seam (`RotorQuantStrategy::iso3()` etc) that the
  attention-integration sub-change can reuse.

## What Changes

- ADD `kv_cache_benchmark::rotorquant` module exposing
  `RotorQuantStrategy` with four constructors (`iso3`, `planar3`,
  `iso4`, `planar4`).
- ADD a binary wire format for the encoded buffer:
  `tag(u8) + n_rows(u32) + head_dim(u32) + per-side {codes_len,
  norms_len, rotation_len}(3×u32) + K{codes,norms,rotation} + V{...}`.
- MODIFY `kv-cache-benchmark/Cargo.toml` to add `larql-rotorquant`
  as a dep.
- ADD inline tests that run two RotorQuant variants through the
  benchmark harness on a synthetic head_dim=32 config, and verify
  the analytical `memory_bytes` is below the Standard baseline.

This is non-breaking. Existing strategies untouched. The accuracy
suite picks up RotorQuant rows for free once the harness wires
them into its iteration.

## Capabilities

### New Capabilities

(none — implements scenarios already on the parent change's
`kv-cache-benchmark-strategies` capability.)

### Modified Capabilities

- `kv-cache-benchmark-strategies`: scenarios for
  `RotorQuantStrategy` round-trip + memory-bytes (declared
  `<!-- test: unbacked -->` in the parent delta) get real test
  annotations on `kv_cache_benchmark::rotorquant::tests::*`.

## Impact

- **Affected files**: new `crates/kv-cache-benchmark/src/rotorquant.rs`
  (~290 lines); Cargo.toml +1 dep; lib.rs +1 mod.
- **Affected systems**: benchmark only. No server / inference /
  router changes.
- **Out of scope**: integration into `larql_inference::attention::KvCache`
  (parent change's `rotorquant-attention-integration` task);
  GPU PTX kernels for RotorQuant (parent change's
  `rotorquant-cuda-kernels` task).
