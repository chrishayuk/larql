## Why

Seventh sub-change of [`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md).
`rotorquant-strategy` showed `larql-rotorquant` plugged into the
benchmark harness. This change goes the next step: it puts a
`KvFormat` parameter on the actual `larql_inference::attention::KvCache`,
so real attention runs can hold compressed K/V in VRAM rather than
FP32.

The integration is **additive** — a parallel side-table on the
existing `KvCache` struct, not a rewrite of the FP32 storage. Layers
that have been `quantize_layer`'d get pulled out of the FP32 `layers`
slot and into a compressed `quantized_kv` slot; reading them back
goes through `dequantize_layer`. Any layer that hasn't been
explicitly compressed keeps its existing FP32 path untouched. This
mirrors the upstream RotorQuant "deferred-K" pattern: prefill writes
FP32; decode token insertion can choose to compress.

This sub-change deliberately stops at the data-layer integration.
Wiring the **attention forward pass** to call `quantize_layer`
automatically (the full deferred-K behaviour described in the parent
design) requires understanding the per-engine prefill/decode paths
in `engines/kv_engines/*` and is left for a follow-up. What this
change ships is the seam — a contract that future engine code can
drive.

## What Changes

- ADD `larql-rotorquant` as a dep of `larql-inference`.
- ADD `KvCache::kv_format: Option<KvFormat>` field +
  `set_kv_format(format)` setter.
- ADD `KvCache::quantized_kv: Vec<Option<(QuantizedKv, QuantizedKv)>>`
  parallel side-table.
- ADD `KvCache::quantize_layer(layer)` — moves a layer from FP32 to
  compressed storage. Returns `false` if format unset, layer empty,
  or out of range.
- ADD `KvCache::dequantize_layer(layer)` — non-destructive readback.
- ADD `KvCache::promote_layer_to_fp32(layer)` — inverse of
  `quantize_layer`.
- ADD `KvCache::is_layer_compressed(layer)`.
- ADD three inline tests covering: format-unset no-op,
  quantize→dequantize cosine ≥ 0.95 round-trip, promote restores
  the FP32 slot.

This is non-breaking. All existing `KvCache` construction sites are
untouched (the new fields default-initialise via `with_layers` and
`with_window`). Existing call sites that read `layers` directly
keep working — they just won't see compressed layers.

## Capabilities

### New Capabilities

(none — implements scenarios already on the parent change's
`inference-attention-and-kv` capability.)

### Modified Capabilities

- `inference-attention-and-kv`: scenarios for KV format
  parameterisation + transparent quantize-on-set / dequantize-on-get
  (parent declared `<!-- test: unbacked -->`) get real test
  annotations on `larql_inference::attention::decode::tests::*`.

## Impact

- **Affected files**: `larql-inference/Cargo.toml` +1 dep;
  `crates/larql-inference/src/attention/decode.rs` adds the new
  fields + 5 methods + 3 tests (~140 lines).
- **Affected systems**: data-layer only. The attention forward
  pass doesn't change behaviour today; engines opt in to
  `quantize_layer` if they want compression.
- **Memory**: when a layer is compressed, the FP32 slot is taken
  (`Option::take`) so the side-table doesn't double memory.
- **Out of scope**: attention-forward integration that calls
  `quantize_layer` automatically (a future sub-change once we've
  picked the engine-level deferred-K policy);
  `RotorQuantEngine` / `IsoQuantEngine` engine wrappers (also
  follow-up); `Capability::KvCompressionRotorQuant` flip on
  `CudaBackend` (gated on the GPU PTX kernels landing in
  `rotorquant-cuda-kernels`).
