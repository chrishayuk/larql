## Context

`KvCache` in `larql-inference::attention::decode` is a compact struct:

```rust
pub struct KvCache {
    pub layers: Vec<Option<SharedKV>>,   // SharedKV = (Array2<f32>, Array2<f32>)
    pub max_window: Option<usize>,
    pub next_position: usize,
}
```

Where `SharedKV` is FP32. The cache is shared by every engine
(Markov, Apollo, UnlimitedContext, …) — a `KvFormat` on the cache
is the natural seam.

Two integration shapes were on the table:

1. **Make `SharedKV` an enum** — a Fp32 variant + a RotorQuant
   variant. Every read site has to match. Touches the entire
   attention forward pass.
2. **Side-table** — a parallel `Vec<Option<...>>` for compressed
   layers; FP32 stays the canonical layer slot. Reads go through
   helpers that know which storage to consult.

We picked option 2. It's smaller, additive, and matches the upstream
"deferred-K" pattern where most code paths see FP32 and only the
specific engine logic that quantises understands the compressed
form.

## Goals / Non-Goals

**Goals:**

- A `KvFormat` parameter on the cache without breaking any
  existing call site.
- A pair of methods (`quantize_layer` / `dequantize_layer`) that
  move a single layer between FP32 and compressed storage.
- The FP32 slot is freed when a layer is compressed (the side-table
  doesn't double cache memory).
- Round-trip cosine ≥ 0.95 on synthetic data (matches
  `larql-rotorquant` test thresholds).

**Non-Goals:**

- Driving the engine forward pass. Engines decide WHEN to
  compress — this change just gives them the API.
- Mixed formats per cache. `set_kv_format` enforces one format
  per cache instance.
- f16 path. Compressed storage is the only deviation from f32; an
  f16 path is its own future work.
- Gating `Capability::KvCompressionRotorQuant`. That requires GPU
  PTX kernels in `larql-rotorquant`'s CUDA module, which are still
  a stub today.

## Decisions

### D1 — Side-table over enum

Two `Vec<Option<...>>` fields side-by-side rather than a tagged
union. Layers in `layers` are FP32; layers in `quantized_kv` are
compressed; both per-layer slots are mutually exclusive (compressing
one layer takes the FP32 slot). Reads consult both — `is_layer_compressed`
is the routing helper.

### D2 — `Option::take` to swap FP32 → compressed

`quantize_layer` does `slot.take()`, so the FP32 entry is removed
from `layers` and the underlying `Array2<f32>` is dropped after
quantization. Memory doesn't double during the swap (briefly, the
flat `Vec<f32>` extracted from the array exists in parallel with
the `QuantizedKv`, but each is on the order of a few MB for typical
layers).

### D3 — `dequantize_layer` is non-destructive

It reconstructs FP32 each call; the compressed side-table is left
in place. The motivation: callers may want to inspect a compressed
layer without paying the promotion cost. `promote_layer_to_fp32` is
the destructive variant that puts the result back in the FP32 slot
and frees compressed.

### D4 — Mixed-format enforcement is "good faith"

`set_kv_format` accepts any `KvFormat`. If someone reuses a cache
across different formats by repeatedly calling `set_kv_format`, the
behaviour is undefined: layers compressed under the old format are
left in the side-table, but `dequantize_layer` reads back through
the format on the per-layer `QuantizedKv`. So compressed layers
remember their format, but new compressions use the latest set.
This is intentional — the test suite covers single-format use; a
future ergonomic improvement could panic on format mismatch.

### D5 — V dequant uses inverse rotation, K uses forward

Inside `dequantize_layer`, K calls `dequantize_k` and V calls
`dequantize_v_with_inverse_rotation`. This is the upstream commit
6e5a4aa lesson — V dequant MUST apply the inverse, not forward. The
contract is enforced architecturally by the rotorquant crate's
function names.

## Risks / Trade-offs

- **Risk: callers reach into `cache.layers` directly and miss
  compressed layers.** → Mitigation: existing call sites all use
  `cache.layers[i]` for FP32 storage; if they want to see a
  compressed layer they must first `promote_layer_to_fp32`. The
  type system doesn't enforce this; consumer-side audit is left
  to the engine sub-changes.
- **Risk: head_dim not divisible by block size.** Iso3/Iso4 require
  `head_dim % 4 == 0`; Planar3/Planar4 require `% 2 == 0`. → On
  failure `quantize_layer` returns false and restores the FP32
  layer. No panic, no silent data loss.
- **Risk: per-call dequant cost.** Each `dequantize_layer` call
  rebuilds the FP32 array. → Acceptable for the data-layer change;
  engine sub-changes can introduce a "promotion cache" if needed.

## Migration Plan

Land. Engines that want compression call:

```rust
cache.set_kv_format(KvFormat::Iso3);
// ... after a layer's K/V is in cache ...
cache.quantize_layer(layer);
// ... reading attends FP32 (no-op for the engine if the layer is
//     compressed and the engine doesn't support compressed reads):
let kv = cache.dequantize_layer(layer);   // when needed
```

Rollback: revert. Existing FP32 path is untouched.

## Open Questions

- **Q1: Should `get_layer` transparently dequantise?** Currently it
  returns `None` for compressed layers (because `layers[i]` is
  `None`). A "smart" version would `dequantize_layer` on miss. We
  chose explicit because automatic dequant on every read defeats
  the point of compression. Engines that want a unified read path
  can wrap.
- **Q2: f16 codes / norms.** Norms are f32 today. f16 would halve
  per-row metadata. **Recommendation:** add when the GPU PTX kernel
  lands and we measure end-to-end cost.
