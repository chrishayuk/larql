## Why

`engine-rotorquant-auto-compress` (proposal-only, ships next) wires
the `RotorQuantEngine` decorator that calls
`cache.quantize_layer(layer)` after each decode step. The
known limitation captured in that change's design.md D5: the
inner engine's next decode step reads `cache.layers[layer]` and
gets `None` because the layer was just compressed.

This sub-change closes the loop. We extend the attention forward
path so that any read of `cache.layers[layer]` that hits a `None`
slot AND a populated `cache.quantized_kv[layer]` slot
auto-promotes the layer back to FP32 transparently.

The combination of `auto-compress` + `promote-on-read` gives the
upstream "deferred-K" pattern end-to-end: K is compressed at rest,
decompressed on demand for the next attention step, recompressed
after.

This is **not** on the critical path for the CUDA + RotorQuant
workstream — it's the natural follow-up to
`engine-rotorquant-auto-compress` once that ships.

## What Changes

- MODIFY `KvCache::get_layer` to check `quantized_kv[layer]` on
  miss and auto-promote when present. Behaviour change is
  observable only when the parallel side-table is populated.
- ADD `KvCache::get_layer_lazy(layer) -> Option<&SharedKV>` as a
  non-promoting alternative for callers that want explicit
  control. The auto-promoting `get_layer` becomes the default.
- MODIFY internal call sites that assume `get_layer` returns
  `None` for "not cached" to handle the case where the layer is
  *compressed* and got promoted lazily.
- ADD a metric counter `KvCache::promote_on_read_count: AtomicU64`
  so engines can introspect how often the auto-promote fired —
  useful for tuning the deferred-K policy.
- MODIFY `inference-attention-and-kv` capability spec.

This is mostly non-breaking. Callers that already handled `None`
correctly keep working; callers that wanted "is this layer fully
in compressed storage" now use `is_layer_compressed` (which still
returns the truth before any promotion).

## Capabilities

### New Capabilities

(none — modifies `inference-attention-and-kv`.)

### Modified Capabilities

- `inference-attention-and-kv`: adds requirements for transparent
  promote-on-read and the metric counter.

## Impact

- **Affected files**: `crates/larql-inference/src/attention/decode.rs`
  (~30 line change to `get_layer` + new `get_layer_lazy` method);
  inline tests for the new behaviour.
- **Affected systems**: inference KV cache only. The decorator
  engine and existing engines work unchanged.
- **Performance**: each promote-on-read pays one
  `dequantize_layer` call (~100 µs on CPU for Gemma 4B-shaped
  inputs). At decode-token granularity that's a measurable cost,
  but it's strictly less than the cost of attending against
  uncompressed FP16 K + paying the bandwidth in attention.
- **Out of scope**: GPU-side promote-on-read (rotorquant CUDA
  kernels need to land first); attention-time inline-decompress
  (that requires fusing dequant into the attention kernel — a
  much larger sub-change).
