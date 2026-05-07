## Context

After `engine-rotorquant-auto-compress` lands, the
`RotorQuantEngine` decorator triggers `cache.quantize_layer` post-
decode. The next decode step's attention forward calls
`cache.get_layer(layer)`, which today returns `None` for compressed
layers — and the inner engine treats `None` as "no cache."

This sub-change makes `get_layer` aware of compressed storage.
The simplest and most additive design: when `cache.layers[layer]`
is `None` AND `cache.quantized_kv[layer]` is `Some`, dequantize
the compressed layer (using `dequantize_v_with_inverse_rotation`
for V), populate the FP32 slot, leave the compressed slot
populated for next time, and return.

Why leave the compressed slot? So the next call doesn't
re-dequantize. The auto-compress decorator will re-`take` the FP32
slot post-decode if the engine hasn't cleared it.

## Goals / Non-Goals

**Goals:**
- `cache.get_layer(layer)` transparently returns FP32 even for
  compressed layers, by promoting on miss.
- `cache.is_layer_compressed(layer)` still reports the *underlying*
  truth — it doesn't lie just because we lazy-promoted.
- A metric counter so engines can observe promote frequency.
- Tests demonstrating the round-trip is silent at the call site.

**Non-Goals:**
- Inline-decompress in the attention kernel (CUDA fusion).
- Eviction of the FP32 promoted copy (it stays until the next
  `quantize_layer` post-decode, or until the layer is cleared).
- Concurrency-safe promote-on-read across multiple threads
  reading the same layer. The KvCache is single-threaded today
  (each engine owns one).

## Decisions

### D1 — Auto-promote inside `get_layer`, not at the engine level

Two options:

1. **`get_layer` auto-promotes** — call sites stay simple; engines
   don't need to be aware.
2. **Engines call `dequantize_layer` explicitly** — more code at
   each engine site; touches every engine.

Chose option 1. Adding a helper to `KvCache` is one place to
maintain. The performance cost is real but bounded; tracking via
the metric counter lets us judge whether to optimise later.

### D2 — Add `get_layer_lazy` for callers that want explicit control

Some call sites — diagnostics, snapshot serialisation — explicitly
DO NOT want the side effect of populating the FP32 slot. They use
`get_layer_lazy` which returns `None` on a compressed layer
without modifying state.

`get_layer` becomes the default for the attention forward; lazy
is the explicit-no-side-effect variant.

### D3 — `is_layer_compressed` returns the underlying truth

Even after a promote-on-read populates the FP32 slot, the
compressed side-table is still populated. `is_layer_compressed`
reports `true` until `quantized_kv[layer]` is taken. This lets
diagnostics distinguish "layer is in compressed storage" from
"layer happened to also have an FP32 copy."

### D4 — Track promote count via `AtomicU64`

`KvCache::promote_on_read_count: AtomicU64` increments on each
auto-promote. Engines (and tests) read it via `.load(Relaxed)`.
The atomic is needed because `get_layer` takes `&self` (immutable)
— the FP32 slot population uses interior mutability via a
`Mutex<...>` on the slot. Wait — that's ugly. Let me revisit.

Actually `cache.layers[layer]` is `Option<SharedKV>` directly, not
behind a Mutex. To populate it from a `&self` we need interior
mutability. Two paths:

1. Make `get_layer` take `&mut self` (breaks current API).
2. Wrap `layers` in `RwLock` (heavy).
3. Add an `OnceLock<SharedKV>` per layer for the auto-populated
   FP32 — but the layer can be cleared and re-populated, so
   OnceLock is wrong.
4. **Change get_layer to take `&mut self`** for the auto-promoting
   variant; keep `get_layer_lazy` as the `&self` variant.

We pick option 4. Existing call sites that call `cache.get_layer`
likely have mutable cache access already (the attention forward
mutates the cache anyway). Audit pass: review every existing
call site, change to `&mut` where needed.

### D5 — Reset compressed side-table when `clear_layer` is called

`clear_layer` already takes the FP32 slot. Symmetric: also clear
`quantized_kv[layer]`. Avoids a stale compressed copy outliving a
clear.

## Risks / Trade-offs

- **Risk: silent dequant on every read = 100 µs per decode step
  per layer.** For 32 layers that's 3.2 ms / token / unconditional
  promote. Acceptable for the correctness-first milestone;
  `engine-rotorquant-deferred-k-policy` lands later with a
  threshold-based policy.
- **Risk: API change `&self` → `&mut self`.** Touches several
  call sites. → Mitigation: audit pass; provide
  `get_layer_lazy(&self)` as the explicit-no-promote variant.
- **Risk: promote-on-read breaks snapshot semantics.** A
  snapshot taken via `get_layer` would unintentionally promote.
  → Mitigation: snapshot uses `get_layer_lazy` + reads the
  compressed side-table directly.

## Migration Plan

Land after `engine-rotorquant-auto-compress`. Audit existing call
sites of `get_layer`; change to `&mut` where they call on a
mutable cache reference, switch to `get_layer_lazy` where they
call on `&self`.

Rollback: revert. The auto-compress decorator silently produces
no benefit (decode reads see `None`), but no test fails.

## Open Questions

- **Q1: Should the FP32 promoted copy expire?** A long-running
  engine could end up with all 32 layers populated as both
  compressed AND FP32, defeating the memory savings.
  → Mitigation: the auto-compress decorator already calls
  `quantize_layer` post-decode, which `take`s the FP32 slot
  again. So at steady state only one layer is in dual storage at
  any moment.
- **Q2: Should the metric counter break out compressed-hit vs
  empty-miss?** Worth doing if anyone tunes against it.
  Recommendation: separate counters for `promote_hits` and
  `promote_misses`.
