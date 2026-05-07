## Why

`rotorquant-attention-integration` (shipped) put the
`KvFormat` parameter and `quantize_layer` /
`dequantize_layer` / `promote_layer_to_fp32` /
`is_layer_compressed` methods on `KvCache`. Today the seam is
**unused** — engines (Markov, Apollo, UnlimitedContext, TurboQuant)
write FP32 K/V into the cache and never call `quantize_layer`. To
get the actual compression benefit we need an engine policy that
drives the seam.

## ⚠ Status: blocked on engine refactor

While drafting the implementation we discovered that the existing
engines do **NOT** hold a `larql_inference::attention::decode::KvCache`
struct — they hold their own per-engine state:

- `UnlimitedContextEngine::current_window_kv: Option<Vec<SharedKV>>`
  (a vec of (K, V) pairs per layer, NOT the `KvCache` type).
- `MarkovResidualEngine` rebuilds K/V from residuals each tick;
  no persistent cache field.
- `TurboQuantEngine` and `ApolloEngine` use bespoke storage.

So the `cache_mut() -> Option<&mut KvCache>` trait method
proposed here would return `None` for every existing engine,
making the decorator a no-op in practice.

**Path forward** before this proposal can be implemented:

1. **`engine-kvcache-unification`** — restructure
   `UnlimitedContextEngine` (and possibly Markov) to use the
   shared `KvCache` type rather than per-engine storage. ~1-2
   days of careful work.
2. After unification ships, this proposal's decorator becomes
   straightforwardly implementable.

The proposal stays on the backlog as designed; the implementation
order is `engine-kvcache-unification` → this change.

This sub-change adds a **decorator engine** `RotorQuantEngine`
that wraps any inner `KvEngine` and applies the upstream
"deferred-K" pattern automatically:

- During prefill, K stays FP32 (so the prefill compute path sees
  the cache exactly as it does today).
- After each decode-step token insertion, the engine compresses
  the recent FP32 region of every layer's K/V via
  `cache.quantize_layer(layer)`.

The decorator pattern means we don't fork every existing engine to
get a compressed variant — `RotorQuant(Markov)`,
`RotorQuant(UnlimitedContext)`, etc. all work for free.

## What Changes

- ADD a `cache_mut(&mut self) -> Option<&mut KvCache>` method to
  the `KvEngine` trait. Default returns `None`; engines that hold
  a `KvCache` override to return `Some(&mut self.cache)`. Existing
  engines:
  - `MarkovResidualEngine` → returns its hot-window cache.
  - `UnlimitedContextEngine` → returns its current-window cache.
  - `TurboQuantEngine` → returns `None` (uses its own quantised
    storage; RotorQuant decoration is conceptually redundant).
  - `ApolloEngine` → returns `None` (uses a fact store, not a
    standard KvCache).
- ADD a new module `engines/kv_engines/rotorquant.rs` with:
  - `pub struct RotorQuantEngine { inner: Box<dyn KvEngine>, format: KvFormat }`
  - `KvEngine` impl that delegates `prefill` / `decode_step` /
    `memory_bytes` / etc. to `inner`, then calls
    `inner.cache_mut()?.quantize_layer(layer)` for every layer
    after each `decode_step`.
- ADD `EngineKind::RotorQuant { inner: Box<EngineKind>, format:
  KvFormat }` variant.
- ADD `EngineKind::from_name` parsing for spec strings like
  `iso3:inner=unlimited-context`,
  `planar3:inner=markov-rs:window=1024`. The default inner is
  `unlimited-context:window=512`.
- ADD inline tests verifying the decorator delegates correctly
  and triggers `quantize_layer` on the inner cache.

This is non-breaking. Existing engines unchanged; the new
`cache_mut()` default impl returns `None`, so engines that don't
override are unaffected.

## Capabilities

### New Capabilities

(none — implements scenarios on the parent change's
`inference-residual-engine` capability.)

### Modified Capabilities

- `inference-residual-engine`: scenarios for `IsoQuantEngine` /
  `PlanarQuantEngine` (declared `<!-- test: unbacked -->` in the
  parent delta) get real annotations on
  `larql_inference::engines::kv_engines::rotorquant::tests::*`.

## Impact

- **Affected files**: `larql-inference/src/engines/mod.rs`
  (trait + EngineKind + parsing); new module
  `engines/kv_engines/rotorquant.rs` (~250 lines incl. tests);
  small overrides on `markov_residual::engine` and
  `unlimited_context::engine` to return their cache.
- **Affected systems**: inference engine registry only. Server,
  router, CUDA, RotorQuant kernels all unchanged.
- **Memory**: when the decorator triggers `quantize_layer` on a
  layer, the FP32 slot is freed (per `quantize_layer`'s
  `Option::take`). So decoration *reduces* memory usage at steady
  state.
- **Out of scope**: engine-side selection of WHICH layers to
  compress (currently every layer post-decode). A future
  `engine-rotorquant-deferred-k-policy` change can add a
  threshold (e.g., compress only when cached_len > N).
