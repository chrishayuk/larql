## Context

`KvCache` (post-`rotorquant-attention-integration`) supports per-layer
compression via `quantize_layer`. To use it productively we need:

1. A way to call `quantize_layer` automatically — engine-level
   policy.
2. A way to do that without modifying every existing engine —
   decorator pattern.
3. A way for the decorator to reach the inner engine's cache —
   trait method.

Today, engines hold their `KvCache` in different shapes:

| Engine | Cache layout |
|---|---|
| `MarkovResidualEngine` | hot window in `MarkovStore`; rebuilds from residuals |
| `UnlimitedContextEngine` | `current_window_kv: Option<KvCache>` |
| `TurboQuantEngine` | quantised K/V in private storage (no `KvCache`) |
| `ApolloEngine` | `FactStore` (no `KvCache`) |

The trait method `cache_mut() -> Option<&mut KvCache>` cleanly
captures the difference: engines that hold a standard `KvCache`
return `Some`; engines using bespoke storage return `None`. The
decorator gracefully no-ops on the latter.

## Goals / Non-Goals

**Goals:**

- A `RotorQuantEngine` that wraps any other `KvEngine` and
  automatically compresses K/V after each decode step.
- A trait method `cache_mut()` exposing the inner cache.
- Configurable via the existing `EngineKind::from_name` spec-string
  syntax.
- Tests that prove the decoration triggers compression without
  needing a real forward pass (mock inner engine).

**Non-Goals:**

- Writing a new prefill / decode loop. The decorator delegates to
  the inner engine.
- Custom compression-aware attention reads. The default attention
  path reads from `cache.layers[i]`; compressed layers are
  **promoted** back to FP32 via `dequantize_layer` /
  `promote_layer_to_fp32` if the attention forward needs them.
  Whether to do that automatically is a future change; the bare
  decorator just compresses on write.
- f16 codebooks (already a future change).
- GPU-side compression — that lives on `larql-rotorquant`'s CUDA
  kernels, which are still a stub.

## Decisions

### D1 — Decorator over per-engine variant

Three options:

1. **Decorator** (`RotorQuantEngine` wraps `Box<dyn KvEngine>`).
   Adds compression to any underlying engine.
2. **Per-engine variants** (`RotorQuantMarkovEngine`,
   `RotorQuantUnlimitedEngine`, ...). Repeats logic 4×.
3. **Internal flag** on each existing engine (`compress: Option<KvFormat>`).
   Per-engine maintenance burden; doesn't compose.

Chose decorator. Composes (`RotorQuant(MarkovResidual(...))` works
out of the box). Single point of policy logic.

### D2 — `cache_mut()` on the trait, default `None`

Adding a method to the public trait is a soft breaking change
because every external implementor needs to acknowledge it. We
mitigate by giving it a default impl returning `None` so existing
implementors compile unchanged.

The trade-off: an engine that *does* hold a `KvCache` but forgets
to override `cache_mut()` will silently break the decorator (it'll
no-op rather than compressing). Documentation in the engine-trait
docstring + a test on each existing engine covers this.

### D3 — Compress every layer per decode step

Naive but correct. The KvCache's `quantize_layer` is a no-op when
the FP32 slot is empty (already compressed or empty layer), so
calling it on every layer per step costs O(num_layers ×
quantize-no-op-overhead) — sub-µs per step. Acceptable.

A finer policy (e.g., "compress only when `cached_len > 256` to
avoid the cost on early decode positions") is a clear follow-up.

### D4 — `EngineKind::RotorQuant { inner: Box<EngineKind>, format }`

Spec string `iso3:inner=unlimited-context` parses to
`EngineKind::RotorQuant { inner: Box::new(EngineKind::UnlimitedContext { ... }), format: Iso3 }`.
Default inner: `unlimited-context:window=512`. Comma-separated
inner params follow the existing pattern; the `inner=` key
delimits the inner engine spec from the outer compression spec.

### D5 — `RotorQuantEngine` does NOT promote layers back to FP32

When the inner engine's `decode_step` reads `cache.layers[i]`, it
gets `None` for any compressed layer, which the engine handles per
its own logic. The decorator does NOT auto-promote. Reason: the
attention forward expects FP32 K/V; reads from compressed layers
either need the upstream caller to know they're compressed (so
they call `promote_layer_to_fp32` first) or the inner engine's
read path needs to be made compression-aware. Both are out of
scope.

In practice, the inner engine's prefill writes FP32 at start of a
turn; decode steps read those FP32 layers. After each decode step
the decorator compresses. The next decode step reads compressed
slots as `None` — meaning the inner engine has to either promote
or treat it as "no cache." This is a known limitation and is
exactly why engine-level deferred-K integration is its own
follow-up. **For this sub-change we wire the seam and demonstrate
compression happens; making it usable end-to-end is the
follow-up.**

## Risks / Trade-offs

- **Risk: decorator triggers compression but inner engine then
  reads `None` from compressed layers.** → This sub-change ships a
  working compression seam, not a working full-attention
  end-to-end run. The follow-up change `engine-rotorquant-promote-on-read`
  adds the auto-promote-on-read dance.
- **Risk: trait method addition is a soft break.** → Default impl
  returning `None` keeps every existing implementor compiling.
- **Risk: nested decorator config strings get unwieldy.** → Tests
  cover the standard combinations; deeper nesting is operator
  responsibility.

## Migration Plan

Land. New decorator is opt-in via spec string. Existing engines
behave identically.

Rollback: revert. No data path changes.

## Open Questions

- **Q1: Should `cache_mut` be `cache(&self)` (immutable) too?**
  Some diagnostics paths want read-only. Recommendation: add when
  needed; YAGNI for now.
- **Q2: What happens with `TurboQuantEngine`?** TurboQuant has its
  own quantised storage; `cache_mut` returns `None`. Wrapping
  `RotorQuant(TurboQuant)` is conceptually redundant. We don't
  reject it; we no-op the compression layer. Document.
