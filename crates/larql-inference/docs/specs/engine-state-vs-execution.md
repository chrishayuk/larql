# Engine State vs Execution — Specification

**Status:** 📝 Draft v0.1 (2026-05-17).
**Audience:** LARQL contributors.
**Scope:** Recut the boundary between `KvEngine` (state policy) and
the per-layer compute path (execution strategy). The existing
`FfnBackend`, `KvDispatch`, and `ComputeBackend` traits already
provide the underlying abstractions; this spec is about engines
consuming them correctly instead of re-coupling.

---

## 1. The diagnosis

The four engines that ship today
([`StandardEngine`](../../../larql-kv/src/engines/standard.rs),
[`MarkovResidualEngine`](../../../larql-kv/src/engines/markov_residual/engine.rs),
[`MarkovResidualCodecEngine`](../../../larql-kv/src/engines/markov_residual_codec/engine.rs),
[`BoundaryKvEngine`](../../../larql-kv/src/engines/boundary_kv/engine.rs))
all hit the same coupling smell. Each engine's `prefill_quant` /
`decode_step_quant` makes execution decisions that are properly the
backend's concern:

```rust
// markov_residual/engine.rs (representative)
fn prefill_quant(
    &mut self,
    weights: &mut ModelWeights,
    _ffn: &dyn FfnBackend,       // (1) IGNORED — engine substitutes its own
    index: &VectorIndex,
    token_ids: &[u32],
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    if !self.force_walk {
        if let Some(h) = fused_prefill(...) {  // (2) engine decides fused vs walk
            self.metal_prefill_done = true;     // (3) engine tracks backend state
            self.store = None;                  // (4) engine voids its own state
            return Some(h);
        }
    }
    ensure_attn_tensors_dequantised(weights, index);
    let result = rs_prefill_walk(            // (5) walk path hardcoded to
        weights, index, token_ids,           //     index — assumes local FFN
        self.window_size, backend,
    );
    ...
}
```

Each numbered concern is something the engine should not own:

1. **Engine ignores the FFN router it was handed.** Caller passes a
   `&dyn FfnBackend` (which may be a `RemoteWalkBackend`, a
   `LayerShardedBackend`, a synthetic test backend, etc.) and the
   engine substitutes a locally-constructed `WalkFfn`. Remote-FFN
   deployments silently degrade because the engine assumes local.

2. **Engine decides fused-vs-walk dispatch.** This is a backend
   capability question (does the backend have a fused fast path that
   bundles attention+FFN, and is it usable in this context?), not a
   state-policy question.

3. **Engine tracks backend internals.** `metal_prefill_done` exists
   because once the backend's fused path runs, the engine's state is
   meaningless until the next prefill. The engine is bookkeeping
   backend state to know how to behave next time.

4. **Engine voids its own state on fast-path success.** When fused
   prefill runs, `self.store = None` — the engine's residual store,
   codec cold tier, and frame archive are all bypassed. The engine
   becomes a transparent wrapper around the backend.

5. **Walk path hardcodes local FFN dispatch.** `rs_prefill_walk`
   constructs `WalkFfn::from_config(weights, index, ...)` internally.
   The caller's FFN choice (passed as `_ffn`) is not used.

The effect: the engine's contract — bounded memory via a residual
store, optional codec compression, optional boundary frames — fires
**only when the fused path fails** AND the deployment is local. Two
production scenarios where the engine should naturally shine are
broken:

- **Remote FFN (`larql bench --ffn http://shard:8080`):** Fused path
  isn't usable (it bundles FFN into a local kernel); walk path is the
  natural fit. But the walk path uses local `WalkFfn`, silently
  bypassing the remote shard.
- **Memory-bounded long context on Metal:** Fused path takes over,
  engine's bounded-memory mechanism is moot. The user explicitly
  asked for the engine's contract and got `Standard` semantics.

## 2. The principle

> **An engine is a state policy. An executor is an execution strategy.
> They compose orthogonally and shouldn't know each other's
> implementation details.**

State policy is "what residuals / K/V / frames do I retain, when do I
evict, what do I compress." Execution strategy is "how do I run one
layer's forward pass — locally fused, locally per-layer, with remote
FFN, with sharded MoE experts."

The existing trait set already isolates the execution side:

```text
ComputeBackend  ← substrate kernels (matvec by format)
KvDispatch      ← engine-facing cache primitives + (coarse_prefill,
                  coarse_decode_step) fused fast paths
FfnBackend      ← per-layer FFN dispatch (local, remote, MoE-sharded)
```

What's missing is a clean per-layer execution surface that lets the
engine ask "run layer L" without knowing whether the backend will
fuse, walk, or remote-dispatch. That's the gap this spec fills.

## 3. The contract

### 3.1 New trait: `LayerExecutor`

```rust
pub trait LayerExecutor {
    /// Run one layer's full forward pass over a prefill chunk.
    ///
    /// Returns the post-layer hidden state and the layer's K/V (which
    /// the engine may store or discard per its state policy).
    fn run_prefill_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)>;

    /// Run one layer's forward for a single decode step.
    ///
    /// `prior_kv` is the K/V state the engine wants to attend against.
    /// The implementation appends the new token's K/V and returns it
    /// alongside the new hidden state.
    fn run_decode_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        prior_kv: &SharedKV,
        abs_position: usize,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)>;

    /// Whether this executor is fused (owns its own K/V state
    /// internally) or per-layer (engines manage K/V externally).
    ///
    /// Engines whose state policy requires per-layer interception
    /// inspect this flag and either refuse fused executors or accept
    /// them and degrade transparently (see §3.2).
    fn dispatch_kind(&self) -> ExecutorDispatchKind;

    fn name(&self) -> &str;
}

pub enum ExecutorDispatchKind {
    /// Backend owns the K/V cache internally (Metal fused, CPU coarse-Q4K).
    /// Engine state policy is unenforceable; engines either accept the
    /// transparency or refuse construction with this executor.
    Fused,
    /// Per-layer Rust dispatch; engine owns K/V state externally.
    /// Engine state policy is fully expressible.
    PerLayer,
}
```

`SharedKV` is the existing `(Array2<f32>, Array2<f32>)` pair from
`larql_inference::attention`.

### 3.2 Engine consumption pattern

A state-policy-driven engine becomes a pure state machine over the
executor's outputs:

```rust
fn decode_step_quant(
    &mut self,
    weights: &mut ModelWeights,
    executor: &dyn LayerExecutor,           // ← was inferred from backend
    ffn: &dyn FfnBackend,                   // ← passed through, no longer ignored
    token_id: u32,
) -> Option<Array2<f32>> {
    if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
        // Engine's state policy cannot fire. Either:
        //   (a) bail out — `requires_per_layer()` would have caught this
        //       at construction
        //   (b) delegate transparently and accept the lost contract
        // Picked per-engine in `EngineConstructionError::FusedExecutorRejected`.
    }

    let mut h = embed(weights, token_id);
    for layer in 0..weights.num_layers {
        let prior_kv = self.kv_for_layer(layer);                  // engine policy
        let (h_out, new_kv) = executor.run_decode_layer(
            weights, layer, &h, &prior_kv,
            self.abs_position, ffn,
        )?;
        self.integrate_layer_output(layer, &h_out, &new_kv);      // engine policy
        h = h_out;
    }
    self.advance_position();
    Some(h)
}
```

The engine touches only state operations (`kv_for_layer`,
`integrate_layer_output`, `advance_position`). The per-layer compute
is opaque.

### 3.3 Concrete executors

| Executor | Dispatch | When |
|---|---|---|
| `LocalFusedExecutor` | `Fused` | Wraps `KvDispatch::coarse_prefill` / `coarse_decode_step`. The existing Metal-fused / CPU-Q4K-coarse path. |
| `LocalWalkExecutor` | `PerLayer` | Per-layer attention + FFN dispatch through the passed `FfnBackend`. The current `rs_prefill_walk` body, generalised to honor the FFN parameter. |
| `RemoteFfnExecutor` | `PerLayer` | Attention local, FFN remote via `RemoteWalkBackend`. Per-layer because each FFN call is a separate HTTP round-trip. |
| `MoeShardedExecutor` | `PerLayer` | MoE expert dispatch sharded across remote nodes (current `RemoteMoeBackend` path). |

Each implements `LayerExecutor` over the existing primitives. None of
them need to know about engine state policy.

### 3.4 Engine-to-executor refusal contract

Some engines genuinely require per-layer dispatch — without it their
state policy cannot fire. The contract:

```rust
pub trait KvEngine {
    /// Whether this engine's state policy requires a `PerLayer`
    /// executor to function as designed. Engines that say "yes" and
    /// receive a `Fused` executor at construction must return
    /// `Err(EngineConstructionError::FusedExecutorRejected)`.
    ///
    /// Default `false` — the engine accepts any executor and degrades
    /// transparently when fused.
    fn requires_per_layer_dispatch(&self) -> bool {
        false
    }
}
```

| Engine | requires_per_layer? | Behaviour with fused executor |
|---|---|---|
| `Standard` | false | Identical (no state policy beyond cache window). |
| `BoundaryKvEngine` | false | Still emits frames from final hidden state per chunk (fused-friendly). |
| `MarkovResidualEngine` | **true** | Memory contract requires the residual store; refuse construction. |
| `MarkovResidualCodecEngine` | **true** | Codec cold tier requires the residual store; refuse construction. |
| `BoundaryPerLayerEngine` | **true** | Per-layer codec choice requires per-layer dispatch. |

Callers that want the engine's contract but get back
`FusedExecutorRejected` choose: change the executor (use a
`LocalWalkExecutor` or `RemoteFfnExecutor`), or use a different engine
that's content with fused dispatch.

### 3.5 What stops being engine concerns

After this refactor, every engine's `prefill_quant` /
`decode_step_quant` body **must not**:

- Call `fused_prefill` / `coarse_prefill` directly. (Executor's job.)
- Construct a local `WalkFfn`. (Caller supplies; executor uses.)
- Pass `Some(index)` to `recompute_kv`. (Executor handles
  format-specific dispatch internally.)
- Track `metal_prefill_done` or any other backend-internal state.
  (Executor is stateless from the engine's view.)
- Branch on `backend.supports_quant(Q4_K)` or similar capability
  checks. (Executor selection happens before the engine sees the
  request.)

If an engine impl needs any of these, it's mixing state policy with
execution strategy.

## 4. What stays the same

The existing traits do not change:

- `ComputeBackend` keeps its substrate kernel surface.
- `KvDispatch` keeps `coarse_prefill` / `coarse_decode_step` etc.
  These become the *implementation* of `LocalFusedExecutor`, not
  something engines call directly.
- `FfnBackend` keeps `forward(layer, x)`. Executors call it.
- `LayerFfnRouter` keeps its per-layer FFN selection. Executors honor
  it.

What changes is the **engine ↔ backend interaction shape**:
engines previously held `&dyn EngineBackend` and reached into both
sub-traits; after the refactor they hold `&dyn LayerExecutor` and
operate purely on the per-layer surface.

## 5. Migration plan

Five phases. Each preserves the existing public API until the last.

### Phase 1 — Add `LayerExecutor` trait

`larql-inference`: introduce the trait + `ExecutorDispatchKind` enum.
No engine touches it yet. Add `LocalFusedExecutor` and
`LocalWalkExecutor` implementations wrapping the existing
`coarse_prefill` / per-layer dispatch primitives respectively.

### Phase 2 — Add executor-aware methods alongside existing

Add `prefill_quant_via_executor` / `decode_step_quant_via_executor` to
the `KvEngine` trait with default impls that fall through to the
existing `prefill_quant` / `decode_step_quant`. Engines that have
opted into the new contract override the new methods; old callers
keep working.

### Phase 3 — Migrate engines one at a time

For each engine:
1. Implement `*_via_executor` over the new contract.
2. Add `requires_per_layer_dispatch()` returning the right answer.
3. Make the old `prefill_quant` / `decode_step_quant` a thin shim
   that picks the right executor and calls `*_via_executor`.

Order: `StandardEngine` first (easiest, fused works), then
`BoundaryKvEngine` (also fused-friendly), then `MarkovResidualEngine`
+ `MarkovResidualCodecEngine` (per-layer-only), then
`BoundaryPerLayerEngine`.

### Phase 4 — Drive site migration

Update `larql-cli/src/commands/primary/bench/engine_runtime.rs` and
`generation.rs` callers to construct an executor explicitly and pass
it through. The existing `EngineKind::build` API gains an executor
parameter; the executor selection logic moves to the driver.

### Phase 5 — Retire old methods

Once all engines and call sites are migrated, mark
`prefill_quant` / `decode_step_quant` `#[deprecated]` and eventually
remove. Trait surface becomes `prefill_via_executor` /
`decode_step_via_executor`.

## 6. Error modes

The refactor introduces one new error variant:

```rust
pub enum EngineConstructionError {
    ...
    /// Engine requires a `PerLayer` executor (its state policy can't
    /// fire under a fused executor) but a `Fused` one was supplied.
    FusedExecutorRejected {
        engine: String,
        executor: String,
    },
}
```

Hard refusal at construction, not at runtime. Caller picks a
different executor or a different engine.

## 7. Use cases that become expressible

After the refactor, these compositions work natively:

| Engine | Executor | Result |
|---|---|---|
| `MarkovResidualCodecEngine` | `LocalWalkExecutor` | Today's `force_walk=true` behaviour, but the engine doesn't know about it. |
| `MarkovResidualCodecEngine` | `RemoteFfnExecutor` | Distributed inference: FFN remote, codec cold tier on the local coordinator. Bounded memory, network-bound throughput, codec runs free. |
| `BoundaryKvEngine` | `LocalFusedExecutor` | Today's "standard + boundary frames" — fused fast path, frames emitted as a side effect. |
| `BoundaryKvEngine` | `RemoteFfnExecutor` | Boundary frames become hand-off tokens between grid nodes. The frame chain is the inter-node session-resume protocol. |
| `BoundaryPerLayerEngine` | `RemoteFfnExecutor` | Per-layer codec policy where each layer's K/V representation is sized to the channel's bandwidth budget. |

None of these require new engine code — they're new executor +
existing engine compositions.

## 8. What this spec does NOT do

Explicit non-scope so future contributors don't accidentally
overload it:

- **Does not change the FFN trait.** `FfnBackend::forward(layer, x)`
  stays.
- **Does not change the K/V cache representation.** Engines that
  store per-layer `SharedKV` continue to; engines that store
  residuals continue to.
- **Does not introduce a new state policy.** The existing engines'
  contracts stay; only the *execution surface they call against*
  changes.
- **Does not promise speedup.** Local-walk-through-executor will run
  at the same speed as today's `force_walk=true`. The win is
  expressiveness, not throughput.
- **Does not auto-pick the executor.** Driver code (CLI, server)
  chooses based on deployment shape; engines don't pick.

## 9. Open questions

- **`SharedKV` interface bloat.** Should the executor return raw
  `(K, V)` tensors or an opaque `KvHandle`? The latter would let
  fused executors keep their cache GPU-resident without copying;
  the former is simpler. v0.1 spec uses `(Array2<f32>, Array2<f32>)`
  for parity with today's `recompute_kv`. Worth revisiting in Phase
  4.
- **Per-layer-policy engines.** `BoundaryPerLayerEngine` configures
  codec per layer. Does the executor need a per-layer
  `LayerCodecConfig` parameter? Probably yes — but only when the
  engine drives it; default `None`.
- **Async executors.** The async backend work (Step A3+) already
  has `AsyncComputeBackend`. Does `LayerExecutor` need an async
  variant? Probably yes for the remote executors (RemoteFfn calls
  are I/O-bound). Out of scope for v0.1; would land as
  `AsyncLayerExecutor` mirroring the existing async backend split.
- **MoE hybrid layers.** `FfnBackend::forward_moe_full_layer`
  exists for hybrid MoE that takes over the whole layer. Should
  this be a separate `MoeLayerExecutor` variant, or stay as an
  FFN-side concern? Lean: stay as FFN — the per-layer executor
  delegates to FFN which optionally subsumes the layer.

## 10. Spec lineage

This spec emerged from the engine-decoupling exercise of 2026-05-17.
Trigger: the `force_walk` flag added that day exposed the underlying
problem — the engines couldn't choose between fused and walk
dispatch without intermingling state policy with execution strategy.
Adding more renames (the session's other thread) was treating
symptoms; this spec describes the structural cut.

The existing `KvDispatch` /  `FfnBackend` /  `ComputeBackend`
separation in `larql-inference` is the right substrate; this spec is
about engines consuming it correctly instead of re-coupling.
