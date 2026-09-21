# larql-kv

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Continuation-state implementations and engine selection. This crate contains
both the `KvEngine` family used with model weights/V2 execution and the
VINDEX3 `CanonicalKvState` provider. They are different integration surfaces.

## Two contracts

`KvEngine` and `AnyEngine` are defined in `larql-inference` and re-exported here.
The implementations under [engines](src/engines/) manage persistent state for
prefill/decode. Engine names and parameters are resolved by `EngineKind` in
[src/lib.rs](src/lib.rs); do not infer a fixed inventory from old benchmark tables.

[VINDEX3 state](src/vindex3/) implements the canonical interpreter's `KvState`
contract: geometry, logical position and continuation state follow the operation
plan. V2 engine selection does not automatically select a corresponding V3
state provider. The [state policy](docs/state-policy.md) distinguishes canonical
state from reconstructible/derivative state; smaller storage is not proof of
semantic equivalence.

The [cache](src/cache.rs), [generation](src/generation/) and
[profiler](src/profiler.rs) modules support cache management, engine composition
and measurement. CPU dispatch traits live in `larql-compute`, Metal in its
sibling backend, and residual codecs in `larql-boundary`.

## Use and checks

Choose an engine through the calling CLI/API's supported selector. In
particular, semantic-promotion enforcing modes must not be inferred from an
observe-only implementation; construction refuses unsupported modes.

```bash
cargo test -p larql-kv
```

The optional `gpu` feature composes Metal on macOS. Model-backed accuracy and
performance suites require their declared models and controls. Historical
numbers in PERFORMANCE.md and baselines retain their original conditions.
See [runtime/state integration](../../docs/inference-engine.md) and
[KV residency](../../docs/kv-residency-contract.md).
