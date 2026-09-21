# VINDEX3 architecture

**Class: CURRENT.** [Generated versions and schemas](../generated/current-facts.md).
The [full stack map](../architecture-stack.md) and [workspace facts](../generated/workspace-facts.md)
cover all crate owners, optional backends and dependency kinds.

```text
checkpoint directories / HF repositories
  → inventory and semantic admission
  → system graph
  → encoded container: objects, representations, provenance
  → component operation plan and operand closure
  → prepared execution image and numerical backend
  → canonical decode
  → generation / observation records / scoped interventions
```

Source admission distinguishes declared configuration, resolved semantics and
what the graph can actually carry. Planning remote HF sources reads config
and tensor headers; encoding fetches payload ranges. A representability plan
(`SystemPlan`) and an executable component plan (`ComponentOpPlan`) serve
different purposes. Their identities must not be conflated with the container
index schema or the graph schema.

| Owner | Responsibility |
|---|---|
| `larql-core` | Generic knowledge graphs and algorithms; a dependency leaf distinct from the V3 system graph |
| `larql-models` | Source inventory, config, architecture and weight loading |
| `larql-vindex` | Container lifecycle, system graph, plans, representations, canonical interpreter and observation seams |
| `larql-compute` | CPU numerical kernels and substrate traits |
| `larql-compute-metal` | Metal arithmetic and lowering backend; a peer of CPU compute |
| `larql-inference` | Opening/preparing V3 runtimes, sessions, generation, records and lenses |
| `larql-kv` | Continuation-state providers and cache engines |
| `vindex-cli` | Format-native planning, encoding, inspection, representation compilation and export |
| `larql-cli`, `larql-lql`, `larql-server` | Command, query and serving interfaces |
| Observatory | Evidence import, validation and replay UI |

The canonical interpreter owns operation ordering and model meaning.
`PlanBackend` owns arithmetic. A representation codec, a lowering provider and
a runtime backend are distinct contracts. Unsupported execution must refuse;
an admitted graph is not proof that every backend can run every operator.

V2 base files remain immutable: patch overlays express changes and compilation
writes a new artifact. V3 canonical and alternate representations carry their
own authority and fidelity; approximate bytes must not be relabelled exact.

See the [container implementation guide](../vindex3-format.md),
[lowering inventory](../lowering-plane-inventory.md),
[codec contract](../represent-codec-contract.md) and
[runtime guide](../vindex3-runtime.md) for the detailed seams.
