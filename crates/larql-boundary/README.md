# larql-boundary

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Residual codecs, confidence metadata and admission gates for BOUNDARY frames.
The crate accepts vectors and caller-supplied logits; it does not load weights,
run a model or implement a complete KV engine. It has no `larql-*` dependencies.

| Source | Responsibility |
|---|---|
| [codec](src/codec/) | Encode/decode residual payloads |
| [metadata](src/metadata.rs) | Confidence and reconstruction metadata from supplied readouts |
| [gate](src/gate.rs) | Decide whether compression is acceptable under a gate configuration |
| [frame](src/frame.rs) | Contract-bearing frame serialization and validation |

Frame size depends on vector width and codec. A codec round-trip is not a
universal behavioral guarantee: gate thresholds, readout choice and fallback
policy must be calibrated for the declared use. Chunking, replay and integration
with hot continuation state belong to callers such as
[larql-kv](../larql-kv/README.md).

```bash
cargo test -p larql-boundary
```

See [the boundary demos](../larql-demos/examples/boundary/README.md) for runnable
examples. VINDEX3 execution records and their receipts have a separate contract;
a BOUNDARY frame does not stand in for an observation record.
