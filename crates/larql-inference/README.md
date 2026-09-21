# larql-inference

**Class: CURRENT.** Runtime and session composition for LARQL: opening model
artifacts, generation, chat, FFN routing, VINDEX3 execution records and lenses.
[Architecture](../../docs/vindex3/architecture.md) ·
[execution](../../docs/vindex3/execution.md) ·
[status](../../docs/vindex3/status.md).

CPU forward math, attention kernels, normalization and substrate dispatch traits
live in `larql-compute`; Metal is a peer backend in `larql-compute-metal`.
This crate composes them with sessions, tokenizers, routing and engine state.
Re-exports retained for compatibility do not change ownership.

## VINDEX3 runtime

The [vindex3 module](src/vindex3/) opens a container's declared component
program rather than reconstructing `ModelWeights`. `Vindex3Runtime` and
`PreparedVindex3` bind the canonical interpreter, operand/representation policy
and numerical realization. `Vindex3Session` advances execution;
`LogitsSession` exposes prefill, step and position to generation machinery.
Continuation state is supplied through the `KvState` seam.

The canonical interpreter and `PlanBackend` contract live in `larql-vindex`.
This crate owns run-record composition, provenance/receipts, vocabulary lenses
and attribution adapters around those execution events. Observation uses the
same decode traversal. See the [runtime guide](../../docs/vindex3-runtime.md)
and [observation guide](../../docs/vindex3/observation-and-intervention.md).

## Other engine surfaces

| Module | Responsibility |
|---|---|
| [layer_graph](src/layer_graph/) | Layer orchestration, generation and distributed routing |
| [ffn](src/ffn/) | Engine-level FFN composition including remote routing |
| [vindex](src/vindex/) | V2 opening and index-backed FFN execution |
| [kv_engine](src/kv_engine/) | Engine traits consumed by the KV implementations |
| [chat](src/chat/) | Chat templates and conversation rendering |
| [trace](src/trace/) | Residual trace handling |
| [vindex3](src/vindex3/) | Container runtime, sessions, records and attribution |

V2 and V3 have different authority models. The V3 path does not translate the
container into a V2 index in order to execute it. Backend capability and
representation compatibility are checked at opening/preparation boundaries.

## Development

```bash
cargo test -p larql-inference
```

Use `--no-default-features` on non-macOS platforms. Model-backed ignored tests
need their declared checkpoints and are separate from the ordinary crate
suite. Runnable capability examples live in [larql-demos](../larql-demos/);
benchmarks remain with the owning crates.

For deeper references see [inference-engine.md](../../docs/inference-engine.md),
[FFN routing](../../docs/ffn-graph-layer.md),
[KV state policy](../larql-kv/docs/state-policy.md), and
[the documentation index](../../docs/README.md).
