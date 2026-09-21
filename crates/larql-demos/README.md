# larql-demos

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Runnable examples of workspace capabilities, with explicit Cargo targets.
The authoritative target list is [Cargo.toml](Cargo.toml), also captured in
[the generated JSON inventory](../../docs/generated/workspace-facts.json).
A target being declared does not mean it can run without model files.

| Catalog | Scope |
|---|---|
| [boundary](examples/boundary/README.md) | Residual codecs and gate decisions |
| [compute](examples/compute/README.md) | Numerical kernels and compute examples |
| [core](examples/core/README.md) | Graph construction and algorithms |
| [inference](examples/inference/README.md) | Generation and runtime composition |
| [kv](examples/kv/README.md) | Continuation-state engines |
| [lql](examples/lql/README.md) | Language parsing/execution |
| [models](examples/models/README.md) | Model descriptions and detection |
| [server](examples/server/README.md) | Serving adapters |
| [vindex](examples/vindex/README.md) | Artifact storage and querying |

```bash
cargo run -p larql-demos --example chat_demo
```

Consult the relevant catalog and target source for required model paths and
features before running it. `gpu` enables the supporting macOS backend paths;
`msgpack` enables the graph serializer feature. The current manifest, rather
than a README total, determines which examples build.

Benchmarks, diagnostics and parity harnesses remain with the crate they measure.
Dated research probes retain their own evidence contract and may live outside
this workspace. A demonstration is not an independent fidelity or performance
witness. The [VINDEX3 CLI recording guide](../../docs/vindex3/observation-and-intervention.md)
is the entry point for canonical observation records.

```bash
cargo check -p larql-demos --examples
```

Model-free and model-backed checks have different prerequisites; compilation
alone never claims that the model-backed examples were executed.
