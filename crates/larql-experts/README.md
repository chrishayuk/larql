# larql-experts

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

A **separate Cargo workspace** of WASM tool experts and their shared interface.
Root `cargo build --workspace`, root tests, clippy and coverage do not include
these packages. The [generated inventory](../../docs/generated/workspace-facts.md)
lists the nested members separately.

## Guest and host boundary

Each expert consumes an operation name and structured JSON arguments. The
shared [expert-interface](expert-interface/) defines metadata/results and the
exports used by the host: `larql_call`, `larql_metadata`, `larql_alloc` and
`larql_dealloc`. Operations are advertised by each expert's metadata; the
registry dispatches by operation name, not natural-language parsing.

The current host is
[larql-inference::experts](../larql-inference/src/experts/). It owns loading,
registry/session lifecycle, dispatch and sandbox limits. Model routing and
prompt-to-operation translation belong above the guest ABI. The separate
[model-compute](../model-compute/README.md) solver host uses a different ABI;
the two are not interchangeable.

## Build and validate

Install the Rust `wasm32-wasip1` target, then run from the repository root:

```bash
cargo build --manifest-path crates/larql-experts/Cargo.toml --target wasm32-wasip1 --release
cargo test --manifest-path crates/larql-experts/Cargo.toml --workspace
```

The first command builds guest artifacts in the nested workspace's target
directory. The second runs host-target unit tests where supported; host dispatch
and WASM ABI behavior require their own integration witnesses. The compiled
module cache, memory/fuel limits and reclamation rules belong to the host loader,
not to claims about every guest's numerical correctness.

Each member's Cargo manifest and source declare its name and supported operations.
Keep finite-output/error behavior and deterministic computation explicit when
adding an expert. Do not infer model-wide routing competence from a successful
direct guest call.

See [virtual expert dispatch](../../docs/virtual-experts-dispatch.md) for the
engine-level integration and its distinction from residual-triggered virtual
experts. Earlier load-time measurements remain historical records in the
[archived README](../../docs/archive/README.md).
