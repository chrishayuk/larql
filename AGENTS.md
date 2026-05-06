# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files that can be queried like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge. The core claim: the model *is* the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning.

Three extraction levels gate which LQL statements work: `browse` (DESCRIBE/WALK/SELECT), `inference` (+INFER), `all` (+COMPILE). Patches (`.vlp` JSON files) stack onto a readonly base vindex — INSERT/DELETE/UPDATE auto-start a patch; base files are never mutated.

## Workspace layout

Cargo workspace at repo root with a strict dependency chain — respect this when adding modules:

```
# LARQL-specific (depend on vindex, LQL, etc.)
larql-models      model config, architecture traits, weight loading, quant/dequant
    ↓
larql-compute     CPU/Metal matmul backends, pipeline
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save, Vindexfile
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   forward pass, BLAS-fused attention, Metal GPU, WalkFfn, trace
    ↓
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/)
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)
kv-cache-benchmark    standalone benchmark crate

# Portable (no LARQL deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime) and optional
                      wasmtime-hosted WASM modules (features: `native`/`wasm`)
```

**`model-compute` never imports `larql-*`.** Dependency flow is one-way:
LARQL may consume it (e.g. for compile-time `sum(1..100)` resolution); it
knows nothing about vindex or LQL. When it moves to a sibling repo, the
name stays the same so imports don't churn. The `install_edge` primitive
that stamps a compiled edge into gate/up/down tensors lives at
[crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs](crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs) —
it's the lowest-level step of the `COMPILE` verb and isn't a separate crate
until a second consumer needs it.

The CLI is a thin dispatcher: each `larql <cmd>` lives in [crates/larql-cli/src/commands/extraction/](crates/larql-cli/src/commands/extraction/) or [crates/larql-cli/src/commands/query/](crates/larql-cli/src/commands/query/) and is wired into the `Commands` enum in [crates/larql-cli/src/main.rs](crates/larql-cli/src/main.rs). `larql serve` exec's into `larql-server`. `larql repl` and `larql lql` delegate to `larql_lql::run_repl`/`run_statement`.

LQL parser and executor are split symmetrically: [crates/larql-lql/src/parser/](crates/larql-lql/src/parser/) and [crates/larql-lql/src/executor/](crates/larql-lql/src/executor/) both have matching `lifecycle.rs`, `query.rs`, `mutation.rs`, `introspection.rs`, `trace.rs`. When adding a statement, touch the AST in [crates/larql-lql/src/ast.rs](crates/larql-lql/src/ast.rs), then both sides.

## Build, test, run

```bash
cargo build --release                             # optimised build
cargo build --release --features metal            # Metal GPU backend (Apple Silicon)
cargo test                                        # entire workspace
cargo test -p larql-lql                           # single crate (272 tests)
cargo test -p larql-inference --features metal    # +Metal GPU tests
cargo test -p <crate> <test_name>                 # single test
make ci                                           # fmt-check + clippy -D warnings + test
make fmt                                          # cargo fmt --all
make lint                                         # cargo clippy --workspace --tests -- -D warnings
```

CLI (after `cargo build --release`): `./target/release/larql extract-index … | repl | lql '…' | convert | hf | build | serve | verify`. See [docs/cli.md](docs/cli.md) for the full surface.

Python bindings are maturin-built under uv (not cargo-run):

```bash
cd crates/larql-python
uv sync --no-install-project --group dev     # create .venv, install dev deps
uv run --no-sync maturin develop --release   # build PyO3 extension into .venv
uv run --no-sync pytest tests/               # run binding tests
```

Or via the Makefile: `make python-setup | python-build | python-test | python-clean`.

## Key architectural invariants

- **Base vindexes are immutable.** All mutation flows through `PatchedVindex` (overlay) — see [crates/larql-vindex/src/patch/core.rs](crates/larql-vindex/src/patch/). `INSERT/DELETE/UPDATE` auto-start a patch; `SAVE PATCH` persists it as `.vlp` JSON. Never write through to base files.
- **`COMPILE CURRENT INTO VINDEX`** bakes patches into a new standalone vindex by hardlinking base weight files (APFS fast path) and rewriting only `down_weights.bin` column-wise. No sidecar at load time.
- **Storage is mmap-first.** Gate vectors, embeddings, down weights are zero-copy `mmap`'d. f16 is the default dtype (`--f16` halves size with negligible accuracy loss). Don't load entire tensors into RAM unless an operation requires it.
- **Three extraction levels, not features.** `browse` (~3 GB), `inference` (~6 GB), `all` (~10 GB) — gated by `ExtractLevel` enum in [crates/larql-vindex/src/config/types.rs](crates/larql-vindex/src/config/types.rs). Check level before attempting an operation; fail loudly if weights aren't present.
- **Walk FFN is sparse-by-design and can beat dense** (517ms vs 535ms on Gemma 4B) because gate KNN (K≈10) skips most of the 10,240 features per layer. If you touch FFN code, preserve this invariant — see [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md).
- **MXFP4 quantized MoE (GPT-OSS) has degraded DESCRIBE/WALK** due to 4-bit precision; `INFER` is the supported path. Don't assume all model families are equivalent — see [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md).

## Spec-first workflow (OpenSpec)

LARQL is spec-driven. The 14-crate workspace is decomposed into
**44 OpenSpec capabilities** under `openspec/specs/<capability>/spec.md`.
Every Requirement uses SHALL/MUST language; every Scenario references at
least one backing test via a `<!-- test: <fqn> -->` HTML comment.

- Authoring rules and the full capability map: [openspec/README.md](openspec/README.md).
- Initial inventory + design: [openspec/changes/backfill-specs/](openspec/changes/backfill-specs/).
- Requirement → test traceability matrix: [openspec/coverage/traceability.md](openspec/coverage/traceability.md).
- Outstanding gaps: [openspec/changes/backfill-specs/gaps-unbacked-scenarios.md](openspec/changes/backfill-specs/gaps-unbacked-scenarios.md) and [gaps-untested-code.md](openspec/changes/backfill-specs/gaps-untested-code.md).

Workflow:

1. Add or change behavior → write a change proposal under `openspec/changes/<id>/`.
2. Tests for new scenarios live in the same crate as the capability they back.
3. `make ci` runs fmt/clippy/tests + `make traceability-check` + `openspec validate`.
4. `make ci-coverage` runs cargo-llvm-cov against per-crate thresholds in `coverage-thresholds.toml` (separate CI job).
5. After merge, `openspec archive <change>` flips the change into `openspec/specs/`.

## Where to find things

Each capability spec already references its own per-crate docs and ADRs. The legacy entrypoints are:

- LQL language spec: [crates/larql-lql/docs/spec.md](crates/larql-lql/docs/spec.md)
- Vindex file format: [crates/larql-vindex/docs/format-spec.md](crates/larql-vindex/docs/format-spec.md)
- Operations + patches: [crates/larql-vindex/docs/operations-spec.md](crates/larql-vindex/docs/operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [crates/larql-vindex/docs/ecosystem-spec.md](crates/larql-vindex/docs/ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format: [crates/larql-inference/docs/trace-format.md](crates/larql-inference/docs/trace-format.md), [docs/residual-trace.md](docs/residual-trace.md)
- Experimental work: [experiments/](experiments/) — numbered 01-07, each self-contained
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
