# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reference the `./AGENTS.md` file for the broader context of this project.

## Quick reference: build and test commands

```bash
cargo build --workspace                           # debug build
cargo build --release -p larql-cli                # release CLI binary
cargo build --release --features metal            # with Metal GPU (Apple Silicon)
cargo test --workspace                            # full test suite (~490 tests)
cargo test -p larql-lql                           # single crate (272 tests)
cargo test -p larql-inference --features metal    # Metal GPU tests
cargo test -p <crate> <test_name>                 # single test by name
make ci                                           # fmt-check + clippy -D warnings + test
make fmt                                          # cargo fmt --all
make lint                                         # cargo clippy --workspace --tests -- -D warnings
make bench-all                                    # criterion benchmarks (parser, vindex ops, matmul)
```

Python bindings (maturin + uv, not cargo):
```bash
make python-setup && make python-build && make python-test
```

## What LARQL is

LARQL decompiles transformer weights into a **vindex** (mmap'd vector index) queryable via **LQL** (SQL-like language). The core idea: weight matrices become queryable data; neural networks become knowledge graphs you can browse, edit, and recompile without fine-tuning or GPUs.

## Architecture overview

14-crate Cargo workspace with strict one-way dependency flow:

```
model-compute          # portable compute (no LARQL deps, never imports larql-*)
    |
larql-models           # architecture detection, ModelArchitecture trait (82 methods), weight loading, quant/dequant
    |
larql-compute          # CPU BLAS + optional Metal GPU kernels, ComputeBackend trait
    |
larql-vindex           # vindex lifecycle: extract, load, KNN, patch overlay, save
    |
larql-core             # graph algorithms (BFS, pagerank, merge, diff) - no NN deps
larql-inference        # forward pass, BLAS-fused attention, WalkFfn, LayerGraph trait
    |
larql-lql              # lexer/parser/executor/REPL, 24 statement types
    |
larql-server           # HTTP + gRPC serving vindexes
larql-cli              # thin dispatcher: 30+ subcommands in commands/
larql-python           # PyO3 bindings (module: larql._native)
```

### Key design patterns

- **Trait-based dispatch**: `ModelArchitecture` (per model family), `ComputeBackend` (CPU/Metal), `LayerGraph` (forward-pass strategy), `FfnBackend` (dense/sparse FFN). Hot paths have zero model-type branching -- `FullPipelineLayer` captures all per-layer params.
- **Immutable base + patch overlay**: Base vindexes are never mutated. All edits go through `PatchedVindex` (recursive HashMap override). `COMPILE` bakes patches into a new vindex via hardlinks + selective rewrite.
- **mmap-first storage**: Gate vectors, embeddings, down weights are zero-copy mmap'd. f16 default dtype. Feature-major layout for down_weights enables sparse walk to beat dense FFN.
- **Symmetric parser/executor split**: LQL modules mirror each other -- `parser/{lifecycle,query,mutation,introspection,trace}.rs` maps to `executor/{lifecycle,query,mutation,introspection,trace}.rs`. Adding a statement means touching `ast.rs` then both sides.

### Adding an LQL statement

1. Add variant to `Statement` enum in `crates/larql-lql/src/ast.rs`
2. Add parser in the matching `crates/larql-lql/src/parser/<category>.rs`
3. Add executor in the matching `crates/larql-lql/src/executor/<category>.rs`
4. Add parser tests in `crates/larql-lql/src/parser/tests.rs`

### Supported model families

Gemma 2/3/4, Llama 2/3, Mistral, Mixtral, Qwen 2/2.5, Phi 2/3, DeepSeek V2/V3, Granite, StarCoder2, GPT-2, GPT-OSS (MoE). Each implements `ModelArchitecture` trait in `crates/larql-models/src/architectures/`.

## Key specs and docs

- LQL language spec: `docs/specs/lql-spec.md`
- Vindex format: `docs/specs/vindex-format-spec.md`
- Inference internals: `docs/inference-engine.md`
- FFN sparse walk design: `docs/ffn-graph-layer.md`
- CLI reference: `docs/cli.md`
- ADRs: `crates/larql-vindex/docs/adr/` (14 architecture decision records)
