# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files that can be queried like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge. The core claim: the model *is* the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning.

Three extraction levels gate which LQL statements work: `browse` (DESCRIBE/WALK/SELECT), `inference` (+INFER), `all` (+COMPILE). Patches (`.vlp` JSON files) stack onto a readonly base vindex — INSERT/DELETE/UPDATE auto-start a patch; base files are never mutated.

## The aim (load-bearing — everything else serves this)

> **Serve the largest models at blazing speed on consumer hardware, with as little GPU as possible — ideally eventually none.**

Frontier-scale models are physically incompatible with consumer hardware under naïve dense matmul: a 671B Q4 model touches ~336 GB per forward pass against ~50 GB/s consumer DDR5 — 6.7 s/token. **The bandwidth wall cannot be beaten by faster compute.** The only path through is touching *fewer weights per token*, which is what vindex, LQL, WalkFfn, the expert grid, quantisation and the KV codecs are all for. When a change makes the engine touch more bytes per token, it is going the wrong way regardless of what the benchmark says.

Two co-equal permanent tracks, neither sacrificed to the other: the **GPU track** keeps a credible baseline against ollama / vLLM / llama.cpp (without it, no measurement on this engine clears the credibility bar), and the **CPU track** drives toward the aim. Full treatment — the per-invention table, the honest achievability assessment, and the baseline-credibility threshold — is in [ROADMAP.md](ROADMAP.md) "Engine purpose".

## Workspace layout

Cargo workspace at repo root with a strict dependency chain — respect this when adding modules:

```
# LARQL-specific (depend on vindex, LQL, etc.)
larql-models      model config, architecture traits, weight loading, quant/dequant,
                  multi-modal trait surface (ModalEncoder, Connector,
                  MultiModalProtocol, EmbeddingPlan types), vision tower
                  config+weights+loader (encoders/vision_tower.rs), projector
                  weights+loader (connectors/projector.rs), shared
                  test_fixtures (behind `test-utils` feature)
    ↓
larql-compute     CPU substrate: BLAS kernels, residual norms, attention spine
                  (rope/gqa/block/decode/gpu), forward-pass primitives (embed,
                  embed_plan, EmbeddingPlan, ops, hooks, ple, layer, predict/raw),
                  kquant_forward Q4_K/Q6_K decode helpers, FfnBackend trait +
                  dense WeightFfn impl, KvDispatch + AsyncComputeBackend traits +
                  CpuBackend impls, KvIndex trait (abstracts VectorIndex for
                  substrate callers), forward_overrides env-var registry,
                  PerLayerDecodeState, vision encoder CPU forward
                  (encoders/vision_tower.rs), projector CPU forward
                  (connectors/projector.rs). ADR-0022 moved substrate down from
                  larql-inference; the substrate is now self-contained.
    ↓
larql-compute-metal  Metal GPU backend (first-class peer, NOT a thin layer).
                     Implements ComputeBackend / KvDispatch / AsyncComputeBackend
                     for MetalBackend; ships custom MSL shaders, multi-layer
                     pipelining, stage-bisected kernels.
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save,
                  Vindexfile. Implements `KvIndex for VectorIndex` (Step 3a).
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   engines (Standard, MarkovResidual, Apollo, etc.), chat,
                  sessions, tokenizer, FFN routing impls (Graph/Remote/MoE),
                  layer_executor, layer_graph orchestration. KvEngine trait
                  (with supports_multimodal + prefill_from_hidden per
                  ADR-0023), AnyEngine dispatch enum (KvEngine | RetrievalEngine).
                  Substrate moves to larql-compute; this crate is the
                  inference-shaped layer that composes substrate primitives +
                  engine state.
    ↓
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/).
                  Multi-modal: `--image` + `--mm-weights` flags on `larql run`,
                  image decode/resize (image_input.rs), plan assembly
                  (run_cmd_image.rs). 3-image regression test in
                  tests/multimodal_e2e.rs (#[ignore], NOT FOR CI).
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)

# Portable (no LARQL deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime) and optional
                      wasmtime-hosted WASM modules (features: `native`/`wasm`)
```

**Metal is a first-class peer** (ADR-0022, 2026-05-18). `larql-compute-metal`
is the same shape as a future `larql-compute-vulkan` / `larql-compute-cuda` —
its own crate, implements the same trait surface, owns its kernels. Inference
factories (`default_engine_backend()`, `default_async_engine_backend()`,
`default_compute_backend()` in `larql-inference/src/lib.rs`) compose Metal +
CPU fallback explicitly; engine-level orchestration in `layer_graph/` still
branches on `#[cfg(feature = "gpu", target_os = "macos")]` where the
hybrid + GPU prefill paths take backend-specific actions.

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
cargo build --release --features gpu              # GPU backend (Metal today; Vulkan/CUDA later)
cargo test                                        # entire workspace
cargo test -p larql-lql                           # single crate (272 tests)
cargo test -p larql-inference --features gpu      # +GPU tests (Metal on Apple Silicon)
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
- **Three extraction levels, not features.** `browse` (~3 GB), `inference` (~6 GB), `all` (~10 GB) — gated by `ExtractLevel` enum in [crates/larql-vindex/src/config/types.rs](crates/larql-vindex-spec/src/lib.rs). Check level before attempting an operation; fail loudly if weights aren't present.
- **Walk FFN is sparse-by-design and can beat dense** (517ms vs 535ms on Gemma 4B) because gate KNN (K≈10) skips most of the 10,240 features per layer. If you touch FFN code, preserve this invariant — see [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md).
- **MXFP4 quantized MoE (GPT-OSS) has degraded DESCRIBE/WALK** due to 4-bit precision; `INFER` is the supported path. Don't assume all model families are equivalent — see [docs/specs/vindex-operations-spec.md](crates/larql-vindex/docs/operations-spec.md).
- **A tensor the extractor doesn't name is silently dropped.** There is no coverage audit: extraction writes what `ModelArchitecture` returns keys for, and anything else in the checkpoint vanishes without a warning. This cost GPT-OSS 5 of its 11 per-layer attention tensors (four projection biases + `self_attn.sinks`) for months, while the module header claimed both existed — fixed 2026-07-29, but the *class* of bug is still open. When adding an architecture, diff the source tensor inventory against the written `weight_manifest.json`; when adding a tensor kind, check both `write_f32.rs` and `write_kquant/norms.rs` ask for it. See [docs/k3-funnel.md](docs/k3-funnel.md) §4.6.
- **Diff the forward before you theorise about it.** `larql shannon layer-dump` + `scripts/dump_layers_hf.py` + `larql shannon layer-diff` give a per-layer f32 comparison against HF and name the *first* drifting capture; `shannon verify` only compares one scalar at the end, so it says *that* two engines disagree and never *where*. Reach for the layer diff first — it closed OLMoE and GPT-OSS in an afternoon each after weeks of scalar-level guessing. See [docs/k3-funnel.md](docs/k3-funnel.md) §4.8–4.9.
- **A short fixture cannot test a long-range behaviour, and will pass.** GPT-OSS's sliding-attention layers use a 128-token window, so an 85-token layer diff computes the identical thing on sliding and full layers — it passed while half the model's attention was still wrong. Before quoting a gate result, ask which behaviours the fixture is *structurally capable* of distinguishing: this is the same failure as an `out_features = 2` split test and a `--bytes 384` corpus slice. Three instances, three subsystems — see [docs/k3-funnel.md](docs/k3-funnel.md) §4.9.1.
- **A config fact belongs in the trait default, not in one architecture.** `norm_topk_prob`, `rope_type: yarn` and `layer_types` were each parsed into `ModelConfig` and then answered by a hardcoded `ModelArchitecture` default, so every family that didn't hand-write an override was silently served wrong. If `config.json` states the answer, read it in `config/architecture.rs`; return `None`/`false` only when the honest answer is "this family has none". Four instances so far — see [docs/k3-funnel.md](docs/k3-funnel.md) §4.7.8, §4.9.1.
- **Plan a logical operator physically, never from tensor format alone.** The plan is a function of `(format, operation, shape, hardware, workspace lifetime)`. One logical operator has several correct physical plans selected by execution phase — decode wants packed Q4K×Q8K matvec (bandwidth-oriented), prefill wants dequant-once + GEMM with batched softmax (compute-oriented). The TTS funnel hit the same pathology three times in one day applying a decode-shaped primitive to a prefill-shaped workload, and the correct plans differed by 1.6–4x; the `workspace lifetime` term was earned separately when per-layer dequant buffers cost ~4 GB of page traffic per prefill that the warm-allocator bench never paid. Corollary for placement: not "a GPU model" but per-phase operator routing — CPU attention + Metal FFN GEMM is a legitimate plan. Measure with `LARQL_PHASE_TIMING=1` (`larql_compute::phase_timing`) before theorising. See [ROADMAP.md](ROADMAP.md) "Standing execution rule".
- **Substrate-vs-engine split** (ADR-0022): all CPU forward-pass math + attention + KvDispatch/AsyncComputeBackend traits live in `larql-compute`, not `larql-inference`. When adding a new substrate primitive (a kernel, an attention variant, a new norm), put it in `larql-compute` and re-export from `larql-inference` for back-compat. When adding engine-shaped code (a new session type, an FFN routing impl, a layer-graph dispatcher), it stays in `larql-inference`. The rule of thumb: substrate consumes `&dyn larql_compute::KvIndex` + `ModelWeights`; engines consume sessions, tokenizers, gRPC clients, layer_graphs.
- **VectorIndex is reached through `KvIndex` from substrate.** `larql-compute`'s `KvDispatch` + `AsyncComputeBackend` + `kquant_forward` take `Option<&dyn KvIndex>` parameters. `larql-vindex` impls `KvIndex for VectorIndex` in `kv_index_impl.rs`. Engine callers passing `&VectorIndex` to substrate traits coerce with `.map(|v| v as &dyn larql_compute::KvIndex)`. Don't reach for `larql_vindex::*` from inside `larql-compute` — that's the cycle the trait was created to avoid.

## Where to find things

Planning documents, by what they hold — an item leaves the roadmap the moment it acquires a gate log:

| Document | Holds |
|---|---|
| [ROADMAP.md](ROADMAP.md) | sequencing and priority — what happens next |
| [CHANGELOG.md](CHANGELOG.md) | what happened |
| [ROADMAP_STATUS.md](ROADMAP_STATUS.md) | live status board — active sequence, P0/P1 boundaries, drift checks |
| [docs/hardening-backlog.md](docs/hardening-backlog.md) | open remediation items from code reviews (still contains unclosed P0s) |
| `docs/*-funnel.md` | gates and evidence for one live programme — [tts](docs/tts-funnel.md), [k3](docs/k3-funnel.md), [dec](docs/dec-funnel.md), [fuse](docs/fuse-funnel.md) |
| `docs/adr/` | decisions |

- LQL language spec: [docs/specs/lql-spec.md](crates/larql-lql/docs/spec.md) (v0.3)
- Vindex file format: [docs/specs/vindex-format-spec.md](crates/larql-vindex/docs/format-spec.md)
- Operations + patches: [docs/specs/vindex-operations-spec.md](crates/larql-vindex/docs/operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [docs/specs/vindex-ecosystem-spec.md](crates/larql-vindex/docs/ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format (.bin/.bndx/.ctxt): [docs/specs/trace-format-spec.md](crates/larql-inference/docs/trace-format.md), [docs/residual-trace.md](docs/residual-trace.md)
- Experimental work: `~/chris-source/chris-experiments/` — numbered 01-45, grouped into foundations, compilation, routing, and shannon series
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
