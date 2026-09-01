# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, opencode, etc.) when working with code in this repository.

## What this project is

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files that can be queried like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge. The core claim: the model *is* the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning.

Extraction tiers gate which LQL statements work: `browse` (DESCRIBE/WALK/SELECT), `inference` (+INFER), `all` (+COMPILE); `attention` is a tier for the client-only slice for `run --ffn URL`. Patches (`.vlp` JSON files) stack onto a readonly base vindex — INSERT/DELETE/UPDATE auto-start a patch; base files are never mutated.

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
larql-vindex-spec the v1 manifest schema contract — ExtractLevel/StorageDtype/quant
                  enums, shard cap. Dependency-light leaf several crates depend on.
larql-execution   execution-refusal semantics shared across runtime crates;
                  deliberately empty of larql-* deps (contract leaf, same precedent
                  as larql-vindex-spec).
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
                  (connectors/projector.rs). CPU-only — Metal lives in the peer
                  below (larql-compute ADR-019). ADR-0022 moved substrate down from
                  larql-inference; the substrate is now self-contained.
    ↓
larql-compute-metal  Metal GPU backend (first-class PEER of larql-compute — same
                     trait surface, owns its kernels, NOT downstream of it).
                     Ships custom MSL shaders, multi-layer pipelining,
                     stage-bisected kernels. Default features workspace-wide pull
                     it in and it only compiles on macOS.
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save,
                  Vindexfile. Implements `KvIndex for VectorIndex` (Step 3a).
                  VINDEX3 container format lives in `src/format/vindex3/`
                  (plan/encode/verify/execute; spec in
                  docs/vindex3-format-spec.md + docs/vindex3-format.md).
                  Payload sources are a trait (`encode/source`): a local
                  checkpoint dir, or an `hf://` repo read by byte range —
                  `plan`/`encode` admit a repo from its staged safetensors
                  headers and never download the checkpoint
                  (docs/vindex3-remote-source.md).
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   engines (Standard, MarkovResidual, Apollo, etc.), chat,
                  sessions, tokenizer, FFN routing impls (Graph/Remote/MoE),
                  layer_executor, layer_graph orchestration, V3 runtime
                  (`src/vindex3/`: Vindex3Runtime, PreparedVindex3,
                  LogitsSession). KvEngine trait (with supports_multimodal +
                  prefill_from_hidden per ADR-0023), AnyEngine dispatch enum
                  (KvEngine | RetrievalEngine). The inference-shaped layer
                  that composes substrate primitives + engine state; the
                  substrate itself lives in larql-compute.
    ↓
larql-kv          pluggable KV-cache engines — 10 implementations (standard,
                  markov-rs, boundary-per-layer, turbo-quant, apollo, …),
                  state-policy classified (canonical vs derivative), W10 mask
                  cascade, CanonicalKvState for the V3 runtime
                  (`src/vindex3/`). Depends on larql-inference (its dev-dep
                  cycle stays out of the substrate).
larql-boundary    confidence-gated BOUNDARY ref codec (used by larql-kv)
    ↓
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes (V2 and VINDEX3
                  containers — bootstrap::load_artifact forks on generation)
larql-router      layer-shard router for distributed larql-server; pairs with
                  larql-router-protocol (generated tonic/prost + QUIC wrapper)
larql-factory     Vindex Factory driver: recipe schema, build_id, structural
                  validation, capability manifest, card generator
larql-cli         top-level `larql` binary (subcommands live in
                  commands/{primary,extraction,query,dev,diagnostics}).
                  Multi-modal: `--image` + `--mm-weights` flags on `larql run`,
                  image decode/resize (image_input.rs), plan assembly
                  (run_cmd_image.rs). 3-image regression test in
                  tests/multimodal_e2e.rs (#[ignore], NOT FOR CI).
larql-factory     Vindex Factory: recipe schema, build_id canonicaliser,
                  structural validator (docs/vindex-factory.md)
larql-boundary    confidence-gated BOUNDARY ref codec (final-layer residuals
                  → contract-bearing protocol objects)
larql-demos       runnable demos of shipped capabilities — every `--example`
                  demo lives here (examples/{boundary,compute,core,inference,
                  kv,lql,models,server,vindex}/); benches stay per-crate
larql-experts     nested workspace of WASM virtual experts (wasm32-wasip1
                  cdylibs, JSON ABI) the engine dispatches to
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)
larql-demos       runnable examples, one per shipped capability

# Portable (no larql-* deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime) and optional
                      wasmtime-hosted WASM modules (features: `native`/`wasm`)
larql-vindex-spec     public vindex on-disk contract: Rust types, JSON Schema,
                      validation thresholds (canonical home of ExtractLevel)
larql-execution       execution-refusal semantics (RefusalKind) shared across
                      the runtime crates
```

**`crates/larql-experts` is its own nested workspace** (own Cargo.toml with `[workspace]` members) — it builds the `wasm32-wasip1` expert modules that `model-compute`'s `wasm` feature hosts. Root `cargo build --workspace` does not include it — which also means the workspace-wide `clippy`, `coverage` and `test` sweeps miss it, so code there is not gated by `make ci`.

**Metal is a first-class peer** (ADR-0022, 2026-05-18). Its crate has its own
[README](crates/larql-compute-metal/README.md) — read that before changing kernels,
dispatch policy or anything under `shaders/`; the operator controls and the
measurement protocol are documented there and nowhere else. `larql-compute-metal`
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

The CLI is a thin dispatcher: each `larql <cmd>` lives in [crates/larql-cli/src/commands/{primary,extraction,query,dev,diagnostics}/](crates/larql-cli/src/commands/) and is wired into the `Commands` enum in [crates/larql-cli/src/main.rs](crates/larql-cli/src/main.rs) under help headings (Run / Build / Query / LQL / Server / Research / Factory). The everyday verbs — `run`, `chat`, `bench`, `serve`, `vindex3`, `shannon` — are in `primary/`, not `extraction/` or `query/`; check where a command's siblings live before adding one. Legacy research subcommands (`larql walk`, `larql weight-extract`, …) trampoline to `larql dev <subcmd>` via an argv rewrite in `main()`, and every name in that trampoline must resolve to a real `dev` subcommand — three did not until 2026-08-23, turning a clean error into a misleading one. `larql serve` exec's into `larql-server`. `larql repl` and `larql lql` delegate to `larql_lql::run_repl`/`run_statement`.

LQL parser and executor are split: [crates/larql-lql/src/parser/](crates/larql-lql/src/parser/) and [crates/larql-lql/src/executor/](crates/larql-lql/src/executor/) both carry `lifecycle`, `query`, `mutation`, `introspection`, `trace` — though on the executor side several are now directories, and the executor additionally owns `vindex3.rs`, `compact.rs`, `knowledge.rs`, `tuning.rs`, `relation_resolver.rs` and `remote/` with no parser twin. The symmetry is a starting point, not an invariant. When adding a statement, touch the AST in [crates/larql-lql/src/ast.rs](crates/larql-lql/src/ast.rs), then both sides.

## Build, test, run

**The toolchain is pinned.** [rust-toolchain.toml](rust-toolchain.toml) fixes it at **1.98.0** with clippy and rustfmt; rustup fetches that version automatically, so do not override it with your own `stable`. This exists because CI installs the newest stable while a developer's `stable` is whenever they last ran `rustup update` — the two drifted to 1.95 vs 1.98, and clippy failed in CI on lints that could not be reproduced locally. (`Cargo.toml`'s `rust-version` is the MSRV — a different claim, checked by the `quality` workflow's MSRV job under an explicit `RUSTUP_TOOLCHAIN`, because the toolchain file otherwise overrides the installed MSRV compiler and the gate ran at 1.98 while claiming 1.88 for weeks. Today the two coincide at 1.98: the NEON dot-product intrinsics the VINDEX3 CPU integer path uses stabilised there.)

```bash
cargo build --release                             # optimised build
cargo build --release --features gpu              # GPU backend (Metal today; Vulkan/CUDA later)
cargo test                                        # entire workspace
cargo test -p larql-lql                           # single crate
cargo test -p larql-inference --features gpu      # +GPU tests (Metal on Apple Silicon)
cargo test -p <crate> <test_name>                 # single test
make ci                                           # fmt-check + clippy -D warnings + test-full
make fmt                                          # cargo fmt --all
make lint                                         # cargo clippy --workspace --tests -- -D warnings
```

- Non-macOS builds need `--no-default-features`. The default gpu feature pulls in larql-compute-metal, which only compiles on macOS; CI uses `--no-default-features` on Linux/Windows.
- `make test` is intentionally fast — `cargo test --workspace --lib --bins` (no integration tests). Use `make test-full` for `cargo test --workspace`, `make test-models` for the `#[ignore]`d model-backed goldens in larql-inference (`-- --ignored`), and `make larql-<crate>-ci` for the per-crate gate CI runs (fmt-check + lint + test + bench-test + coverage). Beyond per-crate workflows, `.github/workflows/quality.yml` adds cargo-audit/deny, MSRV, buf lint, and a dead-doc-link gate (scripts/check_doc_links.py).
- Re-bench across architectures before landing perf claims — `make bench-cross-arch` runs Gemma 3 4B, Gemma 4 31B, Llama 2, Mistral 7B, Gemma 4 26B (ADR-017): an A/B promoted on Gemma 3 4B alone must be re-bench'd here.

CLI (after `cargo build --release`): `./target/release/larql extract-index … | repl | lql '…' | convert | hf | build | serve | verify`, plus the **`larql vindex3`** family (`plan`, `ops`, `exec`, `encode`, …) — the VINDEX3 container surface, and the home of the current perf instrument `vindex3 exec --backend metal-lowered --generate N --profile`. See [docs/cli.md](docs/cli.md) for the full surface, but note it does not yet document `vindex3`; read `crates/larql-cli/src/commands/primary/vindex3_cmd/` for that family.

Python bindings are maturin-built under uv (not cargo-run):

```bash
cd crates/larql-python
uv sync --no-install-project --group dev     # create .venv, install dev deps
uv run --no-sync maturin develop --release   # build PyO3 extension into .venv
uv run --no-sync pytest tests/               # run binding tests
```

Or via the Makefile: `make python-setup | python-build | python-test | python-clean`.

## Key architectural invariants

- **Base vindexes are immutable.** All mutation flows through `PatchedVindex` (overlay, defined in [crates/larql-vindex/src/patch/overlay.rs](crates/larql-vindex/src/patch/overlay.rs)). `INSERT/DELETE/UPDATE` auto-start a patch; `SAVE PATCH` persists it as `.vlp` JSON. Never write through to base files.
- **`COMPILE CURRENT INTO VINDEX`** bakes patches into a new standalone vindex by hardlinking base weight files (APFS fast path) and rewriting only `down_weights.bin` column-wise. No sidecar at load time.
- **Storage is mmap-first.** Gate vectors, embeddings, down weights are zero-copy `mmap`'d. f16 is the default dtype (`--f16` halves size with negligible accuracy loss). Don't load entire tensors into RAM unless an operation requires it.
- **Extraction tiers, not features.** `browse` (~3 GB), `attention` (client-side slice for `run --ffn URL`), `inference` (~6 GB), `all` (~10 GB) — gated by `ExtractLevel`, canonical in [crates/larql-vindex-spec/src/lib.rs](crates/larql-vindex-spec/src/lib.rs) (mirrored in [crates/larql-vindex/src/config/index.rs](crates/larql-vindex/src/config/index.rs)). Check level before attempting an operation; fail loudly if weights aren't present.
- **Walk FFN is sparse-by-design and can beat dense** (517ms vs 535ms on Gemma 4B) because gate KNN (K≈10) skips most of the 10,240 features per layer. If you touch FFN code, preserve this invariant — see [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md).
- **MXFP4 quantized MoE (GPT-OSS) has degraded DESCRIBE/WALK** due to 4-bit precision; `INFER` is the supported path. Don't assume all model families are equivalent — see [crates/larql-vindex/docs/operations-spec.md](crates/larql-vindex/docs/operations-spec.md).
- **A tensor the extractor doesn't name is silently dropped.** There is no coverage audit: extraction writes what `ModelArchitecture` returns keys for, and anything else in the checkpoint vanishes without a warning. This cost GPT-OSS 5 of its 11 per-layer attention tensors (four projection biases + `self_attn.sinks`) for months, while the module header claimed both existed — fixed 2026-07-29, but the *class* of bug is still open. When adding an architecture, diff the source tensor inventory against the written `weight_manifest.json`; when adding a tensor kind, check both `write_f32.rs` and `write_kquant/norms.rs` ask for it. See [docs/k3-funnel.md](docs/k3-funnel.md) §4.6.
- **Diff the forward before you theorise about it.** `larql shannon layer-dump` + `scripts/dump_layers_hf.py` + `larql shannon layer-diff` give a per-layer f32 comparison against HF and name the *first* drifting capture; `shannon verify` only compares one scalar at the end, so it says *that* two engines disagree and never *where*. Reach for the layer diff first — it closed OLMoE and GPT-OSS in an afternoon each after weeks of scalar-level guessing. See [docs/k3-funnel.md](docs/k3-funnel.md) §4.8–4.9.
- **A short fixture cannot test a long-range behaviour, and will pass.** GPT-OSS's sliding-attention layers use a 128-token window, so an 85-token layer diff computes the identical thing on sliding and full layers — it passed while half the model's attention was still wrong. Before quoting a gate result, ask which behaviours the fixture is *structurally capable* of distinguishing: this is the same failure as an `out_features = 2` split test and a `--bytes 384` corpus slice. Three instances, three subsystems — see [docs/k3-funnel.md](docs/k3-funnel.md) §4.9.1.
- **A config fact belongs in the trait default, not in one architecture.** `norm_topk_prob`, `rope_type: yarn` and `layer_types` were each parsed into `ModelConfig` and then answered by a hardcoded `ModelArchitecture` default, so every family that didn't hand-write an override was silently served wrong. If `config.json` states the answer, read it in `config/architecture.rs`; return `None`/`false` only when the honest answer is "this family has none". Four instances so far — see [docs/k3-funnel.md](docs/k3-funnel.md) §4.7.8, §4.9.1.
- **A performance candidate is valid only between two mutually agreeing controls.** The unit of measurement is `baseline / candidate / baseline`, warmed to plateau first; if the two brackets disagree by more than ~1% the block is void and is **not** averaged. This is stronger than interleaving `A/B/A/B`, which has a positional bias — the candidate always sits later in the run — and which only controls for drift that is monotonic. Two further rules that no cheap gate enforces: exclusivity is established by **handshake with every peer session** (`ListAgents` shows peers that `ps` does not, and contention can start after a pre-block idle check passes), and a gate stated in percent cannot be evaluated against a field printed at coarser precision than the gate. A bracket catches drift; only the handshake catches steady contention. Measured cases where each of these flipped a headline — including a three-round block that inverted sign, and a 2.8× same-round split between two arms with every cheap gate reading clean — are in [docs/kv-attention-scaling.md](docs/kv-attention-scaling.md) §Run hygiene.
- **Substrate-vs-engine split** (ADR-0022): all CPU forward-pass math + attention + KvDispatch/AsyncComputeBackend traits live in `larql-compute`, not `larql-inference`. When adding a new substrate primitive (a kernel, an attention variant, a new norm), put it in `larql-compute` and re-export from `larql-inference` for back-compat. When adding engine-shaped code (a new session type, an FFN routing impl, a layer-graph dispatcher), it stays in `larql-inference`. The rule of thumb: substrate consumes `&dyn larql_compute::KvIndex` + `ModelWeights`; engines consume sessions, tokenizers, gRPC clients, layer_graphs.
- **VectorIndex is reached through `KvIndex` from substrate.** `larql-compute`'s `KvDispatch` + `AsyncComputeBackend` + `kquant_forward` take `Option<&dyn KvIndex>` parameters. `larql-vindex` impls `KvIndex for VectorIndex` in `kv_index_impl.rs`. Engine callers passing `&VectorIndex` to substrate traits coerce with `.map(|v| v as &dyn larql_compute::KvIndex)`. Don't reach for `larql_vindex::*` from inside `larql-compute` — that's the cycle the trait was created to avoid.

## Where to find things

- LQL language spec: [crates/larql-lql/docs/spec.md](crates/larql-lql/docs/spec.md) (v0.4)
- Vindex file format: [crates/larql-vindex/docs/format-spec.md](crates/larql-vindex/docs/format-spec.md) (VINDEX2); VINDEX3: [crates/larql-vindex/docs/vindex3-format-spec.md](crates/larql-vindex/docs/vindex3-format-spec.md) (container ABI) + [docs/vindex3-format.md](docs/vindex3-format.md) (model-system spec) + [docs/vindex3-runtime.md](docs/vindex3-runtime.md) (runtime/serving)
- Operations + patches: [crates/larql-vindex/docs/operations-spec.md](crates/larql-vindex/docs/operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [crates/larql-vindex/docs/ecosystem-spec.md](crates/larql-vindex/docs/ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format (.bin/.bndx/.ctxt): [crates/larql-inference/docs/trace-format.md](crates/larql-inference/docs/trace-format.md), [docs/residual-trace.md](docs/residual-trace.md)
- ADRs: [docs/adr/](docs/adr/) (0001–0026 — wire format, grid, compute-trait extraction ADR-0022, multimodal seam ADR-0023, ...). Some crates have their own specifc ADRs in `crates/<crate-name>/doc/adr`.
- KV-cache engines: [crates/larql-kv/README.md](crates/larql-kv/README.md), [crates/larql-kv/docs/state-policy.md](crates/larql-kv/docs/state-policy.md); Vindex Factory: [docs/vindex-factory.md](docs/vindex-factory.md)
- Experimental work: `~/chris-source/chris-experiments/` — numbered 01-45, grouped into foundations, compilation, routing, and shannon series
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
