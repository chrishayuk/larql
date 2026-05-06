## Why

LARQL is shifting to spec-first development: every line of code that ships should
be a known, tested output of a specification we vibe-code from. Today the project
has rich per-crate documentation (specs, ADRs, READMEs) and ~3,670 tests across
~196k lines of source, but no formal capability inventory, no Requirement → Test
traceability, and no measured code coverage gate. Backfilling now — before the
v0.2 surface stabilises — anchors every future change to an OpenSpec capability
and lets us enforce 100% spec coverage of tests and 100% test coverage of code
in CI.

## What Changes

- ADD an OpenSpec capability inventory covering all 14 workspace crates
  (44 capabilities). Each capability gets a `specs/<name>/spec.md` delta with
  formal `### Requirement` blocks and `#### Scenario` blocks derived from existing
  per-crate spec docs, ADRs, and tests.
- ADD a Requirement → Test traceability tool (`scripts/spec-trace.py`) that
  parses the OpenSpec specs, scans the workspace for `#[test]` / `#[tokio::test]`
  / `#[rstest]` functions, and produces a coverage matrix
  (`openspec/coverage/traceability.md` + `traceability.json`). Each scenario
  references at least one backing test by `crate::module::test_name`. Unbacked
  scenarios are flagged in `gaps-unbacked-scenarios.md`.
- ADD `make coverage` and `make ci-coverage` Makefile targets that run
  `cargo-llvm-cov` workspace-wide, fail below a per-crate threshold (configured
  in `coverage-thresholds.toml`), and emit `target/llvm-cov/html`.
- ADD an untested-code gap report (`gaps-untested-code.md`) that lists every
  public symbol without an associated test, generated from `cargo public-api` +
  `cargo-llvm-cov` JSON output.
- MODIFY `make ci` to chain `traceability` and `coverage` checks after the
  existing fmt/clippy/test gates.
- MODIFY `AGENTS.md` and `CLAUDE.md` to document the spec-first workflow:
  every code change must reference an OpenSpec capability and either update
  scenarios or use the `unbacked` annotation.

This is **non-breaking** for runtime behavior — no source files change in this
proposal. It introduces specs, scripts, and CI gates only. Subsequent proposals
will reference these capabilities when adding behavior.

## Capabilities

### New Capabilities

Foundation crates (3):

- `models-architecture`: ModelArchitecture trait, family detection, validation, per-layer geometry (Llama, Gemma, Mistral, Mixtral, Qwen, Phi, DeepSeek, GPT-2, GPT-OSS).
- `models-weight-loading`: Safetensors and GGUF weight loaders, dtype conversion, prefix stripping, mmap.
- `models-quantization-formats`: Half/bf16, GGML (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K), MXFP4, FP4, FP4_block, FP8 codec round-trips.

Compute crates (4):

- `compute-backend-traits`: ComputeBackend, MatMul, QuantMatVec, DecodeBackend, Capability trait split and dispatch.
- `compute-cpu-kernels`: BLAS f32 matmul, hand-rolled Q4/Q6/Q8 matvec, GEGLU, fused attention, vector ops, MoE expert combine.
- `compute-metal-kernels`: Metal GPU pipeline (tiled f32, simdgroup Q4/Q6, fused QKV+norm, QK+rope, KV+attend, FFN gate+up+GEGLU+down, hybrid MoE).
- `compute-decode-pipeline`: FullPipelineLayer orchestration, dual-path Q4_K+Q6_K vs Q4_KF dispatch, prefill/decode loop, dispatch fusion.

Core crates (3):

- `core-graph-model-and-algorithms`: Graph/Node/Edge/Schema; merge, diff, BFS, DFS, walk, pagerank, shortest-path, components, filter.
- `core-inference-engine`: Multi-step inference loop (BFS over prompts), ModelProvider trait, template registry, mock + HTTP providers.
- `core-serialization`: JSON/CSV/packed/msgpack/checkpoint format round-trips and Python-compat layer.

Vindex crates (7):

- `vindex-format`: On-disk file layout, filenames, checksums, manifest, down_meta packing, index.json, golden save/load.
- `vindex-extract-pipeline`: Build vindex from safetensors/GGUF (single-pass + streaming + resumable checkpoints), label staging, metadata aggregation.
- `vindex-index-and-knn`: VectorIndex load + lookup, dense gate KNN, HNSW approximate KNN, mmap residency hints.
- `vindex-patches`: PatchedVindex overlay, .vlp JSON format, INSERT/DELETE/UPDATE, refine pass over residual KNN.
- `vindex-quantization-storage`: FP4/FP8 block codec, Q4_K streaming weights, compliance gate (Q1), precision policy A/B/C, projection-major selection.
- `vindex-compile-engine`: StorageEngine epoch tracking, MEMIT ridge solver, COMPILE CURRENT INTO VINDEX hardlink fast path.
- `vindex-ecosystem`: HuggingFace publish/resolve/download, Vindexfile DSL, clustering + describe relation labelling.

Inference crates (6):

- `inference-forward-pass`: Embedding → layers → logits orchestration, per-layer embeddings (PLE), forward hooks, logit lens.
- `inference-attention-and-kv`: Fused online-softmax GQA, RoPE / partial RoPE, QK norm, KV cache surgery (get/set/clone-position-range).
- `inference-walk-ffn`: Sparse FFN via vindex feature walk, zero-copy mmap'd down projection, Q4K dispatch.
- `inference-trace-format`: TraceStore / BoundaryStore / ContextStore NDJSON format, residual capture, decomposed traces.
- `inference-residual-engine`: Pluggable KV-cache engines (Markov residual, Apollo, Turbo Quant, UnlimitedContext) and prefill/decode loops.
- `inference-layer-graph`: Per-layer graph dispatch (dense, cached, walk, grid, MoE), template-cached prefill for known chat patterns.

LQL crates (4):

- `lql-grammar`: Lexer token classes, AST node shapes, statement / clause / expression types.
- `lql-parser`: Recursive descent parser for all statement families (lifecycle, query, mutation, introspection, trace) and pipe operator.
- `lql-executor`: Session state and verb executors (lifecycle USE/STATS/EXTRACT/COMPILE/DIFF, query SELECT/DESCRIBE/EXPLAIN/INFER/WALK, mutation INSERT/DELETE/UPDATE/MERGE/REBALANCE, introspection SHOW, trace).
- `lql-repl-and-remote`: Interactive REPL loop with statement batching and history; remote backend dispatch via HTTP to larql-server (USE REMOTE).

CLI crates (4):

- `cli-primary-commands`: run, pull, list, show, rm, link, slice, publish, bench, cache, diag, shannon — user-facing verbs.
- `cli-extraction-commands`: extract-index, compile, weight-walk, attention-walk, qk-rank, qk-modes, ov-gate, circuit-discover, attn-bottleneck, ffn-bottleneck — research/extraction subcommands.
- `cli-quantize-cmd`: convert quantize FP4 / Q4K formats with policy and compliance gates.
- `cli-server-and-dev-commands`: serve, lql REPL, dev/ov_rd analysis tools.

Server crates (5):

- `server-http-api`: HTTP query endpoints (describe, walk, select, infer, stats, relations, OpenAI-compat, embed, logits, encode, decode).
- `server-grpc-api`: tonic gRPC service for the same surface.
- `server-expert-service`: FFN expert dispatch (local + sharded MoE topologies, batch and multi-layer-batch).
- `server-vindex-loading`: Lazy bootstrap of vindex files, multi-model dir, embedding store, FFN L2 cache.
- `server-infrastructure`: Auth (API key + grid key), per-IP rate limiting, ETag caching, FPN wire format, grid announce, env flags.

Router crates (1):

- `router-grid`: HTTP request multiplexing across layer-sharded backends (static + grid mode); gRPC GridService for self-assembling enrollment; protocol message types.

Python bindings (1):

- `python-bindings`: PyVindex (describe/insert/relations/gate/walk/infer/embedding), WalkModel (zero-copy mmap'd inference), PySession (LQL via .query()), PyTraceStore / PyBoundaryStore / PyResidualTrace.

Benchmark crates (2):

- `kv-cache-benchmark-strategies`: KvStrategy trait + Standard, TurboQuant, Markov RS, UnlimitedContext, Apollo, GraphWalk strategies.
- `kv-cache-benchmark-accuracy-suite`: Top-1 token match, KL divergence, needle-in-haystack, multi-needle accuracy validation.

Portable crates (4):

- `model-compute-native`: Native Rust kernels (arithmetic + datetime + registry); deterministic, panic-free.
- `model-compute-wasm`: Wasmtime sandbox with per-call fuel/memory caps, alloc-write-solve-read ABI, cwasm cache.
- `experts-wasm-runtime`: WASM expert ABI (larql_call/larql_metadata), expert-interface crate, host-side ExpertRegistry.
- `experts-tier1-and-tier2-modules`: 19 expert modules (arithmetic, conway, date [tier 2], dijkstra, element, finance, geometry, graph, hash, http_status, isbn, logic, luhn, markov, sql, statistics, string_ops, trig, unit).

### Modified Capabilities

None. All capabilities are new — `openspec/specs/` is empty before this change.

## Impact

- **Affected files**: 44 new spec deltas under `openspec/changes/backfill-specs/specs/`; 4 new tooling scripts under `scripts/`; new `coverage-thresholds.toml`; updates to `Makefile`, `AGENTS.md`, `CLAUDE.md`.
- **Affected systems**: CI (new `traceability` + `coverage` gates); developer workflow (every change must reference a capability); Python bindings (Python tests joined into the trace map via test discovery).
- **Affected crates**: All 14 — but only specs and CI, not source.
- **Dependencies**: Adds dev dependencies on `cargo-llvm-cov` and `cargo-public-api` (installed in CI image, not the workspace lockfile).
- **Out of scope for this change**: Closing actual coverage / traceability gaps. Those are tracked in `gaps-untested-code.md` and `gaps-unbacked-scenarios.md` for follow-up changes — this proposal lands the inventory and the gates, not the fixes.
