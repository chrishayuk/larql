## 1. Capability spec backfill

### 1.1 Foundation crates (3 capabilities)
- [ ] 1.1.1 `models-architecture` — write specs/models-architecture/spec.md
- [ ] 1.1.2 `models-weight-loading` — write specs/models-weight-loading/spec.md
- [ ] 1.1.3 `models-quantization-formats` — write specs/models-quantization-formats/spec.md

### 1.2 Compute crates (4 capabilities)
- [ ] 1.2.1 `compute-backend-traits`
- [ ] 1.2.2 `compute-cpu-kernels`
- [ ] 1.2.3 `compute-metal-kernels`
- [ ] 1.2.4 `compute-decode-pipeline`

### 1.3 Core graph crates (3 capabilities)
- [ ] 1.3.1 `core-graph-model-and-algorithms`
- [ ] 1.3.2 `core-inference-engine`
- [ ] 1.3.3 `core-serialization`

### 1.4 Vindex crates (7 capabilities)
- [ ] 1.4.1 `vindex-format`
- [ ] 1.4.2 `vindex-extract-pipeline`
- [ ] 1.4.3 `vindex-index-and-knn`
- [ ] 1.4.4 `vindex-patches`
- [ ] 1.4.5 `vindex-quantization-storage`
- [ ] 1.4.6 `vindex-compile-engine`
- [ ] 1.4.7 `vindex-ecosystem`

### 1.5 Inference crates (6 capabilities)
- [ ] 1.5.1 `inference-forward-pass`
- [ ] 1.5.2 `inference-attention-and-kv`
- [ ] 1.5.3 `inference-walk-ffn`
- [ ] 1.5.4 `inference-trace-format`
- [ ] 1.5.5 `inference-residual-engine`
- [ ] 1.5.6 `inference-layer-graph`

### 1.6 LQL crates (4 capabilities)
- [ ] 1.6.1 `lql-grammar`
- [ ] 1.6.2 `lql-parser`
- [ ] 1.6.3 `lql-executor`
- [ ] 1.6.4 `lql-repl-and-remote`

### 1.7 CLI crates (4 capabilities)
- [ ] 1.7.1 `cli-primary-commands`
- [ ] 1.7.2 `cli-extraction-commands`
- [ ] 1.7.3 `cli-quantize-cmd`
- [ ] 1.7.4 `cli-server-and-dev-commands`

### 1.8 Server crates (5 capabilities)
- [ ] 1.8.1 `server-http-api`
- [ ] 1.8.2 `server-grpc-api`
- [ ] 1.8.3 `server-expert-service`
- [ ] 1.8.4 `server-vindex-loading`
- [ ] 1.8.5 `server-infrastructure`

### 1.9 Router crates (1 capability)
- [ ] 1.9.1 `router-grid`

### 1.10 Python bindings (1 capability)
- [ ] 1.10.1 `python-bindings`

### 1.11 Benchmark crates (2 capabilities)
- [ ] 1.11.1 `kv-cache-benchmark-strategies`
- [ ] 1.11.2 `kv-cache-benchmark-accuracy-suite`

### 1.12 Portable crates (4 capabilities)
- [ ] 1.12.1 `model-compute-native`
- [ ] 1.12.2 `model-compute-wasm`
- [ ] 1.12.3 `experts-wasm-runtime`
- [ ] 1.12.4 `experts-tier1-and-tier2-modules`

## 2. Spec→Test traceability tooling

- [ ] 2.1 Create `scripts/spec-trace.py` — markdown parser + Rust test discovery.
- [ ] 2.2 Implement `--check` mode (fail if committed `traceability.{md,json}` differs from regenerated output).
- [ ] 2.3 Implement `--list-orphans` mode (tests not referenced by any scenario).
- [ ] 2.4 Add `make traceability` target.
- [ ] 2.5 Add `make traceability-check` target invoked by `make ci`.
- [ ] 2.6 Generate initial `openspec/coverage/traceability.{md,json}` and commit.
- [ ] 2.7 Wire `make traceability-check` into the CI workflow at `.github/workflows/ci.yml`.

## 3. cargo-llvm-cov coverage gate

- [ ] 3.1 Create `coverage-thresholds.toml` with per-crate line+branch thresholds.
- [ ] 3.2 Add `make coverage-install` target (rustup component add llvm-tools-preview; cargo install cargo-llvm-cov).
- [ ] 3.3 Add `make coverage` target (runs `cargo llvm-cov --workspace --json --output-path target/llvm-cov/coverage.json` + HTML).
- [ ] 3.4 Create `scripts/coverage-check.py` — read `coverage.json` + `coverage-thresholds.toml`, fail on per-crate drops.
- [ ] 3.5 Add `make coverage-check` target.
- [ ] 3.6 Add `make ci-coverage` target chaining `coverage` + `coverage-check`.
- [ ] 3.7 Wire `make ci-coverage` into the CI workflow as a separate job (so spec-only PRs don't pay the coverage cost).
- [ ] 3.8 Document the install + invocation flow in `crates/*/CONTRIBUTING.md` (or root if a single CONTRIBUTING.md exists).

## 4. Untested-code gap report

- [ ] 4.1 Create `scripts/spec-gap.py` (untested-code mode).
- [ ] 4.2 Run `cargo public-api` per crate, parse JSON.
- [ ] 4.3 Cross-reference public symbols against `cargo llvm-cov` per-symbol coverage.
- [ ] 4.4 Generate `openspec/changes/backfill-specs/gaps-untested-code.md`.
- [ ] 4.5 Verify the report is sorted by largest gap first.
- [ ] 4.6 Add a `make gaps-untested` target.

## 5. Spec-without-test gap report

- [ ] 5.1 Extend `scripts/spec-gap.py` with `--unbacked` mode.
- [ ] 5.2 Walk OpenSpec specs (changes/backfill-specs/specs/* during the change; openspec/specs/* afterwards), collect scenarios with `unbacked` annotation or no annotation.
- [ ] 5.3 Generate `openspec/changes/backfill-specs/gaps-unbacked-scenarios.md`.
- [ ] 5.4 Add a `make gaps-unbacked` target.
- [ ] 5.5 Add `make gaps` rolling up both reports.

## 6. Workflow + CI updates

- [ ] 6.1 Update `Makefile` to chain `traceability-check` and (separately) `coverage-check` into `make ci`.
- [ ] 6.2 Update `AGENTS.md` to document spec-first workflow and capability inventory location.
- [ ] 6.3 Update `CLAUDE.md` to point at the same.
- [ ] 6.4 Add `openspec/coverage/` to `.gitignore` exemption so the generated reports are committed (they are versioned).
- [ ] 6.5 Add a top-level `openspec/README.md` explaining the workflow.

## 7. Validation + archive prep

- [ ] 7.1 Run `openspec validate backfill-specs --strict` and fix any errors.
- [ ] 7.2 Run `openspec status --change backfill-specs` — confirm all artifacts done.
- [ ] 7.3 Spot-check a sample of scenarios: do `<!-- test: -->` annotations resolve to real test functions? Run `python3 scripts/spec-trace.py --check`.
- [ ] 7.4 Run `make ci` end-to-end on a clean clone.
- [ ] 7.5 Tag the merge commit `openspec-backfill-v0.1` so the inventory is referenceable.
- [ ] 7.6 Archive after merge: `openspec archive backfill-specs`.
