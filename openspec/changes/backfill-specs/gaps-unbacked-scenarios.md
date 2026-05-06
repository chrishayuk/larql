# Unbacked Scenarios

**Total: 52 scenario(s).**

Scenarios with no resolved `<!-- test: ... -->` annotation. Either explicitly marked `unbacked`, missing the annotation, or referencing a test that the discovery tool can't find.

## cli-extraction-commands

- **`larql extract` builds vindexes at the requested level / Browse-level extract carves the gate-only footprint** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:14`)
- **`larql extract` builds vindexes at the requested level / Inference-level extract carves projections + gate** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:19`)
- **`larql extract` builds vindexes at the requested level / Full-level extract carries the entire weight set** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:24`)
- **`larql extract` builds vindexes at the requested level / GGUF input flows through the same extract pipeline** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:29`)
- **Weight and attention walk research commands extract circuits without inference / weight-extract refuses to run on a missing-FFN vindex** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:91`)
- **Weight and attention walk research commands extract circuits without inference / attention-extract emits edges scoped to the requested layer range** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:96`)
- **Weight and attention walk research commands extract circuits without inference / vector-extract NDJSON is suitable input for `extract-index`** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:101`)
- **QK template / rank / mode / OV-gate commands probe attention circuits / `qk-templates --layers 0-1` emits one record per (layer, head)** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:116`)
- **QK template / rank / mode / OV-gate commands probe attention circuits / `qk-rank` reports a non-decreasing singular-value ladder** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:121`)
- **QK template / rank / mode / OV-gate commands probe attention circuits / `qk-modes` requires a gate index** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:126`)
- **QK template / rank / mode / OV-gate commands probe attention circuits / `ov-gate` maps each OV head to a gate feature ID** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:131`)
- **Bottleneck and projection-test commands isolate ablation effects / attn-bottleneck reports per-head delta KL** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:147`)
- **Bottleneck and projection-test commands isolate ablation effects / ffn-bottleneck reports per-feature impact** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:152`)
- **Bottleneck and projection-test commands isolate ablation effects / bottleneck-test substitutes early layers with rule-based logic** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:157`)
- **Bottleneck and projection-test commands isolate ablation effects / projection-test exercises rank-k residual flow** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:162`)
- **Bottleneck and projection-test commands isolate ablation effects / ffn-overlap measures entity-vs-truth overlap** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:167`)
- **Predict, walk, and circuit-discover commands compose into research pipelines / `predict --top-k K` emits exactly K candidates** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:183`)
- **Predict, walk, and circuit-discover commands compose into research pipelines / `walk --predict` reports the gate-KNN trajectory** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:188`)
- **Predict, walk, and circuit-discover commands compose into research pipelines / `circuit-discover` emits per-circuit JSON** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:193`)
- **Predict, walk, and circuit-discover commands compose into research pipelines / Cache shorthand resolves the same way as `larql run`** (`openspec/changes/backfill-specs/specs/cli-extraction-commands/spec.md:198`)

## cli-quantize-cmd

- **`larql convert quantize` is the family entry point with per-format flags / `quantize` exposes both FP4 and Q4K subcommands** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:15`)
- **`larql convert quantize` is the family entry point with per-format flags / Each format owns an isolated flag surface** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:20`)
- **`larql convert quantize` is the family entry point with per-format flags / Adding a new format does not touch existing format flags** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:25`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / Default invocation produces an Option B vindex** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:42`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / Existing destination without `--force` exits with code 4** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:47`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / Compliance floor miss under `--strict` exits with code 2** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:52`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / Compliance floor miss without `--strict` downgrades and continues** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:57`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / `--no-sidecar` skips the JSON compliance file** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:62`)
- **`larql convert quantize fp4` enforces the precision policy and compliance gate / Atomic write — partial output is never tagged as complete** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:67`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / Default mix is Q4_K_M with Q6_K down** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:84`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / `--down-q4k` switches FFN down to Q4_K uniformly** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:89`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / `--feature-major-down` emits the W2 sidecar** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:94`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / Browse-only source is rejected with a level hint** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:99`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / Already-quantised source is rejected** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:104`)
- **`larql convert quantize q4k` produces an Ollama-compatible mix / Existing destination without `--force` aborts with exit code 4** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:109`)
- **Quantise CLI surfaces backend describe + diagnostics by default / Default summary prints backend, compression, wall time** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:124`)
- **Quantise CLI surfaces backend describe + diagnostics by default / `--quiet` suppresses the summary** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:129`)
- **Quantise CLI surfaces backend describe + diagnostics by default / Compliance miss hints at the JSON sidecar** (`openspec/changes/backfill-specs/specs/cli-quantize-cmd/spec.md:134`)

## cli-server-and-dev-commands

- **`larql serve` delegates to `larql-server` with documented args / `serve <shorthand>` resolves the cache shorthand to a vindex path** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:13`)
- **`larql serve` delegates to `larql-server` with documented args / TLS, gRPC, layers, and shard flags pass through unchanged** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:18`)
- **`larql serve` delegates to `larql-server` with documented args / Missing `larql-server` binary surfaces a clear install hint** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:23`)
- **`larql serve` delegates to `larql-server` with documented args / Server binary's exit code is propagated** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:28`)
- **`larql repl` and `larql lql` delegate to `larql_lql` / `larql repl` enters the LQL REPL loop** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:41`)
- **`larql repl` and `larql lql` delegate to `larql_lql` / `larql lql "STATEMENT"` prints every output line** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:46`)
- **`larql repl` and `larql lql` delegate to `larql_lql` / `larql lql "BAD STATEMENT"` propagates the parse error** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:51`)
- **`larql dev ov-rd` exposes the residual-decomposition workbench / `ov-rd capture` writes residual artefacts under `--out`** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:119`)
- **`larql dev ov-rd` exposes the residual-decomposition workbench / `ov-rd oracle-pq fit` produces a fit report** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:124`)
- **`larql dev ov-rd` exposes the residual-decomposition workbench / `ov-rd` subcommand list matches the documented tool set** (`openspec/changes/backfill-specs/specs/cli-server-and-dev-commands/spec.md:129`)

## compute-decode-pipeline

- **MoeLayerWeights routing and combination semantics / Split fire/collect default delegates to combined moe_fn** (`openspec/changes/backfill-specs/specs/compute-decode-pipeline/spec.md:142`)

## models-weight-loading

- **Selective in-memory weight release via `drop_*` methods / drop_lm_head replaces lm_head with an empty 0x0 array** (`openspec/changes/backfill-specs/specs/models-weight-loading/spec.md:219`)
- **Selective in-memory weight release via `drop_*` methods / drop_embed replaces embed with an empty 0x0 array** (`openspec/changes/backfill-specs/specs/models-weight-loading/spec.md:224`)

## vindex-ecosystem

- **HuggingFace publish, download, and resolve with checksum verification / PublishOptions exposes a stable default** (`openspec/changes/backfill-specs/specs/vindex-ecosystem/spec.md:34`)

