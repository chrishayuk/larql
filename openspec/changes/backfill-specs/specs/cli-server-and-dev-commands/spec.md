## ADDED Requirements

### Requirement: `larql serve` delegates to `larql-server` with documented args

`larql serve` SHALL resolve the supplied vindex argument (cache
shorthand, `hf://`, `owner/name`, or local path) the same way
`larql run` does, MUST forward every documented option to the
sibling `larql-server` binary unchanged, and MUST exit with the
server binary's exit code. When the sibling binary is missing the
runner SHALL print an installation hint pointing at `cargo install
--path crates/larql-server` and exit non-zero.

#### Scenario: `serve <shorthand>` resolves the cache shorthand to a vindex path
- **WHEN** `larql serve gemma-3-4b-it-vindex --port 8080` is invoked with that vindex cached
- **THEN** the runner SHALL pass the resolved absolute vindex directory path (not the shorthand) to `larql-server`
<!-- unbacked -->

#### Scenario: TLS, gRPC, layers, and shard flags pass through unchanged
- **WHEN** `larql serve … --grpc-port 50051 --layers 0-19 --tls-cert C --tls-key K --moe-shards "0-63=URL"` is invoked
- **THEN** every flag value SHALL appear verbatim in the spawned `larql-server` argv
<!-- unbacked -->

#### Scenario: Missing `larql-server` binary surfaces a clear install hint
- **WHEN** the sibling binary cannot be located on disk or in `PATH`
- **THEN** stderr SHALL include `Make sure larql-server is installed` and the runner SHALL exit non-zero
<!-- unbacked -->

#### Scenario: Server binary's exit code is propagated
- **WHEN** `larql-server` exits with code 7
- **THEN** `larql serve` SHALL fail (Err) carrying the same exit status in its error message
<!-- unbacked -->

### Requirement: `larql repl` and `larql lql` delegate to `larql_lql`

`larql repl` SHALL invoke `larql_lql::run_repl` and exit `Ok(())` on
return; `larql lql "STATEMENT"` SHALL invoke `larql_lql::run_batch`,
print every returned line on stdout, and propagate parser/runtime
errors as the CLI's exit error. Neither command SHALL touch the
filesystem unless the LQL statement itself does so.

#### Scenario: `larql repl` enters the LQL REPL loop
- **WHEN** `larql repl` is invoked with stdin closed
- **THEN** the runner SHALL exit `Ok(())` after `larql_lql::run_repl` returns (EOF on stdin terminates the REPL cleanly)
<!-- unbacked -->

#### Scenario: `larql lql "STATEMENT"` prints every output line
- **WHEN** a statement returning N output lines is executed
- **THEN** stdout SHALL contain those N lines in order
<!-- unbacked -->

#### Scenario: `larql lql "BAD STATEMENT"` propagates the parse error
- **WHEN** the supplied statement fails to parse
- **THEN** the runner SHALL exit non-zero with the LQL error printed on stderr
<!-- unbacked -->

### Requirement: Legacy `dev` argv trampoline preserves pre-redesign invocations

The CLI MUST rewrite legacy top-level research verbs to `larql dev
<verb>` before clap parses argv, MUST NOT rewrite first-class verbs
(`run`, `extract`, `extract-index`, `compile`, `serve`), MUST NOT
rewrite unknown verbs (so clap still produces the canonical
"unrecognized subcommand" error), and SHALL preserve every argument
after the rewritten verb byte-for-byte. The list of legacy verbs
covered SHALL include at minimum `walk`, `weight-extract`,
`attention-extract`, `vector-extract`, `residuals`, `predict`,
`index-gates`, `attention-capture`, `qk-templates`, `qk-rank`,
`qk-modes`, `ov-gate`, `circuit-discover`, `attn-bottleneck`,
`ffn-bottleneck`, `ffn-overlap`, `kg-bench`, `trajectory-trace`,
`projection-test`, `fingerprint-extract`, `bottleneck-test`,
`embedding-jump`, `bfs`, `ffn-latency`.

#### Scenario: Primary verb argv passes through unchanged
- **WHEN** `["larql", "run", "gemma3-4b.vindex", "hello"]` is passed to the trampoline
- **THEN** the output SHALL equal the input
<!-- test: larql_cli::trampoline_tests::primary_verb_is_untouched -->

#### Scenario: Top-level `extract` and `extract-index` are not rewritten to `dev`
- **WHEN** the trampoline sees either verb
- **THEN** the argv SHALL pass through unchanged so clap dispatches to the top-level variant
<!-- test: larql_cli::trampoline_tests::top_level_extract_is_untouched -->
<!-- test: larql_cli::trampoline_tests::extract_index_alias_is_untouched -->

#### Scenario: A legacy research verb is rewritten with `dev` injected
- **WHEN** `["larql", "walk", "--index", "x.vindex", "--prompt", "hi", "--predict"]` is processed
- **THEN** the output SHALL equal `["larql", "dev", "walk", "--index", "x.vindex", "--prompt", "hi", "--predict"]`
<!-- test: larql_cli::trampoline_tests::legacy_research_verb_is_rewritten -->

#### Scenario: Every documented legacy name rewrites
- **WHEN** the trampoline runs with each name in `LEGACY_DEV_NAMES`
- **THEN** the rewritten argv SHALL place `dev` at index 1 and the legacy verb at index 2 with `--help` (or any tail) preserved
<!-- test: larql_cli::trampoline_tests::legacy_research_flag_names_all_rewrite -->

#### Scenario: Empty argv survives
- **WHEN** `["larql"]` is processed
- **THEN** the trampoline SHALL return the same vector unchanged
<!-- test: larql_cli::trampoline_tests::no_args_returns_unchanged -->

#### Scenario: Unknown verbs are not wrapped in `dev`
- **WHEN** `["larql", "typo-command"]` is processed
- **THEN** the argv SHALL pass through unchanged (clap then emits its canonical error)
<!-- test: larql_cli::trampoline_tests::unknown_verb_is_not_rewritten -->

#### Scenario: Argument count grows by exactly one on rewrite
- **WHEN** a legacy verb invocation is rewritten
- **THEN** the output length SHALL equal the input length plus one (the inserted `"dev"` token)
<!-- test: larql_cli::trampoline_tests::rewrite_preserves_argument_count_plus_one -->

### Requirement: `larql dev ov-rd` exposes the residual-decomposition workbench

The `larql dev ov-rd` subcommand group SHALL expose at least the
documented residual-decomposition tools: capture, oracle PQ
(fit/eval/forward/stability/reports), basis, address, gamma-address,
metrics, sanity, edit-catalog, pq-exception, runtime, static-replace,
stats, zero-ablate, and the high-level `oracle-pq` driver. Each
subcommand MUST accept a vindex path plus its own analysis flags and
SHALL emit JSON or NDJSON artefacts under a user-controlled output
directory.

#### Scenario: `ov-rd capture` writes residual artefacts under `--out`
- **WHEN** `larql dev ov-rd capture --index X --prompt P --out OUT` is invoked
- **THEN** `OUT/` SHALL contain at least one JSON or NDJSON artefact
<!-- unbacked -->

#### Scenario: `ov-rd oracle-pq fit` produces a fit report
- **WHEN** the fit subcommand runs against a captured artefact
- **THEN** stdout (or `--out`) SHALL include reconstruction error and rank metrics
<!-- unbacked -->

#### Scenario: `ov-rd` subcommand list matches the documented tool set
- **WHEN** `larql dev ov-rd --help` is invoked
- **THEN** stdout SHALL list at minimum capture, oracle-pq, basis, address, gamma-address, metrics, sanity, edit-catalog, pq-exception, runtime, static-replace, stats, and zero-ablate
<!-- unbacked -->
