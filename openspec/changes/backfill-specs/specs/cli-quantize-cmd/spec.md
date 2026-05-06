## ADDED Requirements

### Requirement: `larql convert quantize` is the family entry point with per-format flags

`larql convert quantize <format>` SHALL be the family entry point
under which each quantisation format is its own subcommand with an
isolated flag surface. The grammar MUST permit adding a new format as
a single `QuantizeCommand::FooBar { ... }` clap variant plus one
`run_quantize_foobar` dispatcher plus one library entry
(`larql_vindex::quant::vindex_to_foobar`) without touching any other
format's surface. Today the wired formats SHALL be `q4k` (Ollama-
compatible Q4_K_M mix) and `fp4` (FP4 + FP8 + optional source-dtype
gate per the precision policy).

#### Scenario: `quantize` exposes both FP4 and Q4K subcommands
- **WHEN** `larql convert quantize --help` is invoked
- **THEN** stdout SHALL list both `fp4` and `q4k` subcommands
<!-- unbacked -->

#### Scenario: Each format owns an isolated flag surface
- **WHEN** `larql convert quantize fp4 --help` and `larql convert quantize q4k --help` are inspected
- **THEN** FP4-only flags (`--policy`, `--compliance-floor`, `--threshold`, `--strict`, `--no-sidecar`) MUST NOT appear under `q4k`, and Q4K-only flags (`--down-q4k`, `--feature-major-down`) MUST NOT appear under `fp4`
<!-- unbacked -->

#### Scenario: Adding a new format does not touch existing format flags
- **WHEN** the source tree adds a `QuantizeCommand::FpN` variant with its own flag set
- **THEN** the existing FP4 and Q4K flag surfaces SHALL remain byte-identical (no shared-flag regressions)
<!-- unbacked -->

### Requirement: `larql convert quantize fp4` enforces the precision policy and compliance gate

`larql convert quantize fp4` SHALL accept `--input` and `--output`
plus the documented optional flags: `--policy {option-a, option-b,
option-c}` (default `option-b`), `--compliance-floor FRAC` (default
`0.99`), `--threshold RATIO` (default `16.0`), `--force`, `--strict`,
`--no-sidecar`, and `--quiet`. The runner MUST refuse to overwrite
the destination unless `--force` is set, MUST emit
`fp4_compliance.json` unless `--no-sidecar` is set, and MUST exit
non-zero (code 2) under `--strict` when any FP4-targeted projection
falls below the compliance floor.

#### Scenario: Default invocation produces an Option B vindex
- **WHEN** `larql convert quantize fp4 --input SRC --output DST` is invoked with no policy flag
- **THEN** the runner SHALL select Policy B (gate=source dtype, up=FP4, down=FP8) and write `DST/fp4_compliance.json`
<!-- unbacked -->

#### Scenario: Existing destination without `--force` exits with code 4
- **WHEN** `DST` already exists and `--force` is not provided
- **THEN** the runner SHALL exit non-zero (documented exit code 4) without rewriting any file
<!-- unbacked -->

#### Scenario: Compliance floor miss under `--strict` exits with code 2
- **WHEN** any FP4-targeted projection falls below `--compliance-floor` and `--strict` is set
- **THEN** the runner SHALL exit non-zero (documented exit code 2) after writing the compliance sidecar
<!-- unbacked -->

#### Scenario: Compliance floor miss without `--strict` downgrades and continues
- **WHEN** the same scenario runs without `--strict`
- **THEN** the affected projection SHALL be downgraded to the manifest fallback precision (FP8) and the run SHALL exit successfully
<!-- unbacked -->

#### Scenario: `--no-sidecar` skips the JSON compliance file
- **WHEN** `--no-sidecar` is provided
- **THEN** `DST/fp4_compliance.json` SHALL NOT be written
<!-- unbacked -->

#### Scenario: Atomic write — partial output is never tagged as complete
- **WHEN** the converter is interrupted mid-write
- **THEN** `DST/index.json` SHALL be absent, distinguishing the partial output from a complete one (writer stages into `DST.tmp/` and renames on success)
<!-- unbacked -->

### Requirement: `larql convert quantize q4k` produces an Ollama-compatible mix

`larql convert quantize q4k` SHALL accept `--input`, `--output`,
plus optional `--down-q4k` (FFN down at Q4_K instead of Q6_K),
`--feature-major-down` (emit `down_features_q4k.bin`), `--force`, and
`--quiet`. The default invocation MUST produce the Ollama-compatible
Q4_K_M mix: attention Q/K/O at Q4_K, attention V at Q6_K, FFN gate/up
at Q4_K, FFN down at Q6_K. The runner MUST reject Browse-only and
already-quantised sources with a clear error pointing at `--level
inference`. Like FP4, the writer MUST stage into `DST.tmp/` and
rename atomically on success.

#### Scenario: Default mix is Q4_K_M with Q6_K down
- **WHEN** `larql convert quantize q4k --input SRC --output DST` is run with no extra flags
- **THEN** the resulting `DST` SHALL contain `interleaved_q4k.bin` whose down portion is Q6_K and whose gate/up portion is Q4_K
<!-- unbacked -->

#### Scenario: `--down-q4k` switches FFN down to Q4_K uniformly
- **WHEN** the flag is supplied
- **THEN** the resulting interleaved file SHALL store down at Q4_K (with the documented walk_correctness gate auto-relaxing from 0.02 to 0.035)
<!-- unbacked -->

#### Scenario: `--feature-major-down` emits the W2 sidecar
- **WHEN** the flag is supplied
- **THEN** `DST/down_features_q4k.bin` SHALL be emitted alongside the standard interleaved file
<!-- unbacked -->

#### Scenario: Browse-only source is rejected with a level hint
- **WHEN** the source vindex has `extract_level: browse`
- **THEN** the runner SHALL fail with an error mentioning `--level inference`
<!-- unbacked -->

#### Scenario: Already-quantised source is rejected
- **WHEN** the source `index.json` reports `quant != none`
- **THEN** the runner SHALL fail rather than silently re-quantising
<!-- unbacked -->

#### Scenario: Existing destination without `--force` aborts with exit code 4
- **WHEN** `DST` exists and `--force` is omitted
- **THEN** the runner SHALL exit non-zero with the documented exit code 4
<!-- unbacked -->

### Requirement: Quantise CLI surfaces backend describe + diagnostics by default

`larql convert quantize` SHALL print the backend-describe summary
line (via `VectorIndex::describe_ffn_backend()`) on stdout after the
write unless `--quiet` is passed, MUST echo the compliance sidecar
path in the summary so the user can find it on a miss, and MUST emit
a one-liner suggesting `LARQL_VINDEX_DESCRIBE=1` for runtime
verification. Verbose tracing (`LARQL_WALK_TRACE=1`) is opt-in only;
the CLI itself MUST NOT spam by default.

#### Scenario: Default summary prints backend, compression, wall time
- **WHEN** `larql convert quantize fp4 --input SRC --output DST` runs to completion without `--quiet`
- **THEN** stdout SHALL include `FFN storage`, `Walk backend`, and `Wall time` lines
<!-- unbacked -->

#### Scenario: `--quiet` suppresses the summary
- **WHEN** the flag is supplied
- **THEN** no summary block SHALL be printed on stdout
<!-- unbacked -->

#### Scenario: Compliance miss hints at the JSON sidecar
- **WHEN** an FP4 run downgrades at least one projection
- **THEN** stderr SHALL include `compliance floor missed` and reference `fp4_compliance.json`
<!-- unbacked -->
