## Context

LARQL is a 14-crate Cargo workspace (~196k LoC, ~3,670 tests) that decompiles
transformer weights into a queryable vindex and executes a SQL-like query
language (LQL) against it. Documentation today is scattered:
- 9 crate-level spec docs in `crates/*/docs/` (vindex format, operations,
  ecosystem, FP4, LQL grammar, server/router, quantize, trace format).
- 30+ ADRs across `docs/adr/` and `crates/*/docs/adr/`.
- A README and ROADMAP at the repo root.
- Inline rustdoc on most public APIs.

There is no formal capability inventory, no Requirement → Test traceability,
no enforced code-coverage threshold. Adding a new feature today happens by
editing prose in a `*-spec.md` and writing matching tests, with no automated
enforcement that the prose and the tests stay in sync. As we move toward
spec-first development this gap is the highest-leverage thing to close.

OpenSpec (`@fission-ai/openspec` 1.2.0) was initialised at repo root with the
`spec-driven` schema. `openspec/specs/` is empty. This proposal lands the
inventory and the gates; subsequent proposals will fix individual gaps.

## Goals / Non-Goals

**Goals:**
- Every coherent feature area in the workspace is named by exactly one
  OpenSpec capability under `openspec/specs/<capability>/spec.md`.
- Every capability spec contains formal `### Requirement` and `#### Scenario`
  blocks. Each scenario points at one or more existing tests via a
  `<!-- test: crate::module::test_name -->` annotation, OR is marked
  `<!-- test: unbacked -->` for the gap report to pick up.
- A traceability tool produces a machine-readable matrix
  (`openspec/coverage/traceability.json`) and a human-readable view
  (`openspec/coverage/traceability.md`) on demand.
- `make ci` enforces line+branch coverage thresholds per crate via
  `cargo-llvm-cov`; PRs that drop coverage below the threshold fail CI.
- Gap reports (`gaps-untested-code.md`, `gaps-unbacked-scenarios.md`) make
  the remaining work visible and prioritisable.

**Non-Goals:**
- Closing the gaps. The proposal lands inventory + gates. Filling tests
  for currently untested code, and writing scenarios for currently
  unspec'd behavior, are tracked as follow-up changes.
- Rewriting existing per-crate spec docs. The OpenSpec specs are the new
  source of truth; the old docs become reference material that we will
  thin out over time. Deleting them is out of scope here.
- Reorganising tests. The traceability tool reads tests where they are,
  not where they "should" be.
- Python test coverage gate. Python tests under `crates/larql-python/tests/`
  are listed in the trace map but not gated by `cargo-llvm-cov` (they run
  in a different harness). A pytest-cov gate is a future change.

## Decisions

### D1 — Capability granularity: capability-grained (~44 capabilities)

We considered three options:
- **Crate-grained** (~14 capabilities): one spec per crate. Rejected —
  larql-vindex (27k LoC, 11 distinct concerns) and larql-inference (44k LoC,
  10 distinct concerns) would have unreadable specs.
- **Module-grained** (~80 capabilities): one spec per top-level src/ module.
  Rejected — tightly coupled modules (e.g., the four `executor/query/*`
  modules) want to share scenarios and dependencies.
- **Capability-grained** (~44 capabilities): one spec per coherent feature
  area, often spanning 2–10 modules within a crate. **Chosen.** Matches the
  existing per-crate spec docs (vindex-format, vindex-operations, vindex-
  ecosystem are already three capabilities) and the ADRs (each ADR is
  scoped to one capability).

The capability map (44 entries, kebab-case) is enumerated in proposal.md.

### D2 — Single mega-change vs. many small changes

Chose a single mega-change `backfill-specs` that ADDs all 44 capabilities at
once. Alternative (one change per capability) was rejected because:
- 44 PRs to land structural inventory is administrative noise.
- The capabilities reference each other (e.g. `inference-walk-ffn` cites
  `vindex-format`); landing them together avoids dangling references.
- Archive flips them all into `openspec/specs/` atomically.

Trade-off: this proposal is large to review. Mitigated by (a) splitting
into 44 self-contained spec files, (b) per-capability sub-tasks in
tasks.md so reviewers can ack capability-by-capability.

### D3 — Test annotation format inside scenarios

Two options for tying scenarios to tests:
- **Option A — Inline HTML comment**: `<!-- test: larql_vindex::format::filenames::tests::canonicalises_gate_path -->`
- **Option B — Trailing `**Test:**` line**: `- **Test:** larql_vindex::format::filenames::tests::canonicalises_gate_path`

Chose **Option A**. Reasons:
- HTML comments survive `openspec validate` (which doesn't parse them).
- Doesn't change the human-readable rendering of the spec.
- Easy to grep and parse: one regex per file.
- Matches the convention used by other spec-driven projects on top of OpenSpec.

Format detail:
```
#### Scenario: <name>
- **WHEN** <condition>
- **THEN** <outcome>
<!-- test: <crate>::<module_path>::<test_fn> -->
<!-- test: <another test> -->
```
Multiple `<!-- test: -->` annotations are allowed when several tests jointly
cover the scenario. `<!-- test: unbacked -->` flags scenarios with no
existing test, picked up by the gap report.

For Python tests, the format is `<!-- test: py:tests/test_bindings.py::test_describe -->`.
For doctests: `<!-- test: doc:crates/larql-vindex/src/format/filenames.rs:42 -->`.

### D4 — Coverage tool: cargo-llvm-cov

`cargo-llvm-cov` over `cargo-tarpaulin` because:
- Generates source-based coverage (LLVM-native), more accurate on
  inline functions and async.
- Already industry-standard for large Rust workspaces (used by tokio,
  rust-analyzer, polars).
- Has stable JSON output (`--json --output-path`) we can consume from
  the gap-report script.
- Supports per-crate filters via `-p` so we can apply different thresholds.

Trade-off: requires `llvm-tools-preview` rustup component. Documented
in the install instructions inside the new `make coverage-install` target.

### D5 — Per-crate coverage thresholds, not global

Not all crates warrant the same bar:
- `larql-router-protocol` is 19 lines of generated proto bindings. Threshold: 0%.
- `kv-cache-benchmark` is exploratory work; tests exist but coverage is uneven. Threshold: 70%.
- Production crates (`larql-vindex`, `larql-compute`, `larql-inference`, `larql-lql`, `larql-models`): threshold: 85%.
- CLI/server entry points: 75% (lots of glue + integration testing).
- `larql-python`: not gated by `cargo-llvm-cov` (tests are in pytest);
  pytest-cov gate is a future change.

Thresholds live in `coverage-thresholds.toml` so they can be raised over
time without touching the Makefile. Schema:
```toml
[crate.larql-vindex]
line = 85
branch = 80
[crate.larql-router-protocol]
line = 0
branch = 0
[default]
line = 85
branch = 80
```

### D6 — Traceability tool implementation

Python (`scripts/spec-trace.py`) over Rust because:
- Reads markdown (regex over `### Requirement:` / `#### Scenario:` / `<!-- test: ... -->`)
  and walks `crates/*/{src,tests}/**/*.rs` for `#[test]` / `#[tokio::test]` /
  `#[rstest]` definitions. Markdown + regex tasks are 5× shorter in Python.
- No build step; runs on a fresh checkout.
- We already require Python (uv) for `larql-python` development, so it's
  not a new dep.

Output:
- `openspec/coverage/traceability.json` — machine-readable. Schema:
  `{ capabilities: [ { name, requirements: [ { name, scenarios: [ { name, tests: [...], unbacked: bool } ] } ] } ], orphan_tests: [...] }`.
- `openspec/coverage/traceability.md` — table per capability, plus orphans
  (tests not referenced by any scenario).

CI check: `make traceability` regenerates and fails if either file diverges
from the committed version. Forces every test-affecting change to update
the trace.

### D7 — Untested-code report

`scripts/spec-gap.py` consumes:
- `cargo llvm-cov --json` per crate → uncovered lines.
- `cargo public-api --simplified --output-format json` per crate → public
  symbols.

Outputs `openspec/changes/backfill-specs/gaps-untested-code.md` while the
change is open; once archived the report is regenerated nightly into
`openspec/coverage/gaps-untested-code.md`.

Format: a markdown table per crate listing `module::symbol` → `file:line`
→ uncovered fraction. Sorted by largest gap first.

### D8 — Unbacked-scenario report

`scripts/spec-gap.py --unbacked` walks the OpenSpec specs, collects
scenarios with `<!-- test: unbacked -->` or no `<!-- test: -->` annotation
at all, and writes
`openspec/changes/backfill-specs/gaps-unbacked-scenarios.md`.

Each entry: `capability::requirement::scenario` plus the WHEN/THEN body
so authors of follow-up changes can see what they need to test.

### D9 — `make ci` chain

After this change lands:
```
make ci  →  fmt-check  →  clippy -D warnings  →  test  →  traceability  →  coverage
```
All five must pass. `traceability` and `coverage` are no-ops if their
inputs haven't changed (cached via mtime).

### D10 — `larql-experts` workspace pulled into capabilities even though it's a sub-workspace

`crates/larql-experts/` is itself a workspace of 21 sub-crates (the WASM
expert modules). The OpenSpec capabilities `experts-wasm-runtime` and
`experts-tier1-and-tier2-modules` cover the public ABI and the union of
op behaviors. We do not generate one capability per expert module — that
would be 19 tiny specs that all say "implements these N ops, runs in
WASM, deterministic." Each expert's specific op semantics are scenarios
under the `experts-tier1-and-tier2-modules` capability.

## Risks / Trade-offs

- **Risk: spec drift.** Specs and tests may drift after the initial backfill.
  → Mitigation: traceability check in CI fails any PR that adds/changes a
  test without updating its scenario annotation, and any PR that adds
  a scenario without a `<!-- test: -->` line.
- **Risk: backfill bias.** Writing scenarios from existing tests means we
  encode current behavior even where it's wrong. → Mitigation: this is
  acceptable for a backfill — the goal is to make behavior visible.
  Subsequent changes will correct wrong behavior with normal MODIFIED
  Requirements deltas, which document the change explicitly.
- **Risk: scenario count explosion.** 3,670 tests → ~3,670 scenarios is
  unwieldy. → Mitigation: scenarios are coarser than tests; one scenario
  may be backed by many tests. Target: ~600–900 scenarios across all
  44 capabilities (avg 15–20 per capability).
- **Risk: coverage gates block development.** Initial coverage may be
  below the proposed thresholds. → Mitigation: thresholds in
  `coverage-thresholds.toml` start at the *measured* current coverage
  per crate, rounded down to the nearest 5%. Raising them is a separate
  change with its own review.
- **Risk: traceability tool false negatives.** A test annotation that
  references a renamed function won't be caught by the regex. →
  Mitigation: the tool resolves test paths against the actual test
  inventory; unresolved annotations fail the check loudly.
- **Trade-off: in-band annotations vs. side-table.** Inline `<!-- test: -->`
  comments couple specs to test names. Renaming a test requires updating
  the spec. We accept this cost because it keeps the link from
  bit-rotting silently. A side-table (`coverage/links.toml`) would be
  more robust to renames but harder to discover when reading a spec.

## Migration Plan

1. Land this proposal: writes 44 spec deltas, the trace/coverage scripts,
   the Makefile changes, and the CI gates. No source files are modified.
2. Archive the change: `openspec archive backfill-specs` flips
   `openspec/changes/backfill-specs/specs/*` into `openspec/specs/*`.
3. Generate first reports: `make traceability && make coverage`. Commit
   the resulting `openspec/coverage/*.md` and the gap reports under
   `openspec/coverage/`.
4. Adjust `coverage-thresholds.toml` to match measured coverage minus 1%
   (the buffer absorbs flaky-test variance).
5. From here on, every PR that touches code or tests must update either
   a scenario annotation, a `<!-- test: -->` line, or a threshold (with
   reviewer ack on threshold changes).
6. Follow-up changes pick up the gap reports and add scenarios + tests
   incrementally. Each picks one or more capabilities and burns down its
   `unbacked` count.

Rollback: this proposal is purely additive (new files + Makefile targets).
Reverting is `git revert` of the merge commit.

## Open Questions

- **Q1: Should doctest names be tracked?** rustdoc doctests don't have
  named functions. We propose using `doc:<file>:<line>` as the test
  identifier, generated by `rustdoc --test --no-run -Z list-tests`.
  Confirm with whoever owns the docs CI when it's added.
- **Q2: Where do MoE per-architecture quirks live?** GPT-OSS MXFP4
  has its own behavior under `vindex-quantization-storage` and its own
  performance profile under `inference-forward-pass`. We split scenarios
  by capability rather than creating an `archs-gpt-oss` capability.
  If a future architecture diverges drastically, we can add a
  per-architecture capability without restructuring existing ones.
- **Q3: Threshold for `model-compute` and `larql-experts`?** These crates
  are small but security-relevant (WASM sandbox). We propose 90% line +
  85% branch. Confirm with the WASM owner before merging.
