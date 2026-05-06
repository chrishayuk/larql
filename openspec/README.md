# OpenSpec workflow

This directory holds the spec-first contract for LARQL.

## Layout

```
openspec/
  config.yaml                       # OpenSpec project config (do not hand-edit)
  specs/<capability>/spec.md        # canonical specs, one per capability
  changes/<change-id>/              # in-flight proposals
    proposal.md                     # why + what
    design.md                       # how (cross-cutting changes only)
    tasks.md                        # implementation backlog
    specs/<capability>/spec.md      # delta with ## ADDED / MODIFIED / REMOVED
    gaps-unbacked-scenarios.md      # auto-generated; lists scenarios with no test
    gaps-untested-code.md           # auto-generated; lists code without tests
  changes/archive/                  # archived (merged) changes
  coverage/
    traceability.md                 # auto-generated requirement → test matrix
    traceability.json               # machine-readable trace data
```

## Capability inventory

44 capabilities across 14 crates. See
[changes/backfill-specs/proposal.md](changes/backfill-specs/proposal.md)
for the full map. After `openspec archive backfill-specs`, every capability
lives under `openspec/specs/<name>/spec.md`.

## Authoring rules

1. **One capability per coherent feature area.** Use kebab-case names.
2. **Each spec is a delta when in `changes/<id>/`.** Top-level header is
   `## ADDED Requirements` (or `MODIFIED` / `REMOVED` / `RENAMED`).
3. **Every Requirement contains SHALL or MUST** (or `SHALL NOT`). The first
   sentence must be normative — `openspec validate` parses the lead line.
4. **Every Requirement has at least one Scenario.** Scenarios use four
   hashes (`####`) — three breaks the parser silently.
5. **Every Scenario references at least one test.** Annotate with an HTML
   comment immediately after the THEN line:

   ```markdown
   #### Scenario: <name>
   - **WHEN** <condition>
   - **THEN** <outcome>
   <!-- test: <crate>::<module_path>::<test_fn> -->
   ```

   Multiple `<!-- test: -->` lines are fine. Use `<!-- test: unbacked -->`
   when there is no test yet — this is what the gap report picks up.

   Forms accepted by the trace tool:

   | Source | Annotation |
   |---|---|
   | Rust integration test | `<!-- test: larql_vindex::test_hnsw::recall_at_k -->` |
   | Rust inline `mod tests` | `<!-- test: larql_vindex::format::filenames::tests::canonicalises_gate_path -->` |
   | Rust inline non-`tests` mod | `<!-- test: larql_compute::cpu::ops::moe::cache::cache_format_tests::bf16_dispatch_round_trip -->` |
   | Python pytest | `<!-- test: py:tests/test_bindings.py::test_describe -->` |
   | Rustdoc doctest | `<!-- test: doc:crates/larql-vindex/src/format/filenames.rs:42 -->` |
   | Explicit gap | `<!-- test: unbacked -->` |

   The fuzzy resolver drops intermediate `mod` segments when there is a
   unique test of that name in the crate, so a slight rename in the
   module hierarchy does not break the link.

## Day-to-day commands

Spec only:
```bash
openspec list                                # list active changes
openspec validate <change> --strict          # validate before archive
openspec archive <change>                    # flip change → openspec/specs/
make traceability                            # regenerate trace matrix
make traceability-check                      # CI: trace must be committed
make gaps-unbacked                           # refresh unbacked-scenarios report
```

With coverage:
```bash
make coverage-install                        # one-time: rustup + cargo-llvm-cov
make coverage                                # full workspace; HTML + JSON
make coverage-check                          # enforce coverage-thresholds.toml
make ci-coverage                             # both
make gaps-untested                           # refresh untested-code report
make gaps                                    # both gap reports
```

CI gates:
```bash
make ci          # fmt + clippy + test + traceability + openspec-validate
make ci-coverage # heavier; runs in a separate job
```

## Authoring a new change

```bash
openspec new change <kebab-case-name>        # scaffold
openspec instructions proposal --change <name> --json   # what proposal needs
# write proposal.md, design.md (if cross-cutting), specs/**/spec.md, tasks.md
openspec validate <name> --strict
make traceability                            # link any new tests to scenarios
make ci
# open PR
# after merge:
openspec archive <name>                      # flip into openspec/specs/
```

## When tests change

If you rename or remove a test:

1. Run `make traceability-check`. It will fail with the orphaned annotation.
2. Update the `<!-- test: ... -->` line in the spec to match the new name,
   or delete the scenario if the behavior is gone.
3. Run `make traceability` to regenerate, then commit both files together.

## When specs change

Every PR that adds or modifies code SHOULD reference an OpenSpec capability:

- Adding new behavior → write a change proposal with `ADDED Requirements`.
- Changing existing behavior → write a change proposal with
  `MODIFIED Requirements` (copy the *whole* requirement block, then edit).
- Removing behavior → write a change proposal with `REMOVED Requirements`
  including `**Reason:**` and `**Migration:**` lines.

The `MODIFIED` workflow is easy to get wrong: if you only paste a partial
block, archive will silently drop the rest. Copy the entire
`### Requirement` block including all scenarios, then edit in place.
