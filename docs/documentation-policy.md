# Documentation authority

**Class: CURRENT.** Documentation describes either the source checkout, a
versioned contract, a recorded experiment, or a superseded design. Those are
different authorities.

| Class | Meaning | Change rule |
|---|---|---|
| CURRENT | Maintained explanation of the implementation | Update with the code; link to generated facts for volatile values |
| NORMATIVE | Explicitly versioned contract | Change through the contract's version/transition process |
| RECORD | Dated protocol, measurement or experimental finding | Preserve original claims; add an explicitly dated correction or successor |
| ARCHIVE | Superseded orientation or design | Retain for provenance and point to its successor |

The small CURRENT spine is the root README, this policy, the documentation
index, [VINDEX3 overview](vindex3/what-is-vindex3.md),
[architecture](vindex3/architecture.md), [execution](vindex3/execution.md),
[representation](vindex3/representation.md),
[observation and intervention](vindex3/observation-and-intervention.md),
[status](vindex3/status.md), [generated facts](generated/current-facts.md),
and the READMEs for [larql-vindex](../crates/larql-vindex/README.md),
[vindex-cli](../crates/vindex-cli/README.md),
[larql-cli](../crates/larql-cli/README.md),
[larql-inference](../crates/larql-inference/README.md) and
[Observatory](../observatory/README.md).

Deep implementation guides remain useful but are not automatically certified
by membership in the index. Frozen specifications, preregistrations, ADRs and
experiment records retain their own scope and date. This policy does not
retroactively reclassify every document or rewrite historical evidence.
[Superseded entry points](archive/README.md) preserve links to the pre-reset
READMEs at an immutable Git revision.

## Mechanical authority

`python3 scripts/current_facts.py --write` generates JSON and Markdown from
the package manifest, Rust constants, candidate specification version and CLI
command declarations. CI checks regeneration. Binary tests independently compare
the facts with Rust constants and Clap's command trees; a source-parser success
alone does not prove the executable's public surface.

CURRENT prose should link to the facts instead of copying versions and schema
numbers. The checks cover the generated facts, not every claim in prose.
Capability, fidelity and performance claims still need a scoped witness.

Downstream consumers can use `--export PATH` to obtain those same facts plus
the checkout SHA, dirty flag and hashes of authority files. A clean checkout
export identifies source; it does not assert that the checkout is `main` or
that its package version is a published release. Websites must keep release
availability separate and preserve the source provenance they consume.

## Maintenance

When a capability changes, update the relevant CURRENT page and its entry
point, regenerate facts when needed, and link the contract or evidence. Do not
promote a local experiment to a supported CLI feature. Do not reinterpret a
record's schema number, benchmark or verdict as a claim about today's build.
