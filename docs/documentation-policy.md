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

The CURRENT spine comprises the root README, this policy, the documentation
index, the [stack architecture](architecture-stack.md),
[compute/source guide](compute-substrate.md), [runtime surface map](runtime-surfaces.md),
[inference guide](inference-engine.md), the [VINDEX3 overview](vindex3/what-is-vindex3.md)
and its architecture/execution/representation/observation/status companions.
The root workspace crate READMEs, [nested experts guide](../crates/larql-experts/README.md)
and [Observatory README](../observatory/README.md) are the entry points to their
respective capabilities. The manifest-derived [workspace inventory](generated/workspace-facts.md)
links every package; [VINDEX3 facts](generated/current-facts.md) track versions,
schemas and CLI commands. A new root member needs a CURRENT README.

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

`python3 scripts/workspace_facts.py --write` derives the separate root/nested
workspace inventories from Cargo manifests, including optional and target-specific
normal/build/dev dependencies. CI checks regeneration and root crate entry-point
coverage. These are manifest declarations, not a feature-resolved build graph.

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
