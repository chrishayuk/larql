# VINDEX3 status

**Class: CURRENT.** This page describes repository capabilities, not release
availability. [Generated facts](../generated/current-facts.md) are the authority
for package version, schemas, planner semantics, extraction default and command
inventories. The [candidate specification](../../crates/larql-vindex/docs/vindex3-format-spec.md)
owns the versioned format contract; its byte-level ABI remains candidate.

| Surface | Implemented scope and boundary |
|---|---|
| Plan / encode / inspect | Local and HF source admission, graph/container construction and inspection; support is determined by admission and closure |
| Execute / serve | Canonical component program, CPU and scoped Metal realizations, sessions and HTTP serving; backend support is operator-specific |
| Represent | Compilation, codecs, selection/accounting and evidence/search machinery; quality and promotion require their own evidence |
| Observe | Canonical decode carrier records, provenance/receipts and optional logit lens through `larql vindex3 observe` |
| Observatory | Recorded-data import, validation, lenses and replay; importing a recording does not execute the model |
| Heads / attribution | Measured head evidence and descriptive-support contracts; separate capture tooling has narrower support than Standard observation |
| Intervene / graph walks | Active research with revision-specific interfaces and witnesses; no standalone `larql vindex3 intervene` verb in the generated inventory |
| V2 / LQL | Existing extraction, query, patch and compilation surfaces remain supported; generation selection is deliberate |

## Current research

The format is the stable conceptual foundation; research tests what can be
learned from execution and its representations. Read these with their own
dates, frozen scopes and stated limitations:

- **Observe:** [carrier observation](../v3-obs-1-carrier-observation.md),
  [logit lens](../v3-lens-1-logit-lens.md), and
  [Observatory bridge](../../observatory/V3-BRIDGE.md).
- **Attribute:** [descriptive head/source support](../v3-attr-1d-descriptive-support.md)
  and [head-capture evidence](../v3-observatory-head-capture.md).
- **Intervene / walk:** [transition-support protocol](../gw-transition-support-paths.md)
  and [dependency freeze](../gw-ts-1-dependencies.md), which separate
  observational prediction, causal attribution and eventual executability;
  the separate causal-read chain records the [GW-V2 result](../gw-v2-results.md)
  and the unsealed [GW-STATE-1 dependency-removal design](../gw-state-1.md).
- **Represent:** [representation contracts](../represent-v1-contract-index.md)
  and [optimizer contracts](../optimizer-contract-index.md).

The [documentation index](../README.md) includes deeper records and proposals.
An index entry or a successful unit test is not a universal model-conformance,
performance or causal witness. See the [documentation policy](../documentation-policy.md)
for CURRENT, NORMATIVE, RECORD and ARCHIVE boundaries.
