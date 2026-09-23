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
| Heads / attribution | `observe --heads`, per-head coverage and reconstruction, measured evidence and descriptive-support contracts; softmax capture has narrower support than Standard observation |
| Intervene | `observe --intervene` supports declared carrier changes and in-kernel CPU softmax head counterfactuals; backend/operator scope is explicit |
| Graph walks | Active transition-support research; its prediction, attribution and executability claims have separate gates |
| V2 / LQL | Existing extraction, query, patch and compilation surfaces remain supported; generation selection is deliberate |

## Planned capabilities

Two capabilities are on the [VINDEX3 roadmap](../../ROADMAP.md#planned-v3-capabilities-added-2026-09-22)
and are **not** part of the current surface. Both are refused explicitly rather
than approximated; the refusal is the truthful boundary until each has a frozen
execution contract and parity gates.

| Planned | Current behaviour |
|---|---|
| **Partial execution / sharding** | `larql serve` refuses `--layers`, `--experts`, `--units`, `--moe-remote`, `--ffn-only`, `--embed-only` and `--no-infer` on a VINDEX3 container, and a V3 server joining a grid announces no shards. Preparation has an internal layer-range slice that consumes hidden states and refuses token ids; it is research substrate, not a supported sharding contract. |
| **Multimodal embedding handoff** | `larql run` refuses `--image` and `--mm-weights` on a VINDEX3 container. Perception towers can be admitted as components, but no V3 execution path accepts externally produced embeddings. Text generation is scoped so that it does not depend on the tower. |

Partial execution comes first: the multimodal handoff should reuse its
boundary, identity and provenance rules rather than add a special-case path.

## Current research

The format is the stable conceptual foundation; research tests what can be
learned from execution and its representations. Read these with their own
dates, frozen scopes and stated limitations:

- **Observe:** [carrier observation](../v3-obs-1-carrier-observation.md),
  [logit lens](../v3-lens-1-logit-lens.md), and
  [Observatory bridge](../../observatory/V3-BRIDGE.md).
- **Attribute:** [descriptive head/source support](../v3-attr-1d-descriptive-support.md)
  and [per-head observation](../v3-head-obs-1-per-head-observation.md).
- **Intervene:** [carrier intervention](../v3-intervene-1-carrier-intervention.md)
  and [in-kernel head intervention](../v3-intervene-2-head-intervention.md),
  including the additive-versus-counterfactual comparison.
- **Walk:** [transition-support protocol](../gw-transition-support-paths.md)
  and [dependency freeze](../gw-ts-1-dependencies.md), which separate
  observational prediction, causal attribution and eventual executability.
- **Represent:** [representation contracts](../represent-v1-contract-index.md)
  and [optimizer contracts](../optimizer-contract-index.md).

The [documentation index](../README.md) includes deeper records and proposals.
An index entry or a successful unit test is not a universal model-conformance,
performance or causal witness. See the [documentation policy](../documentation-policy.md)
for CURRENT, NORMATIVE, RECORD and ARCHIVE boundaries.
