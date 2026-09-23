# GW-TS-1 population-capture amendment

**Status:** frozen before population capture, 2026-09-23. Capture is
authorized only after the bridge witness below returns `bit_identical`.
**Identity:** `sha256:ef83086a42e4f86c791fc4fea3956e46eb8461280e66babce659054d0bc68c90`
**Authority commit:** `5dabdea6` (merge of PR #510)
**Amends:** population contract `sha256:d469118c…`, population manifest
`sha256:01795de6…`. Neither is rewritten.
**Machine artifact:** `bench/gw-ts-1/capture-amendment.json`, built and
checked by `scripts/gwts1_capture_amendment.py`.

The [population contract](gw-ts-1-population-contract.md) froze the capture
design but withheld permission to execute until the ATTR-1D witness it names
was in committed history. This amendment grants that permission. It is an
authority binding plus one recorded reading. It changes no population,
assignment, capture-window, reader, observation, threshold or budget field.

## 1. Authority binding

The amendment names one repository commit. Preflight refuses unless that
commit:

- carries the ATTR-1D witness (`sha256:88065f65…`), the population contract,
  the population manifest and the GW-TS-1 protocol, each byte-identical to the
  frozen bytes;
- is an ancestor of `origin/main` (the final build refuses otherwise; the draft
  skips only this check); and
- is an ancestor of the commit the capture binary is built from.

Preflight also re-verifies the working-tree witness, the ATTR-1D contract
identity, and the frozen counts: 85 identities, 255 prompt instances,
`source_top_k = 14`, 68 parent sites.

A draft first bound #510's head, `ed1b5843`. The final amendment was built
after the merge and binds the merge commit `5dabdea6`. Against that draft it
differs in exactly three fields: `status`, `attr1d_authority.repository_commit`
and the resulting `amendment_sha256`.

## 2. Child-contribution reading

The protocol scores parents as `dot(reader, applied_delta)`. For children it
says only "exact feature contributions" and "reconstructed head contribution
mass", while `SupportMeasurement.signed_contribution` is signed. A vector norm
cannot be signed, so a child's contribution is a reader projection:

```text
signed_contribution(child)   = dot(reader, child_write)
absolute_contribution(child) = |signed_contribution(child)|
mass denominator             = |dot(reader, bias_write)| + Σ_complete child space |signed|
```

This is the attention law ATTR-1D already implements
(`absolute_head_denominators` in `larql-inference/src/vindex3/attribution.rs`),
applied unchanged to FFN features. An FFN child's write is the GW-0B
per-feature vector: `activation(gate)·up` times its down column, carried
through the post-FFN norm at the complete output's statistic and the residual
scale. The per-feature vectors sum exactly to the observed write.

GW-0B's `attribute()` ranks by `contribution_l2`. Capture must not reuse that
ranking. It records the vector L2 mass per child and in total as a
**diagnostic that never selects or admits**.

## 3. Bridge witness

Main changed `AttentionHeadRecord.source_values` from `&[Vec<f32>]` to
`&[&[f32]]`. The layout is the same, but the rows are now borrowed rather than
owned. The sealed witness's `overall_pass` was produced on the owned path, so
it does not qualify a capture built on the borrowed one. Before the first
population execution, the capture binary re-runs the sealed ATTR-1D case in
two arms, and `gwts1_capture_amendment.py bridge` adjudicates them:

| Arm | Requirement |
|---|---|
| candidate | every ATTR-1D gate passes, and every witness field equals the sealed value exactly, type included, except the four run-specific identities (`execution_identity`, `observation_receipt_digest`, `provenance_fingerprint`, `support_observation_id`). Changed gate prose is reported, never counted. |
| control | the same case (`support_coordinate_id`, reader, edge), with each head's `source_values` rotated by one position. The complete gate vector must be exactly `source_split_max_relative_l2` FAIL and every other gate PASS, with `overall_pass` false. Failing a different gate, or failing an extra one, does not count. |

Only one gate can legitimately move. `describe_attention_support` projects
each head's contribution from `mixed_values`, and only the per-source splits
read `source.values`. Rotating values keeps every position and width, so the
coverage and width refusals, head-sum, probability and projection gates are
all untouched. A control that trips anything else has broken something
unrelated, and would satisfy a looser rule by accident.

The bridge runs on the production CPU backend. The sealed witness itself
records that backend ("non-intervened production CPU", in
`gates.backend_agreement`), so a Metal or other-backend run is not a bridge.

The bridge qualifies a binary on one target, not in general. `54d07e91` in
#510 shows production-backend reductions differ in summation order between
x86 SIMD and aarch64. The L24 evidence comes out of those reductions, so
bit-identity is expected only on the target the sealed witness ran on. The
witness JSON records no host ISA: CPU is sealed evidence, while the
Apple-Silicon (aarch64 macOS) host is execution provenance only. The bridge
therefore runs on that host class. On another target the bridge fails closed as
`measurements_differ`. Population capture runs on the same target triple as
the passing bridge run.

| Verdict | Capture |
|---|---|
| `bit_identical` | authorized |
| `measurements_differ` | not authorized. No cross-run tolerance is frozen, and the ATTR-1D gate thresholds are reconstruction bounds, not run-to-run bounds. A difference needs its own amendment naming the cause. |
| `control_not_rejected` | not authorized: the bridge cannot see the failure it exists to catch |
| `candidate_gates_fail` | not authorized |

## Not authorized here

GW-TS-1 extraction, calibration, recurrence, prediction and assessment. The
capture runner records; it computes no quantile and selects no event.
