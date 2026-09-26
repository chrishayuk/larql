# ATTR-1D — deterministic descriptive support normalization

**Status:** protocol frozen before implementation and descriptive witness,
2026-09-21

**Code authority at freeze:** `main` `0337d77bb421ef32b3a4b02725f3d5f2f540f633`
(HEAD-OBS-1 merged)

**Machine contract:**
[`bench/attr-1d/attr1d-contract.json`](../bench/attr-1d/attr1d-contract.json)

**Claim boundary:** descriptive attention-head and source support only; no causal
claim, event selection from intervention evidence, or authorization to alter
execution

ATTR-1D is the normalization layer between HEAD-OBS-1 and the two independent
downstream branches:

```text
HEAD-OBS-1 observation
        ↓
ATTR-1D deterministic transform
        ↓
SupportCoordinate + DescriptiveSupportRecord
        ├── GW-TS-1 population capture
        └── ATTR-1C intervention targets
```

It defines a physical coordinate once and reports descriptive measurements under
it. ATTR-1C may later append causal evidence keyed to the same coordinate ID, but
it cannot mutate the descriptive record or redefine the coordinate.

## Input authority and the stats-only boundary

An admitted transform binds all of the following to one execution:

1. a complete canonical VINDEX3 run record with zero loss, no head-reader failure,
   its receipt digest and execution/provenance identities;
2. HEAD-OBS-1 `HeadsObserved`, `HeadWrite`, `HeadSum`, `CarrierStats` and optional
   true-lens `Readout` events at the same position and attention site;
3. the prepared image and operation-plan identities used by that run;
4. a reader-basis manifest frozen before observation;
5. a prompt-structure manifest frozen before observation;
6. for source-level output, a lossless source-evidence sidecar produced in the
   same run from the borrowed `AttentionHeadRecord` values and bound back to the
   run receipt.

The persisted stats-level `HeadWrite.sources` field contains source positions and
attention weights. It does **not** contain each source's value vector or projected
write. It is sufficient for attention-mass summaries, but not source-token
contribution. ATTR-1D therefore refuses source contribution, source share and
source-role contribution when the lossless source-evidence sidecar is absent or
truncated. It never treats attention weight as contribution.

The sidecar contains the observed attention probability and the corresponding
borrowed value row for every visible source position, plus gate values where the
operator declares an output gate. It is an evidence artifact, not a second
attention execution. Its identity binds the run, position, layer, query head,
KV head, visible source range, prepared image and observation receipt.

## Frozen coordinate identity

`SupportCoordinate` names only the physical location needed to recover the same
observed contribution:

```text
SupportCoordinate
    schema = larql.attr1d.support-coordinate.v1
    model_system_identity
    logical_plan_identity
    component_identity
    input_layout_identity
    position
    operator = attention
    layer
    site
    coordinate_kind = site | bias | head | source
    query_head?          # head/source only
    kv_head?             # head/source only
    source_start?        # source only, inclusive
    source_end?          # source only, exclusive
    source_role?         # source only
    role_map_identity?   # source only
```

The coordinate ID is `sha256:` plus SHA-256 of RFC-8785-style canonical JSON of
that object with UTF-8 strings, sorted object keys and no insignificant
whitespace. Integers remain integers. The hash excludes every measurement and
the coordinate ID itself.

The coordinate is stable across replay and supported backends for the same
logical model, plan and input layout. Backend realization, prepared-image
fingerprint, execution identity, receipt and event sequence belong to a separate
observation binding:

```text
SupportObservationIdentity
    schema = larql.attr1d.support-observation.v1
    support_coordinate_id
    execution_identity
    provenance_fingerprint
    prepared_image_fingerprint
    observation_receipt_digest
    event_sequence
```

Its ID is hashed by the same canonical-JSON rule. Replaying the same sealed
execution reproduces both IDs. Reference and production executions reproduce the
same `SupportCoordinate` ID but have distinct `SupportObservationIdentity` IDs;
their measurements are then compared under the backend-agreement gate.

The key contains no transition identity, subject, relation, semantic target,
answer, prompt-family label, contribution, rank, outcome, support-family label,
intervention identity, causal label or mechanism nickname. Those belong in the
record envelope, its measurements, or a later evidence object.

An ATTR-1C run targets the stable coordinate and names the exact unintervened
`SupportObservationIdentity` used as its baseline. Its intervention execution
identity remains separate. It does not replace either identity.

Source coordinates are exact half-open token spans. V1 emits one-token spans
`[position, position + 1)`; role aggregates are measurements grouped over those
coordinates, not alternative physical coordinates.

## Descriptive record

```text
DescriptiveSupportRecord
    schema = larql.attr1d.descriptive-support.v1
    attr1d_contract_identity
    transition_instance              # envelope, never coordinate key material
    support_coordinate_id
    support_coordinate
    support_observation_id
    support_observation
    parent_coordinate_id?
    reader_identity
    role_map_identity
    contribution_vector_ref?
    contribution_norm
    selected_token_projection[]
    signed_projection_share[]?
    absolute_mass_share[]?
    attention_weight?                 # source only
    source_role?                      # source only
    source_role_projection[]?         # head summary
    source_role_share[]?              # head summary
    selected_token_standing[]?        # site summary; true-lens observation
    reconstruction_error
    observation_receipt
    coverage_status
```

Vectors are content-addressed sidecars when retained. A record without a vector
may still carry lawful statistics. Missing evidence is expressed through
`coverage_status = refused` plus a reason; fields are never filled from a nearby
head, prompt, layer or execution.

## Frozen contribution algebra

For query head `h`, mapped KV head `g(h)`, source position `t`, observed attention
probability `a[h,t]`, value row `v[g(h),t]`, optional activated output gate `q[h]`,
and the prepared image's effective output-projection slice `W_O,h`:

```text
u[h,t] = q[h] ⊙ (a[h,t] · v[g(h),t])     # omit q when no gate exists
c[h,t] = W_O,h · u[h,t]
c[h]   = Σ_t c[h,t]
o      = b_O + Σ_h c[h]
```

For a declared post-attention RMSNorm:

```text
alpha  = 1 / sqrt(mean(o²) + eps)
gain   = weight_offset + w_post
G      = residual_scale · alpha · gain
d[h,t] = G ⊙ c[h,t]
d[h]   = Σ_t d[h,t]
d_bias = G ⊙ b_O
delta_hat = d_bias + Σ_h d[h]
```

Without a post-attention norm, `alpha = 1` and `gain = 1`. Without output bias,
`d_bias = 0`. Once-only bias is a `bias` coordinate; it is never assigned to a
head or source. All products use the prepared production realization that made
the observation. Widened checkpoint weights are not an alternative authority.

The transform records `d[h,t]` as source contribution, `d[h]` as head
contribution, and `delta_hat` as the site reconstruction. It verifies the
persisted `HeadWrite` norm/projection against `d[h]`; it does not trust and copy
those numbers without recomputation when source evidence is requested.

### Reader projections and shares

Each reader vector `r[j]` is content-addressed in the frozen basis manifest. It
is derived only from the prepared effective output head by a declared transform:
`selected_token_row/v1` or `token_contrast/v1`. The primary reader keeps the raw
row scale (`identity/v1`) so its value remains in direct output-head projection
units. An optional `l2_normalize/v1` direction is a separate, always-declared
reader identity and a secondary diagnostic; it cannot replace the primary result.
No reader is fitted to observed support or intervention outcomes.

```text
p_site[j] = dot(r[j], recorded_delta)
p_bias[j] = dot(r[j], d_bias)
p_head[h,j] = dot(r[j], d[h])
p_source[h,t,j] = dot(r[j], d[h,t])
```

`selected_token_projection` means these fixed linear projections. The raw
`selected_token_row/v1 + identity/v1` value may be called direct logit
attribution in the conventional output-head-projection sense; a normalized row
may not. Neither is the nonlinear true-lens logit change or an intervention
effect. True-lens rank/log-probability is copied separately as
`selected_token_standing` from a joined `Readout` event.

For denominator epsilon `1e-12`:

```text
signed head share = p_head / p_site
absolute head mass share = |p_head| / (|p_bias| + Σ_h |p_head|)
signed source share = p_source / p_head
absolute source mass share = |p_source| / Σ_t |p_source|
```

A signed share is absent with refusal reason `unstable_signed_denominator` when
the denominator magnitude is at most epsilon. An absolute share is absent when
its nonnegative denominator is at most epsilon. Raw projections remain present.
Cancellation is reported through signed and absolute surfaces; it is not clipped
or renormalized away.

Source-role projection is the sum of `p_source` over exact source coordinates
carrying that role. Its signed and absolute shares use the same frozen rules.

## Reader and source-role mapping

The reader-basis manifest binds token IDs, row transforms, prepared-image
fingerprint, dtype/realization and exact vector hashes before observation. The
answer emitted by the model cannot add or remove a reader.

The prompt-structure manifest binds exact token spans from construction-time
tokenization. Roles are:

```text
BOS | RELATION | ENTITY | LAST | OTHER
```

No decoded-string search or post-run token inspection assigns roles. BOS,
RELATION and ENTITY spans must each be internally valid and mutually disjoint;
overlap refuses the role map. The deterministic precedence for each visible
source position is:

1. `BOS` when in a declared BOS span;
2. `RELATION` when in the declared relation span;
3. `ENTITY` when in the declared subject/entity span;
4. `LAST` when it is the declared final prompt position and has no earlier role;
5. `OTHER` otherwise.

Thus an entity occupying the final prompt position remains `ENTITY`; `LAST` may
legitimately be empty. Generated positions and unlabelled template tokens are
`OTHER`. Positions outside the execution's visible source range refuse rather
than being clipped.

## Acceptance gates

All gates are conjunctive.

| Gate | Frozen requirement |
|---|---|
| Head reconstruction | `norm(delta_hat - recorded_delta) / max(norm(recorded_delta), 1e-12) <= 1e-4` |
| Source reconstruction | for every head, `norm(Σ_t d[h,t] - d[h]) / max(norm(d[h]), 1e-12) <= 1e-5` |
| Projection reconstruction | absolute error of each reconstructed reader sum `<= 1e-6 * max(1, sum of absolute projected terms)` |
| Attention probability | `abs(Σ_t a[h,t] + sink - 1) <= 1e-6` for every head |
| Coverage | every eligible `HeadsObserved` event has exactly the declared query-head set and every visible source row; no missing or duplicate coordinate |
| Refusal accounting | every uncovered operator/layer, missing sidecar, truncated source surface, unstable denominator or unsupported norm is explicit |
| Replay | canonical serialization and every coordinate ID are byte-identical across two transforms of the same sealed inputs |
| Backend agreement | on a fixture where reference and production are both defined: identical coordinate set and `abs(a-b) <= 1e-5 * max(1, abs(a), abs(b))` for every scalar measurement |

A head-level-only result may pass head reconstruction and replay while declaring
source coverage refused. It cannot pass the full ATTR-1D gate or open source-level
GW-TS-1/ATTR-1C consumers.

## Claim vocabulary

The serialized schema permits descriptive names such as `contribution`, `share`,
`source`, `support`, `projection`, `standing`, `rank`, `positive`, `negative` and
`cancellation`.

It forbids causal or mechanistic-role claims including `necessary`, `necessity`,
`sufficient`, `sufficiency`, `causal`, `cause`, `driver`, `decider`, `mediator`,
`responsible`, `essential`, `redundant`, `transporter` and `amplifier`. The
validator applies this prohibition to schema keys and enum/string labels, not to
free-form refusal messages quoting an upstream error.

ATTR-1D does not select a “winning” head or assign mechanism nicknames. It emits
coordinates, numbers, coverage and refusals.

## Sealed descriptive witness

After implementation, run one non-intervened production witness at zero-based
L24 attention, final prompt position, chosen mechanically as the lexicographically
smallest train `TransitionIdentity` in the frozen GW-0 manifest that has all three
prompt families; use its canonical prompt family. The witness identity, exact row,
tokens, reader basis and role map are sealed before execution.

The witness passes only when every acceptance gate above holds, its output
round-trips exactly, and a second transform of the same sealed evidence has the
same artifact hash. It reports head/source/role projections and true-lens
standings without naming a causal role or comparing intervention outcomes.

This witness establishes that the normalization is lawful on one real-model row.
It does not establish population recurrence, prediction, necessity or
sufficiency.

## Downstream join

GW-TS-1 may copy the coordinate and descriptive measurements into its
`physical_support` payload, subject to its independently frozen event-selection
rule. It cannot alter ATTR-1D coordinates from recurrence results.

ATTR-1C emits a separate object:

```text
CausalSupportEvidence
    support_coordinate_id
    baseline_support_observation_id
    intervention_identity
    observed_effect
    control_effect
    verdict
```

The causal object references the descriptive coordinate ID. It does not replace
the `DescriptiveSupportRecord`, and intervention results never flow backward into
ATTR-1D normalization.

## Non-goals and amendment rule

ATTR-1D does not define FFN feature support, choose GW-TS-1 path events, perform an
intervention, adjudicate GW-HEAD-1, infer correctness, or authorize partial
execution. FFN support remains under its separately frozen GW attribution law.

The ontology, algebra, reader/role mapping, gates and vocabulary are frozen before
implementation and witness execution. Corrections receive an explicit amendment
and a new contract identity; results live in a separate witness/adjudication
artifact and do not rewrite this contract.

## Pre-implementation amendment 1

The first frozen artifact, identity `sha256:caea00f9…`, placed
`baseline_execution_identity` inside `SupportCoordinate` while also requiring
identical coordinate sets across reference and production. Those requirements
cannot both hold when backend realization participates in execution identity.
Before implementation or witness execution, the coordinate was split from the
execution-specific `SupportObservationIdentity` above. This amendment receives a
new contract identity; the earlier artifact has no evidentiary use.
