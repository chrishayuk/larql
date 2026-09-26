# GW-TS-1 — operator-aware multi-site transition support

**Status:** protocol frozen before population capture, 2026-09-21

**Code authority at freeze:** `main` `0337d77bb421ef32b3a4b02725f3d5f2f540f633`
(HEAD-OBS-1 merged)

**Machine contract:**
[`bench/gw-ts-1/gwts1-protocol.json`](../bench/gw-ts-1/gwts1-protocol.json)

**Claim boundary:** held-out structure and prefix prediction; not causality and not
execution substitution

GW-TS-1 asks whether a semantic transition has a reproducible ordered physical
realization across operator families, and whether an observed prefix predicts the
next support event. It succeeds or fails as a path hypothesis. It does not define
the semantic transition by the path used to realize it.

The primary claim is frozen as:

> A held-out prefix predicts subsequent operator-specific support better than
> marginal-frequency, layer-frequency, unordered-set, permutation,
> operator-swap and strongest-single-event controls at matched candidate budget.

The predecessor evidence justifies testing this object but is not evidence that
the primary claim holds. GW-3A-F rejected gate-derived feature postings as a
physical execution index. GW-4A found reusable FFN write families. GW-4B found
strong but non-progressive single-site retrieval. GW-SUP-1 did not establish a
generic basin. GW-CONV-1 nominated an attention landmark. HEAD-OBS-1 closed the
attention head/source decomposition and measured its capture cost.

## Frozen ontology

The identity layers are deliberately separate:

```text
TransitionIdentity
    subject / semantic source state
    relation / semantic operation
    target / semantic destination
    task identity
    preregistered answer identity

TransitionInstance
    transition_identity
    prompt_family_identity
    prompt_identity
    execution_identity
    observed outcome

TransitionSupportPath
    transition_instance
    extraction_contract_identity
    ordered SupportEvent[]
```

`TransitionIdentity` is semantic and realization-agnostic. It contains no prompt
wording, model execution fingerprint, model architecture, layer, site, head,
feature address, support-family ID or path-derived label. `source state` means a
semantic source, never a residual vector. `answer identity` means the target
declared before execution, never the answer emitted by the model.

`TransitionInstance` identifies one realization. Prompt family is here, rather
than in `TransitionIdentity`, so the same transition may be compared across
paraphrase families. The observed answer, ranks and probabilities are outcomes;
they cannot change the parent identity.

`TransitionSupportPath` is evidence about how that instance executed. One
transition may legitimately have different paths across prompt families,
executions, realizations or architectures. Path disagreement is a result, not an
identity error.

### Event identity is not event measurement

```text
SupportEvent
    ordinal
    coordinate: SupportCoordinate
    measurement: SupportMeasurement

SupportCoordinate
    operator: Attention | FFN
    layer
    normalized_layer_depth
    site
    physical_support

SupportMeasurement
    signed_contribution
    absolute_contribution
    operator_normalized_mass
    reconstruction_residual
    observation_identity
```

Contribution, reconstruction error and observation identity are measurements of
an event, not part of its physical coordinate. The same coordinate may recur with
a different magnitude or sign without becoming a different event by definition.
`ordinal` belongs to the path occurrence; it is not part of the reusable physical
support identity.

The tagged `physical_support` value is operator-specific:

- `AttentionSupport` records query-head identity, KV-head identity, source
  position/role support, sink mass where applicable, and the head's reader
  contribution. It is admitted only through the HEAD-OBS-1 reconstruction law.
- `FfnSupport` records exact feature addresses, contribution mass and any
  separately established write-family identity. A write-family label never
  replaces its underlying observed addresses.

MoE, recurrent, state-space, MLA and conv-QKV support require their own typed
variants in a later protocol version. They are not coerced into attention or FFN
coordinates. Unsupported operators produce coverage/refusal records, not guessed
events.

`causal_status` is intentionally absent. GW-TS-1 is observational. Intervention
evidence joins the same event coordinate in ATTR-1C or a later causal-path rung;
it does not rewrite the observational record.

## Identity joins and evidence authority

Every admitted path must bind one exact model/container identity, operation-plan
identity, prepared-image fingerprint, realization policy, prompt-token identity,
execution identity, position, observation receipt and extraction-contract
identity. Attention and FFN records must join on that tuple and on exact ordered
site identity. A mismatch refuses the path rather than falling back to model,
prompt or layer names.

HEAD-OBS-1 is authoritative for attention children and the head-sum/source-sum
laws. Production-effective GW-0B-style attribution is authoritative for FFN
children. The site's canonical carrier write remains the parent write. Child
support must reconstruct that parent within its frozen operator tolerance before
the event is eligible.

The population manifest is a second, immutable artifact. Before any assessment
execution it must bind:

1. semantic transition rows and their `TransitionIdentity` values;
2. prompt families, prompts and tokenizations;
3. discovery and held-out assignments grouped by `TransitionIdentity`;
4. model/container, realization and operation-plan identities;
5. eligible sites and capture windows;
6. the preregistered reader used for contribution scoring;
7. observation level, source `top_k`, byte budget and refusal policy.

This protocol freezes how those fields are used. It does not invent their values
before EXPERIMENT-1 can represent Run subjects. Changing a bound population field
after the first assessment execution creates a new experiment identity.

## Deterministic path extraction

No event is manually added, removed, merged or relabelled after recurrence is
inspected.

For each admitted instance:

1. Read eligible attention and FFN parent writes in canonical execution order.
2. Score each parent with the preregistered target-directed reader. The signed
   score is `dot(reader, applied_delta)`; its absolute value is the selection
   mass.
3. Normalize site admission separately by operator and normalized-depth quartile.
   Discovery rows establish the empirical absolute-contribution distribution for
   each `(operator, depth_quartile)` cell. The primary event threshold is that
   cell's `0.75` quantile. A cell with fewer than 20 discovery observations uses
   the operator-wide discovery distribution; an operator with fewer than 20
   observations is unsupported and produces a refusal. These distributions are
   locked before assessment.
4. Admit a site exactly when its absolute score is at least its locked threshold.
   Apply this decision independently when the site is observed, then retain
   admitted sites in canonical execution order. Later writes, total path mass,
   recurrence and outcomes cannot change whether an earlier site was admitted.
5. At an admitted attention site, select the smallest
   descending-by-absolute-contribution
   set of heads reaching `0.80` of reconstructed head contribution mass. At an
   FFN site, do the same for exact feature contributions, with a maximum of 32
   addresses. Ties use ascending physical address. Zero-mass sites are retained
   only when needed for the parent-write accounting and have empty physical
   support; they cannot count as a recovered next event.
6. Store measurements, the locked threshold identity, selected and total child
   mass, truncation, reconstruction and refusal state. Hash the ordered result.

The `0.75` event quantile, `0.80` child mass and 32-address cap are primary
operating points. Frozen sensitivity points (event quantiles `0.50`, `0.65`,
`0.90`; child-mass thresholds `0.50`, `0.65`, `0.90`, `0.95`; FFN caps `16`,
`64`) are reported as robustness checks and cannot replace the primary verdict.

The rule consumes the locked discovery calibration and the current observation
record only. It cannot use later events, whole-path totals, cross-run recurrence,
the assessment label, intervention outcome or knowledge of which path a later
predictor retrieves.

## Recurrence without manufactured alignment

GW-TS-1 reports two recurrence surfaces.

**Exact-order recurrence** requires equal operator sequence and, within one model
and realization family, equal exact site sequence. Exact physical-support overlap
is weighted Jaccard over head/source or feature-address mass, with contribution
sign agreement reported separately. Across architectures, exact head, feature and
layer identity is undefined and is not scored.

**Aligned recurrence** uses a frozen global sequence alignment. A gap costs `1`.
Matching equal operator kinds costs absolute normalized-layer-depth distance.
An operator mismatch costs `2` and the deterministic tie-break prefers two gaps,
so Attention and FFN events are never made equivalent by alignment. Alignment
uses neither physical-support similarity nor outcomes. After the alignment is
fixed, report operator agreement, layer-depth distance, physical-support
similarity, contribution-sign agreement and gap rate separately. There is no
single tunable path-similarity scalar.

Within a model family, physical support means exact typed IDs. Cross-architecture
comparison is secondary and uses a declared typed equivalence signature only:

```text
AttentionSupportSignature
    semantic/source role
    normalized layer depth
    source-position role
    contribution sign and quantile

FfnSupportSignature
    normalized layer depth
    independently established support-family identity
    contribution sign and quantile
```

The signature never asserts that head H3 in one model is head H11 in another.
Architecture-specific realization remains a valid negative or stratified result.

## Discovery and held-out discipline

Splits are grouped by `TransitionIdentity`; prompt copies never cross the
discovery/assessment boundary as independent facts. Discovery executions may fit
path prototypes, retrieval indices and any permitted alignment-independent
parameters. Assessment executions may only be scored. No prototype, threshold,
equivalence mapping, candidate vocabulary or normalization statistic is updated
from an assessment outcome.

Reproducibility is measured within held-out transitions using at least two
different execution identities. The decisive generalization contrast is:

```text
same TransitionIdentity
different prompt family or prompt identity
different execution identity
```

Controls keep task and relation strata matched but use a different
`TransitionIdentity`. Repeated executions of the identical prompt are reported
separately; they test runtime stability, not paraphrase invariance.

## Frozen comparisons

Every primary result includes these matched controls:

1. ordered path versus the same support events treated as an unordered set;
2. true event order versus within-path permutation;
3. operator labels intact versus Attention/FFN labels swapped;
4. full multi-site path versus its strongest single event;
5. prefix-conditioned next event versus marginal-frequency and layer-frequency
   predictors;
6. same `TransitionIdentity` with a different execution identity;
7. same relation/task stratum with a different `TransitionIdentity`.

Controls use the same eligible universe and are matched on returned event count,
candidate bytes and support-unit budget. An impossible operator swap is an
explicit refusal, not silently omitted or dimension-padded.

The future causal comparison—observed support versus intervention-confirmed
support—is not a GW-TS-1 control. It belongs after ATTR-1C exists.

## Experimental ladder

### TS-0 — census and coverage

Build paths mechanically from the frozen population record. Report admitted,
refused, reconstruction-failed and mass-truncated instances before any recurrence
or prediction statistic. A claim cannot be made on less than `0.80` held-out
instance coverage.

### TS-1 — held-out recurrence

Compare cross-execution and cross-prompt-family paths for the same held-out
transition against matched different-transition controls. Report exact and
aligned recurrence separately, including every component and denominator.

### TS-2 — prefix-to-next-support prediction

For every held-out path and every prefix containing at least two events, expose
only the prefix, semantic task/relation context and structural eligibility mask.
The candidate predictor may retrieve discovery prefixes, but never another
assessment realization of the same `TransitionIdentity`. Score its proposed next
event at candidate-count points `{1, 2, 4, 8, 16}`.

The primary score is true next-event absolute contribution-mass retention. A
candidate receives zero unless it predicts the correct operator and an eligible
site; exact site is required within one architecture, while a separately reported
cross-architecture score uses the frozen typed signature. Also report operator
accuracy, event survival, support weighted-Jaccard, sign accuracy, candidate
fraction, bytes touched and dot products.

The primary prediction claim passes only if all of the following hold:

- held-out path coverage is at least `0.80`;
- at some frozen candidate-count point with mean candidate fraction at most
  `0.25`, the block-bootstrap fifth percentile of contribution-mass-retention
  advantage is greater than zero against every primary control;
- the same point improves operator accuracy over marginal-frequency and
  layer-frequency controls, with block-bootstrap fifth-percentile advantage
  greater than zero;
- the advantage occurs after at least one Attention→FFN or FFN→Attention
  transition, not solely inside one operator class;
- no frozen control is dropped. An inapplicable operator-swap arm makes the
  operator-swap comparison `not_testable`; it cannot be counted as a pass.

Bootstrap blocks are `TransitionIdentity`, not prompt or prefix rows. The random
seed and trial count are in the machine contract. All candidate-count points are
reported; the first point satisfying the conjunction is selected mechanically.

GW-TS-1 ends here. Passing TS-2 establishes predictive addressability, not causal
support and not permission to skip computation.

## Explicit verdicts

The adjudicator must choose the strongest supported statement without promoting a
failed primary claim:

- **No recurrence:** no stable support path for the transition identity.
- **Recurrence without order:** the support set is stable but its sequence is not.
- **Order without prediction:** paths recur retrospectively but prefixes do not
  forecast continuation beyond controls.
- **Predictive observational path:** TS-2 passes; causality remains untested.
- **Prompt-family-specific realization:** semantic recurrence exists but physical
  paths differ across prompt families.
- **Architecture-specific realization:** semantic recurrence exists but the typed
  physical realization differs across architectures.
- **Insufficient coverage:** observation/refusal or reconstruction coverage misses
  the frozen floor; no structural verdict is issued.

“Prediction without causality” is the expected claim boundary of a successful
GW-TS-1, not a defect to be edited away.

## Successor rungs

ATTR-1D supplies descriptive head/source attribution. ATTR-1C combines the same
coordinates with interventions and may establish necessity, sufficiency,
redundancy or cancellation. A later causal-path protocol can then compare a
coherent path with the same marginal events in the wrong order, at wrong sites or
with one operator class removed.

Executability is a separate promotion rung:

```text
predicted support
    -> skip / route / prefetch / substitute work
    -> correctness preserved?
    -> calibration acceptable?
    -> measured work saved?
```

No GW-TS-1 result authorizes those actions. Python WalkModel, reverse sensitivity,
feature providers, state routing and Metal instrumentation are also outside this
freeze.

## Freeze and amendment rule

The ontology, extraction rule, alignment, controls, split discipline, operating
points, gate and claim boundary above are frozen before population capture. The
machine contract hashes the same decisions. A correction after assessment begins
is recorded as an amendment with its reason and timing; the original contract is
not silently rewritten. Results and deviations live in a separate adjudication
artifact.
