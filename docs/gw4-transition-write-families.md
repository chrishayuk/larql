# GW-4 — transition/write-family retrieval

**Status:** GW-4A complete; GW-4B did not open GW-4C, 2026-09-21
**Input:** the 231 reconstructed FFN transition-candidate sites from GW-0B
**Claim boundary:** observational write-family structure, not causal support

GW-3A-F established that gate geometry is a real semantic surface but a poor
physical address space for observed FFN support. GW-4 changes the retrieval
object. It treats the recorded state transition as primary and individual FFN
addresses as operator-specific support beneath it.

```text
TransitionIdentity
    source semantic state
    relation / operation
    destination semantic state
    layer / site context

TransitionSupportObservation
    operator_kind
    carrier_before_ref
    carrier_delta_ref
    carrier_after_ref
    physical_support_ref
    causal_status
```

The first rung does not add this as a stable container ABI. It establishes
whether the object has reusable structure before VINDEX3 promotes it beyond an
experimental evidence schema.

## GW-4A — write-family census

GW-4A asks three questions in order:

1. Do recorded FFN delta directions repeat across prompt variants of the same
   semantic fact?
2. Do same-relation deltas form reproducible neighbourhoods after controlling
   for layer?
3. Does a low-dimensional relation-specific subspace generalise to held-out
   edge families better than wrong-family and random subspaces?

The machine contract is
[`bench/gw0/gemma3-4b-it-phase1/gw4a-preregistration.json`](../bench/gw0/gemma3-4b-it-phase1/gw4a-preregistration.json).
It binds the sealed GW-0 census, production-effective GW-0B attributions, split
assignments, vector extraction rule, controls, thresholds, random seed, and
exact operating points.

### Vector authority

For each reconstructed `(edge_id, layer, ffn)` site, GW-4A reads the exact
`f32-le` slice from the sealed GW-0 `delta` artifact whose execution-record
entry has the same layer and site. The vector must contain 2,560 finite values,
its artifact hash and execution-record hash must verify, and its L2 norm must
match the recorded write norm within the frozen tolerance. Similarity uses the
unit-normalised delta; energy remains a separate reported variable.

No top-feature approximation, promoted-token label, gate score, or GW-3A-F
outcome enters the vector.

### Frozen comparisons

**Prompt repeatability.** Compare different prompt families for the same
`(subject, relation, target)` triple. The primary subset requires the same
physical layer. Its control is a different triple with the same relation and
layer. The all-layer same-fact distribution is descriptive.

**Relation neighbourhood.** Across different semantic triples at the same
layer, classify a pair as same relation or different relation and use delta
cosine as its score. Report ROC AUC and a 1,000-trial permutation null that
shuffles relation labels among `(layer, semantic-triple)` blocks within each
layer. This preserves layer counts and keeps prompt copies together inside an
exact-layer block.

**Held-out family retrieval.** Training rows form the index. Validation and
test rows are a locked assessment set because no model or threshold is fitted
after this preregistration. For each assessment write, rank relation families
by the maximum cosine to a training write at the same layer, excluding its own
semantic triple. Report top-1 through top-4 retention, eligible-query coverage,
balanced accuracy, and a layer-only majority-relation control.

**Shared subspace.** For each relation, fit SVD bases to unit delta directions
from training edge families only at ranks `{1,2,4,8,16}`. Score an assessment
write by squared projection mass. Report correct-family projection, strongest
wrong-family projection, relation retention, layer-prior control, and a
matched-rank random orthonormal-subspace control.

**Matched-energy random writes.** Generate Gaussian directions with the frozen
seed and scale them to observed write norms. Report their direction-cosine
distribution separately; rescaling cannot affect the cosine statistic.

The primary cohort contains all 231 reconstructed FFN sites. Prompt copies are
execution observations, while uncertainty intervals and permutations use
semantic-triple blocks so copies do not become independent facts.

### Frozen gates

The gates select a branch rather than manufacture a general claim:

| Signal | Gate |
|---|---|
| Prompt-stable fact direction | exact-layer same-fact median cosine `>= 0.50` and at least `0.10` above same-relation/different-fact control |
| Relation direction | exact-layer pairwise AUC `>= 0.75` and permutation `p <= 0.01` |
| Neighbour retrieval | held-out top-1 balanced accuracy `>= 0.75`, at least `0.15` above layer prior, with `>= 0.80` query coverage |
| Shared subspace | some frozen rank reaches balanced accuracy `>= 0.75`, improves on layer prior by `>= 0.15`, and has mean correct-minus-wrong projection margin `>= 0.05` |

Passing the first gate supports fact-specific write families. Passing both the
relation-direction and neighbour gates supports direct relation-conditioned
delta indexing. Passing the subspace gate supports a distributed family even
when individual directions vary. GW-4B opens if any of those three family
claims passes. If none passes, direct delta-family indexing is retired and the
next object is a multi-site trajectory or semantic basin.

That final alternative records the frozen preregistration wording. Subsequent
GW-SUP-1 and prior discrete-basin evidence rule out treating generic basin
discovery as the untested successor; the superseding decision is below.

No result here is causal. `causal_status` remains inherited from the sealed
census and all current rows are `untested`.

## GW-4A result

The frozen Gemma 3 4B run passes all three family claims. Its report is
`output/gw4a-gemma3-4b-it-phase1/report.json`,
with SHA-256
`bca0ce2e473879b1899fdcf584fe8fe2e51ded9ec99ba4e71a4881588015d9fb`.
The preregistration identity is
`sha256:c0fa06c358b486919842faa931fc442555bda9aeca0aaab2dc3eb6b7bc46d4e0`;
its file SHA-256 is
`f40054cc0736471da72db5f1ee455cc87772d112064a1186e2880c638b3ad3ff`.

| Test | Frozen gate | Result |
|---|---:|---:|
| Exact-layer, same-fact prompt median cosine | `>= 0.50` | `0.8330` |
| Median gap over same-relation/different-fact control | `>= 0.10` | `0.1757` |
| Exact-layer relation-neighbourhood AUC | `>= 0.75` | `0.9065` |
| Stratified permutation probability | `<= 0.01` | `0.000999` |
| Held-out top-1 balanced relation accuracy | `>= 0.75` | `0.7777` |
| Improvement over exact-layer prior | `>= 0.15` | `0.2197` |
| Held-out query coverage | `>= 0.80` | `0.9091` |

The prompt comparison contains 95 exact-layer prompt pairs across 55 semantic
facts. Same-fact directions have median cosine `0.8330`; the layer- and
relation-matched different-fact control has median `0.6573`. The block
bootstrap's fifth to ninety-fifth percentile interval for the fact-minus-control
gap is `[0.3077, 0.4319]`; this interval bootstraps fact-block medians separately
and is therefore not numerically identical to the raw-pair median gap used by
the frozen gate.

Across different facts at the same layer, 1,526 same-relation pairs and 718
different-relation pairs give AUC `0.9065`. None of the 1,000 permutations,
which preserve exact-layer and prompt-copy block structure, reaches the observed
AUC. The null median is `0.6670` and its maximum is `0.7738`.

Nearest-family retrieval uses 143 training writes and 88 validation/test writes.
It reaches `0.7777` balanced top-1 accuracy with eight explicit refusals counted
as misses, versus `0.5579` for the exact-layer majority-relation prior. Raw
top-1 retention is `70/88 = 0.7955`. This establishes family discrimination,
not a cheap access path: the query still compares against eligible training
writes, so candidate bytes and compute remain questions for GW-4B.

The shared-subspace arm also passes. Frozen ranks 2, 4, 8 and 16 qualify. Rank 2
already reaches `0.8344` balanced accuracy and a mean correct-minus-strongest-
wrong projection margin of `0.2347`; rank 16 reaches `0.8643` and `0.4218`.
Matched-rank random bases remain between `0.1618` and `0.3185` balanced accuracy
across the frozen ladder.

The post-run integrity audit found no semantic triple or edge family crossing
the train/assessment boundary and no duplicated delta artifact hashes. Every
one of the 231 rows retains `causal_status = untested`. The result is therefore
evidence for reusable observational FFN write families on this model and corpus.
It does not establish causal support, cross-model generality, or pre-write
addressability.

The preregistered branch rule opens GW-4B. GW-4B must use only information
available before the write and must keep relation/context, carrier-before, and
layer/site contributions separately ablated. Its required output is a retention
curve against candidate families, bytes touched, and compute; GW-4A's exhaustive
nearest-family classifier is an oracle-quality reference rather than the
proposed index.

## GW-4B — pre-write exemplar retrieval

The machine contract is
[`bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json`](../bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json),
with frozen identity
`sha256:54f975a64a167a9afc18d8dab95774ad587aa35a75e26b5351635bb11baf9e53`.
It was frozen after the GW-4A result and before any carrier-before key
similarity was computed.

The query may use only the carrier immediately before the selected FFN write,
the supplied semantic relation/query class, and the exact layer/site. The
candidate universe is the set of training writes at that layer and site with
the same relation and a different semantic triple. Delta and carrier-after
vectors, target identities, gates and promotions are forbidden during key
construction and ranking.

The evaluation oracle orders eligible exemplars by cosine to the held-out
observed delta. This oracle is used only after retrieval. The primary measure is
the best delta cosine among the returned candidates, accompanied by its gap to
the oracle best. Exact top-1 exemplar retention and top-4 exemplar recall remain
identity diagnostics; GW-3A-F already showed why exact address identity cannot
substitute for recovered transition quality.

The frozen key arms are raw carrier cosine, relation-mean-centred full carrier,
and training-only relation-centred PCA at widths `{2,4,8,16,32}`. A state-only
arm removes the relation mask. A matched random control samples the same number
of candidates from the same structural universe. Candidate counts are
`{1,2,4,8,16,32}`; a count saturates at the eligible universe size rather than
silently changing the denominator.

An operating point progresses only if it satisfies all five conditions:

| Measure | Gate |
|---|---:|
| Query coverage | `>= 0.80` |
| Mean candidate fraction of eligible universe | `<= 0.25` |
| Median best retrieved delta cosine | `>= 0.80` |
| Median gap to oracle-best delta cosine | `<= 0.05` |
| Fifth percentile block-bootstrap advantage over matched random | `>= 0.10` |

The report keeps key dot products, key bytes touched, returned delta bytes,
index bytes and projection-basis bytes separate. Wall time is descriptive. A
pass opens GW-4C; failure retires direct carrier-before exemplar indexing and
moves the representation to multi-site trajectories or semantic basins.

That branch phrase is historical provenance. It is not a current claim that
basin conditioning remains untested.

## GW-4B result

The final report is
`output/gw4b-gemma3-4b-it-phase1/report-final.json`,
with SHA-256
`1c334c983a1cec43af91e01d816cc39aa61a3f7e48924a1c0a29e676b126477b`.
The preregistration file SHA-256 is
`3817e84c645cd73d0091c061d244818ac7ef47eed6211d3ad5873579d7da2bfa`.
All 231 carrier-before slices bind to their sealed artifacts and preceding
same-layer attention outputs; the maximum relative norm error is
`4.13e-15`.

No progressive operating point passes all five frozen conditions, so the
formal decision keeps GW-4C closed. Only `K=1` satisfies the candidate-fraction
limit: it covers `72/88 = 81.82%` of assessment rows and examines a mean
`18.96%` of each exact-layer, same-relation exemplar universe. All larger
candidate counts exceed the `25%` coverage ceiling.

The `K=1` transition-quality signal is substantial but misses the uncertainty
gate:

| Key | Width | Median best delta cosine | Median oracle gap | Bootstrap advantage p05 | Gate |
|---|---:|---:|---:|---:|---|
| Raw carrier | 2,560 | `0.8864` | `0.0000` | `0.0829` | fail |
| Relation-centred full | 2,560 | `0.8864` | `0.0000` | `0.0878` | fail |
| Relation-centred PCA | 2 | `0.8592` | `0.0000` | `0.0812` | fail |
| Relation-centred PCA | 16 | `0.8913` | `0.0000` | `0.0865` | fail |
| Relation-centred PCA | 32 | `0.8753` | `0.0000` | `0.0919` | fail |

Every row in the table clears the `0.80` absolute-quality and `0.05` oracle-gap
gates. The best lower confidence bound is `0.0919`, below the frozen `0.10`
advantage requirement. Raw carrier retains the exact oracle-best exemplar on
`51/88 = 57.95%` of the full assessment denominator. The median expected
matched-random delta cosine within the already relation- and layer-matched pool
is approximately `0.58`, while the carrier-ranked result is approximately
`0.89`.

The state-only control is scientifically important but does not change the
frozen decision. At `K=1`, removing the relation mask gives median best delta
cosine `0.8864`, mean exact-layer candidate fraction `0.1207`, and bootstrap
advantage p05 `0.1784`. The evaluator designated this arm as an ablation and
excluded it from progression before execution. Its result suggests that the
carrier itself contains much of the relation/family key; it requires a fresh
confirmatory cohort rather than post-hoc promotion on these 88 rows.

The failure is therefore narrower than GW-3A-F. Carrier-before similarity is
strongly enriched and low-rank keys preserve much of it, but the preregistered
relation-conditioned improvement is not robust enough to open propagation.
Under the frozen branch rule, direct single-site exemplar retrieval stops here.
The next primary object is ordered, operator-aware multi-site transition
support. A separate replication may preregister the state-only key on new edge
families without reopening this result.

## Deferred rungs

GW-4C will propagate the frozen top-K families and measure candidate entropy,
true-family survival, distance to the observed trajectory, and convergence into
the same semantic basin. It remains closed because GW-4B did not pass its frozen
progression gate.

Generic basin discovery is not the fallback. The completed
[GW-SUP-1 candidate-basin experiment](gw-sup-1.md) failed its full conjunction,
and earlier nearest-basin work did not provide a useful discrete lookup. The
successor is [GW-TS-1](gw-transition-support-paths.md): an ordered sequence of
operator-typed attention and FFN support, gated on sealed HEAD-OBS-1 evidence.

## Commands

```bash
python3 scripts/gw4a_preregister.py \
  bench/gw0/gemma3-4b-it-phase1/gw4a-preregistration.json

python3 scripts/gw4a_write_families.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gw4a-preregistration.json \
  --output output/gw4a-gemma3-4b-it-phase1/report.json

python3 -m unittest \
  scripts.test_gw4a_preregister \
  scripts.test_gw4a_write_families

python3 scripts/gw4b_preregister.py \
  bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json

python3 scripts/gw4b_prewrite_retrieval.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json \
  --output output/gw4b-gemma3-4b-it-phase1/report-final.json

python3 -m unittest \
  scripts.test_gw4b_preregister \
  scripts.test_gw4b_prewrite_retrieval
```
