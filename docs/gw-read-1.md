# GW-READ-1 — pricing the frozen causal read interface

**Status:** amended and frozen before candidate fitting or execution, 2026-09-21

**Predecessor:** GW-KEY-1 compact candidate read path through zero-based L24H1

**Machine contract:**
[`bench/gw-read-1/gwread1-protocol.json`](../bench/gw-read-1/gwread1-protocol.json)

**Claim boundary:** whether Q, K and V for the frozen GW-KEY-1 path can be
constructed economically; not a new semantic search, production correctness,
or authorization to replace canonical execution.

GW-KEY-1 isolated this causal interface:

```text
natural final-position Q
        +
structured reference K
        +
natural subject_entity V
        ↓
zero-based L24H1
        ↓
canonical downstream model
```

It established causal compactness, not computational cheapness. In particular,
the GW-KEY-1 reference K and complement V directions were scaled to each target
execution's natural source-vector norms. Those norms are dynamic natural-state
inputs and are charged as such until an explicitly static replacement removes
them.

The frozen rule is:

> **GW-KEY-1 found a compact causal interface. GW-READ-1 prices that interface
> honestly.**

## Four rungs

```text
READ-1Q: candidate Q + exact GW-KEY K/V
READ-1K: exact natural Q + candidate K + exact GW-KEY V
READ-1V: exact natural Q + fixed template/role K + candidate V
                              │
                              ↓ train-only selection seal
READ-1E: frozen cheap Q + cached K + cheap V, composed without tuning
```

Q, K and V are separate experiments. Each candidate is fit and selected using
train only. Validation and test see only the already-frozen primary candidate
and Pareto set. Held-out outcomes cannot change rank, source layer, cache key,
scale, threshold, or candidate identity.

READ-1E is composition, not selection. It uses the unchanged frozen primary
from each component arm. A fixed 2×2×2 exact/candidate factorial diagnoses
interaction, but only the all-candidate cell enters the READ-1E verdict.

## Authorities and population

The predecessor authorities are GW-KEY-1 preregistration
`sha256:87e80308a9cc684f875a0b16684f3cf2965e3e22b34a50338273c4e1935732d8`,
selection `sha256:eed14eb42f5432960677d47bd328a6b3b74777f542011f7b77b58469cfe69f6f`,
and adjudication
`sha256:a554877b15b69467a6c4f5ecca9427669a1536ff5b9e88056000cf2bb78dd60a`.

The inherited population has 426 rows: 255 train, 87 validation and 84 test.
The unit is the semantic edge, clustered across its three prompt families and
stratified by relation. Capital, currency and language form the factual primary
scope. Hypernym is a transfer diagnostic and cannot rescue or sink that verdict.

Candidate tensor input capture may include every split, provided it is sealed
before fitting and contains no candidate effect, semantic target, or answer
outcome. Candidate fitting, Pareto construction and primary selection use train
only. A selection seal containing component IDs and fitted-artifact hashes is
required before any held-out candidate replay.

## READ-1Q — query construction

The exact control is natural final-position L24H1 Q. Candidate families are
frozen before capture:

- train template×relation, relation and global mean Q;
- native H1 Q at layers 0, 4, 8, 12, 16, 20 and 23;
- centred ridge maps from each listed earlier-layer Q to L24 Q, with ranks 8,
  16, 32, 64 and 128.

Ridge uses train only, f64 fitting, `lambda = 1e-3 × trace(XᵀX)/d`, and a
deterministic sign-canonicalized SVD truncation. Train semantic-effect estimates
for fitted predictors use leave-one-semantic-edge-out predictions. The final
held-out artifact is then fit once on all train rows.

Every Q candidate is replayed with the exact GW-KEY-1 K/V treatment. That exact
treatment remains a positive causal control; all natural K/V rows and target
norms it consumes are charged as dynamic inputs and therefore cannot make Q
alone an end-to-end economical path.

## READ-1K — routing scaffold

The positive control is the GW-KEY-1 template/role reference direction scaled
to each target execution's natural K norm. It is explicitly dynamic.

Candidate K scaffolds contain their complete scale and require no target-row K
or norm:

- template×source-role×within-role ordinal mean;
- prompt-family×source-role×ordinal mean;
- relation×source-role×ordinal mean;
- global source-role×ordinal mean.

Each cell stores the train mean direction and train cell-mean norm. Missing
cells refuse; validation/test donors, fallback, interpolation and target-natural
rescaling are forbidden. The hierarchy directly tests how far routing can be
cached: template, prompt family, relation, then global.

## READ-1V — payload construction

The positive control is natural L24H1 V at the subject/entity source positions,
with the fixed template/role K scaffold. Every candidate carries one of three
cost labels:

- **diagnostic** — mechanistically informative but not eligible for READ-1E;
- **cached/materialized** — eligible only with complete build, footprint,
  refresh and amortization accounting;
- **per-query computed** — eligible only with all executed layers and operands
  charged.

Frozen candidate families are:

- same-fact cross-prompt transplant (**diagnostic**);
- cached entity mean and cached entity×relation mean (**cached/materialized**);
- entity-only-context L24 V, both per-query and cached forms;
- native subject/entity V at layers 0, 4, 8, 12, 16, 20 and 23
  (**per-query computed**);
- centred reduced-rank predictors from those earlier-layer V rows to L24 V at
  ranks 8, 16, 32, 64 and 128 (**per-query computed**).

The frozen non-entity V complement uses static train template/role cells,
including static cell norms. No candidate may inherit target-natural complement
norms silently.

Each cache records construction source, construction compute, refresh policy,
entry key, entry count, bytes per entry, total footprint, lookup bytes, lookup
latency and amortization denominator. A cache built from a held-out execution is
never treated as a deployable candidate.

## Train-only selection

For every metric, signs are normalized so larger means more alignment. The
inherited effect is

```text
E(I) = Δsame-fact(I) − Δmatched-control(I)
R(I) = E(I) / E(exact)
```

with raw and z-scored candidate JS reported separately. A non-positive exact
denominator refuses that stratum.

Within each component arm, train builds the effect-retention/cost Pareto
frontier by direct replay. A candidate is train-eligible when both factual raw
and z-scored retention are at least 0.80. The primary is then selected by:

1. least dynamic matvec-equivalent FLOPs;
2. least dynamic bytes touched;
3. highest minimum raw/z retention;
4. lexicographically smallest candidate ID.

No singleton score is summed into a composite estimate. The selected ID,
fitted artifact and full Pareto frontier are sealed before held-out replay.

## Held-out component gates

The frozen READ-1Q/K/V primary must pass independently on validation and test:

- factual raw and z-scored retention estimates are at least 0.80;
- their semantic-edge cluster-bootstrap lower 95% bounds are at least 0.50;
- terminal transport is positive with a lower 95% bound above zero;
- coverage is complete and all identity/exact controls pass;
- no undeclared natural state, held-out donor, fallback, imputation or
  post-hoc scaling is consumed.

Component semantic-effect gates establish fidelity. Cost is reported as a
curve and applied conjunctively to READ-1E, where avoided work can actually be
measured. Semantic-target recovery is reported separately and cannot rescue a
failed causal gate.

## READ-1E — end-to-end composition

READ-1E combines only the independently sealed Q, K and V primaries. It runs
the real H1 softmax/value aggregation, effective prepared W_O, declared
post-attention transformation, residual write, L24 FFN and every later layer.
There is no joint fit, rescaling, recalibration, candidate swap or combination
search.

The factual validation and test gates are conjunctive:

1. raw and z-scored retention satisfy the component thresholds;
2. terminal transport has a positive lower 95% bound;
3. all inputs are declared, with no hidden held-out full-execution state;
4. total dynamic bytes and matvec-equivalent FLOPs through the replaced
   pre-L24 boundary are each at most 0.25 of canonical L0–23 execution;
5. warmed p50 and p95 end-to-end latency are at most 0.50 and 0.67 of the
   canonical baseline;
6. cache construction, footprint, refresh and amortization are complete.

Passing earns `CREDIBLE MODEL-NATIVE READ PRIMITIVE`, not operational
replacement authorization.

## Physical cost and the shared frontier

Every row reports offline build time and bytes, dynamic construction time,
lookup time, bytes touched, resident and incremental resident bytes, matvecs,
projections, dot products, matvec-equivalent FLOPs, routing/injection overhead,
downstream continuation, and end-to-end latency. Cache costs are reported cold
and amortized at 1, 1,000 and 1,000,000 queries.

Avoided work is the primary accounting view. Component costs are reported, but
READ-1E also constructs the minimum realizable shared schedule:

- original-prompt Q and V prefixes share L0 through the maximum required layer;
- entity-only and original-prompt prefixes do not share execution and are
  summed;
- cached components contribute lookup work plus their separately reported
  build/amortization cost;
- the common H1 and downstream continuation are reported separately and are
  not mislabelled as avoided pre-L24 work.

Thus Q from L12 plus V from L18 costs one original-prompt L0–18 prefix, not two
prefixes. If either needs L23, the honest frontier is L23 even when the final
Q/V tensors are tiny.

Latency uses warmed baseline/candidate/baseline brackets. A block is void when
the two baselines differ by more than 1%. Every peer session must handshake
exclusivity, and measurement precision must be finer than the gate.

## Claim boundary and fixed negative outcomes

Possible outcomes include `q_not_economical`, `k_not_cacheable`,
`v_not_economical`, `components_do_not_compose`,
`quality_retained_without_cost_advantage`,
`cost_advantage_without_causal_retention`, `factual_only_access_method`,
`insufficient_coverage`, and `measurement_invalid`.

Even a pass does not establish production correctness, calibration, scheduler
savings under real traffic, safe layer skipping, architecture generality, or
that the database analogy is universal. Those require a separately frozen
successor using the path as an actual execution route.
