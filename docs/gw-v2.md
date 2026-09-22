# GW-V2 — causal payload formation surface

**Status:** frozen before candidate capture, fitting, replay, or effect observation,
2026-09-21

**Predecessor:** GW-READ-1 localized the unresolved cost to contextual payload
and execution-state materialization after cached Q and K preserved the factual
alignment effect.

**Machine contract:**
[`bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json`](../bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json)

**Fresh population:**
[`bench/gw-v2/gemma3-4b-it-phase1/population-manifest.json`](../bench/gw-v2/gemma3-4b-it-phase1/population-manifest.json)

GW-V2 asks one measurement question:

> **At what source depth and predictor rank does the L24H1 subject/entity V
> payload become causally usable?**

It does not search for or select a better V predictor. Every cell in the frozen
depth-by-rank grid is a first-class result.

## Frozen isolation

The only varied object is the subject/entity V delivered through zero-based
L24H1:

```text
candidate V(source depth, rank)
        + natural final-position L24H1 Q
        + frozen GW-KEY-1 reference K
        + frozen reference V outside subject_entity
        + natural L24 carrier-before
        + natural contributions from the seven non-H1 heads
        ↓
real L24H1 scoring / aggregation / W_O
        ↓
real residual → norm → FFN → later layers
        ↓
inherited control-adjusted semantic-alignment effect
```

The K treatment is the frozen GW-KEY-1 joint arm: every source K direction is
the train-derived `template_id × source_role` reference and is scaled to that
target execution's natural K-row norm. The non-subject V complement uses the
same frozen reference-bank rule. Natural Q, K norms, carrier-before, and the
other seven heads are deliberately retained to isolate payload formation.
They are declared dynamic natural-state inputs, so GW-V2 cannot establish an
economical end-to-end read path.

The exact positive control differs from a grid cell only in using the natural
L24H1 subject/entity V. Zero V, an identity/no-op intervention, and the exact
arm exercise the same intervention machinery.

## Fresh population

The population was frozen without running model prompts. It contains 74
subject-disjoint countries, 222 semantic edges, and 666 executions. Each
country contributes capital, currency, and language edges under canonical,
question, and alternate prompt families.

The source is `mledoze/countries` at commit
`c8015eebdd94c533358406b0d709f441389e1f2e`. The frozen predicate admits every
independent UN member with exactly one capital, one currency, and one official
language after excluding every ISO-3166 country identity used by any factual
GW-0 row. There are no manual inclusions, exclusions, or target adjudications.

Splitting is by country identity, not semantic edge: all relations and prompt
families for a country remain together. Deterministic SHA-256 ordering assigns
44 subjects to train, 15 to validation, and 15 to test. Thus validation and
test are subject-disjoint from both V2 training and the entire earlier GW-0
factual population.

The candidate-readout inventory is the sorted unique set of first continuation
tokens from the frozen V2 target annotations. This changes the population-bound
vocabulary, not the inherited metric: restricted-softmax JS and independently
row-z-scored restricted-softmax JS are computed exactly as before.

## Frozen measurement grid

Source depths are:

```text
L0, L4, L8, L12, L16, L20, L23
```

Predictor ranks are:

```text
8, 16, 32, 64, 128
```

The Cartesian product contains 35 cells. It is immutable. Partial results may
not add a depth, add a rank, remove a cell, change a fit, or promote a cell into
a selected primary.

At source depth `d`, the input is that layer's natural H1 subject/entity V at
each frozen subject-token position. The target is natural L24H1 V at the same
position. Every position is one equally weighted fit row. A centred f64 ridge
map is fit on train only:

```text
lambda = 1e-3 × trace(XᵀX) / input_dimension
```

The fitted map receives a deterministic sign-canonicalized SVD. A grid rank is
the corresponding truncated map. Inference is f32. The intercept, centring,
regularization, fit rows, training split, numeric precision, and cached-state
layout are identical across cells except for the declared source depth and
rank. No semantic targets, candidate effects, validation/test tensors, or
post-hoc scaling enter fitting.

Train results, if reported, use leave-one-subject-out predictions. The sealed
held-out artifact is one fit on all 44 train subjects per source depth, with all
five rank truncations derived from that fit. Train results cannot tune the grid
or alter interpretation rules.

## Inherited causal estimand

For each semantic edge, the three prompt families form the same-fact group.
The frozen different-fact control supplies a different subject at the same
split, relation, and prompt family. Metric signs are normalized so larger means
more alignment:

```text
E(I) = (JSbefore − JSafter)same-fact
     − (JSbefore − JSafter)matched-control

R(depth, rank) = E(candidate depth, rank) / E(exact natural V)
```

Raw and z-scored candidate distributions are separate estimands. A split and
surface refuse retention if the exact-arm point denominator is non-positive;
the confirmatory cell gate additionally requires the exact-arm lower confidence
bound to be positive. Carrier cosine and tensor reconstruction never rescue a
failed semantic-effect result.

Terminal transport repeats the adjusted effect at terminal candidate logits.
It must remain positive; intermediate retention alone is insufficient.

## Analysis plan

All 35 cells are reported independently on validation and test. There is no
winner, Pareto frontier, primary cell, or held-out choice.

Uncertainty uses 10,000 deterministic bootstrap replicates, resampling country
identities with replacement within each split and carrying all nine repeated
executions for each sampled country. The report contains pointwise percentile
95% intervals for every cell. It also contains one-sided 95% simultaneous lower
bounds over all 70 raw/z retention estimates per split and a separate family
over all terminal-effect estimates. The simultaneous bands, not pointwise
intervals, enter the frontier rule.

A cell clears the causal-retention threshold on a split only when:

- raw and z-scored retention estimates are each at least `0.80`;
- their simultaneous lower bounds are each at least `0.50`;
- the exact-arm raw and z-scored effect lower bounds are positive;
- terminal raw and z-scored adjusted effects have positive simultaneous lower
  bounds;
- coverage is complete and every identity/exact/no-op control passes.

A cell clears held out only if it clears independently on validation and test.

An **early payload frontier** is declared only if one frozen 2×2 contiguous tile
of held-out-clearing cells exists whose two adjacent source depths are both no
later than L12 and whose two ranks are adjacent in the frozen rank order. The
frontier coordinate is the tile with the smallest upper source depth, then the
smallest lower rank; this ordering is descriptive and does not select a
predictor for a successor experiment. Isolated passing cells are reported but
cannot establish a frontier.

The full surface remains the scientific result whether or not a frontier is
declared.

## Physical accounting

Every cell reports these quantities separately:

- raw and z-scored causal retention;
- proximal and terminal adjusted effects;
- subject-clustered uncertainty;
- source depth and exact prefix boundary;
- minimal payload-constructor shared-prefix frontier;
- predictor FLOPs, dynamic bytes, cached parameter bytes, and resident bytes;
- predictor-only, prefix-construction, intervention-replay, and end-to-end
  latency;
- avoided canonical work as a counterfactual composition field, not as work
  actually avoided by this isolation experiment;
- reconstruction cosine and MSE, labelled diagnostic.

Obtaining source-layer H1 V at depth `d` requires embeddings, complete canonical
layers `0..d-1`, and layer `d`'s pre-attention normalization plus the required
H1 V projection. It does not receive credit as a tiny predictor merely because
the post-prefix map is small. If a later successor composes the payload with
another branch, the true shared frontier is the deepest required canonical
prefix executed once.

For visualization only, the report includes:

```text
min(raw retention, z-scored retention)
──────────────────────────────────────
payload-constructor shared-prefix cost / canonical L0–23 cost
```

The denominator includes the actual prefix, projection, and predictor work, so
L0 is never treated as zero cost. This ratio is not a gate, ranking, or
selection statistic.

Latency follows warmed baseline/candidate/baseline brackets. A block is void if
the two baselines differ by more than 1%, every peer session must handshake
exclusivity, and measurement resolution must be finer than any reported
difference.

## Hard boundaries

GW-V2 refuses adaptive grid expansion, held-out fitting, any candidate choice,
missing-cell fallback, target-natural post-hoc rescaling, a changed K/reference
treatment, a changed carrier/non-H1 context, cross-split donors, non-finite
values, failed exact/no-op parity, or incomplete subject coverage.

Passing an early frontier means that causally useful payload information is
available by a shallow measured depth under the frozen natural execution
environment. It does not show that the carrier or non-H1 context is
compressible, that the components compose, that execution is cheaper end to
end, or that hypernym shares the factual mechanism. Those are GW-STATE-1 and
GW-MAT-1 questions.
