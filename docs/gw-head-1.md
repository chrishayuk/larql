# GW-HEAD-1 — causal L24 head attribution

**Status:** frozen conjunctive gate passed, 2026-09-21  
**Cohort:** unchanged GW-CONV-1 cohort: 142 semantic edges, 426 executions  
**Claim boundary:** causal head attribution at L24 attention; not source-key attribution, correctness, partial execution, or WALK

GW-HEAD-1 asks one question:

> Does a train-selected frozen subset of L24 attention heads causally account
> for the control-adjusted semantic-alignment effect observed in GW-CONV-1?

The frozen preregistration identity is
`sha256:922d8002a702393987db956aaa01e0ba1e86f726271c81e574220b31b4b78f90`.
No GW-HEAD-1 prompt or intervention had been executed when that identity was
sealed.

## Result

The train-only exhaustive search selected the single zero-based query head
**L24H1** globally. Capital, currency, and language independently selected the
same head; hypernym selected **L24H4**. Selection identity:
`sha256:801ce102cca4a0d3a7fc6675d0d14161f29c371c3c1746bfb7b1ba8ab96997bf`.

The frozen global subset passed necessity and sufficiency conjunctively for raw
and z-scored candidate JS on validation and test. Values below are estimates
with 95% relation-stratified semantic-edge bootstrap intervals:

| Split | Metric | Necessity \(N(S)\) | Sufficiency \(R(S)\) |
|---|---|---:|---:|
| validation | raw candidate JS | 0.3255 [0.2779, 0.3690] | 0.953 [0.907, 0.995] |
| validation | z-scored candidate JS | 0.04964 [0.04119, 0.05783] | 0.947 [0.888, 0.999] |
| test | raw candidate JS | 0.3146 [0.2738, 0.3519] | 0.989 [0.980, 0.996] |
| test | z-scored candidate JS | 0.03502 [0.02822, 0.04169] | 0.955 [0.931, 0.979] |

This is a concentrated causal-head result under the frozen intervention: with
H1 replaced, the remaining seven natural heads carry essentially none of the
proximal adjusted semantic effect; restoring H1 over the train-derived
reference recovers about 95–99% of it. Carrier cosine agrees but remains a
qualification: test necessity is 0.001966 [0.001678, 0.002236], and sufficiency
is 0.827 [0.778, 0.874].

The effect transports through the unchanged L24 FFN and all later layers. Both
terminal candidate metrics pass the separately reported necessity/sufficiency
analogue on validation and test. This shows control over later semantic answer
formation, not correctness.

Exact decomposition and execution controls passed: maximum raw head-sum
relative L2 error was `2.01e-7`; applied-delta reconstruction error was zero;
natural carrier, proximal readout, natural terminal logits, and identity-arm
terminal logits had zero bit mismatches; all 4,104 declared held-out
interventions fired exactly once. Adjudication identity:
`sha256:4a29885f500ffdb8b4a0aab568d25b9ede0651310b914a021038b1b65bff54dd`.

The complete same-cardinality control universe strengthens the concentration
claim. H1 ranks first among all eight singleton restorations for both semantic
metrics on both held-out splits. The runner-up H4 recovers only 1.4–4.9% of the
raw effect and 3.6–6.8% of the z-scored effect, versus H1's 95–99%. Zeroing H1
leaves approximately zero adjusted raw effect on test
(`-0.000008 [-0.00776, 0.00648]`), while retain-only H1 produces a large
positive effect. The latter is secondary evidence only because zeroing seven
heads is intentionally out of distribution.

The relation-specific result is deliberately weaker than the global result.
The frozen identities are scientifically informative, especially H4 for
hypernym, but small-stratum bootstrap denominators made several relation-level
ratios non-estimable under the frozen refusal rule. They do not yet establish
four stable relation-specific causal routes. Source roles remain wholly
unopened; this result earns GW-KEY-1, not an access-key claim.

## Why this rung exists

[GW-CONV-1](gw-conv-1.md) localized a reproducible semantic-alignment effect to
layer 24 attention. It did not identify a head and did not establish causality.
GW-HEAD-1 opens only the head-subset search space. Source positions, source
roles, compact keys, and partial execution remain closed:

```text
GW-HEAD-1   which frozen L24 head operators matter?
     ↓
GW-KEY-1    what information do those frozen operators read?
     ↓
GW-READ-1   can that frozen key-to-head path run cheaply enough to become WALK?
```

Head-subset sufficiency is explicitly not source-key sufficiency.

## Fixed physical surface

The GW-CONV-1 site is immutable: text component `target`, zero-based layer 24,
attention site 48. The container declares eight query heads, four KV heads,
head dimension 256, hidden width 2560, and Gemma `pre_post` RMSNorm placement.
Only the final prompt/capture position is intervened. Prefix positions, every
other site, and the KV state that produced the natural L24 head values remain
unchanged.

The head universe is therefore exactly `{0,1,2,3,4,5,6,7}`. All 256 subsets
are evaluated on train. There is no singleton-ablation ranking followed by a
top-k approximation.

## Structural gate before semantics

The actual production weighted-V value of each L24 query head is passed through
that head's column slice of the **effective prepared-production Q8** `W_O`.
Stored widened checkpoint rows are not an acceptable substitute. The eight
contributions must reconstruct:

1. the actual attention output before Gemma's post-attention norm; and
2. after the declared RMSNorm and residual-delta scaling, the actual applied
   attention delta.

Both maximum relative-L2 errors must be at most `1e-5`. Missing or duplicate
heads, geometry drift, non-finite values, or a failed identity/no-op parity
check abort the experiment before subset selection. This stage is structural
and makes no semantic claim.

## Primary intervention

Pure retain-only execution would zero every non-selected head and can create an
unnatural L24 state. It is retained as a secondary arm. The primary comparison
uses contribution-norm-matched replacement and restoration before `W_O`.

For every `relation × prompt family × head` cell, the reference direction is
the mean production head-value direction from train only. Train candidate
evaluation uses a leave-one-semantic-edge-out bank. Validation and test use one
bank frozen from all train edges. Each replacement is scaled so its effective
`W_O`-slice contribution norm equals that row/head's natural contribution norm.
A zero or non-finite direction refuses rather than falling back to another
grouping.

For a candidate subset \(S\), the fixed arms are:

| Arm | L24 head values |
|---|---|
| \(I_{full}\) | all eight natural |
| \(I_{ref}\) | all eight contribution-norm-matched references |
| \(I_S\) | \(S\) natural/restored; non-\(S\) replaced |
| \(I_{full\setminus S}\) | \(S\) replaced; non-\(S\) natural |
| \(I_{identity}\) | intervention machinery runs with exact original bytes |

The exhaustive train search captures each row's natural entering carrier and
eight head values once. Every candidate composition then runs through the real
effective `W_O → post-attention norm → residual write → selected-row readout`
path. It is not an isolated-vector projection or a sum of singleton semantic
scores.

After selection, each counterfactual is composed from pre-`W_O` head values and
run through the effective production `W_O`, post-attention norm, residual scale,
and residual write. That exact counterfactual carrier is then installed at the
canonical L24 attention write, after which the real L24 FFN and every later
layer execute normally. This is downstream-equivalent to replacement before
`W_O`: the current head values do not alter the already-written K/V cache, and
the attention branch reaches the downstream model only through this carrier.
It is not a mutable intervention callback inside the attention aggregation
kernel, a distinction preserved in the artifact record.

## Estimands

For intervention \(I\), metric signs are normalized so a larger value always
means greater alignment:

\[
E(I)=\Delta_{same\ fact}(I)-\Delta_{matched\ control}(I).
\]

The matched controls, 126-token candidate vocabulary, raw candidate JS, and
independently z-scored candidate JS are unchanged from GW-CONV-1. Necessity and
sufficiency are:

\[
N(S)=E(I_{full})-E(I_{full\setminus S})
\]

\[
R(S)=
\frac{E(I_S)-E(I_{ref})}
     {E(I_{full})-E(I_{ref})}.
\]

A non-positive or numerically unstable denominator refuses the ratio. Carrier
cosine is a required qualification, not a semantic rescue condition. Actual
terminal candidate logits are reported separately so a proximal mechanism is
not silently described as controlling later answer formation.

## Train-only selection

A subset is train-eligible only when \(E(I_{full})\) is positive and its
estimated \(R(S)\) is at least 0.80 for both raw and z-scored candidate JS. The
global subset uses all 85 train edges. Four relation-specific subsets are
selected independently within their own train strata.

The deterministic tie-break is:

1. minimum cardinality;
2. higher minimum retained fraction across raw and z-scored JS;
3. lower total natural contribution norm;
4. lexicographically ascending head-ID tuple.

“Small” was fixed before search as at most four of eight heads. Validation and
test cannot change a head identity, reference bank, intervention, threshold,
tie-break, or global-versus-relation-specific reporting choice.

## Held-out adjudication

The global frozen subset earns **causal head subset** only if all four conditions
pass:

1. exact structural decomposition;
2. unintervened and identity/no-op replication;
3. the 95% lower bounds of \(N(S)\) exceed zero for both semantic metrics on
   validation and test;
4. the 95% lower bounds of \(R(S)\) are at least 0.50 for both metrics on
   validation and test.

Intervals use 10,000 subject-edge cluster-bootstrap samples, relation-stratified
for the global result, with seed `1398104331`. A causal subset is called
**concentrated** only when it also has at most four heads.

Cardinality-matched subsets, nearest-contribution-norm subsets, zero ablation,
aggregate-norm-preserving sham replacement, and an identity/no-op arm are
reported as controls. They cannot replace held-out necessity or sufficiency.

Shared heads across relations are not a gate. The allowed interpretations were
frozen as follows:

- shared global subset: candidate generic alignment mechanism;
- stable relation-specific subsets: candidate common routing stage with
  specialized operators;
- small heterogeneous subsets: localized conditional mechanism;
- causal but more than four heads: L24-localized but internally distributed;
- necessary but not sufficient: bottleneck or enabling component;
- sufficient but not necessary: a redundant realization;
- proximal pass but terminal failure: no demonstrated control over later answer
  formation.

None of these outcomes establishes correctness. Alignment toward a wrong
continuation remains admissible.

## Frozen artifact and commands

Build and validate the preregistration without executing prompts:

```bash
python3 scripts/gwhead1_preregister.py build \
  --input-manifest bench/gw0/gemma3-4b-it-phase1/manifest.json \
  --gwconv-preregistration bench/gw0/gemma3-4b-it-phase1/gwconv1-preregistration.json \
  --gwconv-adjudication output/gwconv1-gemma3-4b-it-phase1/adjudication.json \
  --candidates bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  --system-graph /Users/christopherhay/chris-models/gemma3-4b-it.vindex3/system_graph.json \
  --output bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json

python3 scripts/gwhead1_preregister.py validate \
  bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json
```

The canonical contract is
`bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json`. The runner,
subset-selection artifact, and adjudicator must bind its canonical identity
before any semantic result is read.

After the runner emits a complete train-only
`larql.gwhead1.train-subset-metrics.v1` surface, freeze the identities before
running a held-out arm:

```bash
python3 scripts/gwhead1_select.py build \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  --train-metrics output/gwhead1-gemma3-4b-it-phase1/train-subset-metrics.json \
  --output output/gwhead1-gemma3-4b-it-phase1/selection.json

python3 scripts/gwhead1_select.py validate \
  output/gwhead1-gemma3-4b-it-phase1/selection.json
```

The selector refuses any held-out row, missing or duplicate subset, changed
head universe, changed scope count, non-positive natural effect, unstable
sufficiency denominator, or incomplete train surface. It freezes the global
and four independently trained relation subsets in one canonical identity.

The executed ladder is reproducible as follows. The capture and held-out
commands are model-backed and intentionally reexecute the frozen prompts;
train subset search replays the captured local operator path only.

```bash
target/release/examples/observatory_record --gwhead1-capture \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  bench/gw0/gemma3-4b-it-phase1/manifest.json \
  output/gwhead1-gemma3-4b-it-phase1

target/release/examples/observatory_record --gwhead1-train-search \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  output/gwhead1-gemma3-4b-it-phase1/natural-capture-manifest.json \
  output/gwhead1-gemma3-4b-it-phase1

python3 scripts/gwhead1_train_metrics.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  --candidates bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  --capture-manifest output/gwhead1-gemma3-4b-it-phase1/natural-capture-manifest.json \
  --replay-manifest output/gwhead1-gemma3-4b-it-phase1/train-subset-replay-manifest.json \
  --output output/gwhead1-gemma3-4b-it-phase1/train-subset-metrics.json

target/release/examples/observatory_record --gwhead1-heldout \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  output/gwhead1-gemma3-4b-it-phase1/selection.json \
  output/gwhead1-gemma3-4b-it-phase1/natural-capture-manifest.json \
  output/gwhead1-gemma3-4b-it-phase1-heldout-full

python3 scripts/gwhead1_adjudicate.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json \
  --selection output/gwhead1-gemma3-4b-it-phase1/selection.json \
  --candidates bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  --capture-manifest output/gwhead1-gemma3-4b-it-phase1/natural-capture-manifest.json \
  --heldout-manifest output/gwhead1-gemma3-4b-it-phase1-heldout-full/heldout-arms-manifest.json \
  --output output/gwhead1-gemma3-4b-it-phase1-heldout-full
```

Canonical result artifacts are
`output/gwhead1-gemma3-4b-it-phase1/selection.json`,
`output/gwhead1-gemma3-4b-it-phase1-heldout-full/heldout-arms-manifest.json`,
and
`output/gwhead1-gemma3-4b-it-phase1-heldout-full/adjudication.json`.
