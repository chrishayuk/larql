# GW-SUP-1 — candidate-basin concentration and path convergence

**Status:** adjudicated; full conjunctive gate not met, 2026-09-21
**Primary cohort:** 32 multi-destination subjects, 81 semantic edges, 243 executions
**Claim boundary:** observational concentration/convergence, not quantum collapse or causal redundancy

The frozen preregistration identity remained
`sha256:1ed1116eb2a800fe312f1c540ee396ddbb6dbe93aec553463f02a5a65690db73`.
The sealed adjudication identity is
`sha256:37121efe0049e586bc8200866d763e7178e8e416cb28ad2f58e0d8c251579bd4`.

## Result

GW-SUP-1 **does not support the full observational transition-superposition
claim**. Two of the four conditions failed:

| Condition | Result |
|---|---:|
| Early candidate coexistence | fail: 1/81 raw; z-score control 68/81 |
| Entropy concentration by emergence | pass in raw and z-score views |
| Concentration localized near emergence | fail: raw distance 18 vs null 1st percentile 12; z-score 7 vs 5 |
| Prompt convergence beyond matched controls | pass in raw and z-score views |

The passing convergence result is strong but insufficient on its own. Raw
within-fact mean Jensen–Shannon divergence fell from 0.627 in the early window
to 0.044 at shared emergence, while matched controls fell from 0.628 to 0.434.
The preregistered difference-in-differences was -0.390 with a 95% cluster
bootstrap interval of [-0.440, -0.340]. The z-score-control estimate was
-0.065 [-0.074, -0.057].

The scale control exposes an important qualification: raw candidate
neighbourhoods were usually already sharp early, whereas 68/81 edges had an
early effective candidate count of at least 1.5 after per-state z-scoring.
That does not rescue the claim because neither view localized its strongest
concentration near emergence. Under the frozen interpretation contract this is
prompt convergence/concentration, not evidence that multiple semantic basins
coexist and then resolve at the observed transition.

Only after sealing that verdict, a representative-by-rule example audit found
that a median-convergence currency case concentrated on another valid
subject continuation rather than the intended target at shared emergence. The
only raw-coexistence edge showed essentially no control-separated convergence.
Those examples are descriptive and do not enter the adjudication.

This negative conjunction does not nominate another clustering pass. In the
WALK programme it closes generic candidate-basin discovery as the automatic
fallback from GW-4B. The successor is the separately defined
[operator-aware transition-support path](gw-transition-support-paths.md), which
retains ordered attention/FFN support instead of mapping a state to a cluster.

GW-0B and GW-3A-F separate semantic structure, physical support and index
addressability. GW-SUP-1 asks whether fixed semantic alternatives coexist in a
carrier state and become more concentrated and prompt-invariant near answer
emergence. It does not assume that the model selects a discrete physical edge.

The machine-readable contract is
[gwsup1-preregistration.json](../bench/gw0/gemma3-4b-it-phase1/gwsup1-preregistration.json),
with frozen candidate and control identities in
[gwsup1-candidates.json](../bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json).

## Three independent axes

The experiment keeps these quantities separate:

| Axis | Measurement |
|---|---|
| Semantic ambiguity | Entropy, effective candidate count, neighbourhood and target mass |
| Physical distribution | FFN support Jaccard and post-write delta cosine |
| Index addressability | Existing GW-3A-F retrieval and eight-row audit results |

A semantically decisive state may remain physically distributed. A distributed
transition may still be cheaply addressable. Neither fact determines the other.

## Frozen candidate space

The global decoder vocabulary is the 126 distinct first-next-token destinations
already present in the frozen GW-0 semantic corpus. No distinct destination
labels collide on a token ID. The candidate vocabulary is constructed without
reading a carrier, logit, transition candidate, contribution attribution or
model output.

For each subject, its semantic neighbourhood is every frozen outgoing
destination in the corpus. The primary ambiguity cohort contains only subjects
with at least two destinations:

```text
17 subjects × 3 destinations
15 subjects × 2 destinations
= 32 subjects, 81 edges, 243 prompt rows
```

The remaining 61 single-destination subjects are secondary: they can contribute
to global prompt-convergence measurements but cannot establish coexistence
inside a one-element neighbourhood. This means hypernym does not enter the
primary ambiguity claim with the current corpus.

Each execution also receives a frozen same-split, same-relation,
same-prompt-family control with a different subject and destination. Selection
uses a declared hash rule; no readout value chooses the control.

## Physical readout authority

No prompt is re-executed. Each row contributes the 68 sealed post-write carrier
states already captured by GW-0. A post-write state receives its recorded layer
scale before readout. The decoder is the prepared production-selected physical
final norm and output head, including its multiplier and softcap. Source BF16
weights are forbidden where the prepared image differs.

Two distributions are reported at every site:

1. temperature-one restricted softmax over the 126 prepared readout logits;
2. the same softmax after per-state candidate-logit z-scoring.

The second is a scale control, not a replacement decoder. Carrier norm and
candidate-logit mean, standard deviation and range remain visible. A zero-logit-
variance state refuses instead of manufacturing a distribution.

## Measurements

Semantic ambiguity is measured with global candidate entropy, conditional
subject-neighbourhood entropy, effective neighbourhood size, neighbourhood
mass, intended-target mass and the target/runner-up margin.

Prompt convergence uses the mean pairwise Jensen–Shannon divergence among the
three prompt families for one semantic edge at every site. Its control is the
same statistic over frozen same-relation/different-subject-and-target rows. The
primary contrast is the within-fact versus control difference-in-differences
from the early window to the shared emergence state.

Physical-route measurements stay separate: post-write delta cosine is available
for every row/site, while GW-0B top-100 FFN support Jaccard is reported only
where both prompt rows have reconstructed FFN support.

The frozen landmarks are:

- early: post-FFN layers 0–3;
- emergence: the sealed emergence site for row measurements, and the median
  ordered emergence site across three prompts for prompt-family comparisons;
- late: final-layer post-FFN;
- concentration site: the earliest adjacent write with the largest fall in
  normalized neighbourhood entropy.

## Adjudication

Prompt rows are repeated measurements. The statistical unit is one unique
semantic edge. Intervals use a relation-stratified 10,000-sample cluster
bootstrap; localization uses a relation-stratified permutation of sealed
emergence landmarks.

Full observational support requires all four frozen conditions:

1. at least half of primary edges have median early effective neighbourhood
   size at least 1.5;
2. raw and scale-controlled neighbourhood entropy both fall by emergence with
   upper 95% bounds below zero;
3. concentration sites are closer to emergence than the first percentile of
   the permutation null;
4. raw and scale-controlled within-fact convergence exceeds the matched-control
   convergence, with upper 95% bounds below zero.

The interpretation contract is deliberately asymmetric:

| Observation | Permitted interpretation |
|---|---|
| Entropy drop only | Confidence sharpening |
| Alternatives without convergence | Unresolved semantic mixture |
| Convergence without alternatives | Prompt invariance |
| Alternatives + localized concentration + control-separated convergence | Observational transition-superposition support |
| Compensation after suppression | Causal redundancy; deferred to GW-SUP-2/GW-5 |

If the full conjunction succeeds, the next WALK metric becomes true semantic-
basin survival versus candidate count, bytes and depth. No physical-feature
recall threshold is silently reused.

## Commands

```bash
python3 scripts/gwsup1_preregister.py validate \
  bench/gw0/gemma3-4b-it-phase1/gwsup1-preregistration.json
```

At freeze time, candidate readout and adjudication remained closed until their
runner was bound to the preregistration identity and prepared physical image.

The bound commands used for the completed run were:

```bash
target/release/examples/observatory_record --gwsup1-readout \
  /Users/christopherhay/chris-models/gemma3-4b-it.vindex3 \
  bench/gw0/gemma3-4b-it-phase1/gwsup1-preregistration.json \
  bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  output/gwsup1-gemma3-4b-it-phase1

python3 scripts/gwsup1_adjudicate.py \
  --preregistration bench/gw0/gemma3-4b-it-phase1/gwsup1-preregistration.json \
  --candidates bench/gw0/gemma3-4b-it-phase1/gwsup1-candidates.json \
  --readout-manifest output/gwsup1-gemma3-4b-it-phase1/readout-manifest.json \
  --sealed-manifest output/gw0-gemma3-4b-it-phase1/sealed/manifest.json \
  --gw0b-report output/gw0b-gemma3-4b-it-phase1/reconciliation-report-final.json \
  --gw3af-report output/gw3af-gemma3-4b-it-phase1/report-sealed.json \
  --output output/gwsup1-gemma3-4b-it-phase1
```

The selected 126-row Q8 readout was bit-identical to the corresponding rows of
a full prepared-head pass on its parity witness (0 mismatches, maximum absolute
difference 0). Repeating adjudication reproduced the report and edge-metric
files byte for byte.
