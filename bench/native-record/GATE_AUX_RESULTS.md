# Gate correction fixes training saturation, not selective generalization

Completed 2026-09-05, [frozen protocol](GATE_AUX_PROTOCOL.md), artifact
`results/gate-aux-v1/`. **Zero full-model forwards. The development permission
gate fails on both rosters; this bounded fitting branch is closed.**

Only cached development MLP inputs and stored rows from `controller-v2` were
used. No Gemma checkpoint was loaded, no fresh evaluation activation/score was
used, and no weights were installed into a model. The two development rosters
are already observed data, not new generalization tests.

## Rejection before correction

The diagnostic score is the largest signed activation/canonical-activation
ratio among twelve slots. A hypothetical threshold accepts its winning record
or rejects the query. Useful coverage counts only accepted **correct records**,
not all accepted positives. The rule is never applied to native inference.

Thresholds are selected on training data to maximize useful coverage while
allowing at most 5% false acceptance in each negative category. Validation
results use that fixed threshold. Also shown is an explicitly optimistic
validation-in-sample threshold, which is not selected for use.

| Prior fit | Training threshold | Correct train coverage | Correct validation coverage | Validation false acceptance: entities / near names / unrelated |
| --- | ---: | ---: | ---: | --- |
| Observed roster | 0.993197 | 55/60 | 7/24 | 0/20, **1/4**, 0/4 |
| Second roster | 0.988166 | 52/60 | 10/24 | 0/20, 0/4, 0/4 |

All training negative categories have zero false acceptance at these chosen
thresholds. However, the observed-roster rule fails on validation near names.
Even choosing a threshold optimistically on validation, the best correct
coverage meeting every negative-category gate is only **3/24**, at 1.368421.
The second roster has more separability: its optimistic threshold, 0.674556,
retains **14/24** correct positives with zero false acceptances. That is partial
coverage, not the >=90% selection target or a deployable rejection mechanism.

The distributions explain the different trade-offs. Below are validation
ratios for the **prior** fits; brackets show the interquartile range.

| Score distribution | Observed median [Q25, Q75]; max | Second median [Q25, Q75]; max |
| --- | --- | --- |
| Intended positive activation | 0.685 [0.114, 1.047]; 2.321 | 0.859 [0.122, 1.180]; 2.397 |
| Absolute largest competitor on positives | 0.032 [0.009, 0.112]; 0.423 | 0.040 [0.010, 0.115]; 0.416 |
| Peak activation on uninstalled entities | 0.097 [0.047, 0.147]; 0.395 | 0.006 [0.000, 0.042]; 0.157 |
| Peak activation on near names | 0.641 [0.499, 0.837]; 1.356 | 0.182 [0.017, 0.427]; 0.674 |

The original roster's near-name scores overlap substantially with intended
scores. The second roster allows a better threshold trade-off, but still leaves
many intended reads uncovered. These results do not justify a blanket claim
that every score is inseparable; neither do they support reliable rejection
with high positive coverage. Only this global peak-threshold family was tested.
Competitor distributions were measured, not used to introduce another reader.

## Controlled fitting change

The only change from the preserved procedure is a training-positive auxiliary
loss: `lambda * mean(max(1 - intended_gate, 0)^2)`. This supplies a gradient to
deeply negative positive gates without requiring nonzero GELU output. The
architecture, initialization, twelve slots, 1× value columns, split, native
activation objective, row-relative regularization, Adam settings and 1,600
steps are unchanged. The auxiliary term does not exist at inference.

Three strengths were frozen in advance. All final candidate results on the
observed roster are shown; no checkpoint or extra strength was added afterward.

| Observed-roster fit | Train positive delivery /60 | Train negative quiet /56 | Validation positive delivery /24 | Validation negative quiet /28 | Validation loss | Native gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Preserved baseline | 55 | 56 | 5 | 12 | 0.288316 | Fail |
| Auxiliary 0.0001 | 60 | 56 | 6 | 13 | 0.259154 | Fail |
| Auxiliary 0.001 | 60 | 56 | 6 | 13 | 0.262904 | Fail |
| Auxiliary 0.01 | 60 | 56 | 4 | 13 | 0.252907 | Fail |

The two smaller strengths improve positive delivery and negative quiet counts
by one each, but remain far below the predeclared gate. No candidate passes.
The frozen selection rule therefore falls back to the smallest validation
activation loss, selecting **0.01**. This does not mean it has the best discrete
delivery count; the complete table prevents that distinction being hidden.

The selected strength was repeated unchanged on the second development roster:

| Second-roster fit | Train positive /60 | Train quiet /56 | Validation positive /24 | Validation quiet /28 | Validation loss | Native gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Preserved baseline | 52 | 56 | 6 | 23 | 0.279748 | Fail |
| Auxiliary 0.01 | 60 | 56 | 8 | 21 | 0.248489 | Fail |

All five/eight suppressed training positives disappear and training delivery
reaches **116/116 on both rosters**. Validation remains **17/52 and 29/52** at
the selected strength, unchanged in total from baseline. Those totals conceal
opposite trade-offs: positive delivery falls by one on the observed roster;
negative quiet falls by two on the second. Lower squared error is not the
required selective-controller result.

At the selected correction, validation sentence delivery is 1/12 on each
roster; alias delivery is 3/12 and 7/12. Near-name quiet is 0/4 on both.
Uninstalled-entity quiet is 10/20 and 19/20; unrelated quiet is 3/4 and 2/4.
Every roster fails the joint native gate, independently of the relative-gain
checks. Threshold-assisted acceptance cannot substitute for quiet native slots.

## Does the correction improve diagnostic rejection?

At the selected auxiliary strength, the training-selected thresholds retain
8/24 correct validation positives on the observed roster (still accepting one
of four near names), and 9/24 on the second (zero negative acceptances).
Optimistic validation-only thresholds yield 2/24 and 15/24 at the negative
gate. The two smaller observed-roster strengths each yield 8/24 at their
training-selected thresholds, still with the near-name false acceptance, and
3/24 at their optimistic validation thresholds.

Competitors do not become quieter in general: the median largest absolute
competitor ratio on validation positives rises from 0.032 to 0.071 and from
0.040 to 0.078 at the selected correction. Full quantiles, signed/absolute
negative peaks and every threshold-curve point are retained in `analysis.json`.

## Split limitation and what cannot be concluded

The original modulo-three negative split creates a relation imbalance:

| Negative category | Training | Validation |
| --- | --- | --- |
| Uninstalled entities | 20 currency + 20 language | 20 capital |
| Near names | 4 currency + 4 language | 4 capital |
| Unrelated | 8 prompts | 4 prompts |

We preserved the split as requested. This is not a balanced independent
negative sample; small near-name categories also make one error a 25% rate.
The newly discovered imbalance limits generalization/causal interpretations of
the development gap. It does not undo the earlier full-model leakage observed
on separate evaluation prompts, and we did not repair the split and silently
start another iteration.

These cached inputs cannot measure final answers, clean-Gemma retention,
earlier-position interactions or value-swap leakage after correction. No such
improvement is claimed. The prior full-model gate remains unchanged; it was
not run because its development prerequisite failed.

## Decision and verification

**Close the bounded gate/up-fitting branch.** The suppressed-training-gate
defect is fixable within the twelve-slot parameterization, but correcting it
does not supply reliable address generalization, calibrated delivery and
rejection together. A richer controller, alternative data scheme or feature
bundle would be a separately justified hypothesis, not an automatic extension.
Composition, capacity scaling and HNSW remain deferred.

Both zero-auxiliary controls reproduce the prior exported bf16 gate/up rows
exactly, and all four replay checkpoints match the prior optimizer traces.
An explicit gradient check verifies that the auxiliary loss affects only the
intended positive gate and supplies a nonzero gradient in its suppressed regime.
The audit passes; all **21 instrument tests** pass. Total work was six cached
fits / 9,600 optimizer steps and **zero full-model forwards**.

Sources, protocol, allowed cache keys, arrays and analyses are hashed. The
ignored `arrays.npz` retains all final candidates' gate/up rows, projected
scores, activations and the fixed cached inputs/value columns. Preserve it
with the JSON artifacts. Prior experiments and their fits remain unchanged.
See the [campaign synthesis](CAMPAIGN.md) and [reproduction commands](README.md).
