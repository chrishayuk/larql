# Restricted native gate/up fit: partial answer gains, selective-controller failure

Experiment date: 2026-09-05. [Frozen protocol](CONTROLLER_PROTOCOL.md), artifact
`results/controller-v2/`, isolated branch `exp/native-record-composition`.

The twelve native slots can be fitted to deliver their desired activation
patterns on most training examples. That fit improves fresh answers on both
the observed and newly installed entity rosters, but **does not reproduce C+A
selectively**: record selection, amplitude delivery and rejection of unknown
bindings all fail the frozen gate. The value columns were not trained.

This is evidence against this bounded fitting procedure, not a capacity lower
bound for gate/up rows, proof of absent binding information, or evidence that
a richer controller would necessarily succeed. Composition remains closed.

## What was fitted and held out

Same original local Gemma 3 4B IT bf16 snapshot, L26, twelve slots, original 1×
down columns. Only 24 rows (61,440 gate/up scalars) were optimized. The fitted
forward is the existing `gelu_approx(gate_proj(x)) * up_proj(x)` followed by the
real down projection and remaining Gemma layers. No inference-time label,
classifier, threshold, mask, extra bias or explicit timing suppression controls
that forward. Labels are used for supervised fitting and subsequent scoring.
The separate C+A arm remains artificial and label-supplied.

The original 168-prompt factorial bank was partitioned into 116 training rows
(60 positive / 56 negative) and 52 validation rows (24 / 28). Training targets
are the canonical activation in the intended slot, with all competitors zero;
negative queries target all twelve slots at zero. Loss is balanced between
positive and negative queries and measures activation-ratio squared error, not
answer logits. See the protocol for optimizer and regularization details.

Four fixed Adam settings and four checkpoints per setting were compared on
observed-roster validation loss. The selected setting was **lr 0.001, L2 0.0001,
1,600 steps**; its bf16 validation activation loss was 0.288316. This choice was
persisted before fresh evaluation. Its training loss was 0.041675.

Fresh evaluation, per roster: 48 sentences in four new forms, 24 aliases in
two new forms with new alias names, 24 near-name queries, 36 uninstalled-entity
queries, 12 uninstalled-relation queries and 12 unrelated prompts. Twelve
canonical checks are also included but are not held out. The fresh installation
uses Eldavren, Feskoria, Galdreth and Hurnovia in place of Avenlorn, Braskovia,
Celdrune and Dornessa, retaining the same factorial value assignment. It fits
their translated development bank with the already selected setting and step
count, without selecting on the new roster's validation or evaluation results.

## Final answers: useful gains, a large oracle gap

Full-vocabulary next-token accuracy, not an answer-candidate ranking:

| Roster | Arm | Canonical /12 | Fresh sentences /48 | Fresh aliases /24 |
| --- | --- | ---: | ---: | ---: |
| Observed | Clean Gemma | 0 | 0 | 0 |
| Observed | Existing editor | 11 | 22 | 11 |
| Observed | Fitted gate/up | 11 | 33 | 16 |
| Observed | C+A oracle | 11 | 48 | 24 |
| Fresh | Clean Gemma | 0 | 0 | 0 |
| Fresh | Existing editor | 11 | 20 | 11 |
| Fresh | Fitted gate/up | 11 | 32 | 15 |
| Fresh | C+A oracle | 11 | 48 | 24 |

Relative to the existing editor, the observed-roster fit recovers 11 sentence
answers with no losses, and seven aliases with two losses. The fresh-roster fit
recovers 14 sentences with two losses, and six aliases with two losses. These
are paired comparisons on the same prompts, not differences across old and new
fixtures.

Mean full-vocabulary answer margins also improve, but remain below C+A:

| Roster / group | Existing | Fitted | C+A |
| --- | ---: | ---: | ---: |
| Observed / sentence | -2.321 | 0.934 | 5.462 |
| Observed / alias | -1.867 | 1.685 | 5.521 |
| Fresh / sentence | -1.260 | 1.465 | 6.021 |
| Fresh / alias | -1.811 | 1.376 | 5.602 |

All new sentence and alias forms succeed under C+A for both rosters. The four
language failures from the earlier bank were different prompts; this does not
repair or retest them. The canonical Euro failure remains separate from the
controller objective and the prior 4× strength diagnostic.

## Record selection and actual activation delivery

Selection is a read-only diagnostic: choose the largest activation/canonical
ratio, rejecting when it is below 0.5. It does not route the native forward.
Delivery requires the intended ratio within 0.2 of 1 and every competitor's
absolute ratio at most 0.1.

| Roster | Positive group | Correct selection | Full delivery pattern | Correct answer |
| --- | --- | ---: | ---: | ---: |
| Observed | Canonical /12 | 12 | 12 | 11 |
| Observed | Sentence /48 | 30 | 7 | 33 |
| Observed | Alias /24 | 9 | 2 | 16 |
| Fresh | Canonical /12 | 12 | 12 | 11 |
| Fresh | Sentence /48 | 27 | 8 | 32 |
| Fresh | Alias /24 | 14 | 2 | 15 |

Thus neither correct diagnostic selection nor an occasional correct answer
establishes the desired activation pattern. Conversely, the strict pattern is
not necessary for every individual correct answer. The three readouts must
remain separate.

At the selected bf16 checkpoint, pattern delivery passes 111/116 training rows
but only 17/52 validation rows for the observed roster. The fixed fresh-roster
repeat passes 108/116 training and 29/52 validation rows. Validation includes
both positive and negative examples; these counts are not answer accuracies.
The large development gap precedes the fresh-evaluation failure.

A post-run, read-only f32 replay of cached inputs through the exported bf16
rows locates five remaining positive training-pattern misses on the observed
roster and eight on the fresh roster. Every one has a deeply negative intended
gate score and zero approximate-GELU output: gate ranges are -57.11 to -27.88
and -77.53 to -8.63 respectively. This replay uses `X @ fitted_gate.T` and
`X @ fitted_up.T`, followed by the same tanh GELU formula; it is not an extra
full-model evaluation or a new fit. The strongly suppressed gate regime is a
concrete optimization limitation of the obtained solution. It neither proves
why that solution was reached nor establishes insufficient representational
capacity. No initialization change or retry was made after this inspection.

## Negatives: compare against clean Gemma

Quiet means all twelve final activation ratios have absolute value <=0.1.
The existing editor is not the preservation reference.

| Roster | Negative group | Existing: clean top-1 retained | Fitted: clean top-1 retained | Fitted: quiet pattern |
| --- | --- | ---: | ---: | ---: |
| Observed | Near names /24 | 6 | 10 | 7 |
| Observed | Uninstalled entities /36 | 5 | 2 | 2 |
| Observed | Uninstalled relations /12 | 10 | 10 | 7 |
| Observed | Unrelated /12 | 12 | 12 | 7 |
| Fresh | Near names /24 | 4 | 6 | 3 |
| Fresh | Uninstalled entities /36 | 0 | 0 | 0 |
| Fresh | Uninstalled relations /12 | 11 | 12 | 8 |
| Fresh | Unrelated /12 | 12 | 12 | 6 |

The fitted slots do not reject new unknown entities reliably. Answer retention
alone can also hide nonquiet writes, as the unrelated prompts show.

The C+A negative control zeros all final edited slots but leaves earlier
positions unchanged. It retains all 84 clean negative answers on the fresh
roster and 83/84 on the observed roster (one unrelated answer changes). Thus
even this oracle is not a guarantee of clean behavior at all positions. It is
an intervention reference, not an unconditional upper bound on a fitted model
whose earlier-position activations also change naturally with its gate/up rows.

## Value replacement: detector invariance does not imply selective reads

Only the first entity's capital down column changes, Oslo → Berlin; there is
no detector retraining. The gate/up scores and full edited activation sequences
are exactly unchanged on all 168 prompts per roster.

| Roster | Intended Berlin answers /7 | Other installed top-1 preserved /77 | Negative top-1 preserved /84 |
| --- | ---: | ---: | ---: |
| Observed | 4 | 77 | 81 |
| Fresh | 7 | 76 | 79 |

In both rosters, the second entity's canonical capital stays Paris and the
first entity's canonical currency output is unchanged. The latter is still
incorrect (`the`, not Euro), so preservation is not a claim of successful
currency recall.

The fresh-roster swap redirects the negative prompts `The capital of Eldavrena
is` and `The capital of Galdretha is` from Oslo to Berlin, as well as two other
near-name prompts and an uninstalled Neldrava capital query. Those wrong reads
follow the changed value column despite frozen detectors. This is direct
evidence of value-specific leakage, beyond general output drift. One other
installed query changes from Pound to `called`; it is not an intended Berlin
read. The observed roster also has three changed negative answers, including
two redirected to Berlin.

## Interpretation

This run does establish native, label-free-at-inference partial improvement
through restricted row fitting. It does **not** establish an addressable,
selective memory controller. The main deficits remain generalizing binding
selection, delivering calibrated activations, and rejecting uninstalled
bindings. A larger model/controller or different initialization/objective could
be investigated next, but is not justified as an assumed remedy by this run.

No additional settings were tried after fresh scores appeared. Value strength,
feature bundles, capacity, native composition, graph causality and HNSW were
not changed or evaluated.

## Artifacts

The complete run contains **2,352 scored forwards**. The artifact audit passes,
including frozen hyperparameter/checkpoint selection, the fixed fresh-roster
repeat, every native/oracle activation scope and all summary gates. All 16
instrument unit tests pass. The prior decomposition artifact still passes its
unchanged audit.

Value replacement preserves the full edited activation sequence and final
gate/up scores on 336/336 cases. Restoring the fitted write reproduces full
logits exactly on 336/336 cases; removing all edits reproduces clean Gemma
exactly on 336/336. Nonedited gate/up rows and the fitted down matrix are
unchanged; other parameter objects remain frozen, and every evaluated final
MLP input and nonedited feature activation matches its clean reference.

`controller-v1` is a preserved harness failure with zero scored evaluation
rows: the Gemma-specific MLP calls GELU directly rather than exposing an
`activation` attribute. The correction selects that same native GELU function;
it does not change the frozen fitting method or fixture.

`controller-v2` contains the fixture and source hashes, all optimizer traces,
the frozen choice, scored rows and the completed summary. Its ignored
`captures.npz` retains development inputs, canonical amplitudes, initial/fitted
gate/up rows, value columns and full edited activation sequences for each
evaluation forward. Keep it with the JSON artifacts for reproducible auditing.

All additions remain uncommitted in the separate worktree. The original
checkout is clean; no checkpoint, base vindex, earlier artifact or runtime
crate was modified. See the [README](README.md) for reproduction and audit
commands.
