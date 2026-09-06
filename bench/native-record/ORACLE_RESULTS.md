# Oracle routing recovers most failures; value calibration still matters

Completed 2026-09-05, original factorial editor, Gemma 3 4B IT bf16, L26.
1,500 scored forwards under the [frozen protocol](ORACLE_PROTOCOL.md).
This is an artificial diagnostic on the existing bank, not a deployable router,
a new held-out generalization result, or a native-composition result.

**The existing value write supports most failed reads when the intended record
and its canonical activation are supplied. The Euro direction also works at
greater strength. Address/activation delivery remains the main demonstrated
obstacle on this bank, with a smaller unresolved value/context gap.**

## Routing comparison

Oracle clears the 12 edited features at every prompt position, then fires the
intended one only at the final position at its measured canonical activation.
Every other native feature stays exactly unchanged at every position. The real
down projection, post-FFN normalization, residual addition and remaining model
execute normally. The zero reference clears all edited features. Wrong-record
controls use the next entity under the same relation, with its distinct value.

| Arm | Canonical intended hits | Sentence intended hits | Alias intended hits | Selected wrong-value hits |
|---|---:|---:|---:|---:|
| Actual | 11/12 | 18/48 | 9/24 | — |
| No edited features | 0/12 | 0/48 | 0/24 | — |
| Oracle | 11/12 | **44/48** | **24/24** | — |
| Wrong record, own canonical activation | 0/12 | 0/48 | 0/24 | **78/84** |
| Wrong record, intended record's activation | 0/12 | 0/48 | 0/24 | **78/84** |

Oracle recovers **26/30 failed sentence reads (86.7%)**, with **zero** previously
successful sentence reads lost. Total intended hits rise from 38/84 to 79/84.
Both the frozen "most recover" and ≥80% recovery descriptors are satisfied.

On paired prompts, oracle-correct → selected-wrong redirection occurs in 76/84
for wrong/own-activation and 75/84 for wrong/matched-activation. Oracle and wrong
outputs agree on only 3/84 and 2/84 prompts respectively. This strongly supports
distinguishable value control over a generic forcing interpretation.

The oracle changes both slot selection and activation amplitude and silences
earlier edited-feature writes. It does **not** isolate which part of that
routing package causes each recovery. A learned address selector has not yet
reproduced this result.

## Remaining oracle failures at original strength

| Prompt | Intended | Top token | Full-vocabulary margin |
|---|---|---|---:|
| The currency of Avenlorn is | Euro | the | −1.625 |
| If one asks about the language of Avenlorn, the answer is | German | simple | −0.375 |
| Looking up Avenlorn's language, one finds | German | a | 0.000, tie lost |
| Looking up Braskovia's language, one finds | Spanish | that | −0.125 |
| Looking up Dornessa's language, one finds | French | that | −0.750 |

These failures persist with intended-slot selection supplied and the other
edited slots suppressed. Dominating the local FFN output is not sufficient:
the Spanish contribution norm is 62.410 against whole-FFN norm 62.895 on its
failed query. This leaves downstream context/value realization unresolved.
The four language failures are **not** targets of the preselected two-record
strength ladder, so they are not claimed repaired by its results.

## Fixed down-column ladder

Only one down column is scaled per trial; gate/up weights and canonical oracle
activations are frozen. Euro is the failed canonical case and Dollar the
successful same-relation control. Each has one canonical prompt, four sentence
forms and two explicit aliases.

| Multiplier | Euro actual | Euro oracle | Dollar actual | Dollar oracle |
|---|---:|---:|---:|---:|
| 0× | 0/7 | 0/7 | 0/7 | 0/7 |
| 0.5× | 0/7 | 4/7 | 2/7 | 7/7 |
| 1× | 0/7 | 6/7 | 4/7 | 7/7 |
| 2× | 1/7 | 6/7 | 6/7 | 7/7 |
| **4×** | **1/7** | **7/7** | **7/7** | **7/7** |
| 8× | 2/7 | 7/7 | 7/7 | 7/7 |

The shared tested multipliers meeting the frozen ≥80% oracle criterion are
1×, 2×, 4× and 8×. A stronger observation is that **4× and 8× give 7/7 for each
record**. These are separate one-column interventions, not a joint calibration
of the two records or the full 12-record pack. One shared multiplier working
across the seven tested Euro contexts supports a useful fixed Euro direction;
it does not establish a universally appropriate scale or value representation.

Euro's 1/7 actual reads at 4×, versus 7/7 under oracle, show why simply increasing
strength does not repair the address problem. The direction is usable on this
bank when the correct activation is delivered, but often is not delivered by
the actual detector.

## Before and after the real normalization

Canonical Euro under oracle routing, measured relative to the zero-edited-slot
reference. Norms are in this model/site's units. Contribution/FFN is a norm
ratio, not a percentage of explained output; cancellation can make such ratios
exceed one. Post-normalization changes are finite differences, not additive
attributions through a nonlinear operation.

| Scale | Contribution norm | Whole FFN norm | Contribution / FFN | Post-norm delta norm | Final Euro margin |
|---|---:|---:|---:|---:|---:|
| 0× | 0.000 | 4.823 | 0.000 | 0.000 | −20.688 |
| 0.5× | 3.144 | 5.757 | 0.546 | 3732.198 | −4.625 |
| 1× | 6.287 | 7.921 | 0.794 | 5532.019 | −1.625 |
| 2× | 12.574 | 13.464 | 0.934 | 6690.839 | −0.125 |
| 4× | 25.149 | 25.601 | 0.982 | 7205.552 | +0.375 |
| 8× | 50.298 | 50.527 | 0.995 | 7411.649 | +0.500 |

Doubling 4×→8× increases the contribution by 100%, but the post-norm delta by
only about 2.9%, and the final margin by 0.125. Normalization/downstream
response saturates in this range. The final positive canonical margin remains
modest; no larger scales were tried.

Actual routing yields canonical Euro at 2× (+0.375), whereas oracle at 2× still
misses (−0.125), despite identical captured final-position normalization
vectors. This is consistent with the arms' deliberately different earlier
edited-feature history being consumed downstream. Local final-position
normalization alone does not determine the final logits.

## Collateral under actual routing

Compare with the **unscaled edited** baseline, separately from clean. Both
targets preserve all 11 other canonical outputs at every tested scale. This
does not make those outputs all correct: when scaling Dollar, Euro was already
a failed baseline read.

- Euro at 4× changes 1/12 unrelated outputs; at 8× it also changes 2/12 near-name
  outputs. The 20 unseen-entity currency outputs remain as in the unscaled edit.
- Dollar at 4× changes 1/20 unseen-entity currency outputs; at 8× it changes
  3/20. Near-name and unrelated outputs remain as in the unscaled edit there.
- The baseline edit already has collateral: for these banks it retains only
  3/12 near-name outputs, 19/20 unseen-entity currency outputs and 11/12 unrelated
  outputs against clean. Stability relative to that baseline is not safety or
  clean-model retention.

These small fixed banks measure unintended changes, not population error rates.
No per-prompt strength selection or post-result extension of the ladder occurred.

## Verification and provenance

All 168 actual-baseline prompts match factorial-v1's saved top-five tokens and
log probabilities exactly. Every intervention preserves all 10,228 other native
feature activations at every token position. All 828 ladder forwards preserve
the captured gate/up/raw activation values. Restoring the original written
weights reproduces full actual logits exactly on all 84 labeled prompts.

The audit verifies 1,500 rows, coverage, selected records, source/fixture hashes,
all normalization finite differences from saved vectors, and all routing/ladder
summaries. Ten model-free instrument tests pass. Checkpoint files and previous
experiment artifacts were not modified.

The first attempt (`oracle-v1`) stopped before producing a scored row because
the installed MLX lacked `ArrayAt.set`. The replacement constructs a new
edited-feature tail and concatenates the unchanged native prefix. The failed
attempt's metadata and failure record are preserved. Protocol and fixture did
not change. The completed artifact is **oracle-v2**.

## Decision

Concentrate on reproducing the oracle's address/activation delivery with a real
detector, while retaining explicit value-calibration and downstream-context
checks. This experiment gives that work a stronger basis: most failures recover
without a new value direction. It does not remove the four unresolved language
failures or license a usable memory-system claim. Native composition stays
closed; bundles, capacity sweeps, HNSW and speed were not run.

Artifacts: [summary](results/oracle-v2/summary.json),
[all scored rows](results/oracle-v2/rows.jsonl),
[fixture](results/oracle-v2/fixture.json),
[metadata](results/oracle-v2/metadata.json).
Full pre/post-normalization vectors are retained in the ignored
`results/oracle-v2/normalization_vectors.npz`.
