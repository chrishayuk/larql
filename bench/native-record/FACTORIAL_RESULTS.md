# Factorial write: canonical separation improves, the record gate still fails

2026-09-05, same Gemma 3 4B IT bf16 checkpoint, layer 26. Four fresh entities ×
three relations, 12 edited slots. Each relation has four different values and
each entity has all three relations. The fixture and methods were frozen before
the preceding discrimination run. No method changes followed its results.

**Neither editor passes. Native composition remains gated closed.**

## Comparison

The original recipe already subtracts every other installed address. With a
factorial pack, this includes the relevant canonical binding negatives. The
changed method adds ten negative addresses per slot: the other entities under
the same relation and the same entity under the other relations, each in two
development sentence forms. All positive addresses, scales, values, layer and
slot counts are identical across methods. Both use the original 30 decoys and
11 other canonical record addresses; the candidate adds only these negatives.

| Measurement, original written state | Original factorial recipe | Extra binding negatives |
|---|---:|---:|
| Installation reads | 11/12 | 11/12 |
| Fresh sentence reads | 18/48 | 15/48 |
| Explicit alias reads | 9/24 | 10/24 |
| Near-name agreement with clean | 3/12 | 6/12 |
| Unseen-entity agreement with clean | 36/60 | 36/60 |
| Unrelated agreement with clean | 11/12 | 11/12 |

Clean scores 0/84 on the intended record values. All answers are scored against
the full vocabulary, with single-token targets checked before installation.

The extra negatives cut some collateral and reduce average distribution drift,
but do not create reliable held-out reads. Mean KL(clean || written) on unseen
entities decreases from 1.751 to 1.233 nats while the clean-argmax retention
count stays 36/60. Near-name retention doubles but remains only 50%. The
sentence-read cost means this is a tradeoff within a failed regime.

## Decisive counter-swap, including its failed prerequisite

Both methods show the same canonical behavior:

| Prompt | Written | Replace only Avenlorn capital |
|---|---|---|
| The capital of Avenlorn is | Oslo | Berlin |
| The capital of Braskovia is | Paris | Paris |
| The currency of Avenlorn is | **the (Euro required)** | **the (Euro required)** |

The cross-entity capital separation is real at these canonical addresses.
However, preserving an already-failed currency read does not pass the requested
conjunction. Avenlorn's currency is the sole canonical failure under both
editors and both value states. The other ten non-replaced canonical reads are
correct. Changing the capital is not what caused the currency failure.

The original editor's canonical margins are +6.375 → +6.250 for Avenlorn
Oslo→Berlin, +6.500 → +6.500 for Braskovia Paris, and −1.250 → −1.125 for
Avenlorn Euro. Candidate margins are +6.125 → +5.875, +6.500 → +6.500, and
−1.375 → −1.500 respectively. Raw rows retain the complete scoring context.

After replacement, original sentence/alias hits are 18/48 and 10/24; candidate
hits are 15/48 and 10/24. Neither replacement state repairs the gate. Candidate
unrelated retention declines to 10/12 in its replacement state.

## Detector confusion is not the whole failure

Post-hoc diagnostic: rank the twelve edited slots by actual pre-normalization
contribution magnitude at the answer position. This excludes the other native
FFN features and is not a causal decomposition of the final answer.

| Sentence queries, written state | Original | Extra negatives |
|---|---:|---:|
| Intended slot largest, answer correct | 17 | 15 |
| Intended slot largest, answer wrong | 7 | 16 |
| Other edited slot largest, answer wrong | 23 | 17 |
| Other edited slot largest, answer correct | 1 | 0 |

Additional negatives reduce cases where another edited slot dominates, but
increase failures where the intended slot is largest. On the failed canonical
Euro query, its intended slot has activation 245 with the original editor;
the other edited slots are nearly silent. This observes correct slot isolation
without a successful output read. It leaves value strength/direction and the
downstream interaction unresolved; it does not prove that a larger bundle is
the necessary remedy.

Both mechanisms therefore matter in the observed regime: cross-form detector
confusion and failure to realize the desired answer even with the intended
edited slot dominant. Canonical isolation is feasible, but a robust editable
entity–relation record has not been established.

## Scope and artifacts

168 frozen probes × 9 weight states = 1,512 scored forwards. The two methods
each run write, value-only replacement, removal and restoration, against a
shared clean baseline. Every row includes actual gate/up/activation values,
contribution magnitudes, full-vocabulary target score and clean-distribution
comparison. Full MLP inputs are stored in the ignored `addresses.npz`.

Removal returns exact clean full logits and restoration returns exact written
full logits on all 168 prompts for each method. Gate, up and activation values
are identical between written/replaced states on every prompt, confirming the
value-only intervention at the captured slot interface. The completed artifact
audit verifies 1,512 rows, hashes, target replacement scope, summaries and every
gate. The eight model-free instrument tests also pass. Original address-v1
artifacts retain their original source/protocol hashes and still pass audit.

One 12-record pack, one site and one fixed force profile do not establish a
capacity limit or absence of binding information. No single-slot force sweep,
feature bundle, composition experiment, graph-causality run or speed run was
added after observing the failures. A future experiment should distinguish
detector generalization from value realization, rather than assuming extra
slots repair both.

See also [the discrimination diagnostic](DISCRIMINATION_RESULTS.md) and
[the frozen protocol](DISCRIMINATION_PROTOCOL.md).

Artifacts: [summary and gates](results/factorial-v1/summary.json),
[per-probe rows](results/factorial-v1/rows.jsonl),
[12-record fixture](results/factorial-v1/fixture.json),
[installation details](results/factorial-v1/installation.json),
[provenance](results/factorial-v1/metadata.json).
