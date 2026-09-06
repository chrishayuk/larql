# Native record entry gate: wording generalizes, binding selectivity fails

Completed 2026-09-05. Gemma 3 4B IT, bf16, original chuk-lazarus/native.py
recipe. 114 frozen prompts × 8 weight states = 912 evaluations.
**Do not advance this recipe to the native-composition claim.** The failure is
address discrimination; it is not inability to reproduce the write, generalize
the wording, change its value, or reverse the edit.

## Original write

| Measurement | Result |
|---|---:|
| Original installation reads | 3/3 |
| Held-out sentence reads | 12/12 |
| Explicit alias reads | 6/6 |
| Original retention, agreement with clean | 6/6 |
| Same entity, different relation, agreement with clean | **0/6** |
| Similar names, agreement with clean | **0/9** |
| Held-out entities, agreement with clean | 56/60 |
| Unrelated prompts, agreement with clean | 11/12 |

Clean answered 0/21 intended positive completions. After writing, the weakest
full-vocabulary answer margins were +4.125 on installation prompts, +5.000 on
sentence forms, and +5.625 on aliases. These are confident positive reads, but
their apparent generality must be read together with the selectivity failures.

Each of the three independent value replacements also scored 3/3 installation,
12/12 sentence, and 6/6 alias reads, including preservation of the other two
records' intended reads. Every replacement still changed all six wrong-relation
controls and all nine similar-name controls.

## The binding failure is visible in the counter-swaps

| Prompt | Original three writes | Replace only Zelandia capital Oslo→Paris |
|---|---|---|
| The capital of Zelandia is | Oslo | Paris |
| The capital of Qtaria is | **Oslo** | **Paris** |
| The capital of Vornholt is | **Oslo** | **Paris** |
| The capital of Zelania is | **Oslo** | **Paris** |

All six cross-relation prompts emit the installed value for the requested
relation, even though that value was bound to another entity. Replacing only
Qtaria's currency Yen→Euro makes both Zelandia's and Vornholt's currency reads
follow Euro too. All nine similar-name controls emit the corresponding
installed relation value and follow its replacement.

This supports a narrower interpretation: on this bank, the slots behave as
broad relation-sensitive detectors among these novel/similar entities. They do
not yet establish an isolated entity–relation record. Generalization success
alone cannot distinguish the two interpretations.

The broader controls expose changes missed by the six original retention
prompts. For example, `The language of Russia is` changes from `Russian` to
`Welsh`. Other changes include Poland/Hungary language completions and China's
currency completion. The unrelated change is `A thermometer measures`, from
`the` to `temperature`: an unintended distribution change, not necessarily a
factual error. Retention in this experiment means clean-output agreement.

## Reversibility and instrument checks

- Clean repeat: identical full logits on all 114 prompts.
- Remove all writes: identical to clean on all 114 prompts.
- Restore writes: identical to the first written state on all 114 prompts.
- Four instrument unit tests pass: vocabulary-wide competition, probability
  invariance to logit offsets, evaluation separation, and category-wise gates.
- `audit_run.py` verifies source/protocol/fixture/harness hashes, complete row
  coverage, and recomputes all summaries and gate decisions from the 912 rows.
- Original checkpoint files were read only; edits lived in model memory and
  original array objects were restored in `finally`.

The 95% per-control-category threshold was frozen before the run. Several
categories fail it, but the decisive 0/6 and 0/9 results do not depend on a
borderline threshold choice. No error bars or population-level claims are
warranted by three records and a small fixed template bank. Aliases include
explicit identity declarations, and the metric is next-token completion,
not arbitrary question answering or generation.

## Recovered FHG-2 result

The completed 12B artifact contains 240 cells (4 items × 4 queries × 15 arms).
At absent first edge, supplying the correct intermediate reaches C in 16/16;
supplying the corrupt intermediate reaches C′ in 16/16. However, absent edge
with no prosthesis already chooses C in 11/16 and C′ in 4/16. A sham chooses C
in 1/16, C′ in 4/16, and other tokens in 11/16. Thus the counter-swap is strong,
but necessity is not established by all those ablation cells. This is textual
injection on 12B, separate from the 4B native FFN result.

Audit source:
`/Users/christopherhay/chris-source/chris-experiments/fhg/results/fhg2_intermediate_prosthesis.json`
SHA-256 `c309b5f07411b099f34017100518d390485b4503196eaf2f6a5999bf4934358d`.

## What follows

Experiment 1 is complete and fails its selective-record advancement gate.
Experiment 2 is not run: a downstream answer following these slots could still
be a broad relation-value effect. Experiments 3 and 4 remain separately scoped
in [PROTOCOL.md](PROTOCOL.md); no graph-causality or speed result is claimed.

The next native-write question is whether an address can distinguish entities
within the same relation, including the other installed entities and near-name
neighbors, while preserving these held-out reads. That is a new editing-method
experiment and requires a fresh evaluation bank; this observed bank is now
diagnostic data. No capacity or HNSW sweep was started.

Artifacts: [summary](results/address-v1/summary.json),
[per-probe rows](results/address-v1/rows.jsonl),
[fixture](results/address-v1/fixture.json),
[provenance](results/address-v1/metadata.json).
