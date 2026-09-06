# Oracle-component decomposition: amplitude dominates, competition finishes recovery

Completed 2026-09-05 under the [frozen protocol](DECOMPOSITION_PROTOCOL.md).
Artifact: `results/decomposition-v1/`. Same original factorial editor, Gemma 3
4B IT bf16 snapshot, L26, 12 slots and **1× down-column strength** as the
[previous oracle experiment](ORACLE_RESULTS.md). These are the existing 84
positive prompts, reused for diagnosis, not a fresh generalization test.

Replacing the intended final activation alone recovers 36 of the full oracle's
41 recovered prompts. Adding final-position competitor suppression recovers the
remaining five. That combination, without earlier-position suppression, has
the **same top-1 token on every prompt** as the full oracle. Earlier suppression
changes downstream answers/margins, but recovers no additional correct reads
at any of the four settings of the other interventions.

## All eight arms

E suppresses edited slots before the final position; C suppresses the other
eleven slots at the final position; A supplies the intended slot's canonical
activation. Codes below are ECA. All other native contributions and the actual
normalization/downstream computation are preserved.

| ECA | Interventions | Canonical /12 | Sentence /48 | Alias /24 | Total /84 | Recovered / lost vs 000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 000 | Actual | 11 | 18 | 9 | 38 | 0 / 0 |
| 001 | A | 11 | 42 | 21 | 74 | 36 / 0 |
| 010 | C | 11 | 20 | 11 | 42 | 5 / 1 |
| 011 | C+A | 11 | 44 | 24 | 79 | 41 / 0 |
| 100 | E | 11 | 18 | 8 | 37 | 0 / 1 |
| 101 | E+A | 11 | 42 | 20 | 73 | 35 / 0 |
| 110 | E+C | 11 | 20 | 11 | 42 | 5 / 1 |
| 111 | Full oracle | 11 | 44 | 24 | 79 | 41 / 0 |

A alone recovers 24/30 failed sentence reads, versus 26/30 for C+A and the full
oracle; neither loses a previously successful sentence read. C adds two
sentence and three alias successes when A is already enabled.

Among the 41 prompts recovered by the full oracle, the observed minimal
sufficient intervention subsets are:

- 31: A only.
- 5: either A alone or C alone.
- 5: C+A jointly; neither alone suffices, and adding E to either alone does
  not suffice.

These are minimal subsets of the tested interventions, not a unique allocation
of causes. The five joint cases are Avenlorn's capital alias (`alias-005`), its
currency sentence (`sentence-009`), and Dornessa's capital sentence/aliases
(`sentence-076`, `alias-077`, `alias-078`).

## Conditional effects and interactions

Across all four backgrounds, adding A recovers 36 or 37 prompts and loses none.
Adding C recovers 5 or 6, with one lost success only when added to 000. Adding E
recovers none: it loses one alias at 000 and at A, and changes no hit labels at
C or C+A. The C+A and full-oracle top-1 tokens agree on all 84 prompts, but
their logits are not identical.

For completeness, exact baseline-anchored inclusion–exclusion contrasts are
below. A hit contrast is the sum across 84 prompts, not a count of uniquely
attributable recoveries; margin contrasts are means in final-logit units.

| Term | Hit contrast | Mean answer-margin contrast |
| --- | ---: | ---: |
| E | -1 | -0.409970 |
| C | +4 | +0.316964 |
| A | +36 | +6.047619 |
| E:C | +1 | -0.146577 |
| E:A | 0 | +0.215774 |
| C:A | +1 | +0.374256 |
| E:C:A | 0 | +0.081845 |

Their sums give the full-oracle change: +41 hits and +6.479911 mean margin.
Small aggregate interaction contrasts do not erase the five individual joint
requirements. `analysis.json` retains group and prompt-level contrasts,
conditional effects and minimal subsets. These are descriptive finite-bank
results; no independence assumptions, p-values or population estimates.

## Earlier states matter, but suppressing them does not explain recovery

Every E-only comparison has bit-identical final-position vectors both before
and after the actual post-FFN normalization: **336/336 paired checks**.
Nevertheless, final answer margins change in 296/336 pairs and top-1 tokens in
5/336. Of those five token changes, two lose correct answers; the other three
switch between incorrect answers. None gains a correct answer.

| E-only comparison | Changed margins /84 | Changed top-1 /84 | Mean margin change |
| --- | ---: | ---: | ---: |
| Actual → E | 74 | 3 | -0.409970 |
| A → E+A | 72 | 1 | -0.194196 |
| C → E+C | 76 | 1 | -0.556548 |
| C+A → E+C+A | 74 | 0 | -0.258929 |

This directly isolates sensitivity to the altered earlier-position states
through the remaining computation, despite identical local final-position
outputs. Downstream attention is the cross-position path in this architecture;
this experiment does not localize a particular later layer or attention head.
Mean margins decline at every background, although individual margin effects
can differ. Earlier writes are therefore not harmless, but suppressing them is
not the recovery mechanism on this bank.

## Remaining failures and scope

C+A and the full oracle leave exactly the same five failures:

| Prompt | Target | C+A margin | Full-oracle margin |
| --- | --- | ---: | ---: |
| The currency of Avenlorn is | Euro | -1.25 | -1.625 |
| If one asks about the language of Avenlorn, the answer is | German | -0.25 | -0.375 |
| Looking up Avenlorn's language, one finds | German | 0.0 | 0.0 |
| Looking up Braskovia's language, one finds | Spanish | -0.125 | -0.125 |
| Looking up Dornessa's language, one finds | French | -0.5 | -0.75 |

The zero-margin German case loses the top-1 tie. The prior Euro 4× result is a
separate value-strength diagnostic; no down-column scale changes occur here.

The controller target is now better constrained: infer the intended record
from native state, deliver a suitable final activation, and reject competing
records. Amplitude calibration deserves priority within that problem; a timing
controller is not supported as the first recovery remedy by these contrasts.
However, A still uses the externally known record label and its stored
canonical activation. It is not a universal gain adjustment, learned detector,
or proof that the model can decide when/where a write should execute.

No new negative/collateral bank, generation test or controller was evaluated.
Native composition remains closed; bundles, capacity and HNSW remain deferred.

## Verification and artifacts

Completed 504 new scored combinations plus 336 endpoint-verification forwards:
840 total, comprising the 672-cell cube and 168 closing repeats. Both endpoints
match the prior oracle artifact's top-five scores, answer margins and local
normalization vectors exactly. Closing repeats reproduce opening full logits
exactly for all 168 endpoint cases.

All nonedited 10,228 activations remain bit-exact at every token. Raw edited
activation sequences and final gate/up scores match the actual baseline across
arms. An independent NumPy transformation verifies each intervention's scope
against captured activation sequences. The artifact audit passes, as do all
12 instrument unit tests. Source/protocol/fixture hashes are recorded.

`rows.jsonl`, `analysis.json`, `earlier_position_pairs.json` and `summary.json`
retain scores and analyses. The ignored `vectors.npz` retains every row's
pre/post normalization vector and raw/intervened edited activation sequence;
keep it with the JSON artifacts for the vector-level audit.

All additions are in the separate `exp/native-record-composition` worktree.
No checkpoint, base vindex, runtime crate or earlier experiment artifact was
modified. Reproduction commands are in the [README](README.md).
