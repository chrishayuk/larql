# Three-component oracle decomposition, v1

Frozen before execution, 2026-09-05. Original factorial editor and the existing
84 positive prompts, Gemma 3 4B IT bf16, L26, original down-column scales.
Diagnostic reuse only. No reader/controller training, strength sweep or
composition. The four language failures remain open unless an observed arm
changes their outcome; no extra interventions are added to chase them.

## Complete 2×2×2 design

Codes are **E C A**, in this order, each zero or one:

- E: set all 12 edited-slot activations to zero at positions before the last.
- C: set the other 11 edited-slot activations to zero at the last position.
  Preserve the intended slot's actual activation unless A is also enabled.
- A: replace only the intended final activation with that record's canonical
  installation activation. Preserve competitors unless C is also enabled.

All other activations, weights, positions, normalization and downstream
computation follow the actual edited model. Record label and canonical
activation are supplied by the experiment. 000 must reproduce actual routing;
111 must reproduce the previous oracle.

Run 000 and 111 first, verifying all 84 against oracle-v2's persisted top-five
tokens/log probabilities, answer margins and final-position pre/post norm
vectors. Run the six missing combinations (001, 010, 011, 100, 101, 110):
**504 new scored cells**. Finally repeat 000 and 111 on all 84 prompts and require
full-logit equality with the opening runs. Total: 840 scored forwards, comprising
672 factorial cells and 168 final verification repeats; 336 endpoint forwards
are verification of previously observed conditions.

## Measurements and analysis

Persist full-vocabulary accuracy, target probability/rank/margin, raw gate/up,
before/after edited activations at **every** position, native-feature preservation,
and actual pre/post FFN normalization vectors. The original model runs normally
after the intervention. Never replace residuals or final logits.

Report all eight arms by canonical/sentence/alias group, paired recoveries and
lost successes relative to 000, and conditional effects of each factor at each
setting of the other two. Compute exact 0/1 inclusion–exclusion coefficients
for margins and hits: three singles, three pairs, one triple. Save prompt-level
coefficients and group means; interaction terms are not independent error.
No inferential p-values, population claims, or unique allocation of joint gains.

For each 000-failed / 111-recovered prompt, identify the minimal sufficient
subsets among observed arms. Multiple minimal subsets may exist and are
reported together, not double-counted as exclusive causes.

When only E changes, verify final-position pre/post normalization vectors remain
exactly equal for every setting of C/A. Report final-output flips and margin
changes on those pairs. This tests whether changed earlier positions matter
through the remaining model despite identical local final-position outputs.

## Instrument gates

Nonedited 10,228 features must stay bit-exact at every token. Assert each flag's
scope against an independent NumPy transformation of the captured edited tail.
Raw gate/up/activation must be identical across all arms for each prompt.
000/111 must match the prior artifact as above and reproduce full logits on the
closing repeats. Failed gates invalidate the interpretation; preserve failed
attempts and do not relax thresholds. Source/protocol/fixture hashes and all
scored rows are retained; vectors and activation sequences are saved in an
ignored NPZ. No checkpoint files or prior artifacts are modified.
