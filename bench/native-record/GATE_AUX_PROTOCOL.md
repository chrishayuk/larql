# Cached rejection diagnostic and one gate-loss correction

Frozen 2026-09-05 before execution. No model loading, checkpoint access, new
prompt capture or full-model forward in this experiment. Only the cached
development inputs and row arrays from `controller-v2` are read. Its former
fresh evaluation inputs/scores are not used. Both development rosters have
already been observed; neither is a new held-out evaluation claim.

## Rejection diagnostic

Use bf16 projected gate/up scores and native approximate-GELU activations,
normalized by each record's stored canonical activation. Report distributions
(min, quartiles, median, max) of intended activation, maximum absolute
competitor activation, positive peak score and negative peak/absolute score.
Break negatives down by category and by relation where available.

The only hypothetical rejection rule is: select the maximum signed activation
ratio if it is >= t; otherwise reject. The deployed native forward is not
changed. Enumerate all distinct training peak scores, plus finite endpoints
that accept or reject everything. Choose the threshold maximizing correctly
selected positive coverage subject to <=5% false acceptance in **each**
training negative category; break ties by fewer false acceptances then larger
threshold. Report total positive acceptance, correct coverage, and negative
false acceptance on both training and validation at the frozen threshold.
Reject-all has zero useful coverage. Also retain the full curve and the best
validation-in-sample threshold as an explicitly optimistic diagnostic, not a
selected rule. This tests this scalar-score family, not all possible readers.

## One fitting change

Start each fit from the original editor's stored rows, not the previous fitted
solution. Same architecture, twelve slots, canonical targets, fixed data split,
balanced activation loss, row-relative L2, normalization-free parameterization,
Adam settings and 1,600 steps as the selected previous procedure (lr .001,
regularization .0001). Only add:

`lambda * mean_positive(max(1 - intended_gate_score, 0)^2)`.

The auxiliary term is used only on training positive examples, has a gradient
even when their GELU output is zero, and disappears at inference. No auxiliary
negative/classification loss, initialization change, new input or extra slot.
Exactly three strengths: .0001, .001, .01. Only the final 1,600-step checkpoint
is eligible; steps 400/800/1200 are descriptive logs, not extra choices.

Run a lambda=0 control on each development roster first and require exported
bf16 rows to exactly match the preserved prior fit. Select the strength using
only observed-roster validation: prefer a candidate passing the native
development gate below, then lowest balanced activation-pattern loss, then
strength order. Freeze the choice before the second-roster auxiliary fit.
That repeat uses the same strength/step count with no selection on its scores.
Budget: two zero-aux controls, three candidate fits, one fixed-roster repeat:
9,600 cached optimization steps total, zero model forwards.

## Development permission gate

Preserve the previous delivery/selection thresholds, applied by category on
validation: >=90% correct selection and >=80% delivery on each of its sentence
and alias groups; >=95% quiet activations in each negative category. Delivery
means intended activation ratio within .2 of 1, competitors' absolute ratios
<=.1; quiet means all absolute ratios <=.1. Also require positive delivery and
negative quiet counts not to worsen versus the previous fit, with at least one
of these two counts improving. A threshold-assisted diagnostic cannot satisfy
this native gate. Both development rosters must pass to permit model evaluation.

Answer accuracy, clean-output agreement and value counter-swaps cannot be
measured from these cached final-position inputs. They are not relaxed or
replaced by the development gate: the previous full-model evaluation gates
remain required if a genuinely untouched bank is ever evaluated. If either
development roster fails, close this bounded fitting branch without another
full-model run. A richer controller/bundle is a new hypothesis, not a next step
authorized by failure.

## Known split limitation and audit

The previous modulo-three negative split makes validation near-name and
other-entity negatives exclusively capital queries; their training negatives
are currency/language queries. This is preserved, not silently repaired.
Report the category/relation counts and do not claim a balanced held-out
negative test. The data do not test new uninstalled relations in development.

Persist source/protocol hashes, exact cache keys accessed, prior cache hashes,
all traces, every final candidate's rows and bf16 gate/up/activation arrays,
threshold curves, distributions and native gates. The audit recomputes all
descriptive summaries and decisions from saved arrays. Existing fits, artifacts
and checkpoint files remain unchanged.
