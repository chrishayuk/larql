# Bounded native gate/up fit, v1

Frozen before execution, 2026-09-05. Same original unquantized Gemma 3 4B IT
bf16 checkpoint, L26, twelve factorial records and original 1× down columns.
Fit only the twelve edited gate rows and twelve edited up rows (61,440 scalars).
Keep GELU-approx, multiplication, the dense FFN, all other weights and the
downstream computation unchanged. No added bias, classifier, runtime labels,
gain knob, activation mask or earlier-position suppression in the fitted arm.
No value repair, capacity/feature bundle, composition or speed experiment.

## Data boundary

The observed factorial bank (168 prompts) supplies development data. Training
uses canonical prompts, its first three sentence forms and first alias form;
validation uses its fourth sentence and second alias forms. Negatives are
partitioned by deterministic index modulo three (one third validation). This
is 116 training rows (60 positive, 56 negative), 52 validation (24 positive,
28 negative). All intended positive activations have only one nonzero slot;
every installed slot is targeted to zero on negative queries.

Fresh evaluation has 168 rows per bank: 12 canonical installation checks,
48 sentences in four unused structures, 24 aliases in two unused structures
with new alias names, 24 near-name controls, 36 uninstalled-entity queries,
12 uninstalled relations on installed entities, and 12 unrelated prompts.
Canonical checks are not held-out generalization. Templates, names and splits
are fixed in `controller_fixture.py` before development fitting.

Repeat installation with four fresh entities using the same value assignments
and translated development templates. Fit their addresses with the selected
procedure, but do not use their validation/evaluation outcomes to select a
setting, checkpoint or method. Report the two rosters separately. Never pool
the observed development rows into a fresh-performance claim.

## Fixed fit and selection budget

Inputs are captured actual final-position L26 MLP inputs. They are independent
of the edited L26 rows; full forwards verify that independence. The observed
roster reuses `factorial-v1/addresses.npz`; the fresh roster captures its own.
There is no final-answer/logit loss. Targets are normalized by each record's
canonical activation under its original installation, giving a one-hot target
for positives and zero for negatives.

Minimize squared error summed over all twelve activation ratios, weighted
equally between positive and negative examples, plus a row-relative L2 penalty
from the original editor. Both gate and up are optimized in f32 with native
GELU-approx; each row is parameterized in units of its initial L2 norm. Adam
uses beta1=.9, beta2=.999, epsilon=1e-8 and global gradient clipping at norm 1.
No bias, centering, input normalization or change to deployed architecture.

Exactly four settings: learning rates {0.0003, 0.001} × regularization
{0.0001, 0.01}. Each runs 1,600 full-batch steps from the same original rows.
Checkpoints at 400/800/1200/1600 are scored using bf16 rows and arithmetic.
Choose the smallest balanced validation activation loss, breaking ties by
candidate order then earlier step. Persist this frozen choice before the first
fresh evaluation forward. The fresh roster uses that exact setting and step
count, irrespective of its validation results. Save all training traces and
the installed rows. No retries or new hyperparameters after evaluation.

## Comparisons and value swap

For each roster, score all fresh rows under clean Gemma, the existing editor,
the fitted rows, and C+A on the existing editor. C+A on a negative query zeros
all final edited activations; otherwise it uses the known intended label and
canonical activation. It never suppresses earlier positions. It is a separate
artificial reference, not the fitted implementation or an absolute ceiling.
Fitting gate/up rows can itself change earlier activations, unlike that oracle.

Then replace only record zero's down column (Oslo → Berlin) in the fitted arm,
score all rows without retraining, restore the fitted write, and finally
remove all edits to restore clean Gemma. Both restoration passes score every
row and require exact full-logit equality with their corresponding references.
Replacement must leave gate/up scores and the full edited activation sequence
exact. All phases preserve the captured final MLP input and nonedited native
feature activations. Total: 1,176 scored forwards per roster, 2,352 overall,
plus bounded installation/development capture forwards.

## Separate measurements and gate

Report full-vocabulary target accuracy and margin independently from record
selection and activation delivery. Selection is argmax of activation/canonical
activation, with rejection if the maximum is <0.5. This is a diagnostic only,
never a deployed routing rule. Delivery passes if the intended ratio is within
0.2 of 1 and every competitor has absolute ratio <=0.1. Negative delivery
passes if all absolute ratios are <=0.1. Save ratios and raw gate/up scores.

A predeclared selective-controller gate, assessed separately for each roster:

- On canonical, sentence and alias positives: >=90% correct diagnostic
  selection and >=80% delivery-pattern passes in each group.
- On fresh sentences and aliases: recover >=90% of the C+A-correct prompts in
  each group, while losing no more than one existing-editor success per group.
- For each negative group: >=95% quiet-pattern passes and >=95% top-1 agreement
  with **clean Gemma**. Also report KL, total variation and margins; agreement
  is not a claim of zero distribution drift.
- After replacement: >=6/7 target-record answers become Berlin; >=95% top-1
  preservation on other installed bindings and on negatives relative to the
  fitted arm. Canonical B's capital and A's currency top-1 must be preserved.
- Exact replacement detector invariance, write restoration and clean removal.

This gate intentionally does not require repairing every residual oracle
failure. Failure means this fixed fit/parameterization did not achieve the
tested controller behavior, not that binding information is absent or a larger
controller must succeed. No composition is run even if this gate passes.

## Instrument and provenance

Source, protocol, fixture and prior-artifact hashes accompany the run. Original
and fitted row arrays are persisted, with original down-column equality checked
exactly. Other model parameter objects remain unchanged (MLX arrays are
immutable); nonedited gate/up rows remain bit-identical. Full model forwards,
not a standalone reader, determine reported answers and delivered activations.
The checkpoint, prior experiments and runtime crates are not modified.
