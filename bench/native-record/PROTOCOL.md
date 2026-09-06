# Native FFN record programme

Frozen before the first new model run, 2026-09-05. Start at experiment 1;
experiment 2 depends on its address gate. Experiments 3 and 4 are separate
claims, specified below but not implied by a native-write success.

## Recovered evidence

- `vendor/native.py` is verbatim `the-mechanism` commit
  `3ee95817324e047b5f8f5b587746de2c92051f21:native.py`. The local checkout
  exactly matches the linked source. Original result: 3/3 installation reads,
  6/6 clean-argmax agreement; not a paraphrase or composition result.
- Original model: `google/gemma-3-4b-it`, bf16, chuk-lazarus loader, layer 26,
  gate multiplier 30, value multiplier 0.1, final three MLP slots. Preserve
  the three facts, Gram–Schmidt order, six generic + 24 entity/relation decoys,
  and median row/column scales. Never use evaluation prompts as decoys.
- `chris-experiments/representation_taxonomy/CAP1_VERDICT.md` and
  `cal1_format_calibration.json` contain completed collateral/format studies.
  Expanded retention found damage missed by small probes; format mismatch
  can increase it. The original script's estimated capacity is not a result.
- `chris-experiments/fhg/results/fhg2_intermediate_prosthesis.json` exists:
  12B, textual intermediate injection, reported counter-swap rate 1.0.
  Its absent-edge controls sometimes already choose a destination. Audit
  those separately from the counter-swap; do not inherit necessity or a
  native hidden-intermediate claim.

## 1. Address generalization (run first)

One installation wording per original fact. Freeze all edited weights before
evaluation. Predeclare the entire fixture and save its digest before loading
the model. No template, key, scale, layer, or decoy tuning after observation.

Positive evaluations: four held-out sentence forms and two explicit alias
declarations per fact. Aliases are introduced as aliases in the prompt;
invented synonyms without an identity link would test an impossible binding.
Controls: same entity/different relation, similar names, 20 held-out entities
per relation, unrelated prompts, and the six original retention prompts.
Control agreement means agreement with clean, not factual correctness.

Arms: clean, clean repeat, original three writes, independently replace each
record's value while retaining its key and all other edits, remove all writes,
restore all writes. Restore the original MLX array objects in `finally` too.
All replacement states start from the original written state. Evaluate every
probe in every arm; store per-probe outcomes, full-vocabulary answer margins,
target probability and rank, top tokens, KL(clean || arm), total variation,
and gate/up projections at the actual queried address. No candidate-only
accuracy. Answers must be one token under the original tokenizer convention;
abort on incompatible tokens rather than silently truncate. This is a
next-token completion test, not a free-generation accuracy claim.

Advancement rules (engineering gates, not population estimates):

- reproduce 3/3 installation reads and 6/6 original retention;
- original and each replacement: at least 80% of sentence forms and 80% of
  explicit aliases correct, each fact at least 75% on sentence forms;
- at least 95% clean-argmax agreement in **each** held-out control category
  for original and replacement edits (not just pooled agreement);
- clean repeat/removal and written restoration reproduce full logits exactly.

Failure stops native composition promotion. Report which component failed,
including per-record/per-form failures and collateral. No capacity sweep or
editing-method revision is folded into this experiment. A later revised edit
needs a fresh evaluation bank.

## 2. Native value as intermediate (conditional main target)

Keep 4B and its edit recipe; the recovered 12B FHG result is a separate positive
control. Use at least four seeded randomized A→B→C / B′→C′ graphs and four query
forms. C/C′ must never occur in installation prompts or written value vectors.
First establish full-vocabulary downstream competence on both explicit paths
and correct/corrupt explicit intermediate controls for every retained item.
Keep failed-competence items in the artifact and report coverage.

Pair write A→B, value-only replacement A→B′, removal, restoration, and a
same-layer/same-scale irrelevant entity edit. Supply both downstream bindings
in the query context, with their ordering counterbalanced. Score internal
composition without emitting B separately from emitted/supplied-B controls.
Both C and C′ must follow the swap, removal must reduce their relevant support,
restoration recover, and irrelevant edits stay below the swap effect. Persist
direct first-hop reads and downstream positive controls to distinguish address
failure, interface failure, and downstream failure. No success headline from a
single arm or candidate-restricted score.

## 3. Native feature graph (independent causal experiment)

Propose a small frozen edge set from down-to-next-gate similarity on discovery
prompts. On disjoint prompts compare source suppression, exact contribution
restoration, and restoration with destination suppression against controls
matched on activation and contribution norm. Include predeclared small bundles.
Capture before/after the intervening attention and normalization, and measure
destination activation plus downstream behavior. Advance only if the proposed
edges outperform matched controls across prompts. Otherwise edges remain
descriptive. Selection and evaluation must not share prompts.

## 4. Sparse selection (one bounded feasibility test)

Use the original R4 4B checkpoint for historical comparison and label backend
differences. Compare gate, activation, contribution, cell-pool, and an expensive
selector minimizing the norm of the **summed omitted signed output vectors**.
Freeze K = 1024, 2048, 2816 (below the historical ~2944 crossing), 4096 control;
report selection cost separately without claiming it is executable cheaply.
Test composed layer substitutions and continuations, with per-prompt output
agreement and KL, not only single-layer residual norms. Freeze tolerances and
the prompt bank before execution. Measure today's dense execution budget with
warmed baseline/candidate/baseline brackets; require peer-session handshake,
≤1% bracket drift and sufficient numerical precision. No speed claims from
the old fitted crossing or overlapping workloads.

HNSW is conditional on a feasible selection result: useful-feature recall,
signed scores, connectivity, index bytes, and total routing+execution latency.
Do not implement an index before the selection gate passes.
