# Oracle routing and value strength diagnostic, v1

Frozen before execution, 2026-09-05. Diagnostic reuse of the existing factorial
bank; no new generalization claim. Same original 12-record editor, 4B bf16
checkpoint, L26. The extra-negatives editor is not swept again.

## Routing arms

Score all 84 labeled prompts (12 canonical, 48 sentence, 24 explicit alias):

- Actual: unchanged edited forward, reproduce factorial-v1 exactly.
- Zero: zero only the 12 edited feature activations at every token position.
- Oracle: zero those 12 features at every position, then set the intended
  feature at the last position to its measured canonical installation activation.
- Wrong: same procedure with the next entity's record under the same relation,
  using that wrong record's own canonical activation. Values are distinct.
- Wrong matched: wrong record but the intended record's canonical activation,
  to distinguish value direction from canonical activation magnitude.

Oracle labels are supplied by the experiment; this is not a usable router or a
memory-system success. Other 10,228 native feature activations remain bit-exact
at **all** positions. No residual or answer token is directly injected. The real
down projection, actual post-FFN normalization, residual addition, attention,
and remaining layers execute normally. Earlier edited-slot writes are silenced
so they do not remain an uncontrolled history in the oracle/wrong/zero arms.

Persist raw actual/intervened edited activations, selected record and amplitude,
target and wrong-value full-vocabulary margins and ranks. Capture whole FFN
output and post-feedforward normalization output at the final position. Compare
each intervention against the zero arm at the same weight state: pre/post norm
vector deltas, norms, direction cosine, and contribution norm / whole FFN norm.
These are finite differences, not an additive attribution through normalization.
Also compare final logits and margin changes against zero.

Interpretation: report how many of the 30 actual failed sentence reads recover
under oracle, and how many actual successes are lost. "Most recover" means
strictly more than half; ≥80% is a separate strong-recovery descriptor. Report
wrong-value hits and paired correct→wrong redirection. No outcome reopens native
composition: routing is externally supplied.

## Fixed strength ladder

Use original failed Avenlorn currency→Euro (record 1) and successful same-relation
Braskovia currency→Dollar (record 4) as the preselected control. Multipliers
[0, 0.5, 1, 2, 4, 8] relative to the original down column, always starting from
the same original written weights. Gate/up weights and canonical oracle
activations stay frozen. Change only that one down column, not its activation.

For each record/multiplier, evaluate its seven labeled forms under actual and
oracle routing. Actual-routing collateral bank: all 11 other canonical records,
12 near-name controls, 20 unseen-entity currency prompts, and 12 unrelated
prompts. Record effects on other bindings against the **unscaled edited**
baseline as well as against clean. Near-name/unseen prompts have no oracle
label and are never counted as oracle successes. This is 828 ladder forwards,
plus zero references and actual-baseline checks. Never extend the ladder in
response to its results.

Report per-context response curves, successful control, collateral, and any
saturation. Strength sufficiency on this bank requires one shared tested scale
with ≥80% oracle reads for both selected records, not per-prompt scale choices.
Passing this artificial condition is not a deployable editor or a validation of
new entities. Actual-routing collateral is reported even when oracle improves.

## Instrument gates

Actual arm must match all persisted factorial-v1 top-five tokens/log probabilities
on the reused probes. Routing interventions must leave all nonedited feature
activations exactly equal to their original values at all token positions.
Strength changes must leave captured gate/up/raw activation exactly invariant.
After the ladder, restore baseline weights and reproduce all 84 labeled actual
full logits exactly. Fail these gates before drawing a mechanism conclusion.
Persist vectors in ignored NPZ, all scalar rows/metrics and source hashes in JSON.
Composition, bundles, capacity, graph causality, HNSW and speed remain closed or
outside this experiment.
