# Discrimination diagnostic and factorial write, v1

Frozen before capture on 2026-09-05. Original address-v1 artifacts and protocol
remain unchanged. Same local 4B bf16 snapshot and loader, layer 26.

## Capture and reader diagnostic

Instrument the actual MLP input, actual gate/up projection outputs, and actual
input to down_proj (the post-GELU product). Record each edited slot's signed
activation and `abs(activation) * norm(down column)` contribution magnitude,
before Gemma's post-FFN normalization. Do not reconstruct the activation with
an assumed nonlinearity. Capture clean and written original-bank forwards;
verify instrumented outputs against address-v1 and verify removal exactly.
Persist full MLP input vectors in an ignored NPZ and scalar rows in JSON.

Original-bank analysis is diagnostic, not held-out evidence. For each original
slot compare intended reads against different-entity/same-relation,
same-entity/different-relation, similar-name and other controls. Report signed
activation distributions and AUROC, not just the winning slot.

Readers have canonical enrollment addresses for each binding, with no answers
in their prompts. Closed-set binding identification is a readout experiment,
not proof of a writable record. Use 4 entities × 3 relations in each split:

- Development: Merovia, Tarskeld, Pelvoria, Nuskara.
- Fresh diagnostic evaluation: Quenvar, Drelmora, Surneth, Velkora.
- Fresh factorial write evaluation: Avenlorn, Braskovia, Celdrune, Dornessa.

Each roster has its own one-canonical-example enrollment. Fresh enrollment
introduces new labels, not training queries; a multiclass probe cannot be
expected to name labels it has never been given. Learn a **pairwise binding
match** rule on development entities and apply it to new enrollment/query pairs.
No evaluation-query fitting or tuning is permitted.

Compare cosine nearest address with a ridge-regularized pairwise probe on
`[abs(unit(query)-unit(key)), unit(query)*unit(key)]`. Feature normalization is
fit on development training pairs only. Balance positive/negative pair weights.
Choose lambda from [0.01, 0.1, 1, 10] by development validation binding accuracy,
breaking ties toward stronger regularization. Fit once on training only; do not
refit using validation. Select open-set acceptance thresholds separately on
development validation and predeclared near-name negatives. Report false
acceptance on fresh near-name queries; closed-set accuracy alone is insufficient
to call a reader selective.

Development training forms: possessive and `For ENTITY, the RELATION is`.
Development validation forms: `In ENTITY...` and `...associated with ENTITY...`.
Fresh test structures: `With regard to ENTITY...`, `If one asks about the
RELATION of ENTITY...`, `Looking up ENTITY's RELATION...`, and an explicit alias
declaration. Exact prompt strings and all labels are serialized before loading.

Reader gate: at least 80% fresh binding accuracy and accepted-correct rate, at least 75% within every
relation, and at most 10% fresh near-name false acceptance. These are fixed-bank
engineering gates, not population estimates. Report components independently.
If neither reader passes, evaluate only the fixed additional layers
[22, 24, 28, 30], using the same splits/settings selection and no write changes.
Layer 26 remains the factorial editing site; a layer screen does not silently
promote a new editor. Failure of these readers never proves information absent.

## Factorial write

Four fresh entities × three relations; distinct values within each relation.
Capitals [Oslo, Paris, Rome, Tokyo], currencies [Yen, Euro, Dollar, Pound],
languages [Welsh, French, German, Spanish]. Rotate one relation's assignments
by entity index as defined in the fixture. Validate full answer tokenization;
abort before edits if any answer is not one token. Never silently truncate.

Control: exact original recipe generalized to 12 slots, including all 11
other canonical record addresses plus the original 30 decoys.
Candidate: the same recipe with additional binding negatives captured in the
two development training forms: other entities under the same relation and
the same entity under other relations. This distinction matters: canonical
binding negatives already occur in the factorial control's cross-record set.
Preserve Gram–Schmidt ordering, positive addresses, value vectors, layer,
gate/up/down scales, and slot count. No development or test positive-address
augmentation. No changes based on factorial outcomes.

Evaluate canonical installation, four fresh forms, explicit aliases, fresh
near-name controls, unseen entities under each relation, and new unrelated
prompts. Compare clean, write, replace Avenlorn's capital with Berlin, remove,
and restore for each method. The other 11 records must retain their answers,
especially Braskovia's capital and Avenlorn's currency. Persist every query's
full-vocabulary target score, clean agreement, distribution changes and actual
slot instrumentation. Removal and restoration require exact logits.

Write gate: 12/12 canonical reads, ≥80% sentence and alias reads, ≥75% per
binding on sentence forms, ≥95% agreement in each collateral category; apply
the same rules to the replacement arm and require the 11 unchanged canonical
bindings to remain correct. Only a passing factorial gate can reopen native
composition. A bundle or altered editor would require a new experiment.

Graph-causality and sparse-execution work remain separate. The recovered FHG-2
12B textual-intermediate result is preserved with its unresolved necessity
controls; no 12B result is treated as a 4B native-memory baseline.
