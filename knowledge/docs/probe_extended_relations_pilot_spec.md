# Pilot 1c — Extended Relations Probe Spec

**Status:** spec for `knowledge/scripts/probe_extended_relations_pilot.py` (draft script already exists at that path from prior session; this spec describes the design including refinements not yet in the draft).

**Tests:** [P2 (1c per-axis contribution), P3 (depth stratification)](../../META_MODEL.md) from META_MODEL.md.

**Dependency:** [pilot_2a_polysemy_audit_spec.md](./pilot_2a_polysemy_audit_spec.md) must run first. 1c references the polysemy audit's cutoffs for inline classification of new candidate features. If the audit is not yet run, 1c can technically execute but its output is interpretable only against the comparability count, not the stable count — partial information.

---

## What the probe does

Re-runs the gate-matching probe at L0-L20 against 6 WordNet relations NOT covered by canonical's 5:

**Adjective-side (hypothesized L0-L12 contributors):** pertainym, similar_to, attribute, also_see
**Verb-side (hypothesized L13+ contributors per P3):** entailment, cause

Mirrors multilingual + subword pilot structure with five design deltas baked in.

---

## Design deltas vs prior pilots

### Δ1 — Extended scan depth (L0-L20)

Prior pilots scanned L0-L12 (the "syntax band" per vindex `layer_bands`). This probe extends to L0-L20 because verb-side relations (entailment, cause) are hypothesized to live at deeper layers (per P3 in META_MODEL.md).

Cost: ~1.5x the per-subject encoding time (more residual layers cached); same number of forward passes. Match-attempt cost scales linearly with scan layers.

Per-layer per-relation hit count is recorded in the decision JSON so the depth-stratification test (P3) can run on the same data.

### Δ2 — Convergence signature field

Every labeled feature in the rich JSON records its **convergence signature** = `(primary_label, top_output_token)`. This is the discriminator that distinguishes cross-lingual abstraction (M1) from artifact (M3) when comparing across pilots.

Three outcomes for any feature that overlaps a prior pilot:
- **Same signature, overlapping entities** → normal corroboration (the feature exists, both pilots saw the same content)
- **Same signature, disjoint entities** → cross-lingual abstraction signal (the feature targets the same output via different surface forms)
- **Same label, disjoint entities, disjoint outputs** → artifact-suspect; route to polysemy classification

The probe writes a `content_lineage` block in the decision JSON for every overlap feature with the three-way classification surfaced explicitly. This is the operational definition of M1's cross-lingual count and M2's drift assessment.

### Δ3 — Two-axis threshold (comparability + stable)

Per M3 in META_MODEL.md:

- **Comparability label** (cross-pilot consistency): ≥2 hits + confidence > 0.5. Same as multilingual + subword pilots. The new-vs-cumulative count is computed from this set for direct comparison.
- **Stable label**: ≥3 hits AND ≥2 distinct WordNet synsets among matched entities. The synset-diversity check is the L9_F7535 failure detector — counts the number of distinct WordNet synsets that the matched entities belong to, not just the entity count. Two entities from the same synset count as 1 for the diversity check.

For entities without WordNet coverage (technical, code, morphological lemmas — relevant if 1c sweeps subjects that don't all map to WordNet synsets), the diversity fallback is character n-gram Jaccard between entity strings, with the threshold tuned post-hoc on the labeled inventory. These cases are flagged `diversity_check: "fallback"` in output for auditability.

Both counts are reported. Downstream analysis uses the stable count per M3.

### Δ4 — Inline polysemy classification

For every candidate feature that passes the comparability threshold, run the polysemy audit's three-way classification (mono-semantic / polysemantic / promiscuous) inline using the cutoffs set by `pilot_2a_polysemy_audit_spec.md`. Promiscuous candidates are reported but excluded from the new-vs-cumulative count.

This requires the audit cutoffs to be set before 1c launches. The probe loads them from `feature_labels_polysemy_audit_summary.json` (`cutoffs_used` field).

If the cutoffs file is missing, the probe still runs but skips the inline classification and emits a warning that the output is comparability-count only. This is a graceful degradation, not a hard dependency — the probe should be runnable in degraded mode for debugging without requiring a fresh audit pass.

### Δ5 — Canonical drift caveat noted in output

The decision JSON includes a `notes` field with: "Canonical drift ~30% pooled (per M2 in META_MODEL.md). New-vs-canonical comparisons should be read with this in mind; new-vs-cumulative is the primary number." This is a one-line note, not a methodological change — just makes the caveat visible in the artifact rather than only in memory/docs.

---

## Input

`knowledge/data/wordnet_extended_relations.json` (gitignored build artifact -- `knowledge/data/` is untracked; regenerate it with `fetch_wordnet_extended_relations.py`, draft already exists locally). Contains 6 relations × ~3000 pairs each. Subject filter: alpha + length ≥ 3. No multi-piece filter (1c tests relation coverage, not surface form). No canonical-skip (1c relations don't appear in canonical).

`knowledge/data/wordnet_extended_relations_provenance.json` (gitignored, same generator) records per-relation counts and the adjective/verb split.

---

## Output

All under `larql/output/gemma3-4b-v2.vindex/`:

- `feature_labels_extended_pilot.json` — comparability label set (≥2 hits, conf > 0.5)
- `feature_labels_extended_pilot_rich.json` — per-feature: convergence signature, entities, outputs, relations, first_layer, polysemy classification, polysemy metrics
- `feature_labels_extended_pilot_stable.json` — stable subset (≥3 hits + ≥2 synsets + mono/polysemantic)
- `feature_labels_extended_pilot_decision.json` — branch fired, per-relation breakdown, per-layer per-relation hits, depth-stratification result, content lineage for overlap features, P2 and P3 outcomes

---

## Decision rule (refer to P2 in META_MODEL.md for prediction)

```
new_vs_cumulative = pilot_wn_keys − (canonical_wn ∪ multilingual_wn ∪ subword_wn)

≥ 50  → Branch A: relation coverage was a major axis; per-axis ceiling under-predicted
10-49 → Branch B: contributes per working model; reassess cumulative trajectory
<10   → Branch C: relation coverage was NOT a binding axis; model over-predicted ceiling
```

Per-relation breakdown reported alongside. Per M3, the new-vs-cumulative count uses the **stable** subset only — promiscuous candidates are not counted as labels. The comparability count is reported for cross-pilot continuity but is not the load-bearing number.

**P2 outcome registration in decision JSON:**

```json
{
  "tests_prediction": "P2 in META_MODEL.md",
  "P2_predicted_range": [21, 53],
  "P2_observed_new_vs_cumulative_stable": <N>,
  "P2_outcome": "confirmed | refuted-high | refuted-low | partial",
  "per_relation_predicted": {
    "pertainym": [8, 15], "similar_to": [6, 12], "attribute": [3, 8],
    "also_see": [4, 10], "entailment": [0, 5], "cause": [0, 3]
  },
  "per_relation_observed": {...},
}
```

---

## Depth stratification (refer to P3)

The depth test runs on the per-layer per-relation hit counts. Decision logic:

```python
verb_early = sum(hits[r][l] for r in {entailment, cause} for l in 0..12)
verb_late  = sum(hits[r][l] for r in {entailment, cause} for l in 13..20)

if verb_early + verb_late < 20:
    P3_outcome = "untestable"
elif verb_late > 2 * max(verb_early, 1):
    P3_outcome = "supported"
elif verb_early > 2 * max(verb_late, 1):
    P3_outcome = "refuted-inverse"
else:
    P3_outcome = "refuted-spread"
```

Recorded in decision JSON. The `untestable` outcome is important: it means the methodology does not detect verb relations as features (or there are <20 such features in total), and P3 cannot be tested with this probe. In that case the working model is partially un-tested, not refuted.

---

## What NOT to do

The probe does not:
- Merge into canonical. All output is separate. Merging is a deliberate later session with its own threshold.
- Retroactively re-label multilingual + subword features. The convergence-signature analysis surfaces overlap features but does not change their pilot labels.
- Decide whether to launch 1d or 2c. The decision rule produces the branch fired; the next experiment is decided in a separate session based on the cumulative picture, not auto-triggered by 1c.

---

## Runtime estimate

- Encoding (Phase 2): ~25-35 min for ~10-15K unique subjects across 6 relations (extrapolated from subword pilot's 11.6K subjects at 7-15/s on L0-L12, scaled by 1.5x for the extended scan)
- Gate matching (Phase 3): ~5-10 min
- Total: ~30-45 min

Comparable to multilingual + subword pilot runtimes. Runnable in a single session.

---

## Post-run protocol

1. Spot-check rich JSON: top features by hits, verify they look coherent.
2. Check convergence-signature overlap features: how many cross-lingual corroborations (M1) vs how many artifact-suspects?
3. Check P3 outcome: did verb-side land at L13+?
4. Update META_MODEL.md with P2 and P3 outcomes (append, do not edit).
5. Update `project_larql.md` memory with the result and the next-action decision.
6. If P4 (polysemy audit) hasn't run yet, run it now on the new cumulative set including 1c labels.

The post-run session should be its own short session — design + run + update memory is too much in one sitting (see [[feedback-positive-results-dont-skip-pilots]]).
