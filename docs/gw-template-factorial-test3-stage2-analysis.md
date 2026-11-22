# GW-TEMPLATE-FACTORIAL Test 3 Stage 2: analysis definitions and thresholds (frozen before execution)

**Status:** written 2026-11-18, before any Stage 2 arm executes. This file is hashed into the Stage 2 binding. It implements design d3 (`docs/gw-template-factorial-test3-design-d3.md`) with the owner's Stage 2 decisions of 2026-11-18. Every rule below is fixed before any candidate readout exists. Thresholds are the owner's; nothing here is chosen after seeing results.

The executable form of these definitions is `scripts/gwtemplate_test3_stage2_analyze.py`. Where the two differ, that is a defect to be reported, not resolved silently.

## 1. Scope (owner GO, 2026-11-18)

- **Development only.** The 59 Test 2 countries. The 28-country holdout contributes no capture, label, activation, fit, threshold or selection. Answer classes are taken from the 59 development countries only.
- **Split:** the existing deterministic 39 fit / 20 validation split (salt `gw-template-factorial/test3-fitval/v1|`). Fit-half split: salt `gw-template-factorial/test3-fit-halves/v1|`.
- **Only validation targets decide gates.** Fit-target results are descriptive.
- **Layers:** L1 is primary and gated. L16 is a descriptive contrast with no gate.
- **Map:** window only (all window slots, as the Test 2 window arms).
- **P2, P3 and pooled P2+P3, and both relation directions,** are reported separately.

## 2. Arm inventory (per target cell: 59 countries × 2 relations × 2 phrasings = 236 cells)

| Arm | Kind | Vectors replacing the target window (layers a−1..33, every window slot) | Role |
|---|---|---|---|
| `ID-window-L1` | ID | the target's own captured r window | integrity (must be bit-identical to natural) |
| `S-window-L{1,16}` | S | Y's own r' window | reference (B1 baseline) |
| `C0-D1-L{1,16}`, `C0-D2-L{1,16}` | C0 | donor X'ₖ's r' window, donors as in Test 2 | reference (Test 2 replication) |
| `C1-window-L{1,16}` | C1 | mean over F∖Y of h(e, r', p) | candidate |
| `C1A-window-L{1,16}`, `C1B-window-L{1,16}` | C1A, C1B | the same over F_A∖Y and F_B∖Y | split (B2) |
| `C2-window-L{1,16}` | C2 | h(Y, r, p) + mean_F∖Y h(e, r', p) − mean_F∖Y h(e, r, p) | candidate |
| `C1E-window-L{1,16}` | C1E | mean over F∖Y of h(e, r, p), into Y's r prompt | control (B4) |

17 arms per cell, 4,012 arms. No C3, no C2 halves, no word map, no other layer.

## 3. Token classes

Every accepted first-token set in the development population is a single token, and no two development countries' sets partially overlap (checked by the analysis; it fails closed otherwise). A **class** is therefore one token. Several countries may share a class (for example a shared currency code's first token); the count of development countries sharing it is its multiplicity.

For a target cell (Y, r, p), r' the other relation:
- **r(Y)**, **r'(Y)**: Y's accepted tokens under r and r' in phrasing p.
- **Other-country r' classes:** the distinct r'-tokens, in phrasing p, of the development countries Z ≠ Y, *excluding* the tokens of r(Y) and r'(Y).
- **Other-country r classes:** likewise under r.

Every class token must be in the 325-token candidate set, or the analysis fails closed. So no class is missing for an arm.

## 4. Per-arm quantities

`lp_x(c) = logit_x[c] − logsumexp(all logits)` from the runner readout (full-vocabulary log-probability).

**Relation-switching arms** (S, C0, C1, C1A, C1B, C2):
- **Reference sep** = the natural run (Y, r', p) in the same execution.
- **Outcome** (the greedy token), first match in this order: `r(Y)`, `r'(Y)`, `other_country_r'`, `other_country_r`, then `code_like`, `function_or_punct`, `word` (the result-6270 categories, with development-only classes).
- For C0, also the indicator that the greedy token is the donor's r' token, and B1 (below).
- **Switch margin:** m_switch = lp(r'(Y)) − lp(r(Y)).
- **Gains:** lp_arm(c) − lp_natural(Y, r, p)(c), for c = r(Y), r'(Y).
- **KL:** KL(sep ‖ arm) over the candidate set, each renormalised within it.
- **B3 excess per class:** e(c) = lp_arm(c) − lp_sep(c), for every other-country r' class c.
- **B3 arm mean:** the unweighted mean of e(c) over the eligible classes (each class once). A multiplicity-weighted mean is reported, descriptive only.

**C1E (the E analogue):** the reference is the target's own natural (Y, r, p). Its outcome order is `r(Y)`, `r'(Y)`, `other_country_r`, `other_country_r'`, then the decoded categories. Its B3 uses other-country *r* classes, and its KL is to the natural run. It is descriptive and has no gate.

**B1 (C0 only):** gain_D(r'(X')) − gain_S(r'(X')), with S the same target, layer and window. Arms where r'(X') = r'(Y) or r(X') = r(Y) are excluded and counted.

## 5. B2 membership sensitivity (the declared signed estimand)

Per target cell (Y, r, p) and layer, from the C1A and C1B arms of that cell:
- **A-classes:** the distinct r'-tokens of the countries in F_A∖Y, excluding r(Y), r'(Y) and any token that is also the r'-token of a country in F_B∖Y. B-classes are defined symmetrically.
- d_A = mean over A-classes of [lp_C1A(c) − lp_C1B(c)].
- d_B = mean over B-classes of [lp_C1B(c) − lp_C1A(c)].
- **M(Y, r, p) = (d_A + d_B) / 2,** in nats. It is positive when an average favours the answers of the countries it was built from.
- **Missing rule:** a cell with no A-class or no B-class has no M. It is excluded and counted, not imputed.

**Interval:** a cluster bootstrap over target countries (§7). **Gate:** the whole 95% interval inside [−0.5, +0.5]. A wide interval containing zero fails.

B2 measures the averaging recipe and is reported under C1. **C2's halves are not in the approved Stage 2 inventory, so B2 is *not measured* for C2.** C2's Axis-B verdict is therefore reported as `incomplete: B2 not measured`, never as a pass.

B2 and B1 are different estimands and are never compared numerically.

## 6. B3 aggregation (the eligible class set, aggregation and missing-class rule)

Per (kind, direction, phrasing group, layer, population) group:
- **B3 mean excess:** the cluster mean over the group's arms of the B3 arm mean.
- **B3 maximum per-class mean excess:**
  1. For each class c, average e(c) over the arms in the group where c is eligible (the per-class mean across target countries).
  2. Take the maximum over classes eligible in at least **10 distinct target countries** of the group.
  - A class is ineligible for a target only when it equals r(Y) or r'(Y), so for pooled validation groups (20 countries) nearly every class qualifies. The 10-country support floor is fixed here so that a class seen in one or two targets cannot set the gate.
  - The arg-max class, its multiplicity and its support are reported.
- **Other-country top-1 rate:** the share of arms whose outcome is `other_country_r'`.
- The intervals of all three are cluster-bootstrap; the gates use the point estimates (below).

## 7. Inference

- **Bootstrap:** a cluster bootstrap over target countries (a country's P2 and P3 cells, and both C0 donors, stay together). 10,000 resamples from one `numpy.random.default_rng(20261118)` generator, drawn in the analysis's fixed group order. The interval is the percentile 95% interval (`vals[⌊0.025 B⌋]`, `vals[⌊0.975 B⌋ − 1]`), as in Test 2.
- For the B3 maximum, each resample recomputes the per-class means and the maximum.

## 8. Gates (validation targets, L1, per direction, pooled P2+P3 unless stated)

Applied to the candidates C1 and C2. C0 and S are references; C1A, C1B and C1E are diagnostics.

| Gate | Rule |
|---|---|
| **A: switching** | Share of `r'(Y)`: the lower 95% bound is ≥ 0.80, AND the P2 and P3 point estimates are each ≥ 0.60 |
| **B2** (C1 only) | The whole 95% interval of M is within [−0.5, +0.5] |
| **B3 mean** | The B3 mean excess point estimate is ≤ 0.5 nats |
| **B3 maximum** | The maximum per-class mean excess point estimate is ≤ 1.0 nat |
| **Secondary** | The other-country top-1 rate is ≤ 0.05 |

**Verdict per candidate and direction:**
1. `no switch` if A fails. Axis B is reported but not scored, so a low leak alone never scores.
2. Otherwise `pass` if every applicable B gate passes.
3. `incomplete: B2 not measured` for C2 when every measured gate passes.
4. Otherwise `axis B fail`, naming the gates that failed.

A verdict is a development-validation result. It freezes nothing: freezing and the holdout are later stages, each needing its own GO.

## 9. Reported for every group

Groups: kind × direction × phrasing (P2, P3, P2+P3) × layer (L1, L16) × population (validation, fit). For each:
- the n of arms;
- outcome shares;
- mean m_switch, gains and KL;
- B3 mean, weighted mean, maximum (with arg-max class) and other-country top-1;
- B2 (C1);
- B1 and the donor-answer share (C0).

**Continuous metrics are reported even when a gate fails.**

**Parameter counts** (per direction and phrasing):
- C1, C1A, C1B and C2: n_layers × n_slots × 2,560, where n_layers is 34 at L1 and 19 at L16, and n_slots is 3 (P2) or 4 (P3). C2's count is its displacement field.
- C0 and S carry the same number of values, but they are copied, not estimated.

**Computational costs:** the store build time and size, the vector count, the runner wall time per shard and per arm, and the analysis time.

## 10. Integrity checks (declared; all must pass before any result is reported as admitted)

1. **Binding:** the qualified runner binary, builder, admission checker and capture manifest equal their qualified hashes.
2. **Static:** `--check` passes on every shard. Independent admission (`gwtemplate_test3_derived_admit.admit`) has zero failures, before and after execution.
3. **Recompute:** every store vector is recomputed independently (`gwtemplate_test3_derived_admit.recompute`) and hash-equal.
4. **Inventory audit:** an independent audit of arm names, coverage, recipes, halves membership, donors and information boundaries against the Test 2 plan and the split.
5. **Identity:** 236/236 identity arms bit-identical.
6. **Natural:** every natural readout's `next_logit_sha256` equals the Test 2 target's `expect_natural_logit_sha256`.
7. **Test 2 replication (all, not a sample):** every C0 and S arm at L1 and L16 equals the corresponding Test 2 window arm's `next_logit_sha256` (944 C0 + 472 S).

## 11. Limits on interpretation (owner)

- A successful all-layer residual replacement does not show an independently executable operator.
- A mean residual is not a feature, neuron or relation operator.
- A collapse of switching under C1 and C2 does not show that entity-specific information is necessary.
