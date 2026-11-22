# GW-TEMPLATE-FACTORIAL Test 3: compact relation-state design (d2)

**Status:** d2, 2026-11-18. It incorporates the owner's decisions on d1, recorded in §11.
- **Authorised:** Stage 1b (development-only capture plus reconstruction qualification), and engineering of the derived-vector runner.
- **Not authorised:** any candidate intervention, fitting, selection or holdout access.

No existing binding file, registry result or frozen record is changed.

**Question:** Can a *compact* relation state, one that is not a specific donor's full window, reproduce Test 2's target-specific relation switching? And can it do so while carrying less source-entity information than the full-window transplant?

**Not the question:** whether a neuron, feature or executable graph edge exists. Section 9 covers that, and it is separate.

## 1. What Test 2 fixes as the reference

**Intervention (Test 2 runner `gwtemplate_xpos_runner`, mode `replace`):**
- Let a be the start layer. For every layer l from a−1 to 33, the **post-FFN residual carrier** (the layer's output residual) at each target window position is replaced by the donor's carrier at the corresponding window position.
  - Window positions are right-aligned; slot j of the target window maps to slot j of the source window.
- The target's own states before the window are untouched, and the runner checks this bit-exactly.
- K and V for the window positions at layers ≥ a are *recomputed* from the replaced residuals, with RoPE applied at the **target** positions. So K/V are not copied; residuals are.
- The readout position ("is") keeps its own states at every layer. It learns the relation only by attending to the window.

**Reference behaviour (window, L1, D arms, P2+P3; result 6269 and audit 6271):**

| Direction | Top-1 r'(Y) | Donor answer r'(X') wins | Paired D−S donor leak | E-arm leak r(X') |
|---|---|---|---|---|
| capital → currency | 0.96 [0.93, 0.99] | 0.00 | +0.56 [0.09, 1.03] nats | +0.54 nats (currency-phrase source) |
| currency → capital | 1.00 [1.00, 1.00] | 0.00 | +2.13 [1.77, 2.51] nats | +2.28 nats (capital-phrase source) |

Reporting rules carried over:
- Every candidate is reported per direction and per phrasing (P2, P3, and P2+P3 pooled), against this reference.
- L16 is reported alongside as the late-boundary contrast; it is not a target.

**Test 2 recorded readouts and hashes, not activation tensors.** No candidate in this document can be built from existing records. All of them need Stage 1b capture.

## 2. Populations (deterministic, model-free)

| Population | Countries | Use |
|---|---|---|
| Development | 59 (Test 2 population) | Capture, fitting and validation |
| — fit | 39 | Means, fitted operators, leave-one-out aggregates |
| — validation | 20 | Candidate and rank selection; never in any aggregate or fit |
| Holdout | 28 (`proposed_test3_entity_holdout`, salt `…/test3-entity-holdout/v1\|`) | Final evaluation only, after the freeze in §7 |

- **The split:** sort the 59 by `sha256("gw-template-factorial/test3-fitval/v1|" + ISO2)`. The first 20 form validation and the rest form fit.
  - The split is checked for answer-class balance: every ISO 4217 class shared within the 59 (for example EUR) is reported by side. It is not rebalanced.
  - Rebalancing would make the assignment depend on outcomes, so any imbalance is ledgered instead.
- **Holdout disclosure:** the 28 holdout countries' *natural* baseline readouts already exist in `baseline-01`. That baseline covered all 246 countries, natural runs only, and fed the eligibility counts. They have had no intervention, capture, fit or selection. Their eligibility for Test 3 is computed only at the freeze step, from those existing records, by a rule fixed here: the same joint capital × ISO 4217 rule as Test 2, P2 and P3.
- P1 stays out of scope, for the same reason as Test 2.

## 3. Stage 1b capture specification (for a separate GO)

**Runs:** the 236 natural development runs already in the Test 2 plan: 59 countries × {capital, ISO 4217} × {P2, P3}. No holdout run and no intervened run is captured.

**Per run, captured:**
- The post-FFN carrier `after` vector (f32, 2,560 values) at every window slot, for every layer 0–33. That is 34 × 3 = 102 vectors for P2 and 34 × 4 = 136 for P3.
- The same data for the readout position, for diagnostics only. No candidate may use it.

The capture does not store K/V tensors: candidates inject residuals, and K/V are recomputed. Each run's K/V row hashes are recorded so they can be compared with the Test 2 seal.

**Format and provenance:**
- One file per run: `capture/<run-id>.f32`, raw little-endian, laid out layer-major then slot.
- An immutable `manifest.json` per capture holding:
  - the model and tokenizer hashes and the executable sha;
  - each run's token IDs, window slot → absolute position map and step-logit hashes;
  - a per-vector sha256 using the runner's `vector_sha256`.
- The whole capture is sealed by its manifest sha before any use.
- Size: about 236 × 3.5 × 34 × 10 KB ≈ **290 MB**. It is archived in parts, like Test 2.

**Independent reconstruction checks (all must pass before any candidate is built):**
1. **Seal agreement:** every captured vector's sha equals the carrier hash in the Test 2 seal (`test2-02/run/seal.jsonl`, sha `de92eff7…`). This needs no shared code with the capture executable.
2. **Natural replay:** re-execute every run and compare against the recorded step-logit hashes and K/V hashes.
3. **Re-injection identity:** for every run, inject the captured window carriers back into the *same* run (mode `replace`, all layers). This must reproduce the natural readout bit-exactly.
4. **Test 2 reproduction:** rebuild a declared, salted sample of 12 D arms and 4 E arms from captured vectors loaded from file, not from live donor runs. Each must reproduce Test 2's recorded `next_logit_sha256` bit-exactly. This links the capture to the measured reference.
5. **RoPE independence:** for the cross-position sample, an independent numpy recomputation of K from captured residuals at the target positions agrees with the runner. Test 2 qualification tolerance applies.
6. **Refusals:** a corrupted vector, a holdout run, a wrong slot map and an existing output directory each refuse.

**Cost:** 236 natural forwards plus replays and reinjections, about 20–30 minutes of CPU. It needs the usual quiet-window coordination and the Mac on mains power.

**Engineering:** a capture executable, or a `--capture` mode, plus a **vector-provenance extension** to the runner. Candidates need `Derived{recipe sha, input vector shas}` provenance, alongside the existing `Captured`. That is a runner change, so it needs requalification against the Test 2 qualification suite before any candidate arm executes.

## 4. Candidate hierarchy

Every candidate produces, for each layer l from a−1 to 33 and each window slot j, a vector to replace the target's window carrier. Each is applied with the same all-layer clamp as the reference, so that only the *content* of the replacement varies.

Notation:
- h(e, r, p)ₗⱼ is the captured carrier.
- F is the fit set; F∖Y is the fit set without Y.

| ID | Candidate | Entity information it can carry | Parameters (per direction, phrasing) |
|---|---|---|---|
| C0 | Full donor window, as in Test 2 D arms (replicated in the same execution) | The donor's own | none (the reference) |
| C1 | **Mean donor:** mean over F∖Y of h(e, r', p) | Population average over the fit countries | 34 × n slots × 2,560 means |
| C1-split | Two disjoint-half means, F_A and F_B, each without Y | Each half's population | 2 × C1 |
| C2 | **Relation translation:** h(Y, r, p) + [mean over F∖Y of h(e, r', p) − mean over F∖Y of h(e, r, p)] | The target's own (by design), plus a population shift | Same as C1 |
| C3-k | **Low-rank translation:** ĥ(Y, r', p) = h(Y, r, p) + Δ̄ + U_k ẑ_k(h(Y, r, p)), where Δ̄ is the fit-set mean displacement and U_k the top-k principal subspace of the fit countries' displacements Δ(e) = h(e, r', p) − h(e, r, p) (per layer and slot). ẑ_k is a ridge map fitted on the fit countries only, taking **only** h(·, r, p) as input. k ∈ {1, 2, 4, 8, 16} | The target's own, plus fitted structure | ≤ 34 × n × (2,560 k + k² + k) |

- **Order:** C1 is evaluated before C2. **C3 fitting is deferred until C1 and C2 have been reported and reviewed (owner decision).**
- **C3 information rule:** for any target outside the fit set (validation, holdout), the candidate is computed from h(Y, r, p) and fit-set parameters only. h(Y, r', p) is a training label for fit countries and never an input for prediction. The candidate builder enforces this: it refuses to read any r' capture of a non-fit target when building C2 or C3 vectors. It does read r' captures of non-fit targets for C0 and S, which are references, not candidates.
- **Leave-one-target-out is mandatory** whenever Y ∈ F, for C1, C2 and C3.
  - Validation targets never contribute to any aggregate, so they are out-of-population automatically.
  - Fit targets use F∖Y. Their results are reported but **not** used for selection.
- **Equal data budget:** C1–C3 use exactly the same captured fit-set vectors, so no candidate sees more countries or runs. Parameter counts are reported next to every result.
- **Why both C1 and C2:**
  - C1 asks whether the *new relation's typical window* switches the target. It has no target window at all.
  - C2 asks whether a *relation displacement* applied to the target's own window does. That is closer to an operator, and it contains no other single entity.
  - Either can fail independently.
- **What averaging does not do:** it does not remove entity information by construction. Shared associations, population bias (for example several eurozone countries in F) and correlated entity features survive in a mean. §5 measures them instead of assuming them away.

## 5. Two-axis scoring

All quantities are first-token, using the Test 2 readout, the 325-token candidate union and the Test 2 accepted sets. `lp` is the full-vocabulary log-probability of a class's best accepted token, exactly as in audit 6271.

### Axis A — target-specific relation switching

- **Share of top-1 r'(Y)**, with a cluster bootstrap over target countries.
- **Switch margin:** m_switch = lp(r'(Y)) − lp(r(Y)).
- **KL to the separable reference:** KL(natural(Y, r', p) ‖ arm), over the candidate set.
- **Outcome classes:** r(Y), r'(Y), other-country r', other-country r, and other (decoded, with the other-token categories of result 6270).

### Axis B — leakage, defined per candidate type

A mean has no unique donor, so the paired D−S metric is used only where it is defined.

| Measure | Applies to | Definition |
|---|---|---|
| **B1: source-specific leak** (paired D−S) | C0 | gain_D(r'(X')) − gain_S(r'(X')), as in audit 6271 |
| **B2: membership sensitivity** | C1-split, C2/C3 refitted on halves | For a country Z in F_A: lp under the F_A-built candidate minus lp under the F_B-built candidate, for Z's r' answer. Averaged over Z in each half, then over targets. It measures dependence on *which countries were averaged*. **It is a different estimand from B1 and is never compared numerically with it.** Applied symmetrically to F_B |
| **B3: population bias** | All candidates | For every country Z ≠ Y, the excess of Z's r' answer: lp_arm − lp_sep, with sep = natural(Y, r', p). Reported as a distribution across Z, the top 10 classes, and the share of arms where any other-country r' answer is top-1. Classes shared by several averaged countries (EUR, USD, XOF, XCD and so on) are flagged and weighted by their multiplicity in F |
| **B4: E-arm analogue** | C1, C2 | The same-relation version of the candidate (mean of h(e, r, p) over F∖Y) injected into Y's r prompt. It should leave r(Y) on top. Its B3 measures population bias without a relation change |

- The B1 values at C0 come from the *same* execution, as a replication of the Test 2 reference.
- B1, B2 and B3 are reported side by side, each with its own reference, and no ratio or difference between them is a criterion.
- **A low leak is never scored alone.** A candidate that does not switch is reported as "no switch" whatever its leak. Your zero-vector point is enforced by requiring Axis A to be met first.

### Interpretation table (fixed before capture)

| Result on development validation | Reading |
|---|---|
| High switching, B2 and B3 below the C0 reference | A shared relation-state approximation captures much of the causal effect |
| High switching, persistent B3 bias | Shared relation control exists, but averaging does not isolate it cleanly |
| Switching collapses (C1 and C2) | A mean state is insufficient. This does **not** show that entity-specific information is necessary or that no shared relation representation exists |
| C3-k beats C1/C2 on validation | Learned structure improves on the shared-state baseline |
| C2 switches but C1 does not | Relation control depends on combining with the target's own window; this is a description, not an operator claim |

## 6. Selection rules (proposed; owner to fix before Stage 2)

Selection uses **development validation targets only** (20 countries × 2 directions × 2 phrasings = 80 target cells per candidate and layer). The rules are applied per direction.

1. **Axis A gate:** the lower bound of the 95% CI on top-1 r'(Y) is ≥ 0.80 at L1, pooled P2+P3, and neither phrasing is below 0.60. *Proposed thresholds; owner decision.*
2. **Axis B among gate-passers (owner d2):**
   - **Secondary gate:** B3 other-country top-1 share ≤ 0.05.
   - **Continuous bias criterion**, to be fixed numerically in a d3 revision **before any validation execution**:
     - B2's cluster-bootstrap 95% CI must lie entirely within ± a pre-registered materiality bound, so that no material membership effect is detectable;
     - B3's mean excess and its top-class excess must stay within a pre-registered bound.

     The materiality bounds are proposed in d3 from model-free considerations only. Pass/fail on the ≤ 5% gate is never sufficient on its own: a candidate can favour other countries systematically without their answers winning.
3. **Choosing C3-k:** the smallest k whose Axis A mean is within 0.02 of the best k. Ties go to the lower parameter count.
4. **Final candidates:** at most **two** are frozen. The simplest passing one is mandatory if any passes; the best C3-k is optional.
5. **If nothing passes:** report a negative result for development validation and stop. No holdout run. No new candidate family is added after seeing validation results without a new design revision, which would itself be ledgered.

## 7. Freeze before the holdout

**Frozen before the 28-country holdout is opened:**
- the candidate recipes and their sha;
- k;
- the fit population: **the 39-country fit set (owner decision)**. The 20 validation countries are never used for fitting, before or after selection.
- layers {L1, plus L16 as contrast};
- thresholds;
- the analysis script, plan builder, runner and binding.

**Holdout evaluation:**
- Holdout targets use the candidates with C0 donors drawn *from the holdout*. Pairing uses the Test 2 salted-hash rule with a new salt.
- No holdout country enters any aggregate.
- Holdout capture is limited to the donor runs needed for C0 and to the targets' own windows for C2 and C3. That is a separately authorised Stage 4 capture.

**Accounting:** one result per frozen candidate per direction, with no reselection afterwards. A holdout failure is reported as is.

## 8. Resource budget (estimates)

| Stage | Work | CPU |
|---|---|---|
| 1b capture | 236 natural runs plus checks | ~0.5 h |
| 1c engineering and requalification | Derived provenance plus the candidate builder | ~0.5 h qualification inference (non-population countries, like Test 2) |
| 2 development evaluation (C1, C2 only) | 236 target cells × {C0 D1/D2, S, C1, C1-split ×2, C2, C1-E, C2-E} × {L1, L16} ≈ 236 × 9 × 2 ≈ 4,250 arms | ~1.7 h (Test 2 rate ~1.4 s per arm when awake) |
| 4 holdout | ≤ 2 candidates plus C0 and S on ~28 × 4 cells × 2 layers | ~0.5 h |

## 9. Mechanistic follow-up (separate programmes; not designed here)

- **Where the switch is established:** per-layer and per-slot attribution of the C0/C1 effect. For example, single-layer instead of all-layer clamps, and attention-path ablations from the readout to the window. This is needed before any FFN or neuron claim.
- **Shared-component removal:** remove the projected capital component and the currency component (once identified) and check the capital-vs-currency dissociation table.
- **Neuron-level work** is owned by the neuron-editing session (NEURON-001 / WSP line). The only interface: if attribution localises the switch to specific FFN outputs at specific layers and slots, that location becomes an input to that programme.

Not implied: that an attention-mediated, all-layer residual transplant corresponds to an FFN-neuron circuit; or that a mean residual direction is a feature, neuron or executable relation operator.

## 10. Open items

1. The d3 numeric materiality bounds for B2 and B3. These are model-free and must be set before validation execution.
2. Derived-vector runner qualification and a separate GO before any candidate intervention executes.

## 11. Owner decisions on d1 (2026-11-18)

- **Switching gate:** lower 95% bound ≥ 0.80 at L1, pooled; neither phrasing below 0.60. Kept.
- **Leakage:** B2 (membership) and B1 (C0 donor) are different estimands and are not compared numerically. Both are reported. The continuous bias criterion is fixed before validation execution.
- **Other-country top-1 ≤ 5%:** kept as a secondary gate, with continuous population-bias analysis alongside.
- **Final fit population:** 39 countries. The 20 validation countries stay unseen through final evaluation.
- **C3:** deferred until C1 and C2 are reported. Its specification is corrected so prediction uses only h(Y, r, p) and fit-set parameters.
- **Layers:** L1 is primary; L16 is the only contrast.
- **Stage 1b:** GO, development only.
- **Derived-vector runner:** engineering permitted; any execution of a new intervention needs qualification and a separate GO.
