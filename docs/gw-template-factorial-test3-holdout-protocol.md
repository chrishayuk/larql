# Test 3 Stage 4: final holdout protocol for C1 and C2, capital→currency (frozen-ready, not executed)

**Status:** prepared under the owner's GO of 2026-11-20 for *model-free preparation* only. It becomes binding when the owner approves execution, and `bind` then hashes it into a Stage 4 binding.
- No holdout data has been read for this protocol.
- No holdout capture or inference has run.

**Development selection (accepted and frozen by the owner):**
- C1 and C2 pass every frozen development gate for capital→currency (results 6283 and 6290).
- Neither qualifies for currency→capital (6284, 6286 and 6291).
- The Stage 2 results and the separately recorded, non-blind C2A/C2B amendment (run …01228) are preserved unchanged.

## 1. Question and claim

Do the frozen constructed relation states switch Gemma 3 4B from the capital answer to the target's own ISO 4217 code? The test uses 28 countries reserved before Test 1. They contributed nothing to construction, fitting, development-validation interventions or candidate selection.

- **Disclosure:** the holdout countries' *natural* baseline readouts exist in `baseline-01`. They fed the Test 1 population counts (the joint cohort and the subject-echo lists). That record is now also used for eligibility and the natural-readout integrity check.
- The holdout is therefore **fit-unseen and intervention-unseen**, but not unseen by every part of the programme.
- The C2 membership-sensitivity (B2) evidence comes from a post-hoc amendment that reused a frozen estimand. This holdout is C2's first fully prospective test.

## 2. Frozen inputs (no new rank, threshold, candidate family or fitting rule)

| Item | Frozen value |
|---|---|
| Fit population | the 39-country fit set (salt `gw-template-factorial/test3-fitval/v1\|`), read from the qualified Stage 1b capture (manifest `ce46d692…`) |
| C1 | mean over the fit set of h(e, currency, p) (v1 builder `705a2198…`, formula `test3-candidates/v1`) |
| C2 | h(Y, capital, p) + mean over the fit set of h(e, currency, p) − mean over the fit set of h(e, capital, p) (v1 builder) |
| Intervention | all-layer residual replacement from layer 0 (L1), at every window slot, window map |
| Phrasings | P2 and P3 |
| Direction | capital→currency only |
| Runner | the qualified derived runner (`e240c6a5…`), unchanged |
| Metrics | the unchanged Stage 2 `per_arm`: outcome order, switch margin, gains, KL to the separable reference, B3 excess |
| Gates | the Stage 2 values: switching lower 95% bound ≥ 0.80 pooled, each phrasing ≥ 0.60; B3 mean ≤ 0.5; B3 max ≤ 1.0 with support ≥ 10 targets; other-country top-1 ≤ 0.05 |
| B2 | carried from development (C1: 6283, M 0.046 [0.036, 0.056]; C2: 6290, M 0.031 [0.014, 0.049]); not re-measured, and no holdout analogue |

## 3. Population (computed only at `prepare`, after approval)

**Holdout list:** the 28 countries in `test1-population-01.json` (`proposed_test3_entity_holdout`, salt `…/test3-entity-holdout/v1|`).

**Eligibility rule** (d3 §2, plus the existing readability requirements):
1. The country is in the joint capital × ISO 4217 cohort.
2. It is not a subject-echo country.
3. Its P2 and P3 baseline cells exist for both relations.
4. **Window geometry:** the window lengths are 3 (P2) and 4 (P3), each window ends one position before the readout, and the relation word lies inside the window.
5. Its window token ids equal the development canonical ids.
6. In P2 and P3, each accepted first-token set is a single token.
7. The capital and currency tokens differ.
8. Both answer tokens are in the union candidate set.

Requirements 4–8 are what the Test 2 builder and the Stage 2 analysis already require; Stage 2 refuses any cell that violates them. Exclusions are ledgered with their reasons and never replaced.

**Approval checkpoint:** `prepare` reads only the baseline plan cells, the population file and the baseline natural readout hashes. It prints:
- the eligible count;
- the exclusions;
- the donor availability.

It loads no model. Execution is a separate command.

**References:** C0 donors are drawn from the eligible holdout. The rule is Test 2's (same subject token count, four pairwise-disjoint first-token sets in P2 and P3), with the new salt `gw-template-factorial/test3-holdout-donors/v1|`. Each target gets at most 2 donors, and a target with fewer gets fewer C0 arms. C0 and S are references and never gate.

## 4. Information rule (enforced by the builder, the admission checker and the capture view)

- **C1** reads only fit-country currency windows.
- **C2** reads the target's own capital window h(Y, capital, p) plus fit-country windows. **No candidate ever reads h(Y, currency, p)** or any other holdout or validation country.
- **No validation country** is read anywhere. The capture view loads only the fit set and the eligible holdout countries, and refuses validation and ineligible holdout countries.
- **References:**
  - S copies h(Y, currency, p);
  - C0-Dk copies h(donor_k, currency, p);
  - ID copies h(Y, capital, p).
- **Runner refusal list:** every runner plan's list is the 59 development countries plus any ineligible holdout country. The qualified runners refuse to execute any run whose country is on it, so only eligible holdout runs execute.

## 5. Arms

| Cell | Arms (L1, window map) |
|---|---|
| (Y, capital, p) target | `ID-window-L1` (identity), `S-window-L1`, `C0-D1-L1`, `C0-D2-L1` (when available), `C1-window-L1`, `C2-window-L1` |
| (Y, currency, p) reference | `ID-window-L1` only; its natural readout is the separable reference for KL and B3 |

With 28 eligible countries: 56 target cells × 6 + 56 reference cells = 392 arms.

## 6. Analysis

- **Per-arm metrics:** Stage 2's code, unchanged. The B3 "other-country" classes are the currency tokens of the **59 development countries** minus r(Y) and r'(Y), the same class population as Stage 2. Holdout tokens are used only for the target's own answers.
- **Bootstrap:** a cluster bootstrap over holdout targets, seed 20261118, 10,000 resamples. Draws are consumed in a fixed order: kind (C1, C2, S, C0) → phrasing (P2+P3, P2, P3) → outcome shares, then the scalar metrics, then the B3 max.
- **Verdict per candidate:** `pass`, `no switch` or `axis B fail: …`. That is one result per frozen candidate, with no reselection.
- **Also reported:** S and C0 descriptives; C0 donor-answer top-1; the C1-vs-C2 comparison (descriptive only).

## 7. Execution chain and integrity (fail closed; fresh directories; never retried in place)

1. **bind:** checks the qualified identities (the derived runner, v1 builder, v1 admission, development capture, Stage 2 and C2AB reports, and the Stage 4 qualification record) and runs the tests. No holdout data is read.
2. **prepare:** eligibility, donors, runs, the seal plan and the replication plan (§3).
3. **seal:** the BREADCRUMB-2 seal executable, natural runs only. The last step of every holdout run must equal the baseline natural readout hash. This equivalence is verified on all 236 development runs (QH6).
4. **capture:** the qualified Stage 1b capture executable, against the seal. Then seal agreement on every vector, K/V row and step, and reinject verification.
5. **derived build:** v1 builder over the restricted capture view. Then the runner's `--check`, Stage 4 admission (independent) and an independent recompute of every vector.
6. **derived runner.**
7. **replication:** the qualified Test 2 runner executes S-window-L1 and each Dk-window-L1 in Test 2's arm format, with sealed donor carriers. Every S and C0 derived arm must equal it bit-exactly. Test 2 admission is applied to its records.
8. **integrity gates:**
   - post-run admission clean;
   - identity bit-exact in every cell;
   - every natural readout equals the baseline;
   - S/C0 replication 100%;
   - Test 2 admission clean.
9. **analysis** (§6), then archive and register.

**Power:** execution refuses to start on battery. Every model process runs under a continuous power guard that polls every 15 s and SIGSTOPs the process group on battery until mains returns. Every pause is recorded in `integrity.json`. This is the engineering correction for the C2AB-1 deviation.

**Protocol deviation (owner waiver, 2026-11-20):** the owner approved execution, and binding `321a86c9…` (test3-stage4-01) required mains power. While the Mac was on battery, the owner explicitly chose to waive mains power and rebind rather than wait.
- Binding 01 is preserved **unexecuted**.
- The orchestrator gained a bound `power_policy`. `bind --waive-mains` records `mains_waived_by_owner`. That binding lets execution start on battery, and its guard records battery intervals instead of pausing.
- Requalified as `test3-holdout-qual-02`. Rebound and re-prepared into the fresh `test3-stage4-02`, whose prepared population, donors and plans must be identical to 01's.
- Nothing else changes: recipes, population, arms, metrics and thresholds stay as they were. All integrity checks are bit-exact.

**Estimated cost:** about 112 seal and capture runs plus 392 derived arms plus about 168 replication arms, roughly 15–25 min on mains.

## 8. Qualification (done; development stand-in, no holdout read, no model run)

`output/gw-template-factorial/test3-holdout-qual-01/qualification.json`, re-run after the power-policy change as `test3-holdout-qual-02`. The 20 development-validation countries stand in for the holdout.

| Gate | Result |
|---|---|
| QH1 | eligibility(joint − holdout) reproduces the 59 Test 2 countries exactly (SG excluded as subject echo) |
| QH2 | donors reproduce the Test 2 donors exactly |
| QH3–QH5 | runs, seal plan and capture plan reproduce Test 2 / Stage 1b exactly; the capture plan is byte-equal |
| QH6 | the seal's last step equals the baseline natural readout in 236/236 runs |
| QH7 | stand-in derived plans: ID/S/C1/C2 vector ids equal Stage 2's; runner `--check` passes; a listed target refuses; admission 0 failures; every vector recomputes |
| QH8 | 9 planted defects are each rejected (holdout member in C1, C2 reading the target's currency window in the own term or in plus, incomplete fit membership, undeclared C0 donor, S pointing at a donor, missing arm, target on the refusal list, ineligible target) |
| QH9 | the replication plan equals Test 2's arms and runs; the Test 2 runner's `--check` passes, sealed |
| QH10 | Stage 4 analysis on Stage 2 rows reproduces all 120 Stage 2 per-arm records (S/C1/C2, validation, L1) exactly, and the gate points |
| QH11 | the capture view refuses holdout, validation-r' and other-target reads |
| QH12 | tests pass, including the continuous power-guard tests |

## 9. Not authorised

- C3 fitting;
- currency→capital holdout evaluation;
- layer sweeps;
- new candidate families;
- neuron editing;
- BREADCRUMB-2 confirmation;
- reselection after the holdout.

Holdout capture and inference wait for a separate execution GO.
