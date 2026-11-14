# PATCH-TRAIN-2: retention-aware sequential patch learning (confirmatory protocol)

**Status: DRAFT r2, 2026-11-14. NOT FROZEN. Not reviewed. No calibration, confirmation, native integration or optimiser search has been run
or started.** r2 replaces the r1 draft (`ef508c8f`) after RETAIN-1 (registered `c209ba0b`, results `1807596c`).
Branch `research/structured-real-1`, worktree `larql-structured-real`. CPU only. Gemma 3 270M (`google/gemma-3-270m-it` at
`ac82b4e8`), immutable. Extends `DESIGN.md` (§7 verdicts, §8 calibration); where this document is silent, `DESIGN.md` governs.
The frozen v2 numerics contract (`conformance/vlp-numerics/v2/`, commit `4b67591c`) is not changed by anything here.

## 0. Vocabulary that must not slip

- **emulation-consistent**: produced by the exact frozen v2 emulation (`add_low_rank_columns`, profile `f64-seq-rne1/v1`) in Python,
  self-reproducing and chain-replayable. This is the only numerical status any result here can have.
- **native compiled and independently verified**: built by the engine (#805) and checked by the independent verifier (#806 plus the
  new op kind). Neither exists for this chain. **No result in this document is, or may be called, "verified".**
- **retention of trained behaviours**: the first patch's own training prompt (and its paraphrases) still gives the first answer.
  This is *not* robust continual learning across unseen contexts, and nothing here claims it.

## 1. Claim under test (and its limits)

    W1 = Apply(W0, P_A)          P_A = Learn(W0, D_A)
    W2 = Apply(W1, P_B)          P_B = Learn(W1, D_B, retention obligations from P_A)      <- bound to W1

Claim: **with a declared retention objective, a second rank-1 patch learned against the first patched model acquires its association
(including on held-out wordings) while the first patch's behaviours are retained, without collateral or cross-entity leakage
beyond calibrated limits, on fresh independent entity pairs.** Out of scope: more than two patches, other layers/ranks/models,
scaling to many patches, example-free retention (§3), native verification (§9).

## 2. Evidence this protocol is built on (discovery only; NOT confirmatory)

PATCH-TRAIN-1 discovery, all emulated, all on `tasks_discovery.json`:

- Layer-3 rank-1 patches with a ridge-whitened address can learn a fictional entity's answer; each patch moves only its own entity
  (`single1`, `rep1`, `d2`).
- Independently learned patches do not reliably coexist on address-overlapping pairs (STRESS-CAP-1, rho 0.40: coexist 0/3 and 1/3
  pairs in the two arms; the most recently learned patch wins; no retention term).
- **RETAIN-1** (three pairs, both orders = six pair-order cases that share three entities; NOT independent), both training-prompt
  answers at full-vocabulary top-1:

| Arm | both top-1 | first patch retained | second learned |
|---|---|---|---|
| independent composition | 0/6 | 2/6 | 2/6 |
| new-task only (N) | 1/6 | 1/6 | 6/6 |
| rehearsal (R, lambda 1.0) | 6/6 | 6/6 | 6/6 |
| distillation (D, lambda 1.0) | 6/6 | 6/6 | 6/6 |

- **The observed trade-off.** R and D each preserved 13/14 of the first patch's isolated-top-1 prompts (N: 1/14, composition: 2/14),
  with first-patch log p within about 0.01 of isolated. But the **second patch's own held-out paraphrases that were top-1 fell
  from 8/18 (N) to 5/18 (R) and 5/18 (D)**. Retention protected the first association at some cost to how widely the second one
  generalises. Cross-entity partner leakage was **not** removed (up to +8.0 nats in pair 3 under D, +6.3 under R). Real-fact top-1
  stayed 1.00 and neutral NLL moved <= 0.01 in every arm.
- Limits: the three pairs were chosen by a rule that **maximises address overlap** (an adversarial selection, not a sample), rho was
  raised to 0.40 because 0.20 was cap-limited on those pairs, lambda was declared 1.0 and never tuned, "both top-1 on two training
  prompts" is a weak bar, and every number is emulation-consistent.

**None of these numbers may set a confirmatory threshold or choose a confirmatory parameter** (§7).

## 3. Methods, described exactly

| Arm | Second-patch objective on the first-patch model W1 | What it stores / needs |
|---|---|---|
| **N** new-task only | `L_B` | nothing |
| **R** rehearsal | `L_B + 1.0 * L_A`, `L_A` on the first patch's **original training prompt** | the first patch's training prompt (one string + target) |
| **D** distillation | `L_B + 1.0 * mean KL(P_W1 \|\| P_student)` over a retention set RET of full-vocabulary last-token distributions | the first patch's **training prompt** *and* the training-template prompts of a fixed set of other fictional entities; teacher = W1 |
| **C** independent composition | `P_B` learned on W0, rebound and re-applied on W1 | nothing |

**Distillation here uses a stored first-patch training prompt (it is in RET). It is therefore not example-free retention, and no
example-free claim is made.** A variant of D without the first patch's prompt (D0) is only an *exploratory* arm, in the
calibration phase and not in confirmation (§7, §11). Lambda is the single declared value 1.0 for R and for D; the weight is not
tuned on confirmatory data.

Both R and D give each patch a **retention obligation** (the stored prompts and targets, or the teacher), i.e. a patch carries
not only a weight update but what later patches must preserve. This is the design implication being tested at two patches. It is
**not** shown to scale: with many patches the retention set grows and obligations may conflict.

## 4. Fixed configuration (all phases)

PyTorch CPU, f32, eager attention, batch size 1, BF16 weights widened exactly to f32; layer-3 `down_proj` (640x2048); rank 1;
256 fixed columns (`seed 20261113`); ridge-whitened address from the **training template only** (`mech1.py`/`rep1.py`);
conditional-12 + 0.1 full-vocabulary CE; Adam, lr 3e-3, 200 steps; straight-through BF16-rounded forward; initialisation at half the
cap; cap from the **base W0** footprint RMS; final step is the only endpoint; reported weights are the exact frozen v2 emulation.
**rho = 0.40** (the value RETAIN-1 was registered at). It was chosen from discovery, which is disclosed and not reopened; CAL (§7)
checks isolated learning at 0.40 on untouched entities and it is not re-chosen afterwards. `retain1.py` supplies R and D unchanged.

**Same-layer note.** A and B both write layer 3 `down_proj`, whose input is computed before it; B's address on W1 equals its address
on W0, asserted bitwise at every position used (a failed assertion is a harness defect). Interference is therefore through layers
4..17 and through BF16 rounding on shared columns.

## 5. Confirmatory design

### 5.1 Entity pairs

- **N_pairs >= 8 required; 12 recommended** (see blocker B1), all drawn from `tasks.json` (unopened), **entity-disjoint across
  pairs**, disjoint from every discovery entity (D2, STRESS, RETAIN, CAL, the 22 retention-set entities) and from every
  calibration entity.
- **Deterministic rule**, written before `tasks.json` is read: take the eligible entities of the relation in file order, shuffle
  with a seeded PRNG (seed fixed in the frozen protocol, committed *before* the file is opened), pair consecutively, reject and redraw
  only for the stated reason "the two entities share the same assigned answer". **No address-cosine or any outcome-dependent
  selection.** The address cosine of every pair is *recorded*, and used only for descriptive stratification (§5.4).
- The unit of replication is the **entity pair**. Reverse orders, paraphrases, training prompts and patches are not replications.
- **Primary order** per pair is fixed by the same seeded draw (which entity is A). The reverse order is run as a secondary
  descriptive analysis and is never counted toward N_pairs.

### 5.2 Arms per pair (primary order)

Isolated references (A only, B only), **C**, **N**, **R**, **D**. Stages W0, W1 = A, W2 = A + B. Retention is judged against the first
patch's own isolated result per prompt, never against an ideal (nothing is required retrospectively to reach 3/3).

### 5.3 Measured at every stage, per pair

Target answer's full-vocabulary top-1, rank and log p on the training prompt and on **each** of the 3 held-out paraphrases
separately, for both entities; partner's answer on each entity; in-set leakage over all other entities; selectivity `S`; real-fact
(24) top-1 and log p; neutral NLL; realised-vs-ideal numerics; tensor sha256 per stage; base digest before and after; chain replay;
source-binding refusal.

### 5.4 Strata (descriptive, not gates)

Pairs split by recorded address cosine (above/below the pair-cosine median of the confirmatory draw) and reported separately; no
claim of effect is made unless pre-declared in the frozen protocol.

## 6. Separate verdicts

None implies another. Every pair gets each verdict; thresholds come from CAL (§7) and are frozen before CONF.

| Verdict | Question | Pass for a pair when |
|---|---|---|
| **L: training-prompt learning** | does each association learn in isolation and in the stack? | A and B top-1 on their training prompts in the isolated arms; both top-1 in W2 |
| **P: held-out paraphrase acquisition** | does B generalise to wordings it was not trained on, in the stack? | B's mean held-out-paraphrase log-p gain over W1 `>= f *` its isolated gain **and** above the null floor `T_null_gain` (§7); paraphrase top-1 counts reported alongside |
| **R: first-patch retention** | is A still there? | A's training prompt top-1; every A prompt (training + paraphrases) that was top-1 in A's isolated result still top-1; each A log-p drop `<= Delta_retain` |
| **X: leakage / selectivity** | does either patch spread? | the other entity's answer is not top-1 on the other entity's prompts; per-entity leak `<= isolated leak + T_null_leak`; in-set leakage rate within the calibrated bound |
| **G: collateral** | what else changed? | real-fact top-1 `>= 23/24`, neutral NLL `<= base + 0.15`, at every stage (the `DESIGN.md` §7 limits) |
| **V: numerical reproducibility** | is every stage exactly what its patches declare? | the emulation reproduces itself bit for bit; recompiling each chain prefix from W0 reproduces that stage's tensor; base digest unchanged. Status **emulation-consistent** |
| **S: source binding** | is each patch tied to its true predecessor? | applying B to a source whose digest differs from W1's (W0, or W1 with A removed) is refused |

**Pair-level primary endpoint** ("retention-aware coexist"): L, P, R, X, G, V and S all pass for the pair. Secondary: each verdict
alone, per arm.

### 6.1 Analysis

- Per arm: pairs with primary endpoint / N_pairs, **exact (Clopper-Pearson) 95% interval**, and the per-pair table (all numbers, not
  only counts). Per verdict the same. With 8 pairs, 8/8 has a lower bound of 63%; this is stated up front, not discovered later.
- **Comparisons are paired on pairs**: exact McNemar (one-sided) of R vs N, R vs C, D vs N, D vs C on the primary endpoint; Holm
  correction over those four; R vs D is two-sided and descriptive. An arm is not said to "work" unless its pair-level lower bound and
  its Holm-adjusted comparisons are as pre-declared in the frozen protocol.
- **Intention to treat**: pairs where A or B fails to learn *in isolation* stay in the denominator (they fail L); a secondary analysis
  conditions on both learning in isolation. No pair is excluded after the fact.
- The trade-off (§2) is a pre-declared secondary endpoint: paired difference in B's paraphrase top-1 count and log-p gain, R and D vs N.

## 7. Calibration (CAL) and freeze

CAL is a separate discovery phase on **entities never used by D2, STRESS-CAP-1, RETAIN-1 or any earlier run as trained entities**
(rule: the next unused discovery `trained_entities` by index, paired consecutively with differing answers, at least two pairs; the
earlier stress-pair results are **not** used to set any gate). The comparison entities seen in earlier runs are not patched here.

1. **Null patches** (no learning): `>= 200` random `(a, b)` of the same footprint, rank and norm as the learned patches, applied on
   W0 and on a learned W1, to get the distribution of every continuous metric under "no learning" (per-prompt log-p change, rank
   change, `S`, leakage, real-fact and NLL change).
2. **Floors from the null distribution only**: `T_null_*` = 99th percentile of the absolute null change on that metric.
   `Delta_retain` = `T_null` for first-patch log-p drift. The **method's own CAL results are used only to check feasibility and to
   report; they are never used to loosen a threshold.** If R or D cannot meet the null-derived floor, the floor stays and the pair
   rate is whatever it is.
3. **Isolated learning** at rho 0.40 on the CAL pairs, to check that the configuration learns on untouched entities. A failure is a
   negative result for this configuration, reported, with no re-tuning inside this protocol.
4. **Lambda.** Default and recommended: **lambda = 1.0 for R and D, fixed, no calibration.** If review wants the plasticity/retention
   trade-off addressed, a *registered* grid (e.g. {0.3, 1, 3}) can be run **on CAL pairs only**, with the selection rule (retention
   gate met, then maximal B paraphrase gain) written before running; CONF then uses the chosen value once. Not run until reviewed.
5. **Freeze**: entity-draw seed, N_pairs, orders, rho, steps, lambda, every threshold, every endpoint and the analysis plan are
   committed in a frozen protocol, with a hash recorded, **before `tasks.json` is opened.**

## 8. Declared failure outcomes (each reported as such)

- A or B does not learn in isolation on the fresh pairs: negative for this footprint, rank and rho.
- R and D no better than N or C on the pair-level endpoint: the discovery advantage did not replicate (it came from selected,
  overlapping pairs).
- Training prompts retained but P fails (second patch does not generalise): over-retention; the plasticity/retention trade-off,
  reported with the paraphrase endpoint, not rescued by changing lambda inside CONF.
- X fails: cross-entity leakage is not controlled by these retention objectives (it was not in discovery); reported.
- G fails: collateral. V or S fails: a defect, not a result.
- R succeeds and D fails (or the reverse): reported; they are not merged.

## 9. Emulated vs native

Everything runs in emulation until the engine implements `add_low_rank_columns` and the independent verifier contract exists (a
separate implementer; #805/#806 plus the new op kind are not final). CONF verdict V is **emulation-consistent** and the claim is
limited to *retention-aware two-patch behaviour in emulation*. "Native compiled and independently verified cumulative learning" is a
separate later milestone: a successful CONF chain would be compiled natively and checked by the verifier, and V upgraded only then.

## 10. Reviewer decisions of record

1. All D2 arms kept; confirmation needs at least eight independent entity pairs; the unit is the pair. (2026-11-14)
2. Discovery does not wait for native compilation; V is `emulation-consistent`. (2026-11-14)
3. Thresholds come from calibration set before confirmation; no discovery number sets one. (2026-11-14)
4. After RETAIN-1: commit the evidence, propose this retention-aware protocol, do not start confirmation, native integration or
   optimiser searches before review; describe D as using a stored first-patch prompt; record the 13/14 vs 8/18 vs 5/18 trade-off.

## 11. Open items for review (blockers and decisions)

- **B1 entity supply.** 8 pairs need 16 eligible entities, 12 pairs need 24, none overlapping discovery or the retention set. I have
  not opened `tasks.json`; if it has fewer, `build_tasks.py` must be extended with a new committed seed *before* freeze (counts only,
  no outcomes).
- **B2 paraphrase criterion.** The fraction `f` (proposed 0.5 of isolated gain) is a design choice, not derived from data; review
  must accept or replace it before freeze. RETAIN-1 showed R/D paraphrase top-1 at 5/18 vs 8/18, so P may fail broadly: that is
  informative, not a reason to relax it.
- **B3 leakage.** Retention did not control partner leakage in discovery (several nats). X may fail for all arms; the protocol
  reports that rather than redefining X.
- **B4 rho.** 0.40 was chosen from discovery on adversarial pairs; CAL checks it on untouched entities only.
- **B5 D0 and lambda grid** are optional discovery additions (§7.4, §3), unrun and awaiting review.
- **B6 scale.** Two patches only. Longer chains, patch C, retention-set growth and obligation conflicts are untested.
- **B7 native/verifier** milestone outstanding (§9).

## 12. Out of scope

Patch C and longer chains, a different layer or rank, address discovery without the target's own activation, optimiser tuning,
overlay patches, merging/consolidation, other models, example-free retention claims, and any engine or verifier change.
