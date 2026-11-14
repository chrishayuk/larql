# PATCH-TRAIN-2: retention-aware sequential patch learning (confirmatory protocol)

**Status: DRAFT r3, 2026-11-14. NOT FROZEN. Not reviewed. No calibration, data generation, confirmation, native integration or
optimiser search has been run or started.** r3 revises r2 (`8827975f`) after the reviewer decisions in §10. r2 followed RETAIN-1
(registered `c209ba0b`, results `1807596c`). Companion drafts: `PATCH-TRAIN-2-CALIBRATION-PLAN.md`, `PATCH-TRAIN-2-DATA-PLAN.md`.
Branch `research/structured-real-1`, worktree `larql-structured-real`. CPU only. Gemma 3 270M (`google/gemma-3-270m-it` at
`ac82b4e8`), immutable. Extends `DESIGN.md` (§7 verdicts, §8 calibration); where this document is silent, `DESIGN.md` governs.
The frozen v2 numerics contract (`conformance/vlp-numerics/v2/`, commit `4b67591c`) is not changed by anything here.

## 0. Vocabulary that must not slip

- **emulation-consistent**: produced by the exact frozen v2 emulation (`add_low_rank_columns`, profile `f64-seq-rne1/v1`) in Python,
  self-reproducing and chain-replayable. This is the only numerical status any result here can have.
- **native compiled and independently verified**: built by the engine (#805) and checked by the independent verifier (#806 plus the
  new op kind). Neither exists for this chain. **No result in this document is, or may be called, "verified".**
- **two contracts.** A *numerical contract* says exactly which weights change and how (frozen v2 emulation, bit-exact replay, source
  binding). A *behavioural contract* says what the patch is meant to teach and which behaviours later patches must preserve (the
  retained prompts or teacher of §3). Both are needed; this protocol tests the behavioural contract at two patches and the
  numerical contract only in emulation.
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
example-free claim is made.** A variant of D without the first patch's prompt (D0) is **out of scope** here and is left to a separate later
experiment (§12). Lambda is the single declared value 1.0 for R and for D; the weight is not
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

### 5.1 Entity pairs and the new dataset

- **Target: 12 independent, entity-disjoint confirmatory pairs** (24 patched entities). Fewer than 12 only if the generator cannot
  produce them, and then the shortfall is reported before freeze, not after (§6.2).
- **`tasks.json` and `tasks_discovery.json` are reclassified as DISCOVERY data.** The discovery work evaluated entities of both files
  (and everything else in `tasks_discovery.json`) as leakage comparisons, and those observations informed the layer, address and
  cap choices. **No entity from either file, including entities used only as leakage controls, is reused.** The earlier statement
  that `tasks.json` is "unopened confirmatory data" is withdrawn.
- **A new dataset is generated by a fixed, committed generator with a new committed seed** (`PATCH-TRAIN-2-DATA-PLAN.md`): the
  generator, schema, sampling rule, exclusion check and acceptance rules are committed *before* the generator is run or any
  example is examined, and are not changed after any model has been run on the data. The generator produces, from separate seeds,
  a calibration split and a confirmatory split that are entity-disjoint from each other and from the discovery files.
- **Pairing rule** (deterministic, outcome-independent): the generator's confirmatory entities are paired consecutively in generation
  order; answers are assigned so that pair members always differ (a generator rule stated in advance, so no pair is redrawn after
  the fact). No address cosine or other model quantity enters selection; the address cosine of each pair is *recorded* and used
  only for the descriptive strata (§5.4).
- The unit of replication is the **entity pair**. Reverse orders, paraphrases, training prompts and patches are not replications.
- **Primary order** per pair is fixed by the generator's seeded draw. The reverse order is a secondary descriptive analysis and is
  never counted toward the 12.
- The retention set RET for D and the leakage-comparison entities are drawn from a **third, disjoint generated split** (also in the
  data plan), never from the pair entities and never from discovery.

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
| **P: held-out paraphrase acquisition** | does B generalise to wordings it was not trained on, in the stack? | B's mean held-out-paraphrase log-p gain over W1 `>= f *` its isolated gain, **f = 0.5, fixed**, **and** above the null floor `T_null_gain` (§7). Paraphrase top-1 counts (0-3 of 3) are reported separately, not gated |
| **R: first-patch retention** | is A still there? | A's training prompt top-1; every A prompt (training + paraphrases) that was top-1 in A's isolated result still top-1; each A log-p drop `<= Delta_retain` |
| **X: leakage / selectivity** | does either patch spread? | **both** sub-gates pass, each reported alone: **X_stat** (statistical, §6.3) and **X_func** (functional, §6.3) |
| **G: collateral** | what else changed? | real-fact top-1 `>= 23/24`, neutral NLL `<= base + 0.15`, at every stage (the `DESIGN.md` §7 limits) |
| **V: numerical reproducibility** | is every stage exactly what its patches declare? | the emulation reproduces itself bit for bit; recompiling each chain prefix from W0 reproduces that stage's tensor; base digest unchanged. Status **emulation-consistent** |
| **S: source binding** | is each patch tied to its true predecessor? | applying B to a source whose digest differs from W1's (W0, or W1 with A removed) is refused |

**Pair-level primary endpoint** ("retention-aware coexist"), **fixed here and never redefined after any outcome is observed**: L, P, R, X, G, V and S all pass for the pair. Secondary: each verdict
alone, per arm.

### 6.1 Analysis

- Per arm: pairs with primary endpoint / N_pairs, **exact (Clopper-Pearson) 95% interval**, and the per-pair table (all numbers, not
  only counts). Per verdict the same. With 12 pairs, 12/12 has a lower bound of 73.5% (§6.2); this is stated up front, not discovered later.
- **Comparisons are paired on pairs**: exact McNemar (one-sided) of R vs N, R vs C, D vs N, D vs C on the primary endpoint; Holm
  correction over those four; R vs D is two-sided and descriptive. An arm is not said to "work" unless its pair-level lower bound and
  its Holm-adjusted comparisons are as pre-declared in the frozen protocol.
- **Intention to treat**: pairs where A or B fails to learn *in isolation* stay in the denominator (they fail L); a secondary analysis
  conditions on both learning in isolation. No pair is excluded after the fact.
- The trade-off (§2) is a pre-declared secondary endpoint: paired difference in B's paraphrase top-1 count and log-p gain, R and D vs N.

### 6.2 Statistical feasibility for 12 pairs (computed now, before any data)

Exact one-sided McNemar on the primary endpoint is a sign test on the discordant pairs `d` (pairs where exactly one arm succeeds).
With Holm over four comparisons the first threshold is 0.0125 and the last 0.05. Exact computations (independent pair outcomes):

| Fact | Value |
|---|---|
| discordant pairs needed, all one way, at alpha 0.0125 / 0.0167 / 0.025 / 0.05 | d >= 7 / 6 / 6 / 5 (a 6-0 split fails the first Holm step) |
| two-sided exact 95% interval for 12/12, 11/12, 9/12, 6/12, 3/12 | 73.5-100%, 61.5-99.8%, 42.8-94.5%, 21.1-78.9%, 5.5-57.2% |
| power at alpha 0.0125 / 0.05, method succeeds w.p. 0.9, comparator 0.1 | 0.95 / 0.99 |
| same, 0.8 vs 0.1 | 0.81 / 0.95 |
| same, 0.8 vs 0.3 or 0.7 vs 0.2 | 0.35 / 0.64 |
| same, 0.6 vs 0.2 | 0.18 / 0.44 |
| same, 0.5 vs 0.1 | 0.18 / 0.48 |

So 12 pairs can show a **large** difference (a high pair-level success rate against a low comparator) but are **underpowered** for
moderate ones. The primary endpoint is a conjunction of seven verdicts, including P and X which RETAIN-1 suggests may often fail, so
the realised rates may be low and the comparisons inconclusive. **This is a declared limitation. The sample will not be expanded and
the analysis will not be changed after outcomes are seen**; an inconclusive result is reported as inconclusive, with the per-pair
table and the intervals. Per-verdict rates are secondary and descriptive: they diagnose a failed primary endpoint and never replace it.

### 6.3 Leakage: statistical change and functional interference are separate, independently gated

Measured for every patched entity and every **unrelated** entity (the pair's partner plus all RET/comparison entities of the third
split), on the training prompt and on each held-out template, at each stage, against the model before that stage (W0 for A, W1 for
B) **and** against W0. All of these are always reported:

- continuous: change in log p of each patched answer on each unrelated prompt (nats), `S`, in-set conditional change;
- functional, **F1**: the unrelated prompt's full-vocabulary top-1 token changed; and whether it became a patched answer;
- functional, **F2**: the unrelated prompt's in-set (answer-set) argmax changed; and whether it became a patched answer;
- functional, **F3**: change in the patched answer's margin, `log p(y) - max over the other in-set answers`, on the unrelated prompt.

Gates (frozen from CAL, §7):

- **X_stat**: the continuous log-p changes are at or below the null floors `T_null` (99th percentile of the null-patch distribution).
- **X_func**: (a) **zero** unrelated prompts whose full-vocabulary top-1 *becomes a patched answer*; this defines harm and is not a
  number taken from data; (b) the F2 in-set argmax-flip rate is at or below the null 99th percentile of that rate; (c) the F3 margin
  increase of a patched answer on unrelated prompts is at or below `T_null_margin`.

A patch can pass X_stat and fail X_func (a broad null makes X_stat lenient) and the reverse; they are reported separately and X
passes only if both do. If a null distribution is so broad that its floor would allow a changed answer, the functional gate still binds.

## 7. Calibration (CAL) and freeze

CAL is a separate discovery phase on a **generated calibration split** (`PATCH-TRAIN-2-DATA-PLAN.md`), entity-disjoint from the
discovery files and from the confirmatory split. The stress-pair results set no gate. Detailed steps: `PATCH-TRAIN-2-CALIBRATION-PLAN.md`.

1. **Null patches** (no learning): `>= 200` random `(a, b)` of the same footprint, rank and norm as the learned patches, applied on
   W0 and on a learned W1, to get the distribution of every continuous and functional metric under "no learning".
2. **Floors from the null distribution only**: `T_null_*` = 99th percentile of the absolute null change (the flip rate for F2).
   `Delta_retain` = `T_null` for first-patch log-p drift. The method's own CAL results are used only to check feasibility and to
   report; **they are never used to loosen a threshold.** If R or D cannot meet a floor, the floor stays.
3. **Isolated learning** at rho 0.40 on the CAL pairs, to check the configuration learns on untouched entities. A failure is a
   negative result for the configuration, reported, with no re-tuning inside this protocol.
4. **lambda = 1.0 for R and D, fixed. No loss-weight grid is run, in CAL or CONF.** `f = 0.5` is fixed and not tuned.
5. **Freeze**: generator, seeds, pair count, orders, rho, steps, lambda, `f`, every threshold, the primary endpoint and the analysis
   plan are committed in a frozen protocol with a recorded hash, before the confirmatory split is generated.

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

1. All D2 arms kept; the unit is the entity pair. (2026-11-14)
2. Discovery does not wait for native compilation; V is `emulation-consistent`. (2026-11-14)
3. Thresholds come from calibration set before confirmation; no discovery number sets one. (2026-11-14)
4. After RETAIN-1: commit the evidence, propose this protocol, do not start confirmation, native integration or optimiser searches before
   review; describe D as using a stored first-patch prompt; record the 13/14 vs 8/18 vs 5/18 trade-off. (2026-11-14)
5. **After r2 (2026-11-14):** 12 independent entity-disjoint pairs from a **new generated dataset** (fixed committed generator and seed;
   `tasks.json` reclassified as discovery; no reuse of any discovery entity including leakage controls); `f = 0.5` fixed; lambda 1.0
   with no grid; D0 out of scope; functional leakage (F1-F3) measured and gated separately from statistical change; seven verdicts
   plus a combined primary endpoint that is never redefined; the statistical plan's feasibility for 12 pairs is stated before freeze
   with no post-hoc expansion; native compilation and independent verification remain a separate milestone, and
   `emulation-consistent` is never called verified; calibration and confirmation wait for review of this protocol and the generator.

## 11. Open items

- **Resolved by r3:** entity supply (new generator), `f` (0.5), D0 and the lambda grid (out of scope).
- **Power (§6.2):** 12 pairs detect only large differences; moderate ones are inconclusive. Accepted as a declared limitation.
- **Leakage:** retention did not control partner leakage in discovery; X_stat and X_func may fail for every arm and will be reported.
- **P:** RETAIN-1's 5/18 vs 8/18 suggests P may fail broadly, with the primary endpoint following; that is informative.
- **rho 0.40** was chosen from discovery on adversarial pairs; CAL checks it on untouched entities only.
- **Scale:** two patches only; longer chains, retention-set growth and obligation conflicts are untested.
- **Native/verifier milestone** outstanding (§9).
- **Reviewer to approve before CAL:** the generator spec and exclusion rule, the calibration plan, and the functional-leakage gate forms.

## 12. Out of scope

Patch C and longer chains, a different layer or rank, address discovery without the target's own activation, optimiser tuning, loss-weight grids,
D0 (distillation without a stored original prompt; a separate later experiment), overlay patches, merging/consolidation, other models, example-free retention claims, and any engine or verifier change.
