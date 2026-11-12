# PATCH-TRAIN-1: successive, automatically learned low-rank patches against an immutable base

**Status: DRAFT 2026-11-12. The experimental protocol is NOT frozen.** The competence gate (§3) has been measured on the
baseline model only. Nothing has been trained and no model has been modified.
Branch `research/structured-real-1`, worktree `larql-structured-real`. CPU only. Gemma 3 270M (`google/gemma-3-270m-it`
at `ac82b4e8`), immutable.

## 1. Claim under test

An optimiser can **directly learn the parameters of a column-scoped low-rank patch** (not train weights and then convert them
into a patch) that improves a measurable behaviour; a **second** patch, learned against the *compiled output of the first*,
improves a second behaviour while the stack **retains** the first improvement and stays inside a regression budget.

Four things are claimed or not claimed separately, with separate verdicts (§7): numerical verification, training
improvement, retention, and regression. A pass on one says nothing about the others.

## 2. What is already frozen and what it gives

- **The operation** is `add_low_rank_columns`, frozen at commit `4b67591c` (`conformance/vlp-numerics/v2/`, 91 cases,
  independently reproduced 91/91 blind). `W'[r, J[p]] = chain(widen(W[r, J[p]]) + Σ_k a[r,k]·b[k,p])`, sequential in `k`,
  one rounding to BF16, `J` a strictly increasing column set on the stored `[out, in]` input axis, `a` is `rows × K`, `b` is
  `K × |J|` compact, no scalar alpha (a producer folds it into `a`, `b`). An element whose terms are all zero keeps its
  base bits. **This document does not change that contract.**
- **A stack is a chain of source-bound links.** A patch is bound to its source container by `index_sha256` and the payload
  digest. Patch B is learned on the container compiled from patch A, is bound to A's output, and **may edit the same
  columns**: column-disjointness is enforced only *within* one compilation stage (v2 SPEC §8). Removing A makes B's binding
  stale and the verifier refuses B, so dependency tracking is a property of the binding. "Rebase" means re-learn or
  re-apply against the new state and re-verify.
- **Reversal** is recompilation from the base with a prefix of the chain; nothing is overwritten.
- The learning system emits the **same `.vlp` structured operations** the editing system compiles and verifies. There is no
  training-specific patch format.

**Not yet true** (checklist §9): the engine does not implement the new kind, there is no verifier contract for it, and
neither is started. Until they exist, patches in the learning track are **emulated** (§6) and every result is labelled
"emulated" until the engine reproduces the emulation bit for bit.

## 3. The competence gate, and what was measured (baseline only)

`competence/build_tasks.py` writes `competence/tasks.json` (deterministic, seed 20261112); `competence/run_baseline.py`
writes `competence/baseline.json`. The model is never modified.

**Declared backend, used for every behavioural comparison:** PyTorch 2.14 on CPU, float32 compute, eager attention,
batch size 1, no padding, the container's BF16 weights widened exactly to f32. LARQL's `INFER` is a cross-check only:
the 270M model has a documented strict-f32 equivalence issue between LARQL and the reference, so the two are never assumed
interchangeable. Environment note: the installed `torchvision` (0.21) does not match `torch` (2.14) and breaks the
transformers import; it is masked through `sys.modules` (no package changed), and this is recorded in `baseline.json`.

**Candidate tasks** (all answers are one token including the leading space, so a prompt's loss is one cross-entropy; all
entities are fictional, so the facts cannot already be known):

| Task | Relation | Entities | Answers |
|---|---|---|---|
| A | "The capital of {e} is" (fictional country) | 24 (16 trained, 8 held out) | 12 towns, 2 entities each |
| B | "The occupation of {e} is" (fictional person) | 24 (16 trained, 8 held out) | 12 occupations, 2 entities each |

Each task has the training template, three held-out paraphrases, and a held-out *entity* set (same relation, same
templates, never trained). Collateral sets: 24 real country-capital facts (format check and preservation), and 8 neutral
sentences (per-token NLL).

**Measured, 2026-11-12 (`baseline.json`):**

| Gate | Criterion (set now, before any training) | Result |
|---|---|---|
| G1 identity | all 236 container tensors equal the HF checkpoint bit for bit | **met** (236/236) |
| G2 determinism | two complete runs bitwise identical | **met** |
| G3 format competence | real-fact top-1 ≥ 0.9 | **met**: 24/24, mean log p(target) −0.043 |
| G4 headroom | full-vocabulary top-1 ≤ 0.05 on every fictional split, and in-answer-set top-1 not significantly above chance (1/12; one-sided exact binomial p > 0.01) | **met**: full-vocabulary top-1 is 0 on every split, mean log p(target) −12.7 to −18.9; in-set top-1 1/16, 7/48, 0/8, 1/24 (A) and 1/16, 2/48, 1/8, 2/24 (B), smallest p = 0.10 |
| G5 baselines recorded | neutral NLL per token | 3.4746 |

LARQL `INFER` agreed with the declared backend on the top-1 token of 12 of 12 real-fact prompts (informational).

**Limits of this evidence.** n is small (8 to 48 prompts per split) and the three paraphrases of an entity are not
independent, so an item-level bootstrap over *entities* is the interval to use, not a binomial over prompts. Passing the
gate says the task can show an effect; it does not say 270M can learn the fact through a rank-K patch on a given footprint.
That is the experiment.

## 4. The two training representations

- **Compiled BF16 patches** (the experimental target here). The low-rank update is rounded to BF16 as the frozen v2
  contract defines, compiled into VINDEX3 and verified.
- **Higher-precision overlay patches** (future, **out of scope**). The factors stay in higher precision and are applied at
  inference, so many small updates can accumulate before consolidation. It may be needed because of the constraint below,
  but its semantics are a different execution mode, not an equivalent of the compiled one: it must not be called equivalent
  without measuring both the numerical and the behavioural difference.

**The constraint that shapes the optimiser.** BF16 has an 8-bit significand: an additive update smaller than about half an
ulp of the weight (relative ~2⁻⁹) rounds to no change, and gradient-sized steps usually are. The compiled model is
therefore not the optimiser's f32 solution. Consequences: the loss is measured on the **rounded** result; the optimiser is
rounding-aware (§5); the **optimiser-to-compiled gap** is recorded for every patch; and a feasibility pre-check reports,
for the declared footprint, how much of any plausible update survives rounding.

## 5. The learning method (D3) and its baselines

**Direct patch optimisation.** The parameters are `a` (`rows × K`) and `b` (`K × |J|`) on a declared footprint. Forward
passes use the **emulated BF16-rounded** weights `W' = emulate(W, a, b)`; gradients use a straight-through estimator
across the rounding. A **rounding-aware local refinement** then searches the discrete outcome: it perturbs or sparsifies
`(a, b)` and keeps a change only if the objective on the *emulated, rounded* patch improves.
Objective: cross-entropy on the first answer token for the patch's training prompts (the training template, trained
entities), with an optional preservation term on collateral prompts that every arm receives equally if it is used.

**Matched-budget arms** (same footprint, same parameter count, same evaluation budget, same backend):
unedited base; the learned patch; a **random patch** of the same footprint (negative control); **f32 training then one
rounding** of the same parameters (the "train conventionally, convert" baseline, which may be strong and is mandatory);
and the same with the rounding-aware refinement only. Later arms (gradient-free search, reinforcement learning) emit the
same patch format and enter the same comparison; they are not part of the first run.

## 6. Emulation, and why the engine is not a blocker for the learning track

The optimiser and the calibration need the frozen v2 arithmetic in Python, fast enough to run in a loop. The emulation is a
vectorised implementation that must be **bit-exact** to the frozen contract:

1. it reproduces all 91 frozen vectors;
2. it agrees with the independent evaluator (`conformance/vlp-numerics/v2/independent/evaluate.py`) on a large random
   sample, including signed zero, ties and near-cancellation;
3. every accepted patch is later reproduced bit for bit by the **engine** and passes the **independent verifier**; until then
   every result carries the label "emulated".

Rounding from the f64 working value to BF16 is one rounding (profile `f64-seq-rne1/v1`); it must not pass through f32.
The declared backend's behavioural evaluation of a compiled result loads the compiled container's tensors into the same
model (the container-to-checkpoint mapping is already proved by G1).

## 7. Separate acceptance criteria

A patch or stack gets **five independent verdicts**. None implies another.

| Verdict | Question | Pass when |
|---|---|---|
| **V: numerical** | Is the compiled artefact exactly what the patch declares? | compiled by the engine, **independent verifier SATISFIED** per link against its predecessor, emulation reproduced bit for bit, recompiling from the base with only patch A reproduces container A byte for byte |
| **L: improvement** | Did the intended behaviour improve? | on confirmatory held-out paraphrases of the trained entities, the primary metric rises by at least Δ_improve over the base and the item-bootstrap interval excludes zero |
| **S: specificity** | Did it learn the facts, not a bias? | held-out *entities* of the same relation gain no more than Δ_specific (a rise there means a shifted answer prior, not knowledge) |
| **R: retention** | After patch B, is A's improvement still there? | metric_A(A+B) ≥ metric_A(A) − Δ_retain, and still above the base by more than the noise bound |
| **G: regression** | What else changed? | real-fact top-1 drop ≤ Δ_real, neutral-NLL rise ≤ Δ_nll, and task-B items before B are unaffected by A beyond Δ_cross |

**Cumulative learning** is claimed only if, together: B improves task B *on top of A* (L for B measured against the A-only
model), R holds for A, G holds for the stack, and V holds for both links. Primary metric: in-answer-set top-1 on
held-out paraphrases of trained entities; secondary: mean in-set margin and log p(target). One primary metric is fixed at
the freeze; the others are reported, not selected after the fact.

## 8. Thresholds, calibration and integrity (D5)

- **Thresholds are set from a separate calibration phase, then frozen.** Calibration uses **discovery entities**
  (a different seed from the confirmatory entities, disjoint names) and **null patches** (random operands, same footprint
  and K, norm-matched), giving the distribution of every metric under "no learning". Δ values are set from that
  distribution and from item-bootstrap widths, never after seeing a confirmatory result.
- The run is deterministic on the declared backend, so run-to-run noise is zero; the noise that matters is *sampling over
  entities*, handled by bootstrap over entities.
- **Failure outcomes are declared:** the optimiser cannot reach Δ_improve on the discovery set (a negative result about
  this footprint and K, reported as such); L holds but S fails (bias, not knowledge); L holds but G fails (collateral); A
  holds then R fails after B (interference); V fails (a defect, not a result).
- The footprint (tensor, columns `J`, `K`) is fixed **before** the first confirmatory run. There is no 270M mechanistic
  evidence to choose it from (the existing evidence is for Gemma 3 4B and did not replicate on 12B), so the default is a
  fixed FFN `down_proj` of a middle layer with a seeded column subset, and the choice is a discovery-phase decision frozen at
  calibration. No candidate is selected after observing confirmatory results.
- Every run records the full loss trajectory, patch sizes, rejected candidates and why, optimiser cost, and the
  optimiser-to-compiled gap.

## 9. Dependencies and checklist (not started; none is part of this task)

Engine and verifier work **depends on #805 (engine) and #806 (verifier) being available**, and must not change the frozen
numerical contract.

| # | Item | Notes |
|---|---|---|
| E1 | Interpreter kind `add_low_rank_columns` in `larql-vindex`, with a Rust test running all 91 frozen vectors | semantics exactly v2; refusals by the v2 codes |
| E2 | Generalise `StructuredWeightOp` to carry `rank`, `a`, `b`, `columns` and the kind; patch JSON parse and validation; digest | binding and footprint (= `J` over all rows) unchanged |
| E3 | Conflict rule: disjoint columns within a stage; sequential stages may overlap | already the engine's by-column rule; add the test for sequential overlap |
| E4 | LQL statement (optional) | the harness can write the `.vlp` directly |
| V1 | Verifier contract for the new op, **built by a separate implementer from the frozen spec**, with the SPEC §8 fault-injection tests (flip inside `J`, change outside `J` incl. a `−0`→`+0` flip, other tensor, stale binding, swapped `a`/`b`, wrong `b` layout) | not written by this session |
| L1 | Bit-exact vectorised emulation of v2 and its parity tests (§6) | learning track |
| L2 | Compiled-container loader into the declared backend | learning track; mapping proved by G1 |
| L3 | Feasibility pre-check, discovery/calibration, null-patch distribution | learning track |
| L4 | Optimiser, arms, adjudicator, then the frozen protocol | learning track |

## 10. Out of scope

Overlay patches, agent-generated training data and reward-driven patch optimisation (the intended next step if this works),
patch merging and consolidation, cross-model exchange, quantised representations, models other than 270M, any claim
that the learned patch means anything mechanistically, and any engine or verifier change.
