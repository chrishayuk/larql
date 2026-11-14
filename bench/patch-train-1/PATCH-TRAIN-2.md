# PATCH-TRAIN-2: sequential patch learning and retention

**Status: DRAFT 2026-11-14. The CONFIRMATORY protocol is NOT FROZEN and has not been run.** Reviewer decisions of 2026-11-14 are
recorded in §10: all seven arms are kept, the discovery dry-run (D2) is authorised, and the confirmatory phase needs at least
eight independent entity pairs. The confirmatory protocol is frozen only after the calibration phase (§7) and review, and before
any confirmatory run.
Branch `research/structured-real-1`, worktree `larql-structured-real`. CPU only. Gemma 3 270M (`google/gemma-3-270m-it` at
`ac82b4e8`), immutable. Extends `DESIGN.md` (§7 verdicts, §8 calibration); where this document is silent, `DESIGN.md` governs.
The frozen v2 numerics contract (`conformance/vlp-numerics/v2/`, commit `4b67591c`) is not changed by anything here.

## 1. Claim under test

Cumulative learning through a patch chain, without modifying the foundation checkpoint:

    W1 = Apply(W0, P_A)          P_A = Learn(W0, D_A)
    W2 = Apply(W1, P_B)          P_B = Learn(W1, D_B)      <- learned AGAINST W1, bound to W1

and both behaviours are present in W2. Two independently successful patches are not this claim. This document separates
four things that are easy to conflate: (i) learning each association, (ii) composing independently learned patches, (iii)
cumulative learning (B learned on A's output), (iv) retention and interference. Each gets its own verdict (§6).

## 2. What the discovery work already gives, and what it must not be used for

PATCH-TRAIN-1 discovery (all emulated, all on `tasks_discovery.json`, commits `1d4e1eac` .. `c7d726a1`) established:

- a layer-3 `down_proj` rank-1 patch `dW = a b^T` on a fixed 256-column footprint, with a ridge-whitened `b` built from one
  entity's training-template activation and `a` learned through a BF16-rounded forward (straight-through), can make a
  fictional entity's answer the full-vocabulary top-1 with real-fact top-1 and neutral NLL unchanged;
- the result holds for a second entity and a different answer (Frozoria -> Lyon, Dresalthaia -> Leeds; address cosine 0.032;
  a negative-control answer Bergen is learned equally well);
- each patch moves only its own entity (cross-application: the other entity's answer stays at rank >= 1350);
- the BF16-compiled result equals the f32 ideal to cosine >= 0.9999.

**These results informed this design and are NOT confirmatory evidence for it.** Frozoria and Dresalthaia are discovery
entities. Limits: one layer, one seed per entity, addresses constructed from each target's own activation (not a scalable
address discovery method), paraphrase transfer varies by entity (Frozoria 3/3 at rho 0.20, Dresalthaia 2/3: the possessive
template is not top-1), name-neighbour leakage of up to +3.5 nats exists at rho 0.20, and nothing has been compiled by the
engine or checked by the independent verifier.

## 3. Phases

| Phase | Entities | Purpose | Status |
|---|---|---|---|
| **D2** discovery dry-run | Frozoria (A), Dresalthaia (B) | find breakage in the arms, the harness and the metrics; descriptive only; no threshold is taken from it | **authorised 2026-11-14** (emulation-only, all seven arms) |
| **CAL** calibration | null patches, plus two further discovery entities (`trained_entities[2]`, `[3]`) as an A'/B' pair | set every threshold in §6 from the "no learning" and "isolated learning" distributions | proposed |
| **CONF** confirmatory | **at least eight** independent entity pairs drawn by a fixed rule, without replacement, from `tasks.json` (unopened until freeze); the unit of replication is the **entity pair**, never a paraphrase | the claim | **not run; protocol frozen only after CAL and review** |

D2 results are reported as discovery. CONF is the only phase that can support a claim, and its entities, answers, controls,
rho, steps, thresholds and endpoints are written to the frozen protocol before it runs.

## 4. Fixed configuration (all phases)

Declared backend (PyTorch CPU, f32 compute, eager attention, batch size 1, BF16 weights widened exactly to f32), layer 3
`down_proj`, rank 1, the same 256 columns (`seed 20261113`), the ridge-whitened address recipe of `mech1.py`/`rep1.py`
(training template only), conditional-12 + 0.1 full-vocabulary CE, Adam lr 3e-3, 200 steps, straight-through BF16-rounded
forward, initialisation at half the cap, **rho = 0.20** with the cap defined from the **base W0** footprint RMS (so the same
absolute cap in every arm and every stage), final step is the only endpoint. Reported weights are always the exact frozen v2
emulation (`add_low_rank_columns`, profile `f64-seq-rne1/v1`); the straight-through forward is an optimisation approximation
and says so in every record.

**Same-layer note (a protocol check, not an assumption).** A and B both write layer 3 `down_proj`, whose input `z` is computed
before that matrix. Applying A therefore cannot change the layer-3 input `z` of any position, so B's address is the same on W1
as on W0. The harness asserts this bitwise at every position of every prompt used (a failed assertion is a harness defect). The
interference that can occur is therefore (a) through layers 4..17 (payload interaction), and (b) through BF16 rounding on the
**shared** columns: B's update is rounded on top of A's already-rounded values, so an element may survive in one order and be
absorbed in the other. Both are measured (arms 7 and §5.3).

## 5. Arms

`P_X^W` denotes the patch for X learned with the training forward starting from weights `W`. "Apply" is always the emulator on
the stated source weights, with the source tensor digest recorded and checked (binding).

| # | Arm | Construction | Question |
|---|---|---|---|
| 1 | A only | learn `P_A^W0`; `W1 = Apply(W0, P_A)` | A learns (reproduces rep1 rho 0.20 bit for bit as a determinism check) |
| 2 | B only | learn `P_B^W0`; `Apply(W0, P_B)` | B learns |
| 3 | A then independent B | `W1`, then **re-apply** `P_B^W0`'s `(a, b)` to W1: `Apply(W1, P_B^W0)` (rebound to W1; its payload is recomputed against W1's footprint, never copied as W0's output bits) | does independent composition work |
| 4 | B then independent A | symmetric: `Apply(Apply(W0, P_B^W0), P_A^W0)` | order dependence of composition |
| 5 | A then B learned on A | learn `P_B^W1` with the training forward starting from W1; `W2 = Apply(W1, P_B^W1)` | **cumulative learning** (the claim) |
| 6 | B then A learned on B | symmetric: learn `P_A^(W0+P_B)`; apply | cumulative learning in the other order |
| 7 (diagnostic) | joint single stage | `W_joint = BF16(W0 + dA + dB)`, one stage, rank-2 `[a_A a_B] [b_A; b_B]`, one rounding, versus arm 3's `W_AB = BF16(BF16(W0 + dA) + dB)` | separates "learning interference" from "sequential BF16 rounding on shared columns"; **differences in behaviour alone do not establish the cause**, so the exact per-element weight differences between `W_AB`, `W_BA` and `W_joint` are recorded and every one of those weight sets is evaluated on the same model and prompts |

Arms 1-5 are the five requested; 6 and 7 are kept by reviewer decision (order symmetry for the cumulative arm, and the rounding
control). No arm is learned with a different objective, learning rate, cap or step count from arm 1.

### 5.1 Reference performances (retention is defined on what was actually achieved)

Retention for A is measured against A's own arm-1 result, per prompt (training prompt and each of the three paraphrases): a
prompt that is full-vocabulary top-1 in arm 1 must be top-1 in the stack, and the target log p may fall by at most
`Delta_retain` (set in CAL). Likewise B against arm 2. **Nothing is ever required to reach 3/3 retrospectively**: if B's
isolated result is 2/3 paraphrases, B's reference is those 2/3 plus its other prompts' log p, and the cumulative arms are judged
against that.

### 5.2 Stages and what is measured at each

Stages: `W0`, `W1 = A`, `W2 = A+B` (and the order-swapped chain). At **every stage**, for **both entities**:

- full-vocabulary top-1, rank and log p of the target answer on the training prompt and on **each** held-out paraphrase
  separately; in-set top-1, conditional log p, answer-set mass;
- both entities' **non-target** answers (does B's `Leeds` appear on Frozoria and vice versa), and the continuous selectivity
  `S = d log P(y|target entity) - mean over the other entities d log P(y|x)`, with per-entity leakage over all other entities;
- real-fact top-1 and log p (24 prompts), neutral NLL, and which in-set answer each entity's prompt currently selects;
- numerics: realised vs ideal update (cosine, Frobenius ratio, survival fraction), the tensor sha256 of every stage, payload
  digest per patch, and the base-checkpoint digest before and after.

### 5.3 Rounding control

For each applied patch the harness records, per element of the footprint, whether B's update survives on W1 and on W0, and the
set of elements whose survival differs. Arm 7 compared with arm 3 gives the sequential-rounding contribution.

## 6. Separate verdicts

None implies another (extends `DESIGN.md` §7). Thresholds `Delta_*` come from CAL (§7) and are frozen before CONF.

| Verdict | Question | Pass when (frozen from CAL) |
|---|---|---|
| **V: numerical** | is every stage exactly what its patches declare, and is each patch bound to its true predecessor? | emulation reproduces itself bit for bit; recompiling from W0 with each patch-chain prefix reproduces that stage's tensor exactly; a patch applied to a source whose digest differs is **refused**. *Native compilation by the engine and the independent verifier are NOT available (#805/#806, and the new op kind); every result here is labelled "emulated" and V is "emulation-consistent", never "verified".* |
| **L_A, L_B: learning** | did each association learn in isolation? | arm 1 / arm 2: training prompt top-1 and the per-paraphrase counts at or above the CAL floor |
| **R: retention** | after the second patch, is the first still there? | §5.1: per-prompt top-1 retained and log-p drop `<= Delta_retain`, in arms 3, 4, 5, 6, for each patch as "first" |
| **C: cumulative learning** | does a patch learned against the first patched model learn, and retain? | arm 5 (and 6): B's per-prompt results at or above its arm-2 reference minus `Delta_learn_gap`, **and** R holds for A, **and** G holds |
| **X: interference / selectivity** | does either patch spread? | the other entity's answer is not top-1 on the other entity's prompts; `S` per patch within `Delta_S` of its isolated value; in-set leakage rate within `Delta_leak`; max per-entity leak `<= Delta_leakmax` |
| **G: regression** | what else changed? | real-fact top-1 `>= 0.96` (at most one of 24 prompts lost), neutral NLL `<= base + 0.15`, per stage |
| **Rev: reversibility** | can any stage be recovered exactly? | replaying the chain prefix from W0 reproduces W1 bit for bit; **removing A from the stack makes B's binding stale and the harness refuses B** (B is not silently valid on W0); "rebase" = re-learn, then re-verify |

The sequential-vs-independent comparison is reported, not gated: arms 3/4 vs 5/6 vs 7 are descriptive of **how** the stack
fails or succeeds.

## 7. Calibration (CAL) and freeze

Before any CONF run:

1. **Null patches**: random `(a, b)` of the same footprint, rank and norm as the learned patches, many draws, to give the
   distribution of every metric under "no learning" (per-prompt rank and log-p change, `S`, real-fact and NLL change). Set
   `Delta_S`, `Delta_leak`, `Delta_leakmax` and the noise floor for "no change" from it.
2. **Isolated learning on the CAL pair** (discovery entities 2 and 3): the same arms 1-6 as D2, to set `Delta_retain`,
   `Delta_learn_gap` and the per-paraphrase floors for L from the observed spread of isolated and cumulative results.
3. **Freeze.** Write the CONF entity pair (rule-drawn from `tasks.json`, unopened), answers, negative-control answers,
   comparison entities, rho, steps, every threshold and every endpoint into the frozen protocol, commit it, and only then run.

The run is deterministic on the declared backend, so run-to-run noise is zero; the noise that matters is sampling over
entities. The confirmatory phase therefore uses **at least eight independent entity pairs**, and every rate is over pairs; one
pair supports "worked once", not a rate. CAL uses discovery entities `[2]` and `[3]` only if neither has influenced any tuning
choice. Both were comparison entities in every earlier run (their leakage was observed; entity `[2]`, Pyldredreia, was among the
largest leakers) but neither was trained on, nor was any parameter, threshold or footprint chosen from them; if the reviewer
regards that as influence, CAL instead takes an untouched discovery pair chosen by the predeclared rule 'the next two unused
`trained_entities` by index'.

## 8. Declared failure outcomes (each reported as such)

- A or B does not learn in isolation under the frozen configuration: negative result for this footprint and rank, no
  cumulative claim.
- Both learn, composition (arms 3/4) fails but cumulative (arms 5/6) succeeds: independent patches do not compose, sequential
  learning does: reported with the arm-7 and §5.3 rounding evidence.
- Composition and cumulative both fail on R: interference. Reported with which element survival or which downstream layer
  (§4 same-layer note) explains it, if the data say.
- Order dependence (3 differs from 4, or 5 from 6): non-commutative; reported.
- X fails (leakage grows, e.g. onto name neighbours): reported; this is expected to worsen as patches accumulate.
- G fails: collateral. V fails: a defect, not a result.

## 9. Emulated vs native, and the two claims

Everything runs in emulation until the engine implements `add_low_rank_columns` (E1-E3 in `DESIGN.md` §9) and the independent
verifier contract exists (V1, separate implementer). **Discovery does not wait for them.** Until then **no result may be called
"verified"**: the word is `emulation-consistent`. Two claims are kept apart: *two-patch behavioural retention demonstrated*
(both behaviours survive sequential learning; achievable first) and *numerically verified cumulative patch learning* (the
engine and the independent verifier reproduce the whole chain; a separate later milestone). If the native path is
still unavailable at CONF, the CONF verdict V is "emulation-consistent" and the claim is limited accordingly; it is not
upgraded afterwards without re-running V on the engine.

## 10. Reviewer decisions (2026-11-14)

1. Arms 6 and 7 are kept; all seven arms run.
2. Confirmation targets **at least eight** independent entity pairs, sampled by a fixed rule without replacement; the unit of
   replication is the entity pair.
3. Calibration uses discovery entities `[2]` and `[3]`, subject to the condition in §7 (fallback: the next two unused entities).
4. Discovery does not wait for native compilation; the numerical verdict is explicitly `emulation-consistent`.
5. Every threshold comes from calibration, set before confirmation; no Frozoria/Dresalthaia number is used to choose one.

Still open: the number of confirmatory pairs beyond the minimum of eight, the fixed sampling rule for them, and the form of the
`Delta_*` thresholds (absolute vs relative); settled at the CAL review.

## 11. Out of scope

Patch C and longer chains, a different layer or rank, address discovery without the target's own activation, optimiser
tuning (any blocker found in D2 is reported and decided separately), overlay patches, merging/consolidation, other models, and
any engine or verifier change.
