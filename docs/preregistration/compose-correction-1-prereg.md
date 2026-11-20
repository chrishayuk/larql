# COMPOSE-CORRECTION-1 — Pre-registration

**Status:** 📋 v0.7 DRAFT — becomes FROZEN at the commit that lands it, after the
open items in §10 are closed. After that commit, frozen text is never edited;
changes land only as dated **Addendum** sections at the bottom, each stating what
changed and why, before the affected measurement runs.
**Revision log (draft only; not allowed after freeze).** v0.2: the Gemma admission check
(§4) showed Gemma 3's decoder keeps the leading space, so the v0.1 forecast F2 (A1 bit-identical
to A0 on Gemma), which rested on the opposite assumption, was replaced, and the exploratory
Gemma observations were added to §1 as inputs. v0.3: added §1.1 (exploratory versus
confirmatory), pinned arm builds, trajectory capture, the bank and its rules, and the
canonical-prompt note; none of these responds to a result. v0.4: registered fact 1 was observed during a development check
of the trace sink, so by the rule in §5 it is replaced by the first reserve and recorded; the harness
controls use separate development facts. v0.5: a second matcher failure mode was found on a development fact (§1.2), so every target was
re-authored under a tokenization rule (§5); this used tokenizers only and observed no registered
fact's behaviour. v0.6: status only (the harness's development-stage controls reproduce; the registered-set guard exists
and its first, refused, attempt is ledgered); no design change. v0.7: protocol revision on failure
atomicity (user decision, 2026-11-06), made before the freeze and before any registered run; see §7. Nothing else was changed in response to
results.
**Contains no outputs of this experiment.** Numbers below are inputs from earlier,
cited work, or thresholds fixed before any measurement.
**Implementation is not authorised** until this document is frozen and the prompt
bank (§5) is committed with its hash.

---

## 0. The question

> **Can `INSERT ... MODE COMPOSE` install a targeted behaviour while keeping
> acceptable predictions on related and unrelated prompts — choosing its own
> operating point, on both Qwen3-0.6B and Gemma 3 4B — and when it cannot, refuse
> without changing anything?**

This is separate from #701 (the leading-space token matcher). That fix is an input
here (arm A1), not part of the study.

## 1. Inputs (cited, not re-derived)

From the COMPOSE collateral-damage investigation (`~/chris-models/wsp001/compose/FINDINGS.md`,
PR #701), Qwen3-0.6B Q8_0, `INSERT Atlantis capital-of Poseidon MODE COMPOSE`, layer 26.
Baseline top-1 with no edit: Atlantis `at` 12.77%, France Paris 74.36%, Germany Berlin
72.80%, weather `very` 20.53%.

| Condition | Atlantis | France | Germany |
|---|---|---|---|
| default `ALPHA 0.1`, matcher unfixed | garbage | Pose 92.45% | Pose 99.99% |
| default `ALPHA 0.1`, matcher fixed | Pose 92.31% | `coln` 28.76% | `zheimer` 14.58% |
| `ALPHA 0.01`, matcher fixed | Pose 73.59% | Paris 72.27% | Berlin 72.05% |

Mechanisms found: (1) the balance probe missed a leading-space target token and could
never shrink an install; (2) the default alpha 0.1 and gate scale 30 come from Gemma 3 4B
and over-drive this model (slot gate score 286× a natural feature's typical score on the
Atlantis prompt, ≈100× on neighbouring "capital of" prompts); (3) balance reads a collapsed
target probability as "too weak", and checks no prompt except prior installs.
Same-template residuals are close (cosine 0.775 between Atlantis and France at the layer-26
FFN input), which is real entanglement, not a bug.

From the Gemma 3 4B admission check (exploratory, one fact, `gemma3-4b-it.vindex3`, BF16, 34
layers, default layer 32 and default alpha; baseline top-1 France Paris 80.05%, Germany Berlin
88.91%, identical on repeat):

| Matcher | Balance probe reads | Atlantis | France | Germany |
|---|---|---|---|---|
| old (A0) | 0.0000 ×3, amplifies | Pose 100% | Pose 100% | Pose 100% |
| fixed (A1) | 1.0000 on all 16 iterations | Pose 100% | Paris 80.43% | Berlin 89.12% |

Gemma 3's decoder is `Replace(▁→space) + ByteFallback + Fuse`: decoding one token alone gives
`" Pose"`, `" Paris"`, with the space, as on Qwen. So the matcher defect bites Gemma as well. With
the fix, Gemma's balance never leaves saturation: 16 shrink steps of 0.7 do not undo the
over-drive, and the target ends at 100%, above the 0.95 ceiling.

### 1.1 What is exploratory and what is confirmatory

Everything in §1 is **exploratory**: a handful of installs on one fact (Atlantis), looked at while
diagnosing. None of it is a result of this study. Forecasts F1–F5 were written from the Qwen
observations and F2 and F6 after seeing the Gemma ones, so they cannot be confirmed by those
observations. Their confirmatory value comes only from facts and prompts that were never looked at:
the exploratory prompts (Atlantis, France, Germany, "The weather today is") are excluded from the
bank by construction, and the bank is authored and hashed before any registered run.

**A registered fact was seen during development, and replaced.** While checking the trace sink on
Qwen3-0.6B with the old matcher, the controller's trajectory for the then-registered fact 1
(capital / Zorvania / Meridian) was printed and read (probabilities 0.0020, 0.0018, 0.0003, then
saturated and rolled back). That fact can no longer count as unseen, so it was replaced by the first
reserve under the rule in §5 and the observation is recorded in the bank (`observed_in_development`).
No registered fact was run under the fixed matcher or on Gemma before replacement.

One more exploratory fact matters for design. The relation used in the investigation, `capital-of`,
produces the canonical prompt `The capital of of Atlantis is` (the doubled "of" is pinned by an
existing test), so the slot was captured from that prompt while the results were read on
`The capital of Atlantis is`. The registered facts use bare-noun relations, so the captured prompt
and the evaluated prompt are the same.

### 1.2 A second matcher failure mode (exploratory)

The balance matcher accepts a decoded token that contains the target or starts with the target's
first three characters. A target whose first token is shorter than three characters can therefore
never match, even when the model's top prediction is exactly that token: `Tallow` splits to `' T'`
on both Qwen3-0.6B's and Gemma 3 4B's tokenizers, and on a development fact the controller read
0.0000 on both models. Several of the v0.4 targets had the same property on one or both
tokenizers (for example `Brannoch` → `' Br'`, `Lumen` → `' L'` on Qwen, `Cinder` → `' C'` on Gemma).

This is separate from the leading-space defect, is **not** fixed by #701, and is not fixed by this
study. It is a limitation of arm A1. The registered targets are therefore chosen so that the matcher
can see them (§5), which isolates the mechanisms under test from this defect. Whether to change the
matcher to compare token ids is a separate decision; the acceptance measures here already use ids.

## 2. Arms

Reference arms, no new code:

| Arm | Definition |
|---|---|
| **A0** | the commit on `main` immediately before #701 merges, built clean |
| **A1** | A0 + #701 (matcher fix), built clean. The all-off cell of the factorial below |
| **FA(α)** | A1 with `ALPHA` fixed at each of 0.1, 0.03, 0.01, 0.003, 0.001 and balance as in A1: the non-adaptive sweep |

Candidate mechanisms, each independently switchable:

- **B — balancing.** Bracketed search over the down scale in both directions; a target
  probability that has collapsed with the top-1 destroyed is treated as over-drive, not
  weakness; refuses when no candidate meets the band.
- **G — neighbour guard.** Before accepting, probe the fact's same-template neighbours and
  a fixed unrelated set drawn at install time from the tokenizer vocabulary (not from the
  evaluation bank); refuse if any baseline top-1 changes.
- **C — gate calibration.** Initial gate scale derived from the model's observed
  activation distribution at the install layer (a multiple of the median natural
  `|gate·x|`), in place of the fixed 30.

A0 and A1 are separate builds with identical model, tokenizer, container and inputs, and each is
recorded by commit so either condition can be reproduced independently of the other.

Factorial: the eight cells of {B, G, C} on top of A1. All eight are run and reported;
no cell is dropped after seeing results.

## 3. Hypotheses (forecasts fixed before measurement)

- **F1.** At the default alpha, A0 and A1 each produce an *installed-unsafe* outcome (§6)
  on ≥ 6 of 12 Qwen facts. Falsified by < 6.
- **F2.** On Gemma 3 4B, A0 produces *installed-unsafe* on ≥ 6 of 12 facts (neighbours hijacked
  by an unbalanced install), and A1 produces *installed-safe* on ≥ 9 of 12. Falsified by either
  count. (v0.1 forecast bit-identity of A1 and A0 on Gemma on the premise that its decoder strips
  the leading space; the admission check showed that premise false, so the forecast was withdrawn
  before any registered measurement.)
- **F6.** On both models, A1 leaves the target probability above the 0.95 ceiling on ≥ 6 of 12
  facts at the default alpha, because balance exhausts its iterations while saturated. Falsified by
  < 6. This is the case mechanism B (bracketed search) is meant to fix.
- **F3.** The fixed-alpha sweep on Qwen has a single alpha, in [0.001, 0.01], at which
  ≥ 9 of 12 facts are installed-safe. Falsified otherwise. If this fails there may be no
  safe operating point at layer 26, and the study reports that and stops before building
  B, G or C for Qwen.
- **F4.** C alone (A1 + C) reaches ≥ 9 of 12 installed-safe on Qwen.
- **F5.** G never raises the installed-safe count; it can only turn installed-unsafe into
  refused. (Follows from its definition; a violation is a bug.)
- **No forecast** for B's effect or for interactions; they are reported as found.

## 4. Models and containers

| Model | Container | Notes |
|---|---|---|
| Qwen3-0.6B | `~/chris-models/q8enc-ad82acba/qwen3-0.6b-q8_0.vindex3` | VINDEX3, Q8_0 |
| Gemma 3 4B | `~/chris-models/gemma3-4b-it.vindex3` (V3, BF16; model `093f9f388b31de276ce2de164bdc2081324b9767`) | admitted, see below |

**Admission check (done; read-only, container hashes identical before and after, no patch
written):** the container loads under VINDEX3 (34 layers, execution closed); `INFER` is
deterministic (France Paris 80.05% on repeat, Germany Berlin 88.91%); `INSERT ... MODE COMPOSE`
is admitted and writes a slot at L32/F3694 through the overlay path with refine, balance and
cross-fact passes applied; no refusal or unsupported capability was met. The documented
"10/10 retrieval, 0/4 regression bleed on Gemma 3 4B" was not reproduced as stated: one fact was
tried, retrieval worked, and the regression bleed depends on the matcher (above). The V2 arm
(`gemma3-4b-f16.vindex`) was not exercised and is not part of this study unless a later addendum
adds it.

Identity of each container (index digest, tokenizer digest) and the `larql` build commit
are recorded in the freeze commit and re-checked before every run.

## 5. Facts and prompt banks

The bank is `bench/compose-correction-1/bank.json`, produced by `make_bank.py` (deterministic;
two runs give identical bytes). Its SHA-256 is recorded at freeze; the generator and the file are
committed together so the hash is checkable. No part of it was chosen after seeing a result.

- **12 registered facts:** 4 relations (capital, ruler, currency, language) × 3 invented entities,
  one single-word target each, canonical prompt `The {relation} of {entity} is`. Install each fact
  alone in a fresh session; multi-fact interaction is out of scope.
- **4 reserve facts**, used in order only if a registered fact fails the baseline rule below.
- **Same-template neighbours:** 8 real entities per fact with the same relation.
- **Related neighbours:** 8 prompts per fact with a different template and a related subject.
- **Unrelated held-out bank:** 200 prompts in ten families (factual, arithmetic, code, sentence
  starters, Q&A, translation, lists, definitions, common phrases), disjoint from everything above.
- **Atomicity probe set:** every fourth unrelated prompt, 50 in all.

**Development facts.** The bank holds two further facts, `development`, which the harness uses for
its controls and tests and which are never scored. Anything seen while building or debugging the
harness must be a development fact. If a registered fact is observed in development anyway, it is
moved out of the registered set, replaced by the next unused reserve, and listed in the bank's
`observed_in_development` with what was seen, in the same commit. Replacement is by that rule only;
nothing is swapped because of how a fact behaved.

**Tokenization rule.** Every target (registered, reserve and development) must have a first token,
after dropping leading whitespace, of at least three characters that starts with the target's first
three characters or contains the target, on both the Qwen3-0.6B and the Gemma 3 4B tokenizers, and
registered targets must have distinct first tokens. `bench/compose-correction-1/check_targets.py`
checks this from the two `tokenizer.json` files; it reads no model. It must pass at freeze.

Baseline rules, applied mechanically before the install arms run: a fact is replaced by the next
unused reserve if its canonical prompt gives its target's first token a baseline probability ≥ 0.01 or if
two registered targets share a first token. Every rejected fact and every replacement is recorded
with the measurement that triggered it, so the final set can be audited against the generated one; a neighbour counts toward preservation only if its
baseline top-1 probability is ≥ 0.20 (otherwise it is reported, not scored). Baseline top-1 ids and
probabilities are measured once, before any edit, and stored with the bank.

## 6. Measurements

All acceptance measures use **token ids and probabilities from the execution path**
(`INFER`), never the decoded-string matcher under study.

- **Installed-safe:** the canonical prompt's top-1 id equals the target's first token id
  with p ≥ 0.30, **and** every unrelated-bank prompt keeps its baseline top-1 id, **and**
  ≥ 90% of same-template neighbours keep their baseline top-1 id with |Δp| ≤ 0.05, **and**
  no neighbour's top-1 becomes the target token.
- **Installed-unsafe:** installed, but any of those fails.
- **Refused:** the install returned an error. Whether it left the session unchanged is not assumed:
  it is measured and reported separately (§7.1), and the failed session is discarded.
- **Atomicity violation:** a failed install changed any component of the session fingerprint (§7.1),
  reported by component. Expected on both arms.
- **Balance trajectory (recorded for every install, every arm):** each attempted scale, the target
  probability read at it, the probe's matched token, the effect on the fact's same-template
  neighbours and a fixed unrelated subset at that scale, and whether the controller accepted or
  rejected the step. The point is to separate *never found a safe scale* from *found one and did
  not select it*. Neighbour effects at each step are measured by the harness from the executed
  state, not by the controller.
- Reported, not gated: mean and max KL(baseline ‖ edited) on the unrelated bank at the
  last position; Δp distribution on neighbours; balance iterations; whether the final target probability exceeds the 0.95 ceiling; install
  latency.

## 7. Hard gates

1. **Failure atomicity: a measured outcome, not a precondition.** A failed COMPOSE install is known
   not to be atomic on **both** existing arms: the install writes the decoy cache, the fact's raw
   residual and the overlay slot, and rewrites the layer's other slots, before the steps that can
   fail, and never records the edit in the active patch if it fails after the slot is staged (PR #706
   records the defect, with a real-model demonstration). Waiting for a repair would change the
   implementations being compared, so the study runs on them as they are and **measures** atomicity.
   - *How it is measured.* For each arm and model, with the seven fault points of PR #706 injected
     into installs of the **development facts** (atomicity is a property of the implementation, not of
     a fact, and development facts keep the registered ones unseen): the session's state fingerprint
     (`Session::state_fingerprint`: overlay vectors and counts, weight edits and patches,
     `installed_edges`, the raw-residual and decoy caches, the active patch recording, the mutation
     counters) and the logits on the atomicity probe set (bit for bit) before and after. Any component
     that differs is a violation, listed by name. The expectation is violations on both arms; a result
     of none would itself need explaining.
   - *Natural failures in registered runs.* Any registered install that returns an error is recorded
     with the failed session's fingerprint against the one taken just before the call. It is a
     `refused` outcome (§6), never silently dropped.
   - *Isolation.* Every registered install runs in a fresh session. A session in which an install
     failed is discarded: it is never reused for another install or for any later measurement.
   - *Eligibility versus acceptance.* Atomicity does not decide whether an arm may be studied. It
     does decide production acceptance: no COMPOSE correction is accepted for production (§8)
     until the atomicity requirement passes. The repair (an install on isolated candidate state that
     publishes the whole change, including the patch recording, or nothing) is a separate change with
     its own validation. It is **not** an arm of this study and is not added to A0 or A1.
2. **Determinism.** Installing the same fact twice from a fresh session gives a
   bit-identical overlay digest.
3. **Wiring.** Every arm's run records which mechanisms were active as observed facts of
   the run (the plan that executed, the alpha and gate scale actually written), not as an
   environment variable or flag name.

## 8. Acceptance

A candidate cell is **accepted for a model** when all hold:

- the determinism and wiring gates pass (atomicity is reported, see §7.1);
- installed-unsafe = 0 of 12 facts;
- installed-safe ≥ 9 of 12 facts on Qwen3-0.6B;
- on Gemma 3 4B: no fact that A1 installed safely becomes unsafe or refused (paired exact
  sign test on the 12 facts, reported), and installed-safe is no lower than A1's.

Accepting a cell does not make it the default; that is a separate decision with its own PR. Accepting a
cell is a research finding, not production acceptance: production acceptance additionally requires the
atomicity requirement of §7.1 to pass, which no arm of this study can satisfy today.
The fixed-alpha sweep is the *non-adaptive reference*: its best alpha is chosen on the same
12 facts and is therefore optimistic, so it is a comparison point, not a significance test.

## 9. Analysis

Main effects of B, G and C and their interactions are reported from the eight cells;
per-fact outcomes for every arm are published. Paired comparisons (same fact, same model)
use exact tests; a bit-identical result is reported as such and is *not* counted as
evidence of significance. A refused install is a result, not a missing value; the refusal
rate is reported next to the installed-safe rate so that a cell cannot look safe by
refusing everything.

## 10. Open items before freeze

1. ~~Confirm Gemma admission~~ — done, see §4. Remaining: re-run it on the frozen build and record its commit.
2. ~~Author the 12 facts and banks~~ — drafted (`bench/compose-correction-1/`); hash recorded at freeze.
3. ~~Write the harness and prove it on a control~~ — development stage done (PR #705): all six recorded
   controls reproduce to two decimals on 18 probes under the A0 and A1 builds on both models; the
   saved install is identical with and without the observer; a `--set registered` run is refused and
   ledgered (`registered-attempts.jsonl`). Remaining: build each arm from a single commit that
   contains the harness, with no tracked changes, and re-run the controls on those builds.
4. Measure and store all baselines; confirm `INFER` is bit-deterministic on the machine used.
5. ~~Add a trajectory sink and the fault-injection hooks~~: the sink is merged (PR #704); the hooks and
   the state fingerprint are PR #706, inert when unarmed (an install with an unreached fault armed is
   byte-identical to a plain one, also on the real container). Remaining: the harness must record the
   fingerprint before and after every install and run the §7.1 fault matrix on the development facts.
6. Pin the A0 and A1 commits and the `larql` build for each run.

## 11. Limits

Arm A1 keeps the short-first-token blind spot of §1.2; the targets avoid it by construction, so the
study says nothing about how COMPOSE behaves for targets the matcher cannot see.

One install layer (the default, 26 on Qwen); first-token targets only; canonical-template
prompts; a small model and a quantised container; a bank authored by us. A positive result
shows an adaptive install can find a safe point on these facts, not that COMPOSE is safe
in general.

---

## Addenda

*(none)*
