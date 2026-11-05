# COMPOSE-CORRECTION-1 — Pre-registration

**Status:** 📋 v0.1 DRAFT — becomes FROZEN at the commit that lands it, after the
open items in §10 are closed. After that commit, frozen text is never edited;
changes land only as dated **Addendum** sections at the bottom, each stating what
changed and why, before the affected measurement runs.
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

## 2. Arms

Reference arms, no new code:

| Arm | Definition |
|---|---|
| **A0** | `main` before #701 |
| **A1** | A0 + #701 (matcher fix). The all-off cell of the factorial below |
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

Factorial: the eight cells of {B, G, C} on top of A1. All eight are run and reported;
no cell is dropped after seeing results.

## 3. Hypotheses (forecasts fixed before measurement)

- **F1.** At the default alpha, A0 and A1 each produce an *installed-unsafe* outcome (§6)
  on ≥ 6 of 12 Qwen facts. Falsified by < 6.
- **F2.** On Gemma 3 4B, A1 is bit-identical to A0 on every install (SentencePiece decoding
  strips the leading space, so the matcher change is inert there). Falsified by any
  difference in the installed vectors.
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
| Gemma 3 4B | `~/chris-models/gemma3-4b-it.vindex3` (V3 arm); `~/chris-models/gemma3-4b-f16.vindex` (V2 arm) | which arm(s) are run: open item 1 |

Identity of each container (index digest, tokenizer digest) and the `larql` build commit
are recorded in the freeze commit and re-checked before every run.

## 5. Facts and prompt banks

Authored and hashed before any implementation; the hash is part of the freeze.

- **12 facts:** 4 relations × 3 entities, entities chosen so the baseline model does not
  already produce the target. Each fact has a canonical prompt and a one-token target.
- **Neighbours:** ≥ 8 same-template prompts per fact with real entities, plus ≥ 8
  different-template prompts about related subjects.
- **Unrelated held-out bank:** 200 prompts across topics and formats, disjoint from the
  facts, neighbours and the install-time guard set.
- **Atomicity probe set:** 50 prompts from the unrelated bank.

All baseline top-1 token ids and probabilities are measured once, before any edit, and
stored with the bank.

## 6. Measurements

All acceptance measures use **token ids and probabilities from the execution path**
(`INFER`), never the decoded-string matcher under study.

- **Installed-safe:** the canonical prompt's top-1 id equals the target's first token id
  with p ≥ 0.30, **and** every unrelated-bank prompt keeps its baseline top-1 id, **and**
  ≥ 90% of same-template neighbours keep their baseline top-1 id with |Δp| ≤ 0.05, **and**
  no neighbour's top-1 becomes the target token.
- **Installed-unsafe:** installed, but any of those fails.
- **Refused:** the install returned an error and left state unchanged (below).
- **Atomicity failure:** a refused or failed install changed anything (below).
- Reported, not gated: mean and max KL(baseline ‖ edited) on the unrelated bank at the
  last position; Δp distribution on neighbours; balance iterations; install latency.

## 7. Hard gates

1. **Failure atomicity (per arm, per model).** After every refused install, and after
   faults injected at each phase (after slot insertion, after refine, during each balance
   iteration, during the guard probe, during the cross-fact check): the overlay's
   serialised digest, `installed_edges`, `raw_install_residuals`, the decoy cache, and the
   logits on the atomicity probe set (bit-for-bit) are identical to their values before the
   call. One violation disqualifies the arm on that model regardless of its other results.
2. **Determinism.** Installing the same fact twice from a fresh session gives a
   bit-identical overlay digest.
3. **Wiring.** Every arm's run records which mechanisms were active as observed facts of
   the run (the plan that executed, the alpha and gate scale actually written), not as an
   environment variable or flag name.

## 8. Acceptance

A candidate cell is **accepted for a model** when all hold:

- both hard gates pass;
- installed-unsafe = 0 of 12 facts;
- installed-safe ≥ 9 of 12 facts on Qwen3-0.6B;
- on Gemma 3 4B: no fact that A1 installed safely becomes unsafe or refused (paired exact
  sign test on the 12 facts, reported), and installed-safe is no lower than A1's.

Accepting a cell does not make it the default; that is a separate decision with its own PR.
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

1. Confirm `gemma3-4b-it.vindex3` admits and runs `INSERT ... MODE COMPOSE` and `INFER`
   (earlier notes record an admission gap for Gemma 3 under VINDEX3); if it does not, the
   Gemma leg runs on the V2 arm and B, G and C are ported or the leg is declared V2-only.
2. Author the 12 facts, the neighbour sets, the unrelated bank, and hash them.
3. Write the harness and prove it on a control: it must report the known Qwen result for
   Atlantis at `ALPHA 0.01` and at the default.
4. Measure and store all baselines; confirm `INFER` is bit-deterministic on the machine used.
5. Define the fault-injection hooks for §7.1 without changing install behaviour when unused.

## 11. Limits

One install layer (the default, 26 on Qwen); first-token targets only; canonical-template
prompts; a small model and a quantised container; a bank authored by us. A positive result
shows an adaptive install can find a safe point on these facts, not that COMPOSE is safe
in general.

---

## Addenda

*(none)*
