# GW-ELEMENT-1: chemical element → symbol vs atomic number. Model-free feasibility and draft protocol

**Status: DRAFT for owner review. Not frozen; nothing run.** This is the stage authorised by the owner's GO of 2026-11-21: model-free feasibility and protocol design only.
- No model was loaded.
- The only tool used was the pinned Gemma 3 4B tokenizer (`tokenizer.json`, sha256 `4667f208…`, the same file as Test 3).
- Audit script: `scripts/gwelem_feasibility.py`.
- Output: `output/gw-element-1/feasibility-01/feasibility.json`.

## 1. Hypothesis under test

**Transfer of the construction method, not of the displacement.**

The C2 recipe is refitted on new fit entities, using that domain's own relation pair:

> h(Y, r, p) + mean over F∖Y of h(e, r′, p) − mean over F∖Y of h(e, r, p)

The capital→currency displacement vectors are **not** reused. If it works, the claim is: "the same way of building a relation state, applied to a different entity domain and relation pair, switches the answer." It is **not** "one universal relation vector."

## 2. Audit findings

### 2a. Atomic-number answers are not entity-specific at the first token. This blocks the symmetric design

Gemma tokenizes a number as a bare space followed by **single digits**:
- 8 → `" "`, `"8"`
- 26 → `" "`, `"2"`, `"6"`
- 118 → `" "`, `"1"`, `"1"`, `"8"`

| Answer | First-token classes over 118 elements | Max multiplicity | Single-token answers |
|---|---|---|---|
| Atomic number | **1** (`" "` for every element) | 118 | 0 |
| Chemical symbol | 114 | 2 | 112 |

The runner scores the **next token** only. So:
- **symbol → atomic number** (inject the number relation and read the number) cannot be measured. Every element's "correct" first token is the same `" "`.
- Ending the prompt with a trailing space still yields only the first digit, which gives 10 classes. That is not entity-specific, and it is an unnatural tokenisation.

### 2b. Atomic number → symbol is measurable

Here the target prompt asks for the atomic number and the injected relation is "chemical symbol". The switch target is the element's own symbol token, e.g. `" O"`, which is unique for 110/118 elements.
- The **switched-from** answer r(Y) is the shared `" "`. So "staying on the source relation" is a format token, not an entity-specific fact. The margin m = log p(symbol(Y)) − log p(`" "`) is still well defined.
- **Entity specificity** comes from the r′ side. B3 measures the excess on other elements' symbols, and "other_entity_r′" top-1 detects wrong-element answers.
- This matches Test 3's structure. There, capital→currency was primary and the reverse failed B3.

### 2c. Phrasings and window alignment

A window is the tokens after the subject, up to the readout `is`, right-aligned between the two relations as in Tests 2 and 3. All 118 elements were checked for each candidate.

| Id | Symbol prompt | Number prompt | Window (symbol / number) | Length | ids constant | Subject echo |
|---|---|---|---|---|---|---|
| **A2** | For {x}, the chemical symbol is | For {x}, the atomic number is | `, the chemical symbol` / `, the atomic number` | 4/4 for all 118 | yes | 0 |
| **A3** | {X}'s chemical symbol is | {X}'s atomic number is | `' s chemical symbol` / `' s atomic number` | 4/4 for all 118 | yes | **9** symbol echoes (Al, Sc, Y, I, Pr, Eu, Os, U, Fm) |
| B2 | For {x}, the symbol is | For {x}, the number is | 3/3 | yes | 0 | "the number" is ambiguous; rejected |
| C2 | For the element {x}, the chemical symbol is | … the atomic number is | same window as A2 | 4/4 | yes | 0 | nearly a duplicate of A2 |
| C3 | In chemistry, the symbol for {x} is | … the atomic number of {x} is | **empty** (the relation precedes the subject) | — | — | — | rejected: no window |
| D2 | For {x}, the element symbol is | For {x}, the proton number is | 4/4 | yes | 0 | "proton number" is unnatural; rejected |

- **Proposed phrasings: A2 and A3.** They are structurally different (comma + `the` against a possessive), and the windows align slot for slot. Slot 3 is the relation word (`symbol`/`number`) and slot 2 the modifier (`chemical`/`atomic`).
- Subject token ids are identical across the two relations in every phrasing (118/118).
- Subject lengths: 63 elements are 1 token, 26 are 2, 23 are 3 and 6 are 4.

### 2d. The secondary span from CP3c

CP3c's minimal span was "determiner through the end of the relation phrase, without the comma". It maps as follows:
- **A2:** `the chemical symbol`, window slots 1–3. That is 3 positions against the full window's 4, so it is a real secondary test.
- **A3:** the determiner is the possessive `'s` (`'`, `s`). The span `'s chemical symbol` **is the full window**, so the secondary hypothesis is **identical to the primary in A3**. It is informative only in A2.

The protocol must say this in advance. One alternative is a second comma-phrasing in place of A3. "For the element {x}, …" (C2) has the same window as A2, so that alternative would give two near-duplicate phrasings.

### 2e. Entity eligibility from tokens alone (fixed now, before any baseline)

| Exclusion | Elements | Reason |
|---|---|---|
| Multi-token symbol | Yb, Hf, Tl, Np, Lr, Sg | The first token (` Y`, ` H`, ` T`, ` N`, ` L`, ` S`) is another element's symbol. Excluding these also removes **every** first-token collision |
| A3 subject echo | Al, Sc, Y, I, Pr, Eu, Os, U, Fm | The symbol's token occurs in the subject's tokens. Excluded from the whole population, as Test 3 excluded subject-echo countries |

**103 elements are token-eligible.** Natural-baseline eligibility (§3) needs inference and is not authorised yet.

### 2f. Entity-disjoint split, frozen before any baseline

The split ranks all 118 elements by salted SHA-256, with salt `gw-element-1/split/v1|Z`. It is assigned over all 118 **before** eligibility, so eligibility cannot choose it.

| Split | All 118 | Use |
|---|---|---|
| fit | 59 | Constructing C1/C2 (the F∖Y means) |
| development | 24 | Runner qualification and integrity only; no effect readouts |
| confirmation | 35 | Never read until the confirmation run |

The assignment is in `feasibility.json`. The counts after token eligibility, and later natural eligibility, will be smaller. See the minimums in §5.

## 3. Natural-baseline eligibility (specified now; needs a separate inference GO)

An element is naturally eligible in phrasing p when both of these hold under the natural forward:
- **symbol prompt:** greedy top-1 equals its symbol's first token;
- **number prompt:** greedy top-1 is `" "`, and the greedy continuation's digits equal Z exactly, followed by a non-digit. The existing baseline runner provides an 8-token continuation.

An element must be eligible in **both** A2 and A3. Elements that fail are excluded and listed. Failures are not repaired by changing prompts.

Gemma 3 4B IT may answer in markdown (e.g. `**8**`). If the number prompt's top-1 is not `" "` for most elements, that is a **feasibility failure of this domain**, recorded as such (§5). The prompt is not tuned afterwards.

## 4. Proposed confirmation design (to freeze after the baseline, under its own GO)

- **Direction:** atomic number → symbol only. Target = number prompt; injected relation = chemical symbol; switch = top-1 equals symbol(Y). The reverse direction is **not measurable** (§2a) and is out of scope. It is reported as a design limit, not a failure.
- **Primary:** C2 at L4, the **full window** (4 positions), single carrier layer, band runner unchanged. The rule is set by CP3a–c; nothing is selected on the new domain.
- **Prespecified secondary:** C2 at L4 on `the chemical symbol` (window slots 1–3) in A2 only. It is not separately testable in A3 (§2d).
- **Controls:**
  - identity on every arm shape;
  - S (the natural symbol-prompt state) as the descriptive reference;
  - the comma-slot CTRL;
  - the separable reference, which is the natural readout of the same element's symbol prompt.
- **Thresholds:** the Stage 2 switching gates:
  - pooled lower 95% bound ≥ 0.80;
  - each phrasing ≥ 0.60;
  - other-entity top-1 ≤ 0.05.
- **Bias guard:**
  - **primary:** the original **absolute** B3 limits (mean ≤ 0.5, max ≤ 1.0);
  - **separately:** a non-worsening comparison against S, the natural symbol-state reference, on the same cells. The capital/currency full-window C2 reference is from another domain and is not comparable here, so S stands in for it.

  The relative comparison cannot rescue a failure of the absolute criterion.
- **Confirmation population:** the naturally eligible elements of the confirmation split. It is opened once and never reused.

## 5. Explicit outcomes of feasibility (fixed now)

The domain is **unsuitable**, and that is reported as the result with no prompt or population tuning, if any of these hold:
- fewer than **20** confirmation, **30** fit or **10** development elements are naturally eligible;
- the number prompt's natural top-1 is not `" "` for at least 80% of token-eligible elements in either phrasing;
- the window token ids are not constant over eligible elements (they are, for all 118, under A2 and A3);
- r(Y) and r′(Y) share a first token for any eligible element (impossible here, since `" "` is never a symbol).

## 6. Engineering needed before any inference (each needs qualification)

- **Plan builder** for element runs. The run ids are `number|A2|Z`, and the windows come from §2c.
- **Analysis classes:** `s2.Classes` and `per_arm` assume the relations capital/currency and country classes. They need a generalised entity and class map, with r(Y) = `" "` shared and r′ = symbol. This is a new analysis module that is unit-tested and leaves the frozen Test 3 code unchanged.
- **Runner:** the band runner's built-in refusal list is Test 3's spent holdout. It needs a per-plan refusal list for the element confirmation split. This is a runner change, so it **needs requalification** like compress1-band-qual-02, or an equivalent plan-level gate.
- **Capture:** natural window carriers at L4 (and all layers, for identity) for the fit, development and confirmation runs. Confirmation runs are captured only in the confirmation execution.

## 7. Decisions for the owner

1. Accept **atomic number → symbol** as the only measured direction. The alternative is to replace atomic number with a relation whose answers have entity-specific first tokens, which keeps the experiment symmetric.
2. Accept **A2 + A3**, knowing the secondary span is testable only in A2. The alternative is a different second phrasing.
3. Accept the token-level exclusions (15 elements) and the frozen split.
4. GO, or not, for the **natural baseline**: about 103 × 2 relations × 2 phrasings ≈ 412 natural forwards, no interventions. This is the first inference step.
