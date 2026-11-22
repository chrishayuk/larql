# GW-ELEMENT-1: prospective confirmation protocol, atomic number → chemical symbol (FROZEN-READY, r2)

**Revision r2 (2026-11-21, owner decision "Option 1").** `confirm-qual-02` (NOT QUALIFIED, preserved) found that at layer 4 window slots 0–1 (A2 `, the`; A3 `' s`) precede the first relation-specific token. Those slots are therefore **bit-identical** between the number and symbol prompts for all 60 fit and development elements in both phrasings, so at slots 0–1, C2 and S equal the target's own natural carrier. As a result:
- the r1 control `C2-ctrl-L4` (slot 0) was an exact identity;
- the r1 A2 secondary `C2-np-L4` (`the atomic number`) was numerically the same intervention as the primary.

Both are **removed**, with no replacement hypothesis. The primary is unchanged, and its effective span is disclosed (§4). Admission gains the rule that every intended non-identity arm differs bit-exactly from its matched natural carrier at one or more designated positions (slots 2–3); identity arms are exempt. This degeneracy does not occur in capital→currency: slot-0 C2 differs from identity in CP3c, so CP3c is unaffected.

**Status: frozen-ready (r2).** Written under the owner's GO of 2026-11-21: "confirmation-protocol freezing, engineering and independent qualification only".
- It is hash-bound by the qualification record (`output/gw-element-1/confirm-qual-03/qualification.json`).
- It becomes the binding protocol when the owner authorises execution. The execution binding then re-hashes it, and any edit invalidates that binding.
- **No constructed-state inference has run on this domain.** The confirmation elements have never been captured, intervened on or analysed for effects.

## 1. Question and claim

Does the C2 construction method, refitted on chemical elements, switch Gemma 3 4B IT's answer from the atomic-number format to **the target element's own chemical symbol**? It is tested on 25 reserved elements, with a single post-FFN carrier at layer 4 over the relation window.

| A pass would support | A pass would not show |
|---|---|
| Discovery 1: the method of constructing a relational control state transfers to an unrelated factual domain. Because only the relation-specific tokens (slots 2–3) are numerically changed, this is evidence for a **portable relation-state intervention**, not necessarily for independent transfer of a whole entity–relation representation | A shared mechanism across relation types (discovery 3) |
| | A learned representation transferred across domains (discovery 2). The displacement vectors are refitted here, not reused |
| | A literal graph or a universal relation operator |
| | Replacement of one entity-specific fact by another. The source answer's first token, a bare space, is shared by every element. So switching shows that the injected state supplies the element's own symbol against a strong format preference (median −12 to −15 logits). Own-element specificity (other-element top-1, B3) is therefore primary evidence, not secondary |

**Disclosure.** The 25 confirmation elements were evaluated naturally, for eligibility, in `baseline-01` (results 6315/6316). They are **intervention-unseen and effect-unseen**: excluded from fitting, development and every selection. They are not unseen by every part of the programme, and the final report states this.

## 2. Frozen inputs

| Item | Frozen value |
|---|---|
| Model | `gemma3-4b-it.vindex3`. `index.json`, `system_graph.json` and `tokenizer.json` (`4667f208…`) are bound by hash |
| Split | salted 118-element split `gw-element-1/split/v1\|Z` (feasibility artifact 2748, sha `068ecc94…`) |
| Exclusions | 15 token-level and 18 natural, unchanged, never replaced |
| Fit F (45) | 1 2 4 6 9 11 12 15 16 17 19 20 29 30 32 37 38 41 44 46 48 52 55 56 57 60 62 65 66 67 73 74 77 83 85 88 89 94 98 104 105 108 112 115 118 |
| Development (15) | 3 8 14 18 25 31 35 40 42 43 45 78 82 84 107. Used for integrity and qualification only; no switching effect is ever computed on them |
| Confirmation (25) | 5 7 22 23 24 26 27 28 36 47 49 58 61 64 68 79 80 86 87 90 95 96 97 101 110 |
| Sealed split (35) | the 25 above plus the 10 ineligible members of the confirmation split (10 21 53 63 71 92 93 102 113 116). Sealed in the runner (entity key and subject tokens in four surface forms) and in admission |
| Phrasings | **A2** `For {x}, the atomic number is` / `For {x}, the chemical symbol is`; **A3** `{X}'s atomic number is` / `{X}'s chemical symbol is` |
| Direction | atomic number → chemical symbol only. The target prompt asks for the atomic number; the injected relation is the chemical symbol |
| Window | the 4 tokens between the subject and the readout `is`, aligned slot for slot. A2: `, the atomic number`; A3: `' s atomic number` |
| Carrier | post-FFN residual at **layer 4 only** (`from_layer 5`, `to_layer 4`), readout never replaced (CP3a–c rule; nothing is selected on this domain) |
| Designated positions | slots 2–3 (`atomic number` / `chemical symbol`), the only slots where the relations differ at L4 (r2) |

### Population-level checks

The populations are re-derived from the registered records and must equal the lists above (`gwelem_design.verify_populations`).

## 3. Candidate and information rule

    C2(Y, p, j) = h(Y, number, p, j) + mean_{e∈F} h(e, symbol, p, j) − mean_{e∈F} h(e, number, p, j)     at layer 4, slot j

**Numerics:**
- each mean is a float64 sum over F in ascending atomic number, divided by 45 and cast to f32 once;
- C2 = f32(float64(own) + (float64(mean⁺) − float64(mean⁻)));
- formula id `gw-element-1/c2/v1`.

**Information rule (the core integrity condition).** The target's own **symbol-relation** state never enters a candidate. It is enforced three times, independently:
1. **Builder** (`gwelem_candidates.CandidateView`): the view is built from an allow-list holding the fit runs, in both relations, and the targets' **number** runs. Any other read raises.
   - At execution, it is loaded from the fit capture and the confirmation **number** capture only.
   - The confirmation symbol runs are captured into a separate directory that the candidate view never opens.
2. **Admission** (`gwelem_admit`, which shares no code with the builder). Every C2 recipe must have:
   - exactly one `own` input, the target's own number window;
   - plus inputs equal to F's symbol windows, and minus inputs equal to F's number windows;
   - members equal to F.

   Admission rejects any input that is the target's symbol window, any non-fit aggregate member, and a target in F.
3. **Recompute and dependency audit:**
   - every C2 is recomputed bit-exactly from the fit and number captures only; the target symbol capture is not loaded by that recompute;
   - for each (phrasing, slot), every C2's aggregate inputs are identical across all targets. The displacement is therefore one target-independent vector, and the only target-specific input is the target's own number state.

**References** are built through a separate accessor and are never candidates:
- **S:** the target's own symbol window injected at L4. It is used for the non-worsening comparison and as a descriptive switching reference.
- **ID:** the run's own window at L4, a bit-exact identity.

## 4. Arm inventory (exact; r2)

Per confirmation element Y:

| Target run | Arms |
|---|---|
| `number\|A2\|Y`, `number\|A3\|Y` | `ID-full-L4` (identity), **`C2-full-L4`** (primary, slots 0–3), `S-full-L4` (reference, slots 0–3) |
| `symbol\|A2\|Y`, `symbol\|A3\|Y` | `ID-full-L4` only. Its natural readout is the **separable reference** |

- **Totals: 200 arms over 100 targets:** 100 identity arms (50 on number runs, 50 on symbol runs) and 100 constructed or reference arms (C2-full 50, S-full 50). There are 4 vectors per arm.
- **Effective span (disclosed).** In C2-full and S-full, slots 0–1 are bit-identical to the natural carrier, so the injection numerically changes only slots 2–3. The arm is still declared and executed over the full window, the pre-registered primary. A second "noun phrase" span is not separately testable in this domain.
- **Admission rule (r2).** Every C2 and S arm must differ bit-exactly from the target run's natural carrier at one or more designated positions (slots 2–3). Identity arms are exempt. Removed in r2: `C2-ctrl-L4` and `C2-np-L4`.

## 5. Metrics

Computed by `scripts/gwelem_analyze.py`. It was qualified on synthetic data before any run and does not touch the frozen Test 3 analysis.

**Definitions:**
- r(Y) = the bare space; r′(Y) = Y's symbol token.
- lp = candidate logit − full-vocabulary log-normaliser.
- The candidate set is the baseline "element" set (125 tokens: all symbol first-tokens, the space and the digits).
- **Outcome order:** r(Y), r′(Y), `other_element_symbol` (another of the 118 elements' symbol tokens), `digit`, `function_or_punct`, `word`. **Switch = r′(Y).**

**Fields:**
- m_switch = lp(r′) − lp(r);
- gains relative to the natural number readout;
- kl_ref = KL(separable reference ‖ arm) over the candidate set.

**B3:**
- The class population is the symbol tokens of the **60 naturally eligible fit and development elements**, minus {r(Y), r′(Y)}, mirroring Test 3's use of the non-holdout population.
- The excess is lp(arm, c) − lp(separable reference, c).
- b3_mean is the mean over classes.
- b3_max is the largest per-class mean over the group, among classes with support ≥ 10 targets.

## 6. Gates and decision rule (Test 3 Stage 2 values; nothing relaxed)

**Primary: C2-full-L4, pooled over A2 and A3, 25 elements.** It **passes** iff every one of these holds:

| Gate | Rule |
|---|---|
| A switching | cluster-bootstrap lower 95% bound of the switch share ≥ **0.80**, **and** the A2 and A3 shares are each ≥ **0.60** |
| B3 mean | ≤ **+0.5** nats (point estimate, as frozen) |
| B3 max-class | ≤ **+1.0** nats, support ≥ 10 targets |
| Other-element top-1 | ≤ **0.05** |
| **Non-worsening vs S** | Defined as follows: <ul><li>For each target Y, Δ_Y = the mean, over Y's (A2, A3) cells, of b3_mean(C2-full) − b3_mean(S-full) **on the same cell**.</li><li>Compute the cluster-bootstrap 95% interval of the mean of Δ.</li><li>**Pass iff the upper bound ≤ +0.25 nats.**</li></ul> |

**How the gates combine:**
- The verdict is `pass`, `no switch` (gate A fails), or `axis B fail: …` (naming each failed gate).
- The absolute B3 gates are primary.
- The non-worsening comparison is an **additional required condition**. It can only fail a candidate; it can never rescue an absolute B3 failure.
- The S arm's own B3 is reported, but it does not change the C2 verdict.

**No secondary test (r2).** The primary verdict alone decides the confirmation.

**Descriptive only:**
- S-full groups;
- per-phrasing breakdowns;
- the identity of every other-element top-1 token.

**Bootstrap:**
- `numpy.random.default_rng(20261118)`, 10,000 resamples;
- clusters are target elements;
- percentile intervals;
- the draw order is fixed by `gwelem_analyze.GROUPS`, then the fields in order, then the two non-worsening tests.

**Sensitivity note (frozen now).** The symbol-prompt reference has near-ties in a few elements (smallest own-symbol lead 0.1 nats). This is why non-worsening is computed on the same cells against S, not against another domain's reference.

## 7. Integrity (fail closed; no analysis unless every check passes)

**Pre-run:**
- the execution binding equals this protocol and the qualified identities;
- populations re-derived;
- a seal of every confirmation run, whose prompt steps equal the baseline's and whose last step equals the baseline readout hash;
- capture: number and symbol into separate directories;
- capture–seal agreement and bit-exact reinjection of every run;
- build from the restricted views;
- independent admission, recompute and dependency audit;
- runner `--check`;
- the exact inventory (200 / 100 / 100) and the designated-position rule.

**Runner:**
- `gwelem_band_runner` (qualified in `confirm-qual-03`);
- the plan carries a `sealed_release` naming an authorisation file written from the owner's recorded GO. The authorisation must carry the plan's `runs_digest`, and it is **single-use**: the runner creates `AUTHORISATION.used` before loading the model and refuses any later reuse;
- the seal and capture binaries carry no element seal of their own. Every plan handed to them passes the orchestrator seal (key and four-form subject-token check);
- the spent Test 3 holdout remains unconditionally refused.

**Post-run:**
- the manifest is complete with the bound executable, plan and release;
- records cover the plan exactly;
- the generic band record admission and the sub-window locality admission pass;
- every identity arm is bit-exact (100/100);
- coverage holds: replacements = firings = span length;
- natural number readouts equal the baseline hashes, top-1 is the space, and natural symbol readouts are the element's symbol;
- protected records are unchanged: Test 3, the holdout marker, CP3a–c, baseline-01 and the feasibility record.

**Power:** log only, and battery intervals are disclosed.

## 8. Not permitted

- refitting, or any change to F, the formula, the layer, the spans, the phrasings, the exclusions or the thresholds after development or confirmation outputs exist;
- switching analysis on development elements;
- re-running the confirmation population.

The confirmation population is **opened once**. A failed run is recorded, not repaired.

## 9. Known risks (recorded before execution)

1. **Format-token source.** The model's preference for the bare space is strong: median −12 (A2) and −15 (A3) logits to the own symbol, and at least −7.8. A switch requires overcoming it at a single early layer. Capital→currency started at about −18 and switched, but this is a different answer type.
2. **25 clusters.** A pooled lower bound ≥ 0.80 needs roughly 23 or more of the 25 elements switching in both phrasings. There is little slack.
3. **Single-letter symbol tokens** (` B`, ` N`, ` V`, ` W`, ` K`…) and word-like symbols (` In`, ` As`, ` No`, ` At`, ` Be`) are counted as other-element outcomes when they are top-1 for another element. This is the conservative reading.
4. **Window offset.** The window's absolute positions differ by subject length (1–4 tokens). The slots are aligned by index, as in Tests 2 and 3; RoPE offsets are not equalised.
5. **The S reference is itself an intervention.** If S fails B3 badly, a non-worsening pass is weak evidence. The S arm's B3 is reported next to C2's.
