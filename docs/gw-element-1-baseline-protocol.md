# GW-ELEMENT-1: natural-baseline feasibility, protocol (FROZEN before execution)

**Status:** frozen 2026-11-21 under the owner's restricted GO, "GO for the natural-baseline feasibility stage only". It is bound by hash in `output/gw-element-1/baseline-01/binding.json` and registered before inference.

**Natural inference only:**
- no relation intervention;
- no C2 construction, no fitting;
- no switching experiment;
- no confirmation execution;
- no change to Test 3 or GW-COMPRESS-1.

## Question

Does Gemma 3 4B IT answer chemical-symbol and atomic-number questions naturally, reliably enough, and in the expected format, for the prospective confirmation (atomic number → symbol) to be scientifically meaningful?

## Frozen inputs (accepted by the owner from the model-free audit, result 6314)

| Item | Value |
|---|---|
| Prompts | **A2:** `For {x}, the chemical symbol is` / `For {x}, the atomic number is`<br>**A3:** `{X}'s chemical symbol is` / `{X}'s atomic number is`<br>IUPAC names, lower-case mid-sentence, capitalised at the start |
| Token exclusions (15) | multi-token symbol: Yb, Hf, Tl, Np, Lr, Sg; A3 subject echo: Al, Sc, Y, I, Pr, Eu, Os, U, Fm. These are recomputed at prepare time and must equal the frozen list |
| Population | the **103 token-eligible elements**. That gives 103 × 2 phrasings × 2 relations = **412 element cells**, with no replacement |
| Split | the salted split over all 118 (`gw-element-1/split/v1|Z`; fit 59 / development 24 / confirmation 35), read from the registered feasibility record (artifact 2748, sha `068ecc94…`) |
| Runner | `gwtemplate_baseline`, **unchanged**: source `bd76a134…`, binary `5816e19d…`, qualified in `baseline-qual-01`. The binary was lost with the worktree `target/` and rebuilt from the unchanged source, **reproducing the qualified hash bit-exactly** |
| Model | `gemma3-4b-it.vindex3`. Bound: `index.json`, `system_graph.json` and `tokenizer.json` (`4667f208…`) |
| Decoding | teacher-forced prompt, then **8 greedy continuation tokens**; top-64 recorded at the readout |
| Anchors | four Test 1 baseline cells (`capital|P2|FR`, `capital|P3|JP`, `currency_code_iso4217|P2|BR`, `capital|P1|KE`). Their records must equal the `baseline-01` records **bit-exactly**, proving the model, runner and tokenizer are unchanged |
| Power | log only |

## Frozen answer parser (`scripts/gwelem_parse.py`; qualified by `scripts/test_gwelem_parse.py`)

Each token is decoded separately. Special terminators are `<eos>` and `<end_of_turn>`.

**Symbol:**
- correct iff continuation[0] is the first token of `" "+symbol`;
- **and** the next token does not begin with a lower-case ASCII letter. For example, `" C"`+`"l"` for carbon is refused as `symbol_continues`;
- if the symbol is the final continuation token, it is `truncated` and refused.

**Atomic number:**
1. continuation[0] must be exactly the bare-space token `" "`. Otherwise `format: first_token_not_space`, which includes markdown `**`.
2. The digit run is the maximal run of following all-digit tokens. An empty run is `format: no_digits`.
3. **A terminator must be observed inside the continuation.** A run reaching the end of the continuation is `truncated` and **never accepted**.
4. The terminator is a special token, or its text begins with whitespace or ASCII punctuation. Anything else is `format: bad_terminator` (e.g. `8th`).
5. Correct iff the digits equal str(Z) exactly. A leading zero or any other value is `wrong_value`.

The parser was qualified before any element output existed, with positive, negative and truncated examples on real token ids:
- truncated: `" ","1","1","8"` with no terminator, a symbol as the last token, a full-length digit run;
- negative: markdown, `8th`, `08`, `260`, `" C"+"l"`;
- positive: `.`, `,`, newline and `<end_of_turn>` terminators.

## Eligibility and requirements (frozen; not altered by outputs)

- An element is **naturally eligible** iff all four of its cells are correct (symbol and number in A2 and A3).
- Ineligible elements are excluded with their reasons, without replacement.
- The domain is **UNSUITABLE**, and the result is recorded with no prompt, population or threshold changes, if any of these hold:
  - fewer than **30** fit, **10** development or **20** confirmation elements are naturally eligible;
  - in either phrasing, the number prompt's first token is `" "` for less than **80%** of token-eligible elements.

## Integrity (fail closed; no report unless all pass)

- **Binding:** every bound file is unchanged at execution, and the runner and source identities equal the qualified ones.
- **Prepare:** the plan is reproducible from the frozen inputs at binding.
- **Before the run:** the parser qualification passes, and runner `--check` passes (independent re-tokenisation and span checks).
- **After the run:**
  - the manifest is complete, with the bound executable and plan hashes;
  - records cover the plan exactly;
  - the four anchors are bit-exact;
  - the protected records are unchanged: Test 3 bindings and holdout marker, CP3a–c bindings and integrity, CP3c report, the feasibility record.

## Reported

- natural symbol and atomic-number accuracy, separately for A2 and A3;
- first-token distributions and parser reasons (format failures, truncation, wrong values);
- token-eligible and naturally eligible counts per split;
- every exclusion with reasons;
- the requirements and verdict;
- a recommendation on scientific adequacy for the confirmation.

## Not authorised

- any intervention, C2 construction or fitting;
- single-layer switching;
- confirmation;
- changing prompts, exclusions, split or thresholds;
- a multi-token answer scorer;
- changes to Test 3 or GW-COMPRESS-1 records.
