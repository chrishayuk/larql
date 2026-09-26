# MEASURE-PLAN-3 and AUTO-REP-1c: a pre-registered plan gate for Granite 4.1 3B, and the first campaign

**Frozen 2026-09-26, before any plan-v1 reading of Granite 4.1 3B.** The question, the vocabulary, the gate's
construction rule, the arms, the bank, the tail policy, the budget and the forecasts below are FROZEN. The
anchor's numbers are measured after this commit and before any campaign reading.

Programme: REPRESENT. Slice 3 of MEASURE-PLAN (`docs/measure-plan-1.md`, `docs/measure-plan-2.md`), and
AUTO-REP-1c (`docs/auto-rep-1.md`).

## The question

> Given a quality floor fixed by a hand-written protection map, does AUTO-REP find an admissible map that
> is **strictly cheaper** than that hand map?

The hand map anchors the gate. The anchor is admissible by construction: it is no worse than itself. So
the campaign cannot fail to find *an* admissible map within its budget. The finding is whether the
cheapest admitted map is cheaper than the anchor, and by how much.

## Model, bank and arms

- **Model:** `~/chris-models/granite-4.1-3b.vindex3`, encoded from `ibm-granite/granite-4.1-3b` at
  `c0650403e44e78ec0262dab1c90914c65b196c4e`, the registry pin. Its G4 verification passed: 48/48
  checks, payloads byte-equal.
- **Bank:** Q-BANK-1 `prompts.json` (frozen, 69 prompts), exported with the container's own tokenizer
  (`vindex3 token-bank export`, default `--max-tokens 128`). **All 69 sequences** are measured. UNCERTAINTY-2
  places the p99's bias at fewer than 32 sequences and needs 64–128 for ±25%.
- **Arms** (`vindex3 auto-rep run` / `VerbPlanExecutor`):
  - reference `production` on the source's canonical bytes;
  - candidate `production-nvfp4` on the stored pack.

  Interpreter arms execute any mix of held and compiled groups. A lowered arm might refuse a
  heterogeneous candidate (see MEASURE-PLAN-2's PR 3b record).

## Search vocabulary (AUTO-REP-1c)

Twelve groups. Each is a projection family crossed with a layer quarter of the 40-layer stack:

| family | projections |
|---|---|
| `attn-qkv` | `q_proj`, `k_proj`, `v_proj` |
| `attn-o` | `o_proj` |
| `ffn` | `gate_proj`, `up_proj`, `down_proj` |

The quarters are Q1 = 0–9, Q2 = 10–19, Q3 = 20–29 and Q4 = 30–39.

- Each group is compiled NVFP4 or held at source, so there are 4,096 states.
- The base map is uniform NVFP4 with the default roles.
- No prior, no pins, no structural cuts other than those the layout derives.
- The solver is 1a's branch-and-bound, cheapest first. Each Authority refusal becomes an `ExactNoGood`.

`late10-ffn` from the Q-BANK-1 sweep protects FFN in layers 30–39. In this vocabulary it is exactly one
state, {`ffn`×Q4}.

## The gate: `granite-4.1-3b-plan-v1-vs-late10-ffn/v1`

**The anchor.** Once this document is committed, and before any campaign run, compile the anchor state
({`ffn`×Q4} held, everything else NVFP4). Measure it once with plan-v1 over all 69 sequences with the arms
above, and record its report.

**The gate** is a `PlanGate` with semantics `PLAN_V1` (KL nats, full vocabulary). Its limits are the
anchor's own values:

- `kl_p99_max` = anchor KL p99;
- `kl_mean_max` = anchor KL mean;
- `top1_disagreement_max` = anchor `1 − top1_agreement`;
- `delta_nll_mean_max` = anchor ΔNLL mean;
- `positions_min` = the positions of the 69 sequences.

A candidate is admitted when it is **no worse than the anchor on every criterion**. The gate id and its
numbers are then registered in `gate_of_any_kind`, in a commit that cites the anchor's report digest.

**Tail support:** `min_tail_observations = 5.0`. A p99 needs 500 positions, and the bank holds well over
1,000. Provenance: this document.

**Diagnostic policy:** none. **Promotion:** out of scope. The campaign's claim is admission. Promotion
through `CandidateIndex` is a separate step.

## Forecasts (made before any plan-v1 reading of Granite 4.1)

The Q-BANK-1 sweep measured these arms under a different instrument, so it predicts direction only, never
magnitude.

1. **Uniform NVFP4 fails the gate** on KL p99, KL mean and ΔNLL. In the sweep, R0 had a KL p99 3.7× and a
   KL mean 2.3× those of `late10-ffn`, with ΔNLL +0.118 against −0.008.
2. **The anchor state is admitted when measured**, given deterministic arms. If it is not, the instrument
   is not reproducible, and the campaign stops with that as its finding.
3. **Unknown, and the point:** whether a state strictly cheaper than the anchor is admitted. Plausible
   candidates are {`ffn`×Q4} minus nothing cheaper, or `attn-o`/`attn-qkv` combinations cheaper than
   FFN×Q4. None is forecast to pass. A pass is a result, and a fail is also a result: the anchor is then
   the frontier in this vocabulary.

## Budget and stopping

- At most **40** measurements, the anchor included. The campaign stops at the first admitted proposal
  (cheapest first), when states run out, or when the budget is spent.
- Every proposal is recorded, rejected ones included (the `CampaignRecord`), and so is every reading (the
  advanced `SearchSnapshot`).
- Reported: the admitted state's protected groups, its bytes against the anchor's, and every criterion's
  margin.

## What must be built first (engineering, no result)

1. **Declared group vocabularies in 1a/1b.** A group becomes a named set of (projection, layer range) rules
   instead of one (projection, layer). The 1a gates are kept: exhaustive-enumeration agreement, byte
   agreement with the footprint, `check_against`, and compilability through `Protections`.
2. The producer and `auto-rep init` accept a declared vocabulary.
3. The gate is registered from the anchor's measured numbers (above).

## Not claimed

- That a map admitted under this gate is good in any absolute sense. The gate says "no worse than
  `late10-ffn`", nothing more.
- Anything about other models, or about grains finer than quarter × family.
- Promotion, or any execution-speed win. Bytes are the objective. Speed is a later measurement.
