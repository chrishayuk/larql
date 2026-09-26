# GW-TS-1 population contract — `run_experiment_subjects`

**Status:** frozen before population capture, 2026-09-21

**Bound GW-TS-1 protocol:**
`sha256:d873d238d99406dc8a549681aa746e750a9d4b802e24ff21a8eb26d152a97d55`

**Bound ATTR-1D contract:**
`sha256:4f48973807db0695d5f30b83a00c73637afbd0dc77d116b5a1f9eb6ec913431c`

**Machine contract:**
[`bench/gw-ts-1/population-contract.json`](../bench/gw-ts-1/population-contract.json)

**Materialized manifest:**
[`bench/gw-ts-1/population-manifest.json`](../bench/gw-ts-1/population-manifest.json)

**Generator:** `scripts/gwts1_population_manifest.py` (deterministic; performs
no model execution)

This is `run_experiment_subjects` from the frozen dependency graph
(`bench/gw-ts-1/programme-dependencies.json`): it binds every field
`gwts1-protocol.json`'s `population_binding.required` list names, over the
already-frozen GW-0 census. Population **capture** — actually running the
model over these rows — is a separate, later step this contract does not
authorize.

## Population

Of the 142 `TransitionIdentity` groups in the frozen GW-0 census
(`bench/gw0/gemma3-4b-it-phase1/`), 85 have all three declared prompt
families (canonical, question, alternate) and are entirely in the upstream
`train` split. Those 85 are GW-TS-1's population. Every other upstream split
(validation, test, and any train identity missing a family) is excluded and
reserved for a future replication population — never analyzed here.

## Discovery / held-out assignment

Split unit is `TransitionIdentity` (never prompt family, execution, or
row), per the protocol's `split_discipline.group`. 85 × 0.6 = 51 discovery,
85 × 0.4 = 34 held-out — exact at the population level, but the four
relation strata (capital 23, currency 20, hypernym 24, language 18) don't
divide evenly by 0.6, so:

1. **Apportionment** — largest-remainder (Hamilton) per stratum, ties
   broken by ascending relation name, so the discovery counts (capital 14,
   currency 12, hypernym 14, language 11 → 51 total) are fixed by a rule,
   not a choice, and land exactly on the population-level target.
2. **Within-stratum selection** — `canonical_hash_order/v1`: sort each
   stratum's identities by `sha256` of the sorted-key canonical JSON of
   `{relation, subject, target}`; the first `discovery_count` are
   discovery, the rest held-out. Fully reproducible from the frozen
   population alone; nothing about model behavior, capture outcome, or the
   ATTR-1D witness feeds this ordering.

`stratify_by` is `[task_identity, semantic_operation]` per the frozen
protocol's language; in this population `task_identity` is the single
constant `raw_text_completion_container_tokenizer_with_special_tokens`
protocol shared by every row (so it contributes no further split), and
`semantic_operation` is the `relation` field.

**Note on ATTR-1D's own witness row:** `capital / Algeria / Algiers` — the
row ATTR-1D's sealed witness used — is a member of this population (nothing
excludes it) and its hash landed it in **held-out**, not discovery. That's
the deterministic rule's output, not a decision; it also means ATTR-1D's
descriptive pass over that row cannot have biased which identities
GW-TS-1's discovery set contains. It stays exactly where the split put it —
moving it now, after the fact, would be worse than leaving it, since that
would make the split outcome-dependent in exactly the way the frozen rule
exists to prevent.

## Provenance fact

Recorded explicitly in `population-manifest.json`'s and
`population-contract.json`'s `provenance_facts` (a runtime assertion
refuses if the identity is ever absent, duplicated, or found outside
`held_out`):

> `capital/Algeria/Algiers` received prior instrument-validation exposure
> under ATTR-1D — the sealed descriptive-support witness ran on this
> identity's canonical prompt family. No GW-TS-1 population assignment,
> extraction rule, calibration, predictor, or gate was changed using that
> exposure.

ATTR-1D validated the *measurement instrument* on that row (whether the
descriptive normalization is lawful), not GW-TS-1's thresholds, path
prototypes, candidate vocabulary, or predictor state — both were frozen
independently of that result, so this does not contaminate the primary
split. The fact is on the record anyway, rather than left implicit.

## Preregistered sensitivity check: leave-Algeria-out

`provenance_facts` is paired with a `preregistered_sensitivity_checks`
entry, `leave_algeria_out`: recompute every held-out assessment metric
(held-out path coverage, mean candidate fraction, the block-bootstrap
mass-retention and operator-accuracy advantages, and the
`progression_gate` verdict) with `capital/Algeria/Algiers` excluded from
held-out, and report it alongside the primary result. It demonstrates that
one previously inspected instrument-validation row isn't carrying the
result — it does not replace or gate the primary held-out verdict, and
cannot promote or demote it.

## Capture window

All 34 layers (`0..=33`) × `{attention, ffn}`, final prompt position only:
**68 eligible parent sites per execution**, generated positions excluded.
No layer band is privileged — deliberately: every attention/FFN site this
programme has previously singled out (L23/L24/L29/L35/L41, etc.) came from
narrower prior work, and using that here would let the path-prediction test
rediscover what was already known rather than ask the open question. Site
order is canonical execution order.

## Reader

One reader per `TransitionInstance`: `selected_token_row/v1` +
`identity/v1` (no L2-normalized or contrast reader), on the **preregistered
first continuation token of the semantic destination**
(`semantic_edge.target_token_ids[0]` from the frozen GW-0 row) — never
whatever the model generates. Row values come from the exact resident
output-head representation via `SelectedOutputHead::row_f32` (the same
accessor ATTR-1D's witness added), bound to the token id, its vector hash,
the prepared-image fingerprint and the realization policy. A row that
cannot be represented unambiguously refuses rather than substituting.

## Observation

**Attention** (ATTR-1D authority): `source_top_k` is the maximum visible
prompt length across GW-TS-1's own 255-row population (85 identities × 3
families) — **14 tokens**, computed from the frozen tokenizations before
any capture, not tuned after seeing results. At that value every visible
source position is retained; source truncation is forbidden outright.

**FFN** (production-effective GW-0B-style reconstruction): the top 64 exact
addresses by contribution mass — the largest of the protocol's own frozen
sensitivity caps — plus total absolute child mass and achieved-mass
accounting, so any truncation below full coverage is explicit, never
silent.

## Byte budget

64 MiB per `TransitionInstance`, 16 GiB aggregate. Either ceiling being
exceeded is an explicit instance refusal, never a silent truncation, and
every refusal counts against the frozen 0.80 held-out path-coverage gate.

## What this contract does not authorize

Binding these fields is a design freeze, not a go-ahead to run the model.
`bench/gw-ts-1/population-contract.json`'s `freeze.note` records explicitly
that its `attr1d_witness` authority currently names an **uncommitted**
working-tree artifact — population capture must wait until that ATTR-1D
closure commit has actually landed in history. A later amendment binds a
committed identity before capture begins.
