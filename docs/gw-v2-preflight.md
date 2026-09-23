# GW-V2 execution preflight

**Historical pre-amendment record.** The user subsequently authorized AMEND-1;
see the [execution record](gw-v2-execution.md) for its seal and current stages.
The findings and proposed clarification below describe the outcome-free audit
before that authorization.

**Date:** 2026-09-21  
**Status:** outcome-free audit complete; fit-sham correspondence needs definition.  
**Execution:** no GW-V2 model prompts, captures, fits or replays were run by this audit.

The original [protocol](../bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json)
remains unchanged at
`sha256:830c2ceb97fce4f2d03e74e207d04e54257cd5b38e830a93047a86b6bd4fd006`.
The population identity remains
`sha256:d73e0783adc0c26777ed9f0e4b880133814c39d0283cd7d816cd72cd1d37c23c`.

## Checks completed

The original protocol/population validator passes. The inherited GW-KEY-1
template role table covers all 666 V2 prompts exhaustively and without overlap.
All 49 required template/role reference cells are represented in the original
GW-KEY-1 training role inventory. This checks structural availability, not the
numeric validity of reference vectors; no activation tensors were read.

| Split | Countries | Executions | Subject-token positions |
|---|---:|---:|---:|
| Train | 44 | 396 | 603 |
| Validation | 15 | 135 | 189 |
| Test | 15 | 135 | 216 |

Subject-token counts are consistent across the nine executions of each country.
The training population has 28 one-token, 12 two-token, two three-token, one
four-token and one five-token country names. The latter two countries are
`KNA` and `STP`.

The repository currently contains V2 population/preregistration tooling but no
dedicated V2 capture, fit, replay or adjudication runner. GW-READ-1's existing
runners bind its older population and treatment and cannot be used unchanged.

## Definition needed before execution

The protocol requires:

> subject-identity permutation within train relation strata, seed 27022031

The primary predictor fits one source V to one target V at each subject-token
position. Permuting country identities can pair sequences with different token
counts. The frozen text gives no correspondence for those positions. Dropping,
padding, pooling, repeating or adding token-count strata each supplies an extra
experimental choice. The registered coverage rule also prohibits fallback and
imputation.

The original hash validator does not test this control's implementability.
Passing that validator is therefore insufficient to claim readiness for an
exact execution of every registered control. This is an execution-specification
gap, not a failed V2 result.

## Proposed clarification — not adopted

Use a subject-identity permutation within **relation × subject-token-count**
strata, keeping prompt family and token ordinal fixed. This preserves the
number and equal weighting of training rows and needs no padding or pooling.

For each relation and token-count stratum, order countries by the SHA-256 digest
of the UTF-8 compact JSON array `[27022031, relation, token_count, subject_id]`
(`ensure_ascii=False`, separators `(',', ':')`). Break a digest tie by country
ID. Assign each recipient the next country in this order, wrapping at the end.
Reuse that mapping across all three prompt families, all source depths and all
ranks. Pair each recipient source-token ordinal with the same ordinal in the
donor's natural L24 V sequence for the same relation and prompt family.

Singleton strata remain fixed and must be reported as unshuffled. Thus 42/44
training identities and 522/603 equally weighted token-position fit rows have
different-subject targets. `KNA` and `STP` contribute 81 unchanged rows. This is
a partially shuffled control, not a complete destruction of subject identity.

Fit and seal the sham's full 35-cell family with the same ridge, centring, SVD,
precision and train-only restrictions as the primary family before held-out
replay. Report its complete surface and shuffle coverage separately. No new
sham threshold, primary-cell selection or change to the registered frontier
gate is proposed.

Adopting this rule requires an explicit pre-execution amendment because it adds
token-count strata to the registered control. Preserve the original frozen
artifacts and record the amendment's identity; do not silently rewrite them.

## Reproduction

```sh
python3 scripts/gwv2_preregister.py bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json
python3 scripts/gwv2_preflight.py bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json
python3 -m unittest discover -s scripts -p 'test_gwv2*.py'
```

All 13 preregistration and preflight tests passed. The preflight script reports
the unresolved definition; it does not amend the protocol or execute the model.
