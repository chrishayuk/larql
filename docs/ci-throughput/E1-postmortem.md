# E1 — post-hoc diagnostic

**This document is not a verdict.** E1's verdict is `E1-verdict.json`,
frozen as delivered:

```
C1  cancel superseded PR runs        HELD
C2  macOS scheduling contention      INVALID COMPARISON
C3  VINDEX benches Linux-only        HELD
SYSTEM  the tranche as a whole       HELD
```

Everything below was computed **after** the forecast was scored. It
explains why C2 could not be adjudicated and what the corrected
measurement is used for. It does not re-score E1, and none of these
numbers may be cited as an E1 result.

## What happened

E1 froze five predictions on 2026-09-06. Two of them — P2 and P3, the
pair that scores C2 — name **extensive** metrics: totals of macOS
runner-minutes across the sample.

The AFTER window collected **12 attempts** against a 33-attempt
baseline. For any metric that is flat per attempt, a 12/33 ratio moves
the total by

```
12/33 - 1 = -63.6%
```

before the intervention is considered at all. So:

- **P3 could not have returned HELD.** Its band was "falls by no more
  than 10%". The sample alone spent 63.6% of it.
- **P2 could not have returned FALSIFIED.** Its falsifier was "less than
  20%". The sample alone cleared it three times over.

The adjudicator returned `INVALID COMPARISON` for C2, which is the
behaviour the forecast designed and the selftest proves reachable. But
the veto fired on **sample size**, not on the composition change P3 was
written to detect. C2 is **unadjudicated, not refuted.**

The three predictions that survived are exactly the **intensive** ones:
P1 is scored on a total but its per-attempt mean moves the same way
(-92%), and P4 and P5 are percentiles. Only the totals broke.

## The post-hoc numbers

Per-attempt, from the same two snapshots:

| metric | before /attempt | after /attempt | delta |
|---|---|---|---|
| `queued.macos` | 137.2 min | 80.1 min | -41.6% |
| `executed.macos` | 92.0 min | 76.7 min | -16.6% |
| `queued.linux` | 272.2 min | 129.2 min | -52.5% |
| `executed.linux` | 152.6 min | 107.1 min | -29.8% |
| `queued.windows` | 49.5 min | 18.8 min | -62.0% |
| `executed.windows` | 115.0 min | 69.1 min | -39.9% |

Under P3's own bands, `executed.macos` at -16.6% is neither HELD
(≥ -10%) nor FALSIFIED (< -20%). **Genuinely inconclusive** — which is
a different finding from `INVALID COMPARISON`, and it is why this is a
diagnostic rather than a verdict.

Two further observations that a totals-only reading hides:

- `macos_queue_max` p50 moved the **wrong way**: 20m18s → 21m42s (+7%;
  +16% with `--require-complete`), while its p95 fell 38%. The tail
  improved; the median did not.
- `msrv-metal` — the workflow c2 **created**, by splitting the MSRV Metal
  leg out of `quality` — is now itself the pathology c2 was written to
  remove. In the AFTER window: 6 jobs, **63.3 macOS queue-minutes to buy
  4.3 execution-minutes**. Cheap to run, expensive to schedule.

## What was changed, and what was not

Changed — the **instrument**, so this cannot recur silently:

- every extensive metric gained an intensive companion
  (`queued_per_attempt.*`, `executed_per_attempt.*`);
- `score_prediction` **refuses** an extensive metric whose two samples
  differ in attempt count, returning `UNSCOREABLE` and naming the
  per-attempt metric to use instead;
- `workflow_share.*` was added, so composition — the thing P3 was
  actually reaching for — is a scoreable number rather than a printed
  line;
- `workflow_exec_max.*` was added, counting **successful runs only**, so
  a gate cannot look faster by being cancelled more often;
- the selftest asserts the guard fires at unequal n and does **not** fire
  at equal n, and that every forecast frozen from E2 onward names only
  intensive metrics.

Not changed — **E1**. The forecast, the verdict and both snapshots stand
exactly as they were. Re-scoring a pre-registered experiment on a metric
chosen after seeing its data is the failure this whole apparatus exists
to prevent.

One consequence worth stating plainly: re-running `adjudicate` on E1 with
today's instrument refuses **C1 as well as C2**, because P1 also names a
total (`superseded_exec_total`). The frozen verdict read C1 HELD, and
that finding stands — its intensive companion `superseded_exec_mean`
moves -92% (28m14s -> 2m15s per attempt), so cancellation is established
on a metric the guard accepts. The guard refuses a metric by NAME; it
does not adjudicate the mechanism behind it. For C1 the substance was
already there in the per-attempt mean; for C2 it was not.

`E1-verdict.json` records the verdict as delivered and is not
regenerated.

## What C2 still owes

The macOS-occupancy mechanism has never been scored. It is not being
re-run as E1; it is superseded by CI-E2-D, which proposes the same
mechanism in a stronger form (tiered Metal triggers) and will be scored
against `E2-contract.json` on intensive metrics with a composition
validity gate.
