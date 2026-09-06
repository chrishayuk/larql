# CI throughput — E1

An experiment, not a cleanup. The question is whether three specific CI
changes move measured wall clock, and by how much, so that afterwards the
claim is a number rather than "CI feels faster".

## The instrument

`scripts/ci_throughput.py` — `collect` snapshots the Actions API,
`report` scores a snapshot against a baseline, `selftest` checks the
metric definitions against synthetic runs with known answers (including
two negative controls: a failing gate must leave merge-ready undefined,
and a cancelled run must produce zero superseded execution).

The unit of observation is an **attempt**: one head SHA of one pull
request, with every workflow run created for it. Every metric separates
queue from execution, because the two respond to different changes:

```
queue      job.started_at   - job.created_at
execution  job.completed_at - job.started_at
```

## The intervention

| Commit | Change | Scored by |
|---|---|---|
| c1 | `concurrency` + `cancel-in-progress` on 17 PR workflows | superseded execution |
| c2 | macOS PR occupancy: MSRV Metal split out, 4 crates macOS→main-only | macOS queue minutes |
| c3 | VINDEX benches Linux-only | vindex Windows execution |

`bench-regress.yml` is excluded from c1 until its baseline-restore
semantics are inspected.

## Baseline (frozen 2026-09-06, before any of c1–c3)

`before-e1.json` — 33 attempts across 20 branches, from 2026-08-15.

```
merge-ready p50           61m23s
merge-ready p95           95m07s
macOS queue max p50       20m18s
macOS queue max p95       68m09s
vindex windows exec p50   29m41s
superseded execution      931.6 runner-min total (28.2 mean/attempt)

executed runner-min       linux 5037  windows 3794  macos 3035
queued   runner-min       linux 8984  windows 1632  macos 4527
```

The single most useful number came out of the baseline rather than out of
the plan: **c2 removes 27.9% of macOS queue-minutes but only 5.2% of
macOS execution.** The jobs it drops are cheap to run and expensive to
schedule — `quality`'s msrv-macOS leg is 33 jobs, 544 queue-minutes and
22 execution-minutes. So c2's mechanism is contention relief for a scarce
runner pool, not a reduction in macOS compute. `larql-compute-metal`
alone is 35.3% of macOS execution and c2 deliberately does not touch it.

## Predictions

Pre-registered in `E1-forecast.json`, each with a falsifier. P3 exists
only to catch a composition change masquerading as an effect.

## Scoring

Collect an AFTER snapshot once 10–20 pull requests have run under the new
configuration, then:

```
scripts/ci_throughput.py collect --since <date> --max-prs 20 --out docs/ci-throughput/after-e1.json
scripts/ci_throughput.py report  --before docs/ci-throughput/before-e1.json \
                                 --after  docs/ci-throughput/after-e1.json
```

`report` prints workflow composition for both periods. Read it before
reading any delta: the two periods must be made of comparable pull
requests for the comparison to mean anything.

Then adjudicate, which scores the frozen forecast **per mechanism** so the
headline number cannot swallow the causal information:

```
scripts/ci_throughput.py adjudicate --forecast docs/ci-throughput/E1-forecast.json \
                                    --before docs/ci-throughput/before-e1.json \
                                    --after  docs/ci-throughput/after-e1.json
```

```
C1  cancel superseded pull-request runs
  P1  superseded_exec_total: 931.6 -> X min   HELD / PARTIAL / FALSIFIED
  VERDICT: ...

C2  reduce macOS PR scheduling contention
  P2  queued.macos:   4526.6 -> X min
  P3 (validity)  executed.macos: 3034.9 -> X min
  VERDICT: ... or INVALID COMPARISON
```

It is possible for c1 and c3 to work and for GitHub's macOS pool to be
unusually bad during the AFTER window. Per-mechanism verdicts preserve
that distinction; a single merge-ready delta would not.

Thresholds live in `E1-forecast.json` as machine-readable conditions,
frozen with the forecast, so scoring cannot reinterpret them after the
data exists. `P3` is a **validity** prediction rather than an effect: if
macOS execution collapses, C2 reports `INVALID COMPARISON` and P2 is not
interpreted at all.

Two controls back this up. `selftest` proves each verdict is reachable —
including that a thin AFTER sample makes P2 look `HELD` while P3 vetoes it
to `INVALID COMPARISON`. And scoring the baseline against itself falsifies
all four mechanisms, which is the evidence that the forecast is not
already satisfied by doing nothing.

## Not in E1

vcpkg binary caching and `sccache` are E2, deliberately held back so that
a change in these numbers is attributable to c1–c3 and nothing else. The
five remaining ungated `cargo test --benches` steps (`larql-boundary`,
`larql-core`, `larql-kv`, `larql-lql`, `larql-models`) and an actionlint
gate are also held back for the same reason.
