# CI throughput — E1, E2

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

## E1's verdict (2026-09-07)

`E1-verdict.json`, frozen as delivered, against `after-e1.json`
(12 attempts / 6 branches, `--since 2026-09-07`):

```
C1  cancel superseded PR runs    P1  931.6 -> 26.9 min  (-97%)        HELD
C2  macOS scheduling contention  P2  queued.macos      (-79%)         HELD
                                 P3  executed.macos    (-70%)    FALSIFIED
                                 VERDICT: INVALID COMPARISON
C3  VINDEX benches Linux-only    P4  29.7 -> 21.8 min p50            HELD
SYSTEM                           P5  merge-ready 61.4 -> 40.8 p50    HELD
```

C2's refusal was correct behaviour and fired for the wrong reason: P2 and
P3 name **extensive** totals, and 12 attempts against 33 contributes
-63.6% before any effect exists, so P3 could not have held. C2 is
**unadjudicated, not refuted**. `E1-postmortem.md` carries the post-hoc
diagnostic and is explicitly not a new verdict; E1 itself is not
re-scored.

## E2-0 — what the instrument learned

The defect was in the measurement, so the measurement was fixed:

- every extensive metric gained an intensive companion —
  `queued_per_attempt.<os>`, `executed_per_attempt.<os>`;
- `score_prediction` **refuses** an extensive metric whose two samples
  differ in attempt count, returning `UNSCOREABLE` and naming the
  per-attempt metric to use instead;
- `workflow_share.<workflow>` makes composition scoreable, which is what
  P3 was reaching for;
- `workflow_exec_max.<workflow>` measures a gate's wall clock over
  **successful runs only**, so cancelling more runs cannot read as a
  faster gate;
- `selftest` asserts the guard fires at unequal n, does **not** fire at
  equal n, and that every forecast frozen from E2 onward names only
  intensive metrics.

Re-running `adjudicate` on E1 today returns `UNSCOREABLE` for **both C1
and C2** — P1 also names a total (`superseded_exec_total`) — where the
frozen verdict read HELD and INVALID COMPARISON. That is not a retraction
of C1: its intensive companion `superseded_exec_mean` moves -92%
(28m14s -> 2m15s per attempt), so cancellation is established on a metric
the guard accepts. C2's companion moves only -16.6%, which is
inconclusive. The guard refuses by NAME, not by substance, and the
substance differs between the two.

`E1-verdict.json` is not regenerated; it records the verdict as
delivered.

## E2-A — one piece of evidence, once

`E2-contract.json`, frozen 2026-09-07. Baseline: `after-e1.json` for the
system metrics, `metal-gate-baseline.json` (40 attempts, 34
successful Metal runs spanning 47.0-68.7 min, p50 56.6) for the Metal
gate itself.

The mechanism, measured on run 34098337888 (#449): `cargo test --tests`
selects every target with `test = true`, and a lib has it by default, so
the job ran 1 binary under `--lib`, then 69 under `--tests` **including
`src/lib.rs` again**, then the same 69 instrumented. The 530-test lib
suite executed three times; the 68 integration binaries twice; 2061s of a
3565s job was duplicate execution.

So the coverage run — which already runs the same 69 binaries serially
and fails on a failing test — becomes authoritative, the two plain passes
are deleted, and a small ordinary-profile witness keeps the one signal
that would otherwise be lost. `bench-regress` leaves the pull-request
path until PERF-QUAL-2 qualifies it.

Outcomes live in `E2-A-qualification.json` (machine-readable) and
`E2-A-execution-notes.md` (the reading). The contract itself is frozen
and is not edited by qualification.

Five predictions (A1–A5) plus five **structural** falsifiers (S1–S5) that
this instrument cannot score and which are checked by hand: the same 69
binaries still execute, a deliberately failing test reds the gate, a
deliberately violated coverage policy reds the gate, the smoke witness
runs in the ordinary profile, and no `pull_request` benchmark job
remains.

## Still not done

`sccache` and vcpkg binary caching are deliberately later than they were
in the original plan: measurement says compilation here is tens of
seconds (cold `cargo check --all-targets` 26.1s, plain test build 21.8s)
against tens of minutes of test execution. Caching a 22-second build
before removing a 34-minute duplicate execution optimises the wrong
denominator.

E2-D — tiered Metal triggers — is next after E2-A is scored, not
alongside it: it deliberately changes
`workflow_share.larql-compute-metal`, which is A5's validity gate. The
five remaining ungated `cargo test --benches` steps (`larql-boundary`,
`larql-core`, `larql-kv`, `larql-lql`, `larql-models`) and an actionlint
gate stay held back.
