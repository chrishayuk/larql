# E2-A — execution notes

Outcomes, not the forecast. `E2-contract.json` is frozen and is not
edited by anything here; machine-readable qualification outcomes are in
`E2-A-qualification.json`.

## The first observation (A1, n=1)

Run `34153296231`, branch `ci-e2` @ `b369d21a`, PR #450 — the pull
request that makes the change, so it is its own first sample.

```
                                        baseline        this run
larql-compute-metal, successful run     56.61 min       24.87 min
```

Internal split, against the expectation stated before the run:

| | expected | measured |
|---|---|---|
| setup / checkout / toolchain / fmt / check / clippy | < 2 min | **65s** |
| ordinary-profile smoke | well under 1 min | **47s** |
| install cargo-llvm-cov | — | **1s** (was 49.7s) |
| authoritative estate + coverage policy | ~23–25 min | **22m54s** |
| **total** | ~25–28 min | **24m52s** |

The arithmetic said 3565 − 2061 = 1504s. The run measured 1492s, twelve
seconds away. That agreement is closer than the method earns on one
sample and should be read as luck, not precision.

**A1 holds at ≤ 27 min on n = 1.** That is not yet an effect estimate.
The contract asks for `workflow_exec_max.larql-compute-metal.n >= 8`
before the p50 means anything, and until then the honest statement is
"the first observation fell in the target band", not "56.61 → 24.87".

Structural qualification and statistical effect estimation are different
things. The former is complete (below); the latter accumulates.

## Structural qualification

All five controls established — see `E2-A-qualification.json` for run
IDs, SHAs and exact failing steps.

- **S1** 69 binaries in the authoritative step, matching the pre-change
  `--tests` count; `unittests src/lib.rs` runs once where it ran three
  times.
- **S2** run `34155283543`: smoke green, authoritative step red at 798s
  on `s2_qualification_this_test_must_fail`.
- **S3** run `34153560202`: smoke green, **all 70 test binaries passed**,
  TOTAL 95.46% cleared `--fail-under-lines 90`, and the step went red on
  `lowering/stack.rs: lines 86.63% below minimum 95.00%`.
- **S4** `smoke tests executed: 7 (floor 5)`, binary under
  `target/debug/`, not `target/llvm-cov-target/`.
- **S5** no `bench-regress` check among PR #450's 13.

S2 and S3 fail the same step for different reasons, which is the point of
running both: in S2 the tests never finished, so the coverage policy
never ran, and S2 alone cannot show that coverage can refuse.

## Two runs that established nothing

Recorded because leaving them out would make the qualification look
cleaner than it was.

| run | died at | after | why it proved nothing |
|---|---|---|---|
| `34153558202` | Clippy | 14s | `assert!(false, ..)` trips `clippy::assertions_on_constants` under `-D warnings`; steps 9–11 skipped |
| `34155142711` | Format check | 9s | the replacement was not rustfmt-clean |

Both went red. Neither reached the gate under test. **A red run at the
wrong step resembles qualification and is not** — had either run ID been
filed as "S2 established", the contract would have carried a false
qualification, which is worse than no qualification at all.

The second was avoidable: `cargo fmt --check` needs no compilation and
was run locally before the third attempt. Clippy could not be run
locally — this machine was carrying a peer session's GLM-5 oracle at
~300% CPU — so the third attempt was checked by CI's clippy instead, at a
cost of 14s if wrong.

## An incidental finding worth keeping

The early gates are load-bearing, and this is the reverse of what the
experiment was hunting. Two deliberately broken commits never reached the
test estate because lint and formatting caught them in 14 and 9 seconds.
The expensive gate was never asked, because the cheap ones answered
first.

Total macOS cost of qualification: one full estate run (S3, 24.7 min),
one partial (S2, 15.4 min, stopping at the failure), and ~2 min of
early-gate rejections.

## Outstanding

`S4`'s **failure** path is verified only off-runner: the shell logic was
exercised against synthetic logs (7 → pass, 0 → fail), but no run has
shown a filter selecting nothing actually redding the step. The floor's
success path is proven; its failure path is not. That is the same class
of gap the floor exists to close, and it is worth one dispatch if the
floor is ever relied on to catch a real rename.

## What this does not license

E2-D — tiered Metal triggers — changes which pull requests enter the
Metal population, which is A5's validity gate. Landing it before A has
accumulated observations would hand a freshly-corrected instrument a new
attribution problem. A answers *how long does the Metal gate take*; D
answers *how often should we pay it*. They are cleaner apart.
