# BLOCK-TIME-2: runs

Runs of [`block_time_2.py`](../../block_time_2.py) under
[BLOCK-TIME-2](../../BLOCK-TIME-2.md). **The one admissible run has been made**
(`20261115-02-admissible`, below).
The diagnostic below ran on the draft; the protocol was frozen as revision 1
on 2026-11-15, after it and with rules it led to.

**Erratum to the frozen revision 1 text** (found by the independent
review after freezing; the document is not edited, and the code is not
affected): in "What rule 4 conditions on", the sentence "A systematic
deviation from the model in either of them therefore shows as missing
slopes, never as 'does not follow'" refers to the two costs rule 4 checks,
coordinator time and worker execution. A deviation in the two it does not
check, handler time outside execution and the transport remainder, passes
rule 4 and would show as "does not follow".

| Run | Kind | Outcome |
|---|---|---|
| [`20261115-02-admissible`](20261115-02-admissible/analysis.json) | **Admissible**, revision 1 | Complete, counts; 17 pairs valid, 3 void; `sequential`, `block-2`, `block-4`, `block-8` follow the exchange model; `block-1` missing (2 of 4 valid) |
| [`20261115-01-diagnostic`](20261115-01-diagnostic/analysis.json) | Declared functional diagnostic with `--continue-unsettled` on a heavily loaded host; does not count | Complete: 80 of 80 trials, no stop; 1 pair valid, 19 void; no arm reportable |

## 20261115-02-admissible

The one admissible run under BLOCK-TIME-2 revision 1. Revision `fe9407bd`
(the head CI verified, merged as `e9a4b82d`), clean tree, frozen hashes
checked; binaries `v3_block_time` `4333fbd8…`, `larql-server` `bbbbd57c…`
(built at `242fc0d1`, no Rust changed since), frozen relay `7b1097f8…`, both
block-frame qualifications accepted. Peer handshake in the manifest and
`peer-handshake.txt`: both sessions confirmed quiet against a request that
named the excluded work. 14:42:17 to 15:48:03 UTC, after the host settled
(one-minute load 1.39, five-minute 2.93).

**The run counts**: complete, no stop, warm-up settled, parity in every
trial, no compiler or model process at start or end, no pause, no stall.

| Arm | Predicted | Valid pairs | Pair slopes (valid) | Mean | Ratio | Follows (within 10%) |
|---|---|---|---|---|---|---|
| `sequential` | 24 | 3 of 4 | 24.017, 24.028, 23.952 | 23.999 | 1.000 | **yes** |
| `block-1` | 24 | 2 of 4 | 24.015, 24.006 | missing | | **missing** (3 required) |
| `block-2` | 12 | 4 of 4 | 12.008, 12.010, 12.009, 11.964 | 11.998 | 1.000 | **yes** |
| `block-4` | 6 | 4 of 4 | 6.006, 5.826, 5.999, 5.966 | 5.949 | 0.992 | **yes** |
| `block-8` | 3 | 4 of 4 | 2.997, 3.003, 2.995, 2.975 | 2.992 | 0.997 | **yes** |

Slopes are round trips per evaluated position between 5 and 20 ms. `block-1`
is reported missing by the frozen rule. Its two valid pairs are shown here
for completeness only; they are not a registered statement.

**Void pairs**, all kept with their measurements:

- pair 8 (`block-1`, round 2): brackets differ by 1.86%. Within-pair
  differences were 1.3% (coordinator) and 1.7% (execution). Spotlight
  started 27 workers in that minute (15:12).
- pair 13 (`sequential`, round 3): brackets differ by 3.90%; coordinator
  5.65% and worker execution 7.64% between the delays. Worker execution rose
  from about 84 to 87 to 90 ms per position from 15:26 to 15:29 UTC.
- pair 14 (`block-1`, round 3): brackets differ by 5.34%. Its arm trials
  agreed (0.5% and 0.02%); its opening bracket fell in the same 15:26–15:29
  slowdown (134.2 against 127.2 ms per position).

Telemetry over those pairs shows only light processes (a browser,
`WindowServer`, short `rustfmt` and `grep` runs) and no thermal, performance
or memory-pressure warning. The cause of the 15:26–15:29 slowdown is not
identified. Rule 4 voided one pair, pair 13, which the bracket gate also
voided; its failures were one in each cost, both in a 20→5 pair.

**Secondary reports** (not registered statements):

- **Slopes without rule 4**: identical to the registered ones, because no
  pair was voided by rule 4 alone.
- **Shift rule 4 admits per arm**, from valid pairs' costs: 1.0%
  (`sequential`), 1.1% (`block-1`), 2.1% (`block-2`), 4.0% (`block-4`),
  7.8% (`block-8`) of the prediction. As the protocol states, this does not
  bound handler time outside execution or the transport remainder.
- **Delay order**: the mean slope differences between 5→20 and 20→5 pairs
  are at most 0.07 round trips (`block-4`).
- **Worker execution** at 20 ms, median over valid pairs: 83.3 to 83.9 ms
  per position in every arm.
- **Largest per-exchange remainder** in any valid trial: 9.2 ms, against
  medians around 0.35 ms. So no exchange in a valid pair had a remainder
  larger than that in these records; 9.2 ms would be worth about 0.01
  round trips of slope. This does not show that no stall occurred: one
  absorbed into another timing component, or spread across exchanges,
  would not appear as a single large remainder.

**What this establishes, within its scope**: for exact remote block
execution of GPT-OSS 20B with one CPU expert worker, through the qualified
synthetic relay, per-position wall time grows between 5 and 20 ms at the
rate the exchange model predicts, `24 / W` round trips per evaluated
position, for `sequential`, `block-2`, `block-4` and `block-8` (ratios
0.992 to 1.000).

**What it does not**: no statement for `block-1`; nothing about accepted
tokens or generation throughput; nothing about a physical network or about
delays outside 5 to 20 ms; one machine and one worker. The rules do not
bound handler time outside execution or the transport remainder within a
pair.

Raw trial and warm-up records and the telemetry are gzipped; every file
`verification.json` lists matches its SHA-256 after decompression.
`run.log` and `peer-handshake.txt` were copied in after the run; the
handshake is identical to the one in the manifest.

## 20261115-01-diagnostic

Declared before it started as a functional check of the driver, pair
recording, telemetry and analysis on a busy machine. **Its timings are not
evidence of anything and are not summarised as results.** Revision
`a9e7a185`, clean tree; binaries `v3_block_time` `4333fbd8…`, `larql-server`
`bbbbd57c…` (built at `242fc0d1`; no Rust source changed since), frozen relay
`7b1097f8…`. 11:14:10 to 12:44:39 UTC. No quiet was requested: another
session's work had priority and its builds and real-model runs continued
throughout (one-minute load average 18 to 35).

What it shows, functionally:

- **The schedule ran as recorded.** All 80 trials ran, in exactly the order,
  pairs, rounds, arms and delays the manifest recorded before the first
  trial. No trial was left unrun, and every profiled trial returned worker
  timing for every exchange.
- **Warm-up did not settle** (131.7, 136.9, 131.6, 130.0 ms per position),
  so an admissible run would have stopped there; the diagnostic continued
  as declared.
- **Parity held in every trial**; nothing stopped the run.
- **Every trial and pair is recorded with its verdict.** Each trial carries
  its pair, round, delay order and wall-clock span, and its validity equals
  its pair's. 19 pairs are void by name: brackets drifting 2.8% to 102%,
  and in 14 of them the relay was also behind schedule. One pair (pair 0,
  `block-8`) is valid.
- **Void pairs are kept and shown but never fitted.** Every pair's slope is
  reported, void or not: `block-8`'s four read 8.98 (valid), −31.43, −8.84
  and 6.56 (void). Only the valid one enters the arm's list, and with one
  of four valid the arm's slope is reported as missing, as for every other
  arm (0 of 4 valid).
- **Telemetry was collected throughout**: 1,068 process samples at 5 s and
  178 state samples at 30 s, no sampling errors, no gap over 9 s. A `ps`
  sample cost 54 ms (median), at most 1.7 s under this load. The heaviest
  processes by CPU were `rustc` and `ld` (the other session's builds), then
  this run's relay and trial process. No thermal or performance warning was
  recorded; memory pressure stayed at 0, free memory 94% to 91%. Spotlight
  started 1,507 workers over 70 minutes.

### A pause inside a pair, and what it did to the one valid pair

At about 11:18:27 to 11:18:35 UTC the run's process group was stopped with
`SIGSTOP` for 5 seconds and resumed, to test that the run could be paused
for another session. Telemetry shows it as an 8 s gap between samples. The
stop fell inside trial-002, `block-8` at 20 ms in **pair 0, the one valid
pair**: that trial's coordinator time reads 123.7 ms per position against
34 to 36 in the trials beside it, 5.02 s in all, and the pair's slope reads
8.98 instead of about 3. Its brackets agreed within 0.7%, so it passed every
gate.

This is the case the draft's pausing rule names: a signal stop inside a
pair is invisible to the gates and corrupts that pair's slope. It is
recorded here as observed; the diagnostic's analysis is not altered.

### Files

Raw trial and warm-up records and the telemetry are gzipped; every file
`verification.json` lists matches its SHA-256 after decompression.
`run.log` is the driver's console output, copied in after the run.
