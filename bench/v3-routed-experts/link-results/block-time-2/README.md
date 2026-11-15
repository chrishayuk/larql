# BLOCK-TIME-2: runs

Runs of [`block_time_2.py`](../../block_time_2.py) under the
[BLOCK-TIME-2](../../BLOCK-TIME-2.md) draft. **No admissible run has been
made, and the protocol is not frozen.**

| Run | Kind | Outcome |
|---|---|---|
| [`20261115-01-diagnostic`](20261115-01-diagnostic/analysis.json) | Declared functional diagnostic with `--continue-unsettled` on a heavily loaded host; does not count | Complete: 80 of 80 trials, no stop; 1 pair valid, 19 void; no arm reportable |

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
