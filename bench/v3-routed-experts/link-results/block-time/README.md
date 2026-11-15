# BLOCK-TIME-1: runs

Runs of [`block_time.py`](../../block_time.py) under
[BLOCK-TIME-1 revision 2](../../BLOCK-TIME-1.md).

**The one admissible run has been made: a partial result.** Both 20 ms
repeats are valid, and width 8 took 70.1% and 70.5% less time per evaluated
position than `sequential`. Every 5 ms trial is void by the bracket gate, so
the pre-registered three-point and differential RTT slopes are unavailable.
Revision 2 permits one admissible attempt; no other will be made under it.

| Run | Kind | Outcome |
|---|---|---|
| [`20261115-01-admissible`](20261115-01-admissible/analysis.json) | **Admissible** | Complete; 30 valid, 20 void by the bracket gate; 20 ms comparisons stand, slopes unavailable |
| [`20261114-01-diagnostic`](20261114-01-diagnostic/analysis.json) | Declared diagnostic; does not count | Stopped at warm-up, by the protocol's rule |
| [`20261114-02-diagnostic`](20261114-02-diagnostic/analysis.json) | Declared diagnostic with `--continue-unsettled`; does not count | All 50 trials ran; 40 void by the bracket gate |

## 20261115-01-admissible

GPT-OSS 20B (`gpt-oss-20b.vindex3`, stored weights), one CPU expert worker,
the supplied 216-token stream (152 prompt tokens, 56 measured positions from
position 160), LINK-1's thread settings, the frozen relay `7b1097f8…` at
unlimited rate. Revision `242fc0d1`, clean tree; `v3_block_time`
`4333fbd8…` and `larql-server` `bbbbd57c…`, built at that revision, which
contains the merged worker timing (#842) and block check (#839). Admitted
with the two passing block-frame qualifications
([`20261115-01`](../link-2-block-frames/20261115-01-qualifying-1/calibration.json),
[`20261115-02`](../link-2-block-frames/20261115-02-qualifying-2/calibration.json))
and the peer handshake, saved in the manifest and as `peer-handshake.txt`.
Started 09:27:31 UTC, finished 10:08:07 UTC.

**What this measures:** wall time per evaluated position when verifying a
supplied token stream, through a calibrated synthetic network. It is not
accepted-token or generation throughput, which needs a drafter and an
acceptance policy (VERIFY-N), and not a physical WAN.

### Gates

- **No stop.** Warm-up settled; every trial's logits matched the reference,
  which reproduced the saved control; every frame count was exact; no trial
  was behind the relay's schedule.
- **30 valid, 20 void, all void by the bracket gate** (1% limit). Void
  trials are kept as recorded and are not fitted, pooled or replaced.

| Repeat | Delay | Brackets (ms per position) | Drift | Trials |
|---|---|---|---|---|
| 1 | 0 ms | 128.3 → 127.3 | 0.8% | valid |
| 1 | 5 ms | 127.3 → 133.9 | **5.1%** | void |
| 1 | 20 ms | 133.9 → 135.1 | 0.9% | valid |
| 2 | 20 ms | 129.1 → 129.0 | 0.0% | valid |
| 2 | 5 ms | 129.0 → 158.9 | **20.8%** | void |
| 2 | 0 ms | 158.9 → 126.5 | **22.8%** | void |

So 20 ms has both repeats, 0 ms only repeat 1, and 5 ms none. The host's
one-minute load average was about 3 to 4 for most of the run and reached
14.0 during repeat 2's 0 ms arms (trial-043, about 10:04 UTC); no compiler
or model process was found at start or end. The cause of the drift is not
established here.

### Registered results

Time per evaluated position, valid trials only (ms):

| Arm | 0 ms (repeat 1) | 20 ms, repeat 1 / 2 | 20 ms mean |
|---|---|---|---|
| `sequential` | 127.9 | 617.9 / 617.8 | 617.9 |
| `block-1` | 137.0 | 622.9 / 620.1 | 621.5 |
| `block-2` | 128.1 | 378.7 / 369.3 | 374.0 |
| `block-4` | 123.1 | 249.3 / 251.6 | 250.5 |
| `block-8` | 120.2 | 184.6 / 182.2 | 183.4 |

- **Faster than `sequential` at 20 ms, in both repeats** (minimum effect
  1%): `block-8` by 70.1% and 70.5%, `block-4` by 59.7% and 59.3%,
  `block-2` by 38.7% and 40.2%. As a ratio of evaluated positions per
  second at 20 ms, `block-8` is 3.37 times `sequential` (means).
- **No statement at 0 ms or 5 ms**: neither delay has a valid trial in both
  repeats.
- **Slopes unavailable for every arm**: the three-point fit and the 5-to-20
  ms differential both need a valid 5 ms cell, and there is none. The RTT
  scaling hypothesis is neither confirmed nor refuted by this run.
- **Profiling overhead**, repeat 1 only: −4.0% to +0.4% across arms.

Decomposition per evaluated position, 20 ms, repeat 1 (ms):

| | `sequential` | `block-8` |
|---|---|---|
| Exchanges | 24 | 3 |
| Worker execution | 88.8 | 87.1 |
| Configured delay | 480 | 60 |
| Transport remainder (exchange − worker handler − delay) | 10.1 | 2.0 |
| Coordinator | 37.2 | 34.9 |
| Total | 617.9 | 184.6 |
| Bytes moved (request + response) | 1.39 MB | 1.38 MB |

Worker execution is 84 to 90 ms per position in every valid arm trial, so
the reduction comes from fewer round trips, not less expert work. The
balance of each total is worker handler time outside the execution
interval (1.8 and 0.5 ms here); with it, the parts sum exactly to the
measured wall time. Bytes moved are the same in every arm: the same data in fewer,
larger frames.

### Not registered

Two-point slopes from 0 to 20 ms (repeat 1 at 0 ms) read 24.5, 24.2, 12.3,
6.4 and 3.2 against predictions of 24, 24, 12, 6 and 3. They are reported
as an observation only: the protocol excludes the zero-delay point from the
slope on purpose, because the zero-delay relay forwards large frames
differently (see the block-frame chunk trace), and they rest on one repeat
at 0 ms. They do not substitute for the missing registered slope.

### Files

Raw trial and warm-up records are gzipped; every file `verification.json`
lists matches its SHA-256 after decompression. `run.log` is the driver's
console output and `peer-handshake.txt` the saved handshake; neither is in
`verification.json`, which the driver wrote before they were copied in.

## Earlier diagnostics

The two runs below are declared diagnostics on a heavily loaded machine.
They exercise the driver. Their timings are not evidence of anything and
are not summarised here.

Both used binaries built from a local, unpushed merge of the timing branch
and the worker-timing branch (#842), because the trial binary needs the
worker-timing change and it was not merged. The manifests record the
revision, the binary hashes and the harness hashes. They predate the
provenance record (source hashes and environment) the driver now writes.

Raw trial records are stored gzipped. `verification.json` in each run lists
the SHA-256 of every file as it was written, before compression.

## 20261114-01-diagnostic

Four warm-up trials read 494, 348, 322 and 296 ms per position and never
agreed within 1%, so the driver stopped before the schedule. That is the
rule working on a busy machine. The reference trial reproduced the saved
control under the derived alignment, frame counts were exact, and every
exchange returned worker timing.

## 20261114-02-diagnostic

Run with `--continue-unsettled`, which an admissible run refuses.

- **All 50 scheduled trials ran and left a record.** The run did not stop:
  no trial's logits differed from the reference.
- **40 trials are void, each by a named gate.** Every arm and unprofiled
  trial failed the bracket gate (brackets differed by 3.7% to 48%, against
  1%); seven of them were also behind the relay's schedule. No trial had a
  frame-count defect. The 8 brackets and 2 no-relay controls carry no
  bracket gate of their own.
- **Worker timing was complete** for all 40 profiled trials, every exchange.
  The 10 unprofiled trials did not ask for it.
- **The analysis made no statement.** With every arm trial void, each cell
  is missing, each slope is reported as not fitted, and no "faster" claim is
  evaluated. Void trials were kept, not fitted and not replaced.

What this run shows is that the driver starts the worker and relays, runs
every arm and control through them, applies the parity, frame-count,
bracket and relay gates, and declines to analyse void data. What it does
not show is the analysis on valid real data: that path has only been run on
synthetic records in the tests.
