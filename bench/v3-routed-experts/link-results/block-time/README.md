# BLOCK-TIME-1: runs

Runs of [`block_time.py`](../../block_time.py) under
[BLOCK-TIME-1 revision 2](../../BLOCK-TIME-1.md). **No admissible run has
been made.** Both runs below are declared diagnostics on a heavily loaded
machine. They exercise the driver. Their timings are not evidence of
anything and are not summarised here.

| Run | Kind | Outcome |
|---|---|---|
| [`20261114-01-diagnostic`](20261114-01-diagnostic/analysis.json) | Declared diagnostic; does not count | Stopped at warm-up, by the protocol's rule |
| [`20261114-02-diagnostic`](20261114-02-diagnostic/analysis.json) | Declared diagnostic with `--continue-unsettled`; does not count | All 50 trials ran; 40 void by the bracket gate |

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
