# BLOCK-TIME-1: block verification width against round-trip time (pre-registration, DRAFT)

Status: **draft, revised 2026-11-14 after review. Approved in principle; not
frozen. No timed run has been made.** The instrument is built and tested on
synthetic records but has not run end to end. After freezing, any change to
a parameter, gate or rule is a new revision, not an edit.

Changes in this revision: worker-side timing is part of the decomposition
(it needs the multi-row worker-timing change, which is separate and not yet
merged); every arm also runs unprofiled at 0 ms; the saved-control alignment
is derived and tested, not stated as a position range; the thresholds are
described as a model-fit band and a minimum effect, not significance.

Correctness of the path being timed is recorded in
[BLOCK-CHECK](BLOCK-CHECK.md). This protocol measures time only.

## Question

On GPT-OSS 20B with experts on one remote worker, does elapsed time per
evaluated position fall as verification width rises, by an amount that grows
with round-trip time as the count of dependent exchanges predicts?

## What is and is not measured

Measured: **evaluated positions per second**. A block of width W evaluates W
positions whose tokens the caller already holds.

Not measured, and not to be inferred: accepted tokens per second. There is
no drafter and no acceptance rule here. Every token is supplied, every
position is committed, and no position is rejected. A width-8 block is an
eight-position verification, not eight generated tokens.

## Fixed conditions

| | |
|---|---|
| Model | GPT-OSS 20B VINDEX3 container, identified by SHA-256 of `index.json` and `system_graph.json` |
| Arithmetic | CPU production backend, stored bytes (`RepresentationSource::Stored`), F32 frames; coordinator and worker alike |
| Topology | One worker owning layers 0 to 23 and experts 0 to 31. Two workers are a separate, later decision |
| Workload | The long prompt of `prompts-v2.json` followed by the 64 saved control ids: 216 token ids, supplied to every arm |
| Link | The frozen relay, one per worker, unlimited rate; delays as below |
| Settings | LINK-1's: `LARQL_CPU_WORKERS=8`, `RAYON_NUM_THREADS=8`, `VECLIB_MAXIMUM_THREADS=1`, `TOKIO_WORKER_THREADS=2` |

Every arm consumes the same 216 ids in order. No arm samples: the stream is
supplied, so arms cannot diverge in what they execute.

## Arms

| Arm | Execution | Frames per routed layer |
|---|---|---|
| `sequential` | one position per step | one single-row frame per position |
| `block-1` | block session, width 1 | one multi-row frame per position |
| `block-2`, `block-4`, `block-8` | block session, width 2, 4, 8 | one multi-row frame per block |

`block-1` is a control, not a result: it has the exchange count of
`sequential` and the frame route of the block arms, so it separates the cost
of the route and of the block session from the effect of width.

## Points

Round trips of **0 (relay at zero delay), 1, 5 and 20 ms**, unlimited rate.
One further control, `direct` (no relay), is run for `sequential` only and
prices the relay itself; it is not part of any comparison.

## Trial

Workers start once and stay resident. A trial is one fresh coordinator
process running one arm at one point over the whole stream.

- Positions 0 to 159 are executed in the arm's own mode and **not timed**.
  They warm the connection, the worker and the caches.
- Positions 160 to 215 are the **measured region**: 56 positions, which is
  28, 14 and 7 whole blocks at widths 2, 4 and 8.
- The clock is monotonic wall time around the measured region, inside the
  process. Startup, load and the untimed positions are excluded. Logits are
  retained in memory and hashed after the clock stops.

Recorded per trial: wall time of the measured region and of each step or
block in it; for every exchange, the coordinator-side round-trip time, the
request and response frame bytes and the rows carried; frame counts per
route; SHA-256 and argmax of every position's logits bits; the relay's own
lateness and byte records. A **profiled** trial also asks the worker, per
exchange, for its handler, queue, execution and expert-transform times. The
frame is the same bytes either way.

## Schedule

Warm-up `sequential` trials at 0 ms until two consecutive measured-region
times agree within 1%, at most four. If they never agree the run stops.

Then two repeats. In each, the four delays are visited in order (reversed in
the second repeat). At each delay the five arms run in an order rotated by
one position per delay, reversed in the second repeat. A `sequential` trial
at 0 ms brackets every delay, before and after.

Every arm trial, bracket and control above is profiled. At 0 ms each arm
also runs once per repeat **unprofiled**, directly after the profiled arms
of that delay and inside the same brackets, so the cost of asking the worker
for its timing is measured and not assumed.

That is 40 arm trials, 10 unprofiled trials, 10 bracket trials and 2
`direct` trials: 62 in the order `block_time_model.schedule()` produces.
Estimated duration is 70 to 90 minutes, dominated by `sequential` and
`block-1` at 20 ms.

## Validity gates

A **trial is valid** only if all hold:

1. **Parity.** Every position's logits hash equals the reference recorded by
   the first warm-up trial, profiled or not. A mismatch in any trial stops
   the run: nothing is timed on a path that is not exact.

   The reference itself must reproduce the saved control. The decoder's
   convention is that the logits at position `i` are the distribution over
   the token at position `i + 1`. So for a prompt of `P` ids in a stream of
   `N`, the reference's argmax at each position `i` from `P − 1` to `N − 2`
   must equal the supplied id at `i + 1`: one comparison per generated id,
   the first at the last prompt position, and the last position's logits not
   compared. The rule is `control_alignment` in `block_time_model.py`, which
   takes `P` from the ids file. Its tests shift the alignment one position
   early and one late and require every comparison to fail, and check that a
   wrong first generated id is caught at `P − 1`. A reference that does not
   reproduce the control stops the run before anything is timed.
2. **Frame counts.** `sequential`: 24 single-row frames per position and no
   multi-row frame. `block-W`: 24 multi-row frames per block and no
   single-row frame.
3. **Brackets.** The `sequential` 0 ms trials on either side of its delay
   agree within 1%.
4. **Relay on schedule.** The relay's summed lateness per measured position
   is at most 1% of the bracket's time per position (LINK-1's gate).

A void trial is kept as a diagnostic and never fitted, pooled or replaced.

The **run is admissible** only if two passing
[block-frame qualification](LINK-2-block-frames.md) receipts for the relay
binary in use are supplied, the peer sessions confirmed a quiet machine
before the first trial and the handshake is saved with the receipts, and no
compiler or model process other than this experiment's was running at start
or end.

## Analysis, fixed in advance

`t(arm, delay)` is the measured-region wall time divided by 56, the mean of
the valid repeats. A cell with no valid trial is reported as missing.

1. **Table.** `t` for all twenty cells, with both repeats shown.
2. **Slope.** For each arm, a least-squares line of `t` on configured round
   trip over the four delays. The network model predicts a slope of
   `24 / W` round trips per position (24 for `sequential` and `block-1`).
   Reported: fitted slope, predicted slope, their ratio, and the intercept.
3. **Decomposition per cell**, from the per-exchange records:

   ```text
   wall = coordinator + exchanges
   exchanges = configured delay + worker handler + unattributed
   ```

   - *coordinator*: wall time minus the sum of coordinator-side round trips
     (attention, routing, reduction, head, session bookkeeping);
   - *configured delay*: exchanges × configured round trip;
   - *worker handler*: the worker's own time per request, with its queue,
     execution and expert-transform parts reported beside it;
   - *unattributed*: what remains of the exchanges. It is serialisation,
     HTTP, scheduling, loopback and relay processing together. **It is not
     network latency and is never labelled as such.**

   Worker figures are reported for a trial only if every one of its
   exchanges returned them.

   **This decomposition is valid for one worker only.** With one worker the
   exchanges of a trial are strictly sequential, so their round trips sum to
   the time the coordinator spent waiting. With several workers the
   exchanges of a layer overlap, and summing every worker's round trip or
   handler time would count concurrent time more than once. A multi-worker
   protocol must keep aggregate worker time apart from the completion of the
   slowest worker per layer, which is what the critical path sees. The
   analysis here refuses nothing on that ground because this protocol admits
   one worker; the sum must not be carried into a multi-worker revision.
4. **Bytes**: frame bytes sent and received per position per arm, and the
   bytes the relay carried.
5. **Profiling overhead**: profiled against unprofiled per-position time at
   0 ms, per arm and repeat.

The worker timing depends on the multi-row worker-timing change (profile
header only, frame bytes identical with and without it). That change is a
separate PR and must be merged, and the binaries built from it, before this
protocol is frozen.

## Statements this run can support

Fixed before any data:

- **"Elapsed time per evaluated position follows the exchange model"** for
  an arm only if its fitted slope is within 10% of `24 / W`.
- **"Width W reduced elapsed time per evaluated position at delay D"** only
  if, in each repeat separately, `t(block-W, D)` is below `t(sequential, D)`
  by more than 1% of `t(sequential, D)`. Both repeats' reductions are
  reported individually.

The 10% band is a model-fit criterion and the 1% a minimum effect that must
appear in both repeats. Neither is a significance level: two repeats cannot
support a confidence statement, and none is made.
- **"Block execution costs more (or less) at zero added delay"** from the
  0 ms column and the `block-1` control, stated with its size.

If a slope is outside 10%, or a block arm is not faster at 5 or 20 ms, that
is the result, reported with the decomposition that locates it.

## Not claimed whatever the outcome

Accepted-token or generation throughput; any benefit of speculative
decoding; behaviour on a physical network; rate-limited links; two or more
workers; any reduction in expert weight reads; any model other than
GPT-OSS 20B on this host.

## Rules

- One admissible run. Development and diagnostic runs are declared before
  they start and never count.
- No trial, repeat or cell is excluded after the fact.
- If the run stops on a gate, the stop and its cause are reported with the
  receipts. A second run needs a new decision.
- No change to the relay, to LINK-2 revision 3, or to numerical execution.
- No automatic extension to two workers.

## Implementation

- [`v3_block_time`](../../crates/larql-cli/examples/v3_block_time.rs): one
  arm over the supplied ids, opened with the stored-bytes policy. It times
  each exchange in a transport wrapper and judges nothing.
- [`block_time_model.py`](block_time_model.py): schedule, gates, alignment
  and analysis, pure, with `test_block_time_model.py`.
- [`block_time.py`](block_time.py): starts the worker and relays, runs the
  schedule and applies the model. `--admissible` refuses without two passing
  [block-frame](LINK-2-block-frames.md) receipts for the relay in use, the
  saved peer handshake and a quiet machine. `--diagnostic` runs are labelled
  and never count.

## Before freezing

1. The multi-row worker-timing change is reviewed and merged.
2. A declared diagnostic run completes end to end.
3. The block-frame qualification question below is settled.

## Open: the relay at large frames and short delays

A development check of the frozen relay at the block frame sizes (ten pairs
per point, on a loaded machine, not a diagnostic of record) gave these
errors against the configured 1 ms round trip: width 1 +32 µs, width 2
−3 µs, width 4 −34 µs, width 8 −88 µs. Width 8 read −72 µs at 5 ms and
−18 µs at 20 ms. LINK-2's tolerance at 1 ms is 50 µs.

If a proper diagnostic confirms this, the block-frame qualification fails at
(width 8, 1 ms) by its own rule and this experiment does not run as drawn.
It would also matter directly: 88 µs per exchange is 9% of the configured
delay there, against a 10% slope band. The tolerance is not to be changed.
The options are a decision for the protocol's owner, for example dropping
the 1 ms delay from this experiment or treating it as below the relay's
resolution at width 8.
