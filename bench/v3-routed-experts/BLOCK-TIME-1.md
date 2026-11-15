# BLOCK-TIME-1, revision 2: block verification width against round-trip time (pre-registration, frozen)

Status: **revision 2, approved and frozen on 2026-11-14. No timed run has
been made.** Any change to a parameter, gate or rule below is a new revision,
not an edit. The frozen files and their hashes are in
[`frozen-protocols.json`](frozen-protocols.json), and a test fails if any of
them changes. What is frozen is this document and the files that hold its
parameters and decision rules. The driver and the trial binary are
implementation: they may be corrected without a new revision, and every
receipt records the hashes of the ones that produced it.

Freezing fixes the contract. It does not make a run admissible: the
conditions under "Before the admissible run" must hold first, and the run
needs its own go decision.

**What revision 2 changed from the first draft.** That draft compared the
arms at 0, 1, 5 and 20 ms. The relay could not be qualified at block frame
sizes for 1 ms under the existing calibration
([block-frame addendum, revision 2](LINK-2-block-frames-revision-2.md)). So:

- the comparison is at **0, 5 and 20 ms**, and **no trial is run at 1 ms**:
  every delay in the schedule has a relay qualification behind it;
- the slope is fitted over the three delays with its residuals, and a
  **5-to-20 ms differential slope** is reported beside it, which does not
  use the zero-delay arm;
- the interpretation separates the relay's buffering behaviour from the
  configured delay.

Arms, supplied ids, measured region, profiling controls, one worker and the
10% and 1% thresholds are as first drafted.

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

Round trips of **0 (relay at zero delay), 5 and 20 ms**, unlimited rate. The
relay is qualified at block frame sizes for 5 and 20 ms by the block-frame
addendum. The 0 ms arm is the zero-delay relay, a measured control.

**No trial is run at 1 ms.** The additive-delay calibration failed its
tolerance for width-8 frames at that round trip, so the relay is unqualified
there at block frame sizes. The failed diagnostic and the chunk trace are
kept as history ([runs](link-results/link-2-block-frames/README.md)). This
is not a claim that 1 ms cannot be measured. Any work at 1 ms needs its own
separately declared protocol, with a calibration that qualifies it.

One further control, `direct` (no relay), is run for `sequential` only and
prices the relay itself; it is not part of any comparison.

### The zero-delay arm is a different transport regime

The zero-delay relay and the delayed relay do not forward a large frame
alike. With no delay the relay interleaves reading and writing; with a delay
it reads the whole frame while the chunks wait. A chunk trace measured the
difference at about 100 µs per width-8 exchange, less for smaller frames
([runs](link-results/link-2-block-frames/README.md)). At three exchanges per
position for `block-8` that is roughly 0.3 ms per position, against a
per-position time expected to be of the order of 100 ms.

It is small, but it is systematic and it falls on the wide arms at 0 ms
only. So the 0 ms arm is read as "block execution through the zero-delay
relay", not as the delayed arms extrapolated to zero, and the analysis
reports a slope that does not use it.

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

Then two repeats. In each, the three delays are visited in order (reversed
in the second repeat). At each delay the five arms run in an order rotated by
one position per delay, reversed in the second repeat. A `sequential` trial
at 0 ms brackets every delay, before and after.

Every arm trial, bracket and control above is profiled. At 0 ms each arm
also runs once per repeat **unprofiled**, directly after the profiled arms
of that delay and inside the same brackets, so the cost of asking the worker
for its timing is measured and not assumed.

That is 30 arm trials, 10 unprofiled trials, 8 bracket trials and 2 `direct`
trials: 50 in the order `block_time_model.schedule()` produces. Estimated
duration is 60 to 80 minutes, dominated by `sequential` and `block-1` at
20 ms.

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
[block-frame qualification](LINK-2-block-frames-revision-2.md) receipts for the relay
binary in use are supplied, the peer sessions confirmed a quiet machine
before the first trial and the handshake is saved with the receipts, and no
compiler or model process other than this experiment's was running at start
or end.

## Analysis, fixed in advance

`t(arm, delay)` is the measured-region wall time divided by 56, the mean of
the valid repeats. A cell with no valid trial is reported as missing.

1. **Table.** `t` for all fifteen cells, with both repeats shown.
2. **Slope.** The exchange model predicts `24 / W` round trips per position
   (24 for `sequential` and `block-1`, then 12, 6 and 3). Two estimates per
   arm, each with its ratio to the prediction:
   - **Three-point fit**: a least-squares line of `t` on configured round
     trip over 0, 5 and 20 ms, with its intercept and the residual at each
     of the three delays.
   - **Differential slope**: `S = (t(20) − t(5)) / 15 ms`, from the two
     qualified delays alone, also reported for each repeat separately. It
     does not use the zero-delay arm.
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
  an arm only if **both** its three-point fitted slope and its differential
  slope are within 10% of `24 / W`. Each is also reported on its own, so a
  disagreement between them, which would point at the zero-delay regime, is
  visible and not averaged away.
- **"Width W reduced elapsed time per evaluated position at delay D"**, for
  D of 5 or 20 ms, only if, in each repeat separately, `t(block-W, D)` is below `t(sequential, D)`
  by more than 1% of `t(sequential, D)`. Both repeats' reductions are
  reported individually.

The 10% band is a model-fit criterion and the 1% a minimum effect that must
appear in both repeats. Neither is a significance level: two repeats cannot
support a confidence statement, and none is made.
- **"Block execution costs more (or less) through the zero-delay relay"**
  from the 0 ms column and the `block-1` control, stated with its size and
  with the regime note above. It is a statement about that arm, not about a
  network with no latency.

If a slope is outside 10%, or a block arm is not faster at 5 or 20 ms, that
is the result, reported with the decomposition that locates it.

**Interpretation keeps two things apart.** The configured round trip is what
the relay adds to each exchange by design. How the relay and the sockets
buffer a frame is a property of this instrument and changes with frame size
and with whether a delay is present. A difference between arms is attributed
to the configured delay only as far as the slope and the decomposition
support it; the remainder is reported as unattributed, never as network
latency.

## Not claimed whatever the outcome

Accepted-token or generation throughput; any benefit of speculative
decoding; behaviour on a physical network; anything at a 1 ms round trip;
rate-limited links; two or more
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
  [block-frame](LINK-2-block-frames-revision-2.md) receipts for the relay in use, the
  saved peer handshake and a quiet machine. `--diagnostic` runs are labelled
  and never count.

## Before the admissible run

1. The multi-row worker-timing change is reviewed and merged, and the
   binaries are built from it. The decomposition's worker figures depend on
   it; without it the profiled trials return no worker timing.
2. A declared diagnostic run of the driver completes end to end.
3. Two passing block-frame qualifications under addendum revision 2, on a
   quiet machine.
4. A separate go decision for the run itself.
