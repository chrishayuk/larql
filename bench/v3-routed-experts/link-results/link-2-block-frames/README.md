# LINK-2 block-frame addendum: runs

Runs under the LINK-2 block-frame addendum.

**The frozen relay is qualified at these frame sizes under
[revision 2](../../LINK-2-block-frames-revision-2.md)**: two qualifying runs,
both passing all twelve points (24 of 24), with no threshold, binary or
protocol changed. The qualification covers the frozen relay `7b1097f8…` at
the frame sizes of block widths 1, 2, 4 and 8, unlimited rate, at the null
point and at 5 and 20 ms round trips. It does not cover a 1 ms round trip,
any rate-limited link, a physical network, or the timing of the inference
endpoints.

The two earlier runs were made under
[revision 1](../../LINK-2-block-frames.md), which revision 2 supersedes
because of what they found. They are kept unchanged below; neither counts
under revision 2.

| Run | Revision | Kind | Result |
|---|---|---|---|
| [`20261115-01-qualifying-1`](20261115-01-qualifying-1/calibration.json) | 2 | Qualifying | **Passes**: 12 of 12 points |
| [`20261115-02-qualifying-2`](20261115-02-qualifying-2/calibration.json) | 2 | Qualifying | **Passes**: 12 of 12 points |
| [`20261116-01-candidate-qualifying-1`](20261116-01-candidate-qualifying-1/calibration.json) | 2 | Qualifying, **candidate relay `b4068220…`** | **Passes**: 12 of 12 points |
| [`20261116-02-candidate-qualifying-2`](20261116-02-candidate-qualifying-2/calibration.json) | 2 | Qualifying, candidate relay | **Passes**: 12 of 12 points |
| [`20261114-01-diagnostic`](20261114-01-diagnostic/calibration.json) | 1 | Declared diagnostic; does not count | **Fails**: 15 of 16 points pass; width 8 at 1 ms reads −53.7 µs against a 50 µs tolerance |
| [`20261114-02-chunk-trace`](20261114-02-chunk-trace/) | 1 | Mechanism investigation with a tracing copy of the relay; qualifies nothing | Locates the shortfall: see below |

## Revision 2 qualification (2026-11-15)

Both runs: frozen relay `7b1097f8…`, probe `c1efe687…` (the binaries
revision 3 qualified with; probe sources unchanged since `dc46e89c`), at
revision `a5e9c79f` with no uncommitted change except the runs' own output
directories. Frozen protocol hashes in
[`frozen-protocols.json`](../../frozen-protocols.json) were verified before
the runs. Run 1 started 00:29:31 UTC, run 2 at 00:36:48 UTC, back to back.

Error against the configured round trip (judged half-width in brackets),
run 1 / run 2:

| Width | Null | 5 ms | 20 ms |
|---|---|---|---|
| 1 | +0.2 (±1.2) / +0.8 (±2.8) µs | +28.2 (±5.6) / +31.2 (±6.7) µs | +51.3 (±2.7) / +51.9 (±5.3) µs |
| 2 | +0.4 (±1.7) / +0.6 (±1.3) µs | +6.4 (±3.0) / +11.8 (±2.0) µs | +31.9 (±5.4) / +54.6 (±11.2) µs |
| 4 | +0.5 (±5.0) / −2.5 (±1.5) µs | −35.8 (±3.8) / −9.5 (±2.3) µs | −12.6 (±5.4) / +3.8 (±1.9) µs |
| 8 | −4.0 (±5.8) / −1.7 (±2.4) µs | **−96.3 (±2.7) / −95.5 (±1.9) µs** | −43.4 (±7.3) / −46.5 (±2.6) µs |

Tolerances: 25 µs at the null point, 250 µs at 5 ms, 1,000 µs at 20 ms. The
worst point, width 8 at 5 ms, is at about 39% of its tolerance in both runs.
The error moves from positive at width 1 to negative at width 8, the pattern
the revision 1 diagnostic and the chunk trace below describe.

Host state, recorded in each receipt:

- All three other sessions confirmed quiet before run 1; one had suspended
  its runner. The harness found no compiler or model process at the start
  or end of either run.
- Before run 1 the host was left to settle until the one-minute load was
  below 2.0 and the five-minute load below 3.5. **Load rose during the runs
  with no other heavy process present**, which is consistent with the
  spin-polling probe and relays themselves: run 1 went from 1.39 to 5.20
  (one minute), and run 2, started straight after, from 4.05 to 7.05. Run 2
  therefore began on a machine that was quiet by its process list but still
  carrying run 1's load. The protocol does not gate on load average; this is
  recorded, not judged.
- Console output of each run is in its `run.log`.

## 20261114-01-diagnostic

Declared as a diagnostic before it started. Frozen relay `7b1097f8…`,
revision 3's rules unchanged, all sixteen points measured, nothing excluded.

Error against the configured round trip (judged half-width in brackets):

| Width | Null | 1 ms | 5 ms | 20 ms |
|---|---|---|---|---|
| 1 | −0.6 µs (±2.6) | +18.5 µs (±4.8) | +26.6 µs (±2.2) | +50.6 µs (±13.3) |
| 2 | −0.3 µs (±6.0) | −6.1 µs (±7.5) | +3.0 µs (±7.8) | +38.7 µs (±17.4) |
| 4 | +1.7 µs (±3.5) | −35.3 µs (±9.8) | −49.6 µs (±12.4) | −16.5 µs (±6.7) |
| 8 | −3.2 µs (±3.0) | **−53.7 µs (±2.0)** | −78.4 µs (±1.6) | −85.4 µs (±2.9) |

At every non-zero delay the error moves from positive at width 1 to negative
at width 8. The null points are within 4 µs of zero.

Host state, in `host-before.txt` and `host-after.txt`:

- Three other sessions confirmed quiet beforehand; no compiler or model
  process was running at the start or end.
- 91% of memory free throughout; swap use and the swap-in, swap-out and
  page-out counters were identical before and after.
- One suspended process of another session held 4.9 GB resident at 0% CPU
  for the whole run.
- **The load average was still decaying**: 13.5 at the start and 5.8 at the
  end (one minute), because the other sessions had only just stopped. The
  machine was quiet by its process list but not long-settled, so this run is
  not claimed to be independent of host state.

The failure stands as recorded. No threshold, relay or protocol was changed
in response.

## 20261114-02-chunk-trace

A bounded investigation of why the diagnostic's error moves with frame size.
It used `v3_link_shaper_trace`, a copy of the relay with the same forwarding
logic and four timestamps per chunk, driven by the unchanged probe:
50 pairs per point, widths 1 and 8 at 1 ms, width 8 at 5 ms, and width 8 at
the null point. [`link_chunk_trace.py`](../../link_chunk_trace.py) splits
every exchange into stages on each relay's own clock.

**This is not the frozen relay and the machine was heavily loaded** (load
average about 40, see `host-before.txt`). The numbers below locate where
time goes; they do not measure the relay's accuracy and qualify nothing.
The null control run under the same load is clean, and the probe's error
through the tracing relay at width 8 and 1 ms (−59 µs) agrees with the
quiet diagnostic of the frozen relay (−54 µs).

Median per exchange, microseconds. "Excess" is shaped minus baseline minus
the configured delay for that stage.

| Stage | Width 8, 1 ms: baseline | shaped | excess | Width 1, 1 ms: excess | Width 8, null: excess |
|---|---|---|---|---|---|
| Request taken in | 45.1 | 21.0 | −24.1 | 0.0 | +1.0 |
| Request held | 6.1 | 504.1 | −2.0 | +2.9 | +0.2 |
| Peer turnaround | 46.9 | 59.5 | +12.7 | +10.9 | −0.7 |
| Response taken in | 199.1 | 82.0 | **−117.0** | −8.0 | +1.2 |
| Response held | 5.4 | 535.8 | +30.4 | +0.3 | 0.0 |
| Relay's view of the exchange | 304.6 | 1211.5 | −93.1 | +14.9 | −1.4 |
| Outside the relay (endpoints) | 63.0 | 96.9 | +33.9 | +15.8 | −0.3 |
| Probe error | | | −59.3 | +30.7 | −1.7 |

At 5 ms the same stages read −20.9 (request in), −106.9 (response in),
+35.6 (response held), +26.0 (turnaround) and +34.0 (endpoints).

### What it shows

1. **No chunk is ever under-delayed.** Of 29,607 chunks through the shaped
   relay at width 8 and 1 ms, none was written before its scheduled release;
   the shortest hold was 500.0 µs and the first chunk of a response is held
   506 µs. The delay itself is applied as configured.
2. **The shortfall is in how long a large frame takes to enter the relay.**
   With no delay the relay reads a chunk, writes it, reads the next: taking
   in a 369 kB response is interleaved with sending it on, and takes about
   199 µs. With a delay the relay only reads while the chunks wait, and
   takes the same response in about 82 µs. The request shows the same thing
   on a smaller frame (45 to 21 µs). The last byte therefore leaves about
   120 to 140 µs earlier than "the baseline exchange plus the configured
   round trip" predicts.
3. **Two smaller effects go the other way.** The last chunk of a large
   response is released about 31 µs late, because 23 chunks fall due within
   82 µs and writing them takes longer than that. And the endpoints and the
   peer's turnaround are slower after an idle delay, by 10 to 35 µs each;
   these are the positive errors seen at width 1, and they vary with host
   load.
4. **Frame size, not delay, sets the size of the effect.** It is the same at
   1 ms and 5 ms and absent at the null point, where both relays interleave
   alike.

So the error is not a wrong delay. The calibration's expectation, baseline
exchange plus configured round trip, assumes the relay's own forwarding cost
is the same with and without a delay. For a frame of several chunks it is
not: the zero-delay relay carries an interleaving cost of roughly 100 µs at
width 8 that the delayed relay does not pay.

Not established: how this splits between the relay's loop and kernel socket
buffering (no socket-buffer observations were taken), and the exact sizes on
a quiet machine. The receive side was not timestamped byte by byte; what
happens there is the "outside the relay" residual.

Raw traces are stored gzipped beside each point.

## The candidate relay `b4068220…` (2026-11-16)

This requalifies VERIFY-REMOTE-1 revision 2's relay, which adds the
generation-boundary mark (#865, named the candidate in #868), under the
unchanged frozen addendum revision 2.

**Both qualifying runs pass 12 of 12 points.**
- Run 1: 11:00:10 to 11:07:14 UTC. Run 2: 11:07:14 to 11:14:19 UTC.
- Main at `56ba34fd`, clean tree. The same quiet window as LINK-2's
  candidate runs.
- `check_qualification` accepts the two receipts for `b4068220…`.

Errors in µs (± judged half-width), run 1 / run 2:

| Width | Null | 5 ms | 20 ms |
|---|---|---|---|
| 1 | −1.3 (±6.6) / +0.1 (±1.3) | +32.3 (±4.2) / +34.6 (±8.8) | +54.4 (±4.2) / +57.1 (±3.0) |
| 2 | −0.4 (±1.7) / −0.4 (±2.2) | +19.4 (±7.9) / +17.0 (±6.2) | +28.9 (±8.2) / +36.2 (±3.4) |
| 4 | −1.1 (±1.8) / −0.3 (±1.9) | −25.1 (±3.7) / −42.1 (±2.7) | −16.2 (±5.0) / −21.3 (±5.1) |
| 8 | −2.0 (±3.3) / −2.0 (±2.8) | −89.8 (±2.2) / −94.0 (±3.3) | −75.3 (±3.4) / −81.0 (±3.5) |

**Against the certified build's runs:**
- About three quarters of the run pairs are within about 10 µs of them.
- The largest shifts:
  - width 8 at 20 ms: −75 / −81 µs here, against −43 / −47 there;
  - width 4 at 5 ms: −42.1 against −9.5 (run 2);
  - width 4 at 20 ms: −21.3 against +3.8 (run 2);
  - width 2 at 20 ms: +36.2 against +54.6 (run 2).
- All of them are far inside tolerance, which is 250 µs at 5 ms and
  1,000 µs at 20 ms.
- The cause is not identified: the mark's added loop work, the day, or the
  host. The relay's other work per chunk is unchanged.

Qualifying certifies nothing until a reviewed change makes `b4068220…`
the certified build.
