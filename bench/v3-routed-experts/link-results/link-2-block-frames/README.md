# LINK-2 block-frame addendum: runs

Runs under the LINK-2 block-frame addendum. Both runs below were made under
[revision 1](../../LINK-2-block-frames.md), which
[revision 2](../../LINK-2-block-frames-revision-2.md) supersedes because of
what they found. Nothing here qualifies the relay at these frame sizes, and
nothing here counts under revision 2.

| Run | Kind | Result |
|---|---|---|
| [`20261114-01-diagnostic`](20261114-01-diagnostic/calibration.json) | Declared diagnostic; does not count | **Fails**: 15 of 16 points pass; width 8 at 1 ms reads −53.7 µs against a 50 µs tolerance |
| [`20261114-02-chunk-trace`](20261114-02-chunk-trace/) | Mechanism investigation with a tracing copy of the relay; qualifies nothing | Locates the shortfall: see below |

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
