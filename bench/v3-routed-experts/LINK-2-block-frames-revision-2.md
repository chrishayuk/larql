# LINK-2 block-frame addendum, revision 2 (pre-registration, frozen)

Status: **approved and frozen on 2026-11-14. No diagnostic or qualifying run
has been made under it.** Any change to a parameter or rule below is a new
revision, not an edit. The frozen files and their hashes are in
[`frozen-protocols.json`](frozen-protocols.json), and a test fails if any of
them changes.

It supersedes [revision 1](LINK-2-block-frames.md), which stays as written
with its failed diagnostic. It does not amend
[LINK-2 revision 3](LINK-2-revision-3.md), re-score any run, or touch any
receipt. The relay binary is the one revision 3 qualified (`7b1097f8…`) and
is not modified.

## What changes from revision 1

| | Revision 1 | Revision 2 |
|---|---|---|
| Round trips | 1, 5 and 20 ms | **5 and 20 ms** |
| Points | 16 | **12** |
| Everything else | | unchanged |

Nothing else changes: not the accuracy tolerance, the precision rule, the
bootstrap, the null bound, the frame sizes, the two-run requirement or the
stop rule.

## Why

Revision 1's declared diagnostic passed fifteen points and failed one:
width 8 at a 1 ms round trip read −53.7 µs against the 50 µs tolerance, and
the error moved with frame size at every delay
([runs](link-results/link-2-block-frames/README.md)).

A chunk trace then located it. The relay applies its delay as configured:
no chunk left before its scheduled release. The calibration's expectation is
what does not hold. It takes the cost of an exchange to be the zero-delay
exchange plus the configured round trip, which assumes the relay's own
forwarding cost is the same with and without a delay. For a frame of many
chunks it is not: with no delay the relay interleaves reading and writing
and takes a 369 kB response in about 199 µs; with a delay it only reads
while the chunks wait, and takes it in about 82 µs.

**What this does and does not say about 1 ms.** The additive-delay
calibration failed its tolerance for width-8 frames at a 1 ms round trip.
That is a statement about this calibration model at that point. It is not a
finding that 1 ms cannot be measured: a calibration that does not rest on
the zero-delay baseline, for example one timed from the first byte, might
qualify it. None has been designed. Until one is, 1 ms at these frame sizes
is unqualified, and evidence taken there is diagnostic only.

The same effect is present at 5 and 20 ms, where it is small against the
tolerance: about −78 and −85 µs at width 8 in the diagnostic, which is 1.6%
and 0.4% of the configured round trip, inside a 5% tolerance.

## Question

At unlimited configured rate, does the frozen relay add its configured
round-trip delay, within the unchanged LINK-2 tolerance, to exchanges of the
multi-row frame sizes for block widths 1, 2, 4 and 8, at round trips of 5
and 20 ms?

## Instrument and frame sizes

Revision 3's instrument, unchanged: one long-lived `v3_link_pair` process,
two relay processes per point, one connection through each, default
scheduling class. The frame sizes are revision 1's, from
[`block_frames.json`](block_frames.json), pinned against the wire encoder by
a test in `larql-router-protocol`:

| Width | Request bytes | Response bytes |
|---|---|---|
| 1 | 11,588 | 46,156 |
| 2 | 23,132 | 92,268 |
| 4 | 46,220 | 184,492 |
| 8 | 92,396 | 368,940 |

## Points

For each of the four widths: the **null point** and round trips of **5 and
20 ms**, all at unlimited rate. Twelve points. Expected cost per exchange is
the configured round trip.

## Measurement and rules

Revision 3's, unchanged, read from `link_pair_model.py`:

- 20 warm-up exchanges per connection; blocks of 10; pairs in the order
  baseline, shaped, shaped, baseline; 200 pairs per point; estimate is the
  median pair difference.
- Circular block bootstrap, block 20, 2,000 resamples, fixed recorded seed;
  the half-width is doubled before it is judged.
- Accuracy: `|estimate − expected| ≤ max(5% of expected, 50 µs)`.
- Precision: doubled half-width at most half the tolerance.
- Null point: `|estimate| ≤ 25 µs`, doubled half-width at most 12.5 µs.
- Autocorrelation at lags 1 to 30 is reported and not gated.
- No exchange, pair or point is excluded.

A **run passes** only if all twelve points pass. **Qualification requires two
complete passing runs**, on the frozen relay binary, on a machine the other
sessions have confirmed quiet, with no compiler or model process running.

## Fixed in advance

- Diagnostic runs are declared before they start and never count.
- **Nothing measured under revision 1 counts here.** Its diagnostic passed at
  5 and 20 ms, but it was a diagnostic, under another revision, on a machine
  whose load average was still settling. The harness refuses a revision 1
  receipt by name.
- **If either qualifying run fails, work stops.** BLOCK-TIME-1 is not run,
  and the failure is reported with its receipts before any new decision.
- A failure here does not invalidate revision 3.

## Scope of a pass

The cost the frozen relay adds to an exchange as seen by the paired polling
probe, at these frame sizes, at 5 and 20 ms, at unlimited rate. It does not
certify the timing of the inference endpoints, a 1 ms round trip, any
rate-limited link, or a physical network.

A pass also does not say the zero-delay relay and the delayed relay forward
a large frame alike. They do not, by about 100 µs per width-8 exchange.
[BLOCK-TIME-1](BLOCK-TIME-1.md) states what follows from that.

## Implementation

[`link_block_frames.py`](link_block_frames.py) and
[`link_block_frames_model.py`](link_block_frames_model.py), as for revision
1 with the points above. Tests cover the points, the verdict, the receipt
check, the refusal of revision 1 receipts by name, and that the recorded
revision 1 diagnostic cannot be read as a revision 2 run.

Estimated measuring time: about seven minutes per run.
