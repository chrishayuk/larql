# LINK-2 block-frame addendum (pre-registration, DRAFT)

Status: **superseded on 2026-11-14 by
[revision 2](LINK-2-block-frames-revision-2.md), and kept as the record of
revision 1. It was never frozen. Its one declared diagnostic
([receipt](link-results/link-2-block-frames/20261114-01-diagnostic/calibration.json))
failed at width 8 and a 1 ms round trip, and that failure stands as
recorded.** No run under this revision counts under any other. The text below
is revision 1 as it stood, apart from this paragraph.

It adds evidence beside [LINK-2 revision 3](LINK-2-revision-3.md). It does
not amend revision 3, re-score any run made under it, or touch its receipts.
The relay binary is the one revision 3 qualified (`7b1097f8…`) and is not
modified.

## Why

Revision 3 qualified the relay's added cost with one frame geometry: the
single-row GPT-OSS exchange, 11,576 request bytes and 46,136 response bytes.
[BLOCK-TIME-1](BLOCK-TIME-1.md) sends multi-row frames up to eight times
larger. At unlimited configured rate the relay's delay does not depend on
frame size by design, but socket buffering and segmentation of a larger
frame could change what a peer observes. That is an expectation the existing
qualification did not measure.

## Question

At unlimited configured rate, does the frozen relay add its configured
round-trip delay, within the unchanged LINK-2 tolerance, to exchanges of the
multi-row frame sizes for block widths 1, 2, 4 and 8?

## Instrument

Revision 3's, unchanged: one long-lived `v3_link_pair` process holding the
client and the echo peer, two relay processes per point (baseline and
shaped), one connection through each, default scheduling class.

Only the frame sizes differ. They are the one-worker GPT-OSS multi-row
frames (hidden 2,880, four experts per row), from the frame definition in
[`v3-expert-rows.md`](../../docs/ffn/v3-expert-rows.md):

| Width | Request bytes | Response bytes |
|---|---|---|
| 1 | 11,588 | 46,156 |
| 2 | 23,132 | 92,268 |
| 4 | 46,220 | 184,492 |
| 8 | 92,396 | 368,940 |

HTTP framing is not included, as in revision 3. The sizes live in
[`block_frames.json`](block_frames.json) and are pinned against the wire
encoder: a test in `larql-router-protocol` encodes a `VXR1` and a `VXS1`
frame of each shape and requires their lengths to equal the recorded ones.

## Points

For each of the four widths: the **null point** (the "shaped" relay
configured exactly as the baseline) and round trips of **1, 5 and 20 ms**,
all at unlimited rate. Sixteen points. These are the delays BLOCK-TIME-1
uses. No rate-limited point is measured: bandwidth shaping is outside
BLOCK-TIME-1, and this addendum says nothing about it at these frame sizes.

Expected cost per exchange is the configured round trip, for every width.

## Measurement and rules

Every parameter and rule is revision 3's, unchanged, and is read from
`link_pair_model.py`, not restated in a second implementation:

- 20 warm-up exchanges per connection; blocks of 10; pairs in the order
  baseline, shaped, shaped, baseline; 200 pairs per point; estimate is the
  median pair difference.
- Circular block bootstrap, block 20, 2,000 resamples, fixed recorded seed;
  the half-width is doubled before it is judged.
- Accuracy: `|estimate − expected| ≤ max(5% of expected, 50 µs)`.
- Precision: doubled half-width at most half the tolerance.
- Null point: `|estimate| ≤ 25 µs`, doubled half-width at most 12.5 µs.
- Autocorrelation at lags 1 to 30 is reported and not gated.
- No exchange, pair or point is excluded. A run is void only if the
  instrument fails to complete, and is recorded as void with its cause.

A **run passes** only if all sixteen points pass.

**Qualification of these frame sizes requires two complete passing runs**, on
the frozen relay binary, on a machine the other sessions have confirmed
quiet, with no compiler or model process running.

## Fixed in advance

- Diagnostic runs are declared before they start and never count.
- **If either qualifying run fails, work stops.** BLOCK-TIME-1 is not run.
  The failure is reported with its receipts and the cause investigated
  before any new decision. There is no third attempt, no relay change and no
  threshold change without one.
- A failure here does not invalidate revision 3. It would mean the relay is
  not shown to deliver its delay at the larger geometry.

## Scope of a pass

The same as revision 3's, at these frame sizes and unlimited rate only: the
cost the frozen relay adds to an exchange as seen by the paired polling
probe. It does not certify the timing of the inference endpoints, any
rate-limited link, or a physical network. Every gate of BLOCK-TIME-1 stays
in force independently.

## Implementation

[`link_block_frames.py`](link_block_frames.py) drives `v3_link_pair` with
these geometries and points and judges each with `link_pair_model.judge`,
unchanged. [`link_block_frames_model.py`](link_block_frames_model.py) holds
the points and the receipt check BLOCK-TIME-1 uses; a revision 3 receipt is
not accepted as a block-frame receipt, nor the reverse. No change to the
relay, the probe, revision 3's harness or any runtime crate.

Estimated measuring time: about 17 minutes per run, dominated by the 20 ms
points.

## Development check (does not count)

While building the harness the frozen probe and relay were run briefly at
these sizes to confirm they carry a 369 kB frame: ten pairs per point, on a
machine under heavy load from other sessions. This was not a declared
diagnostic and qualifies nothing. Median error against the configured round
trip:

| Width | 1 ms | 5 ms | 20 ms | Null |
|---|---|---|---|---|
| 1 | +32 µs | | | |
| 2 | −3 µs | | | |
| 4 | −34 µs | | | |
| 8 | −88 µs | −72 µs | −18 µs | +7 µs |

The error moves with frame size, and at width 8 and 1 ms it is beyond the
50 µs tolerance. A possible cause is that the shaped relay releases a large
frame's chunks together after the delay, so the peer receives them faster
than through the baseline relay; this is not established.

If a declared diagnostic on a quiet machine shows the same, a qualifying run
would fail at (width 8, 1 ms). The rules above are not to be adjusted to
avoid that. Whether to run the diagnostic first, and what BLOCK-TIME-1 does
about the 1 ms delay if it is confirmed, is a decision to make before this
addendum is frozen.
