# LINK-2 block-frame addendum: runs

Runs of [`link_block_frames.py`](../../link_block_frames.py) under the
[block-frame addendum](../../LINK-2-block-frames.md). The addendum is not
frozen and nothing here qualifies the relay at these frame sizes.

| Run | Kind | Result |
|---|---|---|
| [`20261114-01-diagnostic`](20261114-01-diagnostic/calibration.json) | Declared diagnostic; does not count | **Fails**: 15 of 16 points pass; width 8 at 1 ms reads −53.7 µs against a 50 µs tolerance |

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
