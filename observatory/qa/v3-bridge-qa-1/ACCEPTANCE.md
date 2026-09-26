# OBSERVATORY-V3-BRIDGE-QA-1

Status: PENDING MANUAL BROWSER ACCEPTANCE. Browser access was unavailable when
this kit was prepared. Parser checks are preparation, not a browser acceptance result.
No frontend, runtime, or live implementation changes were made for this kit.

Target: https://larql-observatory.chrishayuk.chatgpt.site

Record the date, browser/version, published site version, and pass/fail for each
check below. Published version 5 contains the bridge from PR #488.

Download the [original Paris recording](paris-original.jsonl),
[incomplete receipt fixture](QA-ONLY-incomplete-receipt.jsonl), and
[corrupt event fixture](QA-ONLY-corrupt-event.jsonl) before starting.
The [checksum file](SHA256SUMS) identifies the unchanged original.

1. Open **Paris recording** on the published site. Confirm the run is
   `gemma3-4b-france-lens`, schema `larql.run-record.v1`, receipt complete,
   1,446 source events and zero live drops. The source SHA-256 is below.
2. At position 5, use the FFN boundary and scrub forward and backward through
   L0, L24, L26 and L33. Confirm the inspector values in the table exactly.
   Repeat a boundary and confirm it has identical values.
3. At one fixed site, switch between Map, Trace, Context and Lenses and return
   to Map. Keep position, boundary, basis and camera mode fixed. The recorded
   coordinates and displayed trajectory must return unchanged.
4. Refresh the page. Confirm the same run/source hash, basis and values are
   restored. Reselect the same position/site and camera mode for the geometry
   comparison; restoring the prior cursor itself is not a bridge requirement.
5. Choose **Record → Save original record**. Compute its SHA-256 with
   `shasum -a 256 <downloaded-file>`. It must equal the original hash below.
   Reopen that download and repeat one early and one terminal checkpoint.
6. Open `QA-ONLY-incomplete-receipt.jsonl`. Confirm it visibly states incomplete
   evidence/prefix and does not present a completed terminal state. This is a
   deliberate test mutation of the receipt, not a new model capture.
7. Open `QA-ONLY-corrupt-event.jsonl`. Confirm an explicit event-log SHA-256
   refusal. The corrupt recording must not replace the accepted run or be
   presented as valid evidence. This is deliberately corrupt test data.

| Position / site | Carrier norm | Applied-write norm | Recorded x, y, z | Token 9079 rank / log p |
|---|---|---|---|---|
| 5 / L0 FFN | 805.0019365004472 | 338.9479644098229 | -37.302254, -22.289206, 9.100919 | 14055 / -48.97897028849158 |
| 5 / L24 FFN | 40683.051918590936 | 3227.9769670495766 | -1175.3624, -1530.7766, 837.23755 | 1 / -0.36038475576211226 |
| 5 / L26 FFN | 46456.14041240776 | 4000.87707332175 | -1397.9288, -1645.2517, 834.28467 | 1 / -0.0010202578566627096 |
| 5 / L33 FFN | 67842.71073478952 | 4660.802083471362 | -2227.6553, -2349.2349, 1214.0723 | 1 / -0.22265967015508892 |

Original SHA-256:
`12cdf4d9d06f5675233d71556c2158702cff88dd89efe42af72ab9dfb345dbd4`

Basis: `seed-24301-3x2560`, provider `seeded-orthonormal-v1`.
Basis SHA-256:
`95aa4aef2123ff801c6b2a6e8d62496739bdf480ffa6c44351c905a98bcbe06b`

Token 9079 is identified as Paris by the capture study. The canonical file has
no token spellings or prompt text, so the UI must not fabricate them. Terminal
means the completed receipt and final captured write; generated output is not
recorded. Probability may be displayed as exp(recorded log p); entropy and raw
logits remain unavailable.

## Acceptance result

- Date / browser / site version: PENDING
- Original opens: PENDING
- Bidirectional scrub: PENDING
- Stable geometry across views: PENDING
- Refresh restoration: PENDING
- Byte-identical save and reopen: PENDING
- Complete/incomplete/refused states: PENDING
- Verdict: PENDING

Keep LIVE-1 and the separate local Rust merge behind this gate.
