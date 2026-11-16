# VERIFY-REMOTE-1: runs

These are runs of [`verify_remote.py`](../../verify_remote.py) under the
VERIFY-REMOTE-1 draft protocol. **No admissible run has been made, and the
protocol is not frozen.**

| Run | Kind | Outcome |
|---|---|---|
| [`20261115-01-diagnostic`](20261115-01-diagnostic/analysis.json) | Declared functional diagnostic with `--continue-unsettled` on a non-exclusive host; does not count | Complete: 147 of 147 trials plus 4 references, no stop. Ids exact in all 151. No ledger or frame defects. All 3 controls passed. 5 of 24 units valid, 19 void |

## 20261115-01-diagnostic

The user authorised it on 2026-11-15 as the single functional diagnostic,
and it was declared before it started:

> VERIFY-REMOTE-1 functional diagnostic 1 (user-authorised 2026-11-15;
> non-exclusive host; timings unqualified, never counted)

**Its timings are not evidence of a speedup.** They are reported below only
as unqualified observations.

**Provenance.**
- Revision `d2625a22` (main, with #860 and #861 merged), protocol
  "VERIFY-REMOTE-1 revision 1 (draft)".
- Binaries: `v3_verify_remote` `e1e4f614…`, `larql-server` `7ee4cb24…`, and
  the frozen relay `v3_link_shaper` `7b1097f8…`. Source, harness and
  workload hashes are in `manifest.json`.
- GPT-OSS 20B (VINDEX3, stored representation), with one CPU expert worker
  behind the synthetic relay.
- Ran 2026-11-15 23:37:38 to 2026-11-16 03:06:09 UTC.
- No quiet was requested. The one-minute load average was 8.7 at the start
  and at the end, and another session's correctness work ran throughout.
- The run was paused once, between units, from 00:43:04 to 02:05:16 UTC
  (`pauses.json`), for a higher-priority session's measurement window. The
  pause used the PAUSE file, not a signal.

### What it shows, functionally

- **Exact greedy ids in 151 of 151 trials.** Each of the 147 trials and the
  4 sequential references produced ids identical to the independent
  sequential greedy reference for its workload. Gate 1 would have stopped
  the run on any mismatch.
- **0 gate-2 defects (ledger and id counts) and 0 gate-3 defects (frames).**
  - Gate 2: in every trial the requested, emitted, prefill-produced and
    timed id counts reconcile with the event ledger.
  - Gate 3: every unit's exchanges tile the record with 24 frames per unit,
    each of the unit's own width and route. Every exchange has a worker
    record.
- **All three corruption controls passed.** Each control changed the middle
  of every proposal on `json_edit` at 0 ms. In every control, acceptance
  fell, every block was rejected at exactly its changed offset, and the
  final ids still matched the reference:

  | Control | Accepted (clean runs) | Rejected at |
  |---|---|---|
  | `verify-2` (trial-144) | 0 (29, 29) | offset 0 |
  | `verify-4` (trial-145) | 29 (44, 44) | offset 1 |
  | `verify-8` (trial-146) | 43 (51, 51) | offset 3 |

- **Ordinary rejection and rollback were exercised.** Clean trials had
  rejected blocks too, and generation continued with exact ids after each.
  - `prose_free`: no proposal was accepted in any clean trial (0 of 2, 6
    and 14 proposed positions at widths 2, 4 and 8).
  - Copy-heavy workloads: every proposed position was accepted.
- **Stop-token handling was not exercised on the real model.** Every trial
  ended at the generation limit (64 ids; 16 for brackets). No stop id
  (200002, 199999, 200012) came up, so stop handling is covered only by the
  unit tests.
- **The schedule ran as recorded.** The 24 units, 3 controls, warm-up and
  references ran in the order the manifest recorded before the first trial.

### Validity: 5 of 24 units valid

Valid units: 4, 15, 18, 21, 23. Every void reason is in
`analysis.json` → `outcomes.void_units`, and each void trial also carries
its own reasons.

| Void reason | Occurrences |
|---|---|
| Brackets differ by more than 1% | 19 units (every void unit) |
| Relay behind schedule | 25 trials, in 9 units |
| Driver suspended 4.2 s during a trial | 1 trial (trial-055, unit 9) |

- **Bracket drift** across the 24 units: median 2.4%, max 74%; 19 over the
  1% gate.
- **Warm-up never settled**: 249.6, 222.6, 228.8 and 220.0 ms per id. An
  admissible run would have stopped at this point; the diagnostic continued
  as declared.
- **Relay gate.** It compares total relay lateness over the whole trial
  with 1% of the timed generation. 24 of its 25 failures were on the
  long-prompt workloads at 5 and 20 ms: `json_edit` 8 and 6,
  `code_docstring` 7 and 3. The other failure was `explain` at 20 ms.
  - Relay statistics are per connection, so lateness during the untimed
    prefill is counted in the numerator.
  - These records cannot separate prefill lateness from generation
    lateness. The protocol revision addresses this; this diagnostic's
    verdicts are not re-derived under any revised rule.

### Frames and worker timing

These are medians over every worker record in all 147 trials (single-row
frames are prefill and sequential steps; multi-row frames are verify
blocks):

| Rows per frame | Frames | Request / response bytes | Worker execute | Per row |
|---|---|---|---|---|
| 1 | 116,712 | 11,576 / 46,136 | 6.3 ms | 6.3 ms |
| 2 | 9,312 | 23,132 / 92,268 | 12.5 ms | 6.3 ms |
| 4 | 6,288 | 46,220 / 184,492 | 24.9 ms | 6.2 ms |
| 8 | 61,080 | 92,396 / 368,940 | 49.4 ms | 6.2 ms |

Frames of 3, 6 and 7 rows (prefill remainders) follow the same pattern.
Bytes and worker execution are both linear in rows: expert compute has no
batching economy on this worker. A rejected proposed row therefore costs as
much worker compute as an accepted one.

### Unqualified observations (not results)

Each figure is a `verify-W` / `sequential` ratio of committed tokens per
second over timed ids (63 of 64), one per repeat, from the units' own
trials. Units marked * are valid; all others are void. The rates were
measured on a non-exclusive host and support no speedup claim.

| Workload | Delay | Sequential tok/s | `verify-2` | `verify-4` | `verify-8` |
|---|---|---|---|---|---|
| `json_edit` | 20 ms | 1.44, 1.43 | 1.50, 1.47 | 1.98, 1.96 | 2.39, 2.38 |
| `json_edit` | 5 ms | 3.03, 2.97 | 1.18, 1.19 | 1.38, 1.37 | 1.43, 1.48 |
| `json_edit` | 0 ms | 4.21, 4.73 | 0.98, 0.99 | 1.09, 1.02 | 1.16, 1.04 |
| `code_docstring` | 20 ms | 1.46, 1.43 | 1.21, 1.35 | 1.52, 1.63 | 1.81, 1.84 |
| `code_docstring` | 5 ms | 3.04, 3.00 | 1.02, 1.15 | 1.10, 1.27 | 1.21, 1.35 |
| `code_docstring` | 0 ms | 7.32, 4.78* | 0.59, 0.99* | 0.98, 1.02* | 1.04, 1.03* |
| `prose_free` | 20 ms | 1.38, 1.46* | 1.05, 0.99* | 1.01, 0.97* | 1.00, 0.94* |
| `prose_free` | 5 ms | 3.02, 3.04 | 1.00, 0.98 | 0.97, 0.95 | 0.90, 0.88 |
| `prose_free` | 0 ms | 4.54, 4.88* | 1.00, 0.97* | 0.93, 0.92* | 0.59, 0.83* |
| `explain` | 20 ms | 1.40, 1.44 | 0.98, 1.01 | 0.88, 1.01 | 0.99, 1.00 |
| `explain` | 5 ms | 3.13, 2.87 | 0.94, 1.03 | 0.95, 1.05 | 1.00, 1.04 |
| `explain` | 0 ms | 4.66*, 4.88* | 1.00*, 1.00* | 1.00*, 1.01* | 0.97*, 1.00* |

The registered reading of the valid cells:
- `explain` 0 ms: "no difference shown" for all three arms.
- Every other cell: "missing", because at least one of its repeats is void.

Observations, none of them registered statements:

- **Copy-heavy workloads.** Every clean proposal was accepted (`json_edit`
  29/29, 44/44, 51/51; `code_docstring` 23/23, 34/34, 40/40), and the ratio
  rose with delay in both repeats.
- **`explain`.** Prompt lookup proposed nothing, so every arm ran as
  sequential.
- **`prose_free` at width 8.** It was below sequential at 0 and 5 ms in
  both repeats; repeat 2 at 0 ms is a valid unit (0.83).
  - A plausible mechanism is the per-row worker cost above: at 0 ms there
    is no round trip to save, and every rejected row is paid for in full.
  - That mechanism is not established. Host interference on this run was
    large: in void unit 11, `code_docstring` `verify-2` read 0.59 with
    every proposal accepted.

### Files

- Trial, reference and warm-up records and the telemetry are gzipped. Every
  file `verification.json` lists matches its SHA-256 after decompression:
  477 files checked when this was packaged.
- `run.log` is the driver's console output, copied in after the run.
