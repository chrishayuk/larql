# VERIFY-REMOTE-1: runs

These are runs of [`verify_remote.py`](../../verify_remote.py) under the
VERIFY-REMOTE-1 draft protocol. **No admissible run has been made, and the
protocol is not frozen.**

| Run | Kind | Outcome |
|---|---|---|
| [`20261116-04-bracket-study`](20261116-04-bracket-study/analysis.json) | Revision 2 bracket study: 16, 32 and 64-id brackets over 8 stand-in units, in a quiet window agreed with both peer sessions; never counts | **No candidate qualifies**: each length has 6 of 8 drifts within 1%, against 7 required. No length is selected and the protocol is not frozen |
| [`20261116-03-mark-check`](20261116-03-mark-check/analysis.json) | Revision 2 relay mark functional check, under the predicted-bytes rule; declared, never timed; non-exclusive host | **Passes**: every mark `ok`; the split falls on the exchange boundary in every arm; ids exact |
| [`20261116-02-mark-check`](20261116-02-mark-check/analysis.json) | The same check under its first rule, which review found too weak; declared, never timed | Passes under that rule; superseded by `-03`, kept as run |
| [`20261116-01-stop-check`](20261116-01-stop-check/analysis.json) | Revision 2 real-model stop check: declared, functional, never timed; non-exclusive host | **Passes**: natural stop (`<\|return\|>` after 11 ids); all three verify arms exact |
| [`20261115-01-diagnostic`](20261115-01-diagnostic/analysis.json) | Declared functional diagnostic with `--continue-unsettled` on a non-exclusive host; does not count | Complete: 147 of 147 trials plus 4 references, no stop. Ids exact in all 147 trials. No ledger or frame defects. All 3 controls passed. 5 of 24 units valid, 19 void |

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

- **Exact greedy ids in all 147 trials.** Each trial's ids are identical
  to its workload's sequential greedy reference: 4 independent
  `sequential` runs through `RoutedBlockSession::step` at 0 ms, made before
  the units. Gate 1 would have stopped the run on any mismatch. The
  references themselves are the comparison standard, so they are not
  counted as checked.
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

  The offset shown is for full-width blocks. A block cut short near the
  64-id limit is rejected at its own middle: for example, trial-146's
  last block is 3 wide and was rejected at offset 1.

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
| Relay behind schedule | 25 trials, in 8 units |
| Driver suspended 4.2 s during a trial | 1 trial (trial-055, unit 9) |

- **The stall** is recorded in `suspensions.json` with `seconds` 4.24, but
  its start and end monotonic times are only 0.26 s apart. The record is
  kept as written; "4.2 s" quotes its `seconds` field.
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

The table gives medians over every worker record in all 147 trials.

- Prefill runs at width 8, so prompt frames are 8 rows wide, with a
  narrower final remainder of 4 rows (`prose_free`) or 7 rows
  (`code_docstring`, `explain`).
- Every single-row frame is a generation step.
- Of the 61,080 eight-row frames, 58,752 are prefill and 2,328 are verify
  blocks.

| Rows per frame | Frames | Request / response bytes | Worker execute | Per row |
|---|---|---|---|---|
| 1 | 116,712 | 11,576 / 46,136 | 6.3 ms | 6.3 ms |
| 2 | 9,312 | 23,132 / 92,268 | 12.5 ms | 6.3 ms |
| 4 | 6,288 | 46,220 / 184,492 | 24.9 ms | 6.2 ms |
| 8 | 61,080 | 92,396 / 368,940 | 49.4 ms | 6.2 ms |

Frames of 3, 6 and 7 rows follow the same pattern. The 3-row and 6-row
frames are verify blocks cut short near the 64-id limit. Of the 1,176
seven-row frames, 1,152 are prefill remainders and 24 are generation.
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

## 20261116-01-stop-check

The real-model stop check that VERIFY-REMOTE-1 revision 2 adds: one
prompt outside the four performance workloads
([`stop_check.json`](../../verify-remote/functional/stop_check.json)), at
0 ms, through the relay with the generation-boundary mark. It is declared,
functional and never timed as a result.

> VERIFY-REMOTE-1 revision 2 real-model stop check 1 (approved
> 2026-11-16; functional, non-exclusive host, never timed)

**Provenance.**
- Revision `99598315` (branch `feat/verify-remote-stop-check`; the stop
  check's files are unchanged in content by the later rebase), clean tree.
- Binaries: `v3_verify_remote` `127eb712…`; `larql-server` `7ee4cb24…`
  (as diagnostic 01); relay `v3_link_shaper` `b4068220…`.
- The relay build **carries the mark** and is not yet the pinned,
  qualified build, as the manifest records. That does not matter for a
  functional check.
- 2026-11-16, about 09:47 to 09:49 UTC. Another session's correctness run
  shared the CPU, by agreement.

**Result: passes.**
- **Natural stop.** The `sequential` reference generated 11 ids, the
  sentence and then `<|return|>` (200002), within the 64-id limit. No
  synthetic stop was used.
- **Every verify arm is exact.** `verify-2`, `verify-4` and `verify-8`
  each reproduced the reference's 11 ids exactly and ended at the stop
  with nothing after it. Gates 2 and 3 are clean, so no exchange came after
  the last unit.
- **Position.** The continuation position is 85 in every arm: the 75-id
  prompt plus 11 ids, minus the stop id, which is never fed back.
- **How the stop arrived.** In every arm it was the target's correction
  after a block that accepted its whole proposal. `verify-8` ran blocks
  7/7, then 1/1, then the stop. The proposer cut each proposal before
  `<|return|>`, as `greedy_speculate` requires.
- **The relay mark answered `ok` in all four trials.** Each had one
  connection, nothing queued, and no request outstanding.
  - The trial waited 1.5 ms for the answer (`verify-8`) after 240 prefill
    exchanges.
  - This is functional evidence that the mark works on the real model. It
    is not the mechanism's qualification check, which runs at 20 ms on
    `json_edit`.

The records are not gzipped (852 KB). Every file `verification.json`
lists matches its SHA-256. `run.log` is the driver's console output,
copied in after the run.

## 20261116-02-mark-check and 20261116-03-mark-check

Revision 2's functional qualification of the relay's generation-boundary
mark on the real model: one trial per arm on `json_edit` at 20 ms,
through the candidate relay `b4068220…`. Declared before each run and
never timed as a result. The host was non-exclusive: another session's
correctness run shared the CPU, by agreement.

**Two runs, and why.** Run `-02` (revision `1bc10e13`, 11:16:47 to
11:23:43 UTC) used the check's first rule.
- That rule was that relay bytes cover the frames in each phase.
- An independent review, made while `-02` was running, showed the rule
  could not catch a boundary one exchange late on `sequential`, or early
  on any arm.
- The rule was replaced before `-02`'s data was examined: the bytes after
  the mark must match the generation exchanges' frames plus per-route HTTP
  overhead, with constants measured in the stop check's records, to within
  5,000 bytes. That tolerance is under half the smallest frame, so a whole
  exchange on either side of the mark always shows.
- Run `-03` (revision `678a5553`, 11:23:55 to 11:31:18 UTC) is the check
  under that rule. `-02` stays as run and is not re-analysed; it passed
  under its own rule.

**Run `-03` passes.**

| Arm | Mark | Generation exchanges | Request residual | Response residual | Relay lateness, generation / whole trial |
|---|---|---|---|---|---|
| `sequential` | `ok 1` | 1,512 | 0 B | +990 B | 0.09% / 0.76% |
| `verify-2` | `ok 1` | 816 | 0 B | +1,511 B | 0.19% / 1.97% |
| `verify-4` | `ok 1` | 456 | 0 B | +2,055 B | 0.30% / 2.36% |
| `verify-8` | `ok 1` | 288 | 0 B | +1,330 B | 0.36% / 2.15% |

- **Every mark was answered `ok`** after 768 prefill exchanges, within
  1.5 to 2.1 ms. Each trial had one relay connection.
- **Ids are exact** against `sequential`.
- **The split falls on the exchange boundary.** The request bytes after
  the mark are exactly the generation exchanges' predicted bytes. The
  response bytes are within about 2 KB, against a smallest response frame
  of 46,136 B.
- **Relay lateness by phase.** In generation it is 0.09% to 0.36% of each
  trial's timed wall time, so every trial is on schedule under revision
  2's gate. The whole-trial figure is revision 1's gate, with prefill
  included: it reads 0.76% to 2.36%, so three of these four trials would
  have failed it. That is diagnostic 01's relay failure, now separated by
  phase. These timings are unqualified (non-exclusive host); they show
  only that the accounting separates the phases.

Every file `verification.json` lists matches its SHA-256 in both runs.
`run.log` in each is the driver's console output. `-03`'s manifest lists
`-02`'s directory as uncommitted at the time it ran.

## 20261116-04-bracket-study

Revision 2's bracket study, run by
[`verify_remote_bracket.py`](../../verify_remote_bracket.py). It chooses
the length of the bracket that checks each unit for drift, under a rule
fixed before the run.

**Rule.** Candidates are 16, 32 and 64 ids. A candidate's drift in a unit
is the `agreement()` measure: the difference between its before and after
brackets' time per position, over their mean. The selected length is the
shortest candidate whose drift is within 1% in at least 7 of 8 units. If
no candidate qualifies, no length is selected and the protocol is not
frozen.

**Provenance.**
- Revision `144d984d` (main, with #870 merged), clean tree, protocol
  "VERIFY-REMOTE-1 revision 2 (draft)", seed 20261117.
- Binaries: `v3_verify_remote` `127eb712…`; `larql-server` `7ee4cb24…`;
  relay `v3_link_shaper` `b4068220…`, the requalified candidate (#869),
  not the certified build.
- Brackets are `sequential` `prose_free` at 0 ms. The filler between the
  before and after brackets is `explain` at 20 ms, in all four arms, 64
  ids each.
- 2026-11-16, 13:41:30 to 14:25:01 UTC. 80 of 80 trials ran, plus 3
  warm-ups and 2 references. No pause, no suspension, no stall.
- **Quiet window.** Both peer sessions agreed in writing before the start
  (`peer-handshake.txt`, also copied into the manifest). No compiler or
  model process other than the study's own ran at the start or the end.
  The one-minute load average was 1.5 at the start and 3.1 at the end.

**Result: no candidate qualifies.**

| Unit | 16 ids | 32 ids | 64 ids | Filler spread |
|---|---|---|---|---|
| 0 | 0.06% | 0.19% | 0.19% | 0.07% |
| 1 | 0.19% | 0.38% | 0.07% | 0.05% |
| 2 | 0.47% | 0.04% | 0.42% | 0.04% |
| 3 | 0.24% | 0.35% | 0.27% | 0.08% |
| 4 | **6.33%** | **3.23%** | **4.46%** | 0.07% |
| 5 | **2.00%** | **3.46%** | **3.09%** | 0.10% |
| 6 | 0.01% | 0.11% | 0.02% | 0.14% |
| 7 | 0.74% | 0.48% | 0.07% | 0.06% |
| **Within 1%** | 6 of 8 | 6 of 8 | 6 of 8 | |
| Median | 0.35% | 0.36% | 0.23% | |

- Every bracket counted: no ledger, frame, mark or relay defect, and no
  unit was marked disturbed.
- **One episode accounts for every failure.** Units 4 and 5 fail at all
  three lengths together. Every bracket that ran from 14:08:34 to
  14:10:11 UTC was slow: 126 to 131 ms per position, against 123.2 to
  124.9 ms everywhere else. That window holds exactly unit 4's after
  brackets and unit 5's before brackets, which ran back to back between
  the two units' fillers.
  - Unit 4 slowed from before to after (e.g. 16 ids: 123.4 to 131.4 ms).
  - Unit 5 recovered from before to after (16 ids: 126.3 to 123.8 ms).
- **The cause is not identified.** The process telemetry in that window
  shows no competing compiler or model process. CleanMyMac used about 40%
  of one core for one sample at 14:09:29. That is not enough, on its own,
  to explain a 5% slowdown.
- **The study's own disturbance check could not see it.** That check
  compares the filler arms, and no filler ran inside the window, so both
  units' filler spreads are small (0.07%, 0.10%).
  `missed_disturbances` is empty for every candidate.
- **Outside that episode, the lengths do not separate.** In the other six
  units every candidate is within 0.75%, and the medians are within 0.14
  percentage points of each other. The study, therefore, gives no
  evidence that a longer bracket is more stable on this host.

Under the rule, no length is selected and the protocol is not frozen.
Whether to repeat the study, and on what terms, is for review.

### Files

- Trial, reference and warm-up records and the telemetry are gzipped.
  Every file `verification.json` lists matches its SHA-256 after
  decompression: 352 files checked when this was packaged.
- `run.log` is the driver's console output, copied in after the run.
- `peer-handshake.txt` is the agreement record the driver required.
