# VERIFY-REMOTE-1: prompt-lookup speculation over remote experts

Status: **DRAFT, revision 2 proposed. Not frozen; no admissible run.**
Revision 1's one authorised functional diagnostic has run
([diagnostic 01](link-results/verify-remote-1/README.md)): ids exact in all
147 trials, no ledger or frame defect, all three controls passed, but only
5 of 24 units valid. Revision 2's three additions (the bracket study, the relay gate by
phase, and a real-model stop check) were **approved for implementation and
limited qualification diagnostics on 2026-11-16, with the conditions
written into each section below**. Freezing and the admissible run each
need separate approval.

| Seam | PR | State |
|---|---|---|
| 1. Proposer and stop handling moved to `larql_inference::speculation` | #855 | merged |
| 2. `RoutedBlockSession::step` | #856 | merged |
| 3. `greedy_speculate` with event ledger | #857 | merged |
| 4. Trial binary (`v3_verify_remote`) | #860 | merged |
| 5. Driver (`verify_remote.py`, `verify_remote_model.py`) | #861 | merged |

## Question

Does the existing prompt-lookup speculative decoding make **greedy
generation** of GPT-OSS 20B faster, in committed new tokens per second,
when the routed experts run on a remote worker, once drafting,
verification, rejection and rollback are all paid for?

BLOCK-TIME-1 and BLOCK-TIME-2 measured verification of *supplied* tokens:
3.37× more evaluated positions per second at 20 ms for width 8, and the
5-to-20 ms slope following `24 / W`. Neither generated anything. Whether
real acceptance turns that into faster generation is the question here.

**Scope of this first rung:** greedy decoding only; the prompt-lookup
proposer only; no model drafter, no sampled-output acceptance rule, no new
hardware topology, no physical network.

## Interface audit

What exists on main (`4a1279a9`), who owns each step, and what is missing.

| Step | Owner today | Path | Remote-ready? |
|---|---|---|---|
| Proposing | `prompt_lookup::propose` (pure: longest trailing n-gram, 2 ≤ n ≤ 16, earliest occurrence, prompt only) | `crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/prompt_lookup.rs` | **No**: a private module of the local Metal command (`mod prompt_lookup;`), unreachable from the remote path |
| Acceptance | `prompt_lookup::accepted_prefix` (pure: leading agreement of the guess with the target's greedy ids) | same file | **No**, for the same reason |
| Speculative loop | `speculative_decode` over `LoweredSession` | `.../lowered/run.rs:336` | **No**: written for the local Metal session; interleaves its own decode-chain look-ahead (`begin_decode`, `quiesce`) and reads device argmax ids |
| Block verification | `RoutedBlockSession::verify(tokens) → logits per position` | `crates/larql-inference/src/vindex3/routed_block.rs:113` | **Yes**: one multi-row frame per contacted worker per routed layer (BLOCK-CHECK, BLOCK-TIME) |
| Accepted-prefix commit and rejected-suffix rollback | `RoutedBlockSession::commit(count)`, through the continuation's own `truncate` | `routed_block.rs:156` | **Yes**: tests show committing any prefix equals decoding only that prefix, and that chained verify plus partial commit leave no trace of rejected positions (`tests/routed_block.rs:199, 268`) |
| Session position | `RoutedBlockSession::position()` (committed only) | `routed_block.rs:103` | Yes |
| Single step (no proposal) | `DecodeSession::step(token)` over its own continuation | used by `v3_block_time`'s `sequential` arm | **No** on a block session: `RoutedBlockSession` owns its continuation and has no single-step method |
| Stop handling | `truncate_at_stop` | `.../lowered/run.rs` | Pure, `pub(super)`; not reachable |
| Remote dispatch accounting | `Timed` transport (route, rows, bytes, round trip per exchange) and `Recorder` (worker timing) | `crates/larql-cli/examples/v3_block_time.rs:61-155` | Yes, but inside one example |

The two sessions' contracts differ:

| | `LoweredSession` (local) | `RoutedBlockSession` (remote) |
|---|---|---|
| verify | returns argmax ids | returns logits; the caller takes the argmax |
| keep a prefix | `truncate(absolute position)` | `commit(count)`, relative, required before the next verify |
| single step | `step(id)` | none |

### Missing seams, and what this programme would add

1. **Move the proposer, not copy it.** `propose`, `accepted_prefix` and
   `truncate_at_stop` are pure. Move them to a shared module in
   `larql-inference` (proposed: `larql_inference::speculation`) and have
   the local Metal command import them from there, unchanged. The local
   path's behaviour is pinned by its existing tests.
2. **A single step on the block session.** Without one, every position
   with no proposal would run as a width-1 block over the multi-row route.
   BLOCK-TIME-1 measured that as about 7% slower per position than
   `sequential` (coordinator 44.8 against 35.3 ms per position at 0 ms). It
   would penalise the speculative arms for reasons unrelated to
   speculation. Proposed: `RoutedBlockSession::step(token) → logits`, which
   runs `DecodeSession::step` over the session's own continuation (the
   single-row route) and commits that one position. It needs tests that its
   logits and continuation equal the sequential session's, and that it
   refuses while a block is pending.
3. **A remote speculative loop** (built in #857). This is a second loop, and the
   justification is specific: the local loop's structure is its Metal
   decode-chain pipelining (a committed look-ahead step collected by
   `quiesce`), which has no remote counterpart. The remote session reports
   logits, not device ids, and commits relatively. The **policy** is not
   duplicated: it calls the moved `propose`, `accepted_prefix` and
   `truncate_at_stop`, and keeps the same accounting (proposed, accepted,
   verify blocks, single steps, rewinds). Proposed home:
   `larql_inference::vindex3::speculate::greedy_speculate(target, prompt,
   first, n, width, stop, proposer)`, generic over a `BlockTarget` trait
   (`verify`, `commit`, `step`, `position`), which `RoutedBlockSession`
   implements. The greedy choice is shared too (`speculation::greedy_id`:
   first maximum, so ties go to the lowest id; empty or non-finite logits
   refuse). Two properties came out of building it:
   - **The proposer is asked for at most `min(W − 1, still wanted − 1)`
     tokens.** A block yields its accepted proposals plus the target's own
     correction, so a longer proposal would verify a position the output
     cannot use. With one id left, nothing is proposed and that id is a
     single step.
   - **The shared policy owns no K/V.** The loop tells the target what to
     keep; only the session holds continuation state.
4. **A trial binary** (`examples/v3_verify_remote.rs`), sharing `Timed`
   and `Recorder` with `v3_block_time` through one examples support module
   rather than a copy. It records per-exchange routes and sizes, worker
   timing, every block's proposal, acceptance and rejection offset, and the
   generated ids.
5. **A driver** reusing BLOCK-TIME-2's runner, telemetry, stall watch,
   `PAUSE`/`PAUSED` hold, provenance, admission checks and quiet-machine
   handshake.

No second implementation of proposing, acceptance or rollback is
introduced. `RoutedBlockSession` remains the only owner of
verified-against-committed state on the remote path.

## Proposed protocol

### Fixed

- GPT-OSS 20B (`gpt-oss-20b.vindex3`), stored weights (production
  arithmetic); one CPU expert worker owning every routed layer; LINK-1's
  thread settings.
- The frozen relay `7b1097f8…` at unlimited rate, at **0, 5 and 20 ms**.
- Greedy decoding; ties broken to the lowest id, identically in every arm.

### Arms

| Arm | What it does |
|---|---|
| `sequential` | Greedy generation by single steps: the exact reference and the baseline. A loop of its own, independent of `greedy_speculate`, stepping `RoutedBlockSession::step` (which is `DecodeSession::step` over the session's continuation, the single-row route), so every arm prefills the same way |
| `verify-2`, `verify-4`, `verify-8` | `greedy_speculate` with blocks of at most 2, 4 or 8 positions (the fed token plus up to 1, 3 or 7 proposed) |

### Workloads

**Recovered from #729's archive** and committed in
`verify-remote/workloads/` as the exact token ids #729 ran (GPT-OSS chat
template), with the archive file's hash. The text was not archived; it is
decoded for reading at freeze, and the ids are authoritative.

| Workload | Kind | Prompt tokens |
|---|---|---|
| `json_edit` | copy-heavy | 256 |
| `code_docstring` | copy-heavy | 215 |
| `prose_free` | low-copy | 84 |
| `explain` | low-copy | 87 |

All four come from #729's set, chosen there before this programme; none
was picked from remote results. #729's local results (json_edit 1.33×,
code_docstring 1.25×, prose about 1.0×, on a different representation and
backend) are a historical reference only, not a prediction. The selection
and the ids do not change after a run.

**Copy-heavy and low-copy results are reported separately and never
aggregated into one figure.**

### Measure

`committed new tokens per second = timed ids ÷ wall time of generation`

The first generated id is prefill's, produced untimed; the trial records
the ids produced inside the clock as `timed_ids` (one fewer: at most 63 of
the 64 requested). Every trial records the requested, emitted,
prefill-produced and timed counts separately, and whether generation ended
at a stop id or at the limit; an early stop counts only the ids actually
emitted inside the clock. Gate 2 requires the four counts, the ending and
the ids to agree. The report gives the full generation length beside every
rate. Every arm
produces the same ids, so the ratio between arms is the inverse ratio of
their wall times. The trial's own bookkeeping (two counter reads and a
clock read per unit of work) is inside the clock; it is a few hundred
nanoseconds per unit, so it weighs slightly more on `sequential`, which
runs the most units.

This is timed from the first generated position to the last. It includes
proposing, every verify block, every single step, every commit or rollback,
argmax, and coordinator work. **Prompt prefill is excluded, identically in
every arm.** It is done untimed, by the same method in every arm (proposed:
block verification at width 8 and commit, which BLOCK-CHECK showed is exact).

Per trial, recorded:
- generated ids, and blocks with their proposal length, accepted count and
  rejection offset;
- single steps;
- exchanges by route with rows and bytes;
- worker execution per exchange;
- the decomposition used in BLOCK-TIME.

### Correctness gates

1. **Exact greedy identity.** Every trial's generated ids, at every arm
   and delay (brackets and controls included, as far as they run), equal
   the workload's `sequential` reference taken at 0 ms before the units,
   through any stop id. Any mismatch stops the run. A mismatch is a
   correctness finding, reported with the first differing position; the
   mismatching trial's raw record is kept beside the stop message. A record
   that is not the trial the schedule asked for (arm, width, corruption,
   length, stop ids, workload hash, or a block wider than `W`) also stops
   the run, as a harness fault.
2. **Acceptance accounting by event ledger.** Every generated id is
   recorded once, with its source and whether it was emitted (an id after a
   stop is not). The sources are: the prefill prediction, an accepted
   proposal, a target correction (the target's id after a block's accepted
   prefix), or a single step. The emitted events must be exactly the
   generated ids, in order. The counts by source must match the blocks and
   steps run: one prefill, accepted proposals equal to the blocks' accepted
   counts, one correction per block, one per single step.
   `SpeculativeRun::reconcile` checks this; a trial that fails it is void
   by name. Proposed, accepted, rejected and committed positions are
   reported separately; verified positions are never counted as generated
   tokens.
3. **Remote-execution integrity.**
   - every verify block of `k` positions sends exactly one multi-row frame
     of `k` rows per routed layer (24), and no single-row frame;
   - every single step sends exactly 24 single-row frames;
   - no other exchange occurs: the units' exchange ranges tile the whole
     record with no gap, and no routed layer is contacted twice in a unit;
   - one worker: the trial binary refuses more than one `--shards` url.

   A trial that fails this is void by name. It shows a silent local or
   single-row fallback.
4. **Rollback.** This is covered by tests before any run (see the testing
   plan), and by gate 1 during it: a rollback that left rejected state
   would change later ids.

The relay-schedule gate (LINK-1's), BLOCK-TIME-2's stall watch, and the
quiet-machine and handshake rules apply unchanged.

### Schedule

- Warm-up: BLOCK-TIME's rule on the bracket trial (until two agree within
  1%, at most four).
- **References.** After warm-up, one `sequential` trial per workload at
  0 ms. Every later trial of that workload, at any delay and arm, must
  reproduce its ids (gate 1). References are not timed results.
- **The bracket**: `sequential`, 0 ms, `prose_free` (the shortest prompt),
  16 ids. Its time per timed id is the bracket measure. **Provisional:**
  diagnostic 01 put 19 of 24 units over the 1% gate (median 2.4%, max
  74%) on a loaded host. Revision 2 chooses the length by a bracket-only
  diagnostic (see "Revision 2: bracket length"). The gate is not loosened.
- **Units.** One unit is one workload at one delay: the four arms in a
  seeded order, opened and closed by a short `sequential` 0 ms bracket.
  That is 4 workloads × 3 delays = **12 units per repeat, two repeats**.
  Within each repeat every workload × delay cell appears exactly once, in
  an order shuffled from seed 20261116, which also shuffles the arms
  within each unit.
- **Generation length: 64 new ids** (decided at review), or fewer if a stop
  id comes first. **Stop ids**: the container's `generation_config.json`
  end-of-sequence ids, 200002 `<|return|>`, 199999 `<|endoftext|>` and
  200012 `<|call|>`. The stop id is kept, and nothing after it is emitted.
  A longer-generation experiment (for example #729's 256) is a separate
  future decision. Low-copy prompts may produce few or no verify blocks in
  64 ids. Acceptance counts are reported as they are, and an absence of
  proposals is never read as a limit of the remote architecture.
- **Validity**: a unit is void if its brackets differ by more than 1%, or
  if any of its trials fails gate 2, gate 3, the relay gate or the stall
  rule. **Relay gate** (revision 1): the relay's total lateness over a
  trial, prefill included, is at most 1% of that trial's timed generation.
  Revision 2 proposes counting generation only (see "Revision 2: relay
  gate by phase").
- **Pausing** (`PAUSE`/`PAUSED`, BLOCK-TIME-2's) happens only before a
  unit's opening bracket, before warm-up, and before each diagnostic
  control.
  Void units are kept and never rerun. Only a gate 1 failure stops the run.
- **No within-unit cost-stability gate** (decided at review). BLOCK-TIME-2's
  rule 4 compared the same operation at two delays. These arms do
  different work by design, with different acceptance lengths, block
  shapes and expert sharing, so no invariance is established. Worker
  execution per verified position, block widths and acceptance lengths are
  recorded **as diagnostics**. A gate on any of them needs a demonstrated
  invariance first.

### Statements, fixed in advance

For each workload, delay and width, with both repeats' units valid:

- **"Faster than sequential"**: committed new tokens per second higher than
  `sequential`'s by more than 1% in both repeats. **"Slower"**: lower by
  more than 1% in both. Otherwise **no difference shown**.
- The ratio is reported per repeat, along with proposed, accepted,
  rejected and committed-in-blocks counts, blocks and single steps,
  verified positions, the mean accepted length per block, the share of
  committed ids that came from blocks, and the rejection offsets.
- If the prefill id is itself a stop id, no id is timed, and the cell
  reads **"no timed generation"**.
- A width that proposes nothing on a workload is reported as such, with
  its tokens per second. It is not dropped.
- A missing cell (no valid unit in a repeat) is reported as missing.
- **Descriptive only:** the speed the BLOCK-TIME exchange model predicts
  from the observed acceptance, against the measured speed. Revision 1
  implements its delay term only: the arm saves one configured round trip
  per exchange it does not make, so the predicted ratio is
  `sequential wall ÷ (sequential wall − saved exchanges × delay)`. The
  ratio of exchange counts is reported too.
- **Never claimed:** sampled-output exactness, model-drafter behaviour,
  physical-network performance, or an aggregate across copy-heavy and
  low-copy workloads.

### Relay scope

The block-frame qualification covers frames of widths 1, 2, 4 and 8.
Speculative blocks also run at widths 3, 5, 6 and 7, whose frame sizes fall
between qualified sizes. Those geometries are **not qualified**, and the
qualification says nothing about their individual accuracy. Every trial
records each exchange's width. The report gives the share of exchanges and
of bytes at each width, and the share at unqualified widths, per workload,
delay and arm. If unqualified widths carry much of the traffic where an
advantage is narrow, that limits the claim, and the report says so. For
scale, the largest qualified error was −96 µs per exchange (width 8,
5 ms).

### Budget (estimate, to be confirmed by the diagnostic)

Prompts are 84 to 256 tokens, prefilled untimed at width 8. From BLOCK-TIME
timings that is about 11 to 47 s per trial at 20 ms, and less at 0 and
5 ms. Generation is 64 new ids: at 20 ms `sequential` takes about 40 s;
speculative arms take less in proportion to acceptance. Four workloads,
three delays, four arms and two repeats make 96 trials, plus brackets:
roughly **2 to 2.5 hours**, holdable between units.

## Revision 2: bracket length (approved 2026-11-16 for implementation)

**Why.** In diagnostic 01 every void unit had bracket drift over 1%. That
run was on a non-exclusive host (one-minute load about 8.7, with another
session's work running), so it cannot separate host load from a 16-id
bracket being too short to be stable. The 1% gate stays. The length is
chosen by a separate diagnostic whose selection rule is fixed here, before
it runs.

**The bracket diagnostic** (declared, never counted, no throughput
measured or reported):
- **Host.** It needs a quiet window agreed with the WSP-002 session, with
  the handshake recorded. Without a quiet host it does not run: on a busy
  host it could only repeat diagnostic 01's finding.
- **Candidates.** `sequential`, 0 ms, `prose_free`, at **16, 32 and 64
  generated ids** (15, 31 and 63 timed).
- **Units.** **8 stand-in units.** Each opens with all three candidate
  brackets in an order shuffled per unit from seed 20261117, and closes
  with them in the same order, so every candidate's pair is the same
  distance apart.
- **Filler.** Between a unit's brackets runs a filler of the length and
  load of a real unit: `explain` at 20 ms, all four arms in a fixed order.
  Prompt lookup proposes nothing on `explain`, so this is sequential work
  under every arm, and the 20 ms delay gives the longest real unit
  duration. Filler timings are recorded but not analysed or reported as
  rates.
- **Warm-up and gates.** Warm-up as now. Gate 1 applies to every trial.
  Pausing is allowed only between stand-in units.
- **Measure.** Per candidate, the 8 drifts (`|after − before| / before` of
  time per timed id), with their median and maximum.
- **Budget.** About 1 to 1.5 hours.

**Selection rule, fixed in advance:**
- A length **qualifies** if at least 7 of its 8 drifts are at most 1%.
- The bracket is the **shortest qualifying length**. If 16 qualifies, the
  bracket stays 16 ids.
- If **no length qualifies**, none is selected. The protocol is not
  frozen, and the decision returns to review with all three candidates
  reported. The gate is never loosened to make a candidate pass.
- Every candidate is reported, including those that fail.

**Does the bracket see disturbances?** This is the approval's caveat: a
bracket must detect variation, not merely agree with itself. Each filler
runs the same sequential work four times, so the spread of its four times
per timed id measures disturbance inside the unit. Per candidate, the
report counts **missed disturbances**: units whose filler spread exceeds
1% while that candidate's brackets agree within 1%. The count is reported
beside the selection, not added to the rule. A selected length with
missed disturbances goes back to review before any freeze.

**What it does not do.** It does not run the experiment's cells, measure
any arm's speed, or change the 1% gate, the workloads, the widths or the
64-id generation length.

**Implementation.** A `--bracket-study` mode in `verify_remote.py`
reusing the existing trial binary, with its own manifest kind and tests in
the bench suite: the schedule shape and seed, the drift computation, and
the selection rule at, above and below 7 of 8.

## Revision 2: relay gate by phase (approved 2026-11-16 for implementation)

**Why.** The relay records lateness per connection, totalled when the
connection closes. Prefill and generation share one connection, so
revision 1's gate puts untimed prefill lateness over a denominator of
timed generation. In diagnostic 01, 24 of the 25 relay failures were on
the two long-prompt workloads at 5 and 20 ms. The records cannot say how
much of that lateness fell in generation.

**Rule.** Only the relay's lateness during **timed generation**, both
directions summed, is compared with 1% of that trial's timed generation
wall time. The threshold is unchanged, and every trial of the protocol
uses the same rule. Revision 1's whole-trial figure is still reported, as
a diagnostic.

**Mechanism: a generation-boundary mark in the relay.** One warmed
connection and one network path carry prefill and generation, as now.

- **The control listener.** The relay always opens one, on loopback, and
  prints its address after the existing ready line. So the binary that is
  qualified is exactly the binary that is measured, and no mode switch
  exists to differ between them. The listener does nothing until it is
  asked.
  - Harnesses that never mark see no change: their records are
    byte-identical, and mark records go to a separate file.
  - The per-chunk shaping code is untouched. Each connection loop adds one
    atomic load per iteration and one atomic increment per chunk read.
- **The mark.** A control connection that sends `mark` starts one new
  epoch, the logical boundary, for every connection then open. Each
  connection's loop, at its next iteration, snapshots both directions'
  tallies and checks whether it is quiescent. The relay answers `ok`
  only if:
  - every connection open at the mark wrote its snapshot within the
    timeout;
  - in every direction, nothing is queued on the link and nothing is
    unread in the socket (checked with a non-consuming peek);
  - the last chunk each connection delivered was a response, so no request
    is still waiting for its answer;
  - no connection read a chunk between its snapshot and the answer, so no
    generation traffic crossed the boundary;
  - at least one connection was open.

  Otherwise it answers `refused`, with the reasons. Either way, it writes
  every snapshot and a summary line to the mark file before answering.
- **Quiescence from the coordinator's side.** The relay's checks cannot
  see a response still being computed and partly sent. The trial covers
  that case:
  - it marks only after prefill has committed, with every exchange it
    started completed (the count is recorded);
  - its transport is synchronous;
  - it sends no generation exchange and does not start its clock until
    the answer arrives.

  The two sides together establish the boundary. An empty queue alone
  would not.
- **Generation lateness** is the closing tally minus the snapshot, per
  direction.
- **Void by name** if:
  - the mark is missing, duplicated, refused or unanswered;
  - the trial's recorded answer disagrees with the relay's summary;
  - a closing tally is smaller than its snapshot;
  - the set of snapshots does not match the summary's connections.

**Cost.** This changes the frozen relay binary (`7b1097f8…`). Before any
admissible run, the new binary must pass, on a quiet host and under the
existing frozen criteria with no bypass or relaxed threshold:
- **two LINK-2 revision 3 qualifying runs;**
- **the block-frame requalification.**

This is mandatory even though the shaping code is unchanged.

**Considered and not chosen.**
- **A fresh connection for generation.** Its first exchanges would run
  cold, inside the clock.
- **Routing prefill directly to the worker.** Prefill and generation would
  then run on different paths.

**Tests, before any model run:**
- **Relay (Rust, against the relay's own code over loopback, with a fake
  upstream).**
  - With no mark, the closing records are unchanged and no mark file is
    written.
  - Chunks and lateness partition exactly at the mark: closing minus
    snapshot equals post-mark traffic.
  - **Negative controls**, each of which must be refused:
    - a mark while a request awaits its response;
    - a mark while a delayed chunk is queued on the link;
    - a mark with no open connection.
  - A second mark is recorded as a second epoch, so it is detectable.
  - A connection that never answers its snapshot is refused by timeout.
- **Trial binary (Rust).** The client records `ok`, `refused`, no answer
  (timeout) and no listener exactly as received.
- **Gate (Python, bench suite).**
  - Arbitrarily large prefill lateness with generation lateness under 1%
    passes.
  - Generation lateness just over 1% fails, and exactly 1% passes.
  - Each of these voids the trial by name: a missing, duplicated, refused,
    delayed (unanswered) or inconsistent mark, and a non-quiescent mark.

**Qualification of the mechanism, before freeze.**
- The relay requalification above.
- A short declared functional check on the real model:
  - one trial per arm on `json_edit` at 20 ms;
  - every mark answered `ok`;
  - snapshot plus generation lateness equals the closing total;
  - ids exact.

## Revision 2: real-model stop check (approved 2026-11-16 for implementation)

**Why.** Diagnostic 01 never reached a stop id: all 151 runs ended at the
64-id limit. Stop handling is covered by unit tests, but not on the real
model through the remote session.

**The check** (declared, functional, never timed as a result; outside the
four performance workloads, which are unchanged):
- **One prompt**, chosen so the real model's greedy continuation reaches
  one of the configured stop ids within 64 ids. It is chosen by a
  sequential reference run, before the check, and recorded with its ids.
- **Natural or synthetic.**
  - Preferred: a stop the model produces naturally. A chat-formatted
    prompt that opens the final channel and asks for a one-word answer
    should end in `<|return|>` within a few ids.
  - If no prompt does this deterministically, the check uses a
    **synthetic stop**: the stop set is the id the model was observed to
    predict at a chosen position. It is labelled synthetic in every
    record and report, and never presented as evidence of natural
    end-of-sequence behaviour.
- **Arms.** `sequential` and `verify-2`, `verify-4`, `verify-8`, at 0 ms.
  The prompt includes earlier text that makes prompt lookup propose the
  stop id, so the stop can arrive inside an accepted block as well as by
  correction. The record shows which happened.
- **Requirements, per arm:**
  - ids identical to `sequential`'s, ending at the stop id;
  - nothing emitted after it, with `ended_by` = stop id;
  - the event ledger reconciles, with the stop id's event emitted and any
    later accepted proposal not emitted;
  - the committed continuation position equals the prompt length plus the
    committed ids, as `sequential`'s;
  - no exchange after the unit that produced the stop: gate 3's tiling
    ends at that unit.

## Event-ledger invariants## Event-ledger invariants

What `SpeculativeRun::reconcile` (#857) checks on every trial. A trial that
fails it is void by name.

1. **One source per id.** Every produced id has exactly one event, whose
   source is one of: prefill prediction, accepted proposal, target
   correction, single step.
2. **Emitted events are the output.** The emitted events, in order, are
   exactly the generated ids: none missing, none duplicated, none
   reordered.
3. **Unemitted ids only at the end.** No emitted event follows an
   unemitted one. (That unemitted ids occur only after a stop id is a
   property of the loop, tested in #857; `reconcile` does not check it
   separately.)
4. **The prefill prediction comes first, once.**
5. **Counts match what ran.** Accepted-proposal events equal the sum of
   the blocks' accepted counts. Target-correction events equal the number
   of blocks. Single-step events equal the number of single steps.

Two further properties hold by construction and are tested in #857, not
re-checked per trial:
- no proposal is longer than `min(W − 1, still wanted − 1)`;
- a block's rejection offset is the first position at which its proposal
  and the target's greedy ids differ.

## Plan for seams 4 and 5 (after #855 to #857 merge)

**Seam 4, the trial binary** (`crates/larql-cli/examples/v3_verify_remote.rs`):

- **Reused, not copied:** the `Timed` transport (route, rows, bytes and
  round trip per exchange) and the `Recorder` (worker timing) now in
  `v3_block_time.rs`. They move to one shared examples support module, in
  the same pattern as `crates/larql-server/examples/support/`, and
  `v3_block_time` imports them unchanged.
- **Arms:** `sequential` steps the `RoutedBlockSession` with `greedy_id`,
  in a loop independent of `greedy_speculate`. The `verify-W` arms call
  `greedy_speculate` with `propose`. Every arm calls `prefill` on the same
  prompt by blocks of 8, untimed.
- **Records:** the generated ids; for every block, its width, proposal,
  accepted count and rejection offset; single steps; the event ledger;
  every exchange (route, rows, bytes, round trip) tagged with the block or
  step it belongs to; worker timing; the generation wall time.
- **Judges nothing.** It writes what happened; the gates are the driver's.

**Seam 5, the driver** (`verify_remote.py` and `verify_remote_model.py`):

- **From BLOCK-TIME-2, reused as is:** relays, the worker, provenance,
  admission (qualification receipts, handshake, quiet machine at start and
  end), telemetry, the stall watch, the `PAUSE`/`PAUSED` hold between
  units, and failure capture with every record written.
- **New, in the model and pinned by tests:** the unit schedule (seeded,
  balanced across workloads and delays); gates 1 to 3; unit validity; the
  faster/slower/no-difference statements; the per-width traffic shares.
- **Not reused:** BLOCK-TIME-2's within-pair cost rule and its slope
  analysis.

## Failure classification

| What happens | Classification | Effect |
|---|---|---|
| Any arm's ids differ from `sequential`'s (gate 1) | **Correctness failure** | The run stops; reported with the first differing position |
| The ledger does not reconcile (gate 2) | Void trial, by name | Its unit is void |
| A frame-accounting mismatch: unexpected route, rows or worker (gate 3) | Void trial, by name | Its unit is void |
| Relay behind schedule; a driver stall; brackets differing by more than 1% | Void unit, by name | Kept and reported; never rerun |
| A crash, timeout, dead worker or interrupt | **Stop** | Every record written; nothing analysed |
| Warm-up never settles | **Stop** (admissible); continue only if declared (diagnostic) | As BLOCK-TIME |
| A width proposes nothing on a workload | Not a failure | Reported with its tokens per second and zero acceptance |

## Blockers before any model run

1. Seams 1 to 5 built, with tests, reviewed and merged through the normal
   ruleset (1 to 3 merged as #855, #856, #857; 4 and 5 are #860 and #861).
2. ~~Workload prompts recovered and decoded~~ (done: each workload file
   holds the decoded text and the tokenizer's hash. Re-encoding the text
   reproduces the ids exactly, and the ids remain authoritative.)
3. ~~The within-unit cost-stability rule~~ (decided: none; diagnostics
   only). ~~Generation length~~ (decided: 64).
4. ~~One declared functional diagnostic~~ (done:
   [diagnostic 01](link-results/verify-remote-1/README.md), gates 1 to 3
   working end to end).
4a. Revision 2's bracket diagnostic, with a length selected by its rule.
4b. Revision 2's relay mark built and tested, the relay requalified on its
   new binary (two LINK-2 revision 3 runs, block frames), and the
   mechanism's functional check passed.
4c. Revision 2's real-model stop check passed.
4d. Bracket study, relay qualification and stop-check evidence returned for
   review; then a final protocol review and **separate approval** to
   freeze and to run.
5. Independent review of this protocol and the implementation, then a
   freeze with hashes in `frozen-protocols.json`.
6. A quiet window agreed with the WSP-002 session, which has priority.

## Testing plan

**Rust, `larql-inference`, against the existing routed-block fixture (fake
workers, no model):**
- `greedy_speculate` generates exactly the ids of sequential greedy
  decoding for widths 2, 4 and 8, on prompts built so proposals are
  always, sometimes and never accepted;
- **corrupted proposals** (negative control): a proposer wrapper that
  changes one token of every proposal, at the first, a middle and the last
  proposed position. The verifier rejects at exactly that offset, the
  committed ids still equal the sequential reference, and acceptance falls
  as expected;
- **rejection at every offset** 0 to `W - 1`, after which the next block's
  logits equal a fresh sequential session that consumed only the committed
  prefix (KV state carries nothing rejected);
- a stop id inside an accepted block ends generation there, and ids after
  it are not emitted;
- the generation limit and the session's position limit truncate a
  proposal and never overrun;
- `RoutedBlockSession::step`: logits and continuation equal the sequential
  session's; it refuses while a block is pending; a failed step
  invalidates the session;
- **frame accounting** through the fixture's transport: each block of `k`
  positions produces one `k`-row frame per routed layer, each single step
  one single-row frame per layer, and nothing else.

**The local path, unchanged:** the existing `prompt_lookup` and lowered
speculation tests keep passing after the move. CI on the Metal path runs
them.

**Python, bench folder (CI's protocol workflow):** schedule shape, seed and
balance pinned; unit validity; reconciliation of the acceptance accounting;
the "faster/slower/no difference" rule at, below and above 1%; missing and
zero-proposal reporting; and an end-to-end `execute()` with fake trials,
as for BLOCK-TIME-2.

**Before the admissible run:** the functional diagnostic, including one
corrupted-proposer control trial per width on one copy-heavy workload at
0 ms (`json_edit`, corruption at the middle of each proposal). Its ids must
equal the reference, its acceptance must fall below every clean trial of
the same width, and every changed block must reject at or before its
change. The proposal is cut before any stop id first, as the loop cuts it,
and the replacement is never a stop id, so the change always lands on a
verified token; the trial records the changed offset per block. The
control's timing is not a result.
