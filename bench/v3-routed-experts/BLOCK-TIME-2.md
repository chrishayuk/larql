# BLOCK-TIME-2: the RTT differential slope, in bracketed pairs

Status: **approved and frozen on 2026-11-15, revision 1.** A change to this
document, `block_time_2_model.py` or `block_time_2.py` is a new revision,
not an edit. No admissible run has been made. One declared functional
diagnostic was run before freezing; it exercised the driver and does not
count, and its timings are not evidence.

## Why this exists

[BLOCK-TIME-1](BLOCK-TIME-1.md)'s one admissible run
([`20261115-01-admissible`](link-results/block-time/README.md)) measured a
valid 20 ms comparison in both repeats, but **every 5 ms trial was void by
the bracket gate**, so neither registered slope could be computed. The
[bracket investigation](BLOCK-TIME-1-bracket-investigation.md) found that
both void groups coincide with host-wide slowdowns that hit the worker and
the coordinator together, and that BLOCK-TIME-1's brackets enclosed five to
ten trials each, so a single disturbance voided a whole delay across all
arms.

BLOCK-TIME-1's result stands as recorded. BLOCK-TIME-2 does not revisit it,
does not pool with it, and does not set out to reproduce its speed result.

## Question

For exact remote block execution on GPT-OSS 20B with one CPU expert worker,
does per-position wall time grow with the configured round trip at the rate
the exchange model predicts, `24 / W` round trips per evaluated position
(24 for `sequential` and `block-1`, then 12, 6 and 3), **measured between
5 and 20 ms**?

## Fixed from BLOCK-TIME-1, unchanged

- Model `gpt-oss-20b.vindex3`, stored weights; one CPU expert worker owning
  every routed layer; LINK-1's thread settings.
- The supplied 216-token stream (`link-results/control/ids.json`); 56
  measured positions from position 160.
- Arms `sequential`, `block-1`, `block-2`, `block-4`, `block-8`.
- The frozen relay `7b1097f8…` at unlimited rate, with the two passing
  block-frame revision 2 qualifications
  ([`20261115-01`](link-results/link-2-block-frames/20261115-01-qualifying-1/calibration.json),
  [`20261115-02`](link-results/link-2-block-frames/20261115-02-qualifying-2/calibration.json)),
  which cover 5 and 20 ms at these frame sizes.
- Warm-up: `sequential` 0 ms trials until two consecutive agree within 1%,
  at most four; otherwise the run stops.
- Per-trial gates: **parity** (every position's logits hash equals the
  reference; any mismatch stops the run), **frame counts**, and **relay on
  schedule** (LINK-1's gate), exactly as BLOCK-TIME-1 defines them.
- The **1% bracket gate** and the **10% slope band**.
- Every trial is profiled (worker timing requested); BLOCK-TIME-1 measured
  the cost of asking at −4.0% to +0.4%.

## What changes

### The unit is a bracketed pair

A **pair** is four trials, always in this shape:

```
sequential 0 ms bracket → arm at one delay → same arm at the other delay → sequential 0 ms bracket
```

Each pair has its own two brackets; brackets are never shared between
pairs.

### Schedule

- **Four rounds of five pairs, one pair of each arm per round**: twenty
  pairs, eighty trials, after warm-up. No zero-delay arm trials, no
  unprofiled trials, no no-relay controls. Rounds spread a slow drift of the
  host over every arm alike; a global shuffle could, and in an earlier draft
  did, put all four `sequential` pairs in the second half.
- **Arms are shuffled within each round** by one generator seeded
  `20261115`, one shuffle per round. The schedule is computed by
  `block_time_2_model.schedule()` and recorded in the manifest before the
  first trial. With this seed the rounds are:

  | Round | Delay order | Arms, in run order |
  |---|---|---|
  | 1 | 5→20 | `block-8`, `block-4`, `block-1`, `sequential`, `block-2` |
  | 2 | 20→5 | `sequential`, `block-4`, `block-8`, `block-1`, `block-2` |
  | 3 | 20→5 | `block-8`, `block-2`, `block-4`, `sequential`, `block-1` |
  | 4 | 5→20 | `block-4`, `block-2`, `sequential`, `block-1`, `block-8` |

- **Delay order is set by the round, A-B-B-A**: 5→20, 20→5, 20→5, 5→20.
  Every arm has two pairs in each order, and a linear drift across the run
  weighs on the two orders alike.
- Changing the seed, the round composition or the delay pattern is a new
  revision; the contract tests pin all three.
- **Nothing is ever added to the schedule.** A void pair is not rerun or
  replaced, and nothing is run because something else failed.

### Validity

A **pair is valid** only if all hold:

1. Its two brackets agree within 1% (BLOCK-TIME-1's bracket drift).
2. None of its four trials has a frame-count defect.
3. The relay was on schedule in all four.
4. **Within-pair execution-cost stability.** The arm's two trials, at 5 and
   20 ms, agree on **coordinator time per evaluated position** and on
   **worker execution time per evaluated position**, each within **3%**:

   `D = |x(20 ms) − x(5 ms)| / min(x(20 ms), x(5 ms))` ≤ 0.03

   for `x` each of the two. A missing, zero, negative or non-finite value
   fails. Both costs are expected not to depend on the configured round
   trip for the same arm, so a larger difference is a change in execution
   cost inside the pair, which the brackets either side cannot see.
5. **No stall of the driver inside any of its four trials** (see Pausing).

A parity mismatch in any trial stops the run, as in BLOCK-TIME-1. **So does
any other failure**: a trial process that crashes or times out, a worker
that dies, a relay that carried no traffic, an interrupt. A stopped run is
not analysed: no slope or per-arm statement is computed from it, though the
trials it completed are judged into pairs and every record is written
(trials, pairs, stalls, telemetry, the failure itself) as for a complete
run. A void pair is kept, with its measurements and named reasons, and is
never fitted or pooled; every trial of it carries the pair's reasons.

Each pair's brackets are its own, but the relay-lateness gate scales by a
floor taken from the most recent bracket, so a pair's opening bracket is
judged against the previous pair's closing one. This links neighbouring
pairs only through that threshold.

The 3% is an engineering tolerance, not a measured bound on natural
variation. BLOCK-TIME-1's undisturbed trials put the same arm's coordinator
time at 5 and 20 ms within 2 to 3% of each other, so **3% is tight and may
void pairs that were not disturbed. That is accepted: a missing slope is
preferable to a misleading one.** If a run yields too few valid pairs, the
slopes are reported missing by the rule above; the tolerance is not relaxed.
It was first proposed at 5% and tightened to 3% before freezing, because at
5% the two checked costs alone could move `block-8`'s slope by more than the
10% band (below). It was added after the functional diagnostic found a pair corrupted by a 5-second stop
inside its 20 ms trial (coordinator 34.4 against 123.7 ms per position,
slope 8.98 against 3), whose brackets agreed within 0.7%.

**What rule 4 conditions on.** The configured delay per position is fixed
by the exchange count, which rule 2 fixes. So a pair's slope can leave
`24 / W` only through changes between its two delays in four things:
coordinator time, worker execution, the rest of the worker's handler time
(queueing, decoding, encoding), and the transport remainder. Rule 4 voids
pairs on the first two only; the other two are not compared within a pair. **A systematic deviation from the model in
either of them therefore shows as missing slopes, never as "does not
follow".** To keep it visible, the analysis also reports a **secondary,
ungated slope** per arm: the mean over pairs that pass every rule except
rule 4 (with the same three-pair minimum). It is not the registered
statement.

**The shift rule 4 can still admit** in one pair is `0.03 × (C + E) / 15 ms`
round trips per position, with both checked costs moving by the full
tolerance in the same direction. It is a lower bound on what the rules
admit, since handler time outside execution and the transport remainder
are not checked. With BLOCK-TIME-1's undisturbed 20 ms costs that is
about 0.25 round trips for every arm: **1.1% of the prediction for
`sequential` and `block-1`, 2.2% for `block-2`, 4.2% for `block-4`, and
8.1% for `block-8`**, inside the 10% band. **This covers the two checked
costs only. It does not bound the worker's handler time outside execution
or the transport remainder, so these rules do not bound a valid pair's
total distortion below 10%.** Averaging pairs reduces it
only if the shifts vary in sign; a delay-dependent effect would not. The
analysis reports this bound for each arm from the costs of its valid pairs.

**A failure of rule 4 is reported as an observed execution-cost change,
not as external interference.** The rule assumes both costs are invariant
to the configured delay, which is plausible here but not guaranteed:
different pacing could change scheduling and CPU behaviour. If the rule
fails systematically for one arm or one delay order, that is a finding to
report, and not a reason to change the tolerance.

## Analysis, fixed in advance

`t(trial)` is the measured-region wall time divided by 56.

1. **Per pair**: `s = (t(20 ms) − t(5 ms)) / 15 ms`, in round trips per
   evaluated position, for every pair, valid or void; only valid pairs are
   used below. Each pair is reported with its order, bracket drift, both
   times, its within-pair coordinator and worker-execution differences, and
   void reasons if any. The number of pairs failing rule 4 is reported per
   arm and per delay order.
2. **Per arm**, if **at least three of its four pairs are valid**: the mean
   of the valid pairs' `s`, its ratio to `24 / W`, each valid pair's `s`,
   and whether the mean lies within 10% of the prediction ("follows the
   exchange model"). With two or fewer valid pairs the arm's slope is
   reported as **missing**, with the count. The three-of-four rule is a
   minimum for reporting, not a confidence level or significance test.
3. **Secondary, ungated slope** per arm (see "What rule 4 conditions on"),
   and the shift rule 4 admits for that arm, from its valid pairs' costs;
   and for every arm trial, its largest and median per-exchange remainder.
   None of these is the registered statement.
4. **Delay order**: per arm, the mean `s` of its valid 5→20 pairs and of
   its valid 20→5 pairs, and their difference. Reported, not gated.
5. **Worker execution**: for each arm and delay, the worker's execution time
   per position in every trial (minimum, median, maximum and each value),
   for valid pairs and separately for all pairs, with the count of trials
   that lack it.
6. **Times**: mean `t(5 ms)` and `t(20 ms)` per arm over valid pairs,
   reported as a table. **No speed comparison is made or claimed**; that was
   BLOCK-TIME-1's question.
7. **Decomposition** per trial, as in BLOCK-TIME-1: exchanges, coordinator,
   worker handler and execution, configured delay, transport remainder, and
   bytes moved.

Nothing here is a claim about accepted-token or generation throughput.

## Host controls

### Telemetry: recorded, never a gate

Throughout the run, written beside the receipt:

- every 5 s: CPU share and resident memory of the top processes by CPU, and
  the total, from `ps`;
- every 30 s: `pmset -g therm` (thermal and performance warning levels) and
  the kernel's memory-pressure and free-memory levels from `sysctl`;
- each trial's start and end time (UTC and monotonic), so samples can be
  matched to trials;
- after the run: Spotlight worker starts per minute over the run window,
  from the unified log (minutes in the log's local time).

Telemetry is observational. It does not void, exclude or reorder anything,
and no threshold is applied to it. Its cost is recorded with every sample:
one `ps` took about 40 ms in a smoke test, about 0.8% of one core at one
sample per 5 s, in a separate process from the worker and coordinator.
Spotlight's configuration is not changed.

### Quiet machine

The run is admissible only if: both block-frame qualification receipts for
the relay in use are supplied; the peer sessions confirmed quiet before the
first trial and the handshake is saved with the receipt; and no compiler or
model process other than this experiment's runs at start or end. **The end
is checked by the driver after the worker has shut down; a busy process
then means the run does not count**, even if it completed.

**The quiet request names what is excluded**: worktree creation, checkouts,
clones, branch switches that rewrite the working tree, builds and
compilers, model runs, benchmarks, archive creation or extraction, uploads
and downloads of large files, large file writes, and any other CPU- or
disk-intensive work. Reading, editing small files and `git`/`gh` commands
that do not rewrite a working tree are allowed.

### Pausing

Other sessions' work takes priority over this run, so the run can be
held. Creating a file named `PAUSE` in the run directory makes the driver
finish the pair it is in, then wait before the next pair's first bracket
until the file is removed. **While it holds, the driver writes `PAUSED`**
in the run directory, and removes it on resuming: the session that asked
should wait for `PAUSED` before starting heavy work, since the pair in
flight (up to about five minutes) is still being measured until then. The
worker stays resident (about 12 GB) but idle. Pausing is possible only
there and before warm-up, never inside a pair, so no rule voids a pair
because of a pause. **It is not neutral, though**: there is no warm-up
after a hold, and a long hold under other sessions' load leaves the worker
cold. The next pair's own brackets and rule 4 measure that state; they do
not undo it. Every pause is recorded with the pair it preceded and its
start and end times, in `pauses.json` and the analysis.

**There is no way to pause inside a pair, and none may be used in an
admissible run.** Stopping the run's processes with a signal (`SIGSTOP`), or
anything else that suspends the driver mid-trial, is not a pause under this
protocol: the stopped time lands inside a measurement. The driver therefore
watches for its own suspension. A thread in the driver ticks every 0.1 s; a
gap of more than **0.5 s** in either the monotonic or the wall clock is
recorded as a stall, with its length, placed on the monotonic clock (which
a wall-clock correction cannot move), in `suspensions.json` and the
analysis. **Any trial whose monotonic span overlaps a stall voids its
pair**, by name (rule 5). A stall can also be a starved thread on an
overloaded host or a system sleep; it voids the pair either way, and is
reported as a stall, not as a cause. The rule and its thresholds are in
`block_time_2_model.py`, under the freeze.

**What the watch cannot see.** It runs in the driver's process. Stopping
the run's whole process group, the way to stop everything at once, stops
the driver too and is seen. **Stopping the worker, the relay or the trial
process alone is not seen** by rule 5:

- a stop of the trial process lands in its coordinator time, which rule 4
  catches beyond about 0.06 s (3% of about 35 ms per position over 56
  positions);
- a stop of the worker lands in worker execution if it falls inside it
  (caught beyond about 0.14 s), and otherwise in exchange time;
- a stop of the relay while it holds queued chunks makes them late on
  release, which the relay counts; rule 3 then voids the trial once the
  summed lateness passes 1% of the bracket's time per position over 216
  positions (about 0.3 s at 130 ms per position). A stop of the relay while
  it holds nothing, or any other stall while the coordinator waits on an
  exchange, lands in exchange time, which no rule compares within a pair.

How large an undetected stall in exchange time is: one of `d` seconds in a
20 ms `block-8` trial adds `d / 56` s per position and `d / (56 × 15 ms)`
round trips to that pair's slope, so 0.5 s adds 0.60 round trips, 20% of
the predicted 3. A within-pair check on the transport remainder was
considered and not adopted: unlike coordinator and execution time it does
depend on the delay (BLOCK-TIME-1: 6.4 ms per position for `sequential` at
0 ms, 10.1 at 20 ms), so there is no invariant to compare. The quiet
request, the absence of any pause inside a pair, and the per-arm mean over
three or four pairs are the controls for it. To make one visible, every arm
trial reports its **largest per-exchange remainder against the median of
its own exchanges** (round trip less worker handler and delay): a stall in
one exchange stands far above the median at any delay. It is reported, not
a gate. The transport-remainder evidence above compares 0 and 20 ms; it is
the same relay and frames at 5 ms, but that point was not measured.

## Budget

From BLOCK-TIME-1's trial durations (including process start): a bracket
about 30 s; at 5 ms, `sequential` and `block-1` about 57 s, `block-2` 43 s,
`block-4` 34 s, `block-8` 31 s; at 20 ms, `sequential` and `block-1` about
136 s, `block-2` 82 s, `block-4` 55 s, `block-8` 41 s. One pair of each arm
is about 16 minutes; four of each, about **65 minutes**, plus warm-up and
worker start, **about 70 to 75 minutes**. The quiet window to request is
90 minutes.

## Freezing

- Independent review of this document, `block_time_2_model.py`,
  `block_time_2.py` and `host_telemetry.py`, with their tests: two rounds,
  the second approving with minor changes, all of which were made.
- A declared functional diagnostic was run
  ([`20261115-01-diagnostic`](link-results/block-time-2/README.md)); it does
  not count, and its pair 0, which held a 5-second stop, led to rules 4
  and 5.
- Hashes of this document and the contract it depends on
  (`block_time_2_model.py`, `block_time_2.py`, `block_time_model.py`,
  `link_block_frames_model.py`, `link_model.py`, `profile_exact.py`,
  `block_frames.json`) are recorded in `frozen-protocols.json` under
  "BLOCK-TIME-2 revision 1", and checked in CI. The driver is frozen too, because the stall
  watch, the order in which stalls are attached before pairs are judged,
  and the decision whether a run counts are in it. Telemetry is
  implementation; every receipt records its hash.

## Implementation

[`block_time_2_model.py`](block_time_2_model.py) (schedule, pair validity,
analysis; pure), [`block_time_2.py`](block_time_2.py) (driver, reusing
BLOCK-TIME-1's trial runner and per-trial gates), and
[`host_telemetry.py`](host_telemetry.py) (sampler). Tests:
`test_block_time_2_model.py` and `test_host_telemetry.py`.
