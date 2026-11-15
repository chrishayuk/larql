# BLOCK-TIME-2: the RTT differential slope, in bracketed pairs

Status: **DRAFT, revision 1. Not frozen, not approved for a qualifying or
admissible run.** Nothing below has been measured. Once frozen, a change to
this document or to `block_time_2_model.py` is a new revision, not an edit.

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
pairs. **Delay order alternates**: within each arm, its pairs run 5→20,
20→5, 20→5, 5→20 in the order they occur, so every arm has two pairs in
each order.

### Schedule

- **Four pairs per arm, twenty pairs, eighty trials**, after warm-up. No
  zero-delay arm trials, no unprofiled trials, no no-relay controls.
- **The order of the twenty pairs is a seeded shuffle**, seed `20261115`,
  computed by `block_time_2_model.schedule()` and recorded in the manifest
  before the first trial. Changing the seed is a new revision.
- **Nothing is ever added to the schedule.** A void pair is not rerun or
  replaced, and nothing is run because something else failed.

### Validity

A **pair is valid** only if all hold:

1. Its two brackets agree within 1% (BLOCK-TIME-1's bracket drift).
2. None of its four trials has a frame-count defect.
3. The relay was on schedule in all four.

A parity mismatch in any trial stops the run, as in BLOCK-TIME-1. A void
pair is kept, with its reasons, and is never fitted or pooled.

## Analysis, fixed in advance

`t(trial)` is the measured-region wall time divided by 56.

1. **Per pair**: `s = (t(20 ms) − t(5 ms)) / 15 ms`, in round trips per
   evaluated position, for every pair, valid or void; only valid pairs are
   used below. Each pair is reported with its order, bracket drift, both
   times, and void reasons if any.
2. **Per arm**, if **at least three of its four pairs are valid**: the mean
   of the valid pairs' `s`, its ratio to `24 / W`, each valid pair's `s`,
   and whether the mean lies within 10% of the prediction ("follows the
   exchange model"). With two or fewer valid pairs the arm's slope is
   reported as **missing**, with the count. The three-of-four rule is a
   minimum for reporting, not a confidence level or significance test.
3. **Delay order**: per arm, the mean `s` of its valid 5→20 pairs and of
   its valid 20→5 pairs, and their difference. Reported, not gated.
4. **Worker execution**: for each arm and delay, the worker's execution time
   per position in every trial (minimum, median, maximum and each value),
   for valid pairs and separately for all pairs.
5. **Times**: mean `t(5 ms)` and `t(20 ms)` per arm over valid pairs,
   reported as a table. **No speed comparison is made or claimed**; that was
   BLOCK-TIME-1's question.
6. **Decomposition** per trial, as in BLOCK-TIME-1: exchanges, coordinator,
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
- each trial's start and end wall-clock time, so samples can be matched to
  trials;
- after the run: Spotlight worker starts per minute over the run window,
  from the unified log.

Telemetry is observational. It does not void, exclude or reorder anything,
and no threshold is applied to it. Its cost is recorded with every sample:
one `ps` took about 40 ms in a smoke test, about 0.8% of one core at one
sample per 5 s, in a separate process from the worker and coordinator.
Spotlight's configuration is not changed.

### Quiet machine

The run is admissible only if: both block-frame qualification receipts for
the relay in use are supplied; the peer sessions confirmed quiet before the
first trial and the handshake is saved with the receipt; and no compiler or
model process other than this experiment's runs at start or end.

**The quiet request names what is excluded**: worktree creation, checkouts,
clones, branch switches that rewrite the working tree, builds and
compilers, model runs, benchmarks, archive creation or extraction, uploads
and downloads of large files, large file writes, and any other CPU- or
disk-intensive work. Reading, editing small files and `git`/`gh` commands
that do not rewrite a working tree are allowed.

## Budget

From BLOCK-TIME-1's trial durations (including process start): a bracket
about 30 s; at 5 ms, `sequential` and `block-1` about 57 s, `block-2` 43 s,
`block-4` 34 s, `block-8` 31 s; at 20 ms, `sequential` and `block-1` about
136 s, `block-2` 82 s, `block-4` 55 s, `block-8` 41 s. One pair of each arm
is about 16 minutes; four of each, about **65 minutes**, plus warm-up and
worker start, **about 70 to 75 minutes**. The quiet window to request is
90 minutes.

## Before freezing

- Review of this document, `block_time_2_model.py`, `block_time_2.py` and
  `host_telemetry.py`, with their tests.
- One declared diagnostic is optional: it would exercise the driver and the
  telemetry end to end, would never count, and its timings would not be
  summarised.
- On freezing: hashes of this document and the model added to
  `frozen-protocols.json` under "BLOCK-TIME-2 revision 1".

## Implementation

[`block_time_2_model.py`](block_time_2_model.py) (schedule, pair validity,
analysis; pure), [`block_time_2.py`](block_time_2.py) (driver, reusing
BLOCK-TIME-1's trial runner and per-trial gates), and
[`host_telemetry.py`](host_telemetry.py) (sampler). Tests:
`test_block_time_2_model.py` and `test_host_telemetry.py`.
