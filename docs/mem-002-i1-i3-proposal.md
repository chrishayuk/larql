# MEM-002 I1 + I3: trustworthy residency evidence — implementation proposal

**Class: IMPLEMENTATION PROPOSAL, for review before any code.** Dated
2026-11-01 at main `b1e6a524`. Nothing is implemented here. It serves the
cold-state qualification candidate (#670, cited **[Q §n]**), where I1
blocks the freeze and I3 is required for PC2 [Q §4, §9]. **[D §n]** is the
merged cold-state design note (`docs/mem-002-cold-state-design.md`, #669),
and **[R §n]** is the merged MEM-001 reconnaissance
(`docs/mem-001-reconnaissance.md`, #667).

## 1. Scope

**In scope:**

- **I1:** per-region residency reads report success or failure, and the
  witness the residency curve prints carries the failed count.
- **I3:** a minimal per-region residency report, on request.
- Their tests, and the operational procedure for the positive controls PC1
  and PC2.

**Revised 2026-11-01** after maintainer review (APPROVE WITH CHANGES):

- one captured observation per scan, with both views derived from it (§3.2);
- PC1 and PC2 corrected (§5);
- a versioned JSON Lines schema (§3.4);
- an admission-quality total that excludes any failed region (§3.2).

**Revised again 2026-11-01**, after the targeted review of `11376af6`
(preserved in `docs/audits/mem-002-i1-i3-proposal/targeted-review-11376af6.md`):

- the formatting, schema and failure tests move to constructed observations
  (§4, F1);
- PC2 becomes a matched sham-against-treatment comparison, with an explicit
  infeasibility outcome (§5, F3);
- the delta is defined (§3.2, F5);
- region identity is separated from segment-file identity (§3.3, F4);
- the schema is completed (§3.4, F6);
- the empty and default semantics are fixed (§3.1, §3.2, F7);
- the `None → 0` inventory is completed (§2.3).

**No bind-only execution mode is added** (maintainer decision).

**Out of scope:**

- the admission decision and its recording (I2);
- clone tooling (I4);
- the machine-state recorder (I5);
- displacement (I6);
- any change to execution, timing, routing, prefetch or loading;
- the footprint and budget paths that also read residency (§2.3).

**Behavioural constraint.** No timed region changes. Every new read
happens where the existing witness reads already happen, outside the timers
[D §1.1].

## 2. Current behaviour on main (`b1e6a524`)

### 2.1 The read

`WeightRegion::resident_bytes() -> Option<u64>`
(`format/vindex3/represent/physical.rs:269-307`):

- returns `Some(0)` for an empty region;
- returns `None` when `sysconf(_SC_PAGESIZE) <= 0`, or when `mincore`
  returns non-zero;
- returns `None` on non-unix targets;
- otherwise returns the resident whole pages × the page size, clipped to
  the region's length. Boundary pages are counted by every region that
  overlaps them [R §1.1].

**The cause of a failure is discarded:** neither the errno nor the reason
is kept.

### 2.2 The aggregation the witness uses

- `LoadedWeight::resident_bytes() -> usize` maps a mapped region's `None` to
  0 (`opplan/exec/weights/loaded.rs:37`).
- `PreparedOperands::mapped_residency()`
  (`opplan/exec/prepared/residency.rs:417-456`) sums that over every weight
  with `mapped_bytes > 0`. It returns
  `MappedResidency { regions, mapped_bytes, resident_bytes }`
  (`prepared/census.rs:51-55`).
- `DecodeSession::mapped_residency()` delegates to it
  (`decode/steps.rs:121-123`).
- `--residency-curve` reads it at binding (`residency_curve.rs:187`, printed
  at `:195-199`), before each pass's prompt (`:229`), and after each step
  inside `observe` (`:237`).

**So a failed region is indistinguishable from a cold one in every printed
figure.**

### 2.3 Other `None → 0` consumers (not changed by this proposal)

| Site | Use | Why not changed here |
|---|---|---|
| `accounting/budgets.rs:417` | `Observed.resident_bytes`, through `LoadedWeight::resident_bytes()`, feeding `reconcile`'s `mapped_resident` | reported only, never reconciled [R §1.1], and not used by admission |
| `vindex3_cmd/realizations.rs:111` | prints `reconciled.mapped_resident` to the user | the reported end of the row above; a user-visible figure, not a witness |
| `larql-demos/examples/inference/gwv2.rs:257` | reads the aggregate through `mapped_residency()` and prints it | an aggregate consumer, not a direct caller; not a witness |
| `opplan/exec/tests/carrier_write_real.rs:166-173` | a test that prints `mapped.resident_bytes` as run evidence | a test print, not a witness |

`loaded.rs:37` is the only production caller of `WeightRegion::resident_bytes()`.
After this change, the aggregate consumers above read a sum over **read**
regions and still ignore `failed_regions`: they stay blind to failures, but
no longer add a failure as 0.

The review should decide whether these become explicit too, in a later and
separate change. **The admission gate must never read them.**

`SiteResidency::add` (`prepared/census.rs:70-80`) also calls
`resident_bytes()`, but **not for mapped weights**: a weight with
`mapped_bytes() > 0` takes the first branch and is counted as mapped. It is
therefore not a `None → 0` consumer.

## 3. Proposed design

### 3.1 A read that keeps its outcome

The change is in `represent/physical.rs`.

- Add `WeightRegion::residency() -> Result<u64, ResidencyReadError>`, with:
  - `ResidencyReadError::PageSizeUnavailable`;
  - `ResidencyReadError::Mincore { errno: i32 }`;
  - `ResidencyReadError::Unsupported`, for non-unix targets.
- Reimplement `resident_bytes()` as `self.residency().ok()`, so every
  existing caller is unchanged.
- **Empty and non-unix semantics are preserved exactly as today:**
  - on unix, an empty region is `Ok(0)`;
  - on non-unix, **every** region, empty or not, is
    `Err(Unsupported)`, matching today's `None`.
- **errno is captured at the `mincore` call site** (`errno` read
  immediately after a non-zero return), and only for `Mincore`.
- An empty region stays `Ok(0)`.

### 3.2 One observation, two views

The change is in `prepared/residency.rs`.

**One traversal is not enough.** If the aggregate and the per-region report
each called `mincore` separately, residency could change between the calls,
and the two views could disagree. So each observation is captured once, and
both views derive from it.

- **One private iterator over mapped weights** defines the image's mapped
  regions, in a fixed order.
- **One snapshot per observation.**

  ```
  ResidencyObservation {
      started: (wall clock, monotonic),
      ended:   (wall clock, monotonic),
      page_size,
      regions: Vec<RegionResidency>,  // in scan order
  }
  RegionResidency {
      scan_index, store_id, store_offset, len,
      resident: Result<u64, ResidencyReadError>,
  }
  ```

  - `PreparedOperands::observe_residency() -> ResidencyObservation` reads
    every mapped region exactly once, in iterator order, recording
    `scan_index`.
  - `started` and `ended` bracket the scan. **A scan over many regions is
    not atomic**: regions read later in the scan can reflect later state.
    The record says so, and keeps the order.
- **The aggregate is derived from the snapshot.**
  `ResidencyObservation::summary() -> MappedResidency`.
  - `MappedResidency` gains `failed_regions: usize`.
  - `resident_bytes` is the sum over **successfully read** regions only.
    `regions = read + failed`, and a failure is never summed as 0.
  - `MappedResidency::complete_resident_bytes() -> Option<u64>` returns
    `Some` only when `regions > 0` **and** `failed_regions == 0`. An
    observation of no regions, including `MappedResidency::default()`,
    witnesses nothing and returns `None`.
  - **Any admission-quality use must go through `complete_resident_bytes()`**,
    so a failed region can never be silently absorbed into an
    admission-quality total. The admission decision itself remains I2.
- **The existing `mapped_residency()` stays**, as
  `self.observe_residency().summary()`, so its callers keep compiling.
- **`DecodeSession`** gets the matching `observe_residency()` delegate.

**Deltas between two observations** replace today's
`now.resident_bytes.saturating_sub(before.resident_bytes)` on two aggregates
(`residency_curve.rs:255`).

- **Matching key:** the stable region identity of §3.3 (container identity,
  segment file, file offset, length), not `store_id` alone and not
  `scan_index`.
- **Comparable regions:** those present in both observations and read
  successfully in both.
- **Arithmetic:** signed, per region. Δᵢ = residentᵢ(after) − residentᵢ(before),
  as `i64`, with no saturation.
- **Reported quantities:**
  - the comparable-region count;
  - the sum of positive Δᵢ (increase) and the sum of negative Δᵢ (decrease),
    separately;
  - their signed total;
  - the number of regions excluded because they failed in either
    observation, or were present in only one.
- **No comparable regions:** the delta is reported as `n/a`, **never** `0`.
- **The curve's pre-step reading** (`before_res`, `residency_curve.rs:229`
  and inside `observe`) becomes a full `ResidencyObservation`, not a summary,
  so the next delta can be computed per region.

### 3.3 Reading a mapped weight's region

The change is in `weights/loaded.rs`.

- Add `LoadedWeight::mapped_region() -> Option<&WeightRegion>`, returning
  `Some` only for `Mapped`, which is constructed only at
  `experts/routed.rs:347`. The traversal reads residency through it.

**Region identity is not `store_id`.** `WeightRegion::store_id()` is the
store's id, which is the operand **object** name given at
`opplan/exec/operands.rs:282` (`PhysicalStore::map_segment(id, path)`), not
the segment file. `PhysicalStore` keeps no path today
(`represent/physical.rs:73-79`).

- I3 records the segment path at `map_segment`, so
  `PhysicalStore::source_path() -> Option<&Path>` returns `Some` only for
  mapped segments.
- `store_offset()` is an offset into the whole mapping, header included:
  `bytes()` slices `store.all()[offset..offset + len]` (`physical.rs:245-247`).
  So it is a **file offset** within that segment file. The proposal relies on
  this, and the implementation review confirms it.
- **The stable region key** is (container identity, segment path relative to
  the container, file offset, length). That is what maps a `dd` byte range
  of a segment file onto regions (§5).
- To avoid an owned `String` per region per step, each observation keeps a
  store table, and regions refer to it by index. The key is rebuilt from the
  table when needed.
- `LoadedWeight::resident_bytes()` keeps its current meaning for the
  footprint paths in §2.3, with a comment that it is **not** a witness.

### 3.4 What the residency curve prints

The change is in `residency_curve.rs`.

- **Binding line**, from the binding observation:
  `mapped … GB address space over N regions (R read, F failed), Y GB resident in read regions`.
  When `F > 0`, a second line follows:
  `WITNESS INCOMPLETE: F regions could not be read (first: <store_id>@<offset>: <error>)`.
- **Per-step lines** add a column for regions excluded from the delta,
  because they failed in either observation (§3.2). A failure in the middle
  of a run is then visible, and never mixed into a delta.
- **I3 flag:** `--residency-regions <path>` writes observations as JSON
  Lines, at binding and once more after the run.
  - The binding observation is written before the first pass starts, so it
    adds no time to any timed region.
  - It is output only.

**JSON Lines schema, versioned.** Each observation is one header line
followed by one line per region:

```
{"schema":"larql.residency-observation/1","record":"observation",
 "run_id":…,"observation_id":…,"phase":"binding"|"after_run",
 "pid":…,"binary":{"version":…,"commit":…},
 "container":{"path":…,"index_json_sha256":…},
 "started_wall":…,"started_mono_ns":…,"ended_wall":…,"ended_mono_ns":…,
 "page_size":… | null,
 "regions":N,"read":R,"failed":F,
 "mapped_bytes_all":…,"mapped_bytes_read":…,"resident_bytes_read":…}
{"schema":"larql.residency-observation/1","record":"region",
 "run_id":…,"observation_id":…,"scan_index":…,
 "store_id":…,"segment":…,"file_offset":…,"len":…,
 "status":"ok","resident_bytes":…}
{"schema":"larql.residency-observation/1","record":"region",
 "run_id":…,"observation_id":…,"scan_index":…,
 "store_id":…,"segment":…,"file_offset":…,"len":…,
 "status":"failed",
 "error":{"kind":"mincore","errno":…}}
```

- **Identity scope:**
  - `run_id` is unique per process run;
  - `observation_id` is unique within a run;
  - pid, binary and container identity let reports from separate processes
    (A, B and the sham in PC2) be paired and checked against one container.
- **Failure encoding:**
  - `error.kind` is one of `mincore`, `page_size_unavailable` or
    `unsupported`;
  - `errno` is present **only** for `mincore`;
  - `page_size` is `null` when unavailable.
- **Totals:**
  - `mapped_bytes_all` covers every region;
  - `mapped_bytes_read` and `resident_bytes_read` cover read regions only,
    so a resident fraction is computed over read regions.
- **Versioning:** a change to any field's meaning increments the schema
  version, and a reader refuses an unknown version.
- **Nothing here admits or rejects.** The gate, and recording it before the
  timer, is I2 [Q §7].

## 4. Tests

These go in the crates' `tests/` folders, following repository practice.
The new and changed files are held to the per-file coverage floor.

**Formatting, schema and failure reporting are tested on constructed
observations**, not through a whole-model run. The constructed
`ResidencyObservation` values contain mapped regions with successful,
failed, empty and mixed outcomes, so the tests see what the CI fixture
cannot produce.

| Test | What it pins | Where |
|---|---|---|
| Summary with failures | regions `[Ok(a), Err(Mincore), Ok(b)]` give `resident_bytes = a + b`, `failed_regions = 1`, `regions = 3` | `ResidencyObservation::summary()`, on constructed observations |
| **Incomplete never admission quality** | any observation with a failed region, or with no regions (including `MappedResidency::default()`), gives `complete_resident_bytes() = None`; only a non-empty, failure-free observation gives `Some(total)` | the same |
| Delta | for constructed pairs: signed per-region arithmetic; separate increase and decrease sums; the comparable count; exclusion of regions failed in either side or present in one only; `n/a` when no region is comparable | the delta function |
| JSON Lines writer | header and region records for successful, failed, empty and mixed observations. Every line carries the schema version and record type, a failed region carries `status: "failed"` and its error kind, `errno` appears only for `mincore`, and an unavailable page size is `null` | the writer, on constructed observations |
| Line formatter | the binding line names read and failed counts; `WITNESS INCOMPLETE` appears exactly when `failed > 0`, naming the first failure; the per-step column shows excluded regions and prints `n/a` for a delta with no comparable regions | the formatter, on constructed observations |
| Empty region (unix) | `residency()` is `Ok(0)` | `represent/tests/physical_store.rs` (extends the existing test) |
| Warm read | a region over a file just written and read is `Ok(n)`, with `0 < n <= len` | `represent/tests` |
| Preserved public contract | `resident_bytes()` keeps its **pre-change** values on the cases that exist today: `Some(0)` for an empty region on unix, `Some(n)` for the warm read. This replaces a tautological `resident_bytes() == residency().ok()` check | `represent/tests` |
| One observation, real regions | for one `observe_residency()` snapshot on a fixture **with mapped regions** (a mapped per-expert bank, as in `kimi_per_expert_prepared.rs`): the region count equals `summary().regions`, the `Ok` values sum to `summary().resident_bytes`, and the `scan_index` values run 0..N. Both views come from the same snapshot | a prepared fixture with mapped regions |
| Segment identity | a mapped store reports its `source_path()`, and regions' file offsets fall within that file | `represent/tests` |
| CLI smoke test | the existing `the_residency_curve_runs_cold_and_warm_passes_over_one_bound_image` (`vindex3_cmd/tests/generate.rs:79`) still completes, with and without `--residency-regions` | **smoke test only.** Its dense fixture has no mapped regions, and the test only calls `run()`, so it is **not** credited with validating residency, failure reporting or output format |

**Honest limits of CI testing:**

- **A real `mincore` failure is not induced** in CI. Forcing one would need
  an invalid address range, which the safe `WeightRegion` API cannot
  produce. The failure path is pinned on constructed observations, and the
  errno capture by review.
- **A cold region cannot be manufactured in CI.** A file written by the test
  is already in the page cache. So "non-target regions read zero" is not a
  CI test. PC1 and PC2 cover it on the real subject (§5).

## 5. The positive controls these instruments enable

These are operational procedures for phase B of the qualification [Q §8].
They are **not** CI tests, and **not authorised to run** until the protocol
is frozen.

Two general rules:

- **Neither control assumes that pages stay resident.** The OS may reclaim
  pages between the read and the witness. Each control therefore uses a
  documented observation window, the time from read completion to the
  binding observation's `started`, and reports it.
- **Every read in a control is checked for success:** exit status, and
  bytes read equal to bytes requested. A control whose read failed is void,
  not failed.

- **PC1, cross-process visibility, on controlled known ranges.**
  1. Another process reads a set of **known byte ranges** of `.s7` segment
     files (`dd` with a stated block size, skip and count), and its read
     success is recorded.
  2. Within the stated observation window, a fresh process binds `.s7` and
     writes `--residency-regions`.
  3. **Pass:**
     - zero failed regions;
     - **positive detection**: the regions overlapping the read ranges report
       resident bytes, at no less than a fraction fixed at the freeze;
     - **correct attribution**: the detection falls in those regions.

     PC1 does **not** require resident bytes to equal the bytes read, nor a
     whole file to remain resident.
- **PC2, region-level attribution: a matched sham-against-treatment
  comparison.** No bind-only mode is added. The observer's own run is part
  of both conditions, and the contrast between them is what is tested.
  1. **Matched procedure.** Both conditions use the same container, the same
     binary and flags, the same observation window and the same prompt, and
     alternate in a declared order. Each condition is:
     - observation A, a fresh process with `--residency-regions`;
     - the intervention;
     - observation B, a second fresh process.
     - **Treatment:** the intervention is another process reading a known
       byte range of a segment file (`dd`), with read success recorded.
     - **Sham:** the intervention is the same process invocation with a zero
       count, so nothing is read.
  2. **Target regions, fixed before any B is examined.** They are the
     regions whose stable key (§3.3) overlaps the treatment range. Regions in
     a stated readahead distance after the range, and regions sharing a
     boundary page with a target, are classified and reported separately.
  3. **Already-resident targets.** A target near fully resident in A cannot
     show an increase. It is excluded and counted. If too few targets remain,
     by a count fixed at the freeze, the repetition is void.
  4. **The comparison** is, for the **same target regions**, the per-region
     Δ (B − A, §3.2) under treatment against under sham. It is replicated
     over a number of matched pairs fixed at the freeze. Background changes,
     and warming caused by the observer's own run, appear in both
     conditions. Only the read appears in the treatment alone.
  5. **Pass** (all under criteria fixed at the freeze):
     - no report has a failed region;
     - the targets' treatment Δ exceeds their sham Δ;
     - non-target, non-readahead, non-boundary regions show no treatment-
       against-sham difference.
  6. **Infeasible, not relaxed.** If the observer's own run warms the target
     regions in A, or in the sham, so that no contrast can be shown, PC2 is
     **declared infeasible with this procedure**. The acceptance criteria are
     not loosened until it passes. A separate bind-only observation
     capability is then reconsidered, with its own implementation review
     [Q §10].

  The ranges, block size, window, readahead distance, the "near fully
  resident" cut, the replicate count and the criteria are fixed at the
  freeze, with an independent justification [Q §7].

## 6. Review checklist

Resolved in the 2026-11-01 review: the errno-carrying `Result` (kept),
JSON Lines with a schema version (kept, §3.4), and leaving the §2.3 sites
unchanged for this scope (accepted). Open for the implementation review:

1. Is one snapshot per observation enough, or should a re-scan be
   available to bound how much state changes within a scan?
2. Is `complete_resident_bytes()` the right guard, or should the summary
   type make an incomplete total impossible to read as admission quality?
3. Are the CI limits in §4 acceptable, given that PC1 and PC2 cover the
   real-subject cases?

## Sources checked

- `represent/physical.rs:244-321` (`WeightRegion`: `bytes`, `store_id`,
  `len`, `resident_bytes`, `store_offset`)
- `weights/loaded.rs:37`
- `prepared/residency.rs:417-456`
- `represent/physical.rs:73-79,90-111,245-247` (`PhysicalStore` fields and
  `map_segment`; `bytes()` slicing the whole backing)
- `opplan/exec/operands.rs:282` (store id is the object name)
- `opplan/exec/experts/routed.rs:347` (the only `LoadedWeight::Mapped`
  construction)
- `vindex3_cmd/realizations.rs:111`; `opplan/exec/tests/carrier_write_real.rs:166-173`
- `prepared/census.rs:51-55,70-80`
- `decode/steps.rs:121-123`
- `accounting/budgets.rs:410-419`
- `residency_curve.rs:180-200,229,237`
- `vindex3_cmd/tests/generate.rs:79`
- `represent/tests/physical_store.rs:56-67`
- `larql-demos/examples/inference/gwv2.rs:257`
