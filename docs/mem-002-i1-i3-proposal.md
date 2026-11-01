# MEM-002 I1 + I3: trustworthy residency evidence — implementation proposal

**Class: IMPLEMENTATION PROPOSAL, for review before any code.** Dated
2026-11-01 at main `b1e6a524`. Nothing is implemented here. It serves the
cold-state qualification candidate (#670, cited **[Q §n]**), where I1
blocks the freeze and I3 is required for PC2 [Q §4, §9].

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
  (`opplan/exec/prepared/residency.rs:417-…`) sums that over every weight
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
| `accounting/budgets.rs:417` | `Observed.resident_bytes`, feeding `reconcile`'s `mapped_resident` | reported only, never reconciled [R §1.1], and not used by admission |
| `examples/inference/gwv2.rs:257` (`larql-demos`) | a demo's printout | not a witness |

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
    `Some` only when `failed_regions == 0`.
  - **Any admission-quality use must go through `complete_resident_bytes()`**,
    so a failed region can never be silently absorbed into an
    admission-quality total. The admission decision itself remains I2.
- **The existing `mapped_residency()` stays**, as
  `self.observe_residency().summary()`, so its callers keep compiling.
- **`DecodeSession`** gets the matching `observe_residency()` delegate.

**Deltas between two observations** (the per-step resident delta) are
computed only over regions read successfully in **both** observations.

- Each step reports how many regions failed in either observation.
- A region that failed in either is excluded from the delta, and counted.
- That way the delta never mixes a successful reading with a failed one.

### 3.3 Reading a mapped weight's region

The change is in `weights/loaded.rs`.

- Add `LoadedWeight::mapped_region() -> Option<&WeightRegion>`, returning
  `Some` only for `Mapped`. The traversal reads residency through it.
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
 "observation_id":…,"phase":"binding"|"after_run",
 "started_wall":…,"started_mono_ns":…,"ended_wall":…,"ended_mono_ns":…,
 "page_size":…,"regions":N,"failed":F}
{"schema":"larql.residency-observation/1","record":"region",
 "observation_id":…,"scan_index":…,
 "store_id":…,"store_offset":…,"len":…,
 "status":"ok","resident_bytes":…}
{"schema":"larql.residency-observation/1","record":"region",
 "observation_id":…,"scan_index":…,
 "store_id":…,"store_offset":…,"len":…,
 "status":"failed",
 "error":{"kind":"mincore"|"page_size_unavailable"|"unsupported","errno":…}}
```

- A change to any field's meaning increments the schema version.
- A reader refuses an unknown version.
- **Nothing here admits or rejects.** The gate, and recording it before the
  timer, is I2 [Q §7].

## 4. Tests

These go in the crates' `tests/` folders, following repository practice.
The new and changed files are held to the per-file coverage floor.

| Test | What it pins | Where |
|---|---|---|
| Aggregation with failures | for an observation with regions `[Ok(a), Err(_), Ok(b)]`: `resident_bytes = a + b`, `failed_regions = 1`, `regions = 3`, `complete_resident_bytes() = None`; with no failures, `complete_resident_bytes() = Some(total)` | `ResidencyObservation::summary()`, built from constructed observations, tested directly |
| Delta excludes failures | for two constructed observations where a region failed in one of them, the delta sums only regions read in both, and reports the excluded count | the delta function, tested directly |
| Empty region | `residency()` is `Ok(0)`, `resident_bytes()` is `Some(0)` | `represent/tests/physical_store.rs` (extends the existing test) |
| Warm read | a region over a file just written and read is `Ok(n)` with `n > 0` and `n <= len` | `represent/tests` |
| One observation | for **one** `observe_residency()` snapshot on a prepared fixture: the region count equals `summary().regions`, the sum of `Ok` values equals `summary().resident_bytes`, the error count equals `failed_regions`, and the `scan_index` values run 0..N in order. Both views are compared **from the same snapshot**, never across two calls | a prepared fixture already used by the residency tests (for example `kimi_per_expert_prepared.rs`) |
| Schema | every JSON line carries `schema` = `larql.residency-observation/1` and its record type; a failed region carries `status: "failed"` and an error kind | the CLI test below |
| Compatibility | `resident_bytes()` equals `residency().ok()` | `represent/tests` |
| CLI output | the binding line names read and failed counts, `--residency-regions` writes one JSON line per mapped region, and a fixture with no failures prints no `WITNESS INCOMPLETE` line | extends `vindex3_cmd/tests/generate.rs:79` (`the_residency_curve_runs_cold_and_warm_passes_over_one_bound_image`) |

**Honest limits of CI testing:**

- **A real `mincore` failure is not induced** in CI. Forcing one would need
  an invalid address range, which the safe `WeightRegion` API cannot
  produce. The failure path is pinned at the summariser seam, and the errno
  mapping by review.
- **A cold region cannot be manufactured in CI.** A file written by the test
  is already in the page cache. So "non-target regions read zero" is not a
  CI test. It is what PC2 checks on the real subject (§5).

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
- **PC2, region-level attribution, by difference against controls.** This
  needs no cold baseline, so PC2 does not depend on either cold method.
  1. A fresh process binds and writes `--residency-regions` (report A).
  2. Another process reads a **known byte range**. Its read success is
     recorded.
  3. Within the stated window, a second fresh process binds and writes
     report B.
  4. The regions are classified **before** B is examined, from report A and
     the range alone:

     | Class | Definition | Expectation |
     |---|---|---|
     | **known-positive** | overlaps the read range, and was not already near fully resident in A | residency increases |
     | **already-resident** | overlaps the range, but was near fully resident in A | cannot show an increase; reported, and excluded from the attribution test |
     | **readahead window** | does not overlap the range, but lies within a stated distance after the range's end in file order | may increase through readahead; reported separately, and neither a pass nor a fail |
     | **boundary** | shares a boundary page with a known-positive region | may increase by up to the shared pages; reported separately |
     | **unobserved** | everything else | change compared with the null control below |

  5. **Null control (sham):** the same procedure with **no read** in step 2.
     The distribution of per-region change across unobserved regions under
     the sham is the background, from concurrent cache activity and
     reclamation.
  6. **Pass:**
     - neither report has a failed region;
     - the known-positive regions increase, at no less than a fraction fixed
       at the freeze;
     - the unobserved regions' change is not distinguishable from the sham
       background, under a criterion fixed at the freeze.

     Exact zero change outside the range is **not** required.

  The ranges, block size, window, readahead distance, the "near fully
  resident" cut and the criteria are fixed at the freeze, with an
  independent justification [Q §7].

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
- `prepared/residency.rs:412-440`
- `prepared/census.rs:51-55,70-80`
- `decode/steps.rs:121-123`
- `accounting/budgets.rs:410-419`
- `residency_curve.rs:180-200,229,237`
- `vindex3_cmd/tests/generate.rs:79`
- `represent/tests/physical_store.rs:56-67`
- `larql-demos/examples/inference/gwv2.rs:257`
