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

### 3.2 One traversal, two views

The change is in `prepared/residency.rs`.

- Factor the traversal inside `mapped_residency()` into one private
  iterator over **mapped weights**, used by both views below. There is one
  definition of "the image's mapped regions", so the aggregate and the
  per-region report cannot disagree about membership.
- **`MappedResidency` gains `failed_regions: usize`.** `resident_bytes`
  becomes the sum over **successfully read** regions only.
  - `regions` keeps counting every mapped region.
  - `regions = read + failed`.
  - A failure is never summed as 0.
- **I3: `PreparedOperands::region_residency() -> Vec<RegionResidency>`**,
  where each entry has:
  - `store_id` (`WeightRegion::store_id()`);
  - `store_offset` (`store_offset()`);
  - `len`;
  - `resident: Result<u64, ResidencyReadError>`.

  It is computed on request only. Nothing calls it on a timed path.
- **`DecodeSession`** gets the matching `region_residency()` delegate.

### 3.3 Reading a mapped weight's region

The change is in `weights/loaded.rs`.

- Add `LoadedWeight::mapped_region() -> Option<&WeightRegion>`, returning
  `Some` only for `Mapped`. The traversal reads residency through it.
- `LoadedWeight::resident_bytes()` keeps its current meaning for the
  footprint paths in §2.3, with a comment that it is **not** a witness.

### 3.4 What the residency curve prints

The change is in `residency_curve.rs`.

- **Binding line:**
  `mapped … GB address space over N regions (R read, F failed), Y GB resident in read regions`.
  When `F > 0`, a second line follows:
  `WITNESS INCOMPLETE: F regions could not be read (first: <store_id>@<offset>: <error>)`.
- **Per-step lines** add the failed count in a new column, so a failure in
  the middle of a run is visible.
- **I3 flag:** `--residency-regions <path>` writes the per-region report at
  binding, and once more after the run, as JSON Lines. Each line carries
  `phase`, `store_id`, `store_offset`, `len`, and `resident` or `error`.
  - The binding report is written before the first pass starts, so it adds
    no time to any timed region.
  - It is output only.
- **Nothing here admits or rejects.** The gate, and recording it before the
  timer, is I2 [Q §7].

## 4. Tests

These go in the crates' `tests/` folders, following repository practice.
The new and changed files are held to the per-file coverage floor.

| Test | What it pins | Where |
|---|---|---|
| Aggregation with failures | given region outcomes `[Ok(a), Err(_), Ok(b)]`: `resident_bytes = a + b`, `failed_regions = 1`, `regions = 3`; a failure never contributes 0 to the sum as if read | a pure summariser shared by `mapped_residency()`, tested directly |
| Empty region | `residency()` is `Ok(0)`, `resident_bytes()` is `Some(0)` | `represent/tests/physical_store.rs` (extends the existing test) |
| Warm read | a region over a file just written and read is `Ok(n)` with `n > 0` and `n <= len` | `represent/tests` |
| One traversal | `region_residency()` returns exactly `regions` entries. The sum of its `Ok` values equals `mapped_residency().resident_bytes`, and its error count equals `failed_regions` | a prepared fixture already used by the residency tests (for example `kimi_per_expert_prepared.rs`) |
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

- **PC1, cross-process visibility.**
  1. Another process reads selected `.s7` segment files (`cat … > /dev/null`).
  2. A fresh process binds `.s7` with `--residency-curve`.
  3. **Pass:** zero failed regions, and the regions belonging to those files
     report resident bytes of the files' size, within the boundary-page
     convention.
- **PC2, region-level correctness, by difference.** This needs no cold
  baseline, so PC2 does not depend on either cold method.
  1. A fresh process binds and writes `--residency-regions` (report A).
  2. Another process reads a **known byte range** of the expert-bank
     segment (`dd` with a 16 KiB block size, a known skip and a known
     count).
  3. A second fresh process binds and writes report B.
  4. **Pass:**
     - the per-region difference B − A is non-zero exactly in the regions
       whose `[store_offset, store_offset + len)` overlaps the read range;
     - it is zero elsewhere, apart from boundary pages shared with those
       regions;
     - neither report has a failed region.

  The range, block size and tolerance are fixed at the freeze.

## 6. Review checklist

1. Is the errno-carrying `Result` the right granularity, or is a plain
   failed/succeeded flag enough for admission?
2. Is one shared traversal acceptable for the aggregate and the per-region
   report (§3.2)?
3. Is JSON Lines acceptable for the I3 report, and should it carry a schema
   version?
4. Should the §2.3 `None → 0` sites be made explicit in a follow-up?
5. Are the CI limits in §4 acceptable, given that PC1 and PC2 cover the
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
