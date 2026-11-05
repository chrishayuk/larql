# MEM-002: protocol reconciliation after I1 + I3

**Class: RECONCILIATION — NOT A FREEZE.** Dated 2026-11-02 at main
`1c297356`. Nothing was built, run, evicted or measured for it, and it sets
no number.

It reconciles three sources against the code now on main:
- the qualification protocol candidate
  ([`mem-002-cold-state-qualification.md`](mem-002-cold-state-qualification.md),
  #670, cited **[Q §n]**);
- the follow-ups from the #673 review (issue #674, cited **[#674 Fn]**);
- the maintainer's requirements of 2026-11-02 (cited **[M]**).

The cold-state design note ([`mem-002-cold-state-design.md`](mem-002-cold-state-design.md), #669)
is cited **[D §n]**.

The code is I1 + I3 (#672) and the NB6 / NB1 / NB4 corrections (#673).

It answers one question: **for every requirement the protocol makes, what
already satisfies it, what has to be built for it, and what can only ever
be a protocol control?** It also records five gaps the reconciliation
found (§3) and proposes freezing in two stages (§4).

## 1. Status vocabulary

| Status | Meaning |
|---|---|
| **MET** | Satisfied by code on main, with the evidence cited. |
| **I2 … I6** | Needs that instrument [Q §9]. Not built. |
| **EVAL** | Needs the independent evaluator: the tool that scores records [Q §10]. Not built, and **not in I1–I6** (gap G3). |
| **PROTOCOL** | A procedural control. No code can satisfy it; the freeze must state it and the run record must show it was followed. |
| **GAP** | Required, but assigned to nothing yet. Listed in §3 with a proposed owner. |

## 2. Decision table

### 2.1 Residency witness [Q §3]

| Requirement | Status | Evidence or owner |
|---|---|---|
| Each region's read is success with a count, or failure; never 0 for unknown | **MET** | `WeightRegion::residency() -> Result<u64, ResidencyReadError>`. The failed-region count is kept in `MappedResidency` (`prepared/census.rs`), and failures are excluded from sums (`prepared/observation.rs`, `summary`). |
| Binding line reports regions read, regions failed, and the resident total over read regions | **MET** | `residency_record::binding_line`, plus `WITNESS INCOMPLETE` naming the first failure (`incomplete_line`). |
| An attempt with any failed region is **rejected** as "witness failed", counted apart from "above threshold" | **I2** | I1 + I3 *report* the failure; nothing *decides* admission. `complete_resident_bytes()` returns `None` for any failure, empty observation or inconsistent counts, but nothing calls it outside tests (#673 targeted check). |
| An inconsistent observation is never admission-quality | **MET** (instrument), **I2** (decision) | `MappedResidency::is_consistent()` feeds `complete_resident_bytes()`. The record header has `consistent`, and the binding line says `INCONSISTENT COUNTS` (#673). |
| Admission uses a complete observation, not only a consistent one | **I2** + **EVAL** | The header has `consistent` but no `complete` field [#674 F6]. I2's recorded decision must use `complete_resident_bytes()`; the evaluator must refuse `consistent: true` with `failed > 0`. |
| The boundary-page counting convention is named | **PROTOCOL** | Per-region counts can double-count a boundary page shared by two regions [Q §3]. The record carries `page_size`, `file_offset` and `len` for each region, so the evaluator can bound the overcount. It cannot remove it: which boundary pages are resident is not recorded. |
| The scan is not atomic | **PROTOCOL** (stated residual) | Each observation records its start and end bounds and its scan order (`ResidencyObservation`). A region read late may reflect a later state than one read early. |

### 2.2 Records, identity and pairing [Q §2, §4.1; #674 freeze requirements; M]

| Requirement | Status | Evidence or owner |
|---|---|---|
| Stable region identity across fresh clones | **MET** | `RegionKey = (object, SegmentIdentity, file_offset, len)`, with `Declared` being the container-relative segment path (#673). Shown equal across a real copy at another path (`exec/tests/residency_clone_identity.rs`). |
| The absolute path is diagnostic only | **MET** | Record field `segment_path`; `segment` is the declared relative path (schema `/2`). |
| A key alone never authorises comparing two containers | **MET** (docs), **EVAL** (enforcement) | The docs say so (`SegmentIdentity::Declared`, `ResidencyDelta`, `SCHEMA`). No code pairs across processes. Enforcement belongs to the evaluator [M]. |
| Pair only schema `/2` records | **EVAL** | The header carries `schema`. |
| Pair only equal, **non-null** `index_json_sha256`; a null refuses | **EVAL** | Present in every header and null when `index.json` is unreadable (`RunIdentity::for_container`). |
| Pair only equal `page_size` | **EVAL** | Header field `page_size`, null when unavailable. |
| Join region records to their header by `run_id` and `observation_id`; pair on (`store_id`, `segment`, `file_offset`, `len`) with non-null `segment` | **EVAL** | All fields present in every region record. |
| A key appearing twice in one observation refuses pairing for that key | **EVAL** | `ResidencyDelta::between` mishandles duplicates [#674 F10]. In the earlier observation it keeps the last copy. In the later one, every copy is compared against that single entry, so the change is double-counted and `excluded` is under-counted. The in-process console delta is therefore wrong when duplicates occur, though no decision reads it. The evaluator must fail closed. |
| Segment content verified against `segment_sha256` before the run | **MET** (tool, for graph containers), **PROTOCOL** (when and on what) | `larql verify` re-hashes every segment of a **graph-shaped** VINDEX3 container (`commands/extraction/verify_cmd.rs`, `verify_graph` → `inspect_container(path, true)`). A legacy-bank container gets structural checks only. `.s7` carries a `system_graph.json` [Q §2], so it is graph-shaped, but phase A must confirm the verify path it actually took. The re-hash uses plain reads, which leave the subject resident. When it may run is the problem in G1 and G4. |
| Each verification is tied to the exact container instance a run used, so verify → modify → run is detectable and refused [M] | **GAP G4** | The record names the container by path and index digest only. Neither changes when a segment is rewritten in place [#674 F2]. |
| The record names the binary that produced it | **MET** (field), **PROTOCOL** (build rule), **EVAL** (refusal) | `binary.commit` comes from `LARQL_BUILD_COMMIT`, or is `null` with `commit_source: "unavailable"` [#674 F8]. Protocol: measured binaries are built from a clean checkout with the variable set. Evaluator: refuse `unavailable`. |
| Directory and execution-graph identity of `.s7` (`index.json`, `system_graph.json` digests) | **PROTOCOL** (phase A, by `shasum`) | [Q §2]. The record carries the `index.json` digest per run; it does not carry the `system_graph.json` digest. |

### 2.3 PC2 procedure [Q §4, §4.1]

| Requirement | Status | Evidence or owner |
|---|---|---|
| A fresh-process observation A with an `after_run` record, and an observation B with a `binding` record | **MET** | `--residency-regions` writes `binding` before any pass and `after_run` after the run (`residency_curve.rs`). |
| Δ = B(binding) − A(after_run), per region, across two processes | **EVAL** | `ResidencyDelta::between` is the in-process form, and it checks no container identity by design. The cross-process Δ is computed by the evaluator under the §2.2 pairing rule. |
| Target regions fixed in advance by stable key; readahead and boundary neighbours classified | **EVAL** + **PROTOCOL** | Region records carry `file_offset` and `len`, so a byte range of a declared segment maps onto regions. The ranges and the readahead distance are fixed at the freeze (stage 2, §4). |
| Fresh, region-disjoint targets per pair, or a **witnessed** reset | **PROTOCOL** + **EVAL** (disjointness check) | The eligibility observation is an ordinary fresh-process record (**MET**). |
| **A source of non-resident targets at all** | **GAP G1** | Phase A's `larql verify` leaves `.s7` resident before phase B. See G1. |
| Joint eligibility, decided before either arm and narrowed only jointly | **PROTOCOL** + **EVAL** | Wording correction N-a / N-b in §5. |
| Sham arm first in each pair; the order residual stated | **PROTOCOL** | N-a wording, §5. |
| Intervention: a known byte range read by another process, with success recorded (exit status, bytes read = bytes requested); the sham is the same invocation with a zero count | **GAP G2** | No instrument in I1–I6 runs or records the intervention. |
| Two windows recorded separately: read completion → B binding, and A after_run → B binding (N-c) | **GAP G2** (read completion) + **G5** (clock) | Records carry wall and monotonic bounds. The monotonic origin is per process, so cross-process windows can only use wall time (G5). |
| A→B timing matched between arms within a tolerance | **EVAL** + **PROTOCOL** (tolerance, stage 2) | Computable from wall bounds (G5). |
| Pass criteria: target increase under treatment against sham; non-targets compared on increases only | **EVAL** + **PROTOCOL** (criteria, stage 2) | — |
| Declared infeasible, never relaxed, if the observer's run destroys the contrast | **PROTOCOL** | [Q §10]. Reached by G1 if no target source is admitted. |

### 2.4 Other controls, methods and machine state

| Requirement | Status | Evidence or owner |
|---|---|---|
| PC1: cross-process visibility | **MET** (witness), **G2** (the reading process and its record) | Needs the same intervention harness as PC2. |
| PC3: faults against resident delta (diagnostic only) | **MET as console output only**, with caveats | Faults now come from `bracket()` around the step alone (#673), and are process-wide. They appear on the console, **not in the JSONL record**, so an evaluator cannot read them today. F5 (unavailable reads as 0) applies on non-unix targets and wherever `getrusage` fails, macOS included. PC3 is not a gate [Q §4]. |
| PC4: clone isolation | **I4** | — |
| Clone identity per attempt (`clonefile`, extent check, index and graph digests) | **I4** | [Q §2]. Semantics confirmed in I4's review, not assumed. |
| Displacer extent exclusion; displacement | **I4** (check), **I6** (script) | — |
| Machine-state snapshots, quiet gate, peer-process check | **MET as console output only** (quiet gate: `cpu::Environment`), **I5** (`vm_stat`, `iostat`, indexer CPU, process list, all recorded) | The quiet-gate reading is printed, not written to the JSONL record. Recording it is I5's or I2's. |
| Spotlight exclusion, verified | **PROTOCOL** (phase C) | [Q §6]. |
| Gate decision written to the record **before** the timed region | **I2** | [Q §7]. |
| Rejected attempts still executed, timed, latency sealed | **I2** (sealing in the record) + **PROTOCOL** (access rules) | [Q §7, §12.7]. |
| Counterbalanced F/D order, fixed attempt counts, no optional stopping | **PROTOCOL** | [Q §7]. |
| Independent evaluator, named, with the records it scores from | **EVAL** + **PROTOCOL** | [Q §10, §12.8]. |
| The cold arm is pass 1 of a fresh process, `--warmup 0 --repeat 1` | **MET** (flags exist), **PROTOCOL** (their use), **I2** (recorded in the attempt record) | [Q §1]. |
| Model bytes are never written; clones are deleted after each attempt | **PROTOCOL** | [Q §2]. G4's metadata check detects a write to a verified segment. |
| Settling: wait for snapshots within tolerance and the quiet gate, or time out; a timeout is a recorded settle failure | **I5** (snapshots) + **PROTOCOL** (procedure, stage 1; tolerance and timeout, stage 2) | [Q §6]. |
| Unobservable residuals stated with every result | **PROTOCOL** | [Q §6]: device-internal caching, SSD thermal state, GPU contention, refill after settling. |
| One admission threshold for both methods, never tuned per method | **PROTOCOL** | [Q §7]. |
| Every attempt reported, admitted or rejected, with gate decision, settle outcome and snapshots | **I2** + **I5** + **EVAL** | [Q §7]. |
| PC2 rejects an incomplete observation in any of its records, and voids a pair whose intervention read failed | **EVAL** + **G2** | #671 §5 step 5 (no report has a failed region) and #671 §5's general rule ("a control whose read failed is void, not failed"). |

### 2.5 #674 follow-ups

| Item | Classification | Needed before |
|---|---|---|
| F3: end-to-end sampling-boundary test | instrument test | any **scored** use of fault counts (§11 of [Q]). Not cold-binding qualification, which is decided on residency and disturbance [M]. |
| F5: unavailable faults must not read as 0 | instrument | the same. On macOS `getrusage` reports; the fallback zero affects non-unix and a failed call. |
| F6: no `complete` field | **I2** + **EVAL** | qualification: admission must use completeness. |
| F10: duplicate keys merged | **EVAL** | any cross-process pairing (PC2). |
| F2: open never verifies segment hashes | **PROTOCOL** → **G4** | the freeze. |
| F8: commit declared, not checked | **PROTOCOL** + **EVAL** | qualification runs. |
| F4, F7, F9; `/1` help text (`args.rs`); test wording | housekeeping | none. Can ride with I2 or a docs PR. |

## 3. Gaps the reconciliation found

None of these is assigned to I1–I6 today. Each needs a maintainer decision
before stage 1 of the freeze (§4).

**G1. PC2 has no source of cold targets after phase A.** Phase A verifies
`.s7` by re-hashing every extent, which "leaves the subject resident"
[Q §2]. Phase B (PC2) follows, and nothing between them makes any region
non-resident. With the 94 GB bank [Q §5] read through the page cache on a 128 GiB machine,
the eligibility observation would **likely** find targets near fully
resident and void the pairs. Whether pages stay resident depends on memory
pressure, so this is a likely outcome, not a certainty. Candidate sources, each witnessed by the eligibility
observation:

- **(a) One witnessed reset after phase A, then region-disjoint targets.**
  For example `purge` (method E, untested, **needs root** [D §2]). PC2
  tests the witness, not a cold method, so the reset needs only to be
  witnessed, not qualified. The observers' own runs touch only the experts
  they route to, and PC2's targets are drawn from regions the eligibility
  observation shows cold. It needs no I4 or I6.
  - **Residual:** with one reset and the sham arm always first, the
    system-wide refill after the reset is a **time trend confounded with
    arm order**, not a disturbance shared equally by both arms. The protocol
    must state it, and record the refill (I5 snapshots, or at least
    `vm_stat`) across the pairs.
- **(a′) A hash pass with `F_NOCACHE` reads in place of phase A's cached
  re-hash.** It would verify the payload without populating the page cache
  if `F_NOCACHE` behaves as documented. It is untested: whether it leaves
  no residue must itself be witnessed with mincore. It may also serve G4.
- **(b) A fresh APFS clone for PC2.** It needs I4 and PC4 first, which
  inverts the order the maintainer set (PC2 feasibility before I4–I6).
- **(c) Displacement.** It needs I6.
- **(d) Run PC2 before phase A's hashing.** This breaks the identity-first
  order, and so is not recommended.

**Recommendation: (a)**, with the reset named as a PC2-only control
mechanism, never as a qualified cold method. **Decision needed.**

**G2. No instrument runs or records the intervention.** PC1 and PC2 need
another process to read a known byte range of a declared segment and record
exit status, bytes requested and read, segment, offset, block size, and
start and completion wall times. The sham records the same with a zero
count. Today this is an unwritten shell step [Q §4]. **Proposal:** a small
committed intervention script with its own record, reviewed like I4. It is
the "PC harness", needed for PC2 feasibility, so it comes before I4–I6.

**G3. The evaluator is not an instrument in §9.** Every **EVAL** row above
is code someone must write: pairing under the #674 rule, fail-closed on
duplicates and incomplete records, cross-process Δ, target classification,
window matching, and the pass criteria. **Proposal:** the evaluator is
specified, reviewed and named in stage 1 of the freeze. It is written by
someone other than the author of the instruments, and it reads only the
committed record schemas.

**G4. Verification is not bound to the instance a run used.** The record
identifies a container by path and `index.json` digest. A segment rewritten
in place keeps both. `larql verify` reads every extent, so it cannot be run
on a cold instance just before binding without destroying the cold state.
**Proposal:**

- **Verify the source once, in phase A, then bind each run to it by
  metadata that any write would change, without reading payload:**
  - **at verification:** each segment's (inode, size, mtime, ctime), plus
    the volume's identity, written to a **verification record** next to the
    verify output. Nothing writes this today, so it needs an owner (below);
  - **at each run:** the same fields read with `fstat` on the files the
    run **actually maps**, not `stat` by path, so a file replaced by
    rename is caught. They are compared before binding and after the run.
    **The current code cannot provide this:** `PhysicalStore::map_segment`
    drops the `File` right after mapping (`represent/physical.rs`), so no
    descriptor is left to `fstat`. It needs a larql-vindex change, for
    example `fstat` at map time, kept on the store. That is wider than the
    I2 record;
  - ctime cannot be set programmatically (`setattrlist(2)`: change time
    "cannot be set programmatically"), so a write after verification can
    escape detection only by changing the system clock, which needs root;
  - a mismatch voids the attempt.
  - **Writable mappings:** a write through a shared writable mapping may
    update ctime late, so the protocol forbids any writable mapper of the
    subject during the campaign.
  - **Expected false voids, to be named:** an extended-attribute change
    bumps ctime, and `st_dev` can change across a reboot. That is why the
    volume is identified by its UUID rather than `st_dev`. `fstat` does not
    return the UUID; it needs `getattrlist` (`ATTR_VOL_UUID`) or an
    equivalent.
- **For method F, every clone segment is bound the same way.** #670's
  extent check samples offsets, so it cannot by itself show that the whole
  clone still matches the source.
  - At clone creation, after the I4 identity check, each clone segment's
    (inode, size, mtime, ctime) is recorded.
  - It is re-checked by `fstat` on the mapped files before binding and after
    the run.
  - A clone written to after its check is therefore caught even where the
    extent sample would miss it.
  - The run record names the clone and the I4 report it was checked by.
- **Re-verify the source after the campaign** with `larql verify`, where
  reading payload no longer matters. A failure there voids the campaign.
- **Alternative:** an `F_NOCACHE` re-hash of the exact instance before each
  attempt (G1 (a′)), if it is witnessed to leave no residue. It binds
  content to the instance directly, at the cost of one full read per
  attempt and unobservable device-side caching.
- **Owners:**
  - the verification record and the clone-creation record belong to the
    phase-A tooling and I4;
  - the per-run `fstat` metadata belongs in the attempt record I2 writes.
  - That widens I2's scope, which is the maintainer's decision.

**G5. Cross-process windows have only wall time.** `started_mono_ns` and
`ended_mono_ns` are measured from each process's own first observation, so
they cannot be compared across processes. The A→B and read-completion→B
windows [Q §4.1] must use wall time, and wall time can step under clock
adjustment. **Proposal:** the protocol states that cross-process windows are
wall-clock, and records whether time synchronisation adjusted the clock
during the campaign. Alternatively, a run-independent monotonic source
(`mach_continuous_time`) can be added to the record in I2.

## 4. Two-stage freeze [M]

The maintainer's adjustment: settle the complete procedure and the
evaluator first; set the numbers in a separately declared calibration
phase; **never tune them in qualification**. #670 already isolates
calibration as phase C [Q §8]. What changes is that the freeze happens in
two stages.

| Stage | Settles | From [Q §12] and §3 |
|---|---|---|
| **Stage 1: procedure freeze** | every phase's procedure; the PC2 target source (G1); the intervention harness (G2); the evaluator's specification and name (G3); instance binding (G4); the window clock (G5); the boundary-counting convention; the abort conditions; the sealed-latency access rules; the calibration population's definition, **size**, F/D mix and separation from qualification, which cannot be chosen from calibration itself; the Spotlight mechanism to be verified | §12 items 2 (all of it), 6, 7, 8; G1–G5; N-a, N-b, N-c |
| **Calibration (phase C)** | unscored. Its records feed stage 2 only and are never qualification attempts | — |
| **Stage 2: numbers freeze** | admission margin, settling tolerance and timeout, disturbance bounds; PC2 parameters (A→B tolerance, eligible-target count, near-fully-resident cut, readahead distance, replicate count); N_F, N_D and block order; qualification thresholds; the displacer set | §12 items 1, 3, 4, 5; the PC2 parameters §12 omits (noted in the 2026-11-01 PC2 re-check, which was not preserved in the repo) |
| **Phase D** | qualification | — |

**A PC2 feasibility pilot** could run after stage 1, as an unscored
rehearsal. It asks only whether any eligible contrast survives the
observers' runs.
- It is **not** phase B's PC2. PC2 itself stays a phase-B gate under
  stage-2 numbers, so the phase order A → B → C → D is unchanged.
- The pilot's records are calibration-class: they inform stage 2 and are
  never PC2 evidence.
- It needs G1's source and G2's harness, and I2 if admission is recorded.
  It does not need I4–I6.
- **Not authorised by this document.**

## 5. Corrections to apply to #670 at stage 1

The candidate is not edited here. These are listed so that the stage-1
revision applies them.

1. **Stale facts after #672 / #673:**
   - [Q §3] still says `resident_bytes()` returns `Option` and the summing
     caller maps `None` to 0, and that the protocol cannot be frozen until
     I1 is merged. I1 is merged; the witness now keeps failures.
   - [Q §4] PC2 row: "the `None → 0` behaviour still present on main" is
     no longer true of the witness. The other `None → 0` sites listed in the
     I1 + I3 proposal §2.3 are footprint paths, not the witness.
   - [Q §9] marks I1 and I3 as unbuilt.
2. **N-a.** [Q §4.1] "The fixed order is a declared residual ... bounded by
   the matched A→B timing" overstates the bound. Replace with: matched A→B
   timing equalises the windows. It does not bound drift between the arms,
   nor the extra observer runs before the treatment baseline. The after-run
   baselines and the joint exclusion absorb the observer runs' warming of
   the targets. They do **not** absorb drift in machine state between the
   sham slot and the treatment slot, which stays a stated residual with
   every PC2 result. (The wording traces to the 2026-11-01 re-check of the
   PC2 amendment, which was **not preserved in the repo**, so it cannot be
   independently verified. The labels N-a, N-b and N-c are this
   document's.)
3. **N-b.** Rename the joint-eligibility heading: "eligible set decided
   before either arm; narrowed only jointly, from either arm's baseline".
4. **N-c** (point 5 of the PC2 amendment audit,
   `docs/audits/mem-002-cold-state-qualification/`). Record "read completion → B binding" (the #671 §5 window) and
   "A after_run → B binding" (the [Q §4.1] window) as separate quantities.
   For the sham, read completion is the zero-count invocation's completion.
5. **§12 completeness.** List the PC2 parameters, and assign every item to
   stage 1 or stage 2 (§4).
6. **Pairing rule and instance binding.** Adopt the #674 cross-container
   pairing requirements and G4's instance binding as [Q §2] requirements.
7. **Amend [Q §10]'s freeze conditions for two stages.** §10 requires I1–I6
   merged, clone semantics confirmed in I4's review, and calibration done,
   all before "the" freeze. Under §4:
   - stage 1 requires I1–I3 and the G2 harness **merged after their own
     reviews**, the G4 map-time metadata change if adopted, and the
     evaluator's specification;
   - stage 2 requires I4–I6 merged after review, clone semantics confirmed,
     and calibration complete with enough stable observations for every
     number.

## 6. What this authorises, and what it does not

**Proposed next, in order, each needing its own authorisation:**

1. Maintainer decisions on G1 (target source), G2 (harness), G3
   (evaluator ownership), G4 (whether instance metadata goes into I2) and G5
   (clock).
2. **An I2 implementation proposal**, reviewed before any code: the
   admission decision recorded before the timer, from
   `complete_resident_bytes()` (#674 F6), plus G4's instance metadata and
   G5's clock if decided so.
3. **The I2 implementation**, merged after its own code review and CI.
4. **The G2 intervention harness**, and G4's map-time metadata change if
   adopted: each proposed, reviewed, implemented and merged separately.
5. Stage 1 of the freeze, with the §5 corrections. It requires steps 3
   and 4 (§5 item 7).

**Not authorised:** I4–I6 (held until PC2 feasibility is shown [M]), the
PC2 feasibility pilot, calibration, qualification attempts, and any latency
analysis.
