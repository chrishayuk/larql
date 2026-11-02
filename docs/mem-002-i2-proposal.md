# MEM-002 I2: admission recording and run provenance — implementation proposal

**Class: DRAFT PROPOSAL — NOT AUTHORISED FOR IMPLEMENTATION.** Dated
2026-11-02 at main `bbf27efe`. Nothing is built here.

It implements the maintainer's decisions on G1–G5 of the reconciliation
([`mem-002-protocol-reconciliation.md`](mem-002-protocol-reconciliation.md),
#675, cited **[RC §n]**) for the protocol candidate
([`mem-002-cold-state-qualification.md`](mem-002-cold-state-qualification.md),
#670, **[Q §n]**). That candidate stays unfrozen. The maintainer's
2026-11-02 decisions on G1–G5 are cited **[M]**.

## 1. Scope

I2 is split in two, so that the filesystem work stays separately
reviewable, as the maintainer asked [RC §6].

| Part | Purpose | Crates |
|---|---|---|
| **I2a: admission record** | the gate decision, written before any timed region, from a complete observation; a system-wide monotonic timestamp | larql-cli (residency curve and record) |
| **I2b: instance identity** | the identity of every file the run actually mapped, captured from the opened file, and compared with a verification manifest | larql-vindex (map time), larql-cli (record) |

**Out of scope:**
- the threshold value, which is a stage-2 number;
- the verification manifest's producer, which is phase-A tooling with its
  own review (§4.3);
- the intervention harness, which has its own proposal;
- the evaluator;
- I4–I6, the PC2 pilot, calibration and qualification.

## 2. Where the decision sits today

In `run_residency_curve` (`vindex3_cmd/generate/residency_curve.rs`), in
order:
- the quiet-gate reading;
- `PreparedOperands::load_within`;
- the binding observation (`ops.observe_residency()`);
- the `binding` record (`--residency-regions`);
- the console binding lines;
- then the pass loop, whose first timed region is pass 1's prompt
  `bracket`.

Nothing decides admission. `complete_resident_bytes()` exists, but nothing
outside tests calls it [RC §2.1].

## 3. I2a: the admission record

### 3.1 The decision

After the binding observation and **before the pass loop**, the curve
computes one gate decision.

| Outcome | Condition |
|---|---|
| `witness_failed` | `complete_resident_bytes()` is `None`: any failed region, no regions, or inconsistent counts. #674 F6: completeness, not mere consistency |
| `above_threshold` | complete, and the resident total exceeds the declared threshold |
| `admitted` | complete, and at or below the declared threshold |
| `witness_only` | complete, and **no threshold was declared**. Calibration attempts and development runs are this |

- The decision is computed from the binding observation alone.
- It is never revised afterwards.
- **It does not stop the run.** Rejected attempts are still executed and
  timed [Q §7].

### 3.2 The threshold input

- A new flag, `--admission-threshold-bytes <n>`, with `--admission-basis
  <text>` naming where the number came from (a frozen protocol's section
  and commit).
- Both are recorded verbatim.
- The flag sets nothing by default. Without it, the outcome is
  `witness_only`, never `admitted`.
- The counting convention is the existing one: the per-region sum over read
  regions, boundary pages possibly double-counted [Q §3, RC §2.1].
  - It is named in the record as `"per_region_sum"`.
  - The record also carries the overcount bound the evaluator can compute
    from page size and region adjacency: the number of region boundaries
    that share a page, times `page_size`.

### 3.3 Written before the timer

- **Record placement.** The decision goes out as a new record type in the
  `--residency-regions` stream, `"record": "admission"`. It is written,
  **and the stream flushed**, after the `binding` observation records and
  before pass 1 begins.
- **Console line.** It is also printed:
  `admission: <outcome> (complete <bool>, resident <GB> GB over <n> regions, threshold <GB|none>)`.
- **No stream, no admission.** Without `--residency-regions`, the curve
  refuses `--admission-threshold-bytes`. An admission that is not written
  before the timer does not count [Q §7].

### 3.4 A system-wide monotonic timestamp (G5)

Every record header, the observation and the new admission record, gains a
`clock` block:

```
"clock": {
  "system_mono_ns": <u64 | null>,
  "basis": "CLOCK_MONOTONIC_RAW" | null,
  "boot_session": "<id>" | null
}
```

- **Read with `libc::clock_gettime(CLOCK_MONOTONIC_RAW)`** at the same
  points as the existing bounds: `started_system_mono_ns` and
  `ended_system_mono_ns` for an observation, and one reading for an
  admission record.
- **The basis is named in every record, so a reader never infers it.**
  Its exact behaviour on macOS (whether it advances during sleep, and its
  relation to `mach_continuous_time`) is **to be confirmed in the
  implementation review**, not assumed here.
- **`boot_session`** identifies the boot, so values from different boots
  are never compared. On macOS it comes from `sysctl kern.bootsessionuuid`;
  elsewhere it is `null` unless an equivalent is confirmed.
- **The existing `started_mono_ns` and `ended_mono_ns` stay** for
  compatibility. They are documented as process-relative and **not
  comparable across processes**; the evaluator must not compare them [M,
  G5].
- **Non-unix:** the clock reads `null`, never 0.

The harness proposal records the same clock, so interventions and
observations can be ordered on one basis. Ordering is not causality.

### 3.5 Other attempt facts in the admission record

- `--warmup` and `--repeat`. The cold arm is `--warmup 0 --repeat 1`
  [Q §1].
- The quiet-gate reading and disqualifiers. Today they are console only
  [RC §2.4].
- `run_id`, `binary.commit` and `commit_source`, and the container block,
  as in the observation header.

### 3.6 Schema

Adding a record type and a header block changes what a reader must
understand, so the schema becomes **`larql.residency-observation/3`**. A
reader refuses an unknown version.

### 3.7 Tests

On constructed observations, as for I1 + I3:
- each outcome from each condition, including inconsistent counts →
  `witness_failed`, and no threshold → `witness_only`;
- the threshold boundary (equal is `admitted`);
- the admission record's fields and placement: after the binding
  observation, and before any pass output in the stream;
- the refusal of a threshold without a stream;
- the clock block present and non-null on unix, and `null` elsewhere;
- the process-relative fields still present.

A CLI smoke test with and without a threshold, with no residency claims
made for its dense fixture.

## 4. I2b: instance identity (G4)

The maintainer's position: **file metadata detects change; it is not proof
of unchanged bytes.** I2b therefore records identity and compares it.
Whether the bytes are right is established only by the manifest's
verification before the campaign and a payload re-verification after it
[RC §3 G4].

### 4.1 Identity captured from the opened file

`PhysicalStore::map_segment` opens the segment, maps it, and drops the
`File` (`represent/physical.rs`). I2b changes it to:
- `fstat` the open descriptor **before mapping**;
- keep the result on the store as a `FileIdentity`: device, inode, size,
  mtime and ctime (seconds and nanoseconds);
- on macOS, also record the volume UUID from `getattrlist` (`ATTR_VOL_UUID`)
  on the same descriptor.

This is the association between identity and mapping [M]: the identity is
of the file that was mapped, not of a path looked up later.

### 4.2 Recorded and compared

- **Before the timer:** the admission record lists each mapped segment as
  (declared segment, `FileIdentity`), captured at map time.
- **After the run:** the `after_run` record repeats the list, read again by
  `fstat` on a descriptor kept open for the run. That is one descriptor
  per mapped segment; the count for `.s7` is confirmed in review.
- **With `--verification-manifest <path>`:** the curve compares each mapped
  segment's identity with the manifest's entry for that declared segment.
  - The record states the result per segment: `match`, `mismatch`, or
    `absent`.
  - It also states one overall `identity: verified | mismatch | unverified`.
  - Any mismatch makes the attempt invalid. I2b records the fact; the
    evaluator voids the attempt.
- **Without a manifest:** `identity: unverified`. The evaluator must refuse
  it for any scored or paired use [M, G3].

### 4.3 The verification manifest (format only)

Phase-A tooling produces it. Its implementation is a separate reviewed
item, and is **not part of I2**. The format I2b reads:

```
{"schema":"larql.verification-manifest/1",
 "container":{"path":…,"index_json_sha256":…,"volume_uuid":…},
 "verified_wall":…,"verified_system_mono_ns":…,"boot_session":…,
 "verifier":{"command":"larql verify","binary":{"version":…,"commit":…},"path_taken":"graph"},
 "segments":[{"declared":…,"segment_sha256":…,"identity":{dev,inode,size,mtime_ns,ctime_ns}},…]}
```

- **`path_taken`** records that the graph re-hash path ran. A legacy-bank
  container gets structural checks only [RC §2.2].
- **For APFS clones (method F):** a clone's manifest entry is derived from
  the verified source plus the I4 shared-extent report.
  - The shared-extent check is **additional** to the source relationship,
    never a substitute for it [M].
  - The clone's own identity is recorded at creation.

### 4.4 Stated limits

Recorded with every result:
- identity can be restored or forged under unusual conditions;
- a file replaced and restored between two checks is not detected;
- an extended-attribute change bumps ctime, a false void;
- `st_dev` can change across a reboot, which is why the volume UUID is
  recorded.

Only the post-campaign payload re-verification addresses content.

### 4.5 Tests

- A mapped store reports the `FileIdentity` of the file it opened.
- A file rewritten after mapping shows a changed ctime on the kept
  descriptor.
- A file replaced by rename keeps the old identity on the descriptor and
  shows a different identity by path.
- Manifest comparison yields `match`, `mismatch` and `absent`.
- No manifest gives `unverified`.
- Non-unix gives identity `null` and `unverified`.

## 5. Review checklist

1. Is `witness_only` the right outcome when no threshold is declared, rather
   than refusing to run?
2. Is a schema bump to `/3` acceptable for both a new record type and a new
   header block?
3. Is `CLOCK_MONOTONIC_RAW` the right basis, and is
   `kern.bootsessionuuid` the right boot identity?
4. Is holding one descriptor per mapped segment for the run acceptable?
5. Should I2b land as its own PR after I2a, as proposed, or with it?

## 6. Authorisation

This document authorises nothing. Implementation of I2a and I2b each needs
separate approval after this proposal is reviewed. The manifest producer,
the harness, the evaluator, I4–I6, the PC2 pilot, calibration, the freeze
and qualification remain unauthorised.
