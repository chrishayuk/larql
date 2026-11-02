# MEM-002 I2: admission recording and run provenance — implementation proposal

**Class: DRAFT PROPOSAL — NOT AUTHORISED FOR IMPLEMENTATION.** Dated
2026-11-02 at main `bbf27efe`. Nothing is built here. Revised after an
independent review the same day.

It implements the maintainer's decisions on G1–G5 of the reconciliation
([`mem-002-protocol-reconciliation.md`](mem-002-protocol-reconciliation.md),
#675, cited **[RC §n]**) for the protocol candidate
([`mem-002-cold-state-qualification.md`](mem-002-cold-state-qualification.md),
#670, **[Q §n]**). That candidate stays unfrozen. The maintainer's
2026-11-02 decisions on G1–G5 are cited **[M]**.

## 1. Scope

I2 is split in two, so that the filesystem work stays separately
reviewable [M].

| Part | Purpose | Crates |
|---|---|---|
| **I2a: admission record** | the gate decision, written before the run's timed passes, from a complete observation; a system-wide monotonic timestamp | larql-cli (residency curve and record); larql-vindex (observation bounds) |
| **I2b: instance identity** | the identity of every file the run actually mapped, captured from the descriptor it was mapped from, re-checked immediately before the passes and after them | larql-vindex (map time), larql-cli (record) |

**Out of scope:**
- the threshold value, which is a stage-2 number;
- the verification manifest's producer, a separate reviewed item, which
  owns pre- and post-campaign verification (§4.4);
- the intervention harness, which has its own proposal;
- the evaluator: I2 records facts and the evaluator judges them [RC §3 G3];
- I4–I6, the PC2 pilot, calibration and qualification.

## 2. Where the decision sits today

In `run_residency_curve` (`vindex3_cmd/generate/residency_curve.rs`), in
order:
1. the quiet-gate reading;
2. **the binding timer**: `PreparedOperands::load_within` runs under an
   `Instant`, and `weights bound in … s` is printed;
3. the binding observation (`ops.observe_residency()`);
4. the `binding` record (`--residency-regions`), which is not flushed until
   after the run;
5. the console binding lines;
6. the pass loop, whose first timed region is pass 1's prompt `bracket`.

Nothing decides admission. `complete_resident_bytes()` exists, but nothing
outside tests calls it [RC §2.1].

**The binding time is measured before any witness can exist.** The witness
observes the bound image, so no decision can precede the timing of the
binding itself.
- I2 therefore places the decision **before every timed region after
  binding**: all passes, prompt and decode.
- **Bind time is sealed latency.** It is recorded and printed as today,
  carries the same sealing as every other latency [Q §7], and plays no part
  in admission.
- The protocol's unit is cold *binding* as witnessed at binding [Q §1], not
  bind time. Analysing bind time needs its own later freeze [Q §11].
- Review question 1 asks whether the console print of bind time should be
  suppressed under sealing.

## 3. I2a: the admission record

### 3.1 The decision

After the binding observation, and after I2b's pre-pass identity check
(§4.2), and **before the pass loop**, the curve computes one gate decision.

| Outcome | Condition |
|---|---|
| `witness_failed` | `complete_resident_bytes()` is `None`: any failed region, no regions, or inconsistent counts. #674 F6: completeness, not mere consistency |
| `above_threshold` | complete, and the resident total exceeds the declared threshold |
| `admitted` | complete, and at or below the declared threshold |
| `witness_only` | complete, and **no threshold was declared**. Calibration attempts and development runs are this. A fourth outcome that [Q §7] must adopt at stage 1 |

- The decision is computed from the binding observation alone.
- It is never revised afterwards.
- **It does not stop the run.** Rejected attempts are still executed and
  timed [Q §7].
- **Identity is recorded beside the decision, not folded into it.** I2b's
  identity facts go in the same admission record. The evaluator voids an
  attempt on an identity mismatch or an unverified identity [M, G3]; I2
  does not.

### 3.2 The threshold input

- A new flag, `--admission-threshold-bytes <n>`, with `--admission-basis
  <text>` naming where the number came from (a frozen protocol's section
  and commit).
- Both are recorded verbatim.
- The flag sets nothing by default. Without it, the outcome is
  `witness_only`, never `admitted`.
- The counting convention is the existing per-region sum over read regions
  [Q §3], named in the record as `"per_region_sum"`. The record carries each
  region's offset and length and the page size, so the evaluator computes
  any boundary-overlap bound itself.

### 3.3 Written before the passes

- **Record placement.** The decision goes out as a new record type in the
  `--residency-regions` stream, `"record": "admission"`. The stream,
  including the `binding` observation records, is **flushed** after the
  admission record and before pass 1 begins. Today nothing is flushed until
  after the run.
- **Console line.** It is also printed:
  `admission: <outcome> (complete <bool>, resident <GB> GB over <n> regions, threshold <GB|none>)`.
- **No stream, no admission.** Without `--residency-regions`, the curve
  refuses `--admission-threshold-bytes`. An admission that is not written
  before the passes does not count [Q §7].

### 3.4 A system-wide monotonic timestamp (G5)

**Field names, shared with the harness proposal:**

```
"clock": {
  "basis": "CLOCK_MONOTONIC_RAW" | null,
  "boot_session": "<id>" | null,
  "start_system_mono_ns": <u64 | null>,   // observation records
  "end_system_mono_ns":   <u64 | null>,   // observation records
  "system_mono_ns":       <u64 | null>    // admission record: one reading
}
```

- **Read with `clock_gettime(CLOCK_MONOTONIC_RAW)`, in nanoseconds.**
  - For an observation, it is read at the same points as the existing
    `started` and `ended` bounds, inside `ResidencyObservation::capture`.
    That is a larql-vindex change.
  - For the admission record, it is read once, when the decision is made.
- **Semantics by OS:**
  - **macOS:** advances during sleep. Checked on this machine by the
    reviewer: it equals `mach_continuous_time` to within 41 ns, and differs
    from `mach_absolute_time` by the time slept since boot.
  - **Linux:** `CLOCK_MONOTONIC_RAW` stops during suspend.
  - The basis is recorded in every record, so a reader never infers it.
  - MEM-002 runs on macOS.
- **`boot_session`.** On macOS it is `sysctl kern.bootsessionuuid`, which
  exists here. `kern.bootuuid` is not a boot identity: it equals the Data
  volume UUID. On other OSes it is `null` unless an equivalent is confirmed.
  Values from different boot sessions are never compared.
- **The existing `started_mono_ns` and `ended_mono_ns` stay** for
  compatibility. They are documented as process-relative and **not
  comparable across processes**, and the evaluator must not compare them
  [M, G5].
- **Non-unix:** clock fields are `null`, never 0.
- Ordering on this clock is not causality.

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
reader refuses an unknown version. The `--residency-regions` help text,
which still names `/1` (`vindex3_cmd/args.rs`), is corrected in the same
change.

### 3.7 Tests

On constructed observations, as for I1 + I3:
- each outcome from each condition, including inconsistent counts →
  `witness_failed`, and no threshold → `witness_only`;
- the threshold boundary (equal is `admitted`);
- the admission record's fields and placement: after the binding
  observation, before any pass output, and the stream flushed at that
  point;
- the refusal of a threshold without a stream;
- clock fields present and non-null on unix, and `null` elsewhere;
- the process-relative fields still present.

A CLI smoke test with and without a threshold, with no residency claims
made for its dense fixture.

## 4. I2b: instance identity (G4)

The maintainer's position: **file metadata detects change; it is not proof
of unchanged bytes** [M]. I2b records identity and compares it. Whether the
bytes are right is established only by verification before the campaign
and payload re-verification after it, both owned by the manifest producer
(§4.4).

### 4.1 One descriptor per segment, from open to the end of the run

**Today a segment is opened several times by path.**
- `OperandStore::open` reads each segment's header by path
  (`operands/open.rs`).
- `PhysicalStore::map_segment` reads the header again by path
  (`physical.rs` → `encode/segment.rs`).
- It then opens the file again, maps it, and drops the `File`.
- `map_compiled` also drops its `File`.
- A file swapped between those opens would go undetected.

**I2b changes the store to:**
- open each segment **once**;
- `fstat` that descriptor;
- read the header **from that descriptor**, and map **from that
  descriptor**;
- keep the descriptor open on the store for the store's lifetime.

`map_compiled` is changed the same way.

- The store holds a `FileIdentity`: volume UUID, inode, size, mtime_ns and
  ctime_ns. Device id is kept as a diagnostic only.
- On macOS, the volume UUID comes from `fgetattrlist` (`ATTR_VOL_UUID`) on
  the same descriptor. The reviewer confirmed it works on a regular-file
  descriptor here, although the man page says it should return `EINVAL`.
  The implementation pins this with a test, and falls back to the volume's
  UUID by path lookup, recording which source was used.

The header check `OperandStore::open` makes before mapping stays as a
structural check. **Identity is taken only from the mapped descriptor.**

`.s7` has five segment files, and the default soft descriptor limit is 256.

### 4.2 Checked immediately before the passes

At the admission point, after the binding observation and before the pass
loop, the curve re-`fstat`s every kept descriptor and records, per mapped
segment, in the admission record:
- the declared segment;
- the identity captured at map time;
- the identity at admission.

**With `--verification-manifest <path>`:**
- the record also carries the manifest's SHA-256;
- it carries each segment's comparison with the manifest entry for that
  declared segment: `match`, `mismatch` or `absent`. The comparison key is
  (volume UUID, inode, size, mtime_ns, ctime_ns), never device id, which can
  change across a reboot;
- an overall `identity_diagnostic` is recorded **as a diagnostic only**. The
  evaluator recomputes it from the per-segment facts [M, G3].

**Without a manifest,** the per-segment facts are still recorded and the
diagnostic is `unverified`. The evaluator refuses unverified identity for
any scored or paired use [M].

The `after_run` record repeats the per-segment identity from the kept
descriptors, so a change during the run is visible.

### 4.3 Writable mappings

A write through a shared writable mapping may update ctime late. The
protocol forbids any writable mapper of the subject during the campaign
[RC §3 G4]. I2b cannot enforce that, and says so in every record's
documentation.

### 4.4 The verification manifest (format; its producer is separate)

The phase-A producer is a separate reviewed item, and **not part of I2**. It
writes a manifest before the campaign and again after it.

```
{"schema":"larql.verification-manifest/1","phase":"pre"|"post",
 "container":{"path":…,"index_json_sha256":…,"volume_uuid":…},
 "verified_wall":…,"clock":{"basis":…,"boot_session":…,"system_mono_ns":…},
 "verifier":{"command":…,"binary":{"version":…,"commit":…},"path_taken":"graph"},
 "segments":[{"declared":…,"segment_sha256_declared":…,"segment_sha256_computed":…,
              "identity":{volume_uuid,inode,size,mtime_ns,ctime_ns,dev}},…]}
```

- **Identity comes from the descriptor that was hashed.** The producer opens
  each segment once, `fstat`s it, hashes it **through that descriptor**, and
  records the identity of that descriptor. Otherwise a file replaced after
  hashing could pass.
- **`path_taken`** records that the graph re-hash path ran. A legacy-bank
  container gets structural checks only [RC §2.2].
- **Post-campaign re-verification** is the same producer with
  `phase: "post"`. The evaluator requires:
  - every segment's computed hash equal to its declared hash in both
    manifests;
  - every segment's identity in the post manifest equal to the pre manifest
    and to each run's records.
  - A failure voids the campaign. That is the evaluator's rule, not I2's.
- **APFS clones (method F):** their identity relationship to the verified
  source is I4's to specify. The shared-extent check is additional to that
  relationship, never a substitute for it [M].

### 4.5 Stated limits

Recorded with every result:
- identity can be restored or forged under unusual conditions;
- a file replaced and restored between two checks is not detected;
- an extended-attribute change bumps ctime, a false void. `.s7`'s files
  already carry a `com.apple.provenance` xattr, and their ctime is later
  than their mtime;
- writable mappers are forbidden but not enforced.

Only payload verification addresses content.

### 4.6 Tests

- A mapped store reports the `FileIdentity` of the descriptor it read the
  header from and mapped.
- A file rewritten after mapping shows a changed ctime on the kept
  descriptor at admission and after the run.
- A file replaced by rename after mapping keeps the old identity on the
  descriptor, while the path names a different inode.
- Manifest comparison yields `match`, `mismatch` and `absent`, and ignores
  device id.
- No manifest gives `unverified`, with the facts still recorded.
- The `fgetattrlist` volume UUID agrees with the by-path fallback.
- Non-unix gives identity `null` and `unverified`.

## 5. Review checklist

1. Should the console print of bind time be suppressed under latency
   sealing, or kept as today?
2. Is `witness_only` the right outcome when no threshold is declared, rather
   than refusing to run?
3. Is a schema bump to `/3` acceptable for both a new record type and a new
   clock block?
4. Should I2b land as its own PR after I2a, as proposed?

## 6. Authorisation

This document authorises nothing. Implementation of I2a and I2b each needs
separate approval after this proposal is reviewed. The manifest producer,
the harness, the evaluator, I4–I6, the PC2 pilot, calibration, the freeze
and qualification remain unauthorised.
