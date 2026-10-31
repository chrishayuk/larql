# MEM-002 cold-state design note: how to establish and witness a cold start

**Class: DESIGN NOTE (read-only).** Dated 2026-10-31, on the MEM-002
scoping ([`mem-002-scoping.md`](mem-002-scoping.md), #668). Nothing was
built, run or evicted for it, and it does not freeze a protocol. Prior
results are cited to the merged MEM-001 reconnaissance as **[R §n]**, and
all of them are BACKGROUND ONLY.

**The question.** Can a genuinely witnessed cold state be obtained for a
whole-token Kimi `.s7` arm, without more disturbance than the latency
differences the feasibility study is trying to measure?

**Answer, as far as the records go.** Probably yes, but no record shows it.

- The lowest-disturbance candidate is a **fresh APFS clone of the subject
  per cold arm** (§2, method F). The records support it indirectly, and it
  has never been run as a cold method.
- The method the records actually used, displacement, gives a witnessed
  cold state, but with a large disturbance immediately before the arm.
- Either way, the claim is about **mapped-page residency** of the subject's
  file, not about physically cold storage.

## 1. Three states, and which counts

| State | Definition | Counts as a cold arm? |
|---|---|---|
| **Cold by measurement** | mincore over the subject's mapped regions reports negligible resident pages **immediately before execution**, and the instrument is shown to be working (§3) | **yes**, and only this |
| **Cold by procedure** | a specified eviction or displacement step was run, but residency was not witnessed, or was witnessed above the threshold | no |
| **Warm** | the target pages were already resident, or had been touched in an earlier pass | no |

Even "cold by measurement" is a statement about the unified buffer cache's
pages for **that file**. It says nothing about the SSD's internal caches,
or about physical extents shared with another file (§2, F).

**Which step is cold.** At binding, the whole bank can be witnessed at
0.000 GB. The first generated token is not cold: the three-position prompt
leaves 59–67 % of its selection resident [R §2.4]. A protocol must say
whether its cold unit is the first prompt position, the whole prompt, or
the first generated token after a cold prompt. The historical "cold token"
was the last of these.

## 2. Methods compared

| | Method | Recorded behaviour on macOS | Side effects and disturbance | Observable guarantee | Suitability |
|---|---|---|---|---|---|
| A | `MADV_DONTNEED` on the mapping | the records conflict: no eviction in the GLM run; a fresh file left 100 % resident in `vindex3_residency_probe`; "first eviction only" in the KU5 probe [R §1.2] | none | none | **unusable** |
| B | `MADV_FREE_REUSABLE` | returns `EPERM` [R §1.2] | — | — | **unusable** |
| C | `msync(MS_INVALIDATE)` | evicted under memory pressure; stopped reclaiming on an idle machine (frontier §14) [R §1.2] | small | none on an idle machine | **unreliable** exactly when the machine is quietest |
| D | Displacement: stream ~184 GB of other files (`evict-bank.sh`, branch `k3-prefetch-2` only) | gave a 0.000 GB binding witness in PREFETCH rounds 1 and 3 [R §2.3] | **large, and adjacent to the arm**: it evicts every other process's file cache; round 1 recorded load 5–13 from `mds_stores` indexing what the stream had read [R §2.3]. Spotlight indexing is enabled on `/`, and `~/chris-models` carries no `.metadata_never_index` (checked 2026-10-31) | mincore at binding; the script itself treats > 0.1 GB as not cold | **works, at the highest disturbance** |
| E | `sudo purge` | named in `v3-disk-resident-mmap.md`; no recorded use on this machine | system-wide cache flush before the process starts; needs root; refill traffic from every process afterwards | mincore at binding | **untested** |
| F | **Fresh APFS clone per arm** (`cp -c`, so metadata only), mapped by a fresh process | **no record of use as a cold method.** Indirect evidence: `evict-bank.sh` reads the HF directory first and `.s6` **last**, and the subject, recorded as an APFS clone of `.s6`, still witnessed 0.000 GB at binding. If clones shared cached pages, the last read would have left the subject resident [R §1.2, §2.2]. The clone relation at that date is assumed, not recorded [R §2.2] | near zero at the arm: no data read, no other process's cache evicted. A new file may be indexed by Spotlight. **Physical extents are shared with every other clone**, so device-internal caching of those extents is not excluded. Clones accumulate and must be cleaned up. Each clone needs its own payload check (§3) | mincore at binding, per clone | **best candidate; needs a feasibility check** |
| G | Virgin layers | the GLM §14 method: one arm per never-read layer of identical geometry [R §1.2] | none | cold by construction | **not applicable**: a whole Kimi token touches every routed layer. F is its whole-file generalisation |
| H | Reboot | not used as a cold method; round 3 recorded post-boot indexing (`mediaanalysisd`, `mds_stores`) as contention [R §2.3] | very large, for minutes after boot | mincore at binding | **poor** |
| I | Unmount and remount the subject's volume | not used | moves the subject to an external volume, which is a different device (USB 30.0 against internal 26.5 µs per serial fault) [R §1.2] | mincore at binding | changes the device under test |
| J | Memory balloon (wired allocation) | not available; would be new tooling [R §3] | changes Metal headroom, the compressor and every process's cache | mincore | **out of scope** (implementation) |

## 3. What the witness must show

The existing `--residency-curve` binding line, "X GB address space over N
regions, Y GB resident", is the witness [R §1.1]. Three caveats decide
whether a 0.000 GB reading means cold:

1. **A failed read looks cold.** `resident_bytes()` returns `None` when
   mincore fails, and the summing callers map `None` to 0 [R §1.1].
   - **Positive control already in the run:** the same pass's per-step
     resident deltas are non-zero once execution touches the bank (V3: a
     0.976 GB token delta after a 0.000 GB binding [R §2.2]). A
     0.000 GB binding followed by non-zero deltas on the same regions is a
     working instrument.
   - A protocol should require that control on every cold arm.
2. **The sum double-counts boundary pages** [R §1.1]. This doesn't matter at
   ~0, but a threshold above 0 must say what it counts.
3. **A threshold is a protocol choice.** The branch script uses 0.1 GB, about
   0.1 % of the bank. This note chooses none.

Method F adds one more requirement. Each clone is a new file, so its
identity must be re-established: a payload check against the declared
digest (`payload_sha256` `3f12d102…` [R §2.2]). Reading the clone to hash it
would make it warm, so the check is done on the source container once, and
the clone relies on APFS clone semantics. Whether that is acceptable, or
each clone is hashed and then evicted by another method, is a freeze
question.

## 4. Disturbance against the signal

- **The signal.** Historical demand cold tokens ranged from 602 to 1296 ms,
  with advised arms near 550–610 ms [R §2.4]. Under controlled conditions,
  the feasibility study looks for a stable fault-to-latency relationship,
  so the disturbance must be small against run-to-run differences of tens
  of milliseconds, not against the 2× gap.
- **Displacement (D)** disturbs at the worst time, just before the arm:
  - CPU load from indexing (load 5–13 recorded);
  - refill of other processes' caches;
  - device traffic from both.

  A settling interval plus the quiet gate could absorb some of it. How long
  is needed is unmeasured.
- **Fresh clone (F)** reads nothing, so its disturbance is limited to a
  metadata operation and any indexing of the new file. That indexing can be
  removed by excluding the clone directory from Spotlight. Its residual
  risk is device-internal caching of shared extents, which mincore cannot
  see.
- **Neither has been measured as a disturbance.** No record compares the
  same arm across cold methods.

**So whether a witnessed cold state with small enough disturbance exists is
not answered by any record.** F is the method most likely to give one. A
minimal feasibility check would answer it, but it is a measurement, and so
is not authorised by this note. The check would be:

- the same cold arm, repeated, with each of F and D;
- the binding witness and the positive control recorded on every arm;
- the latency and fault distributions compared.

## 5. Questions for the feasibility protocol

1. **Cold method.** F, with D as a comparison arm, or D alone? Is the
   branch's `evict-bank.sh` merged, or rewritten?
2. **Cold unit.** The first prompt position, the whole prompt, or the first
   generated token after a cold prompt (§1)?
3. **Witness threshold** and the positive-control rule (§3).
4. **Subject identity.** `.s7` payload verified against its declared digest,
   and `.s7`'s own system graph treated as the execution definition, not as
   equivalent to lift2's. For F, how the clone inherits that check (§3).
5. **Spotlight and indexing.** Exclude the subject and clone directories
   before any arm?
6. **Settling interval** after any method that reads, and how the quiet
   gate is applied after it.
7. **Clean-up.** Delete clones after each arm. They share extents, so the
   free space is not consumed, but they accumulate as metadata.

## 6. Sources checked

- `aae20ae3:scripts/evict-bank.sh`: the `DISPLACERS` order (HF directory,
  then `.s6`), the 0.1 GB threshold, and the comment "There is no supported
  way to drop macOS's unified page cache for one file".
- `which purge` gives `/usr/sbin/purge`.
- `mdutil -s /` gives "Indexing enabled".
- There is no `~/chris-models/.metadata_never_index`.
- `hw.pagesize` is 16384.
- Everything else is cited to the merged reconnaissance.
