# MEM-002 cold-state qualification: protocol candidate

**Class: PROTOCOL CANDIDATE — NOT FROZEN.** Revised after maintainer
review on 2026-11-01 (§13). Dated 2026-11-01 at main
`b1e6a524`. Nothing was built, run or evicted for it. It sets no number.
Where a number is needed, it says how the number is justified before any
scored arm.

It builds on the merged design note
([`mem-002-cold-state-design.md`](mem-002-cold-state-design.md), #669,
cited **[D §n]**), the scoping ([`mem-002-scoping.md`](mem-002-scoping.md),
#668, **[S §n]**) and the MEM-001 reconnaissance
([`mem-001-reconnaissance.md`](mem-001-reconnaissance.md), **[R §n]**).

## 0. What this experiment is, and is not

**Question.** Can a valid, witnessed cold-binding state be obtained
repeatedly for whole-token VINDEX3 execution of Kimi-Linear `.s7`, by
either of two methods, with measured and acceptable disturbance?

**Is:**
- **Apparatus qualification:** for each cold method, whether it yields
  admissible cold-binding attempts and how much it disturbs the machine.

**Is not:**
- a latency study;
- a test of H1, the memory-cost predictor;
- a test of H2, policy selection;
- an oversized-model experiment [S §0].

**Latency is recorded but sealed.** It is not analysed, and it plays no
part in choosing or qualifying a method, until qualification is decided
(§8).

## 1. Experimental units

| Unit | Status in this protocol | Evidence required |
|---|---|---|
| **Cold binding** | **primary unit** | the binding-line witness, per region, all regions read successfully, at or below the admission threshold (§3, §7) |
| **Cold prefill** | **deferred.** Not qualified by this protocol; cold binding is qualified first | when taken up, it needs a pre-prompt witness recorded **before** the prompt timer starts, under the same per-region rule. It is never inferred from cold binding, nor from the post-prompt derived figure [D §1.1] |
| **Cold decode** | **not claimed** | it would need a pre-step witness of the selected pages, which main lacks [D §1.1]. No step after the prompt is called cold |

The cold arm is **pass 1 of a fresh process**, with `--warmup 0` and
`--repeat 1` [D §1.1]. Each attempt is one fresh process.

## 2. Subject and identity

Identity is established in a **separate phase (A)**, completed well before
any cold arm. Hashing reads every extent and leaves the subject resident
[D §3].

| Check | Method | Existing on main? |
|---|---|---|
| Payload integrity of `.s7` | `larql verify <container>`, which re-hashes every segment against its declared `segment_sha256` (`commands/extraction/verify_cmd.rs:109` → `inspect_container(path, true)`; `inspect.rs:199-201,257-270`) | **yes** |
| Directory identity | SHA-256 of `.s7/index.json` equals the recorded prefix `ddd9cad3e4603d12…` [R §2.2] | yes (shell `shasum`) |
| Execution-graph identity | SHA-256 of `.s7/system_graph.json` equals the recorded prefix `1fe02abad2f3bcdc…` [R §2.2]. `.s7`'s own graph is the execution definition. **No equivalence with lift2 is claimed**: lift2's graph differed [R §2.2] | yes (shell `shasum`) |
| Clone identity, per attempt (method F) | filesystem-level evidence, **without reading the payload**: (a) the clone is created with a call that **fails rather than copies** (`clonefile(2)`); (b) an **independent extent check**: physical block addresses of the clone and the source agree at sampled offsets (`fcntl` `F_LOG2PHYS_EXT`); (c) `index.json` and `system_graph.json` of the clone hash equal to `.s7`'s. (d) `df` before and after is corroboration only, and **never** substitutes for (a) or (b) [D §3] | **no**: (a) and (b) need a small tool (I4). Whether `clonefile` refuses rather than copies, and whether `F_LOG2PHYS_EXT` reports shared extents as identical, are to be **confirmed in I4's implementation review**, not assumed here. **If the filesystem cannot provide a reliable identity check, method F is disqualified** (§10) |
| Displacers share no extents with the subject (method D) | the same extent check (b), run once per displacer file in phase A | no (I4) |

Model bytes are never written: F creates clones, and D only reads. Clones
are deleted after each attempt [D §7].

## 3. Residency witness

**Rule: unknown residency is a failed admission, never zero.**

- Each mapped region's mincore read is reported as **success with a
  resident count**, or **failure**. Today `resident_bytes()` returns
  `Option`, and the summing caller maps `None` to 0 (`weights/loaded.rs:37`)
  [R §1.1, D §3].
- The binding line reports the **number of regions read, the number that
  failed, and the resident total over the successful regions**.
- An attempt with **any failed region is rejected** as "witness failed",
  counted separately from "witness above threshold".
- This is instrumentation **I1** (§9). Until I1 is merged, the protocol
  cannot be frozen: a zero is not distinguishable from a failed read
  [D §3].

The resident total keeps the boundary-page caveat: per-region counts
double-count shared boundary pages [R §1.1]. The admission threshold (§7)
must name the counting convention.

## 4. Positive controls

Run in **phase B**, after identity and before any qualification arm. Each
control has a pass condition fixed at the freeze.

| Control | What it shows | Procedure | Needs |
|---|---|---|---|
| **PC1, cross-process visibility** (**required**, foundational) | mincore in a fresh process sees pages another process cached but this one never mapped [D §3] | another process reads a known set of `.s7` files; a fresh process then binds. The binding witness must report them resident. V3's second run is the historical analogue (10.485 GB at binding) [R §2.2] | I1 |
| **PC2, region-level correctness** (**required**, foundational) | the per-region report attributes residency to the right regions. An aggregate can look right while individual regions are wrong, and the historical `None → 0` behaviour makes that a live risk | another process reads a **known byte range** of the expert-bank segment. The fresh process's per-region report must show residency in exactly the regions covering that range, and none elsewhere beyond boundary pages | I1, I3 |
| **PC3, within-run consistency** (**diagnostic only**) | whether faults and resident changes are broadly consistent | on each admitted attempt, record the prompt's major faults × 16 KiB beside its resident delta [R §1.1]. **Not an equality gate**: the relationship depends on page geometry, fault accounting, residency before the run and concurrent activity. A large discrepancy is reported and investigated, but does not by itself reject an attempt | existing |
| **PC4, clone isolation** (F only) | a clone's cache state is independent of its source's | with `.s7` warm (after phase A), create a clone. Its fresh binding must witness ~0 while `.s7` is still resident. That is the direct test of the indirect evidence in [D §2] | I1, I4 |

If PC1, PC2 or PC4 fails, the corresponding claim cannot be made, and §10
applies.

## 5. Cold methods compared

| | Method | Procedure per attempt | Main disturbance risk |
|---|---|---|---|
| **F** | fresh APFS clone | `clonefile` the container, verify clone identity (§2), bind the clone in a fresh process, delete the clone afterwards | Spotlight indexing of the new file; device-internal caching of shared extents, which is unobservable [D §6] |
| **D** | displacement | stream displacer files through the page cache (I6), settle (§6), bind `.s7` in a fresh process | memory pressure, file-cache churn for other processes, device traffic, thermal state [D §6] |

- **D's displacers** must share no extents with `.s7` (§2). The historical
  list included `.s6`, which is excluded [D §2]. Whether the remaining
  non-clone files are enough to displace a 94 GB bank on a 128 GiB machine
  is **unknown**. Phase C establishes it, and an inadequate displacer set is
  a D failure, not a reason to change the set mid-run (§10).
- **The method is never chosen by latency.** It is chosen by validity, the
  admission and witness results, and by measured disturbance (§8). The
  historical latency results [R §2.4] play no part.

## 6. Machine state, settling and Spotlight

**Recorded per attempt:**

- **Before the method:** the existing quiet-gate reading (`Environment`:
  power source, load, background CPU, busiest process;
  `opplan/exec/cpu/environment.rs:114-118`).
- **Snapshots**, taken before the method, after it, immediately before
  binding, and after the arm:
  - `vm_stat` counters: pageins, pageouts, compressions, swapins and
    file-backed pages;
  - `iostat` throughput on the internal disk;
  - CPU of `mds`, `mds_stores` and `mediaanalysisd`.
- **Peer GPU or LARQL processes:** a process-list check for other `larql`,
  `cargo` or benchmark processes, because the quiet gate cannot see a peer's
  GPU work. There is no direct GPU-contention instrument; this is a
  residual (below).

All of this is the committed machine-state recorder **I5**. It replaces
`contention.sh`, which was never committed [R §1.4].

**Settling.** After the method, the arm waits until two conditions hold,
or a timeout fires:

- the snapshots return to within a tolerance of the pre-method baseline;
- the quiet gate passes.

The tolerance and the timeout are set in **phase C, the calibration
block**. That block is excluded from qualification (§7). A timeout is a
recorded settle failure.

**Spotlight.**

- The clone directory and the displacer locations are excluded from
  indexing before phase B.
- The exclusion is **verified**: no `mds_stores` activity attributable to
  them during a dry clone creation in phase C. `mdutil -s` only reports
  whether indexing is enabled [D §6].
- Whether an exclusion marker or the system privacy list is used, and
  whether it is honoured, is confirmed in phase C.

**Residual, unobservable effects**, stated with every result:

- device-internal caching, in particular of extents shared by clones;
- SSD thermal state;
- GPU contention beyond the process-list check;
- demand refill continuing after the counters settle [D §6].

## 7. Admission, counts and ordering

All of these are fixed at the freeze, before phase D.

- **Gate quantity.** The binding-line resident total over successfully read
  regions, with zero failed regions (§3), under a named boundary-counting
  convention.
- **Admission threshold: how it is justified, without circularity.**
  - It is **one threshold for both methods**. It is never tuned per method,
    so it cannot be chosen to make the method under test look successful.
  - Its basis is instrument properties: the resident total a fresh process
    reports at binding without reading payload, plus a margin derived from
    the boundary-page counting convention.
  - Any empirical input comes from the **calibration population** (phase C):
    attempts that are separate from qualification, drawn from **both**
    methods in a declared mix, and discarded from scoring.
  - It is fixed before any qualification attempt and **never revised after
    qualification data are seen**. Historical PREFETCH results play no part.
  - If calibration cannot yield enough stable observations to justify a
    threshold, the protocol is not frozen. That outcome is recorded (§10),
    and the threshold is not chosen anyway.
- **Attempts.** A fixed number of attempts per method, N_F and N_D, set at
  the freeze. No optional stopping on interim results. The run stops early
  only on a predeclared abort (an identity or positive-control failure, or a
  safety condition).
- **Order.** Attempts interleave F and D in a counterbalanced order fixed in
  advance, for example ABBA blocks, so that drift is not confounded with
  method [D §5].
- **Recording.** The gate decision (admit, above threshold, or witness
  failed) is written to the record **before** the timed region starts
  (I2).
- **Rejected attempts** are still executed and timed. Their latency is kept
  as **sealed evidence**, excluded from the qualification decision and
  unavailable to method selection. Any later secondary analysis needs its
  own declared purpose, and can never retroactively change qualification
  [D §5].
- **Accounting.** Every attempt is reported, admitted or rejected, with its
  gate decision, settle outcome and machine-state snapshots.
- **Qualification outcomes are admission and disturbance only.** Latency is
  never among them.

## 8. Qualification decision

Phases:

| Phase | Content | Scored? |
|---|---|---|
| A | identity and integrity (§2) | pass/fail gate |
| B | positive controls (§4) | pass/fail gate |
| C | calibration: settling tolerance and timeout, Spotlight verification, the cold floor for the threshold, displacer adequacy | **not scored**; sets parameters only |
| D | qualification attempts (§7) | **scored on validity and disturbance only** |

**A method qualifies** if, under criteria and numbers fixed at the freeze,
all of the following hold:

- the admission rate is at least a stated fraction of attempts;
- the witness-failure rate is at most a stated fraction;
- every admitted attempt settled within the timeout;
- the disturbance measures stay within stated bounds of baseline;
- its positive controls passed (PC1, PC2, and PC4 for F);
- for F, clone identity was established on every attempt (§2).

Both methods may qualify. If both do, the preferred one is the less
disturbing, **not** the faster. Latency is unsealed only after the
qualification decision is recorded, and then only for the separately
frozen next stage (§11).

## 9. Minimum new instrumentation

Each item needs its **own implementation review** before the freeze. None
is built here.

| | Instrument | Needed for | Notes |
|---|---|---|---|
| **I1** | per-region witness success/failure: a failed-region count in `MappedResidency`, and on the binding line | every admission (§3) | a change to `prepared/residency.rs` and the residency-curve output; **blocks the freeze** |
| **I2** | the gate decision recorded before the prompt timer starts | §7 recording | a residency-curve output change; no timing change. A pre-prompt witness is added only when cold prefill is taken up (§1) |
| **I3** | a **minimal** per-region residency report (region, mapped bytes, resident bytes or failure) | PC2, **required** | output only; shares I1's observation API |
| **I4** | clone tool: `clonefile` creation and an extent-sharing check (`F_LOG2PHYS_EXT`) | F; the displacer extent check | a script or example; the semantics in §2 are confirmed in review |
| **I5** | machine-state recorder (`vm_stat`, `iostat`, indexer CPU, process list), committed | §6 | replaces the never-committed `contention.sh` |
| **I6** | displacement script on main, with the extent-sharing exclusion | D | a rewrite of branch-only `evict-bank.sh` [R §1.2] |

**Implementation order.** Not all at once, and each step is reviewed
before the next is built:

1. **I1 + I3**, trustworthy residency evidence: explicit per-region errors,
   region-level output, and positive-control tests. They may share one
   narrowly scoped instrumentation PR, because they share one observation
   API. A separate implementation proposal is reviewed first.
2. **I2**, pre-execution admission and recording. It depends on I1, and is
   reviewed separately.
3. **I4**, clone verification: whether method F can provide the required
   file identity and cold-state behaviour.
4. **I5 + I6**, environmental observation and displacement, built
   separately once observation and cloning are understood.

**Not required for qualification:** a routing exporter, a device-read
counter, a pre-step selected-page witness, an explicit fetch path, or a
memory limit [S §3.1].

Routing capture for the prompt step is also not required. It starts only
in the decode loop [D §1.1], and becomes necessary only when prompt
latency is analysed later.

## 10. Freezing, narrowing and abandoning

**The protocol may be frozen when all of these hold:**

- I1, I2, I3, I4, I5 and I6 are merged after their own reviews;
- §2's clone semantics are confirmed in I4's review;
- every number in §6–§8 has a stated justification that does not use
  qualification or historical outcomes;
- the abort conditions and the failure criteria are written;
- the calibration population is defined, and calibration has yielded
  enough stable observations to justify every threshold;
- an independent evaluator is named, with the records it scores from.

**The experiment is narrowed, as a pre-declared condition reported under
its own name, if:**

- PC4 fails, or clone identity cannot be established reliably: F is
  **disqualified**, and D is evaluated alone;
- displacer adequacy fails in phase C: D is dropped, and F is evaluated
  alone;
- D qualifies only with unsettled side effects: its claim is restricted to
  "latency under the complete displacement procedure" [D §6].

**The experiment is abandoned, or the claim recorded as not establishable,
if:**

- PC1 or PC2 fails: the witness cannot support any cold claim on this
  machine;
- neither method qualifies under the frozen criteria. The result is "cold
  binding is not qualified with these methods and instruments on this
  machine", with the failing criterion named. The apparatus is not modified
  until something passes [D §7].

## 11. What follows qualification (not part of this protocol)

1. **Retrospective latency against faults**, a separate freeze: on admitted
   cold-binding attempts (cold prefill only after it is separately qualified), how much
   latency variation the observed faults and resident deltas explain. It is
   feasibility evidence, never H1 [D §1.2].
2. **Prospective H1**, a separate freeze, with information declared as
   available before execution [D §1.2].
3. **H2 and the oversized gate**, later [S §0].

## 12. Freeze decisions still open

1. N_F and N_D, the block order, and the abort conditions.
2. The calibration population: its size, its F/D mix, and its separation
   from qualification.
3. The admission margin, the settling tolerance and timeout, and the
   disturbance bounds, each with an independent justification (§7).
4. The qualification thresholds: admission rate and witness-failure rate.
5. The displacer set, chosen from files with no extents shared with `.s7`.
6. The Spotlight exclusion mechanism.
7. The access rules for sealed latency, including rejected attempts',
   fixed before any execution.
8. **The independent evaluator, named.** Required before the freeze.

## 13. Maintainer decisions, 2026-11-01

| Question | Decision |
|---|---|
| Cold prefill | deferred; cold binding is qualified first |
| PC1, PC2 | required, as the foundational controls on the residency instrument |
| PC3 | diagnostic only, not an equality gate |
| PC4 | retained, as the specific test of method F |
| I3 | required for PC2, kept minimal |
| Clone identity | filesystem-level and independently checkable; `df` is never proof; failure disqualifies F |
| Calibration | a separate population; no threshold selected from qualification outcomes |
| Latency | sealed, unavailable to method selection; rejected attempts' latency is sealed evidence |
| Clone against displacement | both retained as candidates |
| Runtime performance optimisation | out of scope |
| Routing exporter, physical device reads | not required for this qualification |
| Independent evaluator | required before the freeze |
