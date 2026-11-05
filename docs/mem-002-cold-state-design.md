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

### 1.1 The cold unit

Three units are different experimental conditions. A cold binding witness
does not establish cold decode.

| Unit | Meaning | Witness on main (`--residency-curve`) |
|---|---|---|
| **Cold binding** | expert pages witnessed non-resident after binding, before any prompt runs | **yes**: the binding line (`residency_curve.rs:195-199`, printed before execution) reads mincore over every **mapped** bound region. Owned or widened operands are read during load and are not in it (`prepared/residency.rs:417-425`) |
| **Cold prefill** | expert pages witnessed non-resident immediately before prompt processing, in that pass | **yes, mapping-wide.** Each pass reads `mapped_residency()` just before the prompt timer starts (`residency_curve.rs:229`, `:278-292`). That reading is **not printed**. Only the post-prompt residency and its delta are, on the same line as the prompt's latency (`:262-273`). The pre-prompt figure is their difference, and because the delta is a clipped subtraction (`saturating_sub`, `:255`) it is a **lower bound** whenever residency fell during the prompt. The whole prompt is timed as one step, so the per-position residency is not recorded |
| **Cold decode** | the pages **a particular decode step will select** are non-resident immediately before that step | **no.** The pre-step reading is mapping-wide, and after a prompt it is far from zero. Which experts a step selects is known only once its routing runs. A pre-step witness of the selected pages would need the selection in advance, for example from an identical earlier pass, plus a per-selection mincore read. The branch-only `--witness-residency` reads the selected pages inside the timed token, at ~70 ms per token [R §2.3]. Neither exists on main as a pre-step witness |

What the instruments do give for a decode step is its **own major-fault
count and resident delta**, after the fact. It is an outcome measurement,
not a pre-execution witness of coldness, and a model built on it is
retrospective (§1.2).

**The cold arm on main is pass 1 of a fresh process** (for F, mapping a
fresh clone), with `--warmup 0`. From pass 2 on, pass 1's pages are
resident. The instrument's own `"cold"` label (`:214-220`) is assigned by
procedure (`pass == 1`), so by itself it marks **cold by procedure** (§1).

**Passing the binding gate establishes cold binding, not cold prefill.**
In pass 1, binding and the pre-prompt reading are separated only by the KV
build and the session constructor (`:212-213`). A cold-prefill claim needs
evidence that nothing in that gap brings bank pages in. The derived
pre-prompt figure can **corroborate** that afterwards, but it cannot gate
the arm: it is seen only with the outcome, and it is a lower bound.

**Consequences:**

- **The historical "cold token" was not cold decode.** It was the first
  generated token after a cold-prefill prompt. Before it, PREFETCH round 2
  **witnessed** 59.1 % of its selection resident, and V3-N3 **derived** about
  67 % for V3 [R §2.4].
- **The nearest witnessable decode-shaped unit is a one-position prompt.**
  It is one forward step from cold prefill, at position 0, with an empty
  KV state. It is cold by measurement, but at position 0 only, so a later
  position's routing and attention state are not represented. **On main it
  has no routing fingerprint:** routing capture starts only inside the
  decode loop (`:305`), not for the prompt. Its fault count and resident
  delta are recorded (`:286-292`).
- **Witnessing cold decode at position t > 0 needs instrumentation that
  does not exist on main.** It would be a pre-step, outside-the-timer
  witness of the selected pages, which is implementation and outside this
  note.

### 1.2 Retrospective explanation, prospective prediction, policy decision

These are three different models, with different information:

| Model | Information available | Legitimate claim |
|---|---|---|
| **Retrospective** | the step's latency, its observed major faults and resident delta | explains observed latency variation |
| **Prospective** | pre-execution residency of the pages the step will use, routing information available before the step, layout, and calibrated per-fault or per-read costs | predicts latency before execution |
| **H2 decision** | only what is available before the policy is chosen | selects an execution policy |

- **A model that uses the current step's observed faults is retrospective.**
  Its result is feasibility evidence, never H1 success.
- **A prospective model needs the pre-step residency of the selected
  pages.** That is exactly the cold-decode witness main lacks (§1.1).
- **Routing captured from an earlier pass is not decision-time
  information.** It may serve H1 as an oracle input, declared as such, but
  it is not automatically available to an H2 planner.
- **Two ambiguities in the merged scoping, for the H1 freeze:**
  - option 4's "measured fault or read costs" must say whether it means
    calibrated cost parameters (prospective) or the step's observed fault
    count (retrospective);
  - H1's "frozen routing trace" is oracle routing from an earlier pass
    (`mem-002-scoping.md` §0, §5).

## 2. Methods compared

| | Method | Recorded behaviour on macOS | Side effects and disturbance | Observable guarantee | Suitability |
|---|---|---|---|---|---|
| A | `MADV_DONTNEED` on the mapping | the records conflict: no eviction in the GLM run; a fresh file left 100 % resident in `vindex3_residency_probe`; "first eviction only" in the KU5 probe [R §1.2] | none | none | **unusable** |
| B | `MADV_FREE_REUSABLE` | returns `EPERM` [R §1.2] | — | — | **unusable** |
| C | `msync(MS_INVALIDATE)` | evicted in earlier runs while a peer session was running; stopped reclaiming once the machine was idle. Frontier §14 attributes this to memory pressure ("Eviction depends on memory pressure, not on advice") [R §1.2] | small | none on an idle machine | **unreliable** exactly when the machine is quietest |
| D | Displacement: stream ~184 GB of other files (`evict-bank.sh`, branch `k3-prefetch-2` only) | gave a 0.000 GB binding witness in PREFETCH rounds 1 and 3 [R §2.3] | **large, and adjacent to the arm**: it evicts every other process's file cache. Round 1 recorded load 5–13, which its notes **attribute** to `mds_stores` indexing what the stream had read; that cause is the notes' attribution, not established [R §2.3]. Spotlight responds to file changes, so a plain read is not expected to trigger indexing, while F's file creation is. Indexing is enabled on `/` and on the data volume, and `~/chris-models` carries no `.metadata_never_index` (checked 2026-10-31). The branch's displacer list includes `.s6`, which may share extents with `.s7`, so D's displacers must **exclude any file sharing extents with the subject** | mincore at binding; the script itself treats > 0.1 GB as not cold | **works, at the highest disturbance** |
| E | `sudo purge` | named in `v3-disk-resident-mmap.md`; no recorded use on this machine | system-wide cache flush before the process starts; needs root; refill traffic from every process afterwards | mincore at binding | **untested** |
| F | **Fresh APFS clone per arm** (`cp -c`, so metadata only), mapped by a fresh process | **no record of use as a cold method.** Indirect evidence: `evict-bank.sh` reads the HF directory first and `.s6` **last**, and the subject, recorded as an APFS clone of `.s6`, still witnessed 0.000 GB at binding. If clones shared cached pages, the last read would have left the subject resident [R §1.2, §2.2]. The clone relation at that date is assumed, not recorded [R §2.2] | near zero at the arm: no data read, no other process's cache evicted. A new file may be indexed by Spotlight. **Physical extents are shared with every other clone**, so device-internal caching of those extents is not excluded. Clones accumulate and must be cleaned up. Each clone's identity must be established without warming it (§3) | mincore at binding, per clone | **a candidate, not an accepted method; needs a feasibility check** |
| G | Virgin layers | the GLM §14 method: one arm per never-read layer of identical geometry [R §1.2] | none | cold by construction | **not applicable**: a whole Kimi token touches every routed layer. F is its whole-file generalisation |
| H | Reboot | not used as a cold method; round 3's notes record "post-boot indexing (mediaanalysisd ~80%, mds_stores ~33%)" during that round (`e2e0392f:docs/represent/forecasts/k3-prefetch-2-notes.json`) [R §2.3] | very large, for minutes after boot | mincore at binding | **poor** |
| I | Unmount and remount the subject's volume | not used | moves the subject to an external volume, which is a different device (USB 30.0 against internal 26.5 µs per serial fault) [R §1.2] | mincore at binding | changes the device under test |
| J | Memory balloon (wired allocation) | not available; would be new tooling [R §3] | changes Metal headroom, the compressor and every process's cache | mincore | **out of scope** (implementation) |

## 3. What the witness must show

For **cold binding**, and for the admission gate of any cold arm, the
witness is the existing `--residency-curve` binding line, "X GB address
space over N regions, Y GB resident" [R §1.1]. It is printed before
execution. Three caveats decide whether a 0.000 GB reading means cold:

1. **A failed read looks cold.** `resident_bytes()` returns `None` when
   mincore fails, and the summing callers map `None` to 0 [R §1.1].
   - **The gate needs mincore to see pages cached but not mapped by this
     process.** The record has that control: V3's second run read 10.485 GB
     of the bank resident **at binding in a fresh process**, left by earlier
     processes [R §2.2]. A protocol should repeat such a warm-at-binding
     control once per method.
   - **Within a run**, non-zero per-step resident deltas after a 0.000 GB
     binding show the instrument working on **some** regions (V3: 0.976 GB
     after 0.000 GB [R §2.2]). The sums are mapping-wide, so this does not
     exclude a failure on individual regions.
   - On fault-path arms, cross-check major faults × 16 KiB against the
     resident delta [R §1.1].
   - **These controls corroborate; they do not make a zero distinguishable
     from a failed read** (maintainer, 2026-11-01). **Freeze blocker:** no
     protocol may use zero residency as an admission criterion until the
     residency checks behind the gate report success or failure explicitly,
     per region, so that a failed observation cannot admit a run as cold.
     On main the binding line does not; making it do so is instrumentation,
     reviewed separately.
2. **The sum double-counts boundary pages** [R §1.1]. This doesn't matter at
   ~0, but a threshold above 0 must say what it counts.
3. **A threshold is a protocol choice.** The branch script uses 0.1 GB, about
   0.1 % of the bank. This note chooses none.

**Method F adds identity requirements**, and they must not defeat the cold
state:

- **Separate `.s7`'s identity check from the timed cold procedure.** Hashing
  `.s7`'s payload against its declared digests (`payload_sha256` `3f12d102…`,
  `segment_sha256` `e43600cd…` [R §2.2]) reads every physical extent a clone
  shares, and leaves `.s7` resident. It is done once, as a separate step,
  well before any cold arm, and never immediately before one.
- **Verify that `cp -c` cloned rather than copied**, on every arm. A copy
  would consume the bank's size in free space (`df` before and after), and
  would be a different physical subject. **`df` is corroboration only, not
  proof that shared extents were created** (maintainer, 2026-11-01): free
  space moves for other reasons. The feasibility protocol must specify an
  independently checkable clone-identity test before the clone method is
  used. This note does not name one.
- **Hash each clone's small `index.json` and `system_graph.json`.** That is
  cheap identity evidence that does not warm the bank.
- The clone's payload then rests on APFS clone semantics, which hold at the
  moment of creation.

## 4. Disturbance against the signal

- **The signal.** The historical demand "cold tokens" ranged from 602 to
  1296 ms, with advised arms near 550–610 ms [R §2.4]. That unit was the
  first generated token after a cold prefill, **not cold decode** (§1.1).
  The feasibility arm's unit is cold binding, or cold prefill with the
  §1.1 evidence, of pass 1 in a fresh process. Under controlled conditions,
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
  removed by excluding the clone directory from Spotlight, **verified before
  the first arm**. A concurrent indexer reading the clone could otherwise
  warm pages after a passing binding witness and during the timed prompt,
  which the witness cannot see. Its residual risk is device-internal
  caching of shared extents, which mincore cannot see.
  - **Unresolved hypothesis, not a finding:** round 3 had the lowest demand
    cost per fault (8.2 µs). If `.s6` was then the clone of the subject it
    was later recorded as, its displacement re-read extents shared with the
    subject, and shared-extent effects are one candidate explanation
    [R §2.4 (ii)]. Round 1 also displaced with `.s6`, though the script it
    ran was a session file whose equality with the committed one is
    asserted, not shown [R §1.2]. The record does not isolate this
    explanation.
- **Neither has been measured as a disturbance.** No record compares the
  same arm across cold methods.

**So whether a witnessed cold state with small enough disturbance exists is
not answered by any record.** F is the method most likely to give one. A
minimal feasibility check would answer it, but it is a measurement, and so
is not authorised by this note. The check would be:

- the same cold arm (one named unit), repeated, with each of F and D, in a
  counterbalanced order;
- the binding witness and the positive control recorded on every arm;
- the latency and fault distributions compared.

## 5. Classifying cold runs against manufacturing them

A method that **reproducibly causes** non-residency is a different claim
from a witness that **independently shows** it.

- **Manufacture.** A method that reliably makes the unit cold (D, perhaps
  F). The cold state is then an intervention, and its side effects belong
  to the condition (§6).
- **Classification.** Run every attempt, witness residency before the
  timed region, and admit an attempt as cold only if the witness passes. A
  reliable witness can classify runs even when cold starts cannot be
  manufactured reliably.

Classification is defensible only under rules frozen **before any latency
is seen**:

- the gate is evaluated from the pre-execution witness alone, never from
  the outcome;
- every attempt is reported, including how many failed the gate and why;
- the machine state of admitted and rejected attempts is compared, because
  a gate that admits mostly quiet or mostly loaded runs selects on a
  confound even though it never reads latency;
- the number of attempts, or the stopping rule, is fixed in advance;
- the **gate quantity** (the binding line), the **threshold** and the
  boundary-counting convention are frozen with the gate;
- the **gate decision is recorded before the timed region** of each attempt;
- **F and D arms are interleaved or counterbalanced** in a declared order, so
  drift is not confounded with method;
- whether **rejected attempts are still executed and timed** is decided in
  advance. Without it, admitted and rejected attempts can be compared only on
  machine state;
- **settling-interval length and quiet-gate parameters** are fixed before
  the cold arms, or tuned on a separately declared calibration set, never
  on the arms.

Today the binding line can gate cold binding. Cold prefill can be claimed
only with the §1.1 evidence, since its own reading is not seen before the
outcome. Nothing on main can gate cold decode (§1.1).

## 6. Separating the intervention from its side effects

A method can make target pages non-resident and also disturb the machine.
Any such disturbance is either **observed** or **folded into the claim**.

| Side effect | Observable with existing system tools (not run here) | Most at risk |
|---|---|---|
| Memory pressure, compression, swapping | `vm_stat` (pageins, pageouts, compressions, swapins) before the method, after it and before the arm | D, E, J |
| Device traffic | `iostat` on the internal disk across the settling interval | D, E, H |
| Spotlight and indexing | CPU of `mds_stores` and `mds` (`ps`, `top`); `mdutil -s` shows only whether indexing is enabled, not whether it is active | F (file creation), H; D only if reads trigger it, which is not expected |
| File-cache churn for other processes | `vm_stat` file-backed page counts | D, E |
| Device-internal caching of shared extents | **not observable** | F |

Two admissible claim forms follow:

- **Observed side effects settled.** A settling interval ends when the
  observed counters return to a baseline taken before the method, and the
  quiet gate passes. The claim is "latency from a cold unit (named), with
  the **observed** side effects settled". Unobserved effects remain:
  `iostat` shows aggregate traffic, not its source; `vm_stat`'s file-backed
  counts are system-wide; demand refill can continue after the counters
  settle; SSD thermal state is not observed.
- **Side effects not separable.** The claim is restricted to "latency under
  the complete displacement procedure" (or under the clone procedure),
  never "latency from cold storage".

For F, the unobservable device-internal cache means the claim stays
"cold in the page cache for this file", whatever is observed.

## 7. Model bytes, identity and the failure outcome

- **Preserve the model bytes.** D only reads. F creates a clone with
  `cp -c` and never writes the source. Neither modifies `.s7`. Clones are
  deleted after use.
- **Verify `.s7` once, as a separate step well before any cold arm.** Hash
  its payload against the declared `payload_sha256` (`3f12d102…`) and
  `segment_sha256` (`e43600cd…`) [R §2.2]. The hash warms `.s7` and every
  extent its clones share, so it is never the step immediately before a cold
  arm (§3).
- **A clone is byte-identical to its source by APFS clone semantics**, and
  carries `.s7`'s own system graph. Its execution definition is `.s7`'s.
  Neither the clone nor `.s7` is claimed to be execution-equivalent to the
  historical lift2, whose system graph differed [R §2.2].
- **An explicit failure outcome.** If neither D nor F can establish a
  defensible cold condition for the unit the protocol names, that is
  recorded as the result: "the named cold unit (for example cold decode) is
  not establishable with existing instruments on this machine", with the
  missing instrument or control named (§1.1).
  It is not an invitation to keep modifying the apparatus until a gate
  passes. A fallback to a weaker unit (cold prefill, or a one-position
  prompt) is a separate, pre-declared condition, reported under its own
  name.
- **The failure criterion is operational and set at the freeze**, for
  example an admission rate below a stated fraction of attempts, or side
  effects not settled within a stated time. This note sets no number, but a
  freeze without one leaves the apparatus open to iteration.

## 8. Questions for the feasibility protocol

1. **Cold method.** F, with D as a comparison arm, or D alone? Is the
   branch's `evict-bank.sh` merged, or rewritten?
2. **Cold unit.** Cold binding, cold prefill, a one-position prompt, or cold
   decode, which needs instrumentation main lacks (§1.1)? Is a fallback unit
   declared in advance (§7)?
3. **Manufacture or classify** (§5)? If classify: the gate, the attempt count
   or stopping rule, and the report of rejected attempts.
4. **Side effects** (§6): which observers, which baseline, and which claim
   form.
5. **Witness threshold** and the positive-control rule (§3), frozen with
   the gate (§5).
6. **Subject identity.** `.s7` payload verified against its declared digest,
   and `.s7`'s own system graph treated as the execution definition, not as
   equivalent to lift2's. For F, how the clone inherits that check (§3, §7).
7. **Spotlight and indexing.** Exclude the subject and clone directories
   before any arm?
8. **Settling interval** after any method that reads, and how the quiet
   gate is applied after it.
9. **Clean-up.** Delete clones after each arm. They share extents, so the
   free space is not consumed, but they accumulate as metadata.
10. **Model class** (§1.2): retrospective feasibility first; which
    information a prospective H1 model and an H2 planner may use.
11. **Failure criterion** (§7): its operational form and numbers.

## 9. Sources checked

- `aae20ae3:scripts/evict-bank.sh`: the `DISPLACERS` order (HF directory,
  then `.s6`), the 0.1 GB threshold, and the comment "There is no supported
  way to drop macOS's unified page cache for one file".
- `crates/larql-cli/src/commands/primary/vindex3_cmd/generate/residency_curve.rs`: the pre-prompt `mapped_residency()` read (line 229), the prompt timed as one step (lines 278-292), and the per-step mapping-wide witness inside `observe`.
- `prepared/residency.rs:417-425` (`mapped_residency` counts only regions
  with mapped bytes); `residency_curve.rs:195-199` (binding line),
  `:212-220` (session build and the `"cold"` label), `:255`
  (`saturating_sub`), `:262-273` (post-step print), `:305` (routing capture
  starts in the decode loop).
- `frontier-execution-programme.md` §14: "Eviction depends on memory
  pressure, not on advice."
- `e2e0392f:docs/represent/forecasts/k3-prefetch-2-notes.json`: "post-boot
  indexing (mediaanalysisd ~80%, mds_stores ~33%)".
- `which purge` gives `/usr/sbin/purge`. `vm_stat`, `iostat` and `fs_usage` are present in `/usr/bin` and `/usr/sbin`; none was run.
- `mdutil -s /` and `mdutil -s /System/Volumes/Data` both give "Indexing
  enabled".
- There is no `~/chris-models/.metadata_never_index`.
- `hw.pagesize` is 16384.
- Everything else is cited to the merged reconnaissance.
