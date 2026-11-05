# MEM-002 scoping: predicting and improving oversized-MoE execution cost

**Class: SCOPING.** Dated 2026-10-31 at main `bcb67863`. Read-only: no code,
no cache, no memory-pressure test and no scored measurement was run or
written for it. It sets no threshold and picks no policy. "MEM-002" is a
provisional name.

Its evidence base is the merged MEM-001 reconnaissance
([`mem-001-reconnaissance.md`](mem-001-reconnaissance.md), #667), cited as
**[R §n]**. Every prior result there is BACKGROUND ONLY [R §2.0].

## 0. The question and its two hypotheses

> Can LARQL predict the memory movement and execution cost of a routed MoE
> model whose expert weights exceed available physical memory, and can that
> prediction identify a faster execution strategy?

- **H1, prediction.** Given a frozen routing trace, the expert layout and
  measured storage characteristics, LARQL predicts the bytes that must move
  between storage and memory, and the execution time, within a bound fixed
  in advance.
- **H2, optimisation.** Choosing among existing execution policies by that
  prediction improves end-to-end token latency against a declared baseline,
  with numerically unchanged output.

H1 is tested first. A speedup without a validated predictor would not show
that the predictor chose the better policy.

**Four separate validation gates.** Each opens only when the one before it
has succeeded:

| Stage | Question | Scope |
|---|---|---|
| Feasibility | Can stable cold-state and timing measurements be established? | the existing VINDEX3 runtime on Kimi `.s7`; instrumentation and controls only, **not** an H1 result |
| H1 | Does a frozen model predict held-out latency and movement? | a model and conditions fixed in advance |
| H2 | Can its predictions select a policy that beats a baseline? | only after H1 succeeds |
| Oversized execution | Does the approach generalise when the bank exceeds RAM? | a separate, later validation |

**Headline.**

1. **The GLM-5.3-Flash expert bank is not a usable subject today.**
   - Its volume (`model-drive`) is not mounted.
   - The checkpoint (305.78 GiB) would not fit on the internal disk, which
     has 303 GiB free.
   - There is no weight digest.
   - **GLM has no whole-token execution path.** Its DSA indexer is
     registered at extraction tier only, with no executor.
2. **No model on local disk exceeds physical memory** (128 GiB). The
   largest executable routed model is Kimi-Linear `.s7` (94.22 GB bank),
   which fits.
3. **No in-repo method makes a fitting model exceed *available* memory on
   this machine.** LARQL has no runtime cap on mapped pages, eviction is
   unreliable on an idle machine, and no memory balloon exists [R §1.3,
   §1.4, §3].
4. **Under mmap, the kernel, not LARQL, decides what stays resident.** A
   prediction of residency over many tokens would then have to model an
   unspecified macOS replacement policy. An explicit fetch path would give
   LARQL control over **the requests it issues**, not necessarily over
   physical I/O. Four quantities stay distinct under any fetch path:
   - bytes LARQL requests;
   - logical reads the filesystem serves;
   - physical device traffic, which filesystem caching, readahead,
     controller caching and request coalescing can make differ from the
     requests;
   - resident-page changes.

   Such a path exists only in the GLM per-layer probe, not in VINDEX3 [R §2.1,
   §2.4, §6.3].
5. **The historical cold-run range is not evidence that prediction is
   impossible.** Four cold observations of one Kimi token gave 8.2–21.7 µs
   per demand fault [R §2.4]. They were taken under different initial
   residency, machine state, eviction method and instrumentation, so they
   are **observations under differing conditions, not a controlled variance
   distribution**. The open question, and the central objective of the
   first investigation, is whether **controlling** those conditions gives a
   stable relationship between page movement and latency.

So the experiment cannot be frozen on GLM, and it cannot be run in regime A
on this machine at all. The first candidate is a controlled cold-token
timing-feasibility study on Kimi `.s7`, using the existing VINDEX3
execution path (§5).

## 1. Candidate subjects

| Subject | Bytes vs RAM | Whole-token execution in LARQL | Identity | Available 2026-10-31 |
|---|---|---|---|---|
| GLM-5.3-Flash | expert estate 283.5 GiB (42 × 6.75), checkpoint 305.78 GiB, against 128 GiB: **exceeds** [R §3] | **no.** Layer types graded one at a time; KDA executes, and the DSA indexer has no executor (frontier §5; `larql-models` `glm5` registers it "extraction-tier only", `d29b85e6`) | size only, no weight digest [R §2.0] | **no**: `model-drive` not mounted; internal disk has 303 GiB free |
| Inkling-Small | 495.38 GiB checkpoint: **exceeds** | not established here; frontier §5 grades GLM only | none recorded | **no**: on `model-drive` |
| Kimi-Linear `.s7` | 94.22 GB bank [R §1.5], 98.14 GB stored (K3 residency vertical V2): **fits** | **yes.** V3/V4/PREFETCH ran it (on lift2), with HF-oracle logit parity (V3 rung 6) | declared bank identity with lift2 established; system graph differs [R §2.2] | **yes**, in `~/chris-models` |
| Gemma 4 26B-A4B | ~11 GB expert working set: **fits** | legacy path; VINDEX3 lowered path | not pinned | the q4k container named in the locality record is not in `~/chris-models` |

## 2. What is genuinely new, against MEM-001's prior results

| Prior result [R] | What it established | What MEM-002 would add |
|---|---|---|
| GLM §11, coverage 1.0003 [R §2.1] | **which pages** one token touches in one layer, cold, readahead off | bytes moved over **many tokens, under a memory limit**, where reuse and eviction decide movement |
| GLM §12, LRU / Bélády over a 128-token prefill [R §2.1] | an **offline** hit-rate curve per layer at assumed budgets | a hit rate **observed in execution**, compared with the simulated one |
| GLM §14, `pread` × 8 2.36× faster [R §2.1] | fetch shape matters for **one stage** | whether a predictor can **choose** a fetch shape or policy, and whether the gain survives whole-token decode |
| PREFETCH rounds, V3 [R §2.3, §2.4] | advise ≈ 2× demand on one cold token, in two of three rounds | a **time model** with an error bound fixed in advance, and its stability across runs |
| Gemma locality, freq-mass, BW11-1 [R §2.5, §2.6] | little static hot-set potential on one load-balanced router; modest short-window reuse on GPT-OSS | whether a predictor that exploits **short-window** reuse or prefetch beats demand paging when the bank does not fit |

The novelty is turning residency prediction into an execution decision
that is evaluated end to end. That has not been attempted. What would be
new in H1 is the **time** prediction and the **multi-token, constrained**
bytes prediction. A single-token page prediction would replicate GLM §11.

## 3. Instruments, against what H1 and H2 need

| Need | Instrument on main | Gap |
|---|---|---|
| Routing trace | VINDEX3 `routing_trace` capture: every routed layer's selected experts for the characterised step, in process. `--residency-curve` reports its fingerprint, the routed-layer count and the first layer's experts per pass (`residency_curve.rs:305-337`) [R §1.1] | in memory only; no exporter [R §1.4]. **Sufficient for feasibility**, where routing identity across passes in one process is what matters. An exporter becomes necessary for reproducibility across processes, or for feeding an independent simulator |
| Expert layout, bytes per expert | region table; `index.json` [R §1.5] | expert adjacency under the writer's name ordering is **unchecked** [R §1.5] |
| Bytes moved | per-step mincore resident deltas and `ru_majflt` deltas, both printed by `--residency-curve` [R §1.1] | faults track 16 KiB pages one to one on every fault-path arm on record, and stop being a byte proxy once the fetch path changes; summed mincore double-counts boundary pages; **no device-read counter** [R §1.1, §1.4]. Sufficient for a latency-against-faults study; not for a physical-I/O claim |
| Storage characteristics | `examples/readrate.rs` (serial per-fault rate); `scripts/dec8_2_fio_sweep.py` (no result on main) [R §1.2] | queue-depth and request-size curves for the internal SSD are not recorded |
| Reuse simulation | `glm_expert_reuse.py`; retention simulator (`--expert-bytes`); `larql moe-locality` [R §1.2] | none reads a VINDEX3 trace [R §1.4] |
| A memory limit | none in VINDEX3 [R §1.3] | **missing**: a runtime cap, an explicit fetch path, a balloon or another machine |
| Execution policies to choose among (H2) | `MappedAccess {Demand, Advise, Touch}` on main [R §1.1]; `Coalesced`, `Gated` on branch `k3-prefetch-2` only [R §1.2] | no explicit-fetch policy in VINDEX3; the branch policies are unmerged |
| Output identity | routing fingerprint, generated ids, logits bit-identity [R §2.3] | — |
| Latency | per-step wall time; the residency witness costs ~70 ms per token [R §2.3] | an unwitnessed arm is required; one was never recorded for round 2 [R §1.4] |
| Machine state | the quiet gate refuses unqualified warm arms [R §2.3] | **no device-I/O or contention record**; `contention.sh` was never committed [R §1.4] |

### 3.1 Prerequisites, by the claim they support

| Prerequisite | Latency/fault feasibility (existing runtime) | Defensible physical-I/O prediction |
|---|---|---|
| A cold-state method, witnessed per arm | **required.** Eviction is unreliable on an idle machine; displacement is branch-only and may re-read a clone of the subject [R §1.2, §5.2] | required |
| Machine-state record, including device I/O or contention | **required** at least for CPU load, power and background processes; a device-I/O record is needed to test a contention explanation [R §1.4] | required, including device I/O |
| Unwitnessed timing arms | **required** [R §5.4] | required |
| Routing fingerprint per pass | **required**; exists in process | required |
| Routing exporter | not required (in-process capture suffices) | required for cross-process reproducibility or an independent simulator |
| Device-read counter | not required (latency is tested against measured faults) | **required**: without it, requests, logical reads and physical traffic cannot be separated |
| Explicit fetch path | not required | one candidate route, not necessarily required (§5) |
| Memory limit | not required | required only for the oversized gate |

## 4. Environmental controls the protocol would need

Each is drawn from a recorded failure [R §5]:

- **Cold state.** Advice is not a usable eviction method, and
  invalidation stopped on an idle machine [R §1.2]. Displacement reads
  ~184 GB, and one displacer may be a clone of the subject [R §5.2].
  Virgin layers need repeated geometry. **The method must be chosen and
  witnessed per arm.**
- **Machine state.** Record the power source, load, background processes,
  Spotlight indexing on the model volume, and peer GPU sessions. The CPU
  quiet gate cannot see a peer's GPU benchmark.
- **Instrument cost.** Witnessed residency and unwitnessed latency go in
  separate arms [R §5.4].
- **Shared buffer cache.** `pread` and mappings share it, and a fresh
  process inherits earlier residency. Run the cold check before any fetch
  [R §5.1].
- **Routing identity.** Record the fingerprint on every arm; V3 lacked it
  [R §5.8].
- **Repeats.** Single samples are not distributions [R §5.9].

## 5. Ways forward, for the maintainer to choose

None of these is authorised. Each is listed with what it would require.

0. **Controlled cold-token timing feasibility on Kimi `.s7` (first
   candidate).** Use the existing VINDEX3 execution path and
   `--residency-curve`, with no change to weight loading. Cold-token arms
   are repeated under one witnessed cold-state method, a recorded machine
   state, unwitnessed timing and a routing fingerprint per pass. The
   question is whether, once those conditions are controlled, cold-token
   latency against measured major faults and resident deltas is stable
   enough to model. This is the feasibility gate of §0, not H1. If it fails,
   the failure names which measurement or control is missing before any new
   execution path is considered. It overlaps MEM-001A's question [R §6.2]
   and the prerequisites in §3.1.

1. **Mount `model-drive` and make GLM the regime-A subject.** It would
   require:
   - digesting the checkpoint;
   - whole-token GLM execution, at minimum a DSA indexer executor and the
     remaining layer-type rungs (frontier §5–§6).

   This is a programme of implementation, not a measurement.
2. **Regime B on Kimi-Linear `.s7`, with an explicit memory limit.** The
   subject executes whole-token today. "Exceeds available memory" would be
   made true by a wired-memory balloon, which is new tooling and changes
   Metal headroom and every process's file cache [R §3], or by a
   smaller-RAM machine (a separate result).
3. **Explicit fetch path in VINDEX3, under a LARQL-managed byte budget.**
   LARQL then controls and can account for the bytes it **requests** and
   what it keeps resident in its own buffers. Logical reads, physical device
   traffic and page-cache residency remain separately measurable, and can
   still differ from the requests. It extends GLM §14 [R §6.3]. It needs a
   new realisation in VINDEX3, which is implementation. It is **one
   candidate** for a controlled memory limit, conditional on the
   feasibility and H1 evidence, not an inevitable second phase.
4. **H1 timing prediction without memory pressure.** On `.s7`, after the
   feasibility gate succeeds: freeze a model of cold whole-token latency from
   routing, layout and measured fault or read costs, and test it on held-out
   cold runs. No memory limit is needed. An exporter is needed only if the
   held-out runs are in other processes or an independent simulator scores
   them.

**Recommendation for the freeze discussion.** Option 0 first, then option 4
if it succeeds:

- Option 0 establishes whether stable cold-state and timing measurements
  exist on a subject that executes today, with existing instruments. Only
  then is an error bound for H1 meaningful.
- If option 0 fails, its failure decides what to build next. An explicit
  fetch path (option 3) is one candidate among the §3.1 gaps, not the
  default.
- GLM stays the regime-A goal for the oversized gate, but it is gated on
  implementation, not on measurement.

## 6. Candidate outcome measures (no thresholds)

| Measure | Basis | Notes |
|---|---|---|
| Predicted against observed bytes moved per token | kept as four quantities: bytes requested, logical reads, physical device traffic (needs a counter), resident-page changes (mincore, deduplicated per page); faults on fault-path arms | per token and cumulative over a decode window |
| Predicted against observed token latency | unwitnessed wall time, p50 and p95 over repeats | error bound fixed before any arm |
| Hit rate in execution against simulated | the same trace in both | tests the reuse model, not the policy |
| Policy speedup (H2) | unwitnessed latency against the declared baseline policy | each policy arm with its own baseline |
| Output identity | generated ids, logit bit-identity, routing fingerprint | a policy that changes output is out |

## 7. Questions the protocol freeze must answer

1. Is option 0, the feasibility study on `.s7`, the first investigation?
   Which later option follows it, if any?
2. What implementation, if any, is authorised before the freeze? For option
   0 the likely candidates are a committed cold-state method and a
   machine-state record; an exporter, device-read counter, fetch path or
   balloon is needed only for later gates (§3.1). Any of them is reviewed
   separately from the measurement.
3. Whether `.s7`'s payload is re-hashed against its declared digest, and
   whether its system graph is accepted as the subject's definition
   [R §7 Q1].
4. The cold-state method, and its witness.
5. The baseline policy for H2, and the candidate policy set: main only, or
   including the branch's `Coalesced` and `Gated`.
6. Repeats, the quiet gate, and how the error bound and the speedup
   criterion are fixed. They are fixed before any arm runs, and not from the
   numbers in [R].
7. An independent evaluator: who scores H1 and H2, from what records.

## 8. Sources checked for this scoping

- `ls /Volumes` gives `Macintosh HD` and `Recovery`.
- `sysctl hw.memsize` gives 137,438,953,472 (128 GiB).
- `df` gives 303 GiB free on the data volume.
- `ls ~/chris-models` shows no entry larger than RAM. `.s7` is 92 GiB on
  disk by `du`.
- `docs/frontier-execution-programme.md` §1 (sizes), §5 (layer types; DSA
  indexer "none"), §6 (rungs), §15 (disk).
- `docs/glm5-flash-funnel.md` header: "GLM53-PHYSICAL defined and entirely
  unrun".
- `d29b85e6`: "register GLM-5.2's DSA indexer, extraction-tier only". No
  later commit adds an executor (`git log` and `git grep` at `bcb67863`).
- Everything else is cited to the merged reconnaissance.
