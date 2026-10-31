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
   bytes-moved prediction would then have to model an unspecified macOS
   replacement policy. Under an explicit fetch path, LARQL decides, and the
   prediction becomes checkable. That path exists only in the GLM per-layer
   probe, not in VINDEX3 [R §2.1, §6.3].
5. **H1's time prediction meets the variance MEM-001 left unexplained.**
   Four cold observations of one Kimi token give 8.2–21.7 µs per demand
   fault [R §2.4]. A bound fixed in advance cannot be credible until that
   spread is characterised.

So the experiment cannot be frozen on GLM, and it cannot be run in regime A
on this machine at all. §5 sets out the subject choices that remain.

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
| Routing trace, frozen | VINDEX3 `routing_trace` capture plus fingerprint [R §1.1] | **in memory only; no exporter** [R §1.4]. "Given a frozen routing trace" needs one |
| Expert layout, bytes per expert | region table; `index.json` [R §1.5] | expert adjacency under the writer's name ordering is **unchecked** [R §1.5] |
| Bytes moved | mincore resident deltas; `ru_majflt` [R §1.1] | faults stop being a byte proxy once the fetch path changes; summed mincore double-counts boundary pages; **no device-read counter** [R §1.1, §1.4] |
| Storage characteristics | `examples/readrate.rs` (serial per-fault rate); `scripts/dec8_2_fio_sweep.py` (no result on main) [R §1.2] | queue-depth and request-size curves for the internal SSD are not recorded |
| Reuse simulation | `glm_expert_reuse.py`; retention simulator (`--expert-bytes`); `larql moe-locality` [R §1.2] | none reads a VINDEX3 trace [R §1.4] |
| A memory limit | none in VINDEX3 [R §1.3] | **missing**: a runtime cap, an explicit fetch path, a balloon or another machine |
| Execution policies to choose among (H2) | `MappedAccess {Demand, Advise, Touch}` on main [R §1.1]; `Coalesced`, `Gated` on branch `k3-prefetch-2` only [R §1.2] | no explicit-fetch policy in VINDEX3; the branch policies are unmerged |
| Output identity | routing fingerprint, generated ids, logits bit-identity [R §2.3] | — |
| Latency | per-step wall time; the residency witness costs ~70 ms per token [R §2.3] | an unwitnessed arm is required; one was never recorded for round 2 [R §1.4] |
| Machine state | the quiet gate refuses unqualified warm arms [R §2.3] | **no device-I/O or contention record**; `contention.sh` was never committed [R §1.4] |

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
   Movement becomes LARQL's decision, so bytes-moved prediction is exact by
   construction, and the time model is what is tested. It extends GLM §14
   [R §6.3]. It needs a new realisation in VINDEX3, which is
   implementation.
4. **H1 timing without memory pressure first.** On `.s7`, predict the cold
   whole-token latency from the trace, the layout and measured per-fault or
   per-read costs, and test the prediction against repeated cold runs. That
   needs an exporter and an eviction method, but no limit. It overlaps the
   unexplained cold-run variance [R §2.4], and so subsumes MEM-001A's
   question.

**Recommendation for the freeze discussion.** Option 4 first, then option
3:

- Option 4 tests the time half of H1 on a subject that executes today. It
  also decides whether any fixed error bound is attainable, given the
  8.2–21.7 µs per-fault spread.
- Option 3 is the cleanest route to a memory limit LARQL controls. Its
  movement is predictable by construction, unlike the kernel's replacement
  policy.
- GLM stays the regime-A goal, but it is gated on implementation, not on
  measurement.

## 6. Candidate outcome measures (no thresholds)

| Measure | Basis | Notes |
|---|---|---|
| Predicted against observed bytes moved per token | mincore deltas, deduplicated per page; faults on fault-path arms; read accounting on fetch arms | per token and cumulative over a decode window |
| Predicted against observed token latency | unwitnessed wall time, p50 and p95 over repeats | error bound fixed before any arm |
| Hit rate in execution against simulated | the same trace in both | tests the reuse model, not the policy |
| Policy speedup (H2) | unwitnessed latency against the declared baseline policy | each policy arm with its own baseline |
| Output identity | generated ids, logit bit-identity, routing fingerprint | a policy that changes output is out |

## 7. Questions the protocol freeze must answer

1. Which subject and memory regime: option 1, 2, 3 or 4 of §5?
2. What implementation, if any, is authorised before the freeze (exporter,
   eviction method, fetch path, balloon)? It is reviewed separately from the
   measurement.
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
