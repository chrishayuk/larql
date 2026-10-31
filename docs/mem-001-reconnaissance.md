# MEM-001 reconnaissance: what LARQL can already measure about expert residency

**Class: RECONNAISSANCE.** Dated 2026-10-31 at main `51579782`. Read-only. It
is not sealed and not experimental: no arm has run for this document, no
threshold is chosen here, and no cache, policy or runtime is recommended. §7
lists the questions the freeze must answer.

**The question MEM-001 will ask.** How much physical memory does a routed
(MoE) model need to sustain a given decode rate? And does that relationship
survive once the model's bytes exceed what the machine can hold?

Before a protocol can be frozen, three things have to be known:

- which residency, fault and movement facts LARQL can **observe**, and which it
  only **infers** (§1);
- whether the one prior latency result in this area reproduces (§2);
- how a memory budget can be imposed and checked (§3), and what an external
  baseline can be compared on (§4).

**Headline.**

- **The prior cold-latency result does not stand yet.** PREFETCH-1's "advise
  halves the cold token" (1253 → 598 ms) was measured UNQUALIFIED: Spotlight
  was indexing the files that eviction had just read. PREFETCH-2 did not
  reproduce it. Its demand arm ran at 602 ms for the same 73,518 faults. The
  advised arms did not change; the demand arm moved (§2). Until the unrun
  contention control is run, no cold-latency difference between access
  policies is established.
- **Three instruments this programme was going to lean on are not on main.**
  - PREFETCH-2's mechanism and notes are on a branch.
  - The BW10 movement ledger is on a branch.
  - MODALITY-RESIDENCY-1a was **previously reported** (in project notes, on
    a 45.68 GB expert bank), but its **source artifact was not recovered**.
    Absence from the searched refs does not show the work never happened;
    it shows the evidence is not available to cite.
  (§1.)
- **Nothing in LARQL enforces a physical residency budget at run time.**
  `ResidencyBudget` plans and refuses before I/O. `reconcile` compares declared
  bytes with bound objects, not with resident pages. Mapped pages are left to
  the OS (§3).
- **No container on hand is larger than this machine's 128 GB.** The largest
  is 92 GB: the PREFETCH subject, which is a 94.22 GB address space. On this
  machine MEM-001 can test the constrained-budget regime but not a model
  genuinely larger than RAM (§3).
- **Of the three external reports, only one measures real compute under real
  paging on Apple Silicon** (§4).
- **The smallest discriminating experiment** is a trace-predicted versus
  observed fault comparison, run under four cache states, with its own demand
  baseline and a contention control (§6). It needs no new cache.

Paths are at `51579782`. An inventory was collected by one read-only search,
and every load-bearing claim below was re-read by hand against `origin/main`
or the named branch. Numbers quoted from notes files are copied, not
recomputed.

## 1. Instrument inventory: observed, inferred, limited

There are six columns.

- **Implementation:** where the instrument lives, and whether it is on main.
- **Existing evidence:** the recorded result that has used it.
- **Directly observes:** what the instrument itself reports.
- **Inferred from it:** what a reader is tempted to conclude from it, but the
  instrument does not observe.
- **Known limits:** documented failures and blind spots.

"Not found" means a search of `origin/main`, the remote branches and local
history found nothing.

| Instrument | Implementation | Existing evidence | Directly observes | Inferred from it (not observed) | Known limits |
|---|---|---|---|---|---|
| `resident_bytes()`, mincore over one stored region | `represent/physical.rs`; on main | PREFETCH-1 resident deltas (1.199–1.206 GB) | which pages of **one mapped region** are resident at the instant of the call | the process's physical footprint; that this run caused the residency | unified buffer cache keeps file pages across processes; nothing between snapshots; returns `None` without an OS report |
| Binding line "X GB address space over N regions, Y GB resident" | VINDEX3 exec; on main | PREFETCH-1 cold witness: 94.22 GB / 19,968 regions / 0.000 GB, 3/3 | the sum of the above over the plan's bound regions | total memory in use | KV, scratch and Metal buffers outside the plan are not counted |
| `MappedAccess {Demand, Advise, Touch}` | `opplan/exec/realization.rs:247`, `exec/prefetch.rs`; on main | PREFETCH-1 (§2) | which access policy the plan **issued** (`madvise(WILLNEED)` per range; one byte touched per page) | that advised pages were resident before projection | PREFETCH-1 did not witness it; PREFETCH-2's `probed_resident` (branch only) samples 4 pages per run |
| Major/minor fault counts (getrusage) | PREFETCH probes; on main | PREFETCH-1 and -2: ~73,518 major faults per cold token | page-ins that went through the fault path, for the whole process | bytes read from storage; which expert caused a fault | readahead, `pread` and cluster reads move bytes without one fault per page (16.3 KB per fault in PREFETCH-1); counts are per process, not per region |
| `evict()` (msync INVALIDATE + `MADV_FREE_REUSABLE` + DONTNEED, then mincore) | `examples/glm_moe_residency/probe.rs:116`; on main | PREFETCH-1 cold state | that the target regions read 0 resident afterwards | a cold **system** | other files, SSD-side caches and Spotlight's reaction are untouched; DONTNEED is lazy on Darwin; PREFETCH's eviction streams 184 GB first (§2) |
| Routing trace `LARQL_MOE_ROUTE_TRACE` → JSONL `{layer, seq, experts}` | `larql-compute/src/moe_route_observe.rs`, wired in CPU MoE, Metal decode and `kquant_forward`; on main | BW11-1 `dec-bench window-union` | the experts the router selected, per layer and position | which bytes were read (depends on layout, §1.1); routing in another runtime | no byte or time information |
| VINDEX3 routing capture `start_capture` / `take_capture` / `fingerprint` | `opplan/exec/routing_trace.rs`; on main | none found recorded for residency work | the same for VINDEX3 plans, with a fingerprint for identity | as above | as above |
| Retention simulator (DEC-9A): Min / Lru / Lfu / Random / StaticOracle, compulsory-miss floor | `larql-cli/.../k3_ledger/retention.rs`; on main | no recorded result found in `docs/` | the hit rate a policy **would** get on a recorded trace, in experts | latency; bytes on a device | `K3_ROUTED_EXPERT_BYTES = 17,547,264` is a constant, and it is not the PREFETCH subject's figure (§2) |
| `ResidencyBudget` / `select_realizations_within` / `BudgetDeficit` | `opplan/exec/accounting.rs:360`, `prepared/mla.rs`; on main | Residency-into-K3 plans | that **declared** bytes fit a declared budget, or a refusal with the deficit, before any payload byte | physical enforcement during execution | no runtime cap exists (§3) |
| `expectations` / `reconcile` | `prepared/residency.rs`, `accounting/reconcile.rs`; on main | Residency-into-K3 plans | declared bytes against **bound objects** | that bound objects are resident | does not read mincore |
| BW10 movement ledger (semantic / physical / useful bytes; DRAM/NVMe/network tiers; `Regime` gate) | `larql-compute/src/movement_ledger/`; **branch `feat/bw10-ledger-and-exec-policy-seam` only** | BW-A gate on gpt-oss-20b, recorded in project notes only, not in the repo | instrumented byte classes for **MoE experts only**; refuses a verdict line without a declared regime | totals for the other weight surfaces | attention, dense FFN, LM head, KV, embeddings and norms are uninstrumented, so physical totals are floors |
| PREFETCH-2 additions (`Coalesced`, `Gated`, `probed_resident`, `--witness-residency`, `scripts/evict-bank.sh`, `contention.sh`) | **branches `k3-prefetch-2` (`aae20ae3`) and `wip/k3-prefetch-2-2026-09-23` only** | PREFETCH-2 preliminary round (§2) | as described in its notes | — | measurement PAUSED since 2026-09-06; `--witness-residency` costs ~70 ms per token (PF-N4), about one warm token; `contention.sh` unrun |
| MODALITY-RESIDENCY-1a (vision pages untouched by text execution) | **source artifact not recovered**: no file, branch or history entry in the searched refs (`origin/main`, remote branches, `git log --all -S`, local worktrees' `docs/`) | **previously reported** in project notes (2026-08-23, Gemma 4 26B-A4B container, 45.68 GB expert bank); not citable as repository evidence | — | — | an evidence gap, not a negative result; re-run and record it, or drop it as a premise (§7, Q8) |

**No runtime expert cache exists.** `MADV_DONTNEED` appears only in the
server's `release_mmap_after_request` and in the probe's `evict()`.

### 1.1 Expert layout, which decides bytes per expert

There are two layouts. Which one a container uses matters, because the
llama.cpp report in §4 measured 111× read amplification from GGUF's
interleaved expert tensors.

- **`Fused`** (packed GPT-OSS): one `gate_up` stream and one `down` stream.
  Each is expert-major inside its tensor, so one expert's gate/up and down are
  **not** adjacent.
- **`Separate`** (per-expert bank, K3): three whole matrices per expert, each
  its own stored operand. The real container has 19,968 operands.
  - Adjacency on the real container is **inferred, not guaranteed**:
    PREFETCH-2 coalesced 624 ranges into 202 requests, which fits each expert
    being one run.
  - No writer-side ordering guarantee was found. Only the synthetic fixture
    states adjacency (`runtime/fixtures/synthetic.rs:27`).

### 1.2 Coverage against the three review questions

| Question | What exists | The gap |
|---|---|---|
| **Residency:** can the physical working set of a routed model be identified, not just its mapped or declared footprint? | mincore per region (snapshot), the binding-line sum, the routing trace (which experts), and the retention simulator (which experts a budget would keep) | no time-resolved residency (snapshots only, and the witness costs ~70 ms per token); no accounting outside the plan's regions; region sizes per expert are unreconciled (§2) |
| **Movement:** can storage reads, faults and latency be attributed to specific expert accesses? | process-wide fault counts; resident delta **per region**, before against after; per-position routing | faults are not attributed to regions or experts; no device-level read count; no per-layer timing of fault stalls on main. The BW10 ledger's byte classes are the nearest thing, and they are branch-only and cover experts only |
| **Causality:** can a residency change be shown to cause a throughput difference? | access-policy arms with a shared cold witness (PREFETCH-1/2); eviction | both rounds are UNQUALIFIED; there is no contention control yet; the eviction method itself provokes contention; there is no runtime residency budget to intervene on (§3) |

**On present instruments:**
- Residency: answerable at **snapshot granularity**.
- Movement: answerable only at **process granularity**, so the gap is
  attribution.
- Causality: not yet answerable, because the only intervention available is
  the access policy.

These are candidate instrumentation gaps for review. None is proposed for
building here.

## 2. PREFETCH-1 against PREFETCH-2

**Status:**
- **PREFETCH-1** is merged, and its notes are on main.
- **PREFETCH-2**'s mechanism is committed (`aae20ae3`) on a branch, not main.
  Its measurement has been **PAUSED since 2026-09-06**. Its one preliminary
  round did **not reproduce** PREFETCH-1's cold-token improvement.

**The subject.** Neither notes file names the model. The eviction protocol
streams "the HF checkpoint (92 GB) and the .s6 container (92 GB)" through the
page cache. The only model on disk with both at that size is
Kimi-Linear-48B-A3B (`.s6.vindex3` and the HF directory, 92 GB each). Its
26 routed layers × 256 experts × 3 matrices would give the 19,968 regions.
This identification is an **inference**; the freeze must pin it by digest.

**Two byte figures do not reconcile, and both feed §6:**

- **Requested versus paged.** PREFETCH-2's coalesced arm requested
  **2.948 GB**, which is what 8 experts × 26 layers = 208 experts would occupy
  at BF16. Every arm of both rounds showed a resident delta of only
  **1.206 GB**. That is not automatically an error. Four quantities are
  defined differently and cannot stand in for one another:

  | Quantity | Source | What it counts |
  |---|---|---|
  | **Requested bytes** | `PrefetchReport.requested_bytes` (branch) | ranges the plan asked the OS to bring in |
  | **Resident delta** | mincore before against after, over bound regions | pages of those regions that became resident |
  | **Fault activity** | getrusage major faults × page size | page-ins through the fault path, process-wide |
  | **Device reads** | not measured by any instrument here | bytes the SSD actually delivered |

  Over-requesting, a quantised `.s6` payload smaller than the advised ranges,
  pages already resident, and readahead outside the bound regions are all
  consistent with the gap. None was checked here.
- **The simulator's constant.** `K3_ROUTED_EXPERT_BYTES = 17,547,264` is not
  this model's figure. At BF16, one Kimi-Linear expert's three matrices are
  2304 × 1024 × 3 × 2 = 14,155,776 bytes.

Converting simulated misses into bytes needs the container's real region sizes.

Note also that the eviction method itself reads 184 GB through the page cache
just before a cold arm. That read is what Spotlight was reacting to.

Same container, the same 1.2 GB per cold token, the same ~73,518 major faults
in every arm of both rounds:

| | PREFETCH-1 (on main, `k3-prefetch-notes.json`) | PREFETCH-2 preliminary (branch, `k3-prefetch-2-notes.json`) |
|---|---|---|
| Machine state | UNQUALIFIED: load 5–13 from `mds_stores` indexing what eviction had just read | UNQUALIFIED: peer rustc build, `mediaanalysisd`, `mds_stores`; qualified warm arms refused; cold arms run `--unquiet-ok` |
| Cold state | mincore 0.000 GB resident over 94.22 GB / 19,968 regions, 3/3 | via `evict-bank.sh` |
| Demand, cold token | **1253 ms** (loop 1213.6 ms) | **602 ms** |
| Advise, cold token | 598 ms (prefetch stage 504.5 ms) | — |
| Coalesced / Gated | — | 591 / 607 ms (prefetch 514 / 528 ms) |
| Touch, cold token | 1358 ms | — |
| Warm token | 76 / 87 / 80 ms | 74 / 83 / 80 ms |

**What moved was the demand arm.** PREFETCH-2's own reading (PF2-N1) puts
demand at 8.2 µs per fault against PREFETCH-1's 16.4 µs. The advised prefetch
stage barely changed (504 → 514 ms, about 2.4 GB/s).

The simplest reading consistent with both rounds:

1. PREFETCH-1's demand arm was slowed by contention that the advised arms
   escaped, because a large advised read overlaps poorly with page-at-a-time
   faults competing against Spotlight.
2. On a quieter device, demand faulting and advised prefetch arrive at about
   the same cold time.

This is a hypothesis, not a finding. Both rounds are UNQUALIFIED, and the
discriminating control (`contention.sh`: demand on a quiet device, and demand
against a deliberate competing sequential read) is **written and unrun**.

**Carried rules.**
- PREFETCH-2: "no cold latency claim … without its campaign's own demand
  baseline."
- PF-N2: concurrent faulting does not parallelise page-in of one mapping on
  macOS.
- PF-N3: advise costs 8 ms of an 80 ms warm token.

## 3. Imposing a memory budget

There are two regimes, and they make different claims:

| Regime | Claim it can support | Available on this machine (128 GB RAM, 352 GB free disk) |
|---|---|---|
| **A. Model bytes exceed physical RAM** | feasibility of oversized execution | **No subject.** The largest of the 52 entries in `~/chris-models` is 92 GB by `du`, which overstates APFS clones (§7, Q1) |
| **B. Model fits; expert residency is held below its footprint** | how paging and caching behave under controlled pressure | Yes, given a method from the table below |

Methods for regime B:

| Method | Constrains | Verified by | Known interference |
|---|---|---|---|
| Software expert budget inside LARQL (evict outside a resident set) | the expert bank only | mincore on the bank's regions; faults | **does not exist** (no runtime cap); eviction is lazy on Darwin, so "evicted" must be witnessed. It is a software budget, never an RAM limit |
| System-wide memory balloon (wired allocation) | all of RAM | `vm_stat`, mincore | changes Metal allocation headroom, the file cache for every process and the compressor; it confounds what it is meant to isolate |
| A smaller-RAM machine | all of RAM, physically | the machine | a different device, SSD and GPU: a separate result, not an arm |
| No macOS per-process RSS cap equivalent to a cgroup memory limit was found | — | — | — |

A software budget and an actual RAM limit stay two distinct claims in any
protocol.

## 4. External reports (reported findings, not LARQL results)

All three are `ggml-org/llama.cpp` discussions, the primary sources, read on
2026-10-31.

**Every figure in this table is the poster's own report and is independently
unverified.** None was reproduced on this machine or by LARQL. The last column
records how the source itself describes the figure.

| Discussion (primary source) | What it is | Reported (unverified) | Measured or simulated, per the source |
|---|---|---|---|
| [#28248](https://github.com/ggml-org/llama.cpp/discussions/28248) "Persistent expert slot pool for MoE CPU offload" (2026-09-02) | VRAM slot pool for experts that live in CPU RAM; **no SSD tier** | single RTX 4090, Qwen3.8-Flash-Next UD-Q3_K_XL, 8K context: tg512 11.55 ± 0.41 → 21.21 ± 0.46 t/s at 64 slots; reported bit-exact on CUDA and Metal (AMD dGPU) | reported as measured. A commenter reports, also unverified, that PPL moved <1% while argmax moved on 2–6% of tokens in a related fork arm. That argues for token agreement, not PPL, as the acceptance test |
| [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) "MoE offload to disk with on-demand paging" (2026-05-19) | Metal shared-memory slot pool + `pread` from GGUF, on Apple Silicon | Qwen3-30B-A3B Q6_K, M3 Pro 36 GB: vanilla 38.1 tok/s; **128 slots (no saving) 30 tok/s**; 64 slots 27.5; 32 slots 22.1. M1 Pro 16 GB "13 tok/s" is a one-line update with no table | reported as measured; the 128-slot row implies a ~21% pool overhead before any memory is saved |
| [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149) "Expert-Aware SSD Streaming" (2026-08-15) | standalone prototype (TinyGiant) | M1 Pro 16 GB: 2.97 GB/s sequential, 6.06 GB/s double-buffered; headline 4.7 tok/s; GGUF interleaving makes one expert read via mmap touch 6,912 pages (111×) | the headline is **real I/O with simulated compute** ("matrix multiplication is not wired up"). The pipelined C benchmark with real BLAS reports 2.20–4.46 tok/s across 50–100% hit rates |

**What "identical model bytes" can mean across runtimes.** LARQL executes
VINDEX3 containers and llama.cpp executes GGUF. The same weights at the same
quantisation can be compared. The same file cannot, and **layout is one of the
variables under test** (§1.1). The freeze has to define identity as the same
source weights, the same quantisation and the same prompt tokens, with layout
declared per arm. #23324 is the only report that runs real compute under real
paging on Apple Silicon, so it is the natural candidate external arm.

**The two runtimes' instruments are not equivalent, and none of the following
assumes they are.**

| Measurement | LARQL | External runtime (e.g. #23324's fork) | Comparable as is? |
|---|---|---|---|
| Wall time per token, decode rate | yes | yes (`llama-bench`) | only under one harness and quiet gate; the two runtimes' own timers measure different spans |
| Process faults (getrusage) | yes | readable from outside for any process | yes, process-wide |
| File-page residency | mincore inside the process | not built in. mincore is per caller mapping, but file pages are shared, so an **observer process** mapping the same file sees the cache state | yes, if the same observer serves both arms; this needs checking on macOS before it is relied on |
| Expert ids per position | routing trace | not known to be exposed. #28248 mentions a per-layer id readback in its fork | not until shown |
| Bytes per expert | region table | GGUF tensor offsets; **the layout differs by design** | by computation, not observation |
| Declared budget or refusal | yes | slot count `n_slots` | no: different mechanisms that are not the same claim |

**Feasibility:**
- **LARQL side:** feasible on main, except the contention and eviction
  scripts (branch).
- **External side:** feasible for timing, faults and observer residency. Not
  yet shown for routing identity, which §5.6 makes a precondition for any
  working-set comparison.

## 5. Confounds the protocol must separate

1. **File cache versus process residency.** On macOS, mapped file pages are
   unified-buffer-cache pages. A "warm" second process can inherit residency it
   never paid for. Any residency advantage has to beat what the cache provides
   for free.
2. **Device contention.** Spotlight and media analysis react to large reads;
   §2 is the worked example. This needs a quiet gate and a deliberate
   contention arm.
3. **Instrument cost.** `--witness-residency` at ~70 ms per token is about one
   warm token. Witnessing belongs before and after an arm, not inside it.
4. **Fault counts as a proxy for bytes.** Readahead and clustering decouple
   them (§1). Device-level reads need a separate source.
5. **Decode cost unrelated to memory.** Where decode is limited by dispatch,
   not bytes, a perfect residency policy changes little. The fully resident
   arm is LARQL's own ceiling, not the hardware's, and has to be measured,
   not assumed.
6. **Routing identity across runtimes.** Two runtimes can route the same
   prompt differently once numerics differ. Working-set comparisons need
   per-position routing agreement recorded first.
7. **One run is not a distribution.** Cold arms are expensive to repeat; the
   freeze must set repeats and uncertainty before scoring.

## 6. Candidate: the smallest experiment that discriminates

**This is a candidate for review, not an authorised experiment.** It sets no
thresholds, picks no winning policy and runs nothing. If it is accepted, it
enters a freeze with the §7 answers before any arm runs.

MEM-001's question is whether residency can be **predicted**. The smallest
test needs no new cache.

1. **Record** a routing trace on one routed model and one frozen prompt set
   (`LARQL_MOE_ROUTE_TRACE`, on main).
2. **Predict**, offline with the retention simulator (on main), the expert
   misses for each policy at each budget. Convert misses to bytes using the
   container's per-expert region sizes, measured not assumed.
3. **Observe** major faults, resident delta (mincore before and after) and
   token latency under four cache states, each with its own demand baseline:
   - (a) evicted and witnessed cold;
   - (b) file-cache warm, new process: run, exit, re-run without eviction;
   - (c) fully resident, all pages touched;
   - (d) evicted under the deliberate contention read of `contention.sh`.
4. **Compare** predicted with observed bytes per token.

How the outcomes read:

- **Predicted ≈ observed (bytes), and (b) ≈ (c) (latency):** residency is
  predictable from routing, and on this machine the OS cache already provides
  the warm benefit. A residency planner earns its keep only in regime B or A,
  which then becomes the next rung.
- **Predicted ≠ observed:** the layout, readahead or the fault proxy is wrong.
  §1.1 and §5.4 say which to check first. That is a useful negative.
- **(a) ≈ (d):** contention does not explain §2, and the PREFETCH-1/2
  disagreement is still open.
- **(a) ≪ (d):** this supports reading PREFETCH-1's halving as contention.
  Settling §2 also needs the advised arm under the same contention.

"≈" and "≪" are placeholders. What counts as either is a freeze decision
(§7, Q7), not this document's.

This reuses the PREFETCH subject container, the PREFETCH probe arms, the eviction routine
and the simulator, all on main except `contention.sh` and `evict-bank.sh`,
which would be brought from the PREFETCH-2 branch with their provenance.

## 7. Questions for the freeze

1. **Regime A subject.** Is there a container on hand larger than 128 GB,
   and is it runnable end to end? If not, MEM-001 is a regime-B programme on
   this machine, and it should say so in its title.
2. **Regime B method.** Is a software expert budget (to be built, witnessed by
   mincore) acceptable as MEM-001's pressure method? If so, its claims are
   phrased as software budgets, never as RAM limits.
3. **PREFETCH-2.** Is its branch merged first, carried in, or re-derived,
   given that its mechanism is committed and its measurement is paused?
4. **BW10 ledger.** Is it brought to main as part of MEM-001, or does MEM-001
   use faults and mincore only, accepting §1's limits?
5. **External arm.** Is it #23324's fork, pinned to a commit, on the same
   source weights and quantisation as a LARQL arm? Which model can both
   runtimes load?
6. **Quiet gate and repeats.** How many cold and warm repeats per arm, and
   under what qualification? PREFETCH-2 proposed `--repeat 33` warm.
7. **Outcome criteria.** What agreement between predicted and observed bytes
   counts as "predictable"? It must be chosen before any arm in §6 runs, not
   from this document's numbers.
8. **MODALITY-RESIDENCY-1a.** Is it re-run and recorded, or dropped as a
   premise?
9. **Bytes per token.** Before any prediction is scored, reconcile §2's
   2.948 GB requested against 1.206 GB paged from the container's region
   table.

## 8. Review outcome (maintainer, 2026-10-31)

**Accepted as the reconnaissance record. It is not a protocol, and MEM-001 is
not frozen.** The PR stays in draft until the document has been checked
against the implementation and source evidence by a review independent of the
agent that wrote it.

**Readings carried forward:**
- PREFETCH-1's 1253 → 598 ms is a **historical observation, not a baseline
  benefit**.
- The §2 byte gap is a **definitions-and-coverage problem** before it is an
  error.
- The model identity must be **pinned** (exact model, container bytes, tensor
  layout, digest) before any scored run.

**Next: two small investigations, before any freeze.** Neither builds a cache
or a runtime budget, and neither needs a cross-runtime comparison yet.

- **MEM-001A — PREFETCH reproducibility.** Re-establish the actual cold-start
  conditions, and isolate contention, eviction effects and OS caching, with
  the existing PREFETCH instruments and recorded uncertainty. The aim is to
  explain why the two rounds disagreed, not to produce a new headline.
- **MEM-001B — Working-set accounting.** Pin model and container identity,
  reconcile per-expert region sizes against routing traces, and define which
  quantity is requested bytes, resident pages, fault activity and storage
  movement. The simulator's expert-size constant is not a measurement.

MEM-001 is frozen only once the reproducibility, accounting and identity
problems are resolved or explicitly bounded.
