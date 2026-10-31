# MEM-001 reconnaissance: what LARQL can already measure about expert residency

**Class: RECONNAISSANCE.** Dated 2026-10-31 at main `51579782`. Read-only. It
is not sealed and not experimental: no arm has run for this document, no
threshold is chosen here, and no cache, policy or runtime is recommended. §7
lists the questions a later freeze must answer.

**Revision 3 (correction).** Each earlier revision failed an independent
source audit, and every revision and audit stays readable:

| Revision | Commit | Audit | Result |
|---|---|---|---|
| 1 | `43388264` | [`…-audit-1.md`](mem-001-reconnaissance-audit-1.md) | 56 findings, 7 BLOCKING |
| 2 | `b82d70b3` | [`…-audit-2-phase1.md`](mem-001-reconnaissance-audit-2-phase1.md), [`…-audit-2-phase2.md`](mem-001-reconnaissance-audit-2-phase2.md) | 67 findings, 3 new BLOCKING; audit 1's #31 and #34 partly resolved |

This revision adds three prior results that revision 2 omitted: a fourth cold
Kimi run (V3), the lift2 clone record, and a Gemma routing-locality result. It
separates witnessed, unwitnessed and witness-adjusted latency, states the
provenance of every prior result, and makes audit 2's non-blocking
corrections. Each correction was re-checked against its primary source, not
copied from an audit's suggested wording. Findings are cited as **[A1 #n]**
(audit 1) and **[A2 P1-n]** (audit 2, phase 1).

**The question MEM-001 will ask.** How much physical memory does a routed
(MoE) model need to sustain a given decode rate? And does that relationship
survive once the model's bytes exceed what the machine can hold?

LARQL has **already shown, for one layer of one model**, that the pages a
bound expert selection touches can be predicted from the plan (§2.1). So the
open questions are:

- whether that prediction generalises across models, layouts, access paths
  and runtimes;
- whether correct prediction translates into reliable decode-rate gains.

**Headline.**

- **Prior LARQL evidence exists.** All of it is **BACKGROUND ONLY**: none is
  a reproducible baseline (§2.0).
- **GLM-5.3-Flash, one routed layer, readahead off**
  (`docs/frontier-execution-programme.md` §11–14) [A1 #24]:
  - predicted against observed residency: coverage **1.0003**, with
    unselected residency 0.0 MiB;
  - the selection's logical bytes equal the bytes paged in: **ratio 1.000**;
  - demand, advise and touch the same **to within 0.8 %** on an idle machine;
  - an explicit `pread` fetch giving a **2.36× faster** cold routed stage;
  - LRU and Bélády hit rates at six budgets over a 128-token prompt.
- **There are four cold-run observations of one Kimi-Linear token, not
  three** [A2 P1-47]. They are a V3 run (§2.2) and three PREFETCH rounds
  (§2.3), all on the same subject and prompt. They are **not controlled
  repetitions**: conditions, instrument overhead and prior residency differ.
  Cold demand cost per major fault, whole token:

  | Observation | Demand per fault | Basis |
  |---|---|---|
  | V3 rung 7 | 21.7 µs | unwitnessed |
  | PREFETCH round 1 | 17.0 µs | unwitnessed |
  | PREFETCH round 2 | ~15.4 µs | witness-adjusted estimate (16.4 µs raw, witnessed) |
  | PREFETCH round 3 | 8.2 µs | unwitnessed |

  The existing evidence **cannot isolate the cause** of this variation.
- **Advise against demand.** Within each PREFETCH round, rounds 1 and 2 show
  the advised arms at about half the demand cold token. Round 2's figures are
  witnessed: the halving holds with the ~70 ms witness removed (2.0–2.1×).
  Round 3 does not show it for the token, but still does over the prompt
  (1.25–1.43×) (§2.4).
- **The byte accounting is reconciled within the PREFETCH comparison, not
  across the V3 and PREFETCH lineage** [A2 P1-47, A1 #31]:
  - within PREFETCH, 59.1 % of the 2.948 GB selection was resident from the
    prompt before the measured token, and 2.948 × 0.409 = 1.206 GB, the
    observed paged bytes;
  - V3, on the same token and prompt, paged 0.976 GB, with about 67 %
    resident from the prompt. No source explains the difference, and V3
    recorded no routing fingerprint.
- **Subject identity** [A2 P1-37, A1 #34]. The subject,
  `Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`, is no longer on disk. Main
  records it as an APFS clone of `.s6`, matching in segment size and
  spot-checked content but not by digest. `.s6` and `.s7` share a recorded
  bank digest. No lift2 digest exists (§2.2).
- **A recorded locality result bears directly on caching** [A2 P1-67]. On
  Gemma 4 26B-A4B (legacy path, one ~72-token prompt), the union of selected
  experts reached 124.4 of 128 per layer. The diagnosis concludes there is no
  small cacheable hot subset (§2.5). That challenges a static hot-expert
  cache. It does not rule out short-window temporal locality or prefetching.
- **LARQL has runtime expert caches, but none bounds VINDEX3 mapped-page
  residency** [A1 #21] (§1.3).
- **Some instruments that a protocol might need are missing** (§1.4):
  - a VINDEX3-routing → retention-simulator path [A1 #11];
  - the contention script, which is in no git ref [A1 #18];
  - a working eviction method on an idle machine. Main records advice-based
    eviction never working on Darwin, and invalidation stopping when the
    machine was idle [A1 #9].
- **Oversized execution and constrained residency are separate regimes**
  (§3). A genuinely oversized subject is *recorded*: GLM-5.3-Flash has a
  283.5 GiB expert estate on an external volume. That volume was not mounted
  on 2026-10-31, and only one layer has ever been probed.

**How the sources were found.** Paths and line numbers are at `51579782`. The
search was by physical and implementation terms, not only by programme name:

- `git grep -n -i -E 'mincore|madvise|WILLNEED|ru_majflt|major_fault|resident|pread|evict'`
  over `crates`, `scripts` and `docs`;
- the same across remote branches;
- every finding of A1 and A2, re-read against its cited source.

A separate frozen source inventory bounds the completeness check for the
next audit (§8). Numbers quoted from notes files are copied, and the
arithmetic is shown where the document derives anything.

## 1. Instrument inventory

### 1.1 VINDEX3 mapped-page instruments, on main unless marked

Columns:

- **Existing evidence:** the recorded result that has used the instrument.
- **Directly observes:** what the instrument itself reports.
- **Inferred from it:** what a reader is tempted to conclude, but the
  instrument does not observe.

| Instrument | Implementation | Existing evidence | Directly observes | Inferred from it (not observed) | Known limits |
|---|---|---|---|---|---|
| `resident_bytes()`, mincore over one stored region | `format/vindex3/represent/physical.rs:269` | V3 and PREFETCH residency deltas (§2.2, §2.3) | which pages of **one mapped region** are resident at the instant of the call | the process's physical footprint; that this run caused the residency | `None` without an OS report. **Summed callers map `None` to 0** (`weights/loaded.rs:37`, `prepared/residency.rs:417-423`), so a "0.000 GB resident" witness cannot tell cold from a failed read [A1 #4]. It counts **whole pages** overlapping the region (`physical.rs:283-301`), clipped only to the region's length, so two unaligned neighbours both count a shared boundary page: **summed figures are not de-duplicated physical bytes** [A2 P1-4] |
| Binding and step lines "X GB address space over N regions, Y GB resident" | `larql-cli/.../vindex3_cmd/generate/residency_curve.rs:195` | V3: 94.22 GB, 19,968 regions, 0.000 GB resident at binding. PREFETCH cold witness: the same | the sum of the above over the plan's bound regions | total memory in use | KV, scratch and Metal buffers are outside the plan and not counted; both caveats above. The per-step read is taken after the step's timer stops (`residency_curve.rs:306-318`; the same order at `68abfd70`'s `generate.rs`) |
| Fault and RSS per pass (`ru_majflt`, minor faults, peak RSS) | `residency_curve.rs:38-65` (`getrusage(RUSAGE_SELF)`) | V3: 59,670 major faults on the cold token. PREFETCH: ~73,518 | page-ins through the fault path, **for the whole process**; peak RSS is a process peak, not a per-pass delta | bytes from storage; which expert faulted | process-wide only. Faults track 16 KiB pages one to one **in every fault-path arm on record** (V3 "16.4 KB per fault"; PREFETCH 1.199e9 / 73,516 = 16,310 B; GLM demand, advise and touch, 12,295 × 16 KiB = 192.1 MiB). **They are not a byte proxy once the fetch path changes**: GLM §14's `pread` arms moved the same bytes in 48 reads, and V3's second run took 470,846 **minor** faults with 0 major [A2 P1-7] |
| `MappedAccess {Demand, Advise, Touch}` | `opplan/exec/realization.rs:247`, `exec/prefetch.rs:66-127` | PREFETCH-1 (§2.3) | which access policy the plan **issued** (`madvise(WILLNEED)` per range; one byte touched per page) | that advised pages were resident before projection | round 1 did not witness this. Round 2 (branch only, `--witness-residency`) recorded **100.0 % resident before the loop** in both advised arms [A1 #6] |
| VINDEX3 mapping advice | `represent/physical.rs:96,135`; `vindex3/read.rs:121` | — | segments are mapped with plain `memmap2::Mmap::map`, **with no `madvise`**: readahead is the kernel default | — | the only `madvise` under `format/vindex3` is the `WILLNEED` prefetch (`prefetch.rs:93`). The GLM probe set `MADV_RANDOM` on its own mapping, but drives the same `MappedAccess` prefetch code (`glm_moe_residency/probe.rs:7-8,463-469`) [A2 P1-50] |
| VINDEX3 routing capture `routing_trace::record` / `fingerprint` | `opplan/exec/routing_trace.rs`; recorded at `production/plan_backend.rs:375`; fingerprinted at `residency_curve.rs:305-337` | the **routing-identity witness** in V4 and PREFETCH (fingerprint `ebe045dc40cce5b1`) [A1 #12]. Added after V3, so V3 has none | the experts selected per routed call, plus a fingerprint | which bytes were read | in-memory capture. No exporter feeds the retention simulator (§1.4) |
| Expert page attribution `runtime::residency::account` / `ExpertRegion` | `larql-vindex/src/runtime/residency.rs:163,175`; used by `examples/glm_moe_residency/probe.rs` and `examples/vindex3_residency_probe.rs` | GLM §11: coverage 1.0003, unselected 0.0 MiB | resident pages **attributed to the operands (experts) that own them**, against the bound plan's prediction; a shared page counts once | — | a snapshot; the GLM result is one layer with readahead off [A1 #23] |
| `ResidencyBudget` / `select_realizations_within` / `BudgetDeficit` | `opplan/exec/accounting.rs:360`, `prepared/mla.rs:236`, `accounting/budgets.rs:101` | K3 residency vertical V2 on the Kimi container: A1 within budget at a 7.68 GiB physical working set; A2 at 7 GiB re-selects 67 operands; A3 at 3 GiB refuses before any payload byte, with an irreducible deficit of 3.63 GB (page-in per token 2.94 GB is part of the working set) | that **declared** bytes fit a declared budget, or a refusal with the deficit, before any payload byte | physical enforcement during execution | no runtime cap on mapped pages (§1.3) |
| `expectations` / `reconcile` | `prepared/residency.rs`, `accounting/reconcile.rs:63-81` | Residency-into-K3 plans | declared against **bound** bytes; for mappings it also **reports** mincore-resident bytes (`mapped_resident`) | that bound objects are resident | `mapped_resident` is reported beside the declaration, **never reconciled** against it [A1 #15] |

### 1.2 Probes, scripts and prior instruments

| Instrument | Where | What it does | Limits |
|---|---|---|---|
| GLM per-layer residency probe | `examples/glm_moe_residency.rs`, `glm_moe_residency/probe.rs` (on main); `evict()` at `probe.rs:116-128` | maps one routed layer's bank with `MADV_RANDOM`. Runs demand / advise / touch / `pread` / `pread`×8 arms. Byte-for-byte output invariant across arms. A cold check on the selected regions skips an arm it cannot make cold | the results were produced by the single-file probe at `6874285f`. Main's split version (`b3097a3a`) has the same logic, per audit 2's line comparison, which this revision did not repeat. Eviction: see "Eviction methods" below |
| `vindex3_residency_probe` | `examples/vindex3_residency_probe.rs` | "Does the bound plan predict the pages execution actually touches?" On a synthetic layer, it splits pages into router / selected / unselected | synthetic subject |
| GLM reuse trace and simulator | `scripts/glm_expert_trace.py`, `scripts/glm_expert_reuse.py` | the §12 pipeline: a prefill trace of selected experts over a prompt, then LRU and Bélády hit rates by budget, from an `expert_bytes` field | the trace JSON is not committed, and §12 does not record the prompt (§2.0) [A2 O-2] |
| Per-volume cold-fault rate | `examples/readrate.rs` | serial cold-fault rate per volume: the §13 disk table (USB 30.0 µs, internal 26.5 µs per fault) | serial, one fault in flight [A2 O-2] |
| Cold-read probe | `examples/mmap_cold_read_probe.rs`; `docs/diagnoses/v3-disk-resident-mmap.md` | `F_NOCACHE` `pread` cold reads. Records 100 µs p50 / 140 µs p99 per scattered 16 KB page (153 MB/s), and 18 µs p50 for sequential mmap with `MADV_RANDOM`, with 1,048,576 major faults over 1,048,576 pages | different model and file (`bench/aim-validation/v3_granite-30b.json`). It records "macOS can't easily cap mmap residency", and that only the first eviction of a fresh file is reliable [A1 #25, A2 O-1] |
| `larql moe-locality` | `crates/larql-cli/src/commands/diagnostics/moe_locality/`; `docs/cli.md:1761-1786` (research builds) | over a JSONL route trace: the union of experts across `K`-token blocks, and an **LRU hit-rate curve for a resident expert subset**, each against a uniform baseline and a skew-preserving shuffle | reads `LARQL_MOE_ROUTE_TRACE`, which VINDEX3 execution does not write (row below). No recorded result from this tool was found; §2.5's result came from an earlier capture path [A2 P1-67] |
| Routing-locality analysis | `bench/aim-validation/moe-routing/analyze.py`, artefact `v3moe_locality.json` | per-position, per-layer selections from `MOE_DEBUG=1 larql run` (legacy vindex), then reuse and working-set statistics | the source of §2.5 |
| `evict-bank.sh` (displacement eviction) | **not on main**: branch `k3-prefetch-2` (`aae20ae3`) and its WIP descendant `e2e0392f` | streams other large files (`DISPLACERS`: the Kimi-Linear HF directory and `…s6.vindex3`) through the page cache. Rounds 1 and 3 name it as their cold-state method | round 1 used a session scratch file of that name. The committed `aae20ae3` script reconstructs it, and their equality is asserted, not shown [A2 P1-17]. It reads ~183 GiB (~196.5 GB) just before a cold arm [A2 P1-18]. The notes attribute Spotlight's indexing to that read [A1 #39, #41]. `.s6` is recorded as a clone of the subject, so displacement re-read the subject's own extents through a different file (§5.2) [A2 O-7] |
| PREFETCH-2 mechanism: `Coalesced`, `Gated`, `probed_resident` (`GATE_PROBE_PAGES = 4`), `--witness-residency` | `aae20ae3` (`prefetch.rs:138`, `vindex3_cmd/mod.rs:247`). **Notes only on `wip/k3-prefetch-2-2026-09-23` (`e2e0392f`)**, a parked work-in-progress commit. A third branch, `wip/k3-prefetch-2026-09-23` (`4a3823e2`), carries `Coalesced`, the witness and the amended PREFETCH-1 notes [A1 #19] | as named | measurement PAUSED since 2026-09-06. `--witness-residency` costs ~70 ms per token (PF-N4) |
| Retention simulator (DEC-9A): Min / Lru / Lfu / Random / StaticOracle, compulsory floor | `larql-cli/.../k3_ledger/retention.rs` | the hit rate a policy would get on a **dec-bench capture pool** (`retention_report.rs:25`, `routing.bin`) | expert size defaults to `K3_ROUTED_EXPERT_BYTES = 17,547,264` (one K3 routed expert, MXFP4), **overridable with `--expert-bytes`** (`args.rs:124`, `retention_report.rs:27`). The simulator assumes one uniform size per expert [A2 P1-20]. Kimi-Linear's BF16 expert is 14,155,776 B [A1 #13]. No recorded DEC-9A result in `docs/` |
| JSONL route trace `LARQL_MOE_ROUTE_TRACE` → `{layer, seq, experts}` | `larql-compute/src/moe_route_observe.rs`; call sites in CPU MoE, Metal decode and `kquant_forward` | the experts selected per layer; `seq` is a **per-layer call index**, not a position id [A1 #10] | **not called by VINDEX3 opplan execution** [A1 #11]. Evidence: BW11-1 window union (`docs/bw11-weight-traffic.md`) |
| BW10 movement ledger | **branch `feat/bw10-ledger-and-exec-policy-seam` (`41811c1d`) only**; `larql-compute/src/movement_ledger/` | semantic / physical / useful byte classes; DRAM / NVMe / network tiers; refuses a verdict without a declared regime | only `MoeExperts` is instrumented. BW-A gate recorded on that branch in `docs/diagnoses/bw10-live-gate.md` [A1 #17] |
| VINDEX3 Metal path | `opplan/exec/metal_lowered/routed.rs:403-415` (`lowering_register_region`); `larql-compute-metal/src/buffers/residency.rs:46-56` (`LARQL_RESIDENCY_SET`) | the lowered routed path **registers the routed bank regions** (gate_up / down blocks and scales) as Metal buffers. `LARQL_RESIDENCY_SET` selects explicit `MTLResidencySet` arms | the compute-metal README records the explicit arms as "**refuted**, no effect". A "~45 GB wired-memory wall" is recorded and was "contradicted at 79 GB the next day" (`represent/execution_cost.rs:22-24`); `measure/teacher_forced/run.rs:52-58` still refuses on it [A2 P1-24] |
| Device-level read sweep | `scripts/dec8_2_fio_sweep.py` | fio, `psync`, `direct=1` random reads over a large file | no result on main; it counts no reads made inside a LARQL run [A2 O-5] |
| Non-plan memory | `larql-compute-metal/src/buffers/residency.rs` (`MTLResidencySet`); `larql-server/src/runtime_stats.rs:199` (`ru_maxrss`, a **peak**, not live RSS) | GPU residency sets; process peak RSS | outside the VINDEX3 plan's accounting [A1 #26, A2 P1-23] |
| MODALITY-RESIDENCY-1a (vision pages untouched by text execution) | **source artifact not recovered**: no file, branch or history entry in any searched ref | previously reported in project notes (2026-08-23, Gemma 4 26B-A4B, 45.68 GB expert bank); that container is not in `~/chris-models` now [A1 #20] | an evidence gap, not a negative result (§7, Q8) |

**Eviction methods, as recorded** [A2 P1-14, P1-63]:

- **Advice never evicted on Darwin.** `MADV_DONTNEED` did not evict, and
  `MADV_FREE_REUSABLE` returned `EPERM` (frontier §11).
- **Invalidation stopped working on an idle machine.** `msync(MS_INVALIDATE)`
  evicted in the first runs, and stopped reclaiming once nothing competed for
  memory (frontier §14).
- **Virgin layers.** §14's explicit-fetch arms ran on virgin sparse layers
  (identical geometry, never read). §13 does not state its cold method.
- **Displacement.** `evict-bank.sh`, used by the Kimi rounds (row above).
- **Named but not used here:** `sudo purge`, and a Linux cgroup machine for
  capping RAM (`v3-disk-resident-mmap.md`) [A2 O-1].

### 1.3 Runtime caches and budgets: what exists, and what bounds what

| Mechanism | Where | Bounds | Does it bound VINDEX3 mapped pages? |
|---|---|---|---|
| Dequantised-expert LRU | `larql-compute/src/cpu/ops/moe/cache.rs` (`LARQL_MOE_CACHE_ENTRIES`, default 256) | dequantised f32 copies of experts on the legacy CPU MoE path, keyed by mmap pointer | no; the mapped source pages stay with the OS |
| Metal expert-buffer cache warmup | `larql-server/src/routes/expert/warmup.rs:85-94` | the server's Metal expert buffers | no |
| `ResidencyManager` / `auto_pin()` | `larql-vindex/src/index/storage/residency.rs` | pinned legacy-index layers within a memory budget | no; it covers the legacy index, not VINDEX3 expert banks |
| `ResidencyBudget` | §1.1 | declared bytes, at planning | no; it plans and refuses, and does not enforce |
| Metal region registration | §1.2 | which bank regions the GPU may address | no; registration is not a cap, and explicit residency sets were recorded as having no effect |

**No VINDEX3 runtime expert cache, and no cap on mapped-page residency,
exists on main** [A1 #21].

`MADV_DONTNEED` has **five** call sites in `crates/` [A2 P1-27]:

- **one library call**, `mmap_storage.rs:386`
  (`unchecked_advise(UncheckedAdvice::DontNeed)` inside `release_pages`). The
  wrapper `release_mmap_pages` (`index/storage/gate_accessors/mod.rs:312`)
  reaches it by calling `self.storage.release_pages()` (`:327`). The server's
  walk-FFN handlers call the wrapper when `release_mmap_after_request` is set;
- **four probes:**
  - the GLM probe's `evict()` (`glm_moe_residency/probe.rs:127`);
  - `examples/mmap_cold_read_probe.rs:61`;
  - `examples/readrate.rs:20`;
  - `examples/vindex3_residency_probe.rs:191`.

The library function documents it as advisory on Darwin, and frontier §11
records that it did not evict.

### 1.4 Instruments that do not exist, or are not recoverable

Any MEM-001 protocol that needs one of these would first need it built or
recovered, as a **proposed prerequisite**. None is proposed for building here.

| Needed for | Status |
|---|---|
| Feeding a VINDEX3 run's routing into a reuse simulator | **missing.** VINDEX3 records an in-memory `routing_trace` capture. The retention simulator reads dec-bench capture pools, and `larql moe-locality` reads the JSONL trace, which is not written on the VINDEX3 path. It needs an exporter or adapter [A1 #11]. `glm_expert_reuse.py` takes a generic JSON `selected` trace, so its input contract is one possible target [A2 O-2] |
| A contention control | **unrecovered.** `contention.sh` is named only in `e2e0392f:k3-prefetch-2-notes.json` ("written and unrun") and is in no git ref [A1 #18] |
| An unwitnessed latency record for PREFETCH round 2 | **unrecovered.** That round's caveat says "the unwitnessed distributions below are the latency record", but no such distributions are in the file (§2.3) |
| Device-queue or I/O-contention state during a run | **missing.** Every round recorded CPU load or process activity; none recorded device I/O |
| Eviction on an idle machine | **unreliable** by main's own record (§1.2). The working methods are displacement (branch only) and virgin layers (GLM probe) |
| Expert attribution of faults or device reads | **missing.** Faults are process-wide, and nothing counts device reads inside a run |
| A runtime cap on VINDEX3 mapped-page residency | **missing** (§1.3) |

### 1.5 Expert layout, which decides bytes per expert

- **`Fused`** (packed bank). The type documentation describes "one fused
  `[2·intermediate, hidden]` gate/up per expert … and one
  `[hidden, intermediate]` down" (`backend/step_calls.rs:59-69`), passed as
  one slice per expert. No container's region table was checked [A1 §4.10,
  A2 P1-29].
- **`Separate`** (per-expert bank): three whole matrices per expert, each its
  own stored operand. The Kimi-Linear bank has 19,968 operands at
  **4,718,592 B** each (2304 × 1024 × 2, BF16; 26 MoE layers × 256 experts ×
  3 = 19,968). That is one matrix; an expert is three, 14,155,776 B.
  19,968 × 4,718,592 = 94,220,845,056 B, the bank payload in `.s6`'s and
  `.s7`'s `index.json` (lift2's own is unavailable).
- **PREFETCH's requests are consistent with page-rounded contiguous runs**
  [A2 P1-31]:
  - (624 × 288 + 202) × 16,384 = 2,947,710,976 B, which rounds to the 2.948
    GB coalesced request;
  - 624 × 289 × 16,384 = 2.955 GB, the "span".

  The data are given to three decimals (±0.5 MB, about ±30 pages), so the
  "+202" extra pages is assumed from the request count, not resolved by the
  data. This fits each expert's three matrices being one run. No writer-side
  ordering guarantee was found.
- **GGUF.** #27149 contradicts itself on layout [A2 P1-32]. The original post
  says "Each expert is a contiguous byte slice". The update says GGUF "stores
  all 128 experts interleaved in one tensor", that one expert through mmap
  "touches all 6,912 pages (111x amplification)", and that its re-layout
  needs 192 pages (36×). How 111× was derived is not explained.

## 2. Prior evidence

### 2.0 Provenance: every prior result is BACKGROUND ONLY

Each result below can inform a protocol, but none can serve as a
reproducible baseline today. None is untraceable either: each has a
recording commit and a named procedure. The classification is from audit 2
(§B1), checked here against the sources it names.

| Result | Recording commit | What limits reproduction |
|---|---|---|
| GLM §11 coverage and byte ratio | `6874285f` (2026-09-07; code and doc together) | the checkpoint is identified by size only (328,326,771,576 B), with no weight digest; the volume was not mounted on 2026-10-31; the input hidden-state file, invocation, raw output and §11's machine state are not committed |
| GLM §13 access-policy equality | `6874285f` | as above; per-round raw values are not committed; §13's cold method is not stated |
| GLM §14 `pread` 2.36× | `6874285f` | as above; which virgin layers, and the serial sample count, are not recorded |
| GLM §12 reuse table | `6874285f`; scripts unchanged on main | the trace is not committed, and the prompt is not recorded. The script's default prompt is one paragraph repeated three times (`glm_expert_trace.py:86-93`). If the default was used, the 128-token prefill may contain repeated text, which would inflate reuse. That is an inference |
| GLM 283.5 GiB estate | frontier §12 (header plan) | a structural fact from headers; reproducible once the checkpoint is mounted and digested |
| Kimi V3 rung 7 | `68abfd70` (2026-09-06, on main) | subject absent; no lift2 digest; **no routing fingerprint** (added in V4); eviction method and machine state not recorded; one run |
| Kimi PREFETCH round 1 | `63bc7c12` (mechanism, on main); notes on main | subject absent; the eviction script was a session file, committed later as a reconstruction; machine UNQUALIFIED; one sample per arm; raw logs not committed |
| Kimi PREFETCH round 2 | `e2e0392f` (parked WIP, not on main) | the exact code that ran is not recoverable (built uncommitted, ported forward later); machine state not recorded; the witness is inside every latency figure; the promised unwitnessed record is absent. **The weakest of the set** |
| Kimi PREFETCH round 3 | `e2e0392f` notes; mechanism at `aae20ae3` | whether the run used exactly `aae20ae3` is not stated; machine UNQUALIFIED; one retained sample per arm, and one discarded |
| Gemma 4 locality | `docs/diagnoses/moe-routing-locality.md` (2026-05-31), artefact on main | the container (`output/gemma4-26b-a4b-q4k.vindex`) is not in the repository; the passage is described but not given verbatim; one prompt |

For Kimi, "same subject across runs" rests on the clone record (§2.2), and,
for the PREFETCH rounds only, on routing fingerprint `ebe045dc40cce5b1` and
the generated ids [A1 #40].

### 2.1 GLM-5.3-Flash, one routed layer (frontier-execution-programme §11–14, on main, 2026-09-07)

**Subject:**
- `examples/glm_moe_residency.rs`, layer 3, the real 288-expert bank (6.752
  GiB), mapped from the checkpoint with `MADV_RANDOM` (readahead off).
- Top-8: 192.05 MiB of experts per token, 2.78 % of the bank.
- The external USB volume `model-drive`.
- Every arm's routed output is compared byte-for-byte, and a mismatch aborts.

**What it observed:**
- **Residency is predictable for this subject:** coverage 1.0003, with
  unselected residency 0.0 MiB (§11).
- **The selection's logical bytes equal the bytes paged in:** 192.0 MiB
  logical against 192.1 MiB physical, ratio 1.000; 12,295 faults × 16 KiB =
  192.1 MiB (§11). ("Logical bytes" here are what the selection needs. They
  are not `PrefetchReport.requested_bytes`, which §2.4 uses [A2 P1-51].)
- **Access policy makes no difference on an idle machine** (§13: load 2.7–3.6,
  five rounds):

  | Arm | Median | Spread |
  |---|---|---|
  | demand | 210.4 ms | 1.4 % |
  | advise | 209.1 ms | 0.5 % |
  | touch | 208.8 ms | 1.0 % |
  | warm | 17.8 ms | 5.3 % |

  All three cold arms take 12,295 faults. An earlier set taken at load ~30
  was discarded. The times are **one layer's routed-FFN stage**, whose own
  threading reaches about 1.8 faults in flight. The serial cost per fault is
  26.5 µs (internal SSD) to 30.0 µs (USB) (§13) [A2 P1-45].
- **The fetch shape matters** (§14): replacing faults with 48 `pread`s of
  4.0 MiB:

  | Arm | Cold routed stage |
  |---|---|
  | fault path | 212.1 ms |
  | serial `pread` | 121.2 ms |
  | `pread` × 8 threads | 90.0 ms (2,759 MiB/s, 2.36× faster) |

  The output is byte-identical. Parallel n=7, across virgin layers.
- **Reuse** (§12, a 128-token prefill trace): 225 of 288 experts touched.
  LRU and Bélády hit rates at 0.25–6.75 GiB per layer. For example, at 2
  GiB: LRU 42.1 %, LRU excluding first touch 53.9 %, Bélády 62.2 %.
- **Instrument lessons:**
  - A `pread` populates the same unified buffer cache, so the cold check must
    precede the fetch.
  - Eviction depends on memory pressure.

**Scope.** One layer of one model, readahead off, an external drive. This is
direct evidence for that configuration, not for Kimi-Linear, for readahead
on, for internal storage or for whole-token decode.

### 2.2 Kimi-Linear: subject identity, and the V3 cold run

**The subject is confirmed by source** [A1 #34]:
- `k3-v4-warm-latency.json:6` and the V3 notes' `subject` name
  `~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`, with prompt
  [1008, 10484, 318].
- PREFETCH-1's notes record routing fingerprint `ebe045dc40cce5b1`, "the same
  as V4", and the same generated ids.
- The bank is BF16: "bf16 per-expert bank 94.22 GB" (V3 `subject`).

**Identity chain** [A2 P1-37]. Each link is recorded separately, and they are
of different strength:

| Link | Evidence | Strength |
|---|---|---|
| lift2 ≈ `.s6` | `represent-real-evidence-1-notes.json:30`: lift2's "five segments are byte-identical in size and spot-checked content to the historical `s6` container (APFS clones; only `system_graph.json` differs by 33 bytes)" | size and spot-check; **no digest** |
| `.s6` = `.s7` | both `index.json` files are byte-identical on disk: expert bank `payload_sha256` `3f12d102…`, `segment_sha256` `e43600cd…`; segment sizes equal | **recorded digest** |
| lift2's own digest | not found on main, on any searched branch, or on disk | **absent** |

`lift2.vindex3` is not in `~/chris-models` now. `.s6.vindex3` was a displacer
in the eviction script and is now "the declared latent-norm refusal fixture"
(`continuation-mem-1.json:31`).

**V3 rung 7: the first cold measurement of this token** [A2 P1-47]
(`k3-residency-vertical-notes.json`,
`V3_execution_and_residency_curve.rungs.7_predicted_vs_observed`, findings
V3-N3 to V3-N5):

- **Conditions:** 2026-09-05 23:09Z; 0.000 GB of the bank resident at binding
  (mincore); production backend, `--residency-curve`; one run. Eviction
  method, machine state and routing fingerprint are not recorded.
- **Prompt (three positions), cold:** 7702 ms, 459,281 major faults, 7.545 GB.
- **Token 1, cold:** **1296 ms, 59,670 major faults, 0.976 GB**, "16.4 KB per
  fault — one 16 KiB page per fault, no readahead clustering visible".
- **Predicted against observed:** the plan's disjoint bound is 2.944 GB per
  token. V3-N3 attributes the 0.976 GB to "67% of its expert pages … already
  resident from the three-token prompt"; 1 − 0.976 / 2.944 = 66.8 %. V3-N3
  gives "≈22 µs per fault".
- **Warm token:** 140–430 ms across two runs, with 0 major faults in both.
  V3-N5 calls the spread unexplained and "not a throughput claim".
- **Second run** (23:40Z, after other processes had read the bank): 10.485
  GB already page-cache resident at binding, and 470,846 **minor** faults on
  the prompt at about 0.8 µs each. V3-N4 calls this a third residency state:
  on disk (major fault), in the page cache but not mapped (minor fault), and
  mapped.

### 2.3 Kimi-Linear PREFETCH: three rounds

The measured token is the first generated token after the three-position
prompt, as in V3.

| | Round 1: PREFETCH-1 | Round 2: witnessed 5-sample follow-up | Round 3: PREFETCH-2 preliminary |
|---|---|---|---|
| Record | `k3-prefetch-notes.json` on main | `follow_ups.witnessed_distributions…` in `e2e0392f:k3-prefetch-notes.json`; summarised in `aae20ae3:k3-prefetch-2.json` (`already_measured`) | `e2e0392f:k3-prefetch-2-notes.json` (`preliminary_round_UNQUALIFIED`) |
| Samples | 1 per arm | 5 per arm | 1 retained per arm. One further demand sample (3044 ms cold, 2677 ms **warm**, "the V4-N1 whole-process regime") was discarded and the campaign stopped [A2 P1-40] |
| Machine | UNQUALIFIED: AC power, load 5–13, `mds_stores` indexing, which the notes attribute to the eviction read | **not recorded** | UNQUALIFIED: peer `rustc` 78–100 %, `mediaanalysisd` ~80 %, `mds_stores` ~33 %, load 9.42; qualified warm arms refused; cold arms `--unquiet-ok` |
| Cold state | `evict-bank.sh`; binding witness 0.000 GB, 3/3 | eviction-verified cold per sample | `evict-bank.sh`; binding witness 0.000 GB |
| Witness inside the timed token | no | **yes, ~70 ms** (PF-N4) | no |
| Demand, cold token | 1253 ms | 1205 ms p50 (1141–1250) | 602 ms |
| Advise | 598 ms (prefetch stage 504.5) | 634 ms p50 (627–864; prefetch 557) | — |
| Coalesced | — | 619 ms p50 (612–628; prefetch 544) | 591 ms (prefetch 514) |
| Gated | — | — | 607 ms (prefetch 528) |
| Touch | 1358 ms | — | — |
| Cold prompt, three positions | demand 6535 / advise 3300 / touch 6394 ms | demand 5461 / advise 3342 / coalesced 3263 ms p50 | demand 4906 / coalesced 3927 / gated 3429 ms |
| Major faults, token | 73,516 / 73,519 / 73,518 | 73,520 / 73,518 / 73,518 | 73,518 |
| Paged bytes, token | 1.199 / 1.206 / 1.205 GB (`resident_delta_gb`) | 1.204 / 1.206 / 1.206 GB (`paged_gb_p50`) | 1.206 GB ("paged in") |
| Resident before the loop | not witnessed | demand 59.1 %; advised arms 100.0 % | — |
| Warm token | 76 / 87 / 80 ms | ~150 ms (witness-inflated) | 74 / 83 / 80 ms |

Round 1's source key is `resident_delta_gb`; rounds 2 and 3 say "paged".
PREFETCH-1's P5 uses the two interchangeably, so they are probably the same
quantity, but that was not confirmed against the branch-era printing code
[A2 P1-39].

**Round 2 is witnessed residency evidence, not a latency record.** Its
caveat reads: "every figure here carries the ≈70 ms witness cost inside the
token wall (and inside the prefetch stage for the advised arms) … The
unwitnessed distributions below are the latency record". Those distributions
are not in the file (§1.4). Its latency figures are used below only within
the round, or as labelled witness-adjusted estimates.

**A qualified warm reference exists** for the same token:
`A_quiet_QUALIFIED`, 75.4 ms p50, p95/p50 1.03, 2026-09-06, on
`docs/k3-v4-warm-latency-qualified-arm` (`254377ce`) and in `e2e0392f`'s
amended V4 notes [A1 #56].

**Carried observations, each from a single round:**
- **PF-N2:** concurrent faulting does not parallelise page-in of one mapping
  on macOS. This comes from PREFETCH-1's UNQUALIFIED round. GLM §13 on an
  idle machine independently found touch ≈ demand.
- **PF-N3:** advise costs about 8 ms of an 80 ms warm token. This also comes
  from PREFETCH-1's round, and was re-observed once, unqualified, in round 3.
  PREFETCH-2's own falsification list keeps "single-pass artifact" open
  [A1 #38].
- **PF-N5** (round 2): coalescing tightened the distribution but did not make
  it faster. The madvise path "saturates near 2.2–2.4 GB/s", and "the
  remaining gap needs a different primitive (a file-descriptor read path…)".

### 2.4 What the four Kimi observations show, and what they do not

The four observations are **not controlled repetitions** and are not pooled
into one distribution. They differ in date, machine state, eviction method,
instrument overhead and prior residency.

**Three latency bases are kept apart:**

- **raw unwitnessed:** V3, and rounds 1 and 3, as recorded;
- **raw witnessed:** round 2, as recorded, with ~70 ms of witness inside each
  token;
- **witness-adjusted:** round 2 minus 70 ms. This is an **estimate**: PF-N4
  gives the cost as approximate, and it is not measured per arm.

**Direct observations, within a round:**

| Round | Demand ÷ advised, cold token | Demand ÷ advised, cold prompt |
|---|---|---|
| 1 | 1253 / 598 = 2.10 | 6535 / 3300 = 1.98 |
| 2, raw witnessed | 1205 / 634 = 1.90; 1205 / 619 = 1.95 | 5461 / 3342 = 1.63; 5461 / 3263 = 1.67 |
| 2, witness-adjusted | 1135 / 564 = 2.01; 1135 / 549 = 2.07 | not estimated: the witness cost per prompt position is not recorded |
| 3 | 602 / 591 = 1.02; 602 / 607 = 0.99 | 4906 / 3927 = 1.25; 4906 / 3429 = 1.43 |

- For the measured token, rounds 1 and 2 show the advised arms at about half
  the demand time. Round 3 does not.
- Over the prompt (about 7.7 GB and ~470,000 faults), the advised arms are
  faster than demand in **all three** rounds, by less in round 3 [A2 P1-41].
- Across rounds, on comparable bases, the advised and coalesced token times
  fall at about 549–607 ms (round 2 adjusted; rounds 1 and 3 raw). The
  demand token moves by more than 2× (602–1253 ms).

**Per fault, whole token ÷ major faults:**

| Observation | Calculation | Per fault | Basis |
|---|---|---|---|
| V3 rung 7, demand | 1296 / 59,670 | 21.7 µs | unwitnessed |
| Round 1, demand | 1253 / 73,516 | 17.0 µs | unwitnessed |
| Round 2, demand, raw | 1205 / 73,520 | 16.4 µs | witnessed |
| Round 2, demand, adjusted | (1205 − 70) / 73,520 | ~15.4 µs | estimate |
| Round 3, demand | 602 / 73,518 | 8.2 µs | unwitnessed |

The whole-token wall includes warm compute (75.4 ms qualified p50); removing
it changes no ordering. For example, (1253 − 76) / 73,516 = 16.0 µs and
(602 − 74) / 73,518 = 7.2 µs.

PREFETCH-2's addendum (PF2-N1) describes the demand arm moving "from 16.4 us
per fault to 8.2". That 16.4 is round 2's witnessed p50 (1205 / 73,520). It
is the same arithmetic basis as 8.2, not a different one [A2 P1-46].

**The GLM row is not on this basis** [A2 P1-45]. GLM §13's 17.1 µs is a
one-layer routed-stage time at about 1.8 faults in flight, and its serial
cost is 26.5–30.0 µs. It is not compared with the Kimi rows here.

**What the variation shows.** Per-fault cost on the demand arm ranges from
8.2 to 21.7 µs across four observations of one token. Round 3 is the lowest,
and V3 the highest. **The existing evidence cannot isolate the cause.**

**Inferred explanations, none established:**
- **(i) Device I/O contention slowed demand in some runs.** No run recorded
  device I/O. Round 3 ran under heavy **CPU** load with the fastest demand
  arm, which shows only that CPU load and fast demand coincided once
  [A2 P1-49]. Round 2's machine state is unrecorded.
- **(ii) Round 3's demand ran with more faults in flight**, or against a
  different device or cache state. Displacing with `.s6`, an APFS clone of
  the subject, re-reads the subject's own extents (§5.2).
- **(iii) The madvise path helps on the Kimi container but not on GLM**
  because of readahead, the drive or the layout. VINDEX3 maps its segments
  with default readahead, GLM's probe with `MADV_RANDOM`, and both share the
  same `WILLNEED` code (§1.1) [A2 P1-50]. V3 records "no readahead
  clustering visible" on the demand path.
- **(iv) Prior residency differs.** V3 found about 67 % of the token's
  selection resident from the prompt, and PREFETCH 59.1 %.

**GLM's idle result is consistent with a contention reading, but it does not
show that contention caused the PREFETCH differences.** Model, layout, mapping
advice and drive all differ.

**Byte accounting: reconciled within PREFETCH, not across the lineage**
[A1 #31, A2 P1-47]:

- **Within the PREFETCH rounds:** round 2 witnessed 59.1 % of the 2.948 GB
  selection already resident before the demand loop: the prompt's own reuse
  of experts across positions. 2.948 × (1 − 0.591) = 1.206 GB, the observed
  paged bytes. PREFETCH-2's gate "issued 84 of the 202 possible requests,
  asking for 1.219 GB".
- **Across V3 and PREFETCH, on the same token and prompt:**

  | | V3 rung 7 | PREFETCH round 1 |
  |---|---|---|
  | Token paged | 0.976 GB | 1.199 GB |
  | Token major faults | 59,670 | 73,516 |
  | Prompt paged | 7.545 GB | 7.72 GB |
  | Prompt + token | 8.52 GB | 8.92 GB |
  | Resident from the prompt | ~67 % of 2.944 GB (disjoint bound) | 59.1 % of 2.948 GB (round 2, page-rounded request) |
  | Routing fingerprint | **not recorded** | `ebe045dc40cce5b1` |

  The ~0.4 GB difference in the touched set is **unexplained by any
  source**. PREFETCH-1's own P1 notes it ("this run paged 1.199 GB where V3's
  token paged 0.976 GB"). Without V3's fingerprint, a routing difference
  cannot be excluded.

The four quantities stay distinct. "Requested" below means the bytes the
prefetch asked of the OS, not the selection's logical bytes (§2.1):

| Quantity | Source | Counts |
|---|---|---|
| Requested bytes | `PrefetchReport.requested_bytes` (branch) | page-rounded ranges asked of the OS |
| Paged / resident delta | mincore before against after, bound regions | pages that became resident in the measured step (subject to the boundary-page caveat in §1.1) |
| Fault activity | `ru_majflt` × 16 KiB | page-ins via the fault path, process-wide |
| Device reads | no instrument | — |

### 2.5 Routing locality: Gemma 4 26B-A4B (`docs/diagnoses/moe-routing-locality.md`, 2026-05-31)

[A2 P1-67]

**Subject and method:**
- Gemma 4 26B-A4B, 128 experts, top-8, 30 MoE layers, on the **legacy**
  vindex (`output/gemma4-26b-a4b-q4k.vindex`), with experts computed
  in-process.
- Selections captured with `MOE_DEBUG=1` from one forward over a ~72-token
  passage (prefill). Analysis by
  `bench/aim-validation/moe-routing/analyze.py`, artefact `v3moe_locality.json`.

**What it observed:**

| Metric | Measured | Uniform-random null |
|---|---|---|
| Adjacent-position reuse | 21.3 % of top-8 | 6.2 % |
| Working set per layer | 124.4 / 128 | 126.6 / 128 |
| Cumulative cache hit, 72 tokens | 79.5 % | — |
| Top-10 expert mass | 17 % | ~8 % |

**The diagnosis's conclusions, quoted:** "there is no small cacheable hot
subset". For a larger-than-RAM MoE, "you cannot keep a hot fraction resident
and page the rest". It projects ~200 ms per token of sustained thrash from
"V3's ~100µs cold page". (That V3 is the `v3-disk-resident-mmap.md` cold-read
probe, not the Kimi V3 vertical.)

**Its own caveat:** one model, one ~72-token passage. Gemma's router is
load-balanced, and a less regularised MoE "could concentrate more".

**How it bears on MEM-001:**
- It is **contrary evidence for a small static hot-expert cache**, on one
  model.
- It shows adjacent-position reuse well above chance (3.4×). It does not
  rule out short-window temporal locality, or prefetching that exploits it.
- It is a prefill over one passage, like GLM §12. Neither is a decode
  trace.
- GLM §12's 225 of 288 experts over 128 tokens, and Kimi's 59–67 % prompt
  reuse over three positions, are other observations on different models,
  windows and paths. They are not combined with it here.

## 3. Imposing a memory budget

| Regime | Claim it can support | Available |
|---|---|---|
| **A. Model bytes exceed physical RAM** | feasibility of oversized execution | **Recorded, not verified today.** GLM-5.3-Flash: 42 sparse layers × 6.75 GiB = 283.5 GiB expert estate against 128 GiB of RAM (frontier §12). The checkpoint is 328,326,771,576 B (305.78 GiB, 328 GB; frontier §1), on `model-drive` (frontier header and §13) [A2 P1-53]. The volume was **not mounted** on 2026-10-31. Only per-layer probing has run; whole-model execution has not. In `~/chris-models` the largest entries are three of ~91.5 GiB (the Kimi-Linear HF directory, `.s6` and `.s7`); none exceeds RAM [A1 #36] |
| **B. Model fits; expert residency is held below its footprint** | how paging and caching behave under controlled pressure | needs a method from the next table |

Methods for regime B:

| Method | Constrains | Verified by | Known interference |
|---|---|---|---|
| Software expert budget over VINDEX3 mapped pages | the expert bank only | expert page attribution and mincore (§1.1) | **does not exist** (§1.3). Eviction is unreliable on an idle machine (§1.2), so "evicted" must be witnessed. It is a software budget, never an RAM limit |
| An explicit fetch path (`pread` into owned buffers) under a byte budget | the bytes the fetch holds | the fetch's own accounting; the cold check must precede the fetch | shown per layer on GLM (§2.1), not on VINDEX3 execution; `pread` also fills the shared buffer cache |
| System-wide memory balloon (wired allocation) | all of RAM | `vm_stat`, mincore | changes Metal headroom, the file cache for every process and the compressor. A wired-memory wall was once inferred at ~45 GB and contradicted at 79 GB (§1.2) |
| A Linux cgroup machine | all of RAM for the process | the cgroup | named in `v3-disk-resident-mmap.md`; a different OS and device: a separate result, not an arm |
| A smaller-RAM machine | all of RAM, physically | the machine | a different device: a separate result, not an arm |

"macOS can't easily cap mmap residency" (`v3-disk-resident-mmap.md`) [A1
#47]. A software budget and an actual RAM limit stay two distinct claims.

## 4. External reports (reported findings, not LARQL results)

All three are `ggml-org/llama.cpp` discussions, the primary sources, read on
2026-10-31.

**Every figure in this table is the poster's own report and is independently
unverified.** None was reproduced on this machine or by LARQL.

| Discussion (primary source) | What it is | Reported (unverified) | Measured or simulated, per the source |
|---|---|---|---|
| [#28248](https://github.com/ggml-org/llama.cpp/discussions/28248) "Persistent expert slot pool for MoE CPU offload" (2026-09-02) | VRAM slot pool for experts that live in CPU RAM; **no SSD tier** | single RTX 4090, Qwen3.8-Flash-Next UD-Q3_K_XL, 8K context: tg512 11.55 ± 0.41 → 21.21 ± 0.46 t/s at 64 slots | reported as measured. Its "bit-exact" refers to a backend unit matrix (40/40). End-to-end greedy output diverged once on a near-tie, and the RFC reports PPL 3.3096 against 3.3262 [A1 #48]. A commenter reports that argmax moved on 2–6 % of tokens while PPL moved < 1 %, in a related fork. Later in the thread the author reports pool against stock at a 4.45 % top-1 flip rate (KLD 0.0272), and pool against full GPU at KLD 0.000000, attributing the flips to CPU→GPU placement [A2 P1-55] |
| [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) "MoE offload to disk with on-demand paging" (2026-05-19) | Metal shared-memory slot pool, loading experts with `pread` from GGUF (not OS paging), on Apple Silicon | Qwen3-30B-A3B Q6_K, M3 Pro 36 GB: vanilla 38.1 tok/s; 128 slots 30; 64 → 27.5; 32 → 22.1. M1 Pro 16 GB "13 tok/s" is a one-line update | reported as measured. (38.1 − 30)/38.1 = 21 %. The post gives only the pool's run command; the pool requires `--no-mmap`, `--no-warmup` and a small `-ub`, and the vanilla row's flags are not stated. So 21 % is **not** isolated pool overhead [A2 P1-58]. 128 slots holds every expert, so that row has no misses |
| [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149) "Expert-Aware SSD Streaming" (2026-08-15) | standalone prototype (TinyGiant) | I/O on an M1 MacBook Pro 16 GB: 2.97 GB/s sequential, 6.06 GB/s double-buffered; headline 4.7 tok/s | the headline is **real I/O with simulated compute**. The C benchmark (real BLAS, `pread`; an M1 16 GB / MacBook Air per the author) reports **pipelined 2.40–4.45** tok/s (sequential 2.20–4.46) at **controlled** 50–100 % hit rates, not routing-produced ones [A1 #51, #52]. The update also claims an end-to-end Python engine generating text from Qwen3-30B-A3B, with no decode rate given. The layout claims contradict each other (§1.5) |

**Identity across runtimes.** LARQL executes VINDEX3 and llama.cpp executes
GGUF. The same source weights, quantisation and prompt tokens can be
compared; the same file cannot, and layout is one of the variables.
Of the three, #23324 is the only one with a measured end-to-end decode rate
from a real model's routing with real compute and on-demand loads on Apple
Silicon [A2 P1-60].

**The two runtimes' instruments are not equivalent:**

| Measurement | LARQL | External runtime | Comparable as is? |
|---|---|---|---|
| Wall time per token | yes | yes (`llama-bench`) | only under one harness and quiet gate |
| Process faults | `ru_majflt` in-process | for a child the harness spawns (`wait4` / `RUSAGE_CHILDREN`), or `proc_pid_rusage`; not via `getrusage` on an arbitrary process [A1 #45] | process-wide, if collected the same way for both |
| File-page residency | mincore in-process; expert attribution | an **observer process** mapping the same file | LARQL's own V3 second run shows a fresh process's mincore seeing page-cache state left by other processes (10.485 GB at binding), so a shared observer is feasible on macOS. Whether it suits a cross-runtime arm is untested [A2 P1-61] |
| Expert ids per position | `routing_trace` capture | a related fork claims a `moe-trace` exporter; not examined [A2 P1-56] | not until shown |
| Bytes per expert | region table | GGUF tensor offsets; layout differs by design | by computation, not observation |
| Declared budget or refusal | yes | `n_slots` | no: different mechanisms |

## 5. Confounds the protocol must separate

1. **File cache versus process residency.** Mapped file pages and `pread`
   reads share the unified buffer cache (frontier §14), so a new process can
   inherit residency. V3-N4 records three states: on disk (major fault,
   ~22 µs), in the page cache but not mapped (minor fault, ~0.8 µs), and
   mapped.
2. **Eviction state.** Advice never evicted, and invalidation stopped on an
   idle machine (§1.2). Displacement reads ~183 GiB just before the arm, and
   one displacer, `.s6`, is a clone of the subject, so it re-reads the
   subject's own extents through a different file. Virgin layers avoid both,
   but only where the geometry repeats.
3. **Device contention and machine state.** No run recorded device I/O.
   Round 3 ran under heavy CPU load with demand fastest, and round 2's state
   is unrecorded.
4. **Instrument cost.** The residency witness costs ~70 ms per token. The
   residency claims rest on witnessed runs and the latency claims on
   unwitnessed ones. Round 2 enters latency comparisons only within the round
   or as a labelled estimate, because its unwitnessed record is unrecovered.
5. **Mapping advice and readahead.** GLM ran readahead off (`MADV_RANDOM`).
   VINDEX3 maps with no advice (default readahead). The prefetch code is
   shared.
6. **Faults as a proxy for bytes.** They agreed one to one in every
   fault-path arm on record (§1.1). They diverge once the fetch path changes
   (`pread`) or pages are already cached (minor faults). Device reads are
   unmeasured.
7. **Decode cost unrelated to memory.** At about 54 % hit with deep fetch,
   GLM's arithmetic puts the bottleneck flipping to expert compute (frontier
   §14). Warm is the ceiling, and it has to be measured. V3's warm token
   ranged 140–430 ms unexplained.
8. **Routing identity across runs and runtimes.** It needs recording before
   any working-set comparison. V3 lacked it.
9. **One run is not a distribution.** V3 and rounds 1 and 3 are single
   samples, and round 3 discarded one.
10. **Summed residency double-counts boundary pages** (§1.1). Per-region
    mincore sums are not de-duplicated physical bytes.

## 6. Candidate: extending and replicating the existing results

**This is a candidate for review, not an authorised experiment.** It sets no
thresholds, picks no winner and runs nothing. If it is accepted, it enters a
freeze with the §7 answers before any arm runs.

Residency prediction has been **demonstrated once**: GLM, one layer,
readahead off, coverage 1.0003 (§2.1). V3 also tabled predicted against
observed residency on Kimi, at the whole-token level and against a disjoint
bound (§2.2). A candidate MEM-001 therefore **replicates or extends those
results**, and does not present them as new. Every input it would build on
is BACKGROUND ONLY (§2.0). Candidate directions, smallest first:

1. **Replicate GLM §11's predicted-versus-observed page account on the Kimi
   bank.** Use the expert page attribution of §1.1 on one routed layer with
   the GLM probe's method (virgin layers, cold check before the fetch).
   - **Prerequisite:** a pinned subject (§7, Q1).
   - **Question:** does coverage ≈ 1 survive a different model, layout and
     mapping advice?
2. **Explain the variation across the four Kimi cold observations**
   (MEM-001A). Repeated cold demand and advise arms on the pinned subject,
   each with its own demand baseline, under recorded machine and device
   state, with the mapping advice recorded, with the routing fingerprint
   recorded on every arm, and with unwitnessed latency kept separate from
   witnessed residency.
   - **Prerequisites:** a committed contention control (§1.4), a working
     eviction method (§5.2), and an unwitnessed latency arm, since round 2's
     is unrecovered.
   - **Question:** which of §2.4's inferred explanations survives?
3. **Extend GLM §14's `pread` result to VINDEX3 whole-token execution.** PF-N5
   already names a file-descriptor read path as the remaining lever on Kimi.
   - **Prerequisite:** a non-mapped fetch realisation in VINDEX3 (frontier
     §14: "a representation-and-loader question"). That is an implementation
     step, and out of scope until reviewed.
   - **Question:** is the 2.36× per-layer stage gain attributable to fetch
     shape and not to incidental caching, and does it survive whole-token
     decode?
4. **Working-set accounting across tokens** (MEM-001B). Reuse curves on real
   tokens, like GLM §12 and the Gemma locality result (§2.5), for the Kimi
   subject, on a decode trace as well as a prefill.
   - **Prerequisites:** a VINDEX3-routing exporter (§1.4), real region sizes
     (§1.5) passed with `--expert-bytes` if the retention simulator is used,
     and a recorded prompt.
   - **Question:** does the Gemma finding (no small hot subset) hold on Kimi
     and GLM, and how much short-window reuse is there?

Each direction keeps three categories separate: direct observations (pages,
faults, wall time), inferred explanations, and causal claims. A causal claim
needs an intervention arm with its own baseline.

## 7. Questions for the freeze

1. **Subject identity.** lift2 is gone. Is the recorded chain enough to pin
   `.s7` as the subject: lift2 ≈ `.s6` by segment size and spot-checked
   content (no digest), and `.s6` = `.s7` by `payload_sha256`? Or is a digest
   of the subject required, which would mean a new subject pinned by digest?
   Either way: which model, container bytes, tensor layout and digest?
2. **Regime A.** Is GLM-5.3-Flash on `model-drive` the regime-A subject? Is
   whole-model execution of it in scope, or only per-layer probing? Its
   checkpoint would need a weight digest first.
3. **Regime B method.** Software budget over mapped pages, explicit fetch
   under a byte budget, or none? Any claim from it is a software-budget claim.
4. **Eviction method.** Displacement (and with which files, given the clone
   relationship), virgin layers or another, with its failure modes stated and
   witnessed per arm.
5. **Contention control.** Re-written and committed before any arm that
   relies on it. Does it record device I/O, not only CPU load?
6. **PREFETCH-2 and BW10.** Each branch: merged first, carried in, or left?
7. **Mapping advice.** VINDEX3 maps with no advice (§1.1). Is advice an arm
   variable?
8. **MODALITY-RESIDENCY-1a.** Re-run and record it, or drop it as a premise.
9. **Repeats, qualification and outcome criteria.** How many cold and warm
   repeats, under what quiet gate, and what agreement counts as
   "predictable"? These are chosen before any arm runs, not from this
   document's numbers.
10. **External arm.** Whether #23324's fork, pinned to a commit, is in scope,
    and how routing identity would be recorded for it.

## 8. Review outcome

**Revision 1 (maintainer, 2026-10-31).** Accepted as the reconnaissance
record, not a protocol. MEM-001 is not frozen, and the PR stays in draft
pending an independent source-level check.

**Independent audit 1 (`43388264`).** Failed: 7 BLOCKING, 17 INCORRECT.
Preserved in `mem-001-reconnaissance-audit-1.md`.

**Maintainer direction after audit 1 (2026-10-31):**
- Recover LARQL's existing memory-residency baseline rather than rediscover
  it.
- Reframe MEM-001 around whether residency prediction **generalises**, and
  whether it yields **reliable** performance.
- Treat GLM's idle-machine result as consistent with, but not proof of, a
  contention explanation.

**Independent audit 2 (`b82d70b3`), two phases.** Phase 1, independent: 67
findings, 3 BLOCKING (V3 rung 7, the lift2 clone record, the Gemma locality
result). Phase 2: of audit 1's 7 BLOCKING findings, 5 resolved and 2 partly
resolved (#31, #34); four partial resolutions came from adopting audit 1's
suggested wording unchecked. Every prior result classified BACKGROUND ONLY;
the witness-cost presentation refuted in three places. Preserved in
`mem-001-reconnaissance-audit-2-phase{1,2}.md`. This revision corrects it.

**Maintainer direction after audit 2 (2026-10-31):**
- Revision 3 approved, correcting all three new BLOCKING findings, the two
  partly resolved ones, and the non-blocking corrections.
- The aim is a complete and accurately qualified reconnaissance record, not
  one that explains every historical measurement. Revision 3 can be accepted
  with historical experiments irreproducible, provided their provenance
  limits are stated and no important contrary evidence is omitted. It cannot
  be accepted if it misrepresents what instruments measure or what prior
  results establish.
- Audit 3 is bounded by a frozen source inventory built before it starts.
  Discoveries outside the inventory are reported separately, not silently
  added to scope.
- Acceptance of this record is not permission to run MEM-001A or MEM-001B.

MEM-001 is frozen only once the reproducibility, accounting and identity
problems are resolved or explicitly bounded.
