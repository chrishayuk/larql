# MEM-001 reconnaissance: what LARQL can already measure about expert residency

**Class: RECONNAISSANCE.** Dated 2026-10-31 at main `51579782`. Read-only. It
is not sealed and not experimental: no arm has run for this document, no
threshold is chosen here, and no cache, policy or runtime is recommended. §7
lists the questions a later freeze must answer.

**Revision 2 (correction).** Revision 1 (`43388264`) failed an independent
source audit: 7 BLOCKING findings and 17 INCORRECT ones, out of 56. The audit
is preserved unedited in
[`mem-001-reconnaissance-audit-1.md`](mem-001-reconnaissance-audit-1.md), and
revision 1 stays readable at `43388264`.

This revision rebuilds §2 and §6 around evidence revision 1 omitted. It
corrects the subject container, the instrument inventory and the byte
accounting. Finding numbers from the audit are cited as **[A1 #n]**.

**The question MEM-001 will ask.** How much physical memory does a routed
(MoE) model need to sustain a given decode rate? And does that relationship
survive once the model's bytes exceed what the machine can hold?

Since revision 1, the framing has changed. LARQL has **already shown, for one
layer of one model**, that the pages a bound expert selection touches can be
predicted from the plan (§2.1). So the open questions are now:

- whether that prediction generalises across models, layouts, access paths
  and runtimes;
- whether correct prediction translates into reliable decode-rate gains.

**Headline.**

- **Prior LARQL evidence exists, and it is stronger than revision 1
  reported.** `docs/frontier-execution-programme.md` §11–14 (GLM-5.3-Flash,
  one routed layer, readahead off) records the following [A1 #24]:
  - predicted against observed residency: **coverage 1.0003**, with
    unselected residency 0.0 MiB;
  - requested bytes equal to paged bytes: **ratio 1.000**;
  - demand, advise and touch the same **to within 0.8 %** on an idle machine;
  - an explicit `pread` fetch giving a **2.36× faster** cold routed stage;
  - LRU and Bélády hit rates at six budgets on 128 real tokens.
- **The Kimi PREFETCH evidence comes from three rounds, not two** [A1 #30].
  Two show the advised arms at about half the demand cold token (1253 → 598
  ms; 1205 → 634/619 ms p50); one does not (602 / 591 / 607 ms). Measured per
  fault, the round that disagrees is the **third**, not the first (§2.3).
  Reproducibility is unresolved, and nothing here establishes a cause.
- **The PREFETCH byte gap is already explained by its sources** [A1 #31]:
  59.1 % of the 2.948 GB selection was resident from the prompt before the
  measured token, and 2.948 × 0.409 = 1.206 GB, the observed resident delta.
- **The PREFETCH subject is `Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`,
  and it is no longer on disk** [A1 #34]. The `.s6` container that revision
  1 named was one of the files read to *evict* it.
- **LARQL has runtime expert caches, but none bounds VINDEX3 mapped-page
  residency** [A1 #21] (§1.3).
- **Some instruments revision 1 relied on are missing** (§1.4):
  - The routing-trace → retention-simulator chain does not exist for VINDEX3
    execution [A1 #11].
  - The contention script is not in any git ref [A1 #18].
  - The eviction routine on main is the GLM probe's, and main records it
    failing on an idle machine [A1 #9].
- **Oversized execution and constrained residency are still separate
  regimes** (§3). A genuinely oversized subject is *recorded*: GLM-5.3-Flash
  has a 283.5 GiB expert estate on an external volume. That volume was not
  mounted on 2026-10-31, and only one layer has ever been probed.

Paths and line numbers are at `51579782`. Revision 2 was rebuilt from a
search by physical and implementation terms, not by programme name:

- `git grep -n -i -E 'mincore|madvise|WILLNEED|ru_majflt|major_fault|resident|pread|evict'`
  over `crates`, `scripts` and `docs`;
- the same across remote branches;
- every finding of A1, re-read against its cited source.

Numbers quoted from notes files are copied, and the arithmetic is shown where
the document derives anything.

## 1. Instrument inventory

### 1.1 VINDEX3 mapped-page instruments, on main unless marked

Columns:

- **Existing evidence:** the recorded result that has used the instrument.
- **Directly observes:** what the instrument itself reports.
- **Inferred from it:** what a reader is tempted to conclude, but the
  instrument does not observe.

| Instrument | Implementation | Existing evidence | Directly observes | Inferred from it (not observed) | Known limits |
|---|---|---|---|---|---|
| `resident_bytes()`, mincore over one stored region | `format/vindex3/represent/physical.rs:269` | PREFETCH resident deltas (§2.2) | which pages of **one mapped region** are resident at the instant of the call | the process's physical footprint; that this run caused the residency | `None` without an OS report. **Summed callers map `None` to 0** (`weights/loaded.rs:37`, `prepared/residency.rs:417-423`), so a "0.000 GB resident" witness cannot tell cold from a failed read [A1 #4] |
| Binding line "X GB address space over N regions, Y GB resident" | `larql-cli/.../vindex3_cmd/generate/residency_curve.rs:195` | PREFETCH cold witness: 94.22 GB, 19,968 regions, 0.000 GB resident | the sum of the above over the plan's bound regions | total memory in use | KV, scratch and Metal buffers are outside the plan and not counted; the summed-0 caveat above |
| Fault and RSS deltas per pass (`ru_majflt`, minor faults, peak RSS) | `residency_curve.rs:38-65` | PREFETCH, ~73,518 major faults per cold token | page-ins through the fault path and peak RSS, **for the whole process** | bytes from storage; which expert faulted | process-wide only. In every PREFETCH arm and in the GLM probe, faults tracked 16 KiB pages **one to one** (1.199e9 / 73,516 = 16,310 B; GLM 12,295 × 16 KiB = 192.1 MiB). Decoupling by readahead or `pread` is possible in principle, but no LARQL evidence shows it [A1 #8] |
| `MappedAccess {Demand, Advise, Touch}` | `opplan/exec/realization.rs:247`, `exec/prefetch.rs:66-124` | PREFETCH-1 (§2.2) | which access policy the plan **issued** (`madvise(WILLNEED)` per range; one byte touched per page) | that advised pages were resident before projection | the first round did not witness this. The 5-sample follow-up (branch only, `--witness-residency`) recorded **100.0 % resident before the loop** in both advised arms [A1 #6] |
| VINDEX3 routing capture `routing_trace::record` / `fingerprint` | `opplan/exec/routing_trace.rs`; recorded at `production/plan_backend.rs:375`; fingerprinted at `residency_curve.rs:305-337` | the **routing-identity witness** in V4 and PREFETCH (fingerprint `ebe045dc40cce5b1`) [A1 #12] | the experts selected per routed call, plus a fingerprint | which bytes were read | in-memory capture. No exporter feeds the retention simulator (§1.4) |
| Expert page attribution `runtime::residency::account` / `ExpertRegion` | `larql-vindex/src/runtime/residency.rs:163,175`; used by `examples/glm_moe_residency/probe.rs` and `examples/vindex3_residency_probe.rs` | GLM §11: coverage 1.0003, unselected 0.0 MiB | resident pages **attributed to the operands (experts) that own them**, against the bound plan's prediction | — | a snapshot; the GLM result is one layer with readahead off [A1 #23] |
| `ResidencyBudget` / `select_realizations_within` / `BudgetDeficit` | `opplan/exec/accounting.rs:360`, `prepared/mla.rs:236`, `accounting/budgets.rs:101` | Residency-into-K3 plans | that **declared** bytes fit a declared budget, or a refusal with the deficit, before any payload byte | physical enforcement during execution | no runtime cap on mapped pages (§1.3) |
| `expectations` / `reconcile` | `prepared/residency.rs`, `accounting/reconcile.rs:63-81` | Residency-into-K3 plans | declared against **bound** bytes; for mappings it also **reports** mincore-resident bytes (`mapped_resident`) | that bound objects are resident | `mapped_resident` is reported beside the declaration, **never reconciled** against it [A1 #15] |

### 1.2 Probes, scripts and prior instruments

| Instrument | Where | What it does | Limits |
|---|---|---|---|
| GLM per-layer residency probe | `examples/glm_moe_residency.rs`, `glm_moe_residency/probe.rs` (on main); `evict()` at `probe.rs:116` | maps one routed layer's bank with `MADV_RANDOM`. Runs demand / advise / touch / `pread` / `pread`×8 arms. Byte-for-byte output invariant across arms. Cold check on the selected regions | `MADV_DONTNEED` did not evict on Darwin; `MADV_FREE_REUSABLE` returned `EPERM`. `msync(MS_INVALIDATE)` stopped evicting on an idle machine, so cold arms use **virgin layers** (frontier §11, §14) [A1 #9] |
| `vindex3_residency_probe` | `examples/vindex3_residency_probe.rs` | "Does the bound plan predict the pages execution actually touches?" On a synthetic layer, it splits pages into router / selected / unselected | synthetic subject |
| Cold-read probe | `examples/mmap_cold_read_probe.rs`; `docs/diagnoses/v3-disk-resident-mmap.md` | `F_NOCACHE` `pread` cold reads | the diagnosis records "macOS can't easily cap mmap residency" and that only the first eviction of a fresh file is reliable [A1 #25] |
| `evict-bank.sh` (displacement eviction) | **branch `k3-prefetch-2` (`aae20ae3`) only** | streams other large files (`DISPLACERS`: the HF directory and `…s6.vindex3`) through the page cache. This was PREFETCH-1's **and** PREFETCH-2's cold-state method | reads ~184 GB just before a cold arm; the notes attribute Spotlight's indexing to that read [A1 #39, #41] |
| PREFETCH-2 mechanism: `Coalesced`, `Gated`, `probed_resident` (`GATE_PROBE_PAGES = 4`), `--witness-residency` | `aae20ae3` (`prefetch.rs:138`, `vindex3_cmd/mod.rs:247`). **Notes only on `wip/k3-prefetch-2-2026-09-23` (`e2e0392f`)**, a parked work-in-progress commit. A third branch, `wip/k3-prefetch-2026-09-23` (`4a3823e2`), carries `Coalesced`, the witness and the amended PREFETCH-1 notes [A1 #19] | as named | measurement PAUSED since 2026-09-06. `--witness-residency` costs ~70 ms per token (PF-N4) |
| Retention simulator (DEC-9A): Min / Lru / Lfu / Random / StaticOracle, compulsory floor | `larql-cli/.../k3_ledger/retention.rs` | the hit rate a policy would get on a **dec-bench capture pool** (`retention_report.rs:25`, `routing.bin`) | `K3_ROUTED_EXPERT_BYTES = 17,547,264` is "one K3 routed expert, all three branches, MXFP4", not Kimi-Linear's BF16 expert (14,155,776 B) [A1 #13]. No recorded DEC-9A result in `docs/` |
| JSONL route trace `LARQL_MOE_ROUTE_TRACE` → `{layer, seq, experts}` | `larql-compute/src/moe_route_observe.rs`; call sites in CPU MoE, Metal decode and `kquant_forward` | the experts selected per layer; `seq` is a **per-layer call index**, not a position id [A1 #10] | **not called by VINDEX3 opplan execution** [A1 #11]. Evidence: BW11-1 window union (`docs/bw11-weight-traffic.md`) |
| BW10 movement ledger | **branch `feat/bw10-ledger-and-exec-policy-seam` (`41811c1d`) only**; `larql-compute/src/movement_ledger/` | semantic / physical / useful byte classes; DRAM / NVMe / network tiers; refuses a verdict without a declared regime | only `MoeExperts` is instrumented. BW-A gate recorded on that branch in `docs/diagnoses/bw10-live-gate.md` [A1 #17] |
| Non-plan memory | `larql-compute-metal/src/buffers/residency.rs` (`MTLResidencySet`); `larql-server/src/runtime_stats.rs:199` (process RSS) | GPU residency sets; process RSS | outside the VINDEX3 plan's accounting [A1 #26] |
| MODALITY-RESIDENCY-1a (vision pages untouched by text execution) | **source artifact not recovered**: no file, branch or history entry in any searched ref | previously reported in project notes (2026-08-23, Gemma 4 26B-A4B, 45.68 GB expert bank); that container is not in `~/chris-models` now [A1 #20] | an evidence gap, not a negative result (§7, Q8) |

### 1.3 Runtime caches and budgets: what exists, and what bounds what

| Mechanism | Where | Bounds | Does it bound VINDEX3 mapped pages? |
|---|---|---|---|
| Dequantised-expert LRU | `larql-compute/src/cpu/ops/moe/cache.rs` (`LARQL_MOE_CACHE_ENTRIES`, default 256) | dequantised f32 copies of experts on the legacy CPU MoE path, keyed by mmap pointer | no; the mapped source pages stay with the OS |
| Metal expert-buffer cache warmup | `larql-server/src/routes/expert/warmup.rs:85-94` | the server's Metal expert buffers | no |
| `ResidencyManager` / `auto_pin()` | `larql-vindex/src/index/storage/residency.rs` | pinned legacy-index layers within a memory budget | no; it covers the legacy index, not VINDEX3 expert banks |
| `ResidencyBudget` | §1.1 | declared bytes, at planning | no; it plans and refuses, and does not enforce |

**No VINDEX3 runtime expert cache, and no cap on mapped-page residency,
exists on main.** Revision 1's unqualified "no runtime expert cache" was wrong
[A1 #21].

`MADV_DONTNEED` has six call sites in `crates/`, not the two revision 1 listed
[A1 #22].

Two are library functions:
- `release_mmap_pages` (`index/storage/gate_accessors/mod.rs:312`), which
  the server's walk-FFN handlers call when `release_mmap_after_request` is
  set;
- `release_pages` (`index/storage/vindex_storage/mmap_storage.rs:376`).

Four are probes:
- the GLM probe's `evict()` (`glm_moe_residency/probe.rs:127`);
- `examples/mmap_cold_read_probe.rs:61`;
- `examples/readrate.rs:20`;
- `examples/vindex3_residency_probe.rs:191`.

The library functions document it as advisory on Darwin, and frontier §11
records that it did not evict.

### 1.4 Instruments that do not exist, or are not recoverable

Any MEM-001 protocol that needs one of these would first need it built or
recovered, as a **proposed prerequisite**. None is proposed for building here.

| Needed for | Status |
|---|---|
| Feeding a VINDEX3 run's routing into the retention simulator | **missing.** VINDEX3 records an in-memory `routing_trace` capture. The simulator reads dec-bench capture pools, and the JSONL trace is not called on the VINDEX3 path. It needs an exporter or adapter [A1 #11] |
| A contention control | **unrecovered.** `contention.sh` is named only in `e2e0392f:k3-prefetch-2-notes.json` ("written and unrun") and is in no git ref [A1 #18] |
| An unwitnessed latency record for the 5-sample round | **unrecovered.** That round's caveat says "the unwitnessed distributions below are the latency record", but no such distributions are in the file (§2.2) |
| Eviction on an idle machine | **unreliable** by main's own record (frontier §14). The working methods are displacement (`evict-bank.sh`, branch only) and virgin layers (GLM probe) |
| Expert attribution of faults or device reads | **missing.** Faults are process-wide, and nothing counts device reads |
| A runtime cap on VINDEX3 mapped-page residency | **missing** (§1.3) |

### 1.5 Expert layout, which decides bytes per expert

- **`Fused`** (packed GPT-OSS): one `gate_up` stream and one `down` stream,
  each expert-major inside its tensor. This is per the type documentation;
  no container's region table was checked [A1 §4.10].
- **`Separate`** (per-expert bank): three whole matrices per expert, each its
  own stored operand. The Kimi-Linear bank has 19,968 operands at
  **4,718,592 B** each (2304 × 1024 × 2, BF16). That is one matrix; an expert
  is three, 14,155,776 B. 19,968 × 4,718,592 = 94,220,845,056 B, which is the
  bank's payload in `index.json`.
- **PREFETCH's requests are exactly page-rounded contiguous runs:**
  - (624 × 288 + 202) × 16,384 = 2,947,710,976 B = the 2.948 GB coalesced
    request;
  - 624 × 289 × 16,384 = 2.955 GB, the "span".
  This fits each expert's three matrices being one run. No writer-side
  ordering guarantee was found.
- **GGUF interleaving.** #27149 claims 111× read amplification for one expert
  read through mmap. That figure is *claimed, not shown*, and inconsistent
  with the same post (6,912 / 192 = 36) [A1 #53].

## 2. Prior evidence: GLM and the Kimi PREFETCH rounds

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
- **Requested equals paged:** 192.0 MiB logical against 192.1 MiB physical,
  ratio 1.000. 12,295 faults × 16 KiB = 192.1 MiB (§11).
- **Access policy makes no difference on an idle machine** (§13: load 2.7–3.6,
  five rounds):

  | Arm | Median | Spread |
  |---|---|---|
  | demand | 210.4 ms | 1.4 % |
  | advise | 209.1 ms | 0.5 % |
  | touch | 208.8 ms | 1.0 % |
  | warm | 17.8 ms | — |

  All three cold arms take 12,295 faults, about 17.0 µs per fault. An
  earlier set taken at load ~30 was discarded.
- **The fetch shape matters** (§14): replacing faults with 48 `pread`s of
  4.0 MiB:

  | Arm | Cold routed stage |
  |---|---|
  | fault path | 212.1 ms |
  | serial `pread` | 121.2 ms |
  | `pread` × 8 threads | 90.0 ms (2,759 MiB/s, 2.36× faster) |

  The output is byte-identical. Parallel n=7, across virgin layers.
- **Reuse** (§12, 128 real tokens): 225 of 288 experts touched. LRU and
  Bélády hit rates at 0.25–6.75 GiB per layer. For example, at 2 GiB: LRU
  42.1 %, LRU excluding first touch 53.9 %, Bélády 62.2 %.
- **Instrument lessons:**
  - A `pread` populates the same unified buffer cache, so the cold check must
    precede the fetch.
  - Eviction depends on memory pressure.

**Scope.** One layer of one model, readahead off, an external drive. This is
direct evidence for that configuration, not for Kimi-Linear, for readahead
on, for internal storage or for whole-token decode.

### 2.2 Kimi-Linear PREFETCH: three rounds

**The subject is confirmed by source** [A1 #34]:
- `k3-v4-warm-latency.json:6` names
  `~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`, with prompt
  [1008, 10484, 318].
- PREFETCH-1's notes record routing fingerprint `ebe045dc40cce5b1`, "the same
  as V4", and the same generated ids.
- The bank is BF16: `target.expert_bank@BF16`, payload 94,220,845,056 B.

`lift2.vindex3` is **not in `~/chris-models` now**. Its bytes and digest are
unavailable. `.s6.vindex3` was a displacer in the eviction script and is now
"the declared latent-norm refusal fixture" (`continuation-mem-1.json:31`).
Whether `.s7` (expert-bank `payload_sha256` `3f12d102…`, the same as `.s6`)
holds lift2's bank is **unverified**.

**The measured token** is the first generated token after a three-position
prompt. "Same subject across rounds" rests on routing fingerprint and
generated ids, **not on a digest** [A1 #40].

| | Round 1: PREFETCH-1 | Round 2: witnessed 5-sample follow-up | Round 3: PREFETCH-2 preliminary |
|---|---|---|---|
| Record | `k3-prefetch-notes.json` on main | `follow_ups.witnessed_distributions…` in `e2e0392f:k3-prefetch-notes.json`; summarised in `aae20ae3:k3-prefetch-2.json` (`already_measured`) | `e2e0392f:k3-prefetch-2-notes.json` |
| Samples | 1 per arm | 5 per arm | 1 per arm |
| Machine | UNQUALIFIED: AC power, load 5–13, `mds_stores` indexing, which the notes attribute to the eviction read | **not recorded** | UNQUALIFIED: peer rustc build, `mediaanalysisd`, `mds_stores`; qualified warm arms refused; cold arms `--unquiet-ok` |
| Cold state | `evict-bank.sh`; binding witness 0.000 GB, 3/3 | eviction-verified cold per sample | `evict-bank.sh`; binding witness 0.000 GB |
| Witness inside the timed token | no | **yes, ~70 ms** (PF-N4) | no |
| Demand, cold token | 1253 ms | 1205 ms p50 (1141–1250) | 602 ms |
| Advise | 598 ms (prefetch stage 504.5) | 634 ms p50 (627–864; prefetch 557) | — |
| Coalesced | — | 619 ms p50 (612–628; prefetch 544) | 591 ms (prefetch 514) |
| Gated | — | — | 607 ms (prefetch 528) |
| Touch | 1358 ms | — | — |
| Major faults | 73,516 / 73,519 / 73,518 | 73,520 / 73,518 / 73,518 | 73,518 |
| Resident delta | 1.199 / 1.206 / 1.205 GB | 1.204 / 1.206 / 1.206 GB | 1.206 GB |
| Resident before the loop | not witnessed | demand 59.1 %; advised arms 100.0 % | — |
| Warm token | 76 / 87 / 80 ms | ~150 ms (witness-inflated) | 74 / 83 / 80 ms |

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

### 2.3 What the rounds show, and what they do not

**Direct observations.**
- Two rounds (1 and 2) show the advised arms at about half the demand cold
  token. Round 3 does not.
- The advised and coalesced arms sit at about 590–634 ms in **every** round.
  **Only the demand arm moves.**
- Fault counts and resident deltas agree across all three rounds.

**Per fault, on one basis** (whole-token wall time ÷ major faults; revision
1's 16.4 and 8.2 µs figures used two different bases [A1 #28]):

| Measurement | Calculation | Per fault |
|---|---|---|
| Round 1, demand | 1253 / 73,516 | 17.0 µs |
| Round 2, demand | 1205 / 73,520 (includes ~70 ms of witness) | 16.4 µs |
| Round 3, demand | 602 / 73,518 | 8.2 µs |
| GLM, demand, idle | 210.4 / 12,295 | 17.1 µs |

On this basis, **round 3 is the outlier**. The two other Kimi rounds, and
GLM's idle machine on a different model and drive, all sit near 16–17 µs per
demand fault.

**Inferred explanations, none established:**
- **(i) Contention slowed demand in rounds 1 and 2.** Weakened: round 3 was
  also heavily contended and ran demand fastest, and round 2's machine state
  is unrecorded.
- **(ii) Round 3's demand ran with more faults in flight**, or against a
  different device or cache state.
- **(iii) The madvise path helps on the Kimi container but not on GLM**
  because of readahead, mapping advice, the drive or the layout.
  - GLM set `MADV_RANDOM`.
  - Which mapping helper the VINDEX3 segments use was **not checked**:
    `mmap_util.rs` offers both `MADV_RANDOM` and `MADV_SEQUENTIAL`.

**GLM's idle result is consistent with a contention reading, but it does not
show that contention caused the PREFETCH differences.** Model, layout, mapping
advice and drive all differ.

**The byte accounting is reconciled** [A1 #31]:
- Round 2 witnessed 59.1 % of the 2.948 GB selection already resident before
  the demand loop: the prompt's own reuse of experts across positions.
- 2.948 × (1 − 0.591) = 1.206 GB = the observed resident delta.
- PREFETCH-2's gate "issued 84 of the 202 possible requests, asking for
  1.219 GB".

Revision 1 compared a whole-selection request with a cold token's resident
delta without this reuse. The four quantities stay distinct:

| Quantity | Source | Counts |
|---|---|---|
| Requested bytes | `PrefetchReport.requested_bytes` (branch) | page-rounded ranges asked of the OS |
| Resident delta | mincore before against after, bound regions | pages that became resident in the measured token |
| Fault activity | `ru_majflt` × 16 KiB | page-ins via the fault path, process-wide |
| Device reads | no instrument | — |

## 3. Imposing a memory budget

| Regime | Claim it can support | Available |
|---|---|---|
| **A. Model bytes exceed physical RAM** | feasibility of oversized execution | **Recorded, not verified today.** GLM-5.3-Flash: 42 sparse layers × 6.75 GiB = 283.5 GiB expert estate against 128 GiB of RAM (frontier §12). The checkpoint (~306 GB) is recorded on `model-drive` (§15), which was **not mounted** on 2026-10-31. Only per-layer probing has run; whole-model execution has not. In `~/chris-models` the largest entries are three of ~91.5 GiB (the Kimi-Linear HF directory, `.s6` and `.s7`); none exceeds RAM [A1 #36] |
| **B. Model fits; expert residency is held below its footprint** | how paging and caching behave under controlled pressure | needs a method from the next table |

Methods for regime B:

| Method | Constrains | Verified by | Known interference |
|---|---|---|---|
| Software expert budget over VINDEX3 mapped pages | the expert bank only | expert page attribution and mincore (§1.1) | **does not exist** (§1.3). Eviction is unreliable on an idle machine (§1.4), so "evicted" must be witnessed. It is a software budget, never an RAM limit |
| An explicit fetch path (`pread` into owned buffers) under a byte budget | the bytes the fetch holds | the fetch's own accounting; the cold check must precede the fetch | shown per layer on GLM (§2.1), not on VINDEX3 execution; `pread` also fills the shared buffer cache |
| System-wide memory balloon (wired allocation) | all of RAM | `vm_stat`, mincore | changes Metal headroom, the file cache for every process and the compressor |
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
| [#28248](https://github.com/ggml-org/llama.cpp/discussions/28248) "Persistent expert slot pool for MoE CPU offload" (2026-09-02) | VRAM slot pool for experts that live in CPU RAM; **no SSD tier** | single RTX 4090, Qwen3.8-Flash-Next UD-Q3_K_XL, 8K context: tg512 11.55 ± 0.41 → 21.21 ± 0.46 t/s at 64 slots | reported as measured. Its "bit-exact" refers to a backend unit matrix (40/40). End-to-end greedy output diverged once on a near-tie, and the RFC reports PPL 3.3096 against 3.3262 [A1 #48]. A commenter reports that argmax moved on 2–6 % of tokens while PPL moved < 1 %, in a related fork |
| [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) "MoE offload to disk with on-demand paging" (2026-05-19) | Metal shared-memory slot pool, loading experts with `pread` from GGUF (not OS paging), on Apple Silicon | Qwen3-30B-A3B Q6_K, M3 Pro 36 GB: vanilla 38.1 tok/s; 128 slots 30; 64 → 27.5; 32 → 22.1. M1 Pro 16 GB "13 tok/s" is a one-line update | reported as measured. (38.1 − 30)/38.1 = 21 %, but the vanilla and pool rows use different flags (`--no-mmap`, `--no-warmup`, `-ub 10`), so 21 % is **not** isolated pool overhead [A1 #50] |
| [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149) "Expert-Aware SSD Streaming" (2026-08-15) | standalone prototype (TinyGiant) | I/O on an M1 MacBook Pro 16 GB: 2.97 GB/s sequential, 6.06 GB/s double-buffered; headline 4.7 tok/s | the headline is **real I/O with simulated compute**. The C benchmark (real BLAS, `pread`; an M1 16 GB / MacBook Air per the author) reports **pipelined 2.40–4.45** tok/s (sequential 2.20–4.46) at **controlled** 50–100 % hit rates, not routing-produced ones [A1 #51, #52]. The 111× amplification claim is inconsistent with the same post (§1.5) |

**Identity across runtimes.** LARQL executes VINDEX3 and llama.cpp executes
GGUF. The same source weights, quantisation and prompt tokens can be
compared; the same file cannot, and layout is one of the variables.
#23324 is the only report that runs a real model's routing with real compute
and on-demand loads on Apple Silicon [A1 #54].

**The two runtimes' instruments are not equivalent:**

| Measurement | LARQL | External runtime | Comparable as is? |
|---|---|---|---|
| Wall time per token | yes | yes (`llama-bench`) | only under one harness and quiet gate |
| Process faults | `ru_majflt` in-process | for a child the harness spawns (`wait4` / `RUSAGE_CHILDREN`), or `proc_pid_rusage`; not via `getrusage` on an arbitrary process [A1 #45] | process-wide, if collected the same way for both |
| File-page residency | mincore in-process; expert attribution | an **observer process** mapping the same file could see the shared cache state | only if one observer serves both arms; unverified on macOS |
| Expert ids per position | `routing_trace` capture | #28248's fork mentions a per-layer id readback; not shown to be exported | not until shown |
| Bytes per expert | region table | GGUF tensor offsets; layout differs by design | by computation, not observation |
| Declared budget or refusal | yes | `n_slots` | no: different mechanisms |

## 5. Confounds the protocol must separate

1. **File cache versus process residency.** Mapped file pages and `pread`
   reads share the unified buffer cache (frontier §14), so a new process can
   inherit residency.
2. **Eviction state.** Advice-based eviction fails on an idle machine
   (frontier §11, §14). Displacement eviction reads ~184 GB just before the
   arm. Virgin layers avoid both, but only where the geometry repeats.
3. **Device contention and machine state.** Round 3 shows that contention
   alone does not predict a slow demand arm, and round 2's state is
   unrecorded.
4. **Instrument cost.** The residency witness costs ~70 ms per token. The
   residency claims rest on witnessed runs and the latency claims on
   unwitnessed ones, and round 2's unwitnessed record is unrecovered.
5. **Mapping advice and readahead.** GLM ran readahead off. The Kimi mapping's
   advice is unchecked.
6. **Faults as a proxy for bytes.** They agreed one to one in every LARQL
   record (§1.1). That is evidence for those configurations, not a law, and
   device reads are unmeasured.
7. **Decode cost unrelated to memory.** At about 54 % hit with deep fetch,
   GLM's arithmetic puts the bottleneck flipping to expert compute (frontier
   §14). Warm is the ceiling, and it has to be measured.
8. **Routing identity across runtimes.** It needs recording before any
   working-set comparison.
9. **One run is not a distribution.** Rounds 1 and 3 are single samples.

## 6. Candidate: extending and replicating the GLM result

**This is a candidate for review, not an authorised experiment.** It sets no
thresholds, picks no winner and runs nothing. If it is accepted, it enters a
freeze with the §7 answers before any arm runs.

Residency prediction has been **demonstrated once**: GLM, one layer,
readahead off, coverage 1.0003 (§2.1). A candidate MEM-001 therefore
**replicates or extends that result**, and does not present it as new.
Candidate directions, smallest first:

1. **Replicate GLM §11's predicted-versus-observed page account on the Kimi
   bank.** Use the expert page attribution of §1.1 on one routed layer with
   the GLM probe's method (virgin layers, cold check before the fetch).
   - **Prerequisite:** a pinned container whose bank is proven equal to
     lift2's, or a new subject pinned by digest (§7, Q1).
   - **Question:** does coverage ≈ 1 survive a different model, layout and
     mapping advice?
2. **Explain the three PREFETCH rounds** (MEM-001A). Repeated cold demand and
   advise arms on the pinned subject, each with its own demand baseline,
   under recorded machine state, with the mapping advice recorded, and with
   unwitnessed latency kept separate from witnessed residency.
   - **Prerequisites:** a committed contention control (§1.4) and a working
     eviction method (§5.2).
   - **Question:** which of §2.3's inferred explanations survives?
3. **Extend GLM §14's `pread` result to VINDEX3 whole-token execution.** PF-N5
   already names a file-descriptor read path as the remaining lever on Kimi.
   - **Prerequisite:** a non-mapped fetch realisation in VINDEX3 (frontier
     §14: "a representation-and-loader question"). That is an implementation
     step, and out of scope until reviewed.
   - **Question:** is the 2.36× per-layer stage gain attributable to fetch
     shape and not to incidental caching, and does it survive whole-token
     decode?
4. **Working-set accounting across tokens** (MEM-001B). Reuse curves on real
   tokens like GLM §12, for the Kimi subject.
   - **Prerequisite:** the VINDEX3-routing → simulator adapter (§1.4) and real
     region sizes (§1.5), not `K3_ROUTED_EXPERT_BYTES`.

Each direction keeps three categories separate: direct observations (pages,
faults, wall time), inferred explanations, and causal claims. A causal claim
needs an intervention arm with its own baseline.

## 7. Questions for the freeze

1. **Subject identity.** lift2 is gone. Is `.s7`'s bank proven equal to
   lift2's (by digest against a recorded lift2 digest, if one exists), or is
   a new subject pinned? Which model, container bytes, tensor layout and
   digest?
2. **Regime A.** Is GLM-5.3-Flash on `model-drive` the regime-A subject? Is
   whole-model execution of it in scope, or only per-layer probing?
3. **Regime B method.** Software budget over mapped pages, explicit fetch
   under a byte budget, or none? Any claim from it is a software-budget claim.
4. **Eviction method.** Displacement, virgin layers or another, with its
   failure modes stated and witnessed per arm.
5. **Contention control.** Re-written and committed before any arm that
   relies on it.
6. **PREFETCH-2 and BW10.** Each branch: merged first, carried in, or left?
7. **Mapping advice.** Record which `madvise` the VINDEX3 segments carry, and
   whether it is an arm variable.
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

**Independent audit 1 (`43388264`).** Failed: 7 BLOCKING, 17 INCORRECT. It is
preserved in `mem-001-reconnaissance-audit-1.md`. This revision corrects it.

**Maintainer direction after the audit (2026-10-31):**
- Recover LARQL's existing memory-residency baseline rather than rediscover
  it.
- Reframe MEM-001 around whether residency prediction **generalises**, and
  whether it yields **reliable** performance.
- Treat GLM's idle-machine result as consistent with, but not proof of, a
  contention explanation.
- Choose the next experiment only after this revision passes an independent
  re-audit, by deciding which existing result needs replication and which
  capability is genuinely missing. The `pread` result (§2.1, §6.3) and the
  coverage result (§2.1, §6.1) are the leading candidates.

MEM-001 is frozen only once the reproducibility, accounting and identity
problems are resolved or explicitly bounded.
