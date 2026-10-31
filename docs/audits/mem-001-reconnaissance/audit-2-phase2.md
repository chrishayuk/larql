# MEM-001 reconnaissance, revision 2: phase 2, regression against audit 1

| Item | Value |
|---|---|
| Revision under review | `b82d70b3:docs/mem-001-reconnaissance.md` |
| Audit 1 | `b82d70b3:docs/mem-001-reconnaissance-audit-1.md`, 56 findings, written against revision 1 (`43388264`) |
| Revision 1 | read at `43388264:docs/mem-001-reconnaissance.md` |
| Phase-1 report | `phase1-independent.md`, unchanged. "P1-n" below refers to its findings |
| Rules | read-only; no builds or runs; no PR access |

The document's own "[A1 #n]" tags were not accepted as evidence. Every disposition below was re-checked against the primary source named.

## A. Disposition of audit 1's 56 findings

Values:

- **RES**: resolved.
- **PART**: partially resolved.
- **NOTRES**: not resolved.
- **NLA**: no longer applicable.
- **N/A-V**: verified in audit 1 and still correct.

| A1 # | A1 verdict and severity | Disposition | Revision-2 location and evidence | What remains |
|---|---|---|---|---|
| 1 | V/I | N/A-V | Header still reads "at main `51579782`". This is still an ancestor of `origin/main` `5819050f`, and no cited path has drifted | — |
| 2 | V/I | N/A-V | §1.1 now cites `physical.rs:269`, which matches | — |
| 3 | V/I | N/A-V | §1.1 now names `residency_curve.rs:195`, which matches | — |
| 4 | A/NB | RES | §1.1 row 1: "**Summed callers map `None` to 0** (`weights/loaded.rs:37`, `prepared/residency.rs:417-423`), so a '0.000 GB resident' witness cannot tell cold from a failed read". Sources: `loaded.rs:37` `unwrap_or(0)`; `residency.rs:423` | P1-4, which is new: the summed figure also counts boundary pages twice |
| 5 | V/I | N/A-V | §1.1 row unchanged; `realization.rs:247` and `prefetch.rs:66-127` match | — |
| 6 | X/NB | RES | §1.1 MappedAccess row: "the first round did not witness this. The 5-sample follow-up … recorded **100.0 % resident before the loop** in both advised arms". Source: `e2e0392f:k3-prefetch-notes.json` `follow_ups.witnessed_distributions….advise/coalesced.resident_before_loop` = "100.0% in every sample" | — |
| 7 | V/I | N/A-V | §1.1 now names `residency_curve.rs:38-65` | — |
| 8 | X/NB | **PART** | The rev-1 error is gone: 16.3 KB per fault is no longer cited as decoupling. The replacement in §1.1 is "In every PREFETCH arm **and in the GLM probe**, faults tracked 16 KiB pages one to one … Decoupling by readahead or `pread` is possible in principle, **but no LARQL evidence shows it**". §5.6 repeats it ("one to one in every LARQL record"). This is audit 1's suggested text verbatim, and it is wrong. Frontier §14 (`51579782`) records the GLM probe's own `pread` arms fetching the same bytes in "48 reads" in place of "12,295 faults". The V3 `second_run` in `k3-residency-vertical-notes.json` records "0 major faults but 470,846 MINOR faults" | Scope the sentence to fault-path arms, and cite GLM §14 as decoupling evidence (P1-7) |
| 9 | X/**B** | RES | §1.2 GLM row: "`evict()` at `probe.rs:116` … `MADV_DONTNEED` did not evict on Darwin; `MADV_FREE_REUSABLE` returned `EPERM`. `msync(MS_INVALIDATE)` stopped evicting on an idle machine". §1.2 `evict-bank.sh` row: "streams other large files (`DISPLACERS`: the HF directory and `…s6.vindex3`) … This was PREFETCH-1's **and** PREFETCH-2's cold-state method". Headline: "The eviction routine on main is the GLM probe's, and main records it failing on an idle machine". §1.4: "The working methods are displacement (`evict-bank.sh`, branch only) and virgin layers". §6.2 makes a working eviction method a prerequisite; §7 Q4 asks for it. Primary sources: `51579782:…/glm_moe_residency/probe.rs:116-128`; frontier §11 ("`MADV_DONTNEED` did not evict … `MADV_FREE_REUSABLE` returns `EPERM`") and §14 ("`msync(MS_INVALIDATE)` stopped reclaiming pages … virgin sparse layers"); `aae20ae3:scripts/evict-bank.sh` `DISPLACERS=(…Kimi-Linear-48B-A3B-Instruct, …s6.vindex3)`; `51579782:k3-prefetch-notes.json` `cold_state_protocol` ("`evict-bank.sh` streams the HF checkpoint … and the .s6 container"); `aae20ae3:k3-prefetch-2.json` `protocol.cold` ("`evict-bank.sh` before every sample") | Non-blocking residue, all new: "so cold arms use virgin layers" generalises §14's explicit-fetch arms to every cold arm (P1-14). The committed script reconstructs a session scratch file (P1-17). "~184 GB" is GiB (P1-18) |
| 10 | V/I | N/A-V | §1.2 adds that `seq` is a per-layer call index (`ffn/expert_weight/trace.rs:11`) | — |
| 11 | U/**B** | RES | §1.4 row 1: "**missing.** VINDEX3 records an in-memory `routing_trace` capture. The simulator reads dec-bench capture pools, and the JSONL trace is not called on the VINDEX3 path. It needs an exporter or adapter". §1.1: "in-memory capture. No exporter feeds the retention simulator". §1.2 JSONL row: "**not called by VINDEX3 opplan execution**". §6.4 lists the adapter as a prerequisite. Primary sources: `51579782:…/opplan/exec/routing_trace.rs:10-31` (a static `Mutex<Option<Vec<Vec<usize>>>>`, with no writer to disk); `production/plan_backend.rs:375`; `k3_ledger/retention_report.rs:24-25` (`CapturePool::open`, `SelectionTrace::from_routing_pool`); `git grep moe_route_observe 51579782 -- crates/larql-vindex` finds only a JSON fixture list, with no call | The rev-1 "Record → Predict on main" chain is gone. Residue: `--expert-bytes` already overrides the constant (P1-20), and `scripts/glm_expert_reuse.py` takes a generic JSON trace (O-2). Both are non-blocking |
| 12 | X/NB | RES | §1.1: "the **routing-identity witness** in V4 and PREFETCH (fingerprint `ebe045dc40cce5b1`)". Source: PREFETCH-1 `arms.determinism`; V4 `A_quiet_QUALIFIED.routing` | — |
| 13 | V/I | N/A-V | §1.2 now says the constant is a K3 MXFP4 expert (`retention.rs:58-60`) | P1-20 (new): the constant is only a default, overridable by `--expert-bytes` |
| 14 | V/I | N/A-V | `accounting/budgets.rs:101` now cited, and matches | — |
| 15 | X/NB | RES | §1.1: "for mappings it also **reports** mincore-resident bytes (`mapped_resident`) … **never reconciled**". Source: `reconcile.rs:63-81` | — |
| 16 | V/I | N/A-V | Unchanged; `41811c1d` still the branch tip | — |
| 17 | X/NB | RES | §1.2: "BW-A gate recorded on that branch in `docs/diagnoses/bw10-live-gate.md`". The file exists at `41811c1d` | — |
| 18 | X/**B** | RES | §1.4: "**unrecovered.** `contention.sh` is named only in `e2e0392f:k3-prefetch-2-notes.json` ('written and unrun') and is in no git ref". §6.2 makes "a committed contention control" a prerequisite; §7 Q5 says "Re-written and committed before any arm that relies on it". The §1.2 PREFETCH-2 row no longer lists it, and no other occurrence of `contention.sh` exists in revision 2 (grep). Primary sources: `git log --all -S contention.sh` finds only `e2e0392f` (plus mem-001 commits); no path matching `*contention*` exists in any ref; `e2e0392f:k3-prefetch-2-notes.json` `still_to_run_when_the_window_opens[2]`: "(`contention.sh` is written and unrun)" | — |
| 19 | A/NB | RES | §1.2: "**Notes only on `wip/k3-prefetch-2-2026-09-23` (`e2e0392f`)**, a parked work-in-progress commit. A third branch … (`4a3823e2`)". Verified tips and contents | Minor: `evict-bank.sh` is said to be "branch `k3-prefetch-2` only", but it is also on `e2e0392f` (P1-2, informational) |
| 20 | V/I | N/A-V | Adds "that container is not in `~/chris-models` now". Verified by `ls` | — |
| 21 | X/NB | RES | §1.3 table and "**No VINDEX3 runtime expert cache, and no cap on mapped-page residency, exists on main**". Sources: `cpu/ops/moe/cache.rs:19,158`; `warmup.rs:82-94`; `index/storage/residency.rs:13,32` | P1-24 (new): the VINDEX3 Metal path registers expert-bank regions as Metal buffers, and a ~45 GB wired-collector wall is recorded |
| 22 | X/NB | **PART** | §1.3 now lists the sites, but says "**six** call sites … Two are library functions: `release_mmap_pages` (`gate_accessors/mod.rs:312`) … `release_pages` (`mmap_storage.rs:376`)". `gate_accessors/mod.rs:327` shows `release_mmap_pages` only calls `self.storage.release_pages()`; the single library call is `mmap_storage.rs:386`, `unchecked_advise(UncheckedAdvice::DontNeed)` | Five call sites (P1-27). Audit 1's own list made the same double count |
| 23 | X/NB | RES | §1.1 row "Expert page attribution `runtime::residency::account` / `ExpertRegion`" at `runtime/residency.rs:163,175`, used by both probes | — |
| 24 | X/**B** | RES | Headline, first bullet: "`docs/frontier-execution-programme.md` §11–14 … coverage **1.0003** … ratio **1.000** … within **0.8 %** … `pread` … **2.36×** … LRU and Bélády hit rates at six budgets". §2.1 is a full section. §6: "Residency prediction has been **demonstrated once** … replicates or extends that result, and does not present it as new". "The one prior latency result" is gone. Primary source: `51579782:docs/frontier-execution-programme.md` §11 (coverage 1.0003; "Logical bytes requested == physical bytes paged in, ratio 1.000"), §13 ("The three access policies are the same to within 0.8 %"), §14 (212.1 / 90.0 ms = 2.357), §12 table. All numbers re-verified in P1-33 | Prior evidence is still incomplete in ways audit 1 did not raise: V3 rung 7 (P1-47, BLOCKING) and moe-routing-locality (P1-67, BLOCKING). See §C |
| 25 | A/I | RES | Cited in §1.2 (cold-read probe) and §3 ("macOS can't easily cap mmap residency") | O-1 (new): the diagnosis's own numbers (100 µs scattered, 18 µs sequential, one-to-one faults) are not used |
| 26 | A/I | RES | §1.2 "Non-plan memory" row | P1-23 (new, informational): `runtime_stats.rs:199` is peak RSS (`ru_maxrss`), not live RSS |
| 27 | V/I | N/A-V | Round-1 column values re-verified (P1-38) | — |
| 28 | V/I | **PART** | §2.3: "Per fault, **on one basis** (whole-token wall time ÷ major faults; revision 1's 16.4 and 8.2 µs figures used two different bases)". The arithmetic is right (P1-44). But the GLM row (210.4 / 12,295) is a **one-layer, threaded routed-stage** median, not a whole token: frontier §13 says "about 1.8 faults in flight", with a serial cost of 26.5–30 µs per fault. And 16.4 µs = 1205 / 73,520 is round 2's witnessed whole-token p50, which is the same basis as 8.2 = 602 / 73,518, not a different one. The source PF2-N1 / addendum calls it PREFETCH-1's | "One basis" is false for the GLM row (P1-45). "Two different bases" mischaracterises the 16.4 / 8.2 pair (P1-46). Neither audit 1 nor revision 2 identified 16.4 as 1205 / 73,520 |
| 29 | X/NB | RES | §2.2 table, resident delta row: "1.199 / 1.206 / 1.205 GB". Source: PREFETCH-1 `arms.*.resident_delta_gb` | — |
| 30 | X/**B** | RES | Headline: "**The Kimi PREFETCH evidence comes from three rounds, not two** … Two show the advised arms at about half … (1253 → 598 ms; 1205 → 634/619 ms p50); one does not (602 / 591 / 607 ms)". §2.2 has a Round 2 column with "Machine: **not recorded**" and "Witness inside the timed token: **yes, ~70 ms**". Primary sources: `e2e0392f:k3-prefetch-notes.json` `follow_ups.witnessed_distributions_5_samples_each_eviction_verified_cold`: demand `cold_token_ms` {p50 1205, min 1141, max 1250}, advise {634, 627, 864}, coalesced {619, 612, 628}, plus the `caveat`. Also `aae20ae3:k3-prefetch-2.json` `carried_in_uncommitted_from_prefetch_1.already_measured`. Within-round ratios: 1205 / 634 = 1.90 and 1205 / 619 = 1.95; with ~70 ms removed from both arms, 1135 / 564 = 2.01 and 1135 / 549 = 2.07. The halving holds either way | Not closed by #30, all new: the headline omits round 2's witness and qualification caveats (W-1 in §B2); round 3's discarded 3044 ms demand sample (P1-40); the cold-prompt figures (P1-41). V3 is a fourth Kimi cold record outside the PREFETCH framing (P1-47) |
| 31 | X/**B** | **PART** | Headline: "**The PREFETCH byte gap is already explained by its sources**: 59.1 % … 2.948 × 0.409 = 1.206 GB, the observed resident delta". §2.3: "**The byte accounting is reconciled**". §2.2: "The bank is BF16". §1.5 gives region sizes. Primary sources: `e2e0392f:k3-prefetch-notes.json` `…demand.resident_before_loop` = "59.1% of 2.948 GB in every sample — the prompt's cross-position expert reuse, measured"; `e2e0392f:k3-prefetch-2-notes.json` `the_gate_is_selective_not_silent` ("84 of the 202 … 1.219 GB"); `.s6/.s7` `index.json` `target.expert_bank@BF16` with payload 94,220,845,056; `k3-residency-vertical-notes.json` V3 `subject` ("bf16 per-expert bank 94.22 GB"). 2.948 × 0.409 = 1.2057 ✓. **The gap within the PREFETCH rounds is closed with source support** | Two problems remain. (1) The unqualified "reconciled" and "already explained" are contradicted by a main record of the **same token and prompt**: V3 rung 7 shows 0.976 GB paged, 59,670 faults, and 67 % resident from the prompt (`k3-residency-vertical-notes.json` `V3_execution_and_residency_curve.rungs.7_predicted_vs_observed`; V3-N3). PREFETCH-1's own `predictions_scored.P1` on main points at this ("this run paged 1.199 GB where V3's token paged 0.976 GB"). Across LARQL's records the accounting is **not** reconciled (P1-47, BLOCKING, new). (2) "Exactly page-rounded contiguous runs" in §1.5 came from audit 1 and goes beyond 3-d.p. data (P1-31, non-blocking) |
| 32 | V/I | N/A-V | §1.5 now gives the page-rounded arithmetic | See P1-31 |
| 33 | V/I | N/A-V | §1.5: 14,155,776 B per expert; 19,968 regions. Matches config | — |
| 34 | X/**B** | **PART** | Headline: "**The PREFETCH subject is `Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`, and it is no longer on disk.** The `.s6` container that revision 1 named was one of the files read to *evict* it". §2.2: "**The subject is confirmed by source** … `k3-v4-warm-latency.json:6` …". Primary sources: `51579782:docs/represent/forecasts/k3-v4-warm-latency.json:6` (`subject: ~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3 …`); PREFETCH-1 `arms.determinism` ("ebe045dc40cce5b1 (the same as V4)"); `aae20ae3:scripts/evict-bank.sh` `DISPLACERS`; `ls ~/chris-models` has no `lift2`; `continuation-mem-1.json:31`. **The misidentification is fixed with source support** | The replacement "**Its bytes and digest are unavailable.** … Whether `.s7` … holds lift2's bank is **unverified**" (audit 1's suggested text) omits main's own record. `51579782:docs/represent/forecasts/represent-real-evidence-1-notes.json:30`: "Its five segments are **byte-identical in size and spot-checked content** to the historical `s6` container (**APFS clones**; only `system_graph.json` differs by 33 bytes)". `.s6` and `.s7` `index.json` are byte-identical (expert-bank `payload_sha256` 3f12d102…). The lift2 → s6 → s7 chain is therefore recorded, short of a lift2 digest. Audit 1 §4.1 wrongly said the clone fact existed only in user memory. The subject-identity question that §7 Q1 and §6.1 rest on is mis-stated (P1-37, BLOCKING, new) |
| 35 | V/I | N/A-V | Config unchanged and re-verified | — |
| 36 | A/NB | **PART** | §3 fixed: "three of ~91.5 GiB (the Kimi-Linear HF directory, `.s6` and `.s7`); none exceeds RAM" (`du`: 95,951,008 / 95,945,392 / 95,945,396 KiB). But §1.2 and §5.2 still say "reads ~184 GB". That is 2 × 91.5 GiB = 183 GiB = 196.5 GB, so GB and GiB are mixed again, and the figure contradicts §3 | "~183 GiB (~196 GB)" (P1-18) |
| 37 | V/I | NLA | Revision 1's "52 entries … 352 GB free disk" sentence is gone. §3's regime table has no disk-free figure (grep: no "352" or "52 entries") | — |
| 38 | A/NB | RES | §2.2: "**Carried observations, each from a single round**". PF-N2 is "from PREFETCH-1's UNQUALIFIED round"; for PF-N3, "single-pass artifact" is kept open. Sources: PREFETCH-1 `machine`; `aae20ae3:k3-prefetch-2.json` `G6_false_on_advise` | — |
| 39 | X/NB | RES | §2.2 Cold-state row: "`evict-bank.sh`; binding witness 0.000 GB" for rounds 1 and 3. Sources: PREFETCH-1 `cold_state_protocol`; PREFETCH-2 `identical_in_every_arm` | P1-17: the committed script is a later reconstruction |
| 40 | V/I | N/A-V | §2.2: "rests on routing fingerprint and generated ids, **not on a digest**" | — |
| 41 | U/NB | RES | §1.2: "the notes attribute Spotlight's indexing to that read". Source: PREFETCH-1 `arms.machine` | — |
| 42 | A/NB | RES | "Resident delta" is used throughout §2.2 and the four-quantity table | Informational (P1-39): the round 2/3 source keys say "paged" |
| 43 | V/I | N/A-V | §2.3: "**Inferred explanations, none established**" | New: §5.3 states one explanation as shown (P1-49) |
| 44 | V/I | N/A-V | The distinctions are kept in the tables | Exceptions, all new: P1-7, P1-49, P1-51; the witness juxtapositions in §B2 |
| 45 | U/NB | RES | §4 table: "for a child the harness spawns (`wait4` / `RUSAGE_CHILDREN`), or `proc_pid_rusage`; not via `getrusage` on an arbitrary process" | — |
| 46 | V/I | N/A-V | Still hedged ("unverified on macOS") | P1-61 (new): main's V3 `second_run` shows a fresh process's mincore seeing 10.485 GB of page-cache state left by other processes. The hedge is now stale against LARQL's own record |
| 47 | V/I | N/A-V | §3 cites `v3-disk-resident-mmap.md` | — |
| 48 | A/NB | RES | §4 #28248: "Its 'bit-exact' refers to a backend unit matrix (40/40). End-to-end greedy output diverged once on a near-tie, and the RFC reports PPL 3.3096 against 3.3262". Matches the discussion body | Informational (P1-55): the later 4.45 % flip measurement and the pooled-vs-full-GPU KLD 0 decomposition are omitted |
| 49 | V/I | N/A-V | Figures unchanged and re-verified | — |
| 50 | V/I | N/A-V | Figures re-verified | Audit 1's suggested addition, now in revision 2 ("the vanilla and pool rows use different flags"), is not shown by the source. The post gives only the pool command, and the vanilla row's flags are unstated (P1-58, new) |
| 51 | A/I | RES | §4: "I/O on an M1 MacBook Pro 16 GB … (an M1 16 GB / MacBook Air per the author)" | — |
| 52 | X/NB | RES | §4: "**pipelined 2.40–4.45** tok/s (sequential 2.20–4.46) at **controlled** 50–100 % hit rates". Matches the #27149 update table | — |
| 53 | U/NB | **PART** | "Measured" is gone: §1.5 now reads "*claimed, not shown*, and inconsistent with the same post (6,912 / 192 = 36)", which is audit 1's text | 111× and 36× can have different denominators, so they are not shown to be inconsistent. The post's real internal contradiction is "Each expert is a contiguous byte slice" (original) against "GGUF stores all 128 experts interleaved" (update) (P1-32) |
| 54 | A/NB | **PART** | The claim is now "#23324 is the only report that runs a real model's routing with real compute and on-demand loads on Apple Silicon", which is audit 1's text | The #27149 update claims an "End-to-end inference — Python engine generating verified correct text from Qwen3-30B-A3B using the contiguous cache". "Only" holds only for runs with a reported decode rate (P1-60) |
| 55 | V/I | N/A-V | The two prescriptive phrases are gone ("natural candidate", "earns its keep": no grep hit). §6 sets no thresholds | §8's "leading candidates" is attributed maintainer direction. Informational |
| 56 | A/I | RES | §2.2: "`A_quiet_QUALIFIED`, 75.4 ms p50, p95/p50 1.03, 2026-09-06, on `docs/k3-v4-warm-latency-qualified-arm` (`254377ce`) and in `e2e0392f`'s amended V4 notes" | — |

**Counts:**

| Disposition | Count | Findings |
|---|---|---|
| RESOLVED | 25 | — |
| PARTIALLY RESOLVED | 8 | #8, #22, #28, #31, #34, #36, #53, #54 |
| NOT RESOLVED | 0 | — |
| NO LONGER APPLICABLE | 1 | #37 |
| N/A-V | 22 | — |

**Original BLOCKING findings:**

| A1 # | Status |
|---|---|
| #9 | RESOLVED |
| #11 | RESOLVED |
| #18 | RESOLVED |
| #24 | RESOLVED |
| #30 | RESOLVED |
| #31 | PARTIALLY RESOLVED: the PREFETCH-internal gap is closed, but the unqualified "reconciled" is contradicted by V3 |
| #34 | PARTIALLY RESOLVED: the misidentification is fixed, but the identity-equivalence statement omits main's clone record |

Four of the eight partial resolutions (#8, #53, #54, and part of #31 via "exactly") come from revision 2 adopting audit 1's own suggested correction text, which was itself inaccurate.

## B1. Historical result traceability

Each row is classified on four dimensions:

- **model bytes**: checkpoint identity or digest;
- **structure**: container or checkpoint layout;
- **procedure**: code at a commit, invocation, machine state;
- **recording commit**: whether the code at that commit matches main.

| Result | Model bytes | Structure | Procedure | Recording commit / code match | Class | Missing |
|---|---|---|---|---|---|---|
| GLM coverage 1.0003, unselected 0.0 MiB (§11) | GLM-5.3-Flash checkpoint by **size only**: 328,326,771,576 B (305.78 GiB), 76,108 tensors, 62 shards (frontier §1, "measured … from origin/main 1e57b6d1"). No weight digest or HF revision. `glm_reference_sources.sha256` pins transformers sources, not weights. The checkpoint is on `model-drive`, **not mounted** on 2026-10-31 | Layer 3, 288 experts, 6.752 GiB bank, FP8 with scale siblings, "83–98 % density" interleaving (§11). Derivable from headers when the drive is present | Probe code `glm_moe_residency.rs` at `6874285f`. On main it is split into `glm_moe_residency.rs` + `probe.rs` (`b3097a3a`, 2026-09-26) plus `ProductionBackend::new()` (`6c144113`). A whitespace-stripped sorted-line comparison shows 3 lines differ, all module wrapping or the constructor, so the logic is **unchanged**. Usage: `glm_moe_residency <checkpoint-dir> <layer> <input.f32> [arms]`. The `input.f32` hidden state is not committed. The exact invocation, machine model and load for §11's run are not recorded | `6874285f` (2026-09-07), which committed the code and the doc together | **BACKGROUND ONLY** | Weight digest or revision; the input hidden-state file; the invocation; raw output; machine state for §11; a mounted volume |
| GLM ratio 1.000 (192.0 / 192.1 MiB; 12,295 faults) (§11) | as above | as above | as above | `6874285f` | **BACKGROUND ONLY** | as above |
| GLM 0.8 % access-policy equality (§13) | as above | as above | Same probe. Machine state **partially** recorded: "load 2.7–3.6 with nothing but the harness running, five rounds"; the load-30 set was discarded. Per-round raw values are not committed. The cold method for §13 is not stated (virgin layers are described only for §14) | `6874285f` | **BACKGROUND ONLY** | Raw per-round data; the cold method; the invocation; weight identity |
| GLM `pread` 2.36× (§14) | as above | 48 extents of 4.0 MiB ("three code matrices and three scale siblings") | Same probe with `pread-serial` / `pread-parallel` arms (present at `6874285f`, `probe.rs:463-476` on main). Virgin layers (which ones not named); "Medians; parallel n=7 … spread 62.0–79.6 ms"; the serial n is not given | `6874285f` | **BACKGROUND ONLY** | Layer ids used; serial sample count; raw data; weight identity |
| GLM LRU / Bélády table (§12) | as above | 288 experts, 6.75 GiB per layer | `scripts/glm_expert_trace.py` + `scripts/glm_expert_reuse.py` at `6874285f`, unchanged on main (no later commits touch them). The trace JSON is **not committed**. The prompt is not recorded in §12. The script's default prompt (`glm_expert_trace.py:86-93`) is **one paragraph about MoE residency repeated ×3**, with a default `--max-tokens` of 192. "128 real tokens" therefore implies a non-default invocation, and if the default prompt was used with `--max-tokens 128`, the trace contains repeated text, which inflates reuse. That last point is an inference | `6874285f` | **BACKGROUND ONLY** | Trace artifact; prompt and invocation (the repeated-text risk); weight identity |
| 283.5 GiB estate; ~306 GB checkpoint (§12, §13, §15) | Size from headers (frontier §1: 328,326,771,576 B = 305.78 GiB, so "306 GB" is GiB). No digest | 42 sparse layers × 6.75 GiB: arithmetic from config plus the one measured bank (6.752 GiB) | Header plan in 0.74 s at `1e57b6d1`. Reproducible from headers if the volume is mounted | `6874285f` | **BACKGROUND ONLY** (structural fact; it would become a reproducible baseline once the checkpoint is mounted and digested) | Weight digest; volume availability; unit (GiB) |
| Kimi PREFETCH round 1 (PREFETCH-1) | Subject `lift2.vindex3` (`k3-v4-warm-latency.json:6`), **absent**. Recorded as an APFS clone of `.s6` by size and spot-check (`represent-real-evidence-1-notes.json:30`); `.s6` ≡ `.s7` by `payload_sha256`. No lift2 digest. The routing fingerprint `ebe045dc40cce5b1` and generated ids give **behavioural** identity | 19,968 BF16 regions, 94.22 GB mapped | Code: PREFETCH-1 mechanism merged (`63bc7c12`, 2026-09-06; on main). The eviction script was a "session scratch file", committed later as a reconstruction (`aae20ae3`). Machine recorded as UNQUALIFIED (AC, load 5–13, `mds_stores`). Single sample. Raw logs not committed | `63bc7c12`, with the notes on main | **BACKGROUND ONLY** | lift2 digest; the actual eviction script; raw logs; qualified machine; repeats |
| Kimi round 2 (witnessed 5-sample follow-up) | as round 1 (behavioural identity by fingerprint) | as round 1 | Code was "built after #433's head was cut and stranded uncommitted", then ported forward in `aae20ae3` (`k3-prefetch-2-notes.json` `carried_in_from_prefetch_1`). The exact code that ran is **not** recoverable. The notes live only in a parked commit "Unreviewed and not built as a unit" (`e2e0392f` message). Machine state **not recorded**. The witness (~70 ms) is inside every latency figure. The promised unwitnessed distributions are absent | `e2e0392f` (parked WIP, 2026-09-23); not on main | **BACKGROUND ONLY**, the weakest of the set | Machine state; the exact code; unwitnessed latency; raw logs; review |
| Kimi round 3 (PREFETCH-2 preliminary) | as round 1 | as round 1 | Mechanism committed at `aae20ae3` (2026-09-06; "No performance claim is made here"). Whether the round ran on exactly that commit is not stated. Machine recorded as UNQUALIFIED, with process loads. One retained sample per arm, plus one discarded demand sample at 3044 ms | `e2e0392f` notes; mechanism on `aae20ae3`; not on main | **BACKGROUND ONLY** | Code identity of the run; repeats; raw logs |
| Kimi V3 rung 7 (cold token 1296 ms, 59,670 faults, 0.976 GB; prompt 7.545 GB) | lift2, as above. **No routing fingerprint**: the routing trace was added in V4 | as round 1 | Code: K3 vertical V3 (`68abfd70`, 2026-09-06 01:14, on main; `--residency-curve`). Run time recorded ("2026-09-05 23:09Z, page cache holding 0.000 GB … at binding"). Eviction method and machine state not recorded. Single run. Second run (10.485 GB already resident) recorded | `68abfd70` on main | **BACKGROUND ONLY** | Routing fingerprint; cold method; machine state; repeats; lift2 digest |

**Result:** nothing in revision 2's prior evidence is a REPRODUCIBLE BASELINE, and nothing is UNTRACEABLE. Every GLM and Kimi result is BACKGROUND ONLY.

- For GLM, the code is recoverable and matches main. The model bytes have no digest and sit on an unmounted volume, and inputs, traces and raw outputs are uncommitted.
- For Kimi, the subject is absent. Its identity rests on a fingerprint and a spot-checked clone record, and the round-2 code and machine state are unrecoverable.

Revision 2 presents the GLM results as a "baseline" to replicate (§6, §8) without saying any of this. This is new; see N-B4.

## B2. Instrumentation overhead (witnessed vs unwitnessed)

**The maintainer's claim is refuted.** Revision 2 does set witnessed round-2 latency beside unwitnessed figures without accounting for the ~70 ms witness cost, and the missing unwitnessed record is not stated as a gap in every place it matters.

Every place a witnessed figure appears beside an unwitnessed one:

| # | Location | Text | Accounted? |
|---|---|---|---|
| W-1 | Headline, bullet 2 | "Two show the advised arms at about half the demand cold token (1253 → 598 ms; 1205 → 634/619 ms p50); one does not (602 / 591 / 607 ms)" | **No.** The witnessed round-2 figures sit between unwitnessed rounds 1 and 3 with no witness or qualification caveat. Audit 1's own #30 correction asked for "it carries a ~70 ms witness cost". The within-round ratio survives (2.01–2.07× with the witness removed), so the qualitative claim stands, but the caveat is missing |
| W-2 | §2.2 table | Round 2 column beside rounds 1 and 3 | **Yes**: a "Witness inside the timed token: yes, ~70 ms" row and "~150 ms (witness-inflated)" for the warm figure. **But** §2.2 never says that this round's unwitnessed latency record is missing. Only §1.4 does, and §1.4 points back to §2.2 |
| W-3 | §2.3, direct observations | "Two rounds (1 and 2) show the advised arms at about half … Round 3 does not." | Partly: within-round, unaffected. No caveat that round 2 is a witnessed residency record, not the latency record. The source says it is not: PF-N4 says "the latency claim [rests] on unwitnessed ones", and so does revision 2's own §5.4 |
| W-4 | §2.3, direct observations | "The advised and coalesced arms sit at about 590–634 ms in **every** round. **Only the demand arm moves**." | **No.** 634 and 619 are witnessed (≈564 / 549 ms with the witness removed). They are compared in absolute ms with unwitnessed 598 / 591 / 607. The adjusted span is about 549–607 ms, so "590–634" is a cross-witness comparison. Demand round 2 (1205) is also compared unadjusted (≈1135) |
| W-5 | §2.3 per-fault table | "Round 2, demand: 1205 / 73,520 (includes ~70 ms of witness) = 16.4 µs" beside unwitnessed rows, and "all sit near 16–17 µs" | Noted but **not adjusted**: (1205 − 70) / 73,520 = 15.4 µs. The conclusion ("round 3 is the outlier") is unaffected. The "near 16–17 µs" grouping uses the unadjusted value, and the table is a latency comparison that, by §5.4's rule, should not use a witnessed run |
| W-6 | §2.3, byte accounting | "Round 2 witnessed 59.1 % …" | Proper use: a residency claim from a witnessed run |
| W-7 | §5.4 | "The residency claims rest on witnessed runs and the latency claims on unwitnessed ones, and round 2's unwitnessed record is unrecovered." | Explicit, but **contradicted by the document's own use** of round 2 as one of the "three rounds" of latency evidence (headline, §2.3) |
| W-8 | §6.2 | "with unwitnessed latency kept separate from witnessed residency" | Explicit and correct. No reference to the §1.4 gap, though §6.2 is where it matters |
| W-9 | §1.4 | "An unwitnessed latency record for the 5-sample round: **unrecovered** …" | Explicit gap. Verified: `e2e0392f:k3-prefetch-notes.json` has only `witnessed_distributions…` |

**Gap statement coverage:**

| Location | Gap stated? |
|---|---|
| §1.4 | yes |
| §5.4 | yes |
| §6.2 | implicitly |
| headline | **no** |
| §2.2 | **no** |
| §2.3 and its per-fault table | **no** |

**Minimal fixes:**

- Add "(witnessed; ~70 ms inside each figure; machine unrecorded; unwitnessed record unrecovered)" to the headline round-2 figures and to the §2.2 caption.
- In §2.3, either compare only within rounds or give the witness-adjusted values (≈1135 / 564 / 549 ms; 15.4 µs).
- State that round 2 counts as latency evidence only within-round, which is the scope PF-N4 allows.

## C. New in re-audit (not covered by audit 1)

Audit 1 did not raise any of the following. Items marked † were **introduced by adopting audit 1's own suggested correction**.

### BLOCKING

- **N-B1 = P1-47.** A fourth cold Kimi demand measurement on main, the V3 rung 7 (`51579782:docs/represent/forecasts/k3-residency-vertical-notes.json`, `V3_execution_and_residency_curve.rungs.7_predicted_vs_observed` and V3-N3), is omitted.
  - Same lift2 subject and prompt: cold token 1296 ms, 59,670 faults, 0.976 GB; ≈21.7 µs per fault; 67 % of the selection resident from the prompt, against PREFETCH's 59.1 %.
  - The touched set is 8.52 GB against PREFETCH-1's 8.92 GB, a difference no source explains. No routing fingerprint was recorded.
  - It contradicts "byte accounting is reconciled" and "already explained by its sources", and qualifies "Prior evidence … stronger than revision 1 reported".
  - Audit 1 listed this file in its coverage and still missed it.
- **N-B2 = P1-37.** Subject identity. Main records lift2 as an APFS clone of `.s6`, byte-identical in segment size and spot-checked content (`represent-real-evidence-1-notes.json:30`), and `.s6` ≡ `.s7` by `payload_sha256`. Revision 2's "bytes and digest unavailable … unverified" (†, audit 1's text, with audit 1 §4.1 wrongly saying the fact was only in user memory) mis-states the identity question that §7 Q1 and §6.1 depend on.
- **N-B3 = P1-67.** `docs/diagnoses/moe-routing-locality.md` (main, Gemma 4 26B-A4B) records an expert working set of about the full population (124.4 of 128 per layer) and "no small cacheable hot subset". The `larql moe-locality` LRU hit-curve instrument (`crates/larql-cli/src/commands/diagnostics/moe_locality/`, `docs/cli.md:1761`) is also absent from §1.2. Both bear directly on §6.4 (MEM-001B) and on §8's direction to "recover the existing baseline".

### NON-BLOCKING

- **N-B4.** Revision 2 calls the GLM results a "baseline" to replicate (§6, §8) without stating their provenance limits: no weight digest, an unmounted volume, uncommitted inputs, traces and raw outputs (§B1). The §12 reuse curve may come from the script's ×3-repeated default prompt. The prompt is unrecorded, and this is an inference.
- **N-B5.** Witness-cost juxtapositions W-1, W-4 and W-5, and the internal contradiction W-7 (§B2).
- **N-B6 = P1-7†.** "No LARQL evidence shows [decoupling by pread]" is wrong (see A1 #8).
- **N-B7 = P1-31†.** "Exactly page-rounded" goes beyond data reported to 3 d.p.
- **N-B8 = P1-32†.** The stated 111× vs 36× "inconsistency" is not established. The real contradiction in #27149 is contiguous vs interleaved.
- **N-B9 = P1-60†.** "#23324 is the only report …": #27149 claims an end-to-end real-compute engine.
- **N-B10 = P1-58†.** "The vanilla and pool rows use different flags" is not shown by #23324.
- **N-B11 = P1-61.** "Observer mincore unverified on macOS" is stale against V3's `second_run`.
- **N-B12 = P1-45 / P1-46.** The per-fault "one basis" is false for the GLM row, and 16.4 µs = 1205 / 73,520 (round 2), not a different basis.
- **N-B13 = P1-40.** Round 3's discarded demand sample (3044 ms cold, 2677 ms warm) is omitted.
- **N-B14 = P1-41.** Cold-prompt figures are omitted in every round. In round 3 the advised arms are still 1.25–1.43× faster than demand over the prompt.
- **N-B15 = P1-49.** §5.3 "Round 3 shows that contention alone does not predict …" treats CPU load as I/O contention, which no round measured.
- **N-B16 = P1-50.** The mapping advice called "not checked" is answerable. VINDEX3 maps segments with no advice (`physical.rs:96,135`, `read.rs:121`), while GLM used `MADV_RANDOM`. Both used the same WILLNEED code (`probe.rs:7-8,463-469`).
- **N-B17 = P1-24.** The VINDEX3 Metal path registers expert-bank regions (`metal_lowered/routed.rs:403-415`), and a ~45 GB wired-collector wall is recorded (later contradicted at 79 GB; `execution_cost.rs:22-24`). Absent from §1.2 and §3.
- **N-B18 = P1-20.** `--expert-bytes` overrides `K3_ROUTED_EXPERT_BYTES`, so §6.4's prerequisite as written is already met.
- **N-B19 = P1-27.** Five `MADV_DONTNEED` call sites, not six. Audit 1's list double-counted.
- **N-B20 = P1-18.** "~184 GB" is ≈183 GiB / 196.5 GB, which contradicts §3.
- **N-B21 = P1-53.** GLM "~306 GB" is 305.78 GiB / 328 GB, and its location is cited to the wrong section.
- **N-B22 = P1-4.** The summed per-region mincore figure counts shared boundary pages twice.
- **N-B23 = P1-14 / P1-17 / P1-63.** Eviction wording: "cold arms use virgin layers" overgeneralises §14. The committed `evict-bank.sh` is a reconstruction. "Advice-based eviction fails on an idle machine" conflates `MADV_DONTNEED`, which always failed, with msync, which failed when idle.
- **N-B24 = P1-51.** "Requested" carries two definitions (GLM logical bytes vs `PrefetchReport.requested_bytes`).

### INFORMATIONAL

- P1-23 (peak RSS);
- P1-29 (`Fused` type-doc wording);
- P1-34 (GLM warm spread 5.3 %);
- P1-39 (the "paged" keys);
- P1-56 (a `moe-trace` exporter in a related fork);
- P1-55 (later #28248 flip decomposition);
- P1-13 (probe refactored after recording; logic unchanged);
- O-1 (v3-disk-resident numbers);
- O-2 (`readrate.rs`, `glm_expert_reuse.py`, §13 disk table);
- O-3 (V3-N4 three residency states; V2 budget arms);
- O-4 (Metal wired cap ≈75 %, `glm5-flash-funnel.md` §6);
- O-5 (`dec8_2_fio_sweep.py`);
- O-6 (koren1712 Windows data in #23324);
- O-7 (the displacer `.s6` is recorded as a clone of the subject, a device-cache confound).

## D. Overall verdict

1. **Are all seven original BLOCKING findings closed with source support?** Not all.

   | A1 # | Status |
   |---|---|
   | #9, #11, #18, #24, #30 | Closed, each with a primary source quoted in §A |
   | #31 | PARTIALLY: the PREFETCH-internal 59.1 % reconciliation is source-supported, but the replacement claim of a reconciled byte account is contradicted by main's V3 record of the same token |
   | #34 | PARTIALLY: the misidentification is corrected with source support, but the replacement identity statement omits main's lift2 ≡ s6 clone record |

2. **Unresolved BLOCKING items:**

   | Item | Description |
   |---|---|
   | N-B1 (P1-47) | V3 cold record omitted; byte account not reconciled across LARQL records. This carries the residue of A1 #31 |
   | N-B2 (P1-37) | Subject-identity evidence omitted and mis-stated. This carries the residue of A1 #34 |
   | N-B3 (P1-67) | Recorded locality result and `moe-locality` instrument omitted |

   No original BLOCKING finding is NOT RESOLVED outright.

3. **On accuracy grounds alone, revision 2 is not yet an accurate reconnaissance record.** It is much improved: 25 of audit 1's 56 findings are resolved and 22 remain verified. But it still omits main records that bear directly on its stated question, and several of its new sentences, a number of them copied from audit 1's corrections, are wrong.

   **Required changes:**

   - **(a) Add V3 rung 7 as Kimi cold evidence.** Use `k3-residency-vertical-notes.json`: 1296 ms, 59,670 faults, 0.976 GB, 21.7 µs per fault, 67 % prompt reuse, no fingerprint.
     - Qualify "three rounds" as "three PREFETCH rounds plus one V3 run".
     - Replace "The byte accounting is reconciled" and "already explained by its sources" with "explained within the PREFETCH rounds. V3, on the same token and prompt, paged 0.976 GB (67 % reuse); the difference is unexplained, and V3 has no routing fingerprint".
     - Add V3 to the per-fault table.
   - **(b) Cite `represent-real-evidence-1-notes.json:30`.** State it as: lift2 ≡ s6 by segment size and spot-checked content (APFS clone; `system_graph.json` differs by 33 bytes), with no digest; s6 ≡ s7 by `payload_sha256` 3f12d102…. Restate §7 Q1 as whether that chain suffices or a digest is required.
   - **(c) Add `moe-routing-locality.md` to §2 and `larql moe-locality` to §1.2.** Also make §6.4's novelty claim against them.
   - **(d) Fix the witness-cost presentation (§B2).**
     - Caveat the headline and §2.2 for round 2.
     - Compare within rounds only, or use witness-adjusted values, in §2.3 ("590–634 ms in every round", the per-fault table).
     - State the missing unwitnessed record in §2.2 and §2.3.
     - Reconcile §5.4's rule with counting round 2 as a latency round.
   - **(e) Add a provenance statement for each prior result**, using §B1's BACKGROUND ONLY classifications: no GLM weight digest, volume unmounted, inputs and traces uncommitted, the §12 prompt unrecorded.
   - **(f) Non-blocking corrections:**
     - N-B6 decoupling;
     - N-B7 "exactly";
     - N-B8 111×;
     - N-B9 and N-B10 #23324;
     - N-B11 observer mincore;
     - N-B12 one basis and 16.4;
     - N-B13 and N-B14 round-3 sample and prompt figures;
     - N-B15 contention;
     - N-B16 answer the mapping-advice question;
     - N-B17 Metal path;
     - N-B18 `--expert-bytes`;
     - N-B19 five call sites;
     - N-B20 and N-B21 units;
     - N-B22 boundary pages;
     - N-B23 eviction wording;
     - N-B24 "requested".

The document stays within reconnaissance: no thresholds, no winner, no authorised experiment. That is not at issue.
