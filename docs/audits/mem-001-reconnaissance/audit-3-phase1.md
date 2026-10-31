# MEM-001 reconnaissance, audit 3, phase 1 (independent verification)

Subject: `95fd5764:docs/mem-001-reconnaissance.md` (revision 3), dated at main `51579782`.
Bound: `95fd5764:docs/mem-001-reconnaissance-source-inventory.md` (frozen at `f0848d7c`).

Written before any `docs/mem-001-reconnaissance-audit-*.md` file or revision 1/2 was opened.
All repository reads went through git objects (`git show "${c}:path"`, `git grep`, `git log`,
`git diff`, `git ls-tree`). Disk reads: `ls` of `~/chris-models` and `/Volumes`; logical
sizes from `ls -l`; the two small `index.json` and `system_graph.json` files of `.s6` and
`.s7` (read and SHA-256 hashed); Kimi `config.json`. No weight file was opened. External
sources: the three GraphQL JSON copies in `audit667-r2/ext/` (`d28248.json`, `d23324.json`,
`d27149.json`) only. The `[A1 #n]` / `[A2 P1-n]` tags in the document were treated as labels.

## 1. Coverage statement

Every section §0 to §8 was read. Every number, path, line number, commit id, quotation and
derived figure below was checked at its primary source. Arithmetic was recomputed. Rows that
were verified without remark are grouped per section. §8's counts of audit findings
(7 BLOCKING, 17 INCORRECT, 67 findings, 3 BLOCKING) were **not checked in phase 1**, because
that needs the audit files. Phase 2 checks them.

## 2. Findings table

### 2.1 BLOCKING

| id | claim (location, quote) | source checked | evidence / calculation | verdict | severity | minimal correction |
|---|---|---|---|---|---|---|
| P1-1 | §0 "No lift2 digest exists (§2.2)"; §2.2 identity chain: lift2 ≈ `.s6` "size and spot-check; **no digest**", "lift2's own digest: not found on main, on any searched branch, or on disk: **absent**"; §2.0 V3 row "no lift2 digest"; §7 Q1 "lift2 ≈ `.s6` by segment size and spot-checked content (no digest)" | `51579782:docs/v3-obs-1-carrier-observation.md:481-489` (a CORE inventory record; the document cites l.472-476 of the same file in §2.6); on-disk `shasum -a 256` of `.s6`/`.s7` `index.json` and `system_graph.json` | Main records "Kimi container identities (SHA-256 of `index.json` / `system_graph.json`), recorded before `.lift2` was deleted on 2026-09-20": `.lift2` index.json `ddd9cad3e4603d12…`, system_graph `1b4c0fc41da8e83d…`, kda_gate_form absent, kv_a_norm_eps 1e-06, "refused at first step (gate form absent); DELETED"; `.s6` index.json `ddd9cad3e4603d12…`, system_graph `a33b81a92da58237…`, "KEPT as fixture"; `.s7` index.json `ddd9cad3e4603d12…`, system_graph `1fe02abad2f3bcdc…`, kda_gate_form softplus, "the container of record". On disk today both `.s6` and `.s7` `index.json` hash to `ddd9cad3e4603d12dfbc…`, and their system graphs to `a33b81a92da58237…` / `1fe02abad2f3bcdc…`, matching the record. The index.json carries the declared segment digests (expert bank `payload_sha256` `3f12d102…`, `segment_sha256` `e43600cd…`). So lift2 **has** a recorded digest: its index.json was byte-identical to `.s6`/`.s7`'s, declaring the same segment digests. What differs is the declaration (`system_graph.json`): all three hashes differ, and `.s7` adds `kda_gate_form`, `kv_a_norm_eps`, `shared_expert_intermediate_size` (on-disk `diff`). | INCORRECT; omission of in-inventory contrary evidence | **BLOCKING**: it misstates what the record establishes about subject identity, which is the load-bearing premise of §2.0, §2.2 and §7 Q1 | Replace the chain with: lift2, `.s6` and `.s7` share a recorded `index.json` SHA-256 (`ddd9cad3…`, v3-obs-1:487-489), so lift2 declared the same segment digests as `.s7`. The bytes were not re-hashed against those declarations. The three differ in `system_graph.json` (recorded hashes). lift2 lacked `kda_gate_form`, so a later binary refused it. Re-frame Q1 as whether weight identity (declared digests) plus a declaration difference is enough to pin `.s7`. |
| P1-2 | §2.6 freq-mass row, "Bearing": "contrary evidence for a small static hot set … **on a different model and capture from §2.5**"; also implied in §0 "A second record" and §6.4 "the Gemma and freq-mass findings" | `51579782:docs/cli.md:492` (CORE reference); `docs/dec-funnel.md:142-148`; `docs/bw11-weight-traffic.md:71-72`; §2.5's source `docs/diagnoses/moe-routing-locality.md` | The freq-mass pool is `bench/dec0/residuals-gemma4-26b-a4b-q4k-routed` (cli.md:492). bw11:72 places DEC-0's routed capture "on Gemma-4 26B-A4B". dec-funnel's "620 of 3840 layer-expert pairs" gives 3,840 = 30 layers × 128 experts, the Gemma 4 26B-A4B geometry §2.5 states (128 experts, top-8, 30 MoE layers). §2.5's container is `output/gemma4-26b-a4b-q4k.vindex`. **Same model** (and the same q4k encoding family), different capture (64-prompt mixture against one ~72-token passage). | INCORRECT | **BLOCKING**: it presents one model's evidence as a cross-model corroboration. That erases the force of the Gemma diagnosis's own caveat that a load-balanced router "may be more spread" than other MoEs. This is the document's main caching inference (§0 "Recorded locality results bear directly on caching") | "on the **same model** (Gemma 4 26B-A4B, DEC-0 pool `bench/dec0/residuals-gemma4-26b-a4b-q4k-routed`), a different capture". Keep the one-model caveat on both results. |

### 2.2 NON-BLOCKING

| id | claim (location, quote) | source checked | evidence / calculation | verdict | severity | minimal correction |
|---|---|---|---|---|---|---|
| P1-3 | §2.2 "V3-N5 calls the spread unexplained"; §5.7 "V3's warm token ranged 140–430 ms **unexplained**" | `51579782:…/k3-v4-warm-latency-notes.json` V4-N1; `254377ce`/`e2e0392f` same file V4-N5 | The quote of V3-N5 is accurate. But main's V4-N1 says "V3's 140 ms and 430 ms both sit in the slow regime this arm exposed … the machine, not the model". Branch V4-N5 says "CLOSED: V3's 140–430 ms warm spread was the machine, not the model" (qualified arm 75.4 ms). | §2.2 VERIFIED as a quote; §5.7 INCORRECT in the document's own voice | NON-BLOCKING (no conclusion depends on it; the document already uses the qualified 75.4 ms) | §5.7: "V3's 140–430 ms warm spread is attributed to machine state by V4-N1 (main) and closed as such by V4-N5 (branch only)." |
| P1-4 | §0 "only one layer has ever been probed" (GLM) | frontier §14 | §14: "parallel n=7 across virgin layers", "one arm per layer". Several sparse layers have been probed, each on its own. §2.1 itself quotes "Parallel n=7, across virgin layers". §3's "Only per-layer probing has run" is correct. | INCORRECT (internal inconsistency) | NON-BLOCKING | "only single-layer probes have run (layer 3 and, in §14, virgin layers)". |
| P1-5 | §0 "Main records advice-based eviction never working on Darwin"; §1.2 "Advice never evicted on Darwin" | frontier §11; `docs/diagnoses/v3-disk-resident-mmap.md`; `examples/vindex3_residency_probe.rs:81-85,163-177`; `mmap_cold_read_probe.rs:345` | frontier §11 records that `MADV_DONTNEED` did not evict in that run. v3-disk-resident-mmap records it as "lazy (only the *first* eviction on a fresh file is reliable)", with a cold-seq pass "via mmap+MADV_DONTNEED (verified by major faults)". vindex3_residency_probe records `MADV_DONTNEED` alone leaving a fresh file 100 % resident, and says the two probes "reached a different conclusion". §1.2's own cold-read row carries the "first eviction" caveat. | INCORRECT as stated ("never"); the records conflict | NON-BLOCKING (eviction is unreliable either way, and the document's working-method conclusion stands) | "Main's records conflict: advice did not evict in the GLM run, failed on a fresh file in `vindex3_residency_probe`, and was 'first eviction only' in the KU5 probe; it is not a usable eviction method." |
| P1-6 | §2.6 "Identified by the inventory, not summarised here. These are pre-registered or design records with no result read": lists `docs/k3-funnel.md` §4.13 "(unrun)" and `docs/glm5-flash-funnel.md` §8.2.2–§8.2.3 | k3-funnel.md:912-967; glm5-flash-funnel.md §8.2.2l (l.1808), §8.2.2n (l.1912), §8.2.3 (l.2191) | §8.2.2l (ROUTING-UNION) and §8.2.2n (MULTIARM-1) are **measured** results: the cross-arm expert union 1.15× mean, 1.50× worst over 48 tokens, flat with depth over layers 0–22. §8.2.3 is arithmetic from GLM §12's 0.90/8 overlap. k3-funnel §4.13 is the V3 pre-registration summary, plus a V3 result paragraph (l.946-967) that the document covers through the notes file. Neither is "unrun". | INCORRECT characterisation | NON-BLOCKING (the results are about cross-arm union, not temporal residency) | "§8.2.2l/n are measured cross-perturbation union results (not temporal locality); §4.13 is the V3 pre-registration and summary." |
| P1-7 | §2.5 "does not rule out short-window temporal locality"; §1.4 / §6.4 "how much short-window reuse is there?" | `docs/bw11-weight-traffic.md` §3 (BW11-1, CORE) | BW11-1, closed and banked 2026-08-20, on GPT-OSS 20B, 5 prompts × 24 routed layers: the same-sequence window union is 2.40× at K=8 (1.28×/1.72× at K=2/4), 6–10.5 % below a shuffled-marginal control, "real but modest". The document cites BW11-1 only as evidence for the JSONL trace instrument (§1.2). | OMISSION (qualifies, does not contradict) | NON-BLOCKING | Add one row to §2.6: BW11-1 measured modest real short-window locality on a third model (GPT-OSS 20B). |
| P1-8 | §2.0 round 2 "the exact code that ran is not recoverable (built uncommitted, ported forward later)" | `4a3823e2` ("wip(k3-prefetch): uncommitted work parked from worktree cleanup", parent 254377ce); `e2e0392f` notes "carried_in_from_prefetch_1 … stranded uncommitted in the old worktree; ported forward" | 4a3823e2 carries `Coalesced` (realization.rs:229), `--witness-residency` (mod.rs:244), routing_trace.rs and the amended PREFETCH-1 notes, which are byte-identical to e2e0392f's. It is a parked copy of the old worktree's uncommitted state, so it is a candidate for round 2's code. It was parked 17 days after the run, so equality is unestablished. §1.2 names 4a3823e2 but §2.0 does not connect it. | AMBIGUOUS (overstated as "not recoverable") | NON-BLOCKING | "a candidate is parked at `4a3823e2` (2026-09-23); whether it is the code that ran is not established." |
| P1-9 | §0 "V3 … paged 0.976 GB, with about 67 % resident from the prompt"; §2.4 table row "Resident from the prompt: ~67 % of 2.944 GB" beside round 2's witnessed 59.1 % | k3-residency-vertical-notes V3-N3 | V3 had no pre-loop residency witness. 67 % is V3-N3's derivation: 1 − 0.976/2.944 = 66.8 %, which assumes the token touches the full disjoint bound. §2.2 attributes this correctly; the headline and the §2.4 table present it beside a witnessed figure as if both were observed. | AMBIGUOUS (observation vs inference blurred) | NON-BLOCKING | Label the V3 cell "derived (V3-N3), not witnessed". |
| P1-10 | §2.4 "That 16.4 is round 2's witnessed p50 (1205 / 73,520). It is the same arithmetic basis as 8.2" | `aae20ae3:k3-prefetch-2.json` addendum; `e2e0392f:k3-prefetch-2-notes.json` PF2-N1 | 1205/73,520 = 16.39 µs and 602/73,518 = 8.19 µs both match. But PF2-N1 frames 16.4 as PREFETCH-1's figure, and round 1 gives 1253/73,516 = 17.04. The identification is an arithmetic match, not a statement of the source. | AMBIGUOUS (inference stated as fact) | NON-BLOCKING | "16.4 matches round 2's witnessed p50 (1205/73,520), not round 1's 17.0; the source does not say which it used." |
| P1-11 | §1.2 evict-bank row: "It reads ~183 GiB (~196.5 GB) just before a cold arm"; §5.2 "~183 GiB" | `aae20ae3:scripts/evict-bank.sh`; `51579782:k3-prefetch.json` cold_state_protocol; `k3-prefetch-notes.json`; disk `ls -l` | The records say "≈184 GB … 92 GB each" (the pre-registration) and "184 GB" (PF2-N1). Logical sizes today: HF dir 98,253,087,425 B (91.51 GiB) + `.s6` 98,248,054,555 B (91.50 GiB) = 183.0 GiB = 196.5 GB. The script streams only files above `+100M`, which is all the payload. The document's figure is correct for today's disk, but it is unsourced and differs in unit from the records (whose "GB" is evidently GiB). | VERIFIED (arithmetic), UNSOURCED | NON-BLOCKING | Cite "disk sizes on 2026-10-31; the records say ≈184 GB (92 GB each)". |
| P1-12 | §5.2 / §1.2 / §2.4 (ii): "`.s6`, an APFS clone of the subject, re-reads the subject's own extents" (stated for the PREFETCH runs, 2026-09-06) | `represent-real-evidence-1-notes.json:30` (committed 9f2afe97, 2026-09-18); disk mtimes | The clone record is dated 2026-09-18, 12 days after the PREFETCH rounds. That the clone relation held on 2026-09-06 is likely (`.s6` segment mtimes are 2026-08-31) but not recorded. | AMBIGUOUS (time of the record) | NON-BLOCKING | "recorded on 2026-09-18; assumed to hold on 2026-09-06". |

### 2.3 INFORMATIONAL

| id | item | evidence | verdict |
|---|---|---|---|
| P1-13 | §2.0 GLM §12: "If the default was used, the 128-token prefill may contain repeated text" | `glm_expert_trace.py:74` default `--max-tokens 192`; the default prompt is two sentences `* 3` (l.86-93, verified). §12's 128 tokens therefore used at least one non-default argument. The repeated-text inference is still possible (one copy is ~65 tokens). | VERIFIED; add the detail |
| P1-14 | §6 "demonstrated twice" (synthetic and GLM) | Both ran with readahead off: `vindex3_residency_probe.rs:24,198-206` (`MADV_RANDOM`) and GLM probe.rs:55. The document states this for GLM only. §6.1 already asks whether coverage survives different mapping advice. | VERIFIED; note readahead off for both |
| P1-15 | §2.0 provenance: round 2 is singled out as "built uncommitted, ported forward" | V3's code was also uncommitted at run time (V3 status "BUILT 2026-09-06 (uncommitted)"; k3-funnel "uncommitted on `k3-residency-vertical`"). The cold run was 2026-09-05 23:09Z; 68abfd70 was committed 2026-09-06 00:14Z. GLM's code and doc were committed together after the runs (6874285f). The same class of limitation is stated for round 2 only. | Historical uncertainty, stated unevenly |
| P1-16 | §2.2 "Second run (23:40Z, after other processes had read the bank)" | Source: "after the layer and logit dumps". These were evidently separate runs; "other processes" is a mild inference. | VERIFIED in substance |
| P1-17 | §2.0 Gemma row "Recording commit" gives a path and the doc's own date | The file was added in `2ddd0bad` (2026-06-03); the diagnosis is dated 2026-05-31. | Add the commit |
| P1-18 | §1.2 GLM probe: "Main's split version (`b3097a3a`) has the same logic, per audit 2's line comparison, which this revision did not repeat" | Independently re-checked here. The line multiset of `6874285f:glm_moe_residency.rs` (815 lines) against main's `glm_moe_residency.rs` + `glm_moe_residency/probe.rs` (815 lines) differs only in the module declaration, `#[path]`, one brace, and `ProductionBackend` → `ProductionBackend::new()` (6c144113). | VERIFIED (the claim is true; the document still relied on an audit's wording) |
| P1-19 | §1.3 "The 2026-07-23 expert-serving review records it bypassed on the default Q4K-direct path" | The review says the "f32 dequant FIFO" is bypassed (`f32.rs:54-57`, lines since moved); cache.rs calls itself a "Bounded LRU". The same object under two names. | VERIFIED; naming only |
| P1-20 | §1.3 M-CPU-3 "+2 GB RSS per shard" | Copied exactly; the source's own parenthesis gives 9.7 → 13.6 GB (+3.9). The inconsistency is in the source. | VERIFIED as copied |
| P1-21 | §0 "59.1 % of the 2.948 GB selection" | 2.948 GB is the page-rounded coalesced request (§1.5). The logical selection is 624 × 4,718,592 = 2,944,401,408 B = 2.944 GB, which equals V3's disjoint bound. §2.4 labels 2.948 correctly as "page-rounded request". | Wording only |
| P1-22 | §2.3 round 3 "Witness inside the timed token: no" | Not stated by the round-3 notes. Inferable: warm tokens 74/83/80 ms against ~150 ms when witnessed, and the PREFETCH-2 protocol sets `--witness-residency` OFF. | VERIFIED by inference |

### 2.4 Verified without remark, by section

**§0 headline.** GLM coverage 1.0003, unselected 0.0 MiB, ratio 1.000, "to within 0.8 %", 2.36×, six budgets over 128 tokens (frontier §11-14). The four Kimi per-fault figures: 1296/59,670 = 21.72; 1253/73,516 = 17.04; 1205/73,520 = 16.39, (1205−70)/73,520 = 15.44; 602/73,518 = 8.19 µs. Round 2 adjusted ratios 1135/564 = 2.01 and 1135/549 = 2.07. Round 3 prompt ratios 1.25 and 1.43. 2.948 × 0.409 = 1.206. V3 0.976 GB. `.s6`/`.s7` shared bank digest (on disk; identical index.json). Gemma 124.4/128. freq-mass 6.25 %/42 %/35 %/1.03× (dec-funnel:144-146). 283.5 GiB on `model-drive`, not mounted (`/Volumes` = Macintosh HD, Recovery). contention.sh in no ref (`git log --all -S`, ls-tree over all refs). The ~70 ms witness (PF-N4).

**§1.1.** `physical.rs:269` (resident_bytes), `:283-301` (whole pages, `.min(bytes.len())`), `:96`, `:135`. `read.rs:121`: plain `Mmap::map`. The only `madvise` under `format/vindex3` is `prefetch.rs:93`. `loaded.rs:37` (`unwrap_or(0)`); `prepared/residency.rs:417-423`. `residency_curve.rs:38-65` (`RUSAGE_SELF`, `ru_majflt`, `ru_maxrss` peak), `:195` (binding line), `:305-337` (capture/fingerprint), `:306-318` (timer stops at 311, observe at 313). Same order at `68abfd70:generate.rs` (596-603). `plan_backend.rs:375` (`routing_trace::record`). routing_trace.rs was added in 2c290c77 (V4), after V3. `realization.rs:247`, `prefetch.rs:66-127`. `runtime/residency.rs:163,175`, "a page is counted once", used by both probes. `accounting.rs:360`, `mla.rs:236`, `budgets.rs:101`, `reconcile.rs:63-81` ("never reconciled"). V2 arms: A1 7.68 GiB; A2 67 re-selected at 7 GiB; A3 refused at 3 GiB with irreducible deficit 3.63 GB, page-in 2.94 GB. 1.199e9/73,516 = 16,309 B; 12,295 × 16 KiB = 192.11 MiB. V3 470,846 minor / 0 major. GLM probe `MADV_RANDOM` at probe.rs:55; probe.rs:7-8 imports `ProductionBackend`/`MappedAccess`, 463-469 maps arms.

**§1.2.** `evict()` probe.rs:116-128. vindex3_residency_probe added in f13bf385; 200/200. Cold-read probe: 100/140 µs, 153 MB/s, 18 µs, 1,048,576/1,048,576, "macOS can't easily cap mmap residency". moe-locality cli.md:1761-1786, research builds, LRU curve and shuffle. `LARQL_MOE_ROUTE_TRACE` is not called from `format/vindex3`; `seq` is a per-layer call index (`expert_weight/trace.rs:11`). Call sites: CPU MoE (`cpu/ops/moe/mod.rs:183`), Metal decode, kquant_forward. evict-bank.sh: `DISPLACERS` = HF dir + `.s6`, branch only (aae20ae3, identical in e2e0392f). PREFETCH-2: `GATE_PROBE_PAGES = 4` at `aae20ae3:prefetch.rs:138`, `witness_residency` at `mod.rs:247`. 4a3823e2 carries Coalesced, the witness and the amended notes. Retention: `K3_ROUTED_EXPERT_BYTES = 17_547_264` (retention.rs:60), `--expert-bytes` args.rs:124, retention_report.rs:25/27, Min/Lru/Lfu/StaticOracle/Random. No DEC-9A text in `docs/`. BW10 only `moe-experts` wired (41811c1d). Metal: routed.rs:403-415, residency.rs:46-56, README "**refuted**, no effect", execution_cost.rs:22-24, teacher_forced/run.rs:51-59. fio psync direct=1, with no DEC-8.2 result on main. runtime_stats.rs:199 peak. MODALITY-RESIDENCY-1a has no artefact in any ref; the 2026-08-23 / 45.68 GB description matches the out-of-repo project note (§4).

**§1.3.** cache.rs default 256, keyed by mmap pointer. M-CPU-3 11.62 → 2.42 ms. warmup.rs:85-94. ResidencyManager (legacy layers). Five `MADV_DONTNEED` code sites (mmap_storage.rs:386; probe.rs:127; mmap_cold_read_probe.rs:61; readrate.rs:20; vindex3_residency_probe.rs:191); wrapper gate_accessors/mod.rs:312 → :327; walk-ffn handlers call it.

**§1.4.** Every "missing" or "unrecovered" status matches the sources. The PREFETCH round 2 caveat quote is exact; no unwitnessed distributions exist at e2e0392f, 4a3823e2 or aae20ae3.

**§1.5.** step_calls.rs:59-69 (Fused docs). Kimi config: hidden 2304, moe_intermediate 1024, 256 experts, 27 layers, first_k_dense 1 → 26 MoE layers. 2304 × 1024 × 2 = 4,718,592; 26 × 256 × 3 = 19,968; 19,968 × 4,718,592 = 94,220,845,056 = `.s6`/`.s7` expert-bank `payload_bytes`; × 3 = 14,155,776. (624 × 288 + 202) × 16,384 = 2,947,710,976; 624 × 289 × 16,384 = 2,954,625,024. 202 requests and 624 ranges (round 3 warm). #27149 quotes exact; 6,912/192 = 36.

**§2.0.** Commit dates: 6874285f 2026-09-07 (frontier doc, probe, readrate, both scripts in one commit; scripts unchanged since); 68abfd70 2026-09-06 on main; 63bc7c12 carries the notes; e2e0392f parked WIP; aae20ae3 = e2e0392f code (the diff is three notes files only). GLM checkpoint 328,326,771,576 B, with no weight digest in the frontier doc. The Gemma container is not in the repo; the passage is not verbatim.

**§2.1.** Every figure in the §11/§12/§13/§14 tables matches. 212.1/90.0 = 2.357.

**§2.2.** Subject and prompt (k3-v4-warm-latency.json:6, V3 subject). V3 rung 7: 23:09Z, 7702 ms / 459,281 / 7.545 GB; 1296 ms / 59,670 / 0.976 GB; the "16.4 KB per fault" quote; 2.944; ≈22 µs; warm 140–430; second run 10.485 GB, 470,846 minor, ≈0.8 µs. REAL-EVIDENCE quote exact (l.30). continuation-mem-1.json:31 quote exact. `.s6`/`.s7` index.json byte-identical (`cmp`), segment sizes equal. lift2 is absent from `~/chris-models`.

**§2.3.** Every cell of the three-round table matches k3-prefetch-notes (main and e2e0392f), k3-prefetch-2-notes (e2e0392f) and k3-prefetch-2.json (aae20ae3), including machine states, the discarded 3044/2677 ms sample, prefetch stages 504.5/557/544/514/528, faults, paged bytes, resident-before-loop figures and warm figures. A_quiet_QUALIFIED: 75.4 ms, p95/p50 1.03, at 254377ce and e2e0392f. PF-N2 (six threads), PF-N3 (8 ms of 80), PF-N5 (2.2–2.4 GB/s, the quote), G6_false "single-pass artifact".

**§2.4.** Ratio tables recomputed: 2.10, 1.98, 1.90, 1.95, 1.63, 1.67, 2.01, 2.07, 1.02, 0.99, 1.25, 1.43. Range 549–607 ms. 1253/602 = 2.08. (1253−76)/73,516 = 16.0; (602−74)/73,518 = 7.2; the ordering survives subtracting 75.4. Byte table: 7.545 + 0.976 = 8.52; 7.72 + 1.199 = 8.92. The PREFETCH-1 P1 quote is exact. The "84 of the 202 … 1.219 GB" quote is exact. The four-quantity table is consistent with the code.

**§2.5.** Every metric, quote and caveat matches moe-routing-locality.md and v3moe_locality.json (124.43, 0.2131, 0.7950, 0.1714, 203 ms). 21.3/6.2 = 3.4.

**§2.6.** 200/200/zero overshoot/1.63 % (ROADMAP.md:307, ROADMAP_STATUS.md:671; 200 × 16 KiB / 192 MiB = 1.63 %). freq-mass figures (dec-funnel:142-148; 620/3,840 = 16.1 %; 3 to 68 of 128; "least likely to survive to K3"), except the "different model" clause (P1-2). RSL-RP (ROADMAP_STATUS:518-531: 12 × 150 MiB = 1.76 GiB, the "closed negative and narrowly scoped" quote, 2.11×, K3 named) and CR1 1.80× (ROADMAP.md:4006). v3-obs-1 l.472-476 (87.75 GiB = 94.22 GB; 10.56 GiB; 16.9 GB). Expert-serving review quote exact.

**§3.** 42 × 6.75 = 283.5 GiB; 328,326,771,576 B = 305.78 GiB; `model-drive` (frontier header and §13); not mounted. Largest `~/chris-models` entries by logical size: 91.51 / 91.50 / 91.50 GiB (HF, `.s7`, `.s6`), then 63.73 GiB. The cgroup and "can't easily cap" quotes. 45 → 79 GB.

**§4.** #28248: title, 2026-09-02, no SSD tier, RTX 4090, UD-Q3_K_XL, 8K, 11.55 ± 0.41 → 21.21 ± 0.46 at 64, 40/40 matrix, near-tie flip (byte 354), PPL 3.3096 vs 3.3262, JigSawPT 2 %/~6 % argmax with PPL < 1 % ("my fork"), 4.45 % / KLD 0.0272, pool against full-GPU KLD 0.000000, CPU→GPU placement, `moe-trace` example mentioned. #23324: 2026-05-19, Metal shared-memory slots, `pread`, Q6_K on M3 Pro 36 GB, 38.1/30/27.5/22.1, M1 Pro 13 tok/s update, run command flags; (38.1−30)/38.1 = 21.3 %. #27149: 2026-08-15, TinyGiant, 2.97/6.06 GB/s, 4.7 tok/s with simulated compute, C benchmark 2.40–4.45 / 2.20–4.46 at controlled 50–100 %, the M1 MacBook Air comment, the Python engine with no rate. "#23324 is the only … measured end-to-end decode … on Apple Silicon" holds (the Windows/CUDA comment in #23324 is not Apple Silicon).

**§5, §6, §7.** Restatements of verified material. Classification is in §5 of this report.

## 3. Measurement semantics

| distinction | kept? | notes |
|---|---|---|
| (a) mincore snapshot vs process footprint | yes | §1.1 puts "the process's physical footprint" under "Inferred, not observed"; the boundary-page caveat; `ru_maxrss` is a peak. |
| (b) faults vs device reads vs bytes | yes | §1.1 fault row, the §2.4 four-quantity table, §5.6 (it diverges under `pread` and minor faults); no device-read instrument. |
| (c) requested vs paged/resident | mostly | §2.1 note and §2.4 table are correct; the headline calls the 2.948 GB request "the selection" (P1-21). |
| (d) declared budget vs enforcement | yes | §1.1 ResidencyBudget, §1.3, §3 ("a software budget … never an RAM limit"). |
| (e) witnessed / unwitnessed / witness-adjusted | yes | §2.4's three bases, labelled throughout; round 2 is kept within its round. |
| (f) CPU load vs I/O contention | yes | §2.4 (i), §1.4, §5.3. V4's P6 (CPU busy loops barely move the warm token) would strengthen this but is not needed. |
| (g) observation vs inference vs causal | mostly | §2.4 separates inferred explanations. Lapses: V3's 67 % beside a witnessed 59.1 % (P1-9); the 16.4 identification (P1-10); the clone relation at the run date (P1-12). |

## 4. Completeness within the declared inventory (CORE items)

| CORE record | revision 3 | assessment |
|---|---|---|
| frontier §11-14 | §0, §1, §2.1, §3 | covered. §11's synthetic-input control (108 experts, 0.65/8) is omitted; immaterial. |
| frontier §15 (disk) | §3 uses §1 | omission immaterial |
| k3-prefetch.json (pre-registration) | not cited directly | legitimate; it gives "≈184 GB" (see P1-11) |
| k3-prefetch-notes.json (main and branches) | §2.3 | covered |
| k3-v4-warm-latency.json / -notes.json | subject §2.2; qualified arm §2.3 | **V4-N1 omitted where it bears** (P1-3) |
| k3-residency-vertical.json / -notes.json | §1.1, §2.2 | covered |
| represent-real-evidence-1-notes (PERIPHERAL) | §2.2 | covered (P1-12) |
| moe-routing-locality.md, v3moe_locality.json | §2.5 | covered |
| v3-disk-resident-mmap.md, v3_granite-30b.json | §1.2, §3 | covered (P1-5) |
| bw11-weight-traffic.md §3 (BW11-1) | named only as instrument evidence | **material qualification omitted** (P1-7) |
| glm5-flash-funnel §6, §8.2 PHYSICAL, §8.2.2e | absent | legitimate: arithmetic and unbuilt rungs; the "~103 GB wired cap" (§8.2.2e) is consistent with the 45 → 79 GB contradiction |
| glm5-flash-funnel §8.2.2l/n, §8.2.3 | "identified, not summarised" | mischaracterised (P1-6) |
| k3-funnel §4.13 + V3 paragraph; §4.12 / K3c | listed; V3 via notes | mischaracterised as "unrun" (P1-6) |
| dec-funnel freq-mass | §2.6 | covered, **subject misstated** (P1-2) |
| vindex3-experiments E2/E6, lyrw-v2 | listed | legitimate (unrun pre-registrations) |
| ROADMAP KU5/V3/>RAM/C5; l.300-310, 2307; CR1 | §2.6 (200/200, CR1) | KU5 is consistent with §2.5; legitimate |
| ROADMAP_STATUS 518-530, 666-672 | §2.6 | covered |
| ROADMAP_STATUS 713-714, 726, 786 | absent | legitimate: "residency" there means the f32-vs-quantised representation |
| larql-inference ROADMAP M-CPU-3 | §1.3 | covered |
| expert-serving review 92-100 | §1.3, §2.6 | covered |
| remote-moe-bottlenecks | listed | legitimate |
| v3-obs-1 472-476 | §2.6 | covered; **l.481-489 (container digests) omitted and contrary** (P1-1) |
| cli.md (reference) | §1.2 | covered; l.492 contradicts §2.6 (P1-2) |
| vindex3-format-spec, CHANGELOG freq-mass | listed / implicit | legitimate |
| (B) instruments | §1.1-§1.3 | covered. Not named: `runtime/verdict.rs` (`SelectedExpertNotResident`), `codec/residency.rs`, `prepared/census.rs`, `experts.rs`, the `realizations.rs` CLI, `dec_bench/window_union*.rs` (a second consumer of the JSONL trace, which bears on §1.4's adapter question), DEC-8.0/8.6/9.2 ledger modules, `moe_zero_copy.rs`. None contradicts the document. Naming window-union in §1.4 would be more complete. |
| (C) scripts | §1.2, §2.0 | covered |
| (D) branch material | §1.2, §2.3 | covered |
| (E) on-disk artefacts | §2.2, §3 | covered |
| (F) external | §4 | covered |

## 5. Classification

The document remains reconnaissance. §6 states "a candidate for review, not an authorised
experiment … sets no thresholds, picks no winner and runs nothing". §7 leaves repeat counts,
quiet gates and agreement criteria to a freeze ("not from this document's numbers"). The
ordering "smallest first" is not a winner. A non-mapped fetch realisation is called "an
implementation step, and out of scope until reviewed". §8 states that acceptance is not
permission to run MEM-001A/B. No implementation is proposed. **VERIFIED.**

## 6. Out-of-inventory discoveries

1. **The out-of-repo project note for MODALITY-RESIDENCY-1a** (`~/.claude/projects/…/memory/project_modality_residency_1a.md`, named only as a class in inventory §F). It matches the document's description: 2026-08-23, Gemma 4 26B-A4B, `target.expert_bank.bin 45.68 GB`, container `gemma-4-26B-A4B-it.vindex3`, which is absent from `~/chris-models`. It supports §1.2 and is not evidence in the repository.
2. **Disk metadata** (ls only): the `.s6` segment mtimes are 2026-08-31 and its index.json is 2026-08-30. `.s7`'s index.json is dated 2026-09-20, and its segments carry the same 2026-08-31 mtimes. This is consistent with clones, and with `.s6` segments unchanged since before V3.
3. **External sources, content the document does not use:** #23324 contains a Windows/CUDA comment measuring 16 MiB overlapped reads at 2.80 GB/s against 377 MB/s through mmap paging. That is consistent with GLM §14's fetch-shape result on other hardware. #28248 contains a commenter's DeepSeek-V4-Flash routing entropy (6.3–7.9 of 8 bits; 90 % of mass needs 5,854 of 11,008 experts; a static hot list worth ~+3 %), which is further external contrary evidence on static hot sets. Neither is needed for correctness.

## 7. Legitimate unresolved historical uncertainty

Stated accurately by revision 3, or not recoverable. Not blocking.

- GLM: no weight digest; volume unmounted; hidden-state input, invocation, raw output and §11 machine state uncommitted; §12 trace and prompt unrecorded; §13 cold method unstated; §14 virgin-layer identities and serial n unrecorded.
- V3 rung 7: eviction method and machine state unrecorded; one run; no routing fingerprint. Code uncommitted at run time and committed about 1 h later (P1-15, not stated by the document).
- PREFETCH round 1: eviction script was a session file, committed later as a reconstruction (equality asserted, not shown); one sample per arm; raw logs absent.
- PREFETCH round 2: machine state unrecorded; the promised unwitnessed distributions absent everywhere; the code identity is only candidate-recoverable (P1-8).
- PREFETCH round 3: whether exactly aae20ae3 ran is unstated; one retained sample per arm.
- lift2 deleted 2026-09-20. Its bytes cannot be re-hashed against its declared digests, even though the declaration digest itself **is** recorded (P1-1).
- `contention.sh` exists in no ref.
- MODALITY-RESIDENCY-1a: no artefact.
- Gemma locality: container not in repo, passage not verbatim, one prompt.

## 8. Tally

| class | count |
|---|---|
| BLOCKING | 2 (P1-1, P1-2) |
| NON-BLOCKING | 10 (P1-3 … P1-12) |
| INFORMATIONAL | 10 (P1-13 … P1-22) |
| Verified groups without remark | §0-§7 (above) |
| NOT CHECKED | §8 audit counts (phase 2) |

Verdicts across numbered findings: INCORRECT 6 (P1-1, 2, 3, 4, 5, 6), OMISSION 1 (P1-7),
AMBIGUOUS 4 (P1-8, 9, 10, 12), VERIFIED-with-note 11 (P1-11, 13-22).
