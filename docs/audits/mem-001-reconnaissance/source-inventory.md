# MEM-001 source inventory (frozen manifest)

Built independently of `docs/mem-001-reconnaissance*.md` (none of those files were opened).

## Header

| field | value |
|---|---|
| Built | 2026-10-31 ~21:26–22:00 GMT |
| Repository | `/Users/christopherhay/chris-source/larql/.claude/worktrees/mem-001-recon` (all reads via `git show/grep/ls-tree/log` against commits) |
| Base commit | `515797821ef13224adda4ddc61e6fffe3d755db5` (committed 2026-10-31 17:57:46 +0000) |
| Base is ancestor of origin/main | YES (`git merge-base --is-ancestor 51579782 origin/main`) |
| origin/main at search time | `5819050fa95d4d2c39ba5c67a10e9e57ea5e2434` |
| Remote refs searched | 169 (`git for-each-ref refs/remotes`); full sha list in `branches.txt` next to this file |
| Seed-branch tips | `origin/k3-prefetch-2` aae20ae3b03b · `origin/wip/k3-prefetch-2-2026-09-23` e2e0392f280f · `origin/wip/k3-prefetch-2026-09-23` 4a3823e2b1d5 · `origin/docs/k3-v4-warm-latency-qualified-arm` 254377ce1fb6 · `origin/feat/bw10-ledger-and-exec-policy-seam` 41811c1d616d · `origin/mem-001-recon` 45401d4c456c at the start of the search (it moved to c7d0d0da29f5, "revision 3", 2026-10-31 21:32, while the search ran; not re-searched) |

### Exact search commands

All run with `cd <repo>`. `P` is the physical-term regex, saved verbatim in `pterms.txt`:

```
P='mincore|madvise|MADV_|WILLNEED|DONTNEED|F_NOCACHE|msync|ru_majflt|majflt|major fault|minor fault|page-in|page cache|pagecache|purge|evict|prefetch|readahead|MTLResidencySet|working set|hot subset|LRU|Belady|Bélády|hit rate|expert cache|expert bytes|route trace|routing trace|expert trace|expert reuse|page fault'
```

- **S1, strict physical terms at base:** `git grep -l -i -E "$P" 51579782 -- crates scripts docs bench ROADMAP.md` gives 432 files (`phys_strict.txt`).
- **S2, broad terms at base:** `git grep -l -i -E 'resident|residency|pread|mmap|wired|locality' 51579782 -- crates scripts docs bench ROADMAP.md` gives 1,297 files (`phys_broad.txt`).
- **S3, expert co-mention triage** (used to rank S1/S2 docs, scripts and bench files): `git grep -n -i -E 'expert.*(mincore|madvise|WILLNEED|DONTNEED|F_NOCACHE|majflt|fault|page[- ]in|page cache|evict|prefetch|readahead|resident|residency|locality|reuse|hit rate|LRU|Belady|Bélády|working set|hot subset|cache|purge|cold)|(…same…).*expert' 51579782 -- docs scripts bench ROADMAP.md`, counted per file.
- **S4, crate instrument terms:** `git grep -l -E '<t>' 51579782 -- crates`, where `<t>` takes each of: `mincore`, `madvise|MADV_`, `msync|MS_INVALIDATE`, `F_NOCACHE`, `ru_majflt|majflt|ru_minflt`, `MTLResidencySet`, `purge`, `Belady|Bélády`, `LRU`, `readahead`, `pread\b|read_at|read_exact_at`, `evict`; plus `routing trace|route trace|LARQL_MOE_TRACE|expert_trace|MOE_TRACE|routing_trace`, `residency_budget|ResidencyBudget|partial_residency|MappedAccess|LARQL_PREFETCH|prefetch\(`, and `resident expert|expert residency|hot expert|resident_union|selected bytes|expert_bytes`.
- **S5, filename search at base:** `git ls-tree -r --name-only 51579782 | grep -i -E 'residen|prefetch|locality|evict|fault|expert_trace|reuse|madvise|mincore|page_cache|pagecache|warm'`
- **S6, programme names at base:** `git grep -l -i -E '<t>' 51579782 -- crates scripts docs bench ROADMAP.md ROADMAP_STATUS.md` for each `<t>` in: `PREFETCH-1`, `PREFETCH-2`, `K3-PREFETCH`, `k3-prefetch`, `V4 warm|k3-v4|V4-N[0-9]`, `RESIDENCY-VERTICAL|residency-vertical|residency vertical`, `residency curve|residency-curve`, `REAL-EVIDENCE`, `MODALITY-RESIDENCY`, `moe-locality|moe_locality`, `moe-routing-locality|routing locality`, `DEC-9`, `BW10`, `BW11`, `frontier-execution`, `glm_moe_residency|expert-bank residency`, `Kimi-Linear`, `lift2`, `evict-bank`, `witness.residency|witness_residency`, `residency-into-k3`. Hit counts are in `prog.txt`. Notable zero-hit terms: `PREFETCH-2`, `MODALITY-RESIDENCY`, `frontier-execution` (no file links to the frontier doc), `witness_residency`.
- **S7, follow-up greps at base:** `DEC-9|retention|freq-mass|route-aware|hot cache|map-3-4-bridge`; `200 (pages|predicted)|vindex3_residency_probe|mmap_cold_read_probe|glm_moe_residency|readrate`; `DEC-8.x|dec8-[0-9]+|fio`; `unquiet|quiet`; `modality.residency`; path extraction `(~/chris-models|/Volumes/…|output)/…` and `model-drive…` over the CORE files; `https?://…` over the CORE files; `28248|23324|27149`.
- **S8, across all 169 remote tips, files absent at base:** for each tip, `git grep -l -i -E "$P|residen|locality" <tip> -- docs scripts`, keeping the paths not present in `git ls-tree -r --name-only 51579782`. That gives 23 rows (`branch_only.txt`). `docs/represent/forecasts` is inside `docs/`, so it is covered.
- **S9, seed-branch diffs:** `git diff --stat $(git merge-base origin/<b> 51579782) origin/<b>` for the five seed branches. This also catches files that exist at base but differ on the branch, which S8 cannot see.
- **S10, pickaxe:** `git log --all --oneline -S <t>` for `mincore`, `evict-bank`, `contention.sh`, `witness_residency`, `MODALITY-RESIDENCY`.
- **S11, disk:** `ls ~/chris-models` (one level deep, plus the `superseded`, `k3`, `candidates`, `streams`, `kimi-prior` and `measure` subdirectories), `ls /Volumes`, and one small `index.json` read. No weight files were opened.

Search labels in the tables below refer to S1–S11.

---

## (A) Prior-result records

| path | commit | rel | content (one line) | found by |
|---|---|---|---|---|
| docs/frontier-execution-programme.md §11 "Expert-bank residency — the first physical result" (l.776) | 51579782 | CORE | GLM-5.3-Flash layer 3, 288-expert FP8 bank (6.75 GiB) mmapped with MADV_RANDOM. Top-8 = 192.05 MiB (2.78%). demand-cold 219.82 ms / advise-cold 214.12 ms / warm 18.63 ms; 12,295 major faults; coverage 1.0003; unselected residency 0.0 MiB; MADV_DONTNEED does not evict on Darwin, msync(MS_INVALIDATE) does; synthetic input gives 108 distinct experts / 0.65-of-8 overlap | S1, seed |
| same, §12 "The reuse curve, on real hidden states" (l.850) | 51579782 | CORE | 128 real tokens at layer 3: 225/288 distinct experts, consecutive overlap 0.90/8. Window-k table (k=1..64) and LRU / LRU-ex-first-touch / Bélády table at 0.25–6.75 GiB budgets; ~54% steady-state hit at ~2.4 GiB/layer; prefix-loading bug in the trace script recorded | S1, seed |
| same, §13 "Latency, on an idle machine" (l.916) | 51579782 | CORE | Load 2.7–3.6, 5 rounds. demand/advise/touch cold 210.4/209.1/208.8 ms, ~17 µs/fault; warm 17.8 ms (11.8×). Serial readrate: USB `model-drive` 520 MiB/s (30.0 µs/fault) vs internal SSD 588 MiB/s (26.5 µs); tok/s arithmetic 0.11–1.34 | S1, seed |
| same, §14 "Deep fetch" (l.981) | 51579782 | CORE | Explicit pread of 48×4.0 MiB: serial 121.2 ms, 8-thread 90.0 ms (2,759 MiB/s) vs fault path 212.1 ms. Two instrument bugs: pread pollutes the spine check; msync eviction depends on memory pressure, so arms run on virgin layers. Throughput regime table | S1, seed |
| same, §15 "Disk" (l.1051) | 51579782 | CORE | Internal volume 393 GiB free; `model-drive` 1.0 TiB free; GLM container ≈306 GB, Inkling ≈495 GB | seed |
| docs/represent/forecasts/k3-prefetch.json | 51579782 | CORE | PREFETCH-1 pre-registration (frozen 2026-09-06 on 2c290c77, PR #432): selected-expert prefetch as an access realization on Kimi-Linear; arms, cold-state protocol, predictions | S1, S6, seed |
| docs/represent/forecasts/k3-prefetch-notes.json | 51579782 | CORE | PREFETCH-1 result. PF-N1: MADV_WILLNEED over 24 regions takes the cold token 1.25 s → 0.60 s with the same bytes and faults; P1 HELD (1,253 ms, 73,516 faults, 1.199 GB); P2 FALSIFIED (touch does not parallelise page-in). Cold protocol: `evict-bank.sh`, mincore witness `0.000 GB resident` 3/3. PF-N1..N3 | S1, S6, seed |
| docs/represent/forecasts/k3-v4-warm-latency.json | 51579782 | CORE | V4 pre-registration (frozen 2026-09-06 on 43228c97, PR #428): deterministic warm-latency characterisation; subject `~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3`, prompt [1008,10484,318] | S1, S6, seed |
| docs/represent/forecasts/k3-v4-warm-latency-notes.json | 51579782 | CORE | V4 result. V4-N1: warm floor p50 79.5 ms / min 74.9 ms, bit-identical logits; p95/p50 9.57 on BATTERY (P1 PARTIAL). V4-N2: routed experts 38 ms (48%), attention 26 ms (33%) | S1, S6, seed |
| docs/represent/forecasts/k3-residency-vertical.json | 51579782 | CORE | K3-RESIDENCY-VERTICAL-1 forecast (written 2026-09-05 on rung 3d 6669d252, PR #425): a real Kimi-Linear plan through selection, refusal, accounting and execution | S1, S6, seed |
| docs/represent/forecasts/k3-residency-vertical-notes.json | 51579782 | CORE | Waves V1 (semantic truth), V2 (budgeted residency: 98.01 GB mapped is not 98.01 GB resident) and V3 (execution + residency curve: cold token 0.976 GB, 59,670 major faults, ≈22 µs/fault, 1.30 s; warm 0.14–0.43 s, 0 faults; page cache accumulates as reclaimable, V3-N1..N4) | S1, S6, seed |
| docs/represent/forecasts/represent-real-evidence-1-notes.json | 51579782 | PERIPHERAL (verified to exist) | REPRESENT-REAL-EVIDENCE-1: wrong-intervention incident (env leakage); two flagship banks reproduce bit-exactly on the lift2 clone. Not about residency. It only names the lift2 container | seed, S6 (`lift2`, `REAL-EVIDENCE`) |
| docs/represent/forecasts/continuation-mem-1.json (+ `-notes.json`, `-records.json`) | 51579782 | PERIPHERAL (verified to exist) | CONTINUATION-MEM-1: which continuation (KV) bytes exist and are read; closed 2026-09-26, 7/8 forecasts held. KV/continuation memory, not expert weight pages | seed, S2 |
| docs/diagnoses/moe-routing-locality.md | 51579782 | CORE | 2026-05-31, Gemma 4 26B-A4B (128 experts, top-8, 30 layers), one ~72-token passage: adjacent reuse 21.3% (null 6.2%); working set 124.4/128; cumulative hit 79.5%; top-10 mass 17%; verdict: no cacheable hot subset (KU5 locality half) | S1, S3, seed |
| bench/aim-validation/moe-routing/v3moe_locality.json | 51579782 | CORE | Artefact behind moe-routing-locality.md (per-token routing stats; cold latency taken from the V3 granite probe) | S2, S5 |
| docs/diagnoses/v3-disk-resident-mmap.md | 51579782 | CORE | 2026-05-31, granite-4.1-30b-q4k gate_vectors.bin 17.2 GB: cold scattered F_NOCACHE pread p50 100 µs / p99 140 µs (153 MB/s); cold sequential mmap 18 µs (556 MB/s, 1,048,576 verified major faults); warm 0.04 µs (~2380×). Darwin MADV_DONTNEED lazy; >RAM run deferred | S1, seed |
| bench/aim-validation/v3_granite-30b.json | 51579782 | CORE | Artefact for v3-disk-resident-mmap.md (eviction-based cold-read probe; notes its non-coverage of RAM-pressure thrash) | S1 |
| bench/aim-validation/ (other files: README, ave_*, fr*, v1*, v2*, r4_*, matrix.json, moe-latent/) | 51579782 | PERIPHERAL | Other aim-validation results (FFN routing, FP4, zero-out); `matrix.json` lists V3 status. Not expert residency | seed, S2 |
| docs/bw11-weight-traffic.md §3 (BW11-1) | 51579782 | CORE | 2026-08-20, GPT-OSS 20B, 5 prompts × 24 routed layers, `dec-bench window-union`: same-sequence union fraction 0.784/0.583/0.417 at K=2/4/8 (1.28×/1.72×/2.40×), 6–10.5% below a shuffled-marginal control; banked as modest | S1, S6, seed |
| docs/glm5-flash-funnel.md §6 "Residency and traffic" (l.580–728) | 51579782 | CORE | GLM-5.3-Flash checkpoint arithmetic: no LARQL representation fits 128 GB (MXFP4 154.8 GiB); experts 2.0 bpw + rest 4-bit = 75.0 GiB; "expert compression is the residency lever"; wired cap / `iogpu.wired_limit_pct` note (l.725) | S1, S3, seed |
| docs/glm5-flash-funnel.md §8.2 PHYSICAL-4/6 (l.808–870), §8.2.2e (l.1326, "86.6 GB resident … ~103 GB wired cap") | 51579782 | CORE | PHYSICAL rung table: resident expert realization at ≈2.0–2.5 bpw (unbuilt); frozen budget: residency 75.0/92.7 GiB | S1, S3 |
| docs/glm5-flash-funnel.md §8.2.2l/§8.2.2n (l.1808, 1912) and §8.2.3 (l.2191) | 51579782 | CORE | Routing-union and multi-arm union amplification flat with depth; MTP priced from the reuse curve's 0.90/8 overlap (K=2 verify ≈15.1/16 experts, 1.89× routed traffic; MTP ≈1.2–1.3×); provenance caveat (layer 3, one trace) | S3 |
| docs/k3-funnel.md §4.13 + V3 paragraph (l.912–967) | 51579782 | CORE | K3-RESIDENCY-VERTICAL-1 summary: 94.22 GB mapped, 0 resident at binding; cold token 0.976 GB / 59,670 faults / 1.30 s; warm 0.14–0.43 s; points to k3-residency-vertical-notes.json | S1, S3, S6 |
| docs/k3-funnel.md §4.12 (l.891), gate GC / K3c / item 21 (l.66, 937, 1027, 1074, 1086–1093) | 51579782 | CORE | GPU-resident routing + MXFP4 experts (59.8 → 77.2 tok/s); planned coverage curve (resident-expert fraction vs C6 drift) and metric `k3/expert_resident_bytes`, not run | S3 |
| docs/dec-funnel.md "Four conventions" (l.138–150) | 51579782 | CORE | `k3-ledger freq-mass` (2026-08-01) on the DEC-0 Gemma-26B routed pool (8/128): a 6.25% static resident set covers 42% of events, 90% needs 35% of the bank; session-adaptive cache 1.03× (oracle 1.41×); cold tail 620/3840 unobserved; mass_per_residency 8.27→6.76→3.28× | S3, S7 |
| docs/dec-funnel.md R2/R3/R9/R10 (l.43–73), C8 (l.180) | 51579782 | PERIPHERAL | Standing rules about not transferring union and amortisation figures across activation fractions (cites dec8-5, dec8-6); method rules, not a residency result | S3 |
| docs/dec-funnel-v0.2.md, dec-funnel-v0.4.md | 51579782 | PERIPHERAL | Earlier DEC versions: DRAM-resident expert capability claims and cloud-instance eviction; superseded plan text | S1 |
| docs/vindex3-experiments.md E2/E3/E6 (l.313–411, 482–502) | 51579782 | CORE | Pre-registered, unrun experiments: E2 physical layout sweep (SSD bytes/token, page faults, read amplification, prefetch effect, ≥10% bar for a permutation); E6 K3 routing/locality traces (LRU/LFU hit rates, reuse distance) | S1, S3 |
| docs/lyrw-v2.md (l.147–355) | 51579782 | CORE | K3 extraction plan: "cold means cold" (budgets enforced against the unified buffer cache), E2 measurement list, within-run control note from dec8-12; forbids freezing a hot set or permutation; locality stays runtime metadata | S1, S3 |
| ROADMAP.md KU5 (l.166), V3 rows (l.244, 3597), ">RAM models" (l.119–123), C5 (l.3733) | 51579782 | CORE | KU5 "mmap thrash" resolved 2026-05-31 as locality POOR; V3 PARTIAL; C5 lazy disk-resident edges "not started" | S1, S3 |
| ROADMAP.md l.300–310, l.2307 | 51579782 | CORE | VINDEX3 bound plan predicts its page working set: 200 predicted / 200 resident / 0 overshoot, 1.63% of a 192 MiB layer (synthetic/Gemma fixture, `vindex3_residency_probe`) | S1, S7 |
| ROADMAP.md §P1 "Context retirement & route-aware hot tier" (l.3983–4050), l.2500–2507 | 51579782 | CORE | CR1: route touching 1 vs 3 FFN layers 2.11× replay latency; grouping by physical owner 1.80×; registry `rsl-rp` (dense Gemma FFN layers, not routed experts; included because it is a hot-cache / prefetch result) | S7 |
| ROADMAP_STATUS.md l.518–530 (RSL-RP) | 51579782 | CORE | "Route-aware HOT cache justified; speculative storage prefetch is not": 12 FFN layers × 150 MiB pool, V_oracle 0.0 MiB/step once resident, capacity beats prediction 2.7× (dense-layer estate) | S7 |
| ROADMAP_STATUS.md l.666–672 | 51579782 | CORE | 2026-08-02 `f13bf385`: residency probe shows the bound plan predicts its resident page set exactly (200/200/0) | S7 |
| ROADMAP_STATUS.md l.713–714, l.726, l.786 | 51579782 | CORE | 26B CPU MoE "quantized residency" levers (2026-06-11) and C10 "the MoE gap is f32 residency" (2026-06-10); MoE routing locality and V3 status lines | S7 |
| crates/larql-inference/ROADMAP.md l.1046–1054 (M-CPU-3) | 51579782 | CORE | 2026-05-01: `LARQL_MOE_CACHE_ENTRIES` 64 → 256 (dequantised-expert LRU in `cpu/ops/moe/cache.rs`); eviction-driven p99 11.62 → 2.42 ms; +2 GB RSS per shard (26B-A4B) | S4 follow-up |
| docs/audits/expert-serving-review-2026-07-23.md l.92–100 | 51579782 | CORE | Review finding: no cold-cache mode for expert weights (`packed_mmaps` never madvised), so replay measures the warm page cache; the f32 dequant FIFO is bypassed on the Q4K-direct path | S1, S3 |
| docs/diagnoses/remote-moe-bottlenecks.md | 51579782 | CORE | 2026-05-29 remote-MoE diagnosis: recommends multi-layer expert prefetch to hide LAN/WAN round trips (analysis, no residency measurement) | S1, S3 |
| docs/v3-obs-1-carrier-observation.md l.472–476 | 51579782 | CORE | Incidental Kimi residency evidence: 87.75 GiB mapped, 0 resident after preparation; 10.56 GiB resident after the witnesses (experts touched); peak RSS 16.9 GB | S2, S3 |
| docs/diagnoses/walk-ffn-performance.md l.535–559 | 51579782 | PERIPHERAL | Dense WalkFfn: per-layer `MADV_WILLNEED` on layer+1 as a cost candidate; dense FFN, not experts | S1 |
| docs/diagnoses/memory-bandwidth-roofline.md | 51579782 | PERIPHERAL | DRAM vs cache-resident bandwidth roofline; prefetch striding; not page residency | S1 |
| docs/kv-attention-scaling.md l.211, 433–438, 499 | 51579782 | PERIPHERAL | KV ladder measurement hazard: a peer session's warm evicts the other's working set (page-cache drift); KV/attention, not experts | S1 |
| docs/kv-residency-contract.md | 51579782 | PERIPHERAL | KV cache residency contract | S1, S5 |
| docs/vindex3-remote-source.md l.202–266 | 51579782 | PERIPHERAL | Remote/cached/mapped/resident tiers with `cache_hits`/`evictions` counters for remote container compilation; design text, no measurement | S1 |
| docs/represent-optimizer-mcp.md l.294, 1675, 1842 | 51579782 | PERIPHERAL | Optimizer vocabulary `residency.*` (placement/cache/prefetch/external storage) and a worked example "frees 700 MB → 6 more experts resident"; design text, no measurement | S1, S3 |
| docs/represent-codec-contract.md | 51579782 | PERIPHERAL | Per-realization residency declaration (contract text); the instrument is listed under B | S1 |
| docs/positioning.md l.57–71 | 51579782 | PERIPHERAL | Restates KU5 / routing-locality conclusion for positioning | S1 |
| docs/ffn-cache.md | 51579782 | PERIPHERAL | FFN activation cache hit rates (L1/L2); activations, not weights | S1 |
| docs/adr/0018-moe-expert-routing.md, 0014, 0003, 0005 and other ADRs in S1 | 51579782 | PERIPHERAL | Router/grid design (co-resident attention weights, heartbeat eviction, shard memory bounds); no expert-page residency measurement | S1 |
| docs/ffn/v3-routed-experts.md, bench/v3-routed-experts/* | 51579782 | PERIPHERAL | V3 routed-expert workers; "warm-N" files are timing repeats, not residency | S2, S5 |
| bench/baselines/c10_gemma4-26b-a4b_*.json, _c10_26b_*.json, _qres_*.json | 51579782 | PERIPHERAL | 26B CPU throughput baselines (in-process experts, warm); residency here means the f32-vs-quantised representation in RAM, not page state | S3 |
| docs/audits/release-review-2026-09-26/{vindex3,vindex_rest,cli_server,models,inference,compute}.md | 51579782 | PERIPHERAL | Code-review notes; vindex3.md and vindex_rest.md name `glm_moe_residency.rs` (refactor/size), cli_server.md names moe-locality | S1, S6 |
| docs/represent/forecasts/rung3-*.json, rung2-*.json, represent-progressive-1*.json, continuation-window-1*.json, continuation-codec-*.json | 51579782 | PERIPHERAL | Declared decode residency of codecs; KV/continuation residency; no expert-page measurement | S1, S2 |
| docs/arch-conformance/forecasts/k3-*.json, wave*.json | 51579782 | PERIPHERAL | Hits are trait-default / `expert_gate_policy` wording; architecture conformance | S2, S3 |
| docs/audits/codebase-review-*, dec-readiness-review-2026-07-22.md, vindex-walkffn-review-2026-07-30.md | 51579782 | PERIPHERAL | Review notes (lock contention, eviction dead code, walk FFN) | S1 |
| docs/cli.md l.45, 82, 452–504, 760–814, 1761–1780 | 51579782 | CORE (reference) | CLI reference for `moe-locality` (LRU hit-rate curve), `k3-ledger freq-mass`, server flags `--release-mmap-after-request` (madvise DONTNEED; "Darwin: advisory"), `--warmup-walk-ffn`, `/v1/warmup` | S1, S3 |
| crates/larql-vindex/docs/vindex3-format-spec.md l.169, 229, 240, 569, 591 | 51579782 | CORE (spec) | Locality store is runtime metadata, not format; group extent = SSD read / prefetch unit (8/16/32 experts) | S3 |
| crates/larql-cli/CHANGELOG.md l.12–31 | 51579782 | CORE (reference) | 2026-08-01 `k3-ledger freq-mass`: grade a resident expert set on events, not support; `selection_trace.rs` | S7 |

## (B) Instrument implementations (code at 51579782)

| path | type / function | rel | content | found by |
|---|---|---|---|---|
| crates/larql-vindex/examples/glm_moe_residency.rs + glm_moe_residency/probe.rs | `run()`, `evict()`, `advise()`, `pread()`, `resident()` (mincore), `faults()` (getrusage), `hold()` | CORE | GLM expert-bank residency probe behind frontier §11–14: arms demand/advise/touch/warm/pread-serial/pread-parallel; byte-identical output invariant; spine check | S4, S5 |
| crates/larql-vindex/examples/readrate.rs | `run()` | CORE | Cold sparse read rate of a volume by the same method (mmap, readahead off, msync evict, major-fault count); frontier §13 disk table | S4 |
| crates/larql-vindex/examples/mmap_cold_read_probe.rs | `rusage_faults`, `evict`, `pread_cold_scattered`, `touch_pages` | CORE | KU5 V3 cold-read probe (F_NOCACHE pread, MADV_DONTNEED, MADV_RANDOM; cold sequential / cold sparse / warm sparse) | S4, S7 |
| crates/larql-vindex/examples/vindex3_residency_probe.rs | `Environment`, `run()` | CORE | "Does the bound plan predict the pages execution touches?" (mincore, msync, F_NOCACHE); 200/200 result | S4, S5 |
| crates/larql-vindex/src/runtime/residency.rs | `PageSpan`, `ResidencyAccount::{resident_fraction, prediction_coverage}` | CORE | Attributes resident pages to the operands that caused them | S5 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/prefetch.rs | `Range`, `PrefetchReport`, `prefetch(MappedAccess, &[Range], parallelism)`, `advise`, `touch` | CORE | PREFETCH-1 access realization (MADV_WILLNEED or concurrent touch over selected-expert ranges) | S4, S5 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/realization.rs (+ realization/selection.rs) | `MappedAccess` (Demand/Advise/Touch), `decode_residency: ResidencyProfile` | CORE | Realization vocabulary, including the mapped-access policy | S4 |
| crates/larql-vindex/src/format/vindex3/represent/physical.rs | `WeightRegion::resident_bytes()` (mincore) | CORE | Region-level resident-bytes query | S4 |
| crates/larql-vindex/src/format/vindex3/represent/codec/residency.rs | `ResidencyClass`, `ResidencyProfile`, `Acceleration` | CORE | Declared residency per realization | S5 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/prepared/residency.rs, prepared/census.rs | `bound`, `expectations`, `reconcile`; `AllocationCensus`, `ResidencyCensus` | CORE | Prepared-plan residency/allocation census (mapped vs resident pages) | S4, S5 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/accounting.rs, accounting/budgets.rs | `Expectation`/`Observed`; `ResidencyBudget`, `ThroughputBudget` | CORE | Budgeted residency accounting (expert access default Demand) | S4 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/experts.rs | routed packed-bank operand loading | CORE | Binds the packed expert bank as mapped streams | S4 |
| crates/larql-vindex/src/runtime/verdict.rs | `RefusalKind::Residency`, `SelectedExpertNotResident` | CORE | Classifies non-resident selected experts as a residency refusal | S4 |
| crates/larql-vindex/src/format/vindex3/opplan/exec/tests/{prefetch,residency,residency_budget,residency_census,partial_residency,bf16_residency}.rs, coverage_device/diagnostics_and_residency.rs | tests | CORE | Witness tests for the above (tests/prefetch.rs uses mincore) | S4, S5 |
| crates/larql-cli/src/commands/primary/vindex3_cmd/generate/residency_curve.rs (+ generate.rs, args.rs, exec.rs) | `process_resources()` (ru_majflt), `run_residency_curve()`, quiet-machine refusal | CORE | `--residency-curve` per-pass mapped/resident/fault sampling, used for K3 V3/V4 | S4, S5, S6 |
| crates/larql-cli/src/commands/primary/vindex3_cmd/realizations.rs | `ops --realizations` | CORE | Prepared-plan admission priced against a residency budget | S4 |
| crates/larql-cli/src/commands/diagnostics/moe_locality/{mod,trace,stats,shuffle}.rs | `MoeLocalityArgs`, `run`, `Trace::load`, `mean_union`, `uniform_union`, `mean_adjacent_overlap`, `lru_hit_rate`, `shuffle_runs` | CORE | `larql moe-locality`: union growth and LRU hit-rate curve over a routing trace with uniform and shuffled baselines | S4, S5, seed |
| crates/larql-compute/src/ffn/expert_weight/trace.rs (+ options.rs `ENV_MOE_ROUTE_TRACE`; tests test_moe_route_trace_{on,off}.rs) | per-token routing trace writer | CORE | `LARQL_MOE_ROUTE_TRACE` JSONL emitter consumed by moe-locality and window-union | S4 |
| crates/larql-cli/src/commands/primary/dec_bench/window_union.rs, window_union_runtime.rs, replay/denominators.rs | BW11-1 window-union | CORE | Same-sequence consecutive-position expert-union ceiling with shuffled control | S4, S6 |
| crates/larql-cli/src/commands/primary/k3_ledger/retention.rs, retention_report.rs, retention_tests.rs | `Policy`, `ReferenceStream`; DEC-9A | CORE | Retention oracle gate: Belady/MIN vs LRU vs StaticOracle vs Random; compulsory misses; `K3_ROUTED_EXPERT_BYTES` | S4, S6 |
| crates/larql-cli/src/commands/primary/k3_ledger/freqmass.rs, selection_trace.rs, budget.rs (DEC-8.0), touch.rs (DEC-8.6), block.rs (DEC-9.2) | ledger modules | CORE | Static resident-set coverage; per-token fetch (miss) budget for external storage; resident-touch split; speculative block economics | S4, S7 |
| crates/larql-compute/src/cpu/ops/moe/cache.rs | bounded LRU of dequantised experts keyed by mmap pointer (`LARQL_MOE_CACHE_ENTRIES`) | CORE | In-RAM expert cache (decoded representation, not page state) | S4 |
| crates/larql-compute-metal/src/buffers/residency.rs (+ regions.rs, mod.rs) | `ResidencyArm::from_env`, `ResidencySet::{new, add_buffer, commit, request_residency, add_to_queue}` | CORE | Explicit MTLResidencySet for registered mmap weight regions (GPT-OSS layer weights, ~697 MB vs ~87 MB selected) | S4, S5 |
| crates/larql-compute-metal/src/moe_zero_copy.rs, moe_dispatch/entry.rs (`cached_buffer_for_bytes`) | zero-copy expert dispatch | CORE | Experts bound as offsets into registered mmap regions; buffer cache | S4 |
| crates/larql-server/src/routes/expert/warmup.rs, routes/warmup.rs, state/loaded_model.rs, bootstrap/{cli,load}.rs | `warmup_metal_expert_cache`, `warmup_hnsw_unit_cache`; `--release-mmap-after-request` madvise | CORE | Server expert-cache warmup and post-request DONTNEED | S4, S5 |
| crates/larql-vindex/src/mmap_util.rs, index/storage/ffn_store/interleaved*.rs, gate_accessors | `mmap_demand_paged` (MADV_RANDOM), WILLNEED prefetch | PERIPHERAL | Legacy VINDEX1/2 dense-store mmap advice. Named as the foundation for the >RAM follow-up in v3-disk-resident-mmap.md | S4 |
| crates/larql-vindex/src/index/storage/residency.rs | `LayerState`, `ResidencyManager` | PERIPHERAL | Legacy adaptive *layer* residency (pin hot layers, stream cold) for dense Q4; not experts | S5 |
| crates/larql-vindex/examples/glm_moe_probe.rs | GLM sparse-MoE parity probe | PERIPHERAL | Routed-FFN parity over the whole bank (MappedAccess::Demand); correctness, not residency | S4 |
| crates/larql-compute-metal/src/kv_residency_contract.rs, larql-kv/*, larql-compute/src/kv_dispatch/* | KV residency | PERIPHERAL | KV cache residency/eviction | S1, S5 |
| crates/larql-router/src/tasks/rebalancer/eviction.rs, larql-server session/lease purge | router/session eviction | PERIPHERAL | Network/session eviction | S4, S5 |
| crates/larql-inference/src/trace/*, larql-kv markov_residual helpers | madvise in trace store | PERIPHERAL | Residual-trace file advice | S4 |

## (C) Scripts (at 51579782 unless noted)

| path | rel | content | found by |
|---|---|---|---|
| scripts/glm_expert_trace.py | CORE | Real GLM-5.3-Flash layer-3 expert selection over 128 real tokens through the pinned reference (design frozen first); feeds §12 | S1, seed |
| scripts/glm_expert_reuse.py | CORE | Natural-reuse windows and budgeted residency (LRU and Bélády) from the trace | S1, seed |
| scripts/dec8_2_fio_sweep.py | CORE | DEC-8.2 row-granular random read on an external NVMe through fio (explains why a Python pread harness was rejected); target `/Volumes/model-drive/dec8_2.bin` | S1, seed |
| scripts/k3_batch_union.py | CORE | Expert-union growth with batch for K3-from-SSD amortisation arithmetic | S3, S6 |
| bench/aim-validation/moe-routing/analyze.py | CORE | Gemma 26B routing-locality analysis (adjacent reuse, cumulative hit, working set, saturation) | S1, S3 |
| scripts/evict-bank.sh | CORE, BRANCH-ONLY (see D) | Displaces a mapped expert bank from the macOS page cache by streaming ~92 GB files; mincore witness ≤0.1 GB resident. Cited by base k3-prefetch-notes.json but absent at 51579782 | S8, S10 |
| scripts/moe_expert_spectrum.py | PERIPHERAL | Singular spectrum over the expert axis (compression), not residency | S3 |
| scripts/dec0-arm-l.sh, dec0-loopback.sh | PERIPHERAL | DEC-0 replay drivers (expert server loopback) | S3 |
| scripts/kv-ladder-bracket.sh | PERIPHERAL | KV ladder A/B bracket | S1 |
| scripts/kimi_generate_export.py | PERIPHERAL | HF oracle for the K3 vertical (correctness taps) | named in k3-funnel.md |

## (D) Branch-only material (absent at 51579782)

| branch @ tip | path | rel | content | found by |
|---|---|---|---|---|
| origin/k3-prefetch-2 @ aae20ae3b03b (2026-09-06; merge-base 1e57b6d1) | docs/represent/forecasts/k3-prefetch-2.json | CORE | PREFETCH-2 pre-registration: access realization conditioned on residency (frozen 2026-09-06 on main 1e57b6d1) | S8, S9, S10 |
| same | scripts/evict-bank.sh | CORE | See C | S8, S10 |
| same | crates/larql-vindex/src/format/vindex3/opplan/exec/prefetch.rs (modified: `runs_of`, `residency()` mincore, gated/coalesced), exec/routing_trace.rs (new), realization.rs, production.rs, tests/prefetch.rs, tests/stages_and_routing.rs; crates/larql-cli vindex3_cmd/{exec,generate,mod}.rs, tests/generate.rs | CORE | PREFETCH-2 code: ask only for the pages the selected experts are missing; routing-trace capture | S9, S10 |
| origin/wip/k3-prefetch-2-2026-09-23 @ e2e0392f280f (2026-09-23) | all of the above, plus docs/represent/forecasts/k3-prefetch-2-notes.json (new), and modified k3-prefetch-notes.json (+46 lines) and k3-v4-warm-latency-notes.json (+73 lines) | CORE | PREFETCH-2 notes: "CODE COMPLETE AND GATED; MEASUREMENT PAUSED 2026-09-06"; one UNQUALIFIED round (cold token demand 602 / coalesced 591 / gated 607 ms; 73,518 faults, 1.206 GB); names an unrun `contention.sh` contention control (the script itself exists on no branch) | S8, S9, S10 |
| origin/wip/k3-prefetch-2026-09-23 @ 4a3823e2b1d5 (2026-09-23; merge-base 6fd1cd04) | prefetch.rs, routing_trace.rs, generate.rs and related; modified k3-prefetch-notes.json and k3-v4-warm-latency-notes.json | CORE | Parked PREFETCH WIP on top of 254377ce | S9, S10 |
| origin/docs/k3-v4-warm-latency-qualified-arm @ 254377ce1fb6 (2026-09-22) | docs/represent/forecasts/k3-v4-warm-latency-notes.json (+70/−3) | CORE | V4 arm A_quiet_QUALIFIED (2026-09-06 ~13:10, AC): 30 counted passes, p50 75.4 / p95 77.6 ms, p95/p50 1.03, 0 major faults, identical routing | S9 |
| origin/feat/bw10-ledger-and-exec-policy-seam @ 41811c1d616d (2026-08-15) | docs/diagnoses/bw10-live-gate.md | CORE | BW10 movement ledger (`larql_compute::movement_ledger`): DRAM/NVMe/network movement with reuse and prefetch counters, `Regime = resident / capacity-constrained / cold-estate`; BW-E residency horizon named as follow-up | S8, S9 |
| same | docs/adr/0027-execution-policy-seam.md, docs/diagnoses/bwb-compact-dense-oracle.md, docs/diagnoses/bwc-expert-skip-oracle.md, crates/larql-compute/src/movement_ledger/* | PERIPHERAL | Execution-policy seam; compact-dense and expert-skip oracles (skipping experts, not residency) | S8, S9 |
| origin/mem-001-recon @ 45401d4c456c | docs/mem-001-reconnaissance*.md (4 files) | EXCLUDED | The document under review and its audits (not read, by instruction) | S8 |
| origin/main, origin/continuation-codec-3/w2-record | docs/represent/forecasts/continuation-codec-3-records.json | PERIPHERAL | KV codec residency records (after base) | S8 |
| origin/exp/mad-v3-knn-addressability @ ee1a770dd225 | docs/mad-v3-knn.md, scripts/audit_mad_addresses.py | PERIPHERAL | Residual-key addressability | S8 |
| origin/worktree-state1-continuation @ a6a482ded933 | docs/state-continuation.md | PERIPHERAL | Continuation-state residency (sliding layer) | S8 |
| origin/worktree-tts-audio-tokens @ 688fe758d771 | docs/fuse-funnel.md, docs/hardening-backlog.md | PERIPHERAL | TTS; router/session eviction backlog | S8 |
| origin/worktree-vindex2-la @ b65867eb170f | docs/vindex2-la.md, docs/vindex2-la-proposal.md | PERIPHERAL | Lookup-acceleration proposal | S8 |

Pickaxe (S10) commits that touch `mincore` and are not in base: aae20ae3 (PREFETCH-2), e2e0392f and 4a3823e2 (WIP parks), plus the mem-001 commits. In base: 63bc7c12 (PREFETCH-1), 21d73b0a (PREFETCH-1 prereg), 721e43c9 (V4 prereg), 68abfd70 (K3 vertical V3), 6874285f (GLM FP8 / real weights), b3097a3a (release hardening). `witness_residency` appears only in aae20ae3, 4a3823e2 and mem-001 commits. `MODALITY-RESIDENCY` appears only in mem-001 commits; no file in base or on any branch.

## (E) On-disk artefacts named by the records

| artefact | named by | status (ls on 2026-10-31) |
|---|---|---|
| `~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3` | k3-v4-warm-latency.json, k3-residency-vertical*.json, represent-real-evidence-1-notes.json | **NOT PRESENT** at that path |
| `~/chris-models/Kimi-Linear-48B-A3B-Instruct.s6.vindex3` (index v4, 27 layers, segments 92G per `du`), `.s7.vindex3` | k3-prefetch-notes.json ("the .s6 container"), evict-bank.sh | present (APFS clones possible; `du` may overstate) |
| `~/chris-models/Kimi-Linear-48B-A3B-Instruct` (HF checkpoint, 20 safetensors shards) | k3-funnel / evict-bank.sh (streamed for eviction) | present |
| `~/chris-models/Kimi-Linear-48B-A3B-Instruct.vindex3` and `.aligned.vindex3` | k3-residency-vertical | payloads deleted 2026-08-30; metadata tarballs in `~/chris-models/superseded/` (with superseded-containers.md) |
| `~/chris-models/candidates/kimi-map-l20-26q80-lift2.vindex3` | (lift2-family candidate) | present |
| `~/chris-models/k3/ledger-*.log`, `PRE-REGISTRATION.md` | k3-ledger runs | present (not opened) |
| `/Volumes/model-drive/models/GLM-5.3-Flash` (GLM checkpoint, ~306 GB) | frontier-execution §11–15, glm5-flash-funnel | **NOT MOUNTED**: `/Volumes` lists only `Macintosh HD`, `Recovery` |
| `/Volumes/model-drive/dec8_2.bin` | dec8_2_fio_sweep.py | not mounted |
| `output/gemma4-26b-a4b-q4k.vindex`, `output/granite-4.1-30b-q4k.vindex` (repo-relative) | moe-routing-locality.md, v3-disk-resident-mmap.md | not checked (it would live in another worktree's working tree, which is out of scope) |

## (F) Named external sources

| source | where cited | note |
|---|---|---|
| ggml-org/llama.cpp discussions #28248, #23324, #27149 | Listed by the brief. S7 grep at 51579782 found **no occurrence** in any repo file; they appear only in the mem-001 branch audit files (`docs/mem-001-reconnaissance-audit-1.md`, `-audit-2-phase1.md`), which were not read | not fetched |
| Other external URLs in CORE records | S7 `https?://` grep over the CORE files returned **none** | — |
| chuk-experiments registry entries (out-of-repo records): `dec8-0-k3-miss-budget`, `dec8-1`, `dec8-2` (USB4 random read), `dec8-5-k3-batch-union`, `dec8-6-slice-touch-ledger`, `dec8-11`, `dec8-12`, `dec8-13-cross-expert-conditional-census`, `rsl-rp`, `map-3-4-bridge`, `dec0-loopback-mac`, `k3r1-coverage-curve` (planned) | dec-funnel.md, lyrw-v2.md, quant-obs.md, ROADMAP.md, ROADMAP_STATUS.md, k3-funnel.md | Named by id only; their contents are not in the repo |
| `~/chris-source/chris-experiments/rsl/exp14–exp38`, `chris-experiments/larql_probes/...` | ROADMAP.md P1, ROADMAP_STATUS.md | Out-of-repo instruments |
| PRs #425, #428, #432, #433, #439 | k3-residency-vertical, k3-v4, k3-prefetch, k3-prefetch-2 records | GitHub PRs (not fetched) |

---

## Found beyond the seed list (CORE)

1. ROADMAP_STATUS.md l.518–530 (RSL-RP: hot cache justified, storage prefetch not, on a dense-layer estate) and l.666–672 (200/200/0 page prediction, f13bf385). This top-level file was not in the brief's search scope; it was found by S7.
2. ROADMAP.md l.300–310 and l.2307 (200 predicted / 200 resident), KU5/V3/C5 rows, and §P1 CR1 route-aware hot tier (1.80× owner grouping).
3. crates/larql-vindex/examples/vindex3_residency_probe.rs, mmap_cold_read_probe.rs, readrate.rs, and glm_moe_residency.rs + probe.rs (the instruments behind §11–14, KU5 and the 200/200 result).
4. crates/larql-vindex/src/runtime/residency.rs (`ResidencyAccount`), represent/physical.rs (`resident_bytes` via mincore), represent/codec/residency.rs, opplan/exec/prefetch.rs, realization.rs (`MappedAccess`), prepared/{residency,census}.rs, accounting/budgets.rs (`ResidencyBudget`), runtime/verdict.rs (`SelectedExpertNotResident`).
5. crates/larql-cli vindex3_cmd/generate/residency_curve.rs (`--residency-curve`, ru_majflt, quiet refusal) and vindex3_cmd/realizations.rs.
6. crates/larql-cli k3_ledger/retention.rs (DEC-9A Belady/MIN vs LRU vs StaticOracle). The code exists at base; **no DEC-9A result record was found in the repo**. Also freqmass.rs, budget.rs (DEC-8.0), touch.rs (DEC-8.6), block.rs (DEC-9.2).
7. docs/dec-funnel.md l.138–150 (freq-mass static-set coverage, adaptive cache 1.03×, cold tail).
8. crates/larql-cli dec_bench/window_union*.rs (the BW11-1 instrument) and crates/larql-compute ffn/expert_weight/trace.rs (`LARQL_MOE_ROUTE_TRACE`).
9. crates/larql-compute cpu/ops/moe/cache.rs (dequantised-expert LRU) with crates/larql-inference/ROADMAP.md M-CPU-3 (cap 64→256, p99 11.62→2.42 ms).
10. crates/larql-compute-metal buffers/residency.rs (`ResidencySet`, MTLResidencySet), moe_zero_copy.rs, moe_dispatch/entry.rs.
11. crates/larql-server routes/expert/warmup.rs and `--release-mmap-after-request` / `/v1/warmup` (documented in docs/cli.md).
12. docs/audits/expert-serving-review-2026-07-23.md l.92–100 (no cold-cache mode; replay measures the warm page cache).
13. docs/vindex3-experiments.md E2/E6 and docs/lyrw-v2.md (pre-registered, unrun physical-layout and locality experiments; "cold means cold" note).
14. docs/k3-funnel.md §4.13 V3 paragraph, plus GC/K3c coverage-curve gate (unrun).
15. docs/glm5-flash-funnel.md §8.2.2l/n and §8.2.3 (union-with-depth; MTP priced from routing divergence).
16. docs/v3-obs-1-carrier-observation.md l.472–476 (Kimi mapped/resident figures).
17. docs/diagnoses/remote-moe-bottlenecks.md (multi-layer expert prefetch recommendation).
18. scripts/k3_batch_union.py; bench/aim-validation/moe-routing/{analyze.py, v3moe_locality.json}; bench/aim-validation/v3_granite-30b.json.
19. crates/larql-vindex/docs/vindex3-format-spec.md l.169–240, 569–591 (locality store excluded from format; group extent = prefetch unit).
20. Branch material: the PREFETCH-2 code (prefetch.rs gating, new routing_trace.rs) on k3-prefetch-2 and both WIP branches; `k3-prefetch-2.json` / `k3-prefetch-2-notes.json`; the k3-prefetch-notes.json additions on the WIP branches; bw10-live-gate.md (residency regimes).
21. Discrepancies: `scripts/evict-bank.sh` is cited by base k3-prefetch-notes.json but exists only on branches; `contention.sh` is named in the branch notes but exists on no branch; the `lift2.vindex3` container named as the K3 subject is not on disk; `model-drive` is not mounted.

## Counts

- S1 strict-term files at base: 432 (crates 349, non-crate 83). S2 broad: 1,297.
- Rows classified here: A 53 (CORE 35, CORE-reference/spec 3, PERIPHERAL 15 rows; several PERIPHERAL rows group many files); B 31 (CORE 25, PERIPHERAL 6); C 10 (CORE 6 including branch-only evict-bank.sh, PERIPHERAL 4); D 14 rows over 7 branches (CORE 6, PERIPHERAL 7, EXCLUDED 1); E 9 artefacts (2 named paths absent, 1 volume unmounted, 1 not checked); F 5 source groups.
- S8: 23 branch-only (branch, path) rows over 19 distinct paths. S8 found 4 CORE paths (k3-prefetch-2.json, k3-prefetch-2-notes.json, evict-bank.sh, bw10-live-gate.md); S9 adds modified-file content on 4 more CORE branches.
- "Found beyond the seed list": 21 items.

## Limits of this inventory

- **Scope of S1/S2 is crates/, scripts/, docs/, bench/, ROADMAP.md**. ROADMAP_STATUS.md was added via S7. Not term-searched: `examples/`, `probes/`, `tools/`, `tests/`, `knowledge/`, `observatory/`, `registry/`, `data/`, `deploy/`, `META_MODEL.md`, `WORKING_MODEL.md`, `README.md`, `AGENTS.md`.
- **The crate code was triaged, not exhaustively classified**: S1 matched 349 crate files and only the clusters above were opened. KV, router, server-session and legacy dense-store hits were sampled and bulk-marked PERIPHERAL.
- **S8 only catches *new* files on branches whose text matches the terms**, and only under docs/ and scripts/. Branch-only changes to existing files, and branch-only code under crates/, are covered only for the five seed branches (S9). Other branches' code changes were not diffed.
- **Remote refs only**: 90 local `refs/heads` and uncommitted worktree state were not searched (except that `--all` in S10 includes local refs). `origin/mem-001-recon` moved during the run.
- **Out-of-repo records** (the chuk-experiments registry, `~/chris-source/chris-experiments`, auto-memory files, GitHub PR/issue bodies and comments) are named only. DEC-8.x and DEC-9A results, if they exist, would live there.
- **Term vocabulary**: results phrased without any listed term (e.g. "paged", "RSS", "unified buffer cache", "SLC", "spill", "swap", "compressed memory", "vm_stat", "footprint", "wired_limit") were found only incidentally. `wired` was in S2 but not followed up systematically.
- **Disk**: `/Volumes/model-drive` was not mounted, so artefacts named there could not be verified. `du` sizes may overstate because of APFS clones.
- **Relevance cut**: CORE vs PERIPHERAL is a one-line judgement from file headers and matching lines; PERIPHERAL rows carry reasons so the cut can be re-checked.
- Notes and pre-registrations written in commit messages only (not files) are covered only for the five pickaxe terms.
