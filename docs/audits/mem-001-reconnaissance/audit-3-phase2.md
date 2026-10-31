# MEM-001 reconnaissance, audit 3, phase 2 (reconciliation)

Revision under review: `95fd5764:docs/mem-001-reconnaissance.md` (revision 3).
Phase 1: `audit-3-phase1.md` beside this file, unchanged after it was written. "A3 P1-n"
below refers to its findings.

This phase read, after phase 1 was written:
- `95fd5764:docs/mem-001-reconnaissance-audit-1.md` (56 findings, against revision 1 `43388264`);
- `95fd5764:docs/mem-001-reconnaissance-audit-2-phase1.md` (P1-1 … P1-67, against revision 2 `b82d70b3`);
- `95fd5764:docs/mem-001-reconnaissance-audit-2-phase2.md` (dispositions, N-B1 … N-B24, §B1, §B2);
- revision 1 (`43388264`) §8, for context.

An earlier audit's finding was not taken as true. Each disposition was re-checked against the
primary source named, most of them already in phase 1. Where the earlier finding was itself
wrong, this says so.

Disposition codes:
- **RES**: resolved.
- **PART**: partially resolved.
- **NOTRES**: not resolved.
- **NLA**: no longer applicable.
- **VASC**: verified, and still correct in revision 3.

## A. Audit 1 (56 findings, against revision 1)

| A1 # | A1 verdict/sev | Disposition | Revision-3 location; primary source re-checked | Notes |
|---|---|---|---|---|
| 1 | V/I | VASC | header; `51579782` is an ancestor of origin/main | — |
| 2 | V/I | VASC | §1.1; `physical.rs:269` | — |
| 3 | V/I | VASC | §1.1; `residency_curve.rs:195` | — |
| 4 | A/NB | RES | §1.1 row 1; `loaded.rs:37`, `prepared/residency.rs:417-423` | — |
| 5 | V/I | VASC | §1.1; `realization.rs:247`, `prefetch.rs:66-127` | — |
| 6 | X/NB | RES | §1.1 MappedAccess row; `e2e0392f:k3-prefetch-notes.json` follow_ups (100.0 %) | — |
| 7 | V/I | VASC | §1.1; `residency_curve.rs:38-65` | — |
| 8 | X/NB | RES | §1.1 fault row, scoped to fault-path arms; GLM §14 `pread` and V3 minor faults cited | A1's own suggested text ("not shown in LARQL's evidence") was wrong. A2 caught it, and revision 3 uses the correct form |
| 9 | X/**B** | RES | §1.2 "Eviction methods"; §1.4; frontier §11, §14; `aae20ae3:evict-bank.sh`; `k3-prefetch-notes.json` cold_state_protocol | Residue: "Advice never evicted on Darwin" overstates (A3 P1-5, non-blocking) |
| 10 | V/I | VASC | §1.2; `expert_weight/trace.rs:11` | — |
| 11 | U/**B** | RES | §1.4 row 1; §1.1 routing row; `plan_backend.rs:375`; `retention_report.rs:25`; no `moe_route_observe` in `format/vindex3` | — |
| 12 | X/NB | RES | §1.1; fingerprint in V4/PREFETCH | — |
| 13 | V/I | VASC | §1.2; `retention.rs:60`, plus `--expert-bytes` (`args.rs:124`) | — |
| 14 | V/I | VASC | §1.1; `budgets.rs:101` | — |
| 15 | X/NB | RES | §1.1; `reconcile.rs:63-81` | — |
| 16 | V/I | VASC | §1.2; `41811c1d` | — |
| 17 | X/NB | RES | §1.2; `41811c1d:docs/diagnoses/bw10-live-gate.md` | — |
| 18 | X/**B** | RES | §1.4, §6.2, §7 Q5; `git log --all -S contention.sh`; ls-tree over all refs | — |
| 19 | A/NB | RES | §1.2 PREFETCH-2 row; branch tips | — |
| 20 | V/I | VASC | §1.2; no ref; the project note matches | — |
| 21 | X/NB | RES | §1.3; `cache.rs`, `warmup.rs:85-94`, `index/storage/residency.rs` | — |
| 22 | X/NB | RES | §1.3 "five call sites"; `mmap_storage.rs:386`, `gate_accessors/mod.rs:312/327` | A1's own list double-counted the wrapper and cited `:376`. A2 corrected it |
| 23 | X/NB | RES | §1.1 attribution row; `runtime/residency.rs:163,175` | — |
| 24 | X/**B** | RES | §0, §2.1, §6; frontier §11-§14 | Residue: §0 "only one layer has ever been probed" contradicts §14's virgin layers (A3 P1-4, non-blocking) |
| 25 | A/I | RES | §1.2 cold-read row, §3 | — |
| 26 | A/I | RES | §1.2 non-plan memory row (peak RSS) | — |
| 27 | V/I | VASC | §2.3 round 1 column | — |
| 28 | V/I | RES | §2.4 per-fault table (GLM excluded; 16.4 identified) | A1's note ("16.4 not reproducible from PREFETCH-1's fields") was correct. Revision 3 identifies 16.4 with round 2's 1205/73,520 as fact, which is inference (A3 P1-10) |
| 29 | X/NB | RES | §2.3 "1.199 / 1.206 / 1.205" | — |
| 30 | X/**B** | RES | §0, §2.3, §2.4: three PREFETCH rounds plus V3 | — |
| 31 | X/**B** | RES | §0, §2.4: reconciled within PREFETCH (2.948 × 0.409 = 1.206), explicitly not across V3/PREFETCH; e2e0392f follow_ups; V3 rung 7 | Residue: V3's 67 % is presented beside a witnessed figure without "derived" (A3 P1-9, non-blocking) |
| 32 | V/I | VASC | §1.5 | — |
| 33 | V/I | VASC | §1.5; Kimi config | — |
| 34 | X/**B** | **PART** | §0, §2.2. Subject correctly named (lift2), `.s6` correctly a displacer; `k3-v4-warm-latency.json:6` | **Residual BLOCKING (A3 P1-1).** The identity chain says lift2 has no digest, but `51579782:docs/v3-obs-1-carrier-observation.md:481-489` records lift2's `index.json` SHA-256 `ddd9cad3e4603d12…`, identical to `.s6`/`.s7` (on-disk hash confirmed). A1's own correction ("Whether `.s7`'s bank equals lift2's is unverified") and A1 §4.1 ("no digest of it was found in the repo") were **themselves wrong** |
| 35 | V/I | VASC | §1.5; config | — |
| 36 | A/NB | RES | §3 "three of ~91.5 GiB"; §1.2 "~183 GiB (~196.5 GB)" | Unsourced figure (A3 P1-11, non-blocking) |
| 37 | V/I | NLA | the disk-free sentence was removed in revision 2 | — |
| 38 | A/NB | RES | §2.3 "Carried observations, each from a single round" | — |
| 39 | X/NB | RES | §2.3 cold-state row | — |
| 40 | V/I | VASC | §2.0 last paragraph | — |
| 41 | U/NB | RES | §1.2 evict-bank row | — |
| 42 | A/NB | RES | §2.3 footnote; §2.4 quantity table | — |
| 43 | V/I | VASC | §2.4 "Inferred explanations, none established" | — |
| 44 | V/I | VASC | throughout (see A3 phase 1 §3) | — |
| 45 | U/NB | RES | §4 instrument table | — |
| 46 | V/I | RES | §4 file-page row now cites the V3 second run | — |
| 47 | V/I | VASC | §3 | — |
| 48 | A/NB | RES | §4 #28248 row; #28248 body | — |
| 49 | V/I | VASC | §4 #28248 row | — |
| 50 | V/I | RES | §4 #23324: "the vanilla row's flags are not stated" | A1's suggested addition ("rows differ in flags") was **not supported** by the source. A2 corrected it |
| 51 | A/I | RES | §4 #27149 | — |
| 52 | X/NB | RES | §4 #27149 "pipelined 2.40–4.45 (sequential 2.20–4.46)" | — |
| 53 | U/NB | RES | §1.5 "contradicts itself on layout … How 111× was derived is not explained" | A1's "inconsistent (6,912/192 = 36)" framing was itself unsound (different denominators). A2 corrected it |
| 54 | A/NB | RES | §4 "the only one with a measured end-to-end decode rate …" | A1's wording was too strong. A2 corrected it |
| 55 | V/I | VASC | §6 | — |
| 56 | A/I | RES | §2.3 qualified warm reference | — |

Counts: VASC 20, RES 34, PART 1 (#34), NOTRES 0, NLA 1 (#37). Total 56.

## B. Audit 2, phase 1 (P1-1 … P1-67, against revision 2)

| A2 # | A2 verdict/sev | Disposition | Revision-3 location; source re-checked | Notes |
|---|---|---|---|---|
| P1-1 | V/INFO | VASC | header | — |
| P1-2 | V/INFO | RES | §1.2 evict-bank row "branch `k3-prefetch-2` (`aae20ae3`) and its WIP descendant `e2e0392f`" | — |
| P1-3 | V/INFO | VASC | §1.1 | — |
| P1-4 | A/NB | RES | §1.1 row 1 (boundary pages); `physical.rs:283-301` | The A2 mechanism re-checked: 288-page matrices with an unaligned payload give 289-page ranges (624 × 289 × 16,384 = 2.955 GB, observed) |
| P1-5 | V/INFO | VASC | §1.1 | — |
| P1-6 | V/INFO | VASC | §1.1 (peak RSS) | — |
| P1-7 | I/NB | RES | §1.1 fault row; §5.6 | — |
| P1-8 | V/INFO | VASC | §1.1 | — |
| P1-9 | V/INFO | VASC | §1.1 | — |
| P1-10 | V/INFO | VASC | §1.1 | — |
| P1-11 | V/INFO | VASC | §1.1 | — |
| P1-12 | V/INFO | VASC | §1.1 | — |
| P1-13 | V/INFO | RES | §1.2 GLM probe row cites `6874285f` | Independently re-verified: logic identical (A3 P1-18) |
| P1-14 | A/NB | RES | §1.2 "Virgin layers … §13 does not state its cold method" | — |
| P1-15 | V/INFO | VASC | §1.2 | — |
| P1-16 | V/INFO | VASC | §1.2 | — |
| P1-17 | A/NB | RES | §1.2 "equality is asserted, not shown" | — |
| P1-18 | I/NB | RES | §1.2, §5.2 "~183 GiB (~196.5 GB)" | Correct from disk; the source of the figure is not given (A3 P1-11) |
| P1-19 | V/INFO | VASC | §1.2 | — |
| P1-20 | I/NB | RES | §1.2 retention row; `args.rs:124`, `retention_report.rs:27` | — |
| P1-21 | V/INFO | VASC | §1.2 | — |
| P1-22 | V/INFO | VASC | §1.2 | — |
| P1-23 | A/INFO | RES | §1.2 "a **peak**, not live RSS" | — |
| P1-24 | A/NB | RES | §1.2 Metal path row; `routed.rs:403-415`, `execution_cost.rs:22-24`, `teacher_forced/run.rs:51-59` | — |
| P1-25 | V/INFO | VASC | §1.2 | — |
| P1-26 | V/INFO | VASC | §1.3 | — |
| P1-27 | I/NB | RES | §1.3 five call sites | — |
| P1-28 | V/INFO | RES | §1.2 device-level read sweep row | — |
| P1-29 | A/INFO | RES | §1.5 quotes the type doc | — |
| P1-30 | V/INFO | RES | §1.5 "in `.s6`'s and `.s7`'s `index.json` (lift2's own is unavailable)" | The parenthesis is wrong in substance: lift2's index.json digest is recorded and equals s6/s7's (A3 P1-1) |
| P1-31 | A/NB | RES | §1.5 "consistent with" | A writer-side ordering rule does exist (see §E, out of inventory) |
| P1-32 | A/NB | RES | §1.5 | — |
| P1-33 | V/INFO | VASC | §2.1 | — |
| P1-34 | I/INFO | RES | §2.1 warm 5.3 % | — |
| P1-35 | V/INFO | VASC | §2.1 | — |
| P1-36 | V/INFO | VASC | §2.2 | — |
| P1-37 | A/**B** | **PART** | §0, §2.2 identity-chain table, §7 Q1; `represent-real-evidence-1-notes.json:30` now cited | **A2's finding was itself incomplete and partly wrong.** It stated "lift2 ≈ s6 by size and spot-check (not by digest)" and "short of a lift2 digest". `51579782:docs/v3-obs-1-carrier-observation.md:481-489` records lift2's index.json SHA-256 (`ddd9cad3e4603d12…`), equal to `.s6`'s and `.s7`'s, plus three distinct `system_graph.json` hashes. Revision 3 adopted A2's wording ("no digest", "absent"). **Residual BLOCKING = A3 P1-1** |
| P1-38 | V/INFO | VASC | §2.3 | — |
| P1-39 | A/INFO | RES | §2.3 footnote | — |
| P1-40 | I/NB | RES | §2.3 Samples row | — |
| P1-41 | A/NB | RES | §2.3 prompt row; §2.4 prompt ratios | — |
| P1-42 | V/INFO | VASC | §2.3 | — |
| P1-43 | V/INFO | VASC | §2.3 | — |
| P1-44 | V/INFO | VASC | §2.4 | — |
| P1-45 | I/NB | RES | §2.4 "The GLM row is not on this basis" | — |
| P1-46 | A/NB | RES | §2.4 | A2's identification (16.4 = round 2 p50) is an arithmetic match that revision 3 states as fact (A3 P1-10, non-blocking) |
| P1-47 | I/**B** | **RES** | §0 four-observation table; §2.0; §2.2 V3 section; §2.4 byte table ("unexplained by any source"); `k3-residency-vertical-notes.json` rung 7, V3-N3/N4/N5 (all figures re-verified) | Residue: V3's 67 % is derived but placed beside a witnessed figure (A3 P1-9). V3's "unexplained" warm spread is contradicted by V4-N1 (A3 P1-3). Both non-blocking |
| P1-48 | V/INFO | VASC | §2.4 | — |
| P1-49 | U/NB | RES | §2.4 (i), §5.3 | — |
| P1-50 | V/NB | RES | §1.1 mapping-advice row; `physical.rs:96,135`, `read.rs:121`, `prefetch.rs:93` | — |
| P1-51 | A/NB | RES | §2.1 note; §2.4 "Requested" definition | — |
| P1-52 | V/INFO | VASC | §3 | — |
| P1-53 | I/NB | RES | §3 "328,326,771,576 B (305.78 GiB, 328 GB; frontier §1), on `model-drive` (frontier header and §13)" | — |
| P1-54 | V/INFO | VASC | §3 | — |
| P1-55 | V/INFO | RES | §4 #28248 later decomposition added | — |
| P1-56 | A/INFO | RES | §4 instrument table | — |
| P1-57 | V/INFO | VASC | §4 | — |
| P1-58 | A/NB | RES | §4 #23324 | — |
| P1-59 | V/INFO | VASC | §4 | — |
| P1-60 | A/NB | RES | §4 identity paragraph | — |
| P1-61 | I/NB | RES | §4 file-page row | — |
| P1-62 | V/INFO | VASC | §4 | — |
| P1-63 | A/NB | **PART** | §1.2 eviction bullets now separate advice from invalidation | A2's suggested wording "Advice never evicted" overstates: `v3-disk-resident-mmap.md` records the first eviction of a fresh file as reliable, and `vindex3_residency_probe.rs` notes the probes disagree (A3 P1-5, non-blocking) |
| P1-64 | V/INFO | VASC | §5-§7 | Re-confirmed for revision 3 (A3 phase 1 §5) |
| P1-65 | NC/INFO | VASC (now checked) | §8. Audit 1: 7 BLOCKING, 17 INCORRECT, 56 findings (counted: X at #6, 8, 9, 12, 15, 17, 18, 21, 22, 23, 24, 29, 30, 31, 34, 39, 52 = 17). Audit 2: 67 findings, 3 BLOCKING; phase 2: 5 resolved and 2 partial among A1's BLOCKING; "four partial resolutions came from adopting audit 1's suggested wording" (A2 phase 2 l.104); witness presentation refuted (W-1, W-4, W-5, W-7). Revision 1 §8 maintainer outcome present at `43388264` | — |
| P1-66 | NC/INFO | NOTRES (statement now known false) | §1.5 "No writer-side ordering guarantee was found" | `51579782:…/vindex3/encode/segment.rs:8-13,93-97` has "Deterministic payload order: sorted by object-relative name" (since 2026-08-11/12, before lift2's 2026-08-31 encode). That places each expert's three matrices (common prefix `…experts.N.`) adjacent, which supports the document's "consistent with" inference. Out of inventory (§E); non-blocking |
| P1-67 | U/**B** | **RES** | §2.5 (Gemma locality, all figures and quotes re-verified at `moe-routing-locality.md` and `v3moe_locality.json`); §1.2 `larql moe-locality` and routing-analysis rows; §6.4 | The sibling freq-mass row added in §2.6 carries a new BLOCKING error (A3 P1-2: same model, not a different one). That is outside P1-67's scope |

Counts: VASC 31 (including P1-65, now checked), RES 33, PART 2 (P1-37, P1-63), NOTRES 1
(P1-66), NLA 0. Total 67. P1-30 is counted RES. Its leftover parenthesis "(lift2's own is
unavailable)" falls under A3 P1-1.

## C. Audit 2, phase 2 new findings (N-B1 … N-B24)

| N-B | = | Disposition | Revision-3 location | Notes |
|---|---|---|---|---|
| N-B1 | P1-47 | RES | §0, §2.0, §2.2, §2.4 | see P1-47 |
| N-B2 | P1-37 | **PART** | §2.2 | residual BLOCKING A3 P1-1 |
| N-B3 | P1-67 | RES | §2.5, §1.2 | — |
| N-B4 | provenance | RES | §2.0 table (all BACKGROUND ONLY), §6 "Every input … is BACKGROUND ONLY" | A2's §B1 classifications re-checked against commits and files (A3 phase 1 §2.4 §2.0). The GLM default prompt and `--max-tokens 192` were re-verified |
| N-B5 | witness | RES | §0 table "witness-adjusted estimate (16.4 µs raw, witnessed)"; §2.3 "Round 2 is witnessed residency evidence, not a latency record"; §2.4 three bases, adjusted ratios, range "549–607 ms (round 2 adjusted)"; §5.4 | W-1, W-2, W-4, W-5, W-7 are all addressed |
| N-B6 | P1-7 | RES | §1.1 | — |
| N-B7 | P1-31 | RES | §1.5 | — |
| N-B8 | P1-32 | RES | §1.5 | — |
| N-B9 | P1-60 | RES | §4 | — |
| N-B10 | P1-58 | RES | §4 | — |
| N-B11 | P1-61 | RES | §4 | — |
| N-B12 | P1-45/46 | RES | §2.4 | residue A3 P1-10 (inference stated as fact) |
| N-B13 | P1-40 | RES | §2.3 | — |
| N-B14 | P1-41 | RES | §2.3, §2.4 | — |
| N-B15 | P1-49 | RES | §2.4 (i), §5.3 | — |
| N-B16 | P1-50 | RES | §1.1, §2.4 (iii), §5.5 | — |
| N-B17 | P1-24 | RES | §1.2, §3 | — |
| N-B18 | P1-20 | RES | §1.2, §6.4 "passed with `--expert-bytes`" | — |
| N-B19 | P1-27 | RES | §1.3 | — |
| N-B20 | P1-18 | RES | §1.2, §5.2 | residue A3 P1-11 (unsourced) |
| N-B21 | P1-53 | RES | §3 | — |
| N-B22 | P1-4 | RES | §1.1, §2.4, §5.10 | — |
| N-B23 | P1-14/17/63 | **PART** | §1.2 eviction bullets | P1-14 and P1-17 are resolved. P1-63's adopted wording ("never") overstates (A3 P1-5) |
| N-B24 | P1-51 | RES | §2.1, §2.4 | — |

Counts: RES 22, PART 2 (N-B2, N-B23). Total 24. N-B12 and N-B20 carry only non-blocking
residue and are counted RES.

A2 §B1 (traceability) was re-checked: every row's commit, date and missing-item list holds,
with two notes. (a) For Kimi, A2's "No lift2 digest" is wrong (v3-obs-1:487). (b) V3's code
was also uncommitted at run time (A3 P1-15), the same class A2 gave only to round 2. The class
BACKGROUND ONLY stands for every row.

A2 §B2 (witness check) was re-checked against revision 3: all nine places are now accounted
for (see N-B5).

## D. Status of the ten named BLOCKING findings

| Finding | Status in revision 3 | Evidence |
|---|---|---|
| A1 #9 (eviction method) | **RESOLVED** | §1.2 Eviction methods, §1.4, §5.2, §7 Q4. Residual wording "never" is non-blocking (A3 P1-5) |
| A1 #11 (VINDEX3 routing → simulator) | **RESOLVED** | §1.4 row 1, §1.1, §1.2; code re-checked |
| A1 #18 (`contention.sh`) | **RESOLVED** | §1.4, §6.2, §7 Q5; all refs searched |
| A1 #24 (GLM §11-§14 omitted) | **RESOLVED** | §0, §2.1, §6. Residual "only one layer" is non-blocking (A3 P1-4) |
| A1 #30 (three rounds, not two) | **RESOLVED** | §2.3, §2.4 (now three rounds plus V3) |
| A1 #31 (byte gap) | **RESOLVED** | §2.4: reconciled within PREFETCH, explicitly unexplained across V3/PREFETCH. Residual labelling is non-blocking (A3 P1-9) |
| A1 #34 (subject identity) | **PARTIALLY RESOLVED** | The subject is correctly named. The identity chain misstates the record: lift2's digest exists (A3 P1-1, BLOCKING) |
| A2 P1-37 (lift2 clone record) | **PARTIALLY RESOLVED** | The clone record is cited, but A2's own "no digest" framing, which revision 3 adopted, is contradicted by v3-obs-1:481-489 (A3 P1-1, BLOCKING) |
| A2 P1-47 (V3 rung 7) | **RESOLVED** | §0, §2.0, §2.2, §2.4; all figures re-verified |
| A2 P1-67 (Gemma locality) | **RESOLVED** | §2.5, §1.2. The adjacent freq-mass row has a separate new BLOCKING error (A3 P1-2) |

## E. New discrepancies (separate from the reconciliation)

### E.1 Blocking factual errors

1. **A3 P1-1: lift2's recorded digest is omitted, and its absence is asserted.**
   - Revision 3 says "No lift2 digest exists", "no digest", "absent" (§0, §2.0, §2.2, §7 Q1).
   - Main's `docs/v3-obs-1-carrier-observation.md:481-489`, a CORE inventory record, records SHA-256 of `index.json` and `system_graph.json` for `.lift2`, `.s6` and `.s7`. It was recorded before lift2's deletion on 2026-09-20.
   - All three share index.json `ddd9cad3e4603d12…` (hash re-computed on disk for `.s6` and `.s7`). That file carries the declared expert-bank `payload_sha256` `3f12d102…`.
   - The three differ in `system_graph.json`. lift2 lacked `kda_gate_form`, and a later binary refused it at first step.
   - So the weight identity of lift2 with `.s7` is recorded by declared digest; it is not "size and spot-check only". What really separates them is the declaration. §7 Q1 asks the wrong question.
   - This is the same failure mode A2 named: an earlier audit's wording adopted without re-reading the source. Here it was A2's own wording, despite revision 3's header claim that "each correction was re-checked against its primary source".
2. **A3 P1-2: the freq-mass result is described as being on "a different model" from §2.5.**
   - The DEC-0 pool is `bench/dec0/residuals-gemma4-26b-a4b-q4k-routed` (`docs/cli.md:492`). bw11:72 places DEC-0 on Gemma-4 26B-A4B, and 3,840 layer-expert pairs = 30 × 128.
   - It is the same model as §2.5. Both "no small hot set" records are therefore one model with a load-balanced router, which the Gemma diagnosis itself warns may not generalise. Revision 3 presents them as cross-model corroboration.

### E.2 Non-blocking discrepancies (new in audit 3)

- A3 P1-3: V3's warm spread is called "unexplained" in §5.7 (V4-N1 on main; V4-N5 on the branch).
- A3 P1-4: "only one layer has ever been probed".
- A3 P1-5: "advice never evicted".
- A3 P1-6: two funnel sections are mischaracterised as unrun.
- A3 P1-7: BW11-1's short-window locality result is not summarised.
- A3 P1-8: round-2 code is called "not recoverable" despite candidate 4a3823e2.
- A3 P1-9: V3's derived 67 % sits beside a witnessed 59.1 %.
- A3 P1-10: the 16.4 identification is inference stated as fact.
- A3 P1-11: "~183 GiB" is unsourced.
- A3 P1-12: the clone record post-dates the runs.
- **Out of inventory (found while verifying §1.5 / A2 P1-66):** `encode/segment.rs:93-97` defines a writer-side payload order (sorted by relative name). §1.5's "No writer-side ordering guarantee was found" is therefore false, but the rule supports the document's inference. Non-blocking.

### E.3 Legitimate unresolved historical uncertainty

Unchanged from phase 1 §7:
- GLM weight digest, inputs, raw outputs, machine state and trace/prompt;
- V3 eviction method, machine state and fingerprint, and code committed after the run;
- PREFETCH round 1's script reconstruction and single samples;
- round 2's machine state, missing unwitnessed record and code identity;
- round 3's exact commit;
- lift2's bytes (cannot be re-hashed; the declared digest **is** recorded);
- `contention.sh`;
- the MODALITY-RESIDENCY-1a artefact;
- the Gemma passage and container.

None blocks acceptance where it is stated accurately. The lift2 item is accurately bounded only after A3 P1-1 is corrected.

## F. Final recommendation

**REJECT.**

Revision 3 cannot merge as the reconnaissance record while these two blocking factual errors
stand. Each misstates what an in-inventory prior record establishes:

1. **A3 P1-1.** The subject-identity chain asserts that no lift2 digest exists. Main's `docs/v3-obs-1-carrier-observation.md:481-489` records lift2's `index.json` SHA-256, equal to `.s6`'s and `.s7`'s, and records that the three containers differ only in their declarations (`system_graph.json`). This affects §0, §2.0, §2.2 and §7 Q1.
2. **A3 P1-2.** The freq-mass result is described as being on a different model from the Gemma locality result. It is on the same model, Gemma 4 26B-A4B (`docs/cli.md:492`; 3,840 = 30 × 128). This overstates the breadth of the evidence against a static hot-expert cache in §2.6 and §0.

Everything else meets the acceptance criteria:
- all ten named BLOCKING findings from audits 1 and 2 are resolved, except #34 and P1-37, whose residue is item 1;
- every load-bearing number was verified, or is marked as an estimate or derivation (with the non-blocking labelling fixes A3 P1-9 and P1-10);
- irreproducible history is stated accurately;
- the document remains reconnaissance.

Both corrections are text edits against sources already in the inventory, and need no new
measurement. The ten non-blocking items (A3 P1-3 … P1-12) and the out-of-inventory segment-order
note should be fixed in the same pass.

This recommendation concerns the record only. It is not permission to run MEM-001A, MEM-001B
or any experiment.
