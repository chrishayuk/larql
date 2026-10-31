# MEM-001 reconnaissance: revision 4 targeted regression check

## Scope

- Subject: `git diff 95fd5764 20470b22 -- docs/mem-001-reconnaissance.md` (revision 3 to revision 4), plus a consistency search of the complete `20470b22:docs/mem-001-reconnaissance.md` (877 lines).
- Problem statements: `20470b22:docs/mem-001-reconnaissance-audit-3-phase1.md` P1-1..P1-12, and the writer-ordering note in `…-audit-3-phase2.md` (rows P1-31/P1-66, §E). These were treated as problem statements only. Every claim was re-checked at the primary source.
- Sources at main `51579782` unless a branch commit is named. Reads went through git objects only. On disk: `ls ~/chris-models`, `shasum -a 256` of `.s6`/`.s7` `index.json` and `system_graph.json`, a JSON parse of `.s7` `index.json`, and logical file sizes (`stat`) for the displacer arithmetic. No weight file was opened. No build, run, edit, commit or GitHub action.
- Not a completeness audit. No new historical-evidence search was run.

## 1. Changed claims

| # | Location (rev 4) | Claim | Source checked | Verdict | Severity |
|---|---|---|---|---|---|
| C1 | header table | rev 3 `95fd5764`, audit 3: 22 findings, 2 BLOCKING, REJECT | audit-3 phase 1 (P1-1..P1-22, two in §2.1 BLOCKING); phase 2 l.264 "**REJECT.**" | VERIFIED | — |
| C2 | §0 Subject identity; §2.2 table row 1 | lift2 deleted; before deletion main recorded its `index.json` SHA-256, equal to `.s6`'s and `.s7`'s (`ddd9cad3…`, `v3-obs-1-carrier-observation.md:481-489`) | `51579782:docs/v3-obs-1-carrier-observation.md` l.481 heading "recorded before `.lift2` was deleted on 2026-09-20", l.487-489 all three `ddd9cad3e4603d12…`; blame `229ae333` 2026-09-20 00:31; lift2 absent from `~/chris-models` | VERIFIED | — |
| C3 | §2.2 table row 1 | on disk today `.s6`/`.s7` `index.json` hash `ddd9cad3e4603d12dfbc…` | `shasum -a 256`: both `ddd9cad3e4603d12dfbc83535020d93e9bd0c6c2ca9169ccd1cad4c0d55170ec` | VERIFIED | — |
| C4 | §0; §2.2 row 2 | that `index.json` declares the expert bank `payload_sha256` `3f12d102…`, `segment_sha256` `e43600cd…`; declared bank identity established; payload not re-hashed | `.s7` `index.json`: `representations/target.expert_bank@BF16/payload_sha256 3f12d102041b…`, `segment_sha256 e43600cd02b6…` | VERIFIED. The "not re-hashed" limit is stated | — |
| C5 | §0; §2.2 row 3 | system graphs differ: lift2 `1b4c0fc4…`, `.s6` `a33b81a9…`, `.s7` `1fe02aba…`; lift2 lacks `kda_gate_form`, "refused at first step (gate form absent)"; `.s6` lacks latent-norm epsilon; `.s7` "the container of record" | record l.487-489 (exact quotes, `kv_a_norm_eps` absent for `.s6`); on-disk `.s6` `a33b81a92da58237…`, `.s7` `1fe02abad2f3bcdc…` | VERIFIED | — |
| C6 | §2.2 row 4 | `represent-real-evidence-1-notes.json:30` dated 2026-09-18 | blame l.30 = `9f2afe97`, 2026-09-18 23:50 | VERIFIED | — |
| C7 | §2.2 new paragraph | `.s7` carries lift2's declared bank; its system graph differs from the one the historical runs used | as C2/C5 | VERIFIED as to the declared bank. "The one the historical runs used" assumes lift2's system graph was unchanged between the runs (2026-09-05/06) and the record (2026-09-20). Same class as P1-12, not stated here | INFORMATIONAL |
| C8 | §1.5 | the bank payload is in `.s6`/`.s7` `index.json`, "whose recorded hash lift2's matched" | C2 | VERIFIED | — |
| C9 | §2.0 V3 row; text after table | "declared bank identity recorded"; "recorded identity chain (declared bank identity)" | C2-C5 | VERIFIED | — |
| C10 | §7 Q1 | recorded `index.json` hash matches `.s7`'s; system graphs differ; asks whether declared bank identity plus a declaration difference pins `.s7`, and whether to re-hash payload | C2-C5 | VERIFIED. Matches the required framing | — |
| C11 | §0 locality; §2.6 freq-mass row | freq-mass pool is the same model, Gemma 4 26B-A4B: `bench/dec0/residuals-gemma4-26b-a4b-q4k-routed` (`docs/cli.md:492`); 3,840 = 30 × 128; 64-prompt mixture | `cli.md:492` exact pool path; `dec-funnel.md:142` "64-prompt domain mixture", `:146` "620 of 3840 layer-expert pairs"; `moe-routing-locality.md:20` "128 experts, top_k=8, 30 hybrid MoE layers" | VERIFIED | — |
| C12 | §0 | Gemma figures 124.4/128; 6.25 %/42 %/35 %/1.03× | unchanged figures, `moe-routing-locality.md:35`; `dec-funnel.md` freq-mass block | VERIFIED | — |
| C13 | §0; §2.6 new BW11-1 row | BW11-1: GPT-OSS 20B, 4 of 32, 24 routed layers, 5 short prompts, prefill + 8 decode; 1.28/1.72/2.40× at K=2/4/8; 6.2–10.5 % below shuffled-marginal control; "real, controlled, modest"; closed 2026-08-20; `larql dec-bench window-union` | `bw11-weight-traffic.md:4` (status quote, date), `:79` (4-of-32), `:149-161` (§3.5: 5 prompts, 24 layers, prefill + 8 decode, table 6.2/10.4/10.5 %) | VERIFIED | — |
| C14 | §0; §1.2 eviction | advice records conflict: GLM run `MADV_DONTNEED` did not evict, `MADV_FREE_REUSABLE` `EPERM` (frontier §11); `vindex3_residency_probe` records `MADV_DONTNEED` leaving a fresh file fully resident; KU5 cold-read probe records it as lazy, only first eviction on a fresh file reliable | `frontier-execution-programme.md:791-793`; `vindex3_residency_probe.rs:84-85,166-175` (4096/4096 resident); `v3-disk-resident-mmap.md:1` (KU5), `:23-24` | VERIFIED | — |
| C15 | §5.2 | "Advice is not a usable eviction method on the records, which conflict" | C14 | VERIFIED | — |
| C16 | §0 | GLM: only single-layer probes have run (layer 3; in §14 one arm per virgin layer) | frontier l.778 (layer 3), l.1000 "parallel n=7 across virgin layers", l.1025-1026 "one arm per layer" | VERIFIED | — |
| C17 | §2.2 warm token; §5.7 | main V4-N1 attributes 140/430 ms to "the machine, not the model"; branch V4-N5 closes it (qualified 75.4 ms) | `51579782:…/k3-v4-warm-latency-notes.json:112` (exact quote); `254377ce:` same file l.182 "CLOSED: V3's 140–430 ms warm spread was the machine, not the model … 75.4 ms p50" | VERIFIED | — |
| C18 | §0; §2.4 table | V3 67 % is V3-N3's derivation (1 − 0.976/2.944), not witnessed | `k3-residency-vertical-notes.json` V3-N3: "paged in 0.976 GB because 67% of its expert pages were already resident"; 2.944 GB is "the DISJOINT bound"; V3 had no pre-loop witness | VERIFIED in substance. V3-N3 states the 67 % as an explanation and gives no formula; the formula is the document's reconstruction (already in §2.2 rev 3). Calling it "derived, not witnessed" is correct | INFORMATIONAL |
| C19 | §2.4 | 16.4 matches round 2's witnessed p50 (1205/73,520 = 16.39), not round 1's 17.0; source does not say which | `e2e0392f:k3-prefetch-2-notes.json:54` / `aae20ae3:k3-prefetch-2.json:39` "from 16.4 us per fault to 8.2" with no basis named; 1253/73,516 = 17.04 | VERIFIED | — |
| C20 | §2.0 round 2 row | code that ran not established; candidate parked at `4a3823e2` (2026-09-23, 17 days after the run), equality unestablished | `4a3823e2` 2026-09-23 20:09, "uncommitted work parked from worktree cleanup", 11 paths incl. prefetch.rs, realization.rs, routing_trace.rs, mod.rs, k3-prefetch-notes.json; run date 2026-09-06 from `e2e0392f:k3-prefetch.json:4` (frozen 2026-09-06) and `k3-prefetch-2-notes.json:5` (measurement paused 2026-09-06); 09-06 → 09-23 = 17 days; `e2e0392f:k3-prefetch-2-notes.json:10` "stranded uncommitted in the old worktree; ported forward" | VERIFIED | — |
| C21 | §1.2 evict-bank row; §5.2 | records say "≈184 GB … 92 GB each" (GB evidently GiB); today's disk 183.0 GiB (196.5 GB) | `51579782:docs/represent/forecasts/k3-prefetch.json:13` "≈184 GB of other files (the HF checkpoint and the .s6 container, 92 …"; `aae20ae3:scripts/evict-bank.sh:8` "~92 GB each", `MIN_SIZE +100M`; disk: HF 98,248,224,120 B + `.s6` 98,248,019,504 B = 196,496,243,624 B = 183.0 GiB = 196.5 GB | VERIFIED | — |
| C22 | §1.2; §2.4 (ii); §5.2 | `.s6` recorded as a clone on 2026-09-18, after the rounds; clone status during them assumed | C6; rounds 2026-09-06 (C20) | VERIFIED | — |
| C23 | §1.5 writer ordering | writer sorts each segment's payload by object-relative name (`encode/segment.rs:93-97`, `sort_into_payload_order`, since `c554266d`, 2026-08-12); name order only; physical adjacency of an expert's three matrices not checked | `51579782:crates/larql-vindex/src/format/vindex3/encode/segment.rs:93-97` (blame: all five lines `c554266d`); `c554266d` 2026-08-12 00:38 | VERIFIED. Claims only name order | — |
| C24 | §2.6 "Identified by the inventory" | `k3-funnel.md` §4.13 = V3 pre-registration and result summary | `k3-funnel.md:912` heading, `:914-916` "Preregistered in …", `:946` V3 result paragraph | VERIFIED | — |
| C25 | §2.6 same list | GLM §8.2.2l/§8.2.2n measured cross-perturbation union (1.15× mean, 1.50× worst across eight arms, flat with depth); §8.2.3 arithmetic from GLM §12 | `glm5-flash-funnel.md:1808-1820` (8 arms, layer 3, "1.15× mean, 1.50× worst"); `:1912`, `:1936` "FLAT / BOUNDED"; `:2191-2194` (0.90 of 8 from the 128-token reuse curve) | VERIFIED. The 1.15/1.50 figures are §8.2.2l's (layer 3, 48 tokens); "flat with depth" is §8.2.2n's (per-token 1.03–1.15×). Accurate as a combined summary | INFORMATIONAL |
| C26 | §8 audit-3 and maintainer paragraphs | audit-3 summary (22 findings, 2 BLOCKING, eight resolved and two partly resolved, REJECT); direction for rev 4 and this check | audit-3 phase 1 and 2 (§D, l.264, l.273) | VERIFIED as summary. The maintainer direction is the user's own record, not checkable here | — |

## 2. Resolution status

| Finding | Status | Evidence in rev 4 |
|---|---|---|
| P1-1 (BLOCKING) lift2 digest | **RESOLVED** | All four locations corrected: §0 (l.83-90), §2.0 V3 row (l.290) and text after the table (l.296-297), §2.2 identity table (l.368-371) plus new paragraph (l.377-380), §7 Q1 (l.788-793). §1.5 leftover "(lift2's own is unavailable)" also fixed (l.250). No "no digest", "absent" or "no lift2 digest" survives |
| P1-2 (BLOCKING) freq-mass model | **RESOLVED** | §0 (l.92-103) "the same model … not a cross-model replication"; §2.6 row (l.622). §6.4's question (l.778) asks about Kimi and GLM, which stays consistent |
| P1-3 V3 warm "unexplained" | RESOLVED | §2.2 (l.398-400) keeps the V3-N5 quote and adds V4-N1/V4-N5; §5.7 (l.723) |
| P1-4 "only one layer" | RESOLVED | §0 l.117-118 |
| P1-5 advice "never" | RESOLVED | §0, §1.2, §5.2. No "never evict(ed)/work(ing)" remains |
| P1-6 "unrun" characterisation | RESOLVED | §2.6 list split. "unrun" now only on E2/E6/lyrw-v2 and `contention.sh` ("written and unrun", quoted) |
| P1-7 BW11-1 omission | RESOLVED | new §2.6 row; §0 mention |
| P1-8 round 2 "not recoverable" | RESOLVED | §2.0 round 2 row. The heading "§1.4 … or are not recoverable" is a category heading, not this claim |
| P1-9 V3 67 % labelled | RESOLVED, with one residue | §0 and the §2.4 table now say derived. §2.4 (iv) (l.531, unchanged) still reads "V3 found about 67 % … resident from the prompt, and PREFETCH 59.1 %", which puts them side by side without the label (NON-BLOCKING, see §3) |
| P1-10 16.4 basis | RESOLVED | §2.4 l.504-507 |
| P1-11 184 GB / 183 GiB | RESOLVED | §1.2 row, §5.2 |
| P1-12 clone date | RESOLVED | §1.2 row, §2.4 (ii), §5.2 |
| Writer ordering (A3 P2 P1-66, §E) | RESOLVED | §1.5 l.261-267. "No writer-side ordering guarantee" removed. The new text claims name order only and marks adjacency unchecked. It is more conservative than audit 3 phase 2, which asserted adjacency. That is acceptable |

## 3. Consistency search (complete `20470b22` document)

| Term | Hits | Assessment |
|---|---|---|
| lift2 / digest / identity | l.24, 83-89, 250, 290, 296-297, 353-380, 788-796, 833, 855, 865 | All consistent with "declared bank identity established; not complete container identity or execution equivalence (system graphs differ; lift2 lacks `kda_gate_form`)". l.24 "the lift2 clone record" and l.833 are historical descriptions of rev 3 and audit 2. l.285/289/796 "digest" are about the GLM weights |
| "different model" | l.164 (cold-read probe, Granite: correct), l.610 (GLM/Kimi vs Gemma: correct), l.752 (forward question) | none about freq-mass |
| "never evict" / "never work" | 0 | — |
| "only one layer" / "one layer" | l.328, 349 (GLM §13 stage time and §2.1 scope: one layer per probe, consistent) | — |
| "not recoverable" | l.223 section heading only | — |
| "unexplained" | l.398 (quote of V3-N5, now qualified by V4-N1/N5), l.557 (~0.4 GB touched-set difference: separate claim, not in scope) | consistent |
| "unrun" | l.231 (quoted), l.629 (E2/E6/lyrw-v2) | consistent |
| "no writer-side" | 0 | — |
| "67 %" / "67%" | l.80, 554 (labelled derived), l.394 (V3-N3 quote), l.531 (§2.4 (iv), unlabelled "found"), l.609 ("59–67 % prompt reuse") | **l.531 residue of P1-9**. The same section's table labels it, so nothing is misstated about a conclusion: NON-BLOCKING |
| "184 GB" / "183 GiB" | l.167, 704 | consistent |
| "clone" | l.24, 167, 371, 523, 705, 799, 833 | all carry the 2026-09-18 dating where they bear on the rounds |
| "cross-model" / "same model" / "freq-mass" | l.92, 97, 102, 622, 778, 856 | consistent |
| "single-layer" / "virgin" | l.117-118, 188, 234, 287, 340, 707, 750, 800 | consistent |
| "16.4" | l.65, 391, 496, 504-507 | consistent |

Outside the diff, no material contradiction found. One chronological note: §8 "Revision 1 … the PR stays in draft" (l.818) is superseded by the audit-3 direction (l.869). It reads as history, not a contradiction.

## 4. Qualification check

- Class line l.3-6: RECONNAISSANCE; no arm run; no threshold; no cache, policy or runtime recommended. Present.
- l.45-46 and §2.0 heading l.276-279: every prior result is **BACKGROUND ONLY**, "none is a reproducible baseline". §2.6 l.616 repeats it. §6 l.746 repeats it.
- §6 l.736-737: "a candidate for review, not an authorised experiment … sets no thresholds, picks no winner and runs nothing".
- §8 l.852, l.873-874: not permission to run MEM-001A/MEM-001B; merging authorises no implementation or scored measurements. l.876: "MEM-001 is frozen only once …", so it remains unfrozen.
- Intact.

## 5. Informational notes (no action required)

- I1. The lift2 `index.json` hash is recorded truncated to 16 hex digits (`ddd9cad3e4603d12…`). The identity rests on a 64-bit prefix match with the full on-disk hashes. That is adequate, but it is not a full-hash comparison.
- I2. C7: "the system graph the historical runs used" assumes lift2's system graph was unchanged from 2026-09-05/06 to the 2026-09-20 record.
- I3. C18: V3-N3 gives no formula. "1 − 0.976/2.944" is the document's reconstruction, and it is labelled "not witnessed".
- I4. C25: the GLM figures combine §8.2.2l (1.15/1.50×, layer 3) with §8.2.2n (flat with depth). Accurate.
- N1 (NON-BLOCKING). §2.4 (iv) l.531, "V3 found about 67 %", is a residue of P1-9. Suggested wording: "V3-N3 derives about 67 % …".

## Verdict

**PASS.** There are zero unresolved blocking errors. All 26 changed claims are supported by their primary sources, and both BLOCKING findings are corrected in every location. P1-3..P1-12 and the writer-ordering claim are corrected. The lift2 framing is exactly "declared bank identity established, not complete container identity or execution equivalence". The writer-ordering text claims name order only. The reconnaissance-only, BACKGROUND ONLY and unfrozen classification is intact. One non-blocking residue remains: §2.4 (iv) l.531, the unlabelled "V3 found about 67 %".
