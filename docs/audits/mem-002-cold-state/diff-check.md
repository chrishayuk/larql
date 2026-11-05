# PR #669 diff check: `git diff 1ba1739a 25822e8c -- docs/mem-002-cold-state-design.md`

Read-only. Sources read through git objects at `25822e8c` (no code changed in the range), `e2e0392f`, `51579782`, `aae20ae3`. One harmless query: `mdutil -s /System/Volumes/Data` gives "Indexing enabled." Line numbers below are the revised note's unless a file is named.

## 1. Changed or added claims

| Note line | Claim | Source check | Result |
|---|---|---|---|
| 42 | binding line `residency_curve.rs:195-199`, printed before execution | `println!` at :194, format string :195, args to :199; it sits before the pass loop (:211) | VERIFIED |
| 42 | `mapped_residency` counts only mapped regions, `prepared/residency.rs:417-425` | `fn mapped_residency` :417, `if mapped > 0` :421, `regions += 1` :424 | VERIFIED |
| 43 | pre-prompt read :229; prompt timed as one step :278-292; not printed; post-step print :262-273; `saturating_sub` :255, so the derived figure is a lower bound | :229 `let mut before_res = session.mapped_residency();`, :278 comment, :281 `Instant::now()`, :286-292 `observe`; :262-273 `println!`; :255 `resident_delta: ...saturating_sub(...)`. Lower-bound reasoning is right | VERIFIED |
| 52-54 | `"cold"` label :214-220 assigned by `pass == 1` | :214-220 label `if`/`else`; `pass == 1` at :216 | VERIFIED |
| 57-58 | binding and pre-prompt separated by the KV build and session constructor (:212-213) | :212 `select_for(...).build()`, :213 `DecodeSession::over_prepared` | VERIFIED |
| 66-68 | 59.1 % **witnessed** (round 2), ~67 % **derived** (V3-N3) | R :76-80, :556 | VERIFIED |
| 73-75 | routing capture starts only in the decode loop (:305); prompt fault count and delta recorded (:286-292) | :305 `routing_trace::start_capture()` inside `for step`; :286-292 `observe` on the prompt | VERIFIED |
| 98-103 | scoping ambiguities: option 4 "measured fault or read costs"; H1 "frozen routing trace" (§0, §5) | `mem-002-scoping.md` :18 (§0, H1), :194 (§5, option 4) | VERIFIED |
| 111 | C: evicted while a peer session ran, stopped once idle; frontier §14 "Eviction depends on memory pressure, not on advice" | `frontier-execution-programme.md` :1022-1025 | VERIFIED. The quote is now attributed to its source, as the review asked |
| 112 | D: load 5–13 **attributed** by the notes to `mds_stores` | `51579782:k3-prefetch-notes.json:9`; R :417 "which the notes attribute" | VERIFIED. Stated as attribution, not cause |
| 112 | indexing enabled on `/` and on the data volume | `mdutil -s` for both | VERIFIED (§9 still lists only `/`. INFORMATIONAL) |
| 112 | the branch displacer list includes `.s6` | `aae20ae3:scripts/evict-bank.sh` `DISPLACERS`; R :167 | VERIFIED |
| 116 | H: "post-boot indexing (mediaanalysisd ~80%, mds_stores ~33%)" in `e2e0392f:…k3-prefetch-2-notes.json` | that file :27 | VERIFIED. Source now precise |
| 130-132 | V3 second run: 10.485 GB resident at binding in a fresh process, left by earlier processes | R :402-406 ("after other processes had read the bank"); R :692 | VERIFIED |
| 135-136 | V3: 0.976 GB after 0.000 GB binding | R §2.2 (unchanged citation) | VERIFIED |
| 150 | `payload_sha256 3f12d102…`, `segment_sha256 e43600cd…` | R :369 | VERIFIED |
| 185-187 | round 3's demand ran at 8.2 µs per fault, the lowest | `e2e0392f:k3-prefetch-2-notes.json:54` ("from 16.4 us per fault to 8.2 us") | VERIFIED |
| 185-187 | shared-extent effects are one candidate explanation [R §2.4 (ii)] | R :523-526 | VERIFIED. Labelled "Unresolved hypothesis, not a finding". Minor: "whose displacement re-read extents shared with the subject" is stated without R's condition ("If `.s6` was then the APFS clone…"). It sits inside the hypothesis bullet, and the F row says the clone relation is assumed. NON-BLOCKING |
| 188 | "Rounds 1 and 3 used the same displacement script" | Round 3 names the committed script (`e2e0392f:…-2-notes.json:11`). Round 1 used a session scratch file of that name, committed later as a reconstruction (R :291, [A2 P1-17]) | **OVERSTATED**: the scripts' equality is asserted, not shown. This changes no conclusion, since the sentence only says the record does not isolate the explanation. NON-BLOCKING; suggest "the same named displacement script (round 1's is a later reconstruction)" |
| 318 | clones share extents, so free space is not consumed | consistent with §3's `df` check | CONSISTENT |

## 2. Disposition of review findings

| Review item | Disposition in revised note | Status |
|---|---|---|
| 1a pre-prompt read outside the timer; `saturating_sub` lower bound | §1.1 table (line 43), lines 59-61 | ADDRESSED |
| 1b cold arm = pass 1 of a fresh process; `"cold"` label is by procedure; "mapped" bound region | lines 51-54, 42 | ADDRESSED |
| 1c per-step witness is mapping-wide | §1.1 row 3, unchanged and correct | NO CHANGE NEEDED |
| 1d cold decode not witnessable on main | §1.1 row 3, lines 76-79 | NO CHANGE NEEDED |
| 1e one-position prompt has no routing fingerprint | lines 69-75 | ADDRESSED |
| 2a positive control: V3 10.485 GB; warm-at-binding control per method; faults × 16 KiB cross-check; per-region failure not excluded | §3 caveat 1 (lines 129-139) | ADDRESSED |
| 2b boundary double count | §3 caveat 2, unchanged | NO CHANGE NEEDED |
| 2c concurrent indexer warms pages after the witness; exclusion verified before the first arm | §4 lines 179-183 | ADDRESSED |
| 2d gate value seen with the outcome; gate on the binding line | lines 56-61; §3 line 122; §5 lines 226-228, 238-240 | ADDRESSED |
| 3a threshold, quantity and counting frozen with the gate | §5 lines 226-227 | ADDRESSED |
| 3b decision recorded before the timed region | §5 line 228 | ADDRESSED |
| 3c F/D counterbalanced | §5 lines 229-230; §4 line 199 | ADDRESSED |
| 3d rejected attempts timed or not, decided in advance | §5 lines 231-233 | ADDRESSED |
| 3e settling and quiet parameters fixed before arms, or on a separate calibration set | §5 lines 234-236 | ADDRESSED |
| 4a `mds_stores` as attribution; reads not expected to index; exclude displacers that share extents | §2 D (line 112); §6 Spotlight row (line 251) | ADDRESSED |
| 4b-i shared-extent residual tied to the 8.2 µs anomaly | §4 lines 185-189, as an unresolved hypothesis only | ADDRESSED |
| 4b-ii hash-then-cold ordering | §3 lines 148-152; §7 lines 276-280 | ADDRESSED |
| 4b-iii `cp -c` clone-versus-copy check | §3 lines 153-155, "on every arm" | ADDRESSED |
| 4b-iv mincore sees cached-but-unmapped pages (V3 control) | §3 lines 129-133 | ADDRESSED |
| 4b-v displacer order rests on a reconstructed script; `.s6` still cached at the end is not recorded | not in the bounded-changes list. The F row is unchanged and still presents the order without the reconstruction caveat | NOT ADOPTED, no stated reason. NON-BLOCKING (the review did not require it; the row already says "indirect" and "assumed") |
| 4c observer limits (`mdutil` shows enabled not active; `ps`/`top`; `iostat` aggregate; `vm_stat` system-wide; late refill; SSD thermal) | §6 table and lines 260-263 | ADDRESSED |
| 4d claim form 1 → "cold unit, observed side effects settled" | §6 lines 257-260 | ADDRESSED |
| 5 name the unit; §4 signal is not cold decode; fresh process, fresh clone | §3 line 122; §4 lines 163-167, 198; lines 51-52 | ADDRESSED |
| 6 `segment_sha256`; per-clone small-file hashes; hash order | §3 lines 148-157; §7 | ADDRESSED |
| 7 operational failure criterion set at freeze | §7 lines 293-296; §8 Q11 | ADDRESSED |
| 8 design-only | header; §4 line 195; §3 "chooses none" | NO CHANGE NEEDED |
| 9a retrospective / prospective / H2; feasibility ≠ H1; scoping ambiguities | §1.2 (lines 81-103); line 49; §8 Q10 | ADDRESSED |
| 9b consistent unit naming | §1.1 three units, §3, §4, §5 closing paragraph, §7, §8 Q2 | ADDRESSED |
| Change 1 | see 1b, 2d | ADDRESSED |
| Change 2 | see 2a | ADDRESSED |
| Change 3 (F: 8.2 µs link, hash separation, `df` check, Spotlight verified, small-file hashes) | §3, §4 | ADDRESSED (all five) |
| Change 4 (D: attribution, shared-extent exclusion) | §2 D | ADDRESSED |
| Change 5 (§5 rules) | §5 | ADDRESSED (all five) |
| Change 6 (claim form; operational failure criterion) | §6, §7 | ADDRESSED |
| Change 7 (name the unit in §3/§4; historical cold token not cold decode) | §3, §4 | ADDRESSED |
| Change 8 (retrospective vs prospective; scoping flags) | §1.2 | ADDRESSED |
| Change 9 (59 % witnessed / 67 % derived; "post-boot" source; C's "memory pressure") | line 66-68; §2 H; §2 C | ADDRESSED |

The informational citation notes from the review also land: "post-boot" is cited to `e2e0392f`, and C's "memory pressure" is cited to frontier §14. The data-volume `mdutil` result is in the D row, though not in §9.

## 3. Consistency of the complete revised note

- **Design-only.** No tooling, runs, thresholds or freeze. §3 caveat 3 says "This note chooses none", §7 sets no number, and §4 does not authorise the feasibility check. Holds.
- **Clone method is not accepted.** Line 15 says "lowest-disturbance candidate", the F row says "a candidate, not an accepted method", and §4 says "most likely". The old "best candidate" was removed. Holds.
- **The shared-extent explanation appears only once** (lines 185-189), labelled as an unresolved hypothesis. Elsewhere, shared extents appear only as an unobservable risk (lines 33, 114, 184, 253). Holds.
- **Binding is not equated with prefill.** Lines 56-61 and 238-240 agree. The §1.1 table's "yes, mapping-wide" for cold prefill could be read as gateable, but the same cell and the following text qualify it. Holds.
- **The hash ordering is consistent** between §3 and §7. §8 Q6 still asks "for F, how the clone inherits that check", which fits §3's answer (small-file hashes plus clone semantics). No conflict.
- **Spotlight.** §4 requires exclusion, if used, to be verified before the first arm, and §8 Q7 leaves the choice to exclude open. This is compatible, not a contradiction.
- **Minor, pre-existing (not in the diff):** §7's failure text says "cold decode is not establishable" even when the protocol's named unit is cold binding or prefill. It would read better as "the named cold unit is not establishable". INFORMATIONAL.

## Verdict

**PASS.** There are no blocking items. Every review finding and change item 1–9 is either addressed or, for the 4b-v caveat that the review did not require, left unchanged without harm. The changed claims check against their sources.

Non-blocking suggestions:

1. Line 188: "same displacement script" overstates. Round 1's script is a later reconstruction whose equality is asserted, not shown (R :291, [A2 P1-17]).
2. Line 186: carry R §2.4 (ii)'s condition ("if `.s6` was then a clone of `.s7`").
3. §9: add `mdutil -s /System/Volumes/Data`.
4. §7: replace "cold decode" in the failure text with the named unit.
