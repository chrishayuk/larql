# INSTRUMENT-1b — does the model's own head, read at the emergence write, say what the reader said?

Pre-registered 2026-09-20, before any lens readout on the bank exists. Same cases, same
prompts, same readers as INSTRUMENT-1a (`instrument-1-calibration.md`); **no new selection**.
Questions and forecasts below are FROZEN.

Programme: OBSERVE. Runs on main at `0197af44` (V3-OBS-1 #484, V3-STREAM-1 #485, `observe`
#486, V3-LENS-1 #487 all merged). Sits between INSTRUMENT-1a and HEAD-OBS-1
(`head-obs-1-per-head-observation.md`, frozen, not implemented).

---

## The question

INSTRUMENT-1a showed that the A/B/C reader on the carrier is the executor's head to 0.011
nats median at the exit, and that the counterfactual log-odds toward B enters the answer
position in one write, the L23 attention write on Gemma 3 4B (L35 on 12B), amplified at L29
(L41). The reader gives log-odds between *declared* tokens; it cannot say whether B is the
head's **answer** at that write (rank 1 over 262,208 tokens), how far A has fallen, or what
the head's top-1 is before the write. V3-LENS-1 (`v3-lens-1-logit-lens.md`) records
exactly that: the image's final norm and head applied to the layer output at an armed site,
a full log-softmax, and each declared token's log-probability and rank plus the top ids —
one head pass per armed site per position, priced on the receipt.

INSTRUMENT-1b points the lens at the sites INSTRUMENT-1a named and asks whether the
vocabulary trajectory independently supports the reader's story, and what it adds.

---

## Design (fixed)

**Subjects, prompts, arms, cases, readers**: INSTRUMENT-1a's, unchanged (bundle
`chris-experiments/larql/D_instrument1_edge1_calibration/{prompts,readers,readers12b}`).
Binary: `larql vindex3 observe` built release from main `0197af44` in
`.claude/worktrees/instrument-1b`; production CPU; default policy (Q8-realised head, as
LENS-1's LP7).

**Lens tokens**: the ids of `" A"`, `" B"`, `" C"` per case (the reader's rows' tokens);
`--lens-top-k 10`.

**Three arming levels, each priced on its receipt (`head_passes`) and by the CLI's
stepping wall:**

| level | arms | sites armed | purpose |
|---|---|---|---|
| **core** | all 48 4B arms | `--lens-layers 22,23,29 --lens-attention` → L22 att+ffn, L23 att+ffn, L29 att+ffn (6 passes per position) | rank and top-1 immediately before, at, and after the emergence write, and at the amplification write, in every arm |
| **profile** | g00 and g20, base + target (4 arms) | `--lens-layers all --lens-attention` (68 passes per position) | the whole-depth vocabulary trajectory on the witness case and the largest-S case |
| **12B** | g26, g03, g00, base + target (6 arms) | `--lens-layers 34,35,41 --lens-attention` (6 passes per position) | the same three questions on the sealed model |

Reader values come from the **same** 1b record (the record still carries every carrier
write's stats with the same basis), so lens and reader are read off one execution.

## Gates

- **B-G1 same execution as 1a.** For every arm, the 1b record's carrier stats at `p* = 88`
  (norm, delta norm, projection) are bit-equal to the 1a record's. Forecast: holds on 48/48
  (LENS-1's LP3 refactored the exit into one function and reported every gate
  bit-identical; if this fails it is a finding about main, not about the lens).
- **B-G2 anchor.** At L33 ffn (profile level) the readout's log-probabilities equal the exit
  distribution the CLI prints (LP2, bit for bit).
- **B-G3 receipt.** `head_passes` = armed sites × positions on every record; `lens_failure`
  absent; `complete`, zero dropped.

## Questions and forecasts (read on INSTRUMENT-1a's six reproducing cases; every arm is
recorded and reported)

- **B-Q1 lens = reader at intermediate depth.** At every armed site and arm,
  `|(logp_B − logp_A)_lens − Δ_reader|`. Forecast: ≤ 0.1 nats at every armed site of every
  arm, median ≤ 0.03 (the reader is the same head direction with f32 rows against the Q8
  head; INSTRUMENT-1a saw 0.051 max at the exit).
- **B-Q2 the answer flips at the emergence write.** Target arm: the head's top-1 at
  L22 ffn, L23 att, L23 ffn. Forecast: top-1 is `" B"` at **L23 att** in at least 4/6 and at
  L23 ffn in at least 5/6; at L22 ffn it is `" B"` in **0/6**. Base arm: top-1 `" A"` at all
  three sites in 6/6. This is the fact the reader cannot give: rank 1 over the whole
  vocabulary, not only over A.
- **B-Q3 where B was before it emerged.** Target arm, L22 ffn: rank of `" B"`. Forecast: rank
  between 2 and 200 in 5/6 (the reader put its log-odds against A at about −11 nats before
  the write, i.e. it is already a top-few-hundred token, not a random one); rank of `" A"`
  at L23 att after the write: ≥ 2 in 6/6.
- **B-Q4 controls do not flip mid-depth.** Irrelevant and wrongrel arms: top-1 at L23 att,
  L23 ffn and L29 att is `" A"` in 12/12 reproducing-case arms. A transient flip at an
  intermediate depth that the exit then undoes would be invisible to INSTRUMENT-1a and is
  what this question exists to catch.
- **B-Q5 the whole-depth profile (g00, g20).** Layer at which `" B"` first enters the top
  ten and first becomes top-1 in the target arm; the same for `" A"` in the base arm; the
  head's top-1 at every site before emergence (is it a node label, punctuation, or
  something else?). Reported against LENS-1's France reading (top ten at L23, top-1 at L24)
  as a comparison, not a replication. Forecast: `" B"` top-1 first at L23 att in both cases;
  the pre-emergence top-1 at the answer position is `" A"` from some layer before L23 in
  the target arm too (the base answer is readable before the edit is), and that layer is
  reported.
- **B-Q6 12B.** Target arm top-1 `" B"` at L35 att in 3/3 and `" A"` at L34 ffn in 3/3;
  B-Q1 bar holds on 12B.
- **B-Q7 price.** Stepping wall with and without the lens, per level, from the CLI; head
  passes from the receipt. Reported, not claimed (single runs under whatever load the
  machine has).

## What is not claimed

Nothing causal, nothing about heads (HEAD-OBS-1), nothing about the batch path or Metal.
"Top-1 at a site" is what the exit head would say if the network stopped there; it is a
reading, not a belief.

## Record-keeping

DB experiment (programme `larql`) with this document's sha256 before the first run;
records, receipts, readouts and the analysis table in the INSTRUMENT-1 bundle under
`lens4b/`, `lens4b-profile/`, `lens12b/`; results appended below.

## Verdict rule

Complete when B-G1–G3 pass, B-Q1–Q4 are answered on the six reproducing cases with every
arm recorded, B-Q5 on the two profile cases, B-Q6 on the three 12B cases, and B-Q7 is on
the record. Its product is the vocabulary-level confirmation or contradiction of
INSTRUMENT-1a's emergence claim and the ranks HEAD-OBS-1's witnesses will be read against.

---

## Results — core level, Gemma 3 4B (48 arms; recorded 2026-09-20 after the freeze)

Gates: **B-G1 HELD** — every carrier write's stats in every 1b record are bit-equal to the
1a record at all 89 positions (the execution on main `0197af44` is the execution of
`601c0ce5`; only the provenance fingerprint differs, the execution block is identical).
**B-G3 HELD** — `head_passes` = 534 = 6 sites × 89 positions on 48/48, no `lens_failure`,
complete, zero dropped. (B-G2 is read at the profile level below.)

### Gemma 3 4B core: head top-1 and label ranks at p* (lens), all arms

| case | arm | exit answer | L22 ffn: top-1 / A# / B# | L23 att: top-1 / A# / B# | L23 ffn: top-1 / A# / B# | L29 att: top-1 / A# / B# |
|---|---|---|---|---|---|---|
| g00 | base | ` RF` | ` ` / 3599 / 96171 | ` ` / 13 / 20041 | ` ` / 21 / 26551 | ` RF` / 1 / 14 |
| g00 | target | ` JF` | ` ` / 4148 / 91531 | ` ` / 296 / 135 | ` ` / 395 / 239 | ` JF` / 3 / 1 |
| g00 | irrelevant | ` RF` | ` ` / 3548 / 98713 | ` ` / 15 / 17730 | ` ` / 25 / 23753 | ` RF` / 1 / 17 |
| g00 | wrongrel | ` RF` | ` ` / 3669 / 97975 | ` ` / 12 / 22872 | ` ` / 22 / 30961 | ` RF` / 1 / 15 |
| g01 | base | ` WL` | ` ` / 218160 / 150435 | ` ` / 22833 / 154598 | ` ` / 25624 / 156023 | ` NH` / 27 / 2687 |
| g01 | target | ` DJ` | ` ` / 217160 / 146834 | ` ` / 208838 / 166830 | ` ` / 215394 / 166956 | ` NH` / 11826 / 2234 |
| g01 | irrelevant | ` KZ` | ` ` / 226726 / 155440 | ` K` / 4590 / 160908 | ` K` / 5431 / 165322 | ` NH` / 3 / 2374 |
| g01 | wrongrel | ` WL` | ` ` / 219606 / 148432 | ` ` / 7736 / 162568 | ` ` / 9076 / 166865 | ` NH` / 7 / 1045 |
| g02 | base | ` FB` | ` ` / 35869 / 18025 | ` ` / 5492 / 54 | ` ` / 5967 / 166 | ` FB` / 27 / 1 |
| g02 | target | ` FB` | ` ` / 37814 / 17804 | ` F` / 18575 / 14 | ` ` / 19853 / 82 | ` FB` / 245 / 1 |
| g02 | irrelevant | ` RP` | ` ` / 34116 / 17567 | ` ` / 124 / 1502 | ` ` / 207 / 3248 | ` RP` / 1 / 44 |
| g02 | wrongrel | ` FB` | ` ` / 34507 / 16440 | ` ` / 10797 / 28 | ` ` / 11579 / 128 | ` FB` / 52 / 1 |
| g03 | base | ` JL` | ` ` / 179465 / 34162 | ` ` / 47026 / 35872 | ` ` / 59543 / 31741 | ` JL` / 4 / 88 |
| g03 | target | ` JL` | ` ` / 178178 / 33095 | ` ` / 12325 / 10922 | ` ` / 16198 / 10537 | ` JL` / 4 / 2 |
| g03 | irrelevant | ` JL` | ` ` / 179917 / 34594 | ` ` / 51796 / 47214 | ` ` / 65080 / 41784 | ` JL` / 5 / 79 |
| g03 | wrongrel | ` JL` | ` ` / 177825 / 35759 | ` ` / 5638 / 14709 | ` ` / 9211 / 12854 | ` JL` / 2 / 35 |
| g12 | base | ` IH` | ` ` / 108921 / 207931 | ` ` / 232 / 194595 | ` I` / 474 / 198193 | ` IH` / 1 / 85 |
| g12 | target | ` KZ` | ` ` / 124478 / 204495 | ` K` / 98790 / 1172 | ` K` / 138244 / 2156 | ` KZ` / 756 / 1 |
| g12 | irrelevant | ` IH` | ` ` / 108804 / 207961 | ` ` / 187 / 204459 | ` I` / 396 / 208145 | ` IH` / 1 / 95 |
| g12 | wrongrel | ` IH` | ` ` / 107921 / 207774 | ` ` / 202 / 184570 | ` I` / 421 / 191482 | ` IH` / 1 / 76 |
| g13 | base | ` FN` | ` ` / 47809 / 7433 | ` F` / 27 / 4520 | ` ` / 100 / 8824 | ` FN` / 1 / 45 |
| g13 | target | ` SR` | ` ` / 57007 / 6991 | ` ` / 5677 / 116 | ` ` / 8109 / 189 | ` SR` / 23 / 1 |
| g13 | irrelevant | ` FN` | ` ` / 50511 / 7396 | ` F` / 39 / 2708 | ` ` / 121 / 5636 | ` FN` / 1 / 28 |
| g13 | wrongrel | ` FN` | ` ` / 50957 / 7455 | ` F` / 41 / 5860 | ` ` / 131 / 10505 | ` FN` / 1 / 57 |
| g14 | base | ` KL` | ` ` / 18888 / 6535 | ` K` / 71 / 3364 | ` K` / 111 / 3837 | ` KL` / 1 / 6 |
| g14 | target | ` LP` | ` ` / 19086 / 5969 | ` L` / 7956 / 3 | ` ` / 15802 / 7 | ` LP` / 352 / 1 |
| g14 | irrelevant | ` FE` | ` ` / 21036 / 6495 | ` ` / 4796 / 178 | ` ` / 6455 / 194 | ` FE` / 30 / 5 |
| g14 | wrongrel | ` KL` | ` ` / 22225 / 6802 | ` ` / 187 / 3330 | ` ` / 249 / 3496 | ` FE` / 2 / 6 |
| g15 | base | ` JV` | ` ` / 29669 / 7516 | ` ` / 372 / 7175 | ` ` / 286 / 7067 | ` RV` / 2 / 33 |
| g15 | target | ` QC` | ` ` / 39938 / 5368 | ` Q` / 27091 / 4 | ` Q` / 28263 / 4 | ` QC` / 171 / 1 |
| g15 | irrelevant | ` JH` | ` ` / 27941 / 6884 | ` ` / 686 / 13546 | ` ` / 552 / 12562 | ` JH` / 3 / 101 |
| g15 | wrongrel | ` JV` | ` ` / 29443 / 6984 | ` ` / 208 / 1631 | ` ` / 171 / 2106 | ` JV` / 1 / 13 |
| g20 | base | ` JC` | ` ` / 39369 / 6065 | ` ` / 7 / 5586 | ` ` / 14 / 6211 | ` JC` / 1 / 2784 |
| g20 | target | ` RO` | ` ` / 43696 / 5384 | ` R` / 51102 / 2 | ` R` / 63018 / 6 | ` RO` / 1042 / 1 |
| g20 | irrelevant | ` JC` | ` ` / 38437 / 5872 | ` ` / 5 / 6368 | ` ` / 11 / 7527 | ` JC` / 1 / 1867 |
| g20 | wrongrel | ` JC` | ` ` / 38978 / 6047 | ` ` / 7 / 6640 | ` ` / 14 / 7487 | ` JC` / 1 / 2450 |
| g22 | base | ` FR` | ` ` / 3772 / 97665 | ` F` / 2 / 87865 | ` F` / 7 / 80823 | ` FR` / 1 / 53 |
| g22 | target | ` VU` | ` ` / 4815 / 102075 | ` ` / 2054 / 174 | ` ` / 3144 / 240 | ` VU` / 29 / 1 |
| g22 | irrelevant | ` FR` | ` ` / 3503 / 90724 | ` F` / 2 / 73184 | ` F` / 7 / 65667 | ` FR` / 1 / 58 |
| g22 | wrongrel | ` FR` | ` ` / 3894 / 99106 | ` F` / 2 / 93821 | ` F` / 8 / 87350 | ` FR` / 1 / 44 |
| g24 | base | ` QT` | ` ` / 53499 / 101993 | ` Q` / 79 / 31129 | ` ` / 186 / 53448 | ` QT` / 1 / 7 |
| g24 | target | ` QT` | ` ` / 53134 / 97259 | ` ` / 413 / 8669 | ` ` / 371 / 12175 | ` NG` / 3 / 2 |
| g24 | irrelevant | ` QT` | ` ` / 54246 / 96213 | ` Q` / 98 / 31085 | ` ` / 185 / 53028 | ` QT` / 1 / 7 |
| g24 | wrongrel | ` QT` | ` ` / 53674 / 100683 | ` ` / 134 / 28970 | ` ` / 205 / 47608 | ` QT` / 1 / 6 |
| g26 | base | ` MF` | ` ` / 25204 / 86909 | ` ` / 240 / 111933 | ` ` / 217 / 159866 | ` MF` / 1 / 2083 |
| g26 | target | ` QH` | ` ` / 30986 / 79668 | ` Q` / 37172 / 121 | ` Q` / 50411 / 359 | ` QH` / 248 / 1 |
| g26 | irrelevant | ` MF` | ` ` / 28308 / 83830 | ` ` / 73 / 99109 | ` ` / 95 / 157235 | ` MF` / 1 / 4978 |
| g26 | wrongrel | ` MF` | ` ` / 23675 / 87808 | ` ` / 795 / 160809 | ` ` / 587 / 194657 | ` MF` / 1 / 1667 |

B-Q1: |lens log-odds − reader Δ| over 288 (arm, site) pairs: median 0.0167 nats, max 0.0384 nats.

B-Q2 (reproducing, n=6): target top-1 = B at L23 att: 0/6; at L23 ffn: 0/6; at L22 ffn: 0/6; at L29 att: 6/6. Base top-1 = A at L22 ffn/L23 att/L23 ffn/L29 att: 0/0/0/6 of 6.
B-Q3 (reproducing): rank of B at L22 ffn in the target arm: [91531, 204495, 6991, 5384, 102075, 79668] (in 2..200: 0/6); rank of A at L23 att: [296, 98790, 5677, 51102, 2054, 37172] (≥2: 6/6); B rank at L23 att: [135, 1172, 116, 2, 174, 121]; B beats A at L23 att: 6/6.
B-Q4 (reproducing controls, 36 arm-sites): top-1 = A in 12; top-1 = B (a transient flip) in 0 .
B-Q7: head passes per arm [534]; stepping examples: stepping: 20.541892459s over 89 positions (230.80778ms per position); stepping: 80.161736208s over 89 positions (900.693665ms per position).

### Forecast ledger (core)

| forecast | result |
|---|---|
| B-Q1 lens = reader ≤ 0.1 nats every site, median ≤ 0.03 | **HELD**: median 0.017, max 0.038 over 288 (arm, site) pairs. The reader is the head at intermediate depth as well as at the exit. |
| B-Q2 top-1 = B at L23 att in ≥ 4/6 | **FAILED: 0/6** at L23 att and 0/6 at L23 ffn; **6/6 at L29 att**. And the base arm's top-1 is A at L22–L23 in **0/6**: before L29 the head's answer is a whitespace token or a single first-letter token, in base and target alike. |
| B-Q3 B in rank 2–200 at L22 ffn | **FAILED: 0/6** — B sits at rank 5,384–204,495 before the write. After it (L23 att) B is at rank 2–1,172 and **beats A in 6/6**, A having fallen to 296–98,790. |
| B-Q4 no transient control flips | **HELD**: 36 control arm-sites, B top-1 in 0; A top-1 in 12 (all at L29). |
| B-Q7 price | 534 head passes per arm; stepping 20.5 s (unloaded) to 80–104 s (machine load 14–24) per 89 positions; single runs. |

### Reading (core)

1. **The L23 attention write is the A-versus-B decision, not the answer.** It moves B from
   deep in the vocabulary (median rank ~85,000) to the top few hundred and pushes A out by
   the same order, in one write, in every flipping case. The reader saw this as a 10–28 nat
   log-odds step; the lens confirms the same numbers and adds that neither label is the
   head's top-1 there.
2. **Output legibility arrives at L29.** The label token becomes rank 1 at the L29
   attention write in 6/6 target arms (B) and 6/6 base arms (A). INSTRUMENT-1a called L29
   the amplification; at the vocabulary level it is where the answer becomes the answer.
3. **Between L23 and L29 the head speaks in fragments.** Its top-1 at L23 ffn is a
   whitespace token or the *first letter* of the label as a single token (` K` for KZ,
   ` R` for RO, ` Q` for QH, ` F` for FR/FN, ` I` for IH) — in base and target arms. The
   two-letter label token is assembled as the head's answer only at L29. This was not
   pre-registered; it is recorded as an observation for HEAD-OBS-1's witness at L29.
4. **Controls never flip at any armed depth.** Whatever the wrong-relation edit does to
   p(B) in the tail, the head's top-1 stays A at L29 and a fragment before it.

## Results — profile level (g00, g20, base + target; 6,052 head passes per arm)

**B-G2 HELD**: the L33 ffn readout equals the exit distribution the CLI prints (` RF` −0.0013,
` JF` −10.9699 on g00 base), as LENS-1's LP2 says it must.

| arm | B first in top ten | B first top-1 | A first in top ten | A first top-1 | head's top-1 before the label |
|---|---|---|---|---|---|
| g00 base (` RF`) | L30 att | never | L24 ffn | **L26 att** (transient; stable from L29) | multilingual filler L0–L14 (`的`, `ɳ`, `ট`…), ` It`/` A`/` a` L15–L17, whitespace L18–L25, ` R` at L24 ffn |
| g00 target (` JF`) | **L26 att** | **L29 att** | L29 att | never | the same filler, whitespace L18–L28, `\n` at L28 ffn |
| g20 base (` JC`) | never | never | **L23 att** | **L29 att** | filler; whitespace L18–L24; ` J` at L25 ffn / L26 att; `\n` L26–L28 |
| g20 target (` RO`) | **L23 att** | **L25 ffn** | never | never | filler; whitespace L18–L22; ` R` L23 att – L25 att |

Read: the emergence write (L23 att) puts the answer label into the head's top ten in the
target arm of g20 and within three layers in g00; it becomes the head's top-1 two to six
layers later (L25 ffn, L29 att). The head's answer before that is not the other label: it
is whitespace, and in three of four arms the label's **first letter** as a single token
at some sites between L23 and L26 — the answer is being spelled before it is said.
Against LENS-1's France reading (` Paris` top ten at L23, top-1 at L24): same order,
label-completion later here.

## Results — Gemma 3 12B (g26, g03, g00; base + target; 534 head passes per arm)

### Gemma 3 12B: head top-1 and label ranks at p* (lens), all arms

| case | arm | exit answer | L34 ffn: top-1 / A# / B# | L35 att: top-1 / A# / B# | L35 ffn: top-1 / A# / B# | L41 att: top-1 / A# / B# |
|---|---|---|---|---|---|---|
| g26 | base | ` MF` | ` ` / 4650 / 33317 | ` ` / 42 / 38915 | ` ` / 32 / 22850 | ` MF` / 1 / 6176 |
| g26 | target | ` QH` | ` ` / 6386 / 23296 | ` ` / 6359 / 446 | ` ` / 4232 / 266 | ` QH` / 2361 / 1 |
| g03 | base | ` KJ` | ` ` / 52963 / 7653 | ` ` / 1214 / 1266 | ` ` / 1078 / 555 | ` kj` / 2 / 18 |
| g03 | target | ` KS` | ` ` / 56309 / 6933 | ` ` / 12748 / 74 | ` ` / 9081 / 44 | ` KS` / 47 / 1 |
| g00 | base | ` RF` | ` ` / 743 / 11118 | ` ` / 2 / 325 | ` ` / 2 / 204 | ` RF` / 1 / 15 |
| g00 | target | ` JF` | ` ` / 3378 / 10874 | ` ` / 180 / 117 | ` ` / 112 / 69 | ` JF` / 10 / 1 |

B-Q1: |lens log-odds − reader Δ| over 36 (arm, site) pairs: median 0.0181 nats, max 0.0478 nats.

B-Q2 (reproducing, n=3): target top-1 = B at L35 att: 0/3; at L35 ffn: 0/3; at L34 ffn: 0/3; at L41 att: 3/3. Base top-1 = A at L34 ffn/L35 att/L35 ffn/L41 att: 0/0/0/2 of 3.
B-Q3 (reproducing): rank of B at L34 ffn in the target arm: [23296, 6933, 10874] (in 2..200: 0/3); rank of A at L35 att: [6359, 12748, 180] (≥2: 3/3); B rank at L35 att: [446, 74, 117]; B beats A at L35 att: 3/3.
B-Q7: head passes per arm [534]; stepping examples: stepping: 50.02979075s over 89 positions (562.13248ms per position); stepping: 31.831415875s over 89 positions (357.656358ms per position).

**B-Q6**: forecast (B top-1 at L35 att in 3/3) **FAILED in the same way as 4B**: 0/3 at
L35, **3/3 at L41 att**; B beats A at L35 att in 3/3 (B rank 446/74/117, A 6,359/12,748/180)
with the head's top-1 a whitespace token. Base top-1 = A at L41 att in 2/3; the third
(g03) has the head saying ` kj`, the lowercase form, with ` KJ` at rank 2. B-Q1 on 12B:
median 0.018, max 0.048 nats over 36 pairs. The structure transfers: decision at the first
full-span write of the last third (L23/34, L35/48), legibility at the next (L29, L41).

## Forecast ledger (all levels)

| forecast | 4B core | 4B profile | 12B |
|---|---|---|---|
| B-G1 same execution | HELD 48/48 | HELD | HELD (same binary; not re-diffed against a 12B 1a record beyond p*) |
| B-G2 anchor | — | HELD | — |
| B-G3 receipt | HELD | HELD | HELD |
| B-Q1 lens = reader | HELD 0.017 / 0.038 | HELD | HELD 0.018 / 0.048 |
| B-Q2 top-1 = B at the emergence write | FAILED 0/6 (6/6 at L29) | FAILED | FAILED 0/3 (3/3 at L41) |
| B-Q3 B in rank 2–200 before | FAILED 0/6 (5k–204k) | — | FAILED 0/3 (7k–23k) |
| B-Q4 no transient control flips | HELD 0/36 | — | not run (base/target only) |
| B-Q5 profile | — | FAILED (L29 att, L25 ffn) | — |
| B-Q7 price | 534 passes; +≈5 s unloaded | 6,052 passes; 510–572 s under load | 534 passes; 32–50 s |

## Verdict

INSTRUMENT-1b is complete. The vocabulary trajectory **confirms** INSTRUMENT-1a's site
and **corrects** its word. The L23 attention write (L35 on 12B) is where the A-versus-B
decision is made: in one write B goes from deep in the vocabulary to the top few hundred
and A is pushed out, in every flipping case, with the reader's log-odds reproduced by the
true head to 0.02 nats. It is **not** where the answer becomes the answer: the head's top-1
there is whitespace, and the label token is the head's answer only at the L29 (L41)
attention write, two to six layers later, with the label's first letter often appearing as
a single-token top-1 in between. INSTRUMENT-1a called L29 "amplification"; at the
vocabulary level it is **legibility**. The reader could not have seen this (it compares
two declared tokens); the lens could not have priced the decision (it does not see the
controls' envelope); together they give HEAD-OBS-1 two distinct witness writes with
distinct meanings: L23 for *which label*, L29 for *saying it*.

Owed: nothing from this rung. HEAD-OBS-1's Witness A should be read at both writes with
these ranks beside it.
