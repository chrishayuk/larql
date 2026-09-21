# INSTRUMENT-1 — do three views of one model tell one story about a sealed behavioural effect?

Pre-registered 2026-09-20, before any run. Case selection, readers, questions and forecasts
below are FROZEN. Nothing here claims a result.

Programme: OBSERVE. First *experiment* on the observatory substrate (V3-OBS-1 #484,
V3-STREAM-1 #485, `vindex3 observe` #486), run at the observe verb's commit `601c0ce5`
(branch `instrument-1`). It sits beside, not under, the HEAD track in the peer session
(L23 country-corpus recomposition; not duplicated here) and precedes V3-LENS-1
(`v3-lens-1-logit-lens.md`), whose result will be INSTRUMENT-1b: the same cases, no new
selection, through the true lens.

---

## The question

We now have three instruments pointed at the same model:

| Instrument | What it records | Substrate |
|---|---|---|
| **LARQL `vindex3 observe`** | every carrier write on the canonical decode step: norm, delta norm, projection onto a declared basis, provenance, lossless receipt | VINDEX3 container, production CPU (bf16 representation, Q8 requantised projections and head under the default policy) |
| **Anatomist** (chuk-kv-anatomist UI over the `chuk-mcp-lazarus` MCP server) | layer × head direct logit attribution (DLA) to a target token, hot-cell ranking, head content projection onto token directions; also a final-norm logit lens (`track_race`) | HF checkpoint on MLX, loaded `float32` (the bf16 floor hazard in memory applies) |
| **Observatory** (`observatory/`) | Standard capture replay: execution topology, Residual Atlas, run comparison, provenance strip | the LARQL record, adapted through `larql.observatory.standard.v1` |

None of them has been pointed at a behavioural effect whose answer is already sealed.
EDGE-1 (`chris-experiments/sg_invariants/EDGE1_PLAN.md`, sealed 2026-09-19,
`results/edge1_full.json` sha256 `5ca4cf6b…9e7c`, verdict EDGE-SENSITIVE-NONSELECTIVE on
Gemma-3-12b-it) is such an effect, and PR #483 reproduced 12 of its arms through the
VINDEX3 executor at 0.001 nats median. INSTRUMENT-1 asks, with the answers fixed in
advance: **do the instruments locate the same computation, and where they disagree, which
one is wrong and why?**

The sealed bank fixes *outcomes* only (first-answer-token log-probabilities per arm). It
says nothing about depth or mechanism ("Not claimed by any branch: anything about internal
mechanism"). So no instrument is being checked against a sealed *depth*; the instruments
are checked against the sealed *outcome* at the exit and against **each other** in depth.

---

## Substrate and bank

**Calibration subject: Gemma 3 4B IT** — `~/chris-models/gemma3-4b-it.vindex3` (text-only
encode of HF snapshot `093f9f38…`, 34 layers, hidden 2560, no final logit softcap, tied
head), production CPU backend, one arm ≈ 23 s wall for the 89-token prompt (measured,
g00-base). The sealed bank is a 12B result; the 4B is the cheap iteration subject, so every
case first has to reproduce the sealed *behavioural pattern* on 4B (gate G3) before an
instrument question is asked of it.

**Transfer subject: Gemma 3 12B IT** — `~/chris-models/gemma3-12b-it.vindex3`, the sealed
model; four cases replayed (below) to see whether the instrument picture survives scale.

**Prompts**: the sealed EDGE-1 prompt text, byte for byte, from `edge1_full.json` `rows`
(raw text, `perm1.render`, ends in `Answer:`; 89 tokens with BOS under the container
tokenizer for every arm). Four arms per graph: base (truth A), target (edge rewritten to B,
truth B), irrelevant (frequency-matched edit elsewhere, truth A), wrongrel (start node's
other relation retargeted to B, truth A). Node labels are two capital letters and are
single tokens with a leading space (checked for all 32 graphs × 6 labels).

**Case selection (rule fixed before any instrument ran, from sealed 12B numbers only)**:
per tercile of the queried edge's position (early/mid/late), the flipped graph with the
largest target S, the smallest target S, and the median target S; plus the single sealed
target non-flip; plus the graph with the largest |S| under each control. Twelve graphs:

| case | tercile | A | B | C | target S (12B) | irrelevant S | wrongrel S | why |
|---|---|---|---|---|---|---|---|---|
| g00 | early | RF | JF | UM | 23.54 | −0.59 | 0.75 | early median; the #483 witness case |
| g01 | mid | KZ | EY | WL | 31.52 | 1.62 | 0.36 | mid max |
| g02 | late | RP | FB | EH | 24.54 | 0.35 | 1.40 | late min |
| g03 | early | KJ | KS | JL | 21.84 | 0.61 | **5.12** | largest wrong-relation shift |
| g12 | early | IH | KZ | MZ | 32.54 | 0.51 | 1.03 | early max |
| g13 | mid | FN | SR | GB | 25.07 | 2.52 | 3.46 | mid median |
| g14 | late | KL | LP | PX | 28.10 | **4.81** | 1.14 | largest irrelevant shift |
| g15 | early | JV | QC | RV | 22.12 | −1.07 | 1.66 | **sealed non-flip** (answered an unrelated node) |
| g20 | late | JC | RO | TG | 33.00 | 0.07 | 0.92 | late median |
| g22 | mid | FR | VU | JV | 17.78 | 0.19 | 2.23 | mid min |
| g24 | early | QT | JG | RG | 16.18 | −1.31 | 0.50 | early min (sealed minimum) |
| g26 | late | MF | QH | PT | 39.38 | 0.65 | −1.71 | late max (sealed maximum) |

**12B replay set**: g26 (strongest positive), g15 (sealed negative), g03 (strongest
control leak), g00 (median; already witnessed by #483). Sixteen arms.

---

## The readers (declared, hashed into every record)

The LARQL record projects every carrier write onto a caller-supplied basis
(`--basis-rows`, `supplied-rows-v1`, content hash on the record). For each case the basis
is three rows, in the order **A, B, C**:

```
row_X = embed_tokens[id(" X")] ⊙ (1 + final_norm.weight)      (f32, from the HF snapshot's bf16 bytes)
```

Gemma 3's final norm is `x · rsqrt(mean(x²)+ε) · (1+w)` and the head is the tied
embedding with no softcap, so for a carrier `x` with recorded L2 norm `n`:

```
z_X = projection_X · sqrt(hidden) / n  =  the pre-softmax logit of X if the head were applied to x
Δ   = z_B − z_A                           =  the head's log-odds of B over A at that write
```

exactly, under exact arithmetic. The reader is therefore *not* a raw probe: it is the
executor's own head direction with the norm's gain folded in, priced at one dot product
per row per write. Whether it agrees with the executor's real head at the exit (Q8
requantised, bf16 carrier) is question **Q1**, not an assumption. Reader name:
`edge1-ABC-normweighted-head-rows-v1`; rows generated by `make_readers.py` (scratchpad,
copied into the result bundle); token ids recorded in `readers/manifest.json`.

Anatomist's DLA uses the **raw** unembedding row (no norm gain) — `_per_head_dla`:
`(head_ctx @ Wo_hᵀ) · embed[target]`. Its `track_race` applies the real final norm then the
head (`_norm_project`) — a true logit lens on the MLX substrate. These differences are
declared here so a disagreement can be attributed to the reader definition rather than to
the model.

Observatory admits a recorded **3-D** basis only (`lib/standard.ts`: "Map currently
requires a recorded 3D basis"); the A/B/C reader is 3-D by construction, so one record per
arm serves both the trajectory analysis and the Observatory.

---

## Quantities (fixed definitions)

For a record of arm `a` of case `g`, at the final prompt position `p* = 88` (the token
`:` of `Answer:`, whose logits are the first-answer distribution), the 68 carrier writes
`k = 0..67` are `(layer l, site s)` with `k = 2l + {0: attention, 1: ffn}`.

- `Δ_a(k) = z_B − z_A` from the record's projection and norm, as above.
- `S_a(k) = Δ_a(k) − Δ_base(k)` — the in-record analogue of the sealed `S`.
- `E(k) = max(|S_irrelevant(k)|, |S_wrongrel(k)|)` — the control envelope: what a same-kind
  edit that does not change the reachable node does to the reader at that write.
- **First persistent divergence** `k*`: the smallest `k` such that for every `k' ≥ k`,
  `|S_target(k')| > E(k')` and `sign S_target(k') = sign S_target(67)`. Undefined if no
  such `k` exists (reported as such).
- **Half-rise** `k½`: the smallest `k` with `S_target(k) ≥ ½ · S_target(67)`.
- **Largest attention step** `l_jump`: the layer whose attention-site write has the largest
  single-write increase `S_target(2l) − S_target(2l−1)` (with `S_target(−1) := 0`).
- Exit values: `lpA, lpB` from the full-vocabulary log-softmax the CLI prints
  (`--top-k` covering the vocabulary), `S_exit = (lpB − lpA)_a − (lpB − lpA)_base`.

Anatomist, per arm: `batch_dla_scan(prompt, target=" B")` and `(…, target=" A")` over all
34 layers at position −1 → the layer × head DLA matrix and its hot cells; `(l_h, h_h)` = the
cell with the largest **positive** DLA toward B in the target arm; `extract_attention_output`
at that cell → top projected tokens; `track_race(prompt, [" A", " B", " C"], all layers)` →
`Δ_lens(l) = log p_lens(B) − log p_lens(A)` at each layer output.

---

## Gates

- **G1 tokenisation** — every record's header token ids equal the container tokenizer's
  encoding of the prompt file (89 ids, BOS first). Anatomist's `tokenize` of the same text
  gives the same ids (the HF tokenizer is the container tokenizer's source).
- **G2 record integrity** — every record: receipt `complete`, `live_dropped 0`, exactly
  89 × 68 = 6052 carrier writes, basis provider `supplied-rows-v1`, dims 3, the same basis
  hash across a case's four arms and a different hash across cases.
- **G3 behavioural reproduction on 4B** — a case is *reproducing* when its 4B greedy
  first token is A on base, B on target, A on irrelevant, A on wrongrel. Non-reproducing
  cases are still recorded and reported (they are the scale-transfer material) but
  instrument-agreement questions are asked only of reproducing cases. Forecast: ≥ 9/12
  reproduce.
- **G4 (= Q1) reader exit consistency** — see Q1; a reader that fails it is not a lens
  proxy, and Q2–Q6 are then reported under that caveat rather than as calibration.

---

## Pre-registered questions and forecasts

**Q1 — is the reader the head?** At `k = 67` (L33 ffn), `|Δ_a(67) − (lpB − lpA)_exit| ≤
0.5` nats on every arm. Forecast: holds on all 48 arms, median under 0.1 nats (the #483
exit witness put the whole Q8/bf16 pipeline at 0.001 nats median against f32 MLX).

**Q2 — where does the counterfactual first become readable?** For reproducing cases,
report `k*`, `k½`, `l_jump` and `S_target(67)`. Forecast: `k*` lies in layers 20–30 for at
least two thirds of reproducing cases (prior: ADDRESS-BUILD on Gemma-3-4B-IT, relation
readable ~L8, entity binding decisive L24–L28 — a prior, not a claim); `S_target(67)` is
within 2 nats of `S_exit` (a consequence of Q1); the sealed non-flip g15, if it also fails
to flip on 4B, has no defined `k*`.

**Q2b — does the reader agree with a true logit lens?** For reproducing cases, at each
layer's ffn site, `Δ_reader(2l+1)` against `Δ_lens(l)` from `track_race`. Bars: sign
agreement at every layer where both exceed 1 nat in magnitude; median `|Δ_reader − Δ_lens|`
over the last ten layers ≤ 1 nat. Forecast: passes on the last ten layers; early layers may
disagree by more (the substrates differ in representation and the carrier is far from the
head's regime there) and that disagreement is reported per layer, not averaged away.

**Q3 — does DLA implicate a writer where the record shows the state moving?** For
reproducing cases, `(l_h, h_h)` against `l_jump` and `k*`. Agreement bar: `|l_h − l_jump| ≤
2`. Forecast: agreement in ≥ 60% of reproducing cases. The rest are the finding: classify
each as (a) *early DLA* — `l_h < layer(k*) − 2` (a head attributed before the state
diverges: the attribution direction, which lacks the norm gain, disagrees with the head's
own reading); (b) *late DLA* — `l_h > layer(k½) + 2` (the state was already half-decided
before the "hot" head wrote: the hot head is an amplifier, not the decider); (c) neither.

**Q4 — does the hot head's content name the target?** `extract_attention_output` at
`(l_h, h_h)` in the target arm: is the top projected token `" B"`? In the base arm at the
cell with the largest positive DLA toward A: is it `" A"`? Forecast: yes in ≥ 70% of
reproducing cases for the target arm; when it is not B, report whether it is A, the start
node, another node, or a non-node token.

**Q5 — does Observatory admit and place it?** Every arm's record converts to a
`larql.observatory.standard.v1` manifest that passes `scripts/check-standard.mjs
--require-executor --require-complete` with 6052 writes; the manifest carries the record's
`run_provenance` verbatim. The compare view (`compareSamples` in `lib/record.ts`) is then
fed base and target for one reproducing case and the site of its largest projection
difference at `p*` is read from its output and checked against `k*`/`k½`. Forecast: the
preflight passes for 48/48; the compare's largest-difference site is at or after `k*`. What
the UI *shows* a person is a human check, listed as owed, not claimed here.

**Q6 — are any instruments confidently telling a different story?** The tally of Q3
classes (a)/(b)/(c), Q4 misses, and Q2b sign disagreements above 1 nat, each with the case
and site. Forecast: at least one case falls in Q3 class (b): DLA's hottest head is later
than the reader's half-rise, because DLA scores the *direct* write to the target direction
and cannot see indirect contributions routed through later FFNs. No forecast on class (a).

**Q7 — does the picture transfer to 12B?** For the four replay cases: exit values against
the sealed `lpA/lpB` (bar: 0.05 nats on the answer token, as #483), and `k*`, `l_jump`,
`(l_h, h_h)` reported beside the 4B values. Forecast: the sealed exit reproduces; the 12B
`k*` lands in the same *relative* depth band (last third) as 4B; no forecast on head
identity across scales (heads do not correspond across models).

---

## What is not claimed

Nothing causal: no intervention exists on this substrate yet (INTERVENE-1 is a later rung).
Nothing about "belief": reader values are head log-odds at a write, not probabilities the
model holds. Nothing about generality beyond this bank, this format, these two checkpoints.
A DLA hot head is an attribution, not a mechanism. The Lazarus logit lens is a different
substrate's reading and is used as a *foreign reference* for the reader (Q2b), not as truth.

## Out of scope

Per-head observation in LARQL (HEAD-OBS-1), the true lens in LARQL (V3-LENS-1 → 1b),
interventions, any new case selection after looking at an instrument, the Metal path, the
Observatory UI's own correctness (its preflight is used as a gate; its rendering is owed a
human look), and the MCP surface (a separate proposal).

## Record-keeping

Experiment registered in the chuk-experiments DB (programme `larql`) with this document's
sha256 in its design before the first run; the observe verb's commit `601c0ce5` and the
reader manifest are recorded there. Results: per-arm run records (JSONL, receipted),
per-arm full-vocabulary exit distributions, Anatomist tool responses (JSON, verbatim),
Observatory manifests and preflight outputs, an analysis table, and the Results section
appended below after the forecasts above are frozen. The 4B bank first, complete; the 12B
replay after, only on the four named cases.

## Verdict rule

INSTRUMENT-1a is complete when G1–G4 are recorded for all 48 arms, Q1–Q6 are answered for
every reproducing case, and Q7 for the four replay cases. It has no pass/fail: its product
is the disagreement ledger (Q6), which decides whether INSTRUMENT-1b (the true lens) needs
to answer a question or merely confirm one, and whether HEAD-OBS-1 (canonical per-head
evidence in LARQL) is earned by an Anatomist-versus-record disagreement.

---

## Amendments (recorded before any result was read)

- **2026-09-20, basis provider string.** The CLI records a supplied basis with provider
  `cli-supplied-rows` (its own constant), not the observer library's `supplied-rows-v1`.
  G2 accepts the CLI's string; the two spellings of one fact are noted for #486.
- **2026-09-20, Lazarus precision.** `load_model(dtype="float32")` loaded and computed in
  bfloat16: the loader's dtype is not applied to the weights (the E25-D note in
  `e26_common.py`). Every first-pass value sat on the bf16 grid (log-probabilities in
  eighths). Patched locally in `chuk-mcp-lazarus/src/chuk_mcp_lazarus/model_state.py`
  (`set_dtype(float32)` after load, uncommitted, owner's repo); the bf16 pass was discarded
  and every Anatomist number below is float32.
- **2026-09-20, Lazarus token resolution and rounding.** `batch_dla_scan` and `track_race`
  resolve a token string by trying it bare and with an extra leading space and keeping the
  variant with the higher final logit, so an unlikely candidate is silently replaced by the
  double-space token. Every response carries the resolved id; the analysis marks a scan
  whose target id is not the intended label as unreadable. `track_race` rounds
  probabilities to 1e-6, so Q2b compares only layers where both tokens have p ≥ 1e-5.

## Results — Gemma 3 4B (recorded 2026-09-20 after the freeze; 48 arms, all gates)

### Gemma 3 4B: gates and exit values

| case | G1 | G2 | answers b/t/i/w (4B) | reproducing | Q1 max \|reader−exit\| (nats) | S_exit target / irrelevant / wrongrel | sealed 12B S t / i / w |
|---|---|---|---|---|---|---|---|
| g00 | ✓ | ✓ | RF/JF/RF/RF | **yes** | 0.007 | 19.93 / -1.05 / -0.21 | 23.54 / -0.59 / 0.75 |
| g01 | ✓ | ✓ | WL/DJ/KZ/WL | no | 0.023 | 15.09 / -0.19 / 3.03 | 31.52 / 1.62 / 0.36 |
| g02 | ✓ | ✓ | FB/FB/RP/FB | no | 0.011 | 5.87 / -16.11 / 3.14 | 24.54 / 0.35 / 1.40 |
| g03 | ✓ | ✓ | JL/JL/JL/JL | no | 0.051 | 8.63 / 1.39 / 1.42 | 21.84 / 0.61 / 5.12 |
| g12 | ✓ | ✓ | IH/KZ/IH/IH | **yes** | 0.028 | 29.40 / -0.96 / 0.34 | 32.54 / 0.51 / 1.03 |
| g13 | ✓ | ✓ | FN/SR/FN/FN | **yes** | 0.008 | 20.69 / 2.40 / -1.32 | 25.07 / 2.52 / 3.46 |
| g14 | ✓ | ✓ | KL/LP/FE/KL | no | 0.009 | 19.16 / 10.79 / 3.64 | 28.10 / 4.81 / 1.14 |
| g15 | ✓ | ✓ | JV/QC/JH/JV | no | 0.016 | 17.09 / -1.62 / -0.61 | 22.12 / -1.07 / 1.66 |
| g20 | ✓ | ✓ | JC/RO/JC/JC | **yes** | 0.013 | 34.76 / 1.34 / 0.98 | 33.00 / 0.07 / 0.92 |
| g22 | ✓ | ✓ | FR/VU/FR/FR | **yes** | 0.040 | 16.70 / -0.33 / 0.45 | 17.78 / 0.19 / 2.23 |
| g24 | ✓ | ✓ | QT/QT/QT/QT | no | 0.017 | 0.44 / -0.79 / -0.56 | 16.18 / -1.31 / 0.50 |
| g26 | ✓ | ✓ | MF/QH/MF/MF | **yes** | 0.022 | 27.08 / -1.81 / 1.76 | 39.38 / 0.65 / -1.71 |

Q1 over all 48 arms: median 0.011 nats, max 0.051 nats. Reproducing: 6/12 (g00, g12, g13, g20, g22, g26).

### Gemma 3 4B: depth of the counterfactual (all cases; instrument questions are read on reproducing cases only)

| case | rep | k* (frozen) | k>1nat (exploratory) | k½ | l_jump (+nats) | S_target(last) | Emax | DLA hot →B (L,H,dla) | DLA layer-sum top-3 →B | Q3 agree / class | content top (margin) | Q2b n / med\|Δ\| / sign-dis | Q5 max-dist site |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g00 | **yes** | L15 ffn | L23 att | L23 att | L29 (+13.8) | 19.9 | 1.05 | L31 H7 11.3 | L23:4.0, L31:3.9, L24:2.0 | ✓ / late-DLA | ' JF' (0.56) | 11 / 0.14 / 0 | L29 att |
| g01 | no | L23 att | L23 att | L26 att | L29 (+6.0) | 15.1 | 4.44 | L25 H1 1.1 (target mis-resolved) | L25:4.1, L31:2.8, L4:0.8 | ✗ / neither | ' ' (3.46) | A unreadable | L29 att |
| g02 | no | — | — | L23 att | L29 (+5.7) | 5.9 | 30.39 | L31 H7 7.8 | L23:4.8, L25:3.2, L33:1.6 | ✓ / late-DLA | ' FP' (0.62) | A unreadable | L30 ffn |
| g03 | no | L29 att | L29 att | L29 att | L29 (+8.3) | 8.6 | 1.94 | L31 H7 8.9 | L24:2.8, L4:2.2, L33:2.0 | ✓ / neither | ' KJ' (0.80) | 5 / 0.17 / 0 | L30 att |
| g12 | **yes** | L17 att | L23 att | L23 att | L23 (+18.7) | 29.4 | 0.99 | L31 H7 10.1 | L23:4.8, L2:4.5, L24:2.5 | ✗ / late-DLA | ' KZ' (0.92) | A unreadable | L29 att |
| g13 | **yes** | L21 att | L23 att | L23 att | L29 (+21.9) | 20.7 | 2.40 | L31 H7 11.0 | L31:4.1, L23:2.2, L33:2.2 | ✓ / late-DLA | ' SR' (2.21) | 1 / 0.60 / 0 | L29 ffn |
| g14 | no | L23 att | L23 att | L23 att | L23 (+17.2) | 19.1 | 15.92 | L31 H7 8.9 | L23:4.6, L33:2.7, L24:1.8 | ✗ / late-DLA | ' LP' (0.81) | A unreadable | L29 ffn |
| g15 | no | L20 att | L23 att | L23 att | L23 (+16.8) | 17.1 | 1.62 | L31 H7 7.7 | L0:17.6, L23:5.4, L3:2.9 | ✗ / late-DLA | ' RV' (0.70) | 3 / 0.07 / 0 | L29 ffn |
| g20 | **yes** | L20 ffn | L23 att | L23 att | L23 (+27.6) | 34.8 | 1.67 | L31 H7 12.4 | L23:7.0, L31:2.6, L7:1.7 | ✗ / late-DLA | ' RO' (0.70) | A unreadable | L30 ffn |
| g22 | **yes** | L23 att | L23 att | L23 att | L23 (+19.5) | 16.7 | 0.91 | L31 H7 12.7 | L23:4.6, L31:3.3, L33:2.1 | ✗ / late-DLA | ' VU' (0.89) | 4 / 0.05 / 0 | L31 att |
| g24 | no | — | — | L20 ffn | L23 (+4.8) | 0.4 | 0.91 | L31 H7 9.9 | L31:4.1, L24:2.9, L33:2.5 | ✗ / late-DLA | ' JG' (1.11) | 5 / 0.30 / 0 | L29 att |
| g26 | **yes** | L20 ffn | L23 att | L23 att | L23 (+17.3) | 27.1 | 5.70 | L31 H7 8.7 | L0:11.9, L23:4.0, L4:3.4 | ✗ / late-DLA | ' QH' (0.09) | A unreadable | L29 att |

### Gemma 3 4B: S_target at the ffn site of layers 20..33 (nats; the whole depth is in the analysis JSON)

| case | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | L31 | L32 | L33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g00 | 0.2 | 0.5 | 0.3 | 11.9 | 12.8 | 14.7 | 15.1 | 13.5 | 10.9 | 24.0 | 19.1 | 16.7 | 15.4 | 19.9 |
| g01 | 0.2 | 0.0 | 0.0 | 5.5 | 5.2 | 7.3 | 7.4 | 6.6 | 5.8 | 10.7 | 11.2 | 9.6 | 8.9 | 15.1 |
| g02 | 0.1 | 0.1 | 0.1 | 3.5 | 4.3 | 4.9 | 5.2 | 4.2 | 3.1 | 8.4 | 8.8 | 6.0 | 5.1 | 5.9 |
| g03 | -0.1 | -0.2 | 0.0 | -0.5 | -0.1 | 0.1 | 0.9 | 0.6 | 0.6 | 8.2 | 7.9 | 7.7 | 6.9 | 8.6 |
| g12 | 0.5 | 0.7 | 0.5 | 19.6 | 19.6 | 21.9 | 27.7 | 23.1 | 20.8 | 34.6 | 26.5 | 22.8 | 22.0 | 29.4 |
| g13 | 0.0 | 0.4 | 0.4 | 13.1 | 16.3 | 17.6 | 19.8 | 15.1 | 16.5 | 37.5 | 28.7 | 26.0 | 22.9 | 20.7 |
| g14 | 0.2 | 0.2 | 0.2 | 17.7 | 20.3 | 22.7 | 19.7 | 16.4 | 14.9 | 26.5 | 23.4 | 18.1 | 15.6 | 19.1 |
| g15 | 0.4 | 0.9 | 1.0 | 19.7 | 18.8 | 20.4 | 22.2 | 20.7 | 16.7 | 27.8 | 17.2 | 14.1 | 11.3 | 17.1 |
| g20 | 0.1 | 0.3 | 0.3 | 25.4 | 25.7 | 31.2 | 32.8 | 30.8 | 26.9 | 49.6 | 46.8 | 39.2 | 36.4 | 34.8 |
| g22 | 0.2 | 0.3 | 0.1 | 18.6 | 21.0 | 23.2 | 22.7 | 17.3 | 16.3 | 29.2 | 27.7 | 23.6 | 18.6 | 16.7 |
| g24 | 0.2 | 0.2 | 0.1 | 4.0 | 4.1 | 4.6 | 4.4 | 3.9 | 3.7 | 4.8 | 2.5 | 1.2 | 1.1 | 0.4 |
| g26 | 0.1 | 0.5 | 0.5 | 19.3 | 21.1 | 21.1 | 26.5 | 23.6 | 22.5 | 35.2 | 27.0 | 26.5 | 21.3 | 27.1 |

Q2b over every arm whose A and B both resolved in `track_race` (22 of 48 arms; unreadable: 26): layers compared per arm 1–11 (probability ≥ 1e-5 on both tokens), median of per-arm median |Δ_reader − Δ_lens| over the last ten layers 0.133 nats (max 0.601), sign disagreements 0 of 91 checked.

Q5: compare's largest projection distance is at or after k* in 10/10 cases with a defined k*; its site is L29–L31 in every case.


### Forecast ledger (4B)

| forecast | result |
|---|---|
| G3 ≥ 9/12 reproduce | **FAILED: 6/12.** g01 base answers C; g02 base already answers B; g03 answers C (the wrong-relation node) in every arm; g14 and g15 answer a non-node under the irrelevant edit (g14's "frequency-matched" control moves 10.8 nats on 4B); g24 never flips (S 0.44 against a sealed 16.2). The 4B is not the sealed substrate; where it reproduces, the effect size is the sealed one (S 17–35 against 18–39). |
| Q1 reader = head, ≤ 0.5 nats on 48/48, median < 0.1 | **HELD**: median 0.011, max 0.051 nats. |
| Q2 k* in L20–30 for ≥ 2/3 of reproducing | **HELD, barely**: 4/6 (g00 L15, g12 L17 earlier — a ≤ 0.5-nat persistent separation before the control envelope is non-zero). Stronger and unforecast: the half-rise k½ is the **L23 attention write in 6/6 reproducing cases** (10/12 overall); the exploratory 1-nat marker is the same write in 10/12. S_target(last) = S_exit to 0.01 nats in every case. |
| Q2b reader vs MLX f32 lens: signs agree, median ≤ 1 nat | **HELD where readable**: 0 sign disagreements of 91 checked; per-arm medians 0.02–0.60 nats; but only 1–11 layers per arm are readable and A is unreadable in 6 target arms (rounding + resolution). |
| Q3 hot head within 2 layers of l_jump in ≥ 60% | **FAILED: 2/6.** The hottest head toward B is **L31 H7 in every case, reproducing or not** (and the hottest toward A in the base arm is the same L31 H7). It agrees only when l_jump happens to be L29 (g00, g13). |
| Q4 content names B in ≥ 70% | **HELD: 6/6**, by margins of 0.09–2.21 over other two-capital-letter tokens; the head's output is a label-class direction with a small target component (95% of its token-space energy needs ~193k of 262k directions). |
| Q5 48/48 preflight; compare site ≥ k* | **HELD: 48/48** complete, exact replay; compare's largest projection distance is at L29–L31 in every case, at or after k* in 10/10. It flags the amplification, not the emergence. |
| Q6 ≥ 1 late-DLA case | **HELD: 6/6 reproducing are late-DLA** (L31 > k½ = L23 + 2). |

### Reading (4B)

The three instruments tell one story about *where* and a different story about *who*.

1. **Where the counterfactual enters the answer reader.** In every case that flips, the
   log-odds shift toward B is ~0 through L22 and arrives in **one write, the L23
   attention write**, at 10–25 nats (half of its final value or more), then is amplified
   by the **L29 attention write** (+6 to +28 nats) and settles by L33. The record shows
   this; the reader is the executor's own head to 0.01 nats, so this is a statement about
   what the head would say, not about a probe.
2. **Anatomist agrees on the layer, not on the head.** The layer-summed DLA toward B peaks
   at **L23** in 4/6 reproducing target arms (g13: L31; g26: an L0 artefact, then L23),
   i.e. the attribution instrument sees the same site once its heads are summed. Its
   single hottest cell is **L31 H7 in 12/12 target arms and 12/12 base arms**: a head that
   adds ~8–13 nats of direct logit to whichever label is already the answer. DLA ranks
   the copier; the record shows the decider is six layers earlier.
3. **The head content instrument is weakly discriminating here.** L31 H7's output names
   the target in 6/6 flipping cases but by tiny margins over generic labels, and in the
   base arm the same head's top projection is A in only 6/12. "One head, one direction"
   is not what this bank shows; "one head copies a label-class direction" is.
4. **The Observatory compare view sees L29, not L23.** Its Euclidean projection distance
   is dominated by the late writes' magnitude; the emergence at L23 is a smaller vector
   step with a larger log-odds meaning (the reader's z divides by the carrier norm).
   Compare needs a reader-normalised difference beside the raw one.
5. **Early-layer DLA artefacts.** Raw unembedding-row DLA gives L0 heads scores of 12–18
   in two cases (g15, g26 layer sums) — the embedding-scale carrier dotted with the tied
   row. The reader does not see this because the norm gain and the RMS division are in
   its definition. This is the class-(a) hazard and (preview) it dominates on 12B.

### Disagreement ledger (Q6)

| # | instrument | says | record says | attributable to |
|---|---|---|---|---|
| 1 | Anatomist hot cell | L31 H7 is the writer of the answer | state decided at L23 attention; L31 adds direct logit late | DLA scores the direct write to the target row; indirect routing through L23–L28 FFNs is invisible to it |
| 2 | Anatomist layer-sum | L23 | L23 | agree (4/6) |
| 3 | Anatomist content | L31 H7 carries " B" | L31 H7 carries "a two-letter label", slightly B | projection onto raw rows of near-parallel label embeddings |
| 4 | Anatomist raw DLA at L0 | L0 H6 "hot" (g15, g26) | nothing readable at L0 | no norm gain / no RMS normalisation in DLA |
| 5 | Observatory compare | biggest move at L29–31 | first readable divergence at L23 | unnormalised projection distance |
| 6 | Lazarus `track_race` | A unreadable in 6 target arms; C in 20 arms | — | bare/space-prefixed max-logit resolution; 1e-6 rounding |
| 7 | Lazarus `load_model` | float32 | computed bf16 until patched | loader ignores dtype |

None of these is a disagreement about the model. Every one is a property of what the
instrument reads, and each was predictable from the definitions declared in the freeze.

## Results — Gemma 3 12B replay (g26, g15, g03, g00; 16 arms; recorded 2026-09-20)

### Gemma 3 12B replay: gates and exit values

| case | G1 | G2 | answers b/t/i/w (4B) | reproducing | Q1 max \|reader−exit\| (nats) | S_exit target / irrelevant / wrongrel | sealed 12B S t / i / w |
|---|---|---|---|---|---|---|---|
| g26 | ✓ | ✓ | MF/QH/MF/MF | **yes** | 0.027 | 39.40 / 0.71 / -1.72 | 39.38 / 0.65 / -1.71 |
| g15 | ✓ | ✓ | JV/JH/JV/JV | no | 0.028 | 21.66 / -1.15 / 1.65 | 22.12 / -1.07 / 1.66 |
| g03 | ✓ | ✓ | KJ/KS/KJ/KJ | **yes** | 0.012 | 21.81 / 0.62 / 5.19 | 21.84 / 0.61 / 5.12 |
| g00 | ✓ | ✓ | RF/JF/RF/RF | **yes** | 0.035 | 23.51 / -0.60 / 0.75 | 23.54 / -0.59 / 0.75 |

Q1 over all 16 arms: median 0.023 nats, max 0.035 nats. Reproducing: 3/4 (g26, g03, g00).

### Gemma 3 12B replay: depth of the counterfactual (all cases; instrument questions are read on reproducing cases only)

| case | rep | k* (frozen) | k>1nat (exploratory) | k½ | l_jump (+nats) | S_target(last) | Emax | DLA hot →B (L,H,dla) | DLA layer-sum top-3 →B | Q3 agree / class | content top (margin) | Q2b n / med\|Δ\| / sign-dis | Q5 max-dist site |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g26 | **yes** | L29 att | L29 att | L35 att | L41 (+45.4) | 39.4 | 1.74 | L2 H5 6.3 | L2:10.9, L3:5.6, L1:4.9 | ✗ / early-DLA | '猀' (1.44) | A unreadable | L45 att |
| g15 | no | L23 att | L29 att | L35 ffn | L41 (+21.7) | 21.7 | 1.64 | L0 H7 15.7 | L1:13.7, L46:3.3, L44:2.3 | ✗ / early-DLA | '.' (-19.97) | 8 / 0.78 / 0 | L45 att |
| g03 | **yes** | L26 att | L35 att | L41 att | L41 (+22.0) | 21.8 | 5.18 | L0 H4 19.8 | L1:16.9, L44:3.0, L45:2.9 | ✗ / early-DLA | '\n' (-32.51) | A unreadable | L45 att |
| g00 | **yes** | L29 att | L29 att | L35 att | L41 (+29.8) | 23.5 | 1.14 | L44 H10 6.8 | L1:9.7, L35:3.1, L41:2.8 | ✗ / late-DLA | ' JF' (0.93) | A unreadable | L47 att |

### Gemma 3 12B replay: S_target at the ffn site of layers 34..47 (nats; the whole depth is in the analysis JSON)

| case | L34 | L35 | L36 | L37 | L38 | L39 | L40 | L41 | L42 | L43 | L44 | L45 | L46 | L47 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g26 | 1.4 | 19.6 | 23.3 | 22.3 | 20.7 | 19.3 | 20.4 | 53.9 | 59.3 | 57.3 | 55.0 | 47.5 | 34.9 | 39.4 |
| g15 | 2.4 | 11.0 | 9.8 | 9.1 | 8.4 | 8.1 | 6.9 | 25.6 | 25.4 | 24.8 | 24.3 | 23.2 | 16.1 | 21.7 |
| g03 | 0.5 | 10.4 | 10.9 | 9.7 | 10.2 | 8.8 | 8.8 | 27.0 | 32.7 | 28.9 | 28.7 | 29.1 | 20.9 | 21.8 |
| g00 | 2.6 | 14.1 | 14.9 | 13.5 | 12.6 | 13.1 | 12.5 | 37.7 | 38.3 | 35.4 | 31.1 | 31.0 | 24.9 | 23.5 |

Q2b over every arm whose A and B both resolved in `track_race` (4 of 16 arms; unreadable: 12): layers compared per arm 1–8 (probability ≥ 1e-5 on both tokens), median of per-arm median |Δ_reader − Δ_lens| over the last ten layers 0.088 nats (max 0.783), sign disagreements 0 of 15 checked.

Q5: compare's largest projection distance is at or after k* in 4/4 cases with a defined k*; its site is L29–L31 in every case.


### Q7 ledger (12B)

| forecast | result |
|---|---|
| exit within 0.05 nats of sealed on the answer token | **HELD**: 16/16 answers equal the sealed ones (g15's target arm answers the unrelated node JH exactly as sealed); answer-token \|Δ log p\| median 0.00003 nats, max 0.025. Over all six node labels per arm: median 0.14, max 0.89 (g15 target, QC at −4 nats in a spread distribution; deep-tail tokens only). S_exit target 39.40 / 21.66 / 21.81 / 23.51 against sealed 39.38 / 22.12 / 21.84 / 23.54. |
| Q1 on 12B | **HELD**: max 0.035 nats over 16 arms. |
| k* in the last third | **FAILED by the frozen k\***: L29, L23, L26, L29 (a ≤ 3-nat persistent separation from L23–L34). **HELD by the half-rise**: S_target is ~0 through L34 and enters in one write at the **L35 attention write** (g26, g00; L35 ffn g15; L41 g03), then is amplified at the **L41 attention write** (l_jump = 41 in 4/4, +15 to +34 nats). Relative depths 35/48 = 0.73 and 41/48 = 0.85 against 4B's 23/34 = 0.68 and 29/34 = 0.85: the two-step structure transfers at matching relative depth. |
| Anatomist head identity | The hottest head toward B is an **early-layer artefact in 3/4** (L2 H5, L0 H7, L0 H4; raw DLA 6–20; content projections onto unrelated scripts and whitespace): class (a). Only g00 points at a late head, **L44 H10** (class (b), content " JF" by 0.93). The layer-summed DLA peaks at L1–L2 in 4/4 — the raw-row artefact dominates the sum at this scale. The 4B's universal copier L31 H7 has a 12B analogue, L44 H10, which is the hottest toward A in three base arms. |
| Observatory | compare's largest projection distance at L45–L47 attention in 4/4 (after the L41 amplification); preflight not re-run on 12B (records are prompt-only and identical in shape; owed). |
| Q2b | mostly unreadable on 12B (A mis-resolved in every target arm; B mis-resolved in 3 base arms); where readable, no sign disagreements. |

### Verdict

INSTRUMENT-1a is complete: G1–G4 on 48 + 16 arms, Q1–Q6 on the six reproducing 4B cases,
Q7 on the four replay cases. Its product is the disagreement ledger, and it decides the next
rungs as follows.

1. **The reader is the head.** `--basis-rows` with norm-weighted head rows reproduces the
   executor's exit log-odds to 0.011 nats median (0.051 max) on 4B and 0.035 max on 12B,
   through Q8 requantisation, at one dot product per row per write. INSTRUMENT-1b (the true
   lens, V3-LENS-1) is therefore not needed to *confirm* the log-odds trajectory; what it
   adds is the full-vocabulary fact the reader cannot give — rank and top-1 at the
   emergence write (L23 on 4B, L35 on 12B) — and it should be pointed exactly there.
2. **HEAD-OBS-1 is earned.** Anatomist and the record disagree on *who*: DLA's hottest
   head is the late copier (L31 H7 / L44 H10) or an early raw-row artefact; the record
   places the decision one attention write earlier than any hot head, at L23 / L35. Only
   canonical per-head evidence at that write can say which heads carried it.
3. **Two instrument corrections are owed before any further Anatomist comparison**: DLA
   and content projection must use the norm-gained, RMS-normalised head direction (the
   reader's), or its early-layer scores are meaningless; and the token-resolution heuristic
   and 1e-6 rounding must go (tools take ids and return log-probabilities).
4. **Observatory compare needs a reader-normalised difference** beside the projection
   distance; today it shows the amplification (L29 / L41), not the emergence.
5. **The 4B is a proxy for the depth structure, not for the behaviour**: 6/12 sealed cases
   reproduce on it, but every case that flips does so with the same two-write shape at the
   same relative depth as 12B.

### Owed

A human look at the 48 Standard manifests in the Observatory UI (Map / Trace / Compare on
g00 base vs target); the 12B preflight; the quiet-window timing of `observe` (no timing
claim is made here); committing this document and the Lazarus dtype patch (both owner
commits).
