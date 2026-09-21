# HEAD-OBS-1 — which heads write the answer transition, and where do they read from?

Pre-registered 2026-09-20, before any per-head data exists on this executor. Laws, witness
sites, questions and forecasts below are FROZEN. Nothing here claims a result, and nothing
here claims causality.

Programme: OBSERVE. Rung above V3-OBS-1 (#484), V3-STREAM-1 (#485), the `observe` verb
(#486) and V3-LENS-1 (branch `lens-1`); the executor side of the Observatory's *Rich*
capture profile. Earned by INSTRUMENT-1a (`instrument-1-calibration.md`, DB
`instrument-1a-edge1-calibration`), whose disagreement ledger this rung exists to
adjudicate.

---

## The question

INSTRUMENT-1a established, on the sealed EDGE-1 bank, that the counterfactual log-odds
shift toward the rewritten answer enters the answer position's carrier in **one attention
write** — L23 on Gemma 3 4B, L35 on 12B — and is amplified at a second, L29 / L41. It also
established that Anatomist's hottest direct-logit-attribution head is not that write's
author: it is a late head (L31 H7 / L44 H10, henceforth the **late high-DLA head**, a
neutral name) whose direct contribution tracks whichever label is already favoured, and on
12B the raw-row attribution is dominated by layer 0–2 artefacts.

Two things are therefore unknown and cannot be learned from any existing instrument:

1. **Writer identity.** Within the emergence write, which query heads carry the reader
   delta, and how concentrated is it?
2. **Information source.** Where do those heads read from? L23 is the first *large*
   answer-position write; it is not necessarily where the relation was computed. Attention
   moves information across positions, so the write's content was assembled elsewhere and
   earlier. Only the heads' actual source distributions can say whether they read the
   edited relation-bearing line, the start node's mention in the question, or an
   already-resolved representation.

And a third, quieter one: INSTRUMENT-1a's frozen first-divergence marker fired *before*
the emergence write in most cases (a correctly signed, persistent, sub-nat to 3-nat
separation). That precursor is prior evidence now, and this rung asks whether it comes
from the same heads and sources as the emergence or from a different stage.

HEAD-OBS-1 answers these by **observation only**: per-head output contributions and
per-head source distributions, recorded as children of the attention writes V3-OBS-1
already records, under the same parity law.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The attention step returns `AttentionStepOut { key, value, output }` with `output` **post gate, post output-projection**; per-head context never surfaces. V3-OBS-1 said so and refused to infer it. | `opplan/exec/backend.rs` (`AttentionStepCall`/`AttentionStepOut`); `observe.rs` ("`o_proj` … never surfaces"); `v3-obs-1-carrier-observation.md` §"Per-head observation is a separate rung" |
| Production: `project_position` → `(q, k, v, gate)`, then `attend_position(call, position, q, keys, values, pre, gate)` → `output`. Reference and device have their own `attention_step`. | `opplan/exec/production.rs:1046`, `reference.rs:727`, `device.rs:549` |
| The call carries everything the bound operation needs: `num_q_heads`, `num_kv_heads`, `head_dim`, `w_o`, `qk_norm`, `query_scale`, `score_scale`, `logit_softcapping`, `span`, `window`, `gate`, `bias` (O bias after the projection), `sinks`. | `backend.rs` `AttentionCall` |
| After the step, decode applies the family's **post-attention norm** (`state.post_attention`, Gemma 3 has one; Granite/OLMo do not), then `scale_residual_delta`, then `leave_site` → the recorded `CarrierWriteRecord { delta, after, layer_scale }`. | `opplan/exec/decode.rs` (attention site, `leave_site`) |
| Gemma 3 4B: 8 query heads over 4 KV heads, head_dim 256, score scale 1/16, per-head QK-norm with weight offset 1, no softcap, no gate, no bias, no sinks; **full-span attention at layers 5, 11, 17, 23, 29**, sliding (window 1024) elsewhere. 12B: 16 over 8, full-span at 5, 11, 17, 23, 29, 35, 41, 47. | container `system_graph.json` (`components[0].attention[*].span`, `execution.attention`) |
| The observer contract: structural `StepEvent`s plus borrowed-record taps (`carrier_write`, `entering_carrier`, …); recorder assigns sequence; receipt hashes the log. | `observe.rs`, `larql-inference/src/vindex3/record.rs` |

A structural fact worth stating before any result: **every emergence and amplification
write INSTRUMENT-1a found is a full-span attention layer** (23, 29 on 4B; 35, 41 on 12B),
and the two largest precursors on 12B sit at 29 (full-span). The 89-token prompt lies inside
the 1024 window, so sliding layers see the whole prompt too; the difference is the position
encoding (global θ with the linear factor) and what those layers learned. This is read off
the plan, not the record, and it is a fact the witnesses below will be reported against,
not a forecast.

---

## Contract laws (frozen as properties)

Let one attention write at `(layer l, position p)` have query heads `h = 0..H_q−1`, each
bound to KV head `g(h) = h ÷ (H_q / H_kv)`, context `ctx_h ∈ ℝ^{head_dim}` (the softmax-
weighted value sum), and the executor's own output `o ∈ ℝ^{hidden}`.

**HL1 — head-sum law.** The observer computes, beside the executor's fused output and never
in place of it, per-head output contributions `c_h = W_o[:, h·d..(h+1)·d] · ctx_h` (the
family's gate, if any, applied where the plan applies it, per head when it is elementwise on
the concatenated context) and declares the once-only terms (O bias). Property: `‖Σ_h c_h +
bias − o‖ / ‖o‖ ≤ 1e−4` at every observed write, on every backend, with the measured value
recorded; bit identity is **not** required across accumulation orders and is not claimed.
The **children of the recorded write** are `c′_h = s · (1 + w_post) ⊙ c_h · layer_residual_scale`
where `s = 1/rms(o + bias)` is the post-attention norm's scalar for this write (`1` and no
gain when the family has no post-attention norm), so that `Σ_h c′_h + bias′ = delta` exactly
up to the same tolerance, where `delta` is the delta V3-OBS-1 already records. The reader-
projected form `Σ_h ⟨r, c′_h⟩ = ⟨r, delta⟩` follows by linearity and is checked too.

**HL2 — source law.** Each head record carries its actual post-softmax attention
distribution over the positions the bound operation attended: `0..=p` for full span, the
window's positions for a sliding span (positions outside the span are **absent**, never
zero), plus the sink mass when the plan declares sinks. `Σ probs + sink = 1` within 1e−6.
The record carries the KV head index `g(h)` the query head was bound to; for GQA/MQA the
distribution is the query head's, over the shared KV head's rows — the bound operation,
not a transformer-cartoon abstraction. No "source token" is inferred; positions map to
token ids through the record's own token list.

**HL3 — parity law.** With head observation armed on every site, logits at every position,
every recorded carrier write (`delta`, `after`, `layer_scale`), every structural event and
the run's provenance are bit-identical to the unobserved run on the reference and
production CPU backends. The observer reads `ctx_h` and the probabilities the executor
already computed and computes `c_h` separately; the executor's arithmetic is untouched.
Observation is subscription, never a second executor.

**HL4 — accounting law.** The site-level attention write stays authoritative. Head
observations are children keyed to `(run, position, layer, site = attention)`; a record
without them is still a complete V3-OBS-1 record. A backend or path that cannot provide the
decomposition (the Metal whole-stack path, the batch prefill path) records
`head_observation: refused { reason }` for that write. Nothing reconstructs a head
decomposition from `delta` after the fact.

**HL5 — capture is declared and priced.** Two levels, chosen per run: *stats* at every
attention write (per head: `‖c′_h‖`, its projection on the run's basis, `g(h)`, the top-`k`
source positions with their probabilities and the sink mass, `k` declared) and *full* at
armed sites only (the whole source distribution and, on request, `c′_h` itself). The cost
class is one `W_o` slice product per head per observed write — the same multiply-adds as
one `o_proj` — plus the probabilities the executor already holds. Measured, not described:
token wall with stats-level head observation on every site minus `NoopObserver`, release,
production CPU, Granite 4.2 3B and Gemma 3 4B; per token and per write. Not a performance
claim.

**HL6 — on the record.** Head records are events with their own sequence, replay equal,
receipt covering them; the Standard adapter's "attention source capture unavailable" flag
turns into a declared per-record capability rather than a constant.

---

## Frozen acceptance properties (engineering)

- **HP1** HL1 on the golden plan (both backends), on Granite 4.2 3B (no post-attention norm:
  the `s = 1` branch), on Gemma 3 4B (post-attention norm branch), every write of an 8-token
  run; the maximum relative residual reported per subject.
- **HP2** HL2: sums within 1e−6 at every write; sliding-span writes on Gemma 3 at a position
  beyond the window (a 1100-token synthetic prompt on the golden plan's geometry, or a real
  one if cheap) carry exactly the window's positions.
- **HP3** HL3 bit-parity, both CPU backends, golden and Gemma 3 4B.
- **HP4** HL4: the device path refuses with a reason; a record with refusals still replays.
- **HP5** HL5 measured on both real subjects.
- **HP6** HL6 replay identity and receipt coverage; the run record schema gains one event
  kind (`head_write`) and one refusal spelling, nothing else (forecast; if more is needed it
  is recorded as a finding).
- **Unwitnessed by this rung, declared:** families with an attention output gate (Qwen 3.8
  is on disk but large; not run), with O bias, with sinks (no container on disk) — the law's
  terms for them are written, not exercised.

---

## Scientific witnesses (predeclared; no case is selected after looking at head data)

Prompts, readers and arms are INSTRUMENT-1a's, unchanged (`chris-experiments/larql/
D_instrument1_edge1_calibration/{prompts,readers,readers12b}`). Per-head reader
contribution at a write is `⟨r_B − r_A, c′_h⟩ / rms(after)` in the same units as
INSTRUMENT-1a's `Δ`; the per-head **counterfactual contribution** is its target-arm value
minus its base-arm value, in the units of `S`. The site's own `S` step at that write is the
denominator for every "fraction" below and is already on record:

| subject | write | cases (S step at the write, target − base, nats) |
|---|---|---|
| 4B | **L23 attention (emergence)** | g00 11.6, g12 18.7, g13 13.0, g20 27.6, g22 19.5, g26 17.3 (the six reproducing); g14 17.2, g15 16.8 (target flips, base correct, a control arm failed G3 — secondary) |
| 4B | **L29 attention (amplification)** | g00 13.8, g12 16.2, g13 21.9, g20 24.6, g22 13.5, g26 14.2 |
| 4B | L31 attention (the late high-DLA head's site) | all six |
| 12B | **L35 attention (emergence)** | g26 18.3, g03 10.2, g00 11.2 |
| 12B | **L41 attention (amplification)** | g26 45.4, g03 22.0, g00 29.8 |
| 12B | L44 attention (late high-DLA head's site) | the three |

**Witness A — emergence.** For every case and both subjects, at the emergence write and
the amplification write, record per head: `‖c′_h‖`, counterfactual contribution, fraction
of the site's `S` step, `g(h)`, the full source distribution in base and target arms, and
the top five source positions with their tokens. Then, per write:

- **A1 concentration.** Fraction of the site's `S` step carried by the top head, and the
  number of heads needed to reach 90% of it.
- **A2 source.** For the top head in the target arm: attention mass on (i) the rewritten
  label token itself, (ii) the rest of the edited line, (iii) the start node's mention in
  the question line, (iv) the query's relation word in the question, (v) the other edge
  lines, (vi) BOS. These six classes are fixed now from the prompt structure; a position
  belongs to exactly one.
- **A3 identity across arms.** Is the top head toward B in the target arm the same head as
  the top head toward A in the base arm at that write? Same across the six cases?
- **A4 the late high-DLA head.** Its counterfactual contribution and source classes at its
  own site, reported beside the emergence heads, under its neutral name.

**Witness B — precursor.** Prior evidence from INSTRUMENT-1a's frozen `k*` (target arm,
first persistent divergence outside the control envelope): 4B g12 **L17 attention** (full
span, S 0.15), g13 L21 attention (0.32), g00 L15 ffn / g20 L20 ffn / g26 L20 ffn (the
attention write of the same layer is the witness, and the ffn write's own reader delta is
reported beside it), g22 none before L23; 12B g26 and g00 **L29 attention** (full span,
S 1.8 and 3.0, the L29 attention steps being 1.8 and 2.9 nats), g03 L26 attention (0.14).
For each, the same per-head record as Witness A. Question **B1**: does the precursor's
counterfactual contribution sit in the same head index and read from the same source
classes as the emergence head(s) of that case, or in different heads and classes? **No
story is pre-registered as the answer.** The two candidates the source law discriminates
are written down so the reading cannot drift: (a) the emergence head copies the rewritten
label from the edited line (class i/ii dominant), the precursor being an earlier, weaker
read of the same line; (b) the relation was resolved earlier at the start node's position
and the emergence head reads that resolved state from the question line (class iii
dominant), the precursor being the resolution stage leaking into the answer position.
Either, both in different cases, or neither is an admissible result.

---

## Forecasts (falsifiable; deliberately few)

- **HF1** HL1 residual ≤ 1e−5 relative on both CPU backends (f32 accumulation order only).
- **HF2** HL5 stats-level cost on Gemma 3 4B ≤ 15% of token wall (one extra `o_proj`-class
  product per attention write against a forward dominated by FFN and the head).
- **HF3 (A1)** the emergence write's `S` step is carried by **at most two heads for 90%** in
  at least 4 of the 6 reproducing 4B cases. Prior: weak; DLA's layer-sum at L23 was small
  relative to the reader's step, which is compatible with either one dominant head whose
  contribution DLA under-reads or several heads.
- **HF4 (A2)** the top head's source mass is **concentrated**: its top three positions hold
  at least half of the non-sink mass in at least 4 of 6 cases. No forecast on *which*
  class; that is the question.
- **HF5 (A3)** the top head is the same head index in base and target arms in at least 4
  of 6 cases (a label-reading head, not a B-specific one). No forecast across scale.
- **HF6 (B1)** no forecast.

---

## What is not claimed

Nothing causal. A head carrying 90% of the emergence step while attending to the rewritten
label is a **localised observed contribution and source pattern**, not necessity and not
sufficiency; a head with a large contribution may be compensated, and a head with a small
one may be required. The words *decider*, *amplifier*, *transporter/copier* and
*redundant contributor* are reserved for the intervention rung that can distinguish them
(zero the candidate emergence head; zero the late high-DLA head; compare the effect on
emergence and on final amplification), and are not used for any head in this rung's
results. Nothing about layers whose attention this rung does not observe (the Metal path,
batch prefill). Nothing about Gemma 3 beyond this bank and format.

## Out of scope

Interventions of any kind; the Metal whole-stack path (refusal only); the batch prefill
path; per-neuron FFN decomposition; the Observatory UI's Rich rendering (it consumes the
record; owed separately); any change to V3-OBS-1's carrier record or V3-LENS-1's lens
events beyond adding the head event kind.

## Record-keeping

Registered in the chuk-experiments DB (programme `larql`) with this document's sha256
before implementation; results appended below the forecasts after they are frozen;
per-case per-head tables and source distributions in the INSTRUMENT-1 bundle directory
under `head-obs-1/`. Order: golden and Granite engineering witnesses → Gemma 3 4B
engineering witnesses → Witness A on 4B → Witness B on 4B → 12B replay of both.

## Verdict rule

HEAD-OBS-1 is complete when HP1–HP6 pass, Witness A is recorded for the six reproducing 4B
cases at both writes and for the three 12B cases, Witness B is recorded for every case with
a defined precursor, and every head in the results is named by index and site only. It
unlocks INTERVENE-1 with a *named* candidate set: the emergence heads, the precursor heads
and the late high-DLA head, each with a recorded source pattern to be tested rather than
a role to be assumed.

---

## Results (recorded 2026-09-20, after the forecasts above were frozen)

**Execution:** `larql vindex3 observe --heads --heads-top-k 89 --basis-rows <case reader>` at
head-obs-1 `58eb342df5d6b80b51bfbaff860a779f814f9ea2` (the engineering freeze
`v3-head-obs-1-per-head-observation.md` plus the implementation, 18 gated paths; the owner's
in-process witness driver and its bench outputs are a separate commit above it), release,
production CPU, **default policy** (Q8-realised projections, the policy of every INSTRUMENT-1a
and 1b record). The implementation provides the *stats* level only; with `k` = 89 every head
row carries its whole source distribution, so nothing Witness A/B needs was missing. No
refusal arose (CPU path only). Arms: 4B primary g00 g12 g13 g20 g22 g26 and secondary g14
g15, base and target; 12B g26 g03 g00, base and target.

**Gates.** First record (g00 target) and then all 22 records passed the schema gate
(the gate script validate_heads_record.py in the bundle's scripts directory): field identities, head order, KV-head
mapping, source positions within the query, sources plus sink summing to one within 1e−7,
basis hash equal to the case's 1a reader, one `head_sum` and H `head_write` rows per attention
write, head-sum residual ≤ 1e−4, and the reader-projected reconstruction of every site's step
from its head rows within 5.1e−6 of the carrier's recorded step (layer 0 excluded: the record
does not carry the embedding's projection, so that layer is covered by the vector residual
only). Every record's carrier writes are **bit-equal to the 1a record at all 89 positions**
and its provenance fingerprint equals the 1b record's. In reader units, Σ heads equals the
carrier's S step to 0.01 nats at every witnessed write on both models (tables below).

### Forecast ledger

| forecast | result |
|---|---|
| HF1 HL1 residual ≤ 1e−5 | **HELD**: worst 1.03e−6 over 16 4B records (24,208 rows each), 6.6e−7 over 6 12B records (68,352 rows each). |
| HF2 stats-level cost ≤ 15% of token wall on Gemma 3 4B | **FAILED — original bf16 cost arm +34%.** *Additional engineering measurement (not an HF2 re-test):* under the default Q8 policy +9% on Gemma 3 4B (8 heads) and +289% on Granite 4.2 3B s6 (40 heads), carrier stats and provenance invariant in both; the tap itself is ≈ 0 and the cost is the padded per-head projection, which scales with head count and is the next engineering hypothesis. HP5/HP7 figures are the owner's driver's; the default-policy parity and cost figures are the HEAD-OBS-1 session's. |
| HF3 ≤ 2 heads for 90% of the L23 step in ≥ 4/6 | **HELD 6/6** (one head suffices in 5, two in g20). Secondary g14, g15: one head. |
| HF4 top-3 sources ≥ 50% of the top head's mass in ≥ 4/6 | **HELD 6/6** (0.63–0.91). |
| HF5 same top head base/target in ≥ 4/6 | **HELD 6/6**: H3 toward A in every base arm and toward B in every target arm. |
| HF6 (B1) | no forecast; answered below. |

### Gemma 3 4B: Witness A at each write (per case; target − base per head, reader units = nats of the site's S step)

| case | write | S step (carrier) | Σ heads | top head (fraction) | heads to 90% | cf by head | A3 same head base/target |
|---|---|---|---|---|---|---|---|
| g00 | emergence L23 | +11.66 | +11.66 | H3 (1.11) | 1 | H0:+0.1 H1:+0.2 H2:-2.3 H3:+12.9 H4:-0.0 H5:-0.0 H6:+0.4 H7:+0.5 | H3 / H3 same |
| g00 | amplification L29 | +14.12 | +14.12 | H4 (1.02) | 1 | H0:-0.0 H1:+0.1 H2:+0.0 H3:-0.3 H4:+14.3 H5:-0.0 H6:-0.0 H7:-0.0 | H4 / H4 same |
| g00 | late_high_dla L31 | -1.37 | -1.37 | H7 (-1.28) | 8 | H0:-0.0 H1:+0.0 H2:-0.0 H3:+0.1 H4:-0.0 H5:-0.1 H6:-3.1 H7:+1.8 | H7 / H7 same |
| g00 | precursor L15 | +0.01 | +0.01 | H5 (1.20) | 1 | H0:-0.0 H1:+0.0 H2:+0.0 H3:+0.0 H4:-0.0 H5:+0.0 H6:+0.0 H7:-0.0 | H5 / H1 differ |
| g12 | emergence L23 | +18.67 | +18.67 | H3 (0.96) | 1 | H0:+0.3 H1:+0.6 H2:-0.4 H3:+17.9 H4:-0.0 H5:+0.0 H6:-0.3 H7:+0.5 | H3 / H3 same |
| g12 | amplification L29 | +16.83 | +16.83 | H4 (1.03) | 1 | H0:-0.0 H1:+0.1 H2:-0.3 H3:+0.1 H4:+17.4 H5:-0.5 H6:+0.2 H7:+0.0 | H4 / H4 same |
| g12 | late_high_dla L31 | -2.18 | -2.18 | H7 (-1.34) | 8 | H0:-0.0 H1:-0.0 H2:-0.0 H3:-0.0 H4:-0.0 H5:+0.1 H6:-5.2 H7:+2.9 | H7 / H1 differ |
| g12 | precursor L17 | +0.12 | +0.12 | H0 (0.78) | 2 | H0:+0.1 H1:+0.0 H2:-0.0 H3:+0.0 H4:-0.0 H5:+0.0 H6:+0.0 H7:-0.0 | H2 / H6 differ |
| g13 | emergence L23 | +13.00 | +13.00 | H3 (1.02) | 1 | H0:+0.2 H1:+0.5 H2:-3.4 H3:+13.3 H4:+0.0 H5:+0.0 H6:+1.4 H7:+1.1 | H3 / H3 same |
| g13 | amplification L29 | +22.51 | +22.51 | H4 (1.03) | 1 | H0:-0.0 H1:+0.0 H2:+0.5 H3:-0.4 H4:+23.2 H5:-0.8 H6:+0.0 H7:+0.0 | H4 / H4 same |
| g13 | late_high_dla L31 | +2.19 | +2.19 | H7 (2.97) | 1 | H0:+0.0 H1:+0.0 H2:+0.0 H3:+0.2 H4:-0.0 H5:-0.0 H6:-4.5 H7:+6.5 | H7 / H7 same |
| g13 | precursor L21 | +0.29 | +0.29 | H6 (1.00) | 1 | H0:+0.1 H1:+0.1 H2:+0.0 H3:+0.0 H4:-0.0 H5:+0.0 H6:+0.3 H7:-0.2 | H7 / H6 differ |
| g20 | emergence L23 | +27.60 | +27.60 | H3 (0.89) | 2 | H0:+0.7 H1:+0.3 H2:-3.1 H3:+24.5 H4:+0.0 H5:+0.0 H6:+2.0 H7:+3.2 | H3 / H3 same |
| g20 | amplification L29 | +25.31 | +25.31 | H4 (0.99) | 1 | H0:-0.0 H1:-0.0 H2:-0.2 H3:-0.2 H4:+25.2 H5:+0.6 H6:+0.0 H7:-0.0 | H4 / H4 same |
| g20 | late_high_dla L31 | -2.54 | -2.54 | H7 (-3.32) | 8 | H0:+0.0 H1:+0.0 H2:-0.0 H3:+0.1 H4:-0.1 H5:+0.1 H6:-11.0 H7:+8.4 | H7 / H7 same |
| g20 | precursor L20 | +0.08 | +0.08 | H1 (1.45) | 1 | H0:+0.0 H1:+0.1 H2:-0.0 H3:-0.0 H4:-0.0 H5:-0.0 H6:+0.0 H7:-0.1 | H6 / H7 differ |
| g22 | emergence L23 | +19.45 | +19.45 | H3 (0.92) | 1 | H0:+0.1 H1:+0.3 H2:-1.8 H3:+17.9 H4:-0.0 H5:+0.0 H6:+1.3 H7:+1.7 | H3 / H3 same |
| g22 | amplification L29 | +14.06 | +14.06 | H4 (0.96) | 1 | H0:-0.0 H1:+0.0 H2:+1.6 H3:-0.3 H4:+13.5 H5:-0.7 H6:-0.0 H7:+0.0 | H4 / H2 differ |
| g22 | late_high_dla L31 | +0.90 | +0.90 | H7 (6.53) | 1 | H0:+0.0 H1:+0.0 H2:-0.1 H3:+0.2 H4:-0.1 H5:+0.1 H6:-5.2 H7:+5.9 | H7 / H7 same |
| g26 | emergence L23 | +17.33 | +17.33 | H3 (0.99) | 1 | H0:+0.3 H1:+0.4 H2:-4.3 H3:+17.2 H4:-0.0 H5:+0.0 H6:+2.6 H7:+1.2 | H3 / H3 same |
| g26 | amplification L29 | +14.97 | +14.97 | H4 (1.06) | 1 | H0:-0.0 H1:+0.0 H2:-0.0 H3:-0.5 H4:+15.8 H5:-0.4 H6:+0.1 H7:-0.0 | H4 / H4 same |
| g26 | late_high_dla L31 | -0.32 | -0.32 | H7 (-12.35) | 8 | H0:+0.0 H1:+0.0 H2:-0.0 H3:+0.1 H4:-0.1 H5:+0.0 H6:-4.4 H7:+4.0 | H7 / H7 same |
| g26 | precursor L20 | +0.07 | +0.07 | H1 (1.02) | 1 | H0:-0.0 H1:+0.1 H2:-0.0 H3:+0.0 H4:+0.0 H5:+0.0 H6:+0.0 H7:-0.0 | H0 / H5 differ |
| g14 | emergence L23 | +17.19 | +17.19 | H3 (1.07) | 1 | H0:-0.1 H1:+0.3 H2:-3.0 H3:+18.4 H4:-0.0 H5:+0.0 H6:+1.0 H7:+0.6 | H3 / H3 same |
| g14 | amplification L29 | +12.49 | +12.49 | H4 (1.01) | 1 | H0:+0.0 H1:+0.0 H2:-0.1 H3:+0.1 H4:+12.6 H5:-0.1 H6:-0.0 H7:+0.0 | H4 / H4 same |
| g14 | late_high_dla L31 | -2.14 | -2.14 | H7 (-1.49) | 8 | H0:+0.0 H1:-0.0 H2:-0.0 H3:+0.0 H4:-0.1 H5:-0.0 H6:-5.2 H7:+3.2 | H7 / H7 same |
| g15 | emergence L23 | +16.87 | +16.87 | H3 (0.93) | 1 | H0:+0.1 H1:+0.5 H2:-1.7 H3:+15.6 H4:-0.0 H5:-0.0 H6:+1.2 H7:+1.1 | H3 / H3 same |
| g15 | amplification L29 | +12.04 | +12.04 | H4 (1.00) | 1 | H0:-0.0 H1:+0.1 H2:+0.0 H3:+0.0 H4:+12.1 H5:-0.1 H6:+0.0 H7:-0.0 | H4 / H4 same |
| g15 | late_high_dla L31 | -1.65 | -1.65 | H7 (-0.85) | 8 | H0:-0.0 H1:+0.0 H2:-0.0 H3:+0.1 H4:-0.2 H5:+0.1 H6:-3.0 H7:+1.4 | H3 / H7 differ |

### Gemma 3 4B: Witness A2 — source classes of the top head toward B (target arm) at each write

| case | write | top head | rewritten_label | edited_line | start_node_q | relation_q | other_edges | bos | header | question_other | sink | top-3 sources (pos, token, class, weight) | top-3 conc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g00 | emergence L23 | H3 | 0.386 | 0.015 | 0.000 | 0.000 | 0.596 | 0.001 | 0.001 | 0.001 | 0.000 | 67 ' JF' other_edges 0.4224; 35 ' JF' rewritten_label 0.3858; 59 ' MC' other_edges 0.0393 | 0.85 |
| g00 | amplification L29 | H4 | 0.187 | 0.062 | 0.014 | 0.001 | 0.573 | 0.144 | 0.013 | 0.005 | 0.000 | 35 ' JF' rewritten_label 0.1872; 0 '' bos 0.1445; 21 'JF' other_edges 0.0661 | 0.40 |
| g00 | late_high_dla L31 | H7 | 0.006 | 0.030 | 0.011 | 0.001 | 0.376 | 0.230 | 0.027 | 0.319 | 0.000 | 0 '' bos 0.2302; 85 '?' question_other 0.1108; 88 ':' question_other 0.1067 | 0.45 |
| g00 | precursor L15 | H5 | 0.003 | 0.007 | 0.112 | 0.070 | 0.176 | 0.175 | 0.013 | 0.444 | 0.000 | 79 '.' question_other 0.1946; 0 '' bos 0.1746; 73 ' MC' start_node_q 0.112 | 0.48 |
| g12 | emergence L23 | H3 | 0.641 | 0.043 | 0.003 | 0.000 | 0.188 | 0.046 | 0.055 | 0.023 | 0.000 | 23 ' KZ' rewritten_label 0.6413; 29 'KZ' other_edges 0.0484; 0 '' bos 0.0458 | 0.74 |
| g12 | amplification L29 | H4 | 0.306 | 0.102 | 0.019 | 0.000 | 0.308 | 0.231 | 0.031 | 0.004 | 0.000 | 23 ' KZ' rewritten_label 0.3059; 0 '' bos 0.2314; 21 'MZ' edited_line 0.094 | 0.63 |
| g12 | late_high_dla L31 | H7 | 0.018 | 0.029 | 0.027 | 0.001 | 0.426 | 0.175 | 0.065 | 0.258 | 0.000 | 0 '' bos 0.1754; 88 ':' question_other 0.1056; 85 '?' question_other 0.084 | 0.36 |
| g12 | precursor L17 | H2 | 0.128 | 0.320 | 0.006 | 0.001 | 0.417 | 0.019 | 0.072 | 0.038 | 0.000 | 24 '\n' edited_line 0.1618; 22 ' red' edited_line 0.1577; 23 ' KZ' rewritten_label 0.1277 | 0.45 |
| g13 | emergence L23 | H3 | 0.288 | 0.007 | 0.000 | 0.000 | 0.691 | 0.012 | 0.000 | 0.001 | 0.000 | 47 ' SR' rewritten_label 0.2883; 55 ' WG' other_edges 0.1993; 63 ' FN' other_edges 0.1445 | 0.63 |
| g13 | amplification L29 | H4 | 0.219 | 0.011 | 0.011 | 0.000 | 0.545 | 0.205 | 0.003 | 0.005 | 0.000 | 47 ' SR' rewritten_label 0.2187; 0 '' bos 0.2051; 31 ' FN' other_edges 0.0834 | 0.51 |
| g13 | late_high_dla L31 | H7 | 0.009 | 0.029 | 0.007 | 0.002 | 0.316 | 0.256 | 0.031 | 0.351 | 0.000 | 0 '' bos 0.2563; 85 '?' question_other 0.1668; 88 ':' question_other 0.093 | 0.52 |
| g13 | precursor L21 | H7 | 0.015 | 0.019 | 0.025 | 0.050 | 0.432 | 0.208 | 0.009 | 0.242 | 0.000 | 0 '' bos 0.208; 85 '?' question_other 0.1049; 88 ':' question_other 0.0712 | 0.38 |
| g20 | emergence L23 | H3 | 0.802 | 0.042 | 0.000 | 0.000 | 0.153 | 0.001 | 0.000 | 0.001 | 0.000 | 63 ' RO' rewritten_label 0.8023; 59 ' UW' other_edges 0.0733; 61 'IW' edited_line 0.0212 | 0.90 |
| g20 | amplification L29 | H4 | 0.237 | 0.076 | 0.022 | 0.001 | 0.510 | 0.146 | 0.005 | 0.004 | 0.000 | 63 ' RO' rewritten_label 0.2367; 0 '' bos 0.1459; 41 'IW' other_edges 0.0814 | 0.46 |
| g20 | late_high_dla L31 | H7 | 0.003 | 0.062 | 0.010 | 0.001 | 0.414 | 0.090 | 0.018 | 0.403 | 0.000 | 88 ':' question_other 0.1962; 85 '?' question_other 0.1387; 0 '' bos 0.0897 | 0.42 |
| g20 | precursor L20 | H6 | 0.002 | 0.011 | 0.027 | 0.009 | 0.231 | 0.237 | 0.028 | 0.453 | 0.000 | 88 ':' question_other 0.2419; 0 '' bos 0.2374; 85 '?' question_other 0.0568 | 0.54 |
| g22 | emergence L23 | H3 | 0.706 | 0.021 | 0.000 | 0.000 | 0.270 | 0.001 | 0.000 | 0.001 | 0.000 | 51 ' VU' rewritten_label 0.706; 59 ' FR' other_edges 0.071; 43 ' HR' other_edges 0.0459 | 0.82 |
| g22 | amplification L29 | H4 | 0.104 | 0.025 | 0.037 | 0.001 | 0.671 | 0.146 | 0.006 | 0.009 | 0.000 | 0 '' bos 0.1459; 57 'VU' other_edges 0.1066; 37 'JV' other_edges 0.1047 | 0.36 |
| g22 | late_high_dla L31 | H7 | 0.005 | 0.031 | 0.008 | 0.001 | 0.277 | 0.128 | 0.016 | 0.534 | 0.000 | 88 ':' question_other 0.2959; 0 '' bos 0.1278; 85 '?' question_other 0.1251 | 0.55 |
| g26 | emergence L23 | H3 | 0.740 | 0.022 | 0.000 | 0.000 | 0.237 | 0.001 | 0.000 | 0.001 | 0.000 | 59 ' QH' rewritten_label 0.7403; 47 ' NV' other_edges 0.1243; 31 ' NV' other_edges 0.0466 | 0.91 |
| g26 | amplification L29 | H4 | 0.054 | 0.037 | 0.022 | 0.000 | 0.678 | 0.194 | 0.010 | 0.005 | 0.000 | 0 '' bos 0.1938; 35 ' QH' other_edges 0.0909; 33 'QS' other_edges 0.0832 | 0.37 |
| g26 | late_high_dla L31 | H7 | 0.003 | 0.049 | 0.011 | 0.001 | 0.358 | 0.191 | 0.034 | 0.353 | 0.000 | 0 '' bos 0.191; 88 ':' question_other 0.1658; 85 '?' question_other 0.1083 | 0.47 |
| g26 | precursor L20 | H0 | 0.026 | 0.032 | 0.442 | 0.007 | 0.330 | 0.034 | 0.012 | 0.117 | 0.000 | 73 ' NV' start_node_q 0.442; 61 'PT' other_edges 0.0503; 0 '' bos 0.0343 | 0.53 |
| g14 | emergence L23 | H3 | 0.728 | 0.018 | 0.000 | 0.000 | 0.250 | 0.004 | 0.000 | 0.001 | 0.000 | 55 ' LP' rewritten_label 0.7276; 51 ' FE' other_edges 0.0778; 49 'LP' other_edges 0.0445 | 0.85 |
| g14 | amplification L29 | H4 | 0.078 | 0.061 | 0.013 | 0.001 | 0.570 | 0.266 | 0.005 | 0.006 | 0.000 | 0 '' bos 0.2657; 55 ' LP' rewritten_label 0.0779; 41 'LP' other_edges 0.075 | 0.42 |
| g14 | late_high_dla L31 | H7 | 0.005 | 0.043 | 0.021 | 0.001 | 0.378 | 0.180 | 0.029 | 0.343 | 0.000 | 0 '' bos 0.1801; 88 ':' question_other 0.1461; 85 '?' question_other 0.1072 | 0.43 |
| g15 | emergence L23 | H3 | 0.439 | 0.067 | 0.002 | 0.000 | 0.413 | 0.020 | 0.051 | 0.008 | 0.000 | 23 ' QC' rewritten_label 0.4386; 59 ' JH' other_edges 0.1508; 39 ' QC' other_edges 0.1219 | 0.71 |
| g15 | amplification L29 | H4 | 0.165 | 0.047 | 0.031 | 0.000 | 0.505 | 0.229 | 0.019 | 0.004 | 0.000 | 0 '' bos 0.2285; 23 ' QC' rewritten_label 0.1648; 39 ' QC' other_edges 0.0679 | 0.46 |
| g15 | late_high_dla L31 | H3 | 0.004 | 0.018 | 0.023 | 0.006 | 0.249 | 0.070 | 0.053 | 0.576 | 0.000 | 85 '?' question_other 0.1763; 88 ':' question_other 0.1267; 86 '\n' question_other 0.1028 | 0.41 |

### Gemma 3 4B: base arm — top head toward A and its source classes

| case | write | top head toward A | rewritten_label | edited_line | start_node_q | relation_q | other_edges | bos | header | question_other | top-3 sources |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g00 | emergence L23 | H3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.997 | 0.001 | 0.001 | 0.002 | 35 ' RF' other_edges 0.423; 59 ' MC' other_edges 0.2385; 39 ' RF' other_edges 0.0938 |
| g00 | amplification L29 | H4 | 0.000 | 0.000 | 0.007 | 0.001 | 0.781 | 0.195 | 0.009 | 0.007 | 0 '' bos 0.1953; 29 'RF' other_edges 0.1077; 35 ' RF' other_edges 0.0915 |
| g00 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.011 | 0.001 | 0.344 | 0.269 | 0.040 | 0.335 | 0 '' bos 0.269; 88 ':' question_other 0.1187; 85 '?' question_other 0.1041 |
| g00 | precursor L15 | H1 | 0.000 | 0.000 | 0.078 | 0.005 | 0.124 | 0.611 | 0.014 | 0.168 | 0 '' bos 0.6111; 73 ' MC' start_node_q 0.0777; 78 ' link' question_other 0.0256 |
| g12 | emergence L23 | H3 | 0.000 | 0.000 | 0.014 | 0.001 | 0.769 | 0.090 | 0.090 | 0.036 | 23 ' IH' other_edges 0.3295; 0 '' bos 0.0896; 24 '\n' other_edges 0.0732 |
| g12 | amplification L29 | H4 | 0.000 | 0.000 | 0.031 | 0.000 | 0.700 | 0.233 | 0.033 | 0.003 | 0 '' bos 0.2327; 23 ' IH' other_edges 0.1888; 21 'MZ' other_edges 0.165 |
| g12 | late_high_dla L31 | H1 | 0.000 | 0.000 | 0.001 | 0.001 | 0.006 | 0.969 | 0.007 | 0.017 | 0 '' bos 0.9685; 1 'Links' header 0.0032; 81 ' node' question_other 0.0026 |
| g12 | precursor L17 | H6 | 0.000 | 0.000 | 0.008 | 0.003 | 0.085 | 0.763 | 0.031 | 0.110 | 0 '' bos 0.7633; 86 '\n' question_other 0.0301; 88 ':' question_other 0.0256 |
| g13 | emergence L23 | H3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 | 0.004 | 0.000 | 0.000 | 47 ' FN' other_edges 0.753; 63 ' FN' other_edges 0.1037; 59 ' FN' other_edges 0.0394 |
| g13 | amplification L29 | H4 | 0.000 | 0.000 | 0.024 | 0.001 | 0.731 | 0.233 | 0.006 | 0.006 | 0 '' bos 0.2331; 31 ' FN' other_edges 0.1082; 47 ' FN' other_edges 0.0723 |
| g13 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.008 | 0.001 | 0.378 | 0.174 | 0.025 | 0.414 | 0 '' bos 0.1737; 88 ':' question_other 0.1626; 85 '?' question_other 0.1408 |
| g13 | precursor L21 | H6 | 0.000 | 0.000 | 0.002 | 0.001 | 0.851 | 0.087 | 0.006 | 0.053 | 47 ' FN' other_edges 0.2194; 63 ' FN' other_edges 0.1735; 31 ' FN' other_edges 0.1158 |
| g20 | emergence L23 | H3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.999 | 0.000 | 0.000 | 0.001 | 63 ' JC' other_edges 0.9446; 47 ' JC' other_edges 0.012; 61 'IW' other_edges 0.0089 |
| g20 | amplification L29 | H4 | 0.000 | 0.000 | 0.052 | 0.002 | 0.741 | 0.190 | 0.006 | 0.009 | 0 '' bos 0.1903; 63 ' JC' other_edges 0.1531; 41 'IW' other_edges 0.094 |
| g20 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.018 | 0.001 | 0.379 | 0.199 | 0.025 | 0.378 | 0 '' bos 0.1994; 88 ':' question_other 0.1826; 85 '?' question_other 0.0845 |
| g20 | precursor L20 | H7 | 0.000 | 0.000 | 0.011 | 0.004 | 0.020 | 0.233 | 0.009 | 0.722 | 87 'Answer' question_other 0.2364; 0 '' bos 0.2333; 88 ':' question_other 0.1288 |
| g22 | emergence L23 | H3 | 0.000 | 0.000 | 0.001 | 0.000 | 0.997 | 0.001 | 0.000 | 0.001 | 51 ' FR' other_edges 0.7182; 47 ' FR' other_edges 0.0586; 43 ' HR' other_edges 0.0457 |
| g22 | amplification L29 | H2 | 0.000 | 0.000 | 0.027 | 0.001 | 0.302 | 0.555 | 0.057 | 0.058 | 0 '' bos 0.5547; 37 'JV' other_edges 0.0495; 51 ' FR' other_edges 0.0466 |
| g22 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.013 | 0.001 | 0.362 | 0.175 | 0.028 | 0.422 | 88 ':' question_other 0.1949; 0 '' bos 0.1749; 85 '?' question_other 0.0994 |
| g26 | emergence L23 | H3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.998 | 0.001 | 0.000 | 0.001 | 47 ' NV' other_edges 0.4608; 59 ' MF' other_edges 0.4154; 31 ' NV' other_edges 0.0306 |
| g26 | amplification L29 | H4 | 0.000 | 0.000 | 0.016 | 0.000 | 0.735 | 0.236 | 0.008 | 0.005 | 0 '' bos 0.2361; 39 ' MF' other_edges 0.1076; 59 ' MF' other_edges 0.0762 |
| g26 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.010 | 0.001 | 0.368 | 0.280 | 0.043 | 0.297 | 0 '' bos 0.28; 88 ':' question_other 0.1224; 85 '?' question_other 0.0708 |
| g26 | precursor L20 | H5 | 0.000 | 0.000 | 0.001 | 0.000 | 0.004 | 0.482 | 0.000 | 0.513 | 0 '' bos 0.4817; 87 'Answer' question_other 0.3765; 86 '\n' question_other 0.0648 |
| g14 | emergence L23 | H3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.995 | 0.005 | 0.000 | 0.001 | 55 ' KL' other_edges 0.433; 47 ' KL' other_edges 0.1397; 63 ' PX' other_edges 0.1191 |
| g14 | amplification L29 | H4 | 0.000 | 0.000 | 0.007 | 0.001 | 0.722 | 0.263 | 0.005 | 0.003 | 0 '' bos 0.2626; 47 ' KL' other_edges 0.0617; 23 ' KL' other_edges 0.056 |
| g14 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.019 | 0.001 | 0.470 | 0.147 | 0.028 | 0.335 | 88 ':' question_other 0.1657; 0 '' bos 0.1475; 85 '?' question_other 0.1102 |
| g15 | emergence L23 | H3 | 0.000 | 0.000 | 0.002 | 0.000 | 0.890 | 0.039 | 0.060 | 0.009 | 59 ' JH' other_edges 0.2834; 23 ' JV' other_edges 0.1805; 55 ' RV' other_edges 0.1661 |
| g15 | amplification L29 | H4 | 0.000 | 0.000 | 0.026 | 0.000 | 0.732 | 0.221 | 0.019 | 0.002 | 0 '' bos 0.2211; 23 ' JV' other_edges 0.1375; 39 ' QC' other_edges 0.1186 |
| g15 | late_high_dla L31 | H7 | 0.000 | 0.000 | 0.023 | 0.001 | 0.491 | 0.167 | 0.033 | 0.286 | 0 '' bos 0.1668; 85 '?' question_other 0.1127; 88 ':' question_other 0.1089 |

**Tallies at the emergence write, primary cases (n=6):** HF3 (≤ 2 heads for 90% of the S step): 6/6; HF4 (top-3 source positions ≥ 50% of the top head's mass): 6/6; HF5 (same top head index toward A in base as toward B in target): 6/6.
Top emergence head per case: g00:H3, g12:H3, g13:H3, g20:H3, g22:H3, g26:H3.
Dominant source class of the top head (target arm): {'g00': ('other_edges', 0.596), 'g12': ('rewritten_label', 0.641), 'g13': ('other_edges', 0.691), 'g20': ('rewritten_label', 0.802), 'g22': ('rewritten_label', 0.706), 'g26': ('rewritten_label', 0.74)}.
Amplification write top heads: g00:H4(1.02), g12:H4(1.03), g13:H4(1.03), g20:H4(0.99), g22:H4(0.96), g26:H4(1.06).
Late high-DLA head site: contribution of the named head (H7 on 4B / H10 on 12B) as a fraction of that site's S step: g00:-1.28, g12:-1.35, g13:2.97, g20:-3.32, g22:6.53, g26:-12.34.

**Witness B (precursor writes):** g00 L15: S step +0.01, top head H5 (1.20), dominant class question_other vs emergence head H3; g12 L17: S step +0.12, top head H0 (0.78), dominant class other_edges vs emergence head H3; g13 L21: S step +0.29, top head H6 (1.00), dominant class other_edges vs emergence head H3; g20 L20: S step +0.08, top head H1 (1.45), dominant class question_other vs emergence head H3; g26 L20: S step +0.07, top head H1 (1.02), dominant class start_node_q vs emergence head H3.

### Gemma 3 12B: Witness A at each write (per case; target − base per head, reader units = nats of the site's S step)

| case | write | S step (carrier) | Σ heads | top head (fraction) | heads to 90% | cf by head | A3 same head base/target |
|---|---|---|---|---|---|---|---|
| g26 | emergence L35 | +18.44 | +18.44 | H12 (0.52) | 2 | H0:+1.0 H1:-0.0 H2:+0.0 H3:+0.5 H4:-0.0 H5:-0.2 H6:-0.1 H7:+0.5 H8:+8.3 H9:-0.7 H10:-1.2 H11:+0.8 H12:+9.6 H13:-0.0 H14:-0.0 H15:-0.0 | H8 / H12 differ |
| g26 | amplification L41 | +46.01 | +46.01 | H13 (0.34) | 4 | H0:+0.2 H1:+0.1 H2:-3.3 H3:+10.3 H4:-1.6 H5:+14.2 H6:+0.0 H7:+0.0 H8:+0.8 H9:-1.3 H10:+0.0 H11:+10.7 H12:+0.8 H13:+15.5 H14:-0.3 H15:+0.0 | H5 / H13 differ |
| g26 | late_high_dla L44 | +0.40 | +0.40 | H10 (17.66) | 1 | H0:-0.2 H1:+0.1 H2:-0.6 H3:+0.0 H4:+0.2 H5:-0.1 H6:+0.0 H7:-0.0 H8:+0.5 H9:-0.2 H10:+7.0 H11:-6.5 H12:-0.0 H13:+0.0 H14:+0.0 H15:+0.2 | H10 / H10 same |
| g26 | precursor L29 | +1.83 | +1.83 | H1 (0.61) | 2 | H0:-0.3 H1:+1.1 H2:-0.1 H3:+1.1 H4:-0.0 H5:-0.0 H6:-0.0 H7:+0.0 H8:-0.0 H9:+0.0 H10:+0.1 H11:-0.0 H12:+0.0 H13:+0.0 H14:-0.0 H15:-0.0 | H1 / H3 differ |
| g03 | emergence L35 | +10.15 | +10.15 | H12 (0.56) | 2 | H0:+0.4 H1:+0.0 H2:+0.0 H3:+0.1 H4:-0.0 H5:-0.0 H6:-0.1 H7:+0.2 H8:+4.5 H9:-0.2 H10:-0.4 H11:+0.3 H12:+5.6 H13:-0.3 H14:+0.0 H15:-0.0 | H12 / H8 differ |
| g03 | amplification L41 | +22.37 | +22.37 | H5 (0.61) | 3 | H0:+2.2 H1:+0.1 H2:-2.2 H3:+4.4 H4:-1.2 H5:+13.7 H6:-0.0 H7:+0.0 H8:+0.3 H9:-0.5 H10:+0.0 H11:+5.1 H12:+0.7 H13:-0.2 H14:-0.1 H15:-0.0 | H5 / H3 differ |
| g03 | late_high_dla L44 | +3.46 | +3.46 | H10 (1.53) | 1 | H0:-0.0 H1:+0.1 H2:-0.1 H3:+0.0 H4:+0.2 H5:-0.1 H6:-0.0 H7:-0.0 H8:+0.0 H9:+0.0 H10:+5.3 H11:-2.5 H12:+0.0 H13:-0.0 H14:+0.1 H15:+0.3 | H10 / H10 same |
| g03 | precursor L26 | +0.12 | +0.12 | H11 (0.66) | 2 | H0:+0.0 H1:+0.0 H2:+0.0 H3:+0.0 H4:-0.0 H5:+0.0 H6:+0.0 H7:-0.0 H8:-0.0 H9:-0.0 H10:-0.0 H11:+0.1 H12:+0.0 H13:-0.0 H14:-0.0 H15:-0.0 | H12 / H3 differ |
| g00 | emergence L35 | +11.30 | +11.30 | H12 (0.81) | 2 | H0:+0.0 H1:+0.1 H2:-0.1 H3:+0.7 H4:+0.0 H5:-0.0 H6:+0.1 H7:-0.3 H8:+1.9 H9:-0.2 H10:-0.3 H11:+0.2 H12:+9.2 H13:+0.0 H14:+0.0 H15:-0.0 | H12 / H12 same |
| g00 | amplification L41 | +30.17 | +30.17 | H5 (0.57) | 3 | H0:+5.7 H1:-0.4 H2:-2.4 H3:+6.2 H4:-1.2 H5:+17.3 H6:-0.0 H7:+0.0 H8:+0.5 H9:-0.6 H10:+0.0 H11:+4.2 H12:+1.0 H13:-0.1 H14:-0.1 H15:-0.0 | H3 / H5 differ |
| g00 | late_high_dla L44 | +1.92 | +1.92 | H10 (3.29) | 1 | H0:-0.1 H1:+0.2 H2:-0.5 H3:+0.0 H4:+0.5 H5:-0.2 H6:+0.0 H7:-0.0 H8:+0.0 H9:+0.0 H10:+6.3 H11:-4.4 H12:+0.0 H13:-0.0 H14:-0.0 H15:+0.0 | H10 / H10 same |
| g00 | precursor L29 | +2.92 | +2.92 | H1 (0.83) | 2 | H0:-0.4 H1:+2.4 H2:-0.1 H3:+0.9 H4:+0.0 H5:+0.0 H6:-0.0 H7:+0.0 H8:-0.0 H9:-0.0 H10:+0.0 H11:-0.0 H12:+0.0 H13:+0.0 H14:+0.0 H15:+0.0 | H1 / H1 same |

### Gemma 3 12B: Witness A2 — source classes of the top head toward B (target arm) at each write

| case | write | top head | rewritten_label | edited_line | start_node_q | relation_q | other_edges | bos | header | question_other | sink | top-3 sources (pos, token, class, weight) | top-3 conc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g26 | emergence L35 | H8 | 0.609 | 0.002 | 0.004 | 0.000 | 0.265 | 0.074 | 0.018 | 0.029 | 0.000 | 59 ' QH' rewritten_label 0.6088; 35 ' QH' other_edges 0.1364; 0 '' bos 0.0741 | 0.82 |
| g26 | amplification L41 | H5 | 0.143 | 0.009 | 0.049 | 0.001 | 0.393 | 0.375 | 0.015 | 0.015 | 0.000 | 0 '' bos 0.375; 59 ' QH' rewritten_label 0.1428; 35 ' QH' other_edges 0.0902 | 0.61 |
| g26 | late_high_dla L44 | H10 | 0.012 | 0.033 | 0.031 | 0.002 | 0.204 | 0.215 | 0.043 | 0.460 | 0.000 | 0 '' bos 0.2147; 87 'Answer' question_other 0.1174; 85 '?' question_other 0.1148 | 0.45 |
| g26 | precursor L29 | H1 | 0.286 | 0.035 | 0.006 | 0.009 | 0.521 | 0.028 | 0.108 | 0.007 | 0.000 | 31 ' NV' other_edges 0.3001; 59 ' QH' rewritten_label 0.2858; 32 '\n' other_edges 0.0512 | 0.64 |
| g03 | emergence L35 | H12 | 0.366 | 0.021 | 0.060 | 0.003 | 0.424 | 0.093 | 0.006 | 0.027 | 0.000 | 31 ' KS' rewritten_label 0.3656; 49 'KS' other_edges 0.1256; 0 '' bos 0.0934 | 0.58 |
| g03 | amplification L41 | H5 | 0.146 | 0.012 | 0.165 | 0.006 | 0.332 | 0.303 | 0.005 | 0.030 | 0.000 | 0 '' bos 0.3032; 73 ' SV' start_node_q 0.1653; 31 ' KS' rewritten_label 0.1462 | 0.61 |
| g03 | late_high_dla L44 | H10 | 0.009 | 0.010 | 0.044 | 0.004 | 0.110 | 0.132 | 0.018 | 0.673 | 0.000 | 88 ':' question_other 0.2335; 85 '?' question_other 0.1838; 0 '' bos 0.1323 | 0.55 |
| g03 | precursor L26 | H12 | 0.007 | 0.069 | 0.494 | 0.150 | 0.152 | 0.049 | 0.007 | 0.072 | 0.000 | 73 ' SV' start_node_q 0.4937; 77 ' blue' relation_q 0.1498; 29 'SV' edited_line 0.0558 | 0.70 |
| g00 | emergence L35 | H12 | 0.405 | 0.059 | 0.069 | 0.001 | 0.337 | 0.072 | 0.022 | 0.034 | 0.000 | 35 ' JF' rewritten_label 0.4053; 0 '' bos 0.0719; 73 ' MC' start_node_q 0.0692 | 0.55 |
| g00 | amplification L41 | H3 | 0.426 | 0.001 | 0.003 | 0.000 | 0.290 | 0.256 | 0.011 | 0.013 | 0.000 | 35 ' JF' rewritten_label 0.4259; 0 '' bos 0.2562; 21 'JF' other_edges 0.109 | 0.79 |
| g00 | late_high_dla L44 | H10 | 0.009 | 0.021 | 0.031 | 0.001 | 0.171 | 0.196 | 0.036 | 0.535 | 0.000 | 0 '' bos 0.196; 85 '?' question_other 0.1461; 88 ':' question_other 0.1199 | 0.46 |
| g00 | precursor L29 | H1 | 0.650 | 0.039 | 0.003 | 0.002 | 0.234 | 0.029 | 0.038 | 0.006 | 0.000 | 35 ' JF' rewritten_label 0.6496; 67 ' JF' other_edges 0.0648; 59 ' MC' other_edges 0.0449 | 0.76 |

### Gemma 3 12B: base arm — top head toward A and its source classes

| case | write | top head toward A | rewritten_label | edited_line | start_node_q | relation_q | other_edges | bos | header | question_other | top-3 sources |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g26 | emergence L35 | H12 | 0.000 | 0.000 | 0.017 | 0.000 | 0.868 | 0.063 | 0.013 | 0.039 | 59 ' MF' other_edges 0.4415; 39 ' MF' other_edges 0.1831; 31 ' NV' other_edges 0.0682 |
| g26 | amplification L41 | H13 | 0.000 | 0.000 | 0.053 | 0.000 | 0.643 | 0.281 | 0.012 | 0.011 | 0 '' bos 0.2813; 59 ' MF' other_edges 0.2209; 39 ' MF' other_edges 0.1628 |
| g26 | late_high_dla L44 | H10 | 0.000 | 0.000 | 0.037 | 0.002 | 0.206 | 0.172 | 0.031 | 0.552 | 0 '' bos 0.1718; 85 '?' question_other 0.1659; 88 ':' question_other 0.1546 |
| g26 | precursor L29 | H3 | 0.000 | 0.000 | 0.003 | 0.001 | 0.914 | 0.012 | 0.034 | 0.035 | 59 ' MF' other_edges 0.7624; 31 ' NV' other_edges 0.035; 35 ' QH' other_edges 0.0263 |
| g03 | emergence L35 | H8 | 0.000 | 0.000 | 0.025 | 0.000 | 0.739 | 0.131 | 0.005 | 0.100 | 31 ' KJ' other_edges 0.4475; 0 '' bos 0.131; 33 'KJ' other_edges 0.074 |
| g03 | amplification L41 | H3 | 0.000 | 0.000 | 0.008 | 0.001 | 0.688 | 0.270 | 0.008 | 0.026 | 0 '' bos 0.2701; 31 ' KJ' other_edges 0.2653; 33 'KJ' other_edges 0.0877 |
| g03 | late_high_dla L44 | H10 | 0.000 | 0.000 | 0.032 | 0.004 | 0.114 | 0.153 | 0.017 | 0.680 | 88 ':' question_other 0.2762; 85 '?' question_other 0.2076; 0 '' bos 0.153 |
| g03 | precursor L26 | H3 | 0.000 | 0.000 | 0.085 | 0.735 | 0.028 | 0.023 | 0.002 | 0.129 | 77 ' blue' relation_q 0.7347; 79 '.' question_other 0.1139; 73 ' SV' start_node_q 0.0845 |
| g00 | emergence L35 | H12 | 0.000 | 0.000 | 0.013 | 0.001 | 0.897 | 0.053 | 0.018 | 0.018 | 35 ' RF' other_edges 0.5744; 39 ' RF' other_edges 0.073; 0 '' bos 0.0531 |
| g00 | amplification L41 | H5 | 0.000 | 0.000 | 0.022 | 0.001 | 0.617 | 0.322 | 0.017 | 0.021 | 0 '' bos 0.3218; 35 ' RF' other_edges 0.2122; 29 'RF' other_edges 0.0844 |
| g00 | late_high_dla L44 | H10 | 0.000 | 0.000 | 0.027 | 0.001 | 0.224 | 0.193 | 0.034 | 0.522 | 0 '' bos 0.1927; 85 '?' question_other 0.1743; 88 ':' question_other 0.1267 |
| g00 | precursor L29 | H1 | 0.000 | 0.000 | 0.001 | 0.001 | 0.947 | 0.022 | 0.024 | 0.005 | 35 ' RF' other_edges 0.702; 59 ' MC' other_edges 0.1303; 39 ' RF' other_edges 0.0297 |

**Tallies at the emergence write, primary cases (n=3):** HF3 (≤ 2 heads for 90% of the S step): 3/3; HF4 (top-3 source positions ≥ 50% of the top head's mass): 3/3; HF5 (same top head index toward A in base as toward B in target): 1/3.
Top emergence head per case: g26:H12, g03:H12, g00:H12.
Dominant source class of the top head (target arm): {'g26': ('rewritten_label', 0.609), 'g03': ('other_edges', 0.424), 'g00': ('rewritten_label', 0.405)}.
Amplification write top heads: g26:H13(0.34), g03:H5(0.61), g00:H5(0.57).
Late high-DLA head site: contribution of the named head (H7 on 4B / H10 on 12B) as a fraction of that site's S step: g26:17.67, g03:1.53, g00:3.29.

**Witness B (precursor writes):** g26 L29: S step +1.83, top head H1 (0.61), dominant class other_edges vs emergence head H12; g03 L26: S step +0.12, top head H11 (0.66), dominant class start_node_q vs emergence head H12; g00 L29: S step +2.92, top head H1 (0.83), dominant class rewritten_label vs emergence head H12.

### Reading (index and site only; no role words)

1. **The emergence step is one head.** At the L23 attention write on 4B, **H3** carries
   89–111% of the counterfactual reader step in all six primary cases and both secondary
   cases; the remaining heads contribute within ±4 nats and mostly cancel. On 12B the L35
   step is carried by **H12 with H8** (0.52–0.81 and the rest; two heads reach 90% in 3/3).
2. **What it reads.** H3's attention at the answer position goes to occurrences of the label
   it writes: in the target arm the **rewritten label token** is the top source in 5/6 primary
   cases (0.29–0.80 of its mass) and the dominant class in 4/6; where "other edges" dominates
   (g00, g13) the top sources are other lines carrying the same label token (` JF` at 67,
   ` SR`/` FN` lines). In the base arm the same head reads the original label's occurrences
   (` RF` at the queried edge, 0.42, then its other mentions). The start node's mention in the
   question receives ≤ 0.003 of H3's mass in every target arm; the relation word ≤ 0.001. Of
   the two candidates written in the freeze, the sources are candidate (a): the label is read
   from the edge lines, at the rewritten position first. On 12B H12/H8 read the rewritten label
   (0.37–0.61) and its other occurrences the same way.
3. **The amplification step is a different single head.** At L29, **H4** carries 96–106% of
   the step in 6/6 (same head toward A in the base arm in 5/6), reading diffusely: BOS 15–27%,
   other edge lines 51–68%, the rewritten label 5–31% (top-3 concentration 0.36–0.63). On 12B
   the L41 step is spread over H13, H5, H11, H3 (three or four heads to 90%).
4. **The late high-DLA head carries none of the net counterfactual at its site.** At L31 the
   site's own S step is −2.5 to +0.9 nats on 4B; H7 contributes +1.4 to +8.5 there and **H6
   cancels it** with −3.0 to −11.0. Its sources are `?`, `:` and BOS. On 12B the L44 pair is
   H10 (+5.3 to +7.0) against H11 (−2.5 to −6.5) with a site step of +0.4 to +3.5. This is the
   observed contribution INSTRUMENT-1a's DLA ranked first; stated as such, not as a role.
5. **The precursor is a different stage.** Every case with a defined precursor has a
   different top head there (4B: H5, H0, H6, H1, H1; 12B: H1 with H3) with steps of 0.01–0.29
   nats on 4B and 1.8–2.9 on 12B, reading question tokens, other edge lines or (g26 4B,
   g00 12B) the start node or the rewritten label. It shares neither head index nor source
   profile with H3 / H12.
6. **Head-sum law in reader units.** Σ_h ⟨r_B − r_A, c′_h⟩ equals the carrier's recorded step
   to 0.01 nats at every witnessed write on both models (the "S step" and "Σ heads" columns).

### Verdict

HEAD-OBS-1 is **closed**. HL1–HL6 hold on the execution SHA (engineering freeze); every
witness in the tables above is recorded; HF1, HF3, HF4, HF5 HELD; HF2 FAILED on its original
arm with the default-policy figures recorded beside it; HF6 answered without a forecast.
No hypothesis was added after execution. INTERVENE-1 receives a *named* candidate set with
recorded source patterns and no assigned roles: 4B L23 H3 and 12B L35 H12/H8 (emergence),
4B L29 H4 and 12B L41 H13/H5/H11/H3 (amplification), 4B L31 H7/H6 and 12B L44 H10/H11 (the
late high-DLA pair), and the precursor heads. "What information did H3 read to write the
label" beyond the six fixed source classes is a new rung with its own freeze.

Records (gzipped), analyses, gate results and these tables: `chris-experiments/larql/
D_instrument1_edge1_calibration/{head-obs-1, bulk/heads4b, bulk/heads12b}`. DB:
`head-obs-1-per-head-observation`, run RUN-20260920-200222-01126.
