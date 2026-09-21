# V3-HEAD-OBS-1 — can the canonical decode step expose each softmax head's distribution and output, and can the attention write be rebuilt from them exactly?

Pre-registered 2026-09-20, before any implementation on this branch. Properties and
forecasts below are FROZEN. Nothing here claims a result.

Programme: OBSERVE. A sibling of V3-LENS-1 and V3-INTERVENE-1 above V3-OBS-1 (#484) and
V3-STREAM-1 (#485). The first fine-grained internal decomposition on the VINDEX3
execution authority. The migration ledger names it as the home of the legacy walk's
attention-weights capture and per-head analysis; neither exists on VINDEX3's main today.

Specification proceeds now. Implementation against the substrate waits for the `observe`
verb (#486) to land: no rung under moving substrate. An uncommitted candidate
implementation already exists in a shared working tree; the section "Existing candidate"
names it and what it must change to satisfy this freeze.

---

## The question

V3-OBS-1 reads the carrier at every write. Between the attention input it also reads
(`operand_input`, `InputSite::Attention`) and the attention write it reports
(`carrier_write`), the executor does the thing a mechanistic question is usually about:
each head forms a distribution over source positions, mixes the values it attends to,
and the output projection folds the heads into one vector. On main, none of that is
observable. The observation contract says so in plain words: `o_proj`'s input "never
surfaces at this boundary".

The sealed HEAD-1 and HEAD-2 lines on google/gemma-3-4b-it (MLX, f32) are built on
exactly the quantities this rung would expose. HEAD-1 splits block 14's attention write
into per-head parts, `w_h = s(o)·(1+γ)⊙W_{O,h}·x_h`, and gates the split by
`Σ_h w_h = a14` to `1e-3` relative. HEAD-2 splits each head's part by source position,
`c_{h,t} = s(o)·(1+γ)⊙W_{O,h}·(A_h[t]·v_h[t])`, and gates `Σ_t c_{h,t} = w_h` the same way.
Both are descriptive; neither intervenes. They are the acceptance witness for a per-head
tap: if the tap is faithful, the same identities hold on the executor, and the same
role shares can be read from it.

V3-HEAD-OBS-1 asks whether each softmax head's distribution and pre-projection output
can be handed to an observer from inside the one kernel the executor already has,
without changing execution, and whether a consumer holding those records and the
prepared image can rebuild the attention write exactly enough to run HEAD-2's
descriptive attribution on the executor.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| Every standard softmax family (Gemma 3 sliding and global, Granite, OLMo2, Qwen, Llama) lowers to `LayerAttention::Softmax` and runs ONE CPU kernel, `aggregate_heads`: per query head, `scores` is a function-local vector over `source_start..=position` (transient, dropped per head), `head_out` is a slice of the `concat` buffer, the GQA map is `kv_head = q_head / (num_q_heads / num_kv_heads)` | `opplan/exec/production.rs`, `aggregate_heads`, `attend_position` |
| After the head loop the output gate (where a family declares one) multiplies `concat` in place, then `o_proj` consumes it; a consumer reading `concat` afterwards sees the gated value | `opplan/exec/production.rs`, `attend_position` |
| The device backend reuses the production kernel for its attention core | `opplan/exec/device.rs` |
| The reference backend has its own head loop, the oracle the production kernel is gated against | `opplan/exec/reference.rs`, `attend_position_inner` |
| MLA already materialises per-head weights (`heads × visible`) and per-head values (`heads × v_dim`) in its trace; the decode and batch call sites keep only `.output` | `opplan/exec/mla.rs`, `mla_forward_with`; `opplan/exec/decode.rs`; `opplan/exec/mod.rs` |
| Conv-QKV attention has softmax heads too: its per-position output plane survives the call, its scores do not | `opplan/exec/conv_qkv.rs`, `layer_forward_with` |
| KDA, Gated DeltaNet and Mamba2 have no softmax heads; their analogous object is the mixer's normed core output | `opplan/exec/kda.rs`, `gated_delta.rs`, `mamba2.rs` |
| The batch prefill path has no `StepObserver` at all: `traverse` takes a `PlaneEvent` sink whose layer trace carries `post_attention`, `ffn_input`, `post_layer` and nothing inside attention; `step_many` emits no `StepEvent` | `opplan/exec/mod.rs`, `traverse`; `opplan/exec/decode.rs`, `step_many` |
| The Kimi Metal stack fires no observer; the device MLA scratch holds a weights buffer that is read back only by a traced test entry ("Gates only") | `opplan/exec/stack_metal.rs`; `larql-compute-metal`, `mla_attention_step_traced` |
| Geometry lives on the plan's `AttentionOp` (`num_q_heads`, `num_kv_heads`, `head_dim`, `query_scale`, `score_scale`, `logit_softcapping`, `span`, `window`, `qk_norm`, `sinks`) and reaches both paths through ONE builder, `AttentionOperands::call`; head counts are not on `PreparedOperands`; `LayerKvGeometry` is flat (`kv_dim = num_kv_heads × head_dim`), so head identity is erased at the continuation seam | `opplan/mod.rs`; `opplan/exec/mod.rs`; `opplan/exec/kv.rs` |
| Sinks are indexed per query head; the window is applied as `source_start = (position+1) − window` on sliding layers | `opplan/exec/production.rs` |
| The parity gate: an observed step is bit-identical to an unobserved one on the logits; a value-consuming observer leaves every position bit-identical on both backends | `opplan/exec/tests/observe.rs`; `opplan/exec/tests/carrier_write.rs` (P1) |
| The golden fixture crosses a sliding window: hidden 12, three query heads over one KV head, head dim 4, window 3 on layer 0, full span on layer 1, five positions | `opplan/exec/tests/golden.rs` |
| Gemma 3 4B IT text component on disk: 34 layers, hidden 2560, 8 query heads over 4 KV heads, head dim 256, score scale 1/16, window 1024 on the sliding layers, per-head QK norm, a post-attention norm, no sinks, no attention softcap | `~/chris-models/gemma3-4b-it.vindex3`, `system_graph.json` |
| Granite 4.2 3B on disk: 40 softmax attention layers, no state-space layers | `~/chris-models/granite-4.2-3b.vindex3`, `system_graph.json` |
| The sealed witness: HEAD-1's per-head split and gate G9, HEAD-2's per-source split, gates G10/G10b, the four roles (BOS, RELATION, ENTITY, LAST) and the sealed per-head role shares on google/gemma-3-4b-it | `~/chris-source/chris-experiments/semantic_graph/SG_B14_HEAD1_PLAN.md`, `SG_B14_HEAD2_PLAN.md` |

---

## Contract properties (frozen as properties, not as names)

**A1 — one kernel, one tap.** The per-head tap sits inside the head loop of the one
softmax kernel every standard family and the device backend share. There is no second
per-head kernel and no per-architecture tap. The reference backend implements the same
tap inside its own loop, because it is the oracle the split identity is checked against.

**A2 — what one head record says.** Per executed softmax layer, per position, per
query head, the observer is handed, borrowed: the query head index; its KV head `g(h)`;
the first source position the head could read (`source_start`); the distribution the
kernel actually used over `source_start..=position`, after the score scale, the softcap
and the sink where the plan declares them — positions outside the span are ABSENT, never
zero, and the sink mass is carried beside the distribution so that `Σ probs + sink = 1`
within `1e-6` (the evidence freeze's source law, HL2); and the head's mixed value
`ctx_h = Σ_t A_h[t]·v_h[t]` before the output gate, before `o_proj`, before the
post-attention norm and before any layer scale. Where the plan declares an output gate
the record also carries the head's activated gate slice, so the gated per-head input to
`o_proj` is reconstructible; where it declares none the record says so.

**A3 — the sources are reachable.** A consumer holding a head record can obtain the
value row of that head's KV head at every source position it attended, from the run's
own continuation state, so that `c_{h,t} = W_{O,h}·(A_h[t]·v_h[t])` is computable. The
rung exposes them at the tap, borrowed; it does not copy them.

**A4 — the head's projection is the image's projection.** The prepared image exposes
one function that applies the output projection's slice for one query head
(`W_{O,h}·x`) with the pinned realization, and the executor's own `o_proj` is that
projection over all heads. A consumer never carries a second `W_O`. This is LENS-1's H1
applied to `o_proj`.

**A5 — opt-in, declared, priced.** Head observation is requested by the observer
(`wants_attention_heads`), per run. The receipt says heads were captured, how many
records fired, and which executed layers emitted none (A6). The tap is timing-intrusive
by declaration and its cost is measured (HP5), not described.

**A6 — coverage is per layer, on the receipt.** A layer with softmax heads emits
exactly `num_q_heads` records per position. A layer without (KDA, Mamba2, Gated
DeltaNet) emits none and is named on the receipt as uncovered. MLA and conv-QKV layers
are uncovered in THIS rung and named the same way (see Out of scope). A plan that mixes
families is not refused; a request for heads on a backend that cannot serve them is
(A8).

**A7 — decode only.** Head records fire on the decode step. The batch prefill path
emits none, and a run whose prompt positions were prefilled says on its receipt which
positions were head-observed. The witness runs step the prompt so the last prompt
position is observed.

**A8 — refusals are declared.** A backend whose attention core cannot serve the tap
refuses the request before the first token; the Kimi Metal stack refuses by
construction. A request that names no observer capability is a no-op, never a partial
capture.

**A9 — capture is declared and priced, at two levels (the evidence freeze's HL5).** Head
records are borrowed values for in-process consumers; what reaches the run record is
chosen per run. *Stats* level, at every attention write when armed: one `head_write`
event per head carrying `‖c′_h‖`, its projection on the run's basis, `g(h)`, the top-`k`
source positions with their probabilities (`k` declared) and the sink mass. *Full* level,
at declared sites only: the whole source distribution and, on request, `c′_h` itself. The
cost class of the stats level is one `W_o` slice product per head per observed write — the
same multiply-adds as one `o_proj` — plus the probabilities the executor already holds. A
record without head events is still a complete V3-OBS-1 record; nothing reconstructs a
head decomposition from `delta` after the fact. Any per-head tensor file is a later rung.

---

## Frozen acceptance properties

**HP1 — parity (the evidence freeze's HL3).** With heads armed on every softmax layer,
logits at every position, every recorded carrier write (`delta`, `after`, `layer_scale`),
every structural event and the run's provenance are bit-identical to the unobserved run,
on the reference and production CPU backends, on the golden plan and on every real
subject run. The observer reads what the kernel already computed; observation is
subscription, never a second executor.

**HP2 — record identity.** On the golden plan, one record per (layer, position, query
head) for every softmax layer and none for others; `kv_head = head / (num_q_heads /
num_kv_heads)`; the distribution has length `position + 1 − source_start`, every entry
in `[0, 1]`, summing to one within `1e-6` on layers without a sink and to at most one
with; `x_h` has length `head_dim` and is finite; the window truncation is witnessed on
layer 0 from position 3.

**HP3 — the head-sum law (the evidence freeze's HL1).** On the golden plan (both
backends), on Granite 4.2 3B (no post-attention norm: the `s = 1` branch) and on Gemma 3
4B IT (the post-attention norm branch), a consumer using only the head records and A4's
projection computes, beside the executor's fused output and never in place of it, the
per-head contributions `c_h = W_o[:, h·d..(h+1)·d] · (g_h ⊙ ctx_h)` and the children of the
recorded write `c′_h = s · (1 + w_post) ⊙ c_h · residual_scale`, with `s = 1/rms(o + bias)`
the post-attention norm's scalar for this write (`1` and no gain where the family has
none) and the once-only terms (the O bias) declared; property: `‖Σ_h c′_h + bias′ −
delta‖ / ‖delta‖ ≤ 1e-4` at every observed write, with the measured residual recorded,
where `delta` is what `carrier_write` already reports. Bit identity is not required across
accumulation orders and is not claimed. The reader-projected form `Σ_h ⟨r, c′_h⟩ =
⟨r, delta⟩` follows by linearity and is checked too. This is HEAD-1's gate G9 on the
executor, stated through the norm.

**HP4 — the source identity.** On the same subjects, `Σ_t c_{h,t}` rebuilds `W_{O,h}·x_h`
for every head to within `1e-5` relative — HEAD-2's gate G10, on the executor.

**HP5 — cost is measured.** Token wall with heads armed on every softmax layer and an
observer that reads but does not compute, minus token wall with `NoopObserver`, on
Granite 4.2 3B and on Gemma 3 4B IT, release, production CPU; and separately the wall
of an observer that performs HP3's reconstruction on every layer.

**HP6 — on the record (the evidence freeze's HL4 and HL6).** Stats-level `head_write`
events carry their own sequence and position, keyed to `(run, position, layer, site =
attention)` beneath the site's own write, which stays authoritative; a write whose
decomposition a backend or path cannot provide records a refusal with its reason; the
receipt covers them, replay is equal, the record reads back, and a run on a mixed-family
plan names its uncovered layers. The schema gains the `head_write` event kind and the
refusal spelling and nothing else (forecast; more is a recorded finding).

**HP7 — the first reading.** Gemma 3 4B IT, production CPU, `LARQL_CPU_MAX_FORMAT=bf16`,
prompt stepped position by position: one same-entity/different-relation donor pair from
the sealed HEAD-2 set, named in the amendments before the run; at block 14's last
prompt position, per head, the pooled projection share `ρ_{h,R} = ⟨d_{h,R}, Δw_h⟩ /
‖Δw_h‖²` over the four roles, with `Δw_h = w_h^B − w_h^A` and `d_{h,R}` from the per-source
split, computed from head records on the executor. Recorded beside the sealed per-head
labels for that pair's heads (H3 carrier, H7 opposer, and the rest). **No agreement is
forecast** — the sealed rows are MLX f32 on the same checkpoint; the executor's arm is
bf16-capped with Q8-realised projections; the comparison is recorded as a comparison, one
pair, and disagreement is a finding about the arm and the realization, not a failure of
this rung.

---

## Forecasts

- **HF1**: HP1 holds bit for bit on both backends and all witnesses; the tap, unread,
  costs nothing distinguishable from run-to-run noise (HP5's first number).
- **HF2**: the shared kernel gains one optional tap parameter and the backend trait one
  method; the reference loop gains the same tap; `AttentionOperands::call` does not
  change. Nothing else in execution changes.
- **HF3**: HP3 holds at roughly `1e-6` relative on the golden plan (f32 dot-product
  order only) and within `1e-4` on Gemma 3 4B IT under the bf16 cap; if the Q8-realised
  `W_O` puts it outside `1e-3`, that is recorded and the arm is re-run f32-realised
  before the property is judged.
- **HF4**: HP5's reconstruction cost is about one extra output projection per observed
  layer (the per-head slices sum to the whole `W_O`) plus `kv_len × head_dim × hidden`
  per head for the source split — small against the forward on a short prompt, and it
  scales with prompt length, which is why A5 exists.
- **HF5**: the run record gains one event kind and the receipt two fields; the STREAM-1
  schema needs nothing else.
- **HF6**: the existing candidate's whole-plan refusal becomes A6's per-layer coverage;
  its reference-backend refusal becomes A1's second loop; its record gains the gate
  slice (A2) and source reachability (A3).
- **HF7** (the evidence freeze's HF2): stats-level head observation on every attention
  write costs at most 15% of token wall on Gemma 3 4B — one extra `o_proj`-class product
  per attention write against a forward dominated by the FFN and the head — and the
  head-sum residual (HP3) is at most `1e-5` relative on both CPU backends on the golden
  plan, f32 accumulation order only.

---

## Existing candidate (uncommitted, in the shared working tree on branch `v3-obs-1`, 2026-09-20)

Read as evidence that the tap is feasible, not as the rung's result. It adds to the
observer an opt-in `wants_attention_heads` and an `attention_head(layer, record)` with a
record of `{position, head, kv_head, source_start, weights, values}`; to the backend trait
an `attention_step_observed` whose default refuses; to the production kernel an optional
tap fired inside the head loop after the weighted-value accumulation, handing the
head's `scores` and its slice of `concat`; to the decode step a whole-plan precondition
that refuses any plan with a non-softmax layer; and a test that asserts bit-identical
logits under capture, one record per (layer, position, head), the GQA map, the window
truncation and the reference backend's refusal.

Against this freeze it differs in five places: the reference backend refuses (A1 needs
the oracle loop tapped); a mixed-family plan is refused as a whole (A6 names uncovered
layers instead); there is no gate slice on the record (A2) and no source reachability
(A3); there is no per-head projection on the prepared image (A4); and nothing reaches
the receipt (A5, A6, A9). It is a candidate for the production half of A1 and for HP2.
It must not be committed from the shared tree beside unrelated edits; it lands on this
branch, against these properties, after #486.

---

## Reconciliation with the evidence freeze (2026-09-20)

A second HEAD-OBS-1 pre-registration exists: the **HEAD-OBS-1 evidence freeze** on branch
`instrument-1` (its own worktree; sha256
`3c7e9cd9ea7a6e6074429bb8ad939a58af5ad72486c3a29b4fcc4898f288ed31` as of 2026-09-20;
registered in the chuk-experiments DB under the programme `larql` as
`head-obs-1-per-head-observation`, planned), written from the INSTRUMENT-1a/1b calibration
evidence on Gemma 3 4B and 12B. **Decision (2026-09-20): the evidence freeze is the
experiments-database authority** — the hypothesis, the witness sites, the forecasts, the
naming rule and the adjudication criteria are its — and this document is the
implementation and conformance companion that references it by that hash. The two are
reconciled by role, and one file carries each role:

- **This document is the engineering freeze**: the tap (A1–A8), the record contract
  (A9), the acceptance properties (HP1–HP7) and the implementation. The evidence freeze's
  six laws are folded in above by name — HL1 as HP3, HL2 into A2, HL3 as HP1, HL4 and HL6
  as HP6, HL5 as A9 — with its tolerances (`1e-4` on the head sum, `1e-6` on the source
  sum) and its once-only terms (the O bias) as stated there.
- **The evidence freeze is the scientific pre-registration**: its Witness A (emergence and
  amplification writes: 4B L23, L29, L31; 12B L35, L41, L44, per case with the recorded
  `S` step), Witness B (the precursor sites from INSTRUMENT-1a's frozen first divergences),
  the six fixed source classes, its forecasts HF3–HF6, and its naming rule — every head is
  named by index and site only; *decider*, *amplifier*, *transporter* and *redundant* are
  intervention-rung words — are not restated here and are read from it. Its plan fact
  carries: every emergence and amplification write is a full-span attention layer (4B 5,
  11, 17, 23, 29; 12B adds 35, 41, 47).
- The per-head draft in the shared working tree named under "Existing candidate" is
  **unattributed**: the evidence freeze's author did not write it, and nothing lands from
  that tree beside unrelated edits.
- HP7 here (one sealed HEAD-2 pair, descriptive shares) is the engineering reading of the
  descriptive attribution; the evidence freeze's witnesses are the scientific readings and
  run under its record-keeping.
- **Witness A has two writes with two meanings** (INSTRUMENT-1b, the true lens on main
  `0197af44`, same prompts and readers as 1a, carrier writes bit-equal to the 1a records;
  lens against reader 0.017 nats median, 0.048 max over 324 pairs): at the L23 attention
  write on 4B (L35 on 12B) the decision between labels is made — in every flipping case B
  jumps from rank 5k–204k to 2–1172 and A drops to 296–98k — while the head's top-1 is
  still a whitespace token; the label becomes the head's top-1 only at the L29 (L41)
  attention write, in 6/6 target and 6/6 base arms, with the label's first letter often the
  top-1 in between. So a per-head reading of Witness A asks two questions at two writes:
  which label (per-head contribution to the A-versus-B reader step, and sources) at L23/L35,
  and saying it (which heads lift the label token to rank 1) at L29/L41. The per-case lens
  ranks are in the INSTRUMENT-1 bundle's `analysis-lens4b.json` and `analysis-lens12b.json`;
  the frozen `S` steps for both writes are in the evidence freeze.
- **Order (decided 2026-09-20):** commit the evidence freeze and make its hash the database
  authority; commit this document and the implementation on `head-obs-1`; PR, CI, merge;
  run Witness A and B from the frozen bank on the committed build; bank and reconcile the
  per-head evidence; only then merge or use INTERVENE-1, already implemented on its own
  branch, for the causal follow-up — the intervention machinery does not see the heads
  before the observational witness is banked. Neither the +34% fix (a column-sliced `W_o`,
  a separate change under parity after the witness) nor the Granite re-encode blocks the
  witness. Witness B, the precursor sites, is co-equal with Witness A, not an appendix.

## Implementation (recorded 2026-09-20, after the properties above were frozen; LENS-1 merged first)

`opplan/exec/observe.rs`: `AttentionHeadRecord { position, head, kv_head, source_start,
weights, sink, values, gate, source_values }`, `StepObserver::wants_attention_heads` and
`attention_head`, `StepEvent::HeadsObserved { layer, heads }` and `HeadsUncovered { layer }`,
and `fire_head_records` — the ONE place a record is built, called by both kernels.
`backend.rs`: `PlanBackend::serves_attention_heads` (default false) and
`attention_step_observed` (default refuses), forwarded through `Arc`. `production.rs`:
`source_start` shared by kernel and tap; `aggregate_heads_keeping` retains each head's
distribution only when asked; `attend_position_tapped` computes the gate values first,
fires the records between aggregation and the gate multiply, then multiplies and projects
as before; `attention_step_tapped` is the step with an optional tap and
`attention_step` is it with `None`. `reference.rs`: the same tap inside the oracle's own
loop (`attend_position_tapped`), the dead `attend_position` wrapper removed.
`prepared.rs`: `head_projection` (A4: the head's slice placed in a zero vector of the
projection's full width and the executor's own projection kernel run over it),
`attention_output_bias`, `attention_has_heads`, the post-attention norm's kind, weight,
offset and eps. `observe_heads.rs`: `HeadReader` and `HeadStats` — the stats-level reader
that computes `c_h`, `s`, `gain`, `c′_h`, the residual, the rows and the basis projection.
`decode.rs`: the pre-token refusal (A8), the tapped softmax arm with `HeadsObserved` before
the write, `HeadsUncovered` on a layer without softmax heads (A6). `larql-inference`:
`EventKind::{HeadsObserved, HeadsUncovered, HeadWrite, HeadSum}`, `SourceStanding`, receipt
fields `head_records`, `head_layers_uncovered`, `head_failure`, `RunRecorder::with_heads`.
CLI: `vindex3 observe --heads [--heads-top-k N]`, a `heads:` summary line. The model-backed
HP5/HP7 witness is `larql-inference/examples/head_obs_witness.rs`; it drives the same
prepared production-CPU image and canonical tokenwise session.

| Property | Result |
|---|---|
| HP1 parity | PASS — logits, every carrier write (`delta`, `after`, `layer_scale`) and the structural event stream bit-identical with heads armed, reference and production, golden plan |
| HP2 record identity | PASS — one record per (layer, position, head); `kv_head = head / 3`; weights over `position + 1 − source_start`, each in `[0, 1]`, summing with the sink to one within 1e-6; the sliding window truncates on layer 0; `Σ_t weights[t] · source_values[t]` rebuilds `ctx_h` within 1e-6; **the golden plan declares an output gate, so the activated gate slice is witnessed** (the freeze's "unwitnessed gate families" note is narrower than feared: the mechanism is exercised, the real gated families are not) |
| HP3 head-sum law | PASS — residual ≤ 1e-5 at every attention write on both backends, golden plan, through the post-attention norm and the gate; the reader-projected form holds within 1e-5 per basis row |
| HP4 source split | PASS — `Σ_t W_{O,h}·(weights[t]·source_values[t])` rebuilds `W_{O,h}·ctx_h` within 1e-5 relative at every record, both backends |
| A6 coverage | PASS — the hybrid fixture (state-space and softmax layers) names its non-softmax layers `HeadsUncovered` once per position, emits records only for softmax layers, is not refused, and the head-sum law holds on the covered layers |
| A8 refusal | PASS — a loop device backend refuses before the first token with the backend named; the same session runs without heads |
| HP6 on the record | PASS — `HeadsObserved`, then `HeadSum` and one `HeadWrite` per head, then the write's stats and structural event; receipt counts; uncovered layers named once; JSON-lines round trip equal; the verb's `--heads` records it end to end |
| HP5 cost | MEASURED on Gemma 3 4B IT and Granite 4.2 3B (below), each with accepted warmed A/B/A brackets; read-only tap at noise, reconstruction +31.5% / +212.5% |
| HP7 first reading | RECORDED (below): the frozen Denmark pair's four-role donor-difference projection shares at block 14, computed in process from the borrowed head records and the image's own projection; the persisted record remains stats-level |

Gates: full `larql-vindex` library suite 4791 pass, 0 fail, 8 ignored; inference record
tests 33 pass; CLI observe tests 7 pass.

**Amendment to A9 (recorded 2026-09-20).** This build implements the *stats* level only.
`--heads` arms every softmax attention write; there is no declared-site arming and no
separate *full* level. What the record carries per head is the row (`norm`, `projection`,
`kv_head`, the top-`k` sources by weight, `sink`) and per write the residual; `c′_h` itself
is never written (the reader retains it in process only, for tests). Two consequences for a
consumer: the whole source distribution is on the record exactly when `--heads-top-k` is at
least the number of positions the head could read, since a row keeps up to `k` sources
sorted by weight; and `⟨r, c′_h⟩` for a declared reader row `r` is on the record exactly when
`--basis-rows` supplies `r`, because the head rows and the carrier stats project on the same
basis, so `⟨r, delta⟩` sits on the same record for the fraction. The full level, with `c′_h`
on request at declared sites, is owed as its own change and is not claimed here.

**Order and holds (relayed by the evidence freeze's session as Chris's, 2026-09-20).**
Pre-registration stays authoritative; freeze and commit all current evidence; commit this
reconciled engineering freeze; run Witness A and B with `--heads`; adjudicate against the
frozen criteria; commit the evidence; merge. Chris commits. Two holds: the +34% stays a
falsified HF2/HF7 with the padded `W_o` as the next engineering hypothesis, never restated
as success; and HEAD-OBS-1 is not expanded after A/B — if the decomposition works it
closes, and source-token attribution ("what did the head read") is a new rung with its own
freeze, not a results addendum.

**HP5 (recorded 2026-09-20).** Release `head_obs_witness`, production CPU,
`LARQL_CPU_MAX_FORMAT=bf16`, `The capital of France is` plus the same eight fixed greedy
tokens in every arm. Each image is prepared once. After the no-head arm reaches a
consecutive-run plateau within 1%, each cost is one `NoopObserver / candidate /
NoopObserver` bracket; a bracket counts only when its two controls agree within 1%.
Artifacts carry every attempted bracket, the exact ids and acceptance flags.

| Subject / arm | Positions | Baseline controls | Candidate | Drift | Overhead | Records | Worst residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B, read-only borrowed tap | 14 | 1.1000 / 1.0933 s | 1.0939 s | .61% | **−.25%** | 3,808 | — |
| Gemma 3 4B, HP3 reconstruction | 14 | 1.0902 / 1.0874 s | 1.4320 s | .26% | **+31.52%** | 3,808 | 8.18e-7 |
| Granite 4.2 3B, read-only borrowed tap | 13 | .9462 / .9516 s | .9498 s | .58% | **+.09%** | 20,800 | — |
| Granite 4.2 3B, HP3 reconstruction | 13 | .9193 / .9137 s | 2.8640 s | .60% | **+212.49%** | 20,800 | 3.26e-7 |

Exact artifacts: the driver's hp5-gemma.json artifact (kept with the driver in a local forensic commit, not published) and
the driver's hp5-granite.json artifact (kept with the driver in a local forensic commit, not published). The first candidate merely consumes every borrowed
field with no copy or projection; on both subjects its cost is indistinguishable from
run-to-run noise. The second performs the shipped `HeadStats` reconstruction at every
softmax write. It closes the real Gemma post-attention-norm branch and Granite's `s = 1`
branch far inside the `1e-4` law.

**HF7 is falsified on reconstruction cost.** The cause is A4 as implemented: the head's
slice is placed in a zero vector of the full input width and the image's whole `W_o`
product runs, so an eight-head Gemma layer pays eight `o_proj`-class products per write
and a forty-head Granite layer pays forty. A column-sliced projection with the same pinned
realisation is the next engineering hypothesis; it remains a separate change under parity,
not a redefinition of this result. The borrowed kernel tap itself satisfies HF1's cost
forecast on both subjects.

**Additional measurements (this session, `vindex3 observe --heads`, recorded after the
execution commit 58eb342d; the HP5 and HP7 sections above were recorded by the owner's
witness driver, committed separately as 74181943 and not part of this rung's gated paths).**
Realisation is not a constraint of this rung: under the **default policy** (Q8-requantised
projections and head, the witness chain's realisation) on Gemma 3 4B IT, `The capital of
France is` plus 8 greedy tokens, the generated ids are identical with and without heads,
all 952 carrier-stats rows are bit-equal between the two arms, the provenance fingerprint
is identical between them and different from the bf16 arm's, the worst head-sum residual
is 1.13e-6 over 476 writes, and the stats-level cost is +9% (103.9 → 113.4 ms per
position, single runs) against +34% under the bf16 cap on the same prompt. On Granite 4.2
3B (`granite-4.2-3b.s6.vindex3`, schema v6, model id `b7e94730…`; the v5 sibling
`granite-4.2-3b.vindex3` still refuses by declaration) with this session's gated binary:
generated ids identical across arms, 1040 carrier-stats rows bit-equal, 20800 records per
run, worst residual 3.5e-7 under the default policy and 3.26e-7 under bf16 (the driver's
number, reproduced), cost +289% under the default policy on forty query heads — the
reconstruction's price scales with the head count, as the padded-`W_o` cause predicts.
The driver's HP7 artifact was reproduced from its committed source: every four-role share
within 1e-6 of the driver's hp7-denmark.json artifact (kept with the driver in a local forensic commit, not published), residuals identical.

**HP7 (recorded 2026-09-20).** Gemma 3 4B IT, production CPU,
`LARQL_CPU_MAX_FORMAT=bf16`; recipient A `The currency of Denmark is`, donor B `The
capital of Denmark is`, both stepped as six ids with BOS. The witness consumes the
borrowed records in process, applies each source through the prepared image's own
per-head projection, and groups positions as BOS / RELATION / ENTITY / LAST. It does not
add source vectors to the stats-level run record. The exact artifact is
the driver's hp7-denmark.json artifact.

The head-sum residual is `1.20e-7` on A and `9.97e-8` on B. The worst source-sum
residual over both prompts and every head is `2.44e-7`; each row's four shares sum to one
within `3.00e-7`.

| Head (KV) | BOS | RELATION | ENTITY | LAST | Sealed 336-pair HEAD-2 label |
|---|---:|---:|---:|---:|---|
| H0 (0) | .020 | **.487** | .066 | .428 | carrier, TWO_ROLES(RELATION, ENTITY) |
| H1 (0) | .001 | .095 | **.844** | .060 | SINGLE_ROLE(ENTITY) |
| H2 (1) | .131 | .185 | .245 | **.438** | opposer, MIXED |
| H3 (1) | .075 | **.543** | .037 | .345 | carrier, SINGLE_ROLE(RELATION) |
| H4 (2) | -.054 | .310 | **.433** | .311 | SINGLE_ROLE(RELATION) [DIVERGE] |
| H5 (2) | .175 | .216 | .090 | **.519** | SINGLE_ROLE(RELATION) [DIVERGE] |
| H6 (3) | .240 | **.279** | .247 | .234 | carrier, TWO_ROLES(RELATION, BOS) |
| H7 (3) | .245 | **.568** | -.102 | .289 | opposer, SINGLE_ROLE(LAST) |

Read at the pre-registered strength: this is one pair under a different execution
realisation, and agreement with the sealed MLX-f32 aggregate was explicitly not forecast.
H1 and H3 reproduce the aggregate's dominant entity and relation roles. H2 remains mixed.
H0 and H6 distribute their one-pair difference more broadly. H4, H5 and H7 disagree with
the aggregate's dominant role, which is a finding about this pair/realisation rather than
a failure of the observation rung. The identities close tightly enough that the
disagreement cannot be assigned to an unclosed decomposition.

## Out of scope

MLA and conv-QKV taps — both already hold the values, and routing them to the observer
with MLA's absorbed (shared-key) semantics and conv-QKV's plane is V3-HEAD-OBS-2, named
here so a receipt can name the gap; non-softmax families (no heads by declaration); the
batch prefill path; the Kimi Metal stack; per-head readouts through the head (that is
ATTR-1D, which consumes this rung); per-head interventions (`H_h`, `LOO_h`: ATTR-1C,
which needs V3-INTERVENE-1); per-head statistics on the record; any per-head tensor
file; a Python surface.

---

## Verdict rule

V3-HEAD-OBS-1 is complete only when HP1–HP6 PASS and HP7 is recorded with its declared
arm and pair.

**Closed 2026-09-20.** HP1–HP6 PASS on the golden plan (both backends) and on the real
subjects; HP5 and HP7 recorded above. The evidence freeze's Witness A and B ran on the
execution commit `58eb342d` under the default policy (22 records, `--heads-top-k 89` with
the reader rows): every record passed the schema gate, the head-sum residual was at worst
1.03e-6 on 4B and 6.6e-7 on 12B, the reader-projected reconstruction of each site's step
from the head rows held within 5.1e-6, sources plus sink summed to one within 1e-7, the
carriers were bit-equal to the INSTRUMENT-1a records at all 89 positions, and the
fingerprint equalled the 1b record's. The adjudication (HF1, HF3, HF4 and HF5 held six of
six; HF2 failed on cost, with the default-policy and tap-versus-reconstruction measurements
recorded beside it) and the findings, by head index and site only, are in the evidence
freeze's results on branch `instrument-1` and the INSTRUMENT-1 bundle; the database entry
`head-obs-1-per-head-observation` is completed. The candidate set for V3-INTERVENE-1 is
named there without role words. It unlocks ATTR-1D (source-token and head attribution on the executor)
and, with V3-INTERVENE-1, ATTR-1C. If HP3 cannot be brought inside `1e-3` on any
realization, the rung records that the per-head split is not reconstructible on that
realization and ATTR-1D declares the same; it does not relax the identity.
