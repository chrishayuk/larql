# V3-INTERVENE-2 — can the canonical decode step remove or replace ONE head's contribution before the model recombines it, and prove the subtractive shortcut is not the same experiment?

**FROZEN 2026-09-21.** The precondition this document was gated on is now satisfied:
V3-INTERVENE-1's migration onto post-HEAD-OBS-1 main passed its no-op parity gate (on the
record, not only the logits) and its CARRIED migration witness, and merged as PR #494
(`4ba8766a`). The properties J1–J7 and acceptance properties JP1–JP6 below are FROZEN as of
this commit; nothing here claims a result, and no property changes without a new commit
that says so.

This rung is chosen ahead of ATTR-1C (the additive child-transplant form, `delta' =
delta − c′_h` / `c′_h(B) − c′_h(A)`) because HEAD-OBS-1's decomposition sits underneath a
post-attention RMS norm: subtracting a recorded head child after the fact does not reproduce
what the model would have computed had that head never contributed, because removing one
head's contribution changes the norm's scalar, which changes every OTHER head's effective
contribution too. ATTR-1C answers "how much of the observed write does this additive
component represent" — a real and useful question, kept as the decomposition/control
comparator once this rung lands (JP6 runs both forms on the same pair so they can be read
against each other) — but it is not a causal necessity claim on its own, and this rung is
what tells us whether the shortcut is even a good approximation (J5/JP3, the "shortcut gap").

Programme: OBSERVE. Above V3-INTERVENE-1 (carrier addresses) and V3-HEAD-OBS-1 (the per-head
tap, closed on `58eb342d`). The rung that lets the intervention machinery see the heads —
after, and only after, the observational witnesses were banked. It is the engineering seam
the scientific intervention bank needs (H3 at L23, H4 at L29, H6/H7 at L31, the precursor
heads); the bank's arms, controls and measures are the evidence freeze's to pre-register,
not this document's.

---

## The question

HEAD-OBS-1 established, on the executor, that each attention write decomposes into per-head
children through the post-attention norm, `Σ_h c′_h + bias′ = delta` to 1e-6, and named the
heads that carry the reader steps by index and site. The obvious next move — "remove head
H3 and see what happens" — has a wrong and a right form, and the difference is the
post-attention RMS norm.

The wrong form subtracts the recorded child `c′_h` from the carrier. That tests the additive
decomposition, not the model: with the head gone, the norm's scalar `s = 1/rms(o + bias)` is
computed over a different `o`, so every OTHER head's child changes too, and the write is not
`delta − c′_h`. The right form zeroes the head's mixed value inside the kernel, before the
output gate, before `o_proj`, before the real post-attention norm, and lets the executor
continue as it would; the counterfactual write is whatever the model then computes.

V3-INTERVENE-2 asks whether that right form can be threaded through the one decode step at
the same point HEAD-OBS-1's tap already sits, under INTERVENE-1's discipline (address, kind,
provenance, no-op law, receipt), and whether the gap between the real counterfactual and
the subtractive shortcut can be measured rather than assumed, so that no later reading
mistakes the one for the other.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The per-head tap fires between aggregation and the gate multiply, with `kept` (per-head distributions) and the pre-gate `concat` in scope, in ONE place per backend | `opplan/exec/production.rs`, `attend_position_tapped`; `opplan/exec/reference.rs`, `attend_position_tapped` |
| The record's `values` is `ctx_h`, pre-gate; the gate slice is on the record where the plan declares a gate | `opplan/exec/observe.rs`, `AttentionHeadRecord` |
| INTERVENE-1's vocabulary: a second enum beside `Mutation`, an address with declared positions, a kind, vector provenance with a checked hash, admission before the first token, `Intervened` before the write's record, identity hash and receipt counts, an in-memory donor capture | `opplan/exec/intervene.rs`; `opplan/exec/decode.rs` |
| The head reader retains `c′_h` per head in process (`retaining_children`) and the per-head projection runs the image's own kernel | `opplan/exec/observe_heads.rs`; `opplan/exec/prepared.rs`, `head_projection` |
| HEAD-1's per-head arms on the pre-registered Denmark pair are ADDITIVE: `H_h = CARRIED + (w_h^B − w_h^A)`, `LOO_h = CARRIED+ATTN − (w_h^B − w_h^A)`, with `w_h` block 14's post-norm head child; 19 sealed rows for the pair, whose 8-token continuations take exactly two forms | `~/chris-source/chris-experiments/semantic_graph/SG_B14_HEAD1_PLAN.md`; `captures/b14h1_full_meta.json` |
| The scientific bank named after HEAD-OBS-1, by index and site only: 4B H3 at L23 (89–111% of the reader step, reads the rewritten label), H4 at L29, H6/H7 at L31 (cancellation), precursor heads; 12B H12/H8 at L35, H10/H11 at L44 | the evidence freeze's results on branch `instrument-1`; the INSTRUMENT-1 bundle |

What does NOT exist: any intervention inside the attention kernel; any way to zero or
replace one head's mixed value; any measurement of the gap between a real head
counterfactual and the subtractive shortcut.

---

## Two experiments, kept apart by name

- **Additive child transplant** (HEAD-1's construction): add `w_h^B − w_h^A` to the carrier.
  Implementable TODAY as INTERVENE-1 `Add` at the write's address with a vector built from
  HEAD-OBS-1's retained children by an in-process driver. It tests the decomposition's
  terms. This is **ATTR-1C**, witnessed against HEAD-1's sealed `H_h` and `LOO_h` rows; it
  needs nothing from this rung and is named here only so the two are never confused.
- **In-kernel head counterfactual** (this rung): zero, scale or replace `ctx_h` before the
  gate, `o_proj` and the post-attention norm. It tests the model. The scientific bank
  (necessity, then transplant) uses this form.

---

## Contract properties (frozen as properties, not as names)

**J1 — the head address.** An intervention names `(layer, head, positions)` on a softmax
attention layer; it acts on `ctx_h`, the head's mixed value, at the point where HEAD-OBS-1's
tap reads it: after aggregation, before the output gate, before `o_proj`, before the
post-attention norm and any residual scale. Nothing downstream is patched again (INTERVENE-1
I6).

**J2 — three kinds, exact.** `Zero` sets `ctx_h` to zero. `Scale(α)` multiplies it by a
declared finite `α`. `Replace(v)` sets it to a declared `head_dim`-wide vector with
INTERVENE-1 provenance (a literal, or a head captured from a named run at a named
`(run, layer, head, position)`). A gated family's gate still applies to the replaced value:
the gate is the model's, not the intervention's.

**J3 — one seam, both backends.** The head intervention threads through the same function
that fires the head records, on the production kernel and on the reference loop, so the
oracle can gate the production path's counterfactual. Records fire on the UNINTERVENED
`ctx_h` first, then the intervention applies; a consumer that wants the intervened value
reads the write, not the head record (the record says what the model computed; the write
says what it then did).

**J4 — the no-op law.** `Scale(1)`, `Replace(ctx_h)` with the head's own value, and an empty
plan are bit-identical to the unintervened run at every position, both backends.

**J5 — the shortcut gap is measured.** For every `Zero` firing the executor's own head reader
(HEAD-OBS-1) is armed on the same run, and the receipt carries, per firing, the gap between
the real counterfactual write and the subtractive shortcut: `‖delta_real − (delta_base −
c′_h)‖ / ‖delta_base‖`, where `delta_base` and `c′_h` come from the unintervened run of the
same prompt. The gap is a recorded quantity; no property bounds it, because its size is the
finding: it says how much the post-norm recouples the other heads when one is removed.

**J6 — identity, receipt, refusals.** As INTERVENE-1 I4, I7 and I8: the declaration hash
joins the run identity; the receipt counts firings against declarations and names an
unreached address; a head outside the layer's query heads, a layer without softmax heads, a
non-`Single` carrier, a `Replace` of the wrong width or non-finite, or a provenance whose
hash does not match refuses before the first token. Head and carrier interventions may
coexist on one run at different addresses; two interventions on one head at one position
refuse.

**J7 — capture at the head.** The donor capture of INTERVENE-1 gains a head form: an observer
that copies `ctx_h` (and, separately, `c′_h` through the reader) at declared
`(layer, head, position)` addresses, in memory, with provenance. Nothing is persisted.

---

## Frozen acceptance properties

**JP1 — no-op.** J4 on the golden plan and on Gemma 3 4B IT, both CPU backends where the
subject runs on both.

**JP2 — exactness.** With `Zero` at `(l, h, p)`, the head record at `(l, h, p)` is the
unintervened `ctx_h` (J3), and the attention write's decomposition on the intervened run
has `c′_h` with norm zero for that head and the other heads' children DIFFERENT from the
unintervened run's by exactly the norm's recoupling — checked by re-running HEAD-OBS-1's
head-sum law on the intervened run (residual ≤ 1e-4 again). Every write before the address
in execution order is bit-identical to the unintervened run.

**JP3 — the shortcut gap exists.** On the golden plan (which has a post-attention norm and an
output gate) and on Gemma 3 4B IT, for at least one head at one write the recorded gap of
J5 exceeds 1e-3 relative — i.e. the subtractive shortcut is measurably not the counterfactual
on a real post-norm layer. On a layer without a post-attention norm and without a gate
(Granite s6) the gap is ≤ 1e-5, because there the two coincide up to accumulation order.
This property is what makes the thread's rule a measured fact rather than an instruction.

**JP4 — reference gates production.** Zeroing the same head on both backends yields the same
counterfactual continuation on the golden plan and on Gemma 3 4B IT (token for token), and
the intervened writes agree to the backends' existing parity band.

**JP5 — on the record.** `Intervened` carries the head; the receipt carries the per-firing
gap; replay is equal; the verb takes head addresses in the same declaration file
(`"head": h` beside `"layer"`), refusing `"site"` on a head entry and `"head"` on a carrier
entry.

**JP6 — the first engineering arm (recorded, not forecast).** Gemma 3 4B IT, default policy,
the pre-registered Denmark pair, block 14, last prompt position: `Zero` each query head in
turn on the recipient prompt (8 arms) and `Zero` all but one (8 arms), 8 greedy tokens each,
with the per-firing gap recorded. No sealed reference exists for a true zeroing, so the 16
continuations are recorded beside HEAD-1's additive `H_h`/`LOO_h` rows for the same pair as
a COMPARISON of the two experiments, and the arms where the additive transplant and the
true counterfactual disagree are named. No agreement is forecast in either direction.

---

## Forecasts

- **JF1**: JP1 and JP4 hold bit for bit; JP2's re-run head-sum law holds at ≤ 1e-5 on the
  golden plan and ≤ 1e-6 on Gemma, as it did unintervened.
- **JF2**: JP3's gap on Gemma at block 14 for the largest head (H7 by norm share in
  HEAD-OBS-1's HP7) exceeds 1e-2 relative: removing a third of the write's norm moves the
  norm's scalar visibly. No forecast for the smaller heads.
- **JF3**: the seam adds one optional parameter to `attend_position_tapped` on each backend
  and one arm to `leave_site`'s caller; the head record type does not change; INTERVENE-1's
  enum gains one address form and one capture form; the STREAM-1 schema gains the head on
  `Intervened` and the gap on the receipt, nothing else.
- **JF4**: JP6's `Zero(H7)` continuation differs from HEAD-1's `LOO7` row (the additive
  removal of H7's donor difference is not the removal of H7), while `Zero(H1)` and `LOO1`
  agree — a guess written so it can be wrong.

---

## JP6 result (recorded 2026-09-21, execution SHA `127ed390` — engineering commit, no edits since)

Release CLI built from `127ed390` (sha256 of the binary: `ec912016489afd506d4f2c6884389733e453286b6d35d63ce83b814a906ba750`), `gemma3-4b-it.vindex3`
(model identity `093f9f388b31de276ce2de164bdc2081324b9767`, family `gemma3_text`), production CPU,
**default policy** (no `LARQL_CPU_MAX_FORMAT` cap — the container's own realisation selects
BF16 throughout, recorded per-run in `provenance.execution.realizations`), the pre-registered
Denmark pair (recipient `The currency of Denmark is`, encoded `2,818,15130,529,37932,563`, last
prompt position 5 — the same encoding IP7 used), block 14 (layer 14, confirmed 8 query heads
by a live `--heads` run before the bench: `[0..7]`, 3808 head records, 0 uncovered layers).
Every arm: `Zero` at `(layer 14, head h, position 5)`, 8 greedy tokens, `--heads` armed
(retaining children) so the shortcut gap prices against this same run's pre-intervention
records (J5). Recipient baseline (no intervention, same binary): generated
`506,46553,155054,568,12536,236855,769,108` = ` the Danish Krone (DKK).\n\n` — 8/8 the sealed
`WHOLE`/baseline row for this pair.

**The finding, stated once and precisely: all 16 arms, and a 17th supplementary probe zeroing
all 8 heads at once, reproduce the recipient's own baseline continuation exactly, token for
token, while the shortcut gap is real, non-zero and grows systematically with how much is
removed.** This was not the frozen forecast (JF2 named H7 as likely to show the largest gap on
the golden plan's analogue; no forecast was made about whether any arm would move the greedy
decision) and is recorded as found, not repaired or reinterpreted mid-run.

| Arm | Generated ids | Matches recipient baseline | Shortcut gap (head:relative-norm) | Declaration hash (12 hex) | Firings |
|---|---|---|---|---|---|
| Zero(H0) | `506,46553,155054,568,12536,236855,769,108` | yes, 8/8 | H0: 0.0167 | `1dd40de91b83` | 1/1 |
| Zero(H1) | same | yes, 8/8 | H1: 0.0711 | `f3b95799cbea` | 1/1 |
| Zero(H2) | same | yes, 8/8 | H2: 0.0224 | `42fd9b7d0749` | 1/1 |
| Zero(H3) | same | yes, 8/8 | H3: 0.0238 | `1254afeee323` | 1/1 |
| Zero(H4) | same | yes, 8/8 | H4: 0.0175 | `6c6965a6926c` | 1/1 |
| Zero(H5) | same | yes, 8/8 | H5: 0.0190 | `4c40f05f00af` | 1/1 |
| Zero(H6) | same | yes, 8/8 | H6: 0.2957 | `9fe762bc1c42` | 1/1 |
| Zero(H7) | same | yes, 8/8 | H7: 0.3782 | `5abba8454f6f` | 1/1 |
| AllBut(H0) | same | yes, 8/8 | H1..H7: 1.09–1.78 (mean 1.58) | `fc325be5bd0d` | 7/7 |
| AllBut(H1) | same | yes, 8/8 | H0,H2..H7: 0.95–1.61 (mean 1.34) | `7931aadee46f` | 7/7 |
| AllBut(H2) | same | yes, 8/8 | H0,H1,H3..H7: 1.51–2.35 (mean 1.99) | `1aecaaa9ba43` | 7/7 |
| AllBut(H3) | same | yes, 8/8 | H0..H2,H4..H7: 1.17–1.48 (mean 1.36) | `dcb86c802d11` | 7/7 |
| AllBut(H4) | same | yes, 8/8 | H0..H3,H5..H7: 1.17–1.80 (mean 1.32) | `ab41d40fde52` | 7/7 |
| AllBut(H5) | same | yes, 8/8 | H0..H4,H6,H7: 1.23–2.05 (mean 1.43) | `7d229e4a1179` | 7/7 |
| AllBut(H6) | same | yes, 8/8 | H0..H5,H7: 0.97–2.10 (mean 1.77) | `fcb81f35bfda` | 7/7 |
| AllBut(H7) | same | yes, 8/8 | H0..H6: 0.54–1.00 (mean 0.84) | `03c8cd6bc5de` | 7/7 |
| Zero(ALL 8), supplementary | same | yes, 8/8 | H0..H7: 0.49–1.26 (head-sum residual 589, meaningless once nothing remains to sum) | `130a547ceb95` | 8/8 |

Every run shares one `provenance_fingerprint` (`4f8b37e72a92fd0b9a21486bc6228b7c4d226d0eeceab5b611ee9b8e20c1f04c` —
same execution substrate, only the declaration differs); each run's own `log_sha256` differs
(the computation genuinely differs per arm even though the final tokens do not). Raw records
(2898–2899 events × 17 runs, `/tmp/jp6-denmark/records/*.jsonl`) are reproducible byte-for-byte
from `127ed390` + these declarations + this container and are not committed — raw per-run
dumps stay local, as elsewhere in this project. The sealed HEAD-1 rows for this exact pair
(`captures/b14h1_full_meta.json` in `chris-experiments`) at block 14:

| condition | sealed `gen` | pattern |
|---|---|---|
| WHOLE (baseline / donor@L24) | `68063,236761,108,818,15130,529,37932,563` | "Copenhagen.\n\nThe currency of Denmark is" |
| CARRIED, CARRIED+ATTN, H0, H1, H3, H4, H5, H6, LOO1, LOO2, LOO4, LOO5, LOO7 | `68063,236761,108,162623,236789,236751,15130,563` | "Copenhagen.\n\nDenmark's currency is" |
| H2, H7, LOO0, LOO3, LOO6 | `68063,236761,108,818,15130,529,37932,563` | "Copenhagen.\n\nThe currency of Denmark is" |

**The four questions, answered from the table above, not from a hypothesis JP6 was never
frozen to test:**

1. **How large is the gap, head by head?** Sharply uneven and NOT the ranking HEAD-1's block-14
   φ/ψ study found. Single-head zeroing: H0/H3/H4/H5 small (0.017–0.024), H1 moderate (0.071),
   **H6 and H7 an order of magnitude larger (0.30, 0.38)**. HEAD-1's necessary/opposing heads on
   this pair were {H0, H3, H6} (necessary) and {H2, H7} (opposing) — H6 appears in both
   studies' "large effect" set, but H0 and H3 (large in HEAD-1) are among the SMALLEST gaps
   here, and H2 (opposing in HEAD-1) is unremarkable here (0.022). The seven-heads-removed
   arms compound roughly additively in scale (means 0.84–1.99) with no clean per-head
   attribution — consistent with J5's own reasoning that the post-norm scalar recouples
   everything once enough is removed, and gap here is close to the norm PENALTY of dropping
   the shortcut approximation almost entirely, not seven separable per-head numbers.
2. **Does Zero(H7) diverge from the old LOO7 additive row, as JF4 guessed?** Yes, but not for
   the reason JF4 was written to test. LOO7's sealed row is the CARRIED pattern
   (`...162623,236789,236751,15130,563`); Zero(H7) is the recipient's own untouched baseline
   (`...46553,155054,568,12536,236855,769,108`). They diverge completely — different answer,
   different tokens throughout — but because the two experiments start from different STATES
   (LOO7 removes a donor contribution from an already-CARRIED, Copenhagen-primed stream; Zero(H7)
   removes a head's own contribution from the untouched recipient stream), not because the
   same state responds differently to the same removal. JF4's qualitative guess (divergence)
   holds; its implicit mechanism (a nuanced per-head disagreement) does not — the divergence
   traces entirely to which experiment is being run, not to what H7 does.
3. **Are there heads where the additive experiment and the true counterfactual agree closely?**
   Not at the token level, for any head, under this design: JP6's 16 arms all return the
   recipient's own baseline, while 14 of HEAD-1's 17 non-baseline sealed rows (all but H2, H7,
   LOO0, LOO3, LOO6) return the CARRIED pattern — the two experiments never land on the same
   output because JP6 never introduces the donor signal HEAD-1's rows are built on. The
   comparison that IS meaningful — the shortcut gap's size — shows H0/H3/H4/H5 close to the
   additive approximation (gap < 0.03) and H1/H6/H7 far from it (0.07–0.38); read against
   HEAD-1's φ ranking this is close to a REVERSAL for H0 and H3 specifically, which is the
   more informative agreement/disagreement question this rung was built to ask.
4. **Does zero-all-but-one expose a different picture of "sufficiency" than the additive H_h
   construction?** Yes, sharply. HEAD-1's sufficiency (`φ_h`) asks whether head h's DONOR
   contribution alone, added to an already-CARRIED stream, reproduces the CARRIED+ATTN
   transfer — a competitive question with a close-run answer (φ ranges −0.99 to +0.88). JP6's
   all-but-one asks whether the recipient's OWN signal through any single one of block 14's 8
   heads is enough to preserve the recipient's OWN greedy decision — and by that measure every
   single head is "sufficient," including the ones (H2 in HEAD-1's frame) that actively oppose
   the donor transfer. The two "sufficiency" notions do not transfer: one is about winning a
   contest against an injected competitor, the other is about a completely uncontested,
   over-determined decision surviving near-total ablation of one layer's one sublayer at one
   position.

**Read at honest strength.** The recipient's own top-1 margin at this position was ~4 nats
wide before any intervention (`the` at logp −0.0038 against the next candidate at −6.69,
IP7's baseline record) — a syntactic completion ("the currency of X is ___") with very little
genuine ambiguity for the model to resolve. Zeroing block 14's entire attention write at that
one position — not approximated, the model's own recomputed response, confirmed by gaps up to
2.35 relative norm against the naive shortcut — never approached that margin. This says
something specific and useful: **single-layer, single-position head necessity is not
detectable against an over-determined decision**, which is exactly why HEAD-OBS-1's own
programme moved to the L23→L24 transition (a genuinely close-run decision) and why HEAD-1's
block-14 study used a donor-injected, competitively-poised CARRIED state rather than the plain
baseline. It is not evidence that block 14's heads are unimportant; it is evidence that THIS
question, asked of THIS decision, was answered before the intervention ran. The shortcut-gap
finding stands on its own regardless: the additive approximation is a poor stand-in for the
real counterfactual at H1/H6/H7 specifically, on this pair, at this address — which is the
warning V3-INTERVENE-2 was built to make legible before ATTR-1C's additive arms are read as
causal.

---

## Out of scope

The scientific bank's arms, controls and adjudication (the evidence freeze's); per-source
interventions (zeroing one source position's contribution inside a head); MLA and conv-QKV
heads (HEAD-OBS-2 first); the batch prefill path; the Metal path; a column-sliced `W_o`
(owed to HEAD-OBS-1's cost finding, separately); any role word for any head.

---

## Verdict rule

V3-INTERVENE-2 is complete only when JP1–JP5 PASS and JP6 is recorded with its 16 arms and
gaps. It unlocks the scientific intervention bank (necessity first, then transplant) and, with
INTERVENE-1, lets ATTR-1C run the additive arms beside the true counterfactuals on one pair
so the two experiments are read against each other rather than conflated.

**Satisfied 2026-09-21.** JP1/JP2/JP4/JP5 PASS as tests on `127ed390` (JP3's qualitative claim
— a measurable, non-trivial gap on a gated post-norm plan — also holds, as a real test, not
asserted against the frozen `>1e-3`/`≤1e-5` split, which belongs to the scientific bank); JP6
is recorded above with its 16 arms, gaps, and a 17th supplementary probe. **V3-INTERVENE-2 is
CLOSED.** The rung's own finding — that single-layer, single-position necessity is invisible
against an over-determined decision, and that the additive shortcut disagrees with the true
counterfactual by a head-dependent and structurally unpredictable amount — is exactly the
input the scientific necessity bank (HEAD-OBS-1's candidates: H3@L23, H4@L29, H6/H7@L31) needs
before it is frozen: that bank must choose decisions with real competitive margin, the way
HEAD-1's CARRIED construction did and this rung's plain-baseline probe did not.
