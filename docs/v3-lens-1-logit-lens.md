# V3-LENS-1 — can the executor's own head read an intermediate carrier faithfully, and at a declared price?

Pre-registered 2026-09-20, before any implementation. Properties and forecasts below are
FROZEN. Nothing here claims a result.

Programme: OBSERVE. Rung 3, above V3-OBS-1 (#484) and V3-STREAM-1 (#485), beside the
`observe` verb (#486). The first Anatomist-style research lens: what the model would say
at each depth, computed by the model's own head and nothing else.

---

## The question

V3-OBS-1 deliberately refused a logit lens. Its selected-token probe is a raw dot product
of the carrier against supplied rows, with no final norm, no head multiplier, no softcap,
and it says so. That was right for a cheap observation statistic, and it is not what a
researcher means by "P(Paris) at layer 24".

A **true** logit lens applies, to the carrier at an intermediate depth, exactly the
operations the executor applies at the exit: the prepared final norm and the prepared
output head, with the head's multiplier and softcap, on the head's real weights (tied
or not, in whatever representation the image pinned). Anything less is a different
reader with a different name.

V3-LENS-1 asks whether that can be done **through one code path** — the exit's own — so
that the lens at the last layer *is* the executor's logits rather than agreeing with
them, and whether its cost, one head pass per tapped site, can be declared and measured
rather than hidden inside "observation".

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The exit: `final_hidden = final_norm.apply(backend, exit_hidden)` then `backend.output_head(weight.slice(), vocab, hidden, &final_hidden, op.multiplier, op.softcapping)` | `opplan/exec/decode.rs`, the `logits =` match at the end of `run` |
| The same two operations on the batch path | `opplan/exec/mod.rs` |
| `PreparedOperands::final_norm()` and `::output()` are `pub(super)` — reachable inside `exec`, not by a consumer | `opplan/exec/prepared.rs` |
| The carrier at every write, borrowed, with the layer scale where the program has one | `carrier_write` tap (V3-OBS-1); the layer output is `layer_scale × after` (C5) |
| A consumer that records events with a run-scoped sequence and a receipt | `RunRecorder` (V3-STREAM-1) |
| The head's cost class in the timing ledger | `opplan/exec/timing.rs` |

---

## Contract properties (frozen as properties, not names)

**H1 — one head path.** The executor exposes one function on the prepared image that
takes a `[hidden]` carrier and a backend and returns the head's logits by applying the
prepared final norm and the prepared output head. The exit calls that same function.
There is no second implementation of "the head" anywhere.

**H2 — the lens reads the layer output, not the write.** On a site whose program applies
a layer scale after the write, the lens reads `layer_scale × after`, because that is
what the next layer and the exit would read. The lens says which site it read.

**H3 — the lens is a consumer.** It is a `StepObserver` that, at the sites it was
declared for, copies the carrier and calls the head function. It never changes
execution; V3-OBS-1's parity gate covers it like any observer.

**H4 — sites are declared, the price is per site.** A lens is armed with an explicit
site set: every layer, every k-th layer, or a list; attention site, FFN site or both.
Its cost is one head pass per armed site per token, and the record says how many.

**H5 — outputs are the distribution's facts, not a probe's.** Per armed site and per
declared token, the lens records the log-probability under the full log-softmax and the
rank; optionally the top-k ids. Full logits are never persisted by this rung.

**H6 — the head is the image's head.** Whatever representation the image pinned for the
output head (bf16, f32, Q8 under the default policy) is what the lens computes with, and
the run's provenance already names it. The lens does not carry a head of its own.

---

## Frozen acceptance properties

**LP1 — parity.** Logits at every position with the lens armed on every site are
bit-identical to the unobserved run, on the reference and production CPU backends.

**LP2 — the anchor.** At the last layer's FFN site, the lens's logits for a position are
**bit-identical** to the executor's own logits for that position, on the golden plan and
on every real subject run. This is H1 made observable: the same function on the same
vector.

**LP3 — the exit is the function.** The exit's code path is the shared function (a
refactor, witnessed by the existing decode-vs-batch and observed-vs-unobserved gates
staying bit-identical before and after).

**LP4 — a proper distribution at every depth.** At every armed site the full
log-softmax is finite and the declared tokens' log-probabilities are at most zero and
sum, with the rest, to one within f64 rounding.

**LP5 — cost is measured, not described.** Token wall with the lens armed on every FFN
site minus token wall with `NoopObserver`, on Granite 4.2 3B and on Gemma 3 4B IT,
release, production CPU; reported per token and per armed site. The forecast below
says why Gemma's number will be large.

**LP6 — the lens on the record.** A lens readout is a recorded event with its sequence,
site, and per-token log-probability and rank; replay is equal; the receipt covers it.

**LP7 — the first real reading.** Gemma 3 4B IT, `The capital of France is`, the lens
armed on every FFN site for the tokens ` Paris` and ` France` at the final prompt
position: the record carries their log-probability and rank at every layer. **No shape
is forecast**: the ADDRESS-BUILD finding (relation early, entity late) is a prior about
Gemma 3 and is what this reading will be compared against, not assumed.

---

## Forecasts

- **LF1**: LP2 holds bit-for-bit on the golden plan (both backends), Granite and Gemma 3
  4B (production).
- **LF2**: cost per armed site is one output-head pass. On Gemma 3 4B that is a
  262,208 × 2,560 projection, roughly 0.67 GMAC per site; armed on all 34 FFN sites
  that is about 23 GMAC per token against a forward of a few GMAC, so a full-depth lens
  on Gemma is expected to cost **several times** the forward it observes. That is why H4
  exists. Granite's 49,152 × 2,560 head is about 5× cheaper per site.
- **LF3**: no change to the V3-OBS-1 observer contract or the STREAM-1 record schema
  beyond one new event kind; if either needs more, it is recorded as a finding.
- **LF4**: the CLI gains `--lens-tokens`, `--lens-sites` and nothing else.

---

## Amendments (recorded from the witnesses and the runs)

- **2026-09-20, LF2's magnitude.** The forecast said a full-depth lens on Gemma would cost
  "several times" the forward. Measured: on Granite 4.2 3B the lens on all 40 FFN sites
  turned 183.5 ms per position into 1305.7 ms, about 28 ms per head pass, 7.1× the
  forward; on Gemma 3 4B it turned 134.0 ms into 4132.0 ms, about 118 ms per head pass,
  **30.8×** the forward. The direction was right, the size was not: one Gemma head pass
  costs roughly as much as the entire forward at production speed, so 34 of them cost 30
  forwards. H4's declared, priced sites are not a nicety on this model; they are the
  only way to use the lens at all on long prompts.

## Results (recorded 2026-09-20, after the forecasts were frozen)

Implementation: `PreparedOperands::head_logits` and `head_over_normed` (the one head
path; the decode exit and all four batch exits call them), `opplan/exec/observe_lens.rs`
(`LensSites`, `readout_of`, `LogitLens`, `LensReader`), `EventKind::Readout` on the run
record with `head_passes` and `lens_failure` on the receipt, and `--lens-tokens`,
`--lens-layers`, `--lens-attention`, `--lens-top-k` on `vindex3 observe`, which now also
prints the stepping wall time.

| Property | Result |
|---|---|
| LP1 parity, lens armed on every site | PASS on the reference and production backends |
| LP2 anchor | PASS — the last FFN site's lens is the executor's logits bit for bit on both backends, and through the Gemma 4 miniature's layer scale (H2) |
| LP3 the exit is the function | PASS — decode-vs-batch, observed-vs-unobserved, HC, attention-residual and Gemma 4 gates all bit-identical after the refactor (134 tests) |
| LP4 proper distribution | PASS — mass within 1e-9 at every depth; argmax has rank 1; ties rank by strictly-greater |
| LP5 cost | MEASURED, above; single runs, not characterised |
| LP6 on the record | PASS — readouts sequenced between a write's stats and its structural event, priced on the receipt, replay equal; a failing lens is named on the receipt and the record stays complete |
| LP7 first reading | RECORDED, below |

**LP7 — Gemma 3 4B IT, `The capital of France is`, final prompt position, lens on every
FFN site for ` Paris` (9079) and ` France` (7001), production CPU, default policy
(Q8-requantised head).** Selected depths; the full table is in the record
(`gemma3-4b-france-lens`, 1446 events, 204 head passes, fingerprint `4f8b37e7…`).

| Layer | ` Paris` logp / rank | ` France` logp / rank | top-1 |
|---|---|---|---|
| 0 | −48.98 / 14055 | −60.57 / 59343 | 否 |
| 8 | −10.95 / 9070 | −10.31 / 4478 | ` cities` |
| 16 | −10.66 / 4834 | −9.69 / 846 | ` cities` |
| 20 | −9.34 / 846 | −8.59 / 421 | ` ________` |
| 22 | −6.45 / 52 | −8.76 / 225 | ` what` |
| 23 | −3.81 / 10 | −9.11 / 234 | ` what` |
| **24** | **−0.36 / 1** | −4.99 / 8 | ` Paris` |
| 25 | −0.004 / 1 | −7.02 / 3 | ` Paris` |
| 26–31 | ≤ −0.002 / 1 | −8.8 … −10.0 / 3–5 | ` Paris` |
| 33 (exit) | −0.223 / 1 | −7.39 / 34 | ` Paris` |

Read against the ADDRESS-BUILD prior (entity binding decisive at L24–L28): ` Paris`
becomes the head's answer at layer 24, one layer after entering the top ten, and is
near-certain by 26; ` France` is most readable in the teens and twenties and is never
the answer. This is one prompt on one model through a Q8-realised head, recorded as a
comparison with that prior, not as a replication of it. The exit row equals the
executor's own top-3 (` Paris` −0.2227) by construction.

## Out of scope

Per-head attention observation (its own rung), interventions, tuned or learned lenses
(a "tuned lens" is a different reader and would be named as one), the Metal path, any
lens on the batch prefill path.

---

## Verdict rule

V3-LENS-1 is complete only when LP1–LP6 PASS and LP7 is recorded. It unlocks the
Observatory's Logits tab and the depth-wise answer trajectory, both reading the record.
