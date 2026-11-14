# CONSUMPTION-BF16-1 — does a production run compute with the edited BF16 weights?

**Class: instrument qualification.** No behavioural claim, no cohort prompt, no scientific edit. It is a gate on the
execution path SG-EDIT-1 will use, not a result of that programme.

## 1. Question and claims kept apart

For an edit of one stored BF16 matrix, compiled by LARQL into a new container:

> On the production CPU backend, with the loader held to exact BF16, is the edited layer's FFN write the product of
> the edited container's stored `down_proj` and the vector the backend fed it, at every position?

Three claims are separate and this experiment addresses only the last:

| Claim | Established by |
|---|---|
| The canonical weights are the base with exactly the declared edit | `WEIGHT_PATCHES_V4` (run here as a precondition) |
| A stored quantised pack is the declared derivation of the canonical weights | `Q8_0_PACK_DERIVATION` (not applicable: these containers hold no pack) |
| A run computes with the stored weights of the tensor that was edited | this experiment, for one tensor, one layer, one execution policy |

## 2. Exploratory run (2026-11-14), preserved unchanged

A first run used the existing `sg-edit-i2-scale` container (a uniform rank-one update of L12 `down_proj`) and the
three I5 qualification prompts, 20 positions, with thresholds fixed before any record was read (§5). Records and
verdicts are in `~/chris-models/consumption-bf16-1/` and are not regenerated or reinterpreted:

| Record | Loader policy | Verdict | Own-weights error | Other-weights error |
|---|---|---|---|---|
| `record-default-policy.json` | default | `reference_not_calibrated` | 2e-4 .. 1e-2 | 1.8e-3 .. 1.07 |
| `record-max-format-bf16.json` | `LARQL_CPU_MAX_FORMAT=bf16` | `not_shown` | 7e-8 .. 1.2e-6 | 2.7e-3 .. 1.07 |

Two things were learned, and both shape this protocol:

1. **Under the default policy the production loader manufactures a lossy Q8 residency for an operand whose BF16
   image does not fit the L2 cache** (`opplan/exec/cpu/physical/projection.rs`, `q8_permitted` and `streams`). The
   edited `down_proj` is such an operand, so a default run does not compute with its stored BF16 bytes.
   `LARQL_CPU_MAX_FORMAT=bf16` forbids the manufacture. This experiment therefore fixes that setting and the
   reference refuses a record that does not show it. The default path is a separate, currently unverified
   representation and is out of scope here (§8).
2. **The uniform edit is badly conditioned as a stimulus.** Its effect on the write is proportional to the sum of
   the input vector, which is near zero at one position of two prompts; there the wrong-weights error was 2.7e-3,
   inside the inconclusive band. The verdict was `not_shown` and stays `not_shown`. The thresholds are not changed.

## 3. The qualification edit (fixed before it is compiled)

`ADD LOW RANK` of rank one over every column of `12.mlp.down_proj.weight` (`[2560, 10240]`) of
`~/chris-models/gemma3-4b-it.vindex3`, profile `f64-seq-rne1/v1`:

    a[r] = s_a(r) × 2^-1      b[j] = s_b(j) × 2^-2      s(i) = +1 if SHA-256("consumption-bf16-1/<a|b>/<i>")[0] is odd, else −1

written by `scripts/consumption_edit_operands.py`. Every element of the update is ±2^-3 against stored weights of
median magnitude 6.4e-3: the edited model is not a usable model and is not meant to be. The signs are hashed so that
the effect `a·(b·h)` does not vanish for a structured input.

**Why this one.** Chosen on the exploratory positions only: its predicted wrong-weights error there is 0.42 at the
worst position (median 0.82), forty times the mismatch threshold. No fresh prompt was run or examined to choose it.

Layer 12 is kept from the exploratory run. It is a sliding-window layer; every prompt is far shorter than the window.

## 4. Prompts (fresh; fixed before the run)

`bench/consumption-bf16-1/prompts.json`: four prompts, 33 positions, none used in the exploratory run and none
naming a country (the SG-EDIT-1 cohort is countries). Token ids are the container's own tokenizer's, `<bos>`
prepended; the tokenizer reproduces the I5 prompts' known ids.

## 5. Instrument and thresholds (unchanged from the exploratory run)

- **Engine harness** `opplan/exec/tests/consumption_real.rs` (production backend, real containers). It asserts, by
  bits: C1 every carrier write upstream of `(12, Ffn)` and every `down_proj` input at layer 12 is identical between
  the base and edited runs; C2 the layer-12 FFN write differs at every position and the final logits differ; C3 the
  control run is identical to the base at every write and every logit. It records the `down_proj` input and the
  carrier on both sides of the FFN write, the settings that select the arithmetic, and a comparison of the final
  logits. It computes no expected value.
- **Independent reference** `scripts/consumption_reference.py`: its own segment reader and binary64 arithmetic,
  no LARQL code. For each run and each container's weights it recomputes `post_ffn_norm(W·h)` and takes the relative
  error against the recorded write.
- **Thresholds, fixed on 2026-11-14 before the exploratory record was read and not changed since:** a recomputed
  write is a `match` at relative error ≤ 1e-4, a `mismatch` at ≥ 1e-2, and `inconclusive` between.

Changes to the instrument since the exploratory run, none of which touch the arithmetic or the thresholds: the
harness reads its prompts from a file and records the execution policy and the logit comparison; the reference
refuses a record that does not show `LARQL_CPU_MAX_FORMAT=bf16`.

## 6. Procedure

1. Write the operands; compile the edit with `larql lql` (`USE` base, `ADD LOW RANK`, `COMPILE CURRENT INTO VINDEX`)
   into `~/chris-models/consumption-bf16-1-edit.vindex3`.
2. **Precondition:** `larql-verifier ledger --contract WEIGHT_PATCHES_V4` on (base, edit) is `satisfied` on both
   halves. If not, stop: there is no verified edit to consume.
3. Run the harness once, `LARQL_CPU_MAX_FORMAT=bf16`, on base, edit and the control container
   `~/chris-models/consumption-control-vision.vindex3` (a two-element delta on the vision tower's
   `embeddings.position_embedding.weight`, which text inference does not read), with the prompts of §4.
4. Run the reference once on the record.

One run. A rerun is allowed only for an instrument fault (a crash, a missing file), which is recorded; the first
complete record is the result.

## 7. Gate

**PASS** when all of:

- the precondition of §6.2 holds;
- the harness's C1, C2 and C3 all hold (it fails otherwise and writes no record);
- the reference's status is `shown`: at all 33 positions the base run matches the base weights and mismatches the
  edited weights, and the edited run matches the edited weights and mismatches the base weights.

Anything else is the verdict the reference gives, reported as it is. There is no partial pass.

## 8. What a pass would and would not show

It would show that, on this backend with the loader held to exact BF16, the layer-12 FFN write of each run is
computed from that run's own stored `down_proj`, at 33 positions, and that an edit to an unread tensor changes
nothing. It would not show:

- anything about the **default** loader policy, whose manufactured Q8 residency is a different representation with
  no derivation rule written down and no verifier;
- that any other tensor, layer, backend (Metal) or execution mode (batched prefill, decode with a KV cache beyond
  these steps) consumes its stored weights;
- that the record is anything but the engine's own report of its intermediates;
- any behavioural effect, intended or otherwise.

## 9. Implications recorded for earlier SG-EDIT qualification runs

None of their conclusions depends on the loader policy, but none recorded it, so each is labelled:

- **I3 (sham round trip):** `vindex3 exec --backend production`, environment not recorded. Its claim is bit
  identity of a container whose bytes are identical to the base, which holds under any deterministic policy. It
  does not show which arithmetic ran.
- **I5 (capture, replay, restoration):** library test on `ProductionBackend`, environment not recorded. Its claims
  are bit identities and a rounding contract within one process, which hold under either policy. Layers in its S1
  band whose `down_proj` streams were, under the default policy, executed from the manufactured Q8 residency.
- **I2 (scale) and the v4 verdicts:** container-only; no execution involved.

SG-EDIT-1's exact-BF16 path requires `LARQL_CPU_MAX_FORMAT=bf16`, recorded in every run's manifest.

## 10. Record (run 2026-11-14)

**Formal verdict: PASS.** One run, no rerun, no instrument fault.

Frozen before the edit was compiled or any fresh prompt was run (`bench/consumption-bf16-1/prereg-digests.txt`,
2026-11-14T13:00:17Z; every digest re-checked after the run and unchanged). This document's own digest in that
file is of the text above this section. The freeze is a digest file, not a commit: nothing was committed first.

| Step | Outcome |
|---|---|
| §6.2 precondition, `WEIGHT_PATCHES_V4` on (base, edit) | `satisfied` on both halves; 26,214,400 elements compared, 26,214,400 expected to change |
| Harness C1 | every upstream write and every layer-12 `down_proj` input bit-identical, all four prompts |
| Harness C2 | the layer-12 FFN write differs at all 33 positions; final logits differ on every prompt |
| Harness C3 | the control is bit-identical to the base at every write and every logit |
| Execution policy recorded | `LARQL_CPU_MAX_FORMAT=bf16`; the other three policy variables unset |
| Reference | `shown`: 66 of 66 run-positions classified as required, none inconclusive |

| Reference cell | Relative error, min .. max | Required |
|---|---|---|
| base run, base weights | 8.3e-8 .. 8.7e-7 | match (≤ 1e-4) |
| edited run, edited weights | 1.7e-7 .. 2.1e-6 | match (≤ 1e-4) |
| base run, edited weights | 0.25 .. 1.20 | mismatch (≥ 1e-2) |
| edited run, base weights | 0.31 .. 6.17 | mismatch (≥ 1e-2) |

Final logits, descriptive only: base and edited differ at all 262,208 vocabulary entries on every prompt (largest
difference 3.0 to 6.0); the argmax token is the same in both on all four prompts. No behavioural reading is made.

Artifacts, in `~/chris-models/consumption-bf16-1/` (SHA-256):

- `qualification-operands.json` `b574dd72de18f607a73cbbf626d2882a32eb94ebd44cc49be0f3c79a290b4475`
- `qualification-v4-verdict.json` `89a933f6a8edd80bae5bd448ac2483249ed430e92005a102b1a2c8e3a0401eb5`
- `qualification-record.json` `054d35ab0912cefde1ffc03e695e8b3b6189aced65bd00b620f1a71c85d0d3c5`
- `qualification-verdict.json` `c7a0c1e55d884b69699a7d1ec62430cfead7820a4620abf12a48094ece3ef149`
- `qualification-compile.log`, `qualification-harness.log`
- exploratory, unchanged: `record-default-policy.json`, `verdict-default-policy.json`,
  `record-max-format-bf16.json`, `verdict-max-format-bf16.json`

Containers (`index.json` SHA-256): base `4d0287a4…774c693`, edit
`~/chris-models/consumption-bf16-1-edit.vindex3` `5e1c5352…829e902a`, control `5b8573e3…4f97c7112`.

Compiled with a debug `larql` built from `lql/verify-ledger` (engine as on main `7b5334dd`); harness built in
release from this branch. Verified with the verifier as on main.

## 11. Exploratory, after the run: the default policy is reproducible too

Not part of the gate, and judged on the exploratory default-policy record only. The loader's Q8 quantiser
(`opplan/exec/quantise.rs`) was read and its rule re-implemented independently: blocks of 64 elements along a row,
an f32 scale of `peak / 127` per block (1 for an all-zero block), codes `round_half_away(w / scale)` clamped to
±127, a weight read back as `code × scale`. Recomputing the layer-12 write with those weights:

| Default-policy run against | Relative error, min .. max |
|---|---|
| its own container, load-time Q8 rule | 2.9e-8 .. 6.7e-7 |
| its own container, stored BF16 weights | 2.3e-4 .. 1.0e-2 |
| the other container, load-time Q8 rule | 2.4e-3 .. 1.07 |

So under the default policy the run computes with that Q8 derivation of the stored weights, to the same precision
the exact path reproduces BF16. This is a hypothesis-generating result on a record already seen; the verifier it
points to is specified in `docs/cpu-load-time-q8-residency.md`.
