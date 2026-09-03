# PARETO-1 — readiness audit for a REPRESENT vs Unsloth Dynamic 3.0 head-to-head

2026-09-03. Worktree `.claude/worktrees/pareto-1`, branch
`pareto-1/unsloth-headhead`, off `474a0392`.

**No measurement was taken and no candidate was compiled.** This records
what exists, with evidence, and names the gaps between it and the proposed
experiment — *before* a pre-registration commits to arms it cannot run.

The proposal being audited: take the BF16 Qwen3.8-27B source, produce
Unsloth `UD-Q4_K_M` / `UD-Q3_K_XL` / `UD-Q2_K_XL`, conventional GGUF
baselines at the same sizes, and REPRESENT maps at the *same exact byte
budgets*; evaluate every arm through one authority corpus.

---

## The subject and its reference truth already exist

```text
container   ~/chris-models/qwen3.8-27b.s6.vindex3
            payload 54,713,457,120 B (50.96 GiB), model 1d4bf0f2…,
            family qwen3_5_text, 64 layers, hidden 5120
bank        ~/chris-models/qbanks/Qwen3.8-27B/quality-bank-1
            69 prompts, 1,740 positions, logits 248,320 wide, f32,
            NOT top-k truncated, 1.61 GiB
```

The container **is** the one `quality-bank-1`'s provenance names: all four
text representation hashes match it exactly (`target.decoder_stack@BF16`
`8b15bedb…`, `embedding` `a158f24c…`, `final_norm` `d1d9fbde…`,
`output_head` `d922b751…`). The provenance file still names the path
`~/chris-models/Qwen3.8-27B.vindex3`, which no longer exists — the same
object under a later name. *The provenance file should be corrected; a
consumer following it today finds nothing and may conclude the reference
is unreproducible.*

**Qwen3.8-27B is DENSE.** `system_graph.json` contains zero occurrences of
`expert`, `router`, `routed` or `moe`. This is load-bearing below.

A represented container already exists and proves the REPRESENT compile
path runs on this model: `~/chris-models/qwen3.8-nvfp4.vindex3` carries
`precision_map {name: r0-uniform, encoding: NVFP4, roles: [decoder-linear,
recurrence-projection, recurrence-control, expert-weight]}` and a
`target.decoder_stack@NVFP4` of 13,702,470,592 B beside the BF16 original.

---

## GAP 1 — the codec floor is Q4_K, so two of the three budgets have no action

`represent/arena.rs:172-207` dispatches exactly four encoders and refuses
anything else *by name*:

```text
BF16   Q8_0   Q6_K   Q4_K        →  everything else:
                                   "no encoder for `{other}` — refusing
                                    rather than binding source bytes under
                                    a name that claims otherwise"
```

`represent/arena_tests.rs:238` pins that refusal against `Q3_K`.

```text
UD-Q4_K_M   ~4.8 bpw   CONTESTABLE — Q4_K is the vocabulary's floor
UD-Q3_K_XL  ~3.6 bpw   NO ACTION EXISTS
UD-Q2_K_XL  ~2.7 bpw   NO ACTION EXISTS
```

This is not "REPRESENT's codecs are less refined than the IQ family". At
the Q3 and Q2 budgets REPRESENT has **no move at all** — the search space
is empty, so there is nothing to select between and no result to report.
Only the Q4_K_M budget is a real contest today, and there the contest is
genuine: Unsloth's mixed allocation over the K-quant families against
REPRESENT's mixed allocation over {Q8_0, Q6_K, Q4_K} at equal bytes.

Adding Q5_K/Q3_K/Q2_K is *stealing a codec*, which the programme's own
doctrine explicitly permits ("steal codecs freely — win the
selection/planning race, not the codec race"). It is implementation of
published ggml block formats, not new quantiser mathematics.

---

## GAP 2 — the GGUF compile-down cannot carry a K-quant

The granularity is already right, and this is better news than expected.

- `represent/map.rs:57-68` — `Exception { projection: Option<String>,
  layers: Option<(u32,u32)>, encoding: Option<String> }`, first match
  decides. That is **exactly Level 1 (per-layer) and Level 2
  (per-projection)** of the compile-down ladder, and exactly what a stock
  GGUF can carry. The Qwen3.8 NVFP4 container simply has zero exceptions.
- `gguf/walk/mod.rs:391-408` already reads dtype **per tensor**, with a
  comment recording that reading the object's encoding instead once
  inflated a scale-sibling count from 496 to 848. The mixed case is
  structurally handled.

The gap is the *dtype vocabulary at the boundary*:

```text
walk/mod.rs:399-407   understands  NVFP4 | F32 | BF16
                      anything else → refuses by name
emit/mod.rs:45-46,38  emits        TYPE_F32=0 | TYPE_BF16=30 | TYPE_NVFP4
```

A `Q8_0`, `Q6_K` or `Q4_K` tensor reaches `other =>` and errors. So a
searched mixed K-quant map cannot be exported at all.

Separately: `TYPE_NVFP4` is **not a stock ggml type**. `~/chris-models/llama-cli`
is a symlink into `~/chris-source/llama.cpp/build-nvfp4`, and that checkout
is on `master` at `6b80c74f2` with a clean tree. So whatever runs those
exported NVFP4 GGUFs today, *nothing REPRESENT currently emits is
demonstrated to run in stock llama.cpp* — which is precisely the claim the
compile-down principle was written to support ("The exported GGUF must run
in stock llama.cpp — that is the point").

Scope of the fix: three dtype arms in the walk, three ggml type constants,
three byte-size arms, three emit pass-through arms. Bounded. The one real
risk is layout conformance — see `project_ggml_nibble_layout_nonconformance`,
which is the recorded instance of exactly this hazard, and which means the
K-quant blocks must be proved byte-conformant to ggml rather than assumed.

---

## GAP 3 — the contract and the search instrument are Kimi-shaped

The rung-4/5 machinery that makes REPRESENT interesting is built around
`kimi-logit-balanced-v1` and a five-dimension constraint vector:

```text
kl p99                      generic
top-1 probability given up  generic
top-10 mass displaced p99   generic
routed mixture moved p99    REQUIRES A ROUTER
routed mixture moved max    REQUIRES A ROUTER
```

**Qwen3.8-27B is dense.** Two of the five dimensions — including the two
that BALANCED-SEARCH-2 found were the *actually binding* resource at 83%
while KL sat at 68% — do not exist for this model. The constraint vector
collapses toward the scalar-KL case the evidence ladder exists to avoid,
and `route flip rate`, one of only two ordering proxies available at
diagnostic scale after BS2-F1, is likewise unavailable.

The probe itself is Kimi-specific too: `opplan/exec/tests/kda_q8_real.rs`
is driven by `LARQL_KDA_Q8_LAYER` / `LARQL_MLA_Q8_LAYER` and speaks in KDA
/ MLA / expert arms. Qwen3.8 has 48 Gated DeltaNet + 16 M-RoPE softmax
layers and no experts.

**Consequence:** a Qwen3.8 contract has to be *earned* the way
`kimi-logit-balanced-v1` was — a consequence ladder on this model's own
bank, boundary set at the observed movement/character-change transition —
not transplanted. Relaxing Kimi's thresholds onto a dense model would be
the unearned-contract failure the programme has avoided so far.

The CLI already carries generic instruments that help here:
`vindex3 sensitivity` and `vindex3 consequence` (SENSITIVITY-1B′,
activation-weighted per-tensor consequence), both model-agnostic and both
already exercised against `quality-bank-1`.

---

## GAP 4 — bank scale, and the four frozen prompt sets that already solve it

`quality-bank-1` is 1,740 positions, produced on the **production CPU**
backend at `LARQL_CPU_MAX_FORMAT=bf16`, against Kimi's 8,192-position Metal
authority scale. It has no held-out partner, and every ruling since R5-F1
(the band, Ruling 2, R5-F10) needs two banks.

**But three more frozen prompt sets already exist**, and they are mutually
disjoint — verified by comparing prompt text across all six pairs, every
intersection empty:

```text
bank              prompts   positions   Qwen3.8 BF16 reference
quality-bank-1        69       1,740     BANKED (production CPU)
quality-bank-2        69       1,946     prompts only
quality-bank-3a      200       4,859     prompts only
quality-bank-3b      200       4,912     prompts only
                          all six pairwise text overlaps = 0
```

`3a`/`3b` are a matched disjoint pair at ~4,900 positions each — 2.8x
bank-1, within reach of Kimi's 8,192, and p99 tail support of ~49
observations per bank rather than bank-1's 17.

**Revised bank plan: run the BF16 reference arm on 3a (selection) and 3b
(held-out), and do not mix bank-1 into the contract.** Reasons:

- The prompts are already frozen and provably disjoint, so there is no
  authoring step and no overlap risk.
- Both banks get produced on one backend in one campaign, so the
  selection/held-out disagreement that defines the PARETO-1 band measures
  *slice* variation and nothing else. Mixing bank-1 in would confound
  slice with the backend and build that produced it.
- Bank-1 keeps its existing meaning as the CPU-5 / ANE-4A depth reference,
  untouched.

Cost: 4,859 + 4,912 = **9,771 positions** of full-depth 27B BF16
teacher-forced reference at 248,320-wide f32 = **9.7 GB** of logits. Disk
is not the constraint (~223 GB free). Wall-clock is — see GAP 5.

`run_bank.py` already carries `QBANK_DIR`, added after a recorded near-miss
in which `reference` wrote into a bank-2 directory while reading bank-1's
prompts — "destroying exactly the independence the second bank exists to
provide, without any error". It must be set explicitly on every run here.

---

## GAP 5 — Qwen3.8 has no Metal kernel, so there is no GPU arm at all

Stated in the code, at `vindex3_cmd/mod.rs:76-79`, as the reason the
`production-nvfp4` backend exists at all:

> *"Qwen3.8 is the case that forced it: 48 Gated DeltaNet layers with no
> Metal kernel, so before this arm its NVFP4 pack could be compiled and
> verified and then executed nowhere, and its behavioural fidelity was
> unmeasurable in principle."*

Two consequences, and the second is strategic:

1. **Every arm is CPU-bound.** Reference and candidate runs alike use
   `--backend production` / `production-nvfp4`. Wall-clock, not disk, is
   the binding cost of this programme.
2. **The execution-economics half of the REPRESENT story cannot be
   exercised on this model.** "Measured bytes/token -> GPU ms -> predicted
   tok/s", the thing that distinguishes REPRESENT's objective from
   Unsloth's, has no GPU to be measured on here. On Kimi the byte model
   predicted -7.1% GPU time against -6.9% measured; on Qwen3.8 no
   equivalent claim is available.

A Qwen3.8 result is therefore a **behaviour-per-byte** result, not a
behaviour-per-byte-*and*-throughput result. That is still exactly what the
Unsloth comparison is about — Unsloth's own claim is quality at a given
size — but it must not be written up as though the throughput arm were
present.

---

## GAP 6 — the K-quant encoders are reachable only through Kimi's bank compilers

This is the one that blocks rung A immediately, and it was not visible
from the encoder dispatch alone.

`represent/arena.rs` has working `Q8_0`/`Q6_K`/`Q4_K` encoders (GAP 1).
But nothing general can reach them:

```text
represent/mod.rs:221-226    compile_representation() hard-refuses:
                            "encoding `{}` has no representation compiler;
                             known: NVFP4"
vindex3_cmd/mod.rs:339-341  --encoding "NVFP4 is the only compiler today"
prepare.rs:73-75            wanted_representation(): "Only the NVFP4 arms
                            have a compiled counterpart today"
```

The K-quant encoders are called only by the **Kimi-shaped bank
compilers** — `compile_expert_bank` (routed experts) and
`compile_kda_bank` (KDA/MLA arms). Qwen3.8 is dense and has neither.

And the transient route does not rescue it. `RepresentationSource`
{`Auto`, `Stored`, `Transient`} controls *whether* the runtime may
manufacture a representation, not *which* one — the encoding comes from
`wanted_representation(backend)`, and the backend vocabulary has
`production-nvfp4` and the `metal-nvfp4*` family but no
`production-q8`/`-q6k`/`-q4k`. So a Q6_K arm cannot be executed even
transiently.

**Consequence for rung A: the uniform Q8_0 / Q6_K / Q4_K anchors cannot
be produced or executed today.** The only uniform anchor that exists for
Qwen3.8 is NVFP4 at 4.5 bpw, via `production-nvfp4` and the already-built
`~/chris-models/qwen3.8-nvfp4.vindex3` — one point, not a curve, and not
in the K-quant vocabulary the Unsloth comparison eventually needs.

The fix is the same one GAP 1 and GAP 2 point at from their own
directions: **make the general represent path carry the vocabulary the
search already has.** Concretely, three connected pieces —
`compile_representation` routing to `arena::encode` for the K-quant
names, a backend arm that declares it executes them, and the segment
writer handling contiguous K-quant blocks (simpler than NVFP4's split
pack + scale tensors). Once that exists, GAP 2's exporter change makes
the same maps emittable as stock GGUF.

---


## Open, and it bears directly on any KL gate here

`PARITY-FLOOR-1` is **OPEN**: Qwen3.8's per-layer residual `rel_rms`
against HF sits at 1e-3 … 9e-3, growing with depth, unexplained. Its own
memory says: *"revisit before any quantisation claim rests on a parity
margin — a 6e-3 logit floor is large enough to matter for a KL/NLL gate."*

A head-to-head whose whole verdict is a KL threshold is exactly that kind
of claim. This does not block a *relative* comparison in which every arm
carries the same floor, but it does block an absolute fidelity number, and
it must be stated in any result.

---

## What is contestable today, without building anything

One experiment needs none of the four gaps closed, and it is gate (1) of
the "best OPTIMIZER, not best codec" claim as already written down:

> beat sensible baselines on a Pareto plot with the **same encoding
> vocabulary** — uniform Q8, uniform Q6, uniform Q4, hand-tuned mixed,
> REPRESENT — behaviour preservation vs bytes.

All arms inside LARQL, on `quality-bank-1`, with no GGUF export and no
Unsloth download. It also answers the question every Unsloth claim depends
on anyway: **does the rung-4/5 search machinery work on a model that is
not Kimi** — dense, no router, two constraint dimensions fewer, a
different bank shape. "Repeatable across Kimi/GLM/Qwen/Granite before it's
a claim" is the programme's own bar.

---

## Sequencing implied by the above

```text
rung A  uniform-vocabulary Pareto on Qwen3.8            no gaps to close
        + BF16 reference on 3a + 3b                     GAP 4
        + earn a Qwen3.8 contract (consequence ladder)  GAP 3
        + measure the PARETO-1 band from 3a-vs-3b       GAP 4
        behaviour-per-byte only, no throughput arm      GAP 5

rung B  K-quant GGUF compile-down, stock llama.cpp      GAP 2
        → the Q4_K_M head-to-head becomes runnable

rung C  Q5_K / Q3_K / Q2_K encoders                     GAP 1
        → the Q3_XL and Q2_XL budgets become contestable
```

Nothing above argues the head-to-head is a bad idea. It argues that at two
of the three proposed budgets there is currently no experiment to run, and
that the arm which would make the result ecosystem-evaluable cannot be
emitted yet.

---

# ADDENDUM — the K-quant path, and what conformance testing settled (2026-09-03, late)

GAP 6 is closed and GAP 1 turns out to be narrower than stated above.
Both were changed by testing against a **foreign reference** rather than
against ourselves.

## GAP 1, restated: the floor is an ENCODER gap, not a codec gap

`larql_models::quant::ggml::dequantize` already dispatches Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K and Q8_0. The 4.5 bpw floor exists because only three of
those have **encoders** in the workspace. Reaching the Q3_XL / Q2_XL
budgets means writing three encoders against layouts this workspace can
already read — not building a codec stack.

## GAP 6 closed: `represent/kquant.rs` and four wiring points

```text
kquant.rs             the vocabulary, geometry, encode/decode, codec identity
compile_representation  NVFP4-only refusal -> a target dispatch
OperandStore::load    decodes K-quants where the shape is available
prepare.rs            production-q8 / -q6k / -q4k backend arms
```

Two design calls worth restating because they will be load-bearing later:

- **Each K-quant is its own `CodecIdentity` family**, not a revision of a
  shared one. Q4_K and Q6_K are different physical interpretations, so
  the invariant kept is *same family ⇒ the reader can interpret the
  layout*, never *same quantisation lineage ⇒ vaguely related*. This
  matters more as Q2_K/Q3_K/IQ arrive.
- **The backend arms decode to f32** rather than running K-quant kernels.
  The question rung A asks is what behaviour a representation buys per
  stored byte, not how fast a Q4_K kernel is; decoding removes kernel
  implementation quality as a confounder. Native K-quant kernels are a
  separate substrate experiment.

## The row-framing rule — the bug that would have poisoned a Pareto curve

```text
WRONG   N_elements mod B == 0
RIGHT   D_inner    mod B == 0     (each outer row framed independently)
```

`[2, 128]` separates them: 256 elements, exactly one Q6_K super-block, so
the wrong rule admits it and emits a block spanning the end of row 0 and
the start of row 1 under one shared scale. **Nothing crashes.** The bytes
serialise, the segment table is self-consistent, the container loads, and
the arm yields plausible behavioural numbers from semantically invalid
bytes. Once REPRESENT searches automatically, the optimiser would
enumerate such scopes, measure them, and fold them into a curve that is
quietly part fiction. Documented at the planner contract in
`represent/mod.rs`; the test asserts both that the flat check accepts
`[2,128]` and that `plan` refuses it, so it cannot be "simplified" back.

## Foreign-reference conformance — the two claims, separated

Self-consistency (`decode(encode(x)) ≈ x`) passes just as happily if the
encoder and decoder share one misunderstanding. So the fixture's bytes
come from ggml itself: `fixtures/ggml_kquant_golden.gen.c` links against
llama.cpp's ggml and dumps, for one frozen 512-element input, ggml's own
`blck_size`/`type_size`, its quantiser's bytes, and its own decode of
them. The generator is committed beside the fixture.

```text
GEOMETRY            ggml's blck_size/type_size confirm the table
                    32/34, 256/210, 256/144 — all three exact      PASS
LAYOUT CONFORMANCE  decode_larql(bytes_ggml) == decode_ggml(...)
                    bit-for-bit, all three encodings               PASS
ENCODER EQUIVALENCE encode_larql(x) == bytes_ggml
                    Q8_0 10/544, Q6_K 366/420, Q4_K 94/288 differ  FAILS
```

The geometry check matters because the table's numbers were read off this
workspace's own decoders, which therefore cannot confirm it; ggml can.

**Layout conformance holds, encoder equivalence does not.** That is a
legitimate difference, and it fixes the wording of every downstream
claim:

> **VINDEX3's Q4_K encoder, using the ggml Q4_K representation** — not
> "llama.cpp's Q4_K quantisation".

It also settles the `project_ggml_nibble_layout_nonconformance` hazard
for the decode direction: a compiled K-quant pack is readable by the
ecosystem's own decoder, which is what GAP 2's export needs.

**The consequence for the head-to-head is the important part.** REPRESENT's
thesis is that it wins the *selection* race, not the codec race — and that
only holds if the codec is effectively held constant against the
ecosystem's. If LARQL's Q4_K reconstructs materially worse than ggml's,
every point on a behaviour-per-byte curve carries an encoder penalty and
a matched-byte comparison against an Unsloth artifact conflates
allocation quality with encoder quality. So a gate now asserts LARQL's
reconstruction RMS is within 5% of ggml's at the same bit width, in
**either** direction — worse penalises the curve, better flatters LARQL
for a codec reason. The bound was stated before the numbers were read.

---

# WHY THIS INSTRUMENT DESERVES TRUST

Three defects were found while building the K-quant path, at three
different boundaries, and **all three shared one pathology**:

> **Internal agreement is not evidence of external correctness.**

```text
tensor geometry      flattened storage assumptions -> tensor semantics
                     [2,128] divides by element total, not by row
encoder implementation  self-roundtrip -> an independent implementation
                     Q6_K reconstructs 1.1146x worse than ggml's
foreign ABI identity locally named constants -> upstream's actual ABI
                     TYPE_Q8_0 was 6; upstream says 8
```

Each crosses a boundary the previous one cannot see. A round-trip test
cannot detect a shared misunderstanding of the layout. A byte fixture
cannot detect a wrong type *id*, because cases are matched by name and
the id is only ever handed to our own dispatcher. Only passing the value
itself into another implementation tests that.

**This is why the foreign-reference gates are not incidental testing.**
They are part of the validity argument for Rung A: each successive check
defeats a class of shared misconception the preceding one is blind to.
An anchor curve measured on an instrument that had only ever agreed with
itself would be a number without a warrant.

None of the three changed the research hypothesis. All three changed
what the instrument is capable of showing.

---

# THE PRE-REGISTRATION THIS RECORD ANSWERS TO

Frozen outside the repository so amendments are visibly amendments
rather than mutable history:

```text
path    ~/chris-models/pareto1/PRE-REGISTRATION.md
sha256  70d2a49bcaa5ab5b37cad513ba064e6b34c6bc970447e4c92267d8d087b6d72c
lines   471
```

It carries E1-E4, the stopping rule, the control, the band construction,
the exclusions, and two amendments — Amendment 1 (the pre-registered
arms were unrunnable) and Amendment 2 (the encoder was not held
constant, and the direction that biased E2). Both were written **before
any anchor arm existed**, which is the only reason they are evidence
rather than rationalisation.

If the content identity above no longer matches, the pre-registration
has been edited: compare against this hash before trusting any result
that cites it.
