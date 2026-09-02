# VINDEX3 architecture conformance sweep

**Swept 2026-09-02.** 94 checkpoints attempted, 88 scored, across 35
architectural lineages and 60 family-generations — from GPT-2 to Kimi K3.

The question is not "how many models does LARQL support". It is whether
VINDEX3 is an *architecture representation* or a growing switch statement
of model names. A representation earns that name by giving an honest
answer on checkpoints nobody wrote code for.

## Running it

```sh
cargo build --release -p vindex-cli -p larql-cli
scripts/arch_sweep.py run                # resumable; ~25 min cold, network-bound
scripts/arch_sweep.py report             # verdict table + blocking subjects
scripts/arch_sweep.py clusters           # subjects -> semantic ideas
scripts/arch_sweep.py leverage           # greedy cover: which ideas unlock what
```

The matrix is [`matrix.json`](matrix.json). Plans are cached under
`~/chris-models/_conformance/plans/` — one JSON per checkpoint, the
evidence every number below is derived from. `report` re-derives; it
never re-asserts.

## What it cost

`vindex plan hf://…` reads `config.json` and safetensors *headers* over
byte ranges. No weights move.

| | |
|---|---|
| staged | **1.59 GB** |
| stood in for | **16.11 TB** of checkpoint |
| ratio | **10,106 : 1** |

Kimi K3 alone: 135.5 MB of headers standing in for 1,560.9 GB (11,516:1).
Judging a 2.8T model costs about what judging a 7B one does, so the
matrix is bounded by round trips, not by parameters.

## Baseline and movement

The sweep is the regression instrument for the conformance programme. Every
semantic change is measured against the same 88-row corpus, and the two
invariants are the ones that matter most:

| Metric | Baseline | Waves 1+3 | rope | moe | sliding window |
|---|---:|---:|---:|---:|---:|
| semantics version | 1 | 3 | 4 | 5 | **6** |
| GREEN | 17 | 18 | 21 | 26 | **28** |
| AMBER | 6 | 6 | 6 | 6 | 6 |
| RED | 65 | 64 | 61 | 56 | **54** |
| **BUG** | **0** | **0** | **0** | **0** | **0** |
| **silent drops** | **0** | **0** | **0** | **0** | **0** |
| text-closure blockers | 886 | 776 | 756 | 671 | **668** |
| K3 clusters remaining | 7 | 7 | 7 | 7 | **7** |

Wave 2 changed the document, not the verdict: schema 4 → 6, semantics 1 → 3,
and every row identical across it.

`BUG > 0` or a silent-drop regression means do not merge, whatever moved green.

**Wave 1 — architecture identity.** A `model_type` no registry entry matches,
and a container/text pair resolving to different architectures, now block
instead of passing silently into `GenericArch`. That *adds* blockers (16
checkpoints gained one) and is the point: they were being served Llama-shaped
defaults they never declared.

**Wave 3 — inert facts.** Decoding-policy defaults and dropout rates are
preserved and classified rather than graded `Unknown`; `pretraining_tp` is
judged against its value, because HF Llama's forward pass reads it above 1.

**Wave 2 — the sweep became trustworthy (plan schema 6, semantics *held* at 3).**
Every finding now carries an `id` and a `cluster`, and every capability closure
names its `blocker_ids` rather than counting them. The regex table that used to
live in `arch_sweep.py` is gone: the taxonomy is in the compiler, keyed by exact
leaf name and tested. The semantics version deliberately did **not** move — the
document says more, no verdict changed, and the sweep confirms it: 0 rows
changed, 776 blockers before and after. An instrumentation change that shifted
the semantics version would make every stored verdict falsely incomparable.

That makes leverage exact, and splits one number into three that must never be
conflated again:

| idea | reach | blockers removed | clears alone |
|---|---:|---:|---:|
| `position_rope` | 43 | 100 | 3 |
| `unclustered` | 40 | 219 | 1 |
| `moe_routing` | 37 | 66 | 4 |
| `architecture_identity` | 31 | 31 | 0 |
| `representation_quantization` | 19 | 120 | **0** |
| `modality_vision` | 15 | 217 | **0** |

`representation_quantization` deletes 120 blocking findings and moves no verdict
on its own. `moe_routing` deletes half as many and clears four. Reach ranks the
work; blockers-removed measures debt; only clears-alone predicts a row.

`unclustered` at reach 40 is the honest remaining gap in the taxonomy — 40
checkpoints are blocked partly on subjects no table has judged, and that is a
finding about the vocabulary rather than a bucket to grow by pattern-matching.

**Wave 4 — `position.rope`, first pass (semantics 4).** Llama-3 wavelength-band
scaling is now represented: `PositionPolicy::Llama3` carries the block through
the container boundary. No new mathematics —
`larql-compute::attention::rope::llama3` has always implemented the family, and
`RopeFreqScaling::Llama3` was already wired. What was missing was a way for the
schema to *say* it, so every Llama 3.x checkpoint was refused at `plan` rather
than encoded on unscaled frequencies.

Three whole family-generations cleared: **Llama-3.2 (1B, 3B) and Llama-3.3-70B,
5 blockers → 0 each**; Llama-4-Scout 14 → 9.

Two things fell out of it:

- `llama3_rope_scaling()` was a trait default returning `None`, so a family had
  to *opt in* to being served correctly — the exact shape
  `yarn_rope_scaling()`'s own doc-comment warns against three lines below it.
  Only `llama.rs` had opted in. It now reads the config like its YaRN sibling,
  and the duplicate override is gone.
- The two scaling families are resolved once, into `DeclaredRopeScaling`, rather
  than as two `Option`s that between them can express a "both" state no
  `rope_type` can produce.

`llama3` is deliberately **not** folded into `Yarn`. YaRN also rescales `cos`
and `sin`, changing every logit at every position; Llama-3 adjusts frequencies
only. Folding them would apply an attention-temperature change no Llama
checkpoint declares, and `policy.yarn()` returning `None` for a Llama-3 layer is
pinned by test.

**The exact-leverage instrument predicted this.** Wave 2's cover said
`position_rope` "clears alone: 3". It cleared exactly three. That is the first
time the programme predicted a verdict count before the work, and got it right.

**Wave 5 — `moe_routing`, and a falsified forecast (semantics 5).** The first
wave run as a preregistration: before any code, `clears_alone` named exactly four
rows that should go green, frozen in
[`forecasts/wave5-moe-routing.json`](forecasts/wave5-moe-routing.json).

All four hit, none missed, nothing regressed — **and a fifth row cleared that the
forecast did not name.** That triggered a falsifier, and the investigation is the
result worth keeping.

The wave shipped **two** changes, and only one of them was `moe_routing`:

- *The expert schedule.* Qwen3-MoE declares `decoder_sparse_step: 1` and
  `mlp_only_layers: []` — the uniform all-MoE stack the graph already builds.
  Judged by value, like `pretraining_tp`: a stride of 2 makes half the tower
  dense and a non-empty exception list carves out named layers, and both still
  block. This is `moe_routing` proper, and it cleared exactly the two Qwen3 rows.
- *A key declared with no value has nothing to carry.* Gemma 4's dense sizes
  declare `top_k_experts: null` — a dense model saying it has no expert bank —
  and the carriage rule demanded a home at `ExecutionSurface.ffn.moe.top_k`,
  which no dense component can answer. `num_experts: null` sat beside it already
  grading representable, so the two nulls were being judged differently. This is
  a **cross-cutting census rule**, not a routing one. It cleared the two Gemma 4
  rows *and* OLMoE-1B-7B, whose blocker was `clip_qkv: null` in
  `attention_bias`.

So `moe_routing` proper cleared **2**, not 4. The forecast's count was nearly
right and its *composition* was wrong, and only the unpredicted row exposed it.

**The lesson, which changes how forecasts are written.** A cluster names what a
blocker is *about*, not which fix will clear it. `clears_alone` is exact only
when the remediation is confined to one cluster; a cross-cutting rule breaks the
cluster→fix mapping in both directions — over-crediting the cluster it was filed
under, and under-predicting rows elsewhere. Future forecasts must name **the fix
being made**, not only the cluster it was drawn from.

**Wave 6 — the lesson applied, and an exact forecast (semantics 6).** Qwen2.5
ships `sliding_window: 32768` beside `use_sliding_window: false`. VINDEX3
resolves no window — correctly — and the carriage rule reported that agreement
as a dropped fact, refusing the checkpoint over a window it had been told not
to apply. A `CompanionGate` now names the pair: a declaration its own switch
turns off is inert, and the switch is read at the same nesting level so one
component's flag cannot silence another's attention.

The forecast was written against **the fix, not the cluster** — the Wave 5
correction. `attention_schedule` has reach 19, but this fix reaches 3, measured
over every cached config. It predicted two rows green, one row improving from 2
blockers to 1 while staying red, and *nothing else in the corpus moving*.

**Every arm held**, including the negatives: exactly three rows changed, GREEN
28, RED 54, blockers 668 — all as predicted. Forecasting the fix turns
`clears_alone` from an estimate into a statement.

`attention_schedule` still has reach 19: `layer_types`, `attention_policy`,
`attention_chunk_size`, `num_kv_shared_layers` and `attn_layer_period` are
untouched and remain real work.

**What did not happen, and why it is worth recording.** The inert clusters
reach 34 checkpoints, which looked like a large GREEN wave. It was one row
(Yi-1.5-6B). Cluster *reach* counts checkpoints a fix touches; only a
checkpoint whose **every** blocker is retired changes verdict. The greedy
cover already said this — `inert.training_only` cleared 1 — and it was the
better predictor. Reach ranks work; the cover predicts verdicts.

Kimi K3 is the standing witness: 97 blocking subjects → 73, text-closure
blockers 72 → 48, and **7 semantic clusters both before and after**. The
spelling collapsed; the ideas did not. That gap is the whole argument for
counting ideas.

## The verdict

Scored against the **text-generation closure**, which the plan computes
itself — a checkpoint whose only blockers sit in a vision tower is not
evidence about its language model.

| outcome | baseline | current | share now |
|---|---:|---:|---:|
| GREEN — representable | 17 | 18 | 20% |
| AMBER — identified, no implementation | 6 | 6 | 7% |
| RED — semantic gap | 65 | 64 | 73% |
| BUG — should work, doesn't | 0 | 0 | 0% |
| *unreachable (gated/absent)* | *6* | *6* | *not evidence* |

GREEN includes **Qwen3.8-27B, Gemma 4 26B-A4B, Granite 4.1/4.2 (3b/8b/30b),
gpt-oss 20b+120b, Kimi-Linear-48B, Mixtral 8x7B/8x22B, Gemma 2, Granite 3.3,
Ministral-8B** — encoded from their own declarations, no per-model code.
Yi-1.5-6B joined them in wave 3, on `pretraining_tp` alone.

AMBER is **DeepSeek V3.2, V4-Flash, V4-Pro, GLM-5.2, GLM-5.3, GLM-5.3-Flash**
— every frontier sparse-attention-indexer model, and coherently so. VINDEX3
identifies the component and says it has no implementation. That is a
success for the representation, not a failure: `UnsupportedComponent` is a
statement of *knowledge*, not of severity.

## 408 blocking subjects are 23 ideas

*(Baseline run. After waves 1 + 3 the corpus carries 366 subjects over the
same 23 ideas — the spelling shrank, the vocabulary did not.)*

Counting config keys measures spelling. Ten `mamba_*` keys are one idea.

| | 2026-08-31 census | this sweep |
|---|---|---|
| checkpoints | 12 | 88 |
| family-generations | — | 60 |
| distinct blocking subjects | 58 | 408 |
| semantic ideas | 13 | **23** |
| subjects per idea | 4.46 | **17.7** |

The census prediction held at 7× the corpus: checkpoint count grows
quickly, family-generations more slowly, semantic ideas slowest, and
**reuse per idea rises** — 4.46 → 17.7.

Top ideas by reach (checkpoints / family-generations):

| idea | ckpts | gens |
|---|---|---|
| `position.rope` | 43 | 31 |
| `moe.routing` | 35 | 23 |
| `modality.vision` | 26 | 15 |
| `ffn.activation` | 22 | 16 |
| `attention.schedule` | 19 | 14 |
| `inert.training_only` | 18 | 17 |
| `shape.and.tensor_naming` | 18 | 13 |
| `norm.geometry` | 18 | 12 |
| `decode.multi_token_prediction` | 18 | 11 |
| `representation.quantization` | 15 | 11 |

A greedy cover says **21 ideas clear all 71 blocked checkpoints**; the
first five clear 20. This is an estimate — the plan reports its text
closure as a count, not a finding list, so each step is an upper bound.
The *ordering* is what it is for.

The clustering is post-hoc regex over saved plans, and is scratch analysis
rather than authority. It belongs in the parser: a finding should carry
its own `semantic_cluster` so VINDEX can say "these ten findings are one
missing concept" without a script guessing.

## Two doors, and only one of them refuses

**VINDEX3 refuses.** Zero checkpoints passed the gate while carrying a
dropped execution fact (0 GREEN rows with a `mismatched` blocking
finding), and zero rows were BUG. Llama-3.2 is the clean example: it
declares `rope_scaling.rope_type: "llama3"`, VINDEX3 resolves `default`,
and rather than encode a container that would serve the wrong long-context
behaviour, the plan states the mismatch and refuses. `larql-models`
already parses llama3 scaling — the gap is at the *container boundary*,
not in the engine.

**The engine fails open.** Of 42 distinct `model_type` strings swept, 27
resolve through the registry and **15 fall through to `GenericArch`**,
which serves Llama-style defaults rather than refusing — 30 checkpoints,
including every GLM 4.x/5.x, Jamba, OLMo, Phi, Falcon-H1 and Ministral-3.
This is the fail-open class recorded in the 2026-09-01 codebase review,
measured.

Kimi K3 is the sharp case. Its top-level `model_type` is `kimi_k3`; its
`text_config.model_type` is `kimi_linear`. Detection on the nested config
would dispatch a 93-layer, 1.56 TB model to the Kimi-Linear-48B module;
detection on the top level lands on `GenericArch`. Neither door refuses.

## The ladder, closed

`config → detect → semantic judgement → inventory → plan → encode →
reopen → verify`, on Qwen3-0.6B:

```
plan     admissible, 40 representable, 0 blocking
encode   1.50 GB fetched by range · checkpoint never present on this disk
inspect  Qwen3-0.6B · qwen3 · 28 layers · hidden 1024 · graph coherent
verify   yes — the artifact agrees with its own record
```

## The cheapest work, ranked

REDs by distance from the text gate, smallest first — each is a whole
family-generation:

| blocking | checkpoints | subject |
|---|---|---|
| 1 | Yi-1.5-6B | `pretraining_tp` — a *training* parallelism knob, classed `unknown`. Appears in 11 checkpoints |
| 1 | Qwen2.5-0.5B, 7B | `sliding_window` — declared-and-disabled |
| 1 | OLMoE-1B-7B | `clip_qkv` |
| 2 | Qwen3-30B-A3B, 235B-A22B | `decoder_sparse_step`, `mlp_only_layers` |
| 2 | GLM-4.5-Air, GLM-4.6 | `num_nextn_predict_layers`, `use_qk_norm` |
| 4 | Qwen3.5 0.8B/9B/27B | `mtp.*` — multi-token prediction, one concept |

Three of the six are the recurring *kind* of gap the census predicted:
declared-and-inactive, or training-only misfiled as `unknown`.
`inert.training_only`, `inert.dropout` and `inert.generation_defaults`
together block 34 checkpoints and describe nothing a forward pass reads.

## Caveats

- The engine column is computed from the `model_type` the plan reports
  (the text component's). For K3 that is `kimi_linear`, so "recognised"
  there means a string matched — not that the model would be served
  correctly.
- Six rows are unreachable: three gated without access (Command-R ×2,
  Jamba-Mini-1.7), three absent at the id swept (TinyLlama_v1.1,
  Baichuan2-7B-Base, state-spaces/mamba2-780m). They are excluded from
  every percentage.
- GREEN is a claim about *representation*, established from headers.
  Only Qwen3-0.6B was carried through encode/verify in this pass.
