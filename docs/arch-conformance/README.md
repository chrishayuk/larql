# VINDEX3 architecture conformance sweep

**Swept 2026-09-02; widened to the 2026 frontier 2026-09-03.** 117
checkpoints attempted, 109 scored, across 44 architectural lineages and 79
family-generations — from GPT-2 to Kimi K3, Nemotron 3 Ultra and Hy4.

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
scripts/arch_sweep.py envelopes          # coverage by semantic shape
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

| Metric | Baseline | Waves 1+3 | rope | moe | sliding window | frontier census† |
|---|---:|---:|---:|---:|---:|---:|
| semantics version | 1 | 3 | 4 | 5 | **6** | 6 (held) |
| GREEN | 17 | 18 | 21 | 26 | **28** | 28 |
| AMBER | 6 | 6 | 6 | 6 | 6 | 7 |
| RED | 65 | 64 | 61 | 56 | **54** | 74 |
| **BUG** | **0** | **0** | **0** | **0** | **0** | **0** |
| **silent drops** | **0** | **0** | **0** | **0** | **0** | **0** |
| text-closure blockers | 886 | 776 | 756 | 671 | **668** | 1109 |
| K3 clusters remaining | 7 | 7 | 7 | 7 | **7** | 7 |

† Wave 7 widened the corpus from 88 to 109 scored rows. The 88 rows every
earlier column was measured on are unchanged — same verdicts, same 668
blockers — so the earlier columns stay comparable with each other and the
last column is a new baseline, not a movement.

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

**Wave 7 — the census widened to the 2026 frontier (semantics *held* at 6).**
Twenty-three rows added and none of them Llama-shaped: Nemotron 3 Nano-4B /
Nano-30B-A3B / Super / Ultra and 3.5 Lightning, MiniMax M2 / M2.7, Step 3.5 and
3.7 Flash, Hy3 and Hy4-preview, Mistral Large 3, Qwen3.8-2.4T-A95B, Command A
and A+, EXAONE 4.0 (1.2B, 32B) / 4.5 / K-EXAONE 2.0, LFM2 (350M, 2.6B) / 2.5 /
2.5-MoE. Frozen in
[`forecasts/wave7-frontier-census.json`](forecasts/wave7-frontier-census.json)
before any of them was planned: an outcome per row, the aggregate, and the
falsifiers. **23 of 23 outcomes hit**, the aggregate matched to the row, no
existing row moved, BUG 0, silent drops 0.

The two negatives held. Mistral Large 3 ships `params.json` only, in every
variant — a config *dialect* the source does not read, which the sweep now
reports as `no-config.json` rather than filing under "absent". Command A is
gated for this token, like the two Command-R rows. The one AMBER held too:
Hy4-preview's indexer and hyper-connection leaves match the
unsupported-component table, so the plan names the components — and labels
them `(GLM-5.x)` on a Tencent checkpoint. The table is leaf-keyed and
family-labelled; `indexer_types` is named in one table and `unclustered` in the
other. Both are recorded defects for the next semantics wave.

What the wider corpus says about the vocabulary:

| | 88 rows | 109 rows |
|---|---:|---:|
| checkpoints with a blocker | 62 | 83 |
| distinct blocking subjects | 334 | 552 |
| semantic clusters | 20 | 21 |
| subjects per cluster | 16.7 | 26.3 |
| `unclustered` reach (whole model) | 42 | 62 |

One cluster appeared (`shape_and_tensor_naming`, Step's
`attention_other_setting.true_head_dim`), so the closed taxonomy absorbed 218
new subjects into 20 ideas it already had. The gap is where the forecast said
it would be: every new RED row carries `unclustered`, and the subjects there
are whole mixers this schema has no word for yet — Nemotron's
`hybrid_override_pattern` / `layers_block_type`, LFM2's `conv_L_cache` and
`block_*`, MiniMax's `attn_type_list`, Step's `moe_layers_enum` and
`partial_rotary_factors` (plural). Two families spell one generation two ways:
Nemotron Ultra and 3.5 declare `layers_block_type` (a list, and no
`num_hidden_layers`) where Nano and Super declare `hybrid_override_pattern` (a
string); LFM2-350M declares `full_attn_idxs` where LFM2-2.6B declares
`layer_types`.

Two things the forecast reasoned about coarsely, worth keeping. EXAONE-4.0-1.2B
blocks on exactly four subjects — `architecture_family`,
`target.execution_surface`, `hidden_act`, `rms_norm_eps` — none of them a
declaration the model makes unusually; its `sliding_window: null` beside
`sliding_window_pattern: null` passed as inert, Wave 6's companion gate reaching
a third spelling. Everything left is the unregistered family, and `exaone4` is
*not* a Llama alias (QK-Reorder-Norm), so the family entry has to say so.
And Command A+ declares `layer_types` sliding/full with a 4096 window and
`attention_schedule` is **not** in its closure: the schedule lowered. Step 3.5
declares the same strings and blocks — its `layer_types` carries 48 entries
against 45 layers, the three multi-token-prediction layers included, so no
layer aligns. The cluster reads as "attention schedule"; the fix is a length
rule.

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

| outcome | baseline (88) | after wave 7 (109) | share now |
|---|---:|---:|---:|
| GREEN — representable | 17 | 28 | 26% |
| AMBER — identified, no implementation | 6 | 7 | 6% |
| RED — semantic gap | 65 | 74 | 68% |
| BUG — should work, doesn't | 0 | 0 | 0% |
| *unreachable (gated/absent/no config)* | *6* | *8* | *not evidence* |

GREEN includes **Qwen3.8-27B, Gemma 4 26B-A4B, Granite 4.1/4.2 (3b/8b/30b),
gpt-oss 20b+120b, Kimi-Linear-48B, Mixtral 8x7B/8x22B, Gemma 2, Granite 3.3,
Ministral-8B** — encoded from their own declarations, no per-model code.
Yi-1.5-6B joined them in wave 3, on `pretraining_tp` alone.

AMBER is **DeepSeek V3.2, V4-Flash, V4-Pro, GLM-5.2, GLM-5.3, GLM-5.3-Flash,
Hy4-preview** — every frontier sparse-attention-indexer model, and coherently
so. VINDEX3
identifies the component and says it has no implementation. That is a
success for the representation, not a failure: `UnsupportedComponent` is a
statement of *knowledge*, not of severity.

## Coverage by semantic envelope

The claim worth publishing is not a count of model names. It is that the
representation covers the *shapes* the open ecosystem is built from. This
table is an organising view over the same 109 scored rows. Each row's envelope
is declared **on the row** in [`matrix.json`](matrix.json) — the mixer first,
then how its FFN is sparse — so a generation that changes shape (DeepSeek V3.2
adds the indexer; Nemotron Nano-4B has no experts) is filed by what it
declares rather than by its lineage. `scripts/arch_sweep.py envelopes` derives
the table; the verdict columns are the sweep's, not the table's.

| envelope | lineages | rows | gens | GREEN | AMBER | RED |
|---|---|---:|---:|---:|---:|---:|
| Dense Transformer | bitnet, falcon-dense, gpt-neox, gpt2, granite-dense, internlm, llama-dense, olmo, phi, qwen-dense, starcoder2, yi | 28 | 21 | 13 | 0 | 15 |
| Hybrid local ↔ global attention | exaone, gemma2, gemma3, gemma4-dense, gemma4-moe, mistral-dense | 19 | 10 | 6 | 0 | 13 |
| Classic sparse MoE | llama4-moe, mistral-moe, mixtral-moe, olmoe, phi-moe, qwen-moe | 11 | 8 | 5 | 0 | 6 |
| Fine-grained MoE | command-a, exaone-moe, glm, gpt-oss, hunyuan, minimax-m2, step | 12 | 11 | 2 | 0 | 10 |
| MLA + MoE | deepseek-mla, kimi-moe | 7 | 7 | 0 | 0 | 7 |
| Sparse-indexed / DSA + MLA + MoE | deepseek-mla, deepseek-v4, glm-5, hunyuan | 7 | 5 | 0 | 7 | 0 |
| Linear-attention hybrid | qwen-3.8, qwen-dense, qwen-moe | 9 | 4 | 1 | 0 | 8 |
| KDA + MLA | kimi-k3, kimi-linear | 2 | 2 | 1 | 0 | 1 |
| SSM + attention | falcon-hybrid, granite-hybrid, jamba, nemotron-hybrid | 6 | 4 | 0 | 0 | 6 |
| SSM + attention + latent MoE + MTP | nemotron-hybrid | 4 | 4 | 0 | 0 | 4 |
| Conv + attention hybrid | lfm2, lfm2-moe | 4 | 3 | 0 | 0 | 4 |

Read across: the sparse-indexed frontier is **seven for seven AMBER** — every
DeepSeek V3.2/V4, GLM-5.x and Hy4 row names its indexer and is refused, none
is approximated — and the three recurrent envelopes (SSM, latent-MoE
Nemotron, conv) are entirely RED because their blockers are the mixer
vocabulary itself, not any spelling. Both should stay that way until the
mixer exists. Multi-token prediction and the multimodal container cut across
every envelope — 30 and 29 rows respectively — and are the two largest
cross-cutting debts.

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

REDs by distance from the text gate, smallest first — every row blocked on
three subjects or fewer, plus the two four-subject rows whose four are one
cause. Each is a whole family-generation:

| blocking | checkpoints | subjects |
|---|---|---|
| 1 | Falcon3-1B-Base | `activation` — `activation: swiglu` beside `hidden_act: silu` — read by nothing in transformers 5.5.0; agrees with the gated-SiLU FFN. Vestigial, read-and-check |
| 1 | Qwen1.5-MoE-A2.7B | `shared_expert_intermediate_size` — the Qwen-style *gated* shared expert; DeepSeek's ungated `n_shared_experts` already lowers (Kimi-Linear is GREEN) |
| 1 | SmolLM2-135M | `is_llama_config` — `true` under `model_type: llama` — read by nothing in transformers 5.5.0. Vestigial, read-and-check |
| 2 | phi-4 | `architecture_family`, `original_max_position_embeddings` — `phi3` is unregistered (fused `qkv_proj` / `gate_up_proj`) — a family entry, not an alias |
| 3 | GLM-4.5-Air, GLM-4.6 | `architecture_family`, `num_nextn_predict_layers`, `use_qk_norm` |
| 3 | Phi-3-mini-4k-instruct | `architecture_family`, `original_max_position_embeddings`, `sliding_window` |
| 3 | Qwen3.5-0.8B, Qwen3.5-27B, Qwen3.5-9B | `mrope_interleaved`, `mrope_section`, `partial_rotary_factor` — the rule already claims `PositionPolicy::MRope`; no built component answers the probe. One lowering, one family-generation |
| 3 | Qwen3.8-2.4T-A95B | `hidden_act`, `num_experts_per_tok`, `shared_expert_intermediate_size` — the MoE keys the `qwen3_5_moe` parser does not carry; the dense sibling carries them |
| 3 | bitnet-b1.58-2B-4T-bf16 | `hidden_act`, `linear_class`, `quantization_mode` — BitNet's ternary representation |
| 3 | gemma-4-E4B-it | `hidden_size_per_layer_input`, `num_kv_shared_layers`, `per_layer_model_projection` — per-layer input embeddings and KV sharing — execution semantics, not spelling |
| 4 | EXAONE-4.0-1.2B | `architecture_family`, `execution_surface`, `hidden_act`, `rms_norm_eps` — all four follow from the unregistered `exaone4` family |
| 4 | gemma-3-1b-it | `rope_local_base_freq`, `rope_theta`, `sliding_window_pattern` — `gemma3_text` at the ROOT: `rope_theta` is *mismatched* ("resolution does not honour the declared value") where the same keys under `text_config` lower on the 4B and 27B. A flat-layout parse, and the one mismatch in the frontier |

Five of the twelve entries are the recurring *kinds* the census predicted — two
vestigial keys, a gated variant of a component that already lowers, a
lowering the rule already names, and a flat-layout parse of a family that is
green when nested. Together they clear seven rows across five
family-generations without a new kernel. `attention_schedule`, at reach 23,
clears **none** alone: its `layer_types` blockers are missing mixer vocabulary
(`conv`, `mamba`, sparse spans) or Step's length rule, and every carrier has
three to ten other clusters. Reach ranks; the cover predicts.

## Caveats

- The engine column is computed from the `model_type` the plan reports
  (the text component's). For K3 that is `kimi_linear`, so "recognised"
  there means a string matched — not that the model would be served
  correctly.
- Eight rows are unreachable: four gated without access (Command-R ×2,
  Command A, Jamba-Mini-1.7), three absent at the id swept (TinyLlama_v1.1,
  Baichuan2-7B-Base, state-spaces/mamba2-780m), and Mistral Large 3, which
  ships Mistral-native `params.json` with no `config.json` in any variant.
  They are excluded from every percentage.
- GREEN is a claim about *representation*, established from headers.
  Only Qwen3-0.6B was carried through encode/verify in this pass.
