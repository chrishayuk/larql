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

| Metric | Baseline | Waves 1+3 | rope | moe | sliding window | frontier census† | partial rotary | vestigial pair | qwen MoE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantics version | 1 | 3 | 4 | 5 | 6 | 6 (held) | 7 | 8 | 9 | 10 | **11** |
| GREEN | 17 | 18 | 21 | 26 | 28 | 28 | 31 | 33 | 38 | 38 | **38** |
| AMBER | 6 | 6 | 6 | 6 | 6 | 7 | 7 | 7 | 7 | 3‡ | **3** |
| RED | 65 | 64 | 61 | 56 | 54 | 74 | 71 | 69 | 64 | 68‡ | **68** |
| **BUG** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| **silent drops** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| text-closure blockers | 886 | 776 | 756 | 671 | 668 | 1109 | 1091 | 1089 | 1076 | 1058 | **1051** |
| K3 clusters remaining | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | **7** |

‡ **The AMBER and RED columns are not comparable across wave 9 → 10.** Wave
11 tightened the sweep's own classifier: AMBER means "component identified,
no implementation", so it is no longer awarded to a checkpoint whose
`model_type` matches no registered family — that credits an identification
that has not happened. Four rows move AMBER → RED on that change alone
(Hy4-preview, GLM-5.2, GLM-5.3, GLM-5.3-Flash), and none of them moved
because of a semantic change. GREEN, BUG, silent drops and the blocker
count remain comparable throughout.

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

**Wave 8 — `partial_rotary_factor` inside `rope_parameters` (semantics 7).**
The first fix chosen from the 109-row corpus, and the cheapest kind there is:
a key read from the wrong spot. The parser read the legacy top-level
`partial_rotary_factor` and Gemma 4's per-layer-type block, never
`rope_parameters.partial_rotary_factor` — the transformers-5.x flat form, the
spelling the reference reads
(`rope_parameters_dict.get("partial_rotary_factor", 1.0)`), and the only one
every Qwen3.5 checkpoint uses. Qwen3.8 writes both and was GREEN; Qwen3.5
wrote one and lost the fraction, so no layer carried one, the partial and
multi-axis rotary probes answered nothing, and three leaves refused a family
this build executes. Precedence now mirrors `standardize_rope_params`: top
level (the reference copies it *into* the block, overwriting), then the
per-type block, then the flat block.

Frozen before the code in
[`forecasts/wave8-partial-rotary-factor-in-rope-parameters.json`](forecasts/wave8-partial-rotary-factor-in-rope-parameters.json):
three Qwen3.5 dense rows green, three Qwen3.5 MoE rows from six blockers to
three, and three rows that *declare* the key staying exactly where they were
(Qwen3.8-Flash-Next, Nemotron 3, GLM-4.7-Flash). **Every arm held**: GREEN 31,
RED 71, blockers 1091, six rows changed and no others. Both new tests were run
before the parser change and failed with the corpus's own message, then
passed after it.

Two things worth keeping. The earlier parser test
`partial_rotary_factor_is_read_from_the_declared_spot` passed only because
its fixture mirrored Qwen3.8 and declared both spellings — it pinned a
coincidence of the fixture, not the parser; the nested-only test now sits
beside it. And outside VINDEX3 this was a fail-open wrong answer: the engine
path would have rotated all 256 head dims of a Qwen3.5 attention layer where
the checkpoint asks for 64. VINDEX3 refused those checkpoints; the engine
did not. The fix lands in the shared parser, so both doors now read the
reference's spelling.

**Wave 9 — the vestigial pair (semantics 8).** Two keys read by no
implementation, ours or upstream, checked by exact-word grep of transformers
5.5.0: Falcon3-1B-Base's `activation: "swiglu"` beside `hidden_act: "silu"`
under `model_type: llama`, and SmolLM2-135M's `is_llama_config: true`. Same
treatment as the earlier `use_mrope` / `rope_interleaved` pair: read,
stored, credited as consumed, and **checked against what actually
resolves** rather than echoed. `swiglu` is "gated, SiLU on the gate" in one
word, judged against the FFN shape the execution surface carries through one
two-way table (`geglu` on a SiLU-gated stack resolves to `swiglu`; a plain
`silu` is the ungated shape and mismatches too). `is_llama_config` is judged
against the registry entry the declared identity resolved to, so `true`
under a `model_type` nothing matches is refused, not believed.

Frozen before the code in
[`forecasts/wave9-vestigial-pair.json`](forecasts/wave9-vestigial-pair.json):
exactly two rows clear, nothing else moves. **Held**, and held at whole-model
granularity too — the blocker *set* changed on exactly those two rows across
109 plans. That second check mattered, because a fix rode along: the
path-to-component mapping read any first segment ending in `_config` as a
component section, so `is_llama_config` was sent to a component named
`is_llama` that no graph builds. A section is a segment with something after
it. The controls ran in three stages: both tests failed as `Unknown` before
the rules, then as "read by nothing in any registered parser" with the rules
but no parser read — the consumed-key contract holding — then passed.

**Wave 10 — Qwen's MoE facts and the gated shared expert (semantics 9).**
Three defects, all reaching the same five rows, and none of them clearing a
row on its own.

*The FFN presence rule read only the dense width.* `has_ffn` was
`resolved.intermediate_size > 0`, and `Qwen3_5MoeTextConfig` is `@strict`
with no `intermediate_size` field at all — every one of its layers is a
`Qwen3_5MoeSparseMoeBlock`, so its FFN's width lives in
`moe_intermediate_size`. A 397B mixture of experts was therefore graded as
running no FFN op, and `hidden_act` and `num_experts_per_tok` reported
"no built component answered the probe": the routed block was declared and
judged, then had nowhere to be read back from. Qwen3-30B-A3B slipped
through only because it declares *both* widths. Measured before the fix,
the population is exactly four rows — the only primary-text components in
117 plans with an execution surface and no `ffn` on it. `FfnSurface`'s
dense width is now `Option`, so a wholly-routed stack **states** the
absence rather than asserting a zero, and the GGUF geometry and metadata
writers refuse instead of stamping `feed_forward_length: 0`.

*The shared branch's width was derived where it should be declared.*
`SharedExpertOp` sized itself at `expert_intermediate_size *
shared_experts`. That is Kimi's fact — `KimiSparseMoeBlock.__init__` sizes
one wider `KimiMLP` — and it is wrong for the lineage that declares the
width outright: Qwen1.5-MoE states 5632 against a derived 1408, a fourfold
error, and Nemotron-3 Nano states 3712 against 1856. The width is now read
(in both declared spellings) and is the single authority for the op's size
*and* for the shared-expert operand shapes. The derivation survives as the
trait default, which is what the count-only lineage means.

*Qwen3.5-MoE stacks its expert bank.* It ships `mlp.experts.gate_up_proj`
and `mlp.experts.down_proj` as 3-D parameters where its siblings ship a
tensor per expert. Left undeclared, fixing the presence rule would have
made the routed FFN present while the judgment still said `PerExpert`, and
operand closure would have demanded 256 per-expert operands the checkpoint
does not ship. Greening a row on a wrong expert format is the fail-open
this instrument exists to catch, so it is in the same wave.

Beside them, Qwen's shared expert is **gated**: `out = routed +
sigmoid(shared_expert_gate(x)) · shared(x)`. The gate is a `[1, hidden]`
projection one name away from the branch's own SwiGLU gate
(`mlp.shared_expert_gate.weight` against
`mlp.shared_expert.gate_proj.weight`) and differs from it in every
dimension, so it gets its own operand role. Dropping it does not fail —
it runs the branch at full weight on every token — which is why it is
declared rather than inferred from an operand that happens to be present.

Frozen before the code in
[`forecasts/wave10-qwen-moe-shared-expert.json`](forecasts/wave10-qwen-moe-shared-expert.json):
five rows clear, GREEN 38, blockers 1076, and an explicit *nothing moves*
prediction for the five other rows that declare the same key. **Exact on
every arm**, including at whole-model granularity — the blocker set
changed on those five rows and nowhere else, and no row anywhere gained a
blocker.

**The first score disagreed, and the forecast was right.** The first
re-plan came back five blockers lower than predicted, with Qwen3.8-Flash-Next
and the four Nemotron-3 rows each losing their shared-expert blocker —
rows whose components build no execution surface at all, so nothing could
have checked anything. The cause was a classification: the new key had
been filed as `tensor_semantic`, and that bucket's findings are reported
`carriage: parsed` / "read by a registered parser", so the carriage rule
and its probe were never consulted. The key was clearing because a parser
had touched it. That is wave 9's lesson restated — read *and check*, never
echo — and it is the reason a forecast is written down first: the number
that disagreed was the code's, not the prediction's. Moved to the
execution-semantic bucket beside the shared-expert count that already
lives there, the declared width is compared against the width the branch
will be built at, and a component that resolved no branch refuses.
`the_shared_expert_width_carries_only_where_a_branch_resolves` is the
standing guard, with both arms: the same consumed key carries where a
branch resolves and blocks where none does.

**A defect found on the way.** The GGUF orientation path sized the
shared-expert tensors by the *dense* width. That coincides on Qwen1.5-MoE
(5632 either way) and is wrong everywhere else — and on Qwen3.5-MoE, which
has no dense width, it left the shared expert unoriented entirely. Pointed
at the same single authority as everything else.

**What did not happen, and why it is worth recording.** The inert clusters
reach 34 checkpoints, which looked like a large GREEN wave. It was one row
(Yi-1.5-6B). Cluster *reach* counts checkpoints a fix touches; only a
checkpoint whose **every** blocker is retired changes verdict. The greedy
cover already said this — `inert.training_only` cleared 1 — and it was the
better predictor. Reach ranks work; the cover predicts verdicts.

**Wave 11 — a stack may normalise its sublayers' output (semantics 10).**
`NormPlacement` knew two transformer shapes. OLMo-2, OLMo-3 and EXAONE-4
declare a third, and because their operand estate matched neither, their
execution surface refused to build and *every probe on those components
answered nothing at all*. Seven rows, 4–9 blockers each, and most of those
blockers were not facts about the model — they were the silence left by an
absent surface.

The variant is read from which norms EXIST, not from what any one of them
is called, and that is load-bearing: these families' `post_attention_layernorm`
is a true post-norm where a Llama stack's tensor of the same name is the
pre-FFN norm. The discriminator is `post_feedforward_layernorm` without
`input_layernorm`.

**The op plan refuses to lower it.** `LayerPlan.pre_attention_norm` is a
required `NormOp` that every executor reads before its sublayer, and a
post-norm stack carries no such operand; lowering it as the two-norm shape
would find an operand for every site and would *run*, normalising the wrong
tensor at every layer. The closure vocabulary gained
`UnimplementedSemantic` to say this — the exact opposite of
`UnjudgedSemantic`, and a reader must be able to tell them apart: one means
"we do not know what this model does", the other "we know exactly what it
does and cannot yet do it".

Frozen before the code in
[`forecasts/wave11-post-norm-placement.json`](forecasts/wave11-post-norm-placement.json)
as a **revelation** forecast — no row clears, GREEN unchanged, seven
surfaces build, sixteen blockers retire because a probe can finally answer,
and three answer and still refuse. **Exact on every per-row arm**, including
the sharp one: `sliding_window_pattern` ("LLLG", no schema field for a
period string) and `num_nextn_predict_layers` (1, against a schema that
represents only MTP's absence) both started answering and both correctly
stayed blocking.

**The forecast caught a fail-open for the second wave running.** The first
score came back at 1051, seven *better* than predicted, because each row
lost its incomplete-surface blocker with nothing replacing it — OLMo-2 fell
to a single blocker, its unresolved identity. The refusal was real, and the
plan report never consulted it. That row was one registry entry away from
reading `text closure: every declaration has a home` for a model VINDEX3
cannot build a single op for, and the next wave is exactly the one that
would have registered it. `NormPlacement::unimplemented_reason()` is now the
one authority both the op plan and the report read, and a complete-but-
unlowerable surface reports as an unsupported component. A wave scored on
"blockers went down" would have booked the bug as a win.

**And the same error was in the instrument.** Emitting that finding on seven
rows with unresolved identities exposed that the sweep would have called
them AMBER — "component identified, no implementation" — while their
`model_type` matched nothing. Four other rows were already sitting in that
state (Hy4-preview, GLM-5.2, GLM-5.3, GLM-5.3-Flash). AMBER now requires a
resolved identity. That is the ‡ footnote on the ledger above, and it is
declared separately because it moved four verdicts on its own.

**Wave 12 — post-norm placement EXECUTES (semantics 11).** Wave 11 could
represent the placement and refused to lower it. This wave makes it run,
and it is the first in a long while that expands the executable
architecture envelope rather than the representation vocabulary — so the
sweep is *not* its authority. The sweep can only show a refusal was
withdrawn; only parity can show the right thing replaced it.

Two facts made it far smaller than the 27-consumer count suggested, and
one made it necessary.

*The executor was already half right.* The generic path computes
`attn_out = post_attention.apply(raw_attn); h += attn_out` — the norm on
the sublayer output, before the add, which is the OLMo-2 semantic exactly.
What it could not do was run with **no** pre-sublayer norm. Post-norm
execution is the four-norm program minus the pre-norms, not a new residual
topology.

*An epsilon was coupled to a placement.* `QkNormOp` carried no epsilon of
its own; every executor read `layer.pre_attention_norm.eps` as the QK-norm
epsilon, and DeltaNet's gated norm and KDA's read it too. A post-norm stack
has no pre-attention norm to borrow from — and an epsilon and a placement
are unrelated facts. It is now `LayerPlan.declared_norm_eps`, the
component's declared epsilon in its own right. That decoupling is what made
the placement representable in the op plan at all.

*And a name means two causal positions.* `post_attention_norm` is applied
before the residual add in the generic path and **after** it in
`exec/kimi_mla_layer.rs`, where the operand of that name is semantically
the pre-FFN norm. Both are correct for their family, so the witness pins
the generic path by construction rather than by name.

**The witness is exact, not statistical.** Two containers carry the *same
numeric weights* and differ only in which norm names they ship, so any
behavioural difference is placement and nothing else. Under post-norm
placement nothing conditions the residual before attention, so layer 0's
observed attention input must be the embedding row **bit for bit** — and
the fixture states that row from its own generator, so the test compares
the executor against the checkpoint rather than against itself. Its control
is the same assertion against the pre-norm estate, which must *fail* to
hold. A third arm requires the two placements' logits to differ by a
margin, and its failure message says outright that every other assertion in
the file is worthless if it ever trips.

The negative control is the whole `larql-vindex` suite — 3744 tests, every
committed parity oracle (Kimi's 27-layer stack, MLA, KDA, Gemma 4,
gated-delta, conv-QKV), which are the existing pre-norm and four-norm
families. None moved. This is precisely the wave where "support a new
placement" could have changed the default path.

Forecast held exactly on the sweep: seven rows lose the
unsupported-component blocker, 1058 → 1051, and **no row clears** —
`architecture_identity` still blocks all seven. Two things are deliberately
not done here: reference parity against real OLMo-2 weights is gated on
wave 13 (no container can be built for an unregistered identity), and the
Metal trunk now **refuses** a post-norm layer explicitly rather than
lowering it as a shape it is not.

## The rerank, and what wave 11 settled — read this before picking wave 12

The greedy cover over semantic ideas is now **empty**: no single cluster
clears any of the 71 non-admissible rows on its own. Cluster-level fixes
are exhausted as verdict movers, which is what wave 10 already looked like
from the inside — it cut across `ffn_activation`, `moe_routing` and a
storage format, and no one of the three cleared a row alone.

**51 of the 64 RED rows are blocked on `architecture_identity`** — a
`model_type` no registry entry matches. That makes identity the most
*common* blocker, and the first version of this section claimed it was
therefore the causal one: that an unmatched identity means no component
builds, so a registry entry would retire the dependent probes for free.

**That claim was wrong, and checking it is what found the real gate.**
`phi3` and `glm4_moe` match nothing and their components build anyway —
they fall through to `GenericArch`, which is precisely the fail-open door
this programme exists to keep visible. Identity and surface are
independent.

The rows whose surface genuinely does not build say so in their own
finding, and there are **25 of them — 23 on norm placement**, in three
distinct shapes:

| operand evidence | rows | families | blockers each |
|---|---:|---|---|
| no per-layer norm operands at all | 12 | LFM2, Nemotron-H, DeepSeek-V4, Qwen3.8-Flash-Next | 13–36 |
| `post_attn` + `post_ffn`, no pre-norms | **7** | OLMo-2, OLMo-3, EXAONE-4 | **4–9** |
| `pre_attn` only | 4 | Cohere2-MoE, Jamba2, Falcon-H1 | 16–39 |

The middle group is the wave. `NormPlacement` knows two shapes — two-norm
(pre-only) and four-norm (pre+post) — and these seven declare a third the
vocabulary cannot express. Judged from the references rather than from the
name, all three families are byte-identical in shape:

```text
residual = h
h = attn(h)                        # NO pre-norm; the sublayer reads the raw residual
h = post_attention_layernorm(h)    # the norm applies to the sublayer OUTPUT
h = residual + h                   # ...before the add
```

That is `Olmo2DecoderLayer.forward`, `Olmo3DecoderLayer.forward` and
`Exaone4DecoderLayer.forward`, line for line. It is **not** classic
post-LN (`norm(x + attn(x))`, norm after the add) — the difference is
which tensor the norm sees, and picking the wrong one produces fluent
wrong output rather than a failure, so it is a variant to be declared and
not a placement to be inferred.

Neither fix clears a row alone: the placement leaves identity blocking,
and identity leaves the surface refusing. That pairing is the wave-10
shape again.

**What wave 11 settled, and what it leaves for wave 12.** The placement is
now represented and explicitly refused, so those seven rows report what is
actually true of them. Three routes out, and they are not equivalent:

| route | rows | status |
|---|---:|---|
| make post-norm **executable** | 7 | **done — wave 12.** CPU both paths; Metal refuses explicitly, pending its kernel |
| register `olmo2` / `olmo3` / `exaone4` | 7 | **wave 13**, and its expectation CHANGED — see below |
| the 12 rows with **no** per-layer norm operands | 12 | **wave 14**, a discovery wave (LFM2, Nemotron-H, DeepSeek-V4), 13–36 blockers each |

**Wave 13's expectation changed because wave 12 landed first.** While the
placement was unimplemented, registering the families would have moved
seven rows RED → AMBER without making one model runnable. Now the
placement executes, so the registry entry removes the *last* blocker on
some of them:

| row | after wave 13 | why |
|---|---|---|
| OLMo-2-0425-1B, OLMo-2-1124-7B | **GREEN** | identity is their only remaining blocker |
| EXAONE-4.0-1.2B | **GREEN** | same |
| Olmo-3-1025-7B, Olmo-3-1125-32B | RED | `rope_scaling.attention_factor` |
| EXAONE-4.0-32B | RED | `sliding_window_pattern` ("LLLG", no schema field) |
| EXAONE-4.5-33B | RED | five more, including MTP |

That also unlocks the parity wave 12 could not run: with a container
buildable for a registered identity, the real OLMo-2 checkpoint can be
scored against an HF reference oracle, which is a far stronger authority
than any synthetic fixture.

**Wave 14 is a discovery wave and must be preregistered as one.** Not
"support 12 models" — the question is which of four things is true, per
row: a legitimately normless architecture; a norm represented through
another object; a config dialect or naming gap; or an execution semantic
genuinely absent. The outcome to forecast is the *classification*, not a
verdict count.

Kimi K3 is the standing witness: 97 blocking subjects → 73, text-closure
blockers 72 → 48, and **7 semantic clusters both before and after**. The
spelling collapsed; the ideas did not. That gap is the whole argument for
counting ideas.

## The verdict

Scored against the **text-generation closure**, which the plan computes
itself — a checkpoint whose only blockers sit in a vision tower is not
evidence about its language model.

| outcome | baseline (88) | after wave 11 (109) | share now |
|---|---:|---:|---:|
| GREEN — representable | 17 | 38 | 35% |
| AMBER — identified, no implementation | 6 | 3 | 3% |
| RED — semantic gap | 65 | 68 | 62% |
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
| Dense Transformer | bitnet, falcon-dense, gpt-neox, gpt2, granite-dense, internlm, llama-dense, olmo, phi, qwen-dense, starcoder2, yi | 28 | 21 | 15 | 0 | 13 |
| Hybrid local ↔ global attention | exaone, gemma2, gemma3, gemma4-dense, gemma4-moe, mistral-dense | 19 | 10 | 6 | 0 | 13 |
| Classic sparse MoE | llama4-moe, mistral-moe, mixtral-moe, olmoe, phi-moe, qwen-moe | 11 | 8 | 5 | 0 | 6 |
| Fine-grained MoE | command-a, exaone-moe, glm, gpt-oss, hunyuan, minimax-m2, step | 12 | 11 | 2 | 0 | 10 |
| MLA + MoE | deepseek-mla, kimi-moe | 7 | 7 | 0 | 0 | 7 |
| Sparse-indexed / DSA + MLA + MoE | deepseek-mla, deepseek-v4, glm-5, hunyuan | 7 | 5 | 0 | 7 | 0 |
| Linear-attention hybrid | qwen-3.8, qwen-dense, qwen-moe | 9 | 4 | 4 | 0 | 5 |
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
| 1 | Qwen1.5-MoE-A2.7B | `shared_expert_intermediate_size` — the Qwen-style *gated* shared expert; DeepSeek's ungated `n_shared_experts` already lowers (Kimi-Linear is GREEN) |
| 2 | phi-4 | `architecture_family`, `original_max_position_embeddings` — `phi3` is unregistered (fused `qkv_proj` / `gate_up_proj`) — a family entry, not an alias |
| 3 | GLM-4.5-Air, GLM-4.6 | `architecture_family`, `num_nextn_predict_layers`, `use_qk_norm` |
| 3 | Phi-3-mini-4k-instruct | `architecture_family`, `original_max_position_embeddings`, `sliding_window` |
| 3 | Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B, Qwen3.8-2.4T-A95B | `hidden_act`, `num_experts_per_tok`, `shared_expert_intermediate_size` — the MoE keys the `qwen3_5_moe` parser does not carry; the dense sibling carries them. The three Qwen3.5 MoE rows arrived here from six in wave 8 |
| 3 | bitnet-b1.58-2B-4T-bf16 | `hidden_act`, `linear_class`, `quantization_mode` — BitNet's ternary representation |
| 3 | gemma-4-E4B-it | `hidden_size_per_layer_input`, `num_kv_shared_layers`, `per_layer_model_projection` — per-layer input embeddings and KV sharing — execution semantics, not spelling |
| 4 | EXAONE-4.0-1.2B | `architecture_family`, `execution_surface`, `hidden_act`, `rms_norm_eps` — all four follow from the unregistered `exaone4` family |
| 4 | gemma-3-1b-it | `rope_local_base_freq`, `rope_theta`, `sliding_window_pattern` — `gemma3_text` at the ROOT: `rope_theta` is *mismatched* ("resolution does not honour the declared value") where the same keys under `text_config` lower on the 4B and 27B. A flat-layout parse, and the one mismatch in the frontier |

Two of the nine entries are the recurring *kinds* the census predicted — a
gated variant of a component that already lowers, and a flat-layout parse of
a family that is green when nested. Waves 8 and 9 were three more of those
kinds (a key read from the wrong spot, two vestigial keys) and cleared their
five rows exactly as forecast. Everything else in the table is a family
entry or an execution semantic. `attention_schedule`, at reach 23,
clears **none** alone: its `layer_types` blockers are missing mixer vocabulary
(`conv`, `mamba`, sparse spans) or Step's length rule, and every carrier has
three to ten other clusters. Reach ranks; the cover predicts.

## After wave 9: the rerank, from the 109-row closure *(superseded)*

Kept as the record of what wave 10 was chosen from and how well the choice
scored. The current ranking is
[the current rerank](#the-rerank-and-what-wave-11-settled--read-this-before-picking-wave-12);
item 1 below is the wave that landed, and it cleared exactly the five rows
named here.

Every remaining text-closure blocker, by what kind of work retires it
(semantics 8; a row counts under every class it touches):

| class | blockers | rows touched | clears alone |
|---|---:|---:|---:|
| vocabulary — `unclustered`, or class `unknown` | 584 | 75 | **0** |
| real execution semantics — indexers, hyper-connections, SSM/conv mixers, MTP, quantised representations | 266 | 43 | 0 |
| carriage — clustered and judged, not yet carried | 214 | 65 | 6 |
| family entry — an unregistered identity | 25 | 25 | 0 |

The vocabulary class is 54% of the debt and moves **no verdict on its own**:
not one row is blocked only on unclustered subjects, and only two are blocked
on unclustered plus one other cluster. Its largest members are training and
framework-default keys (`ep_size`, `aux_loss_alpha`, `seq_aux`,
`use_mamba_kernels`, the `is_decoder` / `torchscript` / `use_bfloat16` quintet
that `PretrainedConfig` dumps into every Kimi config) and the whole-mixer
vocabularies of Nemotron and LFM2. Worth burning down for the blocker count
and for what the report *says*, never as a GREEN wave.

Verdicts move in the carriage class, and the greedy cover names the order:

| # | fix | rows it clears |
|---|---|---:|
| 1 | the `qwen3_5_moe` family's own MoE facts — `num_experts_per_tok`, `hidden_act` and the *gated* shared expert (`shared_expert_intermediate_size`), which the dense sibling already carries | 5 — Qwen1.5-MoE-A2.7B, Qwen3.5 35B-A3B / 122B-A10B / 397B-A17B, Qwen3.8-2.4T-A95B |
| 2 | `gemma3_text` at the root: `rope_theta` *mismatched* where the same keys lower under `text_config` | 1 — gemma-3-1b-it, and the one genuine mismatch in the frontier |
| 3 | a `phi3` family entry (fused `qkv_proj` / `gate_up_proj`, `original_max_position_embeddings`) | 2 — phi-4, Phi-3-mini |

**Scored.** Item 1 landed as wave 10 and cleared all five, and the note
below about naming the operator was the right instinct for the wrong
reason: the gated shared expert *was* a small operator, but two further
facts had to travel with it (a wholly-routed family has no dense width to
find an FFN by, and Qwen3.5-MoE stacks its expert bank), and neither was
visible from the cluster.

Only the first is a wave in the sense above, and even it is not pure carriage:
the gated shared expert is a small operator (DeepSeek's ungated
`n_shared_experts` already lowers — Kimi-Linear is GREEN — but Qwen gates
the shared branch with a sigmoid), so its forecast must name the operator,
not the cluster. After those three, every remaining row is in the family or
the real-semantics class, and the third class is where engineering time
should start to go: multi-token prediction (22 rows), the sparse indexer
(7, all AMBER by design), SSM and conv mixers (16), quantised representations
(22).

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
