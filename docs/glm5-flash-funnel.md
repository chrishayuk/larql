# GLM-5.3-Flash funnel v0.1 — admission, ledger and the two tracks

**Model:** `zai-org/GLM-5.3-Flash` — 321.3 B parameters, 45 layers, FP8 E4M3, 328.3 GB
**Status:** v0.1 — P0 (admission) **run and recorded below**; no weights downloaded, no code written yet
**Date:** 2026-08-27
**Relation to other ladders:** the KDA half of this model is [`k3-funnel.md`](k3-funnel.md) rung **R2** (Kimi Linear). This document does not replace that rung — it shows the rung is now on the critical path of two models instead of one.

---

## 1. Thesis

The expensive thing about a 328 GB checkpoint is not the download; it is discovering *after* the download that the architecture needs a rewrite. Everything in §3–§6 below was measured today, from **10.7 MB of safetensors headers and a 69 KB config**, using the admission instruments the repo already ships.

The claim this document makes is narrow:

> **GLM-5.3-Flash's admission verdict, tensor census, active-parameter ledger and residency arithmetic are all obtainable before acquiring the weights — and together they relocate the work from where the circulating analysis puts it.**

Three specific relocations, each argued from measurement in §4–§6:

1. The routed experts are **not** where the decode cost is. They are 97 % of the *checkpoint* (94.7 % in the dense stack, the rest in the MTP layer) and 50.5 % of the *active weight per token*. The other 49.5 % is what the tok/s question turns on.
2. The **KDA block is 28 % of active weight per token** — 4.68 B parameters — and the official FP8 checkpoint leaves **100 % of it in BF16**.
3. The KDA block is, tensor name for tensor name, **the block in `Kimi-Linear-48B-A3B-Instruct`, which is already on local disk** with its reference implementation beside it. The riskiest operator in the model has a free, local, layer-diffable ancestor.

## 2. The instrument: admission without the weights

`larql inspect-hf` and `larql vindex3 plan` read `config.json` plus each shard's **safetensors JSON header** — the 8-byte length prefix and the header bytes it announces. `scan_tensors` never touches tensor data ([`larql-models/src/inventory/tensors.rs`](../crates/larql-models/src/inventory/tensors.rs)). So the admission instruments do not need the weights; they need the headers.

[`scripts/hf_metadata_checkpoint.py`](../scripts/hf_metadata_checkpoint.py) fetches those headers over HTTP range requests and writes a stub checkpoint of `<u64 len><header json>` per shard:

```
python scripts/hf_metadata_checkpoint.py zai-org/GLM-5.3-Flash --out stub
larql inspect-hf stub --no-tensor-list --output inventory.json
larql vindex3 plan stub --output plan.json
```

**10.68 MB of stub stands in for 328.33 GB of checkpoint** — a ratio of 30,700:1.

**The fidelity self-check:** the stub's inventory reports `total_bytes = 328,326,771,576`, which equals the index's own declared `total_size` **exactly**, over all 76,108 tensors. Two independent stub builds agree on the tensor count, the byte total and the full plan summary. What a stub cannot do is anything that reads a tensor *value* — encode, verify, execute, and every parity gate still need the real bytes. It answers "would this be admitted, and what would it cost", which is the question worth answering before spending the download.

This generalises: any HF checkpoint can now be put through admission before it is acquired.

## 3. What the checkpoint actually is

Read from `config.json` and the headers, not from the announcement.

| | |
|---|---|
| Architecture | `Glm5NextForConditionalGeneration`, `model_type: glm5_next` |
| Layers | 45 + **1 MTP layer** (`num_nextn_predict_layers: 1`) |
| Attention | 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention`, pattern `L L L S` |
| Full-attention block | **MLA with NoPE** — `q_lora_rank 1536`, `kv_lora_rank 512`, `qk_nope_head_dim 256`, **`qk_rope_head_dim 0`**, `v_head_dim 256`, 64 heads |
| DSA indexer | 32 heads × 128, `index_topk 2048`, `index_kpool 4` with compression |
| MoE | 288 routed + **1 shared** expert, top-8, `moe_intermediate_size 2048`, first 3 layers dense (`intermediate_size 12288`) |
| Router | `scoring_func sigmoid`, `topk_method noaux_tc`, `e_score_correction_bias`, `routed_scaling_factor 2.5`, `norm_topk_prob` |
| Residual | **mHC** — `hc_mult 4`, `hc_sinkhorn_iters 20`; per layer `hc_{attn,ffn}_{fn,base,scale}` |
| Activation | SiLU with `swiglu_limit 10.0` (clamped GLU) |
| Vision | `glm5_next_vision` — 24 blocks, hidden 1024, patch 14, **image *and* video** tokens |
| Context | 1,048,576 |
| Stored | FP8 E4M3, `weight_block_size [128, 128]`, `activation_scheme dynamic`, 1,509 `modules_to_not_convert` |

**Four corrections to the circulating summary**, each material:

- **It is multimodal.** `Glm5NextForConditionalGeneration` with a `vision_config` and video tokens. This is the failure mode that blocks `Qwen3.8` extraction today — but see §4: here vision is only 4 of 46 blockers, not the wall.
- **There is a shared expert on every sparse layer.** The "8 × 25.2 M" arithmetic omits it; it adds 1.06 B active parameters per token, +12.5 % on the expert side.
- **The MTP layer carries a full 288-expert MoE of its own.** Routed experts live on **43** layers (42 sparse + MTP), not 42 — 304.4 B parameters, plus 7.43 B more in the MTP layer's other tensors.
- **The full-attention layers are MLA-NoPE**, not generic "DeepSeek-style sparse attention". `qk_rope_head_dim` is 0: there is no rotary component at all on those layers.

## 4. Admission verdict — measured 2026-08-27

`larql vindex3 plan`, on the branch's own build:

```
plan: 64 representable, 1 mismatched, 47 unrepresented, 0 interfaces — 46 blocking
Error: plan not admissible
```

| capability | admissible | required | blocking |
|---|---|---|---|
| `text_generation` | no | 90 | **42** |
| `image_conditioned` | no | 112 | 46 |
| `audio_conditioned` | no (unavailable) | 90 | 42 |
| `drafting` | no (unavailable) | 90 | 42 |

**Vision is not the wall here.** Only 4 of the 46 blockers are vision-owned, plus 4 root-level image/video binding tokens. Contrast `Qwen3.8`, where 12 of 30 were the tower and the text capability could not be rescued by a component filter. **The text side is the whole job: 42 blockers.**

`inspect-hf` also reports `generic_fallback: true` with two validation errors — `head_dim` "must be greater than 0" (the config declares `head_dim: 0` because MLA carries geometry in `qk_head_dim`/`v_head_dim` instead) and `head_dim_for_layer` "layer 0 returned 0". Per [`inventory/report.rs`](../crates/larql-models/src/inventory/report.rs), a generic fallback on a model with unconsumed keys is "the loudest red flag this report can raise".

### 4.1 The 42 text blockers, grouped

Every one is "declared by the checkpoint, read by nothing in any registered parser" unless noted.

| group | keys | what it is |
|---|---|---|
| **KDA geometry** (6) | `linear_attn_config.{num_heads, head_dim, short_conv_kernel_size, kda_layers, full_attn_layers, gate_lower_bound}` | GLM/Kimi spell the linear-attention block differently from Qwen3.8's `linear_*` keys — see §4.2 |
| **DSA indexer** (9) | `index_{head_dim, kpool, kpool_always_select_tail, kpool_compress, n_heads, topk, share_for_mtp_iteration}`, `indexer_{rope_interleave, types}` | no execution vocabulary at all |
| **mHC** (4) | `mhc`, `hc_mult`, `hc_eps`, `hc_sinkhorn_iters` | 4-wide residual stream |
| **MoE routing** (8) | `scoring_func`, `topk_method`, `routed_scaling_factor`, `norm_topk_prob`, `n_group`, `topk_group`, `moe_router_dtype`, `num_experts_per_tok` | sigmoid + no-aux-loss bias correction |
| **Stack shape** (4) | `first_k_dense_replace`, `mlp_layer_types`, `num_nextn_predict_layers`, `qk_head_dim` | dense/sparse split, MTP |
| **MLA** (1) | `mla_use_nope` | |
| **FP8 carriage** (3) | `quantization_config.{fmt, weight_block_size, activation_scheme}` | see §4.3 |
| **Other** (7) | `swiglu_limit`, `layer_types` (mismatched), `output_router_logits`, `router_aux_loss_coef`, + 3 | |

### 4.2 The plan was *optimistic* about the 34 KDA layers — FIXED 2026-08-27 (P1)

`attention_policy` is classified **representable**, reporting:

> `0 sliding / 0 full / 34 gated-delta recurrent / 11 declared span(s) this schema has no execution vocabulary for, 0 NoPE layer(s)`

and `target.execution_surface` reports **"execution surface complete (attention, ffn, norm, head)"**.

Both claims outrun their evidence, three ways:

1. **The six carriage sites that would populate that operator report their source keys as unread.** [`plan/carriage.rs`](../crates/larql-vindex/src/format/vindex3/plan/carriage.rs) routes `ExecutionSurface.linear_attention.{conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, state_dtype}` into `GatedDeltaOp` — but every one of GLM's `linear_attn_config.*` keys is classified `unrepresented`. The policy line is derived from the `layer_types` *string*, not from a built operator.
2. **`GatedDeltaOp` cannot express KDA** (§5.1). The schema's `dt_bias` is `[Hv]`; GLM's is `[8192] = Hv·Dv`, per-channel. There is no operand at all for the `f_a_proj`/`f_b_proj` decay gate.
3. **`0 NoPE layer(s)`** on a model declaring `mla_use_nope: true` and `qk_rope_head_dim: 0`.

This was [R14, gate–claim congruence](dec-funnel.md) at the instrument level: a gate licenses claims only over what it tests, and this one classified a layer-type spelling while reporting a conclusion about an operator.

**The repair.** `resolve_layer_kind` returned `LayerOperator::GatedDelta` for any layer whose `layer_types` entry spelled `linear_attention` — from the string alone, with no reference to whether the operator's geometry resolved. It now takes that as an argument and returns a new third variant, `LayerOperator::Recurrent`, when the geometry did not resolve: *a declared recurrence whose family this build cannot identify*. `declared_name()` answers `None` for it, so such a layer counts as **unexpressed** rather than as a runnable recurrence, and `attention_policy` — which was unconditionally `Representable` — now grades `Unrepresented` whenever any layer is unexpressed.

Two paired tests pin it (`plan/tests/recurrence_identification.rs`): the same fixture with and without the identifying geometry, differing in nothing else. The negative arm is the one that matters — without it, the positive arm would also pass on a build that had simply deleted `GatedDelta`.

**Consequence for this model, measured:** see §4.4.

### 4.3 FP8 block scales: the codec is half-built

The checkpoint is FP8 E4M3 with `weight_scale_inv` companions at 128×128 block granularity — **one `weight_scale_inv` per FP8 tensor — 37,338 of each**, exactly paired. LARQL has FP8 E4M3 **element** conversion (`larql-models/src/quant/fp8.rs`), and **`weight_scale_inv` appears nowhere in the codebase**. Reading this checkpoint at all needs the block-scale half. It is a small, well-bounded, entirely local piece of work with an exact oracle, and nothing else can start without it.

### 4.4 What the P1 fix changed, measured

Both plans re-run on the branch build, 2026-08-27:

| | before | after |
|---|---|---|
| GLM-5.3-Flash | 64 representable / 1 mismatched / 47 unrepresented — **46 blocking** | 63 / 1 / 48 — **47 blocking** |
| GLM `attention_policy` | `representable` — "34 gated-delta recurrent / 11 declared span(s) … no execution vocabulary" | **`unrepresented`** — "**45** declared span(s) … no execution vocabulary" |
| Kimi Linear | 20 blocking | 20 blocking (unchanged) |

The 34 KDA layers stopped claiming an operator this build does not have, and the finding now blocks. That is P1's whole objective, and it is met for GLM.

### 4.5 Kimi Linear exposes a *second* instance of the same defect — and it is worse

Re-running Kimi with the fix produced no change at all, which is itself the finding:

> `per-layer policy recorded on component target: 0 sliding / 27 full, 0 NoPE layer(s)`

**Kimi Linear has 20 KDA layers and 7 full-attention layers. The plan reports a 27-layer full-attention tower.** It does not disclose, does not block, and does not mention a recurrence — because Kimi declares its hybrid in `linear_attn_config.{kda_layers, full_attn_layers}` and carries **no `layer_types` array at all**. Every layer therefore has `declared_span: None`, `resolve_layer_kind` takes the `None` arm, and each one resolves `Softmax` + `Full`.

This is precisely the defect [`LayerOperator::GatedDelta`] was introduced to remove — *"before it, every one of Qwen3.8's 48 recurrent layers resolved to `AttentionSpan::Full` and the graph reported a 64-layer full-attention tower"* — live again through a different config spelling. It is more dangerous than GLM's version: GLM's stack at least announces that something is unexpressed, whereas Kimi's produces a confident, wrong, non-blocking answer, and a KV planner reading it would size a full per-position cache for 20 layers that have no per-position state.

**P1 does not fix this, and cannot.** The unidentified-recurrence variant keys off a *declared* recurrence; Kimi declares one in keys nothing reads. Closing it means reading `linear_attn_config` — which is P3, and is the reason the KDA rung has to precede any Kimi encode. **A Kimi container cut today would encode as a 27-layer full-attention model.**

### 4.6 The index bases differ between the two models

Measured, not assumed:

| | `num_hidden_layers` | `kda_layers` | `full_attn_layers` | covers |
|---|---|---|---|---|
| GLM-5.3-Flash | 45 | 34 entries, 0–44 | 11 entries, 3–43 | **0…44 — zero-indexed** |
| Kimi Linear | 27 | 20 entries, 1–26 | 7 entries, 4–**27** | **1…27 — one-indexed** |

GLM's `kda_layers` agrees element-for-element with the layer positions its `layer_types` array marks `linear_attention`, so its base is independently corroborated. Kimi's `full_attn_layers` contains `27` against 27 layers, which is out of range zero-indexed and exact one-indexed.

**Same key name, same config section, different base.** Any reader built for one and pointed at the other is off by one on every layer — and off-by-one on an attention interleave is the failure that reads as a plausible model producing subtly wrong output. The P3 reader must take the base as a per-family fact and validate the union covers the stack exactly, rather than trusting either convention.

### 4.8 P3b — KDA is first-class IR (2026-08-27)

`LayerOperator::Kda` and `KdaOp` are siblings of `GatedDelta`/`GatedDeltaOp`, not modes of them. Both checkpoints now resolve the operator:

| | `attention_policy` | blocking |
|---|---|---|
| Kimi Linear | `representable` — `0 sliding / 7 full / **20 KDA recurrent**; 20 layer(s) represented but NOT executable` | 21 → **20** |
| GLM-5.3-Flash | `unrepresented` — `0 sliding / 0 full / **34 KDA recurrent** / 11 unexpressed; 34 layer(s) represented but NOT executable` | 47 → **45** |

Same fifteen operands, same contracts, different geometry: Kimi 32 heads × 128 (value width 4096), GLM 64 × 128 (8192). Kimi proves the operator exists; GLM proves it is not shaped around Kimi.

**`represented` and `executable` are now separate facts.** `LayerOperator::has_executor()` answers the second, and the plan states it in its own clause. A container can describe a KDA layer completely — every operand bound, every dimension stated — and still have nothing able to run it. Collapsing the two is precisely how a merely-*named* operator came to be reported as executable (§4.2). The two execution paths refuse KDA explicitly rather than falling through to another operator.

#### 4.8.1 Role classification had to become layer-aware, and the reason is brutal

KDA's tensors are named `self_attn.q_proj.weight`, `self_attn.o_proj.weight` — **byte-identical to softmax attention**. On Kimi Linear, measured:

| suffix | on its 20 KDA layers | on its 7 full layers |
|---|---|---|
| `self_attn.q_proj.weight` | `[4096, 2304]` — Hv·Dv | `[6144, 2304]` — MLA query |
| `self_attn.o_proj.weight` | `[2304, 4096]` | **`[2304, 4096]` — identical** |

**Neither the name nor the shape separates the recurrence's output projection from the softmax one.** Only the layer's operator does. So `classify_stack_tensor` gained a layer-aware form taking the operator from the graph's per-layer table — which makes P3a's interleave carriage a *precondition* for KDA operand binding, not a convenience. The layer-blind form survives only for norms, which cannot collide, and says so.

#### 4.8.2 What the op states, and why

The rule the op is written to satisfy: **the recurrence must be reconstructible from the op plus its bound operands alone** — no consumer may need to know a container came from Kimi or from GLM. So `KdaOp` carries `num_heads`, `head_dim`, `conv_kernel`, `gate_rank` and `gate_lower_bound` explicitly. `gate_rank` is the interesting one: no config declares it, so it is resolved **once** from `f_a_proj`'s row count at build time and then stated, rather than left for every consumer to recover from a shape.

Eleven of the fifteen operands have shape contracts from config geometry alone, including the discriminator — `dt_bias` is `[Hv·Dv]`, per channel, against Gated DeltaNet's `[Hv]`, per head. The four low-rank gate factors have no per-operand contract (the rank is undeclared) and answer `None`; their agreement is a closure fact between the pair, not a shape invented here.

### 4.7 The invariant, recorded

Three instances of one failure are now on record, across three checkpoints and two mechanisms:

| checkpoint | mechanism | symptom |
|---|---|---|
| Qwen3.8 | `layer_types` read but never consulted | 48 recurrent layers resolved `AttentionSpan::Full` — a 64-layer full-attention tower |
| Kimi Linear | interleave declared in `linear_attn_config`, which nothing read | 20 recurrent layers resolved full — a 27-layer full-attention tower |
| GLM-5.3-Flash | operator inferred from the `layer_types` *spelling* | 34 KDA layers claimed as executable Gated DeltaNet |

The first two are the same bug through different keys; the third is its mirror. All three share one shape, so the rule is worth stating once rather than fixing three times:

> **Unknown topology may block admission. It may never silently degrade into another executable topology.**
>
> Corollaries, each of which one of the three violated:
> 1. A layer-type *label* is not evidence of an *operator*. Identify the operator from geometry or operands; a label alone yields "recurrence, family unknown" (§4.2).
> 2. An absent declaration in one spelling is not an absent declaration. Look for every spelling the family uses before concluding the checkpoint said nothing (§4.5).
> 3. `declared_span: None` must mean **unresolved**, not "therefore full". A default span is only admissible where the family explicitly defines absence as full attention.

Corollaries 1 and 2 are closed (P1, P3a). **Corollary 3 is open**: `resolve_layer_kind`'s `None` arm still answers `Softmax` + `Full`, and it is load-bearing for every single-attention-type model that legitimately declares nothing. Closing it means distinguishing "this family has one attention type, so absence is full" from "this family is hybrid and absence means we failed to read it" — a family fact, not a default. Recorded here rather than fixed, because the two models that would have exercised it now declare their topology explicitly.

### 4.9 Inkling-Small: the philosophy generalised, the mechanism did not (2026-08-27)

`thinkingmachines/Inkling-Small` put through the header-only funnel as an out-of-lineage control — a third architecture family, admitted without downloading it.

**0.12 MB of headers stood in for 531,912,898,740 bytes — 4.4 million to one**, and the inventory's `total_bytes` matched the index's declared `total_size` exactly over all 1,048 tensors. (The first run lost three shards to short range reads that `curl` reported as success; `hf_metadata_checkpoint.py` now retries rather than writing a truncated header, because a stub that silently misreports is worse than one that fails.)

Verdict: **49 blocking, 35 for text**, `generic_fallback: true`, four components (`text`, `vision`, `audio`, and a first-class **`mtp_config`**).

**And it reproduces the §4.7 defect, for the third time through a third mechanism:**

> `attention_policy`: **`representable`** — `0 sliding / 42 full, 0 NoPE layer(s)`

Inkling-Small has **35 sliding layers at a 512-token window and 7 global layers**. It declares them in `text_config.local_layer_ids`, which nothing reads. Measured: 35 entries, zero-based, covering 0–40, with the 7 global layers implied at 5, 11, 17, 23, 29, 35, 41.

This instance is **worse than cosmetic**. Kimi's misreport named the wrong operator; this one tells a KV planner that 35 layers retain an unbounded prefix when their window is 512 — against a 1,048,576-token context. That is a residency error, silently.

**The lesson for P3a's mechanism.** `LinearAttnInterleave::resolve` proves its index base by requiring **two sets to partition the stack exactly**. Inkling declares **one** set with the complement implied, and its two kinds are *sliding vs full*, not *recurrent vs full*. So the reader that fixed Kimi cannot read Inkling, and the invariant's corollary 2 — "an absent declaration in one spelling is not an absent declaration" — is now violated by a spelling P3a did not anticipate. Four scopes are now known:

| checkpoint | key | shape | base |
|---|---|---|---|
| Qwen3.8 | `layer_types` | per-layer array | — |
| GLM-5.3-Flash | `linear_attn_config.{kda,full_attn}_layers` | two sets, partition | zero |
| Kimi Linear | same | two sets, partition | **one** |
| Inkling-Small | `text_config.local_layer_ids` | **one set, complement implied** | zero |
| Inkling-Small MTP | `mtp_config.local_layer_ids` | one set, **for its own sub-stack** | zero |

The generalisation this asks for is a *declared interleave* abstraction over `(scope, kinds, sets-or-array, base)` — not another special case beside `DeclaredInterleave`.

**Other surface this build has no vocabulary for**, recorded so it is not rediscovered: relative position (`d_rel: 16`, `rel_extent: 1024`) with **no RoPE anywhere** — yet the plan reports `0 NoPE layer(s)`, so position resolved to a default; per-layer-type attention geometry (`swa_head_dim`, `swa_num_attention_heads`, `swa_num_key_value_heads`); short convolution on attention (`use_sconv`, kernel 4); 256 experts top-6 with **2 shared** and a `shared_expert_sink`; `route_scale`, `norm_after_topk`, `use_gate_bias`; muP output scaling (`logits_mup_width_multiplier`); a padded vocabulary (`unpadded_vocab_size` 200,058 against `vocab_size` 201,024); log-scaled attention (`log_scaling_n_floor`, `log_scaling_alpha`); and 8 MTP heads.

**The tensor layout is also new**: 1,048 tensors for 532 GB, against GLM-5.3-Flash's 76,108 for 328 GB — experts are packed, not per-expert. `target.execution_surface` is incomplete for a reason that follows: *"stack carries no per-layer norm operands"*. And the stack is named `model.llm.*` rather than `model.language_model.*`, so `model.llm.embed`, `model.llm.unembed` and `model.audio.encoder` are unplaced.

**What this does not change:** KDA remains the critical path. Inkling is recorded, not scheduled.

### 4.10 P3c-0 — topology truthfulness, generalised (2026-08-27)

`DeclaredInterleave` was replaced, not extended. `config/interleave/` is one abstraction over `(scope, kinds, membership, base)`; every spelling is a reader that produces the same `Declaration`s, so a new checkpoint adds a reader and never a second resolution rule.

**The invariant, now load-bearing:** a declared hybrid topology must resolve to exactly one kind for every layer in its scope. Overlap, a hole, an out-of-range index, an ambiguous base, or a length mismatch each make the declaration *unresolved*, and unresolved blocks. None may fall through to full attention.

| model | before | after |
|---|---|---|
| Kimi Linear | `0 sliding / 7 full / 20 KDA recurrent` | unchanged ✓ |
| GLM-5.3-Flash | `34 KDA recurrent / 11 unexpressed` | unchanged ✓ |
| **Inkling-Small** | **`0 sliding / 42 full`, representable** | **`35 sliding / 7 full`**, windows 512 on exactly those 35 |
| Inkling MTP | — | resolves its own 6/2 split in its own 8-layer scope |

**Base resolution stays data-driven** — P3a's partition proof generalised rather than replaced. For a two-set partition at most one base can hold, because `0..n` and `1..=n` differ on whether they contain `0`. For **one set plus an implied complement**, both bases *can* hold — `{1,2,3}` in a 5-layer scope reads as layers 1–3 zero-based and 0–2 one-based, each leaving a well-formed complement — so ambiguity became reachable and is now its own blocking error, with a test.

Two things this rung got right by being forced to:

- **The declaration is authoritative for the span; the resolved boolean is not.** That boolean answers from whichever key the parser happened to read, and on a family whose interleave it cannot read it answers "full" for every layer. Authority moved to the graph; `plan::compare` keeps grading the declared array against the boolean, so the comparison it makes stays a real one.
- **`Unexpressed` is per layer.** An entry with no kind fails *its* layer, not the array. Failing the array made GLM report 45 unexpressed and hid the 34 it understands — worse information from a stricter-looking rule.

Provenance travels with every resolution: sources, encoding, proven base, scope. Kimi records `PartitionSets` / `One`; Inkling records `ExplicitSetWithComplement` / `Zero`.

**And the position lie is closed.** `rope_base` carries a default, so Inkling-Small — which declares `d_rel: 16`, `rel_extent: 1024` and no rope key anywhere — resolved to `Rope { theta }` on all 42 layers. `PositionPolicy::Relative { d_rel, extent }` now carries the declaration; both execution paths refuse it rather than skipping position, which would run the model unpositioned and still produce fluent text.

### 4.11 P3c-1 — Kimi semantic closure, 20 → 1 blocking (2026-08-27)

| | before | after |
|---|---|---|
| Kimi Linear | 20 blocking | **1** |
| GLM-5.3-Flash | 45 | **36** |
| Inkling-Small | 48 | **45** |

The pattern throughout: `source key → canonical semantic → provenance retained → finding closes`. Kimi's spellings became aliases of the fields their DeepSeek-lineage twins fill (`num_shared_experts`→`n_shared_experts`, `moe_renormalize`→`norm_topk_prob`, `moe_router_activation_func`→`scoring_func`, `num_expert_group`→`n_group`), so one execution surface is reached from either family.

**Three defects surfaced on the way, each worse than the noise it made:**

- **A 256-expert model was resolving as dense.** `is_moe()` defaulted to `false` on the trait, so an MoE without a registry entry had no MoE surface at all — Kimi's read `ffn: dense, intermediate_size 9216`, which is one layer's dense width out of twenty-seven. That is not a gap in a report; it is a container that would describe the wrong model. `is_moe`, `num_experts`, `num_experts_per_token`, `num_shared_experts` and `moe_intermediate_size` now answer from the declaration.
- **Fixing that exposed a worse one.** With a surface finally built, it claimed `router_kind: top_k_softmax` for a checkpoint declaring **sigmoid**. Sigmoid scores are independent, so the selected weights do not sum to 1 — a different rule, not a variant of one. `MoeRouterKind::Sigmoid` now carries it, and all three execution sites refuse rather than substituting a softmax policy. Kimi, GLM-5.3-Flash **and** Inkling-Small all declare it.
- **A phantom `linear_attn` component.** `linear_attn_config` ends in `_config` and declares a `num_heads`, so it was read as a sibling sub-model — one with no embedding, no layers and no tensors, whose execution surface was reported *incomplete*. Worse, its keys were credited to that component, so `linear_attn_config.head_dim` graded representable on Kimi and unrepresented on GLM: the same key, two verdicts, decided by where the section sat in the file.

**Two facts became real schema fields** rather than staying unrepresented: `ExecutionSurface.ffn.moe.branch_scale` (`routed_scaling_factor`, 2.446 on Kimi) and `dense_prefix_layers` (`first_k_dense_replace`, 1 on Kimi, 3 on GLM). Both change the forward; neither had a home.

**Expert grouping is representable only at its identity value.** One group is ungrouped routing, so the schema represents its effect exactly by having no field. Any other value states something it cannot, and blocks — asserted with a fixture declaring eight groups.

**The one remaining blocker is honest.** `mla_use_nope` is declared `true` by Kimi Linear while it carries `qk_rope_head_dim: 64`, so what the flag asserts about the rotary is not yet judged. Unjudged blocks. Closing it means deciding what the flag means on a partial-rotary MLA block, which is a question about the operator rather than about the registry.

**A tautological gate, caught before it shipped.** The first version of the renormalisation probe compared the declared flag against the routing policy — a policy *derived from that flag*, so the check could not fail. A gate that cannot fail is worse than none, because it looks like verification. The probe now reports carriage and says so; the test asserts the two settings produce two different policies instead.

### 4.12 P3c-1b — `mla_use_nope` judged from the reference: **Kimi 0 blocking** (2026-08-27)

The config reads as a contradiction: `mla_use_nope: true` beside `qk_rope_head_dim: 64`. Only Kimi Linear's own `modeling_kimi.py` settles it, and it settles it twice over:

1. **The file contains no rotary code whatsoever.** No `apply_rotary`, no cos/sin, nothing. In the MLA forward, `q_rot` and `k_rot` are split out and then `torch.cat`'d straight back — **unrotated**.
2. **`self.use_nope` is read exactly once, as `assert self.use_nope`.** It is a *precondition*, not a switch: the class refuses to run without it.

So `qk_rope_head_dim` is a **structural width, not a rotary subspace**, and the key name is actively misleading. It splits `q_head_dim = 128 + 64 = 192` and gives `kv_a_proj_with_mqa` its extra 64 outputs, broadcast across heads as a shared unrotated K component. The arithmetic closes against the stored tensors exactly:

| | derived | measured |
|---|---|---|
| `q_proj` rows | 32 × (128+64) = 6144 | 6144 |
| `kv_a_proj_with_mqa` rows | 512 + 64 = 576 | 576 |
| `kv_b_proj` rows | 32 × (128+128) = 8192 | 8192 |

**Verdict:** `mla_use_nope: true` → `PositionPolicy::None`. Deliberately keyed on `Some(true)`: `false` is a combination the reference does not implement — its own assert fires — so this build has no ground truth for it and must not answer. Four controls pin it, including the two real shapes (Kimi's non-zero width, GLM's zero width) and the unimplemented `false`.

**And it exposed one more.** With the stack resolving NoPE, Kimi's declared `rope_theta: 10000.0` became a *mismatch* — resolution "failing" to honour a base on a model that performs no rotation. The honest verdict is that the field is **inert**: a leftover the model's own forward never reads. Both the comparator and the carriage probe now say so, **conditional on every in-scope layer being NoPE** — so the failure this comparator was written for (a theta once resolving 50× smaller than declared) stays reachable, asserted by a control that a rotating stack still reaches "declared and resolved agree" and never "provably not applied".

| | | |
|---|---|---|
| **Kimi Linear** | 20 → **0 blocking** | fully representable, deliberately not executable |
| GLM-5.3-Flash | 47 → **35** | |
| Inkling-Small | 49 → **45** | |

**Kimi Linear is admissible.** KDA and sigmoid routing both encode correctly and both refuse to execute — which makes it the first artifact where `represented ≠ executable` is load-bearing rather than theoretical.

### 4.13 Clearing what was clearable on GLM and Inkling (2026-08-27)

| | | |
|---|---|---|
| Kimi Linear | **0** | admissible |
| GLM-5.3-Flash | 35 → **32** | |
| Inkling-Small | 45 → **41** | |

Three defects, plus one real destination:

- **Inkling's `local_layer_ids` was graded against the wrong probe** — the per-layer-array probe, which renders a `layer_types` array and can never equal a declared *set* of indices. It reported a mismatch on a fact carried exactly. Now compared by cardinality against the resolved table, like the two-set spelling.
- **`norm_topk_prob` was refusing on a cross-check that now exists.** Its rule said *"no schema field — not yet cross-checked against routing_policy"*; the routing policy **is** this flag, and `moe_renormalize` is the same fact in Kimi's spelling.
- **`model.llm.embed` / `model.llm.unembed` / `model.audio.encoder` had no placement rule.** Inkling names its stack `model.llm.*`. Added qualified — `"llm.embed"`, not a bare `"embed"`, because Embedding is scanned before Head and a bare pattern would swallow `unembed`, merging the head into the embedding table.
- `linear_attn_config.gate_lower_bound` reaches `KdaOp.gate_lower_bound`, a field that now exists.

**Two things I did not clear, on purpose.**

`swiglu_limit` looked like a free win: GLM declares one, and `ExpertGatePolicy::ClampedGlu` has a `limit`. But that variant is a *specific formula* — `glu = g·sigmoid(alpha·g)`, `out = (u+1)·glu`, `alpha = 1.702` — transcribed from GPT-OSS's reference. GLM-5.3-Flash and Inkling-Small both declare a clamp too, and nothing on hand says they share that activation. Resolving the policy from the bound alone would claim they do **on the strength of one shared field name** — the same inference `layer_types` → Gated DeltaNet made, wrong for the same reason. Reverted, with the reasoning recorded at the trait default.

The `training_only` keys looked like blockers in my own grouping and are not: an existing test already pins that `TrainingOnly` findings do not block. The filter was wrong, not the classifier.

**What remains is architecture, not registry noise.** GLM's 32 are mHC (4), the DSA indexer (9), FP8 block scales (3), the vision tower and its binding tokens (7), MTP, and the 11 `deepseek_sparse_attention` layers. Inkling's 41 are its audio tower (8), its vision tower (5), and a decoder whose every tensor is spelled its own way — `attn_norm` not `input_layernorm`, `wq_du`/`wk_dv`/`wo_ud` for the projections, short convs on attention and MLP, `rel_logits_proj` for relative position, packed 3-D experts (`w13_weight` at `[256, 4096, 4096]`), and a router whose weight is `[258, 4096]` because it scores the 2 shared experts alongside the 256 routed. That is an adapter, and it is Inkling's rung, not this one.

### 4.14 P3c-2 — the Kimi Linear container is cut and verified (2026-08-27)

**Dry run first.** No `--dry-run` flag exists, so the scope check was built from the admission plan's graph plus a full tensor inventory — everything except operand *binding* is provable before writing a byte. It proved: one component and no phantom; 27 layers with **20 KDA at the declared positions** (set equality, not a count); no recurrence carrying a span; every layer NoPE with no rope base; 256 experts / top-8 / 1 shared / width 1024 / sigmoid / branch scale 2.446 / dense prefix 1; the KDA geometry; **placed bytes exactly equal to inventory bytes** (98,245,528,576); all 15 KDA operands present on every KDA layer and the MLA set on every full layer; and that the two colliding suffixes (`q_proj.weight`, `o_proj.weight`) are covered on both sides.

One dry-run assertion failed and it was mine — the position serialises as `{"kind":"none"}` and I compared it against a bare string.

**Encode:** 7m24s, 98.25 GB payload, 20,490 tensors in the decoder stack.

**Reconstruction from the container alone** — no source checkpoint, no architecture registry — is coherent with every payload re-hashed. The semantic round-trip is **identical**: same component set, same role/geometry, all 27 layer policies equal field for field, execution surface equal, and every MoE and KDA fact carried verbatim.

**And the G4 gate earned its keep.** First run: **68/69**, with all four payloads byte-equal and one semantic failure —

> `FAIL resolved ≡ graph  target.attention_policy: first disagreement at layer 0`

The comparator was re-deriving each layer's span from `LayerPolicy::attention`, the parser's **sliding/full boolean** — and a boolean cannot express a recurrence. Before P3c-0 the two sides agreed only because *both* were wrong: the graph also said 27 full-attention layers. With the graph now correct, the stale comparator was demanding it re-adopt the collapse P3c-0 removed.

The comparator now reads `declared_kind` when the checkpoint states one and falls back to the boolean when it does not. That keeps it a real check rather than a tautology: `LayerPolicy::declared_kind` and the graph's `AttentionLayerPolicy` are two separately stored resolutions, written by different code at different times, and either can drift. What changed is *which* fact is compared, not whether one is.

Second run:

> `semantic: 69/69 authority checks pass` · `verified: Declared ≡ Resolved ≡ Graph ≡ Encoded; payloads byte-equal`

**`~/chris-models/Kimi-Linear-48B-A3B-Instruct.vindex3` (92 GB) is the first real KDA conformance container**, and the first artifact where `represented ≠ executable` is load-bearing: it describes a KDA operator and a sigmoid router completely, and five separate execution sites refuse to run either.

### 4.15 P3d-0 — the KDA execution spec, from the reference (2026-08-27)

`fla` is not installed and is Triton/CUDA, so it will not run here regardless: the oracle has to be a transcription. That makes getting the math from source rather than from the config non-negotiable, and doing so immediately falsified one thing the config appears to say.

**The spec, per layer**, with `H = num_heads`, `D = head_dim`, `K = V = D`:

```text
q = silu(causal_depthwise_conv(q_proj(x), q_conv1d))     # and k, v likewise
g = -exp(A_log)[h] * softplus(f_b_proj(f_a_proj(x)) + dt_bias)      # [T, H, D]
beta = sigmoid(b_proj(x).float())                                   # [T, H]

S = initial_state                                        # [H, K, V], zeros at t=0
for t in 0..T:
    S = S * exp(g[t])[..., None]                         # per (head, k-dim) decay
    S = S + outer(beta[t] * k[t],  v[t] - (k[t][...,None] * S).sum(-2))
    o[t] = einsum('hk,hkv->hv', q[t] * K**-0.5, S)

o = o_norm(o, g_b_proj(g_a_proj(x)))                     # gated RMSNorm, sigmoid gate
out = o_proj(flatten(o))
```

Note `q` and `k` are L2-normalised inside the kernel (`use_qk_l2norm_in_kernel=True`), and the `chunk` path is used only above 64 positions — `fused_recurrent` below it, which is the one a parity ladder should target first.

**The falsification: `gate_lower_bound` is never applied.** Kimi Linear declares `gate_lower_bound: -5.0`, and its own code reads that field **nowhere** — neither `modeling_kimi.py` nor `configuration_kimi.py` mentions it. The gate call passes no lower bound:

```python
g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
```

which selects the softplus form, not the `lower_bound · sigmoid(...)` form the same upstream function also offers. An executor built from the config would have clamped the decay gate, computed a different envelope from the model's own reference, and had **every shape still close**. `KdaOp::gate_lower_bound` now documents that it is carried for provenance and is explicitly *not* an input to the recurrence.

This is the third field in this programme — after `qk_rope_head_dim` and `rope_theta` — whose name promises a computation the reference does not perform. The pattern is stable enough to state: **a declared parameter is evidence that the author wrote it down, not that the forward reads it.**

One more thing to carry into P3d: the checkpoint's `modeling_kimi.py` targets an **older `fla` signature** than upstream `main` (`fused_kda_gate(g, A_log, head_dim, g_bias=…)` against today's `(g, A_log, dt_bias, lower_bound, …)`). The transcription must follow the call the checkpoint makes, not the signature upstream currently offers.

### 4.16 P3d-a — the oracle, the fixture, and four controls that fire (2026-08-27)

`scripts/kda_reference.py` transcribes the recurrence; `scripts/kda_fixture.py` freezes an attention-only fixture from real weights; `scripts/kda_controls.py` proves the fixture would catch the defects it exists for.

**Provenance is pinned deliberately.** The transcription follows *the call the checkpoint's own `modeling_kimi.py` makes*, not the signature upstream `fla` currently offers — the two have drifted, and reading today's third positional argument as `dt_bias` would substitute a head width for a bias. Every source file is cached and sha256-pinned beside the transcription, and the checkpoint's `modeling_kimi.py` is hashed too, because it is the call contract.

**The fixture** is one KDA layer's fifteen operands, a seeded input, and nothing else — no router, no MoE, no residual, no MLP. It dumps all fifteen boundaries plus the recurrent and conv states at `N = 1, 2, 8, 32` (correctness) and `N = 64, 65` (the seam where the reference switches from `fused_recurrent_kda` to `chunk_kda`). Runs clean on Kimi layer 0, 32 heads × 128.

**All four controls fire**, and two of the numbers are worth keeping:

| control | output Δ | state Δ |
|---|---:|---:|
| applying the declared `gate_lower_bound` | **1.746** | 2.370 |
| omitting the **q** L2 normalisation | 0.985 | **0.000** |
| omitting the **k** L2 normalisation | 0.980 | 0.979 |
| resetting the state at `t = 16` | 0.355 | — |

The first says the §4.15 finding was not a technicality: applying the declared clamp changes the layer's output by a **relative 1.75** while every shape still closes. An executor built from the config would have been badly, silently wrong.

The second is a free diagnostic the ladder now has: **q does not touch the recurrent state at all** — it is read-only against the recurrence, appearing only in the readout — while k changes both. So a divergence that moves the state cannot be in the q path, and one that moves only the output probably is. That halves the search space on the first real mismatch, and it fell out of writing the control rather than being designed in.

`KdaOp` now states the split its fields carry: every field is an execution input **except** `gate_lower_bound`, which is provenance. The doc gives the 1.75 figure, so a future reader who wires it in "because that is obviously what it is for" has to argue with a measurement.

### 4.17 P3d-b — the KDA executor, boundary-parity green (2026-08-27)

`exec/kda.rs` executes the recurrent path for `T ≤ 64`, attention block only — no chunk path, no router, no MoE, no residual. **All fifteen boundaries and the recurrent state match the pinned oracle at `N = 1, 2, 8`**, to a `2e-5` transcription tolerance.

The fixture is committed and tiny (2 heads × 4, 27 KB): the arithmetic is identical at any width, and a fixture that fits in a repository is one that gets run. It is generated by the transcription pinned to the checkpoint's own call contract (§4.15), so the oracle cannot drift with upstream `fla`.

**Seven controls, all firing, each perturbing the real function rather than a copy:**

| control | caught |
|---|---|
| applying the declared `gate_lower_bound` | output **and** state move — `gate_lower_bound` is provenance, executably |
| omitting the **q** L2 normalisation | output moves, state moves by **exactly 0.0** |
| omitting the **k** L2 normalisation | both move |
| bf16 recurrent state | both move — the f32 promotion cannot be "optimised" away silently |
| writing `v` instead of `v − kᵀS` | **agrees at `N=1`**, caught by `N=8` |
| read-before-write / no-decay / no-beta | all move |

Two of those are worth keeping for their shape rather than their pass:

**The q-normalisation control asserts `state Δ == 0.0` exactly.** That makes the fault-localisation rule *executable* instead of a note someone has to remember: a disagreement that moves the state cannot be in the q path. It was discovered by writing the control, not designed in.

**The delta-rule control asserts agreement first.** Writing `v` instead of the prediction error is the most plausible wrong transcription, and at one position from a zero state the two rules are identical — the test asserts that they agree at `N=1` before asserting they diverge by `N=8`. A ladder that stopped at one position would have certified the wrong recurrence.

**Genericity is asserted without weights.** The same executor accepts GLM-5.3-Flash's 64 × 128 alongside Kimi Linear's 32 × 128 — state and conv-window sizes derived, no family branch, no width constant. Construction only; GLM's weights are not downloaded, and this is the rung where a width assumption would first bite.

`exec/kda.rs` is 395 lines and takes `&[f32]` operands. Reaching the kernel still-compact is the question Gated DeltaNet answered with `WeightRows`; it is a traffic decision, not a numerical one, and it belongs after a correct baseline exists.

### 4.18 P3d-c — full-width parity green: **KDA correctness is closed** (2026-08-27)

The same executor, unchanged, against **real Kimi Linear layer-0 weights at 32 × 128**:

| N | result |
|---|---|
| 8 | 15 boundaries + recurrent state + 3 conv windows match |
| 32 | match |
| **64** | match |
| **65** | match |

`N = 64` and `65` straddle the point where the reference switches from `fused_recurrent_kda` to `chunk_kda`. LARQL implements neither chunking nor a second path — the gate is that it stays **mathematically equivalent across the boundary where the reference changes strategy**, and it does.

**Why full width was a separate gate.** The committed 2 × 4 fixture proves the arithmetic, and the arithmetic is identical at any width. What it cannot prove is indexing, stride, state sizing, convolution layout and flatten order: a transposed head axis or a wrong `h*D + d` is invisible at `D = 4` and fatal at `D = 128`. Passing at 32 × 128 with no change to the executor is what closes that.

The tolerance is `3e-4` here against `2e-5` on the tiny fixture, stated rather than tuned: at hidden 2304 and head dim 128 each value is a sum over hundreds of terms, so two orderings of the same arithmetic separate further. It is still four orders below every control's effect.

Env-gated (`LARQL_KDA_FIXTURE`) because the fixture is ~196 MiB of f32 — too large to commit, regenerable in seconds by `scripts/kda_fixture_export.py`, and skipping cleanly when unset.

**KDA correctness is closed.** Three geometries are now covered by one executor with no family branch: 2 × 4 committed, 32 × 128 verified against real weights, 64 × 128 (GLM-5.3-Flash) asserted at construction.

What remains before a token comes out of the container is not KDA: sigmoid-router execution, one MoE block, one complete Kimi layer, the mixed 20/7 stack, and the token loop — in that order, so that a divergence in the first complete layer is known not to be the attention half.

### 4.19 P3d-c½ — projections onto the trusted matvec: 57.2 s → 4.2 s (2026-08-27)

One function changed. `matvec` now routes to the crate's existing `BlasF32` projector instead of a scalar loop; **the convolution, L2 normalisations, decay gate, delta recurrence and gated norm are byte-for-byte the same code.**

The split is the point. A KDA layer is *ordinary linear algebra* — q/k/v, the two low-rank gate pairs, `b_proj`, `o_proj` — wrapped around *a small amount of KDA-specific arithmetic*. The first group should use infrastructure that is already trusted and already tuned; the second is where the operator actually lives and can stay a plain f32 transcription.

**The regression gate is the frozen full-width fixture, re-run unchanged**: all 15 boundaries, the recurrent state and all three conv windows still match at the same `3e-4` tolerance. Because only the projection function moved, that re-run is a real check that acceleration changed no semantics rather than a formality.

| | before | after |
|---|---|---|
| full-width fixture, N = 8/32/64/65 | 57.25 s | **4.24 s** |

57 s was not a performance problem so much as a debugging one: every later integration rung would have paid it, on every iteration. 4.2 s is the difference between a token loop that can be debugged tonight and one that cannot.

Nothing here was optimisation research — no Metal, no quantisation, no new kernel. The compact-representation question (`WeightRows` beyond `F32`) is still open and still belongs after integration, not before it.

### 4.20 P3d-d — operand closure: a fourth kind of closure, and where Kimi actually stands

Running `vindex3 ops` on the G4-verified container found **20,247 closure defects**. That is not a contradiction of §4.14 — it is a distinction the programme had not yet named:

| closure | question | Kimi |
|---|---|---|
| **admission** | are the model's semantics representable? | green (§4.12) |
| **encoding** | do graph and payload survive the container? | green (§4.14) |
| **operand** | can an executable operator bind its inputs? | **20,227 defects** |
| **execution** | do the operators run? | not reached |

A container can be a perfectly faithful archive of a model that no executor can consume. Admission and G4 never bind operands: they check that every tensor is *placed* and that the semantic graph round-trips byte-equal, which it does. **Operand closure should be a required gate for any container claiming an execution surface**, and it is not one today.

**The cause is naming, again.** Kimi Linear spells its MoE `block_sparse_moe.experts.{E}.w1/w2/w3.weight`, its router `block_sparse_moe.gate.{weight, e_score_correction_bias}`, and its shared branch `block_sparse_moe.shared_experts.*`. The `ROLE_TABLE` knows `mlp.gate_proj.weight`, `mlp.router.weight` and GPT-OSS's packed `mlp.experts.gate_up_proj_blocks`. None of it matches. Its seven MLA layers are equally unmodelled (`kv_a_proj_with_mqa`, `kv_b_proj`, `kv_a_layernorm`).

**One of the three families is closed.** The KDA contract correction: Kimi stores `A_log` as `[1, 1, 32, 1]` — the shape its reference broadcasts against `[B, T, H, D]` — where the contract says `[32]`. Those are the same 32 numbers in the same order, so `shape_satisfies` now accepts a **vector** contract carrying broadcast singletons, and nothing else. Deliberately not a general squeeze: `[2, 16]` still fails `[32]`, and a matrix contract gets no equivalence at all, because a blanket "drop all ones" would accept a genuine relayout as readily as a broadcast form. All 20 `A_log` defects cleared; 20,247 → 20,227.

**What remains, precisely:**

| family | defects | work |
|---|---:|---|
| MoE binding | 20,126 | per-expert roles carrying an expert index, a router pair, shared-expert roles, and an evidence-driven expert-bank prefix — `expert_bank_prefix` today derives only from *packed* keys, so Kimi's per-expert bank was never carved and all 96.7 GB sits in the decoder stack |
| MLA binding | 101 | operand roles for the seven full-attention layers, bound because the layer's operator is MLA rather than because a name looks attention-shaped |
| KDA contract | **0** | done |

**And a correction to what I told you earlier: Kimi *does* ship `e_score_correction_bias`** — 26 of them, one per MoE layer, and `KimiMoEGate` adds it before selection. So the bias-corrected path is not GLM-only. The reference sequence is: `sigmoid(logits)` → **plus bias → top-8 identities** → **gather the unbiased scores as weights** → renormalise → × 2.446. Selection and weighting read different tensors, which the config alone does not say.

## 5. Census and ledger

### 5.1 Where the parameters are

76,108 tensors, 321.34 B parameters, 305.78 GiB stored.

| group | tensors | parameters | % |
|---|---:|---:|---:|
| routed experts | 72,576 | 304,405,807,104 | 94.73 % |
| MTP layer (all tensors) | 1,760 | 7,432,592,416 | 2.31 % |
| **KDA** | 510 | **4,682,897,792** | 1.46 % |
| MLA | 121 | 1,291,868,160 | 0.40 % |
| shared experts | 252 | 1,056,964,608 | 0.33 % |
| lm_head | 1 | 634,388,480 | 0.20 % |
| embeddings | 1 | 634,388,480 | 0.20 % |
| vision tower | 347 | 563,627,008 | 0.18 % |
| dense MLP | 18 | 452,984,832 | 0.14 % |
| DSA indexer | 77 | 82,190,592 | 0.03 % |
| routers | 84 | 49,557,312 | 0.02 % |
| mHC | 270 | 35,391,870 | 0.01 % |
| norms | 91 | 372,736 | 0.00 % |

**KDA carries zero scale tensors: it is entirely BF16 in the official FP8 build.** So is `kv_b_proj`, the routers, mHC and every norm.

### 5.2 Active weight per decoded token — the number that decides tok/s

Dense stack only: no MTP, no vision, top-8 of 288 over 42 sparse layers.

| group | active params | share |
|---|---:|---:|
| routed experts | 8,455,716,864 | **50.5 %** |
| **KDA** | **4,682,897,792** | **28.0 %** |
| MLA | 1,291,868,160 | 7.7 % |
| shared experts | 1,056,964,608 | 6.3 % |
| lm_head | 634,388,480 | 3.8 % |
| dense MLP | 452,984,832 | 2.7 % |
| DSA indexer | 82,190,592 | 0.5 % |
| routers | 49,557,312 | 0.3 % |
| mHC | 35,391,870 | 0.2 % |
| **total active** | **16,742,329,150** | |

**Expert-side 56.8 % / non-expert 43.2 %.** The vendor's "18 B active" is reproduced to within the MTP and embedding accounting.

**This is the relocation.** A 320 B model whose checkpoint is 97 % experts spends only half its per-token weight traffic on them. KDA alone — 4.68 B parameters, left in BF16 by the vendor — is larger than the shared experts, MLA, the head and the dense layers combined.

## 6. Residency and traffic

### 6.1 Can a 320 B model be resident in 128 GB?

Text-only container, vision and MTP excluded (304.41 B routed experts + 8.92 B everything else), GiB:

| experts bpw | rest BF16 | rest 8-bit | rest 6-bit | rest 4-bit |
|---:|---:|---:|---:|---:|
| 4.5 (NVFP4) | 176.1 | 167.8 | 165.7 | 163.6 |
| 4.25 (MXFP4) | 167.2 | 158.9 | 156.8 | 154.8 |
| 3.5 | 140.6 | 132.3 | 130.3 | 128.2 |
| 3.0 | 122.9 | 114.6 | 112.5 | 110.5 |
| 2.5 | 105.2 | 96.9 | 94.8 | 92.7 |
| 2.0 | 87.5 | 79.2 | 77.1 | 75.0 |

**Cross-check against a published build:** the same arithmetic at 4.5 bpw experts with everything else BF16, including vision and MTP, gives **191.0 GiB** against the LibertAI NVFP4 build's ~181 GiB — a 5 % bracket, consistent with that build also excluding or compressing the MTP layer (7.43 B params ≈ 13.8 GiB at BF16). The census reproduces an independently published figure, which is the check that it is sound.

**The consequence is blunt: no representation currently in LARQL fits.** The estate's smallest is MXFP4 at 4.25 bpw — 154.8 GiB even with everything else at 4-bit. Residency on a 128 GB machine, with room left for KV, activations and the OS, needs **routed experts at ≈ 2.5–3.0 bpw**, which is below anything the REPRESENT programme has built or measured. Either that sub-3-bit representation gets built and passes a quality bar, or GLM-5.3-Flash is an out-of-core model on this hardware. Both are legitimate; they are different programmes, and §7 keeps them apart.

### 6.2 Decode weight traffic, and what it implies

GB per token, and tok/s at 300 GB/s of *useful* bandwidth (the repo's measured envelope for good quantised GEMV, not the 400 GB/s headline):

| experts bpw | rest BF16 | rest 8-bit | rest 6-bit | rest 4-bit |
|---:|---|---|---|---|
| 4.25 | 19.51 GB → 15.4 t/s | 12.28 → 24.4 | 10.48 → 28.6 | 8.67 → 34.6 |
| 3.0 | 18.03 GB → 16.6 t/s | 10.80 → 27.8 | 8.99 → 33.4 | 7.18 → 41.8 |
| 2.5 | 17.43 GB → 17.2 t/s | 10.20 → 29.4 | 8.39 → 35.7 | 6.59 → **45.5** |
| 2.0 | 16.84 GB → 17.8 t/s | 9.61 → 31.2 | 7.80 → 38.5 | 5.99 → 50.1 |

Read the table by columns, not rows. **Holding experts at MXFP4 and taking the rest from BF16 to 4-bit: 15.4 → 34.6 tok/s (+125 %). Holding the rest at BF16 and taking experts from 4.25 to 2.0 bpw: 15.4 → 17.8 tok/s (+16 %).** The non-expert side is the dominant decode lever by roughly eight to one, and KDA is 65 % of it.

These are roofline ceilings from weight traffic alone. KDA recurrence, the DSA indexer's top-2048 selection, mHC's Sinkhorn iterations, routing and dispatch all subtract; they are not modelled here and none of them are free.

**What the table licenses:** a bandwidth-derived ordering of levers. **What it does not license:** a tok/s prediction. No kernel has been priced, and per [R14](dec-funnel.md) a roofline is not a measurement.

## 7. Two tracks

Kept separate so that a bad number has one interpretation instead of four.

**GLM53-CORRECTNESS** — architecture → tensors → parity → generation.
**GLM53-PHYSICAL** — expert store → cache → prefetch → eviction → REPRESENT → disk/RAM curves.

They meet only once each is independently trustworthy. The failure this avoids: 1.2 tok/s from an external SSD with no way to tell whether it is the I/O floor, a bad cache, needless materialisation, or a wrong forward pass.

**Standing invariant, adopted now rather than retrofitted:**

> No GLM-5.3-Flash execution component may assume the full expert population is addressable in unified memory.

The 128 GB M3 Max and an 8 GB M1 then run the *same* code path with different cache capacities, and §6.1's "nothing fits" stops being a blocker for the correctness track.

## 8. Rungs

Ordered by what unblocks the most, with the download deferred as far as it will go.

| rung | name | does | exit gate | needs weights? |
|---|---|---|---|---|
| **P0** | Admission | inventory + plan from headers alone | **done** — §4 | no |
| **P1** | Plan honesty | ~~`attention_policy` must not claim `gated-delta recurrent` on unresolved carriage~~ **DONE** (§4.2); `0 NoPE` must not survive `qk_rope_head_dim: 0` — still owed | plan re-run reports the 34 KDA layers as unexpressed, and blocking rises | no |
| **P2** | FP8 block scales | `weight_scale_inv`, 128×128, `activation_scheme dynamic` | decode a shard's tensor bit-exactly against a reference dequant | one shard (~5.4 GB) |
| **P3a** | Topology carriage | read `linear_attn_config.{kda_layers, full_attn_layers}`; prove the index base rather than assume it (§4.6) | **DONE** — Kimi reports `20 recurrent / 7 full`, GLM `34 recurrent / 11 unexpressed`, both blocking | no |
| **P3b** | KDA vocabulary — **DONE** (§4.8) | read `linear_attn_config` (§4.5 — a Kimi encode is wrong until this lands; mind the index base, §4.6); extend or replace `GatedDeltaOp` for split q/k/v, three conv1d, low-rank f/g gates, per-channel `dt_bias`, `gate_lower_bound` | **layer-by-layer f32 diff against Kimi Linear** (§9); Kimi's `attention_policy` reports 20 recurrent / 7 full, not 27 full | **none — local** |
| **P4** | Synthetic GLM-5.3 | same graph and tensor roles, tiny dims/experts: `dense → KDA → MoE → … → DSA → MoE → logits` | plan admissible; generates autoregressively under a bounded expert cache | no |
| **P5** | MLA-NoPE + DSA indexer | `qk_rope_head_dim 0`; top-2048 with kpool-4 compression | MLA against Kimi Linear; DSA has no local ancestor — reference diff on one real layer | one shard |
| **P6** | mHC | `hc_mult 4`, 20 Sinkhorn iterations, per-layer `hc_*` | one real layer against a reference forward | one shard |
| **P7** | Router + MoE | sigmoid, `noaux_tc` bias correction, shared expert, `routed_scaling_factor` | routing trace agrees on a real layer | one shard |
| **P8** | Sparse parity | one real layer of **every** execution type agrees with reference | the gate in §10 | tens of GB |
| **P9** | REPRESENT policy | per-role precision map; §6.1's sub-3-bit expert question | quality bank vs a BF16 target | full checkpoint |
| **P10** | Ingest | | | 328 GB |

P3 and P4 are the two that pay for themselves fastest, and **neither needs a single byte of GLM-5.3-Flash.**

## 9. The Kimi Linear lever

`~/chris-models/Kimi-Linear-48B-A3B-Instruct` (92 GB) is already on disk, with `modeling_kimi.py` and `configuration_kimi.py` beside it. Its KDA block is GLM-5.3-Flash's KDA block:

| | Kimi Linear 48B | GLM-5.3-Flash |
|---|---|---|
| `linear_attn_config` keys | `full_attn_layers, head_dim, kda_layers, num_heads, short_conv_kernel_size` | **identical set** |
| `q/k/v_proj`, `q/k/v_conv1d` | (4096, 2304), (4096,1,4) | (8192, 4096), (8192,1,4) |
| `f_a_proj` / `f_b_proj` | (128, 2304) / (4096, 128) | (128, 4096) / (8192, 128) |
| `g_a_proj` / `g_b_proj` | (128, 2304) / (4096, 128) | (128, 4096) / (8192, 128) |
| `A_log` / `dt_bias` | (1,1,32,1) / (4096,) | (64,) / (8192,) |
| `b_proj` / `o_norm` / `o_proj` | (32,2304) / (128,) / (2304,4096) | (64,4096) / (128,) / (4096,8192) |
| `mla_use_nope` | true | true |

**Same tensor names, same low-rank gate structure at rank 128, same conv kernel 4, same per-channel `dt_bias` at `Hv·Dv`.** Only the widths differ, and Kimi Linear is the *awkward* one (32 heads, hidden 2304) — which makes it the better fixture.

So P3 — the single riskiest operator in GLM-5.3-Flash — is buildable today, against a local checkpoint, with `larql shannon layer-dump` / `layer-diff` giving a per-layer f32 diff instead of an API oracle. This is exactly [`k3-funnel.md`](k3-funnel.md) rung **R2**, and it now unblocks two models.

Differences that do **not** transfer, and need GLM weights: `qk_rope_head_dim` 64 vs 0, MTP, mHC, the DSA indexer, 256 vs 288 experts, and FP8 storage (Kimi Linear is BF16).

## 10. The gate before the download

> **A synthetic full GLM-5.3-Flash topology generates autoregressively under a bounded expert-cache budget, and at least one real GLM layer of every execution type — KDA, MLA-NoPE + DSA, dense MLP, sparse MoE — agrees with a reference implementation.**

Past that, 328 GB arriving is data, not the moment an architectural problem is discovered.

## 11. What is owed

- **P1 is DONE for GLM** (§4.2, §4.4) but **not for Kimi** (§4.5): a stack that declares its recurrence outside `layer_types` still reports a silent all-full tower. That is P3's, and it is why a Kimi container cannot be cut first.
- **Still owed from P1:** `0 NoPE layer(s)` survives `mla_use_nope: true` / `qk_rope_head_dim: 0`, and `target.execution_surface` still reports "complete (attention, ffn, norm, head)" on a stack whose attention policy blocks. Two more claims that outrun their evidence, both untouched.
- The `head_dim: 0` validation error means `generic_fallback: true`, so **every resolved-topology number in the inventory is a generic reading**, not a GLM one. §3's table comes from the raw config and the headers; §5's census comes from the headers. Neither depends on the resolver — but nothing else in the inventory should be quoted until a registry entry exists.
- Sub-3-bit expert representation (§6.1) is unbuilt and unmeasured. Until it exists, "320 B resident on a MacBook" is a target, not a plan.
- No kernel is priced. §6.2 is a roofline.
- The vendor's benchmark claims are vendor claims; nothing here reproduces them.

## 12. Artifacts

Regenerate with §2; nothing is checked in but the tool.

| artifact | how |
|---|---|
| stub checkpoint (10.68 MB) | `scripts/hf_metadata_checkpoint.py zai-org/GLM-5.3-Flash --out stub` |
| `inventory.json` | `larql inspect-hf stub --no-tensor-list --output inventory.json` |
| `plan.json` | `larql vindex3 plan stub --output plan.json` |
