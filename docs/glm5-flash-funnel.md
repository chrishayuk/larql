# GLM-5.3-Flash funnel v0.2 — admission, ledger, and the physical programme

**Model:** `zai-org/GLM-5.3-Flash` — 321.3 B parameters, 45 layers, FP8 E4M3, 328.3 GB
**Status:** v0.2 — P0 (admission) **run and recorded** (§4); the KDA correctness rungs closed against Kimi Linear (§4.15–§4.20); **GLM53-PHYSICAL defined and entirely unrun** (§6.3, §7.1, §8.2)
**Date:** 2026-08-27; physical programme added 2026-09-07
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

**Added 2026-09-07 — the second claim, and the one the physical programme is built on:**

> **20 tok/s is not blocked by weight bandwidth. Residency forces low-bit experts; throughput is dominated by the non-expert trunk — especially KDA — and by non-bandwidth terms such as mHC fusion and DSA selection.**

Relocation 1 is the reason. Because 97 % of the bytes on disk are routed experts, the
intuitive programme is “compress the experts and the tok/s follows.” It does not, or
barely. Capacity and decode traffic are *separate* constraints, they bind on *different*
tensors, and they are satisfied by *different* work (§6.3):

> **Expert compression is primarily the residency lever. Trunk compression — especially KDA — is the throughput lever.**

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

### 6.3 The 20 tok/s budget — capacity and decode traffic are different constraints

**20 tok/s is 50 ms per decoded token.** Everything below is an allocation of that 50 ms.

The two constraints that the checkpoint's shape invites you to merge, kept apart:

| constraint | what binds it | what it forces |
|---|---|---|
| **Capacity** | 128 GiB unified memory, less KV, activations and the OS | routed experts at **≈2.0–2.5 bpw** — §6.1's 75.0–92.7 GiB with the rest at 4-bit |
| **Decode traffic** | ≈300 GB/s of *useful* GEMV bandwidth | the **non-expert trunk** off BF16 — §6.2's lever ratio is ~8:1 against expert bpw, and KDA is 65 % of the trunk |

> **Expert compression is primarily the residency lever. Trunk compression — especially KDA — is the throughput lever.**

§6.2's own columns are the evidence: holding experts at MXFP4 and taking the rest
BF16 → 4-bit is **+125 %**; holding the rest at BF16 and taking experts 4.25 → 2.0 bpw
is **+16 %**.

#### 6.3.1 The corrected traffic point

At **experts 2.0 bpw + rest 4-bit** — the residency-forced operating point — the active
split from §5.2 is:

| | active params | bit width | bytes |
|---|---:|---:|---:|
| expert-side (routed 8.456 B + shared 1.057 B) | 9.513 B | 2 | 2.378 GB |
| non-expert trunk | 7.230 B | 4 | 3.615 GB |
| **total per decoded token** | **16.742 B** | | **5.99 GB** |

**5.99 GB/token, not 4.2 GB.** Pricing all 16.742 B at 2 bpw gives 4.19 GB and
understates traffic by **43 %**. The trunk is 43.2 % of the active *parameters* and is
carried at twice the expert bit width, so it is **60 % of the bytes** — the same
relocation as §5.2, now in the units that decide the budget.

At 300 GB/s useful this is **19.98 ms of weight traffic**, a **50.1 tok/s roofline**, and
**40 % of the 50 ms budget**. Therefore:

> **20 tok/s is not blocked by weight bandwidth.** At the operating point capacity already
> forces, the weight-traffic ceiling is **~2.5× the target**, leaving **≈30 ms/token** for
> everything that is not weight movement.

**What this licenses:** a budget, and the ordering claim that bandwidth is not the binding
term at 2 bpw / 4-bit. **What it does not license:** a tok/s prediction. No kernel is
priced here either.

#### 6.3.2 The 30 ms is where the risk moved — and PHYSICAL-2a relocated it again

A roofline models **bytes**. It does not model **kernel count, dispatch count or serial
dependency depth**, and three terms in this architecture are dominated by exactly those.
v0.2 said so and then mis-priced the largest of them; **PHYSICAL-2a (§8.2.1) measured it
2026-09-07** and the correction is recorded here rather than in a footnote.

**Measured, M3 Max, idle machine (load 1.8–3.1), two runs:**

| unit of cost | measured | instrument |
|---|---:|---|
| dependent dispatch **inside one encoder** | **2–3 µs** | `lowered_dispatch_floor`, n=1024, widths 2880 / 4096 |
| command buffer **commit + wait** | **12–22 µs** | `cb_roundtrip_floor`, empty and empty-encoder arms |
| one submission in the **real** NVFP4 decode | **~427 µs** | of which the Metal floor is **4 %**; the rest is LARQL's own per-call work |

1. **mHC — the alarm was wrong by ~6×, and it is withdrawn.** `hc_sinkhorn_iters: 20` at two
   sites per layer over 45 layers is **1,800 serially dependent normalisation steps per
   token** on 35.4 M parameters; that part stands. But v0.2 priced each step at a *command
   buffer round trip* (10–30 µs ⇒ 18–54 ms), and **that is the wrong unit**. The lowered
   token is already one command buffer holding the whole stack, so the unit is a dependent
   dispatch inside an encoder:

   | encoding | cost/token | share of 50 ms |
   |---|---:|---:|
   | 1,800 separate **submissions** | 29–39 ms | 58–78 % — would kill the target |
   | 1,800 dependent **dispatches, one command buffer** | **3.6–5.4 ms** | **7–11 %** |
   | fused to 90 sites (20 iterations intra-kernel) | 0.18–0.27 ms | < 1 % |

   **⇒ Fusion is desirable, not critical — provided mHC stays inside the token's command
   buffer.** Measured 2–3 µs sits *below* the most optimistic bucket the decision rule
   anticipated.

2. **The real hazard is the submission boundary, not the dispatch count.** The Metal floor is
   4 % of a real submission; the other ~410 µs is per-call work — pool locks, buffer-cache
   lookups, readback allocation. An mHC site that triggers a submission or a readback costs
   **~130× one dispatch**, and 90 of them would be ~36 ms — the budget, arriving through a
   door v0.2 was not watching. **PHYSICAL-2's question is therefore no longer "how many
   dispatches" but "does any site cross a submission or readback boundary".**

3. **DSA selection scales with context, not weights.** `index_topk: 2048` over the live key
   set on 11 layers, `index_kpool: 4`, against a declared context of 1,048,576. A per-token
   byte figure cannot bracket it at all; only a curve in context can. **Unmeasured.**

4. **The dispatch floor is small.** ~450 dispatches/token at 2–3 µs ≈ **0.9–1.4 ms**, not the
   4.5–13.5 ms v0.2 estimated — the same unit error as item 1.

**⇒ GLM53-PHYSICAL tracks four ledgers separately: bytes, kernel count, dispatch count and
serial depth.** That rule survives PHYSICAL-2a; what changed is **which ledger is
dangerous**. The census was right that a roofline cannot see this class of cost, and wrong
about its size by 6× — which is exactly why the cheap rung ran before the fusion was built.

#### 6.3.3 Why KV is not a budget line — arithmetic, not measurement

MLA carries `kv_lora_rank: 512` on the 11 DSA layers, and **KDA is recurrent** — a constant
state, not a per-position cache. Latent KV is therefore ≈11 KiB/token (512 × 2 B × 11),
≈1.4 GB at 128 K context. The KDA state is 34 × 64 × 128 × 128 elements ≈ 143 MB at f32,
**constant in context**. Against the 53.0 GiB (at 2.0 bpw) or 35.3 GiB (at 2.5 bpw) that
§6.1 leaves free, neither is a budget line until something measures otherwise.

This is a genuine structural advantage of the architecture and it is what makes the
residency band affordable: a 45-layer softmax tower at this width would not leave the same
room.

**One operational constraint, recorded before it bites:** macOS caps GPU-wired memory below
installed RAM by default (~75 %, ≈96 GiB on a 128 GiB machine). The 2.5 bpw end of the band
(92.7 GiB) sits inside that cap with ~3 GiB to spare. Residency covers the **bound buffer
object**, not the file, so check `iogpu.wired_limit_pct` before reading an eviction as a
representation failure.

## 7. Two tracks

Kept separate so that a bad number has one interpretation instead of four.

**GLM53-CORRECTNESS** — architecture → tensors → parity → generation. Rungs in §8.1.
**GLM53-PHYSICAL** — budget → trunk → fusion → selection → resident experts → complete
ledger → speculation. Rungs in §8.2.

They meet only once each is independently trustworthy. The failure this avoids: 1.2 tok/s
from an external SSD with no way to tell whether it is the I/O floor, a bad cache, needless
materialisation, or a wrong forward pass.

### 7.1 What GLM53-PHYSICAL is for

> **20 tok/s is not blocked by weight bandwidth. Residency forces low-bit experts;
> throughput is dominated by the non-expert trunk — especially KDA — and by non-bandwidth
> terms such as mHC fusion and DSA selection.**

That is the headline, and it **replaces** the ordering the checkpoint's shape invites
rather than qualifying it. §6.3 splits the constraint in two; the rung order below follows
from the split.

Three consequences:

1. **KDA is the first physical experiment, not the last.** 4.68 B active parameters,
   **28.0 % of active weight**, **~65 % of the dominant lever**, and **100 % BF16 in a
   checkpoint the vendor shipped as FP8** (§5.2). It is also the only operator with a
   bit-exact local ancestor (§9), so it can be qualified before any GLM-specific kernel
   exists. Nothing else in the model has that combination of leverage and cheapness.

2. **Sub-2-bit experts are downgraded** — from *required for 20 tok/s* to **possible
   quality/capacity research beyond what the initial 128 GiB fit target needs**. ≈2.0–2.5
   bpw is the current *residency* requirement (§6.1), and §6.3.1 shows that band already
   clears the traffic constraint by ~2.5×. Whether to go below it is a decision for the
   complete physical ledger at **PHYSICAL-5**, not an assumption made now.

3. **MTP is a multiplier, not a rescue.** GLM declares `num_nextn_predict_layers: 1` —
   depth-1 speculation. Priced against *measured* adjacent-token routing divergence rather
   than assumed expert reuse it is worth **~1.2–1.3×** (§8.2.3). It does not turn a 10 tok/s
   decoder into 20+, and base and MTP-effective tok/s are reported separately so that it
   can never be mistaken for one.

**A caution this programme inherits.** Every number in §6.3 is arithmetic over a measured
census. The census is real; the budget derived from it is not a measurement, and the risk
it identifies — that 1,800 serial mHC steps could consume more wall clock than 5.99 GB of
weight traffic — is exactly the kind of claim that has to be *measured to be believed*.
PHYSICAL-0 freezes the budget precisely so that later measurements can falsify it.

**Standing invariant, adopted now rather than retrofitted:**

> No GLM-5.3-Flash execution component may assume the full expert population is addressable
> in unified memory.

The 128 GB M3 Max and an 8 GB M1 then run the *same* code path with different cache
capacities, and §6.1's "nothing fits" stops being a blocker for the correctness track.

## 8. Rungs

### 8.1 GLM53-CORRECTNESS

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

### 8.2 GLM53-PHYSICAL

Defined 2026-09-07. **Nothing below has been run.** Ordered by leverage per §7.1, which
puts the trunk before the experts and fusion before representation.

| rung | name | does | exit gate | needs weights? |
|---|---|---|---|---|
| **PHYSICAL-0** ✅ | Freeze the budget | record 50 ms/token; §5.2's active decomposition; **5.99 GB/token** at 2 bpw experts + 4-bit trunk; the four ledgers (bytes, kernel count, dispatch count, serial depth) | **DONE 2026-09-07** (§8.2.0) — frozen at v0.2 *before* any probe ran | no |
| **PHYSICAL-1** 📋 | **KDA Q4 on Metal** — scoped §8.2.2 | price one **real** KDA layer at 4-bit; qualify **recurrent-state** fidelity, not merely layer output | measured ms/layer against the PHYSICAL-0 line **and** state-trajectory agreement across positions with a control that fires. **A layer-output-only gate does not close this rung** — see below | one shard |
| **PHYSICAL-2a** ✅ | Dispatch calibration | measure `90 sites × 20 sequential tiny dispatches` against `90 sites × 1 dispatch of 20 iterations`, before building any fusion | **DONE 2026-09-07** (§8.2.1) — dependent dispatch **2–3 µs**, commit+wait **12–22 µs**. §6.3.2's 18–54 mHC alarm **falsified**; real figure 3.6–5.4 ms unfused-in-one-buffer | **no** |
| **PHYSICAL-2** | Fused mHC | the 20 Sinkhorn iterations as an **intra-kernel loop**, not 20 dispatched operations; measure complete attn **and** ffn sites | **Re-aimed by PHYSICAL-2a:** the gate is no longer dispatch count but **no mHC site crosses a submission or readback boundary** — that is the ~410 µs/site term. Fusion to O(sites) is a ~4 ms optimisation, worth doing, no longer load-bearing | one shard |
| **PHYSICAL-3** | DSA cost | once DSA correctness exists, price the >2048-key path **separately from MLA**; sweep context | a cost **curve in context**, not a single number. The fixture must exceed `index_topk` (2048) or the indexer is measurably inert and the rung measures nothing | one shard |
| **PHYSICAL-4** | Resident expert realization | direct Metal low-bit expert kernels at ≈2.0–2.5 bpw; **no f32 staging, no CPU expert loop, no copied slabs** | routed output **byte-identical** to the qualified FP8 path at the same selection, at a measured ms/layer; predicted vs observed resident bytes reconcile | full checkpoint |
| **PHYSICAL-5** | Complete base decode ledger | KDA + MLA/DSA + experts + mHC + router/norm/head in **one** measured 50 ms ledger. **No MTP** | the four ledgers sum to observed wall clock within a stated bracket. An unexplained residual **is** the finding, not an error bar | full checkpoint |
| **PHYSICAL-6** | MTP as multiplier | report **base tok/s and MTP-effective tok/s separately** | verification traffic priced from the **measured** routed-expert union at depth 1, not from assumed reuse (§8.2.3) | full checkpoint |

**PHYSICAL-1 is the decisive experiment and it is cheap.** KDA has a bit-exact ancestor on
local disk (§9), so the 4-bit fidelity question can be asked against `Kimi-Linear-48B-A3B`
before GLM-specific work exists.

**Why its gate is written the way it is.** KDA is a **recurrent** operator. A 4-bit error in
an expert matvec perturbs one token's output; a 4-bit error in a recurrent operator enters
the state and **compounds along the sequence**. A gate that compares only layer output at a
short prefix can pass a representation that diverges at position 200 — and §4.16's own
finding that *position 0 cannot witness a query-path defect* is the same class of trap one
rung earlier. The gate therefore qualifies the **state trajectory**, and the control must
demonstrate it can fail.

**What PHYSICAL-4 is and is not.** It is where REPRESENT enters the 20 tok/s programme, and
it enters **because 128 GiB forces it** (§6.1), not because the traffic constraint demands
it (§6.3.1). Going below ≈2.0 bpw is *not* owed by this programme.

#### 8.2.0 PHYSICAL-0 — the frozen budget (2026-09-07)

**Frozen. Not an experiment.** Everything below was committed to this document at v0.2
**before** any probe was run, and is not revised by later measurement; a rung that
disagrees with it reports the disagreement rather than editing the freeze.

| quantity | frozen value | source |
|---|---:|---|
| target | **20 tok/s = 50 ms/token** | the goal |
| active parameters/token | **16,742,329,150** | §5.2, measured census |
| — expert-side (routed + shared) | 9,512,681,472 (56.8 %) | §5.2 |
| — non-expert trunk | 7,229,647,678 (43.2 %) | §5.2 |
| — of which KDA | 4,682,897,792 (64.8 % of trunk) | §5.2 |
| operating point | experts **2.0 bpw**, trunk **4-bit** | forced by §6.1 capacity |
| **weight traffic/token** | **5.99 GB** | §6.3.1 |
| useful GEMV envelope | **300 GB/s** | repo-measured, not the 400 GB/s headline |
| **weight-traffic time** | **19.98 ms — 40 % of budget** | 5.99 / 300 |
| **roofline** | **50.1 tok/s — 2.50× the target** | |
| **non-weight slack** | **30.0 ms/token** | 50 − 19.98 |
| residency at that point | 75.0 GiB (2.0 bpw) / 92.7 GiB (2.5 bpw) | §6.1 |

**What the freeze asserts:** at the operating point capacity already forces, weight
bandwidth is **not** the binding constraint, and the programme's whole question is whether
the non-weight terms stay inside 30.0 ms.

**What the freeze does not assert:** any tok/s prediction. It is a bandwidth ceiling over a
measured census. No kernel is priced by it, and §6.3.2's item 1 is a live example of the
freeze's *commentary* being wrong by 6× while its *arithmetic* stood.

#### 8.2.1 PHYSICAL-2a — dispatch calibration, RUN 2026-09-07

Run before building any fusion, precisely because §6.3.2's mHC figure was the most
load-bearing assumption in the freeze and the cheapest to falsify. **No weights, no GLM
code, no new instrument** — two probes the repo already had.

`cargo run --release -p larql-compute-metal --example cb_roundtrip_floor`
`cargo run --release -p larql-compute-metal --example lowered_dispatch_floor`

M3 Max, idle (load 1.8–3.1, no peer build or measurement), two runs each:

| arm | run 1 | run 2 |
|---|---:|---:|
| commit + wait, no encoder | 12.4 µs | 17.9 µs |
| commit + wait, empty encoder | 16.3 µs | 21.6 µs |
| dependent dispatch in one encoder, len 2880, n=1024 | 2.88 µs | 2.76 µs |
| dependent dispatch in one encoder, len 4096, n=1024 | 2.03 µs | 2.09 µs |
| one encoder per dispatch, len 4096, n=1024 | 1.33 µs | 1.73 µs |

**Verdict against the decision rule.** The rule anticipated ~5 µs as its most optimistic
bucket ("fusion desirable but not critical"). Measured dependent-dispatch cost is **2–3 µs**
— *below* that bucket. **1,800 mHC steps encoded into the token's existing single command
buffer cost 3.6–5.4 ms, or 7–11 % of the budget.** §6.3.2's 18–54 ms is withdrawn.

**The result that matters more than the verdict.** `cb_roundtrip_floor` also reports that in
the *real* NVFP4 decode a submission costs **~427 µs, of which Metal's floor is 4 %** — the
other ~410 µs is LARQL's own per-call work. So the dangerous quantity was never the dispatch
count; it is **whether a site crosses a submission or readback boundary**. At 90 sites that
boundary is worth ~36 ms and the dispatch count is worth ~4 ms — a **~130× per-occurrence**
difference, and the two would have been indistinguishable in any measurement that only
counted dispatches. PHYSICAL-2's gate is re-aimed accordingly.

**Two cautions on this result.**
1. The chain is *uniform tiny dispatches*. A Sinkhorn iteration is a **single-threadgroup
   reduction**, and `dependency_bubble_probe` exists because such a reduction between wide
   kernels costs a **pipeline bubble** beyond its own execution time. This probe cannot see
   that bubble; PHYSICAL-2 must, on the real geometry.
2. `hc_mult: 4` makes the Sinkhorn operand far smaller than the 2880/4096 widths measured,
   so 2–3 µs is a **fixed-overhead** reading and should hold — but it is an extrapolation
   downward in width, not a measurement at mHC's own shape.

#### 8.2.2 PHYSICAL-1 — KDA Q4 on Metal: SCOPE, pre-registered 2026-09-07

**Nothing below has been run.** Everything in this section is declared *before*
implementation, so the bands can be missed.

> **Question.** Can GLM's KDA weight traffic be reduced to ~4 bits while preserving the
> recurrent state trajectory, and does the resulting Metal path realise enough of the
> 300 GB/s envelope to fit KDA's share of the 50 ms budget?

**Two independent gates. Neither substitutes for the other.**
**QUALITY** — does Q4 preserve the recurrence? **PHYSICAL** — does the kernel deliver the
traffic reduction?

##### What already exists — this is a binding-and-qualification rung, not a kernel rung

Recon 2026-09-07, before scoping:

- **A complete KDA layer already runs on Metal in ONE command buffer**
  (`trait_impl/kda/`, rung 5c): grouped q\|k\|v, conv+SiLU ×3, q/k L2 norm, low-rank
  decay and output gates, `b_proj`→beta, the delta-rule recurrence against
  **device-resident state**, gated RMS norm, `o_proj` — one host crossing per layer.
- **`KdaDeviceState::read_back` exists solely so a gate can check state against the CPU
  path.** The trajectory witness has its hook already.
- **The freeze this rung needs is already the code's own seam.** `KdaDeviceWeights` carries
  `projection_encoding: ExpertEncoding`, documented as governing `qkv_bank` and `o_proj`
  — "the two wide projections and NOTHING else. The convolutions, the low-rank decay/output
  gates, `b_proj`, `A_log`, `dt_bias` and `o_norm`" stay at source precision.
- **`ExpertEncoding::Q4K` is already a variant**, beside `Bf16`, `Q80`, `Q6K`.
- **There is a precedent test to extend**, not invent:
  `q8_projections_track_the_bf16_step_across_tokens` — already a multi-step Q8_0-against-BF16
  trajectory comparison on this exact path.

**⇒ PHYSICAL-1 binds an existing encoding to an existing kernel and qualifies it. It does
not write a KDA kernel.**

##### The frozen representation

| item | frozen value |
|---|---|
| quantised | `qkv_bank` (q, k, v) and `o_proj` — the two wide projections |
| at source precision | 3×`conv1d`, `f_a`/`f_b`, `g_a`/`g_b`, `b_proj`, `A_log`, `dt_bias`, `o_norm` |
| format | **`ExpertEncoding::Q4K`** — ggml Q4_K, 144 B per 256-element block |
| block geometry | 256 along the reduction axis; **`o_proj` transposes it**, so its stride is its own |
| padding | **none needed — verified** |
| execution | direct Metal Q4_K matvec; **no decode-to-f32 intermediate** |
| optimisation | **none** — no fused q/k/v, no conv fusion, no state-update fusion |

**Block legality was checked, not assumed.** Q4_K requires `k % 256 == 0`:
GLM qkv `k`=4096 ✓, GLM `o_proj` `k`=8192 ✓, Kimi qkv `k`=2304 ✓, Kimi `o_proj` `k`=4096 ✓.
**All four legal, both geometries, so no padding rule enters the freeze.**

##### Two corrections to the budget this rung inherits

**1. Q4_K is 4.5 bpw, not 4.0.** 144 bytes per 256 weights. Pricing KDA at a notional 4.0
understates it by 12.5 %.

**2. "Source precision" for the small tensors is a real traffic decision.** The wide
projections are **97.45 %** of KDA's parameters (per-layer 134,217,728 wide + 3,514,560
small; ×34 = 4,682,897,792, **delta 0** against §5.2's independently derived KDA figure).
But the 2.55 % that stays unquantised is not free — at f32 it is 0.478 GB/token:

| configuration | GB/token | ms at 300 GB/s |
|---|---:|---:|
| Q4_K wide + small **f32** (the current signature) | 3.045 | **10.15** |
| Q4_K wide + small **BF16** | 2.806 | **9.35** |
| notional 4.0 bpw + BF16 (the figure this rung was scoped against) | 2.521 | 8.40 |

**Frozen choice: small tensors at BF16** — the checkpoint's own precision for KDA, and it
buys 0.80 ms for no fidelity change against the vendor's bytes. The f32 row is recorded
because the current `KdaDeviceWeights` signature takes `&[f32]` for the gates, so an
unmodified binding lands on the *worse* row and the 0.80 ms would go missing silently.

##### Pre-registered physical bands

**Rebased.** The bands were first drawn against 7.8 ms (all-KDA at 4.0 bpw); at Q4_K's real
4.5 bpw the **roofline is 9.35 ms**, so a "≤10 ms" band would demand ~94 % of the envelope
and is effectively unreachable. Bands are therefore declared as **realised fraction of the
300 GB/s envelope**, with the ms equivalent at 2.806 GB/token:

| all 34 KDA layers | realised | reading |
|---|---:|---|
| ≤ 12.5 ms | ≥ 75 % | **excellent** — 20 tok/s strongly alive |
| 12.5–15.6 ms | 60–75 % | **healthy** |
| 15.6–18.7 ms | 50–60 % | **viable**, eats meaningful slack |
| 18.7–23.4 ms | 40–50 % | **concerning** |
| > 23.4 ms | < 40 % | 20 tok/s needs another representation or kernel strategy |

The headline number is **milliseconds per real KDA layer at Q4_K**, ×34.

##### Subjects — two, for different reasons

**GLM is the primary subject** (real layer, 64 heads × 128, width 8192): it is the
throughput target, and its decay-gate semantics are known to differ from Kimi's
(`reference_kda_gate_form_differs_by_family` — both declare −5.0, only GLM applies it).
**Kimi is the independent transfer control** (32 × 128, width 4096, the *awkward* geometry,
already qualified bit-exact): it catches an implementation that works only at GLM's shape.
**No family branch inside the Q4 kernel** — if one is needed, that is the finding.

##### The QUALITY gate is sequential, and the state is the boundary

A one-step layer-output comparison does not close this rung. Run a fixed token sequence
through the same KDA layer carrying state forward, `x₀,s₀ → y₀,s₁ → … → yₙ,sₙ₊₁`, comparing
Q4_K against the BF16/native authority **at every step**, recording q/k/v post-convolution,
the decay gate, **the recurrent state**, the reduction, and the layer output.

Curves, not scalars: state relative L2 and cosine **vs step**; output relative L2 vs step;
max drift; and the **slope** with sequence length. Short, medium and long trajectories —
an error that is harmless at 8 steps and grows monotonically to 512 is the exact failure
this rung exists to expose, and only the long arm can see it.

##### Controls — the gate is unqualified until all three fire

1. **Known-bad Q4** — degrade one wide projection or use a deliberately poor scale
   geometry. Must move the trajectory.
2. **State-reset control** — run Q4 but reset state between steps. Proves the witness is
   actually reading recurrence rather than per-step arithmetic.
3. **Wrong decay form** — GLM with Kimi's gate form or the reverse. Already known to move
   real GLM materially, so it is the strongest available positive control.

##### The PHYSICAL measurement, once QUALITY is green

Same input and state, Q4_K against the current representation. Record **stored bytes,
bytes actually read, kernel GPU time, realised GB/s, dispatch count, command-buffer count,
temporary allocations and readbacks, and output/state parity.** Command-buffer count and
readbacks are on this list because §8.2.1 measured a submission at ~130× a dispatch.

##### Four outcomes, declared in advance

| QUALITY | PHYSICAL | verdict |
|---|---|---|
| green | green | **PHYSICAL-1 closes** |
| green | red | representation works; kernel engineering remains |
| red | green | **fast wrong answer — the Q4 hypothesis is falsified** |
| red | red | abandon this realization |

A red QUALITY is not softened by a good Metal number. If it comes back red, that is the most
valuable negative result available right now: **the trunk cannot be treated as ordinary Q4
feed-forward weight**, and every trunk-compression estimate in §6.3 needs re-deriving.

**After adjudication, the next rung is PHYSICAL-3 (DSA), not PHYSICAL-2.** §8.2.1 demoted
mHC from budget-killer to bounded optimisation.

#### 8.2.2a PHYSICAL-1 QUALITY, synthetic arm — RUN 2026-09-07, GREEN (mechanism and controls only)

`crates/larql-compute-metal/src/trait_impl/kda/q4_trajectory.rs`, branch
`glm-physical-1` off `b833d3d4`. 6 tests, all green; the full KDA module is 16/16 with no
regression; fmt clean; clippy adds nothing (the 5 warnings are pre-existing `PLE_*` and
`block v0.1.6`).

**Substrate.** `hidden 256, heads 2, head_dim 128, width 256` — `head_dim` is 128 because
that is what GLM (64×128) and Kimi (32×128) run, and 256 is the smallest Q4_K-legal
`hidden`/`width` pair. **The Q8_0 gate's 64/32 shape is illegal for Q4_K**, so this could
not have been a parameter change to the existing test. Both arms are built from ONE draw:
bf16 codes, and Q4_K of the exact values those codes denote.

**Result — the state holds, and it does not accumulate.**

| arm | steps | worst state rel-L2 | worst state cos | drift ratio |
|---|---:|---:|---:|---:|
| Q4_K vs BF16 | 8 | 0.00905 | 0.999961 | 1.000 |
| Q4_K vs BF16 | 128 | 0.01211 | 0.999929 | 1.125 |
| **Q4_K vs BF16** | **512** | **0.01213** | **0.999929** | **1.545** |

Declared bounds were `state rel ≤ 0.10`, `cos ≥ 0.995`, `drift < 3.0`. Measured **0.0121 /
0.99993 / 1.545** — inside all three, with the state rel-L2 essentially **flat from step 8
to step 512** (0.0090 → 0.0121). **Q4_K on KDA's two wide projections does not compound
through the recurrence at this depth.** The decay gate bounds accumulation, which is the
mechanism the drift ratio was written to test.

**Controls, all firing:**

| control | worst state rel | vs honest 0.0121 |
|---|---:|---|
| known-bad Q4 on `v_proj` (coarsened to a 0.25 grid) | **0.04051** | **3.3×** — fires |
| shifted `dt_bias` (the decay gate's own dial) | **0.10677** | **8.8×** — fires |
| state reset between steps | 0.00906 | below carried, as required |

##### Three findings the gate produced about itself

**1. `q` never enters the recurrent state — a control on the query path cannot qualify a
state witness.** The first CONTROL 1 coarsened `q_proj` and moved the state by **exactly
0.000000** while moving the output by 2.6. Scored directly against honest Q4 (so `q` is the
only difference) the state delta is **bit-identical, cosine 1.000000**, and the output moves
0.052. This is KDA's counterpart to §4.16's "position 0 cannot witness a query-path defect",
and it is now pinned as a property rather than left as a war story. **It is also why the
output metric is kept beside the state metric instead of dropped as redundant:** a gate
scoring only the state would grade a badly quantised `q_proj` as perfect.

**2. A per-step relative output error is an artifact generator on this operator.** The
honest Q4 arm read `out_rel` **18.68** at 512 steps while its state sat at 0.012 — purely a
vanishing denominator: `‖out_ref‖` swings from **0.052 to 50.98** across one trajectory, a
~1000× range. Normalised by the trajectory's mean output norm the same arm reads a stable
**0.15–0.18**. The state metric never shows this because the recurrent state accumulates and
does not pass through zero. Same class as normalising a dot-product error by row scale, and
independent evidence that **the state is the right primary boundary**.

**3. Q4_K's output error on this substrate is ~15 %, an order above its state error
(1.2 %).** Synthetic weights are the worst case for a block-scaled format, so this is not a
GLM number — but it means the *output* channel cannot discriminate a coarsened `q` (0.052)
against the shared Q4 floor (0.18) here. On real weights this needs re-checking.

##### What this result does NOT license

- **It is synthetic.** No GLM or Kimi weight has been through it. Real weights are the
  subject of the rung; this qualifies the *instrument* and the *representation* at a real
  head width.
- **The PHYSICAL arm has not run.** No ms/layer, no realised GB/s. QUALITY green says only
  that a fast Q4_K result would be worth having.
- **The BF16-small-tensor residency (§8.2.2) is not implemented.** `KdaDeviceWeights` still
  takes `&[f32]` for the gates, so the current path is the 3.045 GB/token row, not 2.806.
  `encode_bf16_gemv` exists; threading it through is the next commit.

##### BLOCKER FOUND: the GLM arm cannot run correctly on this path at all

`shaders::kda`'s `kda_decay_gate` computes `-exp(a_log) * softplus(f_low + dt_bias)` —
**Kimi's form** — with no `gate_lower_bound` and no family selection, on `origin/main`
(`b833d3d4`). `KdaGateForm::{Softplus, ClampedSigmoid}` landed in `larql-models` with PR
#444 and **has never reached the Metal executor**, so GLM's
`lower_bound · sigmoid(exp(a_log) · (f_low + dt_bias))` cannot be selected.

This is why CONTROL 3 is a `dt_bias` shift rather than the declared GLM-vs-Kimi decay-form
control: **there is only one form on this path to compare.** And the consequence is larger
than a missing control — the error is **2.8× in per-step decay and compounds with context**,
which is precisely the quantity this rung's trajectory gate measures. Running GLM's weights
through the current Metal KDA path would produce a plausible, wrong, monotonically drifting
trajectory and the gate would correctly blame *quantisation*.

**⇒ `KdaGateForm` on Metal is a PREREQUISITE for PHYSICAL-1's GLM arm, not part of Q4.** The
Kimi arm is unaffected — Kimi's form is the one implemented.

#### 8.2.2b PHYSICAL-1 QUALITY, REAL KIMI WEIGHTS — RUN 2026-09-07

`examples/kda_q4_trajectory_real.rs`. Kimi-Linear-48B layer 1 (a declared KDA layer),
geometry read from TENSOR SHAPES: hidden 2304, 32 heads x 128, width 4096, conv kernel 4.
Both projections Q4_K-legal. 512 steps, unit-RMS input, idle machine.

**Verdict: the stability question is answered YES. The magnitude bounds are MISSED at Q4_K.**

| encoding | bpw | MiB | state rel-L2 | state cos | drift | out raw | out traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 8.500 | 38.25 | 0.00840 | 0.999958 | 1.018 | 0.0107 | 0.0085 |
| Q6_K | 6.562 | 29.53 | 0.03071 | 0.999522 | 0.993 | 0.0391 | 0.0301 |
| **Q4_K** | **4.500** | **20.25** | **0.12792** | **0.992011** | **1.092** | 0.1355 | 0.1308 |

Declared bounds were `state rel ≤ 0.10`, `cos ≥ 0.995`, `drift < 3.0`.
**Q4_K misses two of three: 0.1279 > 0.10 and 0.9920 < 0.995. Q6_K passes all three.**
The bounds are NOT amended — this is the rung's answer.

**What passes, and it is the bigger question.** Drift is **≈1.0 at every precision** —
1.018 / 0.993 / 1.092 over 512 steps. **Q4_K does not compound through the recurrence on
trained weights.** The decay gate damps the perturbation rather than amplifying it, which
is exactly the failure mode PHYSICAL-1 was built to catch, and it is absent. The state
error is a *level*, not a *slope*.

**The ladder is monotone in bpw**, which is what licenses reading 0.1279 as a property of
the representation rather than of the binding: 8.5 → 6.56 → 4.5 bpw gives 0.0084 → 0.0307
→ 0.1279, and `o_proj` plus `q|k|v` shrink 72.00 MiB → 20.25 MiB (**3.56x**) at a measured
**4.500 bpw**, confirming §8.2.2's arithmetic on real tensors.

##### The synthetic substrate was OPTIMISTIC by an order of magnitude

Synthetic Q4_K state error was **0.0121**; real is **0.1279** — **10.6x worse**. The
expectation going in was the opposite: that random weights would be the pessimistic case.
They are not. Smooth `sin()`-derived weights have no outlier channels; trained weight
matrices do, and a block-scaled format is hurt precisely by them. **A synthetic substrate
qualifies a mechanism and its controls. It cannot bound a representation's error, and this
rung would have reported a 10x-too-good number had it stopped at 8.2.2a.**

##### Budget consequence, if KDA has to move to Q6_K

| KDA at | KDA GB/token | KDA ms | whole token | token ms | roofline |
|---|---:|---:|---:|---:|---:|
| Q4_K | 2.806 | 9.35 | 5.99 GB | 20.0 | 50.1 tok/s |
| **Q6_K** | **3.982** | **13.27** | **7.17 GB** | **23.9** | **41.8 tok/s** |
| Q8_0 | 5.088 | 16.96 | 8.27 GB | 27.6 | 36.3 tok/s |

**20 tok/s survives KDA at Q6_K with 2.1x roofline headroom**, and Q6_K lands inside
§8.2.2's *healthy* band (12.5–15.6 ms) at the roofline. So a Q4_K miss does not threaten
the programme; it moves KDA one rung up the ladder and costs ~4 ms of the 30 ms slack.
**Capacity is unaffected** — this is the trunk, not the expert bank.

##### Two facts about the checkpoint, found by loading it

1. **The KDA block is not one dtype.** Kimi ships `A_log` and `dt_bias` as **F32** and
   everything else in the block as BF16. "Small tensors at source precision" is therefore a
   **per-tensor** fact, not a per-block one, and the BF16-residency work in §8.2.2 must read
   the declared dtype rather than assume it.
2. **The buffer cache's (ptr, len) aliasing bit again, in a new place.**
   `stable_offset_table` caches by address and length; a loop-local `[ExpertOffset; 3]`
   lands on the same stack address with the same length each iteration, so arms 2 and 3 were
   handed arm 1's cached table — Q8_0 slot strides applied to a Q4_K bank, reading past its
   end. It surfaced as **NaN** on Q6_K and Q4_K while Q8_0, the first arm, read a healthy
   0.0084. **A ladder whose first rung is correct and whose later rungs are NaN is the
   signature.** Every arm's offsets are now built up front and held live at distinct
   addresses.

##### What is still open

- **GLM remains blocked** on `KdaGateForm` (§8.2.2a). This result transfers to GLM only as
  far as the two families share the operator, and their decay forms differ by 2.8x.
- **Unit-RMS input, not real hidden states.** The scale is right; the direction distribution
  is not. Real states need the layers below.
- **No behavioural discriminator yet.** Whether 0.128 state / 0.992 cosine changes the
  emitted token is unmeasured, and it is the evidence that would say whether the declared
  0.10 bound was the right question or merely a plausible one.

#### 8.2.2c KDA-GATE-METAL-1 — CLOSED 2026-09-07, the blocker is fixed

Raised as an independent correctness rung rather than folded into
PHYSICAL-1: it is an executor defect the rung *found*, not test scaffolding it
*needed*.

> **Metal executes the family-declared decay-gate form, or refuses. It may never
> silently substitute one for the other.**

`shaders::kda::kda_decay_gate` now takes `form` and `lower_bound`;
`KdaDeviceWeights.gate_form: KdaGateForm` is a **required field** — deliberately not
`Option` with a default, because a default here *is* the silent substitution — which forced
all four construction sites to state it; `declared_gate_form(Option<KdaGateForm>)` refuses
`None` with `GroupedError::KdaGateFormUndeclared`.

**Five gates, all green** (540 crate tests, fmt and clippy clean):

| gate | what it establishes |
|---|---|
| softplus matches its scalar formula | elementwise, on the **same `f_low` the kernel saw** (`KdaDevicePlanes.f_lowrank`/`.g_decay`), not a re-derivation |
| clamped sigmoid matches its scalar formula | elementwise, **and** every channel stays inside `[lower_bound, 0]` |
| the two forms are different computations | **two-sided** — all `WIDTH` channels differ, distinguishing "declared and honoured" from "declared and ignored for a hard-coded form" |
| swapping the form moves the **trajectory** | state rel > 0.5, cos < 0.99 — **~4x Q4_K's 0.128 on real weights**, which is precisely why a wrong form would have been misread as a representation failure |
| an undeclared form is refused by name | the message names the fact and why the declaration cannot supply it |

**The Softplus path is bit-unchanged**: the 16 pre-existing KDA tests pass untouched and
the real Kimi ladder reproduces exactly (Q4_K 0.127923).

##### A correction to this document's own source

The claim carried until now — *"Kimi and GLM both declare `gate_lower_bound: -5.0`; only
GLM applies it"* — **is wrong about Kimi.** Verified on disk 2026-09-07:
`Kimi-Linear-48B-A3B-Instruct` mentions the key in **no file** — not `config.json` (its
`linear_attn_config` holds only `full_attn_layers`, `head_dim`, `kda_layers`, `num_heads`,
`short_conv_kernel_size`), not `configuration_kimi.py`, not `modeling_kimi.py`.

**The true invariant is stronger than the one it replaces.** It is not that a shared value
cannot select the form; it is that **neither presence nor absence selects it**. Kimi
declares nothing and computes softplus; GLM declares −5.0 and computes the clamped sigmoid
— *and defaults to the same −5.0 when the key is null and `safe_gate` is set*, so an absent
bound still means clamped there. **Absence means opposite things in the two families**,
which is exactly why the tempting inference "declared ⇒ clamped, absent ⇒ softplus" is
unsafe even though it happens to fit this pair.

The real-weight example now derives the form from `config.json`'s `architectures` and
**panics on an unjudged family** rather than defaulting.

#### 8.2.2d PHYSICAL-1 on REAL GLM — baseline GREEN, ladder RUN 2026-09-07

**The oracle is reproducible first.** `tools/oracles/glm5/` — pinned `requirements.txt`
(torch 2.14.0, transformers 5.16.1, numpy 2.5.2, safetensors 0.8.0 on Python 3.12),
`bootstrap.sh`, `verify_sources.sh` checking the installed `glm5_next` against
`scripts/glm_reference_sources.sha256`, and `smoke.py`. The venv is not in the repo; its
construction is. Steps 1–2 pass without the weights mounted and say so, so the environment
verifies separately from the checkpoint.

##### The baseline, and it had to come first

`upstream GLM BF16 → LARQL Metal BF16`, layer 0, the SAME 512 positions:

> **worst rel-L2 9.270e-5 against a 2e-3 floor — BASELINE GREEN.**

The example **returns without scoring any quantised arm** if this misses. It could not have
been green before §8.2.2c: the family line prints
`Glm5NextForConditionalGeneration: KdaGateForm::ClampedSigmoid { -5 }`, and the oracle's own
decay tensor has `absmax = 5.000000` exactly — GLM's gate is pinned at its declared bound,
which softplus cannot produce. **The gate-form rung is independently confirmed by the
reference.**

##### The ladder — GLM reproduces Kimi

512 steps, declared bounds `state rel ≤ 0.10`, `cos ≥ 0.995`, `drift < 3.0`:

| encoding | bpw | GLM state rel | GLM cos | GLM drift | Kimi state rel | verdict |
|---|---:|---:|---:|---:|---:|---|
| Q8_0 | 8.500 | 0.01077 | 0.999945 | 1.171 | 0.00840 | **passes** |
| **Q6_K** | **6.562** | **0.03968** | **0.999221** | **1.113** | 0.03071 | **passes** |
| Q4_K | 4.500 | 0.14928 | 0.989003 | 0.872 | 0.12792 | **misses rel and cos** |

**Two families, two geometries (64×128 vs 32×128), two decay forms, two sets of trained
weights — and the same ladder.** GLM runs 15–30 % worse than Kimi at every rung and the
shape is identical. That is the transfer evidence the Kimi arm could not supply on its own.

**Drift is ≈1.0 everywhere (0.87–1.17), including at Q4_K.** GLM's per-step state error is
essentially constant across the run — 0.1141 at step 0, 0.1166 at step 511. The recurrence
damps the perturbation on the target model too; at Q4_K it mildly *contracts* it.

##### PHYSICAL-1 QUALITY verdict

> **Q6_K is PHYSICAL-1's representation. Q4_K is RED on the pre-registered bounds, on both
> families' real weights, and the bounds are not amended.**

The recurrence-stability risk — the reason this rung existed — **is cleared at every
precision tested.** What Q4_K fails is magnitude, not stability.

Budget at Q6_K: KDA **3.982 GB/token → 13.27 ms**, whole token **7.17 GB → 23.9 ms**,
roofline **41.8 tok/s — 2.1× the target**, and inside §8.2.2's *healthy* band. **The 20 tok/s
programme absorbs the Q4_K miss with room to spare.**

##### Open

- **PHYSICAL timing has not run.** This is QUALITY only.
- **BF16 small-tensor residency still not implemented** — the path remains the 3.045 GB/token
  row. The Kimi load also showed the KDA block is **not one dtype** (`A_log`, `dt_bias` F32,
  the rest BF16), so that work must read the declared dtype per tensor.
- **Unit-RMS input, one layer, one seed.** The oracle drives layer 0 only.
- **No behavioural discriminator.** Whether Q6_K's 0.040 versus Q4_K's 0.149 changes an
  emitted token is unmeasured, and it is now the question worth asking — but as a *new*
  question, not a retrospective amendment to this verdict.

#### 8.2.2e PHYSICAL-1 PHYSICAL — RUN 2026-09-07. **20.1 ms, and Q6_K's win is capacity, not speed**

**Dtype-preserving binding first.** `SmallMatrix::{F32, Bf16}` binds each of KDA's five
small matrices at the precision the *checkpoint* stores, per tensor — not a blanket BF16
conversion, because the block is not one dtype. On GLM all five are BF16: **6.50 MiB/layer
instead of 13.00**, and the trajectory numbers are **bit-identical** to the widened run
(baseline 9.270e-5, Q6_K 0.039682). Pure traffic, zero fidelity cost, exactly as §8.2.2
claimed. `encode_bf16_gemv_into` is a new encode-into sibling so this costs no extra
submission.

**Measurement.** One real GLM KDA layer, median of 200 steady-state steps, GPU span of the
step's single command buffer (host glue excluded). Idle machine, no peer build. Two runs
agree within 0.5 %.

| encoding | ms/layer | **× 34 layers** | wide MiB | realised GB/s | % of the 300 envelope |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.0839 | **36.9** | 256.00 | 254 | 85 % |
| Q8_0 | 0.6260 | **21.3** | 136.00 | 239 | 80 % |
| **Q6_K** | **0.5925** | **20.1** | 105.00 | 198 | **66 %** |
| Q4_K | 0.5274 | **17.9** | 72.00 | 157 | 52 % |

**Q6_K KDA is 20.1 ms — the *viable* band, at its boundary with *healthy*.** And the
borrowed envelope question is answered: a **recurrent** composition of quantised GEMVs
realises about **two-thirds** of the flat-GEMV rate, not all of it.

##### The finding that changes the choice: below Q8_0 the layer stops being bandwidth-bound

| step | ΔMB/layer | Δms/layer | implied marginal GB/s |
|---|---:|---:|---:|
| BF16 → Q8_0 | 125.8 | 0.4585 | **274** — a real bandwidth saving |
| Q8_0 → Q6_K | 32.5 | 0.0318 | **1022** |
| Q6_K → Q4_K | 34.6 | 0.0651 | **532** |

**A marginal rate above the machine's 400 GB/s peak is not a bandwidth saving.** Past Q8_0
the layer has hit a floor made of its non-projection stages — conv+SiLU, the two L2 norms,
the decay and output gates, the recurrence, the gated RMS norm — plus per-dispatch cost.
Byte savings below Q8_0 buy far less time than the roofline predicts, and the realised-GB/s
column falls (85 → 80 → 66 → 52 %) for exactly that reason: the fixed part is a growing
share of a shrinking total.

**Consequence.** Q8_0 costs **1.2 ms more across all 34 layers** than Q6_K and delivers
**3.7× better state fidelity** (0.0108 vs 0.0397). On capacity the two are 85.5 GB vs
86.6 GB resident — **both inside the ~103 GB wired cap**. So neither constraint forces
Q6_K:

> **Q6_K's advantage over Q8_0 is capacity, and KDA is the throughput lever, not the
> capacity one. On this evidence Q8_0 is the better KDA operating point** — 1.2 ms for 4×
> the fidelity margin, on the only term whose error enters a recurrent state.

That does not disturb §8.2.2d's QUALITY verdict (Q6_K passes, Q4_K red). It is a *physical*
argument for spending 1.2 ms of the 30 ms slack to sit further from the bound.

##### Budget, with measured KDA rather than roofline KDA

| KDA at | measured ms | share of 50 ms | left for experts + MLA/DSA + mHC + head |
|---|---:|---:|---:|
| BF16 | 36.9 | 74 % | 13.1 ms |
| Q6_K | 20.1 | **40 %** | **29.9 ms** |
| Q8_0 | 21.3 | 43 % | 28.7 ms |

**Quantising KDA at all is worth 16.7 ms** — the largest measured win in the programme.
Against a ~7.9 ms expert-traffic roofline at 2 bpw, roughly **22 ms** remains for MLA/DSA,
mHC, router, head and dispatch. 20 tok/s is not excluded by this measurement, and the
largest unpriced term is now unambiguously DSA.

**What this does not license.** One layer, measured in isolation, warm and resident, stepped
repeatedly. In a real token the 34 KDA layers contend with expert traffic for the same
bandwidth, so 20.1 ms is an optimistic share, not a schedule. A whole-token measurement is
PHYSICAL-5, not this rung.

#### 8.2.2f DSA cost model (analytic) and KDA-REPRESENT-1 rung 0 — 2026-09-07

##### PHYSICAL-3 cannot run: **DSA is unbuilt**

§4 lists DSA as the one execution type with no implementation. It cannot be timed. What the
pinned oracle *can* supply is its cost structure, read from `Glm5NextTextIndexer`:

- `wk` is `hidden → index_head_dim` — **one 128-dim index key per position, SHARED across
  all 32 index heads**, not per-head. `index_kpool: 4` pools it 4:1, so the scanned set is
  `N/4` entries of 128 values.
- `select_k = index_topk / index_kpool = 512` pools, expanded to 2048 raw indices, plus a
  tail (`index_kpool_always_select_tail: true`).
- The selected-set MLA attention is therefore **constant in context** at ~2048–4095 latents.

| context | pools scanned | index scan MB | selected MLA MB | total MB | ms @300 GB/s | % of 50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 K | 2,048 | 5.8 | 46.1 | 51.9 | 0.173 | 0.3 % |
| 32 K | 8,192 | 23.1 | 46.1 | 69.2 | 0.231 | 0.5 % |
| 128 K | 32,768 | 92.3 | 46.1 | 138.4 | 0.461 | 0.9 % |
| 512 K | 131,072 | 369.1 | 46.1 | 415.2 | 1.384 | 2.8 % |
| 1 M | 262,144 | 738.2 | 46.1 | 784.3 | **2.614** | **5.2 %** |

Plus a constant 152.8 MB of indexer weights across 11 layers (0.509 ms).

**DSA's *traffic* is small and its context-scaling is gentle** — 2.6 ms at the full 1 M
context — because the index key is 128-dim and pooled 4:1. **The unpriced risk is therefore
not bandwidth; it is the top-512-of-262,144 selection** per layer per token (2.9 M
candidates scored across 11 layers at 1 M context), which is latency- and dispatch-shaped.
**Analytic only — no kernel exists and none of this is measured.**

##### KDA-REPRESENT-1 rung 0: where do the bits actually matter?

Uniform Q4_K spends the same bits on all four wide projections. **It should not.** Each
projection round-tripped through Q4_K on its own, every arm bound BF16 so the kernel is
identical and only that projection's error differs (real GLM layer 0, 512 steps):

| quantised | weight rel-L2 | **state_rel** | state_cos | out_traj |
|---|---:|---:|---:|---:|
| `q_proj` | 0.0787 | **0.000000** | 1.000000 | **0.3569** |
| `k_proj` | 0.0787 | **0.125990** | 0.992113 | 0.1773 |
| `v_proj` | 0.0786 | **0.109914** | 0.994308 | 0.2132 |
| `o_proj` | 0.0793 | **0.000000** | 1.000000 | 0.2144 |
| ALL FOUR | 0.0787 | 0.149019 | 0.989009 | 0.3307 |

**Control passes**: the ALL FOUR arm reads 0.149019 against the real Q4_K bank's 0.149277
— **0.17 %** — so the round-trip emulation is sound.

**Half of KDA's wide parameters have ZERO state sensitivity.** `q` reads the recurrent state
and never writes it; `o_proj` sits entirely downstream of it. Their weight error is the
*same* as k's and v's (0.079) and their state contribution is bit-exactly nothing. Only
**k and v** drive the state, and `sqrt(k² + v²) = 0.167` against a measured 0.149 — roughly
independent, with mild cancellation.

**This halves the search space and changes the target's shape:**

| k,v at | q,o at | effective bpw | state quality |
|---|---|---:|---|
| Q6_K | Q4_K | 5.53 | 0.0397 passes |
| Q6_K | 3-bit | 4.78 | 0.0397 passes |
| Q6_K | 2-bit | 4.28 | 0.0397 passes |
| Q6_K | **1-bit** | **3.78** | 0.0397 passes |
| Q4_K | 3-bit | 3.75 | 0.149 **RED** |

**≤3.75 bpw at Q6-level state quality needs q and o under 1 bit** — because k and v must
hold ~6.56 bpw to carry the state gate. So the target is reachable by *two* routes, and they
are different research problems:

1. **Make k and v cheaper than Q6_K at Q6 quality.** This is where outlier separation,
   per-block promotion and sparse residual belong — applied to **half** the matrix, not all
   of it.
2. **Crush q and o.** They are free on the declared gate.

##### And that exposes a hole in the gate

**The pre-registered gate is state-only, so it would score q and o at ZERO bits as passing.**
That is plainly wrong — `q_proj` at Q4_K already produces the *largest* single-projection
output error of the four (`out_traj` 0.357, above k's 0.177 and v's 0.213). q is the most
output-sensitive projection while being state-inert; the two boundaries are perfectly
complementary, which is exactly why §8.2.2a kept the output metric instead of dropping it as
redundant.

> **KDA-REPRESENT-1 needs a second, output-side admission gate before it can allocate bits
> to q and o. With state alone, the optimiser's correct answer is zero bits for half the
> matrix.**

That gate does not exist yet and is the rung's first owed item — ahead of any search.

#### 8.2.2g KDA-REPRESENT-1 rung 1 — the output gate, and it **inverts** rung 0

Two admission boundaries, kept **independent** rather than summed into one score, both set at
Q6_K parity (the representation §8.2.2d qualified: state 0.0397, out_traj 0.0962):

> **STATE** `state_rel ≤ 0.10`, `cos ≥ 0.995`, `drift < 3` · **OUTPUT** `out_traj ≤ 0.10`

Each projection swept with a reference RTN quantiser (per-256 absmax, symmetric) — chosen
because no fixed ggml format can sweep width continuously. **It is strictly weaker than a
real codec, so every figure is a LOWER BOUND**; the Q4_K arms calibrate the offset.

| projection | 2 bits | 3 | 4 | 5 | 6 | 8 | binding |
|---|---:|---:|---:|---:|---:|---:|---|
| `q_proj` state | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| `q_proj` **output** | 1.847 | 0.818 | 0.498 | 0.283 | 0.107 | **0.032** | **8 bits** |
| `k_proj` **state** | 0.841 | 0.413 | 0.329 | 0.116 | **0.047** | 0.017 | **6 bits** |
| `k_proj` output | 1.535 | 0.825 | 0.349 | 0.227 | 0.082 | 0.023 | 6 bits |
| `v_proj` **state** | 0.960 | 0.424 | 0.163 | 0.104 | **0.039** | 0.012 | **6 bits** |
| `v_proj` output | 2.232 | 0.827 | 0.367 | 0.218 | 0.092 | 0.022 | 6 bits |
| `o_proj` state | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| `o_proj` **output** | 2.090 | 0.778 | 0.334 | 0.157 | 0.078 | 0.020 | **6 bits** |

##### Rung 0's conclusion is inverted

**`q_proj` is the most expensive projection, not the cheapest.** It is state-inert at every
width — 0.000000 down to 2 bits — and needs **8 RTN bits** to clear the output gate, more
than any other. `o_proj` is likewise state-inert and still needs 6. The plan that fell out of
rung 0 — *"q and o are free, crush them to 1 bit"* — would have destroyed the layer's output
while every state number stayed perfect.

**This is exactly why the output gate had to exist before any allocation.** With the state
boundary alone an optimiser minimising bpw would have driven half the matrix to zero bits
and passed.

Per-projection allocation is now `q=8, k=6, v=6, o=6` → **6.50 RTN bpw**, against uniform
Q6_K's 6.56. **Asymmetric allocation buys essentially nothing at this gate setting** — the
saving on `k`/`v`/`o` is spent entirely on `q`.

##### Where the room actually is

**1. The codec gap.** Q4_K at 4.5 bpw scores between RTN-4 and RTN-5 on both k and v
(k: 0.126 against 0.330 and 0.116) — **a real codec is worth ~0.4–0.9 bits over RTN**. So the
6-bit RTN requirement is perhaps 5.1–5.6 bits in a K-quant-class format, and less again with
outlier separation or sparse residual.

**2. The state floor on k and v is immovable by allocation.** Sweeping the output bound moves
only q and o:

| output bound | q | o | k | v | effective |
|---:|---:|---:|---:|---:|---:|
| 0.10 (Q6 parity) | 8 | 6 | 6 | 6 | **6.50** |
| 0.20 | 6 | 5 | 6 | 6 | 5.75 |
| 0.35 | 5 | 4 | 6 | 6 | 5.25 |

**k and v sit at 6 in every row.** The route below 4 bpw runs through making them cheaper at
Q6 state quality — which is KDA-REPRESENT-1A, and it is the right first search.

**3. The output bound is the most consequential free parameter in the programme, and it is
currently set by analogy.** `out_traj ≤ 0.10` is Q6 parity, not evidence: nothing yet says
what layer-output deviation the *model* tolerates. Loosening it to 0.35 removes 1.25 bpw on
its own. **The behavioural discriminator has therefore stopped being a later nicety — it sets
the gate that decides whether KDA can go below 4 bpw at all**, and it should now precede the
k/v search rather than follow it.

#### 8.2.2h KDA-BEHAV-1 prerequisite — **the synthetic substrate is off-manifold by 113x, and rung 1 must be re-run**

Before calibrating the output gate against behaviour, one cheap question: `out_traj` is
normalised by the KDA layer's **own** output, so how large is that output next to the stream
it joins? If the layer contributes a small fraction, `0.10` is far stricter than behaviour
needs.

**The cheap estimate said yes, and it was wrong.** From the 512-position smoke run —
unit-RMS input, output RMS 0.0171 — the ratio looked like **0.017**, implying `out_traj 0.10`
is 0.17 % of the stream and the gate is ~60x too strict.

**Measured on REAL tokens through the REAL mHC join** (`tools/oracles/glm5/kda_stream_ratio.py`;
layer 0 is the first layer, so its input stream *is* the embedding and no other layer has to
run):

| quantity | value |
|---|---:|
| real embedding RMS | 0.008832 |
| KDA sub-output RMS | 0.006885 |
| **ratio to input stream** | **0.780** |
| **ratio to output stream** | **1.110** |

**`out_traj = 0.10` is ~11 % of the stream, not 0.17 %.** The surrogate was wrong by **46x**,
in the direction that would have licensed loosening the gate. The output gate at Q6 parity is
*not* obviously too strict.

##### Why: KDA is not scale-invariant, and the synthetic input was 113x off

| input | RMS in | RMS out | out/in |
|---|---:|---:|---:|
| synthetic unit-RMS | 0.99999 | 0.017078 | **0.0171** |
| real embedding | 0.008832 | 0.006885 | **0.7796** |

A scale-linear operator would give the same ratio at both. It differs by **46x**, because
`q` and `k` are **L2-normalised per head** — scale-free — while the decay and output gates
are softplus/sigmoid of projections and **saturate differently at different input scales**.

**⇒ Every magnitude in §8.2.2f and §8.2.2g was measured with the operator in the wrong
regime.** The unit-RMS substrate put a 113x-too-large vector into a scale-sensitive operator.

**What survives, and what does not:**

- **Survives — the structural facts.** `q_proj` and `o_proj` contribute *bit-exactly zero*
  state error at every width. That is topology (q reads the state and never writes it;
  `o_proj` is downstream of it), not a numerical result, and no input distribution changes it.
- **Does not survive — every bit count.** The 6-bit k/v state requirement, the 8-bit q output
  requirement, the 6.50 bpw allocation and the bound-sweep table are all measured off-manifold
  and must be re-run on real hidden states before they mean anything.
- **Also suspect — §8.2.2d's ladder magnitudes.** The GLM Q4_K/Q6_K/Q8_0 state errors
  (0.149/0.040/0.011) come from the same unit-RMS substrate. The *verdict ordering* is
  probably robust; the numbers that were compared against the pre-registered bounds are not
  established on-manifold. **The pre-registered bounds were met by numbers from the wrong
  regime, so PHYSICAL-1's QUALITY verdict is now provisional pending a real-state re-run.**

This is the same failure the expert reuse curve hit — *"the reuse curve needs REAL hidden
states, measured not assumed"* — and the same lesson as §8.2.2b's synthetic-vs-trained 10.6x:
**a synthetic substrate qualifies mechanism and controls; it establishes no magnitude.** Here
it was not even the weights that were synthetic, but the *input scale*, and that was enough.

##### What KDA-BEHAV-1 still needs, and the blocker

Real hidden states for the trajectory arms are cheap at layer 0 (embedding only). **Logits are
not**: `prompt -> candidate KDA -> trusted remainder -> logits` needs all 45 layers and the
288-expert bank — **~305 GB**. That is the same wall the residency programme is built around,
and it is why no behavioural number exists yet. Options, none free: a truncated remainder
(measures propagation, not behaviour), the Kimi 48B path as a proxy family, or out-of-core
streaming of the remainder at minutes per prompt.

**Ordering consequence:** re-run rungs 0–1 on real embedded states *first* — it is cheap and
it is now known to change the answer — then decide whether the behavioural experiment is worth
its cost.

#### 8.2.2i ON-MANIFOLD RE-RUN — **Q4_K passes the state gate, and the allocation is 4.25 bpw**

##### First, a correction to §8.2.2h

That section compared the synthetic `x` against the **raw embedding** (RMS 0.008832). KDA's
actual input is post-`input_layernorm`, post-mHC, and measured at **RMS 0.0982**. So:

| claim in §8.2.2h | corrected |
|---|---|
| synthetic input 113× too large | **10.2×** |
| out/in regime shift 46× | **4.5×** |

The operator *is* scale-sensitive and the substrate *was* off-manifold — both conclusions
stand — but by an order of magnitude less than stated. The KDA-output-to-stream ratio (0.864)
is unaffected: it came from the real run, so `out_traj 0.10` really is ~8.6 % of the stream.

##### The re-run, on 512 real post-norm KDA states

| encoding | bpw | state_rel | state_cos | drift | out_traj | vs off-manifold state |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 | 8.500 | 0.00420 | 0.999974 | 0.980 | 0.0098 | 0.01077 |
| Q6_K | 6.562 | 0.01534 | 0.999874 | 0.997 | 0.0356 | 0.03968 |
| **Q4_K** | **4.500** | **0.05973** | **0.998250** | **0.976** | 0.1404 | 0.14928 |

> **Q4_K passes the STATE gate on-manifold — 0.0598 ≤ 0.10, cos 0.9983 ≥ 0.995, drift 0.976.**
> The off-manifold substrate scored it **2.5× worse** and put it in RED. It still misses the
> output gate (0.140 > 0.10), so it is not simply green — but §8.2.2d's headline, *"Q4_K is
> RED on the pre-registered bounds"*, was an artifact of the substrate and is **withdrawn**.

##### The allocation curve, on-manifold — and `q` inverts again

| projection | 2 bits | 3 | 4 | 5 | 6 | 8 | binding |
|---|---:|---:|---:|---:|---:|---:|---|
| `q` state | 0 | 0 | 0 | 0 | 0 | 0 | — |
| `q` output | 0.185 | **0.053** | 0.017 | 0.009 | 0.004 | 0.001 | **3 bits** |
| `k` **state** | 0.511 | 0.163 | **0.073** | 0.035 | 0.017 | 0.004 | **4 bits** |
| `v` state | 0.528 | 0.169 | 0.072 | 0.035 | 0.017 | 0.005 | 4 |
| `v` **output** | 0.851 | 0.310 | 0.136 | **0.065** | 0.031 | 0.008 | **5 bits** |
| `o` **output** | 1.089 | 0.409 | 0.177 | **0.083** | 0.041 | 0.011 | **5 bits** |

| quantity | off-manifold | **on-manifold** |
|---|---:|---:|
| `q_proj` output error at Q4 | 0.3569 | **0.0124** (29× better) |
| `q` binding width | **8 bits** — most expensive | **3 bits** — cheapest |
| allocation | 6.50 bpw | **4.25 bpw** |

**`q` inverts for the second time.** Rung 0 said "free" (state-only). Rung 1 said "most
expensive" (8 bits, output-bound, off-manifold). On-manifold it is the **cheapest at 3 bits**.
Only the structural fact — `q` and `o` contribute bit-exactly zero state error — has survived
all three readings, which is exactly what a topological fact should do.

##### This is the REPRESENT-D regime, measured

**`q=3, k=4, v=5, o=5` → 4.25 RTN bpw**, already below uniform Q4_K's 4.5, using a *naive*
reference quantiser. Q4_K scores between RTN-4 and RTN-5 on the same tensors, so a
K-quant-class codec at this allocation is roughly **3.35–3.85 effective bpw** — the estimate
is an inference, not yet measured, but the RTN figure that anchors it is not.

**Sub-4-bpw KDA is not a research aspiration; it is approximately where a per-projection
allocation already lands once the operator is measured in its real regime.** The earlier
"6.5 bpw, asymmetry buys nothing" conclusion was entirely a substrate artifact.

##### Consequences

- **PHYSICAL-1's QUALITY verdict is re-opened.** §8.2.2d adjudicated Q6_K on off-manifold
  numbers. On-manifold, Q6_K is comfortable (0.0153) and Q4_K passes state. The
  representation choice must be re-adjudicated against the real-state ladder.
- **The physical timing survives** (§8.2.2e): it is bytes-and-shape dominated on real weights
  and real kernels, and the state substrate does not change 105 MiB of Q6_K traffic. But
  *which* encoding to time has re-opened — Q4_K at 17.9 ms/34 layers is now a live candidate.
- **The output gate is now the binding constraint on three of four projections** (`q`, `v`,
  `o`), and it is still set by analogy at Q6 parity. KDA-BEHAV-1 has become *more* valuable,
  not less.
- **Two synthetic-substrate failures now, in one programme**: synthetic *weights* were 10.6×
  optimistic (§8.2.2b), synthetic *input scale* was 2.5× pessimistic here — in opposite
  directions. The rule that survives both: **a synthetic substrate proves mechanism and fires
  controls; it establishes no magnitude and no admission.**

#### 8.2.2j KDA-BEHAV-1 — roster FROZEN, and the output bound is worth ~1 bpw

Frozen before any downstream measurement, so that experiment qualifies a fixed set rather
than becoming a search over allocations. Widths come from §8.2.2i's on-manifold binding
widths and from deliberately more aggressive points below them. 512 real post-norm states.

| candidate | q/k/v/o | RTN bpw | state_rel | state | out_traj | output |
|---|---|---:|---:|---|---:|---|
| strict | 3/4/5/5 | **4.25** | 0.07947 | pass | 0.1056 | FAIL (marginal) |
| slightly-aggr | 3/4/4/4 | **3.75** | 0.09884 | pass | 0.2272 | FAIL |
| aggressive | 3/4/4/3 | **3.50** | 0.09884 | pass | 0.4386 | FAIL |
| D-probe | 2/3/4/3 | 3.00 | 0.17613 | **FAIL** | 0.4247 | FAIL |

**The state gate permits 3.50 bpw and breaks at 3.00.** Every candidate fails the *output*
gate; three of four pass state. **The entire remaining bit budget is decided by a bound that
is currently set by analogy.**

| if behaviour says | best passing candidate |
|---|---|
| `out_traj ≤ 0.10` (today's, by analogy) | **nothing in the roster** — back to Q6-class |
| `≤ 0.15` | strict, **4.25 bpw** |
| `≤ 0.25` | slightly-aggr, **3.75 bpw** |
| `≤ 0.45` | aggressive, **3.50 bpw** |

**That spread — Q6-class down to 3.50 bpw — is the value of running KDA-BEHAV-1.** It is
worth roughly 1 bpw on KDA, or ~3 bpw against the Q6_K that PHYSICAL-1 provisionally adopted.

##### A control fired inside the roster

`slightly-aggr` (3/4/4/**4**) and `aggressive` (3/4/4/**3**) differ **only in `o`**:

- `state_rel` **0.098840 vs 0.098840** — identical to six decimals
- `out_traj` 0.2272 vs 0.4386 — moved 1.93×

`o_proj`'s state-inertness is re-confirmed independently, on-manifold, inside a combined
candidate rather than a single-projection arm.

##### Errors do not compose — allocation cannot be read off marginal sweeps

| | state_rel |
|---|---:|
| `k` alone at 4 bits | 0.07322 |
| `v` alone at 5 bits | 0.03491 |
| quadrature of the two | 0.08111 |
| **`strict` 3/4/5/5 measured** | **0.07947** |

Close to quadrature, slightly under. So marginal per-projection widths are a *starting point*
for an allocation, not the allocation itself: `strict` was built from rung 1's binding widths
and still misses the output gate at 0.1056. **Any REPRESENT-D search must score joint
candidates, not sum marginal sensitivities.**

##### What KDA-BEHAV-1 costs, honestly

`prompt → candidate KDA → trusted remainder → logits` needs all 45 layers and the routed bank.
Per prompt, prefill over ~32 positions touches roughly half the 288 experts per sparse layer
(the reuse curve measured 225 of 288 distinct over 128 tokens), so **~145–283 GiB streamed per
prompt**. At the deep-fetch rate this programme measured (2,759 MiB/s, §"DEEP FETCH") that is
**~1–2 minutes per prompt of pure I/O**, realistically several times that through the
reference. A 16-prompt bank is therefore an hours-long background job, not an interactive
experiment — but it is bounded, and it decides ~1 bpw on the largest trunk term.

**Design, fixed now:** calibration and holdout banks split before any candidate is scored;
report distributions across prompts (KL, top-1 agreement, top-k overlap, margin erosion,
short greedy continuation), not means; judge candidates *relative* to the Q8/Q6 controls
against BF16 rather than against an invented absolute KL; and keep the state gate independent
— a candidate may not buy behavioural quality by violating it.

#### 8.2.2k NATURAL-PROMPT BANK — the roster survives, with 20x more margin

The 512-state bank in §8.2.2i–j came from **uniformly random vocabulary ids**. Natural text
does not sample the vocabulary uniformly, and on this checkpoint the difference is
measurable:

| token source | median id | embedding row RMS (mean) | spread (min → p95) |
|---|---:|---:|---|
| natural prompts | **982** | 0.007857 | 0.0022 → 0.0125 |
| uniform random | 58,135 | 0.011142 | 0.0065 → 0.0131 |

**Natural tokens carry 0.705× the embedding RMS and a far wider spread.** With KDA's measured
scale sensitivity (4.5× ratio shift over 10.2× input scale), that forecast a **~1.25×** move
— and the roster's state margins were **1.2 %**, so the verdicts were not robust to it.

**Re-run on 12 natural prompts / 198 tokens** (post-norm KDA input RMS 0.0849):

| candidate | q/k/v/o | RTN bpw | random bank | **natural bank** | change | margin | state | out_traj | output |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| strict | 3/4/5/5 | 4.25 | 0.07947 | **0.05823** | −26.7 % | +41.8 % | pass | 0.1142 | FAIL |
| slightly-aggr | 3/4/4/4 | **3.75** | 0.09884 | **0.07712** | −22.0 % | **+22.9 %** | pass | 0.2413 | FAIL |
| aggressive | 3/4/4/3 | **3.50** | 0.09884 | **0.07712** | −22.0 % | **+22.9 %** | pass | 0.4682 | FAIL |
| D-probe | 2/3/4/3 | 3.00 | 0.17613 | 0.13022 | −26.1 % | −30.2 % | **FAIL** | 0.4585 | FAIL |

Uniform ladder moved the same way: Q8_0 0.00420 → 0.00327, Q6_K 0.01534 → 0.01184,
Q4_K 0.05973 → **0.04692**, all −21 to −23 %.

**The forecast was right.** Predicted ~25 %, observed 22–27 % across every arm. That is the
first time in this programme a substrate change was *predicted* before being measured rather
than discovered afterwards — the scale-sensitivity model earned from §8.2.2h/i now has one
successful out-of-sample prediction.

**The conclusions hold and strengthen:**

- **The state gate permits 3.50 RTN bpw with 23 % margin** (was 1.2 %), and still rejects 3.00
  at −30 %.
- **The output gate still binds every candidate**, so the entire remaining budget is still
  decided by a bound set by analogy.
- **The `o`-inertness control fired a third time**: `slightly-aggr` and `aggressive` differ
  only in `o` and read **0.077124 vs 0.077124** — identical — while `out_traj` moves 1.94×.

**Owed:** 12 prompts is a calibration bank, not a holdout; the split must be made before any
candidate is scored behaviourally. And this is still layer 0 only.

#### 8.2.2l ROUTING-UNION — KDA-BEHAV-1 is ~8x cheaper than priced, and routing is already a signal

`tools/oracles/glm5/routing_union.py`. Layers 0–3 loaded **once** (layer 3 without its
288-expert bank — routing needs only `mlp.gate`), then all eight frozen arms fanned through,
perturbing **only layer 0's four KDA projections**. 48 real tokens, layer 3, top-8 of 288.

##### The cost question, measured rather than assumed

| union over all 8 arms, per token | |
|---|---:|
| min / median / p95 / max | 8 / **9** / 11 / 12 |
| if the arms were disjoint | 64 |
| **fetch amplification vs one arm** | **1.15× mean, 1.50× worst** |
| saving vs 8 separate traversals | **85.6 %** |

**The whole roster costs about one BF16 traversal plus 15 %.** A prompt is therefore
**~167–325 GiB**, not ~1,160–2,264 GiB. §8.2.2j priced KDA-BEHAV-1 for isolated arms; fanned
through one traversal it is roughly **one model pass per prompt**, and a 16-prompt
calibration bank becomes a bounded overnight job rather than a multi-day one.

##### And routing agreement is itself a downstream signal — for free

| arm | out_traj | exact top-8 set agreement with BF16 | mean overlap |
|---|---:|---:|---:|
| uniform-8 | 0.0105 | **100.0 %** | 8.00/8 |
| uniform-6 | 0.0381 | **97.9 %** | 7.98/8 |
| strict 4.25 | 0.1142 | **85.4 %** | 7.85/8 |
| uniform-4 | 0.1496 | 52.1 % | 7.50/8 |
| slightly-aggr 3.75 | 0.2413 | 56.2 % | 7.54/8 |
| D-probe 3.00 | 0.4585 | 35.4 % | 7.17/8 |
| aggressive 3.50 | 0.4682 | 35.4 % | 7.12/8 |

This is a **real downstream effect measured without logits and without the expert bank** —
four layers, minutes, no streaming. It already separates the roster: Q6-class perturbations
leave routing essentially intact (97.9 %), `strict` degrades it mildly (85.4 %), and the
aggressive candidates change the routed set on **two tokens in three**.

##### The inversion that matters

`out_traj` correlates strongly with routing agreement — Pearson **−0.922** — but **not
perfectly**:

> `slightly-aggr` carries **1.6× the `out_traj` of `uniform-4`** (0.2413 vs 0.1496) and yet
> routes **better** (56.2 % vs 52.1 %).

**Error direction matters, not only its norm** — exactly the trap flagged before promoting a
local proxy to an admission gate. One inversion in seven arms is enough to say `out_traj`
should stay a **screening** metric: cheap, well-correlated, and not the authority.

**⇒ Routing agreement should join the KDA-BEHAV-1 metric set** as a mid-depth boundary
between the local output error and full logits — and it is cheap enough to run on every
candidate a REPRESENT search proposes, which `out_traj` alone is not trustworthy enough to be.

**Caveats:** one prompt, 48 tokens, layer 3 only, and routing divergence is *not* the same as
behavioural harm — an expert swap may or may not change the emitted token. It is a signal
that discriminates, not a verdict.

#### 8.2.2m BANKS FROZEN — calibration-v1 and holdout-v1

`tools/oracles/glm5/banks/`. Frozen **before any logits were observed**, which is the only
moment at which a holdout can honestly be created.

| bank | role | prompts | tokens | median id | categories | sha256 |
|---|---|---:|---:|---:|---:|---|
| `calibration-v1` | calibration | 12 | 198 | 982 | 10 | `d639576a334895f8` |
| `holdout-v1` | **holdout** | 20 | 511 | 1268 | 10 | `c518b5ec40426242` |

**Overlap: 0 prompts.** `calibration-v1` is the bank §8.2.2k already used for the on-manifold
state re-runs, so it is *already* contaminated as a selection surface — which is exactly why
it is labelled calibration and why a second bank had to exist before the behavioural run.

`holdout-v1` spans factual, instructional, code, maths, reasoning, multilingual,
long-context (up to 75 tokens), numeric/tabular, rare-vocabulary, and **4 `close-call`
prompts**. Its median token id is 1268 against calibration's 982 and uniform-random's 58,135
— natural in both banks, but not identical, which is the point of a holdout.

**The `close-call` label is an INTENT, not a measured property.** Whether those prompts
actually carry small BF16 top-1 margins can only be checked from the BF16 arm at run time,
and any that turn out not to be close calls must be reported as such rather than quietly
re-labelled. That caveat is written into the bank file itself.

**Rules, fixed now:** the roster stays exactly as frozen in §8.2.2j — BF16 / Q8 / Q6 / Q4 /
4.25 / 3.75 / 3.50 / 3.00. No candidate may be added after the first logits are seen. No
REPRESENT search and no gate calibration may read `holdout-v1`.

##### The traversal is designed and costed, and is NOT built

The remaining implementation is the multi-arm traversal: carry all eight candidate residual
states together, and at each sparse layer route every arm, take the union of selected expert
ids, fetch each expert **once**, and execute it only for the arms that chose it.

**What §8.2.2l established:** the *arm* dimension is nearly free — 1.15× mean amplification.
**What it did not establish:** the *token* dimension, which dominates. Across a prompt's
tokens the per-layer distinct-expert count is large (the reuse curve measured 225 of 288
over 128 tokens), so a prompt costs roughly **100–130 GiB streamed for all eight arms
together**, and the two banks together are on the order of **4 TB** — an overnight background
job at the measured 2,759 MiB/s deep-fetch rate, not an interactive one.

**Also unbuilt and required:** selective per-expert loading from the shards (today's scripts
load a layer whole or skip its bank entirely), and per-layer union amplification recorded
**at every sparse layer** rather than only at layer 3 — §8.2.2l is one prompt at one layer,
and whether 1.15× holds with depth is the claim that would make "evaluate a whole frontier
for the cost of one traversal" a REPRESENT primitive rather than a single observation.

#### 8.2.2n MULTIARM-1 — union amplification is **FLAT with depth**, and the traversal is built

`tools/oracles/glm5/multiarm_union.py`. All eight frozen arms carried together through
layers 0–22 of the real checkpoint, each with its own KDA/DSA cache, fetching **only the
union of selected experts** at every sparse layer. 13 real tokens. **3 min 51 s wall**,
RSS flat at 13.1 GiB.

##### The enabling primitive, verified before it was used

`Glm5NextTextExperts` stores `gate_up_proj[n_experts, 2·inter, hidden]`, so a **compact
module holding only the union**, with top-k indices remapped to local positions, was checked
against the full bank and reproduces it **bit-equally** (`rel-L2 0.000e+00`, exact tensor
equality). That is what makes a 288-expert layer runnable without allocating 288 experts —
19 GB of `gate_up_proj` at f32 — and it reuses the reference's own forward, so nothing is
transcribed.

##### The depth curve

| | layers 3–22 (20 sparse layers) |
|---|---|
| bf16 distinct experts / layer | 37 – 84 |
| union over all 8 arms | 38 – 89 |
| **prompt-union amplification** | **1.00× – 1.18×, mean 1.06×** |
| per-token amplification | 1.03× – 1.15× |
| first layer → last layer | 1.08× → 1.07× |
| **trend** | **FLAT / BOUNDED — no growth with depth** |

**The amplification is lower at prompt level than at token level (1.06× vs 1.15×)**, and the
reason is structural: a prompt's own token diversity already forces 37–84 distinct experts
per layer, so eight representation candidates add only two to six more. **The arm dimension
is nearly free, and it stays free with depth.**

> **REPRESENT can evaluate a population of nearby physical candidates for ~6 % more expert
> I/O than evaluating one.** That is infrastructure, not a KDA detail: it is what makes an
> MCP-driven frontier search over a multi-hundred-GB model computationally practical.

##### What it costs to finish

23 layers × 13 tokens took **3:51**, so a full 45-layer prompt is ~8 minutes and both frozen
banks (32 prompts) are **~4 hours** — a bounded background job. That is far cheaper than
§8.2.2m's ~4 TB estimate, because selective union fetch reads 50–90 experts per layer
(1.3–2.3 GB) instead of the whole 6.75 GiB bank.

##### Routing agreement does not collapse with depth

`aggr-3.50` holds 46–92 % exact top-8 set agreement across the twenty layers with no downward
trend. Divergence introduced at layer 0 is **not** compounding through the stack — the same
qualitative finding the recurrent-state drift showed, now at model scale.

**Caveats:** one prompt, 13 tokens, layers 3–22 of 42 sparse layers. Per-layer agreement on 13
tokens moves in 7.7 % steps, so the layer-to-layer variation is mostly sampling noise; the
*absence of a trend* is the claim, not any single layer's value.

#### 8.2.2o KDA-BEHAV-1 — traversal reaches logits; **first behavioural numbers, n=13**

The traversal now runs the full stack: 45 layers, then `Glm5NextTextModel.forward`'s own head
sequence — `norm(hc_head(streams))` where `hc_head` is the unweighted mean over the mHC
streams — then `lm_head`. **7 min 11 s** for all eight arms through all 45 layers, RSS flat at
13.1 GiB.

**Depth curve, now complete over all 42 sparse layers:** prompt-union amplification
**first 1.08× → last 1.08×, mean 1.08×, max 1.18×. FLAT.** §8.2.2n's finding holds to the
bottom of the stack.

##### Saturation checked BEFORE agreement was read

A bank whose BF16 top-1 probability is ≈1.0 everywhere cannot discriminate anything — every
arm would read 100 % agreement whatever it did. Measured first:

> **BF16 confidence: min 0.216, median 0.476, p95 0.889, max 0.987 — only 8 % of positions
> above 0.9.** Not saturated. These are genuinely uncertain positions.

##### First numbers — one prompt, 13 positions

| candidate | KL mean | KL p95 | top-1 | top-5 | rank moved | margin |
|---|---:|---:|---:|---:|---:|---:|
| q8 | 0.00078 | 0.00215 | 100.0 % | 100.0 % | 0.0 % | 0.991 |
| q6 | 0.00517 | 0.02040 | 100.0 % | 96.9 % | 0.0 % | 0.985 |
| q4 | 0.00689 | 0.02536 | 100.0 % | 95.4 % | 0.0 % | 1.011 |
| **strict-4.25** | **0.00361** | 0.01410 | 100.0 % | 95.4 % | 0.0 % | 0.981 |
| **aggr-3.75** | **0.00670** | 0.02525 | 100.0 % | 95.4 % | 0.0 % | 0.984 |
| aggr-3.50 | 0.01683 | 0.08528 | 100.0 % | 96.9 % | 0.0 % | 1.024 |
| probe-3.00 | 0.00786 | 0.03334 | 100.0 % | 96.9 % | 0.0 % | 0.974 |

**Every candidate holds 100 % top-1 and moves no BF16 winner out of first place**, on a bank
that is *not* saturated. KL is ≤0.017 throughout.

**Two orderings that do not match, and neither is resolved at n=13:**

1. **`strict-4.25` (KL 0.0036) beats `q6` (0.0052) and `q4` (0.0069)** — a sub-Q6 allocation
   with *lower* divergence than uniform Q6.
2. **`probe-3.00` (KL 0.0079) beats `aggr-3.50` (KL 0.0168)** — and `probe-3.00` is the arm
   that **FAILS the recurrent-state gate** while `aggr-3.50` passes it.

The second is the one to watch. If it survives more data it means the state gate and
behaviour disagree in ordering, which would be a finding about the *gate*, not the candidate.

**This is 13 positions from one prompt and it settles nothing.** Seven candidates over 13
positions cannot separate 100 % from 92 %, and a single KL mean over 13 samples has a wide
interval. **The full calibration bank (12 prompts, 198 tokens) is running.** No candidate may
be selected, and `holdout-v1` may not be touched, until it completes.

#### 8.2.2p KDA-BEHAV-1 CALIBRATION COMPLETE — 12 prompts, 198 positions

**Not saturated:** BF16 median confidence **0.601** (range 0.207–0.971), 28 % of positions
above 0.9. Agreement numbers are meaningful.

##### Control envelope, read before the candidates

| control | KL mean | KL p95 | top-1 | top-5 | rank moved | worst prompt |
|---|---:|---:|---:|---:|---:|---:|
| q8 (8.50 bpw) | 0.00235 | 0.00974 | 99.3 % | 98.2 % | 0.7 % | 91.7 % |
| q6 (6.56 bpw) | 0.01082 | 0.04491 | 96.5 % | 96.3 % | 3.5 % | 86.7 % |

**Q6 is itself a substantial step down from Q8** — 4.6× the KL, −2.8 pp top-1. That is the
scale against which any candidate's degradation has to be judged.

##### Candidates

| candidate | bpw | KL mean | KL p95 | top-1 | top-5 | rank moved | worst prompt |
|---|---:|---:|---:|---:|---:|---:|---:|
| **strict-4.25** | **4.25** | **0.02011** | **0.06805** | **93.5 %** | **95.1 %** | **6.5 %** | **80.0 %** |
| aggr-3.75 | 3.75 | 0.03362 | 0.20917 | 93.0 % | 93.2 % | 7.0 % | 66.7 % |
| probe-3.00 | 3.00 | 0.03537 | 0.10849 | 93.5 % | 90.7 % | 6.5 % | 75.0 % |
| aggr-3.50 | 3.50 | 0.05989 | 0.29609 | 92.1 % | 90.6 % | 7.9 % | 66.7 % |
| q4 (uniform) | 4.50 | 0.04379 | 0.24916 | 91.8 % | 92.7 % | 8.2 % | 73.3 % |

##### TWO INTERIM CLAIMS OF MINE, FALSIFIED BY THE FULL BANK

1. **"`aggr-3.75` beats uniform Q4_K on every metric" — FALSE on the tail.** At n=6 the
   worst-prompt floor read 91.7 % against q4's 83.3 %; at n=12 it is **66.7 % against 73.3 %**.
   It still wins on every *mean*. The dimension that reversed is precisely the one that means
   were hiding — which is the argument for distributions, made against my own claim.
2. **"`aggr-3.75` sits with Q6" — FALSE.** At full n it carries **3.1× Q6's KL**, −3.5 pp
   top-1, and **−20 pp** on the worst prompt.

##### What survives

> **`strict-4.25` (4.25 bpw) dominates uniform Q4_K (4.50 bpw) on EVERY dimension** — KL
> 0.0201 vs 0.0438, top-1 93.5 % vs 91.8 %, top-5 95.1 % vs 92.7 %, rank moved 6.5 % vs 8.2 %,
> worst prompt 80.0 % vs 73.3 %.

That is the REPRESENT thesis demonstrated on real behaviour: **an asymmetric allocation
strictly better than a uniform one at lower cost.** It is a smaller claim than the interim
data suggested — 4.25 bpw, not 3.75 — and it is the one the full bank supports.

> **No candidate is non-inferior to Q6 on this bank.** `strict-4.25` is the closest and is
> still 1.9× Q6's KL, −3.0 pp top-1, −6.7 pp on the worst prompt. **If the bar is Q6, nothing
> sub-Q6 qualifies.**

##### The tail statistic is one prompt, and that is a weakness

| category | q6 | q4 | strict-4.25 | aggr-3.75 |
|---|---:|---:|---:|---:|
| finance | 86.7 % | 73.3 % | 80.0 % | **66.7 %** |
| science | 100 % | 83.3 % | 83.3 % | 91.7 % |
| legal | 91.7 % | 83.3 % | 83.3 % | 91.7 % |
| (nine others) | ≥90 % | ≥90 % | ≥95 % | ≥90 % |

**"Worst prompt" is n=1** — a single 15-token finance prompt drives every tail figure above,
so 66.7 % is 10 of 15 tokens. The means rest on 198 positions; the tail rests on fifteen.
**Tail safety is the right boundary and this bank is too small to measure it.** A holdout of
20 prompts will not fix that either; measuring a tail needs a bank sized for tails.

##### `probe-3.00` and why admission must precede evidence

`probe-3.00` ties `strict-4.25` on top-1 (93.5 %) and beats `aggr-3.50` on KL — **while
failing the recurrent-state gate** (0.130 against 0.10). Under a purely Pareto treatment it
would sit on the frontier as "cheapest, mixed evidence." It must not. The state gate is an
**admission**, not an evidence dimension to trade against, and this row is the concrete case
that proves the two must stay structurally distinct in `BOUNDARY-CERTIFICATE-1`.

##### Status

**No candidate selected. `holdout-v1` untouched.** The selection rule is not yet written, and
per the frozen procedure it must be derived from the control envelope before the candidate
columns are consulted again — a discipline I have already partly spent by reading candidates
at n=6, and which should be recorded as spent rather than claimed as clean.

#### 8.2.2q H1 ON HOLDOUT — **FAILED**, and the adjudication showed a scale-support defect

**H1 fails.** `strict-4.25` beats uniform Q4_K on top-1 agreement, top-5 overlap and
winner-rank movement, and loses on KL mean and KL p95. The frozen criterion required vector
dominance — no worse on *every* required boundary — so it does not hold. **`holdout-v1` is
spent, and no other candidate may be selected from it.**

`holdout-v1`, 20 prompts, 493 positions, BF16 median confidence 0.590 (not saturated).

| boundary | strict-4.25 | q4 uniform | |
|---|---:|---:|---|
| KL mean | 0.04079 | 0.02778 | **FAIL** |
| KL p95 | 0.17336 | 0.10681 | **FAIL** |
| top-1 agreement | 96.5 % | 96.2 % | pass |
| top-5 overlap | 94.2 % | 93.8 % | pass |
| rank movement | 3.46 % | 3.80 % | pass |

| control | KL mean | KL p95 | top-1 | top-5 | rank moved |
|---|---:|---:|---:|---:|---:|
| q8 | 0.00463 | 0.02278 | 98.4 % | 97.7 % | 1.6 % |
| q6 | 0.03100 | 0.14342 | 97.0 % | 95.7 % | 3.0 % |

##### The separate instrument finding

**The failure does not validate KL mean as a sound summary at this bank size.** The narrow,
defensible claim:

> **A bare prompt-level KL mean is not a stable authority on this 20-prompt bank, because one
> prompt can dominate the aggregate and even reverse the control ordering.**

The controls reversed: **q6 at 6.56 bpw reads a worse KL mean than q4 at 4.5 bpw**, which no
representation story explains. Per-prompt:

> **Prompt 19 — a close-call — contributes 55.6 % of q6's entire KL mean and 41.5 % of
> `strict-4.25`'s, out of twenty prompts.**

| | ordering |
|---|---|
| by **mean** (what H1 was judged on) | q4 0.0278 < q6 0.0310 < strict 0.0408 |
| by **median** | q6 0.0068 < strict 0.0092 < q4 0.0124 |
| dropping prompt 19 alone | q6 0.0145 < strict 0.0251 < q4 0.0262 |

The median ordering matches calibration and matches bit-width. **It is not used to
re-adjudicate.** Switching statistic after seeing the result is exactly the fitting the
freeze exists to prevent; H1 stands failed.

> **H1 failed. The metric that failed it also failed a robustness diagnostic. Neither fact
> cancels the other.**

##### Falsified

- `strict-4.25` does not dominate uniform Q4_K on holdout under the frozen H1 criteria.
- The calibration dominance claim does not generalise.
- `holdout-v1` is spent.

##### Discovered

- **Winner preservation and distribution preservation can disagree materially** — the same
  candidate is better on three boundaries and worse on two.
- **A behavioural boundary cannot be represented by a bare aggregate statistic** without its
  aggregation and support semantics.
- **Control ordering is itself an instrument diagnostic.** A q6-worse-than-q4 reading is not
  a representation result; it is a signal that the aggregate is being driven by an extreme
  observation.
- **The close-call category did its job.** Three of four were genuinely low-confidence
  (median BF16 confidence 0.36 / 0.22 / 0.35, none above 0.9) — the fourth was mislabelled
  (median 0.812, 40 % above 0.9) — and the destabilising outlier came from that category.

##### What survives, stated narrowly

> On 493 held-out positions, `strict-4.25` preserved BF16 winners, top-5 neighbourhoods and
> winner ranks slightly better than uniform Q4_K, while its measured distribution divergence
> was worse under the frozen KL aggregates.

**That is not a promotion result**, and 4.25 is not being rescued. It is another instance of
the finding this programme keeps producing: behavioural evidence is plural and non-scalar.

##### The design input this hands `BOUNDARY-CERTIFICATE-1`

A certificate may not say `KL = 0.04079`. It must carry aggregation and support:

```text
BoundaryEvidence {
    metric              KL
    aggregation         mean_over_prompts | median | quantile(p) | max | per_category
    support             20 prompts / 493 positions
    distribution_summary
    authority
}
```

so a gate can **refuse an aggregation whose support is insufficient for the claim being
made** — exactly analogous to the percentile refusal already in
`represent/measurement.rs`, which was never extended to means.

And one control rule earned here: **if the control ordering contradicts an established
monotonic expectation, mark the measurement diagnostically suspect rather than silently
consuming it.** Not because q6 must beat q4 on every sample, but because a reversal driven by
one extreme prompt belongs in front of the adjudicator.

#### 8.2.3 Pricing MTP from measured routing divergence

The expert reuse curve measured **consecutive routed-expert overlap of 0.90 of 8 — 11.2 %**
on 128 real tokens at layer 3, against a synthetic control at 0.65. That is precisely the
quantity depth-1 speculation faces: the draft token is the accepted token's successor, so
adjacent-token divergence *is* verification cost.

A K=2 verify therefore touches a union of **≈15.1 of 16** routed experts — **1.89× routed
expert traffic** — while the trunk and the always-active shared expert are traversed once.

| operating point | routed share of traffic | verify cost | at 70 % acceptance |
|---|---:|---:|---:|
| experts 2.0 bpw + rest 4-bit | 35.3 % | 1.31× bytes | **≈1.29×** |
| experts 4.25 bpw + rest 4-bit | 51.8 % | 1.46× bytes | **≈1.16×** |

Two things fall out. **MTP is worth ~1.2–1.3×, not 2×.** And its value **rises as experts
get cheaper**, because the routing-divergence penalty applies to a shrinking fraction of the
bytes — low-bit experts and speculation are **complements**, which is not the usual
direction for speculation on an MoE.

**Provenance caveat, and why PHYSICAL-6 re-measures.** The 0.90/8 figure is layer 3, the
first sparse layer, on one 128-token trace. Locality may differ with depth and with corpus.
The acceptance rate of 70 % is an *assumption*, not a measurement — nothing here has run
GLM's MTP head. PHYSICAL-6 measures the union it actually pays for and the acceptance it
actually gets; this table exists to set the expectation, not to satisfy the gate.

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
- Sub-3-bit expert representation (§6.1) is unbuilt and unmeasured. Until it exists, "320 B resident on a MacBook" is a target, not a plan. **It is a capacity requirement, not a throughput one** (§6.3), and going *below* ≈2.0 bpw is research beyond the fit target — not owed by the 20 tok/s programme.
- No kernel is priced. **§6.2 and §6.3 are both rooflines**, and §6.3 is arithmetic over a measured census rather than a measurement of anything.
- **PHYSICAL-0 is frozen and PHYSICAL-2a is run (§8.2.0, §8.2.1); everything else in §8.2 is unrun.** PHYSICAL-2a **falsified v0.2's own mHC estimate by ~6×** — it priced a command-buffer round trip where the unit is a dependent dispatch. Recorded rather than quietly corrected, because the failure mode (right that a roofline is blind to the cost, wrong about its size) is the one this programme is most likely to repeat.
- **The mHC pipeline bubble is still unmeasured.** PHYSICAL-2a's chain is uniform tiny dispatches; a single-threadgroup Sinkhorn reduction between wide kernels is exactly what `dependency_bubble_probe` was built for, and that cost is not in any number here.
- **DSA is the largest remaining unpriced term** (§6.3.2 item 3) and now outranks mHC on the risk list.
- **PHYSICAL-1 is scoped and pre-registered (§8.2.2), not run.** Its bands were **rebased** on discovering Q4_K is 4.5 bpw rather than 4.0 — a "≤10 ms" band would have demanded ~94 % of the envelope and been unmissable-by-construction. Bands are now a realised-bandwidth fraction.
- **The 300 GB/s envelope is itself borrowed.** It is the repo's measured *quantised GEMV* rate; a KDA layer interleaves seven small dependent stages between its projections, and no measurement yet says a recurrent operator reaches the same fraction. PHYSICAL-1's physical gate is the first test of that transfer.
- The MTP multiplier in §8.2.3 assumes 70 % acceptance. **No acceptance rate has been measured**; GLM's MTP head has never been run here.
- The vendor's benchmark claims are vendor claims; nothing here reproduces them.

## 12. Artifacts

Regenerate with §2; nothing is checked in but the tool.

| artifact | how |
|---|---|
| stub checkpoint (10.68 MB) | `scripts/hf_metadata_checkpoint.py zai-org/GLM-5.3-Flash --out stub` |
| `inventory.json` | `larql inspect-hf stub --no-tensor-list --output inventory.json` |
| `plan.json` | `larql vindex3 plan stub --output plan.json` |
