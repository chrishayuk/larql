# GEMMA4-E4B-1: reconnaissance — per-layer input streams and cross-layer K/V in VINDEX3

**Class: RECONNAISSANCE.** Written 2026-11-07 against main `6b62f6d2`, before any
code. It establishes the baseline and the option space only. The decisions,
contracts, tolerances and forecasts are in the [freeze](gemma4-e4b-1.md) and its
machine-readable forecast
[`gemma4-e4b-1-freeze.json`](arch-conformance/forecasts/gemma4-e4b-1-freeze.json).
Where this document and the freeze disagree, **the freeze is the authority**:
§3.4 (schema) and §3.1's "drop the dead K/V weights" were decided the other way
at freeze (v7 for feature-using containers; preserve the weights as retained,
unreachable tensors), and §7's open questions are all answered there.

## Pins and how to read this

- **LARQL code.** Every LARQL citation is at `origin/main` **6b62f6d2**, read with `git show` / `git grep`. The local checkout (`vindex3/qkv-attention-bias`, ff364ba6) is 485 commits behind and was not used for citations.
  - `V` = `crates/larql-vindex/src/format/vindex3`
  - `E` = `$V/opplan/exec`
- **Planner.** The runs used the release binary at `larql-registry-gemma/target/release/larql`, built from b2db1059, four commits off origin/main. It reports planner 0.2.1, `semantics_version` 23 and plan schema 6.
- **Checkpoints.** Headers only were read: the HF API tree, `config.json`, and the safetensors header via HTTP range requests. No weights were downloaded.
  - `google/gemma-4-E4B-it` @ `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`: one `model.safetensors`, 15.99 GB, 2130 tensors.
  - `google/gemma-4-E2B-it` @ `4adf15748de9e1a0097b6274cee9c15836c41c37`: one `model.safetensors`, 10.25 GB, 2011 tensors.
- **Transformers.**
  - Version 5.5.0 (PyPI wheel, global python 3.12), torch 2.14.0.
  - `models/gemma4/modeling_gemma4.py` has sha256 `6a86e033…f935`.
  - `cache_utils.py` has sha256 `7935a009…cf949`.
  - Line numbers below are `mg4:N` for that modeling file and `cu:N` for that cache file.
- **Markers.** Lines marked **[read]** were checked in source. Lines marked **[inferred]** are reasoning that has not been executed.

---

## 0. The central question, answered first

> Can VINDEX3 represent dependencies **between layers** as explicit graph
> relationships, for example layer i's attention consuming K/V produced by
> layer j, or every layer consuming a per-layer slice of a stream computed
> once before the stack? The alternative is Gemma flags or executor special
> cases.

### Today: no. Layers are independent, sequential blocks [read]

Five places in the code assume this:

1. **Per-layer policy is self-contained.** `AttentionLayerPolicy` (`$V/graph/policy.rs:279-338`) holds `operator`, `span`, `window`, `position`, `geometry` and `v_from_k`. Each describes this layer only. No field names another layer.
2. **Operands name tensors, not values.** `OperandRef { object, tensor, dtype, shape }` (`$V/opplan/ops.rs:15-22`) can only point at a stored tensor, never at a value another op produced. `LayerPlan` (`$V/opplan/layer.rs:179-267`) has no input other than the residual.
3. **The only graph edge is cross-component.** `SystemGraph.edges: Vec<HiddenStateEdge>` (`$V/graph/mod.rs:114-115`). `HiddenStateEdge` (`$V/graph/edge.rs:5-27`) is a flow of hidden states from one component's tapped layers into another component, the drafter interface. It is not layer-to-layer inside a component.
4. **The planner says so in words.** `probe_kv_shared_layers` reads: "Every layer in the graph projects its own K/V, so the only declaration the schema agrees with is `0`" (`$V/plan/carriage.rs:428-434`). PLE is "represented as absent" (`$V/plan/carriage/rules_surface.rs:381-390`).
5. **The executors are built the same way.**
   - The KV state is one row-list per absolute layer (`RowKvState`, `$E/kv.rs:389-470`; `prepare` asserts the layer count at `kv.rs:442-457`).
   - An attention step is refused unless its own history view ends exactly at `position` (`AttentionStepCall::new`, `$E/backend/step_calls.rs:108-124`). A layer reading another layer's history, after that layer has appended this position, sees `end = position + 1` and is refused.
   - Token ids are out of scope inside the layer loop. `execute_layer` receives none (`$E/layer_exec.rs:30-47`). Decode consumes the token in `embed_token` before the loop (`$E/decode/run.rs:74-75`).

### Precedents: the cross-layer dependencies VINDEX3 does carry [read]

Two cross-layer relationships exist today. Both are **component-wide topologies chosen from an enum**, not general references:

- `ResidualTopology::{SingleStream, HyperConnection, AttentionResidual{block_size}}` (`crates/larql-models/src/config/residual_topology.rs:91-102`). Kimi K3's attention residuals read earlier layers' snapshots.
- These are lowered to `LayerPlan.hyper_connection` (`layer.rs:243`), the attention-residual site (`layer.rs:256`) and `AttentionResidualExitOp` (`layer.rs:378`).
- Their tensors are placed only when the topology is declared (`$V/graph/build/components.rs:140-177`; role tables gated in `$V/graph/roles/classify.rs:157-176`).

That is the house style:

- a relationship is declared once;
- its tensors are owned and roled conditional on the declaration;
- the closure checks it.

It is a good template for *ownership and closure*. It is a poor template for *generality*, because each topology is a closed enum variant whose cross-layer pattern is implied by a parameter, not stated as edges.

### Three options

| Option | What the graph says | Cost | Verdict |
|---|---|---|---|
| **A. Flag.** `num_kv_shared_layers: 18` and `ple_width: 256` on the surface; the executor derives sources. | A Gemma rule ("last non-shared layer of the same type") becomes executor code. It is the legacy design (`larql-models/src/architectures/gemma4.rs:57-84`). | Smallest | **Rejected.** The derivation becomes an executor authority. No other model can reuse it. Cache ownership stays implicit. |
| **B. Typed cross-layer references** (recommended). Each consuming layer names its producer explicitly. A pre-stack stream is a declared graph value with per-layer slices. Cache ownership and lifetime are *derived from the references*. | Relationships are graph data. The planner's derivation from config is checked against them. Executors read references, not rules. | Medium. Two new typed relationship kinds plus the KV step contract. Roughly 2 × a G4.0-sized rung (§6). | **Recommended.** |
| **C. General dataflow / SSA IR.** Every op output is a named value and every operand is a `ValueRef`. The executor is a scheduler over a DAG. | Fully general. YOCO, CLA, value-residual, PLE and AltUp all fall out. | Large. Every executor (reference, production, device, lowered Metal, distributed, continuation providers) is written as a fixed per-layer block walk (`$E/traverse.rs:276-297`, `$E/decode/run.rs:126ff`, `metal_lowered/*`). This is a rewrite of the opplan and every backend. | **Too expensive now.** B should be shaped so it can become C. |

### Honest answer

A *fully* general construct (C) is too expensive for this rung. The executors are block walkers, and rewriting them for one 4B model is not justified.

A *flag* (A) is the wrong economy: it re-creates the legacy executor-owned rule.

Option B is general along the two axes Gemma 4 needs, and the reference type is shaped as `ValueRef { layer, site }`, so C stays reachable. B adds two typed relationships:

- **`kv_source`**: an attention layer *reads* the K/V history value produced by a named earlier layer's attention site, and produces none of its own.
- **`layer_input_stream`**: a component-level per-token value computed once in the prelude (embedding stage), with each layer *reading its own slice* at a declared injection site.

From these:

- The KV cache and its memory accounting are **derived from the graph**. A producer's retention is the union of its own and its consumers' history ranges. A consumer is `LayerContinuationGeometry::Stateless`, a variant that already exists with nothing producing it (`$E/continuation.rs:159-165`).
- The planner checks each family's *declaration* (`num_kv_shared_layers`, `hidden_size_per_layer_input`) against the references the graph *carries*. Declaration decides the refusal; the graph holds the answer.

### Who else would use these constructs

- **In the repo's corpus**, cross-layer KV sharing and per-layer input embeddings appear **only in Gemma 4 E2B/E4B**:
  - matrix rows `docs/arch-conformance/matrix.json:87-88`, where E2B is the `gemma4-dense` witness and E4B a sibling;
  - ranked list `docs/arch-conformance/README.md:996`;
  - "`num_kv_shared_layers` … remain real work" (`docs/arch-conformance/README.md:295-297`).
  - Of the 117 matrix rows, none other declares these keys (I searched `matrix.json` and `docs/`).
  - The registry (`registry/index.json`) has gemma-3-4b/12b, gemma-4-12b and granite-4.1-3b. None is affected.
- **Gemma 3n** (not in the matrix) is in transformers 5.5 with the same KV-sharing rule (`models/gemma3n/modeling_gemma3n.py:1200-1205, 1232-1248`) and PLE tensors (`:1289-1295`). Its PLE is injected into AltUp streams, a different site, and it also needs AltUp and LAuReL. With B it reuses `kv_source` unchanged and `layer_input_stream` with a different injection site. It still needs AltUp, a residual topology, which is out of scope. **[read for the transformers parts]**
- **Outside the repo** (literature, not verified against code here):
  - **CLA** (cross-layer attention: adjacent layers share K/V) and **MLKV** map directly to `kv_source`.
  - **Hymba**'s cross-layer KV sharing maps directly to `kv_source`.
  - **YOCO**: the cross-decoder layers all read one KV cache produced by a dedicated projection after the self-decoder. The producer is *not* an attention layer's own K/V site, so it needs `ValueRef.site = KvProjection` (a non-attention producer). That is the first step towards C, and the reason the reference carries a `site`.
  - **Value-residual / ResFormer** (layer 1's V mixed into later layers) is a cross-layer `ValueRef` to a V value. That is C territory.
  - **DenseFormer / AttnRes** are already handled as topologies.
  - **[inferred]**: B covers CLA, MLKV, Hymba sharing and Gemma 3n/4 directly. YOCO and value-residual need C-shaped extensions of the same reference type.

---

## 1. What the architecture does (transformers 5.5.0, read)

### 1.1 Shapes from the checkpoints [read, headers]

| | E4B | E2B |
|---|---|---|
| hidden / layers | 2560 / 42 | 1536 / 35 |
| q heads / kv heads | 8 / 2 | 8 / 1 |
| head_dim sliding / full | 256 / 512 | 256 / 512 |
| layer_types | 35 sliding, 7 full at 5,11,17,23,29,35,41 | 28 sliding, 7 full at 4,9,14,19,24,29,34 |
| sliding_window | 512 | 512 |
| `hidden_size_per_layer_input` | 256 | 256 |
| `vocab_size_per_layer_input` | 262144 | 262144 |
| `num_kv_shared_layers` | 18 (layers 24..41) | 20 (layers 15..34) |
| `use_double_wide_mlp` | false | **true** |
| intermediate | 10240 on all layers | 6144 (0..14), **12288 (15..34)** |

Non-layer language-model tensors (BF16):

| tensor | E4B | E2B |
|---|---|---|
| `model.language_model.embed_tokens.weight` | [262144, 2560] | [262144, 1536] |
| `model.language_model.embed_tokens_per_layer.weight` | **[262144, 10752]** = 42×256, 5.64 GB | [262144, 8960] = 35×256, 4.70 GB |
| `model.language_model.per_layer_model_projection.weight` | [10752, 2560], 55 MB | [8960, 1536], 27.5 MB |
| `model.language_model.per_layer_projection_norm.weight` | [256] | [256] |
| `model.language_model.norm.weight` | [2560] | [1536] |

Per layer, every layer has the same 17 tensors. Shapes for E4B (E2B analogous):

- `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `post_per_layer_input_norm`: each [2560]
- `layer_scalar`: [1]
- `mlp.{gate,up}_proj` [10240, 2560]; `mlp.down_proj` [2560, 10240]
- `per_layer_input_gate.weight`: [256, 2560]
- `per_layer_projection.weight`: [2560, 256]
- `self_attn.q_proj`: [2048|4096, 2560]
- `self_attn.k_proj` and `v_proj`: [512|1024, 2560]
- `self_attn.o_proj`: [2560, 2048|4096]
- `self_attn.q_norm` and `k_norm`: [256|512]
- There is no `v_norm` tensor; it is parameter-free (`mg4:1164`).

**KV-shared layers still ship `k_proj`, `v_proj` and `k_norm`.** That is 110 MB on E4B and 38 MB on E2B. In the trained, cached semantics they are never read (§1.3).

### 1.2 Per-layer embeddings (PLE)

**Prelude**, computed once per token before the stack (`mg4:1516-1531, 1566-1569, 1622-1678`):

```
e      = embed_tokens[tok] * bf16(sqrt(H))                 # main embedding (ScaledWordEmbedding; scale cast to weight dtype, mg4:1417)
t      = embed_tokens_per_layer[tok] * sqrt(256)           # [L*256] -> reshape [L,256]; sqrt(256)=16 exact
p      = (e @ per_layer_model_projection.T) * H^-0.5       # [L*256] -> reshape [L,256]; python-float scale
p      = RMSNorm_w(p; per_layer_projection_norm, eps)      # norm over each 256-slice, weight [256] shared by all L slices
ple    = (p + t) * 2^-0.5                                  # [L, 256]
```

**Injection** in layer i, after the FFN residual add and before `layer_scalar` (`mg4:1393-1402`):

```
r = h
g = gelu_tanh(h @ per_layer_input_gate.T)                  # [256]
g = g * ple[i]                                             # elementwise with this layer's slice
g = g @ per_layer_projection.T                             # [H]
g = RMSNorm_w(g; post_per_layer_input_norm)                # weighted norm (Gemma4RMSNorm multiplies by w, no 1+w; mg4:157-175)
h = r + g
h = h * layer_scalar                                       # whole residual, after the PLE add
```

Facts the design must carry:

- The PLE stream depends on the **token id**, not only on the residual. `get_per_layer_inputs` reverse-maps `inputs_embeds` to ids when only embeddings are given (`mg4:1631-1648`).
- In the **multimodal** model, soft-token positions use `pad_token_id` for the table lookup, and the *pad embedding* rather than the soft-token embedding for the projection (`mg4:2185-2195`). So the projection input is "the pre-merge text embedding", which differs from the residual input at image and audio positions. Text-only generation is unaffected. This matters for the `image_conditioned` capability.
- The norm on `p` runs over each 256-wide slice and uses one shared [256] weight.
- The projection scale `H^-0.5` and the `2^-0.5` combine scale are python floats, not bf16-cast. Only the table's `sqrt(256)` goes through the dtype cast, and it is exact.

### 1.3 KV sharing (`num_kv_shared_layers`)

Rule (`mg4:1147-1159`):

- `first = L - num_kv_shared_layers`.
- Layer i ≥ first (and first > 0) is *shared*. Its source is the **last layer before `first` with the same `layer_type`**.
- That source layer has `store_full_length_kv = True`.

Resulting maps [computed from config]:

- E4B: sliding shared layers {24-28, 30-34, 36-40} read **layer 22**; full shared layers {29, 35, 41} read **layer 23**.
- E2B: sliding shared layers read **13**; full shared layers {19, 24, 29, 34} read **14**.
- This matches the legacy test assertions (`crates/larql-models/src/detect/tests/gemma4.rs:284-289`).

Forward (`mg4:1179-1241`):

- A shared layer computes its own Q: `q_proj`, `q_norm`, RoPE with its own layer-type rotary.
- It takes `(K, V) = past_key_values.shared_layers[src]`. These are the source's post-`k_norm`, post-RoPE K and post-`v_norm` V, **after** the source's cache update. So it is the source's full retained history including the current chunk.
- It does not write a cache. Its attention uses its own mask: sliding window 512 for sliding layers, causal for full.
- The cache allocates layers only for `layer_types[:-num_kv_shared_layers]` (`cu:1218-1220, 1331-1332`). E4B's cache therefore has 24 layers, not 42.

Invariants that make the sharing well-defined. They hold for Gemma by construction ("same layer type"), and a general IR must **check** them rather than assume them:

1. Consumer and producer have equal `num_kv_heads` and K/V `head_dim`.
2. Consumer and producer have an identical position policy (K is cached post-RoPE).
3. The consumer's required history is a subset of the producer's retention (sliding 512 ⊆ sliding 512).
4. The producer runs before the consumer in the same step.

**Semantic hazard [read].** The shared path is taken only `if self.is_kv_shared_layer and past_key_values is not None` (`mg4:1198`). With `use_cache=False`, shared layers silently compute K/V from their own shipped `k_proj`/`v_proj`, which is a different function.

- `Gemma4TextModel.forward` creates a `DynamicCache` by default when `config.use_cache` is true (`mg4:1558, 1571-1572`), so a plain `model(input_ids)` takes the shared path.
- The parity harness must pin `use_cache=True` and record it.
- I believe the shared path is the trained semantics, because it is the one generation uses and the one the legacy engine implements. That the shared layers' own K/V weights are untrained is **inferred, not verified**. Checking it needs weights; see open question Q6.

### 1.4 Double-wide MLP (E2B)

`intermediate = config.intermediate_size * 2` on layers that are KV-shared, when `use_double_wide_mlp` is set (`mg4:1016-1033`).

The shapes confirm it: E2B layers 15..34 have 12288, and layers 0..14 have 6144. It is purely a per-layer width with no new operator.

---

## 2. How VINDEX3 represents things today

### 2.1 Planner findings for E4B and E2B (full runs)

`larql vindex3 plan hf://google/gemma-4-{E4B,E2B}-it` was run both with and without `--capability text-generation`. The findings are identical; the capability table is in both.

Totals:

- **E4B**: 127 representable, 3 mismatched, 15 unrepresented; 18 blocking.
- **E2B**: 126 representable, 4 mismatched, 15 unrepresented; 19 blocking.

Blocking findings by capability (`text_generation` requires 64 findings; `drafting` has the same blockers but is "available: false"):

| capability | E4B blockers | E2B blockers |
|---|---|---|
| text_generation | 67, 78, 140 | 67, 78, 91, 140 |
| image_conditioned | + 131 | + 131 |
| audio_conditioned | + 14-18, 20, 22, 23, 36, 38, 40-42, 144 | same |

The findings:

- **67** `text_config.hidden_size_per_layer_input` (256): mismatched. "Graph represents PLE as absent."
- **78** `text_config.num_kv_shared_layers` (18 / 20): mismatched. "No KV-sharing relationship exists."
- **91** (E2B) `text_config.use_double_wide_mlp` (true): mismatched. "Only `false` is representable."
- **140** `model.language_model.per_layer_model_projection`: unrepresented. "No placement rule owns this group."
- **131** `vision_config.use_clipped_linears` (true): mismatched, outside text generation. It is the `Gemma4ClippableLinear` clamp in the vision and audio towers (`mg4:128-155`).
- **Audio** (`gemma4_audio` tower), outside text generation:
  - Unparsed keys: `attention_chunk_size`, `attention_context_left/right`, `attention_invalid_logits_value`, `attention_logit_cap`, `conv_kernel_size`, `gradient_clipping`, `output_proj_dims`, `residual_weight`, `subsampling_conv_channels`.
  - Probes with no built component: `hidden_act`, `rms_norm_eps`, `use_clipped_linears`.
  - Finding 144: execution surface missing `intermediate_size`.

**Silent absorptions not reported as findings [read in plan JSON + code]:**

- `embed_tokens_per_layer` is placed **inside `target.embedding`** because `classify_group` does substring matching, and the embedding patterns include `embed_tokens` (`$V/graph/build.rs:117-132, 325-341`).
- `per_layer_projection_norm` is placed **inside `target.final_norm`** because the norm patterns include `norm` (`build.rs:154`).
- `per_layer_model_projection` matches nothing ("per_layer" ≠ "layers"), so it is unowned (`$V/graph/build/components.rs:189-191`).
- Objects are merged by `{component}.{kind}` (`$V/graph/build/tables.rs:165-190`), so `target.embedding` and `target.final_norm` each get two source bindings.
- The op-plan `single(...)` would then refuse with "expected exactly 1 tensor, found 2" (`$V/opplan/build.rs:657-688`). **[inferred; the plan stops earlier]**
- `canonical_representation` matches with `starts_with(prefix)` and no `.` boundary (`tables.rs:179-181`). **[inferred]** It would also match `embed_tokens_per_layer.*` when looking for `embed_tokens`.
- The per-layer tensors `per_layer_input_gate`, `per_layer_projection` and `post_per_layer_input_norm` land in `target.decoder_stack`, but they have no `OperandRole` (`$V/graph/roles/tables.rs`, `roles/classify.rs:189-225`). Each would be `ClosureDefect::UnclassifiedOperand` (`$V/opplan/build.rs:605-608`). `layer_scalar` is roled (`roles/tables.rs:158`).
- This is the "byte-complete ≠ role-complete" failure. Today the carriage blockers hide it. Once they clear, it must be fixed explicitly and not left to the closure.

### 2.2 Graph schema [read]

- `GRAPH_SCHEMA = 6` is at `$V/graph/mod.rs:103`, with the bump history in the doc comment at lines 60-102.
- The reader accepts exactly 6 (`check_graph_schema`, `$V/inspect.rs:85-102`) and says "re-encode … cannot be upgraded in place".
- 5→6 was one intentional break with a corpus re-encode (`crates/larql-vindex/docs/vindex3-format-spec.md:1158-1174`).
- Additive fields within v6 use serde defaults:
  - `context_length`, `intermediate_size: Option`, `intermediate_size_by_layer` (`$V/graph/surface.rs:131-144`)
  - `residual_topology` (`surface.rs:473-480`)
  - `v_from_k` (`policy.rs:322-323`)
- The graph types do **not** use `deny_unknown_fields`. Only `represent/*` does. So **an older reader silently ignores a new field.**
- Core types:
  - `SystemGraph { schema, components, objects, edges }` (`$V/graph/mod.rs:106-116`).
  - `Component { id, role, source_artifact, num_layers, hidden_size, attention: Option<Vec<AttentionLayerPolicy>>, execution: Option<ExecutionSurface>, perception }` (`$V/graph/component.rs:106-136`).
  - `ExecutionSurface` (`surface.rs:359-475`) contains `attention: Option<AttentionSurface>`, `ffn: Option<FfnSurface>`, `norm`, `head: Option<HeadSurface{vocab_size, embedding_norm, embed_scale, …}>`, `residual_topology` and several mixer-family surfaces.
- E4B's planned surface reads `"head": {"embed_scale": 50.596443, "final_logit_softcapping": 30.0, "head_reuses_embedding": true}` and `"residual_topology": {"kind": "single_stream"}` (plan JSON).

**Already present and directly reusable:** `FfnSurface.intermediate_size_by_layer: Option<Vec<usize>>`. Its doc says the "planner checks every layer's tensors against it and states the width on that layer's `FfnOp`" (`surface.rs:137-144`). **Double-wide MLP needs no new IR**, only a fate rule and a builder fill (§3.3).

### 2.3 Lowered op plan [read]

- `construct_plan` (`$V/opplan/build/construct.rs:55`):
  - The Softmax arm is at `:619-687`. `head_dim` comes from per-layer geometry (`:622`), window and position from the policy (`:640-641`). `v` binds `AttnK` under `v_from_k` (`:648-656`).
  - FFN choice is at `:302-425`. `layer_scale` comes from the `LayerScalar` role (`:427-429`).
  - `EmbeddingOp { table, norm, scale, vocab_size }` is at `:721-726` (type at `layer.rs:271-280`). `OutputOp` is at `:730-737`.
- `AttentionOp` (`ops.rs:62-113`) has `q/k/v: OperandRef` and `v_from_k`. There is no source field.
- The Gemma 4 judgments come from the arch (`larql-models/src/architectures/gemma4.rs`):
  - partial "proportional" RoPE on full layers (`:200-212`)
  - per-layer head_dim and kv heads (`:226-242`)
  - `v_shares_k` (`:251-256`)
  - `embed_scale = sqrt(H)` (`:433-435`)

### 2.4 Executors and KV ownership [read]

- **Reference f32 interpreter.** `ReferenceBackend` (`$E/reference.rs:1-31`, identity `reference/v1`) is naive f32 and shares nothing with larql-compute.
  - Ops are in `$E/reference/ops.rs`: QKV `:101-152` (parameter-free V norm `:138-147`), RoPE / PartialRope `:153-270`, attention `:313-345` (window start from `history().required_start`, `:331`).
  - Embedding is `row * scale` (`$E/reference/plan_backend.rs:240-246`).
- **Traversal.** Batch and prefill go through `traverse` (`$E/traverse.rs:24`; embed `:215-241`; per-layer `execute_layer`, `:276-297`).
  - `execute_layer` (`$E/layer_exec.rs:30`) runs attention at `:329-371` and the FFN at `:453-534`, then `layer_scale` at `:535-558`.
  - **The PLE injection point is between `:534` and `:535`.** The decode twin is `$E/decode/run.rs:520-547`.
- **KV.** The reference and production backends hold no K/V themselves. Rows go to a caller-owned `KvState` / `ContinuationProvider` (`$E/kv.rs:172-278`).
  - The default is `RowKvState`, one `Vec` per absolute layer (`kv.rs:389-470`).
  - Retention is `HistoryRange::of_span` (`kv.rs:62-70`).
  - Per-layer continuation geometry comes from `plan_continuation_geometry` (`$E/continuation.rs:252-263`). The variants are `Kv`, `LatentKv`, `Recurrent`, `KvAndRecurrent` and **`Stateless`** ("No judged operator produces this yet", `continuation.rs:159-165`).
  - The providers live in `crates/larql-kv/src/vindex3/`: `CanonicalKvState`, `WindowKvState` (keeps `w` rows on sliding layers), and the codec states.
  - KV bytes come from `LayerContinuationGeometry::elements_at/bytes` (`continuation.rs:62-110, 174-180`). The residency budgets (`$E/accounting/budgets.rs:15-60`) cover weights only.
- **Production CPU.** `ProductionBackend` (`$E/production.rs`) uses the same traversal and `DecodeSession`.
  - `larql run` builds the continuation provider and decodes (`crates/larql-cli/src/commands/primary/run_cmd_vindex3/mod.rs:529-538`).
  - `--metal` on `run` selects `DevicePlanBackend`, which runs matvecs on the GPU and keeps K/V in the host provider (`$E/device.rs:12-21`).
- **Lowered Metal** (`$E/metal_lowered/mod.rs:127-200`).
  - Every layer has its own device `k_cache`/`v_cache`, allocated at `max_positions` even on sliding layers (`construct.rs:148-149, 237-242`).
  - It **refuses `layer_scale` on a non-hybrid FFN** (`construct.rs:90-100`). E4B is dense, so this refusal applies to E4B too.
  - The encode-ahead design assumes only the embedding row depends on the token (`metal_lowered/step.rs:6-13`).
- **Token ids.**
  - Decode entries are `Entry::{Token, Hidden, Single}` (`$E/decode.rs:112-121`).
  - Embedding-only entries have no id: `step_embedding` (`decode/steps.rs:153-167`), `InputPosition::Embedding` (`crates/larql-inference/src/vindex3/input.rs:15-18, 50-57`), and image soft tokens (`run_cmd_vindex3/inputs.rs:221-238`).
  - Lowered `verify(tokens)` embeds on the host (`metal_lowered/verify.rs:95-124`).

### 2.5 Legacy engine (migration target, not authority) [read]

Both features already run in the legacy (non-VINDEX3) engine:

- **KV sources.** Sources are precomputed per layer (`larql-models/src/architectures/gemma4.rs:57-84`; ADR `crates/larql-models/docs/adr/005-gemma4-precomputed-layers.md`).
  - CPU passes `shared_kv` from a `HashMap<layer, SharedKV>` (`larql-compute/src/forward/predict/raw.rs:189-209`) and skips K/V projection (`larql-compute/src/attention/block.rs:180, 212, 245`).
  - Metal attends against `attend_cache_idx = src` (`larql-compute-metal/src/decode/encode_attn/context.rs:95-115`).
- **PLE.**
  - `larql-compute/src/forward/ple.rs:11-170` matches §1.2 exactly.
  - It is Metal opt-in via `LARQL_METAL_PLE`, otherwise on the CPU (`larql-inference/src/layer_graph/generate/gpu/mod.rs:247-257`).
  - It precomputes per token (`:328-345`).
  - A parity test exists: `crates/larql-kv/tests/engine_ple_parity.rs`.
- **Legacy format.** PLE is stored as `ple_weights.bin` at f16. The KV-shared layers' K/V weights still ship and are skipped (`crates/larql-vindex/docs/format-spec.md:76-78`).

Use the legacy code as a second oracle and as a source of kernels. Do not port its *rule* (the source derivation) into an executor.

---

## 3. Proposal: minimal general IR (Option B)

### 3.1 (b) Cross-layer K/V source: `kv_source`

**Graph** (`AttentionLayerPolicy`, additive):

```rust
/// Where this layer's attention reads K/V history from.
/// Absent = Own (projects, normalises, rotates and retains its own K/V).
#[serde(default, skip_serializing_if = "KvSource::is_own")]
pub kv_source: KvSource,

pub enum KvSource {
    Own,
    /// Read the K/V history value produced by another layer's attention
    /// site in the same component; project no K/V of its own.
    Layer { producer: ValueRef },
}
pub struct ValueRef { pub layer: usize, pub site: ValueSite }   // site = AttentionKv today
```

The `site` field is there so a YOCO-style dedicated KV-projection producer, or a C-style value, can be added later without a new reference type.

**Graph validation** (closure at encode, `$V/opplan/build.rs` family). For every consumer:

1. `producer.layer < consumer` and the producer is `Own`. No chains: a consumer reading a consumer is refused. Chains could be resolved by following them, but refusing is simpler and covers every known model.
2. Equal `HeadGeometry { head_dim, num_kv_heads }`.
3. Identical `PositionPolicy` (K is stored post-RoPE).
4. Equal K-side norm semantics: the producer's `k_norm` and parameter-free V norm are applied once, by the producer.
5. `HistoryRange(consumer) ⊆ retention(producer)`.
6. Same `operator` (Softmax). MLA and latent KV consumers are refused until a model needs them.

**Operands.** A consumer layer has **no `AttnK`/`AttnV` operand**.

- The shipped dead `k_proj`/`v_proj`/`k_norm` (110 MB on E4B) get an explicit **fate**: placed into no object, reported as `dead_under_kv_source` and not encoded. This follows "every declaration has a fate".
- It also gives a fail-closed property. An older v6 reader that ignores `kv_source` finds no K operand on those layers and fails closure, instead of silently computing the `use_cache=False` function. **[inferred; must be proven by a test running the previous release on a new container]**

**Planner.**

- The family parser (Gemma 4 arch, already has `kv_shared_source_layer`, `gemma4.rs:258`) *derives* the per-layer sources. The graph builder writes them as `kv_source` (`$V/graph/build/tables.rs:64-81`, where the policy is filled).
- The carriage rule for `num_kv_shared_layers` (`rules_surface.rs:336-349`) changes its probe from constant 0 to "count of layers whose `kv_source` is `Layer`". It reaches `Lowered` at `Component.attention[].kv_source → AttentionOp.kv_source`.
- **A second, independent check:** a probe that recomputes the HF rule ("last non-shared layer of the same type") from `layer_types` and compares every consumer's producer with it. Then the declaration's *count* and the graph's *sources* both have an authority.
- Semantics bump 23→24 (verdict flips; `$V/plan/report.rs:438`; pinned test `plan/tests/identity.rs:104-105`).

**Op plan.**

- `AttentionOp` gains `kv: KvBinding::{Own{k,v}, Read{producer_layer}}`, replacing the `k`/`v` operands on consumers.
- `plan_continuation_geometry` returns **`Stateless`** for consumers.
- The producer's `LayerKvGeometry.history` becomes the union of its own history and its consumers'. For Gemma it is unchanged, because the spans are equal.

**KV cache and memory** follow from the graph:

- Providers allocate only producers. E4B: 24 cache layers (20 sliding at 512 rows, 4 full). E2B: 15 cache layers.
- `elements_at` sums only `Kv` layers, so the accounting is automatically correct.

Expected KV, BF16 [computed]:

| context | E4B shared | E4B unshared | E2B shared | E2B unshared |
|---|---|---|---|---|
| 4K | 84 MiB | 147 MiB | 30 MiB | 70 MiB |
| 32K | 532 MiB | 931 MiB | 198 MiB | 462 MiB |
| 128K | 2068 MiB | 3619 MiB | 774 MiB | 1806 MiB |

These numbers become an acceptance assertion.

**Step contract.** `AttentionStepCall::new` (`step_calls.rs:108-124`) requires `rows.end() == position` and passes the fresh row separately. For a consumer, the "fresh row" for position p is the producer's row p, already appended this step. The contract change:

- a consumer step receives `KvView` over the producer's `[start, p)`, plus the producer's row p as its fresh K/V;
- it is constructed by a new `AttentionStepCall::reading(producer_view, position)`;
- that constructor asserts `producer_view.end() == position + 1` and splits off the last row.

Then the attention kernel itself is unchanged, and the "view ends at position" invariant still holds for the kernel.

Other effects:

- **Batch prefill**: the producer appends the whole chunk before the consumer runs, so the consumer reads `[start, chunk_end)` with a causal or window mask.
- **Rollback and verify-N**: truncation touches producers only. Consumers have nothing to truncate.

**Executors.**

- **Reference and production**: `layer_exec.rs:335-358` (prefill) and `decode/run.rs:329-399` (decode) branch on `KvBinding`. A consumer computes Q only.
- **Lowered Metal**: consumers allocate no buffers and bind the producer's `k_cache`/`v_cache` with the producer's length (cf. legacy `context.rs:95-115`). The `max_positions` per-layer allocation shrinks to 24 layers.
- **Distributed / layer-sharded** (V3-FFN-SLICE, layer shards): a consumer on a different shard than its producer is a **cross-shard dependency**. The partitioner must either co-locate them or refuse. The reference makes this visible, where a flag would hide it.

### 3.2 (a) Per-layer auxiliary input stream: `layer_input_stream`

**Graph** (`Component`, additive, beside `execution`):

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub layer_input_stream: Option<LayerInputStream>,

pub struct LayerInputStream {
    pub slice_width: usize,                 // 256
    pub layers: usize,                      // must equal num_layers (one slice per layer)
    pub sources: Vec<StreamSource>,         // summed, then scaled
    pub combine_scale: f32,                 // 2^-0.5
    pub injection: StreamInjection,
}
pub enum StreamSource {
    /// Row of a lookup table indexed by the position's TEXT token id.
    TokenLookup { object: String, scale: f32, non_text_position: NonTextId },  // scale 16
    /// Projection of the main embedding output, normalised per slice.
    EmbeddingProjection { object: String, scale: f32, norm: NormSpec /*per-slice, weighted*/,
                          input: ProjectionInput },  // scale H^-0.5
}
pub enum NonTextId { PadToken(u32) }                  // Gemma: pad_token_id at soft-token positions
pub enum ProjectionInput { TextEmbedding /* pre-merge; pad embedding at soft-token positions */ }
pub struct StreamInjection {
    pub site: InjectionSite,                // AfterFfnResidual (before layer_scale)
    pub form: InjectionForm,                // GatedProjection { activation, post_norm: NormSpec }
}
```

The per-layer tensors are roled `LayerInputGate`, `LayerInputProjection` and `LayerInputPostNorm`, and only when the stream is declared. This is the topology-gated pattern of `classify_stack_tensor_under` (`roles/classify.rs:157-176`); otherwise they are an `OperandImpliesAbsentOp` refusal (`opplan/build.rs:598-604`).

Gemma 3n would reuse this with `site: AltUpCorrected` (a new enum arm) once AltUp exists.

**Placement.** Three explicit owning patterns go **ahead of** `embed_tokens` and `norm` in `GROUP_PATTERNS`. This is the same fix as the AttnRes exit (`build.rs:137-153`) and `embed_vision` (`build.rs:91-114`).

- `embed_tokens_per_layer` → `ObjectKind::LayerInputTable`
- `per_layer_model_projection` → `LayerInputProjection`
- `per_layer_projection_norm` → `LayerInputNorm`

They are placed only when the declaration is present. Also add a `.` boundary to `canonical_representation`'s prefix match (`tables.rs:179-181`).

**Planner.**

- The `hidden_size_per_layer_input` rule probes `layer_input_stream.slice_width` (0 when absent) and reaches Lowered.
- `vocab_size_per_layer_input` (`semantics.rs:436-439`) becomes a tensor-semantic check against the table's rows.
- Finding 140 clears through placement.

**Op plan.**

- `EmbeddingOp` gains `layer_inputs: Option<LayerInputsOp>`, so the stream is computed **in the prelude**, at embed time.
- `LayerPlan` gains `layer_input: Option<LayerInputInjectOp { gate, proj, post_norm, activation, slice: usize }>`.

Computing the stream in the prelude keeps the lowered Metal encode-ahead invariant ("only the prelude depends on the token", `metal_lowered/step.rs:6-13`) true with a widened prelude. The prelude cost per token is:

- a table row gather of 10752 BF16 (21.5 KB);
- one [10752 × 2560] matvec (E4B);
- 42 slice norms.

**Token-id plumbing (the real cost of PLE).**

- The prelude needs the token id at every position, so `traverse` and `DecodeSession` thread a per-position `InputId` (text id, or `NonText`) alongside the residual.
- `Entry::Hidden` and `step_embedding` / `InputPosition::Embedding` entries carry `NonText`. If the stream declares `non_text_position: PadToken`, they resolve to the pad id and the pad embedding. If a stream exists and **no rule is declared**, an id-less entry is **refused**, not zero-filled.
- This also removes HF's "reverse-map embeddings to ids" path (`mg4:1631-1648`), which should not be copied.

**Residency.**

- The table is 5.64 GB, about 37% of E4B's language-model bytes, and is read one row per token. It needs a lookup-only residency class: host-mmapped, never device-resident, page-faulted per row.
- The weight budgets (`$E/accounting/budgets.rs`) need that class, or E4B will appear to need 15 GB of device residency.
- Its encoding (BF16 or Q8 rows) is a REPRESENT question for later. Canonical BF16 comes first.

### 3.3 Double-wide MLP (E2B)

There is no new IR. In the builder:

- fill `FfnSurface.intermediate_size_by_layer` from the gate/up/down shapes, as the static-shard path already does (`surface.rs:137-144`);
- change the `use_double_wide_mlp` probe (`rules_surface.rs:391-398`) from constant `false` to "true iff every `kv_source = Layer` layer has width 2 × `intermediate_size` and every other layer has 1 ×".

The second independent authority is the tensor shapes, which the planner already checks against `intermediate_size_by_layer`.

This depends on §3.1's consumer set, so it lands with or after `kv_source`.

### 3.4 Schema: v6 additive or v7

| | Additive within v6 (house precedent) | Bump to v7 |
|---|---|---|
| Old reader on a new E4B container | Ignores `kv_source` and `layer_input_stream` (no `deny_unknown_fields`). It is **fail-closed only by side effect**: consumers lack K/V operands, and the PLE operands are unroled → closure defect. **[inferred; needs a test]** | Refuses by `check_graph_schema` (`inspect.rs:92`), by name. |
| Existing containers (registry: gemma-3-4b/12b, gemma-4-12b, granite-4.1-3b) | Untouched | **Re-encode the whole corpus and the registry** |
| Risk | A future model whose consumers *do* ship the K/V operands we keep, or a reader that tolerates extra operands, breaks the side-effect guarantee | Cost only |

**Recommendation:**

- Additive within v6, *plus* a positive guarantee: a `required_features: Vec<String>` list on `SystemGraph`, and a reader that refuses any feature it does not know. Old readers ignore that list too, which is why the dead-operand omission is still required.
- Add an acceptance gate that runs the **previous release binary** against a new E4B container and requires a named refusal.
- If you prefer the clean guarantee, bump to v7 and fold any other pending breaks into it. **This is open question Q1.**

---

## 4. Verification plan

### Instrument conditions (recorded with every result)

- **Recorded fields**:
  - checkpoint repo and revision;
  - LARQL commit and binary build profile (release);
  - transformers version plus the sha256 of `modeling_gemma4.py` and `cache_utils.py`;
  - torch version;
  - HF dtype (**float32**, loaded via the text config; see the HF composite-dtype memory: assert the loaded parameter dtype, do not trust the `dtype=` argument);
  - `attn_implementation="eager"`;
  - `use_cache=True` (asserted);
  - device (CPU), chip and OS;
  - LARQL backend identity (`reference/v1`, production, device, lowered);
  - prompt bank hash.
- **Tools**:
  - `scripts/dump_layers_hf.py` compared with `vindex3 exec --dump-layers` (`vindex3_cmd/args.rs:199`); this is the Gemma 4 26B precedent, `ROADMAP.md:2192-2277`;
  - legacy `engine_ple_parity` as a second oracle.

### Gates and tolerances

Proposed; freeze them in the freeze rung.

**V1. Reference f32 vs HF f32, identical token ids.**

- Prompts: ≥16, including chat-templated ones.
- Per layer and position, residual after `layer_scalar`: cosine ≥ 0.999999 and max|Δ| / max|h| ≤ 1e-5.
- Final logits after softcap: max|Δ| ≤ 1e-3; top-1 identical at every position; top-5 set identical.
- Precedent: 26B at cos 1.0 and byte-identical ids.

**V2. Sequence lengths across the sliding boundary.**

- Prefill lengths: 1, 7, 511, 512, 513, 1023, 1024, 1025, 2048 and 4096.
- Decode continuations of 64 tokens from prefills of 500 (crosses 512) and 1020 (crosses 1024).
- Every length passes V1, with greedy ids byte-identical to HF `generate(do_sample=False, use_cache=True)`.

**V3. Prefill vs cached decode (sharing exercised).**

- For each step of a 64-token decode, LARQL cached-decode logits must equal a LARQL full re-forward at that length: max|Δ| ≤ 1e-5 in f32 reference.
- They must also equal HF cached decode within V1.
- Run with prefill chunk sizes 1, 7 and the whole prompt, so a consumer reads the producer's partial chunk.

**V4. Execution witnesses (intermediate taps).**

- **K/V tap.** At consumer layer c and position p, the K/V the attention kernel reads must be byte-identical to producer row p. Producer and consumer cache buffers must be **the same allocation** (pointer/identity check on lowered Metal).
- **Negative control.** A deliberately wrong arm with consumers computing own K/V (HF `use_cache=False`) must **fail** V1 against HF `use_cache=True`. If it passes, the instrument cannot see sharing.
- **PLE control.** Zeroing the stream must fail V1, and the per-layer PLE tap must match HF `per_layer_inputs[:, :, i]` at cos ≥ 0.999999.
- **Pad-id control** (image-conditioned capability only, later rung): at soft-token positions the PLE input equals HF's.

**V5. Production parity.**

- CPU production vs reference: layer-diff ≤ 1e-5 rel, ids identical.
- Lowered Metal vs reference: ≤ 1e-6 layer-diff (the G4.3 instrument, `ROADMAP.md:2276-2277`).
- Device backend: identical ids over V2.
- Quantised encodings (Q4_K and so on) are a separate rung with distribution gates. Pair them per memory "deterministic ≠ significant".

**V6. Memory accounting.**

- The provider's reported KV bytes equal the §3.1 table at 4K and 32K exactly. That is 24 (E4B) or 15 (E2B) allocated cache layers.
- Lowered Metal device KV allocation equals the provider bytes at `max_positions`.
- Device-weight residency excludes the PLE table.
- Peak RSS is timed on the binary, not cargo.

**V7. Planner.**

- E4B and E2B `text_generation` admissible.
- A re-plan of the 117-row corpus moves **only** the E4B and E2B rows (forecast written before the code; arch-conformance discipline).
- Old-reader refusal test (§3.4).
- A synthetic config with `num_kv_shared_layers` set but a mismatched graph source must **block**.

---

## 5. Rungs

| Rung | Content | Gate | Size (est.) |
|---|---|---|---|
| **R0 Recon** | This document | Review | done |
| **R1 Freeze** | Decide Q1-Q7. Freeze IR names, tolerances, prompt bank and forecast JSON (`docs/arch-conformance/forecasts/`): exactly which findings flip per rung and which rows move. | Forecast committed before any code | docs + JSON, ~0.5 day |
| **R2 Placement and roles** | Owning patterns for the three PLE groups (declaration-gated); `.`-boundary prefix fix; roles for per-layer PLE tensors; dead-weight fate for consumer K/V | Finding 140 clears; no silent absorption (plan JSON shows three distinct objects); corpus re-plan moves nothing else | ~10-15 files, +600-900 lines |
| **R3 `kv_source` IR + reference** | Policy field; builder fill from the arch derivation; carriage probe plus independent rule probe; `KvBinding`; `Stateless` consumers; producer retention union; step-contract variant; reference executor; miniature fixture (e2b-like, sharing on) | Finding 78 flips (semantics 24); miniature reference ≡ production; V4 K/V tap on the fixture; old-reader refusal test | ~35-50 files, +2-3k (G4.0 b2d41f27 was 46 files, +2.4k) |
| **R4 `layer_input_stream` IR + reference** | Component field; prelude op; injection op; token-id threading through `traverse`/`DecodeSession`; NonText refusal; lookup-only residency class | Finding 67 flips; miniature PLE ≡ legacy `ple.rs` on the same weights; id-less entry refused | ~40-55 files, +2.5-3.5k (the token-id plumbing is the bulk) |
| **R5 Double-wide** | `intermediate_size_by_layer` fill and probe | E2B finding 91 flips; E2B text_generation admissible | ~5-8 files, +200-400 |
| **R6 Encode + HF evidence (E4B, then E2B)** | Canonical BF16 encode; V1-V4 on the reference backend | V1-V4 pass; evidence record committed | ~1-2 days compute plus a results doc |
| **R7 Production CPU and continuation providers** | `CanonicalKvState`/`WindowKvState`/codec providers honour Stateless consumers; verify-N and rollback; V6 | V3, V5 CPU, V6 | ~15-25 files, +1-1.5k |
| **R8 Metal** | Device backend; lowered: consumer buffer binding, PLE prelude, **remove the dense-FFN `layer_scale` refusal** (`construct.rs:90-100`) | V5 lowered ≤ 1e-6; V6 device allocation | ~15-25 files, +1-2k |
| **R9 Registry and multimodal** | Registry entries; image-conditioned with pad-id PLE (finding 131 is separate: clipped linears); audio tower is its own programme (13 findings) | V4 pad-id control; registry attestation | separate |

R3 and R4 are independent and can run in parallel after R2. R5 depends on R3.

---

## 6. Risks and unknowns (what could force a different design)

1. **Transformers is not the Google reference.** The HF shared path depends on `past_key_values is not None`. If Google's reference (gemma.cpp, LiteRT or JAX) differs, for example by sharing pre-RoPE K or using a different source rule, then HF is not the oracle. Mitigation: R1 reads one non-HF reference. The legacy engine agrees with HF, but it was written from HF.
2. **Dead or live shared-layer weights (Q6).** If the shipped consumer `k_proj` weights turned out to be *live* in some deployment path, dropping them would lose information. Recommendation: still drop them, because the graph declares the cached function. Checking needs a weight read in R6.
3. **Step-contract change.** `AttentionStepCall` is shared by every backend and provider, including codec providers that compress history. A consumer reading a codec-compressed producer reads compressed history, which is correct but changes CODEC experiments' meaning. Codec providers must declare whether they support a shared producer.
4. **Token-id threading** touches decode, verify-N, distributed and multimodal entry points. Each of those may have code that assumes a "residual only" input (`metal_lowered/step.rs:6-13`). This is the most likely place for the estimate to grow.
5. **PLE table residency.** The 5.6 GB table on host mmap under a quiet-gate / peer-measurement regime is fine. On constrained devices the per-row page faults could dominate decode latency. The legacy engine's "4205 ms/tok" E2B pathology came from PLE running on the CPU (`crates/larql-compute/CHANGELOG.md:219`). The lowered prelude must be on the device or measured first.
6. **Schema choice (§3.4).** If fail-closed-by-omission cannot be proven for old readers, v7 is forced, and with it a corpus re-encode.
7. **Cross-shard dependencies** for distributed execution (§3.1). If layer-sharded serving of E4B is required, the partitioner work is new.
8. **Option C pressure.** If Gemma 3n (AltUp + LAuReL) or YOCO is wanted next, B's typed references should be designed now as the first two variants of a `ValueRef` space (`site` field). Otherwise the next model forces a second migration.
9. **BF16 embed-scale cast.** HF casts `sqrt(H)` to the weight dtype (`mg4:1417`), giving 50.5 vs 50.596 for BF16 E4B. The graph carries 50.596443. An f32 comparison is consistent, but a BF16 HF comparison is not. Use f32 only for V1.

---

## 7. Open questions for you

- **Q1. Schema**: additive-in-v6 plus a `required_features` list plus fail-closed omission, or bump to v7 with a re-encode?
- **Q2. Generality**: do you accept B (typed `ValueRef` with two relationship kinds) for this programme, or do you want C (a general value DAG) designed first?
- **Q3. Dead weights**: drop the KV-shared layers' `k_proj`/`v_proj`/`k_norm` from the container (110 MB on E4B, fail-closed benefit), or carry them unbound?
- **Q4. Semantics bumps**: one bump per finding-flipping rung (R3→24, R4→25, R5→26), or a single bump when E4B first becomes admissible?
- **Q5. Scope**: is text-generation only acceptable for E4B/E2B now, with image (pad-id PLE plus clipped linears) and audio (a separate tower, 13 findings) as later programmes?
- **Q6. Oracle**: is HF `use_cache=True` the declared oracle, and should R1 also read a non-HF reference (gemma.cpp or LiteRT) to confirm the sharing semantics (§6.1)?
- **Q7. Order**: E4B first as asked, but E2B is the conformance *witness* (`matrix.json:87-88`) and smaller. Should R6 run E2B first for faster evidence, given that E2B also needs R5?
