# Frontier Execution Programme — GLM-5.3-Flash and Inkling-Small

Both checkpoints are now physically on `model-drive`. They stop being
header-only conformance targets and become execution targets.

**Programme statement.** Two coupled goals, neither sufficient alone:

- **Execution closure** — real GLM and Inkling layers running correctly
  against a reference.
- **Residency qualification** — those same real layers proving that
  LARQL's residency model predicts and controls physical behaviour.

**The milestone is not "GLM produces a token".** It is:

> GLM produces a *correct* token under a *declared residency budget*, with
> predicted vs observed resident bytes, page-in, touched bytes and stage
> latency all reconciling.

Blocker count is no longer the objective function. The census did its job:
it made the engine honest enough to say what is actually missing. What
follows is measured, not inferred.

---

## 1. Baseline, frozen 2026-09-06

Measured with `larql` built from `origin/main` `1e57b6d1`, against the real
checkpoints (not headers).

| | GLM-5.3-Flash | Inkling-Small |
|---|---|---|
| bytes | 328,326,771,576 (305.78 GiB) | 531,912,898,740 (495.38 GiB) |
| tensors | 76,108 | 1,048 |
| shards | 62 | 32 + `mtp.safetensors` |
| dtype mix | **F8_E4M3 95.8 %**, BF16 4.2 %, F32 0.0 % | **BF16 100.0 %**, F32 0.0 % |
| plan | 80 representable / 1 mismatched / 32 unrepresented | 34 / 0 / 41 |
| blocking (whole model) | 31 | 41 |
| **blocking (text_generation)** | **28** | **27** |

`larql vindex3 plan` on a 306 GB checkpoint returns in **0.74 s** — headers
only. That property is what makes this programme affordable to iterate.

### GLM stack shape (from `config.json`, cross-checked against the index)

45 decoder layers + 1 MTP layer (`num_nextn_predict_layers: 1`).

- **34 KDA** linear-attention layers — `linear_attn_config.kda_layers`,
  0-indexed, 64 heads × 128.
- **11 `deepseek_sparse_attention`** layers (3, 7, 11 … 43) — MLA-NoPE
  (`qk_rope_head_dim: 0`) with q-LoRA (`q_lora_rank: 1536`,
  `kv_lora_rank: 512`, 64 heads × 256) **plus a DSA indexer**.
- **3 dense MLP** (layers 0–2, `first_k_dense_replace: 3`), **42 sparse
  MoE** in-stack (+1 on the MTP layer = 43 expert banks), 288 routed
  experts + 1 shared, top-8, sigmoid router with
  `e_score_correction_bias`, `routed_scaling_factor: 2.5`.
- **mHC residual topology on every layer** — `hc_mult: 4` parallel residual
  streams, `hc_sinkhorn_iters: 20`, two sites per layer (attn, ffn).

### Inkling stack shape

42 layers + an 8-layer MTP sub-stack; 35 sliding (window 512) + 7 global via
`local_layer_ids`; 256 routed experts (**packed** `w13_weight`/`w2_weight`,
one tensor per layer) + 2 shared with `shared_expert_sink`; its own tensor
dialect (`model.llm.layers.N.attn.wq_du`, `attn_norm`, `mlp_norm`); short
convolutions **on the residual stream** (`attn_sconv`, `mlp_sconv`); a
relative-position scheme (`d_rel: 16`, `rel_extent: 1024`,
`rel_logits_proj`); µP logit scaling.

---

## 2. The reference oracle exists for both models

`transformers 5.16.1` ships **`glm5_next`** *and* **`inkling`**. Neither
checkpoint carries modeling code, so upstream `transformers` — named by each
`config.json`'s `architectures` — **is** the contract.

Pinned in the scratchpad venv, sha256 in `REF_SOURCES.sha256`:

```
2092bbb4…  glm5_next/modeling_glm5_next.py
b62936c9…  glm5_next/configuration_glm5_next.py
234e9f7e…  inkling/modeling_inkling.py
ec8c7edf…  inkling/configuration_inkling.py
```

This removes the transcription risk that dominated the Kimi KDA rungs: the
oracle is executable, not a transcription.

---

## 3. Three corrections the real checkpoints forced

**3.1 GLM applies `gate_lower_bound`. Kimi does not.**

`Glm5NextTextForgetGate.forward`:

```python
if self.safe_gate_lower_bound is not None:
    return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
return -decay_rate * g_softplus
```

GLM declares `gate_lower_bound: -5.0` and `Glm5NextTextConfig` reads it into
`linear_lower_bound`, so **GLM takes the sigmoid branch**. Kimi declares the
same `-5.0` and its own `modeling_kimi.py` reads it nowhere.

LARQL's `KdaOp` documents `gate_lower_bound` as *provenance, not an
execution input*, with a measured control Δ of **1.746** for applying it.
That documentation was correct for Kimi and is **wrong for GLM**. The field
must become a real execution input whose *presence* selects the gate form —
a per-checkpoint fact, carried, never defaulted. The existing 1.746 control
becomes a two-sided gate: it must fire on Kimi and must *not* fire on GLM.

This is the single highest-value finding of the reconnaissance. Reusing the
Kimi KDA executor unchanged on GLM would have produced a silently wrong
recurrence with every shape closing.

**3.2 q-LoRA is genuinely missing, and DSA depends on it.**

`OperandRole` has `MlaQProj` — one `self_attn.q_proj.weight`, Kimi's form
(Kimi has no `q_lora_rank`). GLM spells the query path
`q_a_proj → q_a_layernorm → q_b_proj`. Config carriage is already green
(`q_lora_rank` reads, `mla_use_nope` → `PositionPolicy::None`); the gap is
roles + executor.

It is a hard prerequisite for DSA, not merely adjacent: the indexer's own
query comes from the q-LoRA residual —

```python
q = self.wq_b(q_resid).view(hidden_shape)   # q_resid = q_a_layernorm(q_a_proj(x))
```

**3.3 mHC is most of the way there; the gap is the head, not the layer.**

Finding 111 states `HyperConnection { streams: 4, sinkhorn_iters: 20,
sinkhorn_eps: 1e-6 }` **is executable layer by layer**. What blocks is the
absence of a declared reduction from the 4-stream bundle to the vector the
final norm reads. The reference is three lines:

```python
class Glm5NextTextHyperHead(nn.Module):
    """Unlike DeepSeek-V4, this is an unweighted mean."""
    def forward(self, hidden_streams): return hidden_streams.mean(dim=2)
```

A layer-range image already runs without a head; a whole-stack execution
cannot. Small, and it converts an `unsupported_component` into a fact.

---

## 4. The 28 GLM text blockers, by work item

| work item | findings | note |
|---|---|---|
| **FP8 block-scale ingestion** | 14, 15, 18 | `weight_scale_inv` / `weight_block_size` appear **nowhere** in the Rust sources. 128×128 blocks, `activation_scheme: dynamic`. **95.8 % of the checkpoint's bytes.** |
| **architecture identity** | 8 | `glm5_next_text` matches no family → generic fallback serves Llama-shaped defaults this checkpoint never declared. Root cause of several others. |
| **DSA indexer** | 30–38 (9 keys) | Honest: "architecture work, not a normalisation gap". Greenfield. |
| **attention schedule** | 7, 42, 110 | 11 layers resolve a different kind than declared; `deepseek_sparse_attention` has no vocabulary. |
| **mHC** | 50, 111 | head only (§3.3). |
| **MoE routing** | 55, 78 | `moe_router_dtype: float32`, `topk_method: noaux_tc`. |
| **FFN activation** | 75 | `swiglu_limit: 10.0` → `ExpertGatePolicy::ClampedGlu`; rule exists, no built component answered. |
| **MTP** | 64 | no multi-token-prediction object in the schema. Deferrable. |
| **naming / metadata** | 52, 68, 10, 11, 84, 85 | `mlp_layer_types`, `qk_head_dim`, image/video token ids. |

**Inkling's 27 text blockers are far more concentrated: 20 of them are
"declared, read by nothing", all downstream of finding 7** — `inkling_mm_model`
matches no family, so its entire tensor dialect is unreadable, and finding 73
reports the execution surface incomplete because the stack "carries no
per-layer norm operand" (it spells them `attn_norm` / `mlp_norm`).

That asymmetry is the argument for keeping Inkling in the programme: GLM
alone cannot tell us which abstractions are real and which are Kimi-shaped.

---

## 5. Five layer types — graded against real tensors

The milestone is *one real GLM layer of every type*, not the full model.

| layer type | executor | state | remaining |
|---|---|---|---|
| **KDA** (34 layers) | `exec::kda`, parity-proven 32×128 on Kimi | weights are **BF16** — no FP8 in the way | gate form (§3.1); 64×128 geometry |
| **MLA-NoPE** (11) | `exec::mla`, parity-proven | weights **FP8** | q-LoRA roles + executor; FP8 |
| **dense FFN** (3) | SwiGLU exists | weights **FP8** | `ClampedGlu` binding; FP8 |
| **sparse MoE** (42) | per-expert bank machinery from Kimi | weights **FP8** | `noaux_tc`, router dtype; FP8 |
| **DSA indexer** (11) | none | — | everything |

Read down the "remaining" column: **FP8 is on three of the five rows and
KDA is the only type that can execute today.** That, not q-LoRA, is the
dominant lever — and it is mechanical rather than research.

---

## 6. Rung ladder

Ordering follows the measurement, not the intuition. q-LoRA stays early
because it is a genuine gap that transfers to K3 *and* gates DSA; FP8 moves
ahead of DSA because it unblocks three layer types at once.

- **F1 — `Glm5NextArch` + q-LoRA.** Register the family; add
  `MlaQAProj` / `MlaQANorm` / `MlaQBProj`; branch the MLA executor on
  presence, never on family. Transfers directly to K3.
- **F2 — FP8 block-scale ingestion.** `weight_scale_inv` as a first-class
  scale sibling, `weight_block_size` carried, 128×128 dequant on the
  read path. Gate: bit-exact dequant of one real GLM tensor against torch.
- **F3 — KDA gate form.** `gate_lower_bound` from provenance to execution
  input; two-sided control (fires on Kimi, must not fire on GLM).
- **F4 — mHC head.** Unweighted-mean stream collapse.
- **F5 — `ClampedGlu` + MoE router details.**
- **F6 — DSA indexer.** The research rung: k-pool compression, tail
  selection, top-k → mask.
- **F7 — layer parity.** Each of the five types against the pinned
  reference, real weights, on the boundary ladder the KDA rungs
  established (projection → … → N-position state).
- **F8 — residency qualification.** Predicted vs observed resident bytes,
  page-in, touched bytes, stage latency, on those same real layers.

Inkling follows each generic primitive where it applies; `InklingArch`
(finding 7) is the transfer test for F1's shape.

## 7. Landed 2026-09-06 — F1 (part) and F3

Branch `drive-models-1` off `main` `1e57b6d1`. Gates: **5,694 tests pass, 0
fail**, clippy 0, fmt clean.

### The single-layer oracle (`scripts/glm_layer_oracle.py`)

**A real GLM-5.3-Flash decoder layer executes end-to-end from the real
weights** — layer 0, strict `load_state_dict`, no NaN, 13 s, 1.08 GiB at
f32. One layer covers three of the five types at once: KDA, dense FFN
(FP8-dequantised), and both mHC sites.

Two declared differences between the checkpoint's tensor dialect and the
reference module's parameter layout, both read off the reference rather
than guessed, and both facts LARQL's own role classifier will need:

- `conv1d.weight` is ONE depthwise conv over `cat([q, k, v], dim=-1)`;
  the checkpoint ships `q_conv1d` / `k_conv1d` / `v_conv1d` separately.
  The order comes from the `torch.cat` that feeds it.
- the decay gate lives in a `forget_gate` submodule; the checkpoint keeps
  its four tensors flat on `self_attn`.

The FP8 dequantiser is transcribed from transformers' own
`Fp8Dequantize._dequantize_one`, and it carries one rule worth stating:
**the block size is derived from the scale grid, not from
`quantization_config.weight_block_size`** — the same checkpoint may ship
different grids for different tensors. It is also a *multiply* by
`weight_scale_inv`, despite the name.

### `Glm5NextArch` — the family, registered twice over

`detect_from_json` dispatch **and** `ARCHITECTURE_REGISTRY`. The two are
separate tables and only one direction was tested (every registry pattern
must be honoured by dispatch, but not the reverse) — a dispatch arm with no
registry entry left finding 8 standing while the architecture was already
in use. Both now carry `glm5_next` / `glm5_next_text`, the same two-level
identity Kimi K3 declares.

Facts it declares, each read from the reference:

- `key_prefixes_to_strip` gains `"model.language_model."`. Without it the
  trait default strips `model.` and leaves `language_model.layers.0.…`,
  which matches no layer prefix — every stack tensor reads unclassified.
- MoE keys (`mlp.gate.*`, `mlp.experts.{i}.{gate,up,down}_proj`,
  `mlp.shared_experts.*`), MLA keys including the **q-LoRA pair**
  `q_a_proj` / `q_b_proj` that Kimi has never had
  (`KimiMLAAttention.__init__` asserts `q_lora_rank is None`).
- `mla_kv_a_norm_eps` = `rms_norm_eps`, stated — **not** the `1e-6` class
  default Kimi's latent norm silently runs at against a layer eps of
  `1e-5`. Two families, one tensor name, a factor of ten.
- `default_norm_eps` = `1e-5`, from `Glm5NextTextConfig`'s own default.

**Measured: text blockers 28 → 27, and the byte ledger changed shape.**
`target.expert_bank` now carves 304,480,124,928 bytes out of
`decoder_stack`, which falls from 324.7 GB to **20,181,830,520 bytes**. The
whole ledger closes exactly:

| object | bytes |
|---|---|
| decoder_stack | 20,181,830,520 |
| expert_bank | 304,480,124,928 |
| embedding | 1,268,776,960 |
| output_head | 1,268,776,960 |
| perception_tower | 1,127,254,016 |
| final_norm | 8,192 |
| **total** | **328,326,771,576** |
| checkpoint | 328,326,771,576 |
| **delta** | **0** |

That number is the residency headline: **GLM's non-expert working set is
20.2 GB, not 306 GB.**

### `KdaGateForm` — the gate is now declared, and refused when unjudged

`KdaOp::gate_lower_bound`'s own docs predicted this case: *"if a checkpoint
ever appears whose reference does apply it, that is a second gate form and
belongs in its own field rather than changing what this one means."* GLM is
that checkpoint, so that is exactly the shape built.

- `KdaGateForm::{Softplus, ClampedSigmoid{lower_bound}}` in
  `larql-models::config`.
- `ModelArchitecture::kda_gate_form()` — defaults to `None`, and the
  executor **refuses** on `None` rather than picking. Same contract, and
  the same reasoning, as `mla_kv_a_norm_eps`.
- `KimiLinearArch` → `Softplus`. `Glm5NextArch` → the reference's own rule
  over `gate_lower_bound` + `safe_gate`.
- carried through `ResolvedTopology` → `ExecutionSurface.kda_gate_form` →
  `KdaOp.gate_form` → `KdaWeights.gate_form`.

**Verified on both real checkpoints** — same declared `-5.0`, different
computed gate:

```
GLM-5.3-Flash   kda_gate_form: {"form": "clamped_sigmoid", "lower_bound": -5.0}
Kimi Linear     kda_gate_form: {"form": "softplus"}
```

`safe_gate` is parsed rather than assumed, so the rule is *checked*: GLM
declares no `safe_gate` and its reference defaults it to `True`, but a
checkpoint saying `false` must reach the softplus branch, and it can only
do that if the key is carried.

**Controls, two-sided.** The existing control (apply the clamp on a
Softplus checkpoint) is unchanged and still fires. Its mirror is new:
declare `ClampedSigmoid`, force softplus, and require movement — that arm
is exactly what running the Kimi-shaped executor unchanged on GLM would
compute. A third test asserts the declared form *alone*, with no mutation,
selects the arithmetic, with an identity arm that must read exactly `0.0`.
A one-sided control could not tell "declared and honoured" from "declared
and ignored in favour of the hard-coded one".

All 24 pre-existing KDA parity tests still pass unchanged, which is what
shows the refactor changed no semantics on the Softplus path.

**Two repo gates caught this work and were right to.** `parser_sync`
refused `safe_gate` until it was registered in `CONSUMED_LEAF_KEYS`;
`model_config_persists_every_forward_affecting_field` refused
`kda_safe_gate` until it was classified. The second needed a category that
did not exist — `RESOLVED_INTO_ANOTHER_FIELD`, for an input read only to
resolve a fact that *is* carried — because persisting both the input and
the conclusion would give one fact two sources of truth.

### Still owed on F1

The q-LoRA **operand roles and executor** are not built. `OperandRole` has
`MlaQProj` — one flat `self_attn.q_proj.weight` — and GLM needs
`MlaQAProj` / `MlaQANorm` / `MlaQBProj` with the executor branching on
presence, never on family. `Glm5NextArch` declares the keys; nothing binds
them yet. This is also the hard prerequisite for F6: the DSA indexer's own
query is `wq_b(q_resid)`, where `q_resid = q_a_layernorm(q_a_proj(x))`.

## 8. F2 — FP8 storage, landed and bit-exact

Sequenced ahead of q-LoRA on the user's call: FP8 is 95.8 % of the
checkpoint's bytes and gates three of the five layer types, so it is what
stands between LARQL and the real GLM payload.

### The codec

`larql_models::quant::fp8_finegrained` — E4M3 values against a
**two-dimensional** f32 scale grid. Deliberately named apart from
`fp4_block`'s `FP8_BLOCK_BYTES`, which is LARQL's own packed 257-byte
record with an *E4M3* scale and one-dimensional blocking: the two share
only the element codec.

It is also unlike every other blocked format in the crate. `Q8`, `Q4` and
`NVFP4` block along the input axis alone, so a scale never spans rows;
here a scale covers a `128 × 128` tile, and the grid row is a function of
the weight row. Within a row it still reduces to "scale changes every
`block_cols`", which is why the kernel shape survives.

Two properties, both of which invite a wrong guess and both now asserted:

- **The tile is derived from the scale grid, not from
  `weight_block_size`.** transformers' own dequantiser computes
  `block_m = rows / scale_rows` per tensor, citing MoE experts at
  `[1, 32]` beside dense linears at `[128, 128]` in one checkpoint.
  Reading the config value would be right on GLM and wrong by
  construction.
- **`weight_scale_inv` is MULTIPLIED.** The name says otherwise. A test
  pins it, because dividing is the natural misreading.

### The gate: bit-exact on real tensors

`scripts/glm_fp8_dequant_gate.py` runs transformers' own `Fp8Dequantize`
against `cargo run --example fp8_dequant_probe` over real checkpoint
tensors, comparing **raw f32 bit patterns** — not a tolerance, because
both sides do one f32 multiply per element in the same order, so any
difference is a real disagreement about the format rather than drift.

**All 6 tensors bit-identical, 125,829,120 values, zero differing bits:**

| tensor | shape | grid |
|---|---|---|
| `layers.0.mlp.gate_proj` | [12288, 4096] | [96, 32] |
| `layers.0.mlp.down_proj` | [4096, 12288] | [32, 96] |
| `layers.3.mlp.experts.0.up_proj` | [2048, 4096] | [16, 32] |
| `layers.3.mlp.shared_experts.down_proj` | [4096, 2048] | [32, 16] |
| `layers.3.self_attn.q_a_proj` | [1536, 4096] | [12, 32] |
| `layers.3.self_attn.kv_a_proj_with_mqa` | [512, 4096] | [4, 32] |

One of every FP8 kind GLM ships — dense FFN in both orientations, routed
expert, shared expert, and both MLA down-projections — so a defect
specific to a shape or a grid cannot hide behind a passing sibling.

**The gate is verified to be able to fail** (`--self-test`), because a
comparison that has only ever returned the answer you wanted is not
evidence. Each arm is a real misreading, not noise:

| self-test | values differing (of 2,097,152) |
|---|---|
| `divide` (÷ scale instead of ×) | 2,097,138 |
| `wrong-tile` (tile axes rolled) | 1,490,931 |
| `flip-one-bit` | **1** |

The last arm is the one that matters: it shows the comparison resolves a
single bit in two million values, so "bit-identical" is a measurement and
not a rounding.

### The declaration side, kept honest

`StoredRepresentation` now reads `fmt`, `weight_block_size` and
`activation_scheme`, and the three are classified **differently on
purpose**:

- **`fmt` — represented and enforced.** `e4m3` and `e5m2` are different
  codecs of the same byte width, so decoding one as the other produces
  plausible numbers from every byte rather than an error.
  `is_finegrained_fp8_e4m3()` requires it.
- **`weight_block_size` — represented as provenance *with a check*.**
  `Fp8Grid::check_declared_tile` compares the declared tile against the
  derived one and reports both on disagreement; the derived one stays the
  authority. Exercised on every tensor the gate reads — all six agree at
  `(128, 128)`.
- **`activation_scheme` — NAMED AND REFUSED.** It says the reference
  kernel quantises activations at run time and runs an FP8 GEMM. This
  build dequantises weights and runs f32: numerically close, a different
  route. Classifying it as represented would claim an execution path that
  does not exist, so it stays blocking under its own component name.

**Text blockers 27 → 25** — and the third key deliberately did not move.

### F2 closed — execution and carriage

**`WeightRows::Fp8Block` + `FusedFp8Block`.** Decode E4M3 in registers,
scale per tile, accumulate f32, discard — the same architecture as
`FusedBf16`/`FusedQ8`, for the CPU-1B reason (widen-to-scratch reads half
the bytes and runs slower than plain f32). Unlike `FusedQ8`/`FusedQ4` this
kernel is **not lossy**: the codes and scales are the checkpoint's own, so
it computes what the reference loader materialises — only later.

The variant carries something no other blocked format needs. `Q8`, `Q4`
and `NVFP4` tile along the input axis alone, so a row partition can cut
codes and scales at the same index. Here one scale row serves
`block_rows` output rows, so a slab records **`row_in_tile`** — where its
first row sits within its first tile. Without it, a partition that does
not land on a tile boundary reads its scale slice from row 0 and produces
plausible, wrong numbers.

That field is not speculative: injecting `row_in_tile: 0` into
`slice_rows` fails `an_off_boundary_row_partition_agrees_with_the_whole`
and **nothing else** — five of six FP8 slab tests still pass. The test
isolates exactly the defect it exists for.

**Container carriage.** The encoder already placed the scale siblings —
it places by tensor prefix, which is why the byte ledger closed exactly —
so the gap was binding, not storage. `OperandSource::companion` reaches a
tensor's sibling *within the same object* (deliberately not a free
`(object, tensor)` pair: a scale from elsewhere would be another matrix's,
and the type should not be able to express that), and `load_fp8_block`
derives the tile from the two shapes and refuses every disagreement —
non-F32 scales (E8M0 is a real variant of this scheme and is named, not
guessed), a non-rank-2 weight, a grid that does not tile, a length that
contradicts the declared shape.

**Operand closure had to learn that a scale is not an operand.** The
first carriage run refused with `UnclassifiedOperand` on
`weight_scale_inv` — one defect on the fixture, and ~37,000 on GLM. A
quantisation scale is *part of* the operand it accompanies, so it is
skipped **only when its weight is present in the same object**; an
orphaned scale is still a defect, because a split pair leaves neither
half bindable.

`opplan/exec/tests/fp8_carriage.rs` runs the whole path against an
authority that shares nothing with it below the source file:

```
checkpoint (F8_E4M3 + F32 grid)
  → encode_system → OperandStore → load_weight(Fp8Block)
  → WeightSlice → WeightRows → FusedFp8Block           candidate
the SAME source bytes → fp8_finegrained::dequantize → scalar dot   authority
```

with a control: a scale grid rotated by one entry must **not** agree.

### The milestone: a real GLM dense FFN, natively

`scripts/glm_ffn_fp8_gate.py` feeds the reference's own
`post_attention_layernorm` output into LARQL's FP8 kernel over GLM
layer 0's actual bytes — **no dequantised weight image exists at any
point** — and compares every boundary:

| boundary | values | rel. error |
|---|---|---|
| `gate_proj` | 12,288 | 1.185e-06 |
| `up_proj` | 12,288 | 1.178e-06 |
| `act_fn` | 12,288 | 1.191e-06 |
| `down_proj` | 4,096 | 2.409e-06 |

against a bar of 2e-5 set from the arithmetic (an f32 dot over 4096 terms
reassociated between BLAS blocks and 128-wide tiles), not tuned.
Bit-exactness is claimed for the **decode**, where a disagreement would
mean a format error; this is an ordering effect.

**Two things the first run got wrong, and what they were.** `act` read
rel 1.0 while `down_proj` — which consumes it — agreed at 2.4e-6. The
computation was right and the comparison target was wrong: the
reference's `act_fn` hook captures `silu(gate)` **alone**, because the
module is applied to the gate and multiplied by `up` outside it. And the
reference *does* apply `swiglu_limit`, **asymmetrically**:
`gate.clamp(max=10)` but `up.clamp(-10, 10)`.

**The clamp is now qualified where it bites.** At the oracle's own
activation scale the gate peaks at 0.233 against a limit of 10, so the
first run clamped **0** values — it qualified nothing, and said so.
`--clamp-control 200` drives the gate to 46.7, clamps 1,022 gate and
2,206 up values, and LARQL still agrees at **4.9e-7**.

The asymmetry itself is measured rather than asserted, and the answer is
counter-intuitive: a symmetric gate clamp moves the output by only
**2.06e-5**, because `silu(-10)` is already `-4.5e-4` and `silu(-46)` is
`-1.3e-19`. The asymmetry is real in the source and nearly inert in the
arithmetic — only a measurement separates those, so the gate reports it
every run.

### What F2 does not claim

The path is qualified **operand by operand**, not model-wide: GLM's plan
is still inadmissible (25 text blockers), so no GLM container exists and
the real-weight arithmetic is driven directly off the checkpoint.
Container carriage is proven on a synthetic FP8 fixture through the real
encode path. Joining the two needs admission, which is what F4-F6 buy.

`activation_scheme` remains refused, and that refusal is now a **checked
property**: the same fixture encodes without the key and is refused with
it, so "storage yes, compute path no" is a test rather than a comment.

## 9. F1 — q-LoRA, and a real GLM MLA layer

> **Superseded in part, 2026-09-07.** This rung built a q-LoRA query path
> for GLM independently of `K3-MLA-Q-LORA-1`, which landed the same
> abstraction for Kimi-K3 first. Main's `MlaQueryProjection` /
> `MlaQueryWeights` are what ship; this branch's `MlaQuery` was dropped
> at the rebase, along with its `mla_qlora.rs` (main's
> `mla_parity_q_lora.rs` is the same rung). The findings below stand —
> they are facts about GLM, not about either implementation — but the
> **2.911e-06 figure was measured through the dropped implementation and
> has not been re-run against the merged code**. The probe is ported and
> compiles; re-running it needs the checkpoint remounted.
>
> The K3 cell this section reports emptying was emptied by main's rung,
> not this one.


### The shape

`OperandRole::{MlaQAProj, MlaQANorm, MlaQBProj}`, and `MlaOp.q_proj`
becomes `MlaQuery::{Direct, LowRank}` — an enum, not three `Option`s,
because the forms are mutually exclusive and a consumer must not be able
to hold a half-built one.

**The declaration chooses; the estate checks.** `q_lora_rank` decides
which operand set a layer requires — never the family, and never a sniff
of which tensors turned up. A declared rank with no `q_a_proj` is a
missing operand; a `q_a_proj` under no declaration is an operand implying
an undeclared op. Both refuse, and neither is resolved by preferring
whichever side is easier to believe.

`q_a_layernorm` gets **its own epsilon** (`mla_q_a_norm_eps`), defaulting
to `None` so an unjudged family refuses. GLM's reference constructs both
norms with `config.rms_norm_eps`, so on this family they agree — stated
anyway, because the two being equal is a fact about GLM and not a
property of MLA. Kimi's latent norm runs at `1e-6` against a layer eps of
`1e-5`, which is what that field exists to remember.

`MlaTrace` gains **`q_latent`**, the normalised query latent. It is a
named boundary rather than an intermediate because it has a consumer
outside the attention block: GLM's DSA indexer derives its own query from
exactly this value (`Glm5NextTextIndexer.forward` takes `q_resid`).
Exposing it now is what lets F6 be built without re-deriving it and
risking a second, subtly different one.

### It closed a K3 cell without being aimed at K3

`k3_representable.rs` pinned `MLA_LAYER_UNADDRESSED` as exactly the
q-LoRA triple, documented as "its own capability cell, not this rung's".
That list is now **empty**, and the estate census went 5,382 → 5,379
unclassified (12 → 9 distinct spellings). What remains on K3 is one cell:
K3-LATENTMOE-1's latent expert bank.

That a GLM rung emptied a K3 list is the evidence that the abstraction is
real rather than a second special case — the query path names no family
anywhere in it, so every checkpoint declaring the same rank gets the same
operands.

### Real GLM MLA-NoPE parity

`scripts/glm_mla_gate.py`, layer 3, mixed native storage — `q_a_proj`,
`q_b_proj`, `kv_a_proj_with_mqa` and `o_proj` are fine-grained FP8 while
`kv_b_proj` and both norms are BF16, so one layer exercises both binding
paths and nothing is widened:

| boundary | values | rel. error |
|---|---|---|
| `q_latent` | 1,536 | 1.149e-06 |
| `output[0]` | 4,096 | 2.911e-06 |
| `output[1]` | 4,096 | 2.731e-06 |
| `output[2]` | 4,096 | 2.867e-06 |
| `output[3]` | 4,096 | 2.777e-06 |

Geometry is read from the checkpoint's own tensor shapes — 64 heads,
`qk_nope` 256, `v_head_dim` 256, latent 512 — so a misread config cannot
pass unnoticed.

**Why this is separable from DSA, checked rather than assumed.** Layer 3
is a `deepseek_sparse_attention` layer, so the reference gates attention
with a top-k mask. Below `index_topk` (2,048) that mask selects every
causally-visible key, making sparse and dense causal attention the same
function. The gate does not take this on trust: it captures the mask the
**real forward actually built** and refuses to report parity unless it is
full-causal. At 4 positions it reports `FULL-CAUSAL (indexer inert)`.

**Controls, and a free diagnostic.** `--control omit-q-a-norm` moves
`q_latent` by 0.974 and the later positions by 0.53–0.65; `omit-kv-a-norm`
moves every position by ~0.94 and leaves `q_latent` untouched. Both fire.

But `omit-q-a-norm` leaves **`output[0]` bit-unchanged** — with one
visible key the softmax is 1.0 whatever the query. So *position 0 cannot
witness a query-path defect*, and a disagreement that spares it is in the
query path while one that moves it is not. That is the MLA counterpart of
KDA's "q never touches the recurrent state", and it is recorded in the
script so it is not rediscovered.

### The counter nondeterminism, closed

`the_counters_only_move_for_mapped_images` failed once, off by exactly
4,096 bytes — one page, one concurrent mapped image. `#[serial]` only
excludes other serial tests, and `STAGED_BYTES`/`STAGED_IMAGES` are
process-global atomics that any concurrently running stager moves.

A gate owns its own state, so the fix belongs in the gate: a
**thread-scoped** pair of counters, incremented at the same site as the
globals so the two cannot drift, with the test asserting on those and
requiring the globals to have moved by at least its own contribution.
Staging is synchronous on the caller's thread and each test runs on its
own, so the gate becomes immune to concurrency while still being evidence
about the real counters. **20 full-suite runs, zero failures.**

**The fix in the tree is not this branch's.** `fix-stage-ledger-race`
(#445) landed the same design on main first, as
`staged_bytes_on_this_thread` / `staged_images_on_this_thread`, and it is
stronger: it adds `a_thread_tally_ignores_another_threads_staging`, which
makes the race deterministic by staging from a foreign thread, and it
baselines the process assertion against the process counter rather than
the thread's. This branch's version was dropped on rebase. What belongs to
this work is the diagnosis, not the code.

## 10. Sparse MoE — and the defect the gate was built to find

### `ExpertGatePolicy::ClampedGated`

The first run of the MoE gate disagreed by a relative **31.7**. Not noise,
and the magnitude was the diagnosis: LARQL's `ExpertGatePolicy::ClampedGlu`
is GPT-OSS's

```text
g = gate.clamp(max=limit);  u = up.clamp(-limit, limit)
out = (u + 1) * (g * sigmoid(alpha * g))
```

while GLM applies **the same clamp** and then an ordinary SwiGLU,
`silu(g) * u` — its reference carries the comment *"Simple swiglu instead
of alpha"* over exactly that line. At a residual-scale activation
`(u + 1) ≈ 1` while `u ≈ 0.03`, so the GPT-OSS form is larger by roughly
`1/|u|`. 31.7 is that ratio.

**The existing code had already refused to guess this.** The trait's
`expert_gate_policy` default carried the comment: *"A declared clamp says
a bound exists; it does not say the layer computes `ClampedGlu` … GLM-5.3
-Flash and Inkling-Small both declare a `swiglu_limit` too, and nothing
on hand says they share that activation."* That caution was right, and
the reference now settles it: a third variant, `ClampedGated { limit }`,
declared by `Glm5NextArch` and never derived from `swiglu_limit`.

Every path that could serve one for the other was made to refuse rather
than substitute: the Metal expert-activation dispatch has no shader for
it (`clamped_glu_bias` computes a different function) and refuses it at
**admission**; the bound-MoE inference path returns `UnsupportedFormat`;
the production backend's `require_plain_gate` names it.

Where Metal refuses matters, and the crate had already written down why.
`kernels::ffn::expert_activation_supported` is the one fact, read by the
GPU route's `gpu_route_supported` and by the descriptor dispatch's assert
at the top of its body — both before any command encoder exists. The
three combine matches keep a backstop arm, but it must never be the thing
that reports the refusal: a panic raised mid-encode leaves the encoder
unended and **Metal aborts the process instead of naming the layer**, so
the caller never learns which layer was at fault. That is the same reason
`bind_situ_glu`'s bias assert is paired with an admission check rather
than left to fire alone. Both admission surfaces carry a paired
refuse/admit test, and the refusal arm was verified to fail when the
predicate is stubbed to `true` — a gate that returns the answer you
wanted proves nothing until you show it could return the other. `probe_swiglu_limit` now answers
for **both** clamped policies, because the declared bound is the same
fact in each — answering for only one would have reported GLM's clamp as
uncarried while its executor applied it.

### Real GLM sparse MoE parity

`scripts/glm_moe_gate.py`, layer 3, the whole **288-expert bank bound at
6.78 GiB of native FP8** — the router selects inside the call, so handing
it only the experts the reference chose would have tested the branch
while assuming the selection:

| position | rel. error | selected experts |
|---|---|---|
| 0 | 8.265e-07 | 19, 22, 76, 92, 98, 127, 204, 205 |
| 1 | 7.531e-07 | 22, 25, 66, 107, 109, 140, 172, 225 |

and 4.58e-07 at `--gain 200`, where the clamp actually bites.

The routed branch is compared **apart from** the shared expert: the
reference's `forward` is `experts(x) + shared_experts(x)`, an exact sum,
so the routed half is recovered by subtraction and a disagreement names
which branch it is in. (Branch magnitudes at the fixture's scale:
routed 0.0248, shared 0.0149.)

Two properties that trip a naive comparison, both handled rather than
worked around: the reference's `torch.topk(..., sorted=False)` leaves
expert **order undefined**, so ids are compared as a set; and the
checkpoint's per-expert `gate_proj`/`up_proj` fuse into the module's
`gate_up_proj` as `cat([gate, up], dim=0)`, read off `_apply_gate`'s own
`chunk(2, dim=-1)`.

### Controls

| control | what it changes | relative move |
|---|---|---|
| `gpt-oss-glu` | the defect above | **3.671e+01** |
| `no-renorm` | weights only — selection identical | 3.176e+00 |
| `no-scale` | drops `routed_scaling_factor` 2.5 | 6.000e-01 |
| `no-clamp` (`--gain 200`) | drops the clamp | 2.146e-01 |

`no-renorm` and `no-scale` **leave the selected experts untouched and
change only their weights**, so a gate that merely agreed on top-k could
not pass them. `no-scale` lands on exactly `1 - 1/2.5 = 0.600`, which is
the scaling factor read back out of the measurement.

`no-clamp` is **inert at gain 1** — nothing reaches ±10 at residual
scale — and the gate says so and exits non-zero rather than reporting a
pass. The bias-corrected-selection vs unbiased-weighting split has its
own unit control over the same router code
(`kimi_router::Mutation::GatherBiasedWeights`).

### Four of five types, on real GLM weights

| type | layers | status |
|---|---|---|
| KDA | 34 | executor parity-proven (Kimi); gate form corrected for GLM |
| dense FFN | 3 | **2.409e-06** |
| MLA-NoPE + q-LoRA | 11 | **2.911e-06** |
| sparse MoE | 42 | **8.265e-07** |
| DSA indexer | 11 | unbuilt — and measurably inert below 2,048 |

## 11. Expert-bank residency — the first physical result

`examples/glm_moe_residency.rs`, layer 3, the real 288-expert bank
mmapped from the checkpoint with **readahead off** (`MADV_RANDOM` —
kernel readahead would page in the very sparsity under measurement).
Expert codes are *borrowed from the mapping*, so a read of them is a page
fault on the real file.

**The invariant is enforced, not assumed:** every arm's routed output is
compared byte-for-byte against the first arm's and a mismatch aborts. A
residency result over a changed computation is not a residency result.

### The instrument had to be fixed before any number was believed

The first run reported all three arms identical with `delta +0.0 MiB` and
67.8 % of the bank already resident: **`MADV_DONTNEED` did not evict**,
exactly as this workspace's incumbent probe documents for Darwin, and
`MADV_FREE_REUSABLE` returns `EPERM` here. `msync(MS_INVALIDATE)` does
evict. The probe now runs a **spine check on the SELECTED regions** and
*skips an arm it cannot make cold*, naming why, rather than reporting a
warm latency under a cold label.

### The numbers

Selected working set, one token, top-8 of 288:
**192.05 MiB of experts (+2.25 MiB router) = 2.78 % of the 6.752 GiB bank.**

| arm | resident before | after | major faults | latency |
|---|---|---|---|---|
| demand · cold | 0.0 MiB | 192.1 MiB | 12,295 | 219.82 ms |
| advise · cold | 0.0 MiB | 192.1 MiB | 12,295 | 214.12 ms |
| warm | 192.2 MiB | 192.2 MiB | 0 | **18.63 ms** (median of 9, 17.66–20.42) |

**Three results.**

**1. The prediction is the estate.** `coverage 1.0003` — resident-selected
matches predicted-experts to within one page — and **UNSELECTED residency
is 0.0 MiB**. With readahead off, exactly the selected experts' pages come
in and nothing else. Residency here is a property that can be computed
rather than observed, on a real 6.75 GiB bank.

**2. Logical bytes requested == physical bytes paged in, ratio 1.000.**
192.0 MiB logical against 192.1 MiB physical, and `12,295 × 16 KiB` is
that same 192.1 MiB — the fault count *is* the byte count. There is no
read-ahead amplification and no extent-scatter waste, despite the
checkpoint interleaving other layers' tensors through this bank's span at
83–98 % density. **So the next optimisation is cache hit rate, not layout
coalescing.**

**3. `MADV_WILLNEED` buys almost nothing here: 214.12 ms against 219.82,
with an IDENTICAL fault count of 12,295.** The advice changed the
scheduling of the reads and not their number or volume. Recorded as a
negative result rather than dropped.

Cold costs **≈201 ms for 192.1 MiB ≈ 957 MiB/s** from the external drive,
an **11.8×** penalty over warm. The warm arm's 18.63 ms is the FP8 decode
and matmul for 8 experts — compute, not I/O — so the cold penalty is
essentially all of the difference.

### The reuse curve needs real hidden states, and here is why

The natural next question — *at N GiB resident, what fraction of the next
token's selected bytes are already present?* — cannot be answered from
synthetic input, and that is measured rather than assumed:

> 24 random hidden states at the residual scale select **108 distinct
> experts of 288**, with a mean consecutive overlap of **0.65 of 8**.

That is near-uniform selection. A reuse curve built on it would describe
the router's behaviour off-manifold, understating reuse and overstating
the working set. Real layer-3 hidden states require running embed →
layers 0–2 (KDA + dense MLP + mHC), which the pinned reference can
produce — that is the next step, not this one.

## 12. The reuse curve, on real hidden states

### Getting real layer-3 inputs

`scripts/glm_expert_trace.py` runs embed → layers 0–2 → layer 3's
attention and both mHC sites through the pinned reference, and takes the
vector its MoE actually routes on. Design frozen before any result was
looked at.

**A prefix bug found by the OOM killer, worth recording.**
`load_layer(ckpt, "…layers.1")` used a bare `startswith`, and
`…layers.1` is a prefix of `…layers.10` through `…layers.19` — so it
loaded **17,628 tensors instead of 29**, including sparse layers whose
288-expert banks are ~300 GB once widened to f32. The process was killed
with no output. The rule is the one the Rust side already applies to
`per_expert_bank_prefix`: **a prefix is only a prefix if the divergence
falls on a `.` boundary.**

### The curves

128 real tokens, layer 3, top-8 of 288:

- **225 distinct experts touched (78.1 %)**
- **mean consecutive overlap 0.90 of 8 (11.2 %)**

against the synthetic control's 0.65 of 8 — real routing has *more*
locality than random input, but not much.

| window k | of a token's experts, seen in the last k tokens | still to fetch |
|---|---|---|
| 1 | 11.1 % | 170.6 MiB |
| 4 | 19.4 % | 154.7 MiB |
| 16 | 46.6 % | 102.6 MiB |
| 64 | 72.9 % | 52.1 MiB |

| budget | experts | LRU | LRU ex-first-touch | Bélády optimal |
|---|---|---|---|---|
| 0.25 GiB | 10 | 10.6 % | 13.6 % | 20.1 % |
| 0.50 GiB | 21 | 15.1 % | 19.4 % | 31.2 % |
| 1.00 GiB | 42 | 24.5 % | **31.4 %** | 44.8 % |
| 2.00 GiB | 85 | 42.1 % | **53.9 %** | 62.2 % |
| 4.00 GiB | 170 | 65.7 % | **84.2 %** | 76.5 % |
| 6.75 GiB | 288 | 78.0 % | **100 %** | 78.0 % |

**Read the `ex-first-touch` column.** A finite trace has a compulsory
ceiling — 225 first touches over 1,024 slots is exactly the 78.0 % the
full-bank arm reports — so the raw LRU column understates steady state.
Excluding compulsory misses, a full-size cache is 100 % by construction
and 4 GiB/layer already reaches 84 %.

LRU beats Bélády at 4 GiB (84.2 % against 76.5 %) only because the two
columns are normalised differently — the optimal arm is not
compulsory-corrected. Its value is at the small budgets, where it shows
LRU leaving roughly a third of the achievable hits on the table
(24.5 % against 44.8 % at 1 GiB).

### The arithmetic that follows

Per sparse layer per token: **192 MiB routed + 24 MiB shared expert**
(always active, and therefore trivially cacheable) **+ 2.25 MiB router**.
Across 42 in-stack sparse layers the full estate is **42 × 6.75 =
283.5 GiB**, against 128 GiB of machine — so a per-layer budget of
~2.4 GiB is what actually fits, which the table puts at roughly **54 %
steady-state hit** and ~88 MiB/layer/token still to fetch, or **~3.6 GiB
per token**.

## 13. Latency, on an idle machine — and the lever is not the disk

The first timing set was taken while a peer session compiled and tested
in another worktree (load average **30**) and was discarded. These were
taken at load **2.7–3.6** with nothing but the harness running, five
rounds, and they are tight enough to separate the arms.

| arm | median | spread | effective | per fault |
|---|---|---|---|---|
| demand · cold | 210.4 ms | 1.4 % | 913 MiB/s | 17.1 µs |
| advise · cold | 209.1 ms | 0.5 % | 919 MiB/s | 17.0 µs |
| touch · cold | 208.8 ms | 1.0 % | 920 MiB/s | 17.0 µs |
| warm | **17.8 ms** | 5.3 % | — | — |

**Cold/warm is 11.8×.** Warm reproduces three independent times at
17.8–18.6 ms.

### The three access policies are the same to within 0.8 %

Not "indistinguishable at this noise level" — *measured equal*. Demand,
`MADV_WILLNEED` and concurrent pre-touching land within 1.6 ms of each
other on a 210 ms operation, at per-arm spreads of 0.5–1.4 %, and all
three take exactly **12,295** major faults. Neither the kernel's advice
nor an explicit parallel touch changes the number of requests or their
cost.

### And the disk is not the wall either

Same method, idle machine, alternating, three rounds each:

| volume | rate | per fault |
|---|---|---|
| external USB (`model-drive`) | 520 MiB/s | 30.0 µs |
| internal SSD | 588 MiB/s | 26.5 µs |

**The internal SSD is 13 % faster than a USB drive at this**, and both are
far below what the device can do — an internal NVMe is a multi-GB/s part
delivering 588 MiB/s here. The routed FFN's own threading reaches
913 MiB/s, 1.75× the serial figure, which is about 1.8 faults in flight.

So the cost is **~17–30 µs of per-fault latency at a shallow queue**, not
bandwidth and not the medium. Moving GLM's 306 GB to the internal drive
would buy on the order of 13 %, not the 5–10× a bandwidth roofline
implies. The lever is **request depth and size** — bigger reads, more
of them outstanding — which is a property of how the bank is fetched, not
of where it lives.

This is the question §12 left open, and it is now settled in the
direction that makes the earlier layout result matter: bytes requested
already equal bytes paged in, so there is nothing to win by coalescing
*what* is read; the win is in *how many at once*.

### Order of magnitude

At 17.8 ms warm per sparse layer, 42 layers give **0.75 s → ~1.3 tok/s**
of expert compute before any other layer type. Fully cold at 210 ms gives
8.8 s → **~0.11 tok/s**. At the ~2.4 GiB/layer a 128 GiB machine affords
against a 283.5 GiB estate — roughly 54 % steady-state hit — about
**0.2 tok/s**.

Arithmetic on one measured layer type, with no kernel optimisation pass
and no other layer in it. What it does establish is where the headroom
is: between 0.2 and 1.3 tok/s lies expert I/O, and that I/O is bound by
fault concurrency rather than by the device.

## 14. Deep fetch — the fault path is beatable by 3x

§13 settled that paging is latency-bound at a shallow queue rather than
device-bound. The one discriminating question left: **can a different
fetch realisation turn 12,295 tiny fault events into a few large
outstanding reads, without changing the selected bytes or the output?**

Two arms, both replacing the fault path for the selected experts with
explicit `pread` into owned buffers and re-pointing the bank's slices at
them. Everything else — selection, byte set, kernel, native FP8 — is
untouched. **48 reads of 4.0 MiB** (each expert is three code matrices
and three scale siblings; 8 experts × 6 extents).

| path | requests | rate | fetch | + compute | total stage |
|---|---|---|---|---|---|
| mmap fault (demand) | 12,295 faults | 905 MiB/s | fused | fused | **212.1 ms** |
| `pread` serial | 48 reads | 1,904 MiB/s | 100.8 ms | 20.4 ms | **121.2 ms** |
| `pread` × 8 threads | 48 reads | **2,759 MiB/s** | 69.6 ms | 20.4 ms | **90.0 ms** |

Medians; parallel n=7 across virgin layers, spread 62.0–79.6 ms.

**3.0× the fault path's bandwidth and a 2.36× faster cold routed stage.**
Serial explicit reads alone are already 2.1× the fault path, which is the
cleanest statement of the finding: the cost was never the bytes or the
device, it was **12,295 round trips**.

**The invariant holds.** A `pread-parallel` arm and a `warm` arm on the
same layer produce byte-identical routed output; the probe aborts on any
difference and did not.

Compute costs 20.4 ms in the explicit arms against 17.8 ms warm — a
consistent ~15 % penalty, presumably heap buffers against page-cache-warm
mappings. Small, and not chased.

### Two instrument bugs this rung cost, both worth recording

**A `pread` populates the same unified buffer cache the mapping reads.**
The spine check ran *after* the fetch and therefore saw the fetch's own
bytes, reporting 99.2 % resident on a virgin layer and skipping every
explicit arm. The check must precede the fetch.

**Eviction depends on memory pressure, not on advice.** With the peer
session gone and the machine idle, `msync(MS_INVALIDATE)` stopped
reclaiming pages that earlier runs had evicted successfully — nothing was
competing for them. The arms are therefore run on **virgin sparse
layers** (identical geometry, never read), one arm per layer, which makes
cold cold by construction rather than by request.

### What this does to the throughput picture

| regime | per layer | × 42 | tok/s |
|---|---|---|---|
| fully cold, fault path | 212.1 ms | 8.91 s | **0.11** |
| fully cold, deep fetch | 90.0 ms | 3.78 s | **0.26** |
| 54 % hit (128 GiB budget), deep fetch | 52.3 ms | 2.20 s | **0.46** |
| fully warm | 17.8 ms | 0.75 s | **1.34** |

**About half of the 0.2 → 1.3 tok/s gap is recoverable by fetch shape
alone**, and the arithmetic now says where the other half is: at 54 % hit
the stage is 32 ms of I/O against 20 ms of compute, so **the bottleneck
flips from I/O to expert compute somewhere around this operating point.**
Past it, more fetch tuning buys progressively less and the FP8 expert
kernel — which has had no optimisation pass at all — becomes the subject.

That is the answer this rung was opened to get, so it closes here rather
than becoming a fetch subproject. `MappedAccess` gains no new variant
yet: the finding is that a *non-mapped* realisation wins, which is a
representation-and-loader question for the residency programme rather
than another access policy.

## 15. Disk

Internal volume has **393 GiB** free — it cannot hold a GLM container
(≈306 GB) beside anything else, and cannot hold Inkling's (≈495 GB) at all.
`model-drive` has **1.0 TiB** free. Containers get cut onto `model-drive`.
