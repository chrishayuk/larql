## Background

Qwen 3.6 27B (`qwen35`) and 35B-A3B (`qwen35moe`) use **Gated
DeltaNet** linear attention (NVlabs, arXiv 2412.06464) interleaved
with periodic full softmax-attention layers. The architecture is
NOT Mamba/S4; the GGUF metadata reuses `ssm.*` keys for code-reuse
with llama.cpp's recurrent-state cache infrastructure, but the math
is a delta-rule update on a **matrix-valued state**, not selective
scan with continuous-time discretization.

Ground truth: llama.cpp's `src/models/qwen35.cpp` +
`src/models/delta-net-base.cpp` (function `build_delta_net_base`,
class `llm_build_delta_net_base`).

## 1. Hybrid layer routing

```
recurrent_layer_arr[i] = ((i + 1) % full_attention_interval != 0)
```

With `full_attention_interval = 4` and `n_layer = 64`:

- Linear (DeltaNet) layers: i ∈ {0, 1, 2, 4, 5, 6, 8, …, 62}  (48 layers)
- Full-attn layers:         i ∈ {3, 7, 11, 15, …, 63}        (16 layers)

Attention layer is the **last** of each group of 4. Only the 16
attention layers maintain a softmax KV cache; the 48 linear layers
each maintain `(conv_state, recurrent_state)` instead.

## 2. Shapes (Qwen 3.6 27B)

| Symbol | Source | Value | Meaning |
|---|---|---|---|
| `n_embd` | `qwen35.embedding_length` | 5120 | residual stream |
| `n_layer` | `qwen35.block_count` | 64 | total layers |
| `n_head` (attn) | `qwen35.attention.head_count` | 24 | Q heads on attn layers |
| `n_head_kv` (attn) | `qwen35.attention.head_count_kv` | 4 | KV heads on attn layers (GQA 6:1) |
| `head_dim` (attn) | `qwen35.attention.key_length` | 256 | per-head dim |
| `d_inner` | `qwen35.ssm.inner_size` | 6144 | linear-attn value width |
| `d_conv` | `qwen35.ssm.conv_kernel` | 4 | depthwise Conv1D window |
| `ssm_state_size` | `qwen35.ssm.state_size` | 128 | head_k = head_v dim |
| `n_k_heads` | `qwen35.ssm.group_count` | 16 | K heads in DeltaNet |
| `n_v_heads` | `qwen35.ssm.time_step_rank` | 48 | V heads in DeltaNet |
| `key_dim` | derived | 2048 | `head_k * n_k_heads = 128*16` |
| `value_dim` | derived | 6144 | `head_v * n_v_heads = 128*48` |
| `conv_dim` | derived | 10240 | `2*key_dim + value_dim` |
| `ctx_len` | `qwen35.context_length` | 262144 | 256K (!) |
| `ffn_dim` | `qwen35.feed_forward_length` | 17408 | SwiGLU intermediate |

## 3. Tensor inventory per layer

### Linear-attention layer (48 of 64 layers)

| GGUF name | Shape | Purpose |
|---|---|---|
| `blk.N.attn_norm.weight` | `[5120]` | pre-mixer RMSNorm |
| `blk.N.attn_qkv.weight` | `[5120, 10240]` | fused QKV projection |
| `blk.N.attn_gate.weight` | `[5120, 6144]` | the Z gate projection |
| `blk.N.ssm_conv1d.weight` | `[4, 10240]` | depthwise Conv1D over QKV |
| `blk.N.ssm_dt.bias` | `[48]` | bias added to alpha |
| `blk.N.ssm_a` | `[48]` | per-head log-decay |
| `blk.N.ssm_beta.weight` | `[5120, 48]` | delta-rule learning-rate proj |
| `blk.N.ssm_alpha.weight` | `[5120, 48]` | pre-softplus gate proj |
| `blk.N.ssm_norm.weight` | `[128]` | post-mixer RMSNorm (head_v_dim) |
| `blk.N.ssm_out.weight` | `[6144, 5120]` | output projection |
| `blk.N.attn_post_norm.weight` | `[5120]` | post-mixer (pre-FFN) RMSNorm |
| `blk.N.ffn_gate.weight` | `[5120, 17408]` | SwiGLU gate |
| `blk.N.ffn_up.weight` | `[5120, 17408]` | SwiGLU up |
| `blk.N.ffn_down.weight` | `[17408, 5120]` | SwiGLU down |

### Full-attention layer (16 of 64 layers)

| GGUF name | Shape | Notes |
|---|---|---|
| `blk.N.attn_q.weight` | `[5120, 12288]` | **Q + per-head sigmoid gate fused** (Qwen3-Next style); `12288 = 256*24*2` |
| `blk.N.attn_k.weight` | `[5120, 1024]` | GQA: 4 KV heads × 256 |
| `blk.N.attn_v.weight` | `[5120, 1024]` |  |
| `blk.N.attn_q_norm.weight` | `[256]` | per-head RMSNorm on Q (head_dim, not n_embd) |
| `blk.N.attn_k_norm.weight` | `[256]` | per-head RMSNorm on K |
| `blk.N.attn_output.weight` | `[6144, 5120]` |  |

Plus the same `attn_norm`, `attn_post_norm`, and FFN trio as linear
layers.

### Globals

| GGUF name | Shape | Purpose |
|---|---|---|
| `token_embd.weight` | `[5120, vocab]` | input embeddings |
| `output_norm.weight` | `[5120]` | final RMSNorm |
| `output.weight` | `[5120, vocab]` | lm_head (may be tied to token_embd) |

## 4. The recurrence (per-token autoregressive)

Direct transcription of `build_delta_net_autoregressive`. Inputs:

- Pre-projected `q, k ∈ ℝ^{S_k × H_k}`, `v ∈ ℝ^{S_v × H_v}`
  (with `S_k = S_v = 128`, `H_k = 16`, `H_v = 48`)
- Scalar-per-head gate `g ∈ ℝ^{H_v}` and beta `b ∈ ℝ^{H_v}`
- State `S ∈ ℝ^{S_v × S_v × H_v}`

```
q  ← q / sqrt(S_k)                  # scale Q
g  ← exp(g)                         # per-head scalar decay
S  ← S * g                          # decay state (broadcasts over S_v×S_v)
# K is broadcast from H_k=16 to H_v=48 heads (3:1, GQA-like)
sk ← sum_{S_v}( S ⊙ k )             # [1, S_v, H_v]  = S^T @ k_h
d  ← (v - sk^T) * b                 # delta: [S_v, 1, H_v]
S  ← S + k ⊗ d^T                    # rank-1 update of S
o  ← sum_{S_v}( S ⊙ q )             # [1, S_v, H_v]
return (o, S)
```

Projections (each token):

```
QKV_mixed = wqkv @ x                # [10240]
Z         = wqkv_gate @ x           # [6144]
beta      = sigmoid( ssm_beta @ x ) # [48]
alpha     = ssm_alpha @ x           # [48]
g         = ssm_a * softplus(alpha + ssm_dt)   # [48]
QKV_conv  = SiLU( Conv1D(QKV_mixed, ssm_conv1d) )
q_raw, k_raw, v = split(QKV_conv, [2048, 2048, 6144])
q = L2Norm(reshape(q_raw, [128, 16]))
k = L2Norm(reshape(k_raw, [128, 16]))
v = reshape(v, [128, 48])
# k is repeat-interleaved 3× across heads (16 → 48) before the recurrence
```

Post-mixer:

```
o ← reshape(o, [128*48])
o ← RMSNorm(o, ssm_norm) * SiLU(Z)
y ← ssm_out @ o                     # → [5120]
```

That's the entire linear-attention block body. The L2-norm on Q/K
replaces softmax; this is what makes it linear attention.

## 5. State cache shape and memory budget

Per linear layer per sequence:

- **Conv state**: `[d_conv - 1, conv_dim] = [3, 10240]` = 30,720
  floats → 120 KiB fp32 / 60 KiB fp16.
- **Recurrent state**: `[head_v_dim, head_v_dim, n_v_heads] =
  [128, 128, 48]` = 786,432 floats → 3.0 MiB fp32 / 1.5 MiB fp16.

At 48 linear layers: ~144 MiB fp32 / ~72 MiB fp16 per active
sequence for DeltaNet state.

Plus the 16 attention-layer KV cache: 16 layers × 4 KV heads × 256
head_dim × 2 (K+V) × 2 (fp16) = 64 KiB per token. At full 256K
context that's 16 GiB just for the attention KV — same memory
problem as any other long-context model. Per-token cost is small;
ctx_len drives the budget.

## 6. Subtle correctness traps

1. **GQA in linear layers too.** K is broadcast 3× from 16 heads to
   48 to match V heads. Easy to miss if you treat K and V
   symmetrically.

2. **Q is fused with a per-head sigmoid gate** in full-attention
   layers. `attn_q` outputs `2 × n_embd_head × n_head` columns;
   split, sigmoid the second half, multiply into the attention
   output before `attn_output`.

3. **Q/K per-head RMSNorm in attention layers** has weight shape
   `[head_dim]` not `[n_embd]`. Applied after Q/K split but before
   RoPE.

4. **MRoPE with 4 sections**, NOT vanilla RoPE. Section sizes from
   `rope_dimension_sections` GGUF key. **Not applied in DeltaNet
   layers** — the conv1d is the implicit positional mixer.

5. **`ssm_dt` is a bias, not a weight.** GGUF key suffix is `.bias`
   not `.weight`. Added to `alpha` before softplus; no separate
   `dt_proj` matrix.

6. **No `A` matrix in HiPPO sense.** `ssm_a` is a per-head log-decay
   `[48]`, multiplied into `softplus(alpha+dt_bias)` to give scalar
   gate `g`. Despite name re-use, this isn't Mamba's `A_log`.

7. **Pre-norm AND post-norm around the mixer.** Both `attn_norm`
   (pre) and `attn_post_norm` (post-residual) RMSNorms exist —
   Gemma-2-style sandwich, NOT vanilla LLaMA pre-norm-only. The FFN
   residual adds to the *pre-post-norm* tensor.

8. **`time_step_rank` is misnamed.** Despite the name it's the
   number of V heads (= 48), not a time-step rank in the
   Mamba/dt_proj sense. Pure GGUF metadata legacy.

## 7. Why this isn't Mamba (and why that matters for testability)

The selective-scan literature suggests sequential dependency makes
the recurrence the bottleneck. For Gated DeltaNet:

- **Per-token compute**: `O(S_v² · H_v) = O(128² · 48) ≈ 786K FMA`
  per linear layer, 48 linear layers → ~37.7M FMA/token from
  DeltaNet alone. Plus FFN GEMMs (dominant) and the 16
  attention-layer matmuls.
- **Scalar Rust ceiling** (~5 GFLOPs/core single-thread, no SIMD):
  ~130 tok/s just for the DeltaNet math. Real bottleneck is the
  FFN matmuls.
- **Multi-threading is embarrassingly parallel across the 48
  heads** — each head has its own independent `[128, 128]` matrix
  state.
- **Realistic single-socket CPU target for 27B**: ~3-8 tok/s decode
  at fp16, dominated by FFN GEMMs.

The delta recurrence is NOT the bottleneck on CPU. Phase C's scalar
implementation should be parity-correct even if it's slow.

## 8. Parity oracle

Token-ID parity against llama.cpp, same pattern as
`target_forward_via_speculative_decode_matches_naive_64_seeds`:

- 64 seeded prompts × 32 decoded tokens each
- Compare top-1 argmax per position
- Tolerance: top-1 match per position; cosine ≥ 0.99 on softmax
- Gate: `LARQL_QWEN35_PARITY_LLAMA_CPP=/path/to/llama-cli` env var
  + `LARQL_QWEN35_PARITY_GGUF=/path/to/qwen3.6-27b.gguf`
- On mismatch: dump per-layer hidden states from both, diff the
  first divergence point. Common causes: wrong K-broadcast, wrong
  gate scale, missed pre-norm or post-norm.

## 9. Open questions for design review

- **Should the conv1d state be stored compressed?** It's only
  ~120 KiB per layer; probably not worth quantising until VRAM
  pressure shows it matters.
- **State eviction on context overflow?** Linear attention has
  unbounded effective context but a fixed-size state — no eviction
  needed. Attention layers still need a KV cap.
- **MoE variant first-PR scope.** Phase D as proposed assumes
  Phase C lands first; dense Qwen3.6-27B is the v1 target. If 35B
  is the priority instead, swap A/B/C to handle MoE FFN first;
  DeltaNet code is identical between the two.
