## ADDED Requirements

### Requirement: Gated DeltaNet linear-attention layer

The inference engine SHALL implement Gated DeltaNet linear-attention
layers as a peer of full softmax attention. Per-token forward
through a linear-attention layer SHALL produce output indistinguishable
from llama.cpp's `build_delta_net_autoregressive` reference within a
top-1 argmax and cosine ≥ 0.99 tolerance over 64 seeded prompts.

Per-layer state SHALL consist of:

1. A causal-Conv1D ring buffer of shape `[d_conv - 1, conv_dim]`
   over the concatenated QKV stream.
2. A matrix-valued recurrent state of shape
   `[head_v_dim, head_v_dim, n_v_heads]` per sequence.

The recurrence per token SHALL be:

```
q  ← q / sqrt(S_k)
g  ← exp(g)
S  ← S * g
sk ← sum_{S_v}( S ⊙ k )
d  ← (v - sk^T) * b
S  ← S + k ⊗ d^T
o  ← sum_{S_v}( S ⊙ q )
```

with K broadcast 3× from `n_k_heads` to `n_v_heads` heads.

#### Scenario: linear-attention output matches llama.cpp on 64 seeds

- **WHEN** the engine forwards a Qwen 3.6 27B linear-attention layer
  with weights loaded from `unsloth/Qwen3.6-27B-GGUF` and the prior
  layer's hidden state matches llama.cpp within cosine ≥ 0.99
- **THEN** the output hidden state SHALL match llama.cpp's
  `qwen35.cpp build_layer_attn_linear` output with cosine ≥ 0.99
  per token
<!-- test: unbacked -->

### Requirement: Hybrid layer routing via full_attention_interval

The engine SHALL route each layer to either the DeltaNet path or
the full-attention path per the rule:

```
is_linear(i) = ((i + 1) mod full_attention_interval) != 0
```

where `full_attention_interval` is read from the GGUF
`{arch}.full_attention_interval` key (default 4 if absent).

State management:

1. Linear layers SHALL maintain `(conv_state, recurrent_state)`
   pairs; the engine SHALL NOT allocate KV slabs for them.
2. Full-attention layers SHALL maintain the existing K/V cache
   slabs; the engine SHALL NOT allocate DeltaNet state for them.
3. The two state types SHALL coexist in a single hybrid cache
   wrapper `DeltaNetHybridCache { kv_layers, deltanet_layers,
   layer_kinds }` indexed by global layer number.

#### Scenario: routing matches Qwen 3.6 27B's 48-linear-16-attention pattern

- **WHEN** a Qwen 3.6 27B model (`n_layer=64`,
  `full_attention_interval=4`) is loaded
- **THEN** layers {3, 7, 11, …, 63} SHALL be full-attention (16
  total) AND all other layers SHALL be DeltaNet (48 total)
<!-- test: unbacked -->

### Requirement: Q5_K dequantisation for GGUF tensor loading

The GGUF loader SHALL dequantise Q5_K (GGML type id 13) tensors to
f32 with the 176-byte super-block layout (2 + 2 + 12 + 32 + 128):
f16 d, f16 dmin, 12-byte packed scales+mins (6-bit each, same
`get_scale_min_k4` packing as Q4_K), 32 bytes of high-bits (1 per
element), 128 bytes of low-nibble quants.

This dequant unblocks loading from unsloth's Q4_K_M and Q4_K_S
quants, which mix Q4_K and Q5_K tensors per llama.cpp's mid-grade
quant scheme.

#### Scenario: Q5_K dequant round-trips against synthetic blocks

- **WHEN** `dequantize_q5_k` is invoked on a block with d=1.0, all
  scales=1, all qs nibbles=0xF, and qh bits varying per element
- **THEN** the output f32 SHALL be 15.0 (qh bit 0) or 31.0 (qh bit
  1) per element with absolute error ≤ 1e-5
<!-- test: speculative-tree-out-of-scope -->

### Requirement: Qwen 3.6 architecture detection

The architecture-detection layer SHALL map GGUF arch strings to:

1. `"qwen35"` → `Qwen35Arch` (dense Gated DeltaNet hybrid)
2. `"qwen35moe"` → `Qwen35MoeArch` (MoE variant with top-8-of-N
   routing per layer)

`Qwen35Arch` SHALL expose the SSM metadata fields
(`full_attention_interval`, `ssm_state_size`, `ssm_inner_size`,
`ssm_dt_rank` as `n_v_heads`, `ssm_group_count` as `n_k_heads`,
`ssm_conv_kernel` as `d_conv`, `rope_dimension_sections`) via the
existing `ModelArchitecture` trait extension points.

#### Scenario: detect.rs maps qwen35 arch string

- **WHEN** a GGUF with `general.architecture: "qwen35"` is opened
- **THEN** `detect_arch` SHALL return a `Qwen35Arch` instance with
  `n_layer=64`, `full_attention_interval=4`, `n_v_heads=48`,
  `n_k_heads=16` populated from the GGUF metadata
<!-- test: unbacked -->

### Requirement: Per-layer tensor extraction for qwen35

The extraction pipeline (`larql convert gguf-to-vindex`) SHALL
extract the qwen35 per-layer tensor set:

**Linear layers (48 layers in 27B):**
- `attn_norm`, `attn_qkv`, `attn_gate`, `ssm_conv1d`,
  `ssm_dt` (bias), `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm`,
  `ssm_out`, `attn_post_norm`, `ffn_gate`, `ffn_up`, `ffn_down`

**Full-attention layers (16 layers in 27B):**
- `attn_norm`, `attn_q` (fused Q+gate, output dim
  `2*n_embd_head*n_head`), `attn_k`, `attn_v`, `attn_q_norm`
  (shape `[head_dim]`), `attn_k_norm`, `attn_output`,
  `attn_post_norm`, `ffn_gate`, `ffn_up`, `ffn_down`

Missing tensors SHALL cause extraction to fail loudly (not silently
produce a vindex that misbehaves at inference time).

#### Scenario: extract gguf-to-vindex on Qwen3.6-27B-Q4_K_S succeeds

- **WHEN** `larql convert gguf-to-vindex Qwen3.6-27B-Q4_K_S.gguf
  --output qwen3.6-27b-vindex --level inference` is run
- **THEN** the command SHALL exit 0 AND the resulting vindex
  SHALL contain the full per-layer tensor set above (verified via
  `larql describe`)
<!-- test: unbacked -->
