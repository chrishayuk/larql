## Why

Qwen 3.6 27B (`qwen35`) and 35B-A3B MoE (`qwen35moe`) are hybrid
architectures combining **Gated DeltaNet** linear-attention layers
with periodic full-attention layers (3:1 ratio, attention at layer
`(i+1)%4 == 0`). They are NOT Mamba/S4/selective-scan despite the
GGUF metadata reusing `ssm.*` keys for compatibility with llama.cpp's
recurrent-state cache infrastructure.

Today `larql-models` has zero support for either DeltaNet OR the
hybrid layer routing. Extraction silently drops the `ssm_*` /
`attn_gate` tensors; inference would produce garbage. This change
adds:

1. GGUF extraction for the new tensor names (`attn_qkv`, `attn_gate`,
   `ssm_conv1d`, `ssm_dt` (bias), `ssm_a`, `ssm_beta`, `ssm_alpha`,
   `ssm_norm`, `ssm_out`).
2. A `Qwen35Arch` / `Qwen35MoeArch` architecture handler with
   per-layer `is_linear` routing via `full_attention_interval`.
3. A scalar CPU forward kernel implementing the Gated DeltaNet
   recurrence (`build_delta_net_autoregressive` from llama.cpp).
4. A `DeltaNetStateCache` that stores the per-layer
   `[head_v_dim, head_v_dim, n_v_heads] = [128, 128, 48]` matrix-
   valued recurrent state, analogous to but distinct from `KvCache`.
5. Token-ID parity validation against llama.cpp on 64 seeds (the
   load-bearing correctness gate, same pattern as
   `target_forward_via_speculative_decode_matches_naive_64_seeds`).

## What this change ships

The work splits into 6 phases, each a separate PR. The first three
land extraction + architecture handler + CPU forward and produce a
running (slow) implementation. Phases 4-6 add the GPU path and bench
gates.

### Phase A — GGUF extraction + tensor mapping (~250 LoC)

- New tensor-name constants in `larql-models/src/loading/gguf.rs`
  for the qwen35 set (`ssm_conv1d`, `attn_qkv`, `attn_gate`, etc.).
- Wire `qwen35` and `qwen35moe` arch strings into the loader's arch
  detection. (Currently `gguf.rs:362` only normalises `"qwen"|"qwen2"`.)
- Extract level: `inference` should pull all SSM + attn + gate
  tensors plus the conventional FFN trio. MoE variant pulls the
  expert array (256 experts × per-expert SwiGLU trio per layer for
  the 35B-A3B).
- `index.json` extensions: store `full_attention_interval`,
  `ssm_state_size`, `ssm_inner_size`, `ssm_dt_rank` (= n_v_heads),
  `ssm_group_count` (= n_k_heads), `ssm_conv_kernel`,
  `rope_dimension_sections` (for MRoPE).

**Validation:** `larql convert gguf-to-vindex` on
`Qwen3.6-27B-Q4_K_S.gguf` succeeds; resulting vindex's
`index.json` round-trips the SSM metadata.

### Phase B — `Qwen35Arch` architecture handler (~200 LoC)

- New file `crates/larql-models/src/architectures/qwen35.rs` mirroring
  `qwen.rs`'s pattern. Critical differences from `QwenArch`:
  - `is_linear_attention_layer(layer)` predicate driven by
    `full_attention_interval`.
  - Per-layer weight-name mapping diverges between linear and
    attention layers (linear uses `attn_qkv`/`attn_gate`/`ssm_*`;
    attention uses `attn_q`/`attn_k`/`attn_v`/`attn_output` with the
    Qwen3-Next fused-Q+gate quirk on `attn_q`).
- `Qwen35MoeArch` extends with `num_experts: 256`,
  `expert_used_count: 8`, `expert_feed_forward_length: 512`.
- Architecture-detection in `detect.rs`: map `qwen35` arch string
  to `Qwen35Arch`, `qwen35moe` to `Qwen35MoeArch`.

**Validation:** `cargo run -- describe output/qwen3.6-27b-vindex`
prints the layer layout correctly (48 linear + 16 attention).

### Phase C — Scalar Rust CPU forward (~700 LoC)

The load-bearing implementation. Components:

1. **Helpers** (mostly already exist): RMSNorm, SiLU, sigmoid,
   softplus, L2-norm. Add: causal depthwise Conv1D over a
   `[d_conv=4]` window with state ring-buffer.
2. **DeltaNet block** (the recurrence in `design.md` §4):
   - Project: QKV-mixed (`wqkv @ x`), Z-gate (`wqkv_gate @ x`),
     `beta = sigmoid(ssm_beta @ x)`, `alpha = ssm_alpha @ x`,
     `g = ssm_a * softplus(alpha + ssm_dt)`.
   - Causal Conv1D over QKV stream, SiLU.
   - Split into Q/K/V, L2Norm Q and K, reshape into heads.
   - State update: `S ← S * g; sk = sum(S⊙k); d = (v - sk^T) * b;
     S ← S + k ⊗ d^T; o = sum(S⊙q)`.
   - Post-mixer: flatten heads, RMSNorm, gate by `SiLU(Z)`, project
     `ssm_out`.
3. **Full-attention block** with Qwen3-Next quirks:
   - Fused Q + per-head sigmoid gate projection
     (`attn_q` outputs `head_dim × n_head × 2`).
   - Per-head RMSNorm on Q and K (weight shape `[head_dim]`).
   - MRoPE with 4 sections (driven by `rope_dimension_sections`).
   - GQA: 4 KV heads broadcast to 24 Q heads.
4. **Layer router** dispatching linear vs full attention per
   `(i+1) % 4 == 0`.
5. **`DeltaNetStateCache`** holding per-layer
   `(conv_state: [3, 10240], recurrent_state: [128, 128, 48])`
   plus the per-layer KV slabs for the 16 attention layers.

For prefill, run the autoregressive kernel `n_tokens` times in a
loop. The chunking algorithm in `build_delta_net_chunking` (CS=64,
triangular solve, cumulative-sum-of-gate) is a later optimisation —
not in scope for Phase C.

**Validation:** Token-ID parity against `llama.cpp` on 64 seeded
prompts. Tolerance: top-1 argmax must match per position; cosine ≥
0.99 per-token softmax. Gated on
`LARQL_QWEN35_PARITY_LLAMA_CPP=/path/to/llama-cli` env var.

### Phase D — MoE variant (`qwen35moe`) (~150 LoC)

Identical SSM + attention layers; only the FFN differs:
top-8-of-256 expert routing per layer. The existing `mixtral.rs`
expert dispatch is the closest reference (with adjustments for the
larger expert count and the `expert_shared_feed_forward_length`).

**Validation:** Parity test on Qwen3.6-35B-A3B GGUF, same gates as
Phase C plus an expert-routing-distribution check (top-8 hit rate
≥ 95% of llama.cpp's selection per layer per token).

### Phase E — CUDA acceleration (~1500 LoC, separate spec change)

Out of scope here; reserved for a follow-up `cuda-deltanet-kernels`
change. Will adopt the same Q4_K weight format the existing CUDA
path uses; the new kernels are: depthwise Conv1D-with-state,
delta-rule rank-1 matrix update, fused-Q-gate attention. Phase C's
scalar Rust is the parity oracle.

### Phase F — VRAM + tok/s bench vs llama.cpp (~300 LoC harness)

The deferred benchmark goal from this session's earlier work:
quantify tok/s parity and the FFN-offload VRAM-headroom advantage.
Three configs (llama.cpp all-on-GPU, larql all-on-GPU after Phase
E, larql `--ffn` remote). Sweet-spot target: VRAM headroom ≥ 2×
llama.cpp's at the same context length, tok/s within 1.2× parity.

## Capabilities

### Added

- `inference-gated-deltanet` — new capability covering the DeltaNet
  recurrence + hybrid layer routing. Reusable for future linear-
  attention models (RWKV-7, Mamba-2 if we add it, custom variants).

### Modified

- `inference-residual-engine` — adds linear-attention dispatch
  alongside the existing full-attention path.

## Impact

- **Risk: high.** This adds a fundamentally new architecture family
  to the codebase, not just a new weight format. Six phases over
  multiple weeks. Each phase has a clear validation gate so the
  blast radius of any single PR is bounded.
- **Blast radius if wrong:** silent output corruption on Qwen3.6
  models specifically. Other models (Gemma, Llama, Qwen3 non-hybrid)
  unaffected — the new code paths gate on arch string match.
- **Code change: ~2000-2500 LoC across Phases A-D** (CPU path).
  Phase E adds another ~1500 LoC of CUDA.

## Estimated effort

- Phase A (extraction): 1-2 days
- Phase B (arch handler): 1 day
- Phase C (CPU forward + parity gate): 4-7 days (the hard one)
- Phase D (MoE): 2-3 days
- Phase E (CUDA): 1-2 weeks (separate change)
- Phase F (bench): 1-2 days post-E

Total to working CPU inference + MoE: ~10-13 days focused work.

## Out of scope

- **CUDA acceleration** — separate change.
- **Chunked prefill** — Phase C uses the autoregressive kernel in a
  loop. Throughput on long prompts is poor but correctness is
  preserved. Add chunking in a follow-up.
- **RWKV-7 / Mamba-2 / other linear-attention variants** — the
  capability `inference-gated-deltanet` is named generically so
  future linear-attention work can extend it, but no other model is
  in scope for this change.
- **Qwen 3.5** — separate model line (same arch family, different
  weights). If we want Qwen 3.5 support post-Phase-D, it should be
  another change-level edit, mostly tensor-name registration.
