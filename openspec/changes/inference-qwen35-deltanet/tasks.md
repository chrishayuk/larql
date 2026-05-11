## Phase 0 — Q5_K dequant unblock (this PR)

Bounded prerequisite for any extraction. Lands independently because
it's clean, testable, and useful even before the larger architecture
work.

- [x] 0.1 New file `crates/larql-models/src/quant/ggml/q5_k.rs` with
      `dequantize_q5_k`. 176-byte super-block, same scale/min packing
      as Q4_K, 5 bits per element (low 4 bits from `qs`, high 1 bit
      from `qh`).
- [x] 0.2 Wire `TYPE_Q5_K` (id 13) into `tensor_data_size` and
      `dequantize` dispatch in `quant/ggml/mod.rs`. Add
      `Q5_K_BLOCK_BYTES = 176` constant.
- [x] 0.3 Unit tests: zero-block, uniform mid-value, high-bit lifts
      to 31, multi-block, rejects short / misaligned. 6 tests.
- [x] 0.4 Verified `larql convert gguf-to-vindex
      Qwen3.6-27B-Q4_K_S.gguf` now gets past dequant; fails downstream
      on architecture handler (out of scope here).

## Phase A — qwen35 GGUF extraction (~250 LoC)

Branch: `feat/qwen35-extraction`. Goal: `larql convert
gguf-to-vindex` produces a vindex with the full qwen35 tensor set.
No inference yet.

- [ ] A.1 Add `qwen35` and `qwen35moe` to the arch-string normaliser
      in `crates/larql-models/src/loading/gguf.rs::362` (currently
      only maps `qwen`|`qwen2`).
- [ ] A.2 Extend `VindexModelConfig` (`crates/larql-vindex/src/config/model.rs`)
      with: `full_attention_interval`, `ssm_state_size`,
      `ssm_inner_size`, `ssm_dt_rank` (= `n_v_heads`),
      `ssm_group_count` (= `n_k_heads`), `ssm_conv_kernel`
      (= `d_conv`), `rope_dimension_sections`. All `Option<…>` so
      other architectures don't have to set them.
- [ ] A.3 New tensor-name constants for the SSM/DeltaNet set:
      `attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_dt`,
      `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm`, `ssm_out`.
      Add to `larql-vindex` filename constants and the GGUF→vindex
      tensor-name mapper.
- [ ] A.4 `tokenizer.json` fallback: the GGUF embeds tokenizer
      metadata; extract it via `GgufFile::tokenizer()` (new helper)
      when no `tokenizer.json` sits next to the GGUF. Avoids
      requiring users to download two files.
- [ ] A.5 Smoke test: extract `Qwen3.6-27B-Q4_K_S.gguf` → produced
      vindex has 64 sets of per-layer tensors with the right names
      and shapes per the design.md §3 inventory.

## Phase B — Qwen35Arch architecture handler (~200 LoC)

Branch: `feat/qwen35-arch-handler`. Goal: the `ModelArchitecture`
trait surface reports the right layer-kind distribution and tensor
keys, so downstream code (forward kernels) can dispatch correctly.

- [ ] B.1 New file `crates/larql-models/src/architectures/qwen35.rs`.
      Implement `ModelArchitecture` trait. Critical methods:
      `is_linear_attention_layer(i)`,
      `linear_attention_tensor_keys(i)`,
      `full_attention_tensor_keys(i)`, `is_hybrid()` returns true.
- [ ] B.2 `Qwen35MoeArch` in the same file. Inherits from
      `Qwen35Arch` and overrides `is_moe()` → true,
      `num_experts()` → 256, `num_experts_per_token()` → 8,
      `expert_feed_forward_length()` → 512.
- [ ] B.3 `detect.rs`: map `qwen35` → `Qwen35Arch`, `qwen35moe` →
      `Qwen35MoeArch`. Update test cases.
- [ ] B.4 `larql describe <vindex>` correctly reports layer kinds.

## Phase C — Scalar Rust CPU forward (~700 LoC, the hard one)

Branch: `feat/qwen35-cpu-forward`. Goal: working (slow) inference on
CPU, validated against llama.cpp.

- [ ] C.1 New file `crates/larql-inference/src/forward/deltanet.rs`.
      Pure-Rust scalar implementation of the linear-attention
      block per design.md §4.
- [ ] C.2 New file `crates/larql-inference/src/forward/qwen35_attn.rs`
      for the full-attention layer with the Qwen3-Next quirks
      (fused Q+gate, per-head Q/K RMSNorm, MRoPE).
- [ ] C.3 `DeltaNetStateCache` struct in
      `crates/larql-inference/src/attention/deltanet_state.rs`
      holding `Vec<DeltaNetLayerState>` where each state is
      `(conv_state: ArrayD<f32>, recurrent_state: ArrayD<f32>)`.
- [ ] C.4 `DeltaNetHybridCache { kv_cache: KvCache, dn_state:
      DeltaNetStateCache, layer_kinds: Vec<LayerKind> }` wrapper.
- [ ] C.5 Layer-router in the forward pipeline dispatches per
      `Qwen35Arch::is_linear_attention_layer`.
- [ ] C.6 MRoPE helper with 4 sections (new code, distinct from the
      existing single-section RoPE).
- [ ] C.7 Parity oracle: `test_qwen35_token_parity_vs_llama_cpp` —
      64 seeded prompts × 32 tokens each, top-1 match + cosine ≥
      0.99 per token. Gated on
      `LARQL_QWEN35_PARITY_LLAMA_CPP=/path/to/llama-cli` env var.
- [ ] C.8 Stop-ship gate: if the parity test fails, dump per-layer
      hidden-state diff vs llama.cpp's tensor-dump output. Common
      failure modes: wrong K broadcast, wrong gate scale, missed
      pre/post-norm, MRoPE section indexing.

## Phase D — MoE variant (`qwen35moe`) (~150 LoC)

Branch: `feat/qwen35moe-ffn`. Goal: Qwen3.6-35B-A3B works.

- [ ] D.1 Top-8-of-256 expert routing in
      `crates/larql-inference/src/forward/moe.rs` (or extend the
      existing Mixtral-style helper).
- [ ] D.2 Handle `expert_shared_feed_forward_length` (a few experts
      may share an FFN intermediate per architecture spec).
- [ ] D.3 Parity test on Qwen3.6-35B-A3B GGUF. Same gates as Phase
      C plus expert-routing-distribution check.

## Phase E — CUDA acceleration (separate change)

Out of scope here. Reserved for a follow-up `cuda-deltanet-kernels`
change after Phase D lands and proves out scalar correctness.
Will include: depthwise Conv1D-with-state kernel, delta-rule
rank-1 matrix update kernel, fused-Q-gate full-attention kernel.
Phase C's scalar Rust serves as the parity oracle.

## Phase F — VRAM + tok/s bench vs llama.cpp (~300 LoC harness)

The deferred deliverable from session 2026-05-11. Three-config head-
to-head: llama.cpp all-GPU, larql all-GPU (post Phase E), larql
`--ffn` remote.

- [ ] F.1 `bench-harness/run_bench.sh` end-to-end script (skeleton
      written in this session, runnable on Gemma 3 4B today, ready
      for Qwen 3.6 once Phase E lands).
- [ ] F.2 `nvidia-smi --query-gpu=memory.used` background poller
      capturing peak VRAM.
- [ ] F.3 System-RAM delta via `/proc/PID/status` VmRSS.
- [ ] F.4 Three prompts (chat-style, JSON-structured, RAG) to span
      the workload spectrum.
- [ ] F.5 Sweet-spot metric: **VRAM headroom = 24576 − peak_VRAM
      (MiB)**. Target: larql `--ffn` headroom ≥ 2× llama.cpp's at
      the same context length.

## Validation

- [x] V.0 `cargo test -p larql-models --lib quant::ggml::q5_k`
      passes (6/6 in Phase 0).
- [ ] V.1 `openspec validate inference-qwen35-deltanet --strict` passes.
- [ ] V.2 `make traceability-check` passes after spec.md additions.
- [ ] V.3 Phase A-D PRs each pass their phase-local validation gates
      before merge. Phase E (CUDA) gated on D being green.

## Out of scope

- **Chunked prefill**: Phase C uses the autoregressive kernel in a
  loop. Throughput on long prompts is poor; correctness is preserved.
  Add chunking in a follow-up after Phase D.
- **Other linear-attention architectures** (RWKV-7, Mamba-2): the
  capability `inference-gated-deltanet` is named generically so
  future linear-attention models can extend it, but none are in
  scope here.
- **Qwen 3.5** support (same family, different model line): a
  separate small change post-Phase D if needed; should be mostly
  tensor-name registration.
