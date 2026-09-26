# Metal shader inventory + retention survey

**Class: ARCHIVE.** This inventory is a snapshot, not a current count or dispatch contract. Current [Metal sources](../../larql-compute-metal/src/shaders/) and [backend controls](../../larql-compute-metal/README.md) are the implementation authority.

**Date**: 2026-05-09
**Purpose**: Per-shader audit of `crates/larql-compute-metal/src/shaders/` under the model-agnosticity constraint — shaders need to support not just Gemma 3/4 but Llama 1/2/3, Mistral, DeepSeek, Qwen, and other transformer LM families. The previous Gemma-A/B-falsification cleanup pattern (e.g. NR2 deletion) over-prioritised current-Gemma performance and risks deleting capability that other models need.

This doc is the **retention rationale** for each shader: what it does, which model families it serves, and whether it's currently load-bearing or kept as defensible capability.

## Methodology

1. Read each `.rs` shader file's doc-comment (10–50 lines per file) to capture purpose + author intent.
2. Grep for dispatch sites in `metal/decode/`, `metal/ops/`, `metal/stages/`, `metal/trait_impl/`, `metal/prefill.rs`, `metal/decode_hybrid.rs` to determine production / opt-in / fallback / dead status.
3. Grep tests in `crates/larql-compute/tests/` and `crates/larql-inference/tests/` for kernel name + `LARQL_*` env vars.
4. Cross-reference with `ROADMAP.md` and `PERFORMANCE.md` to identify pinned roadmap tracks.

## Inventory by op-class

### Generic infrastructure (model-agnostic)

| shader | role | status | model-family applicability |
|---|---|---|---|
| `f32_gemv` | f32 mat-vec for lm_head, argmax, top-k | production | All transformer LMs (any tied/untied lm_head). Has `argmax` and `topk` variants. |
| `f16_gemv` | f16 mat-vec for tied-embedding lm_head | production fallback | All transformer LMs with f16 embeddings. Tried as primary lm_head 2026-05-09, regressed (Q4_K is faster); kept as fallback. |
| `sgemm` | f32 tiled matmul | fallback / opt-in | All models (generic prefill). Used when no quantised matmul is available. |
| `sgemm_transb` | f32 tiled matmul with transposed B | fallback / opt-in | All models. Specific shape variant for ops where B is transposed in-place. |
| `quantize_q8` | f32 → int8 + per-block scale | fallback | Layer-chaining for Q8-input kernels. Used when input format mismatches the Q4_K/Q6_K matvec input expectation. |
| `fused_ops` | Fused RMS-norm + Q8 quantise (one-TG) | production | All RMS-norm + Q8-input models (Llama Q4_0 path, some Mistral variants). One-shot dispatch. |
| `residual_inject` | Residual-stream skip-connection helpers (cached residual copy / residual add / scale-vector / standalone rms_norm) | production | All transformer LMs. Used for skip connections after attn/FFN; also load-bearing for the **interpretability install_edge / WASM-in-FFN research path**. |

### Q4_K family — Llama 1/2/3, Mistral, Gemma 3/4, Qwen, most modern Q4_K-quantised models

| shader | role | status | retention reason |
|---|---|---|---|
| `q4k_matvec` | Q4_K mat-vec (4sg) | production | Core Q4_K decode path — used by every Q4_K-quantised model. |
| `q4k_matvec_8sg` | Q4_K mat-vec (8sg, default since 2026-04-28) | production-default for `q4k_matvec_pipeline` | The 8sg dispatch is the production default for `q4k_matvec_pipeline` (per `LARQL_Q4K_MATVEC_8SG`). Same kernel-shape as 4sg, +5.2% end-to-end. |
| `q4k_matvec_stride32` | Q4_K mat-vec with stride-32 reduction | opt-in (`LARQL_LM_HEAD_SKIP_Q4K=1` indirectly) | Was a correctness workaround for the lm_head dispatch-geometry bug (2026-05-01); after the 2026-05-02 fix this is a diagnostic fallback. **Kept** because the stride-32 reduction tree is what would be needed if a future model surfaces an argmax-flip class of bug. |
| `q4k_matmul` | Q4_K matmul (gemm) for prefill | production-default (callable via `MetalBackend::q4k_matmul`) | Wired-in attempts (FFN gate+up, O-proj) failed twice (2026-04-28, 2026-05-09) due to L1-thrash on long prompts. **Kept** because the kernel + parity tests are the foundation for D-PREFILL-MM2 (the `simdgroup_matrix` rewrite). |
| `q4k_qkv_proj` | Fused Q+K+V Q4_K (8 rows/TG, 256 threads) | production | Uniform-Q4_K QKV path. Used when all of Q/K/V are Q4_K (not the Gemma-3/4 mixed-quant path). |
| `q4k_qkv_proj_v2` | Q4_K QKV with `(ix, j, sh)` lane decomposition | **dead** — no dispatch site | Was an experimental alternative for small-K cases. Never promoted; v1 stayed production. **Deletion candidate** — see "Recommendations" below. |
| `q4k_q6k_qkv_proj` | Mixed Q4_K (Q/K) + Q6_K (V) QKV | production | Gemma 3/4, Llama 2, Mistral with ollama-convention extracts. Has both `q4k_q6k_qkv_proj` (non-fused norm; production since 2026-05-09 ADR-016) and `NORMED_SHADER` variant (norm rolled in; opt-in `LARQL_QKV_FUSED=1`). |
| `q4k_ffn_gate_up` | Q4_K gate+up FFN (4sg, original) | production fallback | Older 4sg kernel. Default until 2026-04-28 when 8sg landed. Kept as opt-out (`LARQL_GATE_UP_8SG=0`). |
| `q4k_ffn_gate_up_8sg` | Q4_K gate+up FFN (8sg) | production-default | Production FFN gate+up since 2026-04-28. +2.1% end-to-end vs 4sg on Gemma 3 4B. Same kernel-shape works for any Q4_K gate+up model. |
| `q4k_ffn_gate_up_coop` | Q4_K gate+up with cooperative scale-loading | opt-in (`LARQL_GATE_UP_COOP=1`) | Tried 2026-05-01, null end-to-end on Gemma 3 4B (instance #2 of ADR-015). **Retained** — the cooperative scale-load could win on different K dimensions where amortisation math goes differently (e.g. larger Llama variants with K=4096+). |
| `q4k_ffn_gate_up_f16acc` | Q4_K gate+up with f16 inner accumulator | opt-in (`LARQL_F16_ACC=1`) | Tried 2026-04-28, kernel-isolated 1.79× win, end-to-end at parity (instance #1 of ADR-015). **Retained** — could win on M5+/A19 hardware where ALU/bandwidth balance shifts, or in sustained-load scenarios with thermal pressure. |
| `q4k_geglu_down` | Q4_K fused GEGLU + down projection | production | All gated-FFN Q4_K models (Llama, Mistral, Qwen, Gemma 3/4 — SiLU and GELU-tanh variants). Has separate kernels for SiLU vs GELU-tanh activations. |

### Q4_KF family — llama.cpp-exact GGUF compatibility

| shader | role | status | retention reason |
|---|---|---|---|
| `q4kf_qkv_proj` | Q4_KF (144-byte GGUF layout) fused QKV | production for Q4_KF-format vindexes | Allows direct loading of llama.cpp/ollama GGUFs without re-quantisation. Critical for cross-engine interop. |
| `q4kf_ffn_gate_up` | Q4_KF fused FFN gate+up | production for Q4_KF | Same — Q4_KF-format compatibility path. |

### Q6_K family — typically paired with Q4_K (Gemma 3/4 conventional, Llama 2 with Q6_K down, Mistral)

| shader | role | status | retention reason |
|---|---|---|---|
| `q6k_matvec` | Q6_K mat-vec (4sg) | **production default** | Same kernel shape works for any Q6_K-quantised matrix. |
| `q6k_matvec_8sg` | Q6_K mat-vec (8sg) | **opt-in** via `LARQL_Q6K_8SG=1` | Bit-identical to 4sg, better occupancy — but the win did not translate end-to-end on M3 Max, so it stays opt-in. |
| `q6k_geglu_down` | Q6_K fused GEGLU + down (SiLU) | production | All SiLU-FFN models with Q6_K down (Llama 2 / Mistral with Q6_K down convention). |
| `q6k_geglu_gelu_tanh_down_cached` | Q6_K fused GEGLU + GELU-tanh down + TG-cached activation | opt-in (`LARQL_FUSED_DOWN=1`, blocked) | Production NaN bug on Gemma 3 4B / 31B production weights (D-FFN-FUSE blocked). Kernel correctness verified on synthetic data. **Retained** — the TG-cached activation pattern is sound and applicable to any GELU-tanh FFN once the data-shape bug is found. |

### Q4_0 / Q8_0 family — Llama 1/2 legacy, Mistral early variants, models trained with older quants

| shader | role | status | retention reason |
|---|---|---|---|
| `q4_matvec_v4` | Q4_0 mat-vec (uint4 wide-load) | production | Production Q4_0 path. Used by Llama 1/2 with Q4_0-quantised vindexes. |
| `q4_f32_matvec` | Q4_0 weights × f32 input mat-vec | production | Q4_0 path when input is f32 (rare — sparse activation patterns). |
| `q4_vecmat` | Q4_0 transpose matmul (scatter-accumulate) | production fallback | Gradient/training-style ops; GPU-hostile but functionally complete. |
| `q4_sparse_matvec` | Q4_0 mat-vec with row-index sparsity | production | Sparse activation paths — useful for **DeepSeek-V3 / Qwen MoE candidate routing** where only a subset of rows participate. |
| `q8_matvec` | Q8_0 mat-vec | production fallback | Q8_0 weights. Used in some Mistral and Llama variants. |
| `q8_attn_proj` | Q8_0 fused QKV projection | production fallback | High-precision attention paths where Q8 V is preferred over Q4_K V. Has 1 kernel-handle-contract test, no other dispatch refs in current production paths — but the kernel handle is wired and the GGML convention exists. |

### Attention family

| shader | role | status | retention reason |
|---|---|---|---|
| `fused_attention` | RoPE + optional QK-norm + GQA + optional softcap + causal mask, fused | production prefill | All transformer LMs. Handles Gemma 2 softcap, Gemma 3/4 QK-norm, GQA — most arch variations in one kernel. Used by `metal/prefill.rs` and `stages/attention.rs`. |
| `kv_attention` | Decode attention from KV-cache (T ≤ 1024 / long ≤ 4096) | production decode | All transformer LMs. Has both standard and long-window variants for Gemma 4 global-attention layers. |
| `kv_append_attend_fused` | KV-cache append + decode attention, one dispatch | production for Gemma 3/4 has-post-norms path | Used when the post-norms pipeline is active. Saves 1 dispatch/layer. Pattern applicable to other models with post-norm-on-attention. |
| `causal_attention` | Single-thread-per-(head_dim, position) attention for small seq_len | fallback | Fallback for small prefill (seq_len ≤ 64). Useful for chat-style very-short-prompt warmup. |
| `attn_fused` | Multi-stage attention fusion (qk_norm + rope + kv_append + attend) | opt-in (`LARQL_FUSED_ATTN=1`) | First attempt regressed −1.45 ms (2026-05-01) due to TG-count collapse 12→8. **Pinned by D-ATTN-MTG roadmap** — multi-TG-per-head retry is the next decode-side gap closer (~+5–8 tok/s estimated). Direct analog of llama.cpp's `kernel_flash_attn_ext_vec_reduce`. |

### Norm + RoPE family

| shader | role | status | retention reason |
|---|---|---|---|
| `qk_norm` | Per-head RMS-norm on Q and K | production | Gemma 3/4 attention QK pre-norm. Required to prevent softmax NaN overflow on these models. |
| `qk_norm_rope_fused` | Fused QK-norm + RoPE | production | Gemma 3/4 (or any QK-norm + RoPE model). Saves 1 dispatch/layer. |
| `v_norm` | Parameter-free RMS-norm on V | production for Gemma 4 | Gemma 4 only — V is normalised before attention with no learned weight. |
| `qk_norm` (and friends) for non-Gemma archs | RMS-norm only on K (no QK) | covered by `residual_inject::rms_norm` and `fused_ops` | Most non-Gemma transformer LMs (Llama, Mistral, Qwen) don't use QK-norm — they use plain RMS-norm before QKV. |
| `layer_norm` | LayerNorm (mean + variance) | production | StarCoder2, GPT-2, BERT-style models. Used when `norm_type == LayerNorm`. |
| `post_attn_residual_norm_store` | Post-attn RMS-norm + residual + pre-FFN norm + store (triple fusion) | production for Gemma 3/4 has-post-norms | 2026-05 fusion-wave win (~0.43 ms cumulative). Pattern applicable to any post-norm architecture. |
| `post_ffn_norm_residual_add` | Post-FFN RMS-norm + residual add | production for Gemma 3/4 has-post-norms | Same fusion family, different position. |
| `rope` | RoPE (single vector + matrix in-place + partial rotation) | production | All RoPE-using models. Note: we **don't have** `rope_neox`, `rope_multi`, `rope_vision` variants that llama.cpp ships — adding these would be needed for some models. Tracked as gap. |

### Activation family

| shader | role | status | retention reason |
|---|---|---|---|
| `activation` | SiLU / GELU-tanh / GELU-erf (non-gated) | production | StarCoder2, GPT-2, Phi, BERT — non-gated FFN architectures. |
| `geglu` | Gated FFN activation (gate × up, with SiLU or GELU) | production | All gated-FFN models (Llama, Mistral, Gemma, Qwen). Has SiLU and GELU-tanh variants. |

### KV cache compression (experimental)

| shader | role | status | retention reason |
|---|---|---|---|
| `turboquant_decode` | 4-bit KV decompression (centroid lookup + inverse WHT) | opt-in / not yet wired | Long-context KV memory pressure. Useful for any long-context model (DeepSeek-V3 128k, Qwen 32k, etc.). Not currently in production decode path. |
| `turboquant_encode` | 4-bit KV compression (WHT + Lloyd-Max + pack) | opt-in / not yet wired | Companion to the above. **Retained** — long-context support is a likely future direction; pulling these would lose the implementation work. |

### Interpretability / research path (LARQL-specific)

| shader | role | status | retention reason |
|---|---|---|---|
| `graph_walk_knn` | Dot-product KNN over gate vectors | production for `vindex.infer()` and `walk` paths | Gate KNN feature attribution for the LSM / vindex compilation work. **Load-bearing for the research mission**, not transformer inference. Kept regardless of perf-engineering tracks. |

## Findings

### 1. Only one shader is verifiably orphan

**`q4k_qkv_proj_v2`** is the only shader with **zero dispatch sites** in production code, no test references, no env-var gate, and no roadmap track. The doc-comment frames it as an experimental alternative that never replaced v1. v1 is what production dispatches.

Even under model-agnosticity, v2 has no defensible retention story: it solves the same problem as v1 for the same models, and the alternative lane decomposition was never measured to win.

### 2. The "ADR-015 graveyard" candidates are defensibly retained under model-agnosticity

`q4k_ffn_gate_up_coop` and `q4k_ffn_gate_up_f16acc` lost their Gemma-3-4B end-to-end A/B and have been documented in ADR-015 as instances of the iso-vs-batched pattern. **But the Gemma 3 4B test shape (K=2560) is one specific configuration** — these candidates could plausibly win on:

- Larger K dimensions (Llama 3 70B has hidden_dim=8192; Gemma 4 31B has 4096)
- Different ALU/bandwidth balance (M5+ Apple silicon has different cache hierarchy)
- Sustained-load thermal scenarios (where the ALU savings of f16_acc could materialise)

**This was not the framing when NR2 was deleted earlier this session.** The NR2 deletion (2026-05-09 morning) was justified under "consistency with the iso-vs-batched ADR" but the model-agnosticity constraint suggests that justification was incomplete: NR2's multi-row pattern (mirroring llama.cpp's `N_R0_Q4_K=2`) might also have been useful for non-Gemma K-dimensions.

### 3. The "non-Gemma" shaders are actually production-load-bearing

`causal_attention`, `fused_attention`, `layer_norm`, `q4_matvec_v4`, `q4_f32_matvec`, `q4_vecmat`, `q4_sparse_matvec`, `q8_matvec`, `q8_attn_proj`, `activation` were all marked DEAD or TEST-ONLY in the original audit. They're **not** — they're production-load-bearing for non-Gemma model families. The original audit's grep-for-Gemma-dispatch-sites methodology systematically missed them.

### 4. Cross-model parity test coverage exists at integration level

`crates/larql-inference/tests/test_logits_goldens.rs` covers Gemma 3, Gemma 4 31B, Llama 2 7B, Mistral 7B end-to-end. `test_arch_golden.rs`, `test_decode_consistency.rs`, `test_cpu_metal_parity.rs`, `test_decode_stage_bisect.rs` cover Gemma 3 + Gemma 4. `test_generate_q4k_cpu.rs` covers Gemma 3 + Gemma 4 e2b + Mistral 7B.

**Per-shader coverage is mostly Gemma-only.** Tests in `crates/larql-compute/tests/` use `gemma3-4b-q4k-v2.vindex` as the canonical dispatch target. This means a Gemma-only-failed candidate (like NR2) is currently very hard to revalidate against Llama/Mistral without writing a new test.

### 5. RoPE coverage gap

llama.cpp ships `rope_norm`, `rope_neox`, `rope_multi`, `rope_vision` (4 variants). We have one `rope` shader that supports the standard Llama / Gemma RoPE pattern. **We don't currently support NeoX-style RoPE** (used by some older models like GPT-NeoX, Pythia, some Falcon variants) or vision-LM RoPE patterns. This is a real gap if those models are ever in scope.

## Recommendations

### A. Hard delete (1 shader)

- **`q4k_qkv_proj_v2`** — verifiably orphan, no defensible retention story. Same op as v1 on the same models with an experimental alternative lane decomposition that never won.

### B. Retain with explicit model-agnosticity doc-blocks (3 opt-in shaders)

For each of `q4k_ffn_gate_up_coop`, `q4k_ffn_gate_up_f16acc`, and the `NORMED_SHADER` variant of `q4k_q6k_qkv_proj`, add a doc-block at the top of the shader file:

```rust
//! ## Retention rationale (post 2026-05-09 model-agnosticity audit)
//!
//! - Was tried on Gemma 3 4B (K=2560) on `<date>`; result: `<outcome>`.
//! - Reason this is kept opt-in rather than deleted:
//!   - Could plausibly win on K=`<X>` (e.g. larger Llama variants, Gemma 4 31B with K=4096).
//!   - Could plausibly win on M5+ Apple silicon where ALU/bandwidth balance shifts.
//!   - Could plausibly win in `<sustained-load / cold-cache / specific arch feature>` scenario.
//! - Re-validation gate: `<what would constitute a positive A/B>`.
```

This makes the retention story falsifiable. If a future audit finds the shader still has no plausible scenario, it becomes a deletion candidate with the rationale visible.

### C. Reconsider the NR2 deletion

The 2026-05-09 NR2 deletion was justified under ADR-015 ("opt-in candidates that lost Gemma A/B get deleted") but is inconsistent with the model-agnosticity constraint. Three options:

- **Restore NR2** from git history with a doc-block matching the pattern in (B). Net cost: 1 file restored, 1 env-var re-added.
- **Codify the NR2 deletion as the intended pattern** and apply it to f16_acc + coop too (i.e. delete those as well). Net cost: −2 files, but loses capability.
- **Status quo** — accept the inconsistency, NR2 is gone, the others stay. Cleanest decision: don't relitigate.

### D. Add cross-model per-shader parity tests (follow-up project)

Currently per-shader tests in `crates/larql-compute/tests/` are Gemma-only. A follow-up project: add parametrised tests that exercise each Q4_K-family shader on Llama 2 7B, Mistral 7B, Gemma 3 4B vindexes. Without this, future Gemma A/B regressions can't tell us whether a candidate is Gemma-specific-bad or globally-bad.

### E. Document RoPE coverage gap

Add a `// TODO(rope-neox)` and `// TODO(rope-multi)` to `metal/shaders/rope.rs` flagging the variants llama.cpp has that we don't, with the model families that need them.

### F. Write a new ADR on shader retention policy

ADR-017: **Shader retention under model agnosticity**. Codifies: shaders aren't deleted purely on Gemma-failed-candidate basis. Each opt-in shader needs a documented model-or-hardware scenario where it could revive. Re-litigation is cheap (1-line bench A/B); deletion is expensive (loses implementation work and historical context).

## Out of scope for this audit

- Per-kernel parity testing across non-Gemma vindexes (Llama 2, Mistral) — recommended as follow-up project.
- Profiling each opt-in shader on Llama 2 / Mistral to test the model-agnosticity hypothesis — would need a multi-day effort.
- Adding the missing RoPE variants — separate project.
- Investigating the D-FFN-FUSE NaN bug on production Q6_K weights — separate project.

## Related

- `crates/larql-compute/PERFORMANCE.md` — recent-changes table, current state.
- `crates/larql-compute/ROADMAP.md` "P0: Production gap closers".
- `crates/larql-compute/docs/adr/015-isolated-vs-batched-kernel-perf.md` — the iso-vs-batched ADR (Gemma-frame).
- `crates/larql-compute/docs/adr/016-defused-rms-norm-qkv.md` — the QKV defuse decision.
- `crates/larql-compute/docs/llama-cpp-comparison.md` — kernel-architecture comparison; documents the rope-variant coverage gap.
- `crates/larql-compute/docs/shaders.md` — older shader reference doc (may need consolidation with this one).

> **Corrected 2026-08-23 — this table had the q6k default INVERTED.** `q6k_use_8sg` is `env_opt_in(LARQL_Q6K_8SG)`, i.e. **default-OFF**, and the shader's own header says so: *"Status: opt-in via `LARQL_Q6K_8SG=1`. Default-OFF (4sg)"* — the 8sg occupancy win did not translate end-to-end on M3 Max. **4sg is the production default; 8sg is the opt-in arm.** (The *q4k* 8sg rows on this page are correct — that one is default-on. Do not "fix" them to match.)
