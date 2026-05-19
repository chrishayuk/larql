# Roadmap — larql-inference

## Current: 83.2 tok/s (Metal Q4K, Gemma 3 4B, real vindex, 2026-05-04) | 18.9 tok/s (Gemma 4 26B-A4B MoE, CPU experts) | 6.5 tok/s (Gemma 4 31B remote-FFN batch, Metal GPU server) | Ollama: ~96–104 tok/s | 4 KV engines | 904 lib tests | 65.67% line cov (64 of 127 files at ≥90% line cov)

## Recommended next (priority order, 2026-05-10)

Curated from the open work below. Each item links to the section that owns the
detail.

1. ~~**A1-A3 — heterogeneous attention geometry**~~ — **shipped 2026-05-10**.
   Reproduced 31B end-to-end first: it generates correctly. The "L5
   immediate-EOS" claim was stale — heterogeneous-attention safety had
   quietly landed via per-layer reads in
   `metal/decode/setup.rs:DecodeScratch::new`, `LayerBuffers::allocate`,
   and `ensure_kv_cache_for_layers`. A1 (`Capability::HeterogeneousAttention`
   runtime gate), A2 (drop vestigial scalar geometry from `DecodeBackend`),
   and A3 (KV cache audit) landed as the deferred cleanup pass. See
   `CHANGELOG.md` and "Open: Model architecture independence hardening"
   below (now ✅ shipped section).

2. **G-3 (= G-4) — flash-attention-style fused attention kernel** — highest
   GPU-fwd lever after G-1/G-2 missed. Stub at
   `crates/larql-compute/src/metal/shaders/fused_attention.rs`; collapse
   RoPE + QK_norm + KV_append + KV_attend into 1-2 dispatches → ~0.85 ms/tok
   recoverable, projects to 95-105 tok/s on Gemma 3 4B (ollama parity). See
   "Open: GPU-forward kernel utilization" → G-3 / G-4 (canonical entry: G-3).

3. **H12 closed** (2026-05-10) — H8/H9/H11 + all three orchestration-file
   splits shipped: `layer_graph/predict.rs`, `ffn/moe_remote/shard.rs`,
   `layer_graph/generate/gpu.rs` (999 LOC → 5-file `gpu/` directory, max
   534 LOC; 646 lib tests pass). Inference-crate-hardening backlog is now
   empty of file-split items. See `CHANGELOG.md` entry.

4. **R4 — research trace export contract** — only remaining R-item from the
   mech-interp engine surface; unblocks MI4 onwards. See "Open: Mechanistic
   research engine surface" → R4.

5. **MI4 — golden parity tests for WalkFfn + patched-vindex** — dense and
   custom-backend parity already pinned; add the same for vindex-backed paths
   so MI5/6/7 attribution and patching work has a regression net. See "P0:
   Best-in-class mechanistic interpretability engine".

6. **P1 mechanical backlog** (parallelisable, low risk) — synthetic e2e test
   for `generate()` (currently `#[ignore]` only); per-file 90% floor work
   for the **63 files still below** (was 84; coverage push 2026-05-10 lifted 34 files past
   the floor — see CHANGELOG.md); the deferred structure moves (`capture.rs →
   trace/`, `residual.rs → forward/norm.rs`). See
   "P1: Test coverage gaps", "P1: Structure & file layout", "P1: Quality bugs".

7. **Coverage push: synthetic-Q4K vindex fixture** (parallelisable, ~2-3 hr,
   line cov +1-1.5 pp) — would unlock the ~30 files at 0% line cov that
   are gated on a Q4K vindex on disk: `vindex/q4k_forward/{metal, hidden,
   interventions, walk_ffn, …}.rs`, `layer_graph/generate/gpu/{decode_loop,
   prefill}.rs` deeper paths, `forward/predict/honest.rs`. The fixture
   needs valid Q4_K bytes (144B per 256-element block) for attention +
   FFN + lm_head, written to a tempdir matching the vindex file layout.
   The companion `MockComputeBackend` (advertising `Capability::PrefillQ4`
   + `DecodeToken`, returning `Some` from those methods) is the smaller
   half of the work.

## Open: Mechanistic research engine surface — Q4K interventions for OV/RD

**Status**: In progress as of 2026-05-02.

The existing CPU hook system (`forward::LayerHook`,
`trace_forward_full_hooked`, `RecordHook`, `ZeroAblateHook`,
`SteerHook`) is good for dense in-memory weights. The OV/RD work showed
the next reusable boundary: Q4K/vindex-backed research passes need the
same capture/intervention semantics without each CLI experiment manually
dequantising layers and driving the full forward loop.

Promotion plan:

| # | Item | Status |
|---|------|--------|
| R1 | Public Q4K layer tensor insertion/removal helpers in `vindex::q4k_forward` | shipped |
| R2 | Q4K hidden forward that accepts `LayerHook` for pre-layer/post-attention/post-layer interventions | shipped |
| R3 | Pre-W_O capture/replacement adapters over the existing attention-block primitives | shipped |
| R4 | Research trace export contract for prompt ids, tokens, layer-input rows, pre-W_O rows, logits, and metrics | planned |

Keep PQ codebooks, Mode-D table fitting, address probes, and
code-stability diagnostics in `larql dev ov-rd` until their artifact and
runtime contracts are stable.

---

## Open: inference crate hardening — correctness, API boundaries, model agnosticity

**Status**: Started 2026-05-09 after `larql-inference` review.

The crate now carries stable serving/generation paths, mechanistic-interpretability
research surfaces, Q4K/vindex-backed execution, and remote expert dispatch. The
review found two classes of work: immediate correctness fixes and longer-term
cleanup to keep the crate model-agnostic and maintainable.

Work items:

| # | Item | Status |
|---|------|--------|
| H1 | Restore `cargo test -p larql-inference` after the `GateIndex` trait split by updating walk-FFN test mocks to implement `GateLookup` / `PatchOverrides` / FFN storage traits | shipped 2026-05-09 |
| H2 | Make generation honor `max_tokens == 0` across GPU, CPU fallback, constrained, and streaming paths | shipped 2026-05-09 |
| H3 | Check EOS on the first generated token in unconstrained GPU generation, including special-token raw decode via `EosConfig::is_eos_with_tokenizer` | shipped 2026-05-09 |
| H4 | Stop discarding `EosConfig` in constrained generation and CPU fallbacks; use caller-supplied EOS IDs and stop strings instead of only `vindex::is_end_of_turn` | shipped 2026-05-09 |
| H5 | Convert `optimise_target_delta` invalid inputs (`target_id >= vocab`, empty prompt, unsupported shapes) from panics into early `Err` results | shipped 2026-05-09 |
| H6 | Harden remote FFN/Q8K wire decoders against untrusted length fields with checked arithmetic and bounded allocation | shipped 2026-05-09 |
| H7 | Replace ad hoc `Result<_, String>` / silent empty `GenerateResult` failures in generation paths with typed errors where public callers need to distinguish unsupported backend vs. empty output | shipped 2026-05-09 — `GenerateError` variants added with fallible `try_generate*` wrappers |
| H8 | Move runtime env toggles (`LARQL_PROFILE_*`, `LARQL_MOE_*`, `SKIP_MOE`, `LARQL_LM_HEAD_*`) behind typed debug/config structs passed into hot paths | shipped 2026-05-10 — added `forward::dump_config::DumpConfig` (lifted 7 inline reads of `LARQL_CPU_DUMP_LAYERS` / `LARQL_CPU_STAGE_DUMP` / `LARQL_STAGE_DUMP_LAYER`) and `ffn::moe_remote::runtime::RemoteMoeRuntime` (lifted 6 inline reads of `LARQL_HTTP_TIMING` / `LARQL_MOE_WIRE_F16` / `LARQL_DISABLE_Q8K_WIRE` / `LARQL_VERBOSE`); both `OnceLock` singletons. Earlier 2026-05-09 work covered generation/grid token policy, profiling, LM-head, and MoE config |
| H9 | Narrow the crate root public surface: stop re-exporting experimental/internal modules by default, and distinguish stable inference APIs from research/dev surfaces | shipped 2026-05-10 — dropped 17 `forward::*` + 7 `layer_graph::*` root re-exports with zero external use AND zero in-crate example/test use; `research` module rewritten to source from subpaths (survives further root trims). Earlier 2026-05-09 work added `prelude` and `research` grouped surfaces |
| H10 | Remove backend-name probes and concrete `MetalBackend` downcasts from generic generation dispatch; replace with explicit compute-backend capability methods | shipped 2026-05-09 for generation dispatch |
| H11 | Move model-family workarounds and tokenizer suppression policy out of generic generation loops into architecture/tokenizer policy objects | shipped 2026-05-10 — verified no hardcoded `family() == "..."` branches remain in generic generation/decode/forward paths. The single remaining family-named match (`pipeline_layer.rs::moe_routing_policy`) reads the architecture trait's `router_type` metadata, which is the policy-object pattern this item asks for |
| H12 | Split large orchestration modules (`layer_graph/grid.rs`, `layer_graph/generate/gpu.rs`, remote MoE backend/shard code) into policy, backend dispatch, wire protocol, timing, and token selection units | shipped 2026-05-10 — `layer_graph/predict.rs` (881) → `predict/` directory (mod 376 + split 285 + honest 261); `ffn/moe_remote/shard.rs` (924) → `shard/` directory (mod 376 + layer_batch 215 + multi_layer 140 + expert_batch 136 + stream 123); `layer_graph/generate/gpu.rs` (999) → `gpu/` directory (mod 534 + decode_loop 329 + forced_logits 211 + prefill 193 + sampling_step 56). All earlier 2026-05-09 extractions (token selection policy, GPU setup, constrained generation, grid config/timing/remote-MoE/remote-FFN) preserved. 646 lib tests green |

Acceptance:

- `cargo test -p larql-inference` compiles and passes in a clean target dir.
- Public generation APIs obey token budgets and EOS config deterministically.
- Unsupported model/backend combinations fail with precise errors rather than
  silent empty output, zero-vector substitution, or process panic.
- Model-family-specific behavior is expressed through architecture/tokenizer
  policy, not hardcoded in generic decode loops.

---

## ✅ Model architecture independence hardening — A1-A3 shipped 2026-05-10

**Status**: A1, A2, A3 shipped 2026-05-10 (see CHANGELOG.md). A4 remains
planned. Original "L5 immediate-EOS" framing turned out to be stale — when
the work item was reopened, Gemma 4 31B Q4K Metal already generated
coherent text end-to-end (verified via `larql run gemma4-31b-q4k.vindex`).
The heterogeneous-attention safety had landed quietly via per-layer reads
in `metal/decode/setup.rs:DecodeScratch::new` (sized to
`max(layers[*].q_dim)`), `metal/ops/full_pipeline/buffers.rs` (prefill
same), and `MetalBackend::ensure_kv_cache_for_layers`. A1-A3 then landed as
the deferred API/capability cleanup.

Gemma 4 31B Q4K has 60 layers split into two geometry classes: 50
sliding-attention layers (head_dim=256, num_kv_heads=16,
sliding_window=1024) and 10 full-attention layers at L5, L11, L17, L23,
L29, L35, L41, L47, L53, L59 (head_dim=512, num_kv_heads=4). Both classes
now flow correctly through GPU decode and prefill.

Work items:

| # | Item | Status |
|---|------|--------|
| A1 | Add a runtime capability gate for architectures whose attention is not executable by the active path | shipped 2026-05-10 — `Capability::HeterogeneousAttention` + `gpu_setup::ensure_attention_supported` (see CHANGELOG) |
| A2 | Remove scalar `num_q_heads`, `num_kv_heads`, `head_dim`, `q_dim`, `kv_dim`, and `rope_base` assumptions from decode/prefill call sites | shipped 2026-05-10 — dropped from 10 `DecodeBackend` methods + every `larql-inference` call site; `AttentionGeometry` deleted; bonus: latent `PipelineIntervention` head_dim bug fixed (now per-target-layer) |
| A3 | Ensure all KV cache allocation paths use `layers[*].num_kv_heads` and `layers[*].head_dim`, not the caller's first-layer geometry fallback | already shipped pre-A1/A2; documented + audited 2026-05-10 — production paths all reach `preallocate_kv_cache_per_layer`; legacy uniform `create_kv_cache` retained only for synthetic tests + `populate_kv_layer` lazy bootstrap |
| A4 | Add architecture fixtures for heterogeneous geometry and unsupported-attention failures so GPU, CPU, trace, and vindex-backed paths agree | planned |

Acceptance (met for A1-A3): a heterogeneous model runs through every
selected path using per-layer geometry, or fails before decode/extraction
with a precise unsupported capability error (`GenerateError::UnsupportedBackend`).

---

## P0: Best-in-class mechanistic interpretability engine

**Status**: In progress as of 2026-05-02.

The target is not just "TRACE runs"; the target is that every mechanism LARQL
can execute is also queryable, attributable, patchable, and reproducible as a
research artifact.

| # | Item | Status |
|---|------|--------|
| MI0 | Faithful residual DAG: TRACE routes through the canonical layer runner and `residual[L] = residual[L-1] + attn_delta + ffn_delta` is test-pinned | shipped 2026-05-02 |
| MI1 | Python `WalkModel.trace()` and `patch_activations()` use vindex `WalkFfn`, not dense fallback | shipped 2026-05-02 |
| MI2 | Backend-parametric activation patching helpers for donor capture and recipient intervention | shipped 2026-05-02 |
| MI3 | Trace artifact contract: complete ordered chains only, exact file length checks, `TRACE SAVE` requires `POSITIONS ALL` | shipped 2026-05-02 |
| MI4 | Golden parity tests: TRACE final residual/logits match canonical forward across dense, WalkFfn, patched vindex, Q4K, and MoE paths | partial — dense/custom backend pinned; GPU FFN server path (`run_dense_ffn_q4k`) shipped 2026-05-04, parity tests pending |
| MI5 | Rich attribution objects: per-head attention writes, per-feature FFN activations, router/expert decisions, and path-level provenance | planned |
| MI6 | Expanded causal operators: head patching, feature patching, FFN feature ablation, router/expert patching, and KV/residual boundary patching | planned |
| MI7 | Q4K/MoE interpretability parity: trace and patch support for quantized dense and routed expert paths, with clear precision caveats | planned |
| MI8 | Python experiment ergonomics: batched prompts, donor/recipient alignment helpers, causal effect metrics, artifact metadata, and reproducibility stamps | planned |

Near-term order:

1. Finish MI4 for WalkFfn and patched-vindex paths, because dense and custom
   backend parity are now pinned.
2. Add attribution records where the forward path already exposes data:
   attention captures, WalkFfn feature dispatch, and activation top-k.
3. Extend patching operators one mechanism at a time, starting with
   post-attention/head writes and FFN feature activations.
4. Only then promote Q4K/MoE trace/patch support to first-class status, because
   those paths need parity tests before they can be trusted as evidence.

---

## ✅ Metal lm_head — stride-32 Q4_K matvec, f16 GEMV fallback (correctness + perf fix, 2026-05-01)

> **2026-05-02 follow-up — the root cause was wrong.** What was diagnosed
> as a kernel-level reduction-tree drift turned out to be a dispatch
> geometry mismatch (`MetalBackend::q4k_matvec` hardcoding the 4sg
> shader's `THREADS_PER_TG=128` while dispatching the 8sg pipeline,
> production default since 2026-04-28 — same family as 077884b). With
> the dispatcher fixed to use `pipeline.rows_per_tg` /
> `pipeline.threads_per_tg`, the production `q4k_matvec` is correct AND
> ~1.10 ms/tok faster than stride-32. **Stride-32 is now the diagnostic
> fallback; production default is `lm_head_knn_backend` with
> `q4k_matvec` first.** End-to-end: 76.3 → 84.0 → **88.1** tok/s on
> Gemma 3 4B (last step is the 2026-05-09 QKV defuse), gap to ollama
> 1.30× → 1.18× → **1.17×**. Diagnostic A/B via
> `LARQL_LM_HEAD_SKIP_Q4K=1`. The historical bisect below is preserved
> for context.

Gemma 3 4B Metal end-to-end was producing the wrong continuation
("The Capital of France is:  **") on `"The capital of France is"`
while CPU produced the correct "**Paris**" answer. Bisected:

- Per-layer hidden parity holds (`test_decode_consistency_gemma3_4b`
  and the new 2-step variant pass at cos ≥ 0.99995 across all 34
  layers, 1 and 2 decode steps) — KV cache writes/reads and per-layer
  Metal kernels are correct.
- The single-token logits goldens for Metal pinned a top-5 set whose
  positions 4-5 differed from CPU at the prefill boundary, even though
  top-1 matched (`top1_logit Δ ≈ 5e-4`).
- A/B with `LARQL_LM_HEAD_FORCE_CPU=1` confirmed Metal generated
  "Paris" once the lm_head bypassed the Q4_K matvec path, isolating
  the drift to that specific kernel.

Root cause: `shaders/q4k_matvec.rs` 32-lane simdgroup parallel
reduction with a 2-way inter-superblock split (`ix = lane & 1u`)
accumulates partial sums in a different order than the f32 reference.
Same f32 precision at every step; the difference is reduction-tree
associativity. On a 262K × 2560 lm_head matvec this surfaces as
~1e-3 relative drift on top-1 logits, enough to flip rank-1 on
close-call tokens (e.g. " Capital" vs " capital" at decode step 1
of Gemma 3 4B).

**Fix**: `lm_head_topk` (`layer_graph/generate/lm_head.rs`) routes
through the new `lm_head_knn_backend_skip_q4k` method on `VectorIndex`
when the active backend is non-CPU. That dispatch chain replaces the
production `q4k_matvec` first-path with a 3-step ladder:

  1. **Stride-32 Q4_K matvec** (`backend.q4k_matvec_stride32`,
     `shaders/q4k_matvec_stride32.rs` — new) — same Q4_K bytes as
     production, same bandwidth (330 MB/tok read), but lane `k`
     accumulates the dot-product over elements `i % 32 == k` and the
     final reduction is `simd_sum` across 32 lanes — bit-equivalent
     reduction tree to `f16_gemv`. Recovers rank-1 stability without
     paying the f16 fallback's 4× bandwidth penalty.
  2. **f16 GEMV on `embeddings.bin` mmap** (tied-embed only, ~2×
     bandwidth of Q4_K) — fallback when the stride-32 kernel isn't
     dispatchable.
  3. f32 BLAS fallback (`lm_head_knn`).

**Env vars (post 2026-05-02 dispatch fix):** the production Q4_K path
is the default; `LARQL_LM_HEAD_SKIP_Q4K=1` routes through this
stride-32 chain for diagnostic A/B. Within the chain,
`LARQL_LM_HEAD_STRIDE32=0` disables the stride-32 fallback.

Five attempts on the way to this:
- v1: route through `CpuBackend` via `index.lm_head_knn_backend` —
  picks the **scalar** Q4_K reference (`cpu/ops/q4k_matvec.rs::dispatch`,
  unvectorised), ~510 ms/tok → **1.9 tok/s** end-to-end.
- v2: route through `CpuBackend` via `backend_lm_head_topk` (CPU BLAS
  on f32 `weights.lm_head`), ~30 ms/tok → **23.6 tok/s**.
- v3: route through Metal `backend.f32_gemv` on f32 `weights.lm_head`,
  ~8 ms/tok → **52.2 tok/s** sustained.
- v4: Metal `f16_gemv` on the embed `f16_mmap`, ~4 ms/tok →
  **66.8 tok/s** sustained.
- **v5 (shipped)**: Metal stride-32 `q4k_matvec` on Q4_K mmap, ~3
  ms/tok → **71.5 tok/s** sustained.

**Validation**:
- `arch_gemma3_4b_gpu` now generates `"The capital of France is **Paris**."` (was `"The Capital of France is:  **"`).
- All 4 `gemma3` logits goldens pass for both backends; pinned values are now equal post-fix (per-backend split kept for future drift detection).
- 2-step decode parity (`decode_consistency_gemma3_4b_2steps` — new) confirms KV-cache write/read across decode steps is independently correct.

**Bench (Gemma 3 4B, M3 Max, `larql bench gemma3-4b-q4k-v2 --ollama gemma3:4b -n 50 --warmup 5`, sustained / cold-GPU)**:

| Path | Decode tok/s | lm_head ms/tok | GPU fwd ms/tok | vs ollama |
|---|---|---|---|---|
| Pre-fix (Metal Q4_K matvec, **wrong output**) | ~78 (historic) | ~1 | ~12 | 1.34× slower |
| v1: CPU `index.lm_head_knn_backend` (scalar Q4_K) | **1.9** | 509.3 | 18.6 | 55× slower |
| v2: CPU `backend_lm_head_topk` (BLAS f32) | 23.6 | 30.4 | 12.6 | 4.4× slower |
| v3: Metal `backend.f32_gemv` on f32 lm_head | 52.2 | 8.0 | 12.0 | 2.0× slower |
| v4: Metal `f16_gemv` on embed f16_mmap | 66.8 | 3.8 | 11.8 | 1.57× slower |
| **v5 (shipped)**: Metal stride-32 `q4k_matvec` | **71.5** | 3.0 | 11.7 | **1.44× slower** |
| ollama gemma3:4b | 102.8 | — | — | 1.00× |

(Watch for thermal noise: back-to-back benches on a hot GPU drop
sustained tok/s by 25-30%; cool-GPU numbers above match the historic
~78 baseline structure when adjusting for the 3 ms lm_head cost.)

lm_head is now ~21% of decode (down from 96.5% in v1, 25.5% in v4).
The stride-32 kernel approaches Q4_K's bandwidth floor (330 MB/tok ÷
~400 GB/s ≈ 0.8 ms theoretical; we're at 3 ms ≈ 28% of peak). The
remaining 1.44× gap to ollama (and the ~6 tok/s gap to the historic
~78 baseline) lives entirely in **GPU forward** (75% of decode @
11.7 ms), which is a separate roadmap item — `q4k_matvec` 8sg /
Q4_K matmul for prefill / kernel fusion / encoder coalescing.

**Files**:
- `crates/larql-compute/src/metal/shaders/q4k_matvec_stride32.rs` — new shader, f16_gemv-style stride-32 reduction
- `crates/larql-compute/src/metal/shaders/mod.rs` — register the new module + push to merged source
- `crates/larql-compute/src/metal/mod.rs` — `q4k_matvec_stride32_pipeline` field + KernelHandle init
- `crates/larql-compute/src/metal/trait_impl/matmul.rs` — `MetalBackend::q4k_matvec_stride32` inherent method
- `crates/larql-compute/src/metal/trait_impl/quant_matvec.rs` — `QuantMatVec::q4k_matvec_stride32` trait wire-up
- `crates/larql-compute/src/backend/quant_matvec.rs` — trait method declaration (default returns `None`)
- `crates/larql-inference/src/layer_graph/generate/lm_head.rs` — `lm_head_topk` `prefer_cpu` branch routes to `index.lm_head_knn_backend_skip_q4k(..., backend)`
- `crates/larql-vindex/src/index/storage/lm_head.rs` — new `lm_head_knn_backend_skip_q4k` method (path 1 = stride-32 Q4_K, path 2 = f16 GEMV, path 3 = f32 BLAS); `LARQL_LM_HEAD_STRIDE32=0` opt-out
- `crates/larql-inference/src/residual_diff/capture.rs` — `metal_decode_steps` helper for multi-step parity
- `crates/larql-inference/tests/test_decode_consistency.rs` — `decode_consistency_gemma3_4b_2steps` test
- `crates/larql-inference/tests/test_logits_goldens.rs` — Metal pins re-captured for v5 stride-32 path

---

## Open: GPU-forward kernel utilization — closing the 4.4 ms gap to ollama

**Status**: Open as of 2026-05-01. Diagnosed via
`cargo run -p larql-compute --release --features metal --example diag_profile_kernels`
plus per-step `LARQL_PROFILE_DECODE=1` profiling on Gemma 3 4B; ollama's
fine-grained timings via `/api/generate` (`total_duration`,
`prompt_eval_duration`, `eval_duration`).

**Where the gap *really* lives** (corrected 2026-05-01 after instrumenting
in-pipeline GPU vs CPU timing via `LARQL_GPU_TIMING=1` —
`metal/decode/gpu_timing.rs::TokenGpuTime`):

```
Per-token decode_token (n=12, steady state):
  Wall:    ~10.7 ms
  GPU:     ~10.5 ms  (98% of wall — kernels are GPU-bound)
  CPU:      ~0.5 ms  (5% — dispatch overhead is NOT the bottleneck)
  cmd_bufs: 1 per token (one coalesced buffer covers all 34 layers)
```

So the 14.0 ms/tok vs ollama's 10.4 ms/tok gap breaks down as:

| Stage | larql | ollama (est.) | gap |
|---|---|---|---|
| `decode_token` GPU compute | 10.5 ms | ~7-8 ms | +2.5-3 ms |
| lm_head | 3.0 ms | ~2 ms | +1 ms |
| other | ~0.5 ms | ~0.5 ms | 0 |
| **total** | **14.0 ms** | **10.4 ms** | **+3.5 ms** |

Both gaps are **GPU compute, not CPU dispatch**. Kernel-isolated
`metal/diag/kernel_profile.rs` GB/s overstated the headroom (kernels
run partially pipelined within one cmd buffer; isolated GB/s isn't
the right metric). Our actual decode is at ~75-80% of ollama's
throughput on the same hardware — competitive but not parity.

**Earlier (incorrect) diagnosis preserved for context**:

| Stage | larql peak | ollama | gap | recoverable? |
|---|---|---|---|---|
| GPU forward | 11.6 ms/tok | ~7-8 ms | **+4 ms** | yes — see kernel breakdown |
| lm_head | 3.0 ms/tok | ~1.5-2 | +1.5 ms | mostly tight (~28% of Q4_K bandwidth floor) |
| total/tok | **14.7 ms** | 9.6 ms | +5 ms | most via GPU fwd |
| tok/s | 68 | 104 | 1.53× | |

(Sustained tok/s drops further on hot GPU — thermal throttling doubles
GPU fwd time over ~16 decode steps. Ollama is presumably less affected
because their faster decode finishes the same wallclock budget with less
GPU on-time.)

**Per-kernel utilization** (decode, Gemma 3 4B, M3 Max LPDDR5X ~400 GB/s peak):

| Kernel | Bandwidth | % of peak | ms/tok | Headroom (at 80% peak) |
|---|---|---|---|---|
| q6k_matvec (FFN down, K=10240) | 321 GB/s | 80% | 2.3 | ~0 (already tight) |
| q4k_ffn_gate_up (gate+up, K=2560) | 187 GB/s | **47%** | 5.4 | **-2.2 ms** |
| q4k_matvec (Wo, K=8192) | 184 GB/s | **46%** | 2.2 | **-0.9 ms** |
| q4k_qkv_proj (Q+K+V fused, K=2560) | 114 GB/s | **28%** | 7.1 | **-4.3 ms** |
| **Total recoverable in GPU fwd** | | | | **~7 ms/tok** |

If we hit 80% peak across the three under-utilized kernels: GPU fwd
drops 11.7 ms → ~5 ms, total decode 14 ms → ~8 ms, **125+ tok/s
end-to-end** (ahead of ollama). Realistic target with kernel rewrites:
**80-90 tok/s** as a first milestone (matches the historic memory's
"~78 baseline" pre-correctness-fix).

**Why under-utilized**: per `metal/diag/kernel_profile.rs` annotations,
`q4k_ffn_gate_up` is "COMPUTE-BOUND (K=2560 dequant dominates)". The
Q4_K dequant inline in the shader (decode super-block scale, sub-block
scale via 6-bit unpack, nibble extract, FMA) eats ALU cycles that block
memory issue. Each lane redundantly decodes the per-super-block
`d`/`dmin` and per-sub-block `sc`/`mn`, so the simdgroup spends 32× the
necessary dequant work and the per-row FMA chain stalls waiting for
operands. Llama.cpp's equivalent kernel co-operates one lane per
simdgroup to load scales into threadgroup memory, then broadcasts to
all 32 lanes for a tight FMA loop — eliminates the redundancy.

The same pattern applies to `q4k_qkv_proj` (also K=2560) and `q4k_matvec`
on Wo (K=8192). The three are the largest per-token GPU costs; closing
their utilization is the highest-leverage GPU-fwd work item.

**Optimization paths in priority order** (each independent and stackable):

### G-1 — Cooperative scale-loading in `q4k_ffn_gate_up`

**Status**: ❌ Tried 2026-05-01, no end-to-end win. Kernel kept opt-in
(`LARQL_GATE_UP_COOP=1` → `q4k_ffn_gate_up_coop_pipeline`,
`shaders/q4k_ffn_gate_up_coop.rs`) for future hardware / fusion
scenarios.

**What was tried**: shipped a new `q4k_ffn_gate_up_coop` shader that
keeps the production lane partitioning (`ix = lane & 1u`,
`j = (lane >> 1) >> 1`) but does the per-super-block dequant
cooperatively:
- Lanes 0..7 of each simdgroup each compute one sub-block's
  `(scale = d * sc, mmin = dmin * mn)`.
- Writes go to threadgroup memory (256 B / TG, well under hardware
  limit).
- `threadgroup_barrier(mem_threadgroup)` flushes; all 32 lanes read
  their owned `j`'s `(scale, mmin)`.
- Each writer also re-decodes `d`/`dmin` itself (8× redundant vs
  production's 32×) — using `simd_broadcast` for `d`/`dmin` produced
  wrong output (close-call top-1 flips), likely from the broadcast
  reordering the inner FMA chain enough to drift past the rank-1 gap.

**Result**: bench A/B (3 runs each, cold + warm GPU):
- Coop:     72.1 / 61.8 / 71.8 tok/s, GPU fwd 12.1 / 14.1 / 12.1 ms
- Baseline: 63.2 / 73.0 / 62.2 tok/s, GPU fwd 13.9 / 11.8 / 13.8 ms

Within thermal noise. **No end-to-end win**.

**Why the diagnosis was misleading**: `metal/diag/kernel_profile.rs`
flagged `q4k_ffn_gate_up` as "COMPUTE-BOUND (K=2560 dequant
dominates)" based on isolated-kernel GB/s measurement. In practice the
production kernel's per-lane redundant dequant ALU **runs concurrently
with the per-row weight loads**, filling memory-stall bubbles for free.
Removing the redundant ALU saves cycles in isolation but doesn't
increase memory throughput — the actual bottleneck. Same lesson as the
2026-04-28 `LARQL_F16_ACC=1` kernel-isolated 1.79× → end-to-end parity
finding. Kernel-isolated profiler GB/s alone is not predictive of
end-to-end wins on Apple Silicon GPUs; the right metric is full
end-to-end tok/s on a quiet GPU.

**Implications for G-2 and G-3** (same cooperative pattern proposed
for `q4k_qkv_proj` and `q4k_matvec`-Wo): expect the same null result,
since both kernels share the same per-lane dequant pattern with the
same memory/ALU overlap. Not worth shipping G-2/G-3 as written;
de-prioritise.

**What's actually on the critical path** (revised): the GB/s
under-utilization isn't ALU-driven, it's **memory access pattern /
occupancy**. Possible causes:

- Per-row weight loads are scattered enough that prefetchers don't
  saturate the LPDDR5X channels.
- Threadgroup count too low to hide memory latency across TGs.
- Per-row register footprint blocks higher concurrent-TG counts.

These need a different toolset (Xcode GPU frame capture / Metal
profiler) to localise — kernel-isolated GB/s alone isn't enough.

---

### G-2 — NR0=2 + shared-X-vector port from llama.cpp

**Status**: ❌ Tried 2026-05-01, **slight regression** (~3% slower).
Re-benched 2026-05-09 (8sg 86.3 / 86.2 tok/s vs NR2 83.4 tok/s on
quiet M3 Max, 0.07 ms baseline drift), regression reconfirmed.
**Kernel + `LARQL_GATE_UP_NR2` env var removed 2026-05-09**;
the candidate had two failed benches against it. See ADR-015 for
the iso-vs-batched lesson the kernel pinned.

**Result** (3 runs each, thermal-mixed):
- NR2:           68.6 / 69.2 / 68.3 tok/s, GPU fwd 12.76/12.56/12.84 ms
- Baseline 8sg:  71.1 / 71.1 / 71.0 tok/s, GPU fwd 12.24/12.22/12.26 ms

NR2 is ~0.5 ms/tok slower in GPU forward despite the X-cache-traffic
math predicting a savings.

**Why the diagnosis was wrong**: For Gemma 3 4B's K=2560 input, the
X-vector is 10 KB — easily fits in L1 cache (per-simdgroup or
per-TG). Whatever per-row "X reload" we measured at the kernel
boundary is being served from L1 hits, not LPDDR5X traffic. The
per-row reload doesn't actually consume bandwidth, so eliminating it
via NR0=2 saves nothing.

**This is now the THIRD consecutive miss** on a kernel optimisation
that looked high-confidence from `metal/diag/kernel_profile.rs`'s
isolated GB/s measurement (after `LARQL_F16_ACC=1` 2026-04-28 and
`LARQL_GATE_UP_COOP=1` 2026-05-01). The pattern is now clear:
**isolated kernel GB/s is not predictive of end-to-end tok/s on
Apple Silicon**. The bottleneck must be one of:

- Dispatch / scheduling overhead (not measured by `kernel_profile`)
- Memory subsystem contention across in-flight TGs (not measured)
- Thermal throttling shifting the steady-state target (real but
  doesn't explain peak-cold differences)

**Implications for future kernel work**: stop guessing from isolated
GB/s. Either:
1. Get **actual end-to-end profiling** (Xcode GPU frame capture)
   before any further kernel optimisation work — see G-5.
2. Attack **structural** changes that bypass per-kernel utilisation
   entirely — most notably **G-3** (flash-attention fusion), which
   reduces dispatch count regardless of per-kernel GB/s.

#### Original diagnosis (preserved for context, since the analysis was
correct *for what it measured* — the kernel-isolated GB/s gap is
real, but the gap doesn't translate to end-to-end work)

**Diagnosis**: Side-by-side bench against `ollama gemma3:4b` on
`"The capital of France is"`, num_predict=20:

| | larql | ollama | Δ |
|---|---|---|---|
| Decode tok/s | 71.7 | 96.0 | **+3.53 ms/tok gap** |
| GPU fwd | 12.5 ms | est. 7-8 ms | ~5 ms gap |

llama.cpp's Q4_K matvec
(`ggml/src/ggml-metal/ggml-metal-impl.h::N_R0_Q4_K`) processes **2
output rows per simdgroup** (`NR0=2`) with the X-vector loaded once
into per-lane registers and reused across both rows. Ours processes
1 row per simdgroup; the same 2560-element X-vector is reloaded per
row from cache. With our 8sg / 8-rows-per-TG geometry, that's ~2× the
X-cache traffic of llama.cpp's 2sg / 4-rows-per-TG, which matches our
measured 47% / 28% peak utilization on `q4k_ffn_gate_up` /
`q4k_qkv_proj` (the two biggest decode costs).

**Approach** (mirrors llama.cpp `kernel_mul_mv_q4_K_f32`):

1. Each simdgroup handles 2 output rows (`NR0 = 2`).
2. X-vector slice loaded once into `xl[16]` per lane.
3. For each of 2 rows: separate `sumf[2]` accumulator running the
   per-element FMA against the same `xl[16]`.
4. Two `simd_sum` calls at the end, two row-writes.

**Caveats** to watch:
- Auto-memory note from 2026-04-19: "N_DST=2 caused ~10% regression,
  N_DST=4 caused 24× regression (register spilling)". That earlier
  attempt likely **didn't share the X-vector** across rows — it just
  doubled the per-thread register footprint. The win in llama.cpp
  comes from the **shared X load**, not from naively doubling NR0.
- Verify register count via Xcode's Metal compiler diagnostic
  (`MTLLibrary.functionInfo.maxThreadsPerThreadgroup`) before shipping.
- Inner FMA chain becomes 2 chained FMAs per (lane, element) — same
  total work, but compiler must keep both `sumf[0]` and `sumf[1]` in
  registers without spilling.

**Validation**:
- Kernel-level parity test against current `q4k_ffn_gate_up` on
  synthetic data (cos ≥ 0.9999 — same Q4_K math, just multi-row dispatch).
- `arch_golden_gemma3_4b_gpu` continues to emit "**Paris**".
- `decode_consistency_gemma3_4b_2steps` continues to pass.

**Expected**: 187 GB/s → ~280 GB/s on `q4k_ffn_gate_up` → 5.4 → 3.5 ms/tok
across 34 layers → **+10-15 tok/s end-to-end** on Gemma 3 4B.
Apply the same pattern to `q4k_qkv_proj` (114 → 200 GB/s → +20 tok/s).
Stretch goal: **~95-100 tok/s, ollama parity**.

### G-3 — Flash-attention-style fused attention kernel (HIGH PRIORITY)

**Status**: Open. Canonical entry for the flash-attention work — see
also `G-3' DEPRECATED` below (preserved cross-reference) and `G-4`
which restates the same task with newer dispatch-count framing after
the 2026-05-01 fusion experiments shipped. Treat this entry as the
plan of record; the duplicate `G-4` block exists because the dispatch-
count diagnosis was rewritten after G-1/G-2/G-2' all missed.

Larger lift than G-2 but orthogonal — attacks **dispatch overhead**
(~1.0 ms/tok savings) rather than per-kernel utilization.

**Current decode dispatch chain per layer**: ~11 dispatches × 34
layers = ~374 dispatches/tok × ~5 µs each = **1.87 ms/tok overhead**.
Llama.cpp's flash-attention path collapses RoPE + QK_norm + KV_append +
KV_attend + O_proj fragments into 1-2 dispatches → ~6-7 per layer ×
34 = ~200/tok ≈ 1.0 ms overhead. **~0.85 ms/tok recoverable**.

**Approach** (mirrors llama.cpp `kernel_flash_attn_ext_*`):

1. Single fused kernel takes Q, K, V (already projected and RoPE-rotated),
   and the KV cache. Computes `softmax(QK^T / √d) · V` in one pass.
2. Tile over Q heads × KV blocks; each TG handles one Q head's softmax
   row, accumulating against the V tile in registers.
3. Online softmax (re-normalising incrementally) — avoids the
   per-position Q output allocation our current `kv_attend` materializes.

**File**: `crates/larql-compute/src/metal/shaders/fused_attention.rs`
already exists as a stub — flesh out using llama.cpp's
`kernel_flash_attn_ext_q4_K_f32` as the template (templated over Q
quant type, K head_dim, V head_dim).

**Validation**:
- Per-kernel parity test against current per-stage chain on synthetic
  Q/K/V/cache (cos ≥ 0.9999).
- `arch_golden_gemma3_4b_gpu`, `decode_consistency_gemma3_4b{,_2steps}`
  continue to pass.
- Wider sweep across Gemma 4 31B dense / 26B-A4B (different head
  geometries — global vs sliding-window layers, different head_dim).

**Expected**: -0.85 ms/tok dispatch overhead → **+5-8 tok/s end-to-end**
on Gemma 3 4B.

**Sequencing**: G-2 first (smaller, more bounded), G-3 second
(builds on G-2's NR0 understanding plus the existing
`fused_attention.rs` stub). Both together project to **95-105 tok/s
on Gemma 3 4B** (full ollama parity).

### G-3 — Dispatch-count reduction (✅ first fusion validates the model, 2026-05-01)

**First fusion shipped — `qk_norm_rope_fused`**:
`shaders/qk_norm_rope_fused.rs` collapses `qk_norm_qk` +
`rope_at_pos_batched_qk` into one kernel (each TG handles one head:
RMS-norm → weight scale → in-place RoPE rotation, with one
`threadgroup_barrier` between the norm and rotate phases). Opt-in via
`LARQL_FUSED_QK_NORM_ROPE=1`.

**Measured GPU-only timing** (n=10 each, on Gemma 3 4B M3 Max):

```
                     GPU median   CPU median   Wall median
FUSED QKN+ROPE       10.35 ms     0.55 ms      10.85 ms
BASELINE             10.45 ms     0.70 ms      11.08 ms
─────────────────────────────────────────────────────────────
SAVINGS              -0.10 ms     -0.15 ms     -0.23 ms ✓
```

The 0.23 ms/tok savings matches the theoretical 1-dispatch-saved ×
34-layers × ~7 µs estimate exactly. Splits cleanly into ~0.10 ms GPU
(less inter-dispatch latency in the cmd buffer) and ~0.15 ms CPU
(one fewer `set_compute_pipeline_state` + buffer-bind + dispatch
encode per layer).

`arch_gemma3_4b_gpu` produces "Paris" — bit-equivalent to the
production chain.

**Validation that the diagnosis is right**: the predicted savings
landed exactly where calculated, unlike G-1 (`F16_ACC` no-win), G-2'
(`GATE_UP_COOP` no-win), G-2 (`GATE_UP_NR2` -3% regression). This
confirms dispatch-count was the real bottleneck.

**Second fusion shipped — `residual_norm_store` in post_norms branch**:
The post_norms decode path (Gemma 3/4) was using two dispatches —
`residual_norm` then `residual_add` — when `residual_norm_store`
already does both in one kernel for the `!post_norms` branch.
Routing the post_norms branch through `residual_norm_store` is
mechanically the same fusion as the QK-norm+RoPE one. Saves another
~0.23 ms/tok. Now always-on (no env flag) since the kernel was
already battle-tested on the !post_norms path.

**Third fusion shipped — `post_attn_residual_norm_store`**:
Triple-fusion (post_attn_norm + residual + ffn_norm + h_post_attn
store) into one kernel doing 2 sequential RMS reductions per TG.
`shaders/post_attn_residual_norm_store.rs` + opt-in env
`LARQL_FUSED_POST_ATTN_NORM=1`. Math verified — `arch_gemma3_4b_gpu`
emits "Paris". **Bench result**: end-to-end 70-72 tok/s, ~0.05 ms
savings on top of stacked-2 — real but below thermal-noise floor.
The 2 RMS reductions in one TG add compute density that partially
offsets the dispatch overhead saved. Net: smaller win than the
prior two fusions; kept opt-in for completeness.

**Stacked GPU-only timing summary** (cold-state, 5 samples each):

| Configuration | GPU median | Δ vs baseline |
|---|---|---|
| Baseline (all unfused, post-2026-05-01 lm_head v5) | ~10.45 ms | — |
| + `LARQL_FUSED_QK_NORM_ROPE=1` | ~10.35 ms | -0.10 ms |
| + `residual_norm_store` (always-on) | ~10.07 ms | -0.38 ms |
| + `LARQL_FUSED_POST_ATTN_NORM=1` | ~10.02 ms | -0.43 ms |
| + `LARQL_FUSED_POST_FFN_NORM=1` | ~9.67 ms | -0.78 ms |
| + `LARQL_FUSED_KV_APPEND_ATTEND=1` | **~9.46 ms** | **-0.99 ms** |

**End-to-end tok/s** (Gemma 3 4B, 30 tokens, warm GPU):

| Path | Sustained tok/s |
|---|---|
| Pre-fix Metal (wrong output) | ~78 |
| v5 lm_head fix (correctness) | 71-72 |
| + 2 fusions stacked | 73 |
| + 3 fusions stacked | 71-72 (in noise) |
| + 4 fusions stacked (env-gated) | 74-75 |
| All 4 fusions default-on (shipped 2026-05-01) | 72-74 |
| **+ kv_append+attend fused** (shipped 2026-05-01) | **74-75** |
| Ollama gemma3:4b | 96-104 |

**kv_append+attend fusion measurement** (Gemma 3 4B, warm GPU, n=2):
- on:  GPU fwd 11.55 ms avg, 74.4 tok/s
- off: GPU fwd 11.76 ms avg, 72.7 tok/s
- delta: -0.21 ms GPU, +1.7 tok/s — matches expected dispatch saving
  (one TG per Q-head, cooperative K/V row write at pos = T-1, then
  `threadgroup_barrier(mem_device)`, then standard attention).
  Multiple Q-head TGs sharing one kv_head redundantly write the same
  row — idempotent, race-safe.

**Default-on shipped state** (no env vars needed): all five fusions
land their measured savings without flag friction. End-to-end
~74-75 tok/s sustained, generates "Paris" correctly. Opt-out flags
still wired (`LARQL_FUSED_QK_NORM_ROPE=0`, `LARQL_FUSED_POST_ATTN_NORM=0`,
`LARQL_FUSED_POST_FFN_NORM=0`, `LARQL_FUSED_KV_APPEND_ATTEND=0`)
for diagnostic A/B if regressions ever surface. The Q6_K geglu+down
fusion remains broken
and dead-code — needs kernel-level parity test against
`cpu/ops/q4_common::q6k_matvec` to localise the bug before re-engaging.

**Fourth fusion attempt — `q6k_geglu_gelu_tanh_down_cached`** (❌ both
the new cached kernel AND the existing production
`q6k_geglu_gelu_tanh_down` produce wrong output on
gemma3-4b-q4k-v2 — model collapses to "The" and stops at first decode
step). The prior memory claim "Q6_K fused kernels are
parity-tested" no longer holds against the current
`interleaved_q4k.bin` layout — likely the kernel's Q6_K block-byte
offsets drifted vs the writer in `format/weights/write_q4k` at some
point. Real fix needs a kernel-level parity test against
`cpu/ops/q4_common::q6k_matvec` reference on synthetic data, then a
re-route. Kernel and pipeline kept registered as dead code; env var
`LARQL_FUSED_Q6K_DOWN` is a no-op until the underlying bug is
diagnosed. See `shaders/q6k_geglu_gelu_tanh_down_cached.rs`.

**Remaining gap to 80 tok/s** (revised after `attn_fused` failure
2026-05-02): the simple "fuse adjacent dispatches" lever has likely
played out. Current per-layer chain has 9 dispatches; the cheap
ones (1-TG kernels) are already merged into adjacent multi-phase
kernels (post_attn_residual_norm_store, post_ffn_norm_residual_add).
The remaining 7 dispatches are all multi-TG matvecs or per-head
attention work where merging two of them would cost more in
parallelism than it saves in dispatch overhead — see `attn_fused`
post-mortem below.

**Realistic next options** (in decreasing confidence):
1. **Multi-TG-per-head attention** (split `kv_append_attend_fused`
   across the T dimension so each head uses 4-8 TGs instead of 1).
   Adds parallelism rather than fusing it away. Would let attention
   compete fairly with the matvec stages for SM occupancy. Same
   shape as the early flash-attention work (G-3 deprecated entry).
2. **Q4_K matvec ALU/cache audit** — Xcode GPU frame capture on
   `q4k_ffn_gate_up_8sg` and `q4k_q6k_qkv_proj` (G-5 still open).
   If L2 hit rate or occupancy is the bottleneck, kernel-level
   wins are still on the table. Three earlier kernel-isolated
   attempts came out null but they were optimising in the dark
   (no profiler data); a real frame capture would tell us whether
   the Q4_K matvecs are bandwidth-, cache-, or occupancy-bound.
3. **lm_head reduction** — currently 3 ms (~25% of decode). Q4
   matvec on vocab=262208 × hidden=2560. Hard, but a meaningful
   target if 1+2 don't deliver.

The "1 ms/tok via 3 more dispatch fusions" projection is **withdrawn**
after the attn_fused result — the parallelism cost on Apple Silicon
makes any further per-attention-stage merger a regression at the
current TG counts.

**Current per-layer dispatch count** (~9-10 dispatches × 34 layers):
1. fused input_norm + QKV proj (1)
2. fused QK_norm + RoPE (1, was 2)
3. V_norm (Gemma 4 only) (0-1)
4. fused KV append + attend (1, was 2 — shipped 2026-05-01)
5. O proj (1)
6. fused post_attn residual + ffn_norm + h_post_attn store (1, was 3)
7. gate + up (fused) (1)
8. GEGLU (1)
9. down (1)
10. fused post_ffn norm + residual (1, was 2)

**Where to fuse next** (in priority order, smallest scope first):
- ~~Fuse `QK_norm` + `RoPE`~~ — shipped 2026-05-01, saves 1 dispatch/layer.
- ~~Fuse `KV append` + `KV attend`~~ — shipped 2026-05-01, saves 1
  dispatch/layer × 34 = 34/tok, measured -0.21 ms.
- ❌ **Merge qk_norm_rope + kv_append_attend into one `attn_fused`
  kernel** (attempted 2026-05-02). Built `attn_fused.rs` with a single
  per-Q-head TG doing all of (norm Q+K, rope Q+K, write K/V cache,
  attend). Regressed 74.4 → 63.8 tok/s (-1.45 ms GPU). Even after
  re-using the `(cos, sin)` per rotary pair across Q and K (avoids
  duplicate transcendentals), still slower than the two-dispatch
  pair. **Diagnosis**: the standalone `qk_norm_rope_fused` runs
  `num_q + num_kv = 12` TGs in parallel; the merger collapses to
  `num_q = 8` TGs (one per Q head) with each redundantly doing its
  kv_head's K work. The dispatch saving (~30 µs) is dwarfed by the
  parallelism loss (~1.45 ms). Kernel kept registered as opt-in
  (`LARQL_FUSED_ATTN=1`) for diagnostic A/B and for a future
  multi-TG-per-head retry that preserves parallelism. **Lesson**:
  dispatch fusions only win when they don't reduce TG count for an
  already parallelism-bound stage. The earlier wins (QK_norm+RoPE,
  KV-append+attend, post_attn_residual_norm_store, post_ffn_norm)
  were either already-1-TG kernels (parallelism-free fusions) or
  preserved TG count.
- Fold `V_norm` (Gemma 4 only) into `qk_norm_rope_fused` so all three
  per-head normalisations are one dispatch. Saves 1 dispatch/layer
  × 34 = 34/tok on Gemma 4 only.
- Fuse `GEGLU` + `down`: existing `q4k_geglu_silu_down` /
  `q4k_geglu_gelu_tanh_down` kernels exist but are disabled
  (`encode_ffn.rs::use_fused = false` per a NaN finding on certain
  Q4_K-down configs). Re-test on **gemma3-4b-q4k-v2 (f16 down)**
  where the NaN issue doesn't apply — the fused-down kernel only
  fires when `down_format == Q4_K`, so f16-down vindexes already
  go through the slow path; the gate is empty for them. **G-FFN-1**
  (separate sub-item): rebuild the fused-down kernel for f16 down
  to actually engage. Saves 1-2 dispatches/layer × 34 = 34-68/tok.
- Fuse `O_proj` with `post_attn_residual_norm_store` — O_proj writes
  attn_out into a buffer that `post_attn_residual_norm_store`
  immediately reads. One TG per row could matvec then sum the
  residual in registers before the RMS reduction. 1 dispatch/layer
  × 34 = 34/tok.

**Total savings if all three land**: ~140 dispatches × 7 µs ≈ 1 ms.
Combined with no-loss retention of the v5 lm_head fix, **end-to-end
projection: ~77-80 tok/s**, closing ~1/3 of the gap to ollama.

The original "G-3 = full flash-attention" sequencing was an
overestimate — flash-attn would also need the per-position softmax
re-norm (online softmax) which is a non-trivial precision puzzle for
Gemma 3's softcapped attention logits. The smaller fusion items above
are higher-confidence, lower-risk, and stack toward the same goal.

### G-3' — DEPRECATED entry kept for context (full flash-attention)

After three failed kernel optimizations (`F16_ACC`, `GATE_UP_COOP`,
`GATE_UP_NR2`) — all targeting per-kernel ALU/cache that the
kernel-isolated profiler suggested were bottlenecks — followed by
in-pipeline GPU timing showing our per-dispatch time is already
competitive (~30 µs avg), the picture is now clear: **the gap to
ollama is dispatch count, not per-kernel speed**.

```
                     dispatches/tok    avg µs/dispatch    total
  larql              ~340             ~30 µs            ~10.5 ms
  ollama (est.)      ~200             ~40 µs             ~8.0 ms
                     ────────────────────────────────────────────
  diff               -140             slower per         +2.5 ms
```

So **G-3 (flash-attention fusion)** is the right work item — it
collapses 5-6 attention dispatches per layer (RoPE + QK_norm + V_norm
+ KV_append + KV_attend + sometimes O_proj) into 1-2 dispatches.
Saves ~140 dispatches/tok regardless of per-kernel GB/s.

The earlier "G-3 builds on G-2's NR0 understanding" sequencing note
was wrong; G-2 didn't move the needle so G-3 should go first.

### G-5 — Memory access pattern audit (deferred)

**Status**: Open. Should run before any further kernel rewrites.

**Approach**: Use Xcode's GPU frame capture / Metal Profiler on a
single decode token, focused on `q4k_ffn_gate_up` and `q4k_qkv_proj`.
Look at:
- L2 cache hit rate per dispatch (low = scattered access; high = the
  diagnosis is wrong about memory being the bottleneck).
- Concurrent threadgroup count vs theoretical (low = register
  pressure or threadgroup-mem capping occupancy).
- Memory access stall events on the FMA chain.

The output should distinguish (a) scattered access pattern hurting
prefetch, (b) low occupancy hiding latency poorly, (c) actually
ALU-bound but the existing in-kernel ALU isn't the redundant dequant.

Without this, optimization is guess-and-check. Kernel-isolated GB/s
on `metal/diag/kernel_profile.rs` doesn't predict end-to-end wins on
Apple Silicon (G-1 and the prior `LARQL_F16_ACC=1` attempt both
demonstrated this).

### G-4 — Flash-attention-style fused attention kernel

**Status**: Open. Larger lift, separate from G-1..G-3 / G-5. Promoted
toward the top of the list because it eliminates dispatch overhead
(orthogonal to per-kernel utilization), so it should win regardless
of what G-5 finds about the matmul kernels.

Per-token attention currently dispatches as:
- `q4k_qkv_proj` (Q + K + V projection)
- `qk_norm` (Gemma 3/4)
- `rope_at_pos`
- `kv_append`
- `kv_attend` (the actual `softmax(QK^T)V`)
- `q4k_matvec` (O projection)

Six dispatches per layer × 34 layers = 204 dispatches per token, each
costing ~5-8 µs scheduling overhead = 1-1.6 ms/tok in pure dispatch
time. A flash-attention-style fused kernel (`fused_attention.rs` is a
stub) would collapse RoPE+QK norm+append+attend into one or two
dispatches, saving ~0.5-1 ms/tok dispatch overhead plus the per-stage
buffer round-trips.

**Expected**: +5-10 tok/s end-to-end after G-1..G-3 are in place.

---

## Status

The four KV-cache engines shipped in `engines/kv_engines/` all reach ~93-95 tok/s
on Gemma 3 4B using the Metal Q4K path (matching Ollama within 6%). See bench:

```
larql bench gemma3-4b-q4k --engine markov-rs,unlimited-context,turbo-quant,apollo
```

---

## P0: Mechanistic hooks (lazarus parity)

Driver: replace chuk-mlx as the engine for `chuk-mcp-lazarus`. Lazarus has 77
inference-time MCP tools (capture, ablate, patch, steer, probe, DLA, KV
surgery). Larql today only writes to weights (MEMIT, KNN, LQL) — it has no
mid-forward inspection/intervention API. The whole tool surface collapses to
one missing primitive: a programmatic forward-hook system.

### M1 — `LayerHook` trait + CPU plumbing (read + write)
**Status**: In progress
**File**: `forward/hooks.rs` (new), `forward/layer.rs`, `forward/trace.rs`

Trait shape:
```rust
pub trait LayerHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {}
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {}
    fn on_attention_weights(&mut self, layer: usize, w: &AttentionWeights) {}
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {}
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {}
}
```

Insertion points in `run_layer_with_capture`: pre-layer (h entering),
post-attention (`h_post_attn`, `&mut`), FFN gate activation (`activation`),
post-attention-weights (`attn_weights`), post-layer (`h_out`, `&mut`).

The `&mut` on post-attention and post-layer is what unlocks the entire
intervention surface — ablation, steering, patching, subspace surgery are all
just `LayerHook` impls.

Plumbing strategy: `run_layer_with_capture` and `trace_forward_full` grow an
optional `&mut dyn LayerHook` parameter. Existing call sites pass `None`
(zero overhead — noop when absent). Hot generation paths in `predict.rs`
remain unchanged for slice 1; M6 wires hooks into the Metal `generate` path.

### M2 — Built-in hooks
**Status**: Not started
**File**: `forward/hooks.rs`

- `NoopHook` — never fires, used by tests.
- `RecordHook { layers: HashSet<usize> }` — captures pre/post-layer residuals
  and FFN activations; replaces the file-output path of `capture_residuals`.
- `ZeroAblateHook { layers, positions }` — zeros residual at requested coords.
- `SteerHook { vectors: HashMap<usize, (Array1<f32>, f32)> }` — adds α·v at
  specified layer's `on_post_layer`.

### M3 — Activation patching
**Status**: Not started — blocked on M1
**File**: `forward/patching.rs` (new)

Two-pass primitive: pass 1 with a `RecordHook` collects the donor residual at
(layer L, pos p) from prompt A; pass 2 runs prompt B with a `PatchHook` that
overwrites the same coords. This is the building block for `full_causal_trace`
(2D position × layer grid) — lazarus's flagship causal tool.

### M4 — Full logit lens
**Status**: Not started
**File**: `forward/predict/dense.rs`

Today: `logit_lens_top1(layer)` returns one token. Add:
- `logit_lens_topk(layer, k) -> Vec<(u32, f32)>`
- `track_token(layer, target_id) -> f32` — log-prob of a specific token at
  a specific layer.
- `track_race(layers, k) -> Vec<Vec<(u32, f32)>>` — top-k per layer in one
  pass for streaming top-k diagrams.

All three project the same captured residual through final norm + lm_head; no
new forward passes.

### M5 — KV cache surgery
**Status**: Not started
**File**: `attention/decode.rs:KvCache`

Lazarus `prefill_inject` and `kv_inject_test` need to lift K/V from one cache
into another. Add `get_layer(layer) -> (&[f32], &[f32])`,
`set_layer(layer, k, v)`, `clone_at_position(other, layer, pos_range)`.

### M6 — Hooks during multi-token generation
**Status**: Shipped
**File**: `forward/kv_generate.rs::generate_cached_hooked`,
`crates/larql-python/src/walk.rs::generate_with_hooks`

Final design: **hooks-on-CPU, Metal-stays-fast**. Lazarus-style mech interp
during multi-token generation goes through `generate_cached_hooked` on the
CPU KV-cache path; the Metal-fast `layer_graph::generate::gpu::generate*`
remains hook-free.

Why not propagate hooks into the Metal path: the Metal `decode_token` and
`prefill_q4` calls are end-to-end fused kernels that handle every layer in
one dispatch. Threading hooks in would require either CPU readback per
layer (kills the fusion benefit) or a parallel kernel surface that splits
on layer boundaries (kills the fast path even when no hook is registered).
Mech-interp tools care about correctness over throughput, so paying the
CPU-path cost when hooks are active is the right trade.

Interface mirrors `trace_forward_full_hooked` — same `LayerHook` trait;
`on_pre_layer`, `on_post_attention(&mut)`, `on_post_layer(&mut)` fire on
every layer of every step (prefill + each decode step).
`on_attention_weights` and `on_ffn_activation` do **not** fire on this
path — the production decode kernels don't capture those intermediates.
Use `trace_forward_full_hooked` for a single forward pass when you need
them.

Tests: `forward::kv_generate::tests` — noop matches baseline; record fires
on prefill + every decode step; α=5 steer changes generated tokens vs
baseline. Demo: `examples/mech_interp_demo.rs` § [7] shows
`baseline_ids = [12, 30, 10, 29]` vs `steered_ids = [4, 4, 4, 4]`.

### M7 — `W_E` / `W_U` + `project_through_unembed`

### M7 — `W_E` / `W_U` + `project_through_unembed`
**Status**: Not started
**File**: `forward/predict/dense.rs`, `lib.rs` re-exports

Lazarus tools `head_dla`, `decode_residual`, `embedding_neighbors` need
direct embedding/unembedding matrix access plus a "project this vector
through `W_U`, return top-k tokens" helper. Today both matrices are wrapped
in `VectorIndex` with no public accessor. Add `weights.embed_matrix()` and
`weights.unembed_matrix()` plus a free function `project_to_vocab_topk(vec, weights, k)`.

### M8 — pyo3 `PyLayerHook`
**Status**: Blocked on M1
**File**: `crates/larql-python/src/hooks.rs` (new)

Wrap a Python callable in a `PyLayerHook(PyObject)` that implements
`LayerHook`. Tensors crossed with `numpy.PyArray2<f32>` (zero-copy on
CPU path). MCP tools in lazarus are then just Python that registers a
hook and calls `infer()`.

---

## P0: Generation quality (blocks demo)

### Chat template — inference side
**Status**: Not started  
**Files**: `layer_graph/generate/gpu.rs`, `layer_graph/generate/cpu.rs`  
Read `tokenizer_config.json` from the vindex, parse the `chat_template` Jinja
field with `minijinja` (already in `Cargo.toml`), apply to the token sequence
before generation. `--no-chat-template` flag to bypass for base models or raw
prompts.

### EOS detection
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/eos.rs`  
`EosConfig` reads `eos_token_id` (scalar or array) and `stop_strings` from
`generation_config.json`, layered on top of `BUILTIN_STOP_STRINGS` (covers
Gemma `<end_of_turn>`, ChatML `<|im_end|>`, Llama-3 `<|eot_id|>`/`<|eom_id|>`).
Wired into `generate_with_sampling` via `eos.is_eos(id, &decoded)`. Greedy
`generate` defaults to `EosConfig::builtin()` so existing callers Just Work.

### Token spacing / detokenisation
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/detok.rs`  
`Detokenizer` keeps the cumulative ID buffer and emits only the freshly-grown
suffix on each `push`. Equivalent to llama.cpp `llama_token_to_piece` and HF
Python `decode_stream`. Handles HF leading-space (`▁`) for SP tokenizers and
multi-byte UTF-8 chars that straddle a token boundary. Demo at
`examples/detok_demo.rs` shows the bug ("thecapitaloffranceisparis") and the
fix ("the capital of france is paris").

### Token streaming
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/gpu.rs`  
`generate_streaming(..., on_token: F)` fires `on_token(id, text, prob)` for
every emitted token, including the first (which comes out of prefill). Uses
`Detokenizer::push` so streamed text preserves HF leading-space spacing.
`generate_with_sampling` is a thin wrapper passing a no-op closure so
non-streaming callers are unaffected. Demo at `examples/streaming_demo.rs`
prints tokens live with stdout flushing.

### Sampling
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/sampling.rs`  
`Sampler` + `SamplingConfig` covers greedy / temperature / top-k / top-p with
optional `seed` for reproducibility. Two paths: full-vocab `sample(logits)`
for the OpenAI-API logprob future, sparse `sample_from_topk(hits)` for the
production hot path. Wired into `generate_with_sampling`. Sparse-path
overhead is <2µs/call at top-K=64 (<0.02% of decode budget). CLI flags
(`--temperature`/`--top-p`/`--top-k`) are still owned by `larql-cli`.

### Multi-turn KV state
**Status**: ✅ Done 2026-04-26 (token-buffer) — see `layer_graph/generate/chat_session.rs`  
`ChatSession` owns the running token buffer with whole-turn eviction at
`max_context`. Pluggable `TurnRenderer` covers Gemma / ChatML / Llama-3
templates. The most recent turn is never dropped — eviction is a no-op
when only one turn remains, so a long single prompt is preserved over
silently truncating. `examples/chat_demo.rs` runs a 3-turn conversation.

True KV carryover across turns (so prefill on turn N+1 only processes
the new tokens) is a follow-up — the API surface is in place; it's an
internal optimisation.

### Gemma 3 4B regression smoke test
**Status**: ✅ Done 2026-04-26 — see `tests/test_gemma3_smoke.rs`  
Loads vindex from `LARQL_VINDEX_PATH`, runs single-token greedy generation
on `"The capital of France is"`, asserts first token (trimmed) equals
`"Paris"`. Gated `#[ignore]`; `CI_INTEGRATION=1` flips to fail-loud when
the vindex env isn't set so CI can require the test rather than silently
skip. Defaults configurable via `LARQL_SMOKE_PROMPT` / `LARQL_SMOKE_EXPECTED`.

---

## P0: MoE inference completions

### MoE-aware CPU forward pass
**Status**: Partial — vindex Q4K hidden-forward path shipped; dense `WeightFfn` path not started
**Files**: `vindex/q4k_forward/hidden.rs:169` (shipped), `forward/layer.rs` (open)
The Q4K vindex hidden-forward path already calls
`larql_compute::cpu::ops::moe::cpu_moe_forward`, so any MoE model loaded
through the vindex path has a working CPU MoE branch (this is what the
gRPC grid loopback and `M-CPU-1..6` were measured against). What's still
missing is the dense `WeightFfn::forward` path: non-vindex MoE forward
panics or no-ops. Wire `cpu_moe_forward` into `forward/layer.rs` so dense
loaders also work.

### Wire `RouterIndex` client-side
**Status**: Not started  
`larql-vindex/src/index/router.rs` exists but is not connected to the forward
pass. Connect so MoE router runs locally against the vindex before dispatching.

---

## P0: CPU MoE expert path — close the bandwidth-bound gap (Gemma 4 26B-A4B)

**Why this is P0**: The grid currently runs at **2.3 tok/s** loopback on 26B-A4B
(2 shards same M3 Max). Server compute = 95% of token wall time (250 ms/tok);
network = 2%. Theoretical CPU bandwidth floor for 4B active params at Q4_K is
~10 ms/tok = ~100 tok/s on M3 Max LPDDR5X (~400 GB/s peak), conservatively
~25 tok/s at 50 GB/s effective. We are **40-100× over the bandwidth floor** —
the gap is structural in the CPU expert path, not in kernel quality. Metal
experts measured 3.7× (→ 9.4 tok/s) but stay shipped-off pending the
`inter=704` accuracy bug (see `larql-compute/ROADMAP.md`). Closing this gap
unblocks shipping CPU-only without waiting on the Metal kernel fix and lifts
the Metal-on path proportionally once that lands.

**Target**: 25 tok/s CPU-only on Gemma 4 26B-A4B grid loopback (~10× current).

### M-CPU-1 — stop the `to_vec()` copy on cache hit
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-compute/src/cpu/ops/moe/expert.rs`  
`run_single_expert_into` was doing `let gate_up_w_f32 = v.to_vec()` on every
call, copying ~12 MB *even on cache hit*. Replaced with an
`Option<ExpertF32>` (Arc) held for the call's lifetime; `gate_w` / `up_w`
slice into the cached payload directly. No behavioural change; tests pass.

### M-CPU-2 — K=8 per-layer experts run in parallel + fold/reduce accumulator
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-server/src/routes/expert.rs`  
Confirmed the production gRPC path (`run_experts_cpu_batch`) already uses
rayon `par_iter` over the K active experts with per-rayon-thread
`ExpertScratch`. Refactored from `collect Vec<(Vec<f32>, weight)> + serial
sum` to `par_iter.fold(per-worker hidden-acc).reduce(...)`, eliminating the
per-expert 11 KB Vec allocation (~2.7 MB/token at 30 layers × K=8). Also
parallelised the HTTP `handle_expert_batch` endpoint (was serial `iter().map`).

### M-CPU-3 — `LARQL_MOE_CACHE_ENTRIES` default raised 64 → 256
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-compute/src/cpu/ops/moe/cache.rs`  
Default cap covers one full token's working set (30 layers × top-K=8 = 240
experts) with headroom. Eviction-driven p99 outliers gone (11.62 → 2.42 ms
on `cpu_moe_forward` floor). RSS cost: +2 GB per shard (9.7 → 13.6 GB on
single-shard 26B-A4B bench). Long-term answer is M-CPU-4 (kill the cache
entirely via direct Q4_K matvec); cap=256 is the right default until then.

### M-CPU-4 — NEON-vectorised Q4_K matvec (load-bearing item)
**Status**: ✅ Done 2026-05-01 — measured **8.6× sweep speedup**  
**File**: `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs` (new module);
wired in `expert.rs::run_single_expert`, `expert.rs::run_single_expert_q4k_q8k_into`,
`forward.rs::cpu_moe_forward`, `routes/expert.rs::run_experts_cpu_batch`.  
New isolated module mirrors llama.cpp's `ggml_vec_dot_q4_K_q8_K`:

- `quantize_x_to_q8k(x)` → per-256-element absmax + i8 + per-32-subblock i16 sums.
- `q4k_q8k_matvec_scalar` — scalar reference, integer dot math.
- `q4k_q8k_matvec_neon` — aarch64 SDOT inner loop (16 i8 × i8 → 4 i32 lanes
  per instruction). Implemented via stable `core::arch::asm!` because
  `core::arch::aarch64::vdotq_s32` is still unstable on Rust 1.91 (gated
  behind `stdarch_neon_dotprod`, rust-lang/rust#117224).

Test rig: Q8_K quantiser round-trip; scalar Q4_K×Q8_K vs cached-f32 path
within Q8 quant noise; multi-block matvec parity; **NEON vs scalar bit-exact**
(`to_bits()` equality on non-trivial sin/cos input); zero-dim and short-buffer
edge cases. 7 new tests, all passing.

The Q4_K direct path is on by default for Q4_K weights; `LARQL_DISABLE_Q4K_DIRECT=1`
falls back to the BLAS-on-cached-f32 path for kernel-debug A/B comparison.

Bench measurements (Gemma 4 26B-A4B, M3 Max, single-shard loopback):

| Metric | Baseline (cap=64) | M-CPU-1/2/3 (cap=256) | + M-CPU-4 (NEON Q4_K) | Total Δ |
|---|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 2.43 ms | **0.95 ms** | **2.7×** |
| `cpu_moe_forward` warm floor | 3.52 ms | 1.94 ms | **0.39 ms** | **9.0×** |
| `cpu_moe_forward` p99 | 11.62 ms | 2.42 ms | **0.50 ms** | **23×** |
| **30-layer sweep** | **221 ms** | 205 ms | **25.6 ms (0.85 ms/layer)** | **8.6×** |
| Steady RSS | 11.4 GB | 13.6 GB | **10.5 GB** | -8% |

The sweep at 25.6 ms projects to ~25-30 tok/s end-to-end on the gRPC grid
(vs 2.3 tok/s baseline = ~10-13× end-to-end). RSS dropped below baseline
because the f32 dequant cache is largely inert in the new path —
direct-Q4K reads straight from mmap.

Follow-ups (if further perf needed): (1) shrink `LARQL_MOE_CACHE_ENTRIES`
default back to 64 or 32 once the BF16 fallback path is removed (cache only
serves legacy BF16 vindexes now); (2) reuse per-rayon-thread scratch for the
`gate_out` / `up_out` / `act_q8k` heap allocs in `cpu_moe_forward`'s
direct-Q4K branch (currently per-call); (3) wire AVX2 dot-product equivalent
for x86 hosts (`_mm256_maddubs_epi16`).

### M-CPU-5 — bench harness + per-fix tok/s attribution
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-server/examples/bench_expert_server.rs` (+ pre-existing
`unit_filter` fixture compile fix; two-shard mode has a separate pre-existing
expert-127 off-by-one).  
Single-shard bench on `output/gemma4-26b-a4b-q4k.vindex` (M3 Max, 2026-05-01):

| Metric | cap=64 | cap=256 (new default) | Δ |
|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 2.43 ms | -4% |
| `cpu_moe_forward` warm floor (mean) | 3.52 ms | **1.94 ms** | **-45%** |
| `cpu_moe_forward` p99 (eviction outliers) | 11.62 ms | 2.42 ms | **-79%** |
| 30-layer sweep | 221 ms | 205 ms | -7% |
| Steady RSS | 11.4 GB | 13.6 GB | +19% |

The per-call floor improvement is real; the sweep regression vs the
ROADMAP-published 56 ms (from 2026-04-26) is on current code regardless of
cap, indicating a code drift between then and now that should be bisected
separately. The point of the bench: it falsifies "more cache = more tok/s"
as the path to target, and confirms M-CPU-4 (NEON direct-Q4K, no f32 cache)
as the only structural answer.

### M-CPU-6 — Bottleneck-driven follow-ups (post-NEON profiling round)
**Status**: ✅ Done 2026-05-01  
**Files**: `q4k_q8k_dot.rs`, `cpu/ops/q4_common.rs::f16_to_f32`,
`moe/expert.rs`, `moe/forward.rs`, server `routes/expert.rs` +
`larql-inference/ffn/moe_remote.rs`.

After M-CPU-1..4 landed, samply (`/usr/bin/sample bench_expert_server 30`)
identified the next bottlenecks:

1. **f16-to-f32 was calling `__powisf2`**.  `2.0f32.powi(exp - 15)` lowered
   to a libcall; ~11 M decodes/token at 26B-A4B sizes routed through the
   software powi.  Replaced with pure-integer bit-manipulation
   (`f16_to_f32`).  Bit-exact for all 65,536 inputs (test:
   `f16_to_f32_bit_exact_for_all_inputs`).  Removed the bl from the
   kernel — but wall-clock barely moved, which DIAGNOSED the kernel as
   already DRAM-bandwidth bound (the powi work was hidden in memory
   stalls).

2. **Reusable Q8_K activation buffer (`act_q8k`) in `ExpertScratch`**.
   The per-expert activation Q8_K quantisation was allocating a fresh
   `Q8KActivation` per call; ~5% of calls hit a 150 µs allocator slow
   path that dragged par_iter wall up.  Added `act_q8k` field +
   `quantize_x_to_q8k_into` API + `Q8KActivation::with_capacity`.
   `forward_moe` p99 dropped 23% (1.38 → 1.06 ms).

3. **`cpu_moe_forward` refactored to use thread-local `ExpertScratch` via
   rayon `fold/reduce`**.  Eliminates the per-expert
   `Vec<f32>` allocs in the in-process MoE path AND deduplicates the
   kernel logic (now goes through `run_single_expert_q4k_q8k_into`
   instead of an inlined copy).

4. **`run_single_expert` (HTTP single-expert entry) now uses thread-local
   scratch on the Q4_K path**.  K=8 calls per layer no longer
   re-allocate gate_out / up_out / act / act_q8k; only the final
   returned `Vec<f32>` is allocated.

5. **New `/v1/experts/layer-batch` endpoint + wire format**.  Ships ONE
   residual + K (expert_id, weight) pairs per call (vs K identical
   residuals on the legacy `/v1/expert/batch` path).  Server applies
   `pre_experts_norm` once + Q8_K quantises h_norm once + fans out the
   K experts via `run_experts_cpu_batch`.  `RemoteMoeBackend::forward_moe`
   updated to use the new endpoint.  Saved K-1 redundant pre-norm + Q8K
   quantisations and ~2.6 MB/token of redundant residual on the wire.

6. **Tried fused gate+up matvec**.  Implemented `q4k_q8k_gate_up_into`
   with NEON SDOTs interleaved across both matrices.  Bit-exact parity
   test against back-to-back single-matvec calls (`q8k_gate_up_fused_matches_separate_matvecs`).
   Bench: ~4% slower on the 30-layer sweep.  M3 Max OoO engine
   already extracts ILP from the back-to-back independent matvec calls;
   manual interleaving adds register pressure and hurts the L1 prefetcher
   pattern.  Reverted the wiring; kept the function for future
   architectures where the trade may flip.

End-state bench (Gemma 4 26B-A4B, M3 Max, single-shard loopback):

| Metric | Baseline (cap=64) | M-CPU-1..4 | + M-CPU-6 (post-profile fixes) | Total Δ |
|---|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 0.95 ms | **0.83 ms** | **3.0×** |
| `cpu_moe_forward` warm floor | 3.52 ms | 0.39 ms | **0.38 ms** | **9.3×** |
| **30-layer sweep** | **221 ms** | 25.6 ms | **24.2 ms (0.81 ms/layer)** | **9.1×** |
| Steady RSS | 11.4 GB | 10.5 GB | 10.5 GB | -8% |

Sweep at 24.2 ms projects to **~25-30 tok/s end-to-end on the gRPC grid**
(vs 2.3 tok/s baseline = ~10-13× end-to-end).  Path is now firmly
DRAM-bandwidth bound (~32 GB/s aggregate vs ~50-100 GB/s practical M3
Max CPU peak); further wins require structural changes (multi-row
matvec sharing super-block reads across output rows, prefetch
instructions ahead of SDOT loads, or simply waiting on the Metal MoE
expert kernel fix to land for an additional ~3.7× via GPU dispatch).

---

## P0: Engine performance parity

### TurboQuant Metal K/V checkpoint compression
**Impact**: Reduces boundary checkpoint from 278 KB → 36 KB/window (7.7×) for long contexts.  
**Status**: TurboQuant runs at Metal speed. Compressed boundary checkpoints require
Metal K/V read-back. Add `backend.get_kv_last_position(layer)` to the Metal backend.

### Apollo `prefill_to_layer` — true layer-skip
**Impact**: ~20% faster per step in compressed path.  
**Status**: `forward_from_layer` ships; K/V seeding at `crystal_layer` is a follow-up.

### Apollo store builder
**Impact**: Currently requires pre-built NPY/NPZ files.  
**Status**: Not started. `ApolloEngine::build_from_document(weights, tokenizer, tokens)`.

---

## P0: Evaluation parity (blocks architecture claims)

larql is a research engine for novel architectures (WalkFfn, vindex KV engines, gate
KNN, layer-skip via Apollo). To show an architecture is competitive we need to run
the same eval harnesses other engines run — otherwise we are only ever comparing
synthetic prompts to synthetic prompts. The items below build on the generation-quality
P0 above (sampling, streaming, chat templates, multi-turn KV); without those, none
of the harnesses load at all. Goal is parity for fair evaluation, not feature
parity for its own sake.

### Per-position logprobs / top-k logprobs
**Status**: Not started  
**Files**: `forward/predict/raw.rs`, expose via `lib.rs`  
Add `forward_logprobs(weights, token_ids, target_ids) -> Vec<f32>` returning
per-position log-likelihood of `target_ids[i]` given prefix `token_ids[..i]`.
Also expose top-k logprobs from `forward_raw_logits`. lm-evaluation-harness and
most multiple-choice benchmarks (HellaSwag, ARC, MMLU, WinoGrande, PIQA) score
by sequence log-likelihood, not generation. Without this no likelihood-class
benchmark can run, so no architecture claim has a published comparator.

### OpenAI-compatible HTTP API
**Status**: Not started  
**Files**: `crates/larql-server/src/openai/` (new), thin wrapper over inference  
`larql-server` exposes `/v1/infer` and `/v1/walk`; eval frameworks (lm-eval-harness,
simple-evals, evalplus, AlpacaEval, swe-bench harnesses) plug into
`/v1/chat/completions` and `/v1/completions`. Add OpenAI-shape endpoints as a
wrapper over `generate` + sampling + chat-template rendering + logprob fields.
Unlocks every harness without per-harness adapters.

### Batch inference (independent prompts)
**Status**: Not started  
**Files**: `forward/predict/`, new `predict_batch`  
Distinct from continuous batching. Eval suites issue thousands of independent
prompts; serial execution makes a single benchmark run take hours-to-days. Add
`predict_batch(weights, prompts: &[Vec<u32>]) -> Vec<Vec<f32>>` that prefills each
prompt against the same weight mmap. Each prompt gets its own KV-engine instance,
so all four engines work unchanged.

### LoRA / adapter loading at runtime
**Status**: Not started  
**Files**: `forward/layer.rs`, `larql-models` weight loader  
Many arch papers ship LoRA-tuned variants (instruction-tuned on top of a base).
Without LoRA, larql cannot compare `WalkFfn` on `Gemma-3-4B-base` vs
`Gemma-3-4B-it` without re-quantising a merged model. Add
`WeightSet::with_lora(adapter_path)` wrapping `gate/up/down/q/k/v/o` matmuls as
`W·x + α·B(A·x)`. Stretch: composable adapter stack for ablation
(WalkFfn + LoRA-A vs WalkFfn + LoRA-B on the same base).

### Eval-harness smoke run
**Status**: Not started  
End-to-end test: run lm-eval-harness `hellaswag` (10 samples) against
`larql-server` and assert non-zero accuracy. Gate on `CI_INTEGRATION=1`. This
is what moves "we have logprobs" from a unit test to "harnesses actually plug in."

---

## P1: Eval-class coverage

Each item below unlocks a specific class of evaluation. Land in the order an arch
claim needs them — no need to do all up front. Prerequisite for all of them: the
P0 evaluation-parity stack above.

### Structured output / GBNF grammar / JSON Schema
**Status**: Partial — regex/grammar hook exists in `generate`; not wired to JSON
Schema or BNF.  
**Unlocks**: JSONSchemaBench, BFCL (function-calling leaderboard), any eval
requiring schema-conformant output.  
Apply a constrained-decoding mask over logits before sampling. Minimum viable:
GBNF parser (port from `llama.cpp` grammar.cpp); JSON Schema compiles to GBNF.

### Vision / multimodal forward
**Status**: Not started  
**Unlocks**: MMMU, ChartQA, DocVQA, multimodal subsets of larger suites.
Validates that WalkFfn and the four KV engines work on multimodal weights, not
just text.  
Gemma 3 (4B/12B/27B) and Llama 3.2 ship vision variants; vision-tower weights
are already in safetensors. Add image-embedding pipeline → token-mixing →
existing decoder forward. No new KV-engine work required (image tokens look
like text tokens to the decoder).

### Tool / function calling
**Status**: Not started — depends on chat templates (P0) + structured output
(P1 above).  
**Unlocks**: BFCL, ToolBench, AgentBench, any agent-style eval.  
Once the two prerequisites land this is template glue: parse tool-call markers
in the rendered chat template, emit structured calls via the constrained-decoding
path.

### Speculative decoding
**Status**: Not started  
**Why this matters for arch claims**: any "WalkFfn at X tok/s" comparison
against engines that ship speculative decoding (vLLM, TGI, llama.cpp `--draft`)
is misleading without it. Speculative decoding also interacts non-trivially with
gate KNN — draft and target may diverge on top-k feature selection, which is its
own arch question worth answering.  
**Path**: self-spec via `forward_from_layer` (early-exit verification) is the
cheapest entry; full draft-target spec is a follow-up.

### Trace capture during eval batches
**Status**: Partial — `trace_forward_full` works on single prompts.  
Extend to the batch + logprob path so mechanistic interpretability can use
eval-set inputs without re-running. This is what makes "we ran HellaSwag and
the WalkFfn-replaced layers behaved like X" a single-pass measurement.

---

## P0: Mechanistic trace correctness audit

From the 2026-05-02 interpretability review. These items are correctness
requirements for using TRACE as evidence, not polish.

### Route decomposed TRACE through the real layer sequence
**Status**: Shipped 2026-05-02
**Files**: `trace/capture.rs`, `forward/layer.rs`, `trace/types.rs`,
`crates/larql-lql/src/executor/trace.rs`
`trace_residuals` currently records attention and FFN deltas but stops at
`h_post_ffn`, while the production layer path also applies per-layer embeddings
and layer scalar. Rework trace capture so the recorded residual is the same
state the next layer actually sees. Either add explicit `ple_delta` /
`scalar_delta` components, or route through shared layer intermediates and
derive all deltas from the canonical runner.

### Python WalkModel.trace must use vindex FFN
**Status**: Shipped 2026-05-02
**Files**: `crates/larql-python/src/walk.rs`, `crates/larql-python/src/trace_py.rs`
`WalkModel.trace()` should construct a `WalkFfn` from `self.index` and preserve
patch/overlay semantics. The current dense `WeightFfn` trace is useful as a
baseline, but it is not the trace of the vindex-backed model the user is
querying.

### Trace save contract must fail loudly on incomplete artifacts
**Status**: Shipped 2026-05-02
**Files**: `trace/store.rs`, `trace/types.rs`,
`crates/larql-lql/src/executor/trace.rs`
Persisted chain traces now require complete ordered token chains and exact file
lengths. `TRACE ... SAVE` requires `POSITIONS ALL` so downstream mmap readers do
not silently consume partial traces as if they were complete context graphs.

### Golden parity tests for TRACE as evidence
**Status**: Partial — dense and custom backend parity shipped 2026-05-02
**Files**: `trace/capture.rs`, `tests/test_trace.rs`, Python binding tests
Final trace residuals now project to the same logits as the canonical dense
raw-forward path, and a custom `FfnBackend` trace matches the generic hooked
forward runner. Extend the matrix to WalkFfn, patched-vindex, Q4K, and MoE as
those test fixtures become cheap enough to run in CI.

### Rank displayed gate features by contribution, not raw |dot|
**Status**: Planned  
**Files**: `vindex/walk_ffn/sparse.rs`, `forward/infer_patched.rs`,
`crates/larql-vindex/src/index/compute/gate_knn/dispatch.rs`  
Interpretability displays should not surface strongly negative gate pre-acts as
"active" features when SiLU/GELU makes their contribution near zero. For WALK /
EXPLAIN displays, rank or filter by post-gate activation magnitude and ideally
include each feature's estimated FFN contribution.

### L1 cache must not fabricate zero activations
**Status**: Planned  
**Files**: `vindex/walk_ffn/mod.rs`, `vindex/l1_cache.rs`  
On cache hit, `forward_with_activation` currently returns the cached FFN output
and an all-zero activation matrix. Either store activations with outputs or
bypass the cache whenever activation capture is requested.

### Separate embedding-neighbor labels from logit-lens labels
**Status**: Planned  
**Files**: `capture.rs`, `trace/vocab.rs`  
Residual capture metadata that projects through `W_E` should be named as
embedding-neighbor output, not model belief. Add separate final-norm + lm-head
logit-lens fields where callers need prediction trajectories.

---

## P1: Architecture coverage

### Wire v_shares_k into forward pass
**Effort**: Low — `v_shares_k()` already in larql-models; swap runtime check.

### Validate PLE end-to-end (Gemma 4 E2B)
**Effort**: Medium — config parsed; forward pass not yet wired.

### KV layer sharing for Gemma 4
**Effort**: Medium — `kv_shared_source_layer()` returns correct sources; cache allocation not yet sharing.

### Llama 3 / Gemma 4 engine validation
All four engines validated on Gemma 3 4B. Need empirical `cos h = 1.000000` validation on Llama 3 / Gemma 4.

### MarkovRS batched K/V recompute kernel
**Impact**: Eliminate 2000× FLOP overhead on CPU decode path.  
**Effort**: Medium (new Metal shader for `[W, hidden] @ [hidden, kv_dim]` Q4K projection).

---

## P1: Structure & file layout

From 2026-04-26 code review. All public APIs preserved; changes are internal re-organisation.

### High priority

**`ffn/remote.rs` (893 LOC) — split into `remote/`** ✅ Done 2026-04-26  
`ffn/remote/codec.rs` — binary codec, wire types, latency stats, codec tests.  
`ffn/remote/http.rs` — RemoteFfnConfig, RemoteWalkBackend, RemoteFfnError, HTTP tests.  
`ffn/remote/mod.rs` — thin re-export + protocol doc.  
No magic strings: `BINARY_CT`, `BATCH_MARKER`, `STATS_PATH`, `WALK_FFN_PATH` are named constants.

**`turbo_quant/mod.rs` → `turbo_quant/engine.rs`** ✅ Done 2026-04-26  
TurboQuantEngine + TurboQuant codec moved to `engine.rs`. `mod.rs` is a thin re-export of sub-modules + `pub use engine::{TurboQuantEngine, TurboQuant}`.

**`vindex/walk_ffn/mod.rs` → `walk_ffn/engine.rs`**  
Deferred: walk path submodules use `pub(super) impl WalkFfn` blocks that are
architecturally tied to `mod.rs` as the parent. Requires changing visibility to
`pub(in crate::vindex::walk_ffn)` across 6 files — low risk/reward compared to
other P1 items. Backlog.

**`layer_graph/predict.rs` (881 LOC) — split** ✅ Done 2026-05-10 (H12)
Split into `predict/{mod.rs 376, split.rs 285, honest.rs 261}`. See
CHANGELOG.md `[2026-05-10] — Hardening pass`.

**`residual.rs` at crate root → `forward/norm.rs`**  
It's a collection of norm primitives used exclusively by the forward pass. Moving
it co-locates it with the other forward utilities (`ops.rs`, `layer.rs`).

**`capture.rs` at crate root → `trace/`**  
`InferenceModel` / `CaptureConfig` belong with the trace infrastructure.

### Medium priority

**Softmax in 5 locations — unify**  
`trace/vocab.rs`, `engines/accuracy.rs`, `ffn/moe_remote.rs`,
`layer_graph/logits.rs`, `forward/target_delta.rs` each have a private softmax.
Promote `engines/accuracy.rs::softmax` to `forward/ops.rs` (or `residual.rs`);
have the others `use crate::forward::softmax`.

**`embed_tokens_pub` / `run_attention_public` naming**  
The `_pub` suffix is redundant on public functions. Rename to `embed_tokens` and
`run_attention` or document why the suffix exists. `_pub` vs `_public` is also
inconsistent.

**`ApolloEngine` and `TurboQuantEngine` not re-exported at crate root**  
`MarkovResidualEngine` and `UnlimitedContextEngine` are re-exported; the other
two engines are not. Either export all four or none.

**`experts/` has no module-level docs**  
Add a `//!` header explaining purpose and entry points. (Walkers live in `larql-vindex`, not here.)

**`vindex/` module doc is vague**  
"Vindex integration" says nothing to a new reader. Expand to explain what the
vindex is and what this module provides.

### Low priority

**`forward` re-export block is 70+ items with no sub-grouping**  
Split into clearly commented groups: prediction, tracing, raw logits, analysis
(memit, target_delta, infer_patched).

**`trace as trace_decomposed` alias in `lib.rs`**  
Aliases a naming problem rather than fixing it. Rename the function itself.

**`RawForward` is an implementation detail in the public API**  
Users never construct `RawForward` directly; it's only returned by
`forward_raw_logits`. Consider whether it needs to be pub.

**`generate_cached*` in `forward/` vs `generate` in `layer_graph/`**  
Two generation APIs with similar names but different semantics (CPU KV-cache step
vs Metal fused pipeline). Add a clear doc comment on each explaining the difference.

---

## P1: Quality bugs (from 2026-04-26 review)

### `grid.rs` — hardcoded `eos_id = 1` is a real bug ✅ Fixed 2026-04-26
**File**: `layer_graph/grid.rs`  
Replaced `eos_id: u32 = 1` with `is_end_of_turn(tok_str.trim())` on both the prefill-exit
and decode-loop paths, matching all other generation code.

### Softmax duplicated in 5 locations ✅ Fixed 2026-04-26 (2 of 5)
**Files**: `trace/vocab.rs`, `engines/accuracy.rs` now use `pub use crate::forward::softmax`.  
Canonical implementation lives in `forward/ops.rs`, exported via `forward/mod.rs`.  
`ffn/moe_remote.rs` (in-place `&mut [f32]`), `logits.rs` (single-prob extractor),
`target_delta.rs` (Array1) remain local — different enough to not unify.

### `forward/ple.rs` hardcodes `1e-6` norm epsilon ✅ Fixed 2026-04-26
`1e-6` replaced with `arch.norm_eps()` for consistency.

### `rms_norm` / `layer_norm` default eps in CPU forward path ✅ Fixed 2026-05-16
The CPU forward path (`forward/{layer,ops}.rs`, attention/block.rs,
attention/decode.rs) was reading `crate::residual::rms_norm(...)` which
defaulted to `DEFAULT_EPS = 1e-6`, ignoring `arch.norm_eps()`. New
`rms_norm_for_arch` / `layer_norm_for_arch` helpers in `residual.rs`
consult `arch.norm_eps()` (now config-driven in larql-models) with the
`LARQL_NORM_EPS_OVERRIDE` env var as diagnostic override. Production
callers switched. Closed Mistral 7B's +8.2% bits/char drift and most of
Llama 3.2's +0.59%. See
[`docs/diagnoses/shannon-cross-engine-divergence.md`](../../docs/diagnoses/shannon-cross-engine-divergence.md).

### Shannon cross-engine verify instrument + RoPE scaling support ✅ Landed 2026-05-16
- `attention/rope.rs`: new `Llama3Scaling` struct + `apply_rope_partial_at_full`
  variant that consumes both `position_divisor` (linear `rope_scaling.factor`)
  and `llama3_scaling` (wavelength-dependent per-channel `inv_freq`
  adjustment matching HF's `_compute_llama3_parameters`). Default
  `apply_rope_partial_at` is now a wrapper passing `1.0` / `None`.
- `attention/block.rs` + `attention/decode.rs`: prefill and decode RoPE
  call sites route through `apply_rope_partial_at_full` with
  `pos_divisor` and `llama3` sourced from
  `layer_graph::pipeline_layer::effective_rope_position_divisor_for_layer`
  and `effective_llama3_rope_scaling` (arch-driven, env-var override).
- `layer_graph/pipeline_layer.rs`: five `OnceLock`-gated diagnostic env
  vars (`LARQL_FORCE_GLOBAL_LAYERS`, `LARQL_ROPE_POS_DIVISOR`,
  `LARQL_ROPE_POS_DIVISOR_GLOBAL`, `LARQL_LLAMA3_ROPE_SCALING`,
  `LARQL_NORM_EPS_OVERRIDE`) — production code paths reach for the arch
  first and fall back to env-var overrides for bisection. No behaviour
  change when unset.

The `larql shannon verify` CLI command in larql-cli orchestrates this
forward path against HF/PyTorch + MLX reference scorers. See
[`docs/cli.md`](../../../docs/cli.md#cross-engine-verify) and
[`scripts/README_shannon_score.md`](../../../scripts/README_shannon_score.md).

### `grid.rs` undocumented `SKIP_MOE` env var ✅ Fixed 2026-04-26
Added `# Diagnostics` section to module doc.

---

## P1: Test coverage gaps

Baseline 2026-04-26: 50.45% line coverage.
Current 2026-05-09: **53.11% line coverage, 631 lib tests, 32/116 files at ≥90% per-file floor (84 below).**

Most of the remaining gap is concentrated in:
- 33 files at **0% line coverage** — GPU dispatch (`generate/gpu.rs`), Q4_K integration paths (`vindex/q4k_forward/*`), remote-shard orchestration (`grid/remote_*`, `ffn/moe_remote/*`). Mostly need real model fixtures or running services to cover.
- 9 files at 1-29% — `attention/gpu.rs`, `forward/memit.rs`, `ffn/remote/http.rs`, `vindex/walk_ffn/sparse.rs`. Reachable with synthetic weights + tempfile fixtures.
- 17 files at 70-89% — mechanical floor lifts (the 80-89% bucket cleared 2026-05-09).

### Critical

**`markov_residual/` — zero tests across all 5 new files** ✅ Done 2026-04-26  
`store.rs`: clip_layer edge cases (no-window noop, at-limit, over-limit), memory_bytes, window_tokens.  
`engine.rs`: name, memory lifecycle, prefill→decode cycle, window clipping, multi-step shapes.  
`compute.rs`: recompute_kv shape/finiteness/RoPE shift, rs_prefill result shape + window, rs_decode_step position advance.

**`ffn/sparse_compute.rs` and `ffn/sparse.rs` — zero tests** ✅ Done 2026-04-26  
`sparse_compute.rs`: empty-features→zeros, single/multi-token shape, top-K ordering, dense-fallback equivalence, down-override effect.  
`sparse.rs`: name, all-layers shape/finiteness, top-k vs dense match, with_activation shapes.

**`ffn/graph_backend.rs` — zero tests** ✅ Done 2026-04-26  
Construction (layer count, empty layers), lookup_from_tokens (top-K limit, unknown layer, empty scores, out-of-range tokens), precompute_entity, save/load roundtrip.

**`layer_graph/` — 7 of 17 files untested** ✅ All 7 done 2026-04-26  
`dense.rs` — DenseLayerGraph shape/finiteness/capture, PerLayerGraph bounds.  
`walk.rs` — WalkLayerGraph all-layers, PipelinedLayerGraph in/out-of-range.  
`mod.rs` — trait dispatch, name distinctness.  
`prefill.rs` — CPU path: shape, finiteness, partial range, empty range, logit correctness.  
`template.rs` — detect_template (7 pure tests), TemplateUniverse build/get/total, GuidedWalkLayerGraph shape/finiteness.  
`pipeline_layer.rs` — build_arch_params param extraction, resolve_attn_weights None path, resolve_ffn_weights legacy stride slicing.  
`grid.rs` — error path: no Q4K mmap → `Err(BadResponse)`.  
Integration tests: `tests/test_layer_graph_integration.rs` — real vindex tests for prefill_with_kv, build_pipeline_layers, TemplateUniverse, GuidedWalkLayerGraph (all `#[ignore]`, run with `--ignored`).

### High priority

**`forward/ops.rs` — zero tests** ✅ Done 2026-04-26  
`dot_proj`: shape, identity-weight, value-correctness.  
`add_bias`: all-rows updated, shorter-bias safe, zero-bias noop.  
`apply_norm`: shape, finite output, offset produces different result.

**`forward/ple.rs` — zero tests** ✅ Done 2026-04-26  
precompute returns empty for non-PLE arch, apply_ple None/missing-weight guard paths,
output shape. Softmax tests moved here as a side-effect of unification.

**`engines/kv_engines/unlimited_context/extend.rs` — zero tests** ✅ Done 2026-04-26  
empty_prior shape, empty-tokens/wrong-prior-len → None, single/multi-token extend, kv_cache
row count, checkpoint = last-row, abs_start shifts RoPE, finite logits, chained extends.

### Medium priority

**GQA head grouping (`reps` parameter) not tested** ✅ Done 2026-04-26  
Three tests: output shape (4Q/2KV/reps=2), finiteness, and head-pair sharing — heads 0 & 1
sharing KV-head 0 produce identical output rows.

**RoPE missing property tests** ✅ Done 2026-04-26  
rope_base sensitivity, fraction=1.0 equals full-rope, offset=N matches sequential position N,
partial fractions 0.25/0.5/0.75 all finite.

**No synthetic end-to-end tests for `generate()`**  
`generate()` (Metal GPU path) is only tested with `#[ignore]` real-model tests.
Add a synthetic CPU-backend integration test using `make_test_weights()`.

---

## P2: Spec'd and queued (sequenced behind P0 validation)

Two implementation tracks have shipped specs at
`crates/larql-inference/docs/specs/` but are deliberately queued behind
V1–V4 / R6 / MTP / BR4. Detail and gating preconditions live in the
top-level `ROADMAP.md` § "P1 — Spec'd implementations, sequenced behind
P0 validation" (SQ1, SQ2). Re-promotion conditions are recorded there.

| # | Spec | Status | Why queued |
|---|------|--------|------------|
| SQ1 | [`markov-residual-engine.md`](docs/specs/markov-residual-engine.md) | ✅ shipped | Production impl lives in `larql_kv::engines::markov_residual`. The trait-vs-sibling question resolved via `KvEngine` + `KvDispatch`. The `kv-cache-benchmark` reference impl that the spec lifted from was retired in 2026-05-16. |
| SQ2 | [`vindex-as-ffn.md`](docs/specs/vindex-as-ffn.md) | reviewed, impl not started | §5.4 cost model says it's a wash on typical decode K (256–1024) without large compiled-fact corpora. R6 (depth-fraction probe) needs to land before the per-arch layer policy is automated rather than hand-calibrated. No current video / research workflow needs the paraphrase-reach the lookup buys above the existing L1 i16 cos≥0.999 cache. |

These are not P0/P1-active. Top-level roadmap is the source of truth
for sequencing; this entry exists so the specs are findable from inside
the crate that owns them.

---

## P2: Research

### Hybrid head caching (RS+CA)
95.5% of attention heads are static (cacheable). Would give ~180-370× compression
at 370K tokens — between TurboQuant (4×) and MarkovRS (287×) with near-exact accuracy.

### Graph Walk engine
FFN graph walk is proven (348K features, 34 layers, zero accuracy loss).
Full RS Graph Walk requires cracked attention (static head caching).
`GraphWalkEngine` would eliminate the forward pass entirely for parametric queries.

### Continuous batching + paged attention (deferred)
**Why deferred**: arch claims larql cares about are likelihood-bounded, not
throughput-bounded. PagedAttention-style KV management interacts with all four
KV engines (each has its own checkpoint geometry), and the design work isn't
worth it until a specific eval forces it. Revisit if a throughput-class
benchmark becomes load-bearing for an arch claim.

### Multi-GPU / tensor-parallel (deferred)
`larql-grid` already shards layers across hosts. Tensor-parallel within a layer
is a separate problem and not on the critical path until 70B+ models become the
bottleneck.

---

## Completed work

Dated entries live in [`CHANGELOG.md`](CHANGELOG.md). Headline milestones:

- **2026-05-10** — Hardening pass H8/H9/H11 shipped; H12 down to one file (`generate/gpu.rs`); 639 lib tests.
- **2026-05-09** — Cleanup pass: blanket allows dropped, f16 codec via `half`, `forward/predict/raw.rs` dedup, 8 files crossed 90% coverage floor.
- **2026-05-02** — Mechanistic interpretability engine surface MI0–MI3 + dense/custom-backend MI4 parity.
- **2026-04-30** — gRPC grid + dense Metal chat templates + Gemma 4 four-variant accuracy.
- **2026-04-27** — Q4_K stride validation, strict vindex loader.
- **2026-04-26** — Generation quality (EOS, detok, sampling, streaming, ChatSession) + structural splits + ~100 new tests.
- **2026-04-25** — `KvEngine` trait, MarkovResidual, UnlimitedContext, BackendFfn, profiler.
- **2026-04** — `WalkFfn`, `LayerGraph` trait, `predict_honest`, GPU prefill, Q4_K FFN format wiring, TurboQuant, Apollo, Metal Q4K parity.
- **2026-03** — Forward pass (CPU BLAS) foundation.
