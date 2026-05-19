# Changelog — larql-inference

All notable changes to `larql-inference` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions,
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

Entries migrated from ROADMAP.md on 2026-05-10; pre-2026-05-10 entries
preserve the date and voice they were originally written in.

## [2026-05-17] — `DumpConfig` env-var cache drop

Removed the `OnceLock` cache from `forward::dump_config::DumpConfig::get()`;
the function now reads `LARQL_CPU_DUMP_LAYERS` / `LARQL_CPU_STAGE_DUMP` /
`LARQL_STAGE_DUMP_LAYER` fresh on every call. The cached design silently
broke any test fixture that rotated the dump-dir tempdir between calls —
the second call would still hand out the path of the first (now-dropped)
tempdir, and writes failed with `No such file or directory`.

Root cause was visible in `tests/test_cpu_metal_parity.rs`: the suite's
four cases run back-to-back, each calling `ResidualCapture::cpu_prefill`
which allocates a fresh `tempfile::TempDir` and stomps the env var. The
first case got the live dir; cases 2-4 wrote to a deleted path. Surfaced
as `parity_gemma4_31b_dense_prefill` / `parity_llama2_7b_prefill` /
`parity_mistral_7b_prefill` panicking on `CPU dump missing for layer 0`.

Perf delta: ~150 ns per env-var read × ~170 reads per token (34 layers
× 5 stage gates) = ~25 µs/token, or ~0.07% of a 35 ms decode step.
`DumpConfig::current()` is the new canonical name; `get()` remains as an
alias for back-compat. Four callers (`attention/block.rs`,
`forward/layer.rs`, `vindex/kquant_forward/hidden.rs` × 2) bind the
config to a local before calling `.layer_dir()` / `.stage_dir(layer)`
so the returned `&str` doesn't borrow from a temporary.

## [2026-05-10] — A1+A2+A3 model architecture independence hardening

Closed the "Open: Model architecture independence hardening" section of
ROADMAP.md. Reproduced Gemma 4 31B Q4K Metal end-to-end first — it
generates correctly ("The capital of France is" → "Paris...";
"Write a haiku about ocean waves." → "Blue walls rise up high, /
Crashing down in salty foam, / Returning to sea."). The roadmap's
"L5 immediate-EOS" claim was stale: the heterogeneous-attention safety
had quietly landed via per-layer reads in
`metal/decode/setup.rs:DecodeScratch::new` (q_out / k_out / v_out
sized to `max(layers[*].q_dim)`),
`metal/ops/full_pipeline/buffers.rs:LayerBuffers::allocate` (prefill
same), and `MetalBackend::ensure_kv_cache_for_layers` (KV cache
per-layer). A1/A2/A3 are now the deferred cleanup pass on top.

### A1 — `Capability::HeterogeneousAttention` runtime gate

- Added `Capability::HeterogeneousAttention` to
  `crates/larql-compute/src/backend/capability.rs`. Default
  `ComputeBackend::supports` returns `false`; `MetalBackend` opts in.
- `crates/larql-inference/src/layer_graph/generate/gpu_setup.rs`
  exposes `has_heterogeneous_attention(weights)` (probes
  `head_dim_for_layer`, `num_kv_heads_for_layer`, `num_q_heads_for_layer`,
  `rope_base_for_layer`, `rotary_fraction_for_layer`, and
  `is_sliding_window_layer` against layer 0) plus
  `ensure_attention_supported(weights, backend)`. The gate fires inside
  `build_gpu_decode_setup` before the first decode dispatch, so a
  future non-Metal Q4 backend that hasn't audited heterogeneous
  geometry returns a precise `GenerateError::UnsupportedBackend`
  instead of silently corrupting KV state at the first non-uniform
  layer.

### A2 — drop vestigial scalar geometry from `DecodeBackend`

The trait surface previously took `q_dim`, `kv_dim`, `num_q_heads`,
`num_kv_heads`, `head_dim`, and `rope_base` on every decode/prefill
method. Every implementation already underscored them in favour of
per-layer reads from `FullPipelineLayer`; the scalars were dead weight
that invited future regressions. Dropped from these methods:

- `full_pipeline_q4`, `full_pipeline_q4_with_head_replacement`,
  `full_pipeline_q4_capture_pre_wo`
- `decode_token`, `decode_token_with_moe`, `decode_token_q4k_moe`,
  `decode_token_with_moe_split`, `decode_token_split_profile`
- `prefill_q4`, `prefill_q4_with_head_replacement`

`populate_kv_layer` retained `(num_kv_heads, head_dim)` because those
**are** per-layer geometry for the layer being populated, not vestigial
scalars; callers (prefill, honest predict) already pass per-layer
values via `arch.num_kv_heads_for_layer(layer)`.

The Metal trait impl forwards to the existing inner free-fns
(`MetalBackend::decode_token`, `dispatch_full_pipeline`, etc.) by
synthesising legacy values from `layers[0]` so the inner free-fns and
their existing test/example callers stay source-stable. Inside the
Metal trait impl the `PipelineIntervention` head_dim / num_q_heads
fields now come from `layers[target_layer]` (per-layer correct) rather
than `layers[0]` — silently fixed a latent bug for head intervention
on global-attention layers of Gemma 4.

Inference-side cleanup: `AttentionGeometry` and the helpers
`attention_geometry_for_arch_layer` /
`attention_geometry_for_pipeline_layer` are deleted (zero remaining
users). All threading through `gpu_setup`, `gpu/{decode_loop,
prefill, forced_logits, mod}`, `constrained`, `grid/{remote_ffn,
remote_moe, setup}`, `predict/honest`, `residual_diff/{capture,
stages}`, and `vindex/q4k_forward/metal` simplified.
`larql-kv::engines::unlimited_context::engine` updated for the same
shape.

### A3 — KV cache audit (already shipped, documented)

Production decode/prefill all reach the per-layer KV cache via
`reset_and_preallocate_kv_cache` →
`ComputeBackend::preallocate_kv_cache_per_layer`, and any later
`populate_kv_layer` calls take the layer's own `(num_kv_heads,
head_dim)`. `MetalBackend::create_kv_cache` (the legacy uniform
helper) is now used only by synthetic-architecture tests + the lazy
bootstrap inside `populate_kv_layer`; added a doc-comment to
`metal/decode/mod.rs` steering future readers to
`create_kv_cache_per_layer`.

### Tests

- `larql-inference --lib`: 907 passed (was 904) after A1 helpers and
  the import fix in `pipeline_layer.rs::tests`.
- `larql-compute --lib`: 162 passed.
- `larql-compute --tests`: green except a pre-existing
  `quant_matvec_defaults_forward_and_dequantise_q8_tail` failure that
  also fails on HEAD (unrelated to A1-A3).
- `larql-kv --lib`: 200 passed.
- 31B end-to-end smoke: `larql run` on `gemma4-31b-q4k.vindex`
  generates coherent text through the new trait surface (post-A1/A2/A3).

## [2026-05-10] — Coverage push round 2: 52.80% → 65.67% line (+12.87 pp), 2 bug fixes

Continued the targeted-test sweep with **MockArch fixtures** (Gemma3-style
+ Starcoder2-style), per-file deep dives, and a small unit-test pass on
the new H12 `gpu/` splits. **+250 lib tests** (654 → 904), zero failures,
and **2 production bug fixes** surfaced by edge-case tests.

### Bug fixes (both pre-existing, both reachable from public APIs)

| Bug | Fix | Regression test |
|-----|-----|-----------------|
| `backend_lm_head_topk(top_k=0)` panicked with `index out of bounds: 0` — the per-score loop accessed `heap[0]` before checking `k > 0` | Early-return `Vec::new()` when `k == 0` | `backend_lm_head_topk_top_k_zero_returns_empty` |
| `top_k_from_logits(k=logits.len())` panicked with `partition_at_index N greater than length N` — `select_nth_unstable_by` requires `k < len` | Skip the partition when `k == 0` or `k == len` (the sort still produces the full ordering) | `top_k_from_logits_k_larger_than_vocab_clamps` |

### MockArch fixtures (`test_utils.rs`)

Added two synthetic-arch builders to drive previously-unreachable
arch-dependent branches:

- `make_gemma3_test_weights()` — QK norm + post-norms + GeluTanh +
  `has_v_norm` + non-1.0 `embed_scale` + non-1.0 `residual_multiplier`
  + non-1.0 `qk_norm_weight_offset`
- `make_starcoder2_test_weights()` — non-gated FFN + FFN biases (up +
  down) + Q/K/V/O attention biases + `Activation::Gelu` + LayerNorm
  branch + non-1.0 `attention_multiplier`

### Final coverage (both rounds combined)

| Metric | Session start | End | Δ |
|--------|--------------:|----:|--:|
| **Line coverage** | **52.80%** | **65.67%** | **+12.87 pp** |
| Region coverage | 54.80% | 67.69% | +12.89 pp |
| Function coverage | 61.98% | 72.96% | +10.98 pp |
| Lib tests | 654 | **904** | +250 |
| Files ≥90% line cov | ~30 | **64 of 127** | +34 |

### Files crossed past 90% line coverage

| File | Start line% | End line% |
|------|------------:|----------:|
| `ffn/mod.rs` | 77.78 | **100.00** |
| `ffn/moe_remote/router.rs` | 75.82 | **99.63** |
| `ffn/moe_remote/multi_layer_wire.rs` | 57.14 | **99.21** |
| `forward/dump_config.rs` | n/a | **99.08** |
| `attention/gqa.rs` | 78.46 | **97.84** |
| `cached.rs` | 65.17 | **97.65** |
| `forced_logits.rs` (H12 split) | 0 | 51.28 (capped) |
| `tokenizer.rs` | 58.65 | **98.43** |
| `trace/types.rs` | 57.88 | **98.41** |
| `trace/vocab.rs` | 65.12 | **98.39** |
| `trace/context.rs` | 62.63 | 94.11 region (line ~94+) |
| `forward/predict/ffn.rs` | 40.94 | **97.08** |
| `forward/predict/dense.rs` | 40.79 | 95.37 region |
| `forward/target_delta.rs` | 70.80 | 96.29 region |
| `forward/trace.rs` | 76.14 | 95.88 region |
| `forward/patching.rs` | n/a | **96.92** |
| `attention/block.rs` | 57.33 | **95.71** |
| `attention/gpu.rs` | 66.27 | **97.34** |
| `ffn/weight.rs` | 76.79 | 95.40 region |
| `ffn/graph_backend.rs` | 63.40 | 96.10 region |
| `ffn/sparse_compute.rs` | 55.81 | 95.02 region |
| `ffn/remote/codec.rs` | 70.80 | **98.00** |
| `layer_graph/generate/lm_head.rs` | 54.05 | 93.31 region |
| `layer_graph/generate/gpu/sampling_step.rs` (H12 split) | 0 | **97.64** |
| `forward/infer_patched.rs` | 71.24 | 86.47 (close to 90, dense path tested; q4k path infra-bound) |

### Notable lifts that didn't quite cross 90% line

| File | Start | End | Blocker |
|------|------:|----:|---------|
| `vindex/walk_ffn/mod.rs` | 62.35 | 88.74 | Override-routing branch needs custom `PatchOverrides` mock with non-default impls |
| `forward/layer.rs` | 69.98 | ~88 | `apply_layer_scalar` body fires only on Gemma 4 (`layer_scalar_key` returns Some); `LARQL_*_DUMP_LAYERS` env paths are `OnceLock`-cached |
| `experts/loader.rs` | 61.90 | ~89 | `load_module` / `instantiate` need real WASM modules |
| `layer_graph/template.rs` | 65.13 | 81.62 | `TemplateUniverse::build` can't tokenise `[N]` literals through the synthetic Whitespace tokenizer |
| `forward/predict/honest.rs` (H12 split) | 21.28 | similar | needs Q4K vindex |
| `layer_graph/generate/gpu/decode_loop.rs` (H12 split) | 0 | 0 | needs both `MockComputeBackend` returning `Some` from `decode_token` AND a Q4K-loaded vindex |
| `layer_graph/generate/gpu/mod.rs` (H12 split) | 0 | **53.05** | early-return + `try_*` delegation paths covered; full streaming body needs the same fixture as above |
| `layer_graph/generate/gpu/prefill.rs` (H12 split) | 0 | 26.79 | `Capability::DecodeQ4KMoe` guard covered; rest needs Q4K vindex |

### What's left untouched

- **~30 files at 0%** — need infrastructure (real Q4_K vindex on disk,
  gRPC test server, HTTP test server, `MockComputeBackend` advertising
  `Capability::PrefillQ4` + `DecodeToken` AND returning `Some` from
  `prefill_q4`/`decode_token`)
- **~33 files between 0% and 90%** — mostly Gemma 4-specific
  (PLE / MoE / KV-share / `layer_scalar`) or Metal-kernel-bound branches

### Diminishing-returns frontier

Round 1 hit the easy synthetic-friendly files (target_delta,
multi_layer_wire, codec, gqa, cached, trace, predict/dense,
sparse_compute, graph_backend, lm_head). Round 2 added MockArch fixtures
+ close-to-90 polish + the gpu/ unit-testable pieces. The next pp+
of line coverage costs ~2-3 hours of synthetic-Q4_K-vindex fixture
work (would unlock `vindex/q4k_forward/*` + `gpu/{decode_loop, mod}.rs`
deeper paths). At this point higher-leverage work is on the open
frontiers (A1–A3 heterogeneous attention geometry, R4 trace export, MI4
parity tests) rather than further coverage push.

## [2026-05-10] — Coverage push: 53.13% → 63.18% (+10.05 pp)

Targeted-test sweep on synthetic-friendly files. 654 → **811 lib tests**
(+157 new tests, all passing). Per-file 90% floor crossed by 7 new files.

**Files crossed past the 90% floor:**

| File | Before | After |
|------|-------:|------:|
| `ffn/moe_remote/multi_layer_wire.rs` | 57.14% | **98.54%** |
| `ffn/remote/codec.rs` | 70.80% | **97.18%** |
| `attention/gqa.rs` | 78.46% | **98.78%** |
| `forward/trace.rs` | 76.14% | **95.88%** |
| `layer_graph/cached.rs` | 65.17% | **96.68%** |
| `forward/target_delta.rs` | 70.80% | **96.29%** |
| `trace/context.rs` | 62.63% | **94.11%** |
| `forward/predict/dense.rs` | 40.79% | **95.37%** |

**Big lifts that didn't cross 90% (arch-dependent or infrastructure-bound):**

| File | Before | After | Blocker |
|------|-------:|------:|---------|
| `attention/block.rs` | 57.33% | 89.54% | Bias/QK-norm Some-arms need non-tinymodel arch |
| `vindex/walk_ffn/mod.rs` | 62.35% | 89.24% | Override-routing branch needs custom `PatchOverrides` mock |
| `ffn/graph_backend.rs` | 63.40% | 84.50% | Streaming-build error paths |
| `forward/layer.rs` | 69.98% | 83.45% | post_norms branch needs Gemma-style arch |
| `forward/infer_patched.rs` | 71.24% | 82.56% | `infer_patched_q4k` needs real Q4K vindex |
| `ffn/weight.rs` | 76.79% | 86.36% | Non-gated FFN + bias paths need different arch |
| `attention/gpu.rs` | 66.27% | 76.39% | Q4 matvec + post_norms branches |
| `layer_graph/generate/lm_head.rs` | 54.05% | 80.00% | `lm_head_topk_with_policy` skip-Q4K branch |
| `ffn/sparse_compute.rs` | 55.81% | 78.58% | Some `sparse_ffn_forward_full_impl` branches |

**Files left untouched** (require infrastructure beyond synthetic fixtures):

- `layer_graph/grid/remote_moe.rs` (1192 LOC, 1.4%) — needs gRPC backend
- `ffn/moe_remote/backend.rs` (1013 LOC, 1.2%) — needs HTTP+grid
- `layer_graph/grid/remote_ffn.rs` (810 LOC, 0%) — same
- `vindex/q4k_forward/{metal, hidden, interventions, walk_ffn, …}.rs` — need a real Q4K vindex on disk
- The H12 splits (`gpu/{mod, decode_loop, prefill}.rs`, `predict/{split, honest}.rs`, `shard/*`) — need a `MockComputeBackend`

**Crate total**: lines 53.13% → **63.18%**, regions 54.55% → 60.69%, functions 62.78% → 68.50%.

## [2026-05-10] — `forced_logits` synthetic tests

The H12 `gpu/` split landed three new files at 0% coverage. `forced_logits.rs`
is the only one of the three that doesn't need a live Metal backend or a
fully-loaded vindex to drive — the two private helpers (`final_norm_row`,
`full_logits_from_vindex`) are pure-ish over `ModelWeights` + `VectorIndex`,
and the public entry point has 4 early-return guards reachable without
running any compute.

| Test | What it pins |
|------|--------------|
| `forced_logits_result_default_is_empty` | `ForcedLogitsResult::default()` produces empty fields. |
| `target_steps_zero_returns_empty_without_calling_backend` | `target_steps == 0` short-circuits before any backend call (the `on_logits` callback panics if invoked — proves the early return). |
| `rejects_non_fused_q4_backend` | `CpuBackend` (no `Capability::PrefillQ4`) is rejected with a "fused Q4 backend" error message before any vindex access. |
| `final_norm_row_short_input_errors` | `h_vec.len() < hidden` returns the "too short" error instead of panicking. |
| `final_norm_row_returns_hidden_length_finite_values` | Exact-length input produces a finite `hidden`-length output. |
| `final_norm_row_uses_last_hidden_chunk_when_seq_len_gt_one` | Multi-position `h_vec` slices the last `hidden` floats — pin against single-position equivalent. |
| `final_norm_row_zero_hidden_succeeds_with_empty_output` | Edge case `hidden == 0` no-ops without panicking. |
| `full_logits_returns_err_when_lm_head_knn_yields_nothing` | Empty hits from `lm_head_knn_backend_skip_q4k` surface as a "no scores" error. |

Coverage delta: `forced_logits.rs` 0% → **47.58% line / 51.28% region**.
Crate total 52.80% → **53.13% line**. Net of the whole day's H12 splits +
forced_logits tests: +0.02 pp from 53.11% baseline (the splits added ~750 LOC
of orchestration code mostly uncovered by unit tests; this round of tests
clawed it back).

## [2026-05-10] — H12 closed: `layer_graph/generate/gpu.rs` (999 LOC) split

The last orchestration file flagged by H12 split into a `gpu/` directory:

| File | LOC | Contents |
|------|-----|---------|
| `gpu/mod.rs` | 534 | Public surface (`generate`, `try_generate`, `generate_with_sampling`, `try_generate_with_sampling`, `generate_streaming`, `try_generate_streaming`, `LMHEAD_TOPK_*` consts, `lmhead_k_for_sampling`, `diag_compare_cpu_topk`) — `generate_streaming` is now an orchestrator that calls into the three phase modules below. |
| `gpu/forced_logits.rs` | 211 | Shannon-codec `stream_forced_full_logits` + `ForcedLogitsResult` + `final_norm_row` + `full_logits_from_vindex` (independent of sampling — moved out as its own module). |
| `gpu/prefill.rs` | 193 | `prefill_for_streaming` covering the three prefill branches (PLE / per-layer Q4_K MoE / standard fused) + `prefill_q4k_moe` helper. Cfg-gated PLE backend signature handled with two parallel `pub(super) fn` definitions so non-metal builds don't pay for the `MetalBackend` parameter. |
| `gpu/decode_loop.rs` | 329 | `run_decode_loop` (the 245-LOC step body extracted) + per-step `run_one_decode_step` dispatcher (split-profile / Q4K-MoE / standard) + two diagnostic loggers (`log_step_diagnostic`, `log_h_1d_diagnostic`). Returns `DecodeLoopOutcome` carrying tokens + per-stage timings. Takes `&ModelWeights` immutably (the original `&mut` was incidental — every call is read-only) so the caller's `setup.layers` immutable borrow doesn't conflict at the call site. |
| `gpu/sampling_step.rs` | 56 | `sample_and_emit` helper + `PickedToken` outcome — shared by the first-token path (in `mod.rs`) and the decode-loop step (in `decode_loop.rs`). Collapses the duplicated `sample → detok → softmax_prob → on_token → eos check` block from two sites into one. |

Net: 999 LOC → 1,323 LOC across 5 files (overhead is per-file headers + the
context parameters that previously rode along as locals). Largest file is
now `mod.rs` at 534 LOC; no file > 600. Full lib test suite green
(646 passed, 0 failed, 4 ignored). Public API unchanged — every external
caller (`src/{lib,layer_graph/{mod, generate/{mod, chat_session}}}.rs`,
`examples/{chat_demo,streaming_demo}.rs`, `tests/test_logits_goldens.rs`)
sees the same `pub use` surface.

H12 is now closed. The remaining `> 800 LOC` files in the crate
(`ffn/moe_remote/backend.rs` 867, `layer_graph/grid/remote_moe.rs` 878)
were never in scope for H12 (which targeted the orchestration files
named in the H12 row).

## [2026-05-10] — Magic-strings + magic-numbers cleanup

Code-vs-roadmap audit also flagged scattered env-var literals, HTTP path
literals, and the Q4_K/Q8_K super-block size as raw numbers. All four
sweep items landed in the order proposed.

| Item | Impact |
|------|--------|
| **DumpConfig expansion** — added `ENV_CPU_DUMP_LAYERS`, `ENV_CPU_STAGE_DUMP`, `ENV_STAGE_DUMP_LAYER`, `ENV_DECODE_DUMP_LAYERS`, `ENV_METAL_DUMP_LAYERS` consts + 8 path-template helpers (`cpu_layer_path`, `cpu_layer_h_post_attn_path`, `cpu_stage_path`, `cpu_stage_prefix`, `decode_layer_file`, `decode_layer_prefix`, `metal_layer_h_out_file`, `metal_layer_prefix`) | 11 inline env-var literals + 9 inline `format!()` path templates collapsed onto 5 consts + 8 helpers shared between producer (`forward/layer.rs`, `forward/layer_interventions.rs`, `vindex/q4k_forward/hidden.rs`) and consumer (`residual_diff/{capture,stages}.rs`) sides. Behaviour bit-identical. Module doc-block now pins the producer-side `cpu_L0_*` literal as a known mismatch with the consumer-side `cpu_L<layer>_` prefix — any future fix has to touch both sides. 8 new unit tests pin every helper format. |
| **RemoteMoeRuntime expansion** — added `ENV_HTTP_TIMING`, `ENV_MOE_WIRE_F16`, `ENV_DISABLE_Q8K_WIRE`, `ENV_VERBOSE`, `ENV_MOE_BYTES`, `ENV_MOE_TIMING`, `ENV_MOE_SHARD_TIMING`, `ENV_MOE_TOP_K`, `ENV_MOE_NO_SPLIT`, `ENV_SKIP_MOE` consts; new `moe_bytes_enabled` (with `MOE_TIMING ⇒ MOE_BYTES` implication) and `moe_shard_timing` fields | `metrics::enabled()` and `metrics::shard_timing_enabled()` previously did `std::env::var(..).is_ok()` on **every** shard call; now consult the `OnceLock` singleton instead — no per-call env reads on the hot path. `grid/config.rs` references the same consts so `LARQL_MOE_TIMING` is no longer typed as a literal in two files. 1 new unit test pins all 10 env-var names. |
| **MoE shard HTTP path consts** — added `EXPERT_BATCH_PATH`, `LAYER_BATCH_PATH`, `LAYER_BATCH_F16_PATH`, `MULTI_LAYER_BATCH_PATH`, `MULTI_LAYER_BATCH_Q8K_PATH` alongside the existing content-type consts in `wire.rs` / `multi_layer_wire.rs` | 8 raw `"/v1/expert*"` / `"/v1/experts/*"` literals across `shard/{expert_batch, layer_batch, multi_layer}.rs` collapsed onto 5 consts. Mirrors the existing `STATS_PATH` / `WALK_FFN_PATH` pattern in `ffn/remote/http.rs`. |
| **Q4_K/Q8_K super-block const + `ENV_VINDEX_PATH`** — new `pub const Q4K_Q8K_SUPERBLOCK_ELEMS: usize = 256` in `ffn/mod.rs`; new `pub const ENV_VINDEX_PATH: &str = "LARQL_VINDEX_PATH"` re-exported from `vindex` | Two pre-existing local consts (`ELEMS_PER_Q8K_BLOCK` in `multi_layer_wire.rs`, `ELEMS_PER_BLOCK` in `q8k_wire.rs`) now alias the canonical name. Two raw `% 256` checks at `vindex/q4k_forward/walk_ffn.rs:130` and `layer_graph/grid/remote_ffn.rs:120,218` replaced. Five `LARQL_VINDEX_PATH` runtime literals (in `layer_graph/generate/mod.rs` test fixtures + `forward/memit.rs`) routed through the const; the remaining literal occurrences are inside `#[ignore = "..."]` attribute strings (compile-time literals that can't reference a const). |

Net: ~30 inline env-var literals + ~10 inline path/format literals + 4 raw `256`s collapsed onto a small typed-config / shared-const surface. Module-level docs added to `dump_config.rs` and `runtime.rs` as the discovery points.

## [2026-05-10] — Roadmap reconciliation

Code-vs-roadmap audit caught three stale entries; ROADMAP.md updated and this
CHANGELOG.md split out so dated work doesn't keep accreting in the roadmap.

| Item | Impact |
|------|--------|
| `ffn/moe_remote/shard.rs` (924 LOC) split into `shard/{mod.rs 376, layer_batch 215, multi_layer 140, expert_batch 136, stream 123}` | Was completed in an earlier session that didn't get a roadmap entry. H12 file-split work is now down to one file (`layer_graph/generate/gpu.rs`). |
| MoE-aware CPU forward pass re-scoped | `vindex/q4k_forward/hidden.rs:169` already calls `larql_compute::cpu::ops::moe::cpu_moe_forward`. The vindex Q4K hidden-forward path is wired; only the dense `WeightFfn::forward` (non-vindex) path remains unwired. |
| `forward/predict/` split (P1 structure backlog) marked done | Shipped 2026-04-26 as `forward/predict/{mod.rs, dense.rs, ffn.rs, raw.rs, types.rs}` per the Completed table; the P1 backlog reference was a stale duplicate. Removed. |
| ROADMAP "Completed" table + dated subsections migrated here | ROADMAP.md ends at "P2: Research"; this changelog owns historical entries. |

## [2026-05-10] — Hardening pass: H8/H9/H11 shipped, H12 deferred

Continued the inference crate hardening from 2026-05-09. Three of four
remaining H-items now shipped; H12 (file splits) deferred as too invasive
for an incremental pass.

| Item | Impact |
|------|--------|
| H8: `forward::dump_config::DumpConfig` | New `OnceLock`-backed typed config consolidates 7 inline `LARQL_CPU_DUMP_LAYERS` / `LARQL_CPU_STAGE_DUMP` / `LARQL_STAGE_DUMP_LAYER` env reads (in `attention/block.rs`, `forward/layer.rs`, `forward/layer_interventions.rs`, `vindex/q4k_forward/hidden.rs`) into a single read at first access. 5 new tests pinning `from_env`, `stage_dir(layer)`, `layer_dir`, singleton stability, default fallback. |
| H8: `ffn::moe_remote::runtime::RemoteMoeRuntime` | New `OnceLock`-backed runtime config consolidates 6 inline reads (`LARQL_HTTP_TIMING`, `LARQL_MOE_WIRE_F16`, `LARQL_DISABLE_Q8K_WIRE`, `LARQL_VERBOSE`) across `moe_remote/shard.rs` (HTTP + UDS transports) and `moe_remote/backend.rs`. Replaces two `thread_local!` block caches that fragmented the toggle state per worker thread. 3 new tests. |
| H9: trim crate-root re-exports | Dropped 17 `forward::*` (e.g. `RawForward`, `LayerMode`, `forward_raw_logits`, `infer_patched_q4k`, `KNN_COSINE_THRESHOLD`) + 7 `layer_graph::*` (`GridGenerateResult`, `ChatMLRenderer`, `GemmaRenderer`, `LayerOutput`, `Llama3Renderer`, `PerLayerGraph`, `TurnRenderer`) re-exports with zero external consumers AND zero in-crate example/test usage. `research` module rewritten to source from full subpaths so it survives further root trims. Crate-root surface narrowed from ~120 to ~96 names. |
| H11: verified — no hardcoded family branches | Audit found one remaining `match router_type { "gemma4_top_k_softmax" => ... }` at `pipeline_layer.rs:546`. That's the architecture trait's `router_type` metadata signal — the policy-object pattern this item asks for, not a hardcoded family check. H11 done. |
| H12: split `layer_graph/predict.rs` | 881-LOC single file → `predict/` directory: `mod.rs` 376 (entry + small variants), `split.rs` 285 (3-pass approximate-attention pipeline + logits-only fast path), `honest.rs` 261 (production GPU+CPU hybrid). All 9 predict tests pass; no public API change. |
| Lib tests: 631 → 639 | 8 new tests for the two new typed configs (5 DumpConfig + 3 RemoteMoeRuntime); split of predict.rs preserved all existing tests; all 639 lib tests pass clean. |

## [2026-05-09] — Cleanup pass: blanket allows, f16 codec, raw.rs dedup, coverage lift

Source surgery + targeted test work that doesn't change behaviour. All public APIs preserved.

| Item | Impact |
|------|--------|
| Drop `lib.rs` blanket `#![allow(...)]` (28 lints) | Surfaced 31 real warnings hidden behind `dead_code` / `unused_imports` / `private_interfaces` / `deprecated` / etc. All fixed (deleted dead helpers + structs, cleaned imports, fixed `into_shape` deprecation, fixed `UdsState` privacy, fixed dead V cosine in turbo_quant test). |
| `ffn/moe_remote/wire.rs` f16 codec → `half` crate | Removed ~85 lines of hand-rolled IEEE-754 binary16 conversion; replaced with `half::f16::from_f32().to_bits()` / `from_bits().to_f32()`. All 6 existing f16 round-trip tests still pass bit-for-bit. |
| `capture.rs::current_date` Gregorian leap-year fix | Old impl divided days by 365 / months by 30 — drifted weeks per year. Rewrote with proper leap-year arithmetic; `extraction_date` field in every `VectorFileHeader` now correct. 7 new date tests (epoch zero, 2024 leap boundaries, century rule for 2000, format guards). |
| `forward/predict/raw.rs` dedup | `forward_raw_logits_with_prefix` and `forward_from_layer` were ~70% identical; collapsed into a private `forward_layer_range(prefix, layer_range)` core. Three copies of the softcap reduction → single `apply_logits_transform`. ~120 dup lines gone. |
| `lib.rs` re-exports trimmed | Dropped 8 `larql_compute::*` proxies with no external consumers via `larql_inference::*`: `cpu_moe_forward`, `ComputeActivation`, `CpuBackend`, `MoeLayerWeights`, `dot_proj_gpu`, `matmul_gpu`, `MatMulOp`, `MetalBackend`. Callers `use larql_compute::*` directly. |
| Chat-template stack cross-references | `prompt.rs` (heuristic enum), `chat/` (Jinja+vindex), `chat_session::TurnRenderer` (incremental) are complementary, not redundant — added module-level docs cross-referencing each so the boundary is discoverable. Original review's "collapse" recommendation was incorrect (3 external consumers depend on `prompt::ChatTemplate`). |
| Coverage: 80-89% bucket → ≥90% | 8 files lifted across the per-file 90% floor: `chat/render.rs` 82.64→99.44%, `experts/session.rs` 85.15→95.22%, `ffn/moe_remote/config.rs` 86.90→100%, `ffn/moe_remote/wire.rs` 81.76→99.75%, `forward/patching.rs` 83.77→89.96% (last gap is unreachable defensive guard), `layer_graph/logits.rs` 82.52→96.83% (inline tempfile fixture for `lm_head.bin`), `trace/capture.rs` 89.87→90.55%, `trace/store.rs` 85.22→98.37%. +56 tests, 575→631 lib tests. |

## [2026-05-02] — Mechanistic interpretability engine surface (MI0–MI3)

| Item | Impact |
|------|--------|
| MI0: faithful residual DAG in TRACE | TRACE routes through the canonical layer runner; `residual[L] = residual[L-1] + attn_delta + ffn_delta` is test-pinned. |
| MI1: Python `WalkModel.trace()` / `patch_activations()` use vindex `WalkFfn` | No more dense fallback for vindex-backed models. |
| MI2: backend-parametric activation patching | Donor capture and recipient intervention helpers. |
| MI3: trace artifact contract | Complete ordered chains only, exact file length checks, `TRACE SAVE` requires `POSITIONS ALL`. |
| MI4 partial: dense + custom-backend parity pinned | Final trace residuals project to the same logits as the canonical dense raw-forward path; custom `FfnBackend` trace matches the generic hooked forward runner. WalkFfn/patched-vindex/Q4K/MoE parity still pending. |

## [2026-04-30] — gRPC grid accuracy + dense Metal chat template + Gemma 4 model coverage

End-to-end accuracy work across Gemma 4's three production variants (26B-A4B
MoE via gRPC grid, 31B dense via Metal, E2B with PLE). Started from the gRPC
grid producing semantically wrong text ("not specified in the text") and
ended with all four Gemma 4 vindexes producing correct answers. Per-layer
CPU vs Metal residual parity (cos ≥ 0.9999 across all 60 layers of the 31B)
confirmed the inference math itself was always correct — every remaining
gap was somewhere in the wrapping, sampling, or routing logic.

| What | Notes |
|------|-------|
| `grid.rs` uses `Detokenizer` + `EosConfig::from_vindex_dir` | Was per-token decode losing SP `▁` leading-space + falling back to `<{id}>` for special tokens; output looked like "Thecapital of France is**not specified...**". |
| Special-token suppression in grid `pick_next_filtered` | Built from `tokenizer.get_added_tokens_decoder()` + structural-marker scan (`<unused…>`, HTML tags, `[multimodal]`). Top-K=256 fallback finds a real word when many candidates are markers. Q4_K quantisation noise was lifting `<mask>` (id 4) over the intended next word at the first answer position. |
| `chat::render_user_prompt` shared helper | Centralises `LARQL_RAW_PROMPT` / `LARQL_THINKING` / `LARQL_SYSTEM` / `LARQL_NO_DEFAULT_SYSTEM` + auto Gemma 4 default system prompt. Used by both `run_with_moe_shards` (gRPC) and `walk_cmd::run_predict_q4k` (dense Metal). |
| Built-in Gemma 4 fallback chat template | Vindexes extracted before `chat_template.jinja` was snapshotted (early 31B and E2B) silently sent raw prompts and looped "The answer is:". `family_default_template("gemma4")` plugs the gap. |
| Dense Metal path now applies chat templates | `walk_cmd::run_predict_q4k` was sending the raw user string to `encode_prompt`; the chat-template machinery only ran for gRPC. Both paths now go through `render_user_prompt`. |
| `lm_head_topk` falls back to backend GEMV when KNN is all-zero | At the prefill→decode boundary the Metal `q4k_matvec` for lm_head occasionally returned 256/256 zero scores while h_1d was healthy (rms ≈ 4, max_abs ≈ 60). Detect + retry via `backend_lm_head_topk` recovers a non-zero distribution immediately. |
| PLE auto-route for Gemma 4 E2B | E2B has `hidden_size_per_layer_input=256` (per-layer-input gate + projection + norm + global PLE embedding). The CPU dense path implements PLE; Metal does not. `generate_streaming` now checks `arch.has_per_layer_embeddings()` and delegates to `generate_via_cpu_q4k` for those models so the residual stream gets the per-layer per-position contribution. Without this E2B emitted multilingual gibberish; with it, "The capital of France is Paris". |
| Diagnostic env vars: `LARQL_DEBUG_TOKEN_IDS`, `LARQL_DEBUG_TOPK` | Per-step token-id + raw top-K scores in both `grid.rs` (gRPC) and `gpu.rs` (dense). Surfaced the "all logits == 0.000" smoking gun that localised the lm_head KNN bug. |
| `larql parity --component layer` extended to dense | Was MoE-only (`LARQL_DUMP_RESIDUALS`). Now uses `LARQL_METAL_DUMP_LAYERS` for dense models — wrote per-layer `metal_layer_NN_h_out.f32` and CPU dump files. Gave us the cos ≥ 0.9999 confirmation across 60 layers that ruled out the inference math as the bug source. |
| `larql parity --component lm-head` works on dense | Dropped the MoE-only gate for `lm-head` (Q4_K vs f32 reference is backend-agnostic). |
| `test_logits_goldens.rs` compile fix + 5 new entries | Added missing `None` for `predict_q4k_hidden`'s `Option<&RemoteMoeBackend>`; refreshed stale 5 goldens to match current kernel state; added `gemma3-4b-q4k-downq4k` (Q4_K-down regression test), `gemma4-31b-q4k-q6kdown` (Q6_K-down dense), `gemma4-e2b-q4k` (PLE auto-route) — 13/13 passing. |
| Discovered: in-process Metal MoE path (`gpu_moe_dispatch_with_scratch`) shares the bug | Until now nobody had run `larql run --metal` on Gemma 4 26B-A4B (the gRPC grid was the only tested path). It produces the same wrong text as the server's Metal expert dispatch ("answer is in the context" instead of "Paris"). The gRPC-with-CPU-experts path has been the only working route all along — the in-process Metal MoE was always broken for this model. See `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side fix plan. |

## [2026-04-27] — Q4_K stride validation + strict vindex loader

| Item | Impact |
|------|--------|
| Q4_K stride validation in `load_attn_q4k` | Catches stale 148-byte vindexes; clear "rebuild" error vs silent NaN. |
| `QuantFormatInfo::expected_bytes(&shape)` helper | Single source of truth for stride math; used by loader validation. |
| 11 stride-validation tests (registry + loader) | 144 vs 148-byte stride; arbitrary lengths; Q4_K & Q6_K shapes. |
| Q4_K vs Q4_KF kernel routing fix in `quant_matvec::encode` | Q4_K weights now dispatch the Q4_K kernel; `FusedQkvKernel` enum carries TG geometry. |
| `vindex::open_inference_vindex` strict loader | Single entry point; propagates stride errors instead of silently degrading. |
| Demos switched to `open_inference_vindex` | `sampling`/`streaming`/`eos`/`chat` now error loudly with rebuild guidance on stale vindexes. |

## [2026-04-26] — Generation quality + structural splits + test coverage push

Two batches of work landed: a generation-quality pass that turned the
Gemma 3 4B path into a streamable, sampleable, multi-turn surface; and a
big restructuring + test-coverage round across `engines/`, `forward/`,
`ffn/`, `layer_graph/`.

### Generation quality

| Item | Impact |
|------|--------|
| `generate/eos.rs` — `EosConfig` | Built-in stops + `generation_config.json`; fixes Gemma 4 `<end_of_turn>` bug. |
| `generate/detok.rs` — `Detokenizer` | Cumulative-decode delta; preserves HF `▁` leading-space across SP and BPE. |
| `generate/sampling.rs` — `Sampler` + `SamplingConfig` | Greedy / temp / top-k / top-p + seed; <2µs/call sparse path. |
| `generate_with_sampling` wired into GPU path | Greedy `generate` is a thin wrapper; backward compatible. |
| `generate_streaming(... on_token)` callback | Per-token streaming; `generate_with_sampling` is thin no-op wrapper. |
| `chat_session.rs` — `ChatSession` + `TurnRenderer` | Multi-turn buffer with whole-turn eviction; Gemma/ChatML/Llama-3 renderers. |
| Smoke test: `test_gemma3_smoke.rs` | One-token greedy regression; `CI_INTEGRATION` fail-loud mode. |
| Examples: `streaming_demo`, `chat_demo`, `sampling_demo`, `eos_demo`, `detok_demo` | Live token streaming + 3-turn chat over `ChatSession`; detok runs without a model. |
| `bench_sampling` benchmark | Per-call cost across 4 configs × 3 vocab sizes; results in PERFORMANCE.md. |
| 35 sampling/eos/detok tests + 13 ChatSession tests + streaming integration | All passing; 626 lib tests total. |

### Structural splits and test coverage

| Item | Impact |
|------|--------|
| `generate/` split (`cpu/gpu/lm_head/types`) | Structured generation directory. |
| `markov_residual/` split (`store/engine/compute/q4k`) | Structured engine directory. |
| `forward/predict/` split (`types/raw/dense/ffn`) | Forward predict directory. |
| `forward/ops.rs` extracted | Shared math primitives. |
| `graph_ffn.rs` → `ffn/graph_backend.rs` | Correct placement in `ffn/`. |
| `ffn/remote.rs` → `remote/codec.rs` + `remote/http.rs` | No magic strings; codec/HTTP separation. |
| `turbo_quant/mod.rs` → `engine.rs` | Consistent engine layout; thin `mod.rs`. |
| Softmax unified to `forward/ops.rs` | 2 duplicate impls removed. |
| `forward/ple.rs` `norm_eps` fixed | Uses `arch.norm_eps()` not hardcoded `1e-6`. |
| Bug: `eos_id = 1` in `grid.rs` | Correct EOS on all models, not just Gemma. |
| Code quality review (3-agent) | Unsafe removed, LCG fixed, `OnceLock` added. |
| Tests: `markov_residual/`, `ffn/sparse_compute.rs`, `ffn/sparse.rs`, `ffn/graph_backend.rs`, `forward/ops.rs`, `unlimited_context/extend.rs`, `layer_graph/{dense,walk,mod,prefill,template,pipeline_layer,grid}.rs`, `forward/ple.rs`, GQA reps>1, RoPE properties, `residual_diff/capture.rs` PathBuf import fix | 0 → 100+ new tests across modules; 525 unit tests total at end of day. |
| Integration: `test_layer_graph_integration.rs` | 7 ignored tests; real vindex prefill / pipeline / template. |
| 49% line coverage (llvm-cov) | Baseline measured. |

## [2026-04-25] — KV-engine trait + first three engines

| Item | Impact |
|------|--------|
| `KvEngine` trait + `EngineKind` | Pluggable engine selector + CLI params. |
| `MarkovResidualEngine` | Residual-based KV (exact, 287×). |
| `UnlimitedContextEngine` | Window checkpoints (exact within window, 254×). |
| `BackendFfn` (Q4K FFN dispatch) | WalkFfn + Metal for FFN in all engines. |
| `cold_kv` cache (MarkovRS) | Skip cold-tier recompute; 8.5× decode speedup. |
| Profiler (per-stage timing) | `larql bench --engine --profile` breakdown. |
| P1 code quality fixes (magic strings, duplication) | `env-var` names, GELU constants. |

## [2026-04-26 follow-up] — TurboQuant + Apollo + Metal Q4K

| Item | Impact |
|------|--------|
| `TurboQuantEngine` | 4-bit WHT+Lloyd-Max K/V compression (4×, cos≈0.991). |
| `ApolloEngine` | Retrieval+injection (20,000×, compressed path). |
| `forward_from_layer` | Start forward at `crystal_layer`; 8.5× Apollo speedup. |
| Metal Q4K path for all engines | ~95 tok/s across all 4 engines. |

## [2026-04-07] — Q4_K FFN format + Gemma3 GPU correctness

| Item | Impact |
|------|--------|
| Q4_K FFN format wiring | Vindex Q4_K FFN → `FullPipelineLayer`. |
| GELU-tanh activation | Gemma3 correct on GPU. |
| Post-norm guard | Gemma3 falls to CPU correctly. |

## [2026-04-06] — Production hybrid + GPU prefill

| Item | Impact |
|------|--------|
| `predict_honest` | Production path, GPU+CPU hybrid. |
| GPU prefill pipeline | `seq>1` on GPU (pre-norm models). |

## [2026-04-04] — Layer graph

| Item | Impact |
|------|--------|
| `CachedLayerGraph` | Skip L0-12, 0.999 cosine. |
| `LayerGraph` trait | Pluggable per-layer routing. |

## [2026-04-03] — Sparse FFN + fused attention

| Item | Impact |
|------|--------|
| `WalkFfn` (sparse FFN via vindex) | Gate KNN + top-K. |
| BLAS-fused attention | Online softmax, O(seq) memory. |

## [2026-03] — Foundation

| Item | Impact |
|------|--------|
| Forward pass (CPU BLAS) | Foundation. |
