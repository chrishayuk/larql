# WalkFfn routing analysis — the dense-Q4K trap

**Date:** 2026-05-16
**Author:** captured during the kv-engine Q4K migration session
**Status:** Investigation needed. The kv engines have been routed *around* WalkFfn for Q4K models as a workaround; the WalkFfn-side fix is documented here for a follow-up session.

---

## TL;DR

`WalkFfn::forward_with_activation` for `gemma3-4b-q4k-v2.vindex` falls into the **`zero_features_dense`** branch (line 254 of `crates/larql-inference/src/vindex/walk_ffn/mod.rs`) for every layer. That branch ignores the vindex's Q4K interleaved FFN bytes and dispatches to dense `WeightFfn`, which does **f32 matmul on dequantised gate/up/down tensors** — ~70ms per layer × 34 layers = 2.4s per decode token. **100× slower than the production Q4K FFN** at ~25ms total for the same 34 layers.

This is the root cause of the 0.4 tok/s slowdown on `markov-rs` / `unlimited-context` / `turbo-quant` we measured pre-fix. The fix landed in the engines (route to `larql_inference::vindex::ffn_decode_step_native`) but **WalkFfn itself still falls into the trap for any other caller** (production-path edge cases, research/probe paths, hooks).

## Measured impact (Gemma 3 4B Q4K, M3 Max, 8 threads)

With `LARQL_INSTRUMENT_MARKOV=1` instrumentation on `rs_decode_step_walk` (the markov-rs hot loop):

**Before — WalkFfn → zero_features_dense → WeightFfn (f32):**
```
[markov-rs/decode] s_hot=8  recompute_kv=54.30ms  attention=9.90ms  ffn=2412.50ms  total=2476.70ms
```

**After — engine routes to `ffn_decode_step_native` (production Q4K):**
```
[markov-rs/decode] s_hot=8  recompute_kv=48.02ms  attention=9.05ms  ffn=25.56ms  total=82.62ms
```

FFN dropped from **2412ms → 26ms** (92× faster). Per-engine tok/s gains:

| Engine | Before (tok/s) | After (tok/s) | Speedup |
|---|---:|---:|---:|
| `markov-rs` | 0.4 | 7.3 | 18× |
| `unlimited-context:window=256` | 0.4 | 26.2 | 65× (matches standard) |
| `turbo-quant:bits=4` | 0.4 | 17.7 | 44× |

## The routing ladder today

From `crates/larql-inference/src/vindex/walk_ffn/mod.rs:252-394`. Order is priority — first match wins.

### Pre-ladder gates

| Order | Branch | When it fires | What it does |
|---|---|---|---|
| 0 | `zero_features_dense` (line 254) | `num_features(layer) == 0` | Dispatch to `WeightFfn` — **dense f32 matmul, ignores all vindex compact storage** |
| 0a | `l1_cache_hit` (line 304) | Single-position residual key hits the L1 cache | Return cached output |
| 0b | sparse-with-overrides (line 271) | `index.has_overrides_at(layer)` true | Route to `walk_ffn_sparse` (intercepts overrides correctly) |

### Routing ladder proper

| Order | Branch | When it fires | What it does |
|---|---|---|---|
| 2 | `walk_ffn_sparse` (line 315) | `config.is_sparse(layer)` — user opted in | Sparse walk over top-K gate features |
| 3 | FP4/FP8 sparse (line 327) | `index.has_fp4_storage()` | Sparse walk with FP4 storage |
| 4 | `walk_ffn_q4_interleaved` (line 334) | `has_interleaved_q4() && backend.has_q4()` | Q4_0 (not Q4_K) interleaved + Metal Q4 |
| 5 | `walk_ffn_interleaved` (line 341) | `has_interleaved()` | f32 interleaved |
| 6 | `walk_ffn_full_mmap` (line 348) | `has_full_mmap_ffn()` | Separate gate/up/down mmap files |
| 7 | `walk_ffn_q4k_dequant` (line 355) | `has_interleaved_q4k()` | **Q4K interleaved — DEQUANT to f32 then matmul** |
| 8 | `walk_ffn_exact` (line 362) | `has_down_features()` | Down from mmap, gate/up from safetensors |
| 9 | sparse weights fallback (line 369) | Top-K against safetensors weights | Last resort |

### The two Q4K branches

WalkFfn has two paths for Q4K vindexes:

1. **Branch 7 (`walk_ffn_q4k_dequant`):** Used only if `num_features(layer) > 0`. **Still dequantises to f32 internally** per `interleaved_q4k.rs:13` — the name says "q4k_dequant" because it reads the Q4K bytes but lifts them to f32 before matmul. Some win on memory (Q4K on disk, dequant on demand) but the hot path is still f32 sgemv.

2. **`larql_inference::vindex::ffn_decode_step_native`** (the one we just exposed): The production hot path. Direct Q4K matvec via `q4k_matvec_into` (rayon-parallel) using interleaved Q4K bytes. **No dequant, no f32 staging.** This is what `predict_q4k_decode_step_direct` uses internally — the 24 tok/s path.

WalkFfn doesn't currently route to `ffn_decode_step_native`. The engines now route around WalkFfn for Q4K vindexes as a workaround.

## Why the trap fires for Gemma 3 4B Q4K

The Gemma 3 4B Q4K vindex was built with `extract_level = Browse` (no sparse FFN features extracted). Its FFN payload is the **interleaved Q4K** files (gate / up / down as Q4K blocks), not sparse `down_features.bin`. So:

- `num_features(layer)` → **0** (no sparse features) → triggers `zero_features_dense` immediately.
- `has_interleaved_q4k()` → true (Q4K bytes present), but **the ladder never reaches branch 7** because the `num_features == 0` early exit at line 254 fires first.
- Engines that use WalkFfn get dispatched to dense `WeightFfn` → f32 matmul on `weights.tensors` (after `ensure_attn_tensors_dequantised`-style upfront dequant) → ~70ms per layer.

## What "fix" actually means

Three options, in increasing scope:

### Option A — engines route around WalkFfn (what we just did)

Engines that have a `&VectorIndex` available call `ffn_decode_step_native` first, fall through to `run_ffn(walk_ffn, ...)` only if the native helper returns `None`. **Already shipped** in `markov_residual/q4k.rs`, `unlimited_context/extend.rs`, `turbo_quant/engine.rs`.

Pros: contained to engines, no WalkFfn changes, immediate win.
Cons: WalkFfn itself still wrong for any non-engine caller. Probe paths, capture hooks, ad-hoc forward passes via `WalkFfn::forward(...)` still fall into the trap.

### Option B — reorder WalkFfn's gate

Move the `num_features == 0` early exit *after* the Q4K interleaved check. So the new ladder order:

1. If `has_interleaved_q4k() && backend.has_q4()` and the layer is direct-matvec-eligible → call `ffn_decode_step_native`.
2. Then check `num_features == 0` for the dense fallback.

Pros: fixes WalkFfn itself, every caller benefits.
Cons: changes WalkFfn's routing behaviour. Bit-parity tests need to hold against the f32 path under the same arch. The `ffn_decode_step_native` helper currently lives in `vindex::q4k_forward::cached` — using it from `walk_ffn/mod.rs` means an inter-module call inside the same crate. Manageable but needs structuring (probably move the helper or expose via a smaller surface).

### Option C — collapse the routing ladder

WalkFfn has 10 branches today. Most exist for historical vindex storage formats (Q4_0 interleaved, f32 interleaved, full mmap, down_features, etc.) — each path is bespoke code. A clean rewrite:

1. Inspect the vindex's storage shape once at WalkFfn construction.
2. Pick *one* execution strategy: native Q4K matvec, sparse walk, dense weights, etc.
3. Avoid the per-call ladder traversal entirely.

Pros: removes years of bolted-on routing, makes new storage formats trivial to slot in (one new strategy struct, no ladder edit).
Cons: substantial refactor; every test against WalkFfn pre-strategy-rewrite needs validation.

## Open questions for the dedicated analysis session

1. **Why is the `zero_features_dense` early exit gated *before* the ladder?** Historical reason? Performance optimization for sparse-only vindexes? Documented or implicit?
2. **What other vindex shapes hit `zero_features_dense`?** Llama 2 7B Q4K? Mistral 7B? TinyStories f32? Worth knowing how broad the trap is.
3. **Is `walk_ffn_q4k_dequant` ever the right path?** It exists, requires `num_features > 0` AND `has_interleaved_q4k()` — i.e. vindex has BOTH sparse features AND Q4K bytes. Is that combo ever built?
4. **`ffn_decode_step_native` lives in `q4k_forward/cached.rs` — should it move to `walk_ffn/`?** Architectural question about where the "native quantised FFN forward" primitive belongs. It's used by both `predict_q4k_decode_step_direct` (production path) and now by the engines. Putting it in `walk_ffn/` (as Option B's reorder) makes WalkFfn the canonical entry point. Putting it where it is (in `q4k_forward`) keeps it next to its current sibling helpers.
5. **Does `walk_ffn_q4k_dequant` (branch 7) actually dequant per call?** If so, it's the same trap as `zero_features_dense` just by a different route. Worth measuring.
6. **L1 cache effectiveness.** The cache fires for `seq_len == 1` and an exact-match residual hash. How often does it hit in real generation? Telemetry would tell.

## Instrumentation available for the session

The `LARQL_INSTRUMENT_MARKOV=1` env var, set during `cargo run` of `larql bench`, enables per-stage timing in `rs_decode_step_walk`:

```
[markov-rs/decode] s_hot=8 recompute_kv=48.02ms concat=0.00ms attention=9.05ms ffn=25.56ms total=82.62ms
                   (attn_helper hits/miss=34/0)
```

The instrumentation lives in `crates/larql-kv/src/engines/markov_residual/q4k.rs::rs_decode_step_walk`. Same pattern can be added to `walk_ffn/mod.rs` to log which branch fires per layer per call — useful for the routing analysis.

`WalkFfn` already has `trace_path(layer, label)` (line 255 et al.) which records the path taken. Setting `record_trace = true` on the WalkFfn would let you dump the per-layer branch names without writing new code. See `crate::vindex::walk_ffn::WalkTrace`.

## File pointers

- Routing ladder: `crates/larql-inference/src/vindex/walk_ffn/mod.rs:252-394`
- The trap (zero_features_dense): `crates/larql-inference/src/vindex/walk_ffn/mod.rs:253-260`
- Q4K dequant branch: `crates/larql-inference/src/vindex/walk_ffn/interleaved_q4k.rs`
- Production fast-path FFN: `crates/larql-inference/src/vindex/q4k_forward/cached.rs::run_ffn_decode_step_q4k_direct` (publicly exposed as `vindex::ffn_decode_step_native`)
- Engine workaround examples:
  - `crates/larql-kv/src/engines/markov_residual/q4k.rs:189-210` (decode_step)
  - `crates/larql-kv/src/engines/unlimited_context/extend.rs:175-195` (extend)
  - `crates/larql-kv/src/engines/turbo_quant/engine.rs:380-400` (prefill + decode)
- `WeightFfn::forward_with_activation`: `crates/larql-inference/src/ffn/weight.rs`
- `q4k_matvec_into` (the actual kernel): `crates/larql-compute/src/cpu/ops/q4_common.rs:535`
