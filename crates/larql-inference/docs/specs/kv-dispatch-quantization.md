# KV-dispatch quantization routing

**Status:** Phase 1 + 1b shipped 2026-05-16. `StandardEngine` + `NoCache` Q4K via dispatch at production speed (27.6 tok/s on Gemma 3 4B Q4K, M3 Max). Phase 2 (other engines drop their bespoke Q4K paths) follows — see "Per-engine migration" below. Phase 3 (Q4K-native CpuBackend kernels) is a parallel CPU performance track; see [`q4k-decode-kernel.md`](../../../larql-compute/docs/q4k-decode-kernel.md).
**Audience:** LARQL contributors.
**Driver:** Engines today hard-code per-quantization paths (`prefill_q4k` overrides bypass `KvDispatch` entirely). The new trait should let backends route to Q4K kernels natively; engines should stay quantization-agnostic.

## Problem

`KvDispatch::attention_prefill(weights: &ModelWeights, tokens_embedded, layer, window)` and `attention_step` take f32 inputs and read f32 attention tensors from `weights.tensors`. Q4K-loaded `ModelWeights` doesn't have those tensors — the Q4K data lives in the sibling `VectorIndex` (`index.attn_q4k_layer_data`). The trait can't see the index, so:

- **Engines that need Q4K override `KvEngine::prefill_q4k` and bypass `KvDispatch` entirely** (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`).
- **Engines that don't override get a silent-broken Q4K path** (`StandardEngine`, `NoCache`, `Apollo`): default `prefill_q4k` falls through to `prefill`, which reads missing f32 attention tensors and errors with "Q4K engine prefill failed".

`larql_compute::QuantMatVec` (with `q4k_matvec`, `q6k_matvec`, etc.) is defined but **not consumed by engines for the per-step attention path** — every Q4K-aware engine builds its own `q4k_prefill_metal` / `q4k_decode_token` bypass.

## Decision

Push the `&VectorIndex` into `KvDispatch`'s attention intents:

```rust
fn attention_prefill(
    &self,
    weights: &ModelWeights,
    tokens_embedded: &Array2<f32>,
    layer: usize,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,   // NEW
) -> Option<(Array2<f32>, KvHandle)>;
```

Same for `attention_step`, `attention_step_windowed`, and their `AsyncComputeBackend` siblings.

Backends inspect `index`:
- `Some(idx)` + `backend.has_q4()` → Q4K native path (Metal shaders today; CPU Q4K kernels in a future phase).
- `Some(idx)` + no native support → call shared `ensure_attn_tensors_dequantised(weights, idx)` to populate f32 fallback (memory-heavy; the bridge until CPU Q4K kernels land).
- `None` → today's f32 path.

`ensure_attn_tensors_dequantised` moves from `larql-kv/src/engines/markov_residual/q4k.rs` to `larql-inference/src/vindex/dequant.rs` so the trait impls can call it without a circular dep. `larql-kv` re-exports for callers of the old path.

## Phase 1 (this session) — minimum viable

1. Move `ensure_attn_tensors_dequantised` to `larql-inference`.
2. Widen `KvDispatch::attention_prefill` + `attention_step` + `attention_step_windowed` to take `Option<&VectorIndex>`. Default impls ignore the new param.
3. Same widening for `AsyncComputeBackend::attention_*_async`.
4. `CpuBackend`'s impls: when `Some(idx)`, assert/lazily-ensure attention tensors are dequantised (calls the helper); otherwise unchanged.
5. `MetalBackend`'s impls: scaffold continues to delegate to CpuBackend at A3; future A4/A6 phases can route to Q4K kernels natively.
6. Helpers `kv_prefill_via_dispatch` / `kv_decode_step_via_dispatch` and async siblings take `Option<&VectorIndex>` and thread it to the trait calls.
7. `StandardEngine::prefill_q4k` / `decode_step_q4k` override: dequant via the helper, then delegate to the trait dispatch.
8. Update all call sites (test stubs, CLI bench, existing tests).
9. **Verification:** `larql bench --cpu --engine standard output/gemma3-4b-q4k-v2.vindex` produces tokens (currently fails). Plus the existing 1052+221 tests still pass. Plus the async parity holds on real Q4K model.

## Phase 1b (shipped 2026-05-16) — coarse fused intents on the trait

After Phase 1's per-layer `attention_*` widening, a separate diagnosis on Gemma 3 4B Q4K showed `StandardEngine` running at 0.4 tok/s via the per-layer dispatch path while `larql-cpu` (production path bypassing the trait) hit 24 tok/s — a 60× gap purely from the f32-dequant fallback vs the production Q4K kernels (`predict_q4k_decode_step_direct`).

The fix lives on the trait without leaking quantization into the engine: two **quantization-agnostic coarse intents** on `KvDispatch`:

```rust
fn coarse_prefill(
    &self,
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, KvHandle)>;

fn coarse_decode_step(
    &self,
    weights: &mut ModelWeights,
    token_id: u32,
    index: Option<&larql_vindex::VectorIndex>,
    handle: &mut KvHandle,
    abs_position: usize,
) -> Option<Array2<f32>>;
```

Backends inspect `index` (and `weights.tensors`) and dispatch internally to whatever native kernel matches: Q4K matvec today, Q6K / Q8 / FP4 / future formats slot in without changing the trait or the engine call sites.

`CpuBackend`'s impl routes to the production `predict_q4k_prefill` + `predict_q4k_decode_step_direct` pipeline via a new `CpuQ4kCacheHandle: KvHandleInner` wrapping `CpuKvCache`. Result: **27.6 tok/s through the dispatch trait** — slightly faster than `larql-cpu`'s legacy 24.0 tok/s, with no quantization knowledge anywhere in the engine layer.

Engines that want fast decode call `coarse_prefill` / `coarse_decode_step`; backends that don't support them return `None` and engines fall back to per-layer dispatch.

## Phase 2 — per-engine migration

Status as of 2026-05-16:

| Engine | Dispatch shape | Current tok/s on Gemma 3 4B Q4K (CPU, 8 threads) | What's blocking the fast path |
|---|---|---:|---|
| `StandardEngine` | ✅ coarse intent + per-layer fallback | **27.6** | Done. Matches/beats `larql-cpu`. |
| `NoCache` | per-layer dispatch + WalkFfn (debug fallback) | 0.4 | By design — O(N²) full re-forward per step. Not a candidate for the coarse path (no cache). |
| `MarkovResidual` | bespoke `prefill_q4k` → `rs_prefill_walk` (CPU) / `q4k_prefill_metal` (Metal) | 0.4 | RsStore (residuals, not K/V) is per-step recomputed. Per-layer `recompute_kv` does f32 matmul on dequantised weights — should call `larql_compute::QuantMatVec::q4k_matvec` directly on Q4K bytes (skip dequant). Migration: `recompute_kv` → Q4K-native via the backend's matvec primitive. |
| `UnlimitedContext` | bespoke `prefill_q4k` → `rs_extend_from_checkpoint_q4k` (CPU) / `q4k_prefill_metal` (Metal) | 0.4 | Same shape as MarkovResidual — per-step extension across window-checkpointed K/V uses f32 matmul on dequantised weights. Same fix: route per-layer Q/K/V projection through Q4K matvec. |
| `TurboQuant` | bespoke `prefill_q4k_cpu` / `decode_step_q4k_cpu` with WHT+Lloyd-Max codec on K/V | 0.4 | The codec is the engine's specialised contribution (compress K/V to ~12.7 GB on a 370K-token corpus). Per-step attention currently does f32 matmul on dequantised weights. Same fix as above for the projection matvecs; codec stays engine-side. |
| `Apollo` | uses `forward_from_layer` / `forward_raw_logits` directly (no FfnBackend indirection) | n/a (bench-only) | Apollo is a different shape — boundary-residual injection at a specific layer. The forward functions it calls assume f32 attention + FFN tensors. Migration would require either keeping the upfront-dequant path or rebuilding Apollo's forward to call backend matvec primitives directly. Lower priority because Apollo is bench-only. |

The common pattern across MarkovResidual / UnlimitedContext / TurboQuant: each has a **legitimate engine-side specialisation** (residual store, checkpoints, codec) that doesn't fit the coarse-prefill cached-K/V shape. They keep that specialisation. What changes is the **per-layer Q/K/V/O projection** inside their bespoke decode loops — switching from `dot_proj_gpu(h, dequantised_w)` (f32 matmul, 8× memory bandwidth) to `backend.quant_matvec(QuantFormat::Q4_K, w_q4k_bytes, h_q8, ...)` (Q4K matvec, native bandwidth).

Estimated per-engine effort: 2–5 days. Acceptance: tok/s within 5× of `StandardEngine` (the engines are designed to be slower than Standard due to recompute / re-prefill / codec overhead, but should be in the same order of magnitude).

## Phase 3 (parallel CPU track) — close the per-core kernel gap

Phase 1b gets `StandardEngine` to 27.6 tok/s vs llama.cpp's 41.4 (1.50× behind). The remaining gap is the **Q4K × Q8K matvec inner loop** itself — same NEON SDOT algorithm as llama.cpp, but our Rust intrinsics produce a looser instruction schedule than llama.cpp's hand-asm. See [`q4k-decode-kernel.md`](../../../larql-compute/docs/q4k-decode-kernel.md) for the full kernel-track spec.

Phase 3 is orthogonal to Phase 2: engines benefit automatically when the underlying kernel improves, no engine-side changes needed.

## Non-goals (this work)

- Replacing Metal's fused `decode_token` with per-layer Metal Q4K dispatch (that's A4+A6 in the async-compute-backend spec).
- Removing `prefill_q4k` from the `KvEngine` trait surface (deferred — backward compatibility worth more than the trait simplification today).
- Optimising the dequant-once memory cost (largely obsoleted by the coarse intent path; only the fallback path still uses bulk dequant).
