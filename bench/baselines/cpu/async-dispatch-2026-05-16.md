# Async dispatch — performance + accuracy proof

**Date:** 2026-05-16
**Hardware:** Apple M3 Max (12 perf cores)
**Scope:** `AsyncComputeBackend` A1–A3 + A5 (StandardEngine slice) + Q4K-through-dispatch refactor (Phase 1). Compares the new async dispatch path against the equivalent sync `KvDispatch` path on `CpuBackend`, on both synthetic weights and the real Gemma 3 4B Q4K model.

## TL;DR

- **Accuracy:** 64-step bit-parity tests pass on synthetic weights; **16-token bit-parity also passes on the real Gemma 3 4B Q4K model** (all 16 emitted tokens identical between sync and async).
- **Performance — synthetic:** async dispatch on CPU is statistically indistinguishable from sync (overhead within criterion noise, <2pp).
- **Performance — real model:** 44.9 tok/s sync vs 45.4 tok/s async on Gemma 3 4B Q4K (Δ = -1.1%, within noise).
- **Q4K-through-dispatch refactor** lands `StandardEngine` on Q4K (was broken with "Q4K engine prefill failed"). See "Q4K dispatch fix" section below.

## What was measured

### Criterion microbenchmarks (synthetic 2-layer model)

`cargo bench -p larql-kv --bench engine_decode`

**Full generate loop** (prefill 8 + decode 8 tokens):

| Path | mean ± 95% CI |
|------|---------------|
| `engine_dispatch_standard` (sync) | 446.93 µs [443.31, 450.77] |
| `engine_dispatch_standard_async` | 449.04 µs [446.17, 451.91] |
| **Δ** | **+0.5%** (CIs overlap heavily — within noise) |

**Per-layer helpers** (isolating the `attention_*_async` + `read_hidden` + `flush` overhead):

| Path | mean ± 95% CI |
|------|---------------|
| `helpers/prefill_sync`  | 15.39 µs [15.31, 15.47] |
| `helpers/prefill_async` | 15.57 µs [15.40, 15.82] |
| `helpers/decode_step_sync`  | 13.25 µs [12.86, 13.58] |
| `helpers/decode_step_async` | 13.14 µs [12.73, 13.49] |

Async prefill is +1.2% (noise), async decode_step is −0.8% (noise — async measured marginally *faster*).

Interpretation: on CPU, the `BackendSlot::Async` enum match + `Ready*` handle allocation in the async helpers is **zero-overhead** as designed. The `Ready*` wrappers are bypass-thin — no syscalls, no commits, just an indirection through a `Box<dyn ...>`.

### 64-step bit-parity tests (synthetic weights, multi-step drift detection)

`cargo test -p larql-kv async_parity_long`

- `async_parity_long_run_no_drift` — 16-token prompt, 64 decode steps, `window=None`. Every intermediate hidden state asserted bit-identical between sync and async paths. Post-run cache size and memory must also match. ✅ PASS
- `async_parity_long_run_windowed_no_drift` — same with `window=Some(4)`, exercising `clip_kv` between every step. ✅ PASS

These catch drift that the short-run (4-token) parity tests would miss. None observed.

### Unit + integration tests

| Crate | Tests | Status |
|---|---|---|
| `larql-inference --lib --features metal` | 1052 | ✅ all pass (was 1014 before this session's coverage push) |
| `larql-kv --lib` | 221 | ✅ all pass |

Within those, async-specific tests:
- 19 in `async_compute_backend::tests` (handle/error/Ready helpers, custom inner, panic-default contracts via StubAsyncBackend, commit-control defaults)
- 10 in `async_compute_backend_cpu::tests` (6 bit-parity vs sync KvDispatch + windowed default + commit-control + recompute panic)
- 7 in `async_compute_backend_metal::tests` (compile assertion + 5 Metal-vs-CPU parity + commit-control)
- 5 in `kv_dispatch_helpers::tests` async-vs-sync parity (prefill, windowed prefill, decode_step, multi-step, empty prompt)
- 4 in `engines::standard::tests` (sync-vs-async on engine path including 64-step parity)

## Coverage (new code only — not the whole crate)

`cargo llvm-cov --features metal` against `crates/larql-inference/coverage-policy.json`:

```
Coverage policy passed: total 71.69% lines, included 96.49%,
                       7 files checked, 7 files at 90.0% default, 0 debt baselines.
```

| File | Lines% |
|------|------:|
| `async_compute_backend/mod.rs` | 95.42% |
| `async_compute_backend/cpu.rs` | 100.00% |
| `async_compute_backend/metal.rs` | 92.21% |
| `kv_dispatch/mod.rs` | 96.73% |
| `kv_dispatch/cpu.rs` | 94.41% |
| `kv_dispatch/metal.rs` | 96.70% |
| `kv_dispatch/helpers.rs` | 99.34% |

## Real-model proof — Gemma 3 4B Q4K

`cargo run -p larql-kv --release --example async_parity_real_model -- --vindex output/gemma3-4b-q4k-v2.vindex --tokens 16`

```
Loading Q4K vindex: output/gemma3-4b-q4k-v2.vindex
Prompt tokens: 6 ([2, 818, 5279, 529, 7001, 563]); decode steps: 16

== Sync (StandardEngine::new) ==
  prefill=421.13 ms  mean_decode=22.272 ms  tok/s=44.90

== Async (StandardEngine::with_async_backend(CpuBackend)) ==
  prefill=431.32 ms  mean_decode=22.024 ms  tok/s=45.40

== Verifying parity ==
  ✓ all 16 tokens bit-identical between sync and async paths
  Δ mean_decode = -1.11%  (22.272 ms sync → 22.024 ms async)
  ✓ within ±5% noise threshold
```

This is the strongest proof of the work: real Q4K production model, real f32 attention via lazy dequant, two independent decode paths through the trait, bit-identical token streams, no performance regression. Both `larql bench --cpu --engine standard` and the async opt-in now produce ~45 tok/s on Gemma 3 4B Q4K on M3 Max with 8 threads.

## Q4K dispatch fix (architectural gap closed)

**Before this session:** `StandardEngine` was broken on Q4K vindexes through any dispatch path — confirmed by running `larql bench --cpu --engine standard output/gemma3-4b-q4k-v2.vindex`:

```
Error: Q4K engine prefill failed
```

This is **not async-specific** — sync `StandardEngine::new` exhibits the same failure. Root cause:

1. `KvDispatch::attention_prefill(weights, tokens_embedded, layer, window)` takes f32 inputs and reads f32 attention tensors from `weights.tensors`.
2. `larql_vindex::load_model_weights_q4k` populates `weights` with **graph** tensors (embeddings, norm, lm_head) but leaves attention as Q4K bytes in the sibling `VectorIndex`.
3. The trait's `attention_prefill` has no `&VectorIndex` parameter, so backends can't route to Q4K kernels even when they support them.
4. `KvEngine::prefill_q4k` was added as an **engine-side override** (workaround) — but only `MarkovResidual`, `UnlimitedContext`, `TurboQuant` override it. `StandardEngine`, `NoCache`, `Apollo` inherit the default which falls back to f32 `prefill` and fails.

Engines that DO support Q4K (`MarkovResidual` et al.) **bypass `KvDispatch`** entirely in their `prefill_q4k` body:

```rust
// crates/larql-kv/src/engines/markov_residual/engine.rs:126
fn prefill_q4k(&mut self, weights, _ffn, index, token_ids, backend) {
    if let Some(h) = q4k_prefill_metal(weights, index, token_ids, backend) {
        // Metal fused decode_token path — backend.has_q4() gate.
        ...
    }
    ensure_attn_tensors_dequantised(weights, index);     // CPU fallback: bulk dequant
    rs_prefill_walk(weights, index, token_ids, ...)      // WalkFfn path
}
```

Neither branch uses `KvDispatch::attention_prefill`. The Metal path goes to `decode_token`; the CPU path dequantizes attention tensors into `weights.tensors` then runs the f32 walk path. **`larql_compute::QuantMatVec`** (which has `q4k_matvec`, `q6k_matvec`, etc.) **is not consumed by the engines at all** for the per-step path.

### What "using larql-compute properly" would look like

The spec already anticipated this (`compute-backend-redesign.md` §11.5):

> Resolved 2026-05-16: quantisation stays internal to the backend impl. The new trait's intent vocabulary is uniform (`attention_step`, `matmul`); backends route to f32 or Q4_K paths internally based on tensor type from `weights.tensors` / `VectorIndex`.

Reality has drifted from the spec. The right shape:

1. **`KvDispatch::attention_prefill` carries `Option<&VectorIndex>`** (or a unified "weight source" handle). Backends inspect it: f32 tensors in `weights.tensors` → f32 path; Q4K data in `index` + `backend.has_q4()` → Q4K matvec via `QuantMatVec::q4k_matvec`. Engines don't need to know.
2. **Drop `KvEngine::prefill_q4k`** as a separate trait method. Make `prefill` quantization-agnostic. The Q4K-specific engines (Apollo) that need bespoke logic can still do that internally, but the dispatch shape stays uniform.
3. **Move `q4k_prefill_metal` / `q4k_decode_token`** out of `larql-kv` and into `MetalBackend`'s `KvDispatch` impl. The engine just calls `backend.attention_prefill(weights, embed, layer, Some(index))` and the Metal backend chooses Q4K natively.
4. **`ensure_attn_tensors_dequantised`** becomes an opt-in CPU-fallback helper on the backend trait, not an engine-side concern.

**What landed (this session — Phase 1 of `kv-dispatch-quantization.md`):**

1. `ensure_attn_tensors_dequantised` moved from `larql-kv/src/engines/markov_residual/q4k.rs` to `larql-inference/src/vindex/dequant.rs` (so the trait impls can call it without a `larql-kv → larql-inference → larql-kv` cycle).
2. `KvDispatch::attention_prefill` / `attention_step` / `attention_step_windowed` gained `index: Option<&larql_vindex::VectorIndex>` parameter. Default impls accept and ignore. Same widening on `AsyncComputeBackend::attention_*_async` siblings.
3. Helpers (`kv_prefill_via_dispatch`, `kv_decode_step_via_dispatch`, async variants) thread the index through to the trait calls.
4. `StandardEngine::prefill_q4k` / `decode_step_q4k` now override the `KvEngine` defaults: call `ensure_attn_tensors_dequantised` once, then delegate to `do_prefill` / `do_decode_step` (the shared bodies that match on `BackendSlot::{Sync, Async}` and route through the helpers with `Some(index)`).
5. `larql-kv` re-exports `ensure_attn_tensors_dequantised` from its old location so existing call sites in `MarkovResidual` / `UnlimitedContext` / `TurboQuant` keep working.

**Phase 2 (deferred):** Other engines (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`) still have their bespoke `prefill_q4k` overrides that bypass the dispatch trait (`q4k_prefill_metal`, `rs_prefill_walk`, etc.). They can migrate to the dispatch-trait-only path in a follow-up session.

**Phase 3 (deferred):** Replace bulk dequant in `ensure_attn_tensors_dequantised` with native Q4K matvec on `CpuBackend` via `larql_compute::QuantMatVec::q4k_matvec`. This drops the memory cost of full-f32 attention tensors. Per-call kernel work.

## What this proof does NOT cover

- **Metal real deferred dispatch (A4)** — not yet implemented; the A3 scaffold delegates to CPU.
- **Multi-engine async opt-in** — only `StandardEngine` has `with_async_backend`. Other engines compose on the same `BackendSlot` pattern in subsequent slices.
- **Native Q4K CPU matvec** — Phase 3 of `kv-dispatch-quantization.md`. Today's path still bulk-dequants attention tensors to f32. Memory cost: full f32 attention tensors per layer remain resident.

## Conclusion

The A1–A3 + A5 (StandardEngine) trait machinery is sound and zero-overhead on CPU. Bit-parity holds across 64 synthetic decode steps with no drift, AND across 16 real-model Gemma 3 4B Q4K decode steps. The trait widening unblocked StandardEngine on Q4K through the dispatch path — `larql bench --cpu --engine standard` now produces 45 tok/s on Gemma 3 4B Q4K where it previously failed with "Q4K engine prefill failed". A4 (real Metal tok/s) and Phase 2/3 of the Q4K refactor are documented follow-up work.
