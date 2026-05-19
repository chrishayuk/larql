# AsyncComputeBackend — Specification

**Status:** Trait surface locked 2026-05-16; all 6 open questions resolved. Implementation in progress — A1 (trait + handles), A2 (`CpuBackend` async impl), A3 (`MetalBackend` scaffold), and A5 (`StandardEngine` opt-in slice) landed 2026-05-16. A3's Metal-feature validation gate met (18 async tests pass under `--features metal`, including 4 Metal-aware bit-parity tests vs CPU). A4 (real Metal deferred dispatch — the tok/s-win step) and the remaining engines' A5 migration are next. See migration plan §10.
**Audience:** LARQL contributors who will write per-layer Metal/Vulkan/CUDA kernels for engine intents.
**Scope:** Define the deferred-dispatch trait surface that gives per-layer
[`KvDispatch`](./compute-backend-redesign.md) intents Metal/Vulkan/CUDA-class
performance without forcing the synchronous trait to award a GPU command-buffer
sync per layer.

This spec is the architectural prerequisite for path **B1** of the
ComputeBackend redesign — bringing async/batched dispatch forward from the
deferred CUDA-era position
([`compute-backend-redesign.md` §11.4](./compute-backend-redesign.md)) so that
per-layer engine intents can actually compose into one GPU command buffer per
decode step.

---

## 1. Purpose

Step 5 of the ComputeBackend redesign (real Metal kernels for `KvDispatch`)
discovered a structural problem: the synchronous `KvDispatch` trait forces
each per-layer intent into its own GPU sync. The existing Metal fast path
(`MetalBackend::decode_token`) encodes all 34 layers' attention + FFN into
one command buffer and commits once per token (~13ms on Gemma 3 4B). A
synchronous `attention_step` called per-layer would require 34 command buffers
per token — orders of magnitude slower than the fused path, regardless of
shader quality.

This spec defines the trait shape that fixes that. Engines submit intents to
a backend that *accumulates* them into one in-flight command buffer per
decode session, commits at engine-declared checkpoints, and only blocks on
read-backs the engine actually needs. Per-layer intents become first-class
GPU-fast primitives, not just CPU-fallback debug paths.

## 2. Motivation

Without `AsyncComputeBackend`, three concrete things stay out of reach:

1. **Engine-aware kernel fusion at GPU speed.** `standard:window=N`'s
   decoded-CPU bench shows 1.7× faster decode at small windows. Metal could
   do better via register-resident K/V — but only if windowed attention is
   one kernel inside one command buffer, not one kernel per layer in 34
   command buffers.
2. **Per-engine prefill graphs.** Apollo's boundary upload can pipeline with
   the first attention dispatch on Metal — but only if both ops live in the
   same command buffer. Today's sync trait would force a sync between them.
3. **`markov-rs` Metal recompute path.** The engine recomputes K/V from
   residuals per layer per token. Sync dispatch would mean 34 K/V-recompute
   commits per token. Deferred would be one.

Honest framing of B1's cost: it's a multi-month commitment (trait redesign +
backend reimplementations + engine migration). The value is structural:
every per-layer intent we land becomes a real GPU-speed primitive, not a
research-only fallback.

## 3. Decision

Add `AsyncComputeBackend: ComputeBackend + KvDispatch` as a *sibling* trait
to `KvDispatch`, not a replacement. Engines opt in to async dispatch when
they need GPU-batched per-layer intents. Engines that don't need it stay on
synchronous `KvDispatch`. Both shapes coexist.

The async trait exposes intents that return *handles* (futures over future
GPU work) instead of immediate results. The backend internally maintains
one in-flight command buffer per decode session. Reads (data flowing back
to host) trigger commit + wait + flush. Engines control when commit happens
by choosing when to read.

Synchronous `KvDispatch` continues to work — its methods commit+wait per
call. `AsyncComputeBackend` is the performance path; `KvDispatch` stays
the correctness reference.

## 4. What's in / out of scope

### 4.1 In scope

- `AsyncComputeBackend` trait surface (intent → handle, commit semantics,
  read-back protocol).
- `KvHandleAsync` / `ResidualHandleAsync` / `TokenHandle` etc. — handle
  types that represent pending GPU work.
- Per-backend implementation contracts:
  - `MetalBackend` accumulates into one `MTLCommandBuffer` per session;
    commits on `flush()` or first read.
  - `VulkanBackend` uses one `VkCommandBuffer` + semaphores.
  - `CudaBackend` uses one `cudaStream` + graph capture where appropriate.
  - `CpuBackend` provides a degenerate impl: every intent completes
    synchronously, every handle is a `Ready<T>`. Used for parity testing.
- Engine migration: which engines opt in to async, which stay on sync
  `KvDispatch`, and how the chosen path is exposed at construction.
- Bit-parity tests between sync and async paths on the synthetic substrate.

### 4.2 Out of scope (deliberately)

- Cross-request batching (one server request's command buffer is not shared
  with another's). Session-local async only. Cross-request batching is a
  scheduler concern; deferred to a future "request routing" spec.
- Streaming tokens to the client (the engine's `on_token` callback). Async
  dispatch is internal to the decode loop; token emission stays on
  whatever cadence the engine chooses to flush at.
- Real CUDA implementation. Spec accommodates the shape (CUDA streams + graph
  capture); landing the impl is a separate work item.
- Removing synchronous `KvDispatch`. Both traits coexist permanently —
  engines and backends choose per-call-site.

## 5. The intent vocabulary (async)

Each method on `AsyncComputeBackend` mirrors a sync `KvDispatch` method but
returns a handle instead of the immediate result. Handles compose: a method
that takes a `TokenHandle` and produces an `AttentionHandle` doesn't block
— it just records the dependency in the in-flight command buffer.

| Sync (`KvDispatch`) | Async (`AsyncComputeBackend`) | Returns |
|---|---|---|
| `attention_step(...) -> Option<Array2<f32>>` | `attention_step_async(...) -> AttentionHandle` | Handle to post-attention hidden |
| `attention_step_windowed(...) -> Option<Array2<f32>>` | `attention_step_windowed_async(...) -> AttentionHandle` | Same |
| `attention_prefill(...) -> Option<(Array2<f32>, KvHandle)>` | `attention_prefill_async(...) -> (AttentionHandle, KvHandle)` | Handle + (already-handled) KV |
| `recompute_kv_from_residuals(...) -> Option<KvHandle>` | `recompute_kv_from_residuals_async(...) -> KvHandle` | KV handle |
| `forward_from_layer(...) -> Option<Array2<f32>>` | `forward_from_layer_async(...) -> AttentionHandle` | Hidden handle |
| `upload_boundary_residual(...) -> Option<ResidualHandle>` | `upload_boundary_residual_async(...) -> ResidualHandle` | Residual handle |
| `read_kv_to_host(...) -> Option<(Array2, Array2)>` | (n/a — read-back triggers commit) | |

New trait-level methods:

| Method | Purpose |
|---|---|
| `flush(&self) -> Result<()>` | Commit the in-flight command buffer + wait. Engine calls at decode-step boundaries. |
| `read_hidden(&self, h: &AttentionHandle) -> Array2<f32>` | Trigger commit (if not already), wait, copy hidden to host. |
| `is_pending(&self, h: &AttentionHandle) -> bool` | Non-blocking check — used for diagnostics and bench instrumentation. |

## 6. The trait surface (full Rust)

Trait + handle types live in `crates/larql-inference/src/async_compute_backend/mod.rs`. Backend implementations are sibling submodules: `cpu.rs`, `metal.rs` (`#[cfg(feature = "metal")]`), and — when they're written — `vulkan.rs` and `cuda.rs`.

### 6.1 Handle types

```rust
use ndarray::Array2;
use std::sync::Arc;

/// Pending result from an async attention dispatch — placeholder for a
/// hidden state that will exist after the backend commits its in-flight
/// command buffer.
///
/// Engines compose `AttentionHandle`s without blocking. Reading the
/// underlying `Array2<f32>` triggers commit + wait via
/// [`AsyncComputeBackend::read_hidden`].
pub struct AttentionHandle {
    inner: Arc<dyn AsyncHandleInner<Output = Array2<f32>>>,
}

/// Pending result from a residual-upload dispatch.
pub struct ResidualUploadHandle {
    inner: Arc<dyn AsyncHandleInner<Output = ()>>,
}

/// Backend-side trait for pending result types. Engines never call this
/// directly; backends implement it on their per-platform handle types
/// (`MetalAttentionHandle`, `VulkanAttentionHandle`, `CudaAttentionHandle`,
/// `CpuReadyHandle<T>`).
pub trait AsyncHandleInner: Send + Sync {
    type Output;
    /// Non-blocking completion check. Returns true if the backend's
    /// command buffer covering this handle has been committed AND
    /// completed. False otherwise (including pending and in-flight).
    fn is_complete(&self) -> bool;

    /// Read the output. Blocks on commit + wait if not yet complete.
    /// Consumes the handle.
    ///
    /// Implementations must be idempotent — calling `read` on a handle
    /// whose backend has already committed (e.g. because another handle
    /// in the same batch was read) returns immediately.
    fn read(self: Arc<Self>) -> Self::Output;
}
```

`KvHandle` and `ResidualHandle` from
[`KvDispatch`](./compute-backend-redesign.md) stay as-is. In async usage
they represent backend-side state whose contents are pending; queries
on them follow the rules in §11.2.

### 6.2 The trait

```rust
use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::kv_dispatch::{KvDispatch, KvHandle, ResidualHandle};
use crate::model::ModelWeights;

/// Async/batched dispatch surface — sibling to [`KvDispatch`].
///
/// Implementers maintain an in-flight command buffer (or equivalent on
/// non-Metal backends) per session. Each async method *encodes* an
/// intent into that buffer and returns a handle. The buffer is
/// committed on:
/// - explicit [`flush`](Self::flush) call
/// - first [`read_hidden`](Self::read_hidden) (or other handle read)
/// - backend-internal trigger (buffer overflow — implementation choice).
///
/// Engines opt in by constructing themselves with an `AsyncComputeBackend`
/// (e.g. `StandardEngine::with_async_backend`). Engines that don't opt
/// in stay on synchronous [`KvDispatch`].
///
/// `AsyncComputeBackend` supertraits `ComputeBackend + KvDispatch` so
/// it's also a valid `EngineBackend` — implementers get the sync trait
/// for free (every async method's body in a `Ready` wrapper produces a
/// correct sync impl; backends override per intent as real deferred
/// dispatch lands).
pub trait AsyncComputeBackend: ComputeBackend + KvDispatch + Send {
    // ── Commit / flush control ──────────────────────────────────────

    /// Commit the in-flight command buffer (if any) and wait for GPU
    /// completion. Engines call this at decode-step boundaries to bound
    /// the dispatch cadence at one GPU sync per token.
    ///
    /// Returns `Ok(())` on success. Backend-specific errors (device
    /// removed, command buffer rejected, etc.) surface as `Err`.
    fn flush(&self) -> Result<(), AsyncDispatchError>;

    /// Read the hidden state from an `AttentionHandle`. Triggers commit
    /// + wait if not already complete. Consumes the handle.
    fn read_hidden(&self, handle: AttentionHandle) -> Array2<f32>;

    /// Non-blocking diagnostic: is the backend currently holding an
    /// uncommitted command buffer? Used for instrumentation and
    /// bench-time validation that engines are batching effectively.
    fn has_pending_work(&self) -> bool;

    // ── Async intents (mirror KvDispatch with handle returns) ───────

    /// Async equivalent of [`KvDispatch::attention_step`].
    fn attention_step_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
    ) -> AttentionHandle;

    /// Async equivalent of [`KvDispatch::attention_step_windowed`].
    fn attention_step_windowed_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
    ) -> AttentionHandle {
        // Default: dispatch unbounded variant + clip after.
        let h = self.attention_step_async(weights, query, kv, layer, abs_position);
        self.clip_kv(kv, window);
        h
    }

    /// Async equivalent of [`KvDispatch::attention_prefill`].
    /// `KvHandle` returns immediately (backend-side state); the
    /// `AttentionHandle` is pending until commit.
    fn attention_prefill_async(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
    ) -> (AttentionHandle, KvHandle);

    /// Async equivalent of [`KvDispatch::recompute_kv_from_residuals`].
    fn recompute_kv_from_residuals_async(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> KvHandle;

    /// Async equivalent of [`KvDispatch::upload_boundary_residual`].
    /// The returned handle is pending until commit; subsequent
    /// `forward_from_layer_async` calls referencing it can fuse with the
    /// upload in the same command buffer (Apollo's pipelined upload win).
    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle);

    /// Async equivalent of [`KvDispatch::forward_from_layer`].
    fn forward_from_layer_async(
        &self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle;
}

/// Errors from the deferred-dispatch surface.
#[derive(Debug)]
pub enum AsyncDispatchError {
    /// GPU device removed or unresponsive.
    DeviceError(String),
    /// Command buffer rejected at commit time (typically a backend bug
    /// — encoded operation that the runtime refused).
    CommandBufferRejected(String),
    /// Backend-specific.
    Other(String),
}

impl std::fmt::Display for AsyncDispatchError { /* ... */ }
impl std::error::Error for AsyncDispatchError {}
```

### 6.3 Handle composition

The intended pattern: engines thread handles through the per-layer loop
without blocking; the only sync points are `read_hidden` at sample-time
and `flush` at decode-step boundaries.

```rust
// Engine-side decode loop, abridged.
fn decode_step_async(
    backend: &dyn AsyncComputeBackend,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
) -> Array2<f32> {
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let mut h_step = h_new;

    for layer in 0..weights.num_layers {
        // attention_step_async: encodes into command buffer, returns handle.
        // Does NOT commit. `handles[layer]` is mutated to reflect the
        // queued K/V append; reads on it follow §11.2.
        let h_post_attn_handle = backend.attention_step_async(
            weights, &h_step, &mut handles[layer], layer, abs_position,
        );

        // FFN runs on host (or remote via FfnBackend). To call it, we need
        // the attention result as concrete data. This is the §11.5 question
        // — the answer for v1 is: read here, accept the per-layer sync.
        //
        // The win comes from `attention_step_async`'s K/V append being
        // batched into the same command buffer as the next layer's
        // attention (since `read_hidden` only blocks on this layer's
        // hidden, not on the cache writes).
        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h_step = h_out;
    }

    // End of decode step — explicit flush ensures any deferred work
    // (e.g. background K/V appends) completes before the next call.
    backend.flush().expect("decode step flush");
    h_step
}
```

The above is a *correctness baseline* using v1's "FFN on host" pattern
(§11.5). It's correct and gives some batching (K/V append fuses with
next attention), but the per-layer `read_hidden` still forces commits.

The real win requires deferring FFN too — see §11.5 resolved option
(b): `ffn_step_async` added in a follow-up release. With it, the loop
becomes:

```rust
// Future shape (post-§11.5 implementation):
for layer in 0..weights.num_layers {
    let attn_handle = backend.attention_step_async(...);
    let ffn_handle = backend.ffn_step_async(weights, layer, attn_handle);
    // ffn_handle feeds into next layer's attention as the residual base
    h_step_handle = backend.residual_add_async(h_step_handle, ffn_handle);
}
let final_hidden = backend.read_hidden(h_step_handle);
backend.flush()?;
```

One commit per decode step. One GPU sync. Matches today's fused
Metal cadence — but with per-layer intent granularity, so engines that
need the control (MarkovRS recompute, Apollo boundary upload) can opt in
without losing speed.

## 7. Backend implementations

### 7.1 `MetalBackend`

Maintains per-session state:
- One `MTLCommandQueue` (shared across the process).
- One in-flight `MTLCommandBuffer` per active `AsyncComputeBackend` session
  (per-thread or per-engine instance).
- A pipeline-state cache (same as sync path).
- A pending-handles registry — each async dispatch records (encoder pos,
  output buffer) for `read_hidden` to look up later.

Commit triggers:
- Explicit `flush()` call (engine declares "end of work batch").
- First `read_*` call against any pending handle in the batch.
- Buffer pool exhaustion (backend's choice — eviction policy).

The default commit point engines should use: end of each decode_step. That
gives one GPU sync per token, matching the existing fused path's cadence.

### 7.2 `VulkanBackend`

Same shape with Vulkan primitives:
- One `VkCommandPool` per session.
- Active `VkCommandBuffer` accumulates intents.
- `VkSemaphore` chains express handle dependencies.
- Submit-and-wait on flush / read.

Key Vulkan-specific concern: explicit synchronisation between successive
encoded ops. Backend handles this internally (engines don't see barriers).

### 7.3 `CudaBackend` (future)

CUDA streams map naturally: one `cudaStream_t` per session. Intent encoding
becomes `cudaLaunchKernel` on the stream. `cudaStreamSynchronize` is the
"flush"; `cudaMemcpyAsync` + `cudaStreamSynchronize` is "read_hidden".

CUDA graph capture is an optimization: backend can capture the per-decode-step
pattern once, replay per token. Engines don't see graph capture; backend
chooses internally.

### 7.4 `CpuBackend`

Degenerate. Every intent completes synchronously inside the encode call.
Handles wrap `Ready<T>`. `flush()` is a no-op. `read_hidden` returns the
already-computed value.

This is the parity reference — async vs sync output must be bit-identical
when both run on CPU.

## 8. Engine-side API

Two opt-in mechanisms for engines:

### 8.1 Per-engine async constructor

```rust
impl StandardEngine {
    /// Construct an async-dispatch variant. Backend must implement
    /// `AsyncComputeBackend`. Decode uses async dispatch; tok/s
    /// expected ~equal to today's fused Metal on standard cache.
    pub fn with_async_backend(
        window_size: Option<usize>,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self;
}
```

Engines that don't opt in keep using `with_backend(Box<dyn EngineBackend>)`
and stay on synchronous `KvDispatch`.

### 8.2 Per-call hint

(Out of scope for v1 — would let engines mix sync and async per layer.
Could be added later as `KvDispatch::supports_async() -> Option<&dyn AsyncComputeBackend>`.)

### 8.3 KvEngine trait

`KvEngine::prefill` and `decode_step` signatures stay the same. The async
variant is implementation-internal — engines that opted in via the async
constructor use async dispatch; the trait signature still returns immediate
`Array2<f32>` because the engine internally flushes at decode-step
boundaries.

This keeps `KvEngine` callers (CLI, server, etc.) backward-compatible.

## 9. Compatibility matrix

Adding async columns to the engine × backend matrix:

| Engine | Sync `KvDispatch` | `AsyncComputeBackend` opt-in | Win at opt-in |
|---|---|---|---|
| `Standard` | ✅ (current) | ✅ | Metal/Vulkan/CUDA fast path at per-layer granularity |
| `Standard:window=N` | ✅ | ✅ | Specialised windowed-attention shader, fused |
| `NoCache` | ✅ | ❌ (no win — re-runs full forward each step) | n/a |
| `MarkovResidual` | ✅ | ✅ | Recompute K/V batched into one CB per step |
| `UnlimitedContext` | ✅ | ✅ | Window-checkpoint amortisation on GPU |
| `TurboQuant` | ✅ | ✅ | Codec encoded into the attention CB |
| `Apollo` | ✅ | ✅ | Boundary upload pipelined with first attention |

Every engine has both paths available. Backend support determines whether
the async path is actually faster than sync — engines build with whichever
the substrate offers.

## 10. Migration plan

Each step independently shippable. Each step has a parity guarantee so the
default user experience doesn't regress.

### 10.1 Step A1 — Define the trait + handle types ✅ landed 2026-05-16

`crates/larql-inference/src/async_compute_backend/mod.rs` — trait +
handle types + `Ready*` helpers. No backend impls in `mod.rs`; they
live in sibling `cpu.rs` / `metal.rs` submodules. Compile-only
deliverable.

Spec deviation documented in the module's header comment: the spec
sketched `Arc<dyn AsyncHandleInner<Output = T>>` with
`read(self: Arc<Self>)`; the landed implementation uses per-handle inner
traits (`AttentionHandleInner`, `ResidualUploadHandleInner`) with
`read(self: Box<Self>) -> Output`, which is object-safe on stable Rust
(`self: Arc<Self>` on trait objects requires `arbitrary_self_types`).
Spec semantics (consumed on read, idempotent at the backend level)
preserved.

**Parity:** N/A — type definitions don't change behaviour. 3 unit
tests cover `Ready*` round-trips + error `Display`.

### 10.2 Step A2 — `CpuBackend` async impl ✅ landed 2026-05-16

Trivial — every overridden method is the sync `KvDispatch` impl wrapped
in `Ready*`. Established the parity reference: `CpuBackend` async output
is bit-identical to `CpuBackend` sync output on the synthetic substrate.

Method coverage: overrides `attention_step_async`,
`attention_prefill_async`, `upload_boundary_residual_async`,
`forward_from_layer_async`. `attention_step_windowed_async` uses the
trait's default decomposition (step + clip). `recompute_kv_from_residuals_async`
stays at the trait `unimplemented!()` default since CPU's sync
`KvDispatch` doesn't implement it either (`markov-rs` territory).
Commit-control methods (`flush`, `read_hidden`, `has_pending_work`)
use the trait defaults — correct for any non-deferred backend.

**Parity:** Bit-exact vs sync `KvDispatch` on `CpuBackend`. 6 unit
tests in `async_compute_backend/cpu.rs` enforce this — one per
overridden method plus the windowed default and commit-control sanity.

### 10.3 Step A3 — `MetalBackend` async scaffolding ✅ landed 2026-05-16

`MetalBackend` (now in `larql-compute-metal` after the parallel
metal-extraction settled) implements `AsyncComputeBackend` by
delegating every async method to `CpuBackend`'s async impl (same
CPU-delegation pattern as `kv_dispatch_metal.rs`). Validates the trait
shape against actual `MetalBackend` ownership patterns without writing
new shaders. No tok/s win yet — every call has CpuBackend's cost.

File `crates/larql-inference/src/async_compute_backend/metal.rs` is
behind `#[cfg(feature = "metal")]` and includes a compile-time
`assert_async::<MetalBackend>()` test plus bit-parity tests vs
`CpuBackend` that auto-skip when `MetalBackend::new()` returns `None`.

**Parity:** Bit-exact vs CPU on synthetic. 4 Metal-aware tests pass
under `--features metal` (including bit-parity on prefill hidden + K +
V and bit-parity on decode-step hidden). Tok/s catastrophically worse
than today's fused Metal (each call still commits separately at this
scaffolding stage) — A4 is the gate where the tok/s shape changes.

### 10.4 Step A4 — `MetalBackend` real deferred dispatch

Replace the per-call commits with actual command-buffer accumulation:
- One `MTLCommandBuffer` per session.
- Intent encoding appends to it.
- `flush()` commits + waits.
- First `read_hidden` commits + waits.

This is where the tok/s win starts to land. Engines that opt into async
on Metal see their decode step go from "N commits per token" to "1 commit
per token."

**Parity:** Bit-exact vs CPU on synthetic. Tok/s expected to match or
exceed today's fused Metal path for engines that flush at decode-step
boundaries.

### 10.5 Step A5 — Per-engine opt-in (StandardEngine slice ✅ landed 2026-05-16)

Per-engine constructor `with_async_backend(window_size, Box<dyn AsyncComputeBackend>)`.
Engines internally branch between sync and async dispatch helpers based
on which constructor was used.

**StandardEngine slice landed 2026-05-16:**

- `kv_prefill_via_dispatch_async` + `kv_decode_step_via_dispatch_async`
  helpers added to `larql-inference::kv_dispatch_helpers`. Per spec
  §11.5 v1 semantics: `AttentionHandle` is read per layer to drive FFN
  on host; `backend.flush()` called at end of decode step.
- `StandardEngine` refactored to carry an internal
  `BackendSlot::{Sync(Box<dyn EngineBackend>) | Async(Box<dyn AsyncComputeBackend>)}`
  enum (avoids unstable trait-upcasting under the workspace's pinned
  `rust-version = 1.80`). `prefill`/`decode_step` match on the variant
  and route to the matching helper.
- Existing `new` / `with_backend` constructors unchanged —
  backward-compatible.
- 8 new bit-parity tests in `larql-inference` (5 async-helper vs sync-helper)
  and `larql-kv` (2 sync-engine vs async-engine, 1 backend-name).

**Other engines (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`,
`NoCache`, `Apollo`)** follow the same pattern in subsequent slices —
each adds a `with_async_backend` constructor and a `BackendSlot`
variant. Estimated ~1–2 weeks per engine.

CLI/server async-backend selection follows once the other engines'
slices land.

**Parity:** Existing snapshot tests on `larql run` still pass byte-for-byte.
The async path on `CpuBackend` is a `Ready*`-wrapped pass-through (A2),
so bit-parity is the trait-shape contract — verified by the new tests.

### 10.6 Step A6 — Per-engine specialised shaders

`attention_step_windowed` gets a fused windowed-attention shader. Apollo
gets pipelined boundary upload. MarkovResidual gets the K/V-recompute
kernel. Each new shader is paired with a bench against the real model.

**Parity:** Per-shader bit-parity vs CPU within declared numerical tolerance.

### 10.7 Step A7 — `VulkanBackend` async

Implement `AsyncComputeBackend` for Vulkan. Same shape as Metal; different
primitives. Validates the trait against a second pipeline-based GPU.

### 10.8 Step A8 — `CudaBackend`

Add `CudaBackend` implementing `AsyncComputeBackend`. The async trait was
designed against the CUDA stream model (deferred dispatch + sync points
on read), so this should be the most natural-fitting backend.

## 11. Open questions

### 11.1 What if engine doesn't flush before next prefill?

**Resolved:** backend auto-flushes when the in-flight command buffer
exceeds a backend-defined threshold. Engines don't need to handle this.
Default threshold per backend:
- Metal: 256 MB encoded buffer size OR 4096 encoded ops (whichever first).
- Vulkan: equivalent thresholds expressed via `VkPhysicalDeviceLimits`.
- CPU: n/a (synchronous degenerate impl).

Engines that want deterministic batching should call `flush()` explicitly
at known boundaries. The auto-flush is a safety net, not a guarantee.

### 11.2 Should `KvHandle::cached_len` block?

**Resolved:** option (b) — `KvHandle::cached_len` returns the eventual
length tracked engine-side. Backends update the engine-visible counter
synchronously even when the underlying GPU work is pending. No blocking.

This means `KvHandle::cached_len` is always consistent with the engine's
view of what's been "appended" — even if GPU hasn't yet written the
bytes. Engines treat `cached_len` as an authoritative engine-side state
counter, not as a GPU-state probe. Backends document this in their
`KvHandleInner::cached_len` impl.

Reads via `read_kv_to_host` DO trigger flush (see §11.3) and return the
committed contents. Sync queries on `KvHandle::cached_len` never block.

### 11.3 Can engines mix sync and async on the same handle?

**Resolved:** yes. Sync data-read methods on a handle that has pending
async work trigger flush + wait, then return the committed data.

Concrete sync methods that flush:
- `KvDispatch::read_kv_to_host(&KvHandle)`
- `KvDispatch::clip_kv(&mut KvHandle, ...)` — must reason about current
  contents, so flushes before clipping.

Sync methods that DON'T flush:
- `KvHandle::cached_len()` — see §11.2.
- `KvHandle::backend_name()` — backend-side metadata only.

Engines mixing modes pay one flush cost per crossing. Backends document
this in the trait doc-comment of each method.

### 11.4 Single-threaded session ownership?

**Resolved:** `Send` but not `Sync`. The trait declaration in §6.2 has
`Send` as a supertrait but not `Sync`.

Implications:
- A backend instance moves between threads freely (`Send`).
- A backend instance is NOT shared across threads concurrently — if two
  threads need GPU work, each constructs its own backend (one per
  session).
- Servers handling concurrent requests construct one
  `AsyncComputeBackend` per request handler. The cost is one
  `MTLCommandQueue` per request — Metal allows many cheaply.
- Cross-request batching (would require `Sync`) is deferred to a future
  "scheduler" layer above `AsyncComputeBackend`.

`ComputeBackend` (sync) stays `Send + Sync` — it's stateless per call.
The async sibling's `!Sync` is what's load-bearing here.

### 11.5 How does FFN integrate?

**Resolved:** option (a) ships in v1; option (b) follows in a v2
release. The two stages:

**v1 (Steps A1-A5):** FFN stays synchronous via `FfnBackend`. Engines
await each layer's attention result, run FFN on host, submit next
layer's attention. Decode loop has 1 commit per layer (34 commits
per token on Gemma 3 4B). Some batching: K/V append fuses with the
NEXT layer's attention dispatch in the same command buffer, since
`read_hidden` only forces commit on the hidden, not on the cache write.

This is HALF the win of the fully-deferred shape. The other half (FFN
in the command buffer) lands in v2.

**v2 (Step A6+):** Add `ffn_step_async` to `AsyncComputeBackend`.
Backend encodes the layer's gate/up/down matmuls into the command
buffer. Engines compose attention + FFN handles without any host
round-trip until decode-step end. 1 commit per decode step (matches
today's fused path) with per-layer intent granularity.

Why v1 first: shipping the trait + scaffolding without `ffn_step_async`
proves the deferred-dispatch shape with smaller risk. v2 adds a single
new intent on top, doesn't restructure the trait. Engines that need v1
speed can stay on the existing fused `decode_token` (via the
sync-coarse `decode_token_full` intent — see follow-up below).

**Follow-up clarification:** v1 still benefits from a coarse
`decode_token_full` intent on `AsyncComputeBackend` that maps directly
to today's fused Metal pipeline for engines that don't need per-layer
control. That intent lands as part of Step A4 (Metal real deferred
dispatch) so `StandardEngine` has a fast path even before v2's
`ffn_step_async`.

### 11.6 Where does this spec live?

**Resolved:** stays in `larql-inference/docs/specs/`. Already
cross-referenced from `compute-backend-redesign.md` §11.4.

## 12. Non-goals

- **Not a kernel-perf project.** This spec defines the dispatch shape that
  makes future kernel work possible. Tok/s wins come from Step A4 onward.
- **Not a server-side concurrency redesign.** One session per request.
- **Not a cross-engine batching layer.** Each engine's session is independent.
- **Not a wire-protocol change.** Server / CLI APIs unchanged; engines
  internally use async.

## 13. Acceptance criteria

This spec is accepted (signed off, ready to start Step A1) when:

1. The §5 intent vocabulary covers every async variant of every existing
   `KvDispatch` intent that engines actually use.
2. The §6 handle semantics type-check — `AttentionHandle`, `KvHandle`,
   `ResidualHandle` compose into the per-layer decode loop without
   forcing intermediate flushes.
3. The §7 per-backend impl contracts are concrete enough that a Metal
   implementer can write Step A3 without further design questions.
4. The §10 migration plan is implementable without backwards-incompatible
   breaks for Steps A1-A3. A4 onwards may break the sync trait's behaviour
   if real perf demands it, but with a deprecation path.
5. §11 open questions resolved or explicitly deferred.

---

## Appendix A — Relationship to other specs

- [`compute-backend-redesign.md`](./compute-backend-redesign.md) §11.4
  deferred this work to CUDA-era. This spec brings it forward to enable
  Metal per-layer tok/s. §11.4 in that spec gets an "obsoleted by
  `async-compute-backend.md`" note.
- [`kv-engine-unification.md`](./kv-engine-unification.md) §10.6 deferred
  the server's KV-engine wiring until `KvEngine` had a GPU side. The GPU
  side lands here, via `AsyncComputeBackend`. Server wiring follows.

## Appendix B — Honest scope expectation

- **This spec:** ~1 session of focused writing.
- **Step A1 (trait + handles):** ~1 week.
- **Step A2 (CpuBackend async):** ~1 week (with parity tests).
- **Step A3 (MetalBackend scaffolding):** ~1 week.
- **Step A4 (MetalBackend deferred dispatch — the tok/s win):** ~4-8 weeks
  including command-buffer ownership, pipeline-state cache extension,
  bench validation against real models.
- **Step A5 (engine opt-in):** ~1-2 weeks per engine; 6 engines.
- **Step A6 (specialised shaders):** ongoing. Each shader is its own
  measure-first-then-write effort.
- **Step A7 (VulkanBackend):** ~6-10 weeks (bring-up + bug hunt + shader port).
- **Step A8 (CudaBackend):** ~6-10 weeks.

Total realistic timeline: **6-12 months** end-to-end. The unification PR
(steps 1-7 of `kv-engine-unification.md` + steps 1-4 of
`compute-backend-redesign.md` + 3a/b/c) is shippable today; this work
starts the engine-aware-Metal effort that compounds on top of it.

That number is the load-bearing one. B1 isn't a tweak — it's the proper
foundation for engine-aware multi-backend GPU performance, and pretending
it's smaller would be lying.
