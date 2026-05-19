# ComputeBackend Redesign — Specification

**Status:** Step 1 (trait shape) locked 2026-05-16. Step 2 (CpuBackend conformance) in progress.
**Audience:** LARQL contributors planning multi-backend GPU support.
**Scope:** Define a `ComputeBackend` trait surface that lets `KvEngine`
implementations express compute *intent* (windowed attention, K/V
recompute, boundary upload, fused norm+residual) independently of which
substrate runs it (CPU, Metal, Vulkan). CUDA is deferred — designed
against pipeline-based GPUs (Metal + Vulkan) plus CPU as the
synchronous-backend degenerate case. The output of this spec is the
trait shape that future Metal kernel work targets from day one and that
the Vulkan bring-up has a defined contract to fill in.

This spec is the architectural prerequisite for the `KvEngine`
unification's tok/s value proposition — without it, the unification's
engine choices only affect CPU dispatch, which is not the production
substrate. See
[`kv-engine-unification.md`](./kv-engine-unification.md) §10.4 and the
broader architectural discussion that motivated this spec.

---

## 1. Purpose

The current `ComputeBackend` trait
(`crates/larql-compute/src/lib.rs`) exposes a flat surface of compute
primitives (matmul, softmax, etc.) shared across CPU and Metal. It does
not expose any concept of:

- engine-specific shader specialisation (windowed attention,
  fused-residual recompute);
- pipeline state caching as a first-class resource;
- engine-declared intent ("I need K/V for these positions, with this
  windowing policy") that the backend can route to the optimal kernel.

As a result, `KvEngine` implementations are GPU-opaque. They hand off
to the forward pass and get a result back; the Metal pipeline is
invisible to the policy layer. Engine-aware kernel fusion, compute-aware
engine selection, and per-engine prefill graphs are all foreclosed.

This spec defines the trait surface that fixes that, designed against
Metal + Vulkan + CPU concurrently so the first kernel work landed
against the trait isn't Metal-shaped at the expense of Vulkan.

## 2. Motivation

Three concrete things become possible under this redesign that aren't
today:

1. **Engine-aware kernel fusion.** `standard:window=N` on Metal can
   request a fixed-size windowed-attention shader variant; the
   `MarkovResidualEngine` on Vulkan can request a fused
   residual-to-K/V recompute kernel; today neither has a way to ask.
2. **Compute-aware engine selection.** Auto-pick the engine for the
   substrate. `unlimited-context` may dominate on CPU (cheap recompute,
   amortise prefill); `standard` may dominate on Metal where
   incremental K/V append is one kernel dispatch and checkpoint
   overhead isn't worth it. The choice is measurable per-backend.
3. **Per-engine prefill graphs.** Apollo's boundary upload can pipeline
   with first attention on Metal; same intent on Vulkan compiles to a
   different command-buffer shape; on CPU it's a synchronous load
   followed by forward. Engine declares the intent once; backend
   orchestrates per-substrate.

The honest tok/s framing (from the architectural discussion):
- Steps 1-3 of *this* migration: still no tok/s win. Plumbing for the
  intent-based dispatch and CPU-backend conformance.
- Step 4 (Metal as a real engine target): tok/s matches current Metal
  numbers — no regression, no win, but engine selector now works on GPU.
- Step 5 (per-engine kernel work via the new trait): per-engine wins
  start landing. `standard:window=N` with a specialised window shader
  → measurable decode win on long context. `markov-rs` with Metal
  compressed K/V append → memory win that doesn't cost latency. Apollo
  with pipelined boundary upload → the long-context tok/s number that
  was always there but unreachable.

## 3. Decision

Widen `ComputeBackend` from a flat compute-primitive surface to an
**intent + capability** surface. Engines declare what they need
(attended Q against windowed K/V, recompute K/V from residuals, upload
a boundary residual); backend chooses how (which shader variant, which
pipeline-state cache entry, which command-buffer shape). Pipeline state
becomes a first-class resource the backend owns and caches on engines'
behalf.

The trait is designed against Metal + Vulkan + CPU simultaneously. CUDA
extends the trait with stream/graph-capture variants when added.

## 4. What's in / out of scope

### 4.1 In scope

- `ComputeBackend` trait surface widening (the type signatures and the
  intent vocabulary they express).
- Pipeline state caching as part of the backend (engines never see GPU
  pipeline state objects directly).
- Per-backend capability discovery (an engine asks "do you support
  fused windowed attention?" at build time).
- CPU backend conformance to the new trait (must be a bit-identical
  reference for parity testing).
- The 3D matrix of (engine × backend × model arch) — the spec lays out
  which combinations are first-class supported, which are degraded,
  which are unsupported.
- The migration from today's flat `ComputeBackend` to the new shape, in
  steps that each remain shippable.

### 4.2 Out of scope (deliberately)

- CUDA backend. The trait is designed to admit CUDA streams/graphs as
  a future extension, but no CUDA-specific methods land in this round.
- A specific Vulkan implementation. The Vulkan-side requirements
  inform the trait shape; landing the implementation is a separate work
  item that consumes this spec.
- New compute primitives that don't already have at least one engine
  asking for them. No speculative API surface.
- Async / multi-request dispatch (request-level batching, command
  queue sharing across requests). Single-request decode only; multi-
  request batching is a layer above the backend trait.
- Quantisation format additions. The trait carries quantisation
  through `weights.tensors` as today; new quant tiers (FP4, etc.) go
  through that channel, not the backend trait.

## 5. The intent vocabulary

Engines express compute through *intents* the backend implements per-
substrate. Intents are grouped by category. Each intent has a
correctness contract (what output it produces) and a capability flag
(whether the backend supports it natively or via fallback).

### 5.1 Cache primitives

| Intent | Caller | Why per-backend |
|---|---|---|
| `alloc_kv_buffer(layer, max_tokens, kv_dim) -> KvHandle` | `standard`, `markov-rs`, `turbo-quant`, `unlimited-context` | Metal: pre-allocate `MTLBuffer` with `StorageModePrivate`. Vulkan: `VkBuffer` with `DEVICE_LOCAL`. CPU: `Vec<f32>` allocation. |
| `append_kv(handle, k_row, v_row, position)` | same | GPU: kernel dispatch that writes into pre-allocated buffer at offset. CPU: array slice + memcpy. |
| `clip_kv(handle, window_size)` | `standard:window=N` | GPU: dispatch that shifts entries; CPU: array copy. Backends may no-op if their cache layout supports bounded ring buffers. |
| `read_kv_to_host(handle) -> SharedKV` | fallback paths, debug | GPU: blocking memory copy; CPU: identity. Should not be used in hot loops. |

### 5.2 Attention primitives

| Intent | Caller | Why per-backend |
|---|---|---|
| `attention_step(query, kv_handle, layer, abs_position, attn_mask) -> Array2<f32>` | every engine's decode_step | The whole reason GPUs win. Metal/Vulkan: fused single-kernel attention with K/V from `kv_handle`. CPU: BLAS softmax. |
| `attention_step_windowed(query, kv_handle, layer, abs_position, window) -> Array2<f32>` | `standard:window=N`, future windowed engines | Specialised shader variant — bounded window known at dispatch time enables register-resident K/V on Metal. |
| `attention_prefill(tokens_embedded, layer) -> (last_hidden, KvHandle)` | every engine's prefill | Multi-token attention. Metal: matrix-matrix attention kernel; CPU: BLAS GEMM. |

### 5.3 Engine-specific primitives

| Intent | Caller | Why per-backend |
|---|---|---|
| `recompute_kv_from_residuals(residuals, layer) -> KvHandle` | `markov-rs` | Metal: compute kernel; CPU: BLAS sgemv. The intent is "regenerate K/V for these stored residuals"; backend chooses the kernel. |
| `compressed_kv_append(handle, k, v, codec) -> ()` | `turbo-quant` | Metal: codec kernel (WHT + Lloyd-Max as Metal shader); CPU: vectorised codec. Backends without a native codec fall back to dequant→f32-append→requant. |
| `upload_boundary_residual(residual) -> ResidualHandle` | `apollo` | Metal: shared-mode buffer with first-attention pipelining; Vulkan: explicit barrier; CPU: identity. |
| `forward_from_layer(start_layer, residuals) -> Array2<f32>` | `apollo` compressed path | The "skip first K layers" decode. Backend choice affects whether it pipelines with the boundary upload. |

### 5.4 Norm + residual primitives

| Intent | Caller | Why per-backend |
|---|---|---|
| `rmsnorm(x, weights)` | every engine, every layer | Already in today's `ComputeBackend`. Metal D-RMS-FUSE work targets the fused `residual_add + rmsnorm` variant via this primitive. |
| `residual_norm_store(x, residual, weights, layer)` | every engine after attention/FFN | Fused 3-op intent. CPU may decompose; Metal may have a single kernel. |

### 5.5 Generic compute (kept from today's surface)

`matmul`, `matmul_transb`, `softmax`, `embed_tokens`, `lm_head`. These
are already in `larql_compute::ComputeBackend` and stay there. The
redesign adds the engine-facing intents above; it doesn't remove
existing primitives.

## 6. The trait surface

Working sketch. Names and exact signatures are still negotiable; what's
load-bearing is the *shape*: handle-based cache, intent-based attention,
capability-based feature discovery.

```rust
/// The unified compute surface for inference. Implementations:
///   - `CpuBackend` (always available; reference implementation)
///   - `MetalBackend` (M-series Apple Silicon)
///   - `VulkanBackend` (cross-vendor GPUs)
///   - (future) `CudaBackend` (NVIDIA)
///
/// Engines depend on this trait; the trait depends on `larql-models`
/// and `larql-vindex` for `ModelWeights` / `VectorIndex`.
pub trait ComputeBackend: Send + Sync {
    /// Identity / capability discovery.
    fn name(&self) -> &'static str;
    fn supports(&self, capability: BackendCapability) -> bool;

    // ── Cache primitives ────────────────────────────────────────────
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle;
    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], position: usize);
    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize);
    fn read_kv_to_host(&self, handle: &KvHandle) -> SharedKV;

    // ── Attention primitives ────────────────────────────────────────
    fn attention_step(
        &self,
        query: &Array2<f32>,
        kv: &KvHandle,
        layer: usize,
        abs_position: usize,
    ) -> Array2<f32>;

    /// Optional — backends without windowed shader variants fall back
    /// to `attention_step` with a clipped `KvHandle`. Engines that ask
    /// for windowed attention should still be correct; the win is the
    /// specialised shader when present.
    fn attention_step_windowed(
        &self,
        query: &Array2<f32>,
        kv: &KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
    ) -> Array2<f32> {
        // default: delegate to the unconditional path
        self.attention_step(query, kv, layer, abs_position)
    }

    fn attention_prefill(
        &self,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
    ) -> (Array2<f32>, KvHandle);

    // ── Engine-specific primitives (default-implemented as
    // decompositions; backends override for fused / specialised paths)
    fn recompute_kv_from_residuals(
        &self,
        residuals: &Array2<f32>,
        weights: &ModelWeights,
        layer: usize,
    ) -> KvHandle;

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    );

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> ResidualHandle;

    fn forward_from_layer(
        &self,
        start_layer: usize,
        residuals: &ResidualHandle,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>>;

    // ── Norm primitives ────────────────────────────────────────────
    fn rmsnorm(&self, x: &Array2<f32>, weights: &[f32]) -> Array2<f32>;
    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        weights: &[f32],
    ) -> Array2<f32>;

    // ── Generic compute (existing surface, unchanged) ──────────────
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    fn matmul_transb(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    // ... etc, as today
}

#[derive(Debug, Clone, Copy)]
pub enum BackendCapability {
    /// Backend has a fused single-kernel attention step.
    FusedAttentionStep,
    /// Backend has a specialised windowed-attention shader.
    WindowedAttentionStep,
    /// Backend has a native K/V codec kernel (TurboQuant on GPU).
    NativeKvCodec,
    /// Backend can pipeline boundary upload with first attention dispatch.
    PipelinedBoundaryUpload,
    /// Backend supports fused residual-add + RMSNorm in one kernel.
    FusedResidualNorm,
}
```

### 6.1 KvHandle

```rust
/// Opaque handle to a K/V cache allocation. Layout is backend-specific.
/// Engines treat this as `Send + Sync` data with no observable
/// structure beyond the queries the trait exposes.
pub struct KvHandle {
    inner: Box<dyn KvHandleInner>,
}

trait KvHandleInner: Send + Sync {
    fn shape(&self) -> KvShape;
    fn cached_len(&self) -> usize;
    fn backend_name(&self) -> &'static str;
}
```

Backend implementations of `KvHandleInner`:
- `CpuKvHandle` — wraps an `Array2<f32>` per K and V.
- `MetalKvHandle` — wraps `MTLBuffer` ID + shape metadata.
- `VulkanKvHandle` — wraps `VkBuffer` + memory binding.

The trait doesn't expose the inner type; engines pass handles around
opaquely. `read_kv_to_host` is the explicit cross-backend escape hatch
(blocking copy out to a `SharedKV`).

### 6.2 ResidualHandle

Same pattern as `KvHandle` for boundary residual upload. Owned by the
backend; lifetime tied to the `ComputeBackend` instance.

### 6.3 Capability discovery

Engines query at construction:

```rust
impl MarkovResidualEngine {
    pub fn new(backend: Box<dyn ComputeBackend>, ...) -> Result<Self, EngineBuildError> {
        if !backend.supports(BackendCapability::FusedAttentionStep) {
            // Engine still works, just slower — fallback path.
            log::warn!("backend {} lacks FusedAttentionStep; markov-rs will use decomposed dispatch", backend.name());
        }
        // ...
    }
}
```

Engines do **not** check `backend.name()` to decide behaviour. The
capability flags are the only legitimate gating mechanism. This keeps
new backends (CUDA later) drop-in without engine modifications.

## 7. Backend implementations

### 7.1 `CpuBackend` (reference)

Synchronous. Returns immediately from every method. `KvHandle` wraps
`(Array2<f32>, Array2<f32>)` directly. Capability flags: all `false`
except where the BLAS path is genuinely fused (probably none).

Every other backend's correctness is measured against `CpuBackend`'s
output. Bit-parity is the contract (modulo IEEE-754 reduction-order
caveats already accepted in `larql-compute`).

### 7.2 `MetalBackend`

Owns:
- A `metal::Device` + `metal::CommandQueue`.
- A pipeline-state cache: `HashMap<PipelineKey, MTLComputePipelineState>`.
- A buffer allocator with reuse pools per shape class.

`PipelineKey` includes shader name + specialisation constants (window
size, head dim, dtype). Engine-specific shader variants are
constructed lazily on first use, cached forever (the device's lifetime).

Capabilities probably enabled: `FusedAttentionStep`,
`WindowedAttentionStep`, `FusedResidualNorm`. `NativeKvCodec` and
`PipelinedBoundaryUpload` depend on whether the codec shader / barrier
work has landed yet — gated behind individual capability flags so
engines can fall back cleanly.

### 7.3 `VulkanBackend`

Mirror structure: `VkDevice` + queue, pipeline cache
(`HashMap<PipelineKey, VkPipeline>`), descriptor set allocator. Same
capability bits as Metal in principle; which ones are actually
enabled depends on which kernels have been ported.

Key Vulkan-specific details to validate during implementation (not
this spec):
- Descriptor set layout reuse across engines that share input shape.
- Explicit synchronisation between cache writes and attention reads
  (Metal handles this implicitly via command-buffer ordering).
- SPIR-V module precompilation vs runtime SPIR-V generation.

### 7.4 (Future) `CudaBackend`

Out of scope for this spec, but the trait shape must admit a clean CUDA
add-on. Open question — see §11.4.

## 8. Pipeline state caching

First-class resource owned by the backend. Engines don't see it.

```rust
struct PipelineCache {
    // Per-backend type, e.g. HashMap<PipelineKey, MTLComputePipelineState>
}

impl PipelineCache {
    fn get_or_build(&mut self, key: PipelineKey) -> &Pipeline;
}
```

The cache is constructed at `Backend::new()` and lives as long as the
backend instance. Engines that ask for a windowed-attention shader with
`window=4` populate the cache with one entry on first call; subsequent
calls at the same window size hit the cached pipeline.

Specialisation constants in `PipelineKey` (Metal function constants /
Vulkan specialisation info) let the compiler optimise per-engine
choices without forcing the engine to know about pipeline state at all.

Eviction policy: not specified by this trait. Backends may choose LRU,
unbounded, or pre-warmed-fixed-set. The contract is "engines can ask
for any valid shape; backend will eventually produce a pipeline."

## 9. Engine × backend compatibility matrix

Each engine declares its required capabilities at build time. Backend
build returns an error if required capabilities aren't met (rather than
silently downgrading at decode time).

| Engine | Required | Optional (uses if available) |
|---|---|---|
| `Standard` | (none — works on any backend) | `FusedAttentionStep` |
| `Standard:window=N` | (none) | `WindowedAttentionStep` |
| `NoCache` | (none) | — |
| `MarkovResidual` | (none) | `FusedAttentionStep` |
| `UnlimitedContext` | (none) | `FusedAttentionStep` |
| `TurboQuant` | (none) | `NativeKvCodec` |
| `Apollo` | (none) | `PipelinedBoundaryUpload` |

No engine *requires* a GPU capability — all engines work on CPU as the
baseline. Capabilities accelerate; absence degrades but doesn't break.
This keeps the test matrix tractable: every engine × every backend ×
every architecture combination is a valid configuration.

The exception: architectures with model-specific requirements
(Gemma 4 E2B per-layer embeddings → D-METAL-PLE) intersect with the
backend choice. Those are arch-side concerns handled in
`ModelArchitecture`, not engine-side or backend-side. Backend simply
asks the arch "do you need per-layer-embedding compute?" and dispatches
the corresponding intent.

## 10. Migration plan

Each step is independently shippable. Each step has a parity guarantee
so the default user experience doesn't regress.

### 10.1 Step 1 — Lock the trait shape

This spec. Lock the trait surface against Metal + Vulkan + CPU on
paper. No code yet. Output: the spec doc reviewed and signed off.

**Parity:** N/A — design only.

### 10.2 Step 2 — `KvDispatch` sibling trait + `CpuBackend` conformance

**Refined 2026-05-16, post-implementation finding:** the original
sketch (KvDispatch as a `ComputeBackend` sub-trait in `larql-compute`)
doesn't survive the CPU impl gate. CpuBackend lives in `larql-compute`;
the CPU forward-pass functions its KvDispatch impl needs to call
(`run_attention_*`, `run_ffn`, residual ops) live in `larql-inference`.
Both directions are blocked:
- `larql-compute` can't depend on `larql-inference` (cycle).
- Orphan rules forbid `impl KvDispatch for CpuBackend` in
  `larql-inference` when both trait and type are foreign.

**Correct shape:** `KvDispatch` lives in **`larql-inference`** as a
*sibling* to `ComputeBackend` and `FfnBackend`. CpuBackend +
MetalBackend impls live in `larql-inference` too (orphan rules
satisfied because the trait is local). Substrate `Capability` flags
stay in `larql-compute` — they describe what the substrate supports,
independent of where the dispatch trait lives.

Engines hold three composable abstractions:
- `&dyn larql_compute::ComputeBackend` — kernel primitives + capability probe
- `&dyn larql_inference::KvDispatch` — engine-facing intents
- `&dyn larql_inference::FfnBackend` — FFN routing

Sub-step **2a (trait skeleton):** ✅ landed 2026-05-16 (after one
reverted attempt — see history note above).
- `crates/larql-inference/src/kv_dispatch/mod.rs` — `KvDispatch`,
  `KvHandle`, `ResidualHandle`, `CompressionCodec`, plus
  `KvHandleInner` / `ResidualHandleInner` for backend-side allocation.
  Backend impls live in sibling submodules: `cpu.rs`, `metal.rs`
  (`#[cfg(feature = "metal")]`), and the per-layer prefill/decode
  drivers in `helpers.rs`.
- 6 new `Capability` variants in `crates/larql-compute/src/backend/capability.rs`
  (`FusedAttentionStep`, `WindowedAttentionStep`, `NativeKvCodec`,
  `PipelinedBoundaryUpload`, `FusedResidualNorm`, `KvHandleNative`).
- `larql-inference/src/lib.rs` re-exports the trait surface at the
  crate root.

Sub-step **2b (CPU impl):** Implement `KvDispatch` for
`larql_compute::CpuBackend` inside `larql-inference`. Bodies call the
existing CPU forward-pass functions (`run_attention_*`, `run_ffn`,
residual ops). `KvHandleInner` wraps `larql_inference::attention::KvCache`.

Sub-step **2c (parity):** Verify `CpuBackend`'s `KvDispatch` output
matches the legacy function output bit-for-bit on the synthetic test
fixtures. Engines do NOT migrate to the new trait yet — they keep
dispatching through the current code path. The new conformance is
verified out-of-band.

**Parity:** Existing parity tests must still pass byte-for-byte
(unchanged code path). New parity tests verify `CpuBackend::KvDispatch`
output matches the legacy function output bit-for-bit.

### 10.3 Step 3 — Migrate engines to dispatch through the trait

`Standard`, `NoCache`, `MarkovResidual`, `UnlimitedContext`,
`TurboQuant`, `Apollo` each migrate to call the new trait methods
instead of the legacy functions. CPU backend underneath; no GPU work
yet.

Migration order: smallest first (`NoCache`, `Standard`), then larger
(`MarkovResidual`, `UnlimitedContext`), then most complex (`Apollo`).

**Parity:** Existing engine tests + bit-parity tests from
`kv-engine-unification.md` §8.4 must still pass byte-for-byte.

### 10.4 Step 4 — `MetalBackend`: scaffolding

Implement `MetalBackend` with `CpuBackend`'s behaviour wrapped (every
method copies to host, runs CPU, copies back). No real GPU compute
yet — this exercises the trait shape against actual Metal types
(`MTLDevice`, `MTLBuffer`, `KvHandle::MetalKvHandle`) before any
performance work.

**Parity:** Same output as `CpuBackend`; tok/s catastrophically worse
(every call has a host roundtrip). Acceptance criterion is correctness,
not speed.

### 10.5 Step 5 — `MetalBackend`: real kernels

**Blocked on `async-compute-backend.md` Step A4.** Step-5 work
discovered that the synchronous `KvDispatch` trait can't deliver
Metal-class performance at per-layer granularity — each per-layer
call would force a separate command-buffer commit, slower than
today's fused decode path. The async trait is the prerequisite.

Once `AsyncComputeBackend` Step A4 lands (deferred dispatch on
Metal), this step picks back up: replace
`AsyncComputeBackend::attention_step_async` etc. with real Metal
shader kernels, one at a time. Start with `attention_step_windowed`
(the `standard:window=N` win). Then engine-specific primitives in
priority order.

Each kernel landed is paired with a bench that measures the win on a
real model. Wins must be empirically confirmed end-to-end, not just at
the kernel level (per
[`feedback_isolated_vs_batched_kernel_profile`]).

**Parity:** Each kernel must match `CpuBackend`'s output within
declared numerical tolerance. Tolerance is per-primitive — `matmul`
gets f32 reduction-order tolerance; `softmax` gets bit-parity.

### 10.6 Step 6 — `VulkanBackend`: scaffolding + kernels

Same shape as Steps 4-5 for Vulkan. The Metal kernels from Step 5
inform what shaders to port (and where the abstractions held vs leaked).

If a Vulkan-side gap surfaces ("our trait doesn't express X cleanly"),
the trait revises — Step 6 is when the cross-backend shape gets
validated. If revisions ripple back into Metal, that's expected; better
than discovering the gap after CUDA bring-up.

**Parity:** `VulkanBackend` output matches `CpuBackend` within
tolerance; tok/s on Vulkan-supported hardware (NVIDIA via Vulkan, AMD,
Intel Arc) within X% of Metal on equivalent hardware (X is TBD per
hardware class).

### 10.7 Step 7 — Server wiring

`larql-server`'s `handle_stream_generate` switches from
`generate_streaming` to `generate_with_engine` against a
`ComputeBackend` selected per-request (env var, header, or process-wide
default). `LARQL_KV_ENGINE` finally honoured by the server.

This is the deferred half of the KV-engine unification spec
[`kv-engine-unification.md`](./kv-engine-unification.md) §10.6 — it
lands here, after the GPU side of `KvEngine` exists.

**Parity:** Server's tok/s on the existing `generate_streaming` path
versus the new `generate_with_engine + MetalBackend + Standard` path
must be within Y% (Y is TBD — probably ≤5% as the gate).

## 11. Open questions

### 11.1 Should `ComputeBackend` own the `FfnBackend` dispatch?

**Resolved 2026-05-16:** keep separate. FFN routing is a *network
topology* concern (remote shards, MoE expert dispatch); compute backend
is a *substrate* concern (which silicon runs the math). They compose
orthogonally. Engine API stays at `(&dyn ComputeBackend, &dyn FfnBackend)`.

### 11.2 Synchronous return vs handle-future?

**Resolved 2026-05-16:** synchronous return. Metal + Vulkan can both
submit-and-wait per call; CPU is trivially synchronous. CUDA when it
lands gets a sibling trait (`AsyncComputeBackend`) — see §11.4. Don't
pay the abstraction tax of async return when no caller needs it yet.

### 11.3 Per-request backend vs process-wide backend?

**Resolved 2026-05-16:** process-wide for v1. Per-request routing is a
scheduler concern, not a trait concern. The trait already supports it
(engines take `&dyn ComputeBackend`, which a scheduler can swap per
request), but no work in this spec mandates the scheduler. Deferred to
a future "request routing" spec.

### 11.4 CUDA shape commitments

**Resolved 2026-05-16, revised 2026-05-16:** the original resolution
(defer `AsyncComputeBackend` until CUDA bring-up) was overtaken by a
Step-5 finding: per-layer Metal kernels at the synchronous trait's
granularity are *slower* than today's fused Metal path because each
per-layer call forces a GPU command-buffer commit. The async surface
is therefore needed earlier than originally planned — for Metal +
Vulkan, not just CUDA.

**Revised plan:** `AsyncComputeBackend` becomes a sibling trait now
(not deferred to CUDA). Its design is owned by
[`async-compute-backend.md`](./async-compute-backend.md) — a full
spec covering the intent-collector pattern, handle types, commit
semantics, per-backend implementation contracts (Metal, Vulkan, CUDA,
CPU), and a multi-step migration plan (~6-12 months end-to-end).

This redesign spec's Step 5 (real Metal kernels) is therefore
*blocked* on `async-compute-backend.md` Step A4 (deferred dispatch
landing on Metal). Steps 1-4 of *this* spec remain shippable as the
unification PR; the per-layer-Metal-kernels effort starts after this
spec is signed off and `async-compute-backend.md` is accepted.

**Progress (2026-05-16):** `async-compute-backend.md` steps A1 (trait
+ handles), A2 (`CpuBackend` async impl), A3 (`MetalBackend`
scaffold), and A5 (`StandardEngine` opt-in) have landed. A4 — real
Metal deferred dispatch — is the next gate before Step 5 of this
spec's per-layer Metal kernels can begin.

### 11.5 Quantisation as a backend concern?

**Resolved 2026-05-16:** quantisation stays internal to the backend
impl. The new trait's intent vocabulary is uniform (`attention_step`,
`matmul`); backends route to f32 or Q4_K paths internally based on
tensor type from `weights.tensors` / `VectorIndex`.

**Implication for `KvEngine`:** the current `prefill_q4k` /
`decode_step_q4k` engine-trait split collapses once engines migrate to
dispatch through `ComputeBackend`. The Q4K-vs-f32 choice is no longer
visible at the engine API; it's a backend-internal routing decision.
This simplifies the `KvEngine` trait further — `prefill` /
`decode_step` are the only two methods, and Q4K is just "the backend
loaded a Q4K vindex and dispatches accordingly."

Engine migration in §10.3 should drop the `*_q4k` methods.

### 11.6 Where does this spec live?

**Resolved 2026-05-16:** stays in `larql-inference/docs/specs/`
alongside the unification spec. The trait lives in `larql-compute`
but the *audience* for this spec — engine authors, backend
implementers, server integrators — all read inference specs. Cross-
crate spec location is a documentation concern, not an architectural
one.

## 12. Non-goals

- **Not a kernel improvement project.** This spec doesn't speed up any
  kernel. It defines the surface that kernel work targets. Tok/s wins
  are out of scope until Step 5+.
- **Not a multi-request scheduler.** Single-request decode only.
- **Not a heterogeneous-compute orchestrator.** No "use Metal for
  attention, Neural Engine for FFN" routing. One backend per session.
- **Not a quant-format expansion.** New quant tiers go through
  existing tensor-metadata channels.
- **Not a server-side concurrency redesign.** `larql-server` keeps
  its existing request handling; only the decode-path call changes
  (Step 7).

## 13. Acceptance criteria

This spec is accepted (signed off, ready to implement Step 2) when:

1. The §5 intent vocabulary covers every primitive every shipped
   engine needs. Anyone reading the table can map each
   `KvEngine::{prefill, decode_step}` implementation to a sequence of
   trait calls.
2. The §6 trait surface compiles in principle — the signatures
   type-check against `larql_models::ModelWeights` and
   `larql_vindex::VectorIndex` (verified by writing a stub
   `CpuBackend` impl that satisfies the trait but `unimplemented!()`s
   every method).
3. The §9 compatibility matrix has been reviewed against the actual
   engine code — no engine has a hidden dependency that isn't in the
   table.
4. The §10 migration plan is implementable without backwards-
   incompatible API breaks for at least Steps 2-4. Steps 5-7 may
   introduce breaks if needed but with a deprecation path.
5. The §11 open questions are resolved or explicitly deferred with a
   named owner and deadline.

---

## Appendix A — Relationship to other specs

- [`kv-engine-unification.md`](./kv-engine-unification.md): owns the
  `KvEngine` trait. Steps 1-7 of that spec landed (modulo the server-
  wiring deferral in §10.6). This redesign is what lets the server
  wiring land.
- [`markov-residual-engine.md`](./markov-residual-engine.md): owns the
  correctness contract for `MarkovResidualEngine`. The redesign must
  preserve it — backends ported to the new trait must still hit KL=0
  vs the f32 reference.

## Appendix B — Concrete file pointers

For reviewers, the existing code this spec touches:

- Today's `ComputeBackend` trait:
  `crates/larql-compute/src/lib.rs` (search `pub trait ComputeBackend`)
- Today's CPU backend:
  `crates/larql-compute/src/cpu/`
- Today's Metal backend:
  `crates/larql-compute/src/metal/`
- The KvEngine trait (post-unification):
  `crates/larql-inference/src/kv_engine.rs`
- The engine dispatch entry point:
  `crates/larql-inference/src/forward/kv_generate.rs` (`generate_with_engine`)
- The forward-pass helpers engines currently call directly (and will
  call via the trait post-migration):
  `crates/larql-inference/src/{attention,ffn,forward,residual}/`
