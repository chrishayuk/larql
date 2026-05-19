//! `AsyncComputeBackend` — deferred-dispatch sibling to [`KvDispatch`].
//!
//! See `docs/specs/async-compute-backend.md` for the full design.
//!
//! ## Why a separate trait
//!
//! [`KvDispatch`] is synchronous: every intent commits + waits on the GPU
//! (or completes on the CPU) before returning. That shape is correct but
//! defeats command-buffer batching on Metal / Vulkan / CUDA — one sync
//! per per-layer intent is orders of magnitude slower than today's fused
//! `decode_token` path. `AsyncComputeBackend` lets backends *accumulate*
//! per-layer intents into one in-flight command buffer per decode session
//! and only block on read-back. Engines opt in by constructing themselves
//! with an [`AsyncComputeBackend`] (e.g. `StandardEngine::with_async_backend`).
//!
//! Both traits coexist permanently. `AsyncComputeBackend` is the
//! performance path; [`KvDispatch`] stays the correctness reference.
//!
//! ## Handle ownership: spec deviation
//!
//! The spec sketches `Arc<dyn AsyncHandleInner<Output = T>>` with
//! `read(self: Arc<Self>)`. On stable Rust that combination requires
//! `#![feature(arbitrary_self_types)]`. The stable-Rust translation
//! adopted here is one inner trait per handle type with
//! `read(self: Box<Self>) -> Output`, which is object-safe and consumes
//! the handle on read exactly as the spec intends. The spec's
//! idempotency requirement ("calling `read` on a handle whose backend
//! has already committed returns immediately") survives unchanged —
//! backends record commit state on shared interior state, not on the
//! handle.
//!
//! ## Module layout
//!
//! - [`mod@self`] (this file) — trait surface, handle types, `Ready*`
//!   helpers, error type.
//! - [`cpu`] — `CpuBackend` async impl (degenerate `Ready*` wrapper,
//!   parity reference).
//! - [`metal`] — `MetalBackend` async scaffold (feature-gated behind
//!   `metal`; A3 delegates to CPU at present, real deferred dispatch
//!   lands in A4).
//! - Future siblings: `vulkan`, `cuda` (per spec §7.2, §7.3).
//!
//! ## Status
//!
//! Step A1 of the migration: trait + handle types + `Ready*` helpers.
//! A2 (`CpuBackend`) and A3 (`MetalBackend`) scaffolding shipped
//! 2026-05-16 — see the spec's migration plan §10 for the full
//! sequencing.

pub mod cpu;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

use crate::ffn::FfnBackend;
use crate::kv_dispatch::{KvDispatch, KvHandle, ResidualHandle};
use crate::model::ModelWeights;
use ndarray::Array2;
use std::error::Error;
use std::fmt;

// ── Handle types ─────────────────────────────────────────────────────

/// Pending result from an async attention dispatch — placeholder for a
/// hidden state that will exist once the backend commits its in-flight
/// command buffer.
///
/// Engines compose `AttentionHandle`s without blocking. Reading via
/// [`AttentionHandle::read`] (or
/// [`AsyncComputeBackend::read_hidden`]) triggers commit + wait.
pub struct AttentionHandle {
    inner: Box<dyn AttentionHandleInner>,
}

impl AttentionHandle {
    /// Construct from a backend-specific inner. Backend implementations
    /// call this; engines never do.
    pub fn new<I: AttentionHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Wrap an already-computed hidden state as a handle. Used by
    /// [`CpuBackend`-style degenerate `AsyncComputeBackend` impls](crate::async_compute_backend),
    /// and by bench/test scaffolding.
    pub fn ready(value: Array2<f32>) -> Self {
        Self::new(ReadyAttention(value))
    }

    /// Non-blocking completion check. Returns `true` if the backend's
    /// command buffer covering this handle has been committed AND
    /// completed.
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Read the hidden state. Blocks on commit + wait if the backend
    /// has not yet completed this handle's work. Consumes the handle.
    pub fn read(self) -> Array2<f32> {
        self.inner.read()
    }
}

/// Backend-side trait for [`AttentionHandle`] inner types. Backends
/// implement this on their per-platform handle types
/// (`MetalAttentionHandle`, `VulkanAttentionHandle`, etc.).
///
/// `Send` is required so handles can move between threads (e.g.
/// engine-decode-loop thread to bench harness). `Sync` is intentionally
/// NOT required — backends may hold non-`Sync` GPU primitives inside.
pub trait AttentionHandleInner: Send {
    /// Non-blocking completion probe.
    fn is_complete(&self) -> bool;

    /// Block on commit + wait, return the hidden state. Consumes the
    /// boxed inner so each handle is read at most once.
    ///
    /// Implementations must be idempotent on the *backend* side —
    /// calling `read` on a handle whose backend has already committed
    /// (because another handle from the same batch was read) returns
    /// immediately without forcing a second commit.
    fn read(self: Box<Self>) -> Array2<f32>;
}

/// Pending result from an async residual-upload dispatch. The upload
/// produces no host-visible value; reading it just blocks until the
/// backend has accepted the upload into its in-flight command buffer.
pub struct ResidualUploadHandle {
    inner: Box<dyn ResidualUploadHandleInner>,
}

impl ResidualUploadHandle {
    pub fn new<I: ResidualUploadHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Wrap an already-completed upload (degenerate CPU impl / test
    /// scaffolding).
    pub fn ready() -> Self {
        Self::new(ReadyResidualUpload)
    }

    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Block on commit + wait. Consumes the handle.
    pub fn read(self) {
        self.inner.read();
    }
}

pub trait ResidualUploadHandleInner: Send {
    fn is_complete(&self) -> bool;
    fn read(self: Box<Self>);
}

// ── Ready* helpers ───────────────────────────────────────────────────

/// `AttentionHandleInner` impl wrapping an already-computed value.
/// Used by degenerate `AsyncComputeBackend` impls (CpuBackend) and by
/// scaffolding stages of the Metal migration where async methods
/// internally call sync `KvDispatch` and wrap the result.
pub struct ReadyAttention(pub Array2<f32>);

impl AttentionHandleInner for ReadyAttention {
    fn is_complete(&self) -> bool {
        true
    }

    fn read(self: Box<Self>) -> Array2<f32> {
        self.0
    }
}

/// `ResidualUploadHandleInner` impl for an already-completed upload.
pub struct ReadyResidualUpload;

impl ResidualUploadHandleInner for ReadyResidualUpload {
    fn is_complete(&self) -> bool {
        true
    }

    fn read(self: Box<Self>) {}
}

// ── Errors ────────────────────────────────────────────────────────────

/// Errors from the deferred-dispatch surface.
#[derive(Debug)]
pub enum AsyncDispatchError {
    /// GPU device removed, unresponsive, or otherwise unavailable.
    DeviceError(String),
    /// Command buffer rejected at commit time. Typically a backend bug
    /// — an encoded operation that the runtime refused.
    CommandBufferRejected(String),
    /// Backend-specific failure not covered by the variants above.
    Other(String),
}

impl fmt::Display for AsyncDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsyncDispatchError::DeviceError(s) => write!(f, "async dispatch device error: {s}"),
            AsyncDispatchError::CommandBufferRejected(s) => {
                write!(f, "async dispatch command buffer rejected: {s}")
            }
            AsyncDispatchError::Other(s) => write!(f, "async dispatch error: {s}"),
        }
    }
}

impl Error for AsyncDispatchError {}

// ── The trait ────────────────────────────────────────────────────────

/// Async / batched dispatch surface — sibling to [`KvDispatch`].
///
/// Implementers maintain an in-flight command buffer (or equivalent on
/// non-Metal backends) per session. Each async method *encodes* an
/// intent into that buffer and returns a handle. The buffer is
/// committed on:
///
/// - explicit [`flush`](Self::flush) call;
/// - first [`read_hidden`](Self::read_hidden) (or other handle read);
/// - backend-internal trigger (buffer overflow — implementation choice;
///   see spec §11.1 for default thresholds).
///
/// Engines opt in by constructing themselves with an
/// `AsyncComputeBackend` (e.g. `StandardEngine::with_async_backend`).
/// Engines that don't opt in stay on synchronous [`KvDispatch`].
///
/// Supertraits `ComputeBackend + KvDispatch` so an `AsyncComputeBackend`
/// is also a valid [`EngineBackend`](crate::EngineBackend). Implementers
/// get the sync trait for free by wrapping each async method's body in
/// a `Ready*` helper; they override per-intent as real deferred dispatch
/// lands (Step A4).
///
/// ## Thread-safety
///
/// `Send` only — see spec §11.4. A backend instance moves between
/// threads freely, but is **not** shared concurrently. Servers handling
/// concurrent requests construct one `AsyncComputeBackend` per request
/// handler. Cross-request batching (which would require `Sync`) is
/// deferred to a future scheduler layer above this trait.
///
/// ## Defaults
///
/// Every intent method has an `unimplemented!()` default mirroring
/// [`KvDispatch`]'s pattern. The commit-control methods
/// ([`flush`](Self::flush), [`read_hidden`](Self::read_hidden),
/// [`has_pending_work`](Self::has_pending_work)) have meaningful
/// defaults that are correct for any backend whose handles already
/// carry their own commit state (e.g. degenerate `Ready*`-wrapped
/// CpuBackend). Backends with real deferred dispatch override them.
pub trait AsyncComputeBackend: larql_compute::ComputeBackend + KvDispatch + Send {
    // ── Commit / flush control ──────────────────────────────────────

    /// Commit the in-flight command buffer (if any) and wait for GPU
    /// completion. Engines call this at decode-step boundaries to bound
    /// the dispatch cadence at one GPU sync per token.
    ///
    /// Default: `Ok(())`. Correct for backends without deferred state
    /// (e.g. `Ready*`-wrapped CpuBackend); overridden by real GPU
    /// backends.
    fn flush(&self) -> Result<(), AsyncDispatchError> {
        Ok(())
    }

    /// Read the hidden state from an [`AttentionHandle`]. Triggers
    /// commit + wait if the handle is not already complete. Consumes
    /// the handle.
    ///
    /// Default: delegate to [`AttentionHandle::read`]. Correct when the
    /// backend's commit state lives on the inner handle (the `Ready*`
    /// path). Backends with shared command-buffer state override to
    /// commit-once-then-read-all.
    fn read_hidden(&self, handle: AttentionHandle) -> Array2<f32> {
        handle.read()
    }

    /// Non-blocking diagnostic: is the backend currently holding an
    /// uncommitted command buffer? Used for bench instrumentation and
    /// to validate that engines are batching effectively.
    ///
    /// Default: `false`. Correct for backends without deferred state.
    fn has_pending_work(&self) -> bool {
        false
    }

    // ── Async intents (mirror KvDispatch with handle returns) ───────

    /// Async equivalent of [`KvDispatch::attention_step`]. Encodes a
    /// per-layer attention step into the in-flight command buffer and
    /// returns a handle for the post-attention hidden state. The
    /// `KvHandle` is mutated to reflect the queued K/V append; queries
    /// on it follow spec §11.2. `index` follows the convention from
    /// [`KvDispatch::attention_step`].
    fn attention_step_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> AttentionHandle {
        let _ = (weights, query, kv, layer, abs_position, index);
        unimplemented!("attention_step_async not implemented for this backend")
    }

    /// Async equivalent of [`KvDispatch::attention_step_windowed`].
    ///
    /// Default decomposition: dispatch the unbounded variant, then
    /// [`KvDispatch::clip_kv`] the cache to `window` rows. Backends with
    /// a fused windowed-attention shader (Step A6) override.
    #[allow(clippy::too_many_arguments)]
    fn attention_step_windowed_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> AttentionHandle {
        let handle = self.attention_step_async(weights, query, kv, layer, abs_position, index);
        self.clip_kv(kv, window);
        handle
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
        index: Option<&larql_vindex::VectorIndex>,
    ) -> (AttentionHandle, KvHandle) {
        let _ = (weights, tokens_embedded, layer, window, index);
        unimplemented!("attention_prefill_async not implemented for this backend")
    }

    /// Async equivalent of [`KvDispatch::recompute_kv_from_residuals`].
    fn recompute_kv_from_residuals_async(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> KvHandle {
        let _ = (weights, residuals, layer);
        unimplemented!("recompute_kv_from_residuals_async not implemented for this backend")
    }

    /// Async equivalent of [`KvDispatch::upload_boundary_residual`].
    ///
    /// The returned `ResidualUploadHandle` is pending until commit;
    /// subsequent `forward_from_layer_async` calls referencing the
    /// resulting `ResidualHandle` can fuse with the upload in the same
    /// command buffer (Apollo's pipelined-upload win).
    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle) {
        let _ = residual;
        unimplemented!("upload_boundary_residual_async not implemented for this backend")
    }

    /// Async equivalent of [`KvDispatch::forward_from_layer`].
    fn forward_from_layer_async(
        &self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle {
        let _ = (weights, ffn, start_layer, residuals, token_ids);
        unimplemented!("forward_from_layer_async not implemented for this backend")
    }
}

#[cfg(test)]
mod tests {
    //! Trait-default contract tests.
    //!
    //! `AsyncComputeBackend` supertraits `ComputeBackend + KvDispatch +
    //! Send`. To exercise the `unimplemented!()` intent-method defaults
    //! we build a stub that satisfies the supertraits with panicking
    //! bodies (every supertrait method also `unimplemented!()`) but
    //! overrides none of the async intents — so when a test calls an
    //! intent on the stub, the default body runs and panics with the
    //! documented "not implemented for this backend" message.
    //!
    //! The supertrait methods themselves are never called from these
    //! tests; they exist purely to satisfy the type system. Their
    //! `unimplemented!()` bodies are NOT exercised here.
    //!
    //! Coverage role: every `unimplemented!()` intent default body in
    //! the parent module gets reached, so `mod.rs` lines coverage
    //! crosses 90%.

    use super::*;
    use crate::kv_dispatch::{
        KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner,
    };
    use ndarray::{array, ArrayView2};

    // ── Supertrait-satisfying stub ───────────────────────────────────

    struct StubAsyncBackend;

    // Only `MatMul::matmul` and `MatMul::matmul_transb` are required on
    // the supertraits (no defaults). Everything else on `QuantMatVec`,
    // `DecodeBackend`, and `ComputeBackend` either has a default or is
    // a simple required hook. Stubbing the minimal surface keeps this
    // test module's coverage footprint tight.
    impl larql_compute::MatMul for StubAsyncBackend {
        fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul methods are never invoked from tests")
        }
        fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul methods are never invoked from tests")
        }
    }

    impl larql_compute::QuantMatVec for StubAsyncBackend {}
    impl larql_compute::DecodeBackend for StubAsyncBackend {}

    impl larql_compute::ComputeBackend for StubAsyncBackend {
        fn name(&self) -> &str {
            "stub-async"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl KvDispatch for StubAsyncBackend {}

    impl AsyncComputeBackend for StubAsyncBackend {}

    // ── Tests: async-intent `unimplemented!()` defaults ──────────────

    #[test]
    #[should_panic(expected = "attention_step_async not implemented")]
    fn default_attention_step_async_panics() {
        let backend = StubAsyncBackend;
        let weights = crate::test_utils::make_test_weights();
        let mut kv = KvHandle::new(StubKvInner {
            len: 0,
            dim: weights.hidden_size,
        });
        let query = Array2::zeros((1, weights.hidden_size));
        let _ = backend.attention_step_async(&weights, &query, &mut kv, 0, 0, None);
    }

    #[test]
    #[should_panic(expected = "attention_prefill_async not implemented")]
    fn default_attention_prefill_async_panics() {
        let backend = StubAsyncBackend;
        let weights = crate::test_utils::make_test_weights();
        let tokens = Array2::zeros((2, weights.hidden_size));
        let _ = backend.attention_prefill_async(&weights, &tokens, 0, None, None);
    }

    #[test]
    #[should_panic(expected = "upload_boundary_residual_async not implemented")]
    fn default_upload_boundary_residual_async_panics() {
        let backend = StubAsyncBackend;
        let residual = Array2::zeros((1, 8));
        let _ = backend.upload_boundary_residual_async(&residual);
    }

    #[test]
    #[should_panic(expected = "forward_from_layer_async not implemented")]
    fn default_forward_from_layer_async_panics() {
        let backend = StubAsyncBackend;
        let weights = crate::test_utils::make_test_weights();
        let residuals = ResidualHandle::new(StubResidualInner {
            shape: (1, weights.hidden_size),
        });
        let ffn = crate::ffn::NullFfn;
        let _ = backend.forward_from_layer_async(&weights, &ffn, 0, &residuals, &[0u32]);
    }

    #[test]
    fn default_attention_step_windowed_async_decomposes_via_step() {
        // The trait default delegates to `attention_step_async` (which
        // panics on the stub). Documents the decomposition shape.
        let backend = StubAsyncBackend;
        let weights = crate::test_utils::make_test_weights();
        let mut kv = KvHandle::new(StubKvInner {
            len: 0,
            dim: weights.hidden_size,
        });
        let query = Array2::zeros((1, weights.hidden_size));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = backend.attention_step_windowed_async(&weights, &query, &mut kv, 0, 0, 4, None);
        }));
        let err = result.expect_err("should panic via attention_step_async");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("attention_step_async not implemented"),
            "expected panic from underlying attention_step_async, got: {msg}"
        );
    }

    #[test]
    fn default_flush_returns_ok() {
        let backend = StubAsyncBackend;
        backend.flush().expect("default flush is Ok");
    }

    #[test]
    fn default_has_pending_work_false() {
        let backend = StubAsyncBackend;
        assert!(!backend.has_pending_work());
    }

    #[test]
    fn default_read_hidden_delegates_to_handle_read() {
        let backend = StubAsyncBackend;
        let value = array![[7.0_f32, 8.0]];
        let handle = AttentionHandle::ready(value.clone());
        // `read_hidden` default is `handle.read()`.
        let read = backend.read_hidden(handle);
        assert_eq!(read, value);
    }

    // ── Stub plumbing coverage (exercise the supertrait shims + inners
    //    so they aren't dead lines in this test module's profile) ─────

    #[test]
    fn stub_backend_compute_methods() {
        let backend = StubAsyncBackend;
        assert_eq!(
            <StubAsyncBackend as larql_compute::ComputeBackend>::name(&backend),
            "stub-async"
        );
        let any = <StubAsyncBackend as larql_compute::ComputeBackend>::as_any(&backend);
        assert!(any.downcast_ref::<StubAsyncBackend>().is_some());
    }

    #[test]
    fn stub_kv_inner_accessors() {
        let mut handle = KvHandle::new(StubKvInner { len: 3, dim: 16 });
        assert_eq!(handle.cached_len(), 3);
        assert_eq!(handle.kv_dim(), 16);
        assert_eq!(handle.backend_name(), "stub");
        let _: &dyn KvHandleInner = handle.as_inner();
        let _: &mut dyn KvHandleInner = handle.as_inner_mut();
    }

    #[test]
    fn stub_residual_inner_accessors() {
        let handle = ResidualHandle::new(StubResidualInner { shape: (2, 5) });
        assert_eq!(handle.shape(), (2, 5));
        assert_eq!(handle.backend_name(), "stub");
        let _: &dyn ResidualHandleInner = handle.as_inner();
    }

    // ── Stub handle inners (needed for the panic tests) ──────────────

    struct StubKvInner {
        len: usize,
        dim: usize,
    }
    impl KvHandleInner for StubKvInner {
        fn cached_len(&self) -> usize {
            self.len
        }
        fn kv_dim(&self) -> usize {
            self.dim
        }
        fn backend_name(&self) -> &'static str {
            "stub"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    struct StubResidualInner {
        shape: (usize, usize),
    }
    impl ResidualHandleInner for StubResidualInner {
        fn shape(&self) -> (usize, usize) {
            self.shape
        }
        fn backend_name(&self) -> &'static str {
            "stub"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // ── Existing handle/error/Ready-helper tests ─────────────────────

    #[test]
    fn ready_attention_handle_round_trip() {
        let value = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let handle = AttentionHandle::ready(value.clone());
        assert!(handle.is_complete());
        assert_eq!(handle.read(), value);
    }

    #[test]
    fn ready_residual_upload_round_trip() {
        let handle = ResidualUploadHandle::ready();
        assert!(handle.is_complete());
        handle.read();
    }

    #[test]
    fn async_dispatch_error_display() {
        let err = AsyncDispatchError::DeviceError("metal device removed".into());
        let s = format!("{err}");
        assert!(s.contains("device error"));
        assert!(s.contains("metal device removed"));
    }

    #[test]
    fn async_dispatch_error_display_command_buffer_rejected() {
        let err = AsyncDispatchError::CommandBufferRejected("encoder out of memory".into());
        let s = format!("{err}");
        assert!(s.contains("command buffer rejected"));
        assert!(s.contains("encoder out of memory"));
    }

    #[test]
    fn async_dispatch_error_display_other() {
        let err = AsyncDispatchError::Other("unspecified failure".into());
        let s = format!("{err}");
        assert!(s.contains("async dispatch error"));
        assert!(s.contains("unspecified failure"));
    }

    #[test]
    fn async_dispatch_error_is_std_error() {
        // Compile-time check that the error type satisfies std::error::Error
        // so it composes with `Box<dyn Error>` / `anyhow::Result` callers.
        fn assert_error<E: std::error::Error>() {}
        assert_error::<AsyncDispatchError>();
    }

    #[test]
    fn attention_handle_accepts_custom_inner() {
        // Backend-side API: backends construct `AttentionHandle` from
        // their own inner type, not via `ready()`. Exercise that path
        // with a stub that toggles `is_complete` based on internal state.
        struct StubInner {
            value: Array2<f32>,
            done: bool,
        }
        impl AttentionHandleInner for StubInner {
            fn is_complete(&self) -> bool {
                self.done
            }
            fn read(self: Box<Self>) -> Array2<f32> {
                self.value
            }
        }
        let value = array![[42.0_f32, 1.0]];
        let handle = AttentionHandle::new(StubInner {
            value: value.clone(),
            done: false,
        });
        assert!(!handle.is_complete(), "custom inner reports pending");
        assert_eq!(handle.read(), value);
    }

    #[test]
    fn residual_upload_handle_accepts_custom_inner() {
        struct StubInner {
            done: bool,
        }
        impl ResidualUploadHandleInner for StubInner {
            fn is_complete(&self) -> bool {
                self.done
            }
            fn read(self: Box<Self>) {}
        }
        let handle = ResidualUploadHandle::new(StubInner { done: true });
        assert!(handle.is_complete());
        handle.read();
    }
}
