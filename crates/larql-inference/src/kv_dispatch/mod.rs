//! `KvDispatch` — engine-facing intent surface for K/V cache + attention.
//!
//! Sibling to [`crate::FfnBackend`] (FFN dispatch) and
//! [`larql_compute::ComputeBackend`] (substrate kernel primitives).
//! [`KvEngine`](crate::KvEngine) implementations call `KvDispatch`
//! methods to express *intents* (allocate K/V, append a row, attend Q
//! against K/V with optional windowing, recompute K/V from residuals,
//! upload a boundary residual). The backend decides *how* — which
//! kernel runs, which shader variant from the pipeline cache, whether
//! the K/V append fuses into the attention kernel.
//!
//! ## Why a sibling trait, not a `ComputeBackend` sub-trait
//!
//! The CPU implementation of these intents needs to call into the
//! inference-side forward-pass functions (`run_attention_*`,
//! `run_ffn`, residual ops) that live in this crate. The trait
//! therefore lives here so its CPU impl (and Metal impl, via the same
//! orphan-rule logic) can be authored in this crate. Putting the trait
//! in `larql-compute` would block the CPU impl: orphan rules forbid
//! `impl KvDispatch for CpuBackend` in `larql-inference` when both
//! trait and type are foreign, and `larql-compute` can't depend on
//! `larql-inference` (would be a cycle).
//!
//! The substrate *capability* flags
//! ([`larql_compute::Capability::FusedAttentionStep`] etc.) stay in
//! `larql-compute` — they describe what the substrate supports
//! independently of where the dispatch trait lives.
//!
//! See `docs/specs/compute-backend-redesign.md` for full design rationale.
//!
//! ## Module layout
//!
//! - [`mod@self`] (this file) — trait surface, [`KvHandle`] /
//!   [`ResidualHandle`] types, [`EngineBackend`] umbrella,
//!   [`CompressionCodec`].
//! - [`cpu`] — `CpuBackend` impl (reference, all sync intents implemented).
//! - [`metal`] — `MetalBackend` impl (feature-gated; delegates to CPU
//!   at present, real GPU kernels are step 5 of the redesign).
//! - [`helpers`] — engine-facing per-layer prefill / decode loops:
//!   sync [`helpers::kv_prefill_via_dispatch`] /
//!   [`helpers::kv_decode_step_via_dispatch`] over [`EngineBackend`],
//!   plus async [`helpers::kv_prefill_via_dispatch_async`] /
//!   [`helpers::kv_decode_step_via_dispatch_async`] over
//!   [`crate::AsyncComputeBackend`].
//! - Future siblings: `vulkan`, `cuda`.
//!
//! ## Default behaviour
//!
//! Every method has a default that either returns `None` or panics
//! with `unimplemented!()`. Backends implementing the trait override
//! what they support. Engines should check
//! [`larql_compute::ComputeBackend::supports`] with the matching
//! [`larql_compute::Capability`] flag before calling, unless the
//! method has a meaningful default decomposition documented in its
//! doc-comment.

pub mod cpu;
pub mod helpers;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

use crate::model::ModelWeights;
use ndarray::Array2;

/// Opaque handle to a K/V cache allocation. Layout is backend-specific;
/// engines pass these around without observing structure beyond the
/// queries the trait exposes.
///
/// Backends ship their own inner type (`CpuKvHandle`, `MetalKvHandle`,
/// `VulkanKvHandle`) implementing [`KvHandleInner`]. Engines hold
/// `KvHandle` opaquely and call backend methods to manipulate it.
pub struct KvHandle {
    inner: Box<dyn KvHandleInner>,
}

impl KvHandle {
    /// Construct from a backend-specific inner. Backend implementations
    /// call this; engines never do.
    pub fn new<I: KvHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Number of K/V rows currently cached.
    pub fn cached_len(&self) -> usize {
        self.inner.cached_len()
    }

    /// Hidden dim per K/V row (kv_dim, not full hidden — already
    /// accounts for GQA head count).
    pub fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    /// Which backend allocated this handle. Used for sanity checks
    /// when handles cross backend boundaries (which normally
    /// shouldn't happen — read out to host first via
    /// [`KvDispatch::read_kv_to_host`]).
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    /// Downcast access for backend implementations. Engines never call
    /// this; only the backend that allocated the handle should.
    pub fn as_inner(&self) -> &dyn KvHandleInner {
        &*self.inner
    }

    /// Mutable downcast for backend impls.
    pub fn as_inner_mut(&mut self) -> &mut dyn KvHandleInner {
        &mut *self.inner
    }
}

/// Backend-side trait for K/V handle inner types. Backends implement
/// this on whatever GPU-side or host-side allocation they manage
/// (`MTLBuffer`, `VkBuffer`, `Vec<f32>`, or a wrapper over an engine's
/// `KvCache` from `larql-kv`).
pub trait KvHandleInner: Send + Sync + std::any::Any {
    fn cached_len(&self) -> usize;
    fn kv_dim(&self) -> usize;
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Opaque handle to a residual upload (used by `apollo` for boundary
/// residuals). Same pattern as [`KvHandle`].
pub struct ResidualHandle {
    inner: Box<dyn ResidualHandleInner>,
}

impl ResidualHandle {
    pub fn new<I: ResidualHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    pub fn as_inner(&self) -> &dyn ResidualHandleInner {
        &*self.inner
    }
}

pub trait ResidualHandleInner: Send + Sync + std::any::Any {
    fn shape(&self) -> (usize, usize);
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Engine-facing intent surface.
///
/// All methods are synchronous (return immediately with the result;
/// any GPU work is submitted and waited on internally). Async / stream-
/// graph variants live on a future `AsyncComputeBackend` trait — not
/// part of v1. See `compute-backend-redesign.md` §11.4.
///
/// Engines hold `&dyn KvDispatch` alongside
/// `&dyn larql_compute::ComputeBackend` and [`crate::FfnBackend`].
/// The three abstractions compose orthogonally: substrate kernels +
/// engine intents + FFN routing.
pub trait KvDispatch {
    // ── Cache primitives ────────────────────────────────────────────

    /// Allocate a K/V buffer for `layer`, sized for at most `max_tokens`
    /// positions of `kv_dim`-wide K and V rows. Layout is backend-
    /// specific; engines treat the returned handle opaquely.
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        let _ = (layer, max_tokens, kv_dim);
        unimplemented!("alloc_kv_buffer not implemented for this backend")
    }

    /// Append a single K/V row at `abs_position`. The handle must have
    /// been allocated by *this* backend; cross-backend handles panic.
    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        let _ = (handle, k_row, v_row, abs_position);
        unimplemented!("append_kv not implemented for this backend")
    }

    /// Clip the handle's cached entries to at most `window_size` rows
    /// (keep the tail). Backends with bounded-ring-buffer K/V layouts
    /// may implement this as a no-op; backends with growing K/V apply
    /// a shift or drop.
    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let _ = (handle, window_size);
        unimplemented!("clip_kv not implemented for this backend")
    }

    /// Read the full K/V back to host memory as a `(K, V)` pair.
    /// Blocking copy on GPU backends; identity on CPU. Should NOT be
    /// used in hot loops — it's the cross-backend escape hatch for
    /// fallback paths and debug inspection.
    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let _ = handle;
        None
    }

    // ── Attention primitives ────────────────────────────────────────

    /// Run one decode-step attention: Q (one row, pre-projection
    /// hidden) is projected internally to Q/K/V via the layer's
    /// weights, attended against K/V from `kv` PLUS the new token's
    /// K/V (the backend computes the new K/V from the query and
    /// appends it to `kv` as a side effect), and the post-O-projection
    /// hidden state is returned.
    ///
    /// `kv` is `&mut` because the backend mutates it: K and V grow by
    /// one row to include the current token. After this call the
    /// caller may invoke [`Self::clip_kv`] to enforce a sliding window.
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::FusedAttentionStep`]. Backends
    /// that don't support fused attention return `None`; callers fall
    /// back to decomposed BLAS attention via [`larql_compute::MatMul`]
    /// + manual K/V management.
    ///
    /// `index` is `Some` when the caller has a Q4K (or other
    /// quantised) `VectorIndex` available alongside the f32 fallback
    /// in `weights.tensors`. Backends with native Q4K kernels (e.g.
    /// `MetalBackend` once A4 lands) use it directly; CPU backends
    /// today expect the caller to have already populated
    /// `weights.tensors` via
    /// [`crate::vindex::ensure_attn_tensors_dequantised`] when the
    /// quantised source is present.
    ///
    /// See `docs/specs/kv-dispatch-quantization.md`.
    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<Array2<f32>> {
        let _ = (weights, query, kv, layer, abs_position, index);
        None
    }

    /// Like [`Self::attention_step`] but with a window bound baked
    /// into the dispatch — backend may use a specialised shader variant
    /// that knows the window size at compile time. Backend may also
    /// elide the post-attention `clip_kv` since the window is known.
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::WindowedAttentionStep`]. Default
    /// runs [`Self::attention_step`] then [`Self::clip_kv`] (correct
    /// but not specialised). `index` is forwarded to the underlying
    /// `attention_step` call.
    #[allow(clippy::too_many_arguments)]
    fn attention_step_windowed(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<Array2<f32>> {
        let h = self.attention_step(weights, query, kv, layer, abs_position, index)?;
        self.clip_kv(kv, window);
        Some(h)
    }

    /// Multi-token prefill attention: tokens have been embedded into
    /// `tokens_embedded` (shape `[seq_len, hidden]`). Backend runs full
    /// attention over the sequence, populates a fresh K/V handle, and
    /// returns `(last_hidden_1xH, populated_handle)`.
    ///
    /// `window` selects the K/V cap: `None` = unbounded growth,
    /// `Some(W)` = sliding-window K/V (older positions evicted from
    /// the cache after the prefill).
    ///
    /// `index` follows the same convention as [`Self::attention_step`].
    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = (weights, tokens_embedded, layer, window, index);
        None
    }

    // ── Engine-specific primitives ──────────────────────────────────

    /// Regenerate K/V for a layer from stored pre-layer residuals.
    /// Used by `markov-rs`: residuals are the persistent state, K/V is
    /// recomputed each decode step. Backends without this intent fall
    /// back to running the Q/K/V projection through
    /// [`larql_compute::MatMul`] directly.
    fn recompute_kv_from_residuals(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        let _ = (weights, residuals, layer);
        None
    }

    /// Append compressed K/V to a handle using the given codec.
    /// Used by `turbo-quant`. Backends with native codec kernels
    /// (Metal WHT shader) implement this; others fall back to
    /// dequant → f32 append → requant via the caller.
    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        let _ = (handle, k, v, codec);
        unimplemented!("compressed_kv_append not implemented for this backend")
    }

    /// Upload a boundary residual to backend-managed memory. Returns
    /// a handle the engine can use as the starting state for
    /// [`Self::forward_from_layer`]. Used by `apollo` compressed path.
    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let _ = residual;
        None
    }

    /// Run the forward pass starting at `start_layer` using `residuals`
    /// as the layer-`start_layer` input. Used by `apollo` to skip the
    /// pre-crystal layers when boundaries are available.
    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let _ = (weights, start_layer, residuals, token_ids);
        None
    }

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Coarse-grained, **quantization-agnostic** intents for engines
    // that want backend-fastest decode without per-layer control.
    // The backend inspects `index` (or `weights.tensors`) and dispatches
    // internally to whatever native kernel matches the weight format:
    // Q4K matvec, Q6K matvec, f32 fused, future quant formats — all
    // without changing this trait surface.
    //
    // Engines that DO need per-layer control (MarkovResidual,
    // UnlimitedContext, TurboQuant — recompute, checkpoint, codec
    // mechanisms) continue to use the per-layer `attention_prefill` /
    // `attention_step` intents.
    //
    // Default returns `None` — engines that want a coarse path fall
    // back to per-layer dispatch when the backend doesn't support it.

    /// Coarse prefill: run the prompt through every layer using the
    /// backend's fastest available kernel, populate a backend-specific
    /// K/V cache, return last-row hidden + the populated handle.
    ///
    /// The returned `KvHandle` is opaque to the engine; pass it back to
    /// [`Self::coarse_decode_step`] for subsequent steps. Backends are
    /// free to use any internal cache shape (`CpuKvCache` on CPU,
    /// `MTLBuffer` on Metal once Step A6 lands, etc.).
    ///
    /// `weights` is `&mut` because backends with cached-streaming Q4K
    /// kernels may lazily insert dequantised f32 fallback tensors into
    /// `weights.tensors` over the lifetime of the cache. The per-layer
    /// `attention_prefill` keeps `&weights` because it can't grow
    /// shared state.
    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = (weights, token_ids, index);
        None
    }

    /// One coarse decode step. `handle` must be the `KvHandle` returned
    /// by a prior [`Self::coarse_prefill`] on the same backend.
    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let _ = (weights, token_id, index, handle, abs_position);
        None
    }

    // ── Norm + residual primitives ──────────────────────────────────

    /// Fused `residual_add + rmsnorm` for the post-attention or
    /// post-FFN residual write. Target for D-RMS-FUSE phase 2 work.
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::FusedResidualNorm`]. Default
    /// decomposes into separate add + rmsnorm calls on host (correct
    /// but slow); backends with fused kernels override.
    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        // Default: decompose. add then rmsnorm.
        let added = x + residual;
        let mut out = Array2::<f32>::zeros(added.raw_dim());
        for (i, row) in added.rows().into_iter().enumerate() {
            let row_slice = row.as_slice().expect("non-contiguous row");
            let mean_sq: f32 =
                row_slice.iter().map(|v| v * v).sum::<f32>() / row_slice.len() as f32;
            let scale = (mean_sq + 1e-6).sqrt().recip();
            for (j, (val, w)) in row_slice.iter().zip(norm_weights.iter()).enumerate() {
                out[[i, j]] = val * scale * w;
            }
        }
        out
    }
}

/// Codec hook for [`KvDispatch::compressed_kv_append`]. Backends that
/// implement native compressed K/V append call back into the codec for
/// per-row encode/decode where the kernel isn't fully fused.
pub trait CompressionCodec: Send + Sync {
    fn encode(&self, vec: &[f32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32>;
    fn name(&self) -> &str;
}

/// Umbrella trait combining substrate kernel primitives
/// ([`larql_compute::ComputeBackend`]) and engine-facing dispatch
/// intents ([`KvDispatch`]). Engine implementations
/// ([`crate::KvEngine`] impls) take `&dyn EngineBackend` so they have
/// access to both surfaces through one trait object.
///
/// Any type that implements both `ComputeBackend` and `KvDispatch`
/// automatically implements `EngineBackend` via the blanket impl below.
/// FFN dispatch ([`crate::FfnBackend`]) stays separate per the
/// design's "FFN routing is a network-topology concern, not a substrate
/// concern" resolution
/// (`docs/specs/compute-backend-redesign.md` §11.1).
pub trait EngineBackend: larql_compute::ComputeBackend + KvDispatch {
    /// Trait-object upcast to `&dyn ComputeBackend`. Use when passing
    /// an `&dyn EngineBackend` to an API that takes `&dyn ComputeBackend`
    /// and Rust's trait-object upcasting can't infer the target type
    /// (e.g. inside `Option<&dyn ...>` or generic contexts where the
    /// expected type isn't a direct `&dyn ComputeBackend`).
    ///
    /// In simple call positions you can also write `self as &dyn ComputeBackend`,
    /// but this method is friendlier when the call site is awkward
    /// (e.g. `Some(self.backend.as_compute())`).
    fn as_compute(&self) -> &dyn larql_compute::ComputeBackend;
}

impl<T: larql_compute::ComputeBackend + KvDispatch> EngineBackend for T {
    fn as_compute(&self) -> &dyn larql_compute::ComputeBackend {
        self
    }
}

#[cfg(test)]
mod tests {
    //! Trait-default contract tests. `KvDispatch` has no supertraits, so
    //! a stub backend that overrides nothing exercises every default
    //! body. These tests document the documented "implement-me" contract:
    //! every default either returns `None` (engines treat it as
    //! "backend doesn't support this intent, fall back") or panics with
    //! a `not implemented for this backend` message.
    //!
    //! The `attention_step_windowed` and `residual_norm_store` defaults
    //! have meaningful decompositions; tests check the actual decomposed
    //! behaviour, not just panic semantics.
    //!
    //! Coverage role: this module's lines are dominated by trait-default
    //! bodies. Without these tests, those bodies are unreachable from
    //! the rest of the crate because every concrete backend
    //! (`CpuBackend`, `MetalBackend`) overrides the methods that don't
    //! `unimplemented!()`.

    use super::*;
    use ndarray::Array2;

    // ── Stub backend with all-default `KvDispatch` ───────────────────

    struct StubKvBackend;
    impl KvDispatch for StubKvBackend {}

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

    struct StubCodec;
    impl CompressionCodec for StubCodec {
        fn encode(&self, vec: &[f32]) -> Vec<u8> {
            vec.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .take(dim)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    fn stub_kv_handle(len: usize, dim: usize) -> KvHandle {
        KvHandle::new(StubKvInner { len, dim })
    }

    fn stub_residual_handle(rows: usize, cols: usize) -> ResidualHandle {
        ResidualHandle::new(StubResidualInner {
            shape: (rows, cols),
        })
    }

    // ── KvHandle / ResidualHandle accessor coverage ──────────────────

    #[test]
    fn kv_handle_accessors_through_stub_inner() {
        let mut handle = stub_kv_handle(5, 64);
        assert_eq!(handle.cached_len(), 5);
        assert_eq!(handle.kv_dim(), 64);
        assert_eq!(handle.backend_name(), "stub");
        // `as_inner` + `as_inner_mut` paths.
        let _: &dyn KvHandleInner = handle.as_inner();
        let _: &mut dyn KvHandleInner = handle.as_inner_mut();
    }

    #[test]
    fn residual_handle_accessors_through_stub_inner() {
        let handle = stub_residual_handle(3, 4);
        assert_eq!(handle.shape(), (3, 4));
        assert_eq!(handle.backend_name(), "stub");
        let _: &dyn ResidualHandleInner = handle.as_inner();
    }

    #[test]
    fn stub_codec_round_trips() {
        let codec = StubCodec;
        assert_eq!(codec.name(), "stub");
        let bytes = codec.encode(&[1.5_f32, 2.25, -0.5]);
        let back = codec.decode(&bytes, 3);
        assert_eq!(back, vec![1.5, 2.25, -0.5]);
    }

    // ── KvDispatch default bodies — None returns ─────────────────────

    #[test]
    fn default_read_kv_to_host_returns_none() {
        let backend = StubKvBackend;
        let handle = stub_kv_handle(0, 64);
        assert!(backend.read_kv_to_host(&handle).is_none());
    }

    #[test]
    fn default_attention_step_returns_none() {
        let backend = StubKvBackend;
        let weights = crate::test_utils::make_test_weights();
        let mut handle = stub_kv_handle(0, weights.hidden_size);
        let query = Array2::zeros((1, weights.hidden_size));
        assert!(backend
            .attention_step(&weights, &query, &mut handle, 0, 0, None)
            .is_none());
    }

    #[test]
    fn default_attention_step_windowed_propagates_none() {
        // Default decomposes into `attention_step` (returns None) then
        // `clip_kv`. The `?` on None short-circuits before clip_kv would
        // panic. Tests the default-body's None-propagation branch.
        let backend = StubKvBackend;
        let weights = crate::test_utils::make_test_weights();
        let mut handle = stub_kv_handle(0, weights.hidden_size);
        let query = Array2::zeros((1, weights.hidden_size));
        assert!(backend
            .attention_step_windowed(&weights, &query, &mut handle, 0, 0, 4, None)
            .is_none());
    }

    #[test]
    fn default_attention_prefill_returns_none() {
        let backend = StubKvBackend;
        let weights = crate::test_utils::make_test_weights();
        let tokens = Array2::zeros((2, weights.hidden_size));
        assert!(backend
            .attention_prefill(&weights, &tokens, 0, None, None)
            .is_none());
    }

    #[test]
    fn default_recompute_kv_from_residuals_returns_none() {
        let backend = StubKvBackend;
        let weights = crate::test_utils::make_test_weights();
        let residuals = Array2::zeros((1, weights.hidden_size));
        assert!(backend
            .recompute_kv_from_residuals(&weights, &residuals, 0)
            .is_none());
    }

    #[test]
    fn default_upload_boundary_residual_returns_none() {
        let backend = StubKvBackend;
        let residual = Array2::zeros((1, 8));
        assert!(backend.upload_boundary_residual(&residual).is_none());
    }

    #[test]
    fn default_forward_from_layer_returns_none() {
        let backend = StubKvBackend;
        let weights = crate::test_utils::make_test_weights();
        let residuals = stub_residual_handle(1, weights.hidden_size);
        assert!(backend
            .forward_from_layer(&weights, 0, &residuals, &[0u32])
            .is_none());
    }

    // ── KvDispatch default bodies — `unimplemented!()` panics ────────

    #[test]
    #[should_panic(expected = "alloc_kv_buffer not implemented")]
    fn default_alloc_kv_buffer_panics() {
        let backend = StubKvBackend;
        let _ = backend.alloc_kv_buffer(0, 32, 64);
    }

    #[test]
    #[should_panic(expected = "append_kv not implemented")]
    fn default_append_kv_panics() {
        let backend = StubKvBackend;
        let mut handle = stub_kv_handle(0, 4);
        backend.append_kv(&mut handle, &[0.0; 4], &[0.0; 4], 0);
    }

    #[test]
    #[should_panic(expected = "clip_kv not implemented")]
    fn default_clip_kv_panics() {
        let backend = StubKvBackend;
        let mut handle = stub_kv_handle(0, 4);
        backend.clip_kv(&mut handle, 2);
    }

    #[test]
    #[should_panic(expected = "compressed_kv_append not implemented")]
    fn default_compressed_kv_append_panics() {
        let backend = StubKvBackend;
        let mut handle = stub_kv_handle(0, 4);
        let k = Array2::zeros((1, 4));
        let v = Array2::zeros((1, 4));
        let codec = StubCodec;
        backend.compressed_kv_append(&mut handle, &k, &v, &codec);
    }

    // ── Real default decomposition: residual_norm_store ──────────────

    #[test]
    fn default_residual_norm_store_decomposes_add_plus_rmsnorm() {
        // The trait's default body implements `residual_add` followed by
        // a per-row rmsnorm with eps=1e-6. Test against a hand-computed
        // expected output so the decomposition body is exercised end-to-end.
        let backend = StubKvBackend;
        let x = Array2::from_shape_vec((1, 4), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let residual = Array2::from_shape_vec((1, 4), vec![0.5_f32, 0.5, 0.5, 0.5]).unwrap();
        let norm_weights = vec![1.0_f32; 4];

        let out = backend.residual_norm_store(&x, &residual, &norm_weights);

        // Hand-computed: added = [1.5, 2.5, 3.5, 4.5]
        // mean_sq = (1.5² + 2.5² + 3.5² + 4.5²) / 4 = (2.25+6.25+12.25+20.25)/4 = 10.25
        // scale = 1.0 / sqrt(10.25 + 1e-6) ≈ 0.31234752...
        let added = [1.5_f32, 2.5, 3.5, 4.5];
        let mean_sq: f32 = added.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let scale = (mean_sq + 1e-6).sqrt().recip();
        for j in 0..4 {
            let expected = added[j] * scale;
            assert!(
                (out[[0, j]] - expected).abs() < 1e-5,
                "col {j}: out={} expected={}",
                out[[0, j]],
                expected
            );
        }
    }

    #[test]
    fn default_residual_norm_store_applies_per_column_weights() {
        // Same shape but non-uniform norm_weights — verifies the per-column
        // multiplication branch.
        let backend = StubKvBackend;
        let x = Array2::from_shape_vec((1, 2), vec![1.0_f32, 1.0]).unwrap();
        let residual = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
        let norm_weights = vec![2.0_f32, 0.5_f32];
        let out = backend.residual_norm_store(&x, &residual, &norm_weights);
        // mean_sq = (1 + 1) / 2 = 1.0, scale ≈ 1 / sqrt(1+1e-6)
        let scale = (1.0_f32 + 1e-6).sqrt().recip();
        assert!((out[[0, 0]] - scale * 2.0).abs() < 1e-5);
        assert!((out[[0, 1]] - scale * 0.5).abs() < 1e-5);
    }

    // ── EngineBackend blanket impl ───────────────────────────────────

    #[test]
    fn engine_backend_as_compute_returns_self() {
        // Any type implementing ComputeBackend + KvDispatch auto-implements
        // EngineBackend. Use CpuBackend, then call `as_compute` to exercise
        // the blanket impl's body (one line: `self`).
        let backend = larql_compute::CpuBackend;
        let as_engine: &dyn EngineBackend = &backend;
        let _: &dyn larql_compute::ComputeBackend = as_engine.as_compute();
    }
}
