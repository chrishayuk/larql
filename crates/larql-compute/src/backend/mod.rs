//! Compute backend interface.
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits four narrower traits, each in
//! its own module so it's easy to read what a backend has to provide:
//!
//! | Sub-trait                     | What's there                                  |
//! |-------------------------------|-----------------------------------------------|
//! | [`MatMul`]                    | f32 / f16 matmul, gemv, batch matmul          |
//! | [`QuantMatVec`]               | unified `quant_matvec` + per-format helpers   |
//! | [`DecodeBackend`]             | KV-cached decode + prefill + MoE hook (Metal-shaped) |
//! | (umbrella) `ComputeBackend`   | `name`, `device_info`, [`Capability`] probe   |
//!
//! The engine-facing intent surface (`KvDispatch`) is a sibling of
//! `ComputeBackend`, not a sub-trait. It lives in this crate's
//! `kv_dispatch` module alongside the CPU forward substrate (ADR-0022).
//! `AsyncComputeBackend` is the deferred-dispatch contract; Metal implements
//! both in `larql-compute-metal`. Inference composes the traits and retains
//! compatibility re-exports. Capability flags describe numerical support;
//! they do not move session or routing ownership into the substrate.
//!
//! Most callers stay typed against `&dyn ComputeBackend`; the
//! sub-trait split is mainly an implementation-side organising
//! principle. Callers that want to branch on a specific accelerator
//! (e.g. "use f32_gemv if the backend has it, otherwise fall back to
//! matmul_transb") should use [`Capability`] + [`ComputeBackend::supports`]
//! instead of probing for `None` returns.

pub mod capability;
pub mod decode;
pub mod factory;
pub mod helpers;
pub mod matmul;
pub mod quant_matvec;

pub use capability::Capability;
pub use decode::{DecodeBackend, DecodeHeadPlan, DecodeStateDump, ProfileTimings, StateDumpMask};
pub use factory::{backend_from_spec, BackendCtor, BackendKind, BackendSelectError};
pub use helpers::{dot_proj_gpu, matmul_gpu};
pub use matmul::{MatMul, MatMulOp, SubmissionClock};
pub use quant_matvec::QuantMatVec;

/// Hardware compute backend — the umbrella trait every caller binds.
///
/// Combines [`MatMul`] + [`QuantMatVec`] + [`DecodeBackend`] plus
/// metadata (`name`, `device_info`) and an explicit
/// [`Capability::supports`](Self::supports) probe. Most callers
/// shouldn't care which sub-trait a method comes from.
pub trait ComputeBackend: MatMul + QuantMatVec + DecodeBackend + Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string (for logging/diagnostics).
    fn device_info(&self) -> String {
        self.name().to_string()
    }

    /// Whether this backend accelerates `cap`. Callers can branch on
    /// this *before* calling, instead of pattern-matching on `None`
    /// returns from probe methods.
    ///
    /// Default returns `false` for everything; backends override to
    /// enable. See [`Capability`] for the menu.
    fn supports(&self, _cap: Capability) -> bool {
        false
    }

    /// Register a stable, page-granular weight allocation (an mmap'd
    /// weight file) so the backend can alias sub-slices of it zero-copy
    /// instead of staging per-call copies. GPU backends bind registered
    /// regions as device buffers and resolve expert/tensor slices to
    /// byte offsets; the default is a no-op — CPU reads host memory
    /// directly and has nothing to stage.
    ///
    /// # Contract
    ///
    /// `region` must never move and must outlive the backend (the mmap
    /// stability contract weight caches already rely on). Registering
    /// the same base twice is a cheap no-op.
    fn register_weight_region(&self, _region: &[u8]) {}

    /// Called once after the last [`Self::register_weight_region`] for a
    /// model, before any timed work.
    ///
    /// A backend that prepares resources up front (Metal declares an explicit
    /// residency set over the registered allocations) does it here rather than
    /// per registration, which would rebuild the declaration N times. Default
    /// is a no-op, which is exactly the pre-existing behaviour.
    fn seal_weight_regions(&self) {}

    /// Expose the concrete type for safe downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Upload a Per-Layer Embeddings input table for the next
    /// `decode_token*` / `prefill_*` call.
    ///
    /// `flat` is laid out as `[(position * num_layers + layer) * ple_dim]`
    /// f32 elements. Backends that don't implement PLE do nothing — the
    /// upload is a no-op and the engine must skip the precompute work
    /// instead of relying on this method to validate inputs. Probe
    /// [`Capability::PerLayerEmbeddings`] to decide whether to call at all.
    fn prepare_ple_inputs(&self, _flat: &[f32], _num_layers: usize, _ple_dim: usize) {}

    /// Consume the per-stage timing recorded by the most recent
    /// `decode_token_split_profile` call on the current thread.
    ///
    /// Backends that record split-stage timings (Metal via paired
    /// commit/wait boundaries; future Vulkan/CUDA via their own
    /// timestamp APIs) return `Some(_)` when `LARQL_PROFILE_SPLIT=1`
    /// was honoured on the preceding call. Default returns `None` —
    /// engines treat that as "no instrumentation available."
    fn take_split_timings(&self) -> Option<ProfileTimings> {
        None
    }

    /// Run ONE layer of attention on the GPU and return the
    /// post-attention hidden state. The caller runs FFN externally
    /// (typically a `vindex` walk) and feeds the result back for the
    /// next layer.
    ///
    /// Steps the backend performs: input norm → QKV projection → RoPE →
    /// V-norm → KV append → KV attend → O projection → post-attention
    /// residual + post-attn norm.
    ///
    /// The backing KV cache is owned by the backend and ensured to
    /// match `kv_shapes` (one `(num_kv_heads, head_dim)` per absolute
    /// layer) before the dispatch. Default returns `None` — backends
    /// without GPU attention drop through to the engine's CPU
    /// fallback path. Probe [`Capability::HybridAttention`] before
    /// calling.
    #[allow(clippy::too_many_arguments)]
    fn hybrid_decode_attention_layer(
        &self,
        _layer: &crate::FullPipelineLayer<'_>,
        _layer_idx: usize,
        _x: &[f32],
        _hidden: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _kv_shapes: &[(usize, usize)],
    ) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayView2};

    /// Minimal `ComputeBackend` that uses every default impl. Lets the
    /// tests below exercise the trait's default bodies (which `CpuBackend`
    /// and `MetalBackend` would otherwise shadow with their overrides).
    struct StubBackend;

    impl MatMul for StubBackend {
        fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul never called")
        }
        fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul never called")
        }
    }
    impl QuantMatVec for StubBackend {}
    impl DecodeBackend for StubBackend {}
    impl ComputeBackend for StubBackend {
        fn name(&self) -> &str {
            "stub"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn default_device_info_falls_back_to_name() {
        let b = StubBackend;
        assert_eq!(b.device_info(), "stub");
    }

    #[test]
    fn default_supports_returns_false_for_every_capability() {
        let b = StubBackend;
        for cap in [
            Capability::F32Gemv,
            Capability::F16Gemv,
            Capability::QuantMatVec,
            Capability::Q4VecMat,
            Capability::Q4PairBatch,
            Capability::FullPipelineQ4,
            Capability::MultiLayerQ4Ffn,
            Capability::DecodeToken,
            Capability::DecodeMoe,
            Capability::DecodeQ4KMoe,
            Capability::DecodeProfile,
            Capability::PrefillQ4,
            Capability::HeterogeneousAttention,
            Capability::FusedAttentionStep,
            Capability::WindowedAttentionStep,
            Capability::NativeKvCodec,
            Capability::PipelinedBoundaryUpload,
            Capability::FusedResidualNorm,
            Capability::KvHandleNative,
            Capability::PerLayerEmbeddings,
            Capability::HybridAttention,
        ] {
            assert!(!b.supports(cap), "default supports must reject {cap:?}");
        }
    }

    #[test]
    fn default_prepare_ple_inputs_is_a_no_op() {
        let b = StubBackend;
        // No state to read back — we're just verifying the call returns
        // without panic. The default's job is to be a safe no-op so engines
        // that probe `Capability::PerLayerEmbeddings` and skip can still
        // safely call through.
        b.prepare_ple_inputs(&[0.0; 8], 2, 4);
        b.prepare_ple_inputs(&[], 0, 0);
    }

    #[test]
    fn default_take_split_timings_returns_none() {
        let b = StubBackend;
        assert!(b.take_split_timings().is_none());
    }

    #[test]
    fn default_hybrid_decode_attention_layer_returns_none() {
        let b = StubBackend;
        // The trait default ignores every argument; `FullPipelineLayer::default()`
        // provides the minimal layer fixture.
        let layer = crate::FullPipelineLayer::default();
        let kv_shapes = [(2usize, 4usize)];
        let result = b.hybrid_decode_attention_layer(&layer, 0, &[0.0; 8], 8, 8, 8, &kv_shapes);
        assert!(result.is_none());
    }

    /// `as_any` must downcast back to the concrete type.
    #[test]
    fn as_any_downcasts_to_concrete() {
        let b = StubBackend;
        let any = b.as_any();
        assert!(any.downcast_ref::<StubBackend>().is_some());
    }
}
