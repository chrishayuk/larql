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
//! The engine-facing intent surface (`KvDispatch`) is a *sibling* of
//! `ComputeBackend`, not a sub-trait. It lives in `larql-inference`
//! (sibling to `FfnBackend`) so its CPU and Metal impls can call into
//! the inference-side forward-pass functions without inducing a dep
//! cycle on `larql-compute`. New [`Capability`] flags
//! (`FusedAttentionStep`, `WindowedAttentionStep`, `NativeKvCodec`,
//! `PipelinedBoundaryUpload`, `FusedResidualNorm`, `KvHandleNative`)
//! stay here — they describe what the *substrate* supports, regardless
//! of where the dispatch trait lives. See
//! `crates/larql-inference/docs/specs/compute-backend-redesign.md` §10.2.
//!
//! Most callers stay typed against `&dyn ComputeBackend`; the
//! sub-trait split is mainly an implementation-side organising
//! principle. Callers that want to branch on a specific accelerator
//! (e.g. "use f32_gemv if the backend has it, otherwise fall back to
//! matmul_transb") should use [`Capability`] + [`ComputeBackend::supports`]
//! instead of probing for `None` returns.

pub mod capability;
pub mod decode;
pub mod helpers;
pub mod matmul;
pub mod quant_matvec;

pub use capability::Capability;
pub use decode::DecodeBackend;
pub use helpers::{dot_proj_gpu, matmul_gpu};
pub use matmul::{MatMul, MatMulOp};
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

    /// Expose the concrete type for safe downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}
