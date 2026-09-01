//! # larql-compute
//!
//! [`ComputeBackend`] trait + CPU implementation + pipeline-shape
//! types.  GPU backends ship as sibling crates that implement the
//! same trait — see [`larql-compute-metal`](https://docs.rs/larql-compute-metal)
//! for the Apple-silicon impl.
//!
//! ## Trait split
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits three narrower traits, each
//! in its own module:
//!
//! - [`MatMul`] — f32 / f16 matmul, gemv, batch matmul
//! - [`QuantMatVec`] — unified `quant_matvec` + per-format pre-quantised helpers
//! - [`DecodeBackend`] — KV-cached decode + prefill + MoE hook +
//!   W10 `decode_token_with_state_dump_masked`
//! - umbrella `ComputeBackend` — `name`, `device_info`, [`Capability`] probe
//!
//! `use larql_compute::prelude::*;` brings every sub-trait in scope at once.
//!
//! ## State handles + W10 mask cascade
//!
//! [`state_handle`] — opaque references to per-layer state rows and slabs
//! that may live on different devices or remote nodes. Used together with
//! [`StateDumpMask`] (`{Full, HOnly, None}`) and the
//! `*_with_state_dump_masked` / `*_with_state_masked` trait methods to let
//! engines that treat K/V as derivative state skip the GPU→CPU bridge.
//! Defaults preserve `Full` behaviour everywhere; the mask is a per-engine
//! opt-in (see `crates/larql-kv/docs/state-policy.md` and the per-engine
//! W10 sections in `crates/larql-inference/docs/specs/*-engine.md`).
//!
//! ## Backends
//!
//! | Backend | Crate                  | Operations |
//! |---------|------------------------|------------|
//! | CPU     | `larql-compute`         | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal   | `larql-compute-metal`   | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | Vulkan  | `larql-compute-vulkan` (planned) | — |
//! | CUDA    | `larql-compute-cuda` (planned)   | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use larql_compute::prelude::*;
//! use larql_compute::{default_backend, QuantFormat};
//!
//! // `default_backend()` always returns CPU.  Hosts that want GPU
//! // depend on the relevant backend crate and compose a fallback chain
//! // explicitly — see `larql-compute-metal::metal_backend()`.
//! let backend = default_backend();
//! println!("Using: {} ({})", backend.name(), backend.device_info());
//!
//! // Branch on capability instead of probing for `Option::None`:
//! if backend.supports(Capability::F32Gemv) {
//!     // Specialised LM-head gemv is available on this backend.
//! }
//! ```
//!
//! ## Adding a quant format
//!
//! Adding e.g. FP4 = one [`QuantFormat`] variant + one arm in
//! [`QuantFormat::route`] (the [`FormatRoute`] registry — dequant +
//! matvec + matmul kernel pointers in one place) + one match arm in
//! [`QuantMatVec::quant_matvec`]'s default impl + one CPU kernel +
//! one shader per GPU-backend crate.  The shader-side wiring is
//! local to each backend crate, so a new format doesn't require
//! touching every consumer.  Formats whose weights carry side
//! metadata (I2S trit scales, MXFP4 E8M0 scales) follow the
//! `ternary_matvec` parallel-path template instead — see the
//! unknown-format contract in [`quant_route`].

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "windows"
))]
extern crate blas_src;

pub mod async_compute_backend;
pub mod attention;
pub mod backend;
pub mod connectors;
pub mod cpu;
pub mod encoders;
pub mod ffn;
pub mod forward;
pub mod forward_overrides;
pub mod kquant_forward;
pub mod kv_dispatch;
pub mod kv_index;
pub mod moe_route_observe;
pub mod options;
pub mod packed_attn_index;
pub mod per_layer_decode_state;
pub mod phase_timing;
pub mod pipeline;
pub mod pipeline_layer;
pub mod quant_route;
pub mod residual;
pub mod state_handle;

/// Synthetic test fixtures (Q4K `KvIndex` builder). Behind the
/// `test-utils` feature — production builds never see it.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_fixtures;

pub use kv_index::{KvIndex, FFN_COMPONENTS_PER_LAYER, FFN_DOWN, FFN_GATE, FFN_UP};
pub use per_layer_decode_state::PerLayerDecodeState;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    stored_gate_up_cols, Activation, AttentionSpec, AttentionWeights, ExpertBankOverride,
    ExpertMlp, ExternalScaleKind, FfnSpec, FfnType, FfnWeights, FullPipelineLayer, LayerNorms,
    LayerWeights, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout,
    MoeGateRule, MoeInputSource, MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy,
    MoeRoutingPolicy, MoeSpec, MoeTopKWeightPolicy, MoeWeightLayout, NormType,
    PositionEncodingType, QuantAux, QuantFormat, QuantWeight, RemoteFfnSpec, ScaleStorage,
    RMSNORM_EPSILON_DEFAULT, ROPE_BASE_DEFAULT, ROPE_BASE_GLOBAL,
};

// ── Re-exports: backend ──

pub use backend::{
    backend_from_spec, dot_proj_gpu, matmul_gpu, BackendCtor, BackendKind, BackendSelectError,
    Capability, ComputeBackend, DecodeBackend, DecodeHeadPlan, DecodeStateDump, MatMul, MatMulOp,
    ProfileTimings, QuantMatVec, StateDumpMask,
};

/// Bring every backend sub-trait into scope at once.
///
/// Most test/bench/example code calls methods like `matmul_transb` or
/// `q4_matvec` directly on a concrete backend value, which Rust
/// resolves through the sub-trait that defines the method.
/// `use larql_compute::prelude::*;` saves listing them one by one.
pub mod prelude {
    pub use crate::backend::{
        Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp, QuantMatVec,
    };
}

pub use quant_route::FormatRoute;

pub use cpu::ops::linalg::{cholesky, cholesky_inverse, cholesky_solve, ridge_decomposition_solve};
pub use cpu::ops::moe::{quantize_x_to_q8k, Q8KActivation};
pub use cpu::ops::q4k_matvec::f16_to_f32;
pub use cpu::ops::vector::{cosine, dot, norm};
pub use cpu::ops::KernelShapeError;
pub use cpu::CpuBackend;
pub use packed_attn_index::PackedAttnIndex;

/// Build a CPU backend.  Always returns a usable backend (BLAS on
/// macOS via Accelerate, OpenBLAS on Linux/Windows).
///
/// Callers that want GPU compose an explicit fallback chain via the
/// relevant backend crate (requires a workspace dep on the backend
/// crate, so this is shown `ignore` here — `larql-compute` doesn't
/// take `larql-compute-metal` as a dep to avoid the circular pull):
///
/// ```rust,ignore
/// use larql_compute::{cpu_backend, ComputeBackend};
///
/// let backend: Box<dyn ComputeBackend> =
///     larql_compute_metal::metal_backend()
///         .map(|m| Box::new(m) as Box<dyn ComputeBackend>)
///         .unwrap_or_else(|| cpu_backend());
/// ```
///
/// [`default_backend`] is a synonym kept for backwards compatibility
/// with the pre-split callers.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}

/// Build the default backend.  Returns CPU.
///
/// Before the GPU-backend extraction, this function auto-detected
/// Metal and fell back to CPU.  After the split, GPU selection is the
/// caller's responsibility — see [`cpu_backend`] for the explicit
/// fallback pattern.  Kept as an alias of [`cpu_backend`] so existing
/// CPU-only callers don't need to change.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    cpu_backend()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_exposes_cpu_backend_capabilities() {
        let backend = cpu_backend();

        assert!(backend.name().starts_with("cpu"));
        assert!(!backend.device_info().is_empty());
        assert!(backend.supports(Capability::QuantMatVec));
    }

    #[test]
    fn default_backend_is_usable_through_prelude_traits() {
        fn assert_compute_backend<T: prelude::ComputeBackend + ?Sized>(backend: &T) {
            assert!(backend.supports(prelude::Capability::QuantMatVec));
        }

        let backend = default_backend();
        assert_compute_backend(backend.as_ref());
    }

    /// After the GPU-backend extraction, `default_backend()` and
    /// `cpu_backend()` are synonyms — both return CPU.  Pin that
    /// invariant so a future reintroduction of GPU auto-selection in
    /// this crate is caught.
    #[test]
    fn default_backend_returns_cpu_after_metal_extraction() {
        let default = default_backend();
        let cpu = cpu_backend();
        assert_eq!(default.name(), cpu.name());
    }
}
