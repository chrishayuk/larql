//! larql-inference — full transformer forward pass + mechanistic-interp surface.
//!
//! Two roles:
//!
//! - **Inference**: prefill, decode, sampling, KV engines, Metal GPU path,
//!   chat templates. `predict`, `generate`, `predict_with_temperature`.
//! - **Mechanistic interp**: programmatic hooks at every layer boundary,
//!   logit lens, embedding-neighbor lookups, activation patching, KV-cache
//!   surgery. The primitives lazarus-style MCP servers build on.
//!
//! ## Mechanistic interp surface
//!
//! Five callbacks fire inside [`forward::trace_forward_full_hooked`]; two of
//! them take `&mut Array2<f32>` so a hook can mutate the residual in place:
//!
//! ```text
//! pre_layer  →  attention  →  on_post_attention(&mut h)  →  FFN  →  on_post_layer(&mut h)
//!                                  ^                              ^
//!                                  └─ patching, pre-FFN steer ────┘
//! ```
//!
//! Built-in hooks live in [`forward::hooks`]:
//! [`RecordHook`](forward::RecordHook) (capture),
//! [`ZeroAblateHook`](forward::ZeroAblateHook) (zero-out),
//! [`SteerHook`](forward::SteerHook) (`x + α·v`),
//! [`CompositeHook`](forward::CompositeHook) (compose multiple). Implement
//! [`forward::LayerHook`] for custom transforms.
//!
//! Sibling primitives:
//!
//! - [`forward::lens`] — full logit lens, `track_token`, `track_race`.
//! - [`forward::vocab_proj`] — `W_E` / `W_U` access, `embedding_neighbors`,
//!   raw `project_through_unembed` (DLA without final norm).
//! - [`forward::patching`] — donor/recipient activation patching built on
//!   the hook surface.
//! - `larql_kv::KvCache` — `get_layer` / `set_layer` /
//!   `clone_layer_position_range` for KV-cache surgery (e.g. lazarus's
//!   `prefill_inject` and `kv_inject_test`). Lives in `larql-kv` since
//!   2026-05-16 — it is engine state, not substrate.
//!
//! See `examples/mech_interp_demo.rs` for an end-to-end walkthrough on
//! synthetic weights (no vindex required).

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "windows"
))]
extern crate blas_src;

pub mod async_compute_backend;
pub mod attention;
pub mod capture;
pub mod chat;
pub mod error;
pub mod experts;
pub mod ffn;
pub mod forward;
pub mod forward_overrides;
pub mod kv_dispatch;
pub mod kv_engine;
pub mod layer_executor;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod residual;
pub mod residual_diff;
pub mod test_utils;
pub mod tokenizer;
pub mod trace;
pub mod vindex;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports — only the names with external consumers via
// `larql_inference::*`. Callers wanting other compute types should
// `use larql_compute::...` directly.
pub use larql_compute::cpu::ops::moe::{run_single_expert, run_single_expert_with_norm};
pub use larql_compute::QuantFormat;
pub use larql_compute::{cpu_backend, default_backend, ComputeBackend};

/// CPU backend boxed as `EngineBackend`. Use when you need to construct
/// a backend that satisfies the `EngineBackend` umbrella (so engines
/// can dispatch through both `ComputeBackend` and `KvDispatch`).
/// `cpu_backend()` returns `Box<dyn ComputeBackend>` and can't be
/// upcast to `Box<dyn EngineBackend>` at the trait-object level — use
/// this factory instead.
pub fn cpu_engine_backend() -> Box<dyn EngineBackend> {
    Box::new(larql_compute::CpuBackend)
}

/// Default backend as `Box<dyn EngineBackend>` — Metal on macOS when
/// the `metal` feature is enabled, CPU otherwise. Parallel to
/// `default_backend()` but returns the wider trait object.
pub fn default_engine_backend() -> Box<dyn EngineBackend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(metal) = larql_compute_metal::MetalBackend::new() {
            return Box::new(metal);
        }
    }
    cpu_engine_backend()
}

/// CPU backend boxed as `AsyncComputeBackend`. Use when constructing
/// engines via the `with_async_backend` opt-in (e.g.
/// `StandardEngine::with_async_backend`). Output is bit-identical to
/// the synchronous `cpu_engine_backend()` path — CPU is a degenerate
/// `Ready*`-wrapped impl (A2 parity reference).
pub fn cpu_async_engine_backend() -> Box<dyn AsyncComputeBackend> {
    Box::new(larql_compute::CpuBackend)
}

/// Default async backend as `Box<dyn AsyncComputeBackend>` — Metal on
/// macOS when the `metal` feature is enabled, CPU otherwise. Parallel
/// to [`default_engine_backend`].
///
/// At A3 (scaffolding), the Metal variant delegates every async call
/// to `CpuBackend`'s async impl — same cost as the sync path. The
/// tok/s shape changes at A4 when `MetalBackend` lands real deferred
/// dispatch (one `MTLCommandBuffer` per session).
pub fn default_async_engine_backend() -> Box<dyn AsyncComputeBackend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(metal) = larql_compute_metal::MetalBackend::new() {
            return Box::new(metal);
        }
    }
    cpu_async_engine_backend()
}

/// Default compute backend as `Box<dyn ComputeBackend>` — Metal on
/// macOS when the `metal` feature is enabled, CPU otherwise.
///
/// `larql_compute::default_backend()` lost its Metal auto-detection
/// after the `larql-compute-metal` extraction (the comment in that
/// function recommends callers construct `MetalBackend` directly).
/// This factory restores the convenience for callers that want a
/// runtime-detected GPU backend without `#[cfg(feature = "metal")]`
/// gating in every call site — `larql bench --backends metal` uses it,
/// engines that want a compute backend for `fused_prefill` /
/// `fused_decode_step` use it.
pub fn default_compute_backend() -> Box<dyn larql_compute::ComputeBackend> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(metal) = larql_compute_metal::MetalBackend::new() {
            return Box::new(metal);
        }
    }
    larql_compute::cpu_backend()
}

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(
    arch: &dyn larql_models::ModelArchitecture,
) -> larql_compute::Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    }
}

// Re-export essentials at crate root.
pub use async_compute_backend::{
    AsyncComputeBackend, AsyncDispatchError, AttentionHandle, AttentionHandleInner, ReadyAttention,
    ReadyResidualUpload, ResidualUploadHandle, ResidualUploadHandleInner,
};
pub use attention::AttentionWeights;
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
    DEFAULT_ACTIVATION_TOP_K, DEFAULT_RESIDUAL_TOP_K,
};
pub use chat::{wrap_chat_prompt, wrap_prompt_raw, wrap_with_vindex_template, ChatWrap};
pub use error::InferenceError;
pub use ffn::graph_backend::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use ffn::{
    BackendFfn, FfnBackend, LayerFfnRouter, LayerShardedBackend, MoeRouterWeights, RemoteFfnConfig,
    RemoteFfnError, RemoteLatencyStats, RemoteMoeBackend, RemoteMoeError, RemoteWalkBackend,
    ShardConfig, SparseFfn, WeightFfn, WirePreference,
};
pub use kv_dispatch::{
    CompressionCodec, EngineBackend, KvDispatch, KvHandle, KvHandleInner, ResidualHandle,
    ResidualHandleInner,
};
pub use kv_engine::{DecodeStageSummary, EngineInfo, KvEngine};
// Crate-root forward re-exports — kept for any name with external use OR
// in-crate examples/tests/benches that already import from the root. The
// curated `research` module (below) re-sources these from subpaths so it
// keeps working when individual root re-exports are dropped.
//
// Truly-unused root re-exports (no external + no inference example/test
// usage) were dropped 2026-05-09: `capture_ffn_activation_matrix`,
// `estimate_ffn_covariance`, `forward_raw_logits`, `infer_patched_q4k`,
// `predict_from_hidden_with_ffn`, `predict_with_ffn_trace`,
// `trace_forward_with_ffn`, `InferPatchedResult`, `LayerMode`,
// `MemitFactResult`, `PredictResultWithAttention`,
// `PredictResultWithResiduals`, `RawForward`, `SpecCapture`,
// `TargetDelta`, `TraceResult`, `KNN_COSINE_THRESHOLD`. They remain
// accessible via `larql_inference::forward::*` and `research::*`.
pub use forward::{
    apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals, capture_residuals,
    capture_spec_residuals, forward_from_layer, forward_to_layer, hidden_to_raw_logits,
    infer_patched, logit_lens_top1, predict, predict_from_hidden, predict_with_ffn,
    predict_with_ffn_attention, predict_with_router, predict_with_strategy, run_memit,
    run_memit_with_target_opt, trace_forward, trace_forward_full, walk_trace_from_residuals,
    InferenceWeights, KnnOverride, LayerAttentionCapture, MemitFact, MemitResult, PredictResult,
    TargetDeltaOpts,
};
// Crate-root layer_graph re-exports — kept for any name with external use
// OR in-crate examples/tests/benches that import via the root. Truly-unused
// names (no external + no inference example/test usage) dropped 2026-05-09:
// `GridGenerateResult`, `ChatMLRenderer`, `GemmaRenderer`, `LayerOutput`,
// `Llama3Renderer`, `PerLayerGraph`, `TurnRenderer`. They remain reachable
// via `larql_inference::layer_graph::*`.
pub use layer_graph::{
    build_adaptive_graph,
    detect_template,
    generate,
    generate_streaming,
    generate_with_sampling,
    // Expert grid generation
    grid::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, generate_with_remote_moe,
        generate_with_remote_moe_batch,
    },
    hybrid::predict_hybrid,
    predict_honest,
    predict_pipeline,
    predict_split_cached,
    predict_split_pass,
    predict_with_graph,
    predict_with_graph_vindex_logits,
    trace_with_graph,
    try_generate,
    try_generate_streaming,
    try_generate_with_sampling,
    AttentionCache,
    CachedLayerGraph,
    // Multi-turn chat session
    ChatSession,
    DenseLayerGraph,
    // Generation building blocks (EOS, detok, sampling)
    Detokenizer,
    EosConfig,
    GenerateError,
    GenerateResult,
    GuidedWalkLayerGraph,
    // Production
    LayerGraph,
    PipelinedLayerGraph,
    Sampler,
    SamplingConfig,
    // Analysis/validation
    TemplatePattern,
    TemplateUniverse,
    WalkLayerGraph,
};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};
pub use trace::{
    trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
    ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
    TracePositions, TraceStore, TraceWriter,
};
pub use vindex::{open_inference_vindex, predict_kquant, FfnL1Cache, WalkFfn, WalkFfnConfig};

/// Stable, application-facing inference imports.
///
/// New downstream code should prefer this module over broad crate-root
/// glob imports. The crate root remains source-compatible while the public
/// surface is gradually narrowed.
pub mod prelude {
    pub use crate::{
        default_backend, generate, generate_streaming, generate_with_sampling, load_model_dir,
        load_tokenizer, open_inference_vindex, predict, predict_kquant, resolve_model_path,
        try_generate, try_generate_streaming, try_generate_with_sampling, wrap_chat_prompt,
        wrap_prompt_raw, wrap_with_vindex_template, ChatWrap, ComputeBackend, Detokenizer,
        EosConfig, GenerateError, GenerateResult, InferenceError, ModelWeights, Sampler,
        SamplingConfig, WalkFfn, WalkFfnConfig,
    };
    pub use larql_compute::CpuBackend;
}

/// Mechanistic-interpretability and research-facing imports.
///
/// These APIs are intentionally more experimental than [`prelude`]. Grouping
/// them here makes that boundary visible without breaking existing crate-root
/// users in one large move.
///
/// `KvEngine`, `EngineInfo`, and `DecodeStageSummary` are defined in
/// this crate's [`kv_engine`](crate::kv_engine) module and re-exported
/// at the crate root. Concrete engine implementations
/// (`MarkovResidualEngine`, `UnlimitedContextEngine`, `StandardEngine`,
/// `NoCacheEngine`, `TurboQuantEngine`, `ApolloEngine`) plus
/// `EngineKind` and accuracy helpers (`compare_hidden`,
/// `cosine_similarity`, `kl_divergence`, …) live in the `larql-kv`
/// crate — depend on it directly when you need concrete engines.
pub mod research {
    // Source directly from subpaths so this curated surface keeps working
    // even when individual root re-exports are dropped. Kept as a single
    // import block per source module so the surface is easy to scan.
    pub use crate::forward::{
        apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals,
        capture_ffn_activation_matrix, capture_residuals, capture_spec_residuals,
        estimate_ffn_covariance, forward_from_layer, forward_raw_logits, forward_to_layer,
        hidden_to_raw_logits, infer_patched, infer_patched_q4k, logit_lens_top1,
        predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
        predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router,
        predict_with_strategy, run_memit, run_memit_with_target_opt, trace_forward,
        trace_forward_full, trace_forward_with_ffn, walk_trace_from_residuals, InferPatchedResult,
        InferenceWeights, KnnOverride, LayerAttentionCapture, LayerMode, MemitFact,
        MemitFactResult, MemitResult, PredictResult, PredictResultWithAttention,
        PredictResultWithResiduals, RawForward, SpecCapture, TargetDelta, TargetDeltaOpts,
        TraceResult, KNN_COSINE_THRESHOLD,
    };
    pub use crate::layer_graph::{
        predict_honest, predict_pipeline, predict_with_graph, predict_with_graph_vindex_logits,
        trace_with_graph, AttentionCache, TemplatePattern, TemplateUniverse,
    };
    pub use crate::trace::{
        trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
        ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
        TracePositions, TraceStore, TraceWriter,
    };
}

#[cfg(test)]
mod factory_tests {
    //! Coverage for the engine/backend factory functions at the crate root.
    //!
    //! Each factory exists so engines can construct themselves without the
    //! caller branching on `#[cfg(feature = "metal")]`. The tests verify
    //! that the returned trait object is the CPU backend on the default
    //! build (no `metal` feature on the test runner CI matrix) and that
    //! each factory's pipeline-back name plumbs through.
    //!
    //! On Apple Silicon with the `metal` feature these tests still pass —
    //! the factories fall back to CPU when `MetalBackend::new()` returns
    //! `None`, and on metal-capable hardware the returned backend just
    //! reports a different name. The assertions are deliberately scoped
    //! to "factory returned a usable backend" rather than "the backend is
    //! CPU" so both build configurations pass.
    use super::*;
    use larql_models::Activation as ArchActivation;

    #[test]
    fn cpu_engine_backend_returns_named_backend() {
        let backend = cpu_engine_backend();
        // CpuBackend's name starts with "cpu" — exact suffix varies with
        // BLAS / kernel selection (e.g. "cpu (BLAS + C Q4 kernel)").
        assert!(backend.as_compute().name().starts_with("cpu"));
    }

    #[test]
    fn cpu_async_engine_backend_returns_named_backend() {
        let backend = cpu_async_engine_backend();
        let name = <dyn AsyncComputeBackend as larql_compute::ComputeBackend>::name(&*backend);
        assert!(name.starts_with("cpu"));
    }

    #[test]
    fn default_engine_backend_constructs() {
        // Returns Metal when the feature + hardware align, CPU otherwise.
        // The factory is exercised either way — we only check it returns
        // a backend with a non-empty name.
        let backend = default_engine_backend();
        assert!(!backend.as_compute().name().is_empty());
    }

    #[test]
    fn default_async_engine_backend_constructs() {
        let backend = default_async_engine_backend();
        assert!(
            !<dyn AsyncComputeBackend as larql_compute::ComputeBackend>::name(&*backend).is_empty()
        );
    }

    #[test]
    fn default_compute_backend_constructs() {
        let backend = default_compute_backend();
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn activation_from_arch_maps_gelu_tanh() {
        // Use a tiny ModelWeights so we have a real ModelArchitecture.
        let weights = test_utils::make_test_weights();
        // make_test_weights uses TinyModelArch which reports Silu — verify
        // the non-tanh branch maps to Silu.
        let act = activation_from_arch(&*weights.arch);
        assert!(matches!(act, larql_compute::Activation::Silu));

        // For the GeluTanh branch we need an arch that reports it. Build
        // a Gemma 3 arch via the existing fixture (gemma3 uses GeluTanh).
        let gemma = test_utils::make_gemma3_test_weights();
        assert!(matches!(gemma.arch.activation(), ArchActivation::GeluTanh));
        assert!(matches!(
            activation_from_arch(&*gemma.arch),
            larql_compute::Activation::GeluTanh
        ));
    }
}
