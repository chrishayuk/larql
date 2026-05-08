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
//! - [`attention::KvCache`] — `get_layer` / `set_layer` /
//!   `clone_layer_position_range` for KV-cache surgery (e.g. lazarus's
//!   `prefill_inject` and `kv_inject_test`).
//!
//! See `examples/mech_interp_demo.rs` for an end-to-end walkthrough on
//! synthetic weights (no vindex required).

#![allow(
    deprecated,
    dead_code,
    private_interfaces,
    unused_imports,
    unused_mut,
    unused_variables,
    clippy::doc_nested_refdefs,
    clippy::duplicated_attributes,
    clippy::blocks_in_conditions,
    clippy::collapsible_if,
    clippy::doc_overindented_list_items,
    clippy::erasing_op,
    clippy::if_same_then_else,
    clippy::identity_op,
    clippy::items_after_test_module,
    clippy::large_enum_variant,
    clippy::let_and_return,
    clippy::manual_find,
    clippy::map_identity,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::question_mark,
    clippy::single_char_add_str,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec
)]

extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod chat;
pub mod engines;
pub mod error;
pub mod experts;
pub mod ffn;
pub mod forward;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod residual;
pub mod residual_diff;
pub mod tokenizer;
pub mod trace;
pub mod trie;
pub mod vindex;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports (from larql-compute).
pub use larql_compute::cpu::ops::moe::{
    cpu_moe_forward, run_single_expert, run_single_expert_with_norm,
};
pub use larql_compute::Activation as ComputeActivation;
pub use larql_compute::CpuBackend;
pub use larql_compute::MoeLayerWeights;
pub use larql_compute::QuantFormat;
pub use larql_compute::{
    cpu_backend, default_backend, dot_proj_gpu, matmul_gpu, ComputeBackend, MatMulOp,
};

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(
    arch: &dyn larql_models::ModelArchitecture,
) -> larql_compute::Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    }
}
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use larql_compute::MetalBackend;

// Re-export essentials at crate root.
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
    ShardConfig, SparseFfn, WeightFfn,
};
pub use forward::{
    apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals,
    capture_ffn_activation_matrix, capture_residuals, capture_spec_residuals,
    estimate_ffn_covariance, forward_from_layer, forward_raw_logits, forward_to_layer,
    generate_cached_constrained, hidden_to_raw_logits, infer_patched, infer_patched_q4k,
    logit_lens_top1, predict, predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
    predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router, predict_with_strategy,
    run_memit, run_memit_with_target_opt, trace_forward, trace_forward_full,
    trace_forward_with_ffn, walk_trace_from_residuals, InferPatchedResult, InferenceWeights,
    KnnOverride, LayerAttentionCapture, LayerMode, MemitFact, MemitFactResult, MemitResult,
    PredictResult, PredictResultWithAttention, PredictResultWithResiduals, RawForward, SpecCapture,
    TargetDelta, TargetDeltaOpts, TraceResult, KNN_COSINE_THRESHOLD,
};
pub use layer_graph::{
    build_adaptive_graph,
    detect_template,
    generate,
    generate_streaming,
    generate_with_sampling,
    // Expert grid generation
    grid::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, generate_with_remote_moe,
        generate_with_remote_moe_batch, GridGenerateResult,
    },
    hybrid::predict_hybrid,
    predict_honest,
    predict_pipeline,
    predict_split_cached,
    predict_split_pass,
    predict_with_graph,
    predict_with_graph_vindex_logits,
    trace_with_graph,
    AttentionCache,
    CachedLayerGraph,
    // Multi-turn chat session
    ChatMLRenderer,
    ChatSession,
    DenseLayerGraph,
    // Generation building blocks (EOS, detok, sampling)
    Detokenizer,
    EosConfig,
    GemmaRenderer,
    GenerateResult,
    GuidedWalkLayerGraph,
    // Production
    LayerGraph,
    LayerOutput,
    Llama3Renderer,
    PerLayerGraph,
    PipelinedLayerGraph,
    Sampler,
    SamplingConfig,
    // Analysis/validation
    TemplatePattern,
    TemplateUniverse,
    TurnRenderer,
    WalkLayerGraph,
};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};
pub use trace::{
    trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
    ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
    TracePositions, TraceStore, TraceWriter,
};
pub use vindex::{open_inference_vindex, predict_q4k, FfnL1Cache, WalkFfn, WalkFfnConfig};

// Engine re-exports.
pub use engines::accuracy::{
    compare_hidden, cosine_similarity, js_divergence, kl_divergence, mse, softmax, HiddenAccuracy,
};
pub use engines::markov_residual::MarkovResidualEngine;
pub use engines::unlimited_context::UnlimitedContextEngine;
pub use engines::{EngineInfo, EngineKind, KvEngine};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
