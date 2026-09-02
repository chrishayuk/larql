//! Model architecture trait and shared types.
//!
//! Every model architecture implements [`ModelArchitecture`]. That trait
//! describes *what the model is* — tensor key patterns, norm behaviour,
//! activation functions, scaling — without any compute dependencies.
//!
//! Layout, from data to behaviour:
//!
//! | module | holds |
//! |---|---|
//! | [`model_config`] | [`ModelConfig`], the parsed `config.json` |
//! | [`rope_types`] / [`layer_types`] | HF selector strings, declared once |
//! | [`rope`] | the `rope_scaling` block and its per-family parameter sets |
//! | [`norm`] / [`activation`] / [`experts`] | the behavioural enums |
//! | [`architecture`] | the [`ModelArchitecture`] trait itself |
//!
//! The split exists so a trait default can read a config fact in one short
//! body. Where a behaviour is stated in `config.json`, it is resolved here for
//! every architecture rather than overridden per family — see
//! [`architecture`]'s header for why that ordering is load-bearing.

pub mod activation;
pub mod architecture;
pub mod attention_gate;
pub mod attention_sinks;
pub mod conv_qkv_attn;
pub mod experts;
pub mod interleave;
pub mod layer_types;
pub mod linear_attn;
pub mod mamba2;
pub mod mla;
pub mod model_config;
pub mod moe_router;
pub mod nonfinite_json;
pub mod norm;
pub mod position;
pub mod rope;
pub mod rope_types;

pub use activation::{Activation, FfnType};
pub use architecture::{
    default_position_policy_for_layer, score_scale_from_query_pre_attn_scalar, ModelArchitecture,
};
pub use attention_gate::{
    AttentionGateSpec, GateActivation, GateCombine, GatePlacement, GateSource,
};
pub use attention_sinks::AttentionSinkSpec;
pub use conv_qkv_attn::{ConvQkvAttnGeometry, ConvQkvDialect, ConvQkvProvenance};
pub use experts::{
    ExpertFormat, ExpertGatePolicy, ExpertRoutingPolicy, GateUpBranch, GateUpLayout,
};
pub use interleave::{
    read_declared_interleave, DeclaredInterleave, InterleaveEncoding, InterleaveError,
    InterleaveProvenance, InterleaveScope, LayerIndexBase, LayerKind, RecurrenceFamily,
    ResolvedInterleave,
};
pub use layer_types::{
    LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION,
    LAYER_TYPE_WINDOW_ATTENTION,
};
pub use linear_attn::{KdaGeometry, LAYER_TYPE_UNRESOLVED_INTERLEAVE};
pub use mamba2::{DtBound, Mamba2Dialect, Mamba2FamilyDefault, Mamba2Geometry, Mamba2Provenance};
pub use mla::MlaGeometry;
pub use model_config::ModelConfig;
pub use moe_router::MoeRouterKind;
pub use norm::{EmbeddingNorm, NormSpec, NormType, ParameterFreeQkNorm, PostNormEps, QkNormScope};
pub use position::{
    mrope_axis_table, DeclaredRopeScaling, PositionPolicy, RotaryFrequencyBasis,
    ROPE_PAIRING_INTERLEAVED,
};
pub use rope::{Llama3RopeScaling, RopeScaling, YarnRopeScaling};
pub use rope_types::{
    POSITION_EMBEDDING_TYPE_ROPE, ROPE_TYPE_DEFAULT, ROPE_TYPE_LINEAR, ROPE_TYPE_LLAMA3,
    ROPE_TYPE_PROPORTIONAL, ROPE_TYPE_YARN,
};

#[cfg(test)]
mod kda_geometry_tests;
#[cfg(test)]
mod mamba2_tests;
#[cfg(test)]
mod mla_geometry_tests;
#[cfg(test)]
mod tests;
