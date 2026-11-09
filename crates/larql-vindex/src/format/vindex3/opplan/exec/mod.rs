//! The plan interpreter (V3-G5b-2 Stage A, V3-G5b-3b seam).
//!
//! Executes a [`ComponentOpPlan`] — and **nothing else**. Every argument
//! comes from the plan (which came from the container); every operand
//! loads through the closure-verified `object → representation → segment`
//! path; every judged enum is matched exhaustively so an unjudged variant
//! is a compile error, not a guess. There is no family name, no layer
//! arithmetic, no HF tensor name, and no default anywhere in this module.
//!
//! This file owns *meaning*: operation ordering, residual ordering, layer
//! traversal, whether an optional operation exists, and how position and
//! span policy dispatch. A [`PlanBackend`] owns only arithmetic. One
//! interpreter drives every backend, so a second implementation cannot
//! quietly become a second reading of the model — see [`backend`].
//!
//! The trace mirrors the production forward's hook points
//! (`post_attention` = after the attention residual add, `post_layer` =
//! after the FFN residual add) so parity can compare layer by layer
//! against a checkpoint-driven oracle.

pub mod accounting;
pub mod address;
pub mod attention_residual;
pub mod attested_fidelity;
pub mod backend;
pub mod continuation;
pub mod continuation_authority;
pub mod continuation_handoff;
pub mod continuation_identity;
pub mod continuation_registry;
pub mod controls;
pub mod conv_qkv;
pub mod cpu;
pub mod decode;
pub mod dense_ffn;
pub mod dense_rows;
pub mod device;
pub mod device_refusal;
mod experts;
pub mod fidelity_carriage;
pub mod gated_delta;
pub mod head_replay;
pub mod hyper_connection;
pub mod identity;
pub mod intervene;
pub mod intervene_heads;
pub mod kda;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod kda_metal;
pub mod kernels;
pub mod kimi_kda_layer;
pub mod kimi_mla_layer;
pub mod kimi_moe_block;
pub mod kimi_router;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod kimi_source;
pub mod kv;
pub mod kv_view;
pub mod lowering;
pub mod mamba2;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod metal_lowered;
pub mod mla;
pub mod narrow;
pub mod observe;
pub mod observe_heads;
pub mod observe_lens;
pub mod observe_stats;
pub mod operands;
pub mod payload_prefix;
pub mod portability;
pub mod prefetch;
pub mod prepared;
pub mod production;
pub mod profile;
pub mod provenance;
mod provider_identity;
pub mod quantise;
pub mod realization;
pub mod reference;
pub mod requirements;
pub mod routed_experts;
pub mod routing_trace;
pub mod sequence;
pub mod site_capture;
pub mod stack;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod stack_metal;
pub mod stages;
pub mod timing;
pub mod token;
pub mod weights;

#[cfg(test)]
mod tests;

use super::ComponentOpPlan;

use rayon::prelude::*;

mod attention_ops;
mod batch_site;
mod layer_exec;
pub(crate) mod position_logits;
mod streaming;
mod trace;
mod traverse;
pub use attention_ops::*;
pub use batch_site::*;
use layer_exec::*;
pub use position_logits::*;
pub use streaming::*;
pub use trace::*;
pub(super) use traverse::*;
