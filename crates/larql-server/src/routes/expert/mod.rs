//! `/v1/expert*` endpoints — remote expert dispatch for hybrid-MoE models.
//!
//! Sharding model: a server started with `--experts START-END` (or
//! `--units PATH` for fine-grained per-(layer, expert) ownership) hosts a
//! contiguous slice of the expert table. The inference client routes per
//! expert call to whichever shard owns the (layer, expert_id) pair instead
//! of running all experts locally.
//!
//! ## Endpoints (one file per concern)
//!
//! - `POST /v1/expert/{layer}/{expert_id}` — single expert; see [`single`].
//! - `POST /v1/expert/batch` — pre-2026-05-01 multi-expert wire (one residual
//!   per item); see [`batch_legacy`].
//! - `POST /v1/experts/layer-batch[-f16]` — current MoE wire: one residual +
//!   K (expert_id, weight) pairs → router-weighted sum; see [`layer_batch`].
//!
//! ## Compute paths
//!
//! - [`cpu`] — `run_experts_cpu_batch`, the rayon CPU dispatch with hoisted
//!   pre-norm and shared per-thread `ExpertScratch`. The default path.
//! - [`metal`] — `run_experts_metal_batch`, GPU dispatch behind the
//!   `metal-experts` feature. Currently opt-in via `LARQL_USE_METAL_EXPERTS`
//!   while the inter=704 accuracy bug is being debugged (see ROADMAP).
//! - [`warmup`] — eager-build helpers for the HNSW unit cache and the
//!   Metal expert buffer cache, called from boot.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub mod batch_legacy;
pub mod cpu;
pub mod layer_batch;
pub mod metal;
pub mod multi_layer_batch;
pub mod single;
pub mod warmup;

// ── Public re-exports ─────────────────────────────────────────────────────────
//
// Preserve the historical `routes::expert::*` import shape for callers
// (`grpc_expert.rs`, `main.rs`, `routes/mod.rs`, integration tests).

pub use batch_legacy::handle_expert_batch;
pub use cpu::run_experts_cpu_batch;
pub use layer_batch::{handle_experts_layer_batch, handle_experts_layer_batch_f16};
#[cfg(feature = "metal-experts")]
pub use metal::run_experts_metal_batch;
pub use multi_layer_batch::{
    handle_experts_multi_layer_batch, handle_experts_multi_layer_batch_q8k,
};
pub use single::{handle_expert, run_expert};
pub use warmup::warmup_hnsw_unit_cache;
#[cfg(feature = "metal-experts")]
pub use warmup::warmup_metal_expert_cache;

// ── Request / response types ──────────────────────────────────────────────────
//
// Kept in `mod.rs` because they're shared across the single + batch_legacy
// handlers and trivially small.

#[derive(Deserialize, ToSchema)]
pub struct SingleExpertRequest {
    pub residual: Vec<f32>,
}

#[derive(Serialize, ToSchema)]
pub struct SingleExpertResponse {
    pub output: Vec<f32>,
    pub latency_ms: f64,
}

#[derive(Deserialize, ToSchema)]
pub struct BatchExpertItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize, ToSchema)]
pub struct BatchExpertRequest {
    pub requests: Vec<BatchExpertItem>,
}

#[derive(Serialize, ToSchema)]
pub struct BatchExpertResult {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}

#[derive(Serialize, ToSchema)]
pub struct BatchExpertResponse {
    pub results: Vec<BatchExpertResult>,
    pub latency_ms: f64,
}
