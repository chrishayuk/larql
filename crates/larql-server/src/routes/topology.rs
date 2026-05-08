//! `GET /v1/expert/topology` — advertise this shard's expert ownership range.
//!
//! Returns the expert ID range `[owned_start, owned_end]` (inclusive) that
//! this server was launched with via `--experts START-END`. Clients use this
//! to build the shard map dynamically instead of having it baked into the
//! `--moe-shards` flag.
//!
//! Returns HTTP 404 when the server was not launched with `--experts` (i.e.,
//! it owns all experts or is not operating as an expert shard).

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::state::AppState;

#[derive(Serialize, ToSchema)]
pub struct TopologyResponse {
    /// Model identifier (e.g. `"google/gemma-4-26B-A4B-it"`).
    pub model_id: String,
    /// Total number of experts in the model (0 for non-MoE models).
    pub num_experts: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// First expert ID owned by this shard (inclusive).
    pub owned_start: usize,
    /// Last expert ID owned by this shard (inclusive).
    pub owned_end: usize,
}

#[utoipa::path(
    get,
    path = "/v1/expert/topology",
    tag = "browse",
    responses(
        (status = 200, description = "Expert ownership range for this shard", body = TopologyResponse),
        (status = 404, description = "Server was not launched with --experts"),
    ),
)]
pub async fn handle_topology(
    State(state): State<Arc<AppState>>,
) -> Result<Json<TopologyResponse>, StatusCode> {
    let model = state
        .model_or_err(None)
        .map_err(|_| StatusCode::NOT_FOUND)?;

    // 404 if this server was not launched with --experts (no shard filter set).
    let (start, end_excl) = model.expert_filter.ok_or(StatusCode::NOT_FOUND)?;

    let num_experts = model
        .config
        .model_config
        .as_ref()
        .and_then(|m| m.moe.as_ref())
        .map(|m| m.num_experts)
        .unwrap_or(0);

    Ok(Json(TopologyResponse {
        model_id: model.id.clone(),
        num_experts,
        num_layers: model.config.num_layers,
        owned_start: start,
        owned_end: end_excl.saturating_sub(1), // convert exclusive→inclusive for display
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `owned_end` should be `(end_excl - 1)` to convert the half-open
    /// `expert_filter` tuple `(start, end_excl)` into the inclusive
    /// `[owned_start, owned_end]` range the wire format advertises.
    #[test]
    fn topology_response_inclusive_end() {
        let resp = TopologyResponse {
            model_id: "test/model".into(),
            num_experts: 128,
            num_layers: 30,
            owned_start: 0,
            owned_end: (32usize).saturating_sub(1),
        };
        assert_eq!(resp.owned_start, 0);
        assert_eq!(resp.owned_end, 31);
        // Round-trip via serde to confirm the field names match the
        // documented wire shape.
        let json = serde_json::to_value(&resp).expect("serialise topology");
        assert_eq!(json["owned_start"], 0);
        assert_eq!(json["owned_end"], 31);
        assert_eq!(json["num_experts"], 128);
        assert_eq!(json["num_layers"], 30);
        assert_eq!(json["model_id"], "test/model");
    }

    /// Edge case: `expert_filter = Some((0, 1))` (single-expert shard) →
    /// `owned_end = 0`, not underflow.
    #[test]
    fn topology_response_single_expert_shard() {
        let resp = TopologyResponse {
            model_id: "x".into(),
            num_experts: 1,
            num_layers: 1,
            owned_start: 0,
            owned_end: (1usize).saturating_sub(1),
        };
        assert_eq!(resp.owned_end, 0);
    }

    /// Saturating sub guards against the (illegal but possible) `(0, 0)`
    /// `expert_filter` setting — should not panic and should give 0, not
    /// usize::MAX.
    #[test]
    fn topology_response_zero_filter_saturates() {
        let resp = TopologyResponse {
            model_id: "x".into(),
            num_experts: 0,
            num_layers: 0,
            owned_start: 0,
            owned_end: (0usize).saturating_sub(1),
        };
        assert_eq!(resp.owned_end, 0);
    }
}
