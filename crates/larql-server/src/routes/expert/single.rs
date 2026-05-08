//! `POST /v1/expert/{layer}/{expert_id}` — single expert dispatch.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;

use crate::error::ServerError;
use crate::state::AppState;

use super::{SingleExpertRequest, SingleExpertResponse};

/// Run one expert's gate/up/down compute on the given residual. Used by both
/// the HTTP handler below and the gRPC expert path in `grpc_expert.rs`.
///
/// Ownership precedence: `unit_filter` (`--units` JSON manifest) →
/// `expert_filter` (`--experts START-END`, layer-uniform) → all experts.
/// Mismatched ownership returns 400 rather than silently routing.
pub fn run_expert(
    state: &AppState,
    layer: usize,
    expert_id: usize,
    residual: &[f32],
) -> Result<Vec<f32>, ServerError> {
    let model = state.model_or_err(None)?;

    if let Some(units) = model.unit_filter.as_ref() {
        if !units.contains(&(layer, expert_id)) {
            return Err(ServerError::BadRequest(format!(
                "(layer={layer}, expert={expert_id}) not owned by this shard \
                 (--units manifest defines its ownership set)"
            )));
        }
    } else if let Some((start, end_excl)) = model.expert_filter {
        if expert_id < start || expert_id >= end_excl {
            let end_inclusive = end_excl.saturating_sub(1);
            return Err(ServerError::BadRequest(format!(
                "expert {expert_id} not owned by this shard (owns {start}–{end_inclusive})"
            )));
        }
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let arch = &*weights.arch;

    if !arch.is_hybrid_moe() {
        return Err(ServerError::BadRequest(
            "model is not a hybrid MoE — no expert endpoints available".into(),
        ));
    }

    let hidden = model.config.hidden_size;
    if residual.len() != hidden {
        return Err(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {hidden}",
            residual.len()
        )));
    }

    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);

    // Resolve this expert's per-expert byte slice. Per-layer Q4_K vindexes
    // expose entries at `layers/{layer}/{expert}/...`; legacy BF16 vindexes
    // expose a monolithic `packed_experts_{gate_up,down}_key` blob that we
    // slice by stride. Either way we feed `run_single_expert*` exactly one
    // expert's bytes — no monolith arithmetic in the compute path.
    let (gate_up_bytes, down_bytes, format) = if weights.has_per_layer_ffn() {
        let (gu, dn) = weights
            .get_layer_entry_bytes(layer, expert_id)
            .ok_or_else(|| {
                ServerError::Internal(format!(
                    "per-layer entry missing for layer {layer} expert {expert_id}"
                ))
            })?;
        (gu, dn, larql_inference::QuantFormat::Q4_K)
    } else {
        let gate_up_key = arch.packed_experts_gate_up_key(layer).ok_or_else(|| {
            ServerError::BadRequest(format!("no MoE gate/up weights for layer {layer}"))
        })?;
        let down_key = arch.packed_experts_down_key(layer).ok_or_else(|| {
            ServerError::BadRequest(format!("no MoE down weights for layer {layer}"))
        })?;
        let gu_all = weights.get_packed_bytes(&gate_up_key).ok_or_else(|| {
            ServerError::Internal(format!("gate_up bytes missing for layer {layer}"))
        })?;
        let dn_all = weights.get_packed_bytes(&down_key).ok_or_else(|| {
            ServerError::Internal(format!("down bytes missing for layer {layer}"))
        })?;
        let gu_stride = 2 * inter * hidden * 2; // BF16 = 2 bytes
        let dn_stride = hidden * inter * 2;
        let gu_start = expert_id * gu_stride;
        let dn_start = expert_id * dn_stride;
        if gu_start + gu_stride > gu_all.len() || dn_start + dn_stride > dn_all.len() {
            return Err(ServerError::Internal(format!(
                "expert {expert_id} byte range out of bounds for layer {layer}"
            )));
        }
        (
            &gu_all[gu_start..gu_start + gu_stride],
            &dn_all[dn_start..dn_start + dn_stride],
            larql_inference::QuantFormat::BF16,
        )
    };

    let output = if let Some(norm_key) = arch.moe_pre_experts_norm_key(layer) {
        let pre_experts_norm = weights
            .vectors
            .get(&norm_key)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        larql_inference::run_single_expert_with_norm(
            residual,
            gate_up_bytes,
            down_bytes,
            inter,
            pre_experts_norm,
            arch.norm_weight_offset(),
            arch.norm_eps(),
            format,
            activation,
        )
    } else {
        larql_inference::run_single_expert(
            residual,
            gate_up_bytes,
            down_bytes,
            inter,
            format,
            activation,
        )
    };

    Ok(output)
}

#[utoipa::path(
    post,
    path = "/v1/expert/{layer}/{expert_id}",
    tag = "expert",
    params(
        ("layer" = usize, Path),
        ("expert_id" = usize, Path),
    ),
    request_body = SingleExpertRequest,
    responses(
        (status = 200, description = "Expert output vector", body = SingleExpertResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_expert(
    State(state): State<Arc<AppState>>,
    Path((layer, expert_id)): Path<(usize, usize)>,
    Json(req): Json<SingleExpertRequest>,
) -> Result<Json<SingleExpertResponse>, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    let output =
        tokio::task::spawn_blocking(move || run_expert(&state, layer, expert_id, &req.residual))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SingleExpertResponse { output, latency_ms }))
}
