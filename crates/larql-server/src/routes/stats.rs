//! GET /v1/stats

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel, ServedModel};

/// `/v1/stats` for a VINDEX3 binding.
///
/// A V3 container is an executable program, not a feature index, so the
/// V2 block's vocabulary (features, bands, extract level, q4k caches)
/// has nothing to report. What it *can* state is the program's own
/// shape, read from the opened plan — which is also the only authority
/// that could answer, since a V3 container carries no architecture
/// registry entry.
///
/// The `server` block matters more here than the model block: it is the
/// only surface carrying the N1 continuation counters, and before this
/// existed a V3-only server answered `/v1/stats` with 404 — the
/// instrumentation was unreachable exactly where V3 runs.
fn build_v3_stats(model: &crate::vindex3::V3Model) -> serde_json::Value {
    let plan = model.runtime.plan();
    serde_json::json!({
        "model": model.id,
        "generation": 3,
        "mode": "full",
        "component": plan.component,
        "layers": plan.layers.len(),
        "hidden_size": plan.embedding.as_ref().map(|e| e.table.shape[1]),
        "vocab_size": plan.embedding.as_ref().map(|e| e.table.shape[0]),
        "has_output_head": plan.output.is_some(),
        "path": model.path.display().to_string(),
        "loaded": {
            "browse": false,
            "inference": true,
            "ffn_service": false,
            "embed_service": false,
        },
    })
}

fn build_stats(model: &LoadedModel) -> serde_json::Value {
    let config = &model.config;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    let features_per_layer = if !config.layers.is_empty() {
        config.layers[0].num_features
    } else {
        0
    };

    let layer_bands = config.layer_bands.as_ref().map(|b| {
        serde_json::json!({
            "syntax": [b.syntax.0, b.syntax.1],
            "knowledge": [b.knowledge.0, b.knowledge.1],
            "output": [b.output.0, b.output.1],
        })
    });

    let has_inference = config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All
        || config.has_model_weights;

    let mode = if model.embed_only {
        "embed-service"
    } else if model.ffn_only {
        "ffn-service"
    } else {
        "full"
    };

    let layer_latency: Vec<serde_json::Value> = model
        .layer_latency_tracker
        .snapshot()
        .iter()
        .map(|l| {
            serde_json::json!({
                "layer": l.layer,
                "avg_ms": l.avg_ms,
                "p99_ms": l.p99_ms,
            })
        })
        .collect();

    serde_json::json!({
        "model": config.model,
        "family": config.family,
        "mode": mode,
        "layers": config.num_layers,
        "features": total_features,
        "features_per_layer": features_per_layer,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "extract_level": config.extract_level.to_string(),
        "dtype": config.dtype.to_string(),
        "layer_bands": layer_bands,
        "layer_latency": layer_latency,
        "loaded": {
            "browse": !model.embed_only,
            "inference": has_inference && !model.infer_disabled,
            "ffn_service": !model.embed_only,
            "embed_service": true,
        },
    })
}

/// Per-layer FFN weight byte accounting for the DEC movement-ratio
/// denominator (`dec/movement_ratio`, docs/dec-funnel.md §1).
///
/// `per_layer_dense_bytes[l]` is the exact mmap byte length of layer `l`'s
/// interleaved k-quant gate+up+down weights — the bytes `/v1/walk-ffn`
/// touches for one token on that layer — or `null` where the vindex has no
/// interleaved k-quant data. The `moe` block echoes routing metadata from
/// the vindex config; `per_expert_bytes` is probed from already-loaded model
/// weights (never forces a weight load) and omitted when unavailable.
fn ffn_weights_json(
    model: &LoadedModel,
    base: &larql_vindex::VectorIndex,
) -> Option<serde_json::Value> {
    let config = &model.config;
    let per_layer: Vec<Option<u64>> = (0..config.num_layers)
        .map(|layer| {
            base.interleaved_kquant_layer_data(layer)
                .map(|slices| slices.iter().map(|(bytes, _)| bytes.len() as u64).sum())
        })
        .collect();
    let has_any_dense = per_layer.iter().any(|b| b.is_some());

    let moe = model
        .weights
        .get()
        .and_then(|rw| rw.read().ok())
        .and_then(|w| {
            let arch = &*w.arch;
            if !arch.is_moe() {
                return None;
            }
            // Probe the first OWNED expert entry per layer, not entry 0: a
            // shard started with `--experts 64-127` has no entry 0, and the
            // pre-fix probe reported `per_expert_bytes: null` on a healthy
            // shard (silently missing fact, not an error).
            let per_expert_bytes: Option<u64> = if w.has_per_layer_ffn() {
                (0..config.num_layers).find_map(|l| {
                    w.first_layer_entry_bytes(l)
                        .map(|(gu, dn)| (gu.len() + dn.len()) as u64)
                })
            } else {
                None
            };
            Some(serde_json::json!({
                "num_experts": arch.num_experts(),
                "top_k": arch.num_experts_per_token(),
                "moe_intermediate_size": arch.moe_intermediate_size(),
                "per_expert_bytes": per_expert_bytes,
            }))
        });

    if !has_any_dense && moe.is_none() {
        return None;
    }
    Some(serde_json::json!({
        "per_layer_dense_bytes": per_layer,
        "moe": moe,
    }))
}

/// Async wrapper for the Q4K cache + W2 surface. The base
/// `build_stats` stays sync so the existing single-/multi-model
/// handlers don't change shape; this overlay merges the `q4k_ffn`
/// block in once we have an `.await`-friendly read guard.
async fn add_q4k_ffn(model: &LoadedModel, mut stats: serde_json::Value) -> serde_json::Value {
    let p = model.patched.read().await;
    let (slots, bytes) = p.base.kquant_ffn_cache_stats();
    let has_fm = p.base.has_down_features_kquant();
    let ffn_weights = ffn_weights_json(model, &p.base);
    if let Some(obj) = stats.as_object_mut() {
        obj.insert(
            "q4k_ffn".into(),
            serde_json::json!({
                "cache_slots": slots,
                "cache_bytes": bytes,
                "feature_major_down": has_fm,
            }),
        );
        if let Some(w) = ffn_weights {
            obj.insert("ffn_weights".into(), w);
        }
    }
    stats
}

/// The server-level block: process counters plus every bounded
/// per-client store the maintenance sweeper keeps in check — the
/// operational counterpart of the P1 eviction fix and the N1 KV cache.
async fn server_block(state: &AppState) -> serde_json::Value {
    serde_json::json!({
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "requests_served": state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        "sessions": {
            "active": state.sessions.session_count().await,
            "ttl_secs": state.sessions.ttl().as_secs(),
        },
        "responses_stored": {
            "entries": state.responses.len(),
            "capacity": crate::response_store::MAX_STORED_RESPONSES,
        },
        "v3_kv": {
            "enabled": state.v3_kv.enabled(),
            "entries": state.v3_kv.len(),
            "capacity": state.v3_kv.max_entries(),
            "ttl_secs": state.v3_kv.ttl().as_secs(),
            // `hits` = a resident state was found; `resumptions` = the
            // state also passed the exact ids-prefix check and prefill
            // work was skipped. The difference is the prefix-stability
            // gap under real request construction.
            "hits": state.v3_kv.hits(),
            "misses": state.v3_kv.misses(),
            "resumptions": state.v3_kv.resumptions(),
            // A hit whose state's recorded continuation authority refused
            // to resume under the model's (C4); served by a fresh prefill.
            "refusals": state.v3_kv.refusals(),
            "reused_tokens_total": state.v3_kv.reused_tokens_total(),
        },
    })
}

/// Merge the server-level block into a per-model stats payload.
async fn add_server_block(state: &AppState, mut stats: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = stats.as_object_mut() {
        obj.insert("server".into(), server_block(state).await);
    }
    stats
}

#[utoipa::path(
    get,
    path = "/v1/stats",
    tag = "browse",
    responses(
        (status = 200, description = "Model + vindex statistics", body = crate::openapi::schemas::StatsResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let stats = match state.served_or_err(None)? {
        ServedModel::V2(model) => add_q4k_ffn(&model, build_stats(&model)).await,
        ServedModel::V3(model) => build_v3_stats(&model),
    };
    Ok(Json(add_server_block(&state, stats).await))
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/stats",
    tag = "browse",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    responses(
        (status = 200, body = crate::openapi::schemas::StatsResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_stats_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let stats = match state.served_or_err(Some(&model_id))? {
        ServedModel::V2(model) => add_q4k_ffn(&model, build_stats(&model)).await,
        ServedModel::V3(model) => build_v3_stats(&model),
    };
    Ok(Json(add_server_block(&state, stats).await))
}
