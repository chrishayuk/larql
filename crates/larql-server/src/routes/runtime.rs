//! `GET /v1/runtime` — server + model + backend + memory + performance
//! snapshot. The canonical introspection surface for a LARQL client
//! that isn't the CLI: a desktop menu-bar app, a status dashboard, or
//! `curl` during development.
//!
//! Deliberately distinct from the two existing introspection routes:
//!
//! - `/v1/models` says what models *exist* (OpenAI-compatible listing).
//! - `/v1/stats` says what one *bound vindex* looks like (features,
//!   layers, quant caches) — 404s when no model is loaded.
//! - `/v1/runtime` says what this LARQL *process* is doing right now:
//!   is it up, how long has it been up, what (if anything) is
//!   currently generating, and how fast the last request ran.
//!   `model` is `null` rather than a 404 when no model is bound (zero
//!   models, or an ambiguous multi-model server) — the process facts
//!   are still worth reporting even when the model facts aren't.
//!
//! Every `performance` field is a snapshot of a real measurement taken
//! at the generation call site ([`crate::runtime_stats`]) — this
//! handler never recomputes a rate itself, so `/v1/runtime` cannot
//! silently drift from what a client actually experienced.
//!
//! Read-only by design (first slice): no load/unload, and no
//! `/v1/{model_id}/runtime` multi-model variant yet — `performance`
//! and `generation` are server-wide, not per-model, so a per-id
//! variant would only change the `model` block. Follow-up if that
//! turns out to matter once multi-model serving has real users of
//! this endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::state::{AppState, ServedModel};

const STATUS_READY: &str = "ready";
const FORMAT_VINDEX2: &str = "vindex2";
const FORMAT_VINDEX3: &str = "vindex3";

/// Whether this binary was compiled with V3 Metal execution or MoE expert
/// dispatch. A compile-time fact, not a claim that Metal is currently
/// driving generation — confirming that would need a live probe of a
/// model's lazily-initialised Metal backend, and a read-only status
/// endpoint shouldn't trigger GPU initialisation as a side effect.
/// Follow-up once a probe exists that doesn't force it.
const fn metal_compiled() -> bool {
    cfg!(all(
        any(feature = "metal-experts", feature = "vindex3-metal"),
        target_os = "macos"
    ))
}

/// The `model` block for one resolved binding.
fn model_block(served: &ServedModel) -> serde_json::Value {
    match served {
        ServedModel::V2(m) => serde_json::json!({
            "id": m.id,
            "architecture": m.config.family,
            "format": FORMAT_VINDEX2,
            "quantization": m.config.quant.to_string(),
        }),
        ServedModel::V3(m) => serde_json::json!({
            "id": m.id,
            "architecture": m.family,
            "format": FORMAT_VINDEX3,
            // A V3 container is an executable program, not a feature
            // index with one top-level quant tag the way V2's
            // index.json has — there is no single fact to report here
            // yet. Follow-up once the plan exposes one.
            "quantization": serde_json::Value::Null,
        }),
    }
}

/// Rough resident footprint of the bound model's weights, in bytes.
/// V2: [`larql_vindex::VindexConfig::estimate_resident_bytes`] — the
/// same estimator the startup memcheck pre-flight uses, so this number
/// and a refusal to boot are the same fact, not two competing ones.
/// V3: no equivalent estimator exists yet (a V3 container's operands
/// are lowered into the backend's execution form at bind time, not
/// sized up front the way V2's mmap accounting is) — `None` rather
/// than a guess.
fn model_bytes(served: &ServedModel) -> Option<u64> {
    match served {
        ServedModel::V2(m) => Some(m.config.estimate_resident_bytes()),
        ServedModel::V3(_) => None,
    }
}

/// Build the `/v1/runtime` snapshot. Shared with
/// `routes::runtime_lifecycle`'s `POST`/`DELETE` handlers, which
/// return this same shape after a successful (or idempotent, or
/// refused) mutation — a client that just loaded or unloaded a model
/// doesn't need a second round trip to see the result. Does not bump
/// the request counter; callers do that once, at their own HTTP entry
/// point.
pub(crate) fn runtime_snapshot(state: &AppState) -> serde_json::Value {
    // Same "single binding or none" resolution `/v1/health` implicitly
    // relies on — never an error here, since the process-level facts
    // below are worth reporting regardless of how many models are
    // loaded.
    let served = state.served(None);
    let sample = state.runtime.last_sample();
    let active_requests = state.runtime.active_requests();

    serde_json::json!({
        "status": STATUS_READY,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_ms": state.started_at.elapsed().as_millis() as u64,
        "model": served.as_ref().map(model_block),
        "backend": {
            "metal_compiled": metal_compiled(),
            "selected": served.as_ref().and_then(|m| match m {
                ServedModel::V3(m) => Some(m.backend.as_str()),
                ServedModel::V2(_) => None,
            }),
        },
        "dense_ffn_shard": served.as_ref().and_then(|m| match m {
            ServedModel::V3(m) => m.ffn_shard.as_ref(),
            ServedModel::V2(_) => None,
        }),
        "layer_shard": served.as_ref().and_then(|m| match m {
            ServedModel::V3(m) => m.shard.as_ref(),
            ServedModel::V2(_) => None,
        }),
        "memory": {
            "resident_bytes": crate::runtime_stats::resident_bytes(),
            "model_bytes": served.as_ref().and_then(model_bytes),
        },
        "performance": {
            "prefill_tokens_per_second": sample.and_then(|s| s.prefill_tps_or_none()),
            "decode_tokens_per_second": sample.and_then(|s| s.decode_tps_or_none()),
            "last_request_latency_ms": sample.map(|s| s.latency_ms),
        },
        "generation": {
            "active": active_requests > 0,
            "active_requests": active_requests,
        },
    })
}

#[utoipa::path(
    get,
    path = "/v1/runtime",
    tag = "admin",
    responses(
        (status = 200, description = "Server + model + backend + memory + performance snapshot", body = crate::openapi::schemas::RuntimeResponse),
    ),
)]
pub async fn handle_runtime(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.bump_requests();
    Json(runtime_snapshot(&state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_compiled_matches_the_compile_time_cfg() {
        // Just pins that the const fn doesn't panic and returns a bool
        // consistent with the feature/target gate — the actual value
        // depends on how this test binary was built.
        let compiled = metal_compiled();
        assert_eq!(
            compiled,
            cfg!(all(
                any(feature = "metal-experts", feature = "vindex3-metal"),
                target_os = "macos"
            ))
        );
    }
}
