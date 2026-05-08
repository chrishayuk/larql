//! POST /v1/warmup
//!
//! Pre-touches the lazy state that the `walk-ffn` and `infer` paths
//! would otherwise pay on first request:
//!
//! - **Inference weights** (`get_or_load_weights`) — loads
//!   `lm_head.bin` + `norms.bin` + the f32-decoded gate-vector cache.
//!   On Gemma 26B this is ~2.9 GB / ~1.3 s on first call.
//! - **Q4K mmap pages** for the requested layer range — `madvise
//!   WILLNEED` so the kernel pre-streams the bytes that `walk-ffn`
//!   will read. Cuts the per-layer first-touch cost from ~17 ms to
//!   ~0.3 ms.
//!
//! Idempotent: running it twice is cheap. The warmup also runs at
//! boot when `larql-server --warmup-walk-ffn` is set, which is the
//! recommended posture for production grid shards.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Default, Deserialize, utoipa::ToSchema)]
pub struct WarmupRequest {
    /// Specific layers to prefetch (`madvise WILLNEED`). Defaults to
    /// every owned layer when omitted — the typical case for boot
    /// warmup.
    #[serde(default)]
    pub layers: Option<Vec<usize>>,

    /// Skip the inference-weight load. Use when the server was started
    /// with `--no-infer` and you only want mmap prefetch, not
    /// `lm_head` / `norms` / gate-f32 expansion.
    #[serde(default)]
    pub skip_weights: bool,

    /// Eager-build HNSW for every owned layer (mirrors the existing
    /// `--warmup-hnsw` boot flag, exposed here so operators can warm
    /// a running server without restarting). Requires HNSW already
    /// enabled via `--hnsw`.
    #[serde(default)]
    pub warmup_hnsw: bool,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct WarmupResponse {
    pub model: String,
    pub weights_loaded: bool,
    pub weights_load_ms: u64,
    pub layers_prefetched: usize,
    pub prefetch_ms: u64,
    /// Number of (layer, expert) pairs whose pages were read into the page cache.
    /// Zero for non-MoE models or when `skip_weights = true`.
    pub experts_prefetched: usize,
    pub expert_prefetch_ms: u64,
    pub hnsw_built: bool,
    pub hnsw_warmup_ms: u64,
    pub total_ms: u64,
}

/// Run the warmup steps for one model. Pulled out so the boot-time
/// `--warmup-walk-ffn` flag can call it without going through HTTP.
pub fn warmup_model(model: &LoadedModel, req: &WarmupRequest) -> WarmupResponse {
    let total_t = Instant::now();
    let model_id = model.config.model.clone();

    // ── 1. Inference weights (the 2.9 GB / 1.3 s cost on cold walk-ffn) ──
    let mut weights_load_ms = 0u64;
    let mut weights_loaded = false;
    if !req.skip_weights {
        let t = Instant::now();
        match model.get_or_load_weights() {
            Ok(_) => {
                weights_load_ms = t.elapsed().as_millis() as u64;
                weights_loaded = true;
                info!(
                    "warmup[{model_id}]: inference weights loaded in {}ms",
                    weights_load_ms
                );
            }
            Err(e) => {
                tracing::warn!("warmup[{model_id}]: weight load failed (skipping): {e}");
            }
        }
    }

    // Expert page prefetch is intentionally omitted for MoE shards:
    // total model data (experts + weights + dense FFN + embeddings) exceeds
    // 16 GB on performance-8x machines, so any bulk prefetch causes eviction
    // of other critical pages and degrades steady-state throughput. Demand
    // paging via MADV_RANDOM (set at mmap time) is the right policy here.
    // Upgrade to performance-16x (32 GB) to eliminate cold-fault spikes.
    let (experts_prefetched, expert_prefetch_ms) = (0usize, 0u64);

    // ── 2. Per-layer Q4K mmap prefetch (madvise WILLNEED) ──
    // Uses the existing `prefetch_interleaved_q4k_layer` accessor —
    // it madvises the layer's slice into the page cache without
    // dequantising or decoding anything.
    let prefetch_t = Instant::now();
    let layers: Vec<usize> = match req.layers.as_ref() {
        Some(v) => v.clone(),
        None => (0..model.config.num_layers).collect(),
    };
    let mut prefetched = 0usize;
    {
        let p = model.patched.blocking_read();
        for &layer in &layers {
            if layer >= model.config.num_layers {
                continue;
            }
            p.base.prefetch_interleaved_q4k_layer(layer);
            prefetched += 1;
        }
    }
    let prefetch_ms = prefetch_t.elapsed().as_millis() as u64;

    // ── 3. HNSW eager-build (rayon-parallel, owned layers) ──
    let mut hnsw_built = false;
    let mut hnsw_warmup_ms = 0u64;
    if req.warmup_hnsw {
        let p = model.patched.blocking_read();
        if p.base.is_hnsw_enabled() {
            let t = Instant::now();
            p.base.warmup_hnsw_all_layers();
            hnsw_warmup_ms = t.elapsed().as_millis() as u64;
            hnsw_built = true;
            info!(
                "warmup[{model_id}]: HNSW eager-built in {}ms",
                hnsw_warmup_ms
            );
        } else {
            tracing::warn!(
                "warmup[{model_id}]: warmup_hnsw=true but server was not started with --hnsw"
            );
        }
    }

    WarmupResponse {
        model: model_id,
        weights_loaded,
        weights_load_ms,
        layers_prefetched: prefetched,
        prefetch_ms,
        experts_prefetched,
        expert_prefetch_ms,
        hnsw_built,
        hnsw_warmup_ms,
        total_ms: total_t.elapsed().as_millis() as u64,
    }
}

/// Async wrapper for `warmup_model` that runs the (potentially
/// multi-second) work on a blocking worker so the tokio runtime
/// stays responsive.
pub async fn warmup_model_async(model: Arc<LoadedModel>, req: WarmupRequest) -> WarmupResponse {
    tokio::task::spawn_blocking(move || warmup_model(&model, &req))
        .await
        .expect("warmup spawn_blocking")
}

#[utoipa::path(
    post,
    path = "/v1/warmup",
    tag = "inference",
    request_body = WarmupRequest,
    responses(
        (status = 200, description = "Warmup completed", body = WarmupResponse),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_warmup(
    State(state): State<Arc<AppState>>,
    body: Option<Json<WarmupRequest>>,
) -> Result<Json<WarmupResponse>, ServerError> {
    state.bump_requests();
    let req = body.map(|Json(r)| r).unwrap_or_default();
    let model = state.model_or_err(None)?.clone();
    Ok(Json(warmup_model_async(model, req).await))
}
