//! AppState: loaded vindex + config, shared across all handlers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::embed_store::EmbedStoreF16;

use larql_models::ModelWeights;
use larql_vindex::{
    format::filenames::FEATURE_LABELS_JSON, ndarray::Array2, tokenizers, PatchedVindex,
    VindexConfig,
};
use tokio::sync::RwLock;

use crate::cache::DescribeCache;
use crate::ffn_l2_cache::FfnL2Cache;
use crate::session::SessionManager;

/// A single loaded model.
pub struct LoadedModel {
    /// Model ID derived from config (e.g., "gemma-3-4b-it").
    pub id: String,
    /// Vindex directory on disk.
    pub path: PathBuf,
    /// Vindex config (index.json).
    pub config: VindexConfig,
    /// Base index with patch overlay (starts with no patches).
    pub patched: Arc<RwLock<PatchedVindex>>,
    /// Embeddings matrix + scale factor, loaded once.
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    /// Tokenizer for embedding lookups.
    pub tokenizer: tokenizers::Tokenizer,
    /// Whether inference is disabled (--no-infer).
    pub infer_disabled: bool,
    /// Whether this server is running in FFN-service mode (--ffn-only).
    /// Implies `infer_disabled = true`; advertised in /v1/stats so clients
    /// using `RemoteWalkBackend` can tell they've landed on the right
    /// endpoint. Memory-footprint optimization (skip attention weight
    /// load) is a separate follow-up.
    pub ffn_only: bool,
    /// Whether this server is running in embed-service mode (--embed-only).
    /// Implies `infer_disabled = true`. Loads only embeddings + lm_head +
    /// tokenizer; skips FFN and attention weights.
    pub embed_only: bool,
    /// f16-at-rest embedding store — populated when `--embed-only` and
    /// `embeddings.bin` is an f16 file. Halves embed-server RSS vs the
    /// eager f32 heap copy (ADR-0008). `None` when f32 or not embed-only.
    pub embed_store: Option<Arc<EmbedStoreF16>>,
    /// When true, `madvise(MADV_DONTNEED)` is issued on every mmap after
    /// each walk-ffn request. Opt-in via `--release-mmap-after-request`.
    /// Pairs with `--max-gate-cache-layers` to bound RSS hard; prefer
    /// `--layers START-END` sharding when available.
    pub release_mmap_after_request: bool,
    /// Model weights, lazy-loaded on first INFER request.
    ///
    /// Wrapped in `RwLock` so the OpenAI generation path (which calls
    /// `larql_inference::layer_graph::generate` and friends, all of
    /// which take `&mut ModelWeights` to mutate the per-layer Q4_K
    /// dequant cache) can take a write guard while every other read
    /// path concurrently holds read guards. Read access is the common
    /// case; write access is one-at-a-time per model.
    ///
    /// `OnceLock<RwLock<...>>` rather than `RwLock<Option<...>>` so
    /// the lazy-init logic stays lock-free until first use.
    pub weights: std::sync::OnceLock<std::sync::RwLock<ModelWeights>>,
    /// Probe-confirmed feature labels: (layer, feature) → relation name.
    /// Loaded from feature_labels.json if present.
    pub probe_labels: HashMap<(usize, usize), String>,
    /// L2 FFN output cache — shared across all clients, persists for server lifetime.
    pub ffn_l2_cache: FfnL2Cache,
    /// Per-layer latency tracker — records compute time per walk-ffn layer.
    /// Snapshots are sent to the router in HeartbeatMsg.layer_stats (GT3).
    pub layer_latency_tracker: std::sync::Arc<crate::metrics::LayerLatencyTracker>,
    /// Active walk-ffn request counter — incremented on request entry,
    /// decremented on return. Used by GT6 drain to know when it is safe
    /// to send DroppingMsg(reason="reassigned").
    pub requests_in_flight: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Monotonically-increasing total count of walk-ffn requests seen by
    /// this shard. Read by the grid announce loop to compute
    /// `HeartbeatMsg.req_per_sec` (delta over the heartbeat interval) so
    /// the router's hot-shard rebalancer can detect saturation.
    pub requests_total: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Expert ID range this server owns (from `--experts START-END`).
    /// `None` = serve all experts. Used by the expert endpoint to reject
    /// requests for experts this shard doesn't hold.
    /// Layer-uniform: same range applies to every layer.
    pub expert_filter: Option<(usize, usize)>,
    /// Fine-grained per-(layer, expert) ownership (from `--units PATH`).
    /// When `Some`, takes precedence over `expert_filter` — `run_expert`
    /// rejects any (layer, expert_id) not in this set.  Designed for the
    /// architecture where each shard hosts a tight set of (layer, expert)
    /// units rather than a contiguous expert range.
    pub unit_filter: Option<Arc<std::collections::HashSet<(usize, usize)>>>,
    /// Remote MoE expert backend wired via `--moe-shards` or `--moe-units-manifest`.
    /// When `Some`, the walk-ffn handler uses this for MoE layers instead of local dispatch.
    pub moe_remote: Option<Arc<larql_inference::ffn::RemoteMoeBackend>>,

    /// Lazy-initialised Metal backend for GPU expert dispatch.
    /// `Some(Some(backend))` = initialised, available; `Some(None)` =
    /// initialised, Metal not available; `None` = not yet initialised.
    /// Only present under `--features metal-experts`.
    #[cfg(all(feature = "metal-experts", target_os = "macos"))]
    pub metal_backend: std::sync::OnceLock<Option<larql_compute_metal::MetalBackend>>,
    /// Cached MoE scratch per `(top_k, hidden, inter)` shape — one entry
    /// per architecture in practice.  `MoeScratch` contains mutable Metal
    /// staging buffers, so Metal expert dispatch holds this mutex while
    /// using a scratch entry.
    #[cfg(all(feature = "metal-experts", target_os = "macos"))]
    pub moe_scratches: std::sync::Mutex<
        std::collections::HashMap<(usize, usize, usize), Arc<larql_compute_metal::MoeScratch>>,
    >,
    /// Per-layer pre-loaded Q4K weight buffers for Metal dense FFN dispatch.
    /// `[gate_buf, up_buf, down_buf]` for each layer. Lazily populated on first
    /// Metal FFN request from the interleaved Q4K mmap (zero-copy via
    /// `new_buffer_with_bytes_no_copy` for page-aligned mmap data).
    /// Only populated when the server has interleaved Q4K data loaded.
    #[cfg(all(feature = "metal-experts", target_os = "macos"))]
    pub metal_ffn_layer_bufs: std::sync::OnceLock<Vec<[larql_compute_metal::MetalBuffer; 3]>>,
}

impl LoadedModel {
    /// Get or lazy-load model weights for inference.
    ///
    /// For `--ffn-only` servers the loader filters attention + lm_head
    /// + embed entries from the weight manifest before mmap/decode,
    ///   so peak RSS during load reflects only what the walk-ffn
    ///   endpoint actually needs.
    pub fn get_or_load_weights(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, ModelWeights>, String> {
        let cell = self.ensure_weights_cell()?;
        cell.read()
            .map_err(|e| format!("weights RwLock poisoned: {e}"))
    }

    /// Acquire an exclusive write guard on the loaded weights.
    ///
    /// Used by the OpenAI generation path (`/v1/completions`,
    /// `/v1/chat/completions`) — `larql_inference::layer_graph::generate`
    /// and its variants take `&mut ModelWeights` because the per-layer
    /// Q4_K dequant cache inside `weights.tensors` is mutated as layers
    /// are decoded. Concurrent reads block while a generation is in
    /// flight, but generation requests are typically rare and bounded;
    /// the read fast path (walk-ffn / browse / embed) sees no
    /// contention in steady state.
    pub fn lock_weights_for_gen(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, ModelWeights>, String> {
        let cell = self.ensure_weights_cell()?;
        cell.write()
            .map_err(|e| format!("weights RwLock poisoned: {e}"))
    }

    fn ensure_weights_cell(&self) -> Result<&std::sync::RwLock<ModelWeights>, String> {
        if let Some(cell) = self.weights.get() {
            return Ok(cell);
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;

        // Q4_K vindexes take a dedicated loader that produces a ModelWeights
        // with empty attn/FFN tensors (those live in the Q4K mmap files).
        // The walk-ffn endpoint dequantises FFN per layer on demand.
        let weights = if self.config.quant == larql_vindex::QuantFormat::Q4K {
            if self.ffn_only {
                tracing::info!(
                    "ffn-only (q4k): loading norms + lm_head + embed only; \
                     FFN dequantises per layer from interleaved_kquant.bin on request"
                );
            }
            larql_vindex::load_model_weights_kquant_shard(&self.path, &mut cb, self.expert_filter)
                .map_err(|e| format!("failed to load q4k model weights: {e}"))?
        } else {
            let opts = if self.embed_only {
                // --embed-only: keep lm_head + norm weights (needed for
                // /v1/logits). Skip attn, FFN, and the embed matrix (the
                // embed endpoint reads model.embeddings directly).
                tracing::info!(
                    "embed-only: loading lm_head + norms only; \
                     skipping attn + ffn + embed tensors"
                );
                larql_vindex::LoadWeightsOptions {
                    skip_attn: true,
                    skip_lm_head: false,
                    skip_embed: true,
                    skip_ffn: true,
                }
            } else {
                if self.ffn_only {
                    tracing::info!(
                        "ffn-only: skipping attn + ffn + lm_head + embed at load \
                         (pre-mmap filter — walk uses feature-major mmap instead)"
                    );
                }
                larql_vindex::LoadWeightsOptions {
                    skip_attn: self.ffn_only,
                    skip_lm_head: self.ffn_only,
                    skip_embed: self.ffn_only,
                    skip_ffn: self.ffn_only,
                }
            };
            larql_vindex::load_model_weights_with_opts(&self.path, &mut cb, opts)
                .map_err(|e| format!("failed to load model weights: {e}"))?
        };
        let _ = self.weights.set(std::sync::RwLock::new(weights));
        Ok(self.weights.get().unwrap())
    }
}

/// Shared application state.
pub struct AppState {
    /// Loaded models, keyed by model ID.
    pub models: Vec<Arc<LoadedModel>>,
    /// Server start time for uptime reporting.
    pub started_at: std::time::Instant,
    /// Request counter.
    pub requests_served: std::sync::atomic::AtomicU64,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Per-session PatchedVindex manager.
    pub sessions: SessionManager,
    /// DESCRIBE result cache.
    pub describe_cache: DescribeCache,
}

impl AppState {
    /// Get model by ID, or the only model if single-model serving.
    pub fn model(&self, id: Option<&str>) -> Option<&Arc<LoadedModel>> {
        match id {
            Some(id) => self.models.iter().find(|m| m.id == id),
            None if self.models.len() == 1 => self.models.first(),
            None => None,
        }
    }

    /// Whether this is multi-model serving.
    pub fn is_multi_model(&self) -> bool {
        self.models.len() > 1
    }

    pub fn bump_requests(&self) {
        self.requests_served
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get a model by ID, or return a `NotFound` error.
    ///
    /// Consolidates the 23+ identical `state.model(...).ok_or_else(|| ...)` call
    /// sites scattered across the route handlers.
    pub fn model_or_err(
        &self,
        id: Option<&str>,
    ) -> Result<&Arc<LoadedModel>, crate::error::ServerError> {
        self.model(id).ok_or_else(|| {
            let msg = match id {
                Some(mid) => format!("model '{}' not found", mid),
                None => "no model loaded".into(),
            };
            crate::error::ServerError::NotFound(msg)
        })
    }
}

/// Compute elapsed milliseconds from `start`, rounded to one decimal place.
pub fn elapsed_ms(start: std::time::Instant) -> f64 {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (ms * 10.0).round() / 10.0
}

/// Load probe-confirmed feature labels from feature_labels.json.
/// Format: {"L{layer}_F{feature}": "relation_name", ...}
pub fn load_probe_labels(vindex_path: &std::path::Path) -> HashMap<(usize, usize), String> {
    let path = vindex_path.join(FEATURE_LABELS_JSON);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let obj: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let map = match obj.as_object() {
        Some(m) => m,
        None => return HashMap::new(),
    };

    let mut labels = HashMap::new();
    for (key, value) in map {
        if let Some(rel) = value.as_str() {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() == 2 {
                if let (Some(layer), Some(feat)) = (
                    parts[0]
                        .strip_prefix('L')
                        .and_then(|s| s.parse::<usize>().ok()),
                    parts[1]
                        .strip_prefix('F')
                        .and_then(|s| s.parse::<usize>().ok()),
                ) {
                    labels.insert((layer, feat), rel.to_string());
                }
            }
        }
    }
    labels
}

/// Derive a short model ID from the full model name.
/// "google/gemma-3-4b-it" → "gemma-3-4b-it"
pub fn model_id_from_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

#[cfg(test)]
mod loaded_model_tests {
    //! Unit tests for `LoadedModel` field/flag plumbing.
    //!
    //! The q4k / f32 branch in `get_or_load_weights` keys off
    //! `config.quant == QuantFormat::Q4K`, and `run_full_output` in
    //! `routes/walk_ffn.rs` keys off the same check to decide between
    //! `WalkFfn::new_unlimited` and `kquant_ffn_forward_layer`. Running
    //! either branch end-to-end needs a real on-disk vindex (GBs of
    //! weights), so we cover just the flag plumbing and the selector
    //! expression here; the end-to-end walk is validated by the
    //! `larql bench <model>` example script.
    use super::*;
    use larql_vindex::ndarray::Array2;
    use larql_vindex::{
        ExtractLevel, LayerBands, QuantFormat, VectorIndex, VindexConfig, VindexLayerInfo,
    };

    fn tiny_config(quant: QuantFormat) -> VindexConfig {
        VindexConfig {
            version: 2,
            model: "test/model".to_string(),
            family: "test".to_string(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 4,
            vocab_size: 4,
            embed_scale: 1.0,
            extract_level: ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::default(),
            quant,
            layer_bands: Some(LayerBands {
                syntax: (0, 0),
                knowledge: (0, 0),
                output: (0, 0),
            }),
            layers: vec![VindexLayerInfo {
                layer: 0,
                num_features: 2,
                offset: 0,
                length: 32,
                num_experts: None,
                num_features_per_expert: None,
            }],
            down_top_k: 1,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        }
    }

    fn tiny_loaded_model(quant: QuantFormat, release_mmap: bool) -> LoadedModel {
        let hidden = 4;
        let gate = Array2::<f32>::zeros((2, hidden));
        let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
        let patched = larql_vindex::PatchedVindex::new(index);

        let tok_json =
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

        LoadedModel {
            id: "test".into(),
            path: PathBuf::from("/nonexistent"),
            config: tiny_config(quant),
            patched: std::sync::Arc::new(tokio::sync::RwLock::new(patched)),
            embeddings: Array2::<f32>::zeros((4, hidden)),
            embed_scale: 1.0,
            tokenizer,
            infer_disabled: true,
            ffn_only: false,
            embed_only: false,
            embed_store: None,
            release_mmap_after_request: release_mmap,
            weights: std::sync::OnceLock::new(),
            probe_labels: HashMap::new(),
            ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(1),
            layer_latency_tracker: std::sync::Arc::new(crate::metrics::LayerLatencyTracker::new()),
            requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            moe_scratches: std::sync::Mutex::new(HashMap::new()),
            #[cfg(all(feature = "metal-experts", target_os = "macos"))]
            metal_ffn_layer_bufs: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn release_mmap_flag_round_trips_true() {
        let model = tiny_loaded_model(QuantFormat::None, true);
        assert!(
            model.release_mmap_after_request,
            "true must survive unchanged — the walk-ffn handler reads this \
             post-request to issue MADV_DONTNEED"
        );
    }

    #[test]
    fn release_mmap_flag_round_trips_false() {
        let model = tiny_loaded_model(QuantFormat::None, false);
        assert!(!model.release_mmap_after_request);
    }

    #[test]
    fn quant_format_selects_q4k_branch() {
        // Exact selector used in both `get_or_load_weights` and
        // `run_full_output` to pick the q4k path.
        let q4k_model = tiny_loaded_model(QuantFormat::Q4K, false);
        let f32_model = tiny_loaded_model(QuantFormat::None, false);

        assert!(
            q4k_model.config.quant == QuantFormat::Q4K,
            "Q4K config → q4k branch (load_model_weights_kquant + kquant_ffn_forward_layer)"
        );
        assert!(
            f32_model.config.quant != QuantFormat::Q4K,
            "None config → f32 branch (load_model_weights_with_opts + WalkFfn::new_unlimited)"
        );
    }

    #[test]
    fn weights_not_loaded_by_default() {
        // Lazy-load contract: `weights` is `OnceLock::new()` until the
        // first `get_or_load_weights` call. The `release_mmap_after_request`
        // post-processing in walk_ffn.rs doesn't touch this.
        let model = tiny_loaded_model(QuantFormat::None, true);
        assert!(model.weights.get().is_none());
    }
}
