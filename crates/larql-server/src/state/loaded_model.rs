//! `LoadedModel`: one bound VINDEX2 model, its lazy-loaded weights,
//! and the per-model counters/caches route handlers touch directly.
//! Split out of the top-level `state` module (see `mod.rs`) purely
//! for file size — nothing here changed behavior.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::embed_store::EmbedStoreF16;

use larql_models::ModelWeights;
use larql_vindex::{ndarray::Array2, tokenizers, PatchedVindex, VindexConfig};
use tokio::sync::RwLock;

use crate::ffn_l2_cache::FfnL2Cache;

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
    /// Init guard — held only while one thread is loading tensors
    /// into `weights`.  Without this, two concurrent first-callers of
    /// `get_or_load_weights()` both observe `weights.get() == None`,
    /// both run `load_model_weights_with_opts` (~5 GB of allocation
    /// for a 2 B BitNet vindex), and only the first wins via
    /// `OnceLock::set` — but during the load both allocations are
    /// live, doubling peak heap and OOM-killing the cgroup on tight
    /// hosts.  The init mutex is held only during the load itself;
    /// once `weights` is populated, callers skip the mutex via the
    /// fast-path `OnceLock::get` check.
    pub weights_init: std::sync::Mutex<()>,
    /// BitNet 1.58 model with native ternary weights.  Populated
    /// when the loaded vindex was built with `--keep-quant`
    /// (i.e. `config.bitnet_layout.is_some()`).  When present, the
    /// route handlers prefer this over `weights` for inference
    /// because the native-ternary path runs the full forward at
    /// ~1.4 GB instead of ~5 GB resident.  Eager-loaded by
    /// `force_load_bitnet_model` from `bootstrap::serve` (unless
    /// `--lazy-weights`).
    pub bitnet_model: std::sync::OnceLock<std::sync::RwLock<larql_inference::ternary::BitnetModel>>,
    /// Init guard for the bitnet model load — same pattern as
    /// `weights_init` but for the ternary path.
    pub bitnet_init: std::sync::Mutex<()>,
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

    /// Eagerly load model weights from the request-handling fast
    /// path so the first `/v1/infer` does not face a 5+ GB
    /// allocation under request backpressure.
    ///
    /// Called once by `bootstrap::serve` (unless `--lazy-weights` was
    /// passed) before the HTTP listener binds.  A failure here causes
    /// the process to exit cleanly with a startup error rather than
    /// SIGKILL during the first inference request — operators see a
    /// useful message and can fix the cgroup before any traffic hits
    /// the port.
    pub fn force_load_weights(&self) -> Result<(), String> {
        if self.infer_disabled {
            return Ok(());
        }
        // Skip when there are no model weights to load (browse-only
        // vindex).  `get_or_load_weights` would happily walk the
        // request path and return an error anyway, but eagerly we
        // know in advance and stay quiet.
        let has_weights = self.config.has_model_weights
            || self.config.extract_level == larql_vindex::ExtractLevel::Inference
            || self.config.extract_level == larql_vindex::ExtractLevel::All;
        if !has_weights {
            return Ok(());
        }
        self.ensure_weights_cell().map(|_| ())
    }

    /// Whether this vindex was built with `--keep-quant` and
    /// therefore has the BitNet 1.58 native-ternary artifacts
    /// (`bitnet/` + `bitnet_layout` in index.json).  Route handlers
    /// dispatch on this to pick the ternary forward path.
    pub fn is_bitnet(&self) -> bool {
        self.config.bitnet_layout.is_some()
    }

    /// Whether this vindex was built `--dense-only`: it has the
    /// dense weights + BitNet I2_S artifacts but NO gate vectors /
    /// HNSW clustering, so walk-mode inference cannot run against
    /// it (the KNN store is empty).  Detected by an empty gate-layer
    /// list in index.json (`build_vindex_dense_only` leaves
    /// `layer_infos` empty).  `/v1/infer` serves an omitted mode as
    /// dense on such vindexes and refuses an explicit `walk` /
    /// `compare` with a 400 (see `routes/infer/bitnet.rs`).
    pub fn is_dense_only(&self) -> bool {
        self.config.layers.is_empty()
    }

    /// Get a read guard on the lazy-loaded BitNet model.  Returns
    /// `Err` when the vindex isn't a BitNet (callers should check
    /// `is_bitnet()` first).
    pub fn get_or_load_bitnet(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, larql_inference::ternary::BitnetModel>, String> {
        let cell = self.ensure_bitnet_cell()?;
        cell.read()
            .map_err(|e| format!("bitnet RwLock poisoned: {e}"))
    }

    /// Eager-load the BitNet model from disk before the listener
    /// binds.  Mirrors `force_load_weights` but for the ternary
    /// path; called by `bootstrap::serve` when the vindex is
    /// BitNet-shaped and `--lazy-weights` was not passed.
    pub fn force_load_bitnet_model(&self) -> Result<(), String> {
        if self.infer_disabled || !self.is_bitnet() {
            return Ok(());
        }
        self.ensure_bitnet_cell().map(|_| ())
    }

    fn ensure_bitnet_cell(
        &self,
    ) -> Result<&std::sync::RwLock<larql_inference::ternary::BitnetModel>, String> {
        // Fast path.
        if let Some(cell) = self.bitnet_model.get() {
            return Ok(cell);
        }
        // Single-flight slow path.
        let _init_guard = self.bitnet_init.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(cell) = self.bitnet_model.get() {
            return Ok(cell);
        }
        if !self.is_bitnet() {
            return Err("vindex has no bitnet_layout (not a --keep-quant build)".into());
        }
        let model = larql_inference::ternary::load_bitnet_model(&self.path)
            .map_err(|e| format!("failed to load bitnet model: {e}"))?;
        let _ = self.bitnet_model.set(std::sync::RwLock::new(model));
        self.bitnet_model
            .get()
            .ok_or_else(|| "bitnet cell unset after set".to_string())
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
        // A BitNet `--keep-quant` container has no dense weight manifest to
        // load, so `ensure_weights_cell` would fail here with a bare
        // "No such file or directory" from whichever tensor file it reached
        // first. Every non-streaming generation path funnels through this
        // one method (`openai/completions.rs` batch loop,
        // `openai/chat/handler.rs`, `openai/responses/engine.rs`), so
        // naming the real reason once here covers all of them rather than
        // three separate checks that have to stay in agreement.
        //
        // Refused rather than silently routed to the ternary path: these
        // callers hold a `&mut ModelWeights` for the whole generation, and
        // there is no dense `ModelWeights` to hand them. The ternary
        // engine is reachable through `/v1/infer` and the streaming
        // surfaces, which do not need one.
        if self.is_bitnet() {
            return Err(
                "this vindex is a BitNet --keep-quant build and carries no dense \
                 weights; non-streaming generation is not supported on it. Use \
                 POST /v1/infer, or /v1/completions and /v1/chat/completions \
                 with \"stream\": true, which take the native-ternary path."
                    .to_string(),
            );
        }
        let cell = self.ensure_weights_cell()?;
        cell.write()
            .map_err(|e| format!("weights RwLock poisoned: {e}"))
    }

    fn ensure_weights_cell(&self) -> Result<&std::sync::RwLock<ModelWeights>, String> {
        // Fast path: already loaded.  Lock-free read against the
        // OnceLock; covers the steady-state case where every request
        // after the first hits this branch.
        if let Some(cell) = self.weights.get() {
            return Ok(cell);
        }

        // Slow path: single-flight the load behind `weights_init`.
        // Recovering from a poisoned mutex is fine here — the only
        // operation under the guard is the loader itself, which does
        // not mutate any externally observable state on panic.
        let _init_guard = self
            .weights_init
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Double-check: another thread may have completed the load
        // while we were waiting for the init mutex.
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

#[cfg(test)]
mod tests;
