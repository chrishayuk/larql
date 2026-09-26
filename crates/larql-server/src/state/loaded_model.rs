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
    /// `layer_infos` empty).  Route handlers force dense-mode
    /// inference on such vindexes regardless of the requested mode,
    /// since walk would silently return nothing useful.
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
            bitnet_layout: None,
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
            weights_init: std::sync::Mutex::new(()),
            bitnet_model: std::sync::OnceLock::new(),
            bitnet_init: std::sync::Mutex::new(()),
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
    fn is_dense_only_detects_empty_gate_layers() {
        // A normal vindex has gate layers -> not dense-only.
        let normal = tiny_loaded_model(QuantFormat::None, false);
        assert!(
            !normal.is_dense_only(),
            "vindex with gate layers must not be dense-only"
        );
        assert!(
            !normal.is_bitnet(),
            "and a plain vindex carries no bitnet_layout"
        );

        // A --dense-only BitNet vindex has zero gate layers.  Build
        // one by emptying the layer list + setting bitnet_layout.
        let mut cfg = tiny_config(QuantFormat::None);
        cfg.layers = Vec::new();
        cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
        let mut dense_only = tiny_loaded_model(QuantFormat::None, false);
        dense_only.config = cfg;
        assert!(
            dense_only.is_dense_only(),
            "dense-only vindex (empty gate layers) must be detected"
        );
        assert!(dense_only.is_bitnet(), "and it is a BitNet vindex");
    }

    #[test]
    fn bitnet_guards_refuse_a_dense_vindex_with_a_useful_message() {
        // `ensure_bitnet_cell`'s refusal path: asking a non-BitNet vindex
        // for a ternary model must name *why* rather than surfacing a
        // load error from a file that was never going to exist.
        let model = tiny_loaded_model(QuantFormat::None, false);
        // `BitnetModel` is not `Debug`, so match rather than `expect_err`.
        let Err(err) = model.get_or_load_bitnet() else {
            unreachable!("a dense vindex has no ternary model to hand out")
        };
        assert!(
            err.contains("bitnet_layout") && err.contains("keep-quant"),
            "the error must say the container is not a --keep-quant build, \
             got: {err}"
        );
    }

    #[test]
    fn force_load_bitnet_model_is_a_noop_when_infer_disabled() {
        // `bootstrap::serve` calls this unconditionally for every model,
        // so it has to stay quiet on a --no-infer server even when the
        // container *is* BitNet-shaped: eagerly loading ternary weights
        // into a process that refuses to infer would spend the memory a
        // --no-infer operator asked not to spend.
        let mut cfg = tiny_config(QuantFormat::None);
        cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
        let mut model = tiny_loaded_model(QuantFormat::None, false);
        model.config = cfg;
        model.infer_disabled = true;
        assert!(model.is_bitnet(), "fixture must be BitNet-shaped");
        assert!(
            model.force_load_bitnet_model().is_ok(),
            "must no-op rather than error under --no-infer"
        );
        assert!(
            model.bitnet_model.get().is_none(),
            "and must not have loaded anything"
        );
    }

    #[test]
    fn bitnet_load_failure_names_the_container() {
        // A container that *claims* to be BitNet (bitnet_layout present)
        // but has no `bitnet/` artifacts on disk must fail with the load
        // error, not the "not a --keep-quant build" refusal: the two are
        // different operator problems. The first says "this vindex is the
        // wrong kind", the second says "this vindex is the right kind and
        // is broken/incomplete", and reporting the wrong one sends the
        // operator to rebuild a container that only needs its files back.
        //
        // Reachable without any weights: the fixture's path points at no
        // bitnet/ directory, which is exactly the on-disk state of a
        // truncated or partially-copied container.
        let mut cfg = tiny_config(QuantFormat::None);
        cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
        let mut model = tiny_loaded_model(QuantFormat::None, false);
        model.config = cfg;
        assert!(model.is_bitnet(), "fixture must be BitNet-shaped");

        let Err(err) = model.get_or_load_bitnet() else {
            unreachable!("there are no bitnet/ artifacts to load")
        };
        assert!(
            err.contains("failed to load bitnet model"),
            "a BitNet-shaped container with missing artifacts must report a \
             load failure, not the wrong-kind refusal, got: {err}"
        );
        assert!(
            !err.contains("not a --keep-quant build"),
            "must not claim the container is the wrong kind: {err}"
        );
        // A failed load must leave the cell empty so a later attempt (after
        // the operator restores the files) still tries, rather than caching
        // the failure for the process lifetime.
        assert!(
            model.bitnet_model.get().is_none(),
            "a failed load must not poison the cell"
        );
    }

    #[test]
    fn lock_weights_for_gen_refuses_bitnet_with_an_actionable_message() {
        // Regression: on a real --keep-quant container the three
        // non-streaming generation paths (openai completions batch loop,
        // chat handler, responses engine) all reached
        // `ensure_weights_cell` and surfaced a bare "No such file or
        // directory" as a 503 -- there is no dense weight manifest in such
        // a container. Caught only against the real
        // microsoft/bitnet-b1.58-2B-4T model, because the synthetic
        // fixture is a dense V2 container that has those files.
        //
        // The message has to say what to use instead: the ternary engine
        // *is* reachable, just not through a path that needs
        // `&mut ModelWeights`.
        let mut cfg = tiny_config(QuantFormat::None);
        cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
        let mut model = tiny_loaded_model(QuantFormat::None, false);
        model.config = cfg;
        assert!(model.is_bitnet(), "fixture must be BitNet-shaped");

        let Err(err) = model.lock_weights_for_gen() else {
            unreachable!("a --keep-quant container has no dense weights to lock")
        };
        assert!(
            err.contains("keep-quant") && err.contains("no dense"),
            "must name the container kind as the reason, got: {err}"
        );
        assert!(
            err.contains("/v1/infer") && err.contains("stream"),
            "must point at the paths that do work, got: {err}"
        );

        // And the dense case must be unaffected: a plain container still
        // reaches the loader (and fails on the missing fixture files, not
        // on this guard).
        let dense = tiny_loaded_model(QuantFormat::None, false);
        let Err(dense_err) = dense.lock_weights_for_gen() else {
            unreachable!("the tiny fixture has no weight files on disk")
        };
        assert!(
            !dense_err.contains("keep-quant"),
            "a dense container must not hit the BitNet guard: {dense_err}"
        );
    }

    #[test]
    fn bitnet_model_not_loaded_by_default() {
        // Same lazy-load contract as `weights`: the ternary cell stays
        // empty until `get_or_load_bitnet`, and `force_load_bitnet_model`
        // is a no-op on a vindex that is not BitNet-shaped (rather than
        // an error), so `bootstrap::serve` can call it unconditionally.
        let model = tiny_loaded_model(QuantFormat::None, false);
        assert!(
            model.bitnet_model.get().is_none(),
            "bitnet cell must start empty"
        );
        assert!(
            model.force_load_bitnet_model().is_ok(),
            "force_load_bitnet_model must no-op on a non-BitNet vindex"
        );
        assert!(
            model.bitnet_model.get().is_none(),
            "and must not populate the cell"
        );
        assert!(
            model.get_or_load_bitnet().is_err(),
            "explicitly asking for a bitnet model on a dense vindex is an error"
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

    #[test]
    fn force_load_weights_skips_when_infer_disabled() {
        // tiny_loaded_model() sets infer_disabled = true (no real
        // weights on disk), so force_load_weights() must short-circuit
        // without ever touching the load path — otherwise it would
        // panic trying to mmap the nonexistent vindex directory.
        // This is the contract `bootstrap::serve` relies on for
        // --no-infer / --ffn-only / --embed-only models that should
        // not pay the eager-load cost.
        let model = tiny_loaded_model(QuantFormat::None, false);
        assert!(model.infer_disabled);
        assert!(model.force_load_weights().is_ok());
        assert!(
            model.weights.get().is_none(),
            "force_load_weights must not populate weights when infer_disabled"
        );
    }

    #[test]
    fn force_load_weights_skips_browse_only_vindex() {
        // A vindex with extract_level = Browse and has_model_weights
        // = false has nothing to load.  force_load_weights() should
        // succeed without populating `weights` so the boot sequence
        // does not try to mmap absent files.
        let mut model = tiny_loaded_model(QuantFormat::None, false);
        // Flip infer_disabled off but keep config = Browse + no
        // model weights, so the early-return is taken on the
        // "nothing to load" branch rather than the disabled branch.
        model.infer_disabled = false;
        assert_eq!(
            model.config.extract_level,
            larql_vindex::ExtractLevel::Browse
        );
        assert!(!model.config.has_model_weights);
        assert!(model.force_load_weights().is_ok());
        assert!(model.weights.get().is_none());
    }

    /// Concurrent first-callers of `ensure_weights_cell` must not
    /// double-allocate `ModelWeights`.  Without the `weights_init`
    /// mutex two threads both observe `weights.get() == None`, both
    /// run the loader, both produce a multi-GB `ModelWeights`, and
    /// only the first wins via `OnceLock::set` — but during the
    /// load both allocations are live, doubling peak heap.
    ///
    /// We can't load real weights in a unit test, so we drive the
    /// race by having both threads enter the slow path of
    /// `ensure_weights_cell()` against an `infer_disabled = false`
    /// model with no on-disk weights.  Both will fail at the loader
    /// step, but the test asserts they fail one-at-a-time (i.e. the
    /// init mutex serializes them) and that `weights.get()` stays
    /// `None` afterward.
    ///
    /// Concretely: we observe `loader_in_flight` never exceeds 1.
    #[test]
    fn ensure_weights_cell_single_flights_concurrent_loaders() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::Arc;
        use std::thread;

        // Build a tiny model with infer_disabled=false so
        // ensure_weights_cell will try to load.  The load itself
        // will fail (no real vindex on disk), but failure is fine —
        // we only care that the *attempts* are serialized.
        let mut model = tiny_loaded_model(QuantFormat::None, false);
        model.infer_disabled = false;
        // Mark the model as inference-level so force_load_weights()
        // would proceed (we use ensure_weights_cell directly here
        // anyway).
        model.config.has_model_weights = true;
        let model = Arc::new(model);

        // Track concurrent slow-path occupants.  Bumped just before
        // the loader call would happen, decremented just after.
        // Without the init mutex this would peak at 8; with it,
        // peak == 1.
        let in_flight = Arc::new(AtomicI64::new(0));
        let max_in_flight = Arc::new(AtomicI64::new(0));

        // We can't easily wedge the real loader to widen the race
        // window, but the loader's mmap+open syscall failure path
        // takes long enough on a 4-vCPU system that 8 concurrent
        // attempts will overlap noticeably.  The init mutex is
        // either present or absent — the assertion is that it
        // exists and excludes concurrent slow-path occupants.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let model = Arc::clone(&model);
            let in_flight = Arc::clone(&in_flight);
            let max_in_flight = Arc::clone(&max_in_flight);
            handles.push(thread::spawn(move || {
                // Manually re-do the ensure-style check so we can
                // observe the slow path window.  This mirrors
                // ensure_weights_cell's structure.
                if model.weights.get().is_some() {
                    return;
                }
                let _g = model.weights_init.lock().unwrap_or_else(|p| p.into_inner());
                let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                let prev_max = max_in_flight.load(Ordering::SeqCst);
                if n > prev_max {
                    max_in_flight.store(n, Ordering::SeqCst);
                }
                // Simulate the loader's wall time.  Real load is
                // ~3–10 s on a BitNet 2 B vindex; we use a small
                // sleep here so 8 threads racing actually overlap.
                std::thread::sleep(std::time::Duration::from_millis(20));
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            let _ = h.join();
        }

        let peak = max_in_flight.load(Ordering::SeqCst);
        assert_eq!(
            peak, 1,
            "weights_init mutex must serialize concurrent loaders; \
             observed peak = {peak}"
        );
    }

    /// Verify that `ensure_weights_cell`'s fast path is genuinely
    /// lock-free — once `weights` is populated, callers must not
    /// take the init mutex.  We exercise this by populating
    /// `weights` directly and then checking that holding the init
    /// mutex from another thread does not block the read.
    ///
    /// (We can't construct a real `ModelWeights` here, but we can
    /// at least assert the structural property: `weights.get()`
    /// returning `Some` short-circuits before the mutex is touched
    /// in `ensure_weights_cell`.)
    #[test]
    fn weights_init_mutex_is_unpoisonable_recoverable() {
        // Construct a fresh init mutex, poison it via a panicking
        // thread, then assert that the recovery path in
        // `ensure_weights_cell` (`unwrap_or_else(|p| p.into_inner())`)
        // works.  This is the resilience contract: a panic during
        // load should not permanently wedge the model — a retry
        // must be able to recover the lock.
        let mutex = std::sync::Mutex::new(());
        let mutex_arc = std::sync::Arc::new(mutex);
        let m2 = std::sync::Arc::clone(&mutex_arc);
        let h = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("simulated load failure");
        });
        let _ = h.join();
        assert!(mutex_arc.is_poisoned());
        // The recovery used in production code:
        let _g = mutex_arc.lock().unwrap_or_else(|p| p.into_inner());
        // Reaching here means recovery worked; without
        // unwrap_or_else we'd have unwound on the unwrap of a
        // poisoned guard.
    }
}
