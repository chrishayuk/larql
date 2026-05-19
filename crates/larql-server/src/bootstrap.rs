//! Server bootstrap and vindex loading helpers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::middleware;
use clap::Parser;
use larql_vindex::format::filenames::*;
use larql_vindex::{
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer, PatchedVindex,
    SilentLoadCallbacks, VectorIndex,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::cache::DescribeCache;
use crate::session::SessionManager;
use crate::state::{load_probe_labels, model_id_from_name, AppState, LoadedModel};
use crate::{announce, auth, grpc, grpc_expert, ratelimit, routes};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ── CLI defaults ───────────────────────────────────────────────────────────────
//
// Hoisted out of `#[arg(default_value = "...")]` strings so the same value can
// be referenced from non-clap call sites (e.g. `SessionManager::new`).

pub const DEFAULT_PORT: u16 = 8080;
pub const DEFAULT_HOST: &str = "0.0.0.0";
pub const DEFAULT_MAX_GATE_CACHE_LAYERS: usize = 0;
pub const DEFAULT_MAX_Q4K_CACHE_LAYERS: usize = 0;
pub const DEFAULT_HNSW_EF_SEARCH: usize = 200;
pub const DEFAULT_MAX_CONCURRENT: usize = 100;
pub const DEFAULT_DESCRIBE_CACHE_TTL_SECS: u64 = 0;
pub const DEFAULT_LOG_LEVEL: &str = "info";
pub const DEFAULT_SESSION_TTL_SECS: u64 = 3600;

/// Parse a human-readable RAM size string into bytes.
/// Supports: "24GB", "16384MB", "4096KB", raw decimal bytes.
pub fn parse_ram_bytes(s: &str) -> Result<u64, BoxError> {
    let s = s.trim();
    let (num_str, mult) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("gb")) {
        (n, 1024u64 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("mb")) {
        (n, 1024u64 * 1024)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("kb")) {
        (n, 1024u64)
    } else {
        (s, 1u64)
    };
    let n: u64 = num_str
        .trim()
        .parse()
        .map_err(|_| format!("--available-ram: invalid number '{num_str}'"))?;
    Ok(n * mult)
}

pub fn parse_layer_range(s: &str) -> Result<(usize, usize), BoxError> {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() != 2 {
        return Err(format!("--layers: expected 'START-END' (e.g. '0-19'), got '{s}'").into());
    }
    let start: usize = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid start '{}'", parts[0]))?;
    let end: usize = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid end '{}'", parts[1]))?;
    if end < start {
        return Err(format!("--layers: end ({end}) must be >= start ({start})").into());
    }
    Ok((start, end + 1))
}

#[derive(Clone)]
pub struct LoadVindexOptions {
    pub no_infer: bool,
    pub ffn_only: bool,
    pub embed_only: bool,
    pub layer_range: Option<(usize, usize)>,
    pub max_gate_cache_layers: usize,
    pub max_q4k_cache_layers: usize,
    pub hnsw: Option<usize>,
    pub warmup_hnsw: bool,
    pub release_mmap_after_request: bool,
    pub expert_filter: Option<(usize, usize)>,
    /// Fine-grained per-(layer, expert) ownership.  When `Some`, takes
    /// precedence over `expert_filter` for `run_expert`'s ownership check
    /// and for the HNSW / Metal warmup loops.  Loaded from `--units` JSON.
    pub unit_filter: Option<Arc<std::collections::HashSet<(usize, usize)>>>,
    /// Server-side remote MoE backend. When `Some`, the walk-ffn handler
    /// delegates MoE expert dispatch to remote shard servers.
    pub moe_remote: Option<Arc<larql_inference::ffn::RemoteMoeBackend>>,
}

/// JSON layout for the `--units` manifest.  Each value is a list of inclusive
/// `[start, end]` expert-id ranges, keyed by layer index (as a string for
/// JSON-object compatibility).
#[derive(serde::Deserialize)]
pub struct UnitManifest {
    pub layer_experts: std::collections::BTreeMap<String, Vec<[usize; 2]>>,
}

impl UnitManifest {
    /// Expand the per-layer range list into the flat `(layer, expert_id)`
    /// set used by ownership checks.  Reports the first malformed entry in
    /// the error path so the operator can fix it without grepping.
    pub fn into_unit_set(self) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
        let mut units = std::collections::HashSet::new();
        for (layer_str, ranges) in self.layer_experts {
            let layer: usize = layer_str.parse().map_err(|_| -> BoxError {
                format!("--units: layer key '{layer_str}' is not a valid usize").into()
            })?;
            for [start, end] in ranges {
                if end < start {
                    return Err(format!(
                        "--units: layer {layer}: end ({end}) must be >= start ({start})"
                    )
                    .into());
                }
                for eid in start..=end {
                    units.insert((layer, eid));
                }
            }
        }
        Ok(units)
    }
}

/// Parse `--units PATH` into the canonical `(layer, expert_id)` ownership set.
pub fn parse_unit_manifest(
    path: &Path,
) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
    let bytes = std::fs::read(path)
        .map_err(|e| -> BoxError { format!("--units: read {}: {e}", path.display()).into() })?;
    let manifest: UnitManifest = serde_json::from_slice(&bytes)
        .map_err(|e| -> BoxError { format!("--units: parse {}: {e}", path.display()).into() })?;
    manifest.into_unit_set()
}

pub fn load_single_vindex(
    path_str: &str,
    opts: LoadVindexOptions,
) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex_with_range(&path, &mut cb, opts.layer_range)?;
    if opts.max_gate_cache_layers > 0 {
        index.set_gate_cache_max_layers(opts.max_gate_cache_layers);
        info!(
            "  Gate cache: LRU, max {} layers",
            opts.max_gate_cache_layers
        );
    }
    if opts.max_q4k_cache_layers > 0 {
        index.set_kquant_ffn_cache_max_layers(opts.max_q4k_cache_layers);
        info!(
            "  Q4K FFN cache: LRU, max {} layers",
            opts.max_q4k_cache_layers
        );
    }
    if let Some(ef) = opts.hnsw {
        index.enable_hnsw(ef);
        info!("  HNSW gate KNN: enabled (ef_search={ef})");
        if opts.warmup_hnsw {
            let t0 = std::time::Instant::now();
            index.warmup_hnsw_all_layers();
            let owned = match opts.layer_range {
                Some((s, e)) => e - s,
                None => config.num_layers,
            };
            info!(
                "  HNSW warmup: built {} owned layer(s) in {:.2?}",
                owned,
                t0.elapsed()
            );
        }
    }
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    let has_weights = config.has_model_weights
        || config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All;

    if let Some((start, end)) = opts.layer_range {
        info!("  Layers: {start}–{} (of {})", end - 1, config.num_layers);
    }
    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    if !opts.embed_only {
        match index.load_down_features(&path) {
            Ok(()) => info!("  Down features: loaded (mmap walk enabled)"),
            Err(_) => info!("  Down features: not available"),
        }
        if let Ok(()) = index.load_up_features(&path) {
            info!("  Up features: loaded (full mmap FFN)")
        }
        if index.has_down_features_kquant() {
            info!(
                "  Down features Q4K: loaded (W2 — per-feature decode skips kquant_ffn_layer cache)"
            );
        }

        // For inference-capable vindexes (`/v1/completions`,
        // `/v1/chat/completions`, `/v1/infer mode=walk`), load the
        // attention + interleaved-FFN slices the inference path needs.
        // Mirrors `larql_inference::open_inference_vindex` — without
        // these the Q4K decode panics with "attn Q4K slices missing".
        //
        // `--ffn-only` skips attention weights (no infer path) but MUST
        // still mmap interleaved_kquant so per-layer walk-ffn requests can
        // call `kquant_ffn_forward_layer`.
        let need_ffn_mmap = opts.ffn_only || (!opts.no_infer && has_weights);
        if !opts.no_infer && !opts.ffn_only && has_weights {
            if path.join(LM_HEAD_BIN).is_file() {
                let _ = index.load_lm_head(&path);
            }
            if has_kquant_lm_head(&path) {
                let _ = index.load_lm_head_kquant(&path);
            }
            if has_kquant_attn_weights(&path) {
                if let Err(e) = index.load_attn_kquant(&path) {
                    warn!("  Attn k-quant: failed to load ({e}) — generation may not work");
                } else {
                    info!("  Attn k-quant: loaded (inference path enabled)");
                }
            } else if path.join(ATTN_WEIGHTS_Q8_BIN).is_file() {
                if let Err(e) = index.load_attn_q8(&path) {
                    warn!("  Attn Q8: failed to load ({e}) — generation may not work");
                }
            }
        }
        if need_ffn_mmap {
            if has_kquant_interleaved(&path) {
                if let Err(e) = index.load_interleaved_kquant(&path) {
                    warn!("  Interleaved k-quant: failed to load ({e})");
                } else if opts.ffn_only {
                    info!("  Interleaved k-quant: loaded (ffn-service)");
                }
            } else if path.join(INTERLEAVED_Q4_BIN).is_file() {
                if let Err(e) = index.load_interleaved_q4(&path) {
                    warn!("  Interleaved Q4: failed to load ({e})");
                }
            }
        }
    }

    if opts.ffn_only || opts.embed_only {
        let reason = if opts.embed_only {
            "--embed-only"
        } else {
            "--ffn-only"
        };
        info!("  Warmup: skipped ({reason})");
    } else {
        index.warmup();
        info!("  Warmup: done");
    }

    let (embeddings, embed_scale) = load_vindex_embeddings(&path)?;
    info!(
        "  Embeddings: {}x{}",
        embeddings.shape()[0],
        embeddings.shape()[1]
    );

    let embed_store = if opts.embed_only {
        match crate::embed_store::EmbedStoreF16::open(
            &path,
            embed_scale,
            config.vocab_size,
            config.hidden_size,
            5_000,
        ) {
            Ok(store) => {
                let f16_bytes = config.vocab_size * config.hidden_size * 2;
                info!(
                    "  Embed store: f16 mmap ({:.1} GB, L1 cap 5000 tokens)",
                    f16_bytes as f64 / 1e9
                );
                Some(Arc::new(store))
            }
            Err(e) => {
                info!("  Embed store: f16 mmap unavailable ({e}), using f32 heap");
                None
            }
        }
    } else {
        None
    };

    let tokenizer = load_vindex_tokenizer(&path)?;
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    let infer_disabled = opts.no_infer || opts.ffn_only || opts.embed_only;
    if opts.embed_only {
        info!("  Mode: embed-service (--embed-only)");
        info!("  Infer: disabled (embed-service mode)");
    } else if opts.ffn_only {
        info!("  Mode: ffn-service (--ffn-only)");
        info!("  Infer: disabled (FFN-service mode)");
    } else if opts.no_infer {
        info!("  Infer: disabled (--no-infer)");
    } else if has_weights {
        info!("  Infer: available (weights detected, will lazy-load on first request)");
    } else {
        info!("  Infer: not available (no model weights in vindex)");
    }

    if opts.release_mmap_after_request {
        info!("  Mmap release: enabled (MADV_DONTNEED after each walk-ffn request)");
    }

    if let Some((start, end)) = opts.expert_filter {
        info!("  Experts: {start}–{end} (shard filter)");
        info!("  Endpoints: POST /v1/expert/batch, /v1/experts/layer-batch, GET /v1/stats");
    }

    let num_layers = config.num_layers;
    Ok(LoadedModel {
        id,
        path,
        config,
        patched: Arc::new(RwLock::new(patched)),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled,
        ffn_only: opts.ffn_only,
        embed_only: opts.embed_only,
        embed_store,
        release_mmap_after_request: opts.release_mmap_after_request,
        weights: std::sync::OnceLock::new(),
        probe_labels,
        ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(num_layers),
        layer_latency_tracker: std::sync::Arc::new(crate::metrics::LayerLatencyTracker::new()),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: opts.expert_filter,
        unit_filter: opts.unit_filter.clone(),
        moe_remote: opts.moe_remote.clone(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

pub fn discover_vindexes(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join(INDEX_JSON).exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

pub fn normalize_serve_alias(args: Vec<String>) -> Vec<String> {
    if args.len() > 1 && args[1] == "serve" {
        std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect()
    } else {
        args
    }
}

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "larql-server",
    version,
    about = "HTTP server for vindex knowledge queries and inference"
)]
pub struct Cli {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    pub vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    pub dir: Option<PathBuf>,

    /// Listen port.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Bind address.
    #[arg(long, default_value = DEFAULT_HOST)]
    pub host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    pub no_infer: bool,

    /// Run as an FFN-service endpoint for remote `RemoteWalkBackend`
    /// clients. Disables `/v1/infer` (like `--no-infer`) and advertises
    /// `mode: ffn-service` in `/v1/stats`. This is Act 2 of the demo —
    /// the server holds the FFN weights, clients hold attention.
    ///
    /// Also skips the f16→f32 gate-vector warmup, which is the largest
    /// eager cost on startup (~2x the gate_vectors.bin size). Gate
    /// decode happens lazily per layer on first request instead.
    #[arg(long)]
    pub ffn_only: bool,

    /// Run as an embed-service endpoint.
    ///
    /// Loads only embeddings.bin, lm_head, and the tokenizer — skips all
    /// FFN and attention weights. Advertises `mode: embed-service` in
    /// `/v1/stats`. Enables `/v1/embed`, `/v1/logits`, and `/v1/token/*`.
    ///
    /// Use this to offload the static embedding + lm_head lookup from
    /// attention-only clients (ADR-0007). The embed slice is ~2-5% of the
    /// full model weight — a minimal VPS can host it independently.
    #[arg(long)]
    pub embed_only: bool,

    /// Only load and serve layers in this range (inclusive, e.g. "0-19").
    /// Layers outside the range are not dequantized and their mmap pages are
    /// never touched, keeping RSS proportional to the shard size.
    /// Requests for out-of-range layers are rejected with HTTP 400.
    #[arg(long)]
    pub layers: Option<String>,

    /// Cap the number of decoded f16 gate layers held in the lazy cache.
    /// 0 = unlimited (default; matches historical behaviour). Each decoded
    /// layer is roughly `intermediate × hidden × 4 bytes` — on 31B that's
    /// ~433 MB per layer, so a 60-layer model fully decoded is ~26 GB.
    /// Set to N to cap at N layers via LRU eviction.
    ///
    /// Use when RSS headroom matters (e.g. co-hosting multiple models) at
    /// the cost of re-decode when evicted layers are re-accessed.
    #[arg(long, default_value_t = DEFAULT_MAX_GATE_CACHE_LAYERS)]
    pub max_gate_cache_layers: usize,

    /// Cap the number of layers held in the Q4_K/Q6_K FFN dequant cache.
    /// 0 = unlimited (default). Only fires on the CPU per-position
    /// fallback in walk_ffn — Metal full-K decode does not populate
    /// this cache. Each cached layer holds up to gate+up+down
    /// dequantised to f32 (`intermediate × hidden × 4 bytes` per
    /// component). On Gemma 3 4B that's ~105 MB/component — set to
    /// 8 for ~840 MB ceiling on the down leg.
    #[arg(long, default_value_t = DEFAULT_MAX_Q4K_CACHE_LAYERS)]
    pub max_q4k_cache_layers: usize,

    /// Use HNSW for gate KNN instead of brute-force matmul. Indexes
    /// are built lazily per layer on first query. Approximate (recall
    /// drops from 100% to 80–95% depending on `--hnsw-ef-search`); the
    /// retrieval ranks by |dot| like the brute path, but oversamples
    /// HNSW and re-ranks at the seam. Wins for high-feature MoE
    /// (64-expert ≈ 230 → 60 ms/layer); break-even or net loss for
    /// dense ≤ 10K-feature models.
    #[arg(long)]
    pub hnsw: bool,

    /// HNSW beam width. Higher = better recall, slower search. 50 is
    /// the floor; 200 is the default; 400 is the practical ceiling.
    #[arg(long, default_value_t = DEFAULT_HNSW_EF_SEARCH)]
    pub hnsw_ef_search: usize,

    /// Eager-build the HNSW index for every owned layer at startup
    /// (rayon-parallel across layers). One-shot; trades ~700 ms of boot
    /// time for first-query latency that would otherwise pay ~76 ms /
    /// layer × N lazy builds spread across the first request volume.
    /// Recommended when this server will see traffic on every layer
    /// (e.g. `larql-router` shards behind a steady-state interp pipeline).
    /// Requires `--hnsw`.
    #[arg(long, requires = "hnsw")]
    pub warmup_hnsw: bool,

    /// Pre-load inference weights and prefetch every owned layer's
    /// Q4K mmap pages at boot. Cuts first-`walk-ffn` latency from
    /// ~1.3 s + 17 ms / cold layer down to the warm baseline
    /// (~0.3 ms / layer) at the cost of a ~1–2 s startup delay and
    /// ~3 GB pre-allocated f32 gate cache. Recommended for grid
    /// shards under a steady-state load — operators can also fire
    /// `POST /v1/warmup` later without a restart.
    #[arg(long)]
    pub warmup_walk_ffn: bool,

    /// Ask the kernel to drop resident mmap pages after each walk-ffn
    /// request (calls `madvise(MADV_DONTNEED)` on every mapping). On
    /// Linux RSS drops immediately; on Darwin the kernel may defer.
    /// Pairs with `--max-gate-cache-layers` to enforce a hard bound.
    ///
    /// Prefer `--layers START-END` for real deployments — sharding
    /// prevents out-of-range pages from ever being touched. This flag
    /// is for the single-shard-holds-everything demo topology.
    #[arg(long)]
    pub release_mmap_after_request: bool,

    /// Only load and serve experts in this range (inclusive, e.g. "0-31").
    /// Requests for out-of-range expert IDs are rejected with HTTP 400.
    /// Used to shard the expert bank across multiple servers.
    /// Layer-uniform: same expert range applies to every layer.
    #[arg(long)]
    pub experts: Option<String>,

    /// Path to a JSON manifest specifying per-(layer, expert) ownership for
    /// fine-grained shards.  Format:
    /// ```json
    /// { "layer_experts": { "0": [[0,31]], "1": [[0,15],[64,79]], ... } }
    /// ```
    /// Each value is a list of inclusive `[start, end]` expert-id ranges.
    /// Layers absent from the map own no experts on this shard.
    ///
    /// When set, overrides `--experts` and switches `run_expert` ownership
    /// checks to per-(layer, expert) lookups.  Designed for the architecture
    /// where each shard hosts a tight set of (layer, expert) units rather
    /// than a contiguous expert range across all layers.
    #[arg(long, value_name = "PATH")]
    pub units: Option<std::path::PathBuf>,

    /// Enable CORS for browser access.
    #[arg(long)]
    pub cors: bool,

    /// Disable the built-in Swagger UI and /v1/openapi.json endpoint.
    #[arg(long)]
    pub no_docs: bool,

    /// API key for authentication (clients send Authorization: Bearer <key>).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Rate limit per IP (e.g., "100/min", "10/sec").
    #[arg(long)]
    pub rate_limit: Option<String>,

    /// Trust X-Forwarded-For when rate limiting.
    ///
    /// Enable only when the server is behind a trusted reverse proxy that
    /// strips untrusted client-supplied forwarding headers.
    #[arg(long)]
    pub trust_forwarded_for: bool,

    /// Max concurrent requests.
    #[arg(long, default_value_t = DEFAULT_MAX_CONCURRENT)]
    pub max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value_t = DEFAULT_DESCRIBE_CACHE_TTL_SECS)]
    pub cache_ttl: u64,

    /// Logging level.
    #[arg(long, default_value = DEFAULT_LOG_LEVEL)]
    pub log_level: String,

    /// gRPC port (enables gRPC server alongside HTTP).
    #[arg(long)]
    pub grpc_port: Option<u16>,

    /// Cosine threshold for the Exp 53 `ShardService.Query` KNN cache.
    /// When set, the gRPC server registers a `ShardService` backed by
    /// an in-memory cache; clients hit it when their query vector
    /// matches an indexed entry at `cos >= tau`. Disk-format loaders
    /// are a follow-up — the v1 cache starts empty and is populated
    /// in-process (typically by tests). Common production value is
    /// `0.97` matching the Python prototype.
    #[arg(long, value_name = "TAU")]
    pub shard_query_tau: Option<f32>,

    /// TLS certificate path for HTTPS.
    #[arg(long)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path for HTTPS.
    #[arg(long)]
    pub tls_key: Option<PathBuf>,

    /// ADR-0019: enable an HTTP/3 listener on this port. Routers
    /// opting into the h3 shard transport (`--http3-shards`) connect
    /// here for per-stream-independent fan-out (escapes TCP HoL
    /// blocking on parallel MoE expert sub-requests). Requires
    /// building with `--features http3`. Coexists with the HTTP/1.1
    /// listener on `--port`; both serve the same axum::Router.
    ///
    /// TLS reuse: if `--tls-cert` and `--tls-key` are set, the h3
    /// listener uses the same cert. Otherwise, an in-memory
    /// self-signed cert is generated at startup and its SHA-256
    /// fingerprint is logged — clients pin it via
    /// `--shard-cert-fingerprint` on the router side.
    #[arg(long)]
    #[cfg(feature = "http3")]
    pub http3_port: Option<u16>,

    /// Bind a Unix domain socket alongside the TCP listener for same-host
    /// MoE shard clients.  Skips the kernel TCP stack and saves ~50 µs/call
    /// on loopback.  Path is created at startup; pre-existing socket files
    /// are unlinked.  Clients reach the shard via a `unix:///path/to/sock`
    /// URL in `--moe-shards`.
    #[arg(long, value_name = "PATH")]
    pub uds_path: Option<PathBuf>,

    /// Join one or more router grids (comma-separated gRPC addresses).
    /// Example: "http://router-a:50052,http://router-b:50052"
    /// Each router gets an independent announce stream — stateless fan-out.
    /// Requires --public-url so routers know where to send clients.
    #[arg(long)]
    pub join: Option<String>,

    /// Public HTTP URL clients should use to reach this server.
    /// Used when announcing to the grid with --join.
    /// Example: "http://server-a:8080"
    #[arg(long)]
    pub public_url: Option<String>,

    /// Shared secret matching the router's --grid-key.
    /// Required when the router enforces grid authentication.
    #[arg(long, env = "LARQL_GRID_KEY")]
    pub grid_key: Option<String>,

    /// Mode B: advertise available RAM to the router (no vindex preloaded).
    /// The router will assign a shard via AssignMsg.
    /// Example: "24GB" or "16384MB" or raw bytes "17179869184".
    /// Requires --join and --vindex-store.
    #[arg(long, value_name = "SIZE")]
    pub available_ram: Option<String>,

    /// Mode B: directory where assigned shards will be downloaded.
    /// The router assigns a shard; this server downloads it here.
    /// Example: "/mnt/shards/"
    #[arg(long, value_name = "PATH")]
    pub vindex_store: Option<String>,

    /// ADR-0010: SHA-256 fingerprint (hex) of the router's QUIC server
    /// cert. Required only when `--join` uses the `quic://` scheme.
    /// Without this, the QUIC client skips certificate verification —
    /// LAN / dev only.
    #[arg(long, value_name = "HEX")]
    pub quic_cert_fingerprint: Option<String>,

    /// Server-side MoE expert shard map: `"START-END=URL,START-END=URL,..."`
    /// The walk-ffn handler dispatches MoE expert calls to these remote servers.
    /// Combine with --layers for full 2D (layer × expert) sharding.
    /// Mutually exclusive with --moe-units-manifest.
    #[arg(long)]
    pub moe_shards: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) shard ownership.
    /// Same format as `larql run --moe-units-manifest`. Mutually exclusive with --moe-shards.
    #[arg(long, value_name = "PATH")]
    pub moe_units_manifest: Option<PathBuf>,
}

// ── Server lifecycle ──────────────────────────────────────────────────────────

/// Boot the server: load every vindex named on the command line, build the
/// router, run any opt-in warmups, then bind the TCP listener (plus optional
/// UDS / TLS / gRPC sockets) and run forever.
///
/// `main` is a thin wrapper: parse `Cli`, init tracing, hand off here. Splitting
/// ADR-0019 — spawn an HTTP/3 listener alongside the existing
/// HTTP/1.1 TCP listener when `--http3-port` is set. Reuses the
/// TLS cert from `--tls-cert`/`--tls-key` if both are set;
/// otherwise auto-generates a self-signed leaf cert and prints its
/// fingerprint so the router operator can pin it.
///
/// The h3 listener serves the same `axum::Router` as the dense
/// path — handlers are identical, only the transport differs.
#[cfg(feature = "http3")]
async fn spawn_http3_listener_if_configured(cli: &Cli, app: axum::Router) -> Result<(), BoxError> {
    let Some(port) = cli.http3_port else {
        return Ok(());
    };

    use larql_router_protocol::transport::h3 as h3_transport;
    use larql_router_protocol::transport::quic as quic_transport;

    // Install the rustls ring crypto provider once. Safe to call
    // multiple times — second call is a no-op.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // TLS material: prefer `--tls-cert`/`--tls-key` (reuses the HTTPS
    // pair); fall back to an auto-generated self-signed cert. We
    // print the fingerprint either way so operators have one log
    // line they can hand to the router's `--shard-cert-fingerprint`.
    let tls = if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        let cert_pem = std::fs::read_to_string(cert_path)
            .map_err(|e| format!("read --tls-cert {}: {e}", cert_path.display()))?;
        let key_pem = std::fs::read_to_string(key_path)
            .map_err(|e| format!("read --tls-key {}: {e}", key_path.display()))?;
        // Server name embedded in the cert isn't used by the router
        // when fingerprint-pinning, but we keep the convention here.
        quic_transport::SelfSignedTls {
            cert_pem,
            key_pem,
            fingerprint: String::new(),
            server_name: "larql-server".to_string(),
        }
    } else {
        let generated = quic_transport::self_signed_tls("larql-server")
            .map_err(|e| format!("self-signed cert generation: {e}"))?;
        info!(
            fingerprint = %generated.fingerprint,
            "HTTP/3: generated self-signed cert. Routers must pin this \
             fingerprint via --shard-cert-fingerprint when opting into \
             --http3-shards."
        );
        generated
    };

    let addr: std::net::SocketAddr = format!("{}:{}", cli.host, port).parse()?;
    let endpoint = h3_transport::server_endpoint(addr, &tls)
        .map_err(|e| format!("h3 endpoint bind {addr}: {e}"))?;
    info!("Listening: h3 (HTTP/3 over QUIC) on {addr}");

    tokio::spawn(async move {
        if let Err(e) = h3_transport::serve_axum(endpoint, app).await {
            tracing::error!("h3 listener crashed: {e:#}");
        }
    });
    Ok(())
}

/// the orchestration out lets integration tests drive boot without going
/// through `clap::Parser::parse_from`.
pub async fn serve(cli: Cli) -> Result<(), BoxError> {
    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();

    let layer_range = cli.layers.as_deref().map(parse_layer_range).transpose()?;
    let expert_filter = cli.experts.as_deref().map(parse_layer_range).transpose()?;
    // --units PATH (per-(layer, expert) ownership manifest) takes precedence
    // over --experts START-END; the two are mutually exclusive at parse time
    // so the operator gets a clear error rather than silently picking one.
    if cli.units.is_some() && cli.experts.is_some() {
        return Err("--units and --experts are mutually exclusive — \
             use --experts for layer-uniform ranges, --units for fine-grained ownership"
            .into());
    }
    let unit_filter = cli
        .units
        .as_deref()
        .map(parse_unit_manifest)
        .transpose()?
        .map(Arc::new);
    if let Some(ref u) = unit_filter {
        info!(
            "  Units (--units): {} (layer, expert) pairs across {} layers",
            u.len(),
            u.iter()
                .map(|(l, _)| *l)
                .collect::<std::collections::HashSet<_>>()
                .len(),
        );
    }
    // Build server-side MoE remote backend (--moe-shards or --moe-units-manifest).
    if cli.moe_shards.is_some() && cli.moe_units_manifest.is_some() {
        return Err("--moe-shards and --moe-units-manifest are mutually exclusive".into());
    }
    let moe_remote: Option<Arc<larql_inference::ffn::RemoteMoeBackend>> =
        if let Some(ref s) = cli.moe_shards {
            use larql_inference::ffn::moe_remote::ShardConfig;
            let mut cfgs: Vec<ShardConfig> = Vec::new();
            for segment in s.split(',') {
                let segment = segment.trim();
                if segment.is_empty() {
                    continue;
                }
                let mut parts = segment.splitn(2, '=');
                let range_str = parts.next().ok_or_else(|| -> BoxError {
                    format!("malformed --moe-shards segment: {segment:?}").into()
                })?;
                let url = parts.next().ok_or_else(|| -> BoxError {
                    format!("missing URL in --moe-shards segment: {segment:?}").into()
                })?;
                let (start, end_incl) =
                    ShardConfig::parse_range(range_str).ok_or_else(|| -> BoxError {
                        format!("bad expert range {range_str:?} in --moe-shards").into()
                    })?;
                cfgs.push(ShardConfig::new(start, end_incl, url));
            }
            if cfgs.is_empty() {
                return Err("--moe-shards: no valid segments found".into());
            }
            let n = cfgs.len();
            let backend = larql_inference::ffn::RemoteMoeBackend::connect(cfgs)
                .map_err(|e| -> BoxError { format!("--moe-shards connect: {e}").into() })?;
            info!("  MoE experts: remote ({n} shard(s) via --moe-shards)");
            Some(Arc::new(backend))
        } else if let Some(ref path) = cli.moe_units_manifest {
            use larql_inference::ffn::moe_remote::parse_unit_manifest;
            let cfgs = parse_unit_manifest(path)
                .map_err(|e| -> BoxError { format!("--moe-units-manifest: {e}").into() })?;
            let n = cfgs.len();
            let backend = larql_inference::ffn::RemoteMoeBackend::connect(cfgs)
                .map_err(|e| -> BoxError { format!("--moe-units-manifest connect: {e}").into() })?;
            info!("  MoE experts: remote ({n} shard(s) via --moe-units-manifest)");
            Some(Arc::new(backend))
        } else {
            None
        };

    let load_opts = LoadVindexOptions {
        no_infer: cli.no_infer,
        ffn_only: cli.ffn_only,
        embed_only: cli.embed_only,
        layer_range,
        max_gate_cache_layers: cli.max_gate_cache_layers,
        max_q4k_cache_layers: cli.max_q4k_cache_layers,
        hnsw: if cli.hnsw {
            Some(cli.hnsw_ef_search)
        } else {
            None
        },
        warmup_hnsw: cli.warmup_hnsw,
        release_mmap_after_request: cli.release_mmap_after_request,
        expert_filter,
        unit_filter,
        moe_remote,
    };

    if let Some(ref dir) = cli.dir {
        let paths = discover_vindexes(dir);
        if paths.is_empty() {
            return Err(format!("no .vindex directories found in {}", dir.display()).into());
        }
        info!("Found {} vindexes in {}", paths.len(), dir.display());
        for p in &paths {
            // `LoadVindexOptions` is `Clone` (was `Copy` until `unit_filter`
            // added an `Arc<HashSet<...>>` field) — clone per iteration so
            // the loop owns each call's argument.
            match load_single_vindex(&p.to_string_lossy(), load_opts.clone()) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path, load_opts)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    let rate_limiter =
        cli.rate_limit
            .as_ref()
            .and_then(|spec| match ratelimit::RateLimiter::parse(spec) {
                Some(rl) => {
                    info!("Rate limit: {}", spec);
                    Some(Arc::new(rl))
                }
                None => {
                    warn!(
                        "Invalid rate limit format: {} (expected e.g. '100/min')",
                        spec
                    );
                    None
                }
            });

    let state = Arc::new(AppState {
        models: models.clone(),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(DEFAULT_SESSION_TTL_SECS),
        describe_cache: DescribeCache::new(cli.cache_ttl),
    });

    if cli.cache_ttl > 0 {
        info!("DESCRIBE cache: {}s TTL", cli.cache_ttl);
    }

    let is_multi = state.is_multi_model();
    let mut app = if is_multi {
        info!("Multi-model mode ({} models)", state.models.len());
        for m in &state.models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(Arc::clone(&state))
    } else {
        let m = &models[0];
        info!("Single-model mode: {}", m.config.model);
        routes::single_model_router(Arc::clone(&state))
    };

    // `--warmup-walk-ffn` — pre-load inference weights + prefetch every
    // owned layer's Q4K mmap so the first `/v1/walk-ffn` doesn't pay
    // the ~1.3 s lazy weight load + ~17 ms / cold layer (see
    // ROADMAP G1 / G2). Same code path as `POST /v1/warmup`.
    if cli.warmup_walk_ffn {
        for m in &state.models {
            let req = routes::warmup::WarmupRequest {
                layers: None,
                skip_weights: false,
                warmup_hnsw: false,
            };
            let r = routes::warmup::warmup_model_async(Arc::clone(m), req).await;
            info!(
                "  Warmup walk-ffn[{}]: weights={} ({}ms), prefetched {} layers ({}ms), total {}ms",
                r.model,
                r.weights_loaded,
                r.weights_load_ms,
                r.layers_prefetched,
                r.prefetch_ms,
                r.total_ms,
            );
        }
    }

    // Per-(layer, expert) HNSW unit warmup.
    for m in &state.models {
        if m.expert_filter.is_none() && !cli.warmup_walk_ffn {
            continue;
        }
        let model = Arc::clone(m);
        let model_id = model.id.clone();
        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            crate::routes::expert::warmup_hnsw_unit_cache(&model)
        })
        .await;
        match result {
            Ok(Ok((built, n_layers, n_owned))) if built > 0 => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                info!(
                    "  Warmup hnsw-units[{model_id}]: built {built} units \
                     ({n_layers} layers × {n_owned} experts/shard) in {elapsed_ms:.0}ms"
                );
            }
            Ok(Ok(_)) => {}
            Ok(Err(e)) => warn!("Warmup hnsw-units[{model_id}] failed: {e}"),
            Err(e) => warn!("Warmup hnsw-units[{model_id}] join failed: {e}"),
        }
    }

    // Metal expert cache warmup (cfg=metal-experts only).
    #[cfg(all(feature = "metal-experts", target_os = "macos"))]
    for m in &state.models {
        if m.expert_filter.is_none() {
            continue;
        }
        let model = Arc::clone(m);
        let model_id = model.id.clone();
        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            crate::routes::expert::warmup_metal_expert_cache(&model)
        })
        .await;
        match result {
            Ok(Ok(staged)) => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if staged > 0 {
                    info!(
                        "  Warmup metal-experts[{model_id}]: staged {staged} \
                         (gate_up, down) buffer pairs in {elapsed_ms:.0}ms"
                    );
                }
            }
            Ok(Err(e)) => warn!("Warmup metal-experts[{model_id}] failed: {e}"),
            Err(e) => warn!("Warmup metal-experts[{model_id}] join failed: {e}"),
        }
    }

    // Rate limiting middleware.
    if let Some(ref rl) = rate_limiter {
        let rate_state = Arc::new(ratelimit::RateLimitState {
            limiter: Arc::clone(rl),
            trust_forwarded_for: cli.trust_forwarded_for,
        });
        app = app.layer(middleware::from_fn_with_state(
            rate_state,
            ratelimit::rate_limit_middleware,
        ));
        if cli.trust_forwarded_for {
            info!("Rate limit: trusting X-Forwarded-For");
        }
    }

    // OpenAPI / Swagger UI. Mounted before auth so the docs stay reachable
    // without the API key — consistent with --cors behavior. Flip the
    // ordering if operators want docs gated.
    if !cli.no_docs {
        app = app.merge(crate::openapi::swagger_router());
        info!("OpenAPI: /swagger-ui and /v1/openapi.json enabled");
    }

    // Auth middleware.
    if cli.api_key.is_some() {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::auth_middleware,
        ));
        info!("Auth: API key required");
    }

    // CORS.
    if cli.cors {
        use tower_http::cors::CorsLayer;
        app = app.layer(CorsLayer::permissive());
        info!("CORS: enabled");
    }

    // Concurrency limit.
    app = app.layer(tower::limit::ConcurrencyLimitLayer::new(cli.max_concurrent));
    info!("Max concurrent: {}", cli.max_concurrent);

    // Trace middleware.
    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    // gRPC server (if --grpc-port set).
    if let Some(grpc_port) = cli.grpc_port {
        let grpc_addr = format!("{}:{}", cli.host, grpc_port).parse()?;
        let grpc_state = Arc::clone(&state);
        // Exp 53 ShardService. Vindex-backed: the cache shares the
        // server's loaded `PatchedVindex`, so "compiled facts" live as
        // vindex patches (via `PatchedVindex::add_patch` etc.) and we
        // don't maintain a separate on-disk cache format. Opt-in via
        // `--shard-query-tau`; deployments that don't set it pay zero
        // for the feature.
        let shard_source = cli.shard_query_tau.and_then(|tau| {
            let model = state.models.first()?;
            info!(
                "ShardService: enabled on model {} with tau={tau} (vindex-backed)",
                model.id
            );
            // Share the model's live `Arc<RwLock<PatchedVindex>>` —
            // patches added at runtime via `model.patched.write().await`
            // are immediately visible to the shard service, and the
            // shard service sees the same patch lineage the inference
            // path walks. No snapshot, no clone of the base.
            Some(crate::shard_query::ShardSource::vindex(
                std::sync::Arc::clone(&model.patched),
                tau,
            ))
        });
        info!("gRPC: listening on {}", grpc_addr);
        tokio::spawn(async move {
            let vindex_svc = grpc::VindexGrpcService {
                state: Arc::clone(&grpc_state),
            };
            let expert_svc = grpc_expert::ExpertGrpcService {
                state: Arc::clone(&grpc_state),
            };
            let mut builder = tonic::transport::Server::builder()
                .add_service(
                    grpc::proto::vindex_service_server::VindexServiceServer::new(vindex_svc),
                )
                .add_service(larql_router_protocol::ExpertServiceServer::new(expert_svc));
            if let Some(source) = shard_source {
                let shard_svc = crate::shard_query::ShardGrpcService::new(source);
                builder =
                    builder.add_service(larql_router_protocol::ShardServiceServer::new(shard_svc));
            }
            if let Err(e) = builder.serve(grpc_addr).await {
                tracing::error!("gRPC server error: {}", e);
            }
        });
    }

    let addr = format!("{}:{}", cli.host, cli.port);

    // Grid announce (if --join provided).
    if let Some(join_spec) = cli.join.clone() {
        let listen_url = cli.public_url.clone().unwrap_or_else(|| {
            let host = if cli.host == "0.0.0.0" {
                "127.0.0.1"
            } else {
                &cli.host
            };
            format!("http://{}:{}", host, cli.port)
        });
        let join_urls: Vec<String> = join_spec
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        if join_urls.len() > 1 {
            info!("Joining {} routers (stateless fan-out)", join_urls.len());
        }
        // Mode B: --available-ram without a loaded model → advertise capacity.
        if let Some(ref ram_str) = cli.available_ram {
            match parse_ram_bytes(ram_str) {
                Ok(ram_bytes) => {
                    let store_path = cli
                        .vindex_store
                        .clone()
                        .unwrap_or_else(|| "/tmp/larql-shards".to_string());
                    for join_url in &join_urls {
                        announce::run_announce_available(announce::AvailableConfig {
                            join_url: join_url.clone(),
                            listen_url: listen_url.clone(),
                            ram_bytes,
                            disk_bytes: 0, // TODO: query disk
                            store_path: store_path.clone(),
                            grid_key: cli.grid_key.clone(),
                            quic_cert_fingerprint: cli.quic_cert_fingerprint.clone(),
                        });
                    }
                }
                Err(e) => {
                    warn!("--available-ram parse error: {e} — falling through to Mode A");
                }
            }
        }

        // If the deployer supplied --available-ram alongside a loaded model,
        // build a reusable Mode B fallback config so the server re-enters the
        // available pool after a drain instead of just disconnecting (GT6
        // §Phase B2). The construction logic + tests live in `announce.rs`.
        let available_after_drain = announce::build_available_after_drain(
            cli.available_ram
                .as_deref()
                .and_then(|s| parse_ram_bytes(s).ok()),
            &listen_url,
            cli.vindex_store.as_deref(),
            cli.grid_key.as_deref(),
        );

        for m in &models {
            let (layer_start, layer_end) = match layer_range {
                Some((s, e)) => (s as u32, (e - 1) as u32),
                None => (0, (m.config.num_layers.saturating_sub(1)) as u32),
            };
            let vhash = announce::vindex_identity_hash(&m.id, m.config.num_layers);
            for join_url in &join_urls {
                let avail = available_after_drain.as_ref().map(|base| {
                    let mut a = base.clone();
                    a.join_url = join_url.clone();
                    a
                });
                announce::run_announce(announce::AnnounceConfig {
                    join_url: join_url.clone(),
                    model_id: m.id.clone(),
                    layer_start,
                    layer_end,
                    listen_url: listen_url.clone(),
                    ram_bytes: 0,
                    grid_key: cli.grid_key.clone(),
                    vindex_hash: vhash.clone(),
                    latency_tracker: m.layer_latency_tracker.clone(),
                    requests_in_flight: m.requests_in_flight.clone(),
                    requests_total: m.requests_total.clone(),
                    available_after_drain: avail,
                    quic_cert_fingerprint: cli.quic_cert_fingerprint.clone(),
                });
            }
        }
    }

    // TLS or plain HTTP.
    if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        info!(
            "TLS: enabled ({}, {})",
            cert_path.display(),
            key_path.display()
        );
        info!("Listening: https://{}", addr);

        let tls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_path, key_path).await?;

        axum_server::bind_rustls(addr.parse()?, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        // Optional Unix domain socket alongside TCP (for same-host MoE
        // shard clients). Unix-only — `tokio::net::UnixListener` is
        // gated on `cfg(unix)`. On Windows we warn and serve TCP only;
        // the same-host MoE optimisation is unavailable.
        if let Some(uds_path) = cli.uds_path.clone() {
            #[cfg(unix)]
            {
                let _ = std::fs::remove_file(&uds_path);
                match tokio::net::UnixListener::bind(&uds_path) {
                    Ok(uds_listener) => {
                        info!("Listening: unix://{}", uds_path.display());
                        let uds_app = app.clone();
                        tokio::spawn(async move {
                            if let Err(e) = axum::serve(uds_listener, uds_app).await {
                                tracing::error!(
                                    "UDS listener crashed: {e:#}; same-host MoE shard \
                                     clients will need to fall back to TCP"
                                );
                            }
                        });
                    }
                    Err(e) => warn!(
                        "failed to bind UDS at {}: {e:#}; serving TCP only",
                        uds_path.display()
                    ),
                }
            }
            #[cfg(not(unix))]
            warn!(
                "--uds-path {} ignored: Unix domain sockets are unix-only; \
                 serving TCP only",
                uds_path.display()
            );
        }

        // ADR-0019: optional HTTP/3 listener alongside the HTTP/1.1
        // TCP listener. Spawned only when `--http3-port` is set and
        // the crate is built with `--features http3`. Both listeners
        // share the same `axum::Router`, so request handlers are
        // identical regardless of transport — the only difference is
        // per-stream independence on the wire.
        #[cfg(feature = "http3")]
        spawn_http3_listener_if_configured(&cli, app.clone()).await?;

        info!("Listening: http://{}", addr);
        // `set_nodelay(true)` on every accepted connection — disables
        // Nagle's algorithm so the response tail-packet isn't held
        // waiting for ACK coalescence. The MoE layer-batch path
        // round-trips ~12 KB request + ~11 KB response per layer × 30
        // layers/token; without TCP_NODELAY the last partial packet
        // can be held by the kernel for 40 ms (Linux delayed-ACK timer)
        // or 200 ms (BSD).
        use axum::serve::ListenerExt;
        let listener = tokio::net::TcpListener::bind(&addr)
            .await?
            .tap_io(|stream| {
                if let Err(e) = stream.set_nodelay(true) {
                    tracing::warn!("failed to set TCP_NODELAY on accepted connection: {e:#}");
                }
            });
        axum::serve(listener, app).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ram_bytes_gb() {
        assert_eq!(parse_ram_bytes("24GB").unwrap(), 24 * 1024 * 1024 * 1024);
        assert_eq!(parse_ram_bytes("16gb").unwrap(), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_ram_bytes_mb() {
        assert_eq!(parse_ram_bytes("4096MB").unwrap(), 4096 * 1024 * 1024);
    }

    #[test]
    fn parse_ram_bytes_raw() {
        assert_eq!(parse_ram_bytes("1073741824").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_ram_bytes_invalid() {
        assert!(parse_ram_bytes("notanumber").is_err());
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "larql-server-bootstrap-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── Unit-manifest parser ─────────────────────────────────────────────
    //
    // The JSON shape the operator hands the server must round-trip through
    // `parse_unit_manifest` into a deterministic ownership set.  Tests
    // cover: well-formed multi-range manifest, bad layer key, reversed
    // range, missing file.  The data shape is exercised end-to-end here so
    // ownership-check and warmup loops can rely on it without having to
    // re-validate.

    fn write_units_file(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("units.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn parse_unit_manifest_round_trips_per_layer_ranges() {
        let dir = unique_temp_dir("units-ok");
        let path = write_units_file(
            &dir,
            r#"{"layer_experts": {"0": [[0,2]], "3": [[5,7],[10,10]]}}"#,
        );
        let units = parse_unit_manifest(&path).unwrap();
        // Layer 0: experts 0..=2 → (0,0), (0,1), (0,2)
        // Layer 3: experts 5..=7 + 10 → (3,5), (3,6), (3,7), (3,10)
        let expected: std::collections::HashSet<(usize, usize)> =
            [(0, 0), (0, 1), (0, 2), (3, 5), (3, 6), (3, 7), (3, 10)]
                .into_iter()
                .collect();
        assert_eq!(units, expected);
    }

    #[test]
    fn parse_unit_manifest_rejects_non_numeric_layer_key() {
        let dir = unique_temp_dir("units-bad-layer");
        let path = write_units_file(&dir, r#"{"layer_experts": {"oops": [[0,2]]}}"#);
        let err = parse_unit_manifest(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("layer key 'oops'"), "got: {msg}");
    }

    #[test]
    fn parse_unit_manifest_rejects_reversed_range() {
        let dir = unique_temp_dir("units-bad-range");
        let path = write_units_file(&dir, r#"{"layer_experts": {"0": [[5,2]]}}"#);
        let err = parse_unit_manifest(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("end (2) must be >= start (5)"), "got: {msg}");
    }

    #[test]
    fn parse_unit_manifest_missing_file_reports_path() {
        let bogus = PathBuf::from("/nonexistent/larql-units-not-here.json");
        let err = parse_unit_manifest(&bogus).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("read"),
            "msg should mention read failure: {msg}"
        );
        assert!(
            msg.contains(bogus.to_str().unwrap()),
            "msg should name path: {msg}"
        );
    }

    #[test]
    fn parse_unit_manifest_accepts_empty_object() {
        // Operator may want to test the wiring without owning any units —
        // empty manifest should yield an empty set, not error.
        let dir = unique_temp_dir("units-empty");
        let path = write_units_file(&dir, r#"{"layer_experts": {}}"#);
        let units = parse_unit_manifest(&path).unwrap();
        assert!(units.is_empty());
    }

    #[test]
    fn parse_layer_range_accepts_inclusive_cli_range() {
        assert_eq!(parse_layer_range("0-19").unwrap(), (0, 20));
        assert_eq!(parse_layer_range(" 2 - 2 ").unwrap(), (2, 3));
    }

    #[test]
    fn parse_layer_range_rejects_bad_shapes() {
        assert!(parse_layer_range("0").is_err());
        assert!(parse_layer_range("x-2").is_err());
        assert!(parse_layer_range("2-x").is_err());
        assert!(parse_layer_range("3-2").is_err());
    }

    #[test]
    fn normalize_serve_alias_removes_subcommand() {
        let filtered = normalize_serve_alias(vec![
            "larql-server".into(),
            "serve".into(),
            "model.vindex".into(),
        ]);
        assert_eq!(filtered, vec!["larql-server", "model.vindex"]);
    }

    #[test]
    fn normalize_serve_alias_leaves_non_alias_args_unchanged() {
        let args = vec!["larql-server".into(), "model.vindex".into()];
        assert_eq!(normalize_serve_alias(args.clone()), args);
    }

    #[test]
    fn discover_vindexes_returns_sorted_dirs_with_index_json() {
        let dir = unique_temp_dir("discover");
        let b = dir.join("b.vindex");
        let a = dir.join("a.vindex");
        let ignored = dir.join("ignored.vindex");
        std::fs::create_dir_all(&b).unwrap();
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&ignored).unwrap();
        std::fs::write(b.join(INDEX_JSON), "{}").unwrap();
        std::fs::write(a.join(INDEX_JSON), "{}").unwrap();

        let paths = discover_vindexes(&dir);
        assert_eq!(paths, vec![a, b]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_options_are_copyable() {
        let opts = LoadVindexOptions {
            no_infer: true,
            ffn_only: false,
            embed_only: false,
            layer_range: Some((0, 2)),
            max_gate_cache_layers: 1,
            max_q4k_cache_layers: 2,
            hnsw: Some(200),
            warmup_hnsw: true,
            release_mmap_after_request: true,
            expert_filter: Some((3, 4)),
            unit_filter: None,
            moe_remote: None,
        };
        let copied = opts.clone();
        assert!(copied.no_infer);
        assert_eq!(copied.layer_range, Some((0, 2)));
        assert_eq!(copied.expert_filter, Some((3, 4)));
    }
}
