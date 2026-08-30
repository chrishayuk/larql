//! CLI surface for `larql-server` — the clap definition, argument
//! defaults, and small argv/argument parsers.

use std::path::PathBuf;

use clap::Parser;

use super::BoxError;

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
// Owned by the session module (the manager applies it when the flag is 0);
// re-exported here so the CLI surface stays one import for callers.
pub use crate::session::DEFAULT_SESSION_TTL_SECS;

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

    /// Serve the PUBLIC_EXPLORER surface: the read-only route set plus
    /// `POST /v1/query`, which executes LQL under
    /// `CapabilityProfile::PublicExplorer` (mutation and lifecycle
    /// statements refuse after parsing, before execution). Requires
    /// exactly one VINDEX3 container. Pair with `--rate-limit` and
    /// `--max-concurrent` on anything internet-facing.
    #[arg(long)]
    pub public_explorer: bool,

    /// Defer model-weight loading until the first `/v1/infer` (or
    /// other inference) request, instead of loading at startup.
    ///
    /// The eager startup load is the default because:
    ///
    /// - Lazy load happens on a request thread under HTTP handler
    ///   backpressure, and a 5+ GB allocation under cgroup pressure
    ///   reliably triggers an OOM-kill on memory-constrained hosts
    ///   (see `BUG-infer-deadlock.md`).  Eager load surfaces the
    ///   same condition as a clean startup failure that systemd
    ///   reports loudly, *before* the listener binds.
    /// - Lazy first-callers double-allocated until the single-flight
    ///   `weights_init` guard landed; eager load avoids that path
    ///   entirely on hosts where every inference call is going to
    ///   trigger the load anyway.
    ///
    /// Pass this flag if you want the historical lazy behaviour
    /// (e.g. for `--ffn-only` boxes that *might* be promoted to
    /// inference later, or in tests).
    ///
    /// Note: `--lazy-weights` also skips the startup memory
    /// pre-flight check (there is nothing to size before the
    /// deferred load), so a too-small-RAM condition surfaces on the
    /// first request rather than at startup.
    #[arg(long)]
    pub lazy_weights: bool,

    /// Skip the startup cgroup memory pre-flight check (BUG
    /// `infer-deadlock-oom` §5.5).  By default the server reads
    /// `/sys/fs/cgroup/<self>/memory.max` and refuses to start when
    /// the vindex's estimated resident size + a 512 MiB headroom
    /// reserve exceeds the limit.  Pass `--no-memcheck` to override
    /// (e.g. for cases where the estimate is wrong, or when running
    /// in an environment without cgroup v2).
    #[arg(long)]
    pub no_memcheck: bool,

    /// Headroom (MiB) to reserve below `memory.max` for the OS,
    /// allocator overhead, and the request-handling working set.
    /// Used by the startup pre-flight; ignored when
    /// `--no-memcheck` is set.
    #[arg(long, default_value_t = 512)]
    pub memcheck_headroom_mib: u64,

    /// Per-request hard timeout for `/v1/infer` and other inference
    /// endpoints, in seconds.  When the inference exceeds this, the
    /// handler responds 504 Gateway Timeout and drops the
    /// `spawn_blocking` JoinHandle.  The blocking thread runs to
    /// completion in the background; its result is discarded.
    /// Set to 0 to disable.  See BUG-infer-deadlock §5.6.
    #[arg(long, default_value_t = 60)]
    pub infer_timeout_secs: u64,

    /// Idle TTL for `X-Session-Id` patch sessions, in seconds. A session
    /// with no write access for longer than this is evicted by the
    /// background maintenance sweeper and its patches are dropped.
    /// Set to 0 to use the default (one hour).
    #[arg(long, default_value_t = DEFAULT_SESSION_TTL_SECS)]
    pub session_ttl_secs: u64,

    /// N1 — resident KV continuation states for chained V3 responses
    /// (`previous_response_id` skips re-prefilling the conversation).
    /// Each entry can hold a whole conversation's KV, so size this to
    /// your RAM. Set to 0 to disable (every chained turn re-prefills;
    /// results are identical either way).
    #[arg(long, default_value_t = crate::response_kv::DEFAULT_MAX_ENTRIES)]
    pub v3_kv_cache_entries: usize,

    /// Idle TTL for resident V3 KV continuation states, in seconds.
    /// Set to 0 to use the default (ten minutes).
    #[arg(long, default_value_t = crate::response_kv::DEFAULT_TTL_SECS)]
    pub v3_kv_ttl_secs: u64,

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
