//! larql-router — transparent layer-sharding proxy for larql-server.
//!
//! Two dispatch modes:
//!   --shards  "0-16=http://host-a:8080,17-33=http://host-b:8081"
//!             Static shard map (ADR-0003, backwards-compatible).
//!   --grid-port 50052
//!             Self-assembling grid (ADR-0004). Servers connect via gRPC
//!             and announce their capabilities. No static configuration.
//!
//! Both modes can coexist. Grid takes priority; static shards are fallback.
//!
//! # Wire format
//!
//! The router is wire-transparent for both JSON (`application/json`) and binary
//! (`application/x-larql-ffn`) requests. For single-shard routes the body is
//! forwarded byte-for-byte with no intermediate parsing. Multi-shard fan-out
//! is supported for JSON only; binary multi-shard requests are rejected with
//! HTTP 400 (use the batched JSON format or route per-shard manually).

use larql_router::grid;
use larql_router::tasks::rebalancer;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use parking_lot::RwLock;
use tonic::transport::Server as GrpcServer;
use tracing::{info, warn};

use grid::service::GridServiceImpl;
use grid::GridState;
use larql_router_protocol::GridServiceServer;

#[cfg(feature = "quic")]
fn spawn_quic_listener(
    cli: &Cli,
    state: Arc<RwLock<GridState>>,
    quic_port: u16,
    metrics: Arc<larql_router::metrics::RouterMetrics>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use larql_router_protocol::transport::quic::{
        self_signed_tls, server_endpoint, spawn_accept_loop, SelfSignedTls,
    };
    use tokio_stream::wrappers::ReceiverStream;

    // Install the rustls ring crypto provider once. Safe to call from
    // anywhere; subsequent calls are no-ops.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let tls = match (cli.quic_cert.as_ref(), cli.quic_key.as_ref()) {
        (Some(cert), Some(key)) => {
            let cert_pem = std::fs::read_to_string(cert)
                .map_err(|e| format!("read --quic-cert {}: {e}", cert.display()))?;
            let key_pem = std::fs::read_to_string(key)
                .map_err(|e| format!("read --quic-key {}: {e}", key.display()))?;
            SelfSignedTls {
                cert_pem,
                key_pem,
                fingerprint: String::new(),
                server_name: cli.quic_server_name.clone(),
            }
        }
        (None, None) => {
            let generated = self_signed_tls(&cli.quic_server_name)
                .map_err(|e| format!("self-signed cert generation: {e}"))?;
            info!(
                fingerprint = %generated.fingerprint,
                server_name = %generated.server_name,
                "QUIC: generated self-signed cert. Clients must pin this fingerprint via --quic-cert-fingerprint."
            );
            generated
        }
        _ => {
            return Err(
                "--quic-cert and --quic-key must be provided together (or neither, for self-signed)"
                    .into(),
            );
        }
    };

    let quic_addr: SocketAddr = format!("{}:{}", cli.host, quic_port).parse()?;
    let endpoint = server_endpoint(quic_addr, &tls)
        .map_err(|e| format!("QUIC endpoint bind {quic_addr}: {e}"))?;
    info!("Grid QUIC server listening: {quic_addr}");

    let svc = GridServiceServer::new(
        GridServiceImpl::new_with_key(state, cli.grid_key.clone()).with_metrics(metrics),
    );
    let rx = spawn_accept_loop(endpoint);
    let incoming = ReceiverStream::new(rx);
    tokio::spawn(async move {
        if let Err(e) = GrpcServer::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
        {
            tracing::error!("QUIC server error: {e}");
        }
    });
    Ok(())
}

use larql_router::shards::parse_shards;

// ── CLI ────────────────────────────────────────────────────────────────────────

/// Top-level CLI. When no subcommand is given, the router runs as a daemon
/// using the flags on the `Cli` struct (the historical behavior). The
/// `Admin` subcommands open a one-shot client connection to a running
/// router and exit.
#[derive(Parser)]
#[command(
    name = "larql-router",
    version,
    about = "Layer-sharding proxy for larql-server"
)]
struct CliRoot {
    #[command(subcommand)]
    admin: Option<AdminCmd>,

    #[command(flatten)]
    daemon: Cli,
}

#[derive(clap::Subcommand)]
enum AdminCmd {
    /// Print the current grid status (servers, shards, gaps).
    Status {
        /// Router gRPC URL. Default: `http://localhost:50052`.
        #[arg(long, default_value = "http://localhost:50052")]
        router: String,
    },
    /// Report coverage gaps per model.
    Gaps {
        #[arg(long, default_value = "http://localhost:50052")]
        router: String,
        /// Filter to a single model_id. Empty = all models.
        #[arg(long)]
        model: Option<String>,
    },
    /// Send `UnassignMsg` to a serving server so it drains and exits.
    Drain {
        #[arg(long, default_value = "http://localhost:50052")]
        router: String,
        /// server_id (as returned by `status`).
        #[arg(long)]
        server: String,
        /// Free-form reason; surfaced to the server as `UnassignMsg.reason`.
        #[arg(long, default_value = "admin_drain")]
        reason: String,
    },
    /// Force-assign a layer range to an available server.
    Assign {
        #[arg(long, default_value = "http://localhost:50052")]
        router: String,
        #[arg(long)]
        model: String,
        /// Inclusive layer range, e.g. `0-14`.
        #[arg(long, value_name = "START-END")]
        layers: String,
        /// Optional named available server; otherwise any spare is used.
        #[arg(long)]
        server: Option<String>,
        /// Optional external origin URL (S3, etc.); otherwise resolved
        /// from the live coverage matrix.
        #[arg(long)]
        origin_url: Option<String>,
        /// Hash to pin against when `--origin-url` is set.
        #[arg(long, default_value = "")]
        origin_hash: String,
    },
}

#[derive(Parser)]
struct Cli {
    /// Static shard map: comma-separated "START-END=URL" entries (inclusive bounds).
    /// Example: "0-16=http://host-a:8080,17-33=http://host-b:8081"
    /// Optional when --grid-port is provided.
    #[arg(long)]
    shards: Option<String>,

    /// Enable the self-assembling grid gRPC server on this port.
    /// Servers connect here with --join grpc://router:PORT.
    #[arg(long)]
    grid_port: Option<u16>,

    /// HTTP listen port.
    #[arg(long, default_value = "9090")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Per-request timeout to backend shards, in seconds.
    #[arg(long, default_value = "120")]
    timeout_secs: u64,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Shared secret for the self-assembling grid.
    /// Servers must pass the same key via --grid-key to be accepted.
    /// If not set, the grid port is open to any server (development only).
    #[arg(long, env = "LARQL_GRID_KEY")]
    grid_key: Option<String>,

    /// GT6: seconds between rebalancer checks (default: 30).
    /// Set to 0 to disable dynamic rebalancing.
    #[arg(long, default_value = "30")]
    rebalance_interval: u64,

    /// GT6: latency ratio threshold to trigger rebalancing (default: 2.0).
    /// The slowest replica must be this many times slower than the fastest
    /// for the same layer before the rebalancer acts.
    #[arg(long, default_value = "2.0")]
    rebalance_threshold: f32,

    /// Hot-shard replication: per-replica request rate (req/s) above
    /// which a shard is treated as effectively under-replicated. When
    /// a shard's max req_per_sec across replicas exceeds this value,
    /// the rebalancer pulls one extra spare from the available pool;
    /// when the rate subsides the extra replica is dropped on the next
    /// over-replication tick. Unset (default) disables the check.
    #[arg(long)]
    hot_shard_rps: Option<f32>,

    /// ADR-0020: per-replica in-flight saturation ceiling. When set,
    /// `route()` filters out replicas whose `requests_in_flight` is
    /// at or above this value before picking. If every owning
    /// replica is saturated, the router 503s with `Retry-After: 0.5`
    /// instead of piling more load onto a stuck shard.
    /// Default disabled (matches pre-ADR-0020 behavior). Pick a
    /// value with `2 × target_replicas × rps_per_replica × p99_s`
    /// as a starting point.
    #[arg(long)]
    saturation_ceiling: Option<u32>,

    /// ADR-0021: hedged-dispatch delay in milliseconds. When set, the
    /// multi-shard fan-out picks a secondary replica per sub-request
    /// and dispatches it after `--hedge-after-ms` if the primary
    /// hasn't responded. Halves p99 tail latency in topologies with
    /// `--target-replicas ≥ 2` at the cost of ~2× shard load on the
    /// tail. Default disabled (matches pre-ADR-0021 behavior); only
    /// fires on multi-shard requests, never on single-shard
    /// `proxy_raw`. Operator signals: `larql_router_route_hedge_fires_total`
    /// (how often the hedge fires) and `larql_router_route_hedge_wins_total`
    /// (how often it actually beats the primary).
    #[arg(long)]
    hedge_after_ms: Option<u64>,

    /// ADR-0014 hysteresis: ratio of the elevation threshold below
    /// which a hot shard demotes. Default `0.8` means elevate at
    /// `--hot-shard-rps T`, demote only when the rate falls below
    /// `0.8 × T`. Prevents oscillation at the boundary. Values
    /// outside `(0.0, 1.0]` clamp to the default. `1.0` disables
    /// hysteresis (elevate and demote at the same threshold).
    #[arg(long, default_value = "0.8")]
    hot_shard_demote_ratio: f32,

    /// Active-probe RTT: cadence (in seconds) at which the router
    /// issues `GET {listen_url}/v1/health` against every serving
    /// server. The measured round-trip lands on `ServerEntry.rtt_ms`
    /// and is used by `route()` as a tie-breaker after GT3 per-layer
    /// latency. `0` (default) disables probing — the feature is
    /// opt-in because GT3 already subsumes RTT once heartbeats carry
    /// `layer_stats`, so this mainly helps cold-start and
    /// cross-region tie-breaking.
    #[arg(long, default_value = "0")]
    rtt_probe_interval_secs: u64,

    /// ADR-0019: enable HTTP/3 shard transport. When set, MoE
    /// expert fan-out (ADR-0018) dispatches through an h3 client
    /// per shard host — each parallel sub-request runs as an
    /// independent QUIC stream, escaping TCP HoL blocking on the
    /// HTTP/2-over-TCP path. Requires building with
    /// `--features http3` AND each `larql-server` shard listening
    /// on `--http3-port` (server-side ADR-0019 wiring).
    ///
    /// Dense routing keeps the existing reqwest HTTP/2 path —
    /// HTTP/3 only swings the needle for parallel per-token
    /// fan-outs, which is the MoE workload.
    #[arg(long, default_value = "false")]
    #[cfg(feature = "http3")]
    http3_shards: bool,

    /// ADR-0019: SHA-256 fingerprint (hex) of each shard's
    /// HTTP/3 TLS cert. Required when `--http3-shards` is set
    /// unless the operator wants `AcceptAny` (LAN/dev).
    /// Pin one fingerprint per process — all shards in the grid
    /// must present the same cert.
    #[arg(long)]
    #[cfg(feature = "http3")]
    shard_cert_fingerprint: Option<String>,

    /// Phase 4: number of replicas to maintain per shard range.
    /// 1 = no replication (default). >1 enables auto-replication: when fewer
    /// than N servers cover a range, the router pulls from the available
    /// pool to bring the count back up; when more than N cover it, the
    /// rebalancer drops the least-loaded one.
    #[arg(long, default_value = "1")]
    target_replicas: u32,

    /// ADR-0010: enable the QUIC grid listener on this port. Requires
    /// building with `--features quic`. When set, servers can join via
    /// `quic://router:PORT`. Coexists with the TCP `--grid-port` listener;
    /// neither replaces the other.
    #[arg(long)]
    #[cfg(feature = "quic")]
    quic_port: Option<u16>,

    /// ADR-0010: TLS certificate PEM for the QUIC listener. If omitted,
    /// the router generates a self-signed cert at startup and prints its
    /// SHA-256 fingerprint (which clients pin via
    /// `--quic-cert-fingerprint`).
    #[arg(long)]
    #[cfg(feature = "quic")]
    quic_cert: Option<std::path::PathBuf>,

    /// ADR-0010: TLS private key PEM matching `--quic-cert`.
    #[arg(long)]
    #[cfg(feature = "quic")]
    quic_key: Option<std::path::PathBuf>,

    /// ADR-0010: Server name (TLS SNI) embedded in the auto-generated
    /// self-signed cert. Clients must connect with this name. Default
    /// `"router"`.
    #[arg(long, default_value = "router")]
    #[cfg(feature = "quic")]
    quic_server_name: String,
}

// ── Static shard map ───────────────────────────────────────────────────────────
//
// `Shard`, `parse_shards`, `peek_binary`, `find_shard_for_layer` moved to
// `larql_router::shards` so they can be unit-tested independently of the
// HTTP/gRPC dispatch in this file.

// `AppState`, the HTTP handlers, and the `build_router` factory live in
// `larql_router::http` so they can be exercised by integration tests
// without spawning the binary.
use larql_router::http::{build_router, AppState};

// ── Main ───────────────────────────────────────────────────────────────────────

// ── Admin subcommand dispatch (ADR-0004 Phase 5) ───────────────────────────────

async fn run_admin(cmd: AdminCmd) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use larql_router::admin::{admin_assign, admin_drain, admin_gaps, admin_status};

    match cmd {
        AdminCmd::Status { router } => {
            for line in admin_status(&router).await? {
                println!("{line}");
            }
        }
        AdminCmd::Gaps { router, model } => {
            for line in admin_gaps(&router, model.as_deref()).await? {
                println!("{line}");
            }
        }
        AdminCmd::Drain {
            router,
            server,
            reason,
        } => {
            let ack = admin_drain(&router, &server, &reason).await?;
            if ack.ok {
                println!("ok: drained {server}");
            } else {
                eprintln!("error: {}", ack.message);
                std::process::exit(2);
            }
        }
        AdminCmd::Assign {
            router,
            model,
            layers,
            server,
            origin_url,
            origin_hash,
        } => {
            let ack = admin_assign(
                &router,
                &model,
                &layers,
                server.as_deref(),
                origin_url.as_deref(),
                &origin_hash,
            )
            .await?;
            if ack.ok {
                println!("ok: assigned {model}");
            } else {
                eprintln!("error: {}", ack.message);
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let filtered = larql_router::cli_helpers::filter_legacy_route_arg(args);
    let root = CliRoot::parse_from(filtered);
    if let Some(admin) = root.admin {
        return run_admin(admin).await;
    }
    let cli = root.daemon;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-router v{}", env!("CARGO_PKG_VERSION"));

    if let Err(msg) =
        larql_router::cli_helpers::validate_daemon_inputs(cli.shards.as_deref(), cli.grid_port)
    {
        eprintln!("error: {msg}");
        std::process::exit(1);
    }

    let client = larql_router::cli_helpers::build_shard_client(cli.timeout_secs)?;

    let static_shards = if let Some(spec) = &cli.shards {
        let shards = parse_shards(spec).map_err(|e| format!("--shards: {e}"))?;
        info!("Static shard map:");
        for shard in &shards {
            let status_url = format!("{}/v1/stats", shard.url);
            let healthy = client
                .get(&status_url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            let marker = if healthy { "✓" } else { "✗ UNREACHABLE" };
            info!(
                "  layers {}-{}: {}  {}",
                shard.layer_start,
                shard.layer_end - 1,
                shard.url,
                marker
            );
            if !healthy {
                warn!("  Shard {} is not reachable", shard.url);
            }
        }
        shards
    } else {
        Vec::new()
    };

    let grid_state: Option<Arc<RwLock<GridState>>> = if cli.grid_port.is_some() {
        Some(Arc::new(RwLock::new(GridState::default())))
    } else {
        None
    };

    // ADR-0017: build the metrics registry early so the rebalancer + RTT
    // probe + HTTP route handlers all share one Arc<RouterMetrics>.
    let metrics = larql_router::metrics::RouterMetrics::new();

    if let (Some(grid_port), Some(state)) = (cli.grid_port, &grid_state) {
        // Phase 4: install target_replicas before any servers register so
        // the first under-/over-replication check sees the right target.
        state.write().set_target_replicas(cli.target_replicas);
        if cli.target_replicas > 1 {
            info!(
                target_replicas = cli.target_replicas,
                "Replication: enabled"
            );
        }
        // ADR-0020 — install the saturation ceiling. None = disabled.
        state.write().set_saturation_ceiling(cli.saturation_ceiling);
        if let Some(ceiling) = cli.saturation_ceiling {
            info!(
                ceiling,
                "Saturation backpressure: enabled (route() 503s when every replica >= ceiling)"
            );
        }

        // ADR-0021 — hedged-dispatch info log; the AppState already
        // carries the delay through to handle_walk_ffn_inner.
        if let Some(ms) = cli.hedge_after_ms {
            info!(
                hedge_after_ms = ms,
                "Hedged dispatch: enabled (fires a secondary replica when primary > {ms} ms)"
            );
        }

        let svc = GridServiceServer::new(
            GridServiceImpl::new_with_key(state.clone(), cli.grid_key.clone())
                .with_metrics(metrics.clone()),
        );
        let grpc_addr: SocketAddr = format!("{}:{}", cli.host, grid_port).parse()?;
        info!("Grid gRPC server listening: {grpc_addr}");
        tokio::spawn(async move {
            if let Err(e) = GrpcServer::builder()
                .add_service(svc)
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {e}");
            }
        });

        // ADR-0010: spawn a QUIC accept loop in parallel when --quic-port
        // is set. Same gRPC service implementation, different transport.
        #[cfg(feature = "quic")]
        if let Some(quic_port) = cli.quic_port {
            spawn_quic_listener(&cli, state.clone(), quic_port, metrics.clone())?;
        }

        // GT6: spawn dynamic rebalancer (disabled when interval == 0).
        if cli.rebalance_interval > 0 {
            let rebalance_cfg = rebalancer::RebalancerConfig::from_cli(
                cli.rebalance_interval,
                cli.rebalance_threshold,
            )
            .with_hot_shard_threshold(cli.hot_shard_rps)
            .with_hot_shard_demote_ratio(cli.hot_shard_demote_ratio);
            info!(
                interval_s = cli.rebalance_interval,
                threshold = cli.rebalance_threshold,
                hot_shard_rps = ?cli.hot_shard_rps,
                "Rebalancer: enabled"
            );
            rebalancer::spawn(state.clone(), rebalance_cfg, Some(metrics.clone()));
        }

        // Optional RTT probe loop — opt-in via --rtt-probe-interval-secs.
        if let Some(rtt_cfg) =
            larql_router::tasks::rtt_probe::RttProbeConfig::from_cli(cli.rtt_probe_interval_secs)
        {
            info!(
                interval_s = cli.rtt_probe_interval_secs,
                "RTT probe: enabled"
            );
            larql_router::tasks::rtt_probe::spawn(state.clone(), rtt_cfg, Some(metrics.clone()));
        }
    }

    // Snapshot grid gauges now so /metrics has values before the
    // first rebalancer tick fires. The rebalancer tick (if enabled)
    // keeps refreshing these every interval.
    if let Some(g) = grid_state.as_ref() {
        metrics.refresh_gauges(&g.read());
    }

    // ADR-0019 — build the H3Client when `--http3-shards` is on.
    // Held as an `Option`: dense routing keeps reqwest unchanged.
    #[cfg(feature = "http3")]
    let h3_client = if cli.http3_shards {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let h3 = larql_router_protocol::transport::h3::H3Client::new(
            "0.0.0.0:0".parse().unwrap(),
            cli.shard_cert_fingerprint.clone(),
        )
        .map_err(|e| format!("HTTP/3 client setup: {e}"))?;
        info!(
            fingerprint = ?cli.shard_cert_fingerprint,
            "HTTP/3 shard transport: enabled"
        );
        Some(Arc::new(h3))
    } else {
        None
    };

    let state = Arc::new(AppState {
        static_shards,
        grid: grid_state,
        client,
        metrics: Some(metrics.clone()),
        #[cfg(feature = "http3")]
        h3_client,
        hedge_after: cli.hedge_after_ms.map(std::time::Duration::from_millis),
    });

    let app = build_router(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("HTTP listening: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
