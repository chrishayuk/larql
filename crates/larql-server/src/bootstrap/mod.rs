//! Server bootstrap and vindex loading helpers.
//!
//! Module layout:
//!
//! ```text
//! bootstrap/
//! ├── mod.rs       — `serve()` orchestration + re-exports
//! ├── cli.rs       — clap definition, arg defaults, argv parsers
//! ├── load.rs      — vindex/model loading (V2/V3 artifact fork)
//! ├── models.rs    — boot phase 1: load models, memcheck, pre-load weights
//! ├── listeners.rs — optional HTTP/3 listener (ADR-0019)
//! └── tests/       — unit tests (module tests folder)
//! ```

mod cli;
mod listeners;
mod load;
mod models;

#[cfg(test)]
mod tests;

pub use cli::{
    normalize_serve_alias, parse_layer_range, parse_ram_bytes, Cli,
    DEFAULT_DESCRIBE_CACHE_TTL_SECS, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HOST, DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_GATE_CACHE_LAYERS, DEFAULT_MAX_Q4K_CACHE_LAYERS,
    DEFAULT_PORT, DEFAULT_SESSION_TTL_SECS,
};
pub use load::{
    classify_source, discover_vindexes, load_artifact, load_single_vindex, parse_unit_manifest,
    LoadVindexOptions, LoadedArtifact, SourceKind, UnitManifest,
};

use std::sync::Arc;

use axum::middleware;
use tracing::{info, warn};

use crate::cache::DescribeCache;
use crate::session::SessionManager;
use crate::state::AppState;
use crate::{announce, auth, grpc, grpc_expert, ratelimit, routes};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ── Server lifecycle ──────────────────────────────────────────────────────────

/// Boot the server: load every vindex named on the command line, build the
/// router, run any opt-in warmups, then bind the TCP listener (plus optional
/// UDS / TLS / gRPC sockets) and run forever.
///
/// `main` is a thin wrapper: parse `Cli`, init tracing, hand off here. Splitting
/// the orchestration out lets integration tests drive boot without going
/// through `clap::Parser::parse_from`.
pub async fn serve(cli: Cli) -> Result<(), BoxError> {
    cli.check_network_exposure()?;
    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));
    // No DEC number should ever be recorded on an unlogged scalar
    // fallback — see docs/audits/dec-readiness-review-2026-07-22.md §1b.
    info!(
        "  Q4K/Q6K×Q8K kernel class: {}",
        larql_compute::cpu::ops::q4k_q8k_dot::kernel_class_summary()
    );
    // Same discipline for flag state: every env toggle that changes a
    // number is logged before any request is served.
    info!(
        "  decode options: {}",
        larql_compute::options::decode_options_summary()
    );

    let layer_range = cli.layer_range()?;
    let (models, v3_models) = models::load_models(&cli)?;
    models::memory_preflight(&cli, &models)?;
    models::preload_weights(&cli, &models)?;

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

    // Frozen once, from the exact same boot-time count that decides
    // which axum `Router` gets built below — computing both from one
    // call means the topology `AppState` freezes and the router that
    // actually exists can never disagree with each other.
    let router_topology =
        crate::state::RouterTopology::for_boot_count(models.len() + v3_models.len());
    // The lifecycle flag's initial value: `Ready` only when boot
    // loaded exactly the one model a single-model topology can ever
    // hold; `Idle` for zero models *and* for a multi-model boot (2+).
    // A multi-model boot's value is never actually consulted —
    // `validate_lifecycle_mutation` refuses every mutation before
    // anything reads `lifecycle` — but `Idle` is still the honest
    // placeholder: `Ready` names exactly one binding, and a
    // multi-model boot doesn't have one to name.
    let initial_lifecycle = match (
        models.first(),
        v3_models.first(),
        models.len() + v3_models.len(),
    ) {
        (Some(m), _, 1) => crate::state::LifecycleState::Ready {
            model_id: m.id.clone(),
            path: m.path.clone(),
        },
        (_, Some(m), 1) => crate::state::LifecycleState::Ready {
            model_id: m.id.clone(),
            path: m.path.clone(),
        },
        _ => crate::state::LifecycleState::Idle,
    };

    let mut model_set = crate::state::ModelSet {
        models: models.clone(),
        v3_models: v3_models.clone(),
        aliases: Default::default(),
    };
    if let Some(alias) = cli.model_alias.as_deref() {
        model_set.alias_sole_model(alias)?;
        info!("  Model alias: '{alias}'");
    }

    let state = Arc::new(AppState {
        model_set: std::sync::RwLock::new(model_set),
        router_topology,
        lifecycle: std::sync::Mutex::new(initial_lifecycle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(cli.session_ttl_secs),
        describe_cache: DescribeCache::new(cli.cache_ttl),
        infer_timeout: std::time::Duration::from_secs(cli.infer_timeout_secs),
        patch_sources: crate::routes::patches::PatchSources {
            dir: cli.patch_dir.clone(),
            allow_hf: cli.allow_hf_patches,
        },
        responses: crate::response_store::ResponseStore::new(),
        v3_kv: crate::response_kv::ResponseKvCache::new(
            cli.v3_kv_cache_entries,
            cli.v3_kv_ttl_secs,
        ),
        runtime: Arc::new(crate::runtime_stats::RuntimeRecorder::new()),
    });

    if cli.infer_timeout_secs == 0 {
        info!("Infer timeout: disabled");
    } else {
        info!("Infer timeout: {}s", cli.infer_timeout_secs);
    }

    if cli.cache_ttl > 0 {
        info!("DESCRIBE cache: {}s TTL", cli.cache_ttl);
    }

    // Background maintenance: evict idle sessions and stale rate-limit
    // buckets so the per-client maps stay bounded (ROADMAP §Open defects P1).
    {
        let sessions_state = Arc::clone(&state);
        let mut sweep_targets = vec![crate::maintenance::SweepTarget::new(
            "sessions",
            move || {
                let state = Arc::clone(&sessions_state);
                async move { state.sessions.evict_expired().await }
            },
        )];
        let kv_state = Arc::clone(&state);
        sweep_targets.push(crate::maintenance::SweepTarget::new("v3-kv", move || {
            let state = Arc::clone(&kv_state);
            async move { state.v3_kv.evict_expired() }
        }));
        if let Some(ref rl) = rate_limiter {
            let limiter = Arc::clone(rl);
            sweep_targets.push(crate::maintenance::SweepTarget::new(
                "ratelimit-buckets",
                move || {
                    let limiter = Arc::clone(&limiter);
                    async move { limiter.evict_stale() }
                },
            ));
        }
        crate::maintenance::spawn(
            std::time::Duration::from_secs(crate::maintenance::DEFAULT_SWEEP_INTERVAL_SECS),
            sweep_targets,
        );
        info!(
            "Maintenance sweeper: every {}s (session TTL {}s)",
            crate::maintenance::DEFAULT_SWEEP_INTERVAL_SECS,
            state.sessions.ttl().as_secs(),
        );
    }

    // One snapshot for every boot-time loop below — the model set
    // never changes after this point in this rung, but going through
    // the same accessor every real caller uses keeps boot from being
    // a second, divergent way to read `AppState`'s model list.
    let boot_models = state.models_snapshot().models;

    // The router-shape decision reads the frozen fact, not a live
    // recount — `state.is_multi_model()` would (once mutation exists)
    // answer a different question than "which router did we build".
    let is_multi = state.router_topology == crate::state::RouterTopology::MultiModel;
    let mut app = if cli.public_explorer {
        // The public surface serves exactly one VINDEX3 container: the
        // LQL bridge binds it as its whole universe, and a multi-model
        // catalogue would make /v1/query and the REST routes disagree
        // about what "the model" is.
        if is_multi || !models.is_empty() || v3_models.len() != 1 {
            return Err(
                "--public-explorer requires exactly one VINDEX3 container (no V2, no multi-model)"
                    .into(),
            );
        }
        let bridge = Arc::new(crate::lql_bridge::spawn(
            &v3_models[0].path,
            std::time::Duration::from_secs(cli.infer_timeout_secs),
        )?);
        info!(
            "PUBLIC_EXPLORER mode: read routes + /v1/query only ({})",
            v3_models[0].id
        );
        routes::public_explorer_router(Arc::clone(&state), bridge)
    } else if is_multi {
        info!("Multi-model mode ({} models)", boot_models.len());
        for m in &boot_models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(Arc::clone(&state))
    } else {
        match models.first() {
            Some(m) => info!("Single-model mode: {}", m.config.model),
            None => info!("Single-model mode: {} (VINDEX3)", v3_models[0].id),
        }
        routes::single_model_router(Arc::clone(&state))
    };

    // `--warmup-walk-ffn` — pre-load inference weights + prefetch every
    // owned layer's Q4K mmap so the first `/v1/walk-ffn` doesn't pay
    // the ~1.3 s lazy weight load + ~17 ms / cold layer (see
    // ROADMAP G1 / G2). Same code path as `POST /v1/warmup`.
    if cli.warmup_walk_ffn {
        for m in &boot_models {
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
    for m in &boot_models {
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
    for m in &boot_models {
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
        let client_ip = match (&cli.client_ip_header, cli.trust_forwarded_for) {
            (Some(header), _) => ratelimit::ClientIpSource::Header(header.clone()),
            (None, true) => ratelimit::ClientIpSource::ForwardedFor,
            (None, false) => ratelimit::ClientIpSource::Socket,
        };
        match &client_ip {
            ratelimit::ClientIpSource::Header(header) => {
                info!("Rate limit: client address from the {header} header")
            }
            ratelimit::ClientIpSource::ForwardedFor => {
                info!("Rate limit: trusting X-Forwarded-For")
            }
            ratelimit::ClientIpSource::Socket => {}
        }
        let rate_state = Arc::new(ratelimit::RateLimitState {
            limiter: Arc::clone(rl),
            client_ip,
        });
        app = app.layer(middleware::from_fn_with_state(
            rate_state,
            ratelimit::rate_limit_middleware,
        ));
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
        let grpc_api_key = cli.api_key.clone();
        let grpc_max_concurrent = cli.max_concurrent;
        // Exp 53 ShardService. Vindex-backed: the cache shares the
        // server's loaded `PatchedVindex`, so "compiled facts" live as
        // vindex patches (via `PatchedVindex::add_patch` etc.) and we
        // don't maintain a separate on-disk cache format. Opt-in via
        // `--shard-query-tau`; deployments that don't set it pay zero
        // for the feature.
        let shard_source = cli.shard_query_tau.and_then(|tau| {
            let model = boot_models.first()?;
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
            // Same key and concurrency bound as the HTTP listener.
            let auth = auth::grpc_interceptor(grpc_api_key);
            let mut builder = tonic::transport::Server::builder()
                .concurrency_limit_per_connection(grpc_max_concurrent)
                .add_service(
                    grpc::proto::vindex_service_server::VindexServiceServer::with_interceptor(
                        vindex_svc,
                        auth.clone(),
                    ),
                )
                .add_service(
                    larql_router_protocol::ExpertServiceServer::with_interceptor(
                        expert_svc,
                        auth.clone(),
                    ),
                );
            if let Some(source) = shard_source {
                let shard_svc = crate::shard_query::ShardGrpcService::new(source);
                builder = builder.add_service(
                    larql_router_protocol::ShardServiceServer::with_interceptor(shard_svc, auth),
                );
            }
            if let Err(e) = builder.serve(grpc_addr).await {
                tracing::error!("gRPC server error: {}", e);
            }
        });
    }

    let addr = format!("{}:{}", cli.host, cli.port);

    // Grid announce (if --join provided).
    if let Some(join_spec) = cli.join.clone() {
        // The announce loop below iterates `models` (V2) only, and the
        // ShardService registration above reads `boot_models.first()` —
        // a server whose only artifact is a VINDEX3 container would
        // otherwise join silently with nothing to announce.
        if models.is_empty() && !v3_models.is_empty() {
            warn!(
                "--join configured, but only VINDEX3 containers are loaded; \
                 V3 containers do not join the grid yet — no shards will be announced"
            );
        }
        let listen_url = cli.public_url.clone().unwrap_or_else(|| {
            // An unspecified bind (0.0.0.0 / ::) is not an address peers
            // can dial; announce loopback for it.
            let unspecified = cli
                .host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_unspecified());
            let host = if unspecified { "127.0.0.1" } else { &cli.host };
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
        let unverified_shards =
            crate::shard_loader::UnverifiedShards::from_flag(cli.allow_unverified_shards);
        if cli.allow_unverified_shards {
            warn!("--allow-unverified-shards: Mode B will load shards that carry no content hash");
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
                            unverified_shards,
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
            unverified_shards,
        );

        for m in &models {
            let (layer_start, layer_end) = match layer_range {
                Some((s, e)) => (s as u32, (e - 1) as u32),
                None => (0, (m.config.num_layers.saturating_sub(1)) as u32),
            };
            let vhash = announce::vindex_identity_hash(&m.id, m.config.num_layers);
            // N0-router: this server can serve complete OpenAI requests
            // only when it holds the whole model with inference enabled —
            // any layer/expert/unit slice or inference-disabling mode
            // makes it a compute shard, not an OpenAI backend.
            let serves_openai = layer_range.is_none()
                && m.expert_filter.is_none()
                && m.unit_filter.is_none()
                && !m.infer_disabled
                && !m.ffn_only
                && !m.embed_only;
            let mut configs = Vec::with_capacity(join_urls.len());
            for join_url in &join_urls {
                let avail = available_after_drain.as_ref().map(|base| {
                    let mut a = base.clone();
                    a.join_url = join_url.clone();
                    a
                });
                configs.push(announce::AnnounceConfig {
                    join_url: join_url.clone(),
                    model_id: m.id.clone(),
                    layer_start,
                    layer_end,
                    listen_url: listen_url.clone(),
                    ram_bytes: 0,
                    grid_key: cli.grid_key.clone(),
                    vindex_hash: vhash.clone(),
                    // Filled by `run_announce_after_content_hash` below.
                    shard_sha256: String::new(),
                    serves_openai,
                    latency_tracker: m.layer_latency_tracker.clone(),
                    requests_in_flight: m.requests_in_flight.clone(),
                    requests_total: m.requests_total.clone(),
                    available_after_drain: avail,
                    quic_cert_fingerprint: cli.quic_cert_fingerprint.clone(),
                });
            }
            // The content hash covers exactly what `/v1/shard` streams for
            // this model: the whole vindex directory.
            announce::run_announce_after_content_hash(configs, m.path.clone());
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
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
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
                        // A Unix socket has no peer address and only local
                        // peers; mark its requests so the rate limiter does
                        // not put them all in the unknown-client bucket.
                        let uds_app = app
                            .clone()
                            .layer(axum::Extension(ratelimit::LocalTransport));
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
        listeners::spawn_http3_listener_if_configured(&cli, app.clone()).await?;

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
        // Served with the peer address attached: the rate limiter's default
        // bucket is the socket peer, and without this there is none.
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await?;
    }

    Ok(())
}
