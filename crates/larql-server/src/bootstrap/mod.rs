//! Server bootstrap and vindex loading helpers.
//!
//! Module layout:
//!
//! ```text
//! bootstrap/
//! ├── mod.rs       — `serve()` orchestration + re-exports
//! ├── cli.rs       — clap definition, arg defaults, argv parsers
//! ├── load.rs      — vindex/model loading (V2/V3 artifact fork)
//! ├── listeners.rs — optional HTTP/3 listener (ADR-0019)
//! └── tests/       — unit tests (module tests folder)
//! ```

mod cli;
mod listeners;
mod load;

#[cfg(test)]
mod tests;

pub use cli::{
    normalize_serve_alias, parse_layer_range, parse_ram_bytes, Cli,
    DEFAULT_DESCRIBE_CACHE_TTL_SECS, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HOST, DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_GATE_CACHE_LAYERS, DEFAULT_MAX_Q4K_CACHE_LAYERS,
    DEFAULT_PORT, DEFAULT_SESSION_TTL_SECS,
};
pub use load::{
    discover_vindexes, load_artifact, load_single_vindex, parse_unit_manifest, LoadVindexOptions,
    LoadedArtifact, UnitManifest,
};

use std::sync::Arc;

use axum::middleware;
use tracing::{info, warn};

use crate::cache::DescribeCache;
use crate::session::SessionManager;
use crate::state::{AppState, LoadedModel};
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

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();
    let mut v3_models: Vec<Arc<crate::vindex3::V3Model>> = Vec::new();

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
            match load_artifact(&p.to_string_lossy(), load_opts.clone()) {
                Ok(LoadedArtifact::V2(m)) => models.push(Arc::new(*m)),
                Ok(LoadedArtifact::V3(m)) => v3_models.push(Arc::new(*m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        match load_artifact(vindex_path, load_opts)? {
            LoadedArtifact::V2(m) => models.push(Arc::new(*m)),
            LoadedArtifact::V3(m) => v3_models.push(Arc::new(*m)),
        }
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() && v3_models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    // Cgroup memory pre-flight (BUG-infer-deadlock §5.5).  Refuses to
    // start when the configured cgroup leaves no room to load weights;
    // converts a 10-second OOM-kill loop into a one-line startup error.
    if !cli.no_memcheck && !cli.lazy_weights {
        let total_estimate: u64 = models
            .iter()
            .filter(|m| !m.infer_disabled)
            .map(|m| m.config.estimate_resident_bytes())
            .sum();
        if total_estimate > 0 {
            let headroom = cli.memcheck_headroom_mib * 1024 * 1024;
            let outcome = crate::memcheck::check_memory_headroom(total_estimate, headroom);
            match &outcome {
                crate::memcheck::MemCheckOutcome::Ok {
                    cgroup_max_bytes,
                    estimate_bytes,
                } => {
                    info!(
                        "Memcheck: estimated {:.1} GB resident vs cgroup memory.max {:.1} GB \
                         (headroom {} MiB, ok)",
                        (*estimate_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
                        (*cgroup_max_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
                        cli.memcheck_headroom_mib,
                    );
                }
                crate::memcheck::MemCheckOutcome::Skipped { reason } => {
                    info!("Memcheck: skipped ({reason})");
                }
                crate::memcheck::MemCheckOutcome::Tight { .. } => {
                    return Err(crate::memcheck::explain_tight_outcome(&outcome).into());
                }
            }
        }
    } else if cli.no_memcheck {
        info!("Memcheck: disabled (--no-memcheck)");
    }

    // Eager-load model weights at startup so the first /v1/infer
    // request does not face a multi-GB allocation under HTTP-handler
    // backpressure.  Failure here is a clean startup error rather
    // than an OOM-kill during the first request.  See
    // `BUG-infer-deadlock.md` and `LoadedModel::force_load_weights`.
    if cli.lazy_weights {
        info!("Lazy weight load: enabled (--lazy-weights)");
    } else {
        for m in &models {
            if m.infer_disabled {
                continue;
            }
            let load_start = std::time::Instant::now();
            info!("Pre-loading model weights for '{}' …", m.id);
            if let Err(e) = m.force_load_weights() {
                return Err(format!(
                    "failed to load weights for '{}': {} \
                     (pass --lazy-weights to defer until first request)",
                    m.id, e
                )
                .into());
            }
            info!(
                "  Pre-loaded weights for '{}' in {:.1}s",
                m.id,
                load_start.elapsed().as_secs_f64(),
            );
        }
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

    let state = Arc::new(AppState {
        model_set: std::sync::RwLock::new(crate::state::ModelSet {
            models: models.clone(),
            v3_models: v3_models.clone(),
        }),
        router_topology,
        lifecycle: std::sync::Mutex::new(initial_lifecycle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(cli.session_ttl_secs),
        describe_cache: DescribeCache::new(cli.cache_ttl),
        infer_timeout: std::time::Duration::from_secs(cli.infer_timeout_secs),
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
            let host = if cli.host == DEFAULT_HOST {
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
                    serves_openai,
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
        axum::serve(listener, app).await?;
    }

    Ok(())
}
