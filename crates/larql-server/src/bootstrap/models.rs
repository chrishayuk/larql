//! Boot phase 1: load every vindex named on the command line, check the
//! memory budget, and pre-load weights, so a request never pays for a
//! multi-GB allocation.

use std::sync::Arc;

use tracing::{info, warn};

use super::{
    discover_vindexes, load_artifact, parse_layer_range, parse_unit_manifest, BoxError, Cli,
    LoadVindexOptions, LoadedArtifact,
};
use crate::state::LoadedModel;

/// The boot-time model set: V2 vindexes and VINDEX3 containers.
pub(super) type BootModels = (Vec<Arc<LoadedModel>>, Vec<Arc<crate::vindex3::V3Model>>);

/// Load every artifact `cli` names, with its layer/expert/unit filters and
/// remote-MoE backend. Refuses an empty result.
pub(super) fn load_models(cli: &Cli) -> Result<BootModels, BoxError> {
    let mut models: Vec<Arc<LoadedModel>> = Vec::new();
    let mut v3_models: Vec<Arc<crate::vindex3::V3Model>> = Vec::new();

    let layer_range = cli.layer_range()?;
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
    let moe_remote = moe_remote_backend(cli)?;

    let load_opts = LoadVindexOptions {
        v3_backend: cli.v3_backend,
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
    Ok((models, v3_models))
}

/// The server-side remote MoE backend (`--moe-shards` or
/// `--moe-units-manifest`), if one is configured.
fn moe_remote_backend(
    cli: &Cli,
) -> Result<Option<Arc<larql_inference::ffn::RemoteMoeBackend>>, BoxError> {
    if cli.moe_shards.is_some() && cli.moe_units_manifest.is_some() {
        return Err("--moe-shards and --moe-units-manifest are mutually exclusive".into());
    }
    Ok(if let Some(ref s) = cli.moe_shards {
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
    })
}

/// Refuse to start when the cgroup leaves no room for the weights.
pub(super) fn memory_preflight(cli: &Cli, models: &[Arc<LoadedModel>]) -> Result<(), BoxError> {
    // Cgroup memory pre-flight (BUG-infer-deadlock §5.5).  Refuses to
    // start when the configured cgroup leaves no room to load weights;
    // converts a 10-second OOM-kill loop into a one-line startup error.
    if !cli.no_memcheck && !cli.lazy_weights {
        let total_estimate: u64 = models
            .iter()
            // BitNet (--keep-quant) containers are included, sized by
            // the bytes their I2_S weights actually occupy rather than
            // the dense path's f32 sizes (see `memcheck::bitnet`).
            .filter(|m| !m.infer_disabled)
            .map(|m| crate::memcheck::estimate_model_resident_bytes(&m.config))
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
    Ok(())
}

/// Pre-load weights unless `--lazy-weights`.
pub(super) fn preload_weights(cli: &Cli, models: &[Arc<LoadedModel>]) -> Result<(), BoxError> {
    // Eager-load model weights at startup so the first /v1/infer
    // request does not face a multi-GB allocation under HTTP-handler
    // backpressure.  Failure here is a clean startup error rather
    // than an OOM-kill during the first request.  See
    // `BUG-infer-deadlock.md` and `LoadedModel::force_load_weights`.
    if cli.lazy_weights {
        info!("Lazy weight load: enabled (--lazy-weights)");
    } else {
        for m in models {
            if m.infer_disabled {
                continue;
            }
            let load_start = std::time::Instant::now();
            // BitNet vindex (--keep-quant) skips the dense load and
            // pre-loads the native ternary path instead.  Saves ~5 GB
            // of dense allocation per model on a 2 B BitNet.
            if m.is_bitnet() {
                info!("Pre-loading BitNet model for '{}' …", m.id);
                if let Err(e) = m.force_load_bitnet_model() {
                    return Err(format!(
                        "failed to load bitnet model for '{}': {} \
                         (pass --lazy-weights to defer until first request)",
                        m.id, e
                    )
                    .into());
                }
                info!(
                    "  Pre-loaded BitNet model for '{}' in {:.1}s",
                    m.id,
                    load_start.elapsed().as_secs_f64(),
                );
                continue;
            }
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
    Ok(())
}
