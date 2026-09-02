//! Vindex/model loading — the V2/V3 artifact fork, the single-vindex
//! loader, ownership manifests, and `--dir` discovery.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_vindex::format::filenames::*;
use larql_vindex::{
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer, PatchedVindex,
    SilentLoadCallbacks, VectorIndex,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::state::{load_probe_labels, model_id_from_name, LoadedModel};

use super::BoxError;

#[derive(Clone, Default)]
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

/// One bound model artifact — which runtime family serves it. The
/// V2/V3 decision is made HERE, at binding time, from the container's
/// own generation marker; nothing downstream re-detects it.
pub enum LoadedArtifact {
    V2(Box<LoadedModel>),
    V3(Box<crate::vindex3::V3Model>),
}

/// Detect the artifact's container generation and bind it with the
/// matching loader. A VINDEX3 container binds as an executable
/// program ([`crate::vindex3::load_v3_model`]) — it structurally
/// cannot take the V2 path, whose `load_vindex_config` refuses
/// non-V2 generations.
/// Options a VINDEX3 binding cannot honour, named for the refusal.
///
/// The V3 branch of [`load_artifact`] takes only a path — slicing,
/// service modes, and cache knobs have no V3 implementation. Accepting
/// the flag and ignoring it is the dangerous failure: a `--layers 0-9`
/// shard silently loads the *whole* model and answers complete
/// requests, and `--no-infer` does not disable inference. Fail closed
/// until V3 sharding exists (ROADMAP §N1 / V3 sharding).
fn unsupported_v3_options(opts: &LoadVindexOptions) -> Vec<&'static str> {
    let mut named = Vec::new();
    if opts.no_infer {
        named.push("--no-infer");
    }
    if opts.ffn_only {
        named.push("--ffn-only");
    }
    if opts.embed_only {
        named.push("--embed-only");
    }
    if opts.layer_range.is_some() {
        named.push("--layers");
    }
    if opts.expert_filter.is_some() {
        named.push("--experts");
    }
    if opts.unit_filter.is_some() {
        named.push("--units");
    }
    if opts.moe_remote.is_some() {
        named.push("--moe-remote");
    }
    named
}

/// The single choke point that decides V2 vs V3 for serving — every
/// `load_artifact` caller (this binary's own CLI arg, `--dir` bulk
/// discovery, and the `/v1/runtime/model` HTTP lifecycle endpoint) goes
/// through here, so a resolution improvement made once benefits all
/// three (`docs/vindex3-registry-design.md` §10, rung 2B).
pub fn load_artifact(path_str: &str, opts: LoadVindexOptions) -> Result<LoadedArtifact, BoxError> {
    let path = resolve_artifact_path(path_str)?;
    // The V2 loader re-derives its own path from the string it's given
    // (it has to run standalone from `larql vindex3 cmd` call sites too)
    // — passing it the ALREADY-resolved directory, not the original
    // `path_str`, means a claimed registry name or an `hf://` reference
    // is fetched exactly once, not re-resolved a second time under a
    // string `load_single_vindex`'s own (narrower) resolution has no way
    // to understand.
    let resolved_path_str = path.to_string_lossy().into_owned();
    match larql_vindex::format::generation::detect_generation(&path)? {
        larql_vindex::format::generation::ContainerGeneration::V3 => {
            let unsupported = unsupported_v3_options(&opts);
            if !unsupported.is_empty() {
                return Err(format!(
                    "VINDEX3 containers do not support {} — a V3 binding serves the whole \
                     model, so accepting these would silently ignore them (a `--layers` shard \
                     would load the full model and answer complete requests). Remove them, or \
                     serve a VINDEX2 container: {}",
                    if unsupported.len() == 1 {
                        "this option"
                    } else {
                        "these options"
                    },
                    unsupported.join(", "),
                )
                .into());
            }
            info!("Loading VINDEX3 container: {}", path.display());
            Ok(LoadedArtifact::V3(Box::new(crate::vindex3::load_v3_model(
                &path,
            )?)))
        }
        larql_vindex::format::generation::ContainerGeneration::V2 => Ok(LoadedArtifact::V2(
            Box::new(load_single_vindex(&resolved_path_str, opts)?),
        )),
    }
}

/// Resolve a `load_artifact` argument to a literal local directory.
///
/// A bare name (`qwen3.8`, optionally `:variant`) the VINDEX3 registry
/// has claimed resolves through it **exclusively** — any failure
/// (unknown variant, incompatible ABI) is a real refusal, never rescued
/// by falling through to the plain `hf://`/literal-path handling below.
/// The dispatch (`larql_vindex::registry::resolve_claimed`) is shared
/// with `larql-cli`'s `serve` trampoline (rung 2A) — the same claimed
/// name must mean the same thing whether reached via `larql serve`, this
/// binary invoked directly, or the `/v1/runtime/model` HTTP endpoint.
///
/// An `hf://` reference or an existing local path is untouched: neither
/// is a bare registry-shaped name (`resolve_claimed` returns `Ok(None)`
/// for both, structurally — see the reference grammar), so they fall
/// through to exactly today's behaviour.
fn resolve_artifact_path(path_str: &str) -> Result<PathBuf, BoxError> {
    resolve_artifact_path_with(path_str, &larql_vindex::registry::production_registry())
}

/// Testable core of [`resolve_artifact_path`]. Takes `registry`
/// explicitly so the claimed/unclaimed dispatch can be proven without
/// depending on the (currently empty) production registry.
fn resolve_artifact_path_with(
    path_str: &str,
    registry: &larql_vindex::registry::RegistryManifest,
) -> Result<PathBuf, BoxError> {
    match classify_with(path_str, registry)? {
        SourceResolution::Registry(resolved) => Ok(resolved),
        SourceResolution::Hf => Ok(larql_vindex::resolve_hf_vindex(path_str)?),
        SourceResolution::Local => Ok(PathBuf::from(path_str)),
    }
}

/// Which branch a reference takes, carrying whatever that branch
/// already resolved. Split out of [`resolve_artifact_path_with`] so
/// that `/v1/capabilities` can report the reference forms this server
/// accepts by *asking the resolver* ([`classify_source`]) rather than
/// restating its branches in a second list that can drift.
///
/// The registry arm carries its resolved path because
/// `resolve_claimed` has already done the work — classifying and then
/// re-resolving would run it twice on the load path.
enum SourceResolution {
    /// A registry-claimed name, resolved by the registry.
    Registry(PathBuf),
    /// An `hf://` reference to a *published container*. Not fetched
    /// yet: classification is free, resolution is a download.
    Hf,
    /// A literal filesystem path.
    Local,
}

fn classify_with(
    path_str: &str,
    registry: &larql_vindex::registry::RegistryManifest,
) -> Result<SourceResolution, BoxError> {
    if let Some(resolved) = larql_vindex::registry::resolve_claimed(path_str, registry)? {
        return Ok(SourceResolution::Registry(resolved));
    }
    if larql_vindex::is_hf_path(path_str) {
        Ok(SourceResolution::Hf)
    } else {
        Ok(SourceResolution::Local)
    }
}

/// How `load_artifact` will interpret `reference`.
///
/// Reported by `GET /v1/capabilities` as the source forms this
/// server's load verb accepts. Note what an `Hf` answer does and does
/// not promise: `resolve_hf_vindex` fetches an **already-encoded
/// VINDEX container** (it looks for `index.json`), so `hf://` here
/// means "a published container", never "a raw checkpoint". Encoding
/// a checkpoint is `vindex encode`, a capability this server does not
/// have — which is why the capability table keeps `load` and `encode`
/// as separate verbs over separate objects.
pub fn classify_source(reference: &str) -> SourceKind {
    match classify_with(reference, &larql_vindex::registry::production_registry()) {
        // A refusal from the registry is still the registry
        // answering: the name was claimed, and failed. Reporting it
        // as `Local` would say a claimed-but-broken name is a
        // filesystem path.
        Ok(SourceResolution::Registry(_)) | Err(_) => SourceKind::Registry,
        Ok(SourceResolution::Hf) => SourceKind::Hf,
        Ok(SourceResolution::Local) => SourceKind::Local,
    }
}

/// The reference forms [`classify_source`] distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// A name the VINDEX3 registry has claimed.
    Registry,
    /// An `hf://owner/repo` reference to a published container.
    Hf,
    /// A literal filesystem path.
    Local,
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
        weights_init: std::sync::Mutex::new(()),
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

#[cfg(test)]
mod v3_option_tests {
    use super::*;
    use larql_inference::prompt::ChatTemplate;

    #[test]
    fn default_options_are_all_supported() {
        assert!(unsupported_v3_options(&LoadVindexOptions::default()).is_empty());
    }

    #[test]
    fn cache_knobs_do_not_block_a_v3_binding() {
        // These are V2 tuning that a V3 binding simply has no use for —
        // refusing them would reject a perfectly serveable container.
        let opts = LoadVindexOptions {
            max_gate_cache_layers: 8,
            max_q4k_cache_layers: 8,
            hnsw: Some(64),
            warmup_hnsw: true,
            release_mmap_after_request: true,
            ..LoadVindexOptions::default()
        };
        assert!(unsupported_v3_options(&opts).is_empty());
    }

    #[test]
    fn every_slicing_and_mode_option_is_named() {
        let opts = LoadVindexOptions {
            no_infer: true,
            ffn_only: true,
            embed_only: true,
            layer_range: Some((0, 10)),
            expert_filter: Some((0, 31)),
            unit_filter: Some(std::sync::Arc::new(std::collections::HashSet::new())),
            ..LoadVindexOptions::default()
        };
        let named = unsupported_v3_options(&opts);
        for flag in [
            "--no-infer",
            "--ffn-only",
            "--embed-only",
            "--layers",
            "--experts",
            "--units",
        ] {
            assert!(named.contains(&flag), "{flag} must be named in the refusal");
        }
    }

    #[test]
    fn no_infer_alone_is_refused() {
        // The dangerous one: silently ignoring it served inference from a
        // server the operator had told not to.
        let opts = LoadVindexOptions {
            no_infer: true,
            ..LoadVindexOptions::default()
        };
        assert_eq!(unsupported_v3_options(&opts), vec!["--no-infer"]);
    }

    #[test]
    fn family_beats_the_id_heuristic() {
        // The container declares what it is; the id is just a folder name.
        assert!(matches!(
            crate::vindex3::resolve_chat_template("gemma3", "some-container"),
            ChatTemplate::Gemma
        ));
    }

    #[test]
    fn id_heuristic_still_applies_when_the_family_is_unknown() {
        assert!(matches!(
            crate::vindex3::resolve_chat_template("", "gemma-2-2b.vindex3"),
            ChatTemplate::Gemma
        ));
    }

    #[test]
    fn an_unmapped_family_and_id_falls_back_to_plain() {
        // Granite: declared by the container, but no renderer maps to it.
        // The fallback is legitimate — the bind path warns about it.
        assert!(matches!(
            crate::vindex3::resolve_chat_template("granite", "granite-4.1-3b.vindex3"),
            ChatTemplate::Plain
        ));
    }
}

/// Rung 2B of the vindex3-registry initiative's "resolver convergence"
/// step (`docs/vindex3-registry-design.md` §10): `load_artifact`'s
/// claimed/unclaimed dispatch, shared with `larql-cli`'s `serve`
/// trampoline via `larql_vindex::registry::resolve_claimed`.
#[cfg(test)]
mod resolve_artifact_path_tests {
    use std::collections::BTreeMap;

    use larql_vindex::registry::{
        Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel,
        RegistryVariant, Vindex3Abi, CURRENT_VINDEX3_ABI, REGISTRY_MANIFEST_SCHEMA_VERSION,
    };

    use super::resolve_artifact_path_with;

    fn registry_claiming_qwen38(abi: Vindex3Abi) -> RegistryManifest {
        let mut variants = BTreeMap::new();
        variants.insert(
            "27b-nvfp4".to_string(),
            RegistryVariant {
                artifact: RegistryArtifactRef {
                    repo: "larql/qwen3.8-27b-nvfp4".to_string(),
                    revision: "abc123f0".to_string(),
                },
                abi,
                source: Provenance {
                    repo: "Qwen/Qwen3.8-27B".to_string(),
                    revision: "8c4fdeadbeef".to_string(),
                    attestation: Attestation::Mechanical,
                },
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "qwen3.8".to_string(),
            RegistryModel {
                default_variant: "27b-nvfp4".to_string(),
                variants,
            },
        );
        RegistryManifest {
            schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
            models,
        }
    }

    fn empty_registry() -> RegistryManifest {
        RegistryManifest {
            schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
            models: BTreeMap::new(),
        }
    }

    // ── Unclaimed forms: today's behaviour, unchanged ───────────────────

    #[test]
    fn an_existing_local_directory_passes_through_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let registry = empty_registry();
        let out = resolve_artifact_path_with(dir.path().to_str().unwrap(), &registry).unwrap();
        assert_eq!(out, dir.path());
    }

    #[test]
    fn a_bare_name_the_registry_has_never_claimed_passes_through_as_a_literal_path() {
        // Matches today's `load_artifact` behaviour exactly: an unclaimed
        // bare name becomes `PathBuf::from(path_str)`, not an error here —
        // `detect_generation` (called by `load_artifact`, not this helper)
        // is what actually refuses a nonexistent directory.
        let registry = empty_registry();
        let out = resolve_artifact_path_with("not-a-registered-name", &registry).unwrap();
        assert_eq!(out, std::path::PathBuf::from("not-a-registered-name"));
    }

    // ── Claimed names: registry-exclusive, no fallback on any failure ──

    #[test]
    fn a_claimed_name_with_an_unknown_variant_hard_errors() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let err = resolve_artifact_path_with("qwen3.8:does-not-exist", &registry).unwrap_err();
        assert!(err.to_string().contains("does-not-exist"), "{err}");
    }

    #[test]
    fn a_claimed_name_with_an_incompatible_abi_hard_errors() {
        let incompatible = Vindex3Abi(CURRENT_VINDEX3_ABI.get() + 1);
        let registry = registry_claiming_qwen38(incompatible);
        let err = resolve_artifact_path_with("qwen3.8", &registry).unwrap_err();
        assert!(err.to_string().contains("ABI"), "{err}");
    }

    // ── Explicit hf:// bypasses the claim check structurally ───────────

    #[test]
    fn an_hf_reference_is_not_treated_as_a_registry_name_even_if_claimed() {
        // `resolve_claimed` returns `Ok(None)` for any `hf://`-prefixed
        // string — it is never a bare `ModelReference::Registry` form —
        // so this falls through to the existing `is_hf_path` branch
        // untouched. Proven here via a malformed hf:// (would error
        // before ever attempting a real network fetch) rather than a
        // real repo, keeping this hermetic.
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let err = resolve_artifact_path_with("hf://", &registry).unwrap_err();
        // Not the registry's "unknown model"/"incompatible ABI" wording —
        // proves the claim check never engaged.
        assert!(!err.to_string().contains("ABI"), "{err}");
    }
}
