//! `larql pull` — fetch a Hugging Face repo into the local cache, or
//! refresh what is already there: a vindex (or a slice, or a whole
//! collection) is downloaded with ollama-style progress bars and free
//! resume; a raw checkpoint is encoded to VINDEX3.
//!
//! Four resolution paths, in order of specificity:
//!
//!   1. `pull <repo>`                   — plain pull, one repo
//!   2. `pull <repo> --preset client`   — pull the `-client` sibling instead
//!   3. `pull <repo> --all-slices`      — pull full + default slice siblings
//!   4. `pull --collection <slug|url>`  — pull every dataset in a collection
//!
//! After a single-repo pull, `pull` probes HF for the standard sibling
//! suffixes and prints a hint if any exist — so the slice convention is
//! self-announcing. A user landing on `chrishayuk/gemma-4-31b-it-vindex`
//! discovers `-client` / `-server` / `-browse` without having to read a
//! README.
//!
//! A repo that is a raw Hugging Face checkpoint rather than a vindex is
//! encoded to VINDEX3 into the encoded cache instead of downloaded — the
//! same first-run path `larql run hf://…` takes (`super::hf_auto`). For a
//! checkpoint already encoded, `pull` is the explicit refresh: the
//! revision is re-resolved against the hub and re-encoded only if it moved
//! (`super::hf_auto::refresh`). `run` itself never checks the hub.
//!
//! Progress + resume: `indicatif::MultiProgress` gives one bar per file;
//! hf-hub 0.5 handles `.incomplete` partial-file resume internally so an
//! interrupted pull picks up where it left off on the next run.

use std::path::PathBuf;

use clap::Args;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::hf_auto::HF_SCHEME;

/// Default sibling presets to probe / pull when the caller doesn't pass
/// `--preset`. Matches the `publish` default set; the symmetry matters
/// so `publish` and `pull` stay in lock-step.
const DEFAULT_SIBLING_PRESETS: &[&str] = &["client", "attn", "embed", "server", "browse"];

/// Same sibling-naming template as `publish` so `pull` can reverse what
/// `publish` produced without a separate configuration handshake.
const DEFAULT_SIBLING_TEMPLATE: &str = "{repo}-{preset}";

#[derive(Args)]
pub struct PullArgs {
    /// `hf://owner/name[@rev]`, `owner/name`, or the shorthand of an
    /// encoded checkpoint (`Qwen3-0.6B`) to refresh. Omit when passing
    /// `--collection`.
    pub model: Option<String>,

    /// Pull a sibling slice instead of the named repo. Options: `client`,
    /// `server`, `browse`, `router`, `all`. Resolves via the
    /// `--sibling-template` (default `{repo}-{preset}`).
    #[arg(long)]
    pub preset: Option<String>,

    /// Pull the full vindex *and* every default slice sibling
    /// (`-client`, `-attn`, `-embed`, `-server`, `-browse`) in one
    /// command. Missing siblings are warned-about, not fatal.
    #[arg(long)]
    pub all_slices: bool,

    /// Pull every dataset item in an HF collection. Accepts the slug
    /// (`namespace/slug-id`) or the full
    /// `https://huggingface.co/collections/…` URL. Mutually exclusive
    /// with `<model>`.
    #[arg(long)]
    pub collection: Option<String>,

    /// Override the sibling-resolution template. `{repo}` and `{preset}`
    /// substitute. Must match whatever `larql publish` wrote — defaults
    /// align, override only if you changed `publish --slice-repo-template`.
    #[arg(long, default_value = DEFAULT_SIBLING_TEMPLATE)]
    pub sibling_template: String,

    /// Download the vindex to this path instead of the default local cache.
    /// Useful for container deployments where weights live on a mounted volume.
    /// If the path already exists it is left unchanged (idempotent).
    #[arg(long, value_name = "PATH")]
    pub output: Option<std::path::PathBuf>,
}

pub fn run(args: PullArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ref slug_or_url) = args.collection {
        return pull_collection(slug_or_url);
    }

    let model = args
        .model
        .as_deref()
        .ok_or_else(|| "pull needs <model> or --collection".to_string())?;

    if args.all_slices {
        return pull_all_slices(model, &args.sibling_template);
    }

    if let Some(ref preset) = args.preset {
        let sibling = render_sibling_repo(model, preset, &args.sibling_template)?;
        eprintln!("Resolving --preset {preset} → {sibling}");
        return pull_one(&sibling, /*print_siblings=*/ false, None);
    }

    pull_one(model, /*print_siblings=*/ true, args.output.as_deref())
}

/// HuggingFace repos look like `owner/name` — exactly one `/`, neither
/// side empty, no leading `/`, no dot in the owner segment. Used by both
/// `render_sibling_repo` and `normalise_hf_path` so filesystem paths
/// never get confused for HF refs.
fn looks_like_hf_repo(s: &str) -> bool {
    if s.starts_with('/') {
        return false;
    }
    let mut parts = s.splitn(2, '/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    !owner.is_empty() && !name.is_empty() && !owner.contains('.') && !name.contains('/')
}

/// Render `{repo}-{preset}` (or the caller's override). Strips any
/// existing `hf://` prefix so the template operates on bare `owner/name`.
fn render_sibling_repo(
    model: &str,
    preset: &str,
    template: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let bare = model.trim_start_matches("hf://");
    if !looks_like_hf_repo(bare) {
        return Err(
            format!("--preset needs an `owner/name` repo, not a local path: {model}").into(),
        );
    }
    Ok(template.replace("{repo}", bare).replace("{preset}", preset))
}

/// `indicatif::ProgressBar` wrapper that implements hf-hub's `Progress`
/// trait. We can't use hf-hub's built-in `impl Progress for ProgressBar`
/// directly because hf-hub 0.5 pins indicatif 0.18 while the workspace
/// is on 0.17 — different types.
struct BarProgress(ProgressBar);

impl larql_vindex::DownloadProgress for BarProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.0.set_length(size as u64);
        self.0.set_style(
            ProgressStyle::with_template(
                "  {msg:28} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
                 {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>10} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        let msg = if filename.len() > 28 {
            format!("…{}", &filename[filename.len() - 27..])
        } else {
            filename.to_string()
        };
        self.0.set_message(msg);
    }
    fn update(&mut self, size: usize) {
        self.0.inc(size as u64);
    }
    fn finish(&mut self) {
        self.0.finish();
    }
}

fn download_with_indicatif(hf_path: &str) -> Result<PathBuf, larql_vindex::VindexError> {
    let mp = MultiProgress::new();
    larql_vindex::resolve_hf_vindex_with_progress(hf_path, |_filename| {
        BarProgress(mp.add(ProgressBar::new(0)))
    })
}

/// Recursively copy a directory tree (used when rename() crosses filesystems).
fn copy_dir_all(
    src: &std::path::Path,
    dst: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dst_path = dst.join(entry.file_name());
        let meta = entry.metadata()?;
        if meta.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

/// Resolve + download a single repo, then optionally probe for siblings.
fn pull_one(
    model: &str,
    print_siblings: bool,
    output: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // If --output is set and already populated, skip the download.
    if let Some(out) = output {
        if out.join("index.json").exists() {
            eprintln!(
                "Vindex already present at {} — skipping download.",
                out.display()
            );
            return Ok(());
        }
    }

    let hf_path = resolve_pull_hf_path(model)?;
    eprintln!("Pulling {hf_path}...");
    let resolution = super::hf_auto::pull_hf_model(&hf_path, &mut |spec| {
        download_with_indicatif(spec).map_err(Into::into)
    })?;
    let encoded = matches!(resolution, super::hf_auto::Resolution::Encoded { .. });
    let cached: PathBuf = resolution.into_path();
    eprintln!("Cached at: {}", cached.display());

    // VINDEX3 has no metadata/weight split the way VINDEX2 does — the
    // repo's own file listing IS the required download (design doc
    // §10.5), so "downloaded" must mean "usable": validate before
    // handing the result back as a success. VINDEX2 keeps its existing
    // (unvalidated-here) behaviour. An encoded checkpoint was not
    // downloaded as a container; the encoder's closure gate vouched for it.
    match larql_vindex::format::generation::detect_generation(&cached)? {
        larql_vindex::format::generation::ContainerGeneration::V3 if !encoded => {
            larql_vindex::format::vindex3::validate_downloaded_container(&cached)?;
        }
        _ => {}
    }

    // If --output is set, move the downloaded vindex to the requested path.
    if let Some(out) = output {
        if out != cached.as_path() {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Try rename first (fast, same filesystem); fall back to copy.
            // An encoded entry is copied, never moved: it belongs to the
            // encoded cache, which `run` keeps answering from.
            if encoded {
                copy_dir_all(&cached, out)?;
            } else if std::fs::rename(&cached, out).is_err() {
                eprintln!("Rename failed (cross-device?), copying...");
                copy_dir_all(&cached, out)?;
            }
            eprintln!("Vindex available at: {}", out.display());
            return Ok(());
        }
    }

    if let Ok(cfg) = larql_vindex::load_vindex_config(&cached) {
        eprintln!(
            "  {} layers, hidden_size={}, dtype={:?}, level={}",
            cfg.num_layers, cfg.hidden_size, cfg.dtype, cfg.extract_level
        );
    }

    if print_siblings {
        hint_siblings(model);
    }
    Ok(())
}

/// Pull every dataset item in an HF collection. A single-item failure
/// logs a warning but doesn't abort — one unavailable sibling shouldn't
/// fail the whole collection pull.
fn pull_collection(slug_or_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Fetching collection: {slug_or_url}");
    let items = larql_vindex::fetch_collection_items(slug_or_url)?;
    let datasets: Vec<String> = items
        .into_iter()
        .filter(|(kind, _)| kind == "dataset")
        .map(|(_, id)| id)
        .collect();
    if datasets.is_empty() {
        eprintln!("  (no dataset items in collection)");
        return Ok(());
    }
    eprintln!("  Found {} dataset repo(s):", datasets.len());
    for id in &datasets {
        eprintln!("    {id}");
    }

    let mut ok = 0usize;
    let mut failed: Vec<(String, String)> = Vec::new();
    for id in datasets {
        let hf_path = format!("hf://{id}");
        match download_with_indicatif(&hf_path) {
            Ok(cached) => {
                eprintln!("  ✓ {id} → {}", cached.display());
                ok += 1;
            }
            Err(e) => {
                eprintln!("  ✗ {id}: {e}");
                failed.push((id, e.to_string()));
            }
        }
    }
    eprintln!("\nPulled {ok} of {} repos.", ok + failed.len());
    if !failed.is_empty() {
        eprintln!("Failures:");
        for (id, err) in &failed {
            eprintln!("  {id}: {err}");
        }
    }
    Ok(())
}

/// Pull the full repo + every default sibling preset. Missing siblings
/// log a warning; only the full repo is hard-required.
fn pull_all_slices(model: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    pull_one(model, /*print_siblings=*/ false, None)?;
    for preset in DEFAULT_SIBLING_PRESETS {
        let sibling = match render_sibling_repo(model, preset, template) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  skip {preset}: {e}");
                continue;
            }
        };
        eprintln!("\n→ Pulling sibling `{preset}` ({sibling})");
        if let Err(e) = pull_one(&sibling, /*print_siblings=*/ false, None) {
            eprintln!("  skipped: {e}");
        }
    }
    Ok(())
}

/// After a successful pull, probe HF for standard sibling suffixes and
/// print what's available. Fail-silent — an HTTP error here shouldn't
/// mask the successful pull we just did.
fn hint_siblings(model: &str) {
    let bare = model.trim_start_matches("hf://");
    if !looks_like_hf_repo(bare) {
        return;
    }

    let (base, pulled_preset) = split_sibling_suffix(bare);
    let mut candidates: Vec<(String, String)> = Vec::new(); // (label, repo)
    if pulled_preset.is_some() {
        candidates.push(("full".into(), base.to_string()));
    }
    for preset in DEFAULT_SIBLING_PRESETS {
        if Some(*preset) == pulled_preset {
            continue;
        }
        candidates.push((preset.to_string(), format!("{base}-{preset}")));
    }

    let mut found: Vec<(String, String)> = Vec::new();
    for (label, repo) in &candidates {
        if let Ok(true) = larql_vindex::dataset_repo_exists(repo) {
            found.push((label.clone(), repo.clone()));
        }
    }
    if !found.is_empty() {
        eprintln!("\n  Also available on HuggingFace:");
        for (label, repo) in &found {
            eprintln!("    --preset {label:<8} → hf://{repo}");
        }
        eprintln!("  Use `larql pull <repo> --all-slices` to grab them all.");
    }
}

/// If `bare` ends in one of the known preset suffixes, return `(base,
/// Some(preset))`. Otherwise `(bare, None)`. Lets `hint_siblings`
/// suggest the full repo when the user pulled a specific slice directly.
fn split_sibling_suffix(bare: &str) -> (&str, Option<&'static str>) {
    for preset in DEFAULT_SIBLING_PRESETS {
        let suffix = format!("-{preset}");
        if let Some(base) = bare.strip_suffix(&suffix) {
            let preset_static: &'static str = preset;
            return (base, Some(preset_static));
        }
    }
    (bare, None)
}

fn normalise_hf_path(model: &str) -> Result<String, Box<dyn std::error::Error>> {
    if model.starts_with("hf://") {
        return Ok(model.to_string());
    }
    if looks_like_hf_repo(model) {
        return Ok(format!("hf://{model}"));
    }
    Err(format!(
        "pull expects `hf://owner/name`, `owner/name` or the shorthand of an encoded \
         checkpoint, got: {model}"
    )
    .into())
}

/// Resolve `model` to the `hf://` reference [`pull_one`] actually
/// downloads — the resolver-convergence rung's third and final seam
/// (`docs/vindex3-registry-design.md` §10.5), sharing the exact
/// claimed/unclaimed dispatch `larql-cli`'s `serve` trampoline and
/// `larql-server`'s `load_artifact` already use.
///
/// A bare name the VINDEX3 registry has claimed resolves through it
/// **exclusively** — its pinned `hf://repo@revision` reference, any
/// failure (unknown variant, incompatible ABI) a real refusal. It is
/// never rescued by falling through to [`normalise_hf_path`]'s legacy
/// `owner/name`/`hf://` heuristic (`looks_like_hf_repo`) below — that
/// heuristic is now purely the *unclaimed* path, not `pull`'s only
/// identity concept.
fn resolve_pull_hf_path(model: &str) -> Result<String, Box<dyn std::error::Error>> {
    resolve_pull_hf_path_with(
        model,
        &larql_vindex::registry::production_registry(),
        &super::hf_auto::EncodedStore::open_default()?,
    )
}

/// Testable core of [`resolve_pull_hf_path`]. Takes `registry` explicitly
/// so the claimed/unclaimed dispatch can be proven against a non-empty
/// registry without depending on the (currently empty) production one —
/// `resolve_claimed_hf_reference` never touches the network, so this is
/// hermetic either way.
///
/// An unclaimed bare name that is the shorthand of an encoded checkpoint
/// (`Qwen3-0.6B` for an encoded `hf://Qwen/Qwen3-0.6B`) pulls that entry —
/// the explicit refresh — at the revision it tracks.
fn resolve_pull_hf_path_with(
    model: &str,
    registry: &larql_vindex::registry::RegistryManifest,
    encoded: &super::hf_auto::EncodedStore,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(hf_ref) = larql_vindex::registry::resolve_claimed_hf_reference(model, registry)? {
        return Ok(hf_ref);
    }
    if !model.starts_with(HF_SCHEME) && !model.contains('/') {
        if let Some(spec) = super::hf_auto::refresh::pull_spec_for_shorthand(encoded, model)? {
            return Ok(spec);
        }
    }
    normalise_hf_path(model)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_sibling_uses_default_template() {
        let got = render_sibling_repo(
            "chrishayuk/gemma-4-31b-it-vindex",
            "client",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap();
        assert_eq!(got, "chrishayuk/gemma-4-31b-it-vindex-client");
    }

    #[test]
    fn render_sibling_strips_hf_prefix() {
        let got = render_sibling_repo(
            "hf://chrishayuk/gemma-4-31b-it-vindex",
            "server",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap();
        assert_eq!(got, "chrishayuk/gemma-4-31b-it-vindex-server");
    }

    #[test]
    fn render_sibling_custom_template() {
        let got = render_sibling_repo("me/model", "browse", "{repo}/{preset}").unwrap();
        assert_eq!(got, "me/model/browse");
    }

    #[test]
    fn render_sibling_rejects_local_path() {
        let err = render_sibling_repo(
            "/local/path/model.vindex",
            "client",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap_err();
        assert!(err.to_string().contains("owner/name"), "got: {err}");
    }

    #[test]
    fn split_sibling_suffix_recognises_known_presets() {
        assert_eq!(
            split_sibling_suffix("chrishayuk/gemma-4-31b-it-vindex-client"),
            ("chrishayuk/gemma-4-31b-it-vindex", Some("client")),
        );
        assert_eq!(
            split_sibling_suffix("me/model-server"),
            ("me/model", Some("server")),
        );
        assert_eq!(
            split_sibling_suffix("me/model-browse"),
            ("me/model", Some("browse")),
        );
    }

    #[test]
    fn split_sibling_suffix_leaves_full_repo_untouched() {
        assert_eq!(
            split_sibling_suffix("chrishayuk/gemma-4-31b-it-vindex"),
            ("chrishayuk/gemma-4-31b-it-vindex", None),
        );
    }

    #[test]
    fn normalise_hf_path_accepts_hf_prefix_and_owner_name() {
        assert_eq!(normalise_hf_path("hf://me/model").unwrap(), "hf://me/model");
        assert_eq!(normalise_hf_path("me/model").unwrap(), "hf://me/model");
    }

    #[test]
    fn normalise_hf_path_rejects_single_word() {
        assert!(normalise_hf_path("nomodel").is_err());
    }

    #[test]
    fn normalise_hf_path_rejects_local_path() {
        assert!(normalise_hf_path("/abs/path/model.vindex").is_err());
    }

    #[test]
    fn default_sibling_presets_match_publish_defaults() {
        // Symmetry guard: if publish's default slice set changes, pull
        // must change in lock-step so sibling hints don't go stale.
        // Keep in sync with `publish_cmd::DEFAULT_SLICES`.
        assert_eq!(
            DEFAULT_SIBLING_PRESETS,
            &["client", "attn", "embed", "server", "browse"]
        );
    }

    // ── resolve_pull_hf_path: the claimed/unclaimed convergence seam ────
    //
    // `resolve_claimed_hf_reference` never touches the network (identity
    // only, no fetch — see `larql_vindex::registry`'s own tests for that
    // contract), so every case here is hermetic.

    /// An encoded store with nothing in it (a path that does not exist).
    fn no_encoded() -> crate::commands::primary::hf_auto::EncodedStore {
        crate::commands::primary::hf_auto::EncodedStore::new(
            std::env::temp_dir().join("larql-pull-tests-no-encoded-store"),
        )
    }

    fn empty_registry() -> larql_vindex::registry::RegistryManifest {
        larql_vindex::registry::RegistryManifest {
            schema_version: larql_vindex::registry::REGISTRY_MANIFEST_SCHEMA_VERSION,
            models: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn an_encoded_shorthand_pulls_the_revision_it_tracks() {
        let tmp = tempfile::tempdir().unwrap();
        let store = crate::commands::primary::hf_auto::EncodedStore::new(tmp.path());
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let snap = store.snapshot_dir("Qwen/Qwen3-0.6B", commit).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("index.json"), b"{}").unwrap();
        store.record_ref("Qwen/Qwen3-0.6B", "main", commit).unwrap();
        assert_eq!(
            resolve_pull_hf_path_with("Qwen3-0.6B", &empty_registry(), &store).unwrap(),
            "hf://Qwen/Qwen3-0.6B"
        );
        // Not an encoded shorthand: the old error stands.
        assert!(resolve_pull_hf_path_with("nomodel", &empty_registry(), &store).is_err());
    }

    fn registry_claiming_qwen38(
        abi: larql_vindex::registry::Vindex3Abi,
    ) -> larql_vindex::registry::RegistryManifest {
        use larql_vindex::registry::{
            Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel,
            RegistryVariant, REGISTRY_MANIFEST_SCHEMA_VERSION,
        };
        let mut variants = std::collections::BTreeMap::new();
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
        let mut models = std::collections::BTreeMap::new();
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

    #[test]
    fn a_claimed_name_resolves_to_its_pinned_hf_reference() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let hf_path = resolve_pull_hf_path_with("qwen3.8", &registry, &no_encoded()).unwrap();
        assert_eq!(hf_path, "hf://larql/qwen3.8-27b-nvfp4@abc123f0");
    }

    #[test]
    fn an_unclaimed_name_falls_through_to_the_legacy_owner_name_heuristic() {
        let empty = larql_vindex::registry::RegistryManifest {
            schema_version: larql_vindex::registry::REGISTRY_MANIFEST_SCHEMA_VERSION,
            models: std::collections::BTreeMap::new(),
        };
        let hf_path = resolve_pull_hf_path_with("me/model", &empty, &no_encoded()).unwrap();
        assert_eq!(hf_path, "hf://me/model");
    }

    #[test]
    fn a_claimed_name_with_an_unknown_variant_never_falls_through_to_the_heuristic() {
        // If this fell through, "qwen3.8:does-not-exist" would fail
        // `looks_like_hf_repo` (a `:` isn't a `/`) with a *different*,
        // misleading error. The registry's own error must win.
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        let err = resolve_pull_hf_path_with("qwen3.8:does-not-exist", &registry, &no_encoded())
            .unwrap_err();
        assert!(err.to_string().contains("does-not-exist"), "{err}");
    }

    #[test]
    fn a_claimed_name_with_an_incompatible_abi_never_falls_through_to_the_heuristic() {
        let incompatible = larql_vindex::registry::Vindex3Abi(
            larql_vindex::registry::CURRENT_VINDEX3_ABI.get() + 1,
        );
        let registry = registry_claiming_qwen38(incompatible);
        let err = resolve_pull_hf_path_with("qwen3.8", &registry, &no_encoded()).unwrap_err();
        assert!(err.to_string().contains("ABI"), "{err}");
    }

    #[test]
    fn an_explicit_hf_reference_bypasses_the_claim_check() {
        let registry = registry_claiming_qwen38(larql_vindex::registry::CURRENT_VINDEX3_ABI);
        // Not a bare registry-shaped reference — `resolve_claimed_hf_reference`
        // is structurally `None` for it, so the legacy hf:// passthrough wins
        // even though the registry claims the same short name.
        let hf_path =
            resolve_pull_hf_path_with("hf://owner/repo", &registry, &no_encoded()).unwrap();
        assert_eq!(hf_path, "hf://owner/repo");
    }
}
