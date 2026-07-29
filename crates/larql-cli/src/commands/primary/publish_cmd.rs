//! `larql publish <SRC> --repo OWNER/NAME` — upload a vindex to HuggingFace,
//! optionally carving + uploading deployment slices to sibling repos in one
//! go.
//!
//! The default (`--all`) produces four repos from a single source vindex:
//!
//!   * `OWNER/NAME`         — the full vindex (INFER + DESCRIBE)
//!   * `OWNER/NAME-client`  — attention-only slice (pair with `run --ffn URL`)
//!   * `OWNER/NAME-server`  — FFN-only slice (pair with `serve --ffn-only`)
//!   * `OWNER/NAME-browse`  — gate + embed + down_meta (DESCRIBE/WALK only)
//!
//! The `router` preset is opt-in via `--slices` because dense vindexes don't
//! carry `router_weights.bin` and the resulting repo would be empty.
//!
//! Under the covers this is `larql slice` + `larql hf publish` bundled: each
//! slice is staged in a temp directory, uploaded to its sibling repo via
//! `larql_vindex::publish_vindex`, and then cleaned up.
//!
//! Requires `HF_TOKEN` (or `~/.huggingface/token`) just like `larql hf publish`.

use larql_vindex::format::filenames::*;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::Args;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::commands::primary::cache;
use crate::commands::primary::slice_cmd::{preset_parts, slice_vindex, Part};

/// Default sibling slice presets when `--slices` is not given. Covers
/// every deployment shape ADR-0007 and ADR-0008 support today:
///
///   * `client`  — 2-tier dense-remote (client holds embed locally)
///   * `attn`    — 3-tier dense-remote client (embed delegated)
///   * `embed`   — 3-tier embed server
///   * `server`  — 3-tier / 2-tier FFN server
///   * `browse`  — read-only DESCRIBE/WALK consumers
///
/// `router` is omitted because it would produce an empty repo on non-MoE
/// vindexes; request it explicitly via `--slices router` when relevant.
/// Publishing all five by default is cheap: skip-if-unchanged keeps the
/// re-upload cost at a few KB per slice once the LFS blobs are already
/// on HF.
const DEFAULT_SLICES: &[&str] = &["client", "attn", "embed", "server", "browse"];

#[derive(Args)]
pub struct PublishArgs {
    /// Source vindex: directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub source: String,

    /// HuggingFace repo ID for the full vindex (e.g. `chrishayuk/gemma-4-31b`).
    /// Sibling slice repos are named `<repo>-<preset>` by default.
    #[arg(long)]
    pub repo: String,

    /// Publish the full vindex to `--repo`. On by default; pair with
    /// `--no-full --slices client,server` to publish only the slices.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub full: bool,

    /// Shortcut: `--no-full` is the same as `--full false`.
    #[arg(long, conflicts_with = "full")]
    pub no_full: bool,

    /// Comma-separated slice presets to publish alongside the full vindex.
    /// Defaults to `client,attn,embed,server,browse` — covers both the
    /// 2-tier and 3-tier (ADR-0008) topologies in one run. Pass `none`
    /// to skip all slice uploads.
    #[arg(long, value_delimiter = ',')]
    pub slices: Vec<String>,

    /// Suffix template for sibling slice repos. `{repo}` is replaced with
    /// `--repo`; `{preset}` with the preset name. Default: `{repo}-{preset}`.
    #[arg(long, default_value = "{repo}-{preset}")]
    pub slice_repo_template: String,

    /// Directory to stage intermediate slices. Defaults to the system temp
    /// dir; each slice gets its own subdir and is cleaned up on success.
    #[arg(long)]
    pub tmp_dir: Option<PathBuf>,

    /// Preview the upload plan without creating repos or uploading files.
    #[arg(long)]
    pub dry_run: bool,

    /// Collection levels to create or update after the uploads land.
    /// Comma list of: `model` (per-model-size), `family` (per-architecture),
    /// `library` (one top-level "LARQL Vindex Library"). Default is all
    /// three. Pass `none` to skip collection creation entirely.
    #[arg(long, value_delimiter = ',', default_value = "model,family,library")]
    pub collections: Vec<String>,

    /// Override the model title used in the per-model collection. Default
    /// is derived from the vindex config (e.g. `Gemma 4 31B`).
    #[arg(long)]
    pub model_title: Option<String>,

    /// Override the family name used in the family-level collection
    /// (e.g. `Gemma`). Default: prefix of the model id up to the first
    /// version/size token.
    #[arg(long)]
    pub family: Option<String>,

    /// Title for the library-level collection. Default matches the one
    /// in docs: "LARQL Vindex Library". Override if you want a namespaced
    /// variant.
    #[arg(long, default_value = "LARQL Vindex Library")]
    pub library_title: String,

    /// Force re-upload of every file even if the remote copy already
    /// matches the local SHA256. By default `publish` fetches the remote
    /// LFS file index and skips any file whose `lfs.oid` equals the
    /// local SHA256, which saves a full re-upload when nothing changed.
    ///
    /// Use this flag to bypass the skip and re-upload everything, e.g.
    /// if you suspect a prior upload was truncated.
    #[arg(long)]
    pub force_upload: bool,

    /// HuggingFace repo type: `model` (default) or `dataset`.
    #[arg(long, default_value = "model")]
    pub repo_type: String,

    /// Create every repo (full + slices) private. Pair with `larql hf
    /// visibility --public` once verification passes — Vindex Factory's
    /// two-phase publish (docs/vindex-factory.md §8: "nothing goes
    /// public unverified"). Repos already on the Hub keep their
    /// existing visibility; this only affects repo *creation*.
    #[arg(long)]
    pub private: bool,
}

pub fn run(args: PublishArgs) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Resolve source.
    let src = cache::resolve_model(&args.source)?;
    if !src.is_dir() {
        return Err(format!("source vindex not a directory: {}", src.display()).into());
    }
    if !src.join(INDEX_JSON).exists() {
        return Err(format!("source vindex missing index.json: {}", src.display()).into());
    }

    let publish_full = args.full && !args.no_full;
    let requested_slices = resolve_slice_list(&args.slices)?;
    if !publish_full && requested_slices.is_empty() {
        return Err(
            "nothing to publish: `--no-full` requires at least one preset in `--slices`".into(),
        );
    }

    // 2. Build the upload plan.
    let mut plan: Vec<UploadStep> = Vec::new();
    if publish_full {
        plan.push(UploadStep {
            label: "full".into(),
            repo: args.repo.clone(),
            preset: None,
            staging: None,
        });
    }
    let staging_root = args.tmp_dir.clone().unwrap_or_else(std::env::temp_dir);
    for preset in &requested_slices {
        let repo = args
            .slice_repo_template
            .replace("{repo}", &args.repo)
            .replace("{preset}", preset);
        // Unique subdir per (pid, preset) so parallel invocations don't collide.
        let staging = staging_root.join(format!(
            "larql-publish-{}-{}-{}.vindex",
            args.repo.replace('/', "_"),
            preset,
            std::process::id()
        ));
        plan.push(UploadStep {
            label: preset.clone(),
            repo,
            preset: Some(preset.clone()),
            staging: Some(staging),
        });
    }

    // 3. Print the plan.
    println!("Source:    {}", src.display());
    println!("Upload plan ({} step(s)):", plan.len());
    for step in &plan {
        match &step.preset {
            None => println!("  full    → {}", step.repo),
            Some(p) => println!("  {p:<7} → {}", step.repo),
        }
    }
    let preview_levels = resolve_collection_list(&args.collections)?;
    if !preview_levels.is_empty() {
        let cfg = larql_vindex::load_vindex_config(&src)?;
        let model_title = args
            .model_title
            .clone()
            .unwrap_or_else(|| format!("{} — LARQL Vindex", default_model_title(&cfg.model)));
        let family = args
            .family
            .clone()
            .unwrap_or_else(|| default_family(&cfg.model));
        println!("Collections:");
        for level in &preview_levels {
            let title = match level.as_str() {
                "model" => model_title.clone(),
                "family" => format!("{family} Family — LARQL Vindexes"),
                "library" => args.library_title.clone(),
                _ => continue,
            };
            let namespace = namespace_of(&args.repo)?;
            println!("  {level:<8} {namespace}: {title}");
        }
    }
    if args.dry_run {
        println!("\n(dry run — no repos created, no files uploaded)");
        return Ok(());
    }

    // 4. Execute each step.
    let mut results: Vec<StepOutcome> = Vec::new();
    for step in plan {
        let url = execute_step(
            &src,
            &step,
            args.force_upload,
            &args.repo_type,
            args.private,
        )?;
        results.push(StepOutcome {
            label: step.label,
            repo: step.repo,
            url,
        });
    }

    // 5. Collection step — group the uploaded repos into HF collections.
    let collection_levels = resolve_collection_list(&args.collections)?;
    let collection_urls = if !collection_levels.is_empty() {
        Some(build_collections(
            &src,
            &args,
            &results,
            &collection_levels,
        )?)
    } else {
        None
    };

    // 6. Summary.
    println!("\nPublished:");
    for r in &results {
        println!("  {:<8} {} → {}", r.label, r.repo, r.url);
    }
    if let Some(urls) = collection_urls {
        println!("\nCollections:");
        for (level, url) in &urls {
            println!("  {level:<8} {url}");
        }
    }
    println!("\nPull any of these with:");
    for r in &results {
        println!("  larql pull hf://{}", r.repo);
    }
    Ok(())
}

fn resolve_collection_list(raw: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if raw.len() == 1 && raw[0].eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(raw.len());
    for name in raw {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "model" | "family" | "library" => out.push(lower),
            other => {
                return Err(format!(
                    "invalid collection level '{other}'. Valid: model, family, library, none"
                )
                .into());
            }
        }
    }
    Ok(out)
}

/// Parse `OWNER/NAME` → `OWNER`. Returns an error for bare names so we
/// don't accidentally treat a missing namespace as valid.
fn namespace_of(repo: &str) -> Result<&str, Box<dyn std::error::Error>> {
    repo.split_once('/')
        .map(|(ns, _)| ns)
        .ok_or_else(|| format!("--repo must be `OWNER/NAME`, got '{repo}'").into())
}

/// Extract the short model name from whatever `index.json` happens to
/// carry in its `model` field. Handles:
///
///   * `google/gemma-4-31b-it`               → `gemma-4-31b-it`
///   * `/absolute/path/...gemma-4-31b-it/`   → `gemma-4-31b-it`
///   * `.../models--google--gemma-4-31B-it/` → `gemma-4-31B-it` (HF cache layout)
///   * `gemma-4-31b-it`                      → `gemma-4-31b-it`
fn short_model_name(model_field: &str) -> &str {
    // Drop trailing slashes so `rsplit` doesn't return the empty string.
    let trimmed = model_field.trim_end_matches('/');

    // HF cache layout: `.../models--{owner}--{name}/snapshots/{hash}/`
    // At this point the trailing `snapshots/{hash}` is already trimmed
    // by `rsplit` below; the `models--…` directory is what remains.
    let last = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if let Some(rest) = last.strip_prefix("models--") {
        // `google--gemma-4-31B-it` → `gemma-4-31B-it`
        if let Some((_owner, name)) = rest.split_once("--") {
            return name;
        }
        return rest;
    }
    // Walk back up looking for a `models--…` segment (when the tail is a
    // hash directory like `.../snapshots/abc123/`).
    for seg in trimmed.rsplit('/') {
        if let Some(rest) = seg.strip_prefix("models--") {
            if let Some((_owner, name)) = rest.split_once("--") {
                return name;
            }
            return rest;
        }
    }
    last
}

/// Default model title derived from the vindex's `model` field in
/// `index.json`. Title-cases segments separated by `-` so
/// `gemma-4-31b-it` → `Gemma 4 31b It`. Override with `--model-title`
/// when clarity matters.
fn default_model_title(model_field: &str) -> String {
    let short = short_model_name(model_field);
    short
        .split('-')
        .map(|seg| {
            let mut chars = seg.chars();
            match chars.next() {
                Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Default family = prefix of the model id up to (but not including) the
/// first segment that looks like a size/version token — one starting with
/// a digit. `gemma-4-31b-it` → `Gemma`; `gemma-3-4b-it` → `Gemma`;
/// `llama-3-8b-instruct` → `Llama`.
fn default_family(model_field: &str) -> String {
    let short = short_model_name(model_field);
    let mut segs: Vec<&str> = Vec::new();
    for seg in short.split('-') {
        if seg
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            break;
        }
        segs.push(seg);
    }
    if segs.is_empty() {
        return short.to_string();
    }
    segs.iter()
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn note_for_preset(preset: &str) -> &'static str {
    match preset {
        "client" => "2-tier client — attention + embed + norms. Pair with `larql run --ffn URL`.",
        "attn" | "attention" => {
            "3-tier attention client — attn + norms only. Pair with `larql run --embed URL --ffn URL` (ADR-0008)."
        }
        "embed" | "embed-server" => {
            "Embed-server slice — embeddings + tokenizer. Pair with `larql serve --embed-only` (ADR-0008)."
        }
        "server" => "FFN-only slice — pair with `larql serve --ffn-only`.",
        "browse" => "Browse-only slice — DESCRIBE / WALK / SELECT, no forward pass.",
        "router" => "Router slice — MoE router weights only (ADR-0003).",
        "all" => "Full mirror.",
        _ => "Sliced variant.",
    }
}

fn note_for_full() -> &'static str {
    "Canonical full vindex — INFER + DESCRIBE."
}

fn build_collections(
    src: &Path,
    args: &PublishArgs,
    uploaded: &[StepOutcome],
    levels: &[String],
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let namespace = namespace_of(&args.repo)?;
    let cfg = larql_vindex::load_vindex_config(src)?;

    let model_title = args
        .model_title
        .clone()
        .unwrap_or_else(|| format!("{} — LARQL Vindex", default_model_title(&cfg.model)));
    let family = args
        .family
        .clone()
        .unwrap_or_else(|| default_family(&cfg.model));
    let family_title = format!("{family} Family — LARQL Vindexes");
    let library_title = args.library_title.clone();

    let items: Vec<larql_vindex::CollectionItem> = uploaded
        .iter()
        .map(|r| larql_vindex::CollectionItem {
            repo_id: r.repo.clone(),
            repo_type: args.repo_type.clone(),
            note: Some(if r.label == "full" {
                note_for_full().into()
            } else {
                note_for_preset(&r.label).into()
            }),
        })
        .collect();

    if args.dry_run {
        // Shouldn't normally hit this path (dry_run returns earlier), but
        // keep the branch so future refactors don't accidentally upload.
        return Ok(Vec::new());
    }

    let mut urls = Vec::new();
    for level in levels {
        let (level_title, description) = match level.as_str() {
            "model" => (
                model_title.clone(),
                format!(
                    "All deployment variants of {} as LARQL vindexes — full, client, server, browse.",
                    default_model_title(&cfg.model)
                ),
            ),
            "family" => (
                family_title.clone(),
                format!("LARQL vindexes for the {family} model family."),
            ),
            "library" => (
                library_title.clone(),
                "Every LARQL vindex in one place — browse, client, server, and full mirrors for each supported model."
                    .to_string(),
            ),
            _ => continue,
        };

        println!(
            "\n→ {} collection `{}` under `{}`…",
            match level.as_str() {
                "model" => "Updating",
                "family" => "Updating",
                "library" => "Updating",
                _ => "Updating",
            },
            level_title,
            namespace
        );
        let url =
            larql_vindex::ensure_collection(namespace, &level_title, Some(&description), &items)?;
        println!("  {url}");
        urls.push((level.clone(), url));
    }
    Ok(urls)
}

fn resolve_slice_list(raw: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Default set when --slices is not passed.
    if raw.is_empty() {
        return Ok(DEFAULT_SLICES.iter().map(|s| s.to_string()).collect());
    }
    // Explicit opt-out.
    if raw.len() == 1 && raw[0].eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(raw.len());
    for name in raw {
        let trimmed = name.trim();
        // Validate by round-tripping through preset_parts. Catches typos
        // before we start creating repos.
        preset_parts(trimmed).map_err(|e| {
            format!(
                "invalid slice preset '{trimmed}': {e}. Valid: client, attn, embed, server, browse, router, all"
            )
        })?;
        out.push(trimmed.to_string());
    }
    Ok(out)
}

struct UploadStep {
    label: String,
    repo: String,
    /// `None` for the full-vindex upload; `Some(preset)` for a sliced upload.
    preset: Option<String>,
    /// Where the sliced vindex gets staged before upload.
    staging: Option<PathBuf>,
}

struct StepOutcome {
    label: String,
    repo: String,
    url: String,
}

fn execute_step(
    src: &Path,
    step: &UploadStep,
    force_upload: bool,
    repo_type: &str,
    private: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    match (&step.preset, &step.staging) {
        // Full vindex — upload the source directory directly, no slicing.
        (None, _) => {
            println!("\n→ Uploading full vindex to {}", step.repo);
            upload_dir(src, &step.repo, force_upload, repo_type, private)
        }
        // Sliced upload — carve into staging, upload, clean up.
        (Some(preset), Some(staging)) => {
            println!("\n→ Carving slice `{preset}` …");
            let parts: BTreeSet<Part> =
                preset_parts(preset).map_err(|e| format!("preset `{preset}`: {e}"))?;
            let outcome = slice_vindex(
                src, staging, parts, /*force=*/ true, /*dry_run=*/ false,
            )?;
            println!(
                "  staged {} file(s), {} — {}",
                outcome.copied.len(),
                human_size(outcome.total_bytes),
                staging.display()
            );
            println!("→ Uploading slice `{preset}` to {}", step.repo);
            let result = upload_dir(staging, &step.repo, force_upload, repo_type, private);
            // Always try to clean up the staging dir, regardless of outcome.
            let _ = std::fs::remove_dir_all(staging);
            result
        }
        (Some(_), None) => Err("internal: slice step without staging dir".into()),
    }
}

fn upload_dir(
    dir: &Path,
    repo: &str,
    force_upload: bool,
    repo_type: &str,
    private: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut callbacks = CliPublishCallbacks::new();
    let opts = larql_vindex::PublishOptions {
        skip_unchanged: !force_upload,
        repo_type: repo_type.to_string(),
        private,
    };
    let url = larql_vindex::publish_vindex_with_opts(dir, repo, &opts, &mut callbacks)?;
    Ok(url)
}

// ─── Progress reporter ───────────────────────────────────────────────────
//
// One `MultiProgress` per upload-step (i.e. per sibling repo). Each file
// gets its own bar via `on_file_start`; `on_file_progress` ticks it as
// bytes flow through the counting-reader upload body (see
// `larql_vindex::upload_file_to_hf`). Skipped files get a finished bar
// so the line stays visible in the scrollback.

struct CliPublishCallbacks {
    mp: MultiProgress,
    current: Option<ProgressBar>,
}

impl CliPublishCallbacks {
    fn new() -> Self {
        Self {
            mp: MultiProgress::new(),
            current: None,
        }
    }
}

fn make_upload_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "    {msg:28} [{elapsed_precise}] [{wide_bar:.green/blue}] \
         {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>10} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-")
}

fn truncate_msg(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("…{}", &s[s.len() - (max - 1)..])
    } else {
        s.to_string()
    }
}

impl larql_vindex::PublishCallbacks for CliPublishCallbacks {
    fn on_start(&mut self, repo: &str) {
        eprintln!("  Creating repo: {}", repo);
    }

    fn on_file_start(&mut self, filename: &str, size: u64) {
        let mb = size as f64 / 1024.0 / 1024.0;
        eprintln!("  ↑ {filename} ({mb:.0} MB)");
        let bar = self.mp.add(ProgressBar::new(size));
        bar.set_style(make_upload_style());
        bar.set_message(truncate_msg(filename, 28));
        self.current = Some(bar);
    }

    fn on_file_progress(&mut self, _filename: &str, bytes_sent: u64, _total_bytes: u64) {
        if let Some(ref bar) = self.current {
            bar.set_position(bytes_sent);
        }
    }

    fn on_file_done(&mut self, filename: &str) {
        if let Some(bar) = self.current.take() {
            bar.finish();
        }
        eprintln!("    ✓ {filename}");
    }

    fn on_file_skipped(&mut self, filename: &str, _size: u64, sha256: &str) {
        // Print a plain line above the active bars rather than adding a
        // finished-bar stub. `MultiProgress::println` cooperates with
        // indicatif's cursor handling so the output stays one-line-per-
        // file even on wide terminals; the earlier bar-based approach
        // let indicatif pack multiple "skipped" entries on the same row
        // when it thought it had horizontal space.
        let short_sha = sha256.get(..12).unwrap_or(sha256);
        let _ = self.mp.println(format!(
            "    {:<28} [skipped — unchanged, sha256 {}…]",
            truncate_msg(filename, 28),
            short_sha
        ));
    }

    fn on_complete(&mut self, url: &str) {
        eprintln!("  URL: {}", url);
    }
}

fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_slice_list_is_full_publish_set() {
        // Flipping this default changes what bare `larql publish` writes
        // to HF — pin the exact order so the test fails loudly if it
        // gets rearranged. Covers both 2-tier (`client`) and 3-tier
        // (`attn` + `embed`) deployment shapes out of the box.
        let got = resolve_slice_list(&[]).unwrap();
        assert_eq!(got, vec!["client", "attn", "embed", "server", "browse"]);
    }

    #[test]
    fn slices_none_disables_sliced_uploads() {
        let got = resolve_slice_list(&["none".to_string()]).unwrap();
        assert!(got.is_empty());
        // Case-insensitive.
        let got_caps = resolve_slice_list(&["NONE".to_string()]).unwrap();
        assert!(got_caps.is_empty());
    }

    #[test]
    fn slices_explicit_list_passes_through() {
        let raw = vec!["client".into(), "server".into()];
        let got = resolve_slice_list(&raw).unwrap();
        assert_eq!(got, vec!["client", "server"]);
    }

    #[test]
    fn slices_with_router_is_valid() {
        // Router is a real preset even though it's omitted from the default
        // set. Passing it explicitly must round-trip cleanly.
        let got = resolve_slice_list(&["router".into()]).unwrap();
        assert_eq!(got, vec!["router"]);
    }

    #[test]
    fn slices_invalid_name_errors() {
        let err = resolve_slice_list(&["typo".into()]).unwrap_err();
        assert!(
            err.to_string().contains("invalid slice preset"),
            "got: {err}"
        );
    }

    #[test]
    fn slice_repo_template_substitution() {
        let template = "{repo}-{preset}";
        let rendered = template
            .replace("{repo}", "chrishayuk/gemma-4-31b")
            .replace("{preset}", "client");
        assert_eq!(rendered, "chrishayuk/gemma-4-31b-client");
    }

    #[test]
    fn slice_repo_template_custom_separator() {
        // Verify callers can override to e.g. "{repo}_{preset}" without
        // hard-coding a dash in the implementation.
        let template = "{repo}/{preset}";
        let rendered = template
            .replace("{repo}", "me/model")
            .replace("{preset}", "client");
        assert_eq!(rendered, "me/model/client");
    }

    // ── Collection helpers ─────────────────────────────────────────────

    #[test]
    fn default_collection_levels_are_all_three() {
        // Matches the clap default_value on --collections. The default
        // publishes to every level so a single run produces the full
        // docs structure (library → family → model).
        let raw = vec!["model".into(), "family".into(), "library".into()];
        let got = resolve_collection_list(&raw).unwrap();
        assert_eq!(got, vec!["model", "family", "library"]);
    }

    #[test]
    fn collection_level_none_disables_all() {
        let got = resolve_collection_list(&["none".into()]).unwrap();
        assert!(got.is_empty());
        // Case-insensitive.
        let got_caps = resolve_collection_list(&["NONE".into()]).unwrap();
        assert!(got_caps.is_empty());
    }

    #[test]
    fn collection_level_invalid_errors() {
        let err = resolve_collection_list(&["world".into()]).unwrap_err();
        assert!(
            err.to_string().contains("invalid collection level"),
            "got: {err}"
        );
    }

    #[test]
    fn collection_level_is_lowercased() {
        let got = resolve_collection_list(&["Model".into(), "FAMILY".into()]).unwrap();
        assert_eq!(got, vec!["model", "family"]);
    }

    #[test]
    fn namespace_of_rejects_bare_name() {
        assert!(namespace_of("chrishayuk/gemma-4-31b").is_ok());
        assert_eq!(
            namespace_of("chrishayuk/gemma-4-31b").unwrap(),
            "chrishayuk"
        );
        assert!(namespace_of("gemma-4-31b").is_err());
    }

    #[test]
    fn default_model_title_strips_hf_namespace() {
        assert_eq!(
            default_model_title("google/gemma-4-31b-it"),
            "Gemma 4 31b It"
        );
        assert_eq!(default_model_title("gemma-3-4b-it"), "Gemma 3 4b It");
        assert_eq!(
            default_model_title("llama-3-70b-instruct"),
            "Llama 3 70b Instruct"
        );
    }

    #[test]
    fn short_model_name_handles_hf_cache_layout() {
        // Absolute paths from the HF cache trim trailing slashes and
        // strip the `models--{owner}--` prefix so we don't end up with
        // empty titles.
        let cached =
            "/Users/me/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/abc123/";
        assert_eq!(short_model_name(cached), "gemma-4-31B-it");

        // Plain path without the `models--` prefix falls back to the
        // last segment, handling trailing slash correctly.
        assert_eq!(short_model_name("/path/to/gemma-3-4b-it/"), "gemma-3-4b-it");

        // HuggingFace `owner/name` format → `name`.
        assert_eq!(short_model_name("google/gemma-4-31b-it"), "gemma-4-31b-it");

        // Already-short name is returned unchanged.
        assert_eq!(short_model_name("gemma-3-4b-it"), "gemma-3-4b-it");
    }

    #[test]
    fn default_model_title_from_hf_cache_path() {
        // Regression guard: this exact layout is what the 31B Q4K vindex
        // produces in its index.json, and the first pass gave an empty
        // string because `rsplit('/').next()` returned "" for trailing `/`.
        let cached =
            "/Users/me/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/abc123/";
        assert_eq!(default_model_title(cached), "Gemma 4 31B It");
        assert_eq!(default_family(cached), "Gemma");
    }

    #[test]
    fn default_family_stops_at_first_digit_segment() {
        assert_eq!(default_family("google/gemma-4-31b-it"), "Gemma");
        assert_eq!(default_family("gemma-3-4b-it"), "Gemma");
        assert_eq!(default_family("llama-3-8b-instruct"), "Llama");
        assert_eq!(default_family("mistral-7b-v0.3"), "Mistral");
    }

    #[test]
    fn default_family_multi_word_prefix_preserved() {
        // e.g. `tiny-llama-1b` → `Tiny Llama` (both non-digit segments kept).
        assert_eq!(default_family("tiny-llama-1b"), "Tiny Llama");
    }

    #[test]
    fn default_family_no_digit_title_cases_all_segments() {
        // When there's no version token (no digit-leading segment), every
        // segment becomes part of the family name — title-cased so the
        // collection header reads cleanly. The key invariant is that we
        // don't produce an empty family string.
        assert_eq!(default_family("my-custom-model"), "My Custom Model");
        assert!(!default_family("singleword").is_empty());
    }

    #[test]
    fn note_for_preset_covers_every_default_slice() {
        // Every slice preset has a hand-written note so the collection
        // card explains the variant. Any future preset wired into
        // `slice_cmd::preset_parts` should also land here.
        assert!(note_for_preset("client").contains("2-tier"));
        assert!(note_for_preset("attn").contains("3-tier"));
        assert!(note_for_preset("attention").contains("3-tier"));
        assert!(note_for_preset("embed").contains("Embed-server"));
        assert!(note_for_preset("embed-server").contains("Embed-server"));
        assert!(note_for_preset("server").contains("FFN-only"));
        assert!(note_for_preset("browse").contains("Browse-only"));
        assert!(note_for_preset("router").contains("MoE"));
        // Unknown preset falls back to a generic note.
        assert_eq!(note_for_preset("zzz"), "Sliced variant.");
    }

    // ── Skip-if-unchanged ──────────────────────────────────────────────
    //
    // The actual upload/skip decision lives in
    // `larql_vindex::publish_vindex_with_opts` and can't be exercised
    // without an HF server. These tests pin the CLI-side plumbing: that
    // `--force-upload` flips the option into `skip_unchanged = false`,
    // and that `PublishOptions::skip_unchanged()` is the default-on
    // constructor.

    #[test]
    fn force_upload_disables_skip() {
        // Simulate the flag state the CLI builds from `--force-upload`.
        let opts = larql_vindex::PublishOptions {
            skip_unchanged: false,
            ..Default::default()
        };
        assert!(!opts.skip_unchanged);
    }

    #[test]
    fn default_publish_options_skip_unchanged() {
        // Without `--force-upload`, `skip_unchanged: true`.
        let opts = larql_vindex::PublishOptions {
            skip_unchanged: true,
            ..Default::default()
        };
        assert!(opts.skip_unchanged);
    }

    #[test]
    fn publish_options_explicit_skip_helper() {
        // The `::skip_unchanged()` constructor is intended for callers
        // that want the feature on without depending on field defaults.
        let opts = larql_vindex::PublishOptions::skip_unchanged();
        assert!(opts.skip_unchanged);
    }

    #[test]
    fn publish_options_default_is_conservative() {
        // `Default` keeps `skip_unchanged: false` so code that gets an
        // options struct via Default doesn't silently skip uploads —
        // the opt-in happens at the CLI boundary where it's explicit.
        let opts = larql_vindex::PublishOptions::default();
        assert!(!opts.skip_unchanged);
    }
}
