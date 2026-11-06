//! Shared cache scan for the primary verbs.
//!
//! `run`, `show`, `rm`, `list`, and `link` all need to look at three cache
//! locations and ask "is this vindex here?".
//!
//! 1. **HuggingFace hub cache** — `~/.cache/huggingface/hub/`, populated
//!    by `larql pull` (and by `hf-hub` transitively). Layout:
//!    ```
//!    {models,datasets}--<owner>--<name>/snapshots/<sha>/{index.json,…}
//!    ```
//!    `larql publish` defaults to model repos; older vindexes and a few
//!    docs examples live as dataset repos. The scan accepts either.
//! 2. **LARQL local cache** — `~/.cache/larql/local/`, populated by
//!    `larql link <path>`. Each entry is a symlink (or directory) named
//!    `<name>.vindex/` containing the usual vindex files. Owner-less;
//!    this is where locally-extracted vindexes live after registration.
//! 3. **LARQL encoded cache** — `~/.cache/larql/encoded/`, populated when
//!    `run`/`chat`/`serve`/`show`/`pull` meet an `hf://` repo that is a
//!    raw Hugging Face checkpoint rather than a vindex: LARQL encodes it
//!    to VINDEX3 once and keeps it here. Same layout as the hub cache
//!    (`models--<owner>--<name>/snapshots/<commit>/`, plus `refs/<rev>`),
//!    so entries list and match exactly like HF entries. See
//!    [`super::hf_auto`].
//!
//! A snapshot / directory counts as a cached vindex iff it contains
//! `index.json`. Same invariant `larql_vindex::resolve_hf_vindex` uses.
//!
//! Resolution order for a user-supplied `<model>` string is:
//!
//! 1. Starts with `hf://` → [`super::hf_auto::resolve_hf_model`]: an
//!    already-encoded checkpoint is answered from the encoded cache with
//!    no network; a published vindex (`index.json` in the repo) goes to
//!    [`larql_vindex::resolve_hf_vindex`] exactly as before; a raw
//!    checkpoint (`config.json`, no `index.json`) is encoded once.
//! 2. Existing local directory path → use as-is.
//! 3. Contains `/` (e.g. `chrishayuk/gemma-3-4b-it-vindex`) → check the
//!    caches first; fall back to step 1 on `hf://<owner>/<name>`.
//! 4. Plain name (no slash) → search **all** caches for a unique match
//!    on the entry name. Local entries match on their full name; HF and
//!    encoded entries match on the `name` half of `owner/name` (so
//!    `hf://Qwen/Qwen3-0.6B` is afterwards runnable as `Qwen3-0.6B`).
//!    Ambiguous shorthands error out and list candidates.

use larql_vindex::format::filenames::*;
use std::path::{Path, PathBuf};

use super::hf_auto::HF_SCHEME;

/// Suffix every entry in the LARQL local cache carries
/// (`~/.cache/larql/local/<name>.vindex`), whatever generation the linked
/// container is. It is the registry's own naming, not the container's: the
/// shorthand is the entry name with exactly this suffix removed.
pub const LOCAL_LINK_SUFFIX: &str = ".vindex";

/// Which cache an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    /// `~/.cache/huggingface/hub/{models,datasets}--<owner>--<name>/`
    HuggingFace,
    /// `~/.cache/larql/local/<name>.vindex/`
    Local,
    /// `~/.cache/larql/encoded/models--<owner>--<name>/` — a Hugging Face
    /// checkpoint LARQL encoded to VINDEX3 on first use.
    Encoded,
}

impl CacheSource {
    pub fn label(self) -> &'static str {
        match self {
            CacheSource::HuggingFace => "hf",
            CacheSource::Local => "local",
            CacheSource::Encoded => "encoded",
        }
    }
}

/// A vindex that already exists on disk (HF cache or local registry).
#[derive(Debug, Clone)]
pub struct CachedVindex {
    /// For HF entries: `owner/name`. For local entries: just `name`
    /// (owner-less). Shorthand matching collapses both to the trailing
    /// segment.
    pub repo: String,
    /// Directory you actually load from. For HF this is the newest
    /// snapshot; for local it is the entry directory (or what the
    /// symlink resolves to).
    pub snapshot: PathBuf,
    /// Total byte size on disk.
    pub size_bytes: u64,
    /// Which cache produced this entry.
    pub source: CacheSource,
}

/// Return the HF hub cache root (`~/.cache/huggingface/hub/` by default,
/// honoring `HF_HOME`).
pub fn hf_hub_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(h) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(h).join("hub"));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "no HOME env var".to_string())?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Env var naming the LARQL cache root (the dir holding `local/`,
/// `encoded/` and `hf-headers/`).
pub const LARQL_HOME_ENV: &str = "LARQL_HOME";

/// LARQL cache root under `$HOME` when [`LARQL_HOME_ENV`] is unset.
const DEFAULT_LARQL_HOME: &str = ".cache/larql";

/// Subdir of the LARQL cache root holding `larql link` registrations.
const LOCAL_CACHE_SUBDIR: &str = "local";

/// Return the LARQL cache root (`~/.cache/larql/` by default, honoring
/// `LARQL_HOME`). The same root `larql_vindex`'s header staging uses.
pub fn larql_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(h) = std::env::var(LARQL_HOME_ENV) {
        return Ok(PathBuf::from(h));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "no HOME env var".to_string())?;
    Ok(PathBuf::from(home).join(DEFAULT_LARQL_HOME))
}

/// Return the LARQL local cache root (`~/.cache/larql/local/` by default,
/// honoring `LARQL_HOME` which should point at a dir that will hold the
/// `local/` subdir).
pub fn larql_local_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(larql_home()?.join(LOCAL_CACHE_SUBDIR))
}

/// Scan every cache for every cached vindex. Sorted by `(source, repo)`
/// — HF entries first, then local, then encoded, alphabetically within
/// each group.
pub fn scan_cached_vindexes() -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    let hub = hf_hub_dir()?;
    let local = larql_local_dir()?;
    let encoded = super::hf_auto::EncodedStore::open_default()?;
    scan_cached_vindexes_at_all(&hub, &local, encoded.root())
}

/// Testable core of [`scan_cached_vindexes`]: all three caches, merged.
pub fn scan_cached_vindexes_at_all(
    hub: &Path,
    local: &Path,
    encoded: &Path,
) -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    let mut out = scan_hf_hub_at(hub)?;
    out.extend(scan_local_at(local)?);
    out.extend(scan_hub_layout_at(encoded, CacheSource::Encoded)?);
    sort_by_source_then_repo(&mut out);
    Ok(out)
}

fn sort_by_source_then_repo(entries: &mut [CachedVindex]) {
    entries
        .sort_by(|a, b| (a.source as u8, a.repo.as_str()).cmp(&(b.source as u8, b.repo.as_str())));
}

/// Prefixes hf-hub uses for the per-repo cache dir. Order doesn't matter
/// — the two prefixes are mutually exclusive on a single dir name.
pub(crate) const HF_REPO_PREFIXES: &[&str] = &[HF_MODEL_REPO_PREFIX, HF_DATASET_REPO_PREFIX];

/// Hub-layout directory prefix of a model repo.
pub(crate) const HF_MODEL_REPO_PREFIX: &str = "models--";

/// Hub-layout directory prefix of a dataset repo.
const HF_DATASET_REPO_PREFIX: &str = "datasets--";

/// Separator standing in for `/` in a hub-layout repo directory name.
pub(crate) const HUB_REPO_SEP: &str = "--";

/// Per-repo subdir holding one directory per revision.
pub(crate) const HUB_SNAPSHOTS_DIR: &str = "snapshots";

/// Strip whichever HF repo prefix matches and return the remainder
/// (`<owner>--<name>`), or `None` if `name` is something else under the
/// hub root (e.g. `.locks`, `blobs`).
fn strip_hf_repo_prefix(name: &str) -> Option<&str> {
    HF_REPO_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(*prefix))
}

/// Scan the HuggingFace hub cache only.
pub fn scan_hf_hub_at(hub: &Path) -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    scan_hub_layout_at(hub, CacheSource::HuggingFace)
}

/// Scan any directory laid out like the HF hub cache
/// (`{models,datasets}--<owner>--<name>/snapshots/<rev>/index.json`),
/// labelling every entry with `source`. The encoded cache shares the
/// layout precisely so this scan serves it too.
pub fn scan_hub_layout_at(
    hub: &Path,
    source: CacheSource,
) -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    if !hub.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(hub)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        // `larql publish` defaults to model repos, so `models--` is the
        // common case; `datasets--` covers older vindexes that pre-date
        // the publish default flip. Either prefix is a valid HF cache
        // dir, and `hf_hub` itself supports both.
        let Some(rest) = strip_hf_repo_prefix(&name) else {
            continue;
        };
        let repo = rest.replacen(HUB_REPO_SEP, "/", 1);
        let snapshots = entry.path().join(HUB_SNAPSHOTS_DIR);
        if !snapshots.is_dir() {
            continue;
        }
        // Pick the most recently modified snapshot that has an index.json.
        let latest = std::fs::read_dir(&snapshots)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join(INDEX_JSON).exists())
            .max_by_key(|e| {
                e.metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            });
        if let Some(snap) = latest {
            let path = snap.path();
            let size_bytes = dir_size_bytes(&path).unwrap_or(0);
            out.push(CachedVindex {
                repo,
                snapshot: path,
                size_bytes,
                source,
            });
        }
    }
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    Ok(out)
}

/// Scan the LARQL local cache only. Each entry is a directory (or
/// symlink to one) under `local/` whose name ends in `.vindex` and which
/// contains an `index.json`.
pub fn scan_local_at(local: &Path) -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    if !local.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(local)? {
        let entry = entry?;
        let path = entry.path();
        // Resolve symlinks so `metadata()` sees the target, but keep the
        // symlink path as the canonical one so `rm` can unlink cleanly.
        let target_is_dir = std::fs::metadata(&path)
            .map(|m| m.is_dir())
            .unwrap_or(false);
        if !target_is_dir {
            continue;
        }
        if !path.join(INDEX_JSON).exists() {
            continue;
        }
        let entry_name = entry.file_name().to_string_lossy().to_string();
        // Strip the registry's link suffix if present — the shorthand
        // shouldn't include it. Fall back to the raw name otherwise.
        let repo = entry_name
            .strip_suffix(LOCAL_LINK_SUFFIX)
            .unwrap_or(&entry_name)
            .to_string();
        let size_bytes = dir_size_bytes(&path).unwrap_or(0);
        out.push(CachedVindex {
            repo,
            snapshot: path,
            size_bytes,
            source: CacheSource::Local,
        });
    }
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    Ok(out)
}

/// The last segment of a cache entry's name — what shorthand matches on.
/// HF entries (`owner/name`) expose the `name` half; local entries
/// (`name` only) expose themselves.
fn shorthand_key(repo: &str) -> &str {
    match repo.rsplit_once('/') {
        Some((_, n)) => n,
        None => repo,
    }
}

/// Testable core: match a plain shorthand name against a pre-scanned list
/// of cached vindexes. Returns `Ok(path)` on unique match,
/// `Err(reason)` on zero or multiple matches.
pub fn resolve_shorthand_from(
    name: &str,
    cache: &[CachedVindex],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let matches: Vec<_> = cache
        .iter()
        .filter(|c| shorthand_key(&c.repo) == name)
        .collect();
    match matches.as_slice() {
        [hit] => Ok(hit.snapshot.clone()),
        [] => Err(format!(
            "no cached vindex matches `{name}`.\n\
             Try `larql run hf://owner/{name}` (a Hugging Face checkpoint \
             is encoded on first use), `larql pull hf://owner/{name}` \
             (HF cache) or `larql link <path>` (local cache), or pass the full \
             `owner/name` / path explicitly."
        )
        .into()),
        multiple => {
            let candidates = multiple
                .iter()
                .map(|c| format!("  - {} [{}]", c.repo, c.source.label()))
                .collect::<Vec<_>>()
                .join("\n");
            Err(format!(
                "shorthand `{name}` is ambiguous — matches multiple cached \
                 vindexes:\n{candidates}\nUse the full `owner/name` to disambiguate."
            )
            .into())
        }
    }
}

/// Resolve a user-supplied `<model>` string to a local vindex directory.
///
/// See the module docstring for the precedence order. Plain-name lookups
/// that match multiple cached repos return an error listing the matches so
/// the user can pick one.
pub fn resolve_model(model: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // 1. hf:// URI — defer to the vindex crate. Downloads if not cached.
    if model.starts_with(HF_SCHEME) {
        return resolve_hf(model);
    }

    // 2. Already a local directory.
    let direct = PathBuf::from(model);
    if direct.is_dir() {
        return Ok(direct);
    }

    // 3. Contains `/` — treat as `owner/name`. Step 2 already absorbed
    //    actual local paths that exist, so anything landing here is
    //    either a cached repo name or a hub repo we should download.
    //    (On Unix MAIN_SEPARATOR is `/`, so we can't distinguish a
    //    non-existent local path from a hub repo — err on the HF side.)
    if model.contains('/') {
        let cache = scan_cached_vindexes().unwrap_or_default();
        if let Some(hit) = cache.iter().find(|c| c.repo == model) {
            return Ok(hit.snapshot.clone());
        }
        return resolve_hf(&format!("{HF_SCHEME}{model}"));
    }

    // 4. Plain name — look up by cache shorthand.
    resolve_shorthand(model)
}

/// An `hf://` spec: encoded cache, published vindex, or a first-run
/// encode — [`super::hf_auto`] decides which.
fn resolve_hf(spec: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let resolution = super::hf_auto::resolve_hf_model(spec, &mut |s| {
        larql_vindex::resolve_hf_vindex(s).map_err(Into::into)
    })?;
    Ok(resolution.into_path())
}

/// Match a plain name against the cache. The match is on the `name` half
/// of `owner/name`, e.g. `gemma-3-4b-it-vindex` matches
/// `chrishayuk/gemma-3-4b-it-vindex`.
pub fn resolve_shorthand(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache = scan_cached_vindexes()?;
    resolve_shorthand_from(name, &cache)
}

/// Resolve a user-supplied string to a single `CachedVindex` entry —
/// never touches the network. Used by `rm` where we explicitly don't want
/// to download something in order to delete it.
pub fn resolve_cached(model: &str) -> Result<CachedVindex, Box<dyn std::error::Error>> {
    let cache = scan_cached_vindexes()?;
    resolve_cached_from(model, &cache)
}

/// Testable core of [`resolve_cached`].
pub fn resolve_cached_from(
    model: &str,
    cache: &[CachedVindex],
) -> Result<CachedVindex, Box<dyn std::error::Error>> {
    let key = model.strip_prefix(HF_SCHEME).unwrap_or(model);

    // Full owner/name match (HF entries only have this form).
    if key.contains('/') {
        if let Some(hit) = cache.iter().find(|c| c.repo == key) {
            return Ok(hit.clone());
        }
        return Err(format!("not cached: {key}").into());
    }

    // Shorthand match — hits local entries by name, HF entries by name half.
    let matches: Vec<_> = cache
        .iter()
        .filter(|c| shorthand_key(&c.repo) == key)
        .collect();
    match matches.as_slice() {
        [hit] => Ok((*hit).clone()),
        [] => Err(format!("not cached: {key}").into()),
        multiple => {
            let candidates = multiple
                .iter()
                .map(|c| format!("  - {} [{}]", c.repo, c.source.label()))
                .collect::<Vec<_>>()
                .join("\n");
            Err(format!("shorthand `{key}` is ambiguous — matches:\n{candidates}").into())
        }
    }
}

/// Total bytes of the files under `path`.
///
/// A file reached through a symlink counts at its target's size: an HF hub
/// snapshot is nothing *but* symlinks into `../../blobs/`, so counting the
/// links themselves reported every pulled vindex as 0 bytes. Directory
/// symlinks are not followed — a link cycle must not hang `larql list` —
/// but real subdirectories are walked.
pub fn dir_size_bytes(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            total += dir_size_bytes(&entry.path()).unwrap_or(0);
        } else if kind.is_file() || kind.is_symlink() {
            // `fs::metadata` follows the link; a dangling link or one that
            // names a directory contributes nothing.
            total += std::fs::metadata(entry.path())
                .ok()
                .filter(|m| m.is_file())
                .map_or(0, |m| m.len());
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
