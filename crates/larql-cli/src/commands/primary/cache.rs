//! Shared cache scan for the primary verbs.
//!
//! `run`, `show`, `rm`, `list`, and `link` all need to look at two cache
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
//!
//! A snapshot / directory counts as a cached vindex iff it contains
//! `index.json`. Same invariant `larql_vindex::resolve_hf_vindex` uses.
//!
//! Resolution order for a user-supplied `<model>` string is:
//!
//! 1. Starts with `hf://` → [`larql_vindex::resolve_hf_vindex`] (hits the
//!    network only if not already cached).
//! 2. Existing local directory path → use as-is.
//! 3. Contains `/` (e.g. `chrishayuk/gemma-3-4b-it-vindex`) → check the
//!    HF cache first; fall back to `resolve_hf_vindex` (download).
//! 4. Plain name (no slash) → search **both** caches for a unique match
//!    on the entry name. Local entries match on their full name; HF
//!    entries match on the `name` half of `owner/name`. Ambiguous
//!    shorthands error out and list candidates.

use larql_vindex::format::filenames::*;
use std::path::{Path, PathBuf};

/// Which cache an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    /// `~/.cache/huggingface/hub/{models,datasets}--<owner>--<name>/`
    HuggingFace,
    /// `~/.cache/larql/local/<name>.vindex/`
    Local,
}

impl CacheSource {
    pub fn label(self) -> &'static str {
        match self {
            CacheSource::HuggingFace => "hf",
            CacheSource::Local => "local",
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

/// Return the LARQL local cache root (`~/.cache/larql/local/` by default,
/// honoring `LARQL_HOME` which should point at a dir that will hold the
/// `local/` subdir).
pub fn larql_local_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(h) = std::env::var("LARQL_HOME") {
        return Ok(PathBuf::from(h).join("local"));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "no HOME env var".to_string())?;
    Ok(PathBuf::from(home).join(".cache/larql/local"))
}

/// Scan both caches for every cached vindex. Sorted by `(source, repo)`
/// — local entries come first, then HF entries alphabetically within
/// each group.
pub fn scan_cached_vindexes() -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    let hub = hf_hub_dir()?;
    let local = larql_local_dir()?;
    scan_cached_vindexes_at_both(&hub, &local)
}

/// Testable core: scan both a hub-shaped dir and a local-shaped dir and
/// return the merged list.
pub fn scan_cached_vindexes_at_both(
    hub: &Path,
    local: &Path,
) -> Result<Vec<CachedVindex>, Box<dyn std::error::Error>> {
    let mut out = scan_hf_hub_at(hub)?;
    out.extend(scan_local_at(local)?);
    out.sort_by(|a, b| (a.source as u8, a.repo.as_str()).cmp(&(b.source as u8, b.repo.as_str())));
    Ok(out)
}

/// Prefixes hf-hub uses for the per-repo cache dir. Order doesn't matter
/// — the two prefixes are mutually exclusive on a single dir name.
const HF_REPO_PREFIXES: &[&str] = &["models--", "datasets--"];

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
        let repo = rest.replacen("--", "/", 1);
        let snapshots = entry.path().join("snapshots");
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
                source: CacheSource::HuggingFace,
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
        // Strip trailing `.vindex` if present — the shorthand shouldn't
        // include the suffix. Fall back to the raw name otherwise.
        let repo = entry_name
            .strip_suffix(".vindex")
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
             Try `larql pull hf://owner/{name}` (HF cache) or \
             `larql link <path>` (local cache), or pass the full \
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
    if model.starts_with("hf://") {
        return Ok(larql_vindex::resolve_hf_vindex(model)?);
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
        return Ok(larql_vindex::resolve_hf_vindex(&format!("hf://{model}"))?);
    }

    // 4. Plain name — look up by cache shorthand.
    resolve_shorthand(model)
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
    let key = model.strip_prefix("hf://").unwrap_or(model);

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

pub fn dir_size_bytes(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size_bytes(&entry.path()).unwrap_or(0);
        }
    }
    Ok(total)
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake HF hub layout under `root` with the given
    /// `owner/name` entries. Each entry gets one snapshot containing
    /// `index.json` plus a small `stub.bin` so size calculations have
    /// something to report. Uses the `models--` prefix to match the
    /// `larql publish` default; for dataset-prefix coverage see
    /// [`build_fake_hub_with_prefix`].
    fn build_fake_hub(root: &Path, repos: &[&str]) {
        build_fake_hub_with_prefix(root, "models--", repos);
    }

    fn build_fake_hub_with_prefix(root: &Path, prefix: &str, repos: &[&str]) {
        for repo in repos {
            let (owner, name) = repo.split_once('/').expect("owner/name");
            let dir = root.join(format!("{prefix}{owner}--{name}/snapshots/abc123"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
            std::fs::write(dir.join("stub.bin"), vec![0u8; 1024]).unwrap();
        }
    }

    /// Build a fake local cache under `root` with the given bare names.
    /// Each is a `<name>.vindex/` directory with an `index.json`.
    fn build_fake_local(root: &Path, names: &[&str]) {
        for name in names {
            let dir = root.join(format!("{name}.vindex"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
            std::fs::write(dir.join("stub.bin"), vec![0u8; 512]).unwrap();
        }
    }

    // ── HF-only scan (legacy single-cache shape) ────────────────────

    #[test]
    fn scan_returns_empty_for_missing_dir() {
        let root = std::path::PathBuf::from("/definitely/not/a/path");
        let out = scan_hf_hub_at(&root).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn scan_finds_cached_vindexes_and_sorts() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["zebra/last", "acme/first", "beta/mid"]);
        let out = scan_hf_hub_at(tmp.path()).unwrap();
        let repos: Vec<_> = out.iter().map(|c| c.repo.as_str()).collect();
        assert_eq!(repos, vec!["acme/first", "beta/mid", "zebra/last"]);
        assert!(out.iter().all(|c| c.source == CacheSource::HuggingFace));
    }

    #[test]
    fn scan_skips_snapshots_without_index_json() {
        let tmp = tempfile::tempdir().unwrap();
        let bare = tmp.path().join("models--foo--bar/snapshots/deadbeef");
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(bare.join("not-a-vindex.txt"), b"hi").unwrap();
        let out = scan_hf_hub_at(tmp.path()).unwrap();
        assert!(
            out.is_empty(),
            "snapshot without index.json should be skipped"
        );
    }

    #[test]
    fn scan_finds_both_models_and_datasets_prefixes() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub_with_prefix(tmp.path(), "models--", &["chrishayuk/gemma-3-4b-it-vindex"]);
        build_fake_hub_with_prefix(tmp.path(), "datasets--", &["legacy/older-vindex"]);
        let out = scan_hf_hub_at(tmp.path()).unwrap();
        let repos: Vec<_> = out.iter().map(|c| c.repo.as_str()).collect();
        assert_eq!(
            repos,
            vec!["chrishayuk/gemma-3-4b-it-vindex", "legacy/older-vindex"]
        );
        assert!(out.iter().all(|c| c.source == CacheSource::HuggingFace));
    }

    #[test]
    fn scan_ignores_non_repo_hub_entries() {
        // hf-hub also drops `.locks/`, `blobs/`, and `version.txt` into
        // the hub root. None of these are repo dirs and must be skipped
        // (they have no `--` and no `snapshots/`).
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".locks")).unwrap();
        std::fs::create_dir_all(tmp.path().join("blobs")).unwrap();
        std::fs::write(tmp.path().join("version.txt"), b"1\n").unwrap();
        let out = scan_hf_hub_at(tmp.path()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn scan_records_nonzero_size_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["o/one"]);
        let out = scan_hf_hub_at(tmp.path()).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].size_bytes >= 1024);
    }

    // ── Local cache scan ────────────────────────────────────────────

    #[test]
    fn scan_local_empty_when_dir_missing() {
        let out = scan_local_at(Path::new("/does/not/exist")).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn scan_local_finds_bare_name_entries() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_local(tmp.path(), &["gemma3-4b-f16", "v10c-tinystories"]);
        let out = scan_local_at(tmp.path()).unwrap();
        assert_eq!(out.len(), 2);
        // sort_by repo; alphabetical → gemma first, v10c second
        assert_eq!(out[0].repo, "gemma3-4b-f16");
        assert_eq!(out[1].repo, "v10c-tinystories");
        assert!(out.iter().all(|c| c.source == CacheSource::Local));
    }

    #[test]
    fn scan_local_skips_non_vindex_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        // no index.json
        std::fs::create_dir_all(tmp.path().join("junk.vindex")).unwrap();
        std::fs::write(tmp.path().join("loose-file.txt"), b"nope").unwrap();
        let out = scan_local_at(tmp.path()).unwrap();
        assert!(out.is_empty());
    }

    /// Unix-only — Windows directory symlinks require admin or
    /// developer-mode and aren't available on the default
    /// `windows-latest` runner. The "resolves symlinks" behaviour
    /// itself is unix-specific (`std::os::unix::fs::symlink`); on
    /// Windows the scan path falls back to walking the directory
    /// normally, which is covered by `scan_local_finds_bare_name_entries`.
    #[cfg(unix)]
    #[test]
    fn scan_local_resolves_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let local = tmp.path().join("local");
        let target = tmp.path().join("src/my-model.vindex");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join(INDEX_JSON), b"{}").unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::os::unix::fs::symlink(&target, local.join("my-model.vindex")).unwrap();
        let out = scan_local_at(&local).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].repo, "my-model");
    }

    // ── Merged scan ─────────────────────────────────────────────────

    #[test]
    fn scan_both_merges_and_orders_local_first() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let local = tmp.path().join("local");
        std::fs::create_dir_all(&hub).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        build_fake_hub(&hub, &["chrishayuk/gemma-3-4b-it-vindex"]);
        build_fake_local(&local, &["gemma4-31b-f16"]);
        let out = scan_cached_vindexes_at_both(&hub, &local).unwrap();
        assert_eq!(out.len(), 2);
        // Local first.
        assert_eq!(out[0].source, CacheSource::HuggingFace);
        assert_eq!(out[1].source, CacheSource::Local);
    }

    // ── Shorthand resolution ────────────────────────────────────────

    #[test]
    fn shorthand_unique_match_returns_snapshot_path() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["alice/cool-vindex"]);
        let cache = scan_hf_hub_at(tmp.path()).unwrap();
        let path = resolve_shorthand_from("cool-vindex", &cache).unwrap();
        assert!(path.ends_with("snapshots/abc123"));
    }

    #[test]
    fn shorthand_matches_bare_local_name() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_local(tmp.path(), &["gemma4-31b-f16"]);
        let cache = scan_local_at(tmp.path()).unwrap();
        let path = resolve_shorthand_from("gemma4-31b-f16", &cache).unwrap();
        assert!(path.ends_with("gemma4-31b-f16.vindex"));
    }

    #[test]
    fn shorthand_no_match_mentions_both_registration_paths() {
        let cache: Vec<CachedVindex> = Vec::new();
        let err = resolve_shorthand_from("missing", &cache).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("no cached vindex matches `missing`"));
        assert!(s.contains("larql pull"));
        assert!(s.contains("larql link"));
    }

    #[test]
    fn shorthand_ambiguous_across_hf_and_local_errors_with_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let local = tmp.path().join("local");
        std::fs::create_dir_all(&hub).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        build_fake_hub(&hub, &["someone/gemma4-31b"]);
        build_fake_local(&local, &["gemma4-31b"]);
        let cache = scan_cached_vindexes_at_both(&hub, &local).unwrap();
        let err = resolve_shorthand_from("gemma4-31b", &cache).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("ambiguous"));
        assert!(s.contains("[hf]"));
        assert!(s.contains("[local]"));
    }

    // ── resolve_cached ──────────────────────────────────────────────

    #[test]
    fn resolve_cached_accepts_owner_slash_name() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["alice/x", "bob/y"]);
        let cache = scan_hf_hub_at(tmp.path()).unwrap();
        let hit = resolve_cached_from("alice/x", &cache).unwrap();
        assert_eq!(hit.repo, "alice/x");
        assert_eq!(hit.source, CacheSource::HuggingFace);
    }

    #[test]
    fn resolve_cached_strips_hf_scheme() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["alice/x"]);
        let cache = scan_hf_hub_at(tmp.path()).unwrap();
        let hit = resolve_cached_from("hf://alice/x", &cache).unwrap();
        assert_eq!(hit.repo, "alice/x");
    }

    #[test]
    fn resolve_cached_rejects_uncached_owner_slash_name() {
        let cache: Vec<CachedVindex> = Vec::new();
        let err = resolve_cached_from("not/here", &cache).unwrap_err();
        assert!(err.to_string().contains("not cached: not/here"));
    }

    #[test]
    fn resolve_cached_accepts_hf_shorthand() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_hub(tmp.path(), &["alice/unique-name"]);
        let cache = scan_hf_hub_at(tmp.path()).unwrap();
        let hit = resolve_cached_from("unique-name", &cache).unwrap();
        assert_eq!(hit.repo, "alice/unique-name");
    }

    #[test]
    fn resolve_cached_accepts_local_shorthand() {
        let tmp = tempfile::tempdir().unwrap();
        build_fake_local(tmp.path(), &["my-extract"]);
        let cache = scan_local_at(tmp.path()).unwrap();
        let hit = resolve_cached_from("my-extract", &cache).unwrap();
        assert_eq!(hit.repo, "my-extract");
        assert_eq!(hit.source, CacheSource::Local);
    }

    #[test]
    fn shorthand_key_strips_owner_slash() {
        assert_eq!(shorthand_key("owner/name"), "name");
        assert_eq!(shorthand_key("just-name"), "just-name");
        assert_eq!(shorthand_key("a/b/c"), "c"); // rsplit_once — last segment wins
    }
}
