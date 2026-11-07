//! Resolve a model argument to a model **already on this machine**.
//!
//! One identifier everywhere: every verb that reads an existing model,
//! vindex or container accepts what `larql run` accepts — a directory,
//! `hf://owner/name[@revision]`, `owner/name`, or a cache shorthand such
//! as `Qwen3-0.6B`. There are two resolvers and they differ in one thing
//! only, whether they may fetch:
//!
//! - [`super::resolve_model`] — the front door (`run`, `chat`, `serve`,
//!   `bench`, `show`, `pull`, …). A miss goes to the hub: a published
//!   vindex is downloaded, a raw checkpoint is encoded on first use.
//! - [`resolve_installed_model`] — every inspection and research verb
//!   (`larql vindex3 …`, `shannon decode-diff`). It never touches the
//!   network and never encodes; a miss is an error naming the front-door
//!   command that would fetch it. Pointing `vindex3 inspect` at a name
//!   must not start a multi-GB download as a side effect.
//!
//! Rule, first match wins:
//!
//! 1. The argument names an existing path → that path, unchanged. A
//!    directory that happens to share a cached model's name always wins,
//!    and every path that worked before keeps working byte for byte.
//! 2. Anything that can only be a filesystem path (absolute, `.`- or
//!    `~`-relative, containing a Windows separator, or not UTF-8) →
//!    unchanged, existing or not. It is never looked up as a name, and a
//!    missing one reaches the verb exactly as before, which reports it
//!    after its own argument checks.
//! 3. `hf://owner/name[@revision]` → the encoded cache at that revision;
//!    for the default revision, also any cached vindex whose repo is
//!    `owner/name`.
//! 4. `owner/name` → the cache entry with exactly that repo.
//! 5. A plain name → the unique cache entry whose shorthand it is;
//!    ambiguity lists the candidates.
//!
//! Only rules 3–5 can fail, and a miss says how to fetch the model.

use std::path::{Path, PathBuf};

use larql_vindex::format::huggingface::range::{parse_spec, DEFAULT_REVISION};

use super::super::hf_auto::{EncodedStore, HF_SCHEME};
use super::{ambiguous, lookup_cached, scan_cached_vindexes_at_all, CacheLookup, CachedVindex};

type BoxErr = Box<dyn std::error::Error>;

/// Prefix of a path relative to the working directory (`./x`, `../x`).
const RELATIVE_PATH_PREFIX: char = '.';
/// Prefix of a home-relative path the shell did not expand.
const HOME_PATH_PREFIX: char = '~';
/// Windows' path separator. No repo id or shorthand contains it, so it
/// marks a path on every host — a Windows path typed on Unix included.
const WINDOWS_SEPARATOR: char = '\\';

/// The three cache roots a name can resolve into.
#[derive(Debug, Clone)]
pub struct CacheRoots {
    /// HF hub cache (`HF_HOME/hub`).
    pub hub: PathBuf,
    /// `larql link` registrations (`LARQL_HOME/local`).
    pub local: PathBuf,
    /// Checkpoints encoded on first use (`LARQL_HOME/encoded`).
    pub encoded: PathBuf,
}

impl CacheRoots {
    /// The roots the environment names (`HF_HOME`, `LARQL_HOME`).
    pub fn from_env() -> Result<Self, BoxErr> {
        Ok(Self {
            hub: super::hf_hub_dir()?,
            local: super::larql_local_dir()?,
            encoded: EncodedStore::open_default()?.root().to_path_buf(),
        })
    }

    fn scan(&self) -> Result<Vec<CachedVindex>, BoxErr> {
        scan_cached_vindexes_at_all(&self.hub, &self.local, &self.encoded)
    }
}

/// Resolve `model` to a model already on this machine, without network.
/// See the module docs for the rule.
pub fn resolve_installed_model(model: &Path) -> Result<PathBuf, BoxErr> {
    match as_name(model) {
        None => Ok(model.to_path_buf()),
        Some(_) => resolve_installed_model_in(model, &CacheRoots::from_env()?),
    }
}

/// Testable core of [`resolve_installed_model`]: the caches are `roots`.
pub fn resolve_installed_model_in(model: &Path, roots: &CacheRoots) -> Result<PathBuf, BoxErr> {
    let Some(spec) = as_name(model) else {
        return Ok(model.to_path_buf());
    };
    if spec.starts_with(HF_SCHEME) {
        return resolve_hf_installed(spec, roots);
    }
    let cache = roots.scan()?;
    match lookup_cached(spec, &cache) {
        CacheLookup::Hit(hit) => Ok(hit.snapshot.clone()),
        CacheLookup::Miss => Err(not_installed(spec)),
        CacheLookup::Ambiguous(matches) => Err(ambiguous(spec, &matches)),
    }
}

/// `hf://owner/name[@revision]` from disk: the encoded cache answers any
/// revision it recorded; a cached vindex answers only the default one,
/// because its snapshot carries no revision to match against.
fn resolve_hf_installed(spec: &str, roots: &CacheRoots) -> Result<PathBuf, BoxErr> {
    let parsed = parse_spec(spec)?;
    if let Some(hit) = EncodedStore::new(&roots.encoded).lookup(&parsed.repo, &parsed.revision) {
        return Ok(hit);
    }
    if parsed.revision != DEFAULT_REVISION {
        return Err(not_installed(spec));
    }
    let cache = roots.scan()?;
    cache
        .iter()
        .find(|c| c.repo == parsed.repo)
        .map(|hit| hit.snapshot.clone())
        .ok_or_else(|| not_installed(spec))
}

/// `model` as a name to look up, or `None` when it is a path to pass
/// through: one that exists, or one that can only be a path (rules 1–2).
fn as_name(model: &Path) -> Option<&str> {
    if model.exists() {
        return None;
    }
    // A name is UTF-8 by construction; anything else can only be a path.
    let spec = model.to_str()?;
    (spec.starts_with(HF_SCHEME) || !looks_like_path(spec)).then_some(spec)
}

/// Whether `spec` can only be a filesystem path, never a model name.
fn looks_like_path(spec: &str) -> bool {
    Path::new(spec).is_absolute()
        || spec.starts_with(RELATIVE_PATH_PREFIX)
        || spec.starts_with(HOME_PATH_PREFIX)
        || spec.contains(WINDOWS_SEPARATOR)
}

/// The miss: what was looked for and how to fetch it.
fn not_installed(spec: &str) -> BoxErr {
    let repo = spec.strip_prefix(HF_SCHEME).unwrap_or(spec);
    let fetch = if repo.contains('/') {
        format!("{HF_SCHEME}{repo}")
    } else {
        format!("{HF_SCHEME}<owner>/{repo}")
    };
    format!(
        "`{spec}` is not a directory or a model already on this machine.\n\
         This command never downloads. Fetch the model first:\n  \
         larql run {fetch}   — a Hugging Face checkpoint, encoded on first use\n  \
         larql pull {fetch}  — a published vindex\n  \
         larql link <dir>  — a container you built\n\
         `larql list` shows what is already here."
    )
    .into()
}

#[cfg(test)]
mod tests;
