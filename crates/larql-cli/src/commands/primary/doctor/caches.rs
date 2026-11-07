//! The caches the quick start writes to — the HF hub cache (`larql pull`)
//! and the encoded-model cache under `LARQL_HOME` (`larql run hf://…`) —
//! their sizes, whether they can be written, and the free disk under them.

use std::path::{Path, PathBuf};

use super::report::{Check, Group};
use super::requirements::{QUICKSTART_DOWNLOAD_BYTES, QUICKSTART_FREE_DISK_BYTES};
use super::units::{display_path, format_bytes};

/// What one cache directory looks like on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirFacts {
    pub path: PathBuf,
    pub exists: bool,
    /// Whether the directory (or, before it exists, the nearest existing
    /// ancestor it would be created under) accepts a new file.
    pub writable: bool,
    /// Bytes stored under it, when it exists.
    pub size_bytes: Option<u64>,
    /// Bytes free on its volume.
    pub free_bytes: Option<u64>,
}

/// A cache location, and the variable that moved it there, if one did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheDir {
    pub facts: Result<DirFacts, String>,
    pub env_override: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheFacts {
    pub hub: CacheDir,
    pub encoded: CacheDir,
    pub home: Option<PathBuf>,
}

/// The nearest ancestor of `path` (itself included) that exists.
pub(crate) fn nearest_existing(path: &Path) -> Option<&Path> {
    path.ancestors().find(|p| p.exists())
}

/// Whether a file can be created in `dir`. The probe file is anonymous
/// and gone when this returns.
pub(crate) fn writable(dir: &Path) -> bool {
    dir.is_dir() && tempfile::tempfile_in(dir).is_ok()
}

/// Bytes in the regular files under `path`. Symlinks are not followed:
/// an HF hub snapshot is symlinks into `blobs/`, so following them would
/// count every downloaded file twice.
pub(crate) fn physical_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            total += physical_size(&entry.path()).unwrap_or(0);
        } else if kind.is_file() {
            total += entry.metadata().map_or(0, |m| m.len());
        }
    }
    Ok(total)
}

/// Look at one cache directory, through `free` for the volume's space.
pub(crate) fn probe_dir(path: &Path, free: impl Fn(&Path) -> Option<u64>) -> DirFacts {
    let exists = path.is_dir();
    let base = nearest_existing(path);
    DirFacts {
        path: path.to_path_buf(),
        exists,
        writable: base.is_some_and(writable),
        size_bytes: exists.then(|| physical_size(path).ok()).flatten(),
        free_bytes: base.and_then(free),
    }
}

pub(crate) fn checks(c: &CacheFacts) -> Vec<Check> {
    let home = c.home.as_deref();
    let mut out = vec![
        location(&c.hub, "HF hub cache", "HF_HOME", false, home),
        location(&c.encoded, "Encoded models", "LARQL_HOME", true, home),
    ];
    out.extend(free_disk(c, home));
    out
}

/// Where a cache lives, how big it is, and whether it can be written.
/// `required`: an unwritable cache stops the quick start (the encoded
/// cache, which `larql run hf://…` writes) rather than only `larql pull`.
fn location(dir: &CacheDir, name: &str, env: &str, required: bool, home: Option<&Path>) -> Check {
    let facts = match &dir.facts {
        Ok(f) => f,
        Err(why) => {
            return Check::fail(
                Group::Caches,
                name,
                format!("no location: {why}"),
                format!("set HOME, or set {env} to a writable directory"),
            )
        }
    };
    let from = dir
        .env_override
        .map(|var| format!(", from ${var}"))
        .unwrap_or_default();
    let contents = match (facts.exists, facts.size_bytes) {
        (false, _) => "not created yet".to_string(),
        (true, Some(bytes)) => format_bytes(bytes),
        (true, None) => "size unknown".to_string(),
    };
    let summary = format!("{} ({contents}{from})", display_path(&facts.path, home));
    if facts.writable {
        return Check::ok(Group::Caches, name, summary);
    }
    let hint = format!("fix its permissions, or set {env} to a writable directory");
    if required {
        Check::fail(
            Group::Caches,
            name,
            format!("{summary} is not writable"),
            hint,
        )
    } else {
        Check::warn(
            Group::Caches,
            name,
            format!("{summary} is not writable"),
            hint,
        )
    }
}

/// Free space where the quick start writes: required on the encoded
/// cache's volume; on the hub cache's volume, reported only when it is short,
/// since only `larql pull` writes there.
fn free_disk(c: &CacheFacts, home: Option<&Path>) -> Vec<Check> {
    const NAME: &str = "Free disk";
    let need = format_bytes(QUICKSTART_FREE_DISK_BYTES);
    let Ok(encoded) = &c.encoded.facts else {
        return Vec::new();
    };
    let shown = display_path(&encoded.path, home);
    let mut out = vec![match encoded.free_bytes {
        None => Check::warn(
            Group::Caches,
            NAME,
            format!("could not read free space under {shown}"),
            format!("the quick start needs about {need} free there"),
        ),
        Some(free) if free < QUICKSTART_FREE_DISK_BYTES => Check::fail(
            Group::Caches,
            NAME,
            format!(
                "{} free under {shown}; the quick start needs about {need}",
                format_bytes(free)
            ),
            "free some space, or set LARQL_HOME to a directory on a larger volume",
        ),
        Some(free) => Check::ok(
            Group::Caches,
            NAME,
            format!(
                "{} free under {shown} (quick start needs {need})",
                format_bytes(free)
            ),
        ),
    }];
    if let Ok(hub) = &c.hub.facts {
        if let Some(free) = hub.free_bytes.filter(|f| *f < QUICKSTART_DOWNLOAD_BYTES) {
            out.push(Check::warn(
                Group::Caches,
                "Free disk (hub)",
                format!(
                    "{} free under {}",
                    format_bytes(free),
                    display_path(&hub.path, home)
                ),
                format!(
                    "`larql pull` downloads there ({} for the quick start model); \
                     free some space or set HF_HOME",
                    format_bytes(QUICKSTART_DOWNLOAD_BYTES)
                ),
            ));
        }
    }
    out
}
