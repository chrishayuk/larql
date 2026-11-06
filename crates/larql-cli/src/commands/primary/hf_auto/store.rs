//! The encoded cache: where a Hugging Face checkpoint lives once LARQL
//! has encoded it to VINDEX3.
//!
//! ```text
//! <LARQL_HOME>/encoded/                       (~/.cache/larql/encoded/)
//!   models--<owner>--<name>/
//!     refs/<revision>        commit the revision resolved to at encode
//!     snapshots/<commit>/    a complete VINDEX3 container
//!     .partial-<pid>-<n>/    an encode in flight — never read as an entry
//! ```
//!
//! The layout is the HF hub cache's own, so `cache::scan_hub_layout_at`
//! lists it and `larql rm` evicts it with no special case.
//!
//! **Atomicity.** An encode writes into a `.partial-*` directory beside
//! `snapshots/` and is renamed into `snapshots/<commit>/` only once it
//! succeeded. A snapshot therefore exists only whole: an interrupted or
//! refused encode leaves at most a partial directory, which nothing scans,
//! which [`PartialEncode`]'s drop removes on an error, and which the next
//! [`EncodedStore::begin`] sweeps once its owning process is gone (a
//! killed process runs no destructor).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use larql_vindex::format::filenames::INDEX_JSON;

use super::super::cache::{self, HF_MODEL_REPO_PREFIX, HUB_REPO_SEP, HUB_SNAPSHOTS_DIR};

/// Subdir of the LARQL cache root holding encoded checkpoints.
pub const ENCODED_CACHE_SUBDIR: &str = "encoded";

/// Where revision names are pinned to the commit they resolved to.
pub const REFS_DIR: &str = "refs";

/// Prefix of an in-flight encode directory. Leading `.` keeps it out of
/// any listing that skips hidden entries, and it is never under
/// `snapshots/`, so no scan can mistake it for an entry.
pub const PARTIAL_PREFIX: &str = ".partial-";

/// Separates the owning pid from the uniquifier in a partial's name.
const PARTIAL_FIELD_SEP: char = '-';

/// Distinguishes two partials one process opens for the same repo.
static PARTIAL_SEQ: AtomicU64 = AtomicU64::new(0);

type BoxErr = Box<dyn std::error::Error>;

/// The encoded cache rooted at one directory.
#[derive(Debug, Clone)]
pub struct EncodedStore {
    root: PathBuf,
}

impl EncodedStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store under the LARQL cache root (`LARQL_HOME` honoured).
    pub fn open_default() -> Result<Self, BoxErr> {
        Ok(Self::new(cache::larql_home()?.join(ENCODED_CACHE_SUBDIR)))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `models--<owner>--<name>/` for `owner/name`. Refuses a repo id that
    /// would escape the store or collapse two repos onto one directory.
    pub fn repo_dir(&self, repo: &str) -> Result<PathBuf, BoxErr> {
        let (owner, name) = repo
            .split_once('/')
            .filter(|(owner, name)| is_safe_component(owner) && is_safe_component(name))
            .ok_or_else(|| format!("`{repo}` is not an `owner/name` Hugging Face repo id"))?;
        Ok(self
            .root
            .join(format!("{HF_MODEL_REPO_PREFIX}{owner}{HUB_REPO_SEP}{name}")))
    }

    /// Where the container for `repo` at `commit` lives (whether or not
    /// it exists yet).
    pub fn snapshot_dir(&self, repo: &str, commit: &str) -> Result<PathBuf, BoxErr> {
        if !is_safe_component(commit) {
            return Err(format!("`{commit}` is not a usable commit id").into());
        }
        Ok(self.repo_dir(repo)?.join(HUB_SNAPSHOTS_DIR).join(commit))
    }

    /// The complete container for `repo` at `commit`, if one exists.
    pub fn complete(&self, repo: &str, commit: &str) -> Option<PathBuf> {
        let dir = self.snapshot_dir(repo, commit).ok()?;
        dir.join(INDEX_JSON).is_file().then_some(dir)
    }

    /// Answer `repo@revision` from disk alone: `revision` names a commit
    /// already encoded, or a ref recorded when one was.
    pub fn lookup(&self, repo: &str, revision: &str) -> Option<PathBuf> {
        if let Some(hit) = self.complete(repo, revision) {
            return Some(hit);
        }
        let commit = self.read_ref(repo, revision)?;
        self.complete(repo, &commit)
    }

    /// The commit `revision` resolved to when `repo` was last encoded.
    fn read_ref(&self, repo: &str, revision: &str) -> Option<String> {
        let path = self.ref_path(repo, revision)?;
        let text = std::fs::read_to_string(path).ok()?;
        let commit = text.trim();
        is_safe_component(commit).then(|| commit.to_string())
    }

    fn ref_path(&self, repo: &str, revision: &str) -> Option<PathBuf> {
        if !is_safe_component(revision) {
            return None;
        }
        Some(self.repo_dir(repo).ok()?.join(REFS_DIR).join(revision))
    }

    /// Pin `revision` to `commit`, so the next lookup needs no network.
    ///
    /// A revision that cannot be a file name (`refs/pr/1`) is simply not
    /// recorded: the lookup then goes to the hub, which is correct, just
    /// not offline.
    pub fn record_ref(&self, repo: &str, revision: &str, commit: &str) -> Result<(), BoxErr> {
        let Some(path) = self.ref_path(repo, revision) else {
            return Ok(());
        };
        if revision == commit {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, commit)?;
        Ok(())
    }

    /// Open a fresh partial directory for encoding `repo`, first sweeping
    /// partials whose owning process has exited.
    pub fn begin(&self, repo: &str) -> Result<PartialEncode, BoxErr> {
        let repo_dir = self.repo_dir(repo)?;
        std::fs::create_dir_all(&repo_dir)?;
        sweep_orphaned_partials(&repo_dir)?;
        let pid = std::process::id();
        let seq = PARTIAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = repo_dir.join(format!("{PARTIAL_PREFIX}{pid}{PARTIAL_FIELD_SEP}{seq}"));
        // The encoder creates its output itself; a leftover of the same
        // name (a reused pid) is removed so it starts empty.
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(PartialEncode {
            dir,
            committed: false,
        })
    }
}

/// An encode in flight. Removed on drop unless [`Self::commit`] ran.
#[derive(Debug)]
pub struct PartialEncode {
    dir: PathBuf,
    committed: bool,
}

impl PartialEncode {
    /// Where the encoder writes.
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Publish the finished container at `dest` by rename.
    ///
    /// If `dest` already holds a complete container — a concurrent run
    /// finished first — that one is kept and this one discarded: both
    /// were encoded from the same commit by the same encoder.
    pub fn commit(mut self, dest: &Path) -> Result<PathBuf, BoxErr> {
        if !self.dir.join(INDEX_JSON).is_file() {
            return Err(format!(
                "encode produced no {INDEX_JSON} in {}; nothing cached",
                self.dir.display()
            )
            .into());
        }
        if dest.join(INDEX_JSON).is_file() {
            return Ok(dest.to_path_buf());
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&self.dir, dest)?;
        self.committed = true;
        Ok(dest.to_path_buf())
    }
}

impl Drop for PartialEncode {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort: a destructor cannot report, and the sweep in
            // `begin` catches whatever this misses.
            let _ = std::fs::remove_dir_all(&self.dir);
            // And the repo directory, if this partial was all it held — a
            // refused first run leaves no trace. `remove_dir` refuses a
            // non-empty directory, so a sibling entry is never touched.
            if let Some(repo_dir) = self.dir.parent() {
                let _ = std::fs::remove_dir(repo_dir);
            }
        }
    }
}

/// Remove partials in `repo_dir` whose owning process no longer runs.
/// A partial whose name carries no pid is left alone: it is not ours.
fn sweep_orphaned_partials(repo_dir: &Path) -> Result<(), BoxErr> {
    for entry in std::fs::read_dir(repo_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = partial_owner(&name) else {
            continue;
        };
        if pid != std::process::id() && !process_alive(pid) {
            std::fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

/// The pid that owns a partial directory, from its name.
pub fn partial_owner(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(PARTIAL_PREFIX)?;
    let (pid, _) = rest.split_once(PARTIAL_FIELD_SEP)?;
    pid.parse().ok()
}

/// Whether a process with this pid exists.
#[cfg(unix)]
pub fn process_alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 delivers nothing; `kill` only reports whether
    // `pid` names a process this user could signal. No memory is touched.
    let rc = unsafe { libc::kill(pid, 0) };
    // EPERM means it exists under another user — alive, not ours to sweep.
    rc == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Without a liveness probe, every partial is presumed live and left for
/// `larql rm` — never sweep what might still be writing.
#[cfg(not(unix))]
pub fn process_alive(_pid: u32) -> bool {
    true
}

/// A single path component that cannot escape or alias: non-empty, no
/// separators, not `.`-led (which also excludes `.` and `..`).
pub fn is_safe_component(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('.') && !s.contains(['/', '\\']) && !s.contains(HUB_REPO_SEP)
}
