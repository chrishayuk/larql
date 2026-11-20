//! Writing a new vindex directory beside a source one, without ever
//! damaging what is already on disk.
//!
//! Every verb that derives one vindex from another (`COMPILE … INTO`,
//! `COMPACT INTO`) hard-links untouched files out of the source and
//! promotes a finished directory into place. Both steps are destructive
//! when the destination is chosen badly: a link attempt onto the source
//! falls back to copying a file over itself, and promoting over an
//! existing directory means deleting it first.
//!
//! [`write_fresh_directory`] is the one place that decides whether a
//! destination is acceptable and how the result reaches it:
//!
//! ```text
//! check_fresh_destination(source, out)      refuse before any write
//!     → create an exclusive sibling staging directory
//!     → work(staging)                       all writes land here
//!     → sync every staged file and directory
//!     → validate(staging)                   reopen and check the result
//!     → promote: rename staging → out       the only step touching `out`
//!     → sync out's parent directory
//! ```
//!
//! The sequence and its two failure classes (before the rename: nothing
//! changed; the final sync: published, durability not confirmed) are
//! Gate B, B6 §3.6 (`docs/gate-b-b6.md`).
//!
//! What it refuses (fail closed):
//!
//! - `out` is a symbolic link — its target is not this call's to replace;
//! - `out` resolves to the source, contains it, or sits inside it, with
//!   symlinks in either path resolved first;
//! - `out` exists and is anything but an empty directory. A previously
//!   written vindex is *not* an exception: recognising an artifact is
//!   not authority to replace it.
//!
//! What it never does: recursively delete a directory it did not create.
//! The staging directory is made with an exclusive `create_dir`, so the
//! cleanup on failure only ever removes this call's own work, and
//! promotion replaces an empty `out` in a way that fails — rather than
//! deletes — if the directory gained contents after the check.

//!
//! What it guarantees, and its limits, stated so they are not mistaken for
//! more:
//!
//! - **Atomic and durable on Unix local filesystems.** Promotion is one
//!   `rename` on one filesystem, so a reader never sees a partial
//!   directory, and every staged file and directory is synced before it
//!   and the parent after it, so a successful return survives a crash.
//!   On other platforms files are synced but directories cannot be, so
//!   the rename itself is not made durable there.
//! - **Not a defence against a concurrent adversary.** The check and the
//!   promotion are separate steps. Paths are resolved through symlinks
//!   at check time; a process that swaps a parent directory for a
//!   symlink afterwards can redirect the rename. The one race that is
//!   closed is the common one: a destination that gains contents during
//!   the work makes promotion fail instead of deleting them.

#[cfg(test)]
mod tests;

use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use crate::error::VindexError;

/// Infix that marks a staging directory: `.<output name>.tmp.<pid>.<n>`.
/// Hidden, and beside the output so the promoting rename stays on one
/// filesystem.
pub const STAGING_INFIX: &str = ".tmp.";

/// How many staging names to try before giving up. Names differ by a
/// counter, so this bounds the search when earlier runs of the same
/// process id crashed and left their directories behind.
const MAX_STAGING_ATTEMPTS: u32 = 64;

/// Staging name stem when the output path has no final component.
const UNNAMED_OUTPUT: &str = "output";

const STEP_CREATE_PARENT: &str = "create the output's parent directory";
const STEP_CREATE_STAGING: &str = "create staging dir";
#[cfg(not(unix))]
const STEP_REMOVE_EMPTY_OUT: &str = "remove the empty output directory";
const STEP_PROMOTE: &str = "promote the staging dir to";
const STEP_RESOLVE: &str = "resolve";
const STEP_INSPECT: &str = "inspect";
const STEP_SYNC: &str = "sync";
const STEP_INJECTED: &str = "publish";

fn refuse(out: &Path, reason: impl Into<String>) -> VindexError {
    VindexError::UnsafeOutputDestination {
        out: out.to_path_buf(),
        reason: reason.into(),
    }
}

fn staging_error(step: &'static str, path: &Path, source: std::io::Error) -> VindexError {
    VindexError::OutputStaging {
        step,
        path: path.to_path_buf(),
        source,
    }
}

/// Resolve `path` to the location it names once symlinks are followed.
///
/// An existing path canonicalises directly. For one that does not exist
/// yet, the deepest existing ancestor is canonicalised and the missing
/// tail appended — so a not-yet-created output under a symlinked parent
/// still compares equal to the real location it would be created at.
///
/// A `..` in the missing tail is refused rather than interpreted: what
/// it means depends on directories that do not exist.
fn resolve(path: &Path) -> Result<PathBuf, VindexError> {
    let absolute = std::path::absolute(path).map_err(|e| staging_error(STEP_RESOLVE, path, e))?;

    let mut existing = absolute.as_path();
    let mut missing: Vec<OsString> = Vec::new();
    loop {
        match existing.symlink_metadata() {
            Ok(_) => break,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                let (Some(parent), Some(Component::Normal(name))) =
                    (existing.parent(), existing.components().next_back())
                else {
                    return Err(refuse(
                        path,
                        "the path uses `..` below a directory that does not exist, so where \
                         it points cannot be determined",
                    ));
                };
                missing.push(name.to_os_string());
                existing = parent;
            }
            Err(e) => return Err(staging_error(STEP_RESOLVE, existing, e)),
        }
    }

    let mut resolved = existing
        .canonicalize()
        .map_err(|e| staging_error(STEP_RESOLVE, existing, e))?;
    resolved.extend(missing.into_iter().rev());
    Ok(resolved)
}

/// `path` without trailing separators. `lstat("link/")` follows the
/// link, so a destination spelled with a trailing slash would slip past
/// the symlink refusal and be written *through*; every check and every
/// write below uses this spelling instead.
fn without_trailing_separator(path: &Path) -> PathBuf {
    path.components().collect()
}

/// Decide whether `out` may receive a directory derived from `source`.
/// Reads the filesystem; writes nothing.
pub fn check_fresh_destination(source: &Path, out: &Path) -> Result<(), VindexError> {
    let out = without_trailing_separator(out);
    let out = out.as_path();
    refuse_symlink(out)?;
    refuse_nesting(source, out)?;
    refuse_contents(out)
}

/// [`check_fresh_destination`] for a destination with no source on this
/// filesystem to protect — a model named by a hub id, say. Only the
/// destination's own state is judged: not a symlink, and absent or an
/// empty directory.
pub fn check_empty_destination(out: &Path) -> Result<(), VindexError> {
    let out = without_trailing_separator(out);
    let out = out.as_path();
    refuse_symlink(out)?;
    refuse_contents(out)
}

fn refuse_symlink(out: &Path) -> Result<(), VindexError> {
    match out.symlink_metadata() {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(refuse(
                out,
                "it is a symbolic link; name the real directory to write to",
            ));
        }
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(staging_error(STEP_INSPECT, out, e)),
    }
    Ok(())
}

fn refuse_nesting(source: &Path, out: &Path) -> Result<(), VindexError> {
    let source_real = resolve(source)?;
    let out_real = resolve(out)?;

    if out_real == source_real {
        return Err(refuse(
            out,
            "it resolves to the source vindex, which would be overwritten in place",
        ));
    }
    if source_real.starts_with(&out_real) {
        return Err(refuse(
            out,
            format!("it contains the source vindex ({})", source_real.display()),
        ));
    }
    if out_real.starts_with(&source_real) {
        return Err(refuse(
            out,
            format!("it is inside the source vindex ({})", source_real.display()),
        ));
    }

    Ok(())
}

fn refuse_contents(out: &Path) -> Result<(), VindexError> {
    match std::fs::read_dir(out) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return Err(refuse(
                    out,
                    "the directory already exists and is not empty; existing contents are \
                     never replaced — choose a new path, or remove the directory yourself \
                     first",
                ));
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(_) if out.is_file() => {
            return Err(refuse(out, "a file already exists at that path"));
        }
        Err(e) => return Err(staging_error(STEP_INSPECT, out, e)),
    }
    Ok(())
}

/// Undoes this call's filesystem footprint unless promotion disarmed
/// it: the staging directory, then any parent directories this call
/// created for it.
///
/// The staging directory was created exclusively by this call, so its
/// recursive removal can only ever delete this call's own partial work —
/// including when `work` panics. Created parents are removed
/// non-recursively, deepest first, so one that gained other contents
/// simply stays.
struct StagingGuard {
    staging: PathBuf,
    created_parents: Vec<PathBuf>,
    armed: bool,
}

impl StagingGuard {
    /// Remove the staging directory and the parents created for it, and
    /// disarm. Returns the error if the staging directory could not be
    /// removed (it is then left on disk).
    fn clean_up(&mut self) -> std::io::Result<()> {
        self.armed = false;
        let removed = match take_fault(PublishFault::CleanUp) {
            Some(fault) => Err(fault.io_error()),
            None if self.staging.as_os_str().is_empty() => Ok(()),
            None => std::fs::remove_dir_all(&self.staging),
        };
        for parent in &self.created_parents {
            let _ = std::fs::remove_dir(parent);
        }
        removed
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.clean_up();
        }
    }
}

/// The ancestors of `parent` that do not exist yet, deepest first —
/// what `create_dir_all(parent)` is about to create.
fn missing_ancestors(parent: &Path) -> Vec<PathBuf> {
    parent
        .ancestors()
        .take_while(|dir| !dir.as_os_str().is_empty() && dir.symlink_metadata().is_err())
        .map(Path::to_path_buf)
        .collect()
}

/// Create a staging directory beside `out` that did not exist before
/// this call, creating `out`'s missing parents on the way.
fn create_staging(out: &Path) -> Result<StagingGuard, VindexError> {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // Armed before the parents are created, so a failure part-way
    // through creating them takes the ones already made back down. An
    // empty `staging` names nothing, so removing it is a no-op.
    let mut guard = StagingGuard {
        staging: PathBuf::new(),
        created_parents: missing_ancestors(parent),
        armed: true,
    };
    std::fs::create_dir_all(parent).map_err(|e| staging_error(STEP_CREATE_PARENT, parent, e))?;

    let stem = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| UNNAMED_OUTPUT.to_string());
    let pid = std::process::id();

    let mut last = parent.to_path_buf();
    for attempt in 0..MAX_STAGING_ATTEMPTS {
        let candidate = parent.join(format!(".{stem}{STAGING_INFIX}{pid}.{attempt}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => {
                guard.staging = candidate;
                return Ok(guard);
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => last = candidate,
            Err(e) => return Err(staging_error(STEP_CREATE_STAGING, &candidate, e)),
        }
    }
    Err(staging_error(
        STEP_CREATE_STAGING,
        &last,
        std::io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "{MAX_STAGING_ATTEMPTS} staging directories from earlier runs are in the way; \
                 remove them"
            ),
        ),
    ))
}

/// Move the finished staging directory to `out`.
///
/// `out` was checked to be absent or an empty directory, and nothing
/// here can delete contents it gained since: on Unix `rename` replaces
/// an empty directory atomically and fails on a non-empty one; elsewhere
/// the empty directory is first removed non-recursively, which fails the
/// same way.
fn promote(staging: &Path, out: &Path) -> Result<(), VindexError> {
    #[cfg(not(unix))]
    match std::fs::remove_dir(out) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(staging_error(STEP_REMOVE_EMPTY_OUT, out, e)),
    }
    std::fs::rename(staging, out).map_err(|e| staging_error(STEP_PROMOTE, out, e))
}

/// Run `work` against a staging directory and publish the result to
/// `out`, provided `out` is a safe destination for something derived
/// from `source`. [`write_fresh_directory_validated`] with no validation
/// beyond the sync.
pub fn write_fresh_directory<T, E>(
    source: &Path,
    out: &Path,
    work: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<VindexError> + std::fmt::Display,
{
    write_fresh_directory_validated(source, out, work, |_, _| Ok(()))
}

/// Run `work` against a staging directory, then publish it to `out` by
/// the sequence of Gate B, B6 §3.6:
///
/// ```text
/// work(staging)                 write every staged file
/// sync every file               contents and the metadata to read them
/// sync every staged directory
/// validate(staging, &value)     reopen and check the staged output
/// rename staging → out          the only step touching `out`
/// sync out's parent directory
/// ```
///
/// A failure before the rename leaves `out` exactly as it was, and the
/// staging directory and any parent directories created for it are
/// removed. If that removal fails too, the error is
/// [`VindexError::StagingLeftBehind`], naming the leftover.
///
/// A failure of the final sync is different: the rename has happened, so
/// the output may already be visible. It is not rolled back. The error is
/// [`VindexError::PublishedNotDurable`].
///
/// Directory syncs need a directory handle, which only Unix offers; on
/// other platforms files are synced and directories are not.
pub fn write_fresh_directory_validated<T, E>(
    source: &Path,
    out: &Path,
    work: impl FnOnce(&Path) -> Result<T, E>,
    validate: impl FnOnce(&Path, &T) -> Result<(), E>,
) -> Result<T, E>
where
    E: From<VindexError> + std::fmt::Display,
{
    let out = without_trailing_separator(out);
    let out = out.as_path();
    check_fresh_destination(source, out)?;
    let mut guard = create_staging(out)?;

    let staged = (|| {
        inject(PublishFault::AfterStagingCreated)?;
        let value = work(&guard.staging)?;
        inject(PublishFault::AfterWork)?;
        sync_tree(&guard.staging)?;
        inject(PublishFault::Validate)?;
        validate(&guard.staging, &value)?;
        inject(PublishFault::Rename)?;
        promote(&guard.staging, out)?;
        Ok(value)
    })();

    let value = match staged {
        Ok(value) => value,
        Err(cause) => {
            return Err(match guard.clean_up() {
                Ok(()) => cause,
                Err(cleanup) => E::from(VindexError::StagingLeftBehind {
                    cause: cause.to_string(),
                    staging: guard.staging.clone(),
                    cleanup,
                }),
            });
        }
    };
    guard.armed = false;

    sync_parent(out).map_err(|source| VindexError::PublishedNotDurable {
        out: out.to_path_buf(),
        source,
    })?;
    Ok(value)
}

// ── Durability ──────────────────────────────────────────────────────────

/// Sync every file under `dir`, then every directory, deepest first, so a
/// directory is synced after the entries it lists.
fn sync_tree(dir: &Path) -> Result<(), VindexError> {
    let mut files = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    let mut next = 0;
    while next < dirs.len() {
        let current = dirs[next].clone();
        next += 1;
        let entries =
            std::fs::read_dir(&current).map_err(|e| staging_error(STEP_SYNC, &current, e))?;
        for entry in entries {
            let path = entry
                .map_err(|e| staging_error(STEP_SYNC, &current, e))?
                .path();
            let kind = path
                .symlink_metadata()
                .map_err(|e| staging_error(STEP_SYNC, &path, e))?
                .file_type();
            if kind.is_dir() {
                dirs.push(path);
            } else if kind.is_file() {
                files.push(path);
            }
        }
    }

    inject(PublishFault::SyncFile)?;
    for file in &files {
        sync_file(file).map_err(|e| staging_error(STEP_SYNC, file, e))?;
    }
    inject(PublishFault::SyncStagingDir)?;
    for dir in dirs.iter().rev() {
        sync_dir(dir).map_err(|e| staging_error(STEP_SYNC, dir, e))?;
    }
    Ok(())
}

fn sync_file(path: &Path) -> std::io::Result<()> {
    // Windows flushes a file only through a handle with write access;
    // opening one changes nothing in the file.
    #[cfg(not(unix))]
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    #[cfg(unix)]
    let file = std::fs::File::open(path)?;
    file.sync_all()
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Sync the directory that now lists `out`, making the rename durable.
fn sync_parent(out: &Path) -> std::io::Result<()> {
    take_fault(PublishFault::SyncParent).map_or(Ok(()), |fault| Err(fault.io_error()))?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    sync_dir(parent)
}

// ── Fault injection ─────────────────────────────────────────────────────

/// A step of publication at which a test can make it fail (Gate B, B6
/// §4). Arm one with [`arm_publish_fault`]; the next publication on this
/// thread fails there, once. With none armed, publication runs exactly
/// its normal steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishFault {
    /// The staging directory exists; nothing is written into it yet.
    AfterStagingCreated,
    /// `work` has written its files into staging.
    AfterWork,
    /// Syncing the staged files.
    SyncFile,
    /// Syncing the staged directories.
    SyncStagingDir,
    /// Validating the staged output.
    Validate,
    /// Renaming staging to the destination.
    Rename,
    /// Removing the staging directory after a failure. Arm it with another
    /// point to make that failure's cleanup fail too.
    CleanUp,
    /// Syncing the destination's parent after the rename.
    SyncParent,
}

impl PublishFault {
    fn io_error(self) -> std::io::Error {
        std::io::Error::other(format!("injected publish fault: {self:?}"))
    }
}

thread_local! {
    static ARMED: std::cell::RefCell<Vec<PublishFault>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Arm `faults` for the next publication on this thread; each fires once.
/// An empty slice disarms every fault.
pub fn arm_publish_fault(faults: &[PublishFault]) {
    ARMED.with(|armed| *armed.borrow_mut() = faults.to_vec());
}

/// Remove and return `point` if it is armed.
fn take_fault(point: PublishFault) -> Option<PublishFault> {
    ARMED.with(|armed| {
        let mut armed = armed.borrow_mut();
        let at = armed.iter().position(|f| *f == point)?;
        Some(armed.remove(at))
    })
}

fn inject(point: PublishFault) -> Result<(), VindexError> {
    match take_fault(point) {
        Some(fault) => Err(staging_error(
            STEP_INJECTED,
            Path::new(""),
            fault.io_error(),
        )),
        None => Ok(()),
    }
}
