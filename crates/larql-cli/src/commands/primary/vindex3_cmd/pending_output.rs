//! An output file that exists only once it is complete (#744).
//!
//! The teacher-forced logit dumps (`--logit-dump`, `--bank`) used to
//! `File::create` their output before the first logit existed, so a run
//! that refused or failed part-way left a zero-byte or truncated file at
//! the requested path — an artifact a pipeline checking for the file
//! rather than the exit status would read as a result (the #730 class).
//!
//! A [`PendingOutput`] writes to a hidden temporary sibling and renames it
//! onto the destination only in [`PendingOutput::commit`]; dropping it
//! uncommitted deletes the temporary. Creating one first removes whatever
//! is already at the destination, so a failed run leaves *nothing* there
//! rather than a previous run's output that would read as this one's.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Prefix of the temporary sibling, so a leftover is recognisable.
pub(super) const PARTIAL_PREFIX: &str = ".larql-partial-";

/// A buffered writer whose bytes appear at `dest` only on [`Self::commit`].
pub(super) struct PendingOutput {
    file: BufWriter<tempfile::NamedTempFile>,
    dest: PathBuf,
}

/// The directory a destination's temporary sibling lives in: its parent,
/// or the working directory for a bare file name.
pub(super) fn parent_dir(dest: &Path) -> &Path {
    match dest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// Remove `path` if it exists; any other failure is an error.
pub(super) fn remove_stale(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

impl PendingOutput {
    /// Remove any existing file at `dest` and open a temporary sibling in
    /// the same directory (so the final rename stays on one filesystem).
    pub(super) fn create(dest: &Path) -> io::Result<Self> {
        remove_stale(dest)?;
        let tmp = tempfile::Builder::new()
            .prefix(PARTIAL_PREFIX)
            .tempfile_in(parent_dir(dest))?;
        Ok(Self {
            file: BufWriter::new(tmp),
            dest: dest.to_path_buf(),
        })
    }

    /// Flush and move the bytes onto the destination.
    pub(super) fn commit(self) -> io::Result<()> {
        let tmp = self.file.into_inner().map_err(|e| e.into_error())?;
        tmp.as_file().sync_all()?;
        tmp.persist(&self.dest).map_err(|e| e.error)?;
        Ok(())
    }
}

impl Write for PendingOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
