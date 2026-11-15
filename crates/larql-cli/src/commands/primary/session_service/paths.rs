//! Every path a request names stays inside the policy's workdir.
//!
//! A request names a file relative to the workdir, or absolutely under it. The name must pass the container
//! path rule (`larql_vindex::format::vindex3::container_path::check`: ordinary components only, no root, `.`,
//! `..`, backslash or NUL) after the workdir prefix is removed.
//!
//! The file is then opened one component at a time with `openat`, starting from a descriptor of the workdir
//! taken at start, with `O_NOFOLLOW` on every component. So no symlink is followed anywhere on the way, in
//! either direction, and there is no window between a check and the open in which the agent, who can write to
//! the workdir, could swap a directory for a link. The file opened must be a regular file with one link: a hard
//! link to a file elsewhere is refused. This is stricter than the container rule, which follows symlinks for
//! readers (the HF cache is made of them): a workdir is written by the party the service is protecting the
//! model from.
//!
//! The service never hands these paths to LARQL. A patch is read here and copied into the service's private
//! staging directory, and LARQL reads the copy; a saved patch is written by LARQL into staging and copied out
//! here, into a file opened as above.

use std::ffi::{CString, OsStr};
use std::fs::File;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};

use larql_vindex::format::vindex3::container_path;

use super::protocol::Failure;

/// Permissions of a directory or file the service creates in the workdir (before the umask).
const NEW_DIR_MODE: libc::mode_t = 0o755;
const NEW_FILE_MODE: libc::c_uint = 0o644;

/// The workdir: its canonical path, the path as the policy wrote it, and a descriptor of it.
pub struct Workdir {
    canonical: PathBuf,
    given: PathBuf,
    root: File,
}

/// A file opened inside the workdir for writing, and enough to remove it again.
pub struct WriteTarget {
    pub file: File,
    dir: File,
    name: CString,
    /// Whether the open created it.
    pub created: bool,
}

impl WriteTarget {
    /// Remove the file this open created. A file that existed before is left alone.
    pub fn discard(self) {
        if self.created {
            // SAFETY: `dir` is an open directory descriptor and `name` a NUL-terminated component.
            unsafe { libc::unlinkat(self.dir.as_raw_fd(), self.name.as_ptr(), 0) };
        }
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Why opening `name` failed, as a refusal.
fn open_refusal(name: &str, code: i32) -> Failure {
    let why = match code {
        libc::ELOOP => "a symlink is on its path, and the service follows none".to_string(),
        libc::ENOTDIR => "a component on its path is a symlink or not a directory".to_string(),
        libc::ENOENT => "it does not exist".to_string(),
        libc::EACCES | libc::EPERM => "permission denied".to_string(),
        _ => std::io::Error::from_raw_os_error(code).to_string(),
    };
    Failure::refused(format!("`{name}` in the workdir: {why}"))
}

impl Workdir {
    /// Open the workdir. It must exist and be a directory.
    pub fn open(given: &Path) -> Result<Self, String> {
        let canonical = given
            .canonicalize()
            .map_err(|e| format!("the workdir {}: {e}", given.display()))?;
        if !canonical.is_dir() {
            return Err(format!(
                "the workdir {} is not a directory",
                given.display()
            ));
        }
        let root = File::open(&canonical)
            .map_err(|e| format!("open the workdir {}: {e}", canonical.display()))?;
        Ok(Self {
            canonical,
            given: given.to_path_buf(),
            root,
        })
    }

    pub fn canonical(&self) -> &Path {
        &self.canonical
    }

    /// The workdir-relative name of `path`, refused unless it stays inside the workdir.
    pub fn name(&self, path: &str) -> Result<String, Failure> {
        if path.is_empty() {
            return Err(Failure::refused("the path is empty"));
        }
        let p = Path::new(path);
        let relative = if p.is_absolute() {
            [&self.canonical, &self.given]
                .iter()
                .find_map(|root| p.strip_prefix(root).ok())
                .ok_or_else(|| {
                    Failure::refused(format!(
                        "`{path}` is outside the workdir {}",
                        self.canonical.display()
                    ))
                })?
                .to_path_buf()
        } else {
            p.to_path_buf()
        };
        let relative = relative
            .to_str()
            .ok_or_else(|| Failure::refused(format!("`{path}` is not valid UTF-8")))?;
        if relative.is_empty() {
            return Err(Failure::refused(format!(
                "`{path}` names the workdir itself"
            )));
        }
        container_path::check(relative)
            .map_err(|e| Failure::refused(format!("`{path}` is refused: {e}")))?;
        Ok(relative.to_string())
    }

    /// Open every directory on the way to `name` (a checked workdir-relative name), creating missing ones
    /// when `create`, and give the last directory and the file's own component.
    fn walk(&self, name: &str, create: bool) -> Result<(File, CString), Failure> {
        let parts: Vec<&OsStr> = Path::new(name)
            .components()
            .map(|c| match c {
                Component::Normal(part) => Ok(part),
                _ => Err(Failure::refused(format!(
                    "`{name}` is not a plain relative name"
                ))),
            })
            .collect::<Result<_, _>>()?;
        let (file_part, dirs) = parts
            .split_last()
            .ok_or_else(|| Failure::refused("the path is empty"))?;
        let mut dir = self.root.try_clone().map_err(Failure::failed)?;
        for part in dirs {
            let c = cstring(part)?;
            let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
            // SAFETY: `dir` is an open directory descriptor and `c` NUL-terminated.
            let mut fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), flags) };
            if fd < 0 && errno() == libc::ENOENT && create {
                // SAFETY: as above.
                if unsafe { libc::mkdirat(dir.as_raw_fd(), c.as_ptr(), NEW_DIR_MODE) } < 0
                    && errno() != libc::EEXIST
                {
                    return Err(open_refusal(name, errno()));
                }
                // SAFETY: as above.
                fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), flags) };
            }
            if fd < 0 {
                return Err(open_refusal(name, errno()));
            }
            // SAFETY: `fd` was just returned by `openat` and is owned by nothing else.
            dir = unsafe { File::from_raw_fd(fd) };
        }
        Ok((dir, cstring(file_part)?))
    }

    /// The bytes of the regular file `name` names, at most `max_bytes` of them.
    pub fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, Failure> {
        let (dir, c) = self.walk(name, false)?;
        // O_NONBLOCK: opening a FIFO the agent left there must not hang the service.
        let flags = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
        // SAFETY: `dir` is an open directory descriptor and `c` NUL-terminated.
        let fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), flags) };
        if fd < 0 {
            return Err(open_refusal(name, errno()));
        }
        // SAFETY: `fd` was just returned by `openat` and is owned by nothing else.
        let file = unsafe { File::from_raw_fd(fd) };
        single_regular_file(name, &file)?;
        let mut bytes = Vec::new();
        file.take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| Failure::failed(format!("read `{name}`: {e}")))?;
        if bytes.len() as u64 > max_bytes {
            return Err(Failure::refused(format!(
                "`{name}` is larger than the policy's max_patch_bytes ({max_bytes})"
            )));
        }
        Ok(bytes)
    }

    /// Open `name` for writing, creating it (and its directories) if it does not exist. Nothing is
    /// truncated: the caller writes only once it knows what to write.
    pub fn open_for_write(&self, name: &str) -> Result<WriteTarget, Failure> {
        let (dir, c) = self.walk(name, true)?;
        let base = libc::O_WRONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
        // SAFETY: `dir` is an open directory descriptor and `c` NUL-terminated.
        let mut fd = unsafe {
            libc::openat(
                dir.as_raw_fd(),
                c.as_ptr(),
                base | libc::O_CREAT | libc::O_EXCL,
                NEW_FILE_MODE,
            )
        };
        let mut created = true;
        if fd < 0 && errno() == libc::EEXIST {
            created = false;
            // SAFETY: as above.
            fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), base | libc::O_NONBLOCK) };
        }
        if fd < 0 {
            return Err(open_refusal(name, errno()));
        }
        // SAFETY: `fd` was just returned by `openat` and is owned by nothing else.
        let file = unsafe { File::from_raw_fd(fd) };
        single_regular_file(name, &file)?;
        Ok(WriteTarget {
            file,
            dir,
            name: c,
            created,
        })
    }
}

fn cstring(part: &OsStr) -> Result<CString, Failure> {
    CString::new(part.as_bytes()).map_err(|_| Failure::refused("the path holds a NUL byte"))
}

/// Refuse anything but a regular file with exactly one link.
fn single_regular_file(name: &str, file: &File) -> Result<(), Failure> {
    use std::os::unix::fs::MetadataExt;
    let meta = file
        .metadata()
        .map_err(|e| Failure::failed(format!("stat `{name}`: {e}")))?;
    if !meta.file_type().is_file() {
        return Err(Failure::refused(format!("`{name}` is not a regular file")));
    }
    if meta.nlink() != 1 {
        return Err(Failure::refused(format!(
            "`{name}` has {} hard links; the service reads and writes only files with one",
            meta.nlink()
        )));
    }
    Ok(())
}
