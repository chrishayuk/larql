//! Opening and reading store files without trusting what they are or how long
//! they say they are.

use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;

use super::store::StoreLoadError;
use super::store_limits::StoreLimits;

/// An open handle that `fstat` says is a regular file, and its length.
#[derive(Debug)]
pub(super) struct RegularFile {
    pub(super) file: File,
    pub(super) len: u64,
    pub(super) path: String,
}

/// Open `path` and refuse anything but a regular file, judging the handle that
/// will be read so a swap between check and open cannot get past it.
///
/// On Unix, `O_NONBLOCK` makes opening a FIFO with no writer return at once, and
/// it does not affect reads of regular files; `O_NOCTTY` stops a terminal device
/// becoming the controlling terminal. Symlinks are followed, so one to a regular
/// file still loads and one to a device or FIFO is refused here.
///
/// Elsewhere there is no such flag, so the path is also checked before opening,
/// which leaves a race, and the handle is checked after. On Windows the open can
/// block on some device paths, and `is_file` is true for anything not a directory
/// or symlink, so a device can pass. Length limits still bound what is read from
/// it, not how long a read takes. None of this has been run on Windows locally.
pub(super) fn open_regular(path: &Path) -> Result<RegularFile, StoreLoadError> {
    let display = path.display().to_string();
    let io_err = |source| StoreLoadError::Io {
        path: display.clone(),
        source,
    };
    #[cfg(not(unix))]
    {
        let metadata = std::fs::metadata(path).map_err(io_err)?;
        if !metadata.is_file() {
            return Err(StoreLoadError::NotRegularFile {
                path: display,
                file_type: format!("{:?}", metadata.file_type()),
            });
        }
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
    }
    let file = options.open(path).map_err(io_err)?;
    let metadata = file.metadata().map_err(io_err)?;
    if !metadata.is_file() {
        return Err(StoreLoadError::NotRegularFile {
            path: display,
            file_type: format!("{:?}", metadata.file_type()),
        });
    }
    let len = metadata.len();
    Ok(RegularFile {
        file,
        len,
        path: display,
    })
}

/// Reserve exactly `additional` more elements, refusing with a typed error where
/// `Vec::with_capacity` would abort. For a size known up front.
pub(super) fn try_reserve_for<T>(
    vec: &mut Vec<T>,
    additional: usize,
    path: &dyn Display,
) -> Result<(), StoreLoadError> {
    vec.try_reserve_exact(additional)
        .map_err(|_| allocation_refused::<T>(additional, path))
}

/// Make room to push one more, with the amortised growth a plain `push` has.
pub(super) fn try_reserve_push<T>(
    vec: &mut Vec<T>,
    path: &dyn Display,
) -> Result<(), StoreLoadError> {
    vec.try_reserve(1)
        .map_err(|_| allocation_refused::<T>(vec.len().saturating_add(1), path))
}

fn allocation_refused<T>(elements: usize, path: &dyn Display) -> StoreLoadError {
    StoreLoadError::AllocationRefused {
        path: path.to_string(),
        bytes: u64::try_from(elements.saturating_mul(size_of::<T>())).unwrap_or(u64::MAX),
    }
}

/// Read exactly `len` bytes from the reader's current position. A file that has
/// shrunk since it was measured reads short and is refused.
pub(super) fn read_exactly(
    reader: impl Read,
    len: u64,
    path: &dyn Display,
) -> Result<Vec<u8>, StoreLoadError> {
    let capacity = usize::try_from(len).map_err(|_| StoreLoadError::AllocationRefused {
        path: path.to_string(),
        bytes: len,
    })?;
    let mut buf = Vec::new();
    try_reserve_for(&mut buf, capacity, path)?;
    let read = reader
        .take(len)
        .read_to_end(&mut buf)
        .map_err(|source| StoreLoadError::Io {
            path: path.to_string(),
            source,
        })?;
    if read != capacity {
        return Err(StoreLoadError::MemberLengthMismatch {
            path: path.to_string(),
            declared: len,
            read: read as u64, // Widening: usize is at most 64 bits on supported targets.
        });
    }
    Ok(buf)
}

/// One load's limits and the bytes charged against them.
#[derive(Debug)]
pub(super) struct StoreReader<'a> {
    limits: &'a StoreLimits,
    charged: u64,
}

impl<'a> StoreReader<'a> {
    pub(super) fn new(limits: &'a StoreLimits) -> Self {
        Self { limits, charged: 0 }
    }

    pub(super) fn limits(&self) -> &StoreLimits {
        self.limits
    }

    /// Open `path` as a regular file within `limit`, charging its whole logical
    /// length to the store's total. Nothing has been read yet when this refuses.
    pub(super) fn open(&mut self, path: &Path, limit: u64) -> Result<RegularFile, StoreLoadError> {
        let file = open_regular(path)?;
        if file.len > limit {
            return Err(StoreLoadError::FileTooLarge {
                path: file.path,
                len: file.len,
                limit,
            });
        }
        let charged = self.charged.saturating_add(file.len);
        if charged > self.limits.max_store_bytes {
            return Err(StoreLoadError::StoreTooLarge {
                path: file.path,
                total: charged,
                limit: self.limits.max_store_bytes,
            });
        }
        self.charged = charged;
        Ok(file)
    }

    pub(super) fn read_file(&mut self, path: &Path, limit: u64) -> Result<Vec<u8>, StoreLoadError> {
        let RegularFile { file, len, path } = self.open(path, limit)?;
        read_exactly(file, len, &path.as_str())
    }
}
