//! Which of a region's pages are physically resident, as the OS reports
//! them — with the OUTCOME kept.
//!
//! A residency witness that turns a failed read into zero cannot tell a
//! cold region from one it never managed to observe, which is the one
//! distinction a cold-state admission gate exists to make. So a read
//! here either returns the resident byte count or says why it could
//! not, and nothing downstream is allowed to collapse the second into
//! the first.

/// Why a region's residency could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyReadError {
    /// `sysconf(_SC_PAGESIZE)` returned no usable page size.
    PageSizeUnavailable,
    /// `mincore` returned non-zero; `errno` is what it set.
    Mincore { errno: i32 },
    /// The target has no residency report at all (non-unix).
    Unsupported,
}

impl ResidencyReadError {
    /// The stable name an observation record carries for this failure.
    pub fn kind(&self) -> &'static str {
        match self {
            ResidencyReadError::PageSizeUnavailable => "page_size_unavailable",
            ResidencyReadError::Mincore { .. } => "mincore",
            ResidencyReadError::Unsupported => "unsupported",
        }
    }

    /// The OS error number, present only for a failed `mincore`.
    pub fn errno(&self) -> Option<i32> {
        match self {
            ResidencyReadError::Mincore { errno } => Some(*errno),
            _ => None,
        }
    }
}

impl std::fmt::Display for ResidencyReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidencyReadError::PageSizeUnavailable => write!(f, "page size unavailable"),
            ResidencyReadError::Mincore { errno } => write!(f, "mincore failed (errno {errno})"),
            ResidencyReadError::Unsupported => write!(f, "no residency report on this target"),
        }
    }
}

impl std::error::Error for ResidencyReadError {}

/// The OS page size, if it reports a usable one.
pub fn page_size() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf reads a process-independent constant.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page > 0).then_some(page as u64)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Resident bytes of `bytes`, which must lie inside a live mapping:
/// whole resident pages times the page size, clipped to the slice's
/// length. A page shared with a neighbouring region counts for both.
pub(super) fn resident_bytes_of(bytes: &[u8]) -> Result<u64, ResidencyReadError> {
    #[cfg(unix)]
    {
        if bytes.is_empty() {
            return Ok(0);
        }
        let page = page_size().ok_or(ResidencyReadError::PageSizeUnavailable)? as usize;
        let start = bytes.as_ptr() as usize;
        let first = start & !(page - 1);
        let end = start + bytes.len();
        let pages = end.div_ceil(page) - first / page;
        let mut vec = vec![0u8; pages];
        // SAFETY: `[first, first + pages*page)` covers the slice's pages
        // within the live mapping it borrows from, and `vec` holds one
        // byte per page as `mincore` requires.
        let rc = unsafe {
            libc::mincore(
                first as *mut libc::c_void,
                pages * page,
                vec.as_mut_ptr().cast(),
            )
        };
        if rc != 0 {
            // Read immediately: nothing between the call and here can
            // overwrite errno.
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(ResidencyReadError::Mincore { errno });
        }
        let resident_pages = vec.iter().filter(|b| **b & 1 == 1).count();
        Ok((resident_pages * page).min(bytes.len()) as u64)
    }
    #[cfg(not(unix))]
    {
        // No report on this target, for an empty slice as for any other:
        // the same answer `resident_bytes` has always given here.
        let _ = bytes;
        Err(ResidencyReadError::Unsupported)
    }
}
