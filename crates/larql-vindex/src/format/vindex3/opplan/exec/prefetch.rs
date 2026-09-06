//! Bringing a mapped bank's selected experts into memory ahead of the
//! projection loop — the ACCESS realization of the same lossless bytes.
//!
//! V3 measured the cold Kimi-Linear token at 1.30 s: 0.976 GB through
//! 59,670 major faults of one 16 KiB page each, ≈22 µs per fault, taken
//! one at a time in row order by the loop. Nothing here changes what is
//! read or stored; only the shape of the request — whether the kernel is
//! told what is coming, or the pages are faulted concurrently before the
//! loop needs them.

use super::realization::MappedAccess;

/// One selected expert's matrix, as a range of the mapping: its first
/// byte's address and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub address: usize,
    pub bytes: usize,
}

impl Range {
    pub fn of<T>(slice: &[T]) -> Self {
        Self {
            address: slice.as_ptr() as usize,
            bytes: std::mem::size_of_val(slice),
        }
    }
}

/// What a prefetch did, for the reconciliation beside it: how many
/// ranges it covered and how many bytes those spanned once page-aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefetchReport {
    pub ranges: usize,
    pub bytes: usize,
}

/// The page the OS faults by, read once.
fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf reads a process-independent constant.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size > 0 {
            return size as usize;
        }
    }
    // Every platform this runs on pages by at least 4 KiB; a platform
    // that will not say pages by that.
    4096
}

/// `range` rounded out to whole pages.
fn aligned(range: Range, page: usize) -> Range {
    let start = range.address / page * page;
    let end = (range.address + range.bytes).div_ceil(page) * page;
    Range {
        address: start,
        bytes: end - start,
    }
}

/// Bring `ranges` in under `access`, ahead of whoever reads them.
/// `parallelism` bounds the touch arm's concurrency; Demand does nothing
/// and reports nothing, which is the point of naming it.
pub fn prefetch(access: MappedAccess, ranges: &[Range], parallelism: usize) -> PrefetchReport {
    if access == MappedAccess::Demand || ranges.is_empty() {
        return PrefetchReport::default();
    }
    let page = page_size();
    let mut aligned_ranges: Vec<Range> = ranges.iter().map(|r| aligned(*r, page)).collect();
    // In address order, so the request follows the file's physical layout.
    aligned_ranges.sort_by_key(|r| r.address);
    let bytes = aligned_ranges.iter().map(|r| r.bytes).sum();
    match access {
        MappedAccess::Demand => unreachable!("returned above"),
        MappedAccess::Advise => advise(&aligned_ranges),
        MappedAccess::Touch => touch(&aligned_ranges, page, parallelism.max(1)),
    }
    PrefetchReport {
        ranges: aligned_ranges.len(),
        bytes,
    }
}

#[cfg(unix)]
fn advise(ranges: &[Range]) {
    for r in ranges {
        // SAFETY: the range is page-aligned and lies inside a mapping this
        // process owns for the life of the prepared image; WILLNEED is a
        // hint and cannot change the mapping's contents or protection.
        unsafe {
            libc::madvise(r.address as *mut libc::c_void, r.bytes, libc::MADV_WILLNEED);
        }
    }
}

#[cfg(not(unix))]
fn advise(_ranges: &[Range]) {}

/// Touch one byte per page of every range, concurrently over up to
/// `parallelism` threads, each thread taking a contiguous share of the
/// address-ordered ranges so the device sees sequential runs.
fn touch(ranges: &[Range], page: usize, parallelism: usize) {
    let threads = parallelism.min(ranges.len()).max(1);
    let per_thread = ranges.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for share in ranges.chunks(per_thread) {
            scope.spawn(move || {
                for r in share {
                    let mut at = r.address;
                    let end = r.address + r.bytes;
                    while at < end {
                        // SAFETY: `at` lies inside a mapping this process
                        // owns for the life of the prepared image; a
                        // volatile read cannot be elided and reads one byte
                        // the mapping already holds.
                        unsafe {
                            std::ptr::read_volatile(at as *const u8);
                        }
                        at += page;
                    }
                }
            });
        }
    });
}
