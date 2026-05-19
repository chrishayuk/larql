//! Optimized mmap helpers for vindex file loading.
//!
//! Two access patterns:
//! - `mmap_optimized`: MADV_SEQUENTIAL + MADV_WILLNEED — for files that must
//!   be fully resident (embeddings, norms, attn weights). Prefaults pages at
//!   load time so the first query doesn't stall on page faults.
//! - `mmap_demand_paged`: MADV_RANDOM — for large sparse files (gate vectors,
//!   feature payloads). Pages fault in only when accessed, keeping RSS low at
//!   load time. Gate KNN touches all pages during a linear scan but only a
//!   logarithmic subset when HNSW is active.

/// Create an mmap with SEQUENTIAL + WILLNEED hints — prefaults all pages.
///
/// Use for files that will be read fully on every forward pass (embeddings,
/// norms, attention weights). Not suitable for large sparse files where only
/// a fraction of pages are touched per token.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive.
pub unsafe fn mmap_optimized(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    advise_sequential(&mmap);
    Ok(mmap)
}

/// Create an mmap with RANDOM hint — no prefaulting, demand-paged only.
///
/// Use for large sparse files (gate_vectors.bin, interleaved_kquant.bin) where
/// RSS should reflect only the pages actually touched during inference, not
/// the full file size. Pages fault in on first access and are evictable under
/// memory pressure without any explicit unmap.
///
/// On non-unix targets the madvise hint is a no-op — the page-cache
/// policy is OS-managed there.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive.
pub unsafe fn mmap_demand_paged(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    #[cfg(unix)]
    {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        unsafe {
            libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
    }
    Ok(mmap)
}

/// Apply sequential + willneed hints to an existing mmap. Unix-only;
/// on Windows the function is a no-op (the OS handles readahead).
#[cfg_attr(not(unix), allow(unused_variables))]
pub fn advise_sequential(mmap: &memmap2::Mmap) {
    #[cfg(unix)]
    {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        unsafe {
            // Sequential: tell OS we stream linearly (enables aggressive readahead)
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
            // Willneed: prefault pages into cache (background page-in)
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }
    }
}
