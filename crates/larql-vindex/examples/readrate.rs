//! Cold sparse-read rate of a volume, by the SAME method the expert-bank
//! residency probe uses: mmap, readahead off, msync(MS_INVALIDATE) to
//! evict, then fault in a strided selection of pages and count major
//! faults. Comparable to the expert measurement by construction.
// POSIX-only, like `vindex3_residency_probe` beside it: the measurement
// IS the page-fault behaviour, so there is no portable shape for it.
#[cfg(unix)]
mod probe {
    use std::time::Instant;
    pub fn run() {
        let a: Vec<String> = std::env::args().collect();
        let path = &a[1];
        let want_mib: usize = a.get(2).map_or(192, |v| v.parse().unwrap());
        let file = std::fs::File::open(path).unwrap();
        let map = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        unsafe {
            libc::madvise(map.as_ptr() as *mut _, map.len(), libc::MADV_RANDOM);
            libc::msync(map.as_ptr() as *mut _, map.len(), libc::MS_INVALIDATE);
            libc::madvise(map.as_ptr() as *mut _, map.len(), libc::MADV_DONTNEED);
        }
        let pages = (want_mib * 1024 * 1024) / page;
        // Contiguous run, like an expert's own extent.
        let f0 = faults();
        let t0 = Instant::now();
        let mut acc = 0u64;
        for p in 0..pages.min(map.len() / page) {
            acc += map[p * page] as u64;
        }
        let dt = t0.elapsed();
        let f1 = faults();
        let mib = (f1 - f0) as f64 * page as f64 / (1024.0 * 1024.0);
        println!(
            "{path}\n  {:.1} MiB in {:.3} s = {:.0} MiB/s   ({} major faults, checksum {acc})",
            mib,
            dt.as_secs_f64(),
            mib / dt.as_secs_f64(),
            f1 - f0
        );
    }
    fn faults() -> i64 {
        let mut u: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
        u.ru_majflt
    }
}

#[cfg(unix)]
fn main() {
    probe::run()
}

#[cfg(not(unix))]
fn main() {
    // madvise / msync / mincore / getrusage are POSIX. Windows would need
    // QueryWorkingSetEx.
    eprintln!("readrate is unix-only.");
}
