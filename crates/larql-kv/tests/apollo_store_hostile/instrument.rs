//! Per-thread allocation accounting and refusal, and a watchdog.
//!
//! Thread-local so that tests running in parallel cannot see each other, because
//! a process-global counter here would race. The state is const-initialised with
//! no destructor, so reading it cannot itself allocate.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ops::RangeInclusive;
use std::sync::mpsc;
use std::time::{Duration, Instant};

thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
    /// Inclusive byte range to refuse; start above end refuses nothing.
    static REFUSE_FROM: Cell<usize> = const { Cell::new(usize::MAX) };
    static REFUSE_TO: Cell<usize> = const { Cell::new(0) };
}

fn refused(size: usize) -> bool {
    REFUSE_FROM.with(Cell::get) <= size && size <= REFUSE_TO.with(Cell::get)
}

fn record_grown(by: usize) {
    let live = LIVE.with(Cell::get).saturating_add(by);
    LIVE.with(|cell| cell.set(live));
    PEAK.with(|cell| cell.set(cell.get().max(live)));
}

fn record_shrunk(by: usize) {
    LIVE.with(|cell| cell.set(cell.get().saturating_sub(by)));
}

pub struct ThreadCountingAllocator;

// SAFETY: every call delegates to `System`; the bookkeeping does not allocate,
// and a refusal returns null, which the `GlobalAlloc` contract allows.
unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if refused(layout.size()) {
            return std::ptr::null_mut();
        }
        let block = unsafe { System.alloc(layout) };
        if !block.is_null() {
            record_grown(layout.size());
        }
        block
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        unsafe { System.dealloc(block, layout) };
        record_shrunk(layout.size());
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if refused(new_size) {
            return std::ptr::null_mut();
        }
        let moved = unsafe { System.realloc(block, layout, new_size) };
        if !moved.is_null() {
            record_shrunk(layout.size());
            record_grown(new_size);
        }
        moved
    }
}

/// What a watched closure cost its thread.
#[derive(Debug)]
pub struct Measured<T> {
    pub value: T,
    pub peak_bytes: usize,
    pub elapsed: Duration,
}

/// Run `work` on a fresh thread, refusing allocations in `refuse`, and fail if it
/// is not done within `deadline`. A blocked thread is abandoned; it cannot hold up
/// the process exiting.
pub fn watched<T: Send + 'static>(
    deadline: Duration,
    refuse: Option<RangeInclusive<usize>>,
    work: impl FnOnce() -> T + Send + 'static,
) -> Measured<T> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        if let Some(range) = refuse {
            REFUSE_FROM.with(|cell| cell.set(*range.start()));
            REFUSE_TO.with(|cell| cell.set(*range.end()));
        }
        let started = Instant::now();
        let value = work();
        let elapsed = started.elapsed();
        REFUSE_FROM.with(|cell| cell.set(usize::MAX));
        let measured = Measured {
            value,
            peak_bytes: PEAK.with(Cell::get),
            elapsed,
        };
        // A send fails only if the watchdog has already given up.
        let _ = sender.send(measured);
    });
    match receiver.recv_timeout(deadline) {
        Ok(measured) => measured,
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("load still running after {deadline:?}"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("load panicked"),
    }
}
