//! A capture of which experts every routed layer selected, in execution
//! order, for the step under observation. Off unless a caller starts it,
//! costs one atomic load per routed call when off, and exists so a
//! latency figure can say what work it timed: two passes over "the same
//! token" that routed differently measured different things.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CAPTURE: Mutex<Option<Vec<Vec<usize>>>> = Mutex::new(None);
static RESIDENCY: Mutex<Vec<super::prefetch::Residency>> = Mutex::new(Vec::new());
/// Whether an open capture also reads residency — a page-table walk over
/// every selected page, ≈70 ms on a 3 GB selection, so it is asked for
/// by name and never rides a latency measurement unasked.
static WITNESS_RESIDENCY: AtomicBool = AtomicBool::new(false);
static REQUESTS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Begin capturing; any earlier capture is discarded.
pub fn start_capture() {
    *CAPTURE.lock().expect("routing capture lock") = Some(Vec::new());
    RESIDENCY.lock().expect("residency capture lock").clear();
    REQUESTS.lock().expect("request capture lock").clear();
    ACTIVE.store(true, Ordering::Release);
}

/// Stop capturing and return every routed call's selected experts, in
/// the order the calls happened.
pub fn take_capture() -> Vec<Vec<usize>> {
    ACTIVE.store(false, Ordering::Release);
    CAPTURE
        .lock()
        .expect("routing capture lock")
        .take()
        .unwrap_or_default()
}

/// Record one routed call's selection. A no-op unless a capture is open.
pub fn record(selected: &[(usize, f32)]) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    if let Some(capture) = CAPTURE.lock().expect("routing capture lock").as_mut() {
        capture.push(selected.iter().map(|(e, _)| *e).collect());
    }
}

/// A short, order-sensitive fingerprint of a capture, so two passes can
/// be compared at a glance and a mismatch located by index.
///
/// Each expert is hashed as the word (layer, slot, expert) — never the
/// bare id — so moving a boundary between layers changes the words, not
/// just their order. (A separator XORed between layers is not enough:
/// XOR-then-multiply lets `[3, 1 | 0]` and `[3 | 1, 0]` collide.)
pub fn fingerprint(capture: &[Vec<usize>]) -> u64 {
    // FNV-1a over position-aware words.
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const LAYER_SHIFT: u32 = 48;
    const SLOT_SHIFT: u32 = 32;
    let mut h = OFFSET;
    for (layer, selected) in capture.iter().enumerate() {
        h ^= (selected.len() as u64) << LAYER_SHIFT | 0xffff_ffff;
        h = h.wrapping_mul(PRIME);
        for (slot, &expert) in selected.iter().enumerate() {
            h ^= (layer as u64) << LAYER_SHIFT | (slot as u64) << SLOT_SHIFT | expert as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

/// Whether a capture is open — so a caller can skip a measurement that
/// only a capture would read.
pub fn is_capturing() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Ask the next captures to read residency too (see
/// [`record_residency`]); off by default because the reading has a cost
/// a latency figure must not carry unannounced.
pub fn set_witness_residency(on: bool) {
    WITNESS_RESIDENCY.store(on, Ordering::Release);
}

/// Whether an open capture wants residency read.
pub fn wants_residency() -> bool {
    ACTIVE.load(Ordering::Acquire) && WITNESS_RESIDENCY.load(Ordering::Acquire)
}

/// Record one routed call's request shape: how many ranges its prefetch
/// covered and how many requests it issued for them. A no-op unless a
/// capture is open.
pub fn record_requests(ranges: usize, requests: usize) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    REQUESTS
        .lock()
        .expect("request capture lock")
        .push((ranges, requests));
}

/// Every routed call's (ranges, requests) since the capture opened.
pub fn take_requests() -> Vec<(usize, usize)> {
    std::mem::take(&mut *REQUESTS.lock().expect("request capture lock"))
}

/// Record what one routed call found resident of its selected experts
/// after its prefetch and before its loop. A no-op unless a capture is
/// open.
pub fn record_residency(residency: super::prefetch::Residency) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    RESIDENCY
        .lock()
        .expect("residency capture lock")
        .push(residency);
}

/// Every routed call's residency reading since the capture opened, in
/// call order; empty when nothing recorded one.
pub fn take_residency() -> Vec<super::prefetch::Residency> {
    std::mem::take(&mut *RESIDENCY.lock().expect("residency capture lock"))
}
