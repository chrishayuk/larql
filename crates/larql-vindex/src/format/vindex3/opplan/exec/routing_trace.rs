//! A capture of which experts every routed layer selected, in execution
//! order, for the step under observation. Off unless a caller starts it,
//! costs one atomic load per routed call when off, and exists so a
//! latency figure can say what work it timed: two passes over "the same
//! token" that routed differently measured different things.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CAPTURE: Mutex<Option<Vec<Vec<usize>>>> = Mutex::new(None);

/// Begin capturing; any earlier capture is discarded.
pub fn start_capture() {
    *CAPTURE.lock().expect("routing capture lock") = Some(Vec::new());
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
