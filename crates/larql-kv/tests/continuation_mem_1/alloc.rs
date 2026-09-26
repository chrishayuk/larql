//! CONTINUATION-MEM-1 I1: a counting global allocator with scopes.
//!
//! A scope is opened on one thread (its owner). While it is open, every
//! allocation, free and reallocation on the owner thread is counted and
//! logged; an allocation on any OTHER thread marks the run foreign
//! (I1_thread_integrity), because thread-local attribution would not have
//! seen it. Nothing here allocates: counters, the event log and the
//! live-allocation table are fixed-size statics of atomics, so the recorder
//! cannot measure itself (the recorder-silence control proves it).
//!
//! The live table holds every allocation MADE inside a scope, keyed by
//! pointer, and forgets it when it is freed or reallocated wherever that
//! happens. It is how an allocation's capacity is read back when the type
//! that owns it (an ndarray matrix) exposes none (deviation D3).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering::Relaxed};

pub struct MeasuringAllocator;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static OWNER: AtomicUsize = AtomicUsize::new(0);

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static FREE_BYTES: AtomicU64 = AtomicU64::new(0);
static MOVED: AtomicU64 = AtomicU64::new(0);
static MOVED_BYTES: AtomicU64 = AtomicU64::new(0);
static IN_PLACE: AtomicU64 = AtomicU64::new(0);
static FOREIGN: AtomicU64 = AtomicU64::new(0);

/// Events of the CURRENT scope, reset on entry.
const EVENT_CAPACITY: usize = 1 << 16;
static EVENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static EVENT_KIND: [AtomicU8; EVENT_CAPACITY] = [const { AtomicU8::new(0) }; EVENT_CAPACITY];
static EVENT_OLD_PTR: [AtomicUsize; EVENT_CAPACITY] =
    [const { AtomicUsize::new(0) }; EVENT_CAPACITY];
static EVENT_NEW_PTR: [AtomicUsize; EVENT_CAPACITY] =
    [const { AtomicUsize::new(0) }; EVENT_CAPACITY];
static EVENT_OLD_SIZE: [AtomicUsize; EVENT_CAPACITY] =
    [const { AtomicUsize::new(0) }; EVENT_CAPACITY];
static EVENT_NEW_SIZE: [AtomicUsize; EVENT_CAPACITY] =
    [const { AtomicUsize::new(0) }; EVENT_CAPACITY];

/// Open-addressing pointer → size table. `TOMBSTONE` keeps probe chains
/// intact after a removal.
const TABLE_CAPACITY: usize = 1 << 21;
const EMPTY: usize = 0;
const TOMBSTONE: usize = usize::MAX;
static TABLE_PTR: [AtomicUsize; TABLE_CAPACITY] =
    [const { AtomicUsize::new(EMPTY) }; TABLE_CAPACITY];
static TABLE_SIZE: [AtomicUsize; TABLE_CAPACITY] = [const { AtomicUsize::new(0) }; TABLE_CAPACITY];
static TABLE_LIVE: AtomicUsize = AtomicUsize::new(0);
static TABLE_OVERFLOW: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static TOKEN: u8 = const { 0 };
    static IN_ALLOCATOR: Cell<bool> = const { Cell::new(false) };
}

fn thread_token() -> usize {
    TOKEN.with(|t| t as *const u8 as usize)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Alloc,
    Free,
    ReallocMoved,
    ReallocInPlace,
}

impl EventKind {
    fn code(self) -> u8 {
        match self {
            Self::Alloc => 1,
            Self::Free => 2,
            Self::ReallocMoved => 3,
            Self::ReallocInPlace => 4,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Alloc,
            2 => Self::Free,
            3 => Self::ReallocMoved,
            _ => Self::ReallocInPlace,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub kind: EventKind,
    pub old_ptr: usize,
    pub new_ptr: usize,
    pub old_size: usize,
    pub new_size: usize,
}

/// What one scope saw. Plain data, built only after the scope closed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeDelta {
    pub allocs: u64,
    pub alloc_bytes: u64,
    pub frees: u64,
    pub free_bytes: u64,
    pub moved: u64,
    pub moved_bytes: u64,
    pub in_place: u64,
    pub foreign: u64,
    pub events: usize,
    pub events_dropped: bool,
}

#[derive(Clone, Copy)]
struct Counters([u64; 8]);

fn snapshot() -> Counters {
    Counters([
        ALLOCS.load(Relaxed),
        ALLOC_BYTES.load(Relaxed),
        FREES.load(Relaxed),
        FREE_BYTES.load(Relaxed),
        MOVED.load(Relaxed),
        MOVED_BYTES.load(Relaxed),
        IN_PLACE.load(Relaxed),
        FOREIGN.load(Relaxed),
    ])
}

/// An open scope. Leaving it is the only way to read what it saw.
#[must_use = "a scope must be left to be read"]
pub struct Scope {
    start: Counters,
}

/// Open a scope on this thread. Allocation-free.
pub fn enter() -> Scope {
    assert!(
        !ACTIVE.load(Relaxed),
        "measuring scopes do not nest; one is already open"
    );
    EVENT_COUNT.store(0, Relaxed);
    OWNER.store(thread_token(), Relaxed);
    let start = snapshot();
    ACTIVE.store(true, Relaxed);
    Scope { start }
}

/// Is a scope open (on any thread)?
pub fn active() -> bool {
    ACTIVE.load(Relaxed)
}

impl Scope {
    /// Close the scope. Allocation-free up to the returned value.
    pub fn leave(self) -> ScopeDelta {
        let end = snapshot();
        ACTIVE.store(false, Relaxed);
        let d = |i: usize| end.0[i] - self.start.0[i];
        let events = EVENT_COUNT.load(Relaxed);
        ScopeDelta {
            allocs: d(0),
            alloc_bytes: d(1),
            frees: d(2),
            free_bytes: d(3),
            moved: d(4),
            moved_bytes: d(5),
            in_place: d(6),
            foreign: d(7),
            events: events.min(EVENT_CAPACITY),
            events_dropped: events > EVENT_CAPACITY,
        }
    }
}

/// The events of the most recently closed scope. Call only after `leave`.
pub fn events(delta: &ScopeDelta) -> Vec<Event> {
    assert!(!active(), "events are read after the scope closes");
    (0..delta.events)
        .map(|i| Event {
            kind: EventKind::from_code(EVENT_KIND[i].load(Relaxed)),
            old_ptr: EVENT_OLD_PTR[i].load(Relaxed),
            new_ptr: EVENT_NEW_PTR[i].load(Relaxed),
            old_size: EVENT_OLD_SIZE[i].load(Relaxed),
            new_size: EVENT_NEW_SIZE[i].load(Relaxed),
        })
        .collect()
}

/// The size of a live allocation made inside some scope, if the table
/// holds it. `None` means the allocation was never made in a scope, or
/// has been freed — never "zero bytes".
pub fn live_size(ptr: usize) -> Option<usize> {
    let mut slot = hash(ptr);
    for _ in 0..TABLE_CAPACITY {
        match TABLE_PTR[slot].load(Relaxed) {
            EMPTY => return None,
            p if p == ptr => return Some(TABLE_SIZE[slot].load(Relaxed)),
            _ => slot = (slot + 1) & (TABLE_CAPACITY - 1),
        }
    }
    None
}

pub fn table_overflowed() -> bool {
    TABLE_OVERFLOW.load(Relaxed) > 0
}

/// Attributed allocations and bytes since process start (negative control).
pub fn attributed_totals() -> (u64, u64) {
    (ALLOCS.load(Relaxed), ALLOC_BYTES.load(Relaxed))
}

fn hash(ptr: usize) -> usize {
    ((ptr >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15)) & (TABLE_CAPACITY - 1)
}

fn table_insert(ptr: usize, size: usize) {
    let mut slot = hash(ptr);
    for _ in 0..TABLE_CAPACITY {
        let current = TABLE_PTR[slot].load(Relaxed);
        if current == EMPTY || current == TOMBSTONE || current == ptr {
            TABLE_SIZE[slot].store(size, Relaxed);
            TABLE_PTR[slot].store(ptr, Relaxed);
            if current != ptr {
                TABLE_LIVE.fetch_add(1, Relaxed);
            }
            return;
        }
        slot = (slot + 1) & (TABLE_CAPACITY - 1);
    }
    TABLE_OVERFLOW.fetch_add(1, Relaxed);
}

/// Forget `ptr`; returns whether it was held.
fn table_remove(ptr: usize) -> bool {
    if TABLE_LIVE.load(Relaxed) == 0 {
        return false;
    }
    let mut slot = hash(ptr);
    for _ in 0..TABLE_CAPACITY {
        match TABLE_PTR[slot].load(Relaxed) {
            EMPTY => return false,
            p if p == ptr => {
                TABLE_PTR[slot].store(TOMBSTONE, Relaxed);
                TABLE_LIVE.fetch_sub(1, Relaxed);
                return true;
            }
            _ => slot = (slot + 1) & (TABLE_CAPACITY - 1),
        }
    }
    false
}

fn log(kind: EventKind, old_ptr: usize, new_ptr: usize, old_size: usize, new_size: usize) {
    let i = EVENT_COUNT.fetch_add(1, Relaxed);
    if i < EVENT_CAPACITY {
        EVENT_KIND[i].store(kind.code(), Relaxed);
        EVENT_OLD_PTR[i].store(old_ptr, Relaxed);
        EVENT_NEW_PTR[i].store(new_ptr, Relaxed);
        EVENT_OLD_SIZE[i].store(old_size, Relaxed);
        EVENT_NEW_SIZE[i].store(new_size, Relaxed);
    }
}

/// `Some(true)` on the owner thread of an open scope, `Some(false)` on a
/// foreign thread while one is open, `None` when no scope is open.
fn attribution() -> Option<bool> {
    if !ACTIVE.load(Relaxed) {
        return None;
    }
    Some(OWNER.load(Relaxed) == thread_token())
}

unsafe impl GlobalAlloc for MeasuringAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record_alloc(ptr, layout.size());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record_alloc(ptr, layout.size());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        table_remove(ptr as usize);
        if attribution() == Some(true) {
            FREES.fetch_add(1, Relaxed);
            FREE_BYTES.fetch_add(layout.size() as u64, Relaxed);
            log(EventKind::Free, ptr as usize, 0, layout.size(), 0);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if new.is_null() {
            return new;
        }
        let held = table_remove(ptr as usize);
        match attribution() {
            Some(true) => {
                let kind = if new == ptr {
                    IN_PLACE.fetch_add(1, Relaxed);
                    EventKind::ReallocInPlace
                } else {
                    MOVED.fetch_add(1, Relaxed);
                    MOVED_BYTES.fetch_add(layout.size() as u64, Relaxed);
                    EventKind::ReallocMoved
                };
                log(kind, ptr as usize, new as usize, layout.size(), new_size);
                table_insert(new as usize, new_size);
            }
            Some(false) => {
                FOREIGN.fetch_add(1, Relaxed);
                if held {
                    table_insert(new as usize, new_size);
                }
            }
            None if held => table_insert(new as usize, new_size),
            None => {}
        }
        new
    }
}

fn record_alloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    // A thread-local access during thread teardown must not recurse.
    let reentrant = IN_ALLOCATOR.try_with(|f| f.replace(true)).unwrap_or(true);
    if reentrant {
        return;
    }
    match attribution() {
        Some(true) => {
            ALLOCS.fetch_add(1, Relaxed);
            ALLOC_BYTES.fetch_add(size as u64, Relaxed);
            log(EventKind::Alloc, 0, ptr as usize, 0, size);
            table_insert(ptr as usize, size);
        }
        Some(false) => {
            FOREIGN.fetch_add(1, Relaxed);
        }
        None => {}
    }
    let _ = IN_ALLOCATOR.try_with(|f| f.set(false));
}
