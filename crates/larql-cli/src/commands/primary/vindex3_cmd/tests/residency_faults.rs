//! NB1: a step's fault counts cover the step and nothing the observer
//! does around it.
//!
//! The curve used to read the counters inside its observer, after the
//! residency scan and before the delta and the printing, so the scan was
//! charged to the step just measured and the rest to the next one. Now
//! the counters are read by [`bracket`], in the timer's own window. This
//! test re-creates the observer's position: heavy allocation immediately
//! before and immediately after a bracketed step, with a positive control
//! showing the same allocation IS counted when it runs inside a bracket.
//! Process-wide counters can pick up another test thread's faults, so
//! the bound is a fraction of the control rather than zero.

#[cfg(unix)]
use std::hint::black_box;

use super::super::generate::bracket;
#[cfg(unix)]
use super::super::generate::process_resources;

#[cfg(unix)]
/// Fresh pages the stand-in observer touches: 64 MiB.
const TOUCHED_BYTES: usize = 64 << 20;
#[cfg(unix)]
/// Write one byte in every 4 KiB, which touches every page at any page
/// size this runs on.
const STRIDE: usize = 4096;
#[cfg(unix)]
/// An empty step may pick up stray faults from other threads; it may not
/// pick up this share of the observer's own.
const MAX_LEAK_FRACTION: u64 = 10;

/// Map and touch [`TOUCHED_BYTES`] of fresh anonymous memory, then unmap
/// it, as an observer building its per-region maps does on a large
/// container. A fresh mapping every call: a heap allocation could be
/// served from pages the allocator kept from the last call, which take
/// no fault, and the control would then witness nothing.
#[cfg(unix)]
fn observer_allocates() {
    // SAFETY: an anonymous private mapping of TOUCHED_BYTES, written only
    // within its bounds and unmapped before return; nothing else holds it.
    unsafe {
        let base = libc::mmap(
            std::ptr::null_mut(),
            TOUCHED_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(base, libc::MAP_FAILED, "anonymous mmap failed");
        let bytes = base.cast::<u8>();
        for i in (0..TOUCHED_BYTES).step_by(STRIDE) {
            std::ptr::write_volatile(bytes.add(i), 1);
        }
        black_box(bytes);
        assert_eq!(libc::munmap(base, TOUCHED_BYTES), 0, "munmap failed");
    }
}

#[cfg(unix)]
#[test]
fn the_observers_own_allocation_faults_land_outside_every_bracket() {
    // The counters must be live, or every assertion below passes vacuously.
    let before = process_resources();
    observer_allocates();
    assert!(
        process_resources().minor_faults > before.minor_faults,
        "getrusage reports no faults: this test cannot witness anything"
    );

    // Positive control: the observer's work inside a bracket IS counted.
    let control = bracket(observer_allocates);
    let observer_faults = control.minor_faults();
    assert!(
        observer_faults > MAX_LEAK_FRACTION,
        "the control touched {TOUCHED_BYTES} bytes and counted {observer_faults} faults"
    );

    // The curve's sequence: observer, step, observer, step.
    for _ in 0..2 {
        observer_allocates();
        let step = bracket(|| black_box(1u64 + 1));
        observer_allocates();
        assert!(
            step.minor_faults() < observer_faults / MAX_LEAK_FRACTION,
            "an empty step was charged {} faults; the observer alone makes {observer_faults}",
            step.minor_faults()
        );
        assert!(step.after.max_rss >= step.before.max_rss);
        assert_eq!(step.value, 2);
    }
}

#[test]
fn a_bracket_returns_its_steps_value_and_a_wall_time() {
    let step = bracket(|| "done");
    assert_eq!(step.value, "done");
    assert!(step.elapsed_seconds >= 0.0);
    assert_eq!(
        step.major_faults(),
        step.after.major_faults - step.before.major_faults
    );
}
