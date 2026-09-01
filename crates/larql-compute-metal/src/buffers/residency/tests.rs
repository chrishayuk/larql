//! Tests for [`super`] — the `MTLResidencySet` arms.
//!
//! This module is a **retained instrument for a refuted hypothesis**: the
//! A/B/C ladder it exists to run showed explicit residency buys nothing
//! (19.61 / 19.63 / 19.80 ms/token), so nothing here should be read as
//! claiming otherwise. What the tests pin is the part that still matters —
//! that the instrument reports its own engagement honestly, because the
//! null result is only trustworthy if the arm actually attached. A
//! silently-inactive residency set and a residency set that does nothing
//! produce the same timing and must not be confused.
//!
//! The Objective-C calls themselves are not asserted against Metal's
//! internal state (there is no public way to read a set's allocations
//! back); what is asserted is that every one of them is reachable, that
//! the lifecycle flags track the calls, and that the whole thing degrades
//! to `None` rather than panicking on a runtime without the class.

use larql_compute::options::env_usize;

use super::{ResidencyArm, ResidencySet, ENV_RESIDENCY_SET};
use crate::MetalBackend;

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Shared with `buffers::regions::tests`, which seals residency under each
/// arm: one lock across both files, because they mutate the same variable
/// and `cargo test` runs them on different threads.
pub(in crate::buffers) fn with_residency_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    with_env(value, f)
}

fn with_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var_os(ENV_RESIDENCY_SET);
    match value {
        Some(v) => unsafe { std::env::set_var(ENV_RESIDENCY_SET, v) },
        None => unsafe { std::env::remove_var(ENV_RESIDENCY_SET) },
    }
    let out = f();
    match prev {
        Some(v) => unsafe { std::env::set_var(ENV_RESIDENCY_SET, v) },
        None => unsafe { std::env::remove_var(ENV_RESIDENCY_SET) },
    }
    out
}

/// The arm selector is the thing an experimenter reads to know which arm
/// ran, so each mapping is pinned individually — including the fallback,
/// which must be A rather than "whatever was last set".
#[test]
fn arm_selection_maps_each_documented_value() {
    with_env(None, || {
        assert_eq!(ResidencyArm::from_env(), ResidencyArm::Implicit)
    });
    with_env(Some("0"), || {
        assert_eq!(ResidencyArm::from_env(), ResidencyArm::Implicit)
    });
    with_env(Some("1"), || {
        assert_eq!(ResidencyArm::from_env(), ResidencyArm::QueueSet)
    });
    with_env(Some("2"), || {
        assert_eq!(ResidencyArm::from_env(), ResidencyArm::QueueSetRequested)
    });
}

/// Anything outside the documented vocabulary falls back to A. A typo
/// must not silently select an arm, and must not select a *different*
/// arm than the one the experimenter believes they asked for.
#[test]
fn an_unrecognised_value_falls_back_to_the_implicit_arm() {
    for v in ["3", "99", "yes", "true", ""] {
        with_env(Some(v), || {
            assert_eq!(
                ResidencyArm::from_env(),
                ResidencyArm::Implicit,
                "LARQL_RESIDENCY_SET={v:?} must not select an explicit arm"
            );
        });
    }
}

/// `env_usize` is what the selector reads; pinning it here documents why
/// `"1"` selects B while `"true"` does not.
#[test]
fn the_selector_reads_a_number_not_a_flag() {
    with_env(Some("1"), || {
        assert_eq!(env_usize(ENV_RESIDENCY_SET), Some(1))
    });
    with_env(Some("true"), || {
        assert_eq!(env_usize(ENV_RESIDENCY_SET), None)
    });
}

/// The full lifecycle, in the order the production caller uses it. On a
/// runtime without `MTLResidencySet` construction returns `None` and the
/// caller keeps implicit residency — correct, just slower — so the test
/// treats that as a pass rather than forcing a macOS-15 floor on CI.
#[test]
fn a_residency_set_supports_the_whole_lifecycle() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let Some(mut set) = ResidencySet::new(&metal.bufs.device) else {
        // Pre-macOS-15 runtime: the degradation path is the behaviour
        // under test here, and it did not panic.
        return;
    };

    assert!(
        !set.is_committed(),
        "a fresh set has nothing applied yet — the flag must not start true"
    );

    let buf = metal.bufs.output(4096);
    set.add_buffer(&buf);
    assert!(
        !set.is_committed(),
        "adding an allocation is pending until commit; reporting committed \
         here would let an experiment record an arm it never applied"
    );

    set.commit();
    assert!(set.is_committed());

    // Arm C's extra step. Idempotent from the caller's side.
    set.request_residency();
    assert!(set.is_committed());
}

/// Arm B's engagement signal. `add_to_queue` returning `false` is the
/// documented "this runtime cannot attach" answer, and the run that
/// produced the null result verified this line before trusting it.
#[test]
fn attaching_to_a_queue_reports_whether_it_took() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let Some(mut set) = ResidencySet::new(&metal.bufs.device) else {
        return;
    };
    let buf = metal.bufs.output(4096);
    set.add_buffer(&buf);
    set.commit();

    // Either verdict is legitimate depending on the runtime, so there is
    // nothing to assert about the value itself — asserting `attached ||
    // !attached` would be a tautology dressed as a check. What is worth
    // pinning is the consequence: when it reports success, the queue it
    // attached to must still function.
    if set.add_to_queue(&metal.queue) {
        // Attaching must not disturb the queue: a command buffer from it
        // still runs to completion.
        let cmd = metal.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/buffers/residency/tests.rs:157",
        )
        .expect("command buffer completed");
    }
}

/// Dropping releases the `+1` reference. Nothing observable is asserted —
/// the point is that the Drop impl runs on a committed, queue-attached set
/// without over-releasing, which is what a double-free would show as.
#[test]
fn dropping_a_committed_attached_set_is_clean() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    for _ in 0..4 {
        let Some(mut set) = ResidencySet::new(&metal.bufs.device) else {
            return;
        };
        let buf = metal.bufs.output(1024);
        set.add_buffer(&buf);
        set.commit();
        let _ = set.add_to_queue(&metal.queue);
        drop(set);
    }
}
