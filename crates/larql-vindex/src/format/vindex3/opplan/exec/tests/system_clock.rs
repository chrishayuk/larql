//! The system-wide monotonic clock that orders records across processes
//! within one boot. A missing clock reads as `None`, never as 0, and an
//! observation's system-wide bounds are read at the same points as its
//! process-relative ones.

use super::super::prepared::{
    boot_session, system_mono_basis, system_mono_ns, ResidencyObservation, SYSTEM_MONO_BASIS,
};

const SUPPORTED: bool = cfg!(any(target_os = "macos", target_os = "linux"));

#[test]
fn the_basis_is_named_exactly_where_the_clock_is_read() {
    assert_eq!(system_mono_basis(), SUPPORTED.then_some(SYSTEM_MONO_BASIS));
    assert_eq!(system_mono_ns().is_some(), SUPPORTED);
}

#[test]
fn readings_never_go_backwards() {
    if let (Some(first), Some(second)) = (system_mono_ns(), system_mono_ns()) {
        assert!(second >= first, "{second} < {first}");
    }
}

#[test]
fn a_boot_session_is_named_only_where_one_is_confirmed() {
    let session = boot_session();
    if cfg!(target_os = "macos") {
        let session = session.expect("macOS names its boot session");
        assert!(!session.is_empty());
        assert!(!session.contains('\0'), "{session:?}");
        assert_eq!(boot_session(), Some(session), "stable within a process");
    } else {
        assert_eq!(session, None);
    }
}

#[test]
fn an_observation_carries_ordered_system_wide_bounds() {
    let before = system_mono_ns();
    let observation = ResidencyObservation::capture(std::iter::empty());
    let after = system_mono_ns();
    match (
        before,
        observation.started_system_mono_ns,
        observation.ended_system_mono_ns,
        after,
    ) {
        (Some(before), Some(start), Some(end), Some(after)) => {
            assert!(before <= start && start <= end && end <= after);
        }
        (None, None, None, None) => assert_eq!(system_mono_basis(), None),
        other => panic!("bounds read on a different basis than the clock: {other:?}"),
    }
    assert!(observation.started_mono_ns <= observation.ended_mono_ns);
}
