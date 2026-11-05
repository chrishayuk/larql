//! A system-wide monotonic clock, for ordering records written by
//! different processes within one boot.
//!
//! [`super::ResidencyObservation`]'s `started_mono_ns` / `ended_mono_ns`
//! count from the observing process's first observation, so they are
//! comparable only within that process. Readings of this clock are
//! comparable across processes, but only within one boot session: values
//! from different [`boot_session`]s are never compared. Ordering on it is
//! not causality.
//!
//! The basis is named in every record so a reader never infers it:
//! - macOS: `CLOCK_MONOTONIC_RAW` advances during sleep (it equals
//!   `mach_continuous_time`);
//! - Linux: `CLOCK_MONOTONIC_RAW` stops during suspend.
//!
//! Elsewhere there is no basis and every reading is `None` — never 0.

use std::sync::OnceLock;

/// The clock every system-wide reading is taken from, as records name it.
pub const SYSTEM_MONO_BASIS: &str = "CLOCK_MONOTONIC_RAW";

/// The sysctl naming this boot on macOS. `kern.bootuuid` is NOT a boot
/// identity — it is the Data volume's UUID.
#[cfg(target_os = "macos")]
const BOOT_SESSION_SYSCTL: &[u8] = b"kern.bootsessionuuid\0";

#[cfg(any(target_os = "macos", target_os = "linux"))]
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// The basis [`system_mono_ns`] reads on this target, or `None` where it
/// reads nothing.
pub fn system_mono_basis() -> Option<&'static str> {
    cfg!(any(target_os = "macos", target_os = "linux")).then_some(SYSTEM_MONO_BASIS)
}

/// Nanoseconds on [`SYSTEM_MONO_BASIS`], or `None` when the target has no
/// such clock or the read failed.
pub fn system_mono_ns() -> Option<u64> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a valid, writable timespec for the call.
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut now) } != 0 {
            return None;
        }
        let seconds = u64::try_from(now.tv_sec).ok()?;
        let nanos = u64::try_from(now.tv_nsec).ok()?;
        seconds.checked_mul(NANOS_PER_SECOND)?.checked_add(nanos)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// This boot's identity, read once per process: on macOS
/// `kern.bootsessionuuid`; elsewhere `None` until an equivalent is
/// confirmed.
pub fn boot_session() -> Option<String> {
    static SESSION: OnceLock<Option<String>> = OnceLock::new();
    SESSION.get_or_init(read_boot_session).clone()
}

#[cfg(target_os = "macos")]
fn read_boot_session() -> Option<String> {
    let name = BOOT_SESSION_SYSCTL.as_ptr().cast::<libc::c_char>();
    let mut len: libc::size_t = 0;
    // SAFETY: a null buffer asks only for the value's length.
    let sized = unsafe {
        libc::sysctlbyname(
            name,
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if sized != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` holds `len` writable bytes; `len` is updated to the
    // bytes written.
    let read = unsafe {
        libc::sysctlbyname(
            name,
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if read != 0 {
        return None;
    }
    buf.truncate(len);
    let text = String::from_utf8(buf).ok()?;
    let session = text.trim_end_matches('\0').trim();
    (!session.is_empty()).then(|| session.to_string())
}

#[cfg(not(target_os = "macos"))]
fn read_boot_session() -> Option<String> {
    None
}
