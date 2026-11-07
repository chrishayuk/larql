//! Free space on the volume holding a path, from the OS.

use std::path::Path;

/// Bytes available to this user on the volume holding `path`, or `None`
/// when the OS will not say.
#[cfg(unix)]
// `statvfs`'s field widths differ by platform (`f_bavail` is u32 on macOS,
// u64 on Linux), so the widening `u64::from` is a no-op on some targets.
#[allow(clippy::useless_conversion)]
pub(crate) fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `statvfs` is a plain C struct, so all-zero is a valid value;
    // `c_path` is NUL-terminated and outlives the call.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    (rc == 0).then(|| u64::from(stat.f_bavail) * u64::from(stat.f_frsize))
}

/// Bytes available to this user on the volume holding `path`, or `None`
/// when the OS will not say.
#[cfg(windows)]
pub(crate) fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let mut available: u64 = 0;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the two
    // totals this caller does not need may be null per the API.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some(available)
}

/// No free-space API on this platform.
#[cfg(not(any(unix, windows)))]
pub(crate) fn free_bytes(_path: &Path) -> Option<u64> {
    None
}
