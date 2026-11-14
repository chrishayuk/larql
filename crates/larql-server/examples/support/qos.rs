//! Optional thread scheduling class for the DEC-V3-1 link instruments.
//!
//! A polling thread that the OS treats as background work can be moved,
//! preempted or run on a slower core, which would show up as link time. This
//! lets a calibration run ask for the interactive class so that hypothesis
//! can be tested. It is a request, not a guarantee: a QoS class does not pin
//! a thread to a core or a core type, so the instruments report the class
//! that was requested and calibration still has to pass on its own.

/// Environment variable naming the class; unset keeps the OS default.
pub const ENV: &str = "LARQL_LINK_QOS";
pub const DEFAULT: &str = "default";
pub const USER_INTERACTIVE: &str = "user-interactive";

/// Apply the requested class to the calling thread and return its name.
pub fn apply() -> Result<&'static str, String> {
    match std::env::var(ENV).ok().as_deref() {
        None | Some(DEFAULT) => Ok(DEFAULT),
        Some(USER_INTERACTIVE) => user_interactive().map(|()| USER_INTERACTIVE),
        Some(other) => Err(format!("{ENV}={other} is not a known scheduling class")),
    }
}

#[cfg(target_os = "macos")]
fn user_interactive() -> Result<(), String> {
    // SAFETY: sets the QoS class of the calling thread only; no pointers.
    let status = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    if status == 0 {
        Ok(())
    } else {
        Err(format!("pthread_set_qos_class_self_np failed: {status}"))
    }
}

#[cfg(not(target_os = "macos"))]
fn user_interactive() -> Result<(), String> {
    Err(format!(
        "{ENV}={USER_INTERACTIVE} is only implemented on macOS"
    ))
}
