//! Command-buffer completion status — the check every `wait_until_completed`
//! site was missing.
//!
//! `waitUntilCompleted` returns for a *failed* buffer just as it does for a
//! finished one: `status == Error`, immediately. Once the GPU has faulted,
//! later buffers on the queue may be dropped outright ("ignored for causing
//! prior/excessive GPU errors"), which also completes instantly. Nothing
//! downstream can tell — the output buffers simply hold whatever was there
//! before — so a caller that only waits will read stale or garbage results
//! at full speed. See #229 and `docs/kv-attention-scaling.md`, "The fault is
//! on main and predates seqpar": impossible ~0.5 ms/token decode steps,
//! token ids past the vocabulary, EOS at token 1.
//!
//! This module names the condition and refuses on it. The rule for every
//! wait in this crate: any number LARQL reports must come from an
//! execution that actually succeeded, so a failed buffer never becomes a
//! result. [`wait_checked`] hands the failure to callers that have an
//! error channel; [`wait_or_abort`] stops the step for callers that do
//! not (a `Vec<f32>`-returning stage, or an `Option`-returning trait
//! method whose `None` would be taken as "fall back to the CPU" and
//! report the fallback's number as the GPU's). `cb_status::tests` pins
//! that no production site waits without one of the two, and that none
//! discards the result.
//!
//! A real fault cannot be produced deterministically — an out-of-bounds
//! kernel may be tolerated, or may poison the queue for the rest of the
//! process — so the refusal paths are witnessed through
//! [`inject_fault_at_for_test`] instead.

use metal::foreign_types::ForeignTypeRef;
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Buffers observed in a non-`Completed` state since process start.
static NON_COMPLETED: AtomicUsize = AtomicUsize::new(0);

/// A fault a test asked for: the next [`check_completed`] whose `site`
/// contains this fragment reports a failure instead of reading the
/// buffer. Consumed on first match.
static INJECTED_FAULT: Mutex<Option<String>> = Mutex::new(None);

/// How many command buffers this process has seen finish in any state other
/// than `Completed`. Zero on a healthy process. Injected test faults are
/// not counted: this number reports what the GPU actually did.
pub fn non_completed_count() -> usize {
    NON_COMPLETED.load(Ordering::Relaxed)
}

/// Make the next wait whose site name contains `site_fragment` report a
/// failure, without any GPU fault. Tests use this to witness that a
/// refusal path refuses: no readback, no cache advance, no result.
#[doc(hidden)]
pub fn inject_fault_at_for_test(site_fragment: &str) {
    *INJECTED_FAULT.lock().unwrap_or_else(|p| p.into_inner()) = Some(site_fragment.to_string());
}

/// Whether an injected fault is armed and has not yet been consumed. A
/// test that arms one asserts this is `false` afterwards, so a fault
/// that never fired cannot leak into the next test.
#[doc(hidden)]
pub fn injected_fault_pending() -> bool {
    INJECTED_FAULT
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .is_some()
}

fn take_injected_fault(site: &str) -> bool {
    let mut slot = INJECTED_FAULT.lock().unwrap_or_else(|p| p.into_inner());
    match slot.as_deref() {
        Some(fragment) if site.contains(fragment) => {
            *slot = None;
            true
        }
        _ => false,
    }
}

/// Wait for `cmd`, then inspect it. This is the only sanctioned way to
/// wait on a command buffer in this crate: `wait_until_completed` alone
/// cannot distinguish a finished buffer from a failed or ignored one, and
/// a test pins that no production site calls it directly. Callers with a
/// result channel propagate the `Err`; callers without one use
/// [`wait_or_abort`]. Discarding the result is not an option — a test
/// pins that too.
#[must_use = "a failed command buffer must refuse the step, not be logged past"]
pub fn wait_checked(cmd: &CommandBufferRef, site: &'static str) -> Result<(), String> {
    cmd.wait_until_completed();
    check_completed(cmd, site)
}

/// Wait for `cmd` and abort the step if it did not complete. For sites
/// whose enclosing function has no error channel to its caller. A fault
/// surfacing as a CPU fallback would report the fallback's number as the
/// GPU's, and a fault surfacing as a stale output would report garbage;
/// neither is a result, so the step ends here.
pub fn wait_or_abort(cmd: &CommandBufferRef, site: &'static str) {
    cmd.wait_until_completed();
    require_completed(cmd, site);
}

/// [`check_completed`] for a buffer the caller has already waited on by
/// other means (a spin loop on `status()`), aborting the step on failure.
pub fn require_completed(cmd: &CommandBufferRef, site: &'static str) {
    if let Err(msg) = check_completed(cmd, site) {
        panic!("[metal] refusing to continue past a failed command buffer: {msg}");
    }
}

/// Inspect `cmd` after `wait_until_completed`. Returns `Ok(())` for
/// `Completed`; otherwise records the event, prints one line naming the
/// site, the status and Metal's own error description, and returns that
/// description so a caller can decide what to do with a poisoned step.
pub fn check_completed(cmd: &CommandBufferRef, site: &'static str) -> Result<(), String> {
    if take_injected_fault(site) {
        let msg =
            format!("command buffer at {site} finished with status <injected fault, test hook>");
        eprintln!("[metal] {msg}");
        return Err(msg);
    }
    let status = cmd.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let n = NON_COMPLETED.fetch_add(1, Ordering::Relaxed) + 1;
    let desc = error_description(cmd).unwrap_or_else(|| "<no NSError>".to_string());
    let msg = format!("command buffer at {site} finished with status {status:?} (#{n}): {desc}");
    eprintln!("[metal] {msg}");
    Err(msg)
}

/// UTF-8 contents of an `NSString*`, or `None` for nil.
///
/// # Safety
/// `ns` must be nil or a live `NSString*`; the returned bytes are copied
/// before the autorelease pool can reclaim it.
unsafe fn ns_string(ns: *mut Object) -> Option<String> {
    if ns.is_null() {
        return None;
    }
    let utf8: *const std::os::raw::c_char = msg_send![ns, UTF8String];
    if utf8.is_null() {
        return None;
    }
    Some(
        std::ffi::CStr::from_ptr(utf8)
            .to_string_lossy()
            .into_owned(),
    )
}

/// `-[MTLCommandBuffer error].localizedDescription`, or `None` when the
/// buffer carries no NSError (e.g. `NotEnqueued`, or an ignored buffer on
/// some OS versions).
fn error_description(cmd: &CommandBufferRef) -> Option<String> {
    // SAFETY: `cmd` is a live MTLCommandBuffer; `error` returns an
    // autoreleased NSError* or nil, and `localizedDescription` an
    // autoreleased NSString*. Both are read, never retained or released.
    unsafe {
        let obj: *mut Object = cmd.as_ptr() as *mut Object;
        let err: *mut Object = msg_send![obj, error];
        if err.is_null() {
            return None;
        }
        ns_string(msg_send![err, localizedDescription])
    }
}

#[cfg(test)]
mod tests;
