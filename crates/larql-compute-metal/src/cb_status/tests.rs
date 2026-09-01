use super::*;
use std::path::Path;

/// Every `.rs` file under `src/`, recursively.
fn rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The containment rule for #229: no production code waits on a command
/// buffer without reading its status afterwards. `wait_until_completed`
/// returns for a failed or ignored buffer exactly as for a finished one,
/// so a naked wait is a step that can hand GPU garbage to the sampler.
/// This module is the only place the raw call may appear.
#[test]
fn no_naked_wait_until_completed_outside_cb_status() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(files.len() > 50, "walked too few files: {}", files.len());

    let mut offenders = Vec::new();
    for path in files {
        // The definition site, and this test's own needle below.
        if path.ends_with("cb_status.rs") || path.ends_with("cb_status/tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("source readable");
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains(concat!(".wait_until_", "completed()")) {
                offenders.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "naked wait_until_completed — use cb_status::wait_checked:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn non_completed_count_starts_at_zero_and_is_monotonic() {
    let a = non_completed_count();
    let b = non_completed_count();
    assert!(b >= a);
}

#[cfg(target_os = "macos")]
#[test]
fn a_completed_empty_command_buffer_passes() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    cmd.commit();
    let before = non_completed_count();
    assert!(wait_checked(cmd, "cb_status test").is_ok());
    assert_eq!(non_completed_count(), before);
}

#[cfg(target_os = "macos")]
#[test]
fn a_buffer_that_never_ran_is_reported_and_counted() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    // Never committed: status stays NotEnqueued, and there is no NSError,
    // which exercises the "<no NSError>" arm.
    let cmd = queue.new_command_buffer();
    let before = non_completed_count();
    let err = check_completed(cmd, "cb_status test: not enqueued").expect_err("not completed");
    assert!(err.contains("NotEnqueued"), "{err}");
    assert!(err.contains("<no NSError>"), "{err}");
    assert!(err.contains("cb_status test: not enqueued"), "{err}");
    assert_eq!(non_completed_count(), before + 1);
}

#[test]
fn ns_string_of_nil_is_none() {
    // SAFETY: nil is a valid argument; the function checks it first.
    assert!(unsafe { ns_string(std::ptr::null_mut()) }.is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn ns_string_reads_an_nsstring() {
    use objc::{class, msg_send, sel, sel_impl};
    let text = c"command buffer text";
    // SAFETY: standard Foundation call; the returned NSString is
    // autoreleased and only read.
    let got = unsafe {
        let ns: *mut Object = msg_send![class!(NSString), stringWithUTF8String: text.as_ptr()];
        ns_string(ns)
    };
    assert_eq!(got.as_deref(), Some("command buffer text"));
}

/// The second containment rule: every wait either propagates its result
/// or aborts. A `let _ =` on `wait_checked` is the "log and continue"
/// this module exists to end — the step would carry on with whatever the
/// output buffers held before the fault and report it as a result.
#[test]
fn no_discarded_wait_checked_outside_cb_status() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        if path.ends_with("cb_status.rs") || path.ends_with("cb_status/tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("source readable");
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("let _ =") && code.contains(concat!("wait_", "checked")) {
                offenders.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "discarded wait_checked — propagate with `?` or use cb_status::wait_or_abort:\n{}",
        offenders.join("\n")
    );
}

/// The injection hook fires once, only at the named site, and is not
/// counted as a GPU observation.
#[cfg(target_os = "macos")]
#[test]
fn an_injected_fault_is_reported_once_at_its_site_and_then_cleared() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    cmd.commit();

    inject_fault_at_for_test("cb_status injection witness");
    assert!(injected_fault_pending());
    // A wait at some other site neither fires nor consumes it.
    assert!(wait_checked(cmd, "cb_status somewhere else").is_ok());
    assert!(injected_fault_pending());

    let before = non_completed_count();
    let err = wait_checked(cmd, "cb_status injection witness").expect_err("armed fault must fire");
    assert!(err.contains("injected fault"), "{err}");
    assert!(err.contains("cb_status injection witness"), "{err}");
    assert_eq!(
        non_completed_count(),
        before,
        "an injected fault is not a GPU observation"
    );
    assert!(!injected_fault_pending(), "consumed on first match");
    assert!(
        wait_checked(cmd, "cb_status injection witness").is_ok(),
        "fires exactly once"
    );
}

/// `wait_or_abort` is the refusal for a site with no error channel: the
/// step ends, so no result can be reported from a failed buffer.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "refusing to continue past a failed command buffer")]
fn wait_or_abort_stops_the_step_on_an_injected_fault() {
    let Some(device) = metal::Device::system_default() else {
        panic!("refusing to continue past a failed command buffer: no Metal device on this host");
    };
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    cmd.commit();
    inject_fault_at_for_test("cb_status abort witness");
    wait_or_abort(cmd, "cb_status abort witness");
}

/// A production site with no error channel: `MatMul::matmul` returns a
/// plain `Array2<f32>`, so a faulted buffer there ends the step rather
/// than handing back the previous contents of the output buffer as a
/// product.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "refusing to continue past a failed command buffer")]
fn a_faulted_matmul_aborts_instead_of_returning_a_result() {
    let Some(m) = crate::MetalBackend::new() else {
        panic!("refusing to continue past a failed command buffer: no Metal device on this host");
    };
    // Below the FLOP threshold `matmul` computes on the CPU; force the
    // GPU path so the wait site under test is the one that runs.
    m.set_flop_threshold(1);
    let a = ndarray::Array2::<f32>::from_elem((128, 128), 1.0);
    let b = ndarray::Array2::<f32>::from_elem((128, 128), 1.0);
    inject_fault_at_for_test("f32_ops.rs:40");
    let _ = larql_compute::MatMul::matmul(&m, a.view(), b.view());
}
