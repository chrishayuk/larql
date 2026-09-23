//! Per-backend accounting of the command buffers `MatMul` calls submit.
//!
//! A device call's wall time splits three ways, and each needs a different
//! fix: host work around the submission, the wait between `commit` and
//! completion that is not GPU work, and the GPU's own execution. The
//! caller's wall time gives the first. This records the other two for
//! every buffer committed through [`MetalBackend::commit_and_wait`].
//!
//! Scoped to one backend instance, never a process-global ledger, so two
//! backends (or two tests) cannot read each other's submissions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use larql_compute::SubmissionClock;
use metal::foreign_types::ForeignTypeRef;
use metal::CommandBufferRef;
use objc::{msg_send, sel, sel_impl};

use crate::MetalBackend;

/// Seconds (`CFTimeInterval`) to nanoseconds.
const NANOS_PER_SECOND: f64 = 1e9;

/// Cumulative counters behind [`SubmissionClock`].
#[derive(Debug, Default)]
pub(crate) struct SubmissionCounters {
    submissions: AtomicU64,
    commit_to_done_nanos: AtomicU64,
    gpu_nanos: AtomicU64,
}

impl SubmissionCounters {
    pub(crate) fn snapshot(&self) -> SubmissionClock {
        SubmissionClock {
            submissions: self.submissions.load(Ordering::Relaxed),
            commit_to_done_nanos: self.commit_to_done_nanos.load(Ordering::Relaxed),
            gpu_nanos: self.gpu_nanos.load(Ordering::Relaxed),
        }
    }

    fn record(&self, commit_to_done_nanos: u64, gpu_nanos: u64) {
        self.submissions.fetch_add(1, Ordering::Relaxed);
        self.commit_to_done_nanos
            .fetch_add(commit_to_done_nanos, Ordering::Relaxed);
        self.gpu_nanos.fetch_add(gpu_nanos, Ordering::Relaxed);
    }
}

/// `GPUStartTime` and `GPUEndTime` of a completed command buffer, in
/// seconds. metal-rs 0.29 does not expose them, so they are read through
/// `objc`. Both are valid only once the buffer has completed.
pub(crate) fn gpu_times(cmd: &CommandBufferRef) -> (f64, f64) {
    let raw: *mut objc::runtime::Object = cmd.as_ptr() as *mut _;
    // SAFETY: both selectors exist on MTLCommandBuffer and return
    // CFTimeInterval (double); callers pass a completed buffer.
    unsafe {
        let start: f64 = msg_send![raw, GPUStartTime];
        let end: f64 = msg_send![raw, GPUEndTime];
        (start, end)
    }
}

/// The GPU's own execution span of a completed buffer, in nanoseconds.
/// A buffer the GPU never started reports zero rather than a negative.
fn gpu_span_nanos(cmd: &CommandBufferRef) -> u64 {
    let (start, end) = gpu_times(cmd);
    ((end - start).max(0.0) * NANOS_PER_SECOND) as u64
}

impl MetalBackend {
    /// Commit `cmd`, wait for it, refuse on failure, and record where the
    /// wait went. The one commit-and-wait for `MatMul` calls, so every
    /// submission they make is counted and none is inferred.
    pub(crate) fn commit_and_wait(&self, cmd: &CommandBufferRef, site: &'static str) {
        let committed = Instant::now();
        cmd.commit();
        crate::cb_status::wait_or_abort(cmd, site);
        let commit_to_done = committed.elapsed().as_nanos() as u64;
        self.submission_counters
            .record(commit_to_done, gpu_span_nanos(cmd));
    }
}
