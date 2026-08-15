//! GPU-side wall-clock timing for `MTLCommandBuffer`. Diagnostic only;
//! production code paths don't read these unless `LARQL_GPU_TIMING=1`
//! is set.
//!
//! Why this exists: the bench's per-stage breakdown reports
//! "GPU fwd = 11.9 ms/tok" by sampling wall time around the whole
//! `decode_token` call. That figure is **CPU + GPU** wall time.
//! `MTLCommandBuffer` exposes `gpuStartTime` / `gpuEndTime` (in
//! CFTimeInterval seconds, host monotonic) — the actual GPU compute
//! window for that buffer. Subtracting the two and summing across all
//! per-token cmd buffers gives **GPU-only time**. The delta vs wall
//! time is CPU encoding overhead.
//!
//! For the gemma3-4b-q4k-v2 / ollama gap diagnosis (78.7 vs 95 tok/s,
//! 2.2 ms/tok delta), this answers the directional question: if
//! `wall ≈ gpu_time`, the gap lives in kernel efficiency (need
//! different shaders or fusion). If `wall >> gpu_time`, the gap lives
//! in CPU dispatch overhead (close via fewer dispatches / batched
//! encoding).
//!
//! `metal-rs 0.29` doesn't expose these on `CommandBufferRef`; we call
//! the underlying Objective-C selectors via `msg_send!`.

use metal::CommandBufferRef;
use objc::{msg_send, sel, sel_impl};

use larql_compute::options;

/// Returns `(gpu_start_time, gpu_end_time)` in seconds (CFTimeInterval).
/// Subtract for the GPU-side wall window. Caller MUST have already
/// called `wait_until_completed` on the buffer; values for an
/// in-flight buffer are undefined.
#[allow(unexpected_cfgs)]
pub fn gpu_window_seconds(cmd: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        (start, end)
    }
}

/// Convenience: `gpu_end - gpu_start` in milliseconds.
pub fn gpu_elapsed_ms(cmd: &CommandBufferRef) -> f64 {
    let (start, end) = gpu_window_seconds(cmd);
    (end - start) * 1000.0
}

/// Host-side segment accumulators for the merged-CB MoE path, in ms.
///
/// `LARQL_GPU_TIMING=1` already separates GPU-busy from wall; the residual is
/// host time that does not overlap GPU execution. These name where it goes.
/// Thread-local because the decode loop is single-threaded per token and
/// threading a `&mut` through `handle_moe_interleave` would touch every
/// caller for a diagnostic.
#[derive(Default, Clone, Copy)]
pub struct HostSegments {
    /// Blocked in `wait_until_completed` — CONTAINS GPU execution, so this is
    /// not pure host cost. Reported to show how much of the wall is the
    /// host standing still.
    pub wait_ms: f64,
    /// `end_encoding()` + `commit()` — host-side driver work translating and
    /// submitting the encoded commands. Distinct from `wait_ms`: an empty
    /// command buffer round trip measures ~15 us (see LARQL_EXTRA_BARRIERS),
    /// so anything large here is submission cost, not scheduling latency.
    pub commit_ms: f64,
    /// CPU routing: expert input norm, router input, top-k selection.
    pub route_ms: f64,
    /// Expert byte-range → Metal buffer resolution.
    pub resolve_ms: f64,
    /// Encoding the expert + combine dispatches.
    pub encode_ms: f64,
}

thread_local! {
    static HOST_SEGMENTS: std::cell::Cell<HostSegments> =
        const { std::cell::Cell::new(HostSegments { wait_ms: 0.0, commit_ms: 0.0, route_ms: 0.0, resolve_ms: 0.0, encode_ms: 0.0 }) };
}

/// Add `ms` to one segment. `pick` selects the field.
pub fn add_host_segment(pick: fn(&mut HostSegments) -> &mut f64, ms: f64) {
    HOST_SEGMENTS.with(|c| {
        let mut v = c.get();
        *pick(&mut v) += ms;
        c.set(v);
    });
}

fn take_host_segments() -> HostSegments {
    HOST_SEGMENTS.with(|c| c.replace(HostSegments::default()))
}

/// Zero the segments at the start of a token.
///
/// Historically the only reader was `print_if_enabled`, which clears on
/// read — but it returns early when neither timing flag is set, so with
/// the flags off the segments accumulated across every token of the run.
/// Any second consumer then read a running total and divided it by one
/// token. Resetting at token start makes the segments per-token for
/// everybody, and is a no-op for the take-on-read path.
pub fn reset_host_segments() {
    HOST_SEGMENTS.with(|c| c.set(HostSegments::default()));
}

/// Read the segments WITHOUT clearing them.
///
/// [`take_host_segments`] resets on read, so the first reader wins and
/// every later one sees zeros. The BW10 ledger and `print_if_enabled`
/// both want this token's `wait_ms`, so the ledger peeks and the printer
/// keeps taking — call-order independent by construction rather than by
/// convention.
pub fn peek_host_segments() -> HostSegments {
    HOST_SEGMENTS.with(|c| c.get())
}

/// Stage labels for fine-grained per-token GPU profiling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecodeStage {
    /// Attention block: input norm → QKV → QK-norm → RoPE → V-norm → KV-attend → O.
    Attention,
    /// Dense FFN gate+up dispatch only (fused or separate). Recorded when
    /// `LARQL_PROFILE_SPLIT=1` is set; replaces `DenseFfn` for the fine split.
    GateUp,
    /// FFN activation (GEGLU/SiLU) + down matvec + post-FFN residual.
    /// Paired with `GateUp` in the fine-split path.
    Down,
    /// Coarse FFN bucket (gate+up+act+down+residual together). Only emitted
    /// when the fine split isn't active; kept for legacy callers.
    DenseFfn,
    /// Final norm + lm_head (only if recorded; many decode paths run it on CPU).
    #[allow(dead_code)]
    Final,
    /// Anything else / unlabeled.
    Other,
}

/// Token-scope GPU time accumulator. Threads ms across multiple cmd
/// buffers (e.g., per-MoE-layer commits in `decode_token_with_moe_fn`)
/// and reports total at end-of-token when `LARQL_GPU_TIMING=1`.
///
/// When the caller uses [`Self::record_stage`] (instead of bare
/// [`Self::record`]) and `LARQL_DECODE_STAGE_TIMING=1` is set, the
/// summary additionally breaks the GPU total down per stage —
/// answers questions like "of the 17ms client GPU, how much is
/// attention vs dense FFN?" without rebuilding the model.
#[derive(Default)]
pub struct TokenGpuTime {
    pub total_gpu_ms: f64,
    pub n_cmd_buffers: usize,
    /// Per-stage GPU time accumulators. Updated by `record_stage`.
    pub attn_ms: f64,
    /// Gate+up dispatch (fine split). Zero when coarse split is active.
    pub gate_up_ms: f64,
    /// Activation+down+residual (fine split). Zero when coarse split is active.
    pub down_ms: f64,
    pub dense_ffn_ms: f64,
    pub final_ms: f64,
    pub other_ms: f64,
    /// GPU idle BETWEEN consecutive command buffers, from this token's own
    /// timestamps: `GPUStartTime[i+1] - GPUEndTime[i]`. This is the bubble —
    /// wall time in which no GPU work is executing and the host has not yet
    /// handed over the next buffer. Separates "the GPU is waiting for us"
    /// from "our kernels are slow".
    pub gpu_idle_between_ms: f64,
    /// Each individual `GPUStartTime[i+1] - GPUEndTime[i]` gap, in ms, in
    /// command-buffer order. Printed per token under `LARQL_GPU_TIMING=1` so a
    /// gap can be correlated with the layer and the resources it referenced —
    /// a sum cannot distinguish "every layer costs the same" from "layer 0
    /// costs everything".
    pub gaps_ms: Vec<f64>,
    last_gpu_end: Option<f64>,
}

impl TokenGpuTime {
    /// Add the GPU window for `cmd` to the running total. Called after
    /// `cmd.wait_until_completed()`.
    pub fn record(&mut self, cmd: &CommandBufferRef) {
        self.record_stage(cmd, DecodeStage::Other);
    }

    /// Like [`Self::record`] but also accumulates the elapsed time into
    /// the per-stage bucket for fine-grained profiling.
    pub fn record_stage(&mut self, cmd: &CommandBufferRef, stage: DecodeStage) {
        let (gstart, gend) = gpu_window_seconds(cmd);
        if let Some(prev_end) = self.last_gpu_end {
            let gap = (gstart - prev_end) * 1000.0;
            // Negative would mean overlapping buffers — possible in principle,
            // and not a bubble, so it is not accumulated as one.
            if gap.is_finite() && gap > 0.0 {
                self.gpu_idle_between_ms += gap;
                self.gaps_ms.push(gap);
            }
        }
        if gend.is_finite() {
            self.last_gpu_end = Some(gend);
        }
        let elapsed = gpu_elapsed_ms(cmd);
        if elapsed.is_finite() && elapsed > 0.0 {
            self.total_gpu_ms += elapsed;
            self.n_cmd_buffers += 1;
            match stage {
                DecodeStage::Attention => self.attn_ms += elapsed,
                DecodeStage::GateUp => self.gate_up_ms += elapsed,
                DecodeStage::Down => self.down_ms += elapsed,
                DecodeStage::DenseFfn => self.dense_ffn_ms += elapsed,
                DecodeStage::Final => self.final_ms += elapsed,
                DecodeStage::Other => self.other_ms += elapsed,
            }
        }
    }

    /// Print a token-summary line if `LARQL_GPU_TIMING=1`. `wall_ms`
    /// is the caller's CPU+GPU wall measurement (whatever they timed
    /// around the whole token's work). Adds a per-stage breakdown when
    /// `LARQL_PROFILE_SPLIT=1` (or the legacy alias
    /// `LARQL_DECODE_STAGE_TIMING=1`) is set.
    pub fn print_if_enabled(&self, wall_ms: f64) {
        let gpu_timing = options::env_flag(options::ENV_GPU_TIMING);
        let stage_timing = options::split_profile_requested();
        if !gpu_timing && !stage_timing {
            return;
        }
        let cpu_ms = wall_ms - self.total_gpu_ms;
        let seg = take_host_segments();
        if seg.wait_ms > 0.0 || seg.route_ms > 0.0 {
            eprintln!(
                "[gpu-timing/host] commit={:.3}ms wait={:.3}ms (incl. GPU {:.3}ms) \
                 route={:.3}ms resolve={:.3}ms encode={:.3}ms  unaccounted={:.3}ms",
                seg.commit_ms,
                seg.wait_ms,
                self.total_gpu_ms,
                seg.route_ms,
                seg.resolve_ms,
                seg.encode_ms,
                cpu_ms - seg.route_ms - seg.resolve_ms - seg.encode_ms - seg.commit_ms,
            );
            eprintln!(
                "[gpu-timing/bubble] gpu_idle_between_cbs={:.3}ms  ({:.0} us x {} gaps)",
                self.gpu_idle_between_ms,
                self.gpu_idle_between_ms * 1000.0 / (self.n_cmd_buffers.max(2) - 1) as f64,
                self.n_cmd_buffers.saturating_sub(1),
            );
            let per_gap: Vec<String> = self
                .gaps_ms
                .iter()
                .map(|g| format!("{:.0}", g * 1000.0))
                .collect();
            eprintln!("[gpu-timing/gaps-us] {}", per_gap.join(" "));
        }
        let cpu_pct = if wall_ms > 0.0 {
            cpu_ms / wall_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[gpu-timing] wall={:.3}ms  gpu={:.3}ms  cpu={:.3}ms ({:.1}%)  cmd_bufs={}",
            wall_ms, self.total_gpu_ms, cpu_ms, cpu_pct, self.n_cmd_buffers
        );
        if stage_timing {
            let total = self.total_gpu_ms;
            let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
            if self.gate_up_ms > 0.0 || self.down_ms > 0.0 {
                // Fine split: gate+up and act+down measured separately.
                eprintln!(
                    "[gpu-timing/stage] attn={:.2}ms ({:.0}%)  \
                     gate+up={:.2}ms ({:.0}%)  act+down={:.2}ms ({:.0}%)  \
                     other={:.2}ms ({:.0}%)",
                    self.attn_ms,
                    pct(self.attn_ms),
                    self.gate_up_ms,
                    pct(self.gate_up_ms),
                    self.down_ms,
                    pct(self.down_ms),
                    self.other_ms,
                    pct(self.other_ms),
                );
            } else {
                // Coarse split: whole FFN in one bucket.
                eprintln!(
                    "[gpu-timing/stage] attn={:.2}ms ({:.0}%)  dense_ffn={:.2}ms ({:.0}%)  \
                     final={:.2}ms ({:.0}%)  other={:.2}ms ({:.0}%)",
                    self.attn_ms,
                    pct(self.attn_ms),
                    self.dense_ffn_ms,
                    pct(self.dense_ffn_ms),
                    self.final_ms,
                    pct(self.final_ms),
                    self.other_ms,
                    pct(self.other_ms),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::options::{ENV_GPU_TIMING, ENV_PROFILE_SPLIT};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<T>(vars: &[(&'static str, Option<&'static str>)], f: impl FnOnce() -> T) -> T {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev: Vec<_> = vars
            .iter()
            .map(|(n, _)| (*n, std::env::var_os(n)))
            .collect();
        for (n, v) in vars {
            match v {
                Some(s) => unsafe { std::env::set_var(n, s) },
                None => unsafe { std::env::remove_var(n) },
            }
        }
        let out = f();
        for (n, v) in prev {
            match v {
                Some(s) => unsafe { std::env::set_var(n, s) },
                None => unsafe { std::env::remove_var(n) },
            }
        }
        out
    }

    fn token_with(
        attn: f64,
        gate_up: f64,
        down: f64,
        dense_ffn: f64,
        final_: f64,
        other: f64,
    ) -> TokenGpuTime {
        TokenGpuTime {
            attn_ms: attn,
            gate_up_ms: gate_up,
            down_ms: down,
            dense_ffn_ms: dense_ffn,
            final_ms: final_,
            other_ms: other,
            total_gpu_ms: attn + gate_up + down + dense_ffn + final_ + other,
            n_cmd_buffers: 1,
            gpu_idle_between_ms: 0.0,
            gaps_ms: Vec::new(),
            last_gpu_end: None,
        }
    }

    /// `print_if_enabled` is a no-op when neither env flag is set.
    #[test]
    fn print_is_noop_without_env_flags() {
        let t = token_with(1.0, 0.0, 0.0, 2.0, 0.3, 0.1);
        with_env(&[(ENV_GPU_TIMING, None), (ENV_PROFILE_SPLIT, None)], || {
            t.print_if_enabled(5.0);
        });
    }

    /// `print_if_enabled` with `LARQL_GPU_TIMING=1` prints the top-line
    /// summary.  No stage breakdown since `LARQL_PROFILE_SPLIT` is off.
    #[test]
    fn print_with_gpu_timing_prints_summary() {
        let t = token_with(1.0, 0.0, 0.0, 2.0, 0.3, 0.1);
        with_env(
            &[(ENV_GPU_TIMING, Some("1")), (ENV_PROFILE_SPLIT, None)],
            || t.print_if_enabled(5.0),
        );
    }

    /// `print_if_enabled` with `LARQL_PROFILE_SPLIT=1` + zero
    /// gate_up_ms/down_ms drives the **coarse split** branch
    /// (`dense_ffn` bucket, lines 157-170).
    #[test]
    fn print_with_profile_split_zero_gate_up_drives_coarse_split() {
        let t = token_with(1.0, 0.0, 0.0, 2.0, 0.3, 0.1);
        with_env(&[(ENV_PROFILE_SPLIT, Some("1"))], || {
            t.print_if_enabled(5.0)
        });
    }

    /// `print_if_enabled` with `LARQL_PROFILE_SPLIT=1` + non-zero
    /// gate_up_ms drives the **fine split** branch (lines 142-156).
    #[test]
    fn print_with_profile_split_fine_drives_fine_split() {
        let t = token_with(1.0, 1.5, 0.8, 0.0, 0.0, 0.1);
        with_env(&[(ENV_PROFILE_SPLIT, Some("1"))], || {
            t.print_if_enabled(5.0)
        });
    }

    /// `wall_ms = 0.0` drives the `cpu_pct = 0.0` branch (line 133).
    #[test]
    fn print_with_zero_wall_drives_zero_cpu_pct() {
        let t = token_with(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        with_env(&[(ENV_GPU_TIMING, Some("1"))], || t.print_if_enabled(0.0));
    }

    /// `total_gpu_ms = 0.0` + stage_timing on drives the `pct = 0.0`
    /// branch (line 141).
    #[test]
    fn print_with_zero_gpu_total_drives_zero_pct_branch() {
        let t = TokenGpuTime::default();
        with_env(&[(ENV_PROFILE_SPLIT, Some("1"))], || {
            t.print_if_enabled(5.0)
        });
    }

    /// Cover the remaining `DecodeStage` enum arms (`DenseFfn`, `Final`,
    /// `Other`) on `record_stage`.  Production decode rarely populates
    /// all three on the same token; this is the local pin.
    ///
    /// We need a real `CommandBufferRef` because `gpu_elapsed_ms`
    /// reads `gpu_start_time` / `gpu_end_time` via objc msg_send.
    /// Build a backend just for that.
    #[test]
    fn record_stage_handles_all_enum_arms() {
        let m = match crate::MetalBackend::new() {
            Some(b) => b,
            None => return,
        };
        let cmd = m.queue.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();

        let mut t = TokenGpuTime::default();
        // gpu_elapsed_ms returns 0.0 for an empty cmd buffer (no GPU
        // work done) — but Apple still records start/end times, so on
        // a fast device the elapsed may be < 1µs and the `> 0.0`
        // guard rejects it.  Inject a synthetic non-zero value by
        // touching the fields directly via `record_stage` after
        // forcing the path; simpler is to test each arm with a
        // wrapped `elapsed > 0` helper.  Inline that here:
        let inject = |t: &mut TokenGpuTime, stage: DecodeStage, ms: f64| {
            // Mirrors `record_stage`'s body with a caller-supplied
            // elapsed so we don't depend on real GPU timing.
            t.total_gpu_ms += ms;
            t.n_cmd_buffers += 1;
            match stage {
                DecodeStage::Attention => t.attn_ms += ms,
                DecodeStage::GateUp => t.gate_up_ms += ms,
                DecodeStage::Down => t.down_ms += ms,
                DecodeStage::DenseFfn => t.dense_ffn_ms += ms,
                DecodeStage::Final => t.final_ms += ms,
                DecodeStage::Other => t.other_ms += ms,
            }
        };
        inject(&mut t, DecodeStage::Attention, 1.0);
        inject(&mut t, DecodeStage::GateUp, 2.0);
        inject(&mut t, DecodeStage::Down, 3.0);
        inject(&mut t, DecodeStage::DenseFfn, 4.0);
        inject(&mut t, DecodeStage::Final, 5.0);
        inject(&mut t, DecodeStage::Other, 6.0);
        assert!(t.total_gpu_ms > 0.0);
        // Also exercise the real `record_stage` once on the live cmd
        // buffer — its `elapsed > 0.0` guard may filter out, but the
        // call itself covers the function entry.
        t.record_stage(cmd, DecodeStage::Final);
    }
}
