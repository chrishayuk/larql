//! Per-stage GPU attribution for a lowered token.
//!
//! The lowered token is one command buffer of serially dependent kernels,
//! and its cost splits into bytes moved (the bandwidth floor) and a
//! residual. On every container measured 2026-08-19 that residual is
//! roughly constant across representations — ~10 ms on Gemma 4 26B-A4B
//! whether attention is f16 or NVFP4, ~8 ms on gpt-oss — which is the
//! signature of per-kernel fixed cost rather than bytes. Closing it needs
//! to know *which* kernels carry it, and this GPU only exposes timestamps
//! at **stage (encoder) boundaries**: `MTLCounterSamplingPointAtDispatchBoundary`
//! is unsupported on Apple silicon (`examples/counter_probe.rs`).
//!
//! So the lowering encodes through a [`StageEncoders`] seam instead of a
//! bare encoder. Production hands in [`SingleEncoder`] — one encoder for
//! the whole token, `stage()` is a label and nothing else, scheduling
//! unchanged. Profiling hands in [`StageProfiler`], which opens a fresh
//! sampled encoder each time the stage changes and resolves the GPU
//! timestamps after completion.
//!
//! ## What the numbers mean
//!
//! - Encoder boundaries themselves are free (`examples/lowered_dispatch_floor.rs`:
//!   one encoder per dependent dispatch runs *no slower* than one encoder
//!   for all of them). Sampling is not: each sampled boundary drains the
//!   pipeline and costs ~15 us (`examples/counter_stage_probe.rs`). The
//!   report therefore carries its own distortion — the profiled token's
//!   GPU span against the unprofiled one — and a stage's span is an
//!   upper bound inflated by at most a few microseconds of drain.
//! - Units are nanoseconds: the sum of samples reproduces
//!   `GPUEndTime - GPUStartTime` (the probe checks this).

use std::collections::BTreeMap;

use metal::foreign_types::ForeignTypeRef;
use metal::{CommandBuffer, ComputeCommandEncoderRef};

/// A class of work inside a lowered token. Ordered as the stack encodes
/// them, so a report reads top to bottom like the layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Pre-attention RMS norm.
    AttnNorm,
    /// Q/K/V (and gate) projections with their biases.
    AttnProj,
    /// Per-head QK/V norms, query scale, RoPE.
    AttnQkOps,
    /// Attention over the KV cache.
    AttnCore,
    /// Sigmoid gate, output projection + bias, post-norm, residual. On
    /// the single-position path the projection itself is [`Stage::AttnOProj`]
    /// and this keeps the glue around it.
    AttnOut,
    /// The attention output projection GEMV alone (single-position path).
    AttnOProj,
    /// Dense gated FFN: pre-norm, gate/up, activation, down, post-norm,
    /// residual. On the single-position dense path the projections are
    /// [`Stage::FfnGateUp`] and [`Stage::FfnDown`] and this keeps the
    /// glue: pre-norm, activation, post-norm, residual.
    DenseFfn,
    /// The dense FFN's gate and up projection GEMV(s) alone.
    FfnGateUp,
    /// The dense FFN's down projection GEMV alone (with the residual when
    /// the plan folds it into the write).
    FfnDown,
    /// A routed FFN encoded by the served descriptor path as one block
    /// (gpt-oss) — router, experts and combine together.
    RoutedFfn,
    /// Hybrid layers: the norms feeding router and experts.
    FfnNorms,
    /// Router logits and top-k selection.
    Router,
    /// Expert gather, expert matvecs, weighted combine.
    Experts,
    /// Hybrid layers: post-experts norm, branch sum, post-FFN norm,
    /// residual, layer scale.
    FfnOut,
    /// Layer-output copies for `--dump-layers`.
    Checkpoint,
    /// Final norm, output projection, softcap.
    Head,
}

impl Stage {
    /// Every stage, in encode order.
    pub const ALL: [Stage; 16] = [
        Stage::AttnNorm,
        Stage::AttnProj,
        Stage::AttnQkOps,
        Stage::AttnCore,
        Stage::AttnOut,
        Stage::AttnOProj,
        Stage::DenseFfn,
        Stage::FfnGateUp,
        Stage::FfnDown,
        Stage::RoutedFfn,
        Stage::FfnNorms,
        Stage::Router,
        Stage::Experts,
        Stage::FfnOut,
        Stage::Checkpoint,
        Stage::Head,
    ];

    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Stage::AttnNorm => "attn.norm",
            Stage::AttnProj => "attn.proj",
            Stage::AttnQkOps => "attn.qk_ops",
            Stage::AttnCore => "attn.core",
            Stage::AttnOut => "attn.out",
            Stage::AttnOProj => "attn.o_proj",
            Stage::DenseFfn => "ffn.dense",
            Stage::FfnGateUp => "ffn.gate_up",
            Stage::FfnDown => "ffn.down",
            Stage::RoutedFfn => "ffn.routed",
            Stage::FfnNorms => "ffn.norms",
            Stage::Router => "ffn.router",
            Stage::Experts => "ffn.experts",
            Stage::FfnOut => "ffn.out",
            Stage::Checkpoint => "checkpoint",
            Stage::Head => "head",
        }
    }
}

/// Where a lowering's dispatches go.
///
/// `stage()` returns the encoder the next dispatches must use. The
/// production implementation always returns the same one; the profiler
/// may return a new encoder, so callers must not hold an encoder across
/// a `stage()` call.
pub trait StageEncoders {
    fn stage(&mut self, stage: Stage) -> &ComputeCommandEncoderRef;
}

/// Production: one encoder for the whole token. Stage marks are labels.
pub struct SingleEncoder<'a>(pub &'a ComputeCommandEncoderRef);

impl StageEncoders for SingleEncoder<'_> {
    fn stage(&mut self, _stage: Stage) -> &ComputeCommandEncoderRef {
        self.0
    }
}

/// Stage-granular profiler: one sampled encoder per run of a stage.
pub struct StageProfiler {
    cmd: CommandBuffer,
    sample_buffer: metal::CounterSampleBuffer,
    capacity: usize,
    /// `(stage, sample index of its start)` in encode order; the end
    /// sample is the next index.
    runs: Vec<(Stage, usize)>,
    open: Option<metal::ComputeCommandEncoder>,
    current: Option<Stage>,
    /// Stages that were requested after the sample buffer filled; they
    /// still execute (on the last open encoder) but are not attributed.
    overflowed: usize,
}

/// A resolved profile: nanoseconds per stage, summed over the token.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageProfile {
    /// Total sampled span per stage, in nanoseconds.
    pub stage_ns: BTreeMap<Stage, u64>,
    /// Encoder runs per stage.
    pub stage_runs: BTreeMap<Stage, u32>,
    /// First sample to last sample, in nanoseconds — the profiled token's
    /// GPU span as the counters saw it.
    pub span_ns: u64,
    /// Stage requests that did not fit the sample buffer.
    pub overflowed: usize,
}

impl StageProfile {
    /// Sum of all stage spans, in nanoseconds.
    pub fn attributed_ns(&self) -> u64 {
        self.stage_ns.values().sum()
    }

    /// Time between sampled runs — the sampling drain plus whatever the
    /// GPU did outside any stage — in nanoseconds.
    pub fn gap_ns(&self) -> u64 {
        self.span_ns.saturating_sub(self.attributed_ns())
    }

    /// Accumulate another token's profile into this one.
    pub fn add(&mut self, other: &StageProfile) {
        for (s, ns) in &other.stage_ns {
            *self.stage_ns.entry(*s).or_insert(0) += ns;
        }
        for (s, n) in &other.stage_runs {
            *self.stage_runs.entry(*s).or_insert(0) += n;
        }
        self.span_ns += other.span_ns;
        self.overflowed += other.overflowed;
    }
}

impl StageProfiler {
    /// A profiler over `cmd` able to attribute up to `max_runs` stage
    /// runs. `None` if the device exposes no timestamp counter set; see
    /// [`StageProfiler::try_new`] for the reason.
    pub fn new(device: &metal::Device, cmd: CommandBuffer, max_runs: usize) -> Option<Self> {
        Self::try_new(device, cmd, max_runs).ok()
    }

    /// As [`StageProfiler::new`], naming the refusal: no timestamp counter
    /// set, no stage-boundary sampling, or the sample buffer itself (the
    /// device caps it — 4096 samples on an M3 Max).
    pub fn try_new(
        device: &metal::Device,
        cmd: CommandBuffer,
        max_runs: usize,
    ) -> Result<Self, String> {
        // Order matters, and the counter-set probe must be nil-safe: on
        // GitHub's macos-14 runners the paravirtualized GPU can return a
        // nil `counterSets` array, which `metal-rs`'s accessor derefs —
        // a non-unwinding abort, not a catchable panic. Check the
        // sampling capability (a plain bool) first, then read the array
        // through a raw nil check.
        if !device.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary) {
            return Err("device does not sample counters at stage boundaries".into());
        }
        let set = nil_safe_counter_sets(device)
            .into_iter()
            .find(|s| s.name() == TIMESTAMP_COUNTER_SET)
            .ok_or_else(|| format!("device exposes no `{TIMESTAMP_COUNTER_SET}` counter set"))?;
        let desc = metal::CounterSampleBufferDescriptor::new();
        desc.set_counter_set(&set);
        desc.set_sample_count((2 * max_runs) as u64);
        desc.set_storage_mode(metal::MTLStorageMode::Shared);
        let sample_buffer = device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .map_err(|e| format!("timestamp sample buffer ({} samples): {e}", 2 * max_runs))?;
        Ok(Self {
            cmd,
            sample_buffer,
            capacity: max_runs,
            runs: Vec::with_capacity(max_runs),
            open: None,
            current: None,
            overflowed: 0,
        })
    }

    fn close_open(&mut self) {
        if let Some(enc) = self.open.take() {
            enc.end_encoding();
        }
    }

    /// End the encoding side: closes the open encoder and returns the
    /// command buffer for the caller to commit and wait on, plus the
    /// handle that resolves the samples afterwards.
    pub fn finish(mut self) -> (CommandBuffer, StageSamples) {
        self.close_open();
        let StageProfiler {
            cmd,
            sample_buffer,
            runs,
            overflowed,
            ..
        } = self;
        (
            cmd,
            StageSamples {
                sample_buffer,
                runs,
                overflowed,
            },
        )
    }
}

/// The timestamp counter set every Apple GPU exposes.
const TIMESTAMP_COUNTER_SET: &str = "timestamp";

/// `device.counter_sets()` with a nil guard: `metal-rs` 0.29 dereferences
/// a nil `counterSets` NSArray (seen on paravirtualized CI GPUs).
fn nil_safe_counter_sets(device: &metal::Device) -> Vec<metal::CounterSet> {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    let raw: *mut objc::runtime::Object = device.as_ptr() as *mut _;
    // SAFETY: `counterSets` exists on MTLDevice and returns NSArray<...>
    // or nil; nil is checked before any element access.
    let arr: *mut objc::runtime::Object = unsafe { msg_send![raw, counterSets] };
    if arr.is_null() {
        return Vec::new();
    }
    device.counter_sets()
}

impl StageEncoders for StageProfiler {
    fn stage(&mut self, stage: Stage) -> &ComputeCommandEncoderRef {
        if self.current == Some(stage) && self.open.is_some() {
            return self.open.as_deref().expect("open encoder");
        }
        if self.runs.len() >= self.capacity {
            // Out of samples: keep executing, on an *unsampled* encoder so
            // the last attributed stage is not polluted by work that is
            // not its own; count the loss.
            self.overflowed += 1;
            if self.current.is_some() || self.open.is_none() {
                self.close_open();
                self.open = Some(self.cmd.new_compute_command_encoder().to_owned());
                self.current = None;
            }
            return self.open.as_deref().expect("open encoder");
        }
        self.close_open();
        let idx = 2 * self.runs.len();
        // The pass descriptor is autoreleased and retains the sample
        // buffer; without a pool per stage those pile up for the life of
        // the thread and the device refuses new sample buffers after a
        // few dozen ("Cannot allocate sample buffer" at token ~33). The
        // encoder is retained explicitly, so it survives the pool.
        let enc = objc::rc::autoreleasepool(|| {
            let pass = metal::ComputePassDescriptor::new();
            let att = pass
                .sample_buffer_attachments()
                .object_at(0)
                .expect("compute pass sample attachment 0");
            att.set_sample_buffer(&self.sample_buffer);
            att.set_start_of_encoder_sample_index(idx as u64);
            att.set_end_of_encoder_sample_index(idx as u64 + 1);
            self.cmd
                .compute_command_encoder_with_descriptor(pass)
                .to_owned()
        });
        self.open = Some(enc);
        self.current = Some(stage);
        self.runs.push((stage, idx));
        self.open.as_deref().expect("open encoder")
    }
}

/// The sampled stage runs of one committed token, resolvable once the
/// command buffer has completed.
pub struct StageSamples {
    sample_buffer: metal::CounterSampleBuffer,
    runs: Vec<(Stage, usize)>,
    overflowed: usize,
}

impl StageSamples {
    /// Resolve the samples into per-stage nanoseconds. Call only after
    /// the command buffer has completed; `None` if the device returned no
    /// data.
    pub fn resolve(&self) -> Option<StageProfile> {
        let n = 2 * self.runs.len();
        let ts = resolve_timestamps(&self.sample_buffer, n)?;
        Some(profile_from_samples(&self.runs, &ts, self.overflowed))
    }
}

/// Fold `(stage, start index)` runs over a timestamp array into a
/// profile. Pure, so the arithmetic is testable without a device.
pub fn profile_from_samples(
    runs: &[(Stage, usize)],
    ts: &[u64],
    overflowed: usize,
) -> StageProfile {
    let mut p = StageProfile {
        overflowed,
        ..Default::default()
    };
    let mut first: Option<u64> = None;
    let mut last: u64 = 0;
    for &(stage, idx) in runs {
        let (Some(&s), Some(&e)) = (ts.get(idx), ts.get(idx + 1)) else {
            continue;
        };
        *p.stage_ns.entry(stage).or_insert(0) += e.saturating_sub(s);
        *p.stage_runs.entry(stage).or_insert(0) += 1;
        first.get_or_insert(s);
        last = last.max(e);
    }
    p.span_ns = last.saturating_sub(first.unwrap_or(last));
    p
}

/// `GPUEndTime - GPUStartTime` of a completed command buffer, in
/// milliseconds — the whole token's GPU span, sampling included.
pub fn gpu_span_ms(cmd: &metal::CommandBufferRef) -> f64 {
    use objc::{msg_send, sel, sel_impl};
    let raw: *mut objc::runtime::Object = cmd.as_ptr() as *mut _;
    // SAFETY: both selectors exist on MTLCommandBuffer and return a
    // CFTimeInterval; the buffer has completed, so they are populated.
    unsafe {
        let start: f64 = msg_send![raw, GPUStartTime];
        let end: f64 = msg_send![raw, GPUEndTime];
        (end - start) * 1e3
    }
}

/// `resolveCounterRange:` is not bound by `metal` 0.29; read the
/// `MTLCounterResultTimestamp` array (one `u64` each) by hand.
fn resolve_timestamps(sb: &metal::CounterSampleBufferRef, n: usize) -> Option<Vec<u64>> {
    use objc::{msg_send, sel, sel_impl};
    let sbp: *mut objc::runtime::Object = sb.as_ptr() as *mut _;
    let range = metal::NSRange::new(0, n as u64);
    // SAFETY: `resolveCounterRange:` exists on MTLCounterSampleBuffer and
    // returns an autoreleased NSData (or nil); `bytes`/`length` are read
    // inside the pool that owns it and copied out before it drains.
    objc::rc::autoreleasepool(|| unsafe {
        let data: *mut objc::runtime::Object = msg_send![sbp, resolveCounterRange: range];
        if data.is_null() {
            return None;
        }
        let bytes: *const u8 = msg_send![data, bytes];
        let length: usize = msg_send![data, length];
        if bytes.is_null() || length < n * std::mem::size_of::<u64>() {
            return None;
        }
        let slice = std::slice::from_raw_parts(bytes as *const u64, n);
        Some(slice.to_vec())
    })
}

#[cfg(test)]
mod tests;
