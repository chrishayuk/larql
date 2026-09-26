//! `--profile`: where a lowered token's GPU time goes, by stage.
//!
//! The lowering's [`StageEncoders`] seam lets the same encode run under a
//! [`StageProfiler`] — one sampled encoder per stage run — so each
//! token's GPU time is attributed to the stage classes of
//! [`Stage`]. This module keeps the ledger across tokens and prints it
//! against the bytes each class reads, so a stage reads as either
//! bandwidth-bound (its GB/s is near the roofline, only bytes can move
//! it) or latency-bound (far below it — fusion, geometry, occupancy).
//!
//! The ledger is honest about its own cost: sampled stage boundaries
//! drain the pipeline (~15 us each), so the profiled token's GPU span is
//! printed next to the stage sum and the reader compares it with an
//! unprofiled run's ms/token.

use larql_compute_metal::lowering::profile::{Stage, StageProfile};

/// Bytes one token reads per stage class — from the resident matrices,
/// not the container, so a policy that quantises at load is priced as
/// executed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StageBytes {
    /// Q, K, V (and gate) projections.
    pub attn_proj: usize,
    /// Output projection.
    pub attn_out: usize,
    pub dense_ffn: usize,
    /// The part of `dense_ffn` the gate and up projections read — priced
    /// on [`Stage::FfnGateUp`] when the lowering split it out.
    pub dense_gate_up: usize,
    /// The part of `dense_ffn` the down projection reads.
    pub dense_down: usize,
    pub experts: usize,
    pub head: usize,
}

impl StageBytes {
    /// Bytes priced on `stage`. When the projections ran as their own
    /// stages (`split_o`, `split_ffn`), their bytes go there and the glue
    /// stage that used to carry them reads none — no byte is priced twice.
    fn for_stage(&self, stage: Stage, split_o: bool, split_ffn: bool) -> Option<usize> {
        match stage {
            Stage::AttnProj => Some(self.attn_proj),
            Stage::AttnOut if split_o => None,
            Stage::AttnOut | Stage::AttnOProj => Some(self.attn_out),
            Stage::DenseFfn if split_ffn => None,
            Stage::DenseFfn => Some(self.dense_ffn),
            Stage::FfnGateUp => Some(self.dense_gate_up),
            Stage::FfnDown => Some(self.dense_down),
            Stage::Experts | Stage::RoutedFfn => Some(self.experts),
            Stage::Head => Some(self.head),
            _ => None,
        }
    }

    pub fn total(&self) -> usize {
        self.attn_proj + self.attn_out + self.dense_ffn + self.experts + self.head
    }
}

/// Accumulated per-token stage profiles plus the whole-token GPU spans.
#[derive(Debug, Default)]
pub(super) struct StageLedger {
    pub profile: StageProfile,
    pub tokens: usize,
    pub gpu_span_ms: f64,
    pub bytes: StageBytes,
}

/// Achieved bandwidth the roofline is judged against: the probe-measured
/// GPU read ceiling on this class of machine (`membw_probe`, M3 Max).
const GPU_READ_CEILING_GB_S: f64 = 367.0;

impl StageLedger {
    pub fn record(&mut self, token: &StageProfile, gpu_span_ms: f64) {
        self.profile.add(token);
        self.tokens += 1;
        self.gpu_span_ms += gpu_span_ms;
    }

    /// Render the ledger as lines for stdout. Pure, so the layout is
    /// testable.
    pub fn render(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.tokens == 0 {
            out.push("profile: no tokens recorded".into());
            return out;
        }
        let n = self.tokens as f64;
        let attributed_ms = self.profile.attributed_ns() as f64 / 1e6 / n;
        let span_ms = self.profile.span_ns as f64 / 1e6 / n;
        let gpu_ms = self.gpu_span_ms / n;
        out.push(format!(
            "stage profile over {} token(s) — per token, GPU time; floor = bytes / {GPU_READ_CEILING_GB_S:.0} GB/s",
            self.tokens
        ));
        out.push(format!(
            "  {:<12} {:>9} {:>6} {:>6} {:>10} {:>9} {:>8}",
            "stage", "ms/tok", "%", "runs", "MB/tok", "GB/s", "floor ms"
        ));
        let split_o = self.profile.stage_ns.contains_key(&Stage::AttnOProj);
        let split_ffn = self.profile.stage_ns.contains_key(&Stage::FfnGateUp);
        for stage in Stage::ALL {
            let Some(ns) = self.profile.stage_ns.get(&stage) else {
                continue;
            };
            let ms = *ns as f64 / 1e6 / n;
            let runs = self.profile.stage_runs.get(&stage).copied().unwrap_or(0) as f64 / n;
            let pct = if attributed_ms > 0.0 {
                100.0 * ms / attributed_ms
            } else {
                0.0
            };
            let (mb, gbs, floor) = match self.bytes.for_stage(stage, split_o, split_ffn) {
                Some(b) if b > 0 => {
                    let gb = b as f64 / 1e9;
                    (
                        format!("{:.1}", gb * 1e3),
                        format!("{:.0}", gb / (ms / 1e3)),
                        format!("{:.2}", gb / GPU_READ_CEILING_GB_S * 1e3),
                    )
                }
                _ => ("-".into(), "-".into(), "-".into()),
            };
            out.push(format!(
                "  {:<12} {:>9.3} {:>5.1}% {:>6.1} {:>10} {:>9} {:>8}",
                stage.label(),
                ms,
                pct,
                runs,
                mb,
                gbs,
                floor
            ));
        }
        let total_floor = self.bytes.total() as f64 / 1e9 / GPU_READ_CEILING_GB_S * 1e3;
        out.push(format!(
            "  {:<12} {:>9.3}        {:>6} {:>10.1} {:>9} {:>8.2}",
            "attributed",
            attributed_ms,
            "",
            self.bytes.total() as f64 / 1e6,
            "",
            total_floor
        ));
        out.push(format!(
            "  sampled span {span_ms:.3} ms/tok (gaps between stages {:.3}); command-buffer GPU span {gpu_ms:.3} ms/tok",
            span_ms - attributed_ms
        ));
        out.push(format!(
            "  residual over byte floor: {:.3} ms/tok — compare with an unprofiled run: sampling drains at each stage boundary",
            attributed_ms - total_floor
        ));
        if self.profile.overflowed > 0 {
            out.push(format!(
                "  WARNING: {} stage run(s) exceeded the sample buffer and ran unattributed",
                self.profile.overflowed
            ));
        }
        out
    }
}
