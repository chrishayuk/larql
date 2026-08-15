//! The time half of the BW10 ledger, and the join that turns "bytes
//! moved" into "time attributable to moving them".
//!
//! # These fields are NOT a partition
//!
//! Host wait contains GPU execution; I/O wait sits inside host time. The
//! only true identity here is
//!
//! ```text
//! wall = gpu_busy + gpu_bubble + host_outside_gpu
//! ```
//!
//! where `host_outside_gpu` is a RESIDUAL, not a measurement. Everything
//! else is reported "of which". A decomposition that silently double-
//! counts is worse than none, because it sums to something plausible.

use super::regime::TierBandwidth;

/// Measured time terms for one token (or one aggregation window).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimeAttribution {
    /// End-to-end wall time for the window.
    pub wall_ms: f64,
    /// Sum of GPU command-buffer execution windows.
    pub gpu_busy_ms: f64,
    /// GPU idle between consecutive command buffers — scheduling loss.
    /// This is the term S2 removed (24 starvation bubbles → 1 boundary).
    pub gpu_bubble_ms: f64,
    /// Blocking time in storage reads / major faults. OVERLAPS host time;
    /// reported "of which", never added.
    pub io_wait_ms: f64,
    /// Host time spent blocked in `wait_until_completed`. Contains GPU
    /// execution — reported "of which", never added.
    pub host_wait_ms: f64,
    /// A sampler reported `host_wait_ms`. Without it a zero means "not
    /// instrumented on this path", not "the host never blocked" — the
    /// same distinction the byte side draws for reuse and prefetch.
    pub host_wait_reported: bool,
    /// A sampler reported `io_wait_ms`. A resident run's true value IS
    /// zero, but only a registered sampler can say so.
    pub io_wait_reported: bool,
}

impl TimeAttribution {
    /// Wall time in neither a GPU execution window nor an inter-buffer
    /// gap: host work at the token boundary, before the first buffer or
    /// after the last. A residual by construction.
    pub fn host_outside_gpu_ms(&self) -> f64 {
        (self.wall_ms - self.gpu_busy_ms - self.gpu_bubble_ms).max(0.0)
    }

    /// Fraction of the window in which the GPU was executing.
    pub fn gpu_occupancy(&self) -> Option<f64> {
        (self.wall_ms > 0.0).then(|| self.gpu_busy_ms / self.wall_ms)
    }

    /// Sum of two windows, for aggregating tokens into a steady state.
    pub fn add(&self, other: &TimeAttribution) -> TimeAttribution {
        TimeAttribution {
            wall_ms: self.wall_ms + other.wall_ms,
            gpu_busy_ms: self.gpu_busy_ms + other.gpu_busy_ms,
            gpu_bubble_ms: self.gpu_bubble_ms + other.gpu_bubble_ms,
            io_wait_ms: self.io_wait_ms + other.io_wait_ms,
            host_wait_ms: self.host_wait_ms + other.host_wait_ms,
            host_wait_reported: self.host_wait_reported || other.host_wait_reported,
            io_wait_reported: self.io_wait_reported || other.io_wait_reported,
        }
    }

    /// Divide every term by `n`, turning an aggregate into a per-token
    /// mean. `n == 0` yields all zeros rather than NaN.
    pub fn per(&self, n: usize) -> TimeAttribution {
        if n == 0 {
            return TimeAttribution::default();
        }
        let d = n as f64;
        TimeAttribution {
            wall_ms: self.wall_ms / d,
            gpu_busy_ms: self.gpu_busy_ms / d,
            gpu_bubble_ms: self.gpu_bubble_ms / d,
            io_wait_ms: self.io_wait_ms / d,
            host_wait_ms: self.host_wait_ms / d,
            host_wait_reported: self.host_wait_reported,
            io_wait_reported: self.io_wait_reported,
        }
    }
}

/// Declared sustainable bandwidth per tier. A tier with no declared
/// bandwidth cannot have its transfer floor computed — the ledger reports
/// that as an uncomputable term rather than assuming a number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rooflines {
    pub dram: Option<TierBandwidth>,
    pub nvme: Option<TierBandwidth>,
    pub network: Option<TierBandwidth>,
}

impl Rooflines {
    /// The common resident case: only DRAM is in play.
    pub fn dram_only(dram: TierBandwidth) -> Self {
        Self {
            dram: Some(dram),
            nvme: None,
            network: None,
        }
    }
}

/// Bytes attributed to moving data, derived by joining the byte counters
/// with the time counters and the declared rooflines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementCost {
    /// Time the measured bytes would take at the DECLARED roofline — the
    /// floor byte movement imposes on this window if nothing else bound.
    pub floor_ms: f64,
    /// Physical bytes whose tier had no declared bandwidth, so they
    /// contributed nothing to `floor_ms`. Non-zero invalidates any
    /// "movement is X% of the token" reading.
    pub bytes_without_roofline: u64,
    /// `floor_ms / wall_ms` — the share of the window that byte movement
    /// could possibly account for. THE number this ledger exists for:
    /// it caps what any byte reduction can claim in latency.
    pub share_of_wall: Option<f64>,
    /// Physical bytes divided by GPU-busy time, in GB/s. Charges ALL GPU
    /// time to streaming, so it is a LOWER bound on the true streaming
    /// rate and an upper bound on attributable inefficiency. Compute-
    /// bound stages inside the window depress it.
    pub implied_stream_gbps: Option<f64>,
    /// `implied_stream_gbps / declared DRAM roofline` — the kernel
    /// efficiency term η. The MXFP4 down-kernel sat at 0.57 here while
    /// Q6_K sat at 0.82, which is why an equal-η byte projection
    /// over-predicted the end-to-end win.
    pub roofline_utilisation: Option<f64>,
    /// Upper bracket on movement's share of the window: GPU-busy time as
    /// a fraction of wall, i.e. every GPU millisecond charged to
    /// streaming.
    ///
    /// The true share lies in `[share_of_wall, gpu_busy_share]`. The
    /// lower end assumes the engine streams at roofline (it does not);
    /// the upper end assumes GPU time is nothing but streaming (it is
    /// not). Reporting one number here would be a guess wearing a
    /// measurement's clothes — and note the two ends differ by exactly
    /// `roofline_utilisation`, so the bracket width IS the kernel
    /// efficiency question.
    pub gpu_busy_share: Option<f64>,
}

/// Bytes per decimal gigabyte, matching [`TierBandwidth`]'s convention.
const BYTES_PER_GB: f64 = 1.0e9;
/// Milliseconds per second.
const MS_PER_S: f64 = 1.0e3;

impl MovementCost {
    /// Join bytes, time and rooflines into the causality terms.
    pub fn derive(
        bytes: &super::bytes::ByteMovement,
        time: &TimeAttribution,
        rooflines: &Rooflines,
    ) -> Self {
        let mut floor_ms = 0.0;
        let mut unroofed = 0u64;
        for (n, bw) in [
            (bytes.dram, rooflines.dram),
            (bytes.nvme, rooflines.nvme),
            (bytes.network, rooflines.network),
        ] {
            match bw {
                Some(bw) => floor_ms += bw.transfer_floor_ms(n),
                None => unroofed += n,
            }
        }
        unroofed += bytes.tier_unattributed();

        let share_of_wall = (time.wall_ms > 0.0).then(|| floor_ms / time.wall_ms);
        let implied_stream_gbps = (time.gpu_busy_ms > 0.0).then(|| {
            (bytes.physical_touched as f64 / BYTES_PER_GB) / (time.gpu_busy_ms / MS_PER_S)
        });
        let roofline_utilisation = match (implied_stream_gbps, rooflines.dram) {
            (Some(g), Some(bw)) if bw.gb_per_s() > 0.0 => Some(g / bw.gb_per_s()),
            _ => None,
        };
        let gpu_busy_share = (time.wall_ms > 0.0).then(|| time.gpu_busy_ms / time.wall_ms);

        Self {
            floor_ms,
            bytes_without_roofline: unroofed,
            share_of_wall,
            implied_stream_gbps,
            roofline_utilisation,
            gpu_busy_share,
        }
    }

    /// Latency a byte saving of `saved_bytes` can claim in this window,
    /// priced at the OBSERVED streaming rate rather than the roofline.
    ///
    /// This is the guard against the §4.11 class of error in its
    /// forward-looking form: a proposal that will remove N bytes gets its
    /// honest ms here, and if that number is small relative to the window
    /// the proposal is not a latency lever no matter how large N looks.
    pub fn predicted_saving_ms(&self, saved_bytes: u64) -> Option<f64> {
        let gbps = self.implied_stream_gbps.filter(|g| *g > 0.0)?;
        Some((saved_bytes as f64 / BYTES_PER_GB) / gbps * MS_PER_S)
    }
}

#[cfg(test)]
#[path = "tests/timing.rs"]
mod tests;
