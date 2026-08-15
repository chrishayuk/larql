//! The regime declaration — BW10's load-bearing guard.
//!
//! A byte reduction is not evidence of a performance win unless byte
//! movement is on the critical path. That sentence exists because this
//! project already paid for its absence: `docs/k3-funnel.md` §4.11 called
//! 11.5 ms of MoE decode "genuine bandwidth" and licensed a whole
//! representation programme as the primary lever, when the path was at
//! 62% GPU occupancy and larql was reading FEWER bytes than a faster
//! engine.
//!
//! So the ledger prints raw counters unconditionally — raw arms are
//! always readable — but refuses to print the derived verdict line until
//! the caller has named the regime the measurement targets. The verdict
//! inherits the regime's assumptions, so the regime is part of the claim,
//! not context around it.

use std::fmt;

/// Which physical constraint the measured workload is operating under.
///
/// This is a claim about the WORKLOAD, not about the machine: the same
/// M3 Max serves gpt-oss resident and K3 cold, and those two answer
/// "does saving bytes save time?" oppositely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// Whole model estate fits in the memory tier the compute reads from.
    /// Critical path = compute + scheduling/bubbles + memory-stall time;
    /// bytes are diagnostic, not the optimisation target. Reducing bytes
    /// helps only to the extent memory stall is on the critical path,
    /// which the ledger's movement share reports directly.
    Resident,
    /// Estate exceeds the fast tier but the working set is schedulable —
    /// residency policy, eviction and prefetch decide the outcome. Both
    /// byte movement AND scheduling are live; neither dominates a priori.
    CapacityConstrained,
    /// Estate cannot be resident; every token pulls new bytes across an
    /// external tier. Here `external_bytes / sustainable_tier_bandwidth`
    /// is a hard lower bound on token time and bytes are existential.
    ColdEstate,
}

impl Regime {
    /// The short banner token printed on every ledger line.
    pub const fn label(self) -> &'static str {
        match self {
            Regime::Resident => "resident",
            Regime::CapacityConstrained => "capacity-constrained",
            Regime::ColdEstate => "cold-estate",
        }
    }

    /// One line stating what a byte reduction is permitted to claim in
    /// this regime. Printed beside the verdict so the reading and the
    /// licence travel together.
    pub const fn byte_claim_licence(self) -> &'static str {
        match self {
            Regime::Resident => {
                "bytes are diagnostic — a byte saving claims latency only up to the movement share"
            }
            Regime::CapacityConstrained => {
                "bytes and scheduling both live — attribute against the measured movement share"
            }
            Regime::ColdEstate => {
                "external bytes / tier bandwidth is a hard floor on token time — bytes are existential"
            }
        }
    }

    /// Parse a regime from a CLI/env token. Deliberately strict: an
    /// unrecognised name yields `None` so the caller refuses rather than
    /// silently defaulting to the permissive reading.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "resident" => Some(Regime::Resident),
            "capacity" | "capacity-constrained" | "capacity_constrained" => {
                Some(Regime::CapacityConstrained)
            }
            "cold" | "cold-estate" | "cold_estate" => Some(Regime::ColdEstate),
            _ => None,
        }
    }
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Sustainable bandwidth of the tier the measured bytes cross, with its
/// provenance attached.
///
/// A roofline is a MEASURED property of one machine and one access
/// pattern, never a constant of the codebase — so this type carries where
/// the number came from and the report prints it. Comparing an implied
/// bandwidth against an unlabelled ceiling is how a GiB/s-vs-GB/s slip
/// turns into a 7.4% phantom effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierBandwidth {
    gb_per_s: f64,
    source: &'static str,
}

/// Attainable DRAM read bandwidth measured on the M3 Max development
/// machine. This is a PROBE RESULT, not a spec figure (spec peak is 400
/// GB/s); it is the banked attainable number this project's roofline
/// arithmetic uses. Re-measure on any other host before quoting it.
pub const M3_MAX_ATTAINABLE_DRAM_GBPS: f64 = 367.0;

/// Bytes per decimal gigabyte. The probe reports decimal GB/s, so the
/// ledger converts with the same convention — mixing GiB/s and GB/s is a
/// 7.4% error, exactly the size of the effects being argued about.
const BYTES_PER_GB: f64 = 1.0e9;

/// Milliseconds per second, for transfer-floor arithmetic.
const MS_PER_S: f64 = 1.0e3;

impl TierBandwidth {
    /// Declare a bandwidth ceiling together with the provenance of the
    /// number. `source` should name the probe or datasheet, not just the
    /// machine.
    pub const fn measured(gb_per_s: f64, source: &'static str) -> Self {
        Self { gb_per_s, source }
    }

    /// The banked M3 Max attainable DRAM figure.
    pub const fn m3_max_dram() -> Self {
        Self::measured(
            M3_MAX_ATTAINABLE_DRAM_GBPS,
            "M3 Max attainable DRAM read probe",
        )
    }

    pub const fn gb_per_s(self) -> f64 {
        self.gb_per_s
    }

    pub const fn source(self) -> &'static str {
        self.source
    }

    /// Milliseconds this tier needs to move `bytes` — the floor byte
    /// movement alone imposes. Says nothing about whether that floor is
    /// on the critical path; that is the movement share's job.
    pub fn transfer_floor_ms(self, bytes: u64) -> f64 {
        if self.gb_per_s <= 0.0 {
            return 0.0;
        }
        (bytes as f64) / (self.gb_per_s * BYTES_PER_GB) * MS_PER_S
    }
}

#[cfg(test)]
#[path = "tests/regime.rs"]
mod tests;
