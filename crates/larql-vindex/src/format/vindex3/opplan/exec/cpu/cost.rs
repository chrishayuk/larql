//! The physical cost model: what a plan's byte budget PREDICTS.
//!
//! Twelve rungs measured how fast each arithmetic streams its weights.
//! This turns that into the number a mixed-precision plan has to be
//! judged on — not "did Q4 pass quality" but **what does the
//! quality-valid plan cost**, given that every operand restored from Q4
//! to Q8 buys accuracy and spends bytes.
//!
//! ```text
//! arithmetic            bytes/token   ms/token    GB/s
//! BF16 x F32 -> F32       51.20 GB     420.84    121.66
//! Q8   x F32 -> F32       27.20        332.97     81.69
//! Q8   x Q8  -> I32       27.20        224.75    121.02
//! Q4   x Q8  -> I32       14.40        135.10    106.59
//! ```
//!
//! **The rates are DERIVED from CPU-4Y's own bytes and times, not copied
//! from the rung that first reported each one.** Those differ: CPU-3A
//! measured Q8 x F32 at 83.4 GB/s and CPU-4X measured Q8 x Q8 at 118.0,
//! while CPU-4Y's single table implies 81.7 and 121.0. The gaps are
//! real — separate builds, and a rebuild touching only `workers_from`
//! once moved an untouched function 14% — so mixing them would build a
//! cost model out of four different machines. One table, one run, one
//! set of rates.
//!
//! **These are rates, not times.** That is the whole point: a rate
//! multiplied by a plan's own byte census predicts a plan nobody has
//! timed yet, which is what makes an exception-set search affordable.
//! A recipe that had to be benchmarked end to end per candidate would
//! cost an hour a candidate.
//!
//! The prediction is stated against a MEASURED correction, not an
//! assumed one: CPU-PERF-3B priced the gap between the isolated harness
//! and the real decode at 4.7%, splitting into 2.6% residency/topology
//! and 2.1% interleaving.

use super::ledger::PlanTally;
use super::physical::PhysicalProjectionPlan;

/// Synthetic-harness rate to real-decode rate.
///
/// **Measured** (CPU-PERF-3B), not assumed. bf16's replay came in 0.8%
/// FASTER than its synthetic harness — traversing 497 large linear
/// streams costs nothing extra — while Q8's 866 allocations over 369
/// operands cost ~8 ms. The cost model needs a topology term and it is
/// a small one.
pub const SYNTHETIC_TO_REAL: f64 = 1.047;

/// Bytes per gigabyte, decimal — the unit every measured rate above is
/// quoted in.
const BYTES_PER_GB: f64 = 1e9;

/// The measured streaming rate for one arithmetic, in GB/s.
///
/// Every value is from a harness whose bf16 arm reproduces the real
/// model's projection class to -3.9%, so a ratio taken against it
/// licenses something.
pub fn measured_rate_gbps(plan: PhysicalProjectionPlan) -> Option<f64> {
    match plan {
        // The literal transcription: a flat 5.6 GB/s at every shape,
        // which is why it is the oracle and not a strategy.
        PhysicalProjectionPlan::ScalarF32 => Some(5.6),
        // Accelerate's sgemv reading from cache — this plan is only ever
        // chosen for operands whose image FITS cache, so the cache-
        // resident rate is the right one.
        PhysicalProjectionPlan::BlasF32 => Some(262.0),
        PhysicalProjectionPlan::FusedBf16 => Some(121.66),
        PhysicalProjectionPlan::FusedQ8 => Some(81.69),
        // CPU-4A: Q4 against an f32 activation is SLOWER than Q8, because
        // the kernel was already conversion-bound and Q4 adds a nibble
        // split. 14.40 GB at 36.9 GB/s.
        PhysicalProjectionPlan::FusedQ4 => Some(36.9),
        // **No rate has been measured for this plan.** `None` rather
        // than a plausible-looking constant: every other value here
        // comes from a harness, and one invented number in a table of
        // measurements poisons every ratio taken against it. NVFP4 is
        // reached by observation — a compiled pack the loader binds —
        // so it can genuinely appear in a tally, and the budget omits it
        // for the same reason it omits a plan with no calls.
        PhysicalProjectionPlan::FusedNvfp4 => None,
        // Unmeasured for the same reason as NVFP4: reached by observation
        // of a compiled pack, and no harness has priced it. PARETO-1's v3
        // qualification is an EQUIVALENCE gate, not a rate.
        PhysicalProjectionPlan::FusedKQuant => None,
        // Unpriced, deliberately, and not by omission: no harness has run
        // this kernel at a rate yet. Quoting a number here would let a
        // roofline claim a throughput nothing measured — the FP8 rung's
        // gate is EQUIVALENCE against the reference decode, and its cost
        // belongs to the residency work that comes after.
        PhysicalProjectionPlan::FusedFp8Block => None,
        PhysicalProjectionPlan::Q8xQ8 => Some(121.02),
        PhysicalProjectionPlan::Q4xQ8 => Some(106.59),
        // The A1 control runs the exact bf16 kernel over a reconstructed
        // activation, so it streams bf16 bytes at the bf16 rate. It is
        // never a deployment plan and its cost is quoted only so a
        // control run's ledger still adds up.
        PhysicalProjectionPlan::Bf16xQ8 => Some(121.66),
    }
}

/// One arithmetic's contribution to a token's projection cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BudgetRow {
    pub plan: PhysicalProjectionPlan,
    pub bytes: u64,
    pub calls: u64,
    pub rate_gbps: f64,
    /// Predicted milliseconds, before the synthetic-to-real correction.
    pub synthetic_ms: f64,
}

/// What one token's projections are predicted to cost, by arithmetic.
#[derive(Clone, Debug, Default)]
pub struct ProjectionBudget {
    pub rows: Vec<BudgetRow>,
    pub total_bytes: u64,
    pub synthetic_ms: f64,
    /// The number a decode claim is made against.
    pub predicted_ms: f64,
}

impl ProjectionBudget {
    /// Bytes carried by one arithmetic — the quantity an exception set
    /// spends.
    pub fn bytes_for(&self, plan: PhysicalProjectionPlan) -> u64 {
        self.rows
            .iter()
            .find(|r| r.plan == plan)
            .map(|r| r.bytes)
            .unwrap_or(0)
    }
}

/// Price a ledger snapshot.
///
/// Plans with no calls are dropped rather than reported as zero: a table
/// of eight arithmetics where six are empty invites reading the empty
/// ones as measurements.
pub fn budget(tallies: &[(PhysicalProjectionPlan, PlanTally)]) -> ProjectionBudget {
    let mut out = ProjectionBudget::default();
    for (plan, tally) in tallies {
        if tally.calls == 0 {
            continue;
        }
        // An unmeasured plan is omitted, never given a stand-in rate:
        // the row would read as a measurement and the totals would carry
        // a number nobody took.
        let Some(rate) = measured_rate_gbps(*plan) else {
            continue;
        };
        let ms = tally.bytes as f64 / BYTES_PER_GB / rate * 1e3;
        out.rows.push(BudgetRow {
            plan: *plan,
            bytes: tally.bytes,
            calls: tally.calls,
            rate_gbps: rate,
            synthetic_ms: ms,
        });
        out.total_bytes += tally.bytes;
        out.synthetic_ms += ms;
    }
    out.predicted_ms = out.synthetic_ms * SYNTHETIC_TO_REAL;
    out
}

/// Predicted tokens per second for a whole step, given the projection
/// budget and what the rest of the token costs.
///
/// The non-projection floor is a MEASURED range on this build (17-24 ms)
/// and is passed in rather than baked, because it is the one term that
/// does not scale with the representation — CPU-2D1 left the recurrence
/// at 13.3 ms and quantisation does not touch it.
pub fn predicted_tokens_per_second(budget: &ProjectionBudget, non_projection_ms: f64) -> f64 {
    let token_ms = budget.predicted_ms + non_projection_ms;
    if token_ms <= 0.0 {
        return 0.0;
    }
    1e3 / token_ms
}
