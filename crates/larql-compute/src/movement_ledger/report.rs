//! Rendering. Raw arms first, verdict last and gated.
//!
//! The ordering is deliberate: a derived verdict inherits every
//! assumption of the classifier that produced it, so the raw counters are
//! printed above it and are always readable. The verdict line refuses to
//! render at all without a declared [`Regime`], because the identical
//! byte delta licenses opposite conclusions across regimes.

use super::{
    ByteMovement, DecisionCounts, LedgerConfig, MovementCost, Phase, TimeAttribution, TokenRecord,
};

/// Decimal-unit byte formatting, matching the GB/s convention the
/// bandwidth probes report in. Mixing binary and decimal units here is a
/// 7.4% error — the size of the effects this ledger adjudicates.
fn fmt_bytes(n: u64) -> String {
    const UNITS: [(&str, f64); 4] = [("GB", 1.0e9), ("MB", 1.0e6), ("kB", 1.0e3), ("B", 1.0)];
    for (suffix, scale) in UNITS {
        if (n as f64) >= scale {
            return format!("{:.3} {}", n as f64 / scale, suffix);
        }
    }
    format!("{n} B")
}

fn fmt_ratio(v: Option<f64>, unit: &str) -> String {
    match v {
        Some(x) => format!("{x:.3}{unit}"),
        None => "n/a".to_string(),
    }
}

/// The byte arms.
fn render_bytes(b: &ByteMovement, out: &mut String) {
    out.push_str(&format!(
        "[bw10/bytes]  semantic={}  physical={}  useful={}  ampl={}  eff={}\n",
        fmt_bytes(b.semantic_requested),
        fmt_bytes(b.physical_touched),
        fmt_bytes(b.useful_physical),
        fmt_ratio(b.amplification(), "x"),
        fmt_ratio(b.useful_ratio(), ""),
    ));
    out.push_str(&format!(
        "[bw10/tier]   dram={}  nvme={}  network={}  unattributed={}\n",
        fmt_bytes(b.dram),
        fmt_bytes(b.nvme),
        fmt_bytes(b.network),
        fmt_bytes(b.tier_unattributed()),
    ));
    // A zero from an absent reporter is not a measured zero. Say which.
    let reused = if b.reuse_observed {
        fmt_bytes(b.reused)
    } else {
        "n/a (no reuse reporter registered)".to_string()
    };
    let prefetch = if b.prefetch_observed {
        format!(
            "{} (unused {}, waste {})",
            fmt_bytes(b.prefetched),
            fmt_bytes(b.prefetched_unused),
            fmt_ratio(b.prefetch_waste_ratio(), ""),
        )
    } else {
        "n/a (no prefetcher registered)".to_string()
    };
    out.push_str(&format!(
        "[bw10/cache]  reused={reused}  prefetched={prefetch}\n"
    ));
}

/// The execution-policy arm: what the runtime chose NOT to do, and what
/// that decision removed.
///
/// Printed under the byte arms because it qualifies them in exactly the
/// way coverage does. A physical total that shrank because a policy
/// deleted operations is a different fact from one that shrank because
/// the representation got smaller, and without this block the two render
/// identically.
fn render_decisions(d: &DecisionCounts, b: &ByteMovement, c: &MovementCost, out: &mut String) {
    let policy = crate::exec_policy::installed_name()
        .unwrap_or_else(|| "none (canonical execution)".to_string());
    if !d.is_measured() {
        out.push_str(&format!(
            "[bw10/policy] policy={policy}  NOT MEASURED — no routed expert group reached the \
             execution seam in this window. Not the same as a 0% skip rate.\n"
        ));
        return;
    }
    out.push_str(&format!(
        "[bw10/policy] policy={policy}\n[bw10/policy] expert groups: requested={} executed={} \
         skipped={}  skip_rate={}\n",
        d.requested,
        d.executed,
        d.skipped,
        fmt_ratio(d.skip_rate(), ""),
    ));
    if !d.is_consistent() {
        out.push_str(
            "[bw10/policy] BUG requested != executed + skipped — a dispatch site recorded a \
             decision outside larql_compute::exec_policy::resolve_expert_group. Every number \
             on this line is unreliable.\n",
        );
    }
    if d.skipped == 0 {
        return;
    }
    out.push_str(&format!(
        "[bw10/avoided] semantic={}  physical={}  share_of_canonical={}\n",
        fmt_bytes(d.semantic_avoided),
        fmt_bytes(d.physical_avoided),
        fmt_ratio(d.avoided_share(b.physical_touched), ""),
    ));
    // Time attribution for the avoided bytes, priced at THIS arm's own
    // observed streaming rate. It is a projection, and the label says so:
    // a latency saving is a difference between two runs, and this window
    // only contains one of them. Quoting it as a measured saving would be
    // the §4.11 error in its forward-looking form — which is precisely
    // why `predicted_saving_ms` prices at the observed rate rather than
    // the roofline, so the number is at least not inflated by pretending
    // the engine streams at ceiling.
    match c.predicted_saving_ms(d.physical_avoided) {
        Some(ms) => out.push_str(&format!(
            "[bw10/avoided] PROJECTED time for those bytes at this arm's observed stream rate: \
             {ms:.3}ms. NOT a measured saving — run the same prompt with the policy uninstalled \
             and compare wall times for that.\n"
        )),
        None => out.push_str(
            "[bw10/avoided] projected time not computable — no observed stream rate in this \
             window\n",
        ),
    }
}

/// The time arms. Terms that overlap are printed as "of which" and never
/// summed into the decomposition above them.
fn render_time(t: &TimeAttribution, out: &mut String) {
    out.push_str(&format!(
        "[bw10/time]   wall={:.3}ms = gpu_busy {:.3}ms + bubble {:.3}ms + host_outside_gpu {:.3}ms\n",
        t.wall_ms,
        t.gpu_busy_ms,
        t.gpu_bubble_ms,
        t.host_outside_gpu_ms(),
    ));
    // A zero from an uninstrumented sampler is not a measured zero.
    let host_wait = if t.host_wait_reported {
        format!("{:.3}ms", t.host_wait_ms)
    } else {
        "n/a (no sampler)".to_string()
    };
    let io_wait = if t.io_wait_reported {
        format!("{:.3}ms", t.io_wait_ms)
    } else {
        "n/a (no sampler)".to_string()
    };
    out.push_str(&format!(
        "[bw10/time]   of which (overlapping, not additive): host_wait={host_wait}  \
         io_wait={io_wait}  gpu_occupancy={}\n",
        fmt_ratio(t.gpu_occupancy(), ""),
    ));
}

/// The join: what the bytes cost, and how far the engine runs from the
/// declared ceiling.
fn render_cost(c: &MovementCost, cfg: &LedgerConfig, out: &mut String) {
    let roofline = match cfg.rooflines.dram {
        Some(bw) => format!("{:.0} GB/s ({})", bw.gb_per_s(), bw.source()),
        None => "undeclared".to_string(),
    };
    out.push_str(&format!(
        "[bw10/cost]   floor={:.3}ms @ {}  implied_stream={}  utilisation_eta={}\n",
        c.floor_ms,
        roofline,
        fmt_ratio(c.implied_stream_gbps, " GB/s"),
        fmt_ratio(c.roofline_utilisation, ""),
    ));
    if c.bytes_without_roofline > 0 {
        out.push_str(&format!(
            "[bw10/cost]   WARNING {} crossed a tier with no declared bandwidth — \
             the movement share below is a LOWER bound only\n",
            fmt_bytes(c.bytes_without_roofline),
        ));
    }
    // Under partial coverage the numerator of implied_stream is missing
    // surfaces while its denominator (GPU busy) covers all of them, so
    // both it and eta are floors. Saying "coverage is partial" once at
    // the byte arms is not enough — a reader who sees eta=0.38 will
    // conclude the kernels are inefficient unless told here.
    if !super::coverage::is_complete() {
        out.push_str(
            "[bw10/cost]   NOTE coverage is partial: implied_stream and utilisation_eta charge \
             ALL GPU time against only the covered surfaces' bytes, so both are LOWER bounds. \
             Compare eta across arms, not against 1.0.\n",
        );
    }
}

/// The gated verdict.
fn render_verdict(c: &MovementCost, cfg: &LedgerConfig, out: &mut String) {
    let Some(regime) = cfg.regime else {
        out.push_str(
            "[bw10/verdict] REFUSED — no regime declared. Set LARQL_MOVEMENT_REGIME to one of \
             resident | capacity-constrained | cold-estate, or call \
             LedgerConfig::with_regime. A byte delta licenses opposite conclusions across \
             regimes, so no verdict is emitted without one.\n",
        );
        return;
    };
    out.push_str(&format!(
        "[bw10/regime] {} — {}\n",
        regime,
        regime.byte_claim_licence(),
    ));
    match (c.share_of_wall, c.gpu_busy_share) {
        (Some(lo), Some(hi)) => {
            out.push_str(&format!(
                "[bw10/verdict] movement share of wall is bracketed [{lo:.3}, {hi:.3}] \
                 (lo = bytes at roofline; hi = all GPU time charged to streaming). \
                 Removing ALL measured byte movement cannot buy more than {:.1}% of the token.\n",
                hi * 100.0,
            ));
            if !super::coverage::is_complete() {
                // The upper end survives partial coverage — it never reads
                // the byte counters — so the ceiling on any byte lever
                // still holds. Only the lower end is understated.
                out.push_str(
                    "[bw10/verdict] the UPPER end of that bracket is unaffected by partial \
                     coverage (it reads no byte counter), so the ceiling on any byte lever \
                     stands; the lower end is understated.\n",
                );
            }
        }
        _ => {
            out.push_str(
                "[bw10/verdict] movement share not computable — zero wall time in the window\n",
            );
        }
    }
}

impl TokenRecord {
    /// Full multi-line ledger entry: raw byte arms, raw time arms, the
    /// cost join, then the gated verdict.
    pub fn render(&self, cfg: &LedgerConfig) -> String {
        let mut out = String::new();
        render_bytes(&self.bytes, &mut out);
        // Coverage sits directly under the byte arms because it qualifies
        // them: a physical total is only a token's traffic if every
        // surface that moves bytes is instrumented and fired.
        out.push_str(&super::coverage::render());
        out.push('\n');
        let cost = self.cost(cfg);
        render_decisions(&self.decisions, &self.bytes, &cost, &mut out);
        render_time(&self.time, &mut out);
        render_cost(&cost, cfg, &mut out);
        render_verdict(&cost, cfg, &mut out);
        out
    }

    /// One-line form for per-token streaming, where the multi-line block
    /// would swamp the output. Carries the two headline currencies only.
    ///
    /// The bracket tag names the phase (`decode`, `prefill`, or `?` for
    /// unattributed) — without it a per-token stream cannot be told apart
    /// from the exact defect this ledger's phase tracking exists to
    /// catch: 129 prefill positions rendering identically to 129 decode
    /// steps.
    pub fn render_compact(&self, cfg: &LedgerConfig) -> String {
        let cost = self.cost(cfg);
        let tag = match self.phase {
            Some(Phase::Decode) => "bw10:decode",
            Some(Phase::Prefill) => "bw10:prefill",
            None => "bw10:?",
        };
        // The skip term appears only when a policy actually deleted
        // something, so an uninstrumented stream keeps its old shape —
        // but when it does appear it is on the SAME line as the physical
        // total, because a reader who sees the byte count drop must not
        // have to scroll to find out that the engine was told to skip.
        let skipped = if self.decisions.skipped > 0 {
            format!(
                "  skipped={}/{} groups (avoided {})",
                self.decisions.skipped,
                self.decisions.requested,
                fmt_bytes(self.decisions.physical_avoided),
            )
        } else {
            String::new()
        };
        format!(
            "[{tag}] physical={}  wall={:.3}ms  gpu_busy={:.3}ms  bubble={:.3}ms  \
             eta={}  share=[{}, {}]{skipped}",
            fmt_bytes(self.bytes.physical_touched),
            self.time.wall_ms,
            self.time.gpu_busy_ms,
            self.time.gpu_bubble_ms,
            fmt_ratio(cost.roofline_utilisation, ""),
            fmt_ratio(cost.share_of_wall, ""),
            fmt_ratio(cost.gpu_busy_share, ""),
        )
    }
}

#[cfg(test)]
#[path = "tests/report.rs"]
mod tests;
