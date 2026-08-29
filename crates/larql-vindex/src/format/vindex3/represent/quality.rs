//! **Versioned acceptance criteria, and the evidence they judge.**
//!
//! `quality_proven` must never be a boolean somebody sets. It has to
//! mean *passed a named gate*, because acceptance criteria change: they
//! get tightened when a representation ships, loosened for a draft
//! model, and compared across models. A record saying "quality: ok" with
//! no gate behind it is unfalsifiable a month later, and worse, is
//! indistinguishable from one that passed a much weaker bar.
//!
//! So the evidence carries the gate it was judged by, and the verdict is
//! **derived, never stored** — [`QualityEvidence`] holds only the
//! criteria and the measurements, so it is not possible to construct a
//! passing verdict over a failing bank.
//!
//! ## Routing evidence is kept apart from logit evidence
//!
//! Both live in one bank, but they answer different questions, and for a
//! MoE representation the difference is the whole diagnosis:
//!
//! - **logits moved, routing stable** — an arithmetic effect. The same
//!   experts ran and their outputs shifted. Usually smooth in the
//!   representation's error, and usually the thing more bits fix.
//! - **routing moved** — a decision changed. A different expert ran, so
//!   the downstream effect is not a small perturbation of the same
//!   function, and more bits *on the experts* may not be the response at
//!   all; the router or its inputs may be what needs protecting.
//!
//! Collapsing them into one "quality" number would make those two look
//! identical and send the precision-map search in the wrong direction.

use serde::{Deserialize, Serialize};

/// Acceptance criteria, identified so a claim can name what it passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    /// e.g. `kimi-logit-v1`. Changing a threshold means a NEW id.
    pub id: String,
    /// Fewest positions the bank must cover for its tail statistics to
    /// mean anything. A p99 over nineteen positions is not a p99.
    pub positions_min: u64,
    pub kl_p99_max: f64,
    pub top1_flip_max: u64,
    pub top10_change_max: u64,
    pub route_flip_max: u64,
}

/// What the bank measured about the output distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitEvidence {
    pub kl_p50: f64,
    pub kl_p95: f64,
    pub kl_p99: f64,
    pub max_logit_delta: f64,
    /// Positions where the argmax changed.
    pub top1_flips: u64,
    /// Positions where the top-10 SET or its order changed.
    pub top10_changes: u64,
}

/// What the bank measured about routing — deliberately separate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingEvidence {
    /// Selected-expert-id changes, summed over positions and layers.
    pub route_flips: u64,
    /// Positions where ANY layer routed differently. A handful of flips
    /// concentrated in one position is a different fact from the same
    /// count spread across all of them.
    pub positions_with_route_change: u64,
    /// Layers that ever routed differently, so a fix can be scoped by
    /// depth instead of applied to the whole role.
    pub layers_with_route_change: u64,
}

/// One bank of measurements over a fixed token sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBank {
    pub positions: u64,
    pub logits: LogitEvidence,
    pub routing: RoutingEvidence,
}

/// Which criterion a bank failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Criterion {
    Positions,
    KlP99,
    Top1Flips,
    Top10Changes,
    RouteFlips,
}

impl Criterion {
    pub fn name(self) -> &'static str {
        match self {
            Criterion::Positions => "positions",
            Criterion::KlP99 => "kl_p99",
            Criterion::Top1Flips => "top1_flips",
            Criterion::Top10Changes => "top10_changes",
            Criterion::RouteFlips => "route_flips",
        }
    }
}

/// The outcome of judging a bank by a gate.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityVerdict {
    pub gate_id: String,
    /// Empty means passed. Each entry names the criterion and what it
    /// saw against what it required.
    pub failures: Vec<(Criterion, String)>,
}

impl QualityVerdict {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn describe(&self) -> String {
        if self.passed() {
            return format!("passed {}", self.gate_id);
        }
        let f: Vec<String> = self
            .failures
            .iter()
            .map(|(c, d)| format!("{}: {d}", c.name()))
            .collect();
        format!("failed {} — {}", self.gate_id, f.join("; "))
    }
}

impl QualityGate {
    /// Judge a bank. Every criterion is checked, not short-circuited, so
    /// a report says everything that is wrong rather than the first
    /// thing.
    pub fn evaluate(&self, bank: &QualityBank) -> QualityVerdict {
        let mut failures = Vec::new();
        if bank.positions < self.positions_min {
            failures.push((
                Criterion::Positions,
                format!("{} < {} required", bank.positions, self.positions_min),
            ));
        }
        if bank.logits.kl_p99 > self.kl_p99_max {
            failures.push((
                Criterion::KlP99,
                format!("{:.3e} > {:.3e}", bank.logits.kl_p99, self.kl_p99_max),
            ));
        }
        if bank.logits.top1_flips > self.top1_flip_max {
            failures.push((
                Criterion::Top1Flips,
                format!("{} > {}", bank.logits.top1_flips, self.top1_flip_max),
            ));
        }
        if bank.logits.top10_changes > self.top10_change_max {
            failures.push((
                Criterion::Top10Changes,
                format!("{} > {}", bank.logits.top10_changes, self.top10_change_max),
            ));
        }
        if bank.routing.route_flips > self.route_flip_max {
            failures.push((
                Criterion::RouteFlips,
                format!("{} > {}", bank.routing.route_flips, self.route_flip_max),
            ));
        }
        QualityVerdict {
            gate_id: self.id.clone(),
            failures,
        }
    }
}

/// A bank together with the gate it is judged by.
///
/// The verdict is NOT a field. Storing it would allow a record whose
/// stored verdict and stored numbers disagree, which is exactly the
/// unfalsifiable claim this module exists to prevent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidence {
    pub gate: QualityGate,
    pub bank: QualityBank,
}

impl QualityEvidence {
    pub fn verdict(&self) -> QualityVerdict {
        self.gate.evaluate(&self.bank)
    }

    /// The gate id this evidence passed, if it passed.
    pub fn proven_by(&self) -> Option<&str> {
        self.verdict().passed().then_some(self.gate.id.as_str())
    }

    /// Whether the logits moved while routing stayed put — the two
    /// mechanisms a MoE precision decision has to tell apart.
    pub fn is_arithmetic_only(&self) -> bool {
        self.bank.routing.route_flips == 0
    }
}

/// **`kimi-logit-v1` — the first acceptance contract for Kimi Linear.**
///
/// Named, and therefore FROZEN. If these thresholds turn out too strict
/// or too lax, the answer is `kimi-logit-v2`; editing this function
/// would silently re-date every claim that ever cited v1, which is the
/// one thing a versioned gate exists to prevent.
///
/// The values are a stated PRIOR, not a calibration — no candidate has
/// been through a bank yet, and guessing thresholds from a distribution
/// nobody has seen is how a bad contract gets frozen. They are written
/// as fractions of an 8192-position bank so the reasoning is inspectable:
///
/// | criterion | value | fraction | why |
/// |---|---|---|---|
/// | `positions_min` | 4096 | half the bank | a p99 needs a tail; 19 positions has none |
/// | `kl_p99_max` | 1e-3 nats | — | at the 99th percentile, well under where sampled text visibly diverges |
/// | `top1_flip_max` | 8 | 0.1 % | greedy decoding changes at all here, so it is the strictest of the four |
/// | `top10_change_max` | 82 | 1 % | reordering inside the top-10 is real but rarely decisive |
/// | `route_flip_max` | 82 | 1 % | a routing change is a decision change, held to the same 1 % |
///
/// The bar this must clear before it is used on anything: a NULL arm
/// (BF16 against itself) has to pass it with every count at zero. A gate
/// its own reference cannot satisfy is measuring the harness.
pub fn kimi_logit_v1() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 4096,
        kl_p99_max: 1e-3,
        top1_flip_max: 8,
        top10_change_max: 82,
        route_flip_max: 82,
    }
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
