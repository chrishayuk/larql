//! **Which candidate to measure next, when some of its costs cannot be
//! priced.**
//!
//! [`super::assessment`] prices what the evidence licenses pricing. At
//! diagnostic scale, under ROUTE-CAL-1, that is currently nothing:
//! `kl p99` is an ordering proxy and `routed mixture moved at p99` is
//! unusable, so every candidate comes back [`MoveClass::Unscorable`].
//!
//! That leaves a hole, and it is the statistical twin of the
//! exhausted-budget bug: **an unpriced dimension must not create an
//! advantage by being invisible.** Two candidates whose route cost is
//! equally unpriceable will tier equally and then sort on physical
//! gain — so the one that moves routing hardest wins, precisely because
//! nothing could see it.
//!
//! ```text
//! A   gain 1.00 ms   kl 5%   route p99 unscorable   route proxy DANGEROUS
//! B   gain 0.90 ms   kl 6%   route p99 unscorable   route proxy benign
//! ```
//!
//! A must not win. The proxy has to change its class, not its score —
//! turning ROUTE-CAL-1's rho +0.991 flip-rate result into an
//! approximate route-mass number would be the exact conflation of
//! ordering with magnitude that [`super::search_evidence`] exists to
//! prevent.
//!
//! So pricing and proxy ordering stay separate, and the search combines
//! them in stages rather than in a formula:
//!
//! ```text
//! 1. priceable      economics decide
//! 2. proxy-supported   every unpriceable dimension has benign proxy evidence
//! 3. proxy-risky       a proxy warns
//! 4. uninformed        an unpriceable dimension nobody can speak to
//! 5. worthless         buys nothing, whatever its evidence
//! ```

use serde::{Deserialize, Serialize};

use super::assessment::{CandidateAssessment, MoveClass};
use super::quality::Statistic;
use super::search_evidence::SearchEvidence;

/// Which way a proxy points for the criterion it stands in for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyRisk {
    /// The proxy did not worsen.
    Benign,
    /// The proxy worsened, so the criterion it stands in for probably
    /// did too. Not a magnitude — a direction.
    Elevated,
}

/// An ordering-only observation standing in for a criterion that cannot
/// be priced at this evidence scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProxyObservation {
    /// The proxy itself, e.g. `"route flip rate"`.
    pub statistic: Statistic,
    /// The contract criterion it stands in for.
    pub for_criterion: Statistic,
    pub parent: f64,
    pub candidate: f64,
    /// Why this proxy may be believed for ordering. Anything that is
    /// not an ordering proxy or better carries no weight.
    pub evidence: SearchEvidence,
}

impl ProxyObservation {
    pub fn delta(&self) -> f64 {
        self.candidate - self.parent
    }

    /// The direction only. A magnitude here would be a route-mass
    /// estimate wearing a flip rate's clothes.
    pub fn risk(&self) -> ProxyRisk {
        if self.evidence.orders() && self.delta() > 0.0 {
            ProxyRisk::Elevated
        } else {
            ProxyRisk::Benign
        }
    }

    /// Whether this observation may be believed at all.
    pub fn usable(&self) -> bool {
        self.evidence.orders()
    }
}

/// How ready a candidate is to be promoted to an authority measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionReadiness {
    /// Every dimension the contract judges was priceable.
    Priceable,
    /// Some dimension was not priceable, and every such dimension has
    /// benign proxy evidence.
    ProxySupported,
    /// A proxy warns about an unpriceable dimension.
    ProxyRisky,
    /// An unpriceable dimension that no proxy can speak to. The search
    /// knows it does not know.
    Uninformed,
}

impl PromotionReadiness {
    /// Higher promotes first.
    fn tier(self) -> u8 {
        match self {
            Self::Priceable => 3,
            Self::ProxySupported => 2,
            Self::ProxyRisky => 1,
            Self::Uninformed => 0,
        }
    }
}

/// A candidate, its assessment, and whatever proxy evidence speaks to
/// the parts of it that could not be priced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionCandidate {
    pub assessment: CandidateAssessment,
    pub proxies: Vec<ProxyObservation>,
}

impl PromotionCandidate {
    pub fn new(assessment: CandidateAssessment, proxies: Vec<ProxyObservation>) -> Self {
        Self {
            assessment,
            proxies,
        }
    }

    /// Proxy evidence for one unpriceable criterion, if any is usable.
    pub fn proxy_for(&self, criterion: Statistic) -> Option<&ProxyObservation> {
        self.proxies
            .iter()
            .find(|p| p.for_criterion == criterion && p.usable())
    }

    /// **How ready this candidate is**, decided by the weakest
    /// unpriceable dimension rather than the average.
    pub fn readiness(&self) -> PromotionReadiness {
        let unpriceable: Vec<&super::assessment::MarginalConstraintCost> =
            self.assessment.marginal.unpriceable_costs().collect();
        if unpriceable.is_empty() {
            return PromotionReadiness::Priceable;
        }
        let mut worst = PromotionReadiness::ProxySupported;
        for c in unpriceable {
            // A dimension whose OWN evidence orders is its own proxy —
            // diagnostic kl cannot be priced and can still say which of
            // two candidates is worse. Only a dimension nothing can
            // order needs an external stand-in.
            if c.orders() {
                continue;
            }
            match self.proxy_for(c.what).map(ProxyObservation::risk) {
                Some(ProxyRisk::Benign) => {}
                Some(ProxyRisk::Elevated) => {
                    if worst != PromotionReadiness::Uninformed {
                        worst = PromotionReadiness::ProxyRisky;
                    }
                }
                None => worst = PromotionReadiness::Uninformed,
            }
        }
        worst
    }

    /// Tier for ranking. A move that buys nothing ranks last whatever
    /// its evidence, so worthlessness outranks every readiness.
    fn tier(&self) -> u8 {
        if self.assessment.ranking_score.class == MoveClass::Worthless {
            return 0;
        }
        self.readiness().tier() + 1
    }

    /// Sum of usable proxy deltas — the secondary key WITHIN a tier,
    /// where lower is preferred. Never combined with the economics into
    /// one number, because their units are not comparable and pretending
    /// otherwise is how a proxy becomes a price.
    fn proxy_pressure(&self) -> f64 {
        self.proxies
            .iter()
            .filter(|p| p.usable())
            .map(ProxyObservation::delta)
            .sum()
    }

    /// **Total order, best first.**
    ///
    /// Readiness tier, then the assessment's own economics, then proxy
    /// pressure, then the candidate's identity so a search trace
    /// reproduces exactly.
    pub fn cmp_rank(&self, other: &Self) -> std::cmp::Ordering {
        other
            .tier()
            .cmp(&self.tier())
            .then_with(|| {
                self.assessment
                    .ranking_score
                    .cmp_rank(&other.assessment.ranking_score)
            })
            .then_with(|| self.proxy_pressure().total_cmp(&other.proxy_pressure()))
            .then_with(|| {
                self.assessment
                    .candidate_map
                    .cmp(&other.assessment.candidate_map)
            })
    }

    /// Why this candidate ranks where it does, in one line a search
    /// trace can carry.
    pub fn why(&self) -> String {
        let r = self.readiness();
        let unpriced: Vec<&super::assessment::MarginalConstraintCost> =
            self.assessment.marginal.unpriceable_costs().collect();
        if unpriced.is_empty() {
            return format!("{r:?}: every judged dimension was priceable");
        }
        let detail: Vec<String> = unpriced
            .iter()
            .map(|c| {
                if c.orders() {
                    return format!("{} unpriceable, orders itself", c.what);
                }
                match self.proxy_for(c.what) {
                    Some(p) => format!("{} unpriceable, {} {:?}", c.what, p.statistic, p.risk()),
                    None => format!("{} unpriceable, no proxy", c.what),
                }
            })
            .collect();
        format!("{r:?}: {}", detail.join("; "))
    }
}

#[cfg(test)]
#[path = "promotion_tests.rs"]
mod tests;
