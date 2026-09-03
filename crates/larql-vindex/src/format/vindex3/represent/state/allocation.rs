//! **R5-F11 — what evidence must we purchase next?**
//!
//! A distinct question from the two either side of it, and it had no home
//! before this module:
//!
//! ```text
//! Selection            what does the search frontier contain?
//! EvidenceAllocation   what evidence must we PURCHASE next?      <- here
//! Measurement          what was observed?
//! AuthorityState       did the frozen contract pass?
//! Robustness           how decisive is that evidence?
//! Promotion            what map may become parent?
//! ```
//!
//! Each stage's authority, stated once:
//!
//! ```text
//! search        may propose
//! selection     may preserve ambiguity
//! allocation    may expand measurement
//! budget        may refuse expenditure
//! measurement   observes
//! authority     decides contract truth
//! robustness    characterises evidence
//! promotion     changes parent
//! ```
//!
//! > **A resource decision may restrict WHETHER evidence is purchased; it
//! > may never change WHAT THE EVIDENCE SAYS.**
//!
//! That is the boundary this module exists to hold, and there is
//! deliberately nowhere in the pipeline for a resource decision to
//! masquerade as behavioural evidence.
//!
//! [`super::search_policy::Selection`] answers the first and always names
//! ONE experiment, correctly: it orders experiments by priority.
//! [`super::super::decision::decide_promotion`] answers the last and owns
//! ADMISSIBILITY ambiguity. Neither answers what happens when the
//! ordering evidence cannot legitimately collapse the frontier at all.
//!
//! # Where this came from
//!
//! Rung 5's N3 put two candidates on the frontier that were
//! Pareto-incomparable on two validly participating proxies — U2 better
//! on kl, U1 better on route flip rate. Choosing U2 because kl was the
//! binding constraint would have installed a new policy invented after
//! seeing the numbers: *when proxies conflict, prefer the proxy attached
//! to the currently binding constraint*. That is scalarization wearing a
//! different hat, and the 0.49% byte lead and the one-flip difference die
//! to the same objection.
//!
//! Measuring both was what revealed the frontier did not exist at
//! authority scale — U2 dominated U1 on BOTH statistics at 8,192
//! positions. One run would have adjudicated an artefact.
//!
//! # The invariant
//!
//! > **Allocation may EXPAND measurement when evidence cannot
//! > legitimately collapse the frontier. It may not MANUFACTURE an
//! > ordering.**
//!
//! # What this deliberately does NOT say
//!
//! The historical ruling read "ambiguous -> measure all at authority".
//! That fused two things. N3 needed paired *selection-authority* runs
//! because authority was the next registered scale for that ladder, not
//! because ambiguity implies authority. A later programme might resolve a
//! 256-position ambiguity on a 2,048-position bank before buying two
//! 8,192-position authority runs.
//!
//! So [`EvidenceAllocation`] says WHICH states need evidence, and
//! [`AllocationPolicy`] says at what scale — two questions, two types,
//! because fusing "frontier members" with "8,192 authority" would bake a
//! ladder's accident into the vocabulary.

use serde::{Deserialize, Serialize};

use super::search_policy::SelectionShape;
use crate::format::vindex3::represent::measurement::EvidenceScale;

/// What evidence the next round must purchase.
///
/// Generic over the subject so the allocator is not tied to one
/// experiment representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceAllocation<S> {
    /// Nothing to buy. The neighbourhood is closed.
    None,
    /// One experiment resolves the round.
    One(S),
    /// The ordering evidence cannot collapse these, so every one of them
    /// is measured. **Not "several promoted"** — evidence acquisition
    /// required BECAUSE the search cannot legitimately choose.
    Frontier(Vec<S>),
}

impl<S> EvidenceAllocation<S> {
    /// How many experiments this round buys.
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Frontier(v) => v.len(),
        }
    }

    /// Whether nothing is to be measured.
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Whether the round had to expand because evidence could not choose.
    pub fn is_expanded(&self) -> bool {
        matches!(self, Self::Frontier(_))
    }

    /// The subjects, in allocation order.
    pub fn subjects(&self) -> &[S] {
        match self {
            Self::None => &[],
            Self::One(s) => std::slice::from_ref(s),
            Self::Frontier(v) => v,
        }
    }
}

/// At what scale, and within what budget, an allocation is purchased.
///
/// Separate from [`EvidenceAllocation`] on purpose: which states need
/// evidence is a property of the ordering, and what it costs to look is a
/// property of the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationPolicy {
    /// The next sanctioned scale for this round. NOT necessarily
    /// authority — a ladder may register an intermediate resolution bank.
    pub next_scale: EvidenceScale,
    /// The most experiments one round may buy. `None` is unbounded.
    ///
    /// A cap may refuse a round outright; it may never TRIM a frontier to
    /// fit, because trimming is choosing, and choosing is what the
    /// frontier says cannot be done.
    pub max_experiments: Option<usize>,
}

impl AllocationPolicy {
    /// A policy buying at `next_scale` with no cap.
    pub fn at(next_scale: EvidenceScale) -> Self {
        Self {
            next_scale,
            max_experiments: None,
        }
    }

    /// Cap how many experiments a round may buy.
    pub fn with_budget(mut self, max_experiments: usize) -> Self {
        self.max_experiments = Some(max_experiments);
        self
    }

    /// Whether this allocation fits the budget.
    ///
    /// Returns the allocation unchanged when it fits, and
    /// [`BudgetRefusal`] when it does not. **There is no third answer**:
    /// silently returning a shortened frontier is the failure this type
    /// exists to prevent.
    pub fn admits<S>(
        &self,
        allocation: EvidenceAllocation<S>,
    ) -> Result<EvidenceAllocation<S>, BudgetRefusal> {
        match self.max_experiments {
            Some(cap) if allocation.len() > cap => Err(BudgetRefusal {
                required: allocation.len(),
                budget: cap,
                scale: self.next_scale,
            }),
            _ => Ok(allocation),
        }
    }
}

/// A round the budget cannot buy.
///
/// Carries what it would have cost, so the decision to raise the budget
/// or defer the round is made with the number in hand — and so that a
/// deferred ambiguity is visibly deferred rather than quietly resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetRefusal {
    /// Experiments the frontier requires.
    pub required: usize,
    /// Experiments the policy allows.
    pub budget: usize,
    /// The scale they would have been bought at.
    pub scale: EvidenceScale,
}

/// Whether the frontier's members can be ordered against each other.
///
/// Supplied by the caller because comparability is a property of the
/// evidence layer — [`super::super::participation`] decides which
/// statistics a candidate may be ranked on, and this module must not
/// re-derive that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierOrdering {
    /// The evidence resolves them; take the leader.
    Resolved,
    /// Valid evidence disagrees. The frontier stands.
    Incomparable,
}

/// Turn a [`Selection`] into what must actually be measured.
///
/// `unresolved` is the frontier the selection could not collapse, used
/// only when `ordering` is [`FrontierOrdering::Incomparable`].
pub fn allocate<S: Clone>(
    selection: SelectionShape,
    leading: S,
    unresolved: &[S],
    ordering: FrontierOrdering,
) -> EvidenceAllocation<S> {
    match (selection, ordering) {
        (SelectionShape::Exhausted, _) => EvidenceAllocation::None,
        // A sole candidate is selected because there is nothing to rank
        // (Ruling 3). Ambiguity cannot arise with one member, whatever
        // the caller claims about ordering.
        (SelectionShape::Sole, _) => EvidenceAllocation::One(leading),
        (SelectionShape::Ranked, FrontierOrdering::Resolved) => EvidenceAllocation::One(leading),
        (SelectionShape::Ranked, FrontierOrdering::Incomparable) => {
            if unresolved.len() <= 1 {
                // Nothing to expand to. An "incomparable" claim over one
                // member is a caller error, not a frontier.
                EvidenceAllocation::One(leading)
            } else {
                EvidenceAllocation::Frontier(unresolved.to_vec())
            }
        }
    }
}

#[cfg(test)]
#[path = "allocation_tests.rs"]
mod allocation_tests;
