//! The decision itself: how a semantic operation is physically satisfied
//! on this invocation, and the site description a policy decides from.

use crate::movement_ledger::Phase;

/// How the runtime chose to satisfy one semantic operation.
///
/// The router still says WHICH experts conceptually participate; this
/// says how that participation is physically realised. The two are
/// different questions and this type exists so they cannot be conflated:
/// a `Skip` is not "the router picked fewer experts", it is "the routed
/// group's contribution was not computed".
///
/// # Why this is deliberately exhaustive
///
/// No `#[non_exhaustive]`, no catch-all arms at the dispatch sites. When
/// BW-B's compiled compact-dense representation lands as a third variant
/// (`CompactDense(..)` — see [`super`]'s module doc), every backend that
/// matches on this must fail to compile until it decides what to do with
/// it. A `_ => canonical` fallback would silently serve compact-dense as
/// canonical on whichever backend nobody remembered to update, and the
/// resulting measurement would look like a null result rather than an
/// unimplemented one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Execute as written — the only strategy with no installed policy,
    /// and the only one whose numerics are the model's own.
    Canonical,
    /// Omit the operation entirely. For a routed MoE expert group under
    /// an additive combine (`MoePostExpertNormPolicy::None`) this is
    /// residual/identity pass-through: `h_out = h_post_attn`.
    ///
    /// Precisely NOT the same perturbation as "the router selected these
    /// experts with weight zero" on an architecture that renormalises
    /// selected top-k weights before the combine — see
    /// `cpu::ops::moe::expert_override`'s module doc, which
    /// makes the same distinction for the CPU research hook.
    Skip,
}

impl ExecutionStrategy {
    /// Short label for reports and test failure messages.
    pub const fn label(self) -> &'static str {
        match self {
            ExecutionStrategy::Canonical => "canonical",
            ExecutionStrategy::Skip => "skip",
        }
    }

    /// Whether the operation's work is omitted under this strategy.
    pub const fn is_skip(self) -> bool {
        matches!(self, ExecutionStrategy::Skip)
    }
}

/// The routed expert group a policy is being asked about.
///
/// # Why the group, not the expert
///
/// On the production GPU route the selected expert IDs are chosen by a
/// kernel and consumed by a kernel — `encode_moe_router_select`'s output
/// buffer is never read by the host. A per-expert seam would therefore
/// have to either read that buffer back (reinstating exactly the host
/// round-trip S2 removed, for a 1.40x cost) or decide from stale routing.
/// The whole-group unit is what the production dispatch path can actually
/// address without giving back the scheduling win, and it is also the
/// unit BW-C3/C4/C5 measured — 66.7% of late-layer checkpoints had the
/// WHOLE top-4 group jointly removable, so it is the unit with evidence
/// behind it. A per-expert seam is a separate, later question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGroupSite {
    /// Index of the layer whose routed expert group is about to dispatch.
    pub layer: usize,
    /// Which generation phase is executing, from the ledger's
    /// [`Phase`] scope. `None` when no driver loop declared one — a
    /// phase-selective policy must refuse rather than guess, exactly as
    /// [`crate::movement_ledger::phase`] refuses to attribute a token.
    pub phase: Option<Phase>,
    /// Token index within the current phase, from [`super::step`].
    /// `None` when no token boundary has been declared — same refusal
    /// contract as `phase`.
    pub step: Option<u64>,
    /// Expert slots this group would dispatch if executed. Not the
    /// expert IDs (see the type doc for why the host does not have
    /// them) — the cardinality only.
    pub slots: usize,
}

#[cfg(test)]
#[path = "tests/strategy.rs"]
mod tests;
