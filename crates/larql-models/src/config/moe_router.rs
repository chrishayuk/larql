//! How a mixture-of-experts router turns logits into expert weights.
//!
//! **This is a typed enum because it used to be a `&str`.** The compute layer
//! matched the literal `"gemma4_top_k_softmax"` and fell through to a default
//! for anything else, so `gpt_oss`'s `"gpt_oss_topk_then_softmax"` — a
//! genuinely different rule — silently selected the ordinary policy. That is
//! the [§4.7.8] shape once more: a behaviour that is a property of the
//! architecture, expressed as a string, with a default quietly answering for
//! every value nobody remembered to add.
//!
//! The string survives as the *wire* form only: `MoeConfig.router_type` in a
//! vindex is serialised text, so [`MoeRouterKind::as_str`] and
//! [`MoeRouterKind::from_wire`] own that conversion in one place.
//!
//! [§4.7.8]: ../../../../docs/k3-funnel.md

/// Wire value for [`MoeRouterKind::TopKSoftmax`].
pub const ROUTER_WIRE_TOP_K_SOFTMAX: &str = "top_k_softmax";
/// Wire value for [`MoeRouterKind::Gemma4Hybrid`].
pub const ROUTER_WIRE_GEMMA4_TOP_K_SOFTMAX: &str = "gemma4_top_k_softmax";
/// Wire value for [`MoeRouterKind::TopKThenSoftmax`].
pub const ROUTER_WIRE_GPT_OSS_TOPK_THEN_SOFTMAX: &str = "gpt_oss_topk_then_softmax";

/// The routing rule an architecture's MoE block uses.
///
/// Serialises to the same wire values as [`MoeRouterKind::as_str`] — one
/// spelling per fact, whether it travels in a VINDEX2 `MoeConfig` or a
/// VINDEX3 execution surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MoeRouterKind {
    /// Softmax over all experts, then take the top-k weights as they are.
    #[default]
    #[serde(rename = "top_k_softmax")]
    TopKSoftmax,
    /// Gemma 4's hybrid: normalised softmax plus a per-expert scale.
    #[serde(rename = "gemma4_top_k_softmax")]
    Gemma4Hybrid,
    /// Select the top-k logits *first*, then softmax over just those — the
    /// selected weights sum to 1. GPT-OSS.
    #[serde(rename = "gpt_oss_topk_then_softmax")]
    TopKThenSoftmax,
}

impl MoeRouterKind {
    /// The vindex wire value. Exhaustive on purpose: a new variant fails to
    /// compile here rather than serialising as something else.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopKSoftmax => ROUTER_WIRE_TOP_K_SOFTMAX,
            Self::Gemma4Hybrid => ROUTER_WIRE_GEMMA4_TOP_K_SOFTMAX,
            Self::TopKThenSoftmax => ROUTER_WIRE_GPT_OSS_TOPK_THEN_SOFTMAX,
        }
    }

    /// Parse a wire value. `None` for anything unrecognised — a caller must
    /// decide what an unknown router means rather than inheriting a default
    /// that silently computes a different rule.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            ROUTER_WIRE_TOP_K_SOFTMAX => Some(Self::TopKSoftmax),
            ROUTER_WIRE_GEMMA4_TOP_K_SOFTMAX => Some(Self::Gemma4Hybrid),
            ROUTER_WIRE_GPT_OSS_TOPK_THEN_SOFTMAX => Some(Self::TopKThenSoftmax),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant must round-trip, or a vindex written today fails to load
    /// tomorrow.
    #[test]
    fn every_variant_round_trips_through_the_wire_form() {
        for k in [
            MoeRouterKind::TopKSoftmax,
            MoeRouterKind::Gemma4Hybrid,
            MoeRouterKind::TopKThenSoftmax,
        ] {
            assert_eq!(MoeRouterKind::from_wire(k.as_str()), Some(k), "{k:?}");
        }
    }

    /// The wire strings are a published format — pin them so a rename of the
    /// Rust variant cannot silently change what lands in a vindex.
    #[test]
    fn wire_values_are_pinned() {
        assert_eq!(MoeRouterKind::TopKSoftmax.as_str(), "top_k_softmax");
        assert_eq!(MoeRouterKind::Gemma4Hybrid.as_str(), "gemma4_top_k_softmax");
        assert_eq!(
            MoeRouterKind::TopKThenSoftmax.as_str(),
            "gpt_oss_topk_then_softmax"
        );
    }

    /// An unknown router must not resolve to the default — that silent
    /// fallback is the defect this type replaces.
    #[test]
    fn unknown_wire_value_does_not_become_the_default() {
        assert_eq!(MoeRouterKind::from_wire("some_future_router"), None);
        assert_eq!(MoeRouterKind::from_wire(""), None);
    }

    /// The default is the ordinary rule, and it is the one most families use.
    #[test]
    fn default_is_top_k_softmax() {
        assert_eq!(MoeRouterKind::default(), MoeRouterKind::TopKSoftmax);
    }
}
