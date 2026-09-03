//! **K3-ACTIONS-1 — the complete physical action catalogue, weight-free.**
//!
//! Every dense-side representation action K3 admits, with its exact
//! physical opportunity, derived from headers alone. No forward pass, no
//! weights, no behavioural claim of any kind.
//!
//! ```text
//! family x layer x codec  ->  resident saving, activated saving
//! ```
//!
//! This is what a beam search will generate candidates from, and what
//! prices them. Behaviour is measured later and separately; nothing here
//! predicts it.
//!
//! # Representation is not execution role
//!
//! The LatentMoE wrapper — `routed_expert_up_proj` / `down_proj`, 9.45
//! GB/token across 92 layers — is dense BF16 by REPRESENTATION and routed
//! by FUNCTION. It wraps the MXFP4 expert bank, at 36.6% of the size of
//! the traffic it serves.
//!
//! ```text
//! representation   dense BF16
//! execution role   routed-path always-on
//! ```
//!
//! Filing it simply under "dense" would lose the fact that compressing it
//! changes the balance between compute and expert-fetch on the routed
//! path, not just total bandwidth. [`ExecutionRole`] keeps the two axes
//! apart, the way [`super::access::AccessMode`] keeps existence apart from
//! touch.

use serde::{Deserialize, Serialize};

use super::access::{AccessMode, DenseCensus};

/// Where a family sits in the execution graph.
///
/// Orthogonal to how it is represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRole {
    /// Always-on and not attached to routing: attention, norms.
    DensePath,
    /// BF16 machinery on the ROUTED path, read every token whatever the
    /// router picks: the LatentMoE wrapper and the router itself.
    RoutedPathAlwaysOn,
    /// The vocabulary surfaces — head and embeddings.
    Vocabulary,
}

impl ExecutionRole {
    /// How a report names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::DensePath => "dense-path",
            Self::RoutedPathAlwaysOn => "routed-path always-on",
            Self::Vocabulary => "vocabulary",
        }
    }

    /// Classify a census family by name.
    ///
    /// The wrapper and router are the two BF16 surfaces attached to routed
    /// execution; everything else on the dense side is genuinely dense.
    pub fn of(family: &str) -> Self {
        match family {
            "LatentMoE wrapper" | "router" => Self::RoutedPathAlwaysOn,
            "lm_head" | "embed_tokens" => Self::Vocabulary,
            _ => Self::DensePath,
        }
    }
}

/// What one action COVERS.
///
/// The distinction is load-bearing and easy to lose:
///
/// ```text
/// KDA family    -> Q8   saves 28.71 GB/token   a physical CEILING
/// one KDA layer -> Q8   saves    415.9 MB      a search CANDIDATE
/// ```
///
/// A family-wide figure is what the opportunity would be if EVERY member
/// moved and every one of them earned it behaviourally. It is not a thing
/// anyone can measure in one authority run. Reading it as an atomic
/// candidate would let a beam policy — or an agent — propose "quantise
/// KDA" as though it were one decision with one verdict, which is the
/// scalarization error one level up from proxies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionScope {
    /// One unit. The natural atomic search candidate.
    Layer,
    /// Every unit of a family. A ceiling, never a candidate.
    Family,
    /// Every family sharing an [`ExecutionRole`]. Also a ceiling.
    RoleGroup,
}

impl ActionScope {
    /// Whether one authority run could decide this action.
    ///
    /// Only `Layer` can. The others aggregate members that must each earn
    /// their own admission.
    pub fn is_atomic_candidate(self) -> bool {
        matches!(self, Self::Layer)
    }

    /// How a report names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Layer => "layer",
            Self::Family => "family (CEILING)",
            Self::RoleGroup => "role group (CEILING)",
        }
    }
}

/// A serving format and its ALL-IN width.
///
/// All-in includes block scales — R0: never quote a width without saying
/// which convention it is in. Q8_0 is 34 bytes per 32 values, not 32.
///
/// `Serialize` only: the `&'static str` name keeps the table a `const`,
/// and an [`Action`] records the codec by value rather than borrowing it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Codec {
    /// How a report names it.
    pub name: &'static str,
    /// All-in bits per weight.
    pub all_in_bits: f64,
}

/// The codecs a dense BF16 surface could move to.
pub const DENSE_CODECS: [Codec; 4] = [
    Codec {
        name: "Q8_0",
        all_in_bits: 8.5,
    },
    Codec {
        name: "Q6_K",
        all_in_bits: 6.5625,
    },
    Codec {
        name: "Q4_K",
        all_in_bits: 4.5,
    },
    Codec {
        name: "MXFP4",
        all_in_bits: 4.25,
    },
];

/// One representation action, priced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// Which census family.
    pub family: String,
    /// What this action covers — and therefore whether it is a candidate
    /// or a ceiling.
    pub scope: ActionScope,
    /// Where it sits in the execution graph.
    pub role: ExecutionRole,
    /// How it is touched.
    pub access: AccessMode,
    /// How many units the action covers. A whole-family action covers all
    /// of them; a per-layer action covers one.
    pub units: usize,
    /// Bytes resident today, across `units`.
    pub resident_before: u64,
    /// Bytes activated per token today, across `units`.
    pub activated_before: u64,
    /// The target format's name.
    pub codec: String,
    /// Its all-in width.
    pub codec_all_in_bits: f64,
    /// Bytes resident after.
    pub resident_after: u64,
    /// Bytes activated per token after.
    pub activated_after: u64,
}

impl Action {
    /// Resident bytes freed.
    pub fn resident_saving(&self) -> u64 {
        self.resident_before.saturating_sub(self.resident_after)
    }

    /// Activated bytes per token removed.
    pub fn activated_saving(&self) -> u64 {
        self.activated_before.saturating_sub(self.activated_after)
    }
}

/// Price one family at one codec, from its BF16 figures.
///
/// `stored_bits` is what the family is stored at now — read from the
/// checkpoint, never assumed (K3-LEDGER-1b).
pub fn price(
    family: &str,
    scope: ActionScope,
    access: AccessMode,
    units: usize,
    resident_before: u64,
    activated_before: u64,
    codec: Codec,
    stored_bits: f64,
) -> Option<Action> {
    if stored_bits <= 0.0 || codec.all_in_bits <= 0.0 {
        return None;
    }
    let ratio = codec.all_in_bits / stored_bits;
    Some(Action {
        family: family.to_string(),
        scope,
        role: ExecutionRole::of(family),
        access,
        units,
        resident_before,
        activated_before,
        codec: codec.name.to_string(),
        codec_all_in_bits: codec.all_in_bits,
        resident_after: (resident_before as f64 * ratio) as u64,
        activated_after: (activated_before as f64 * ratio) as u64,
    })
}

/// The complete catalogue: every census family at every dense codec.
///
/// Emits BOTH scopes: the family-wide ceiling AND the per-unit candidate,
/// so a consumer never has to divide by `units` itself and never has to
/// guess which kind of figure it is holding.
pub fn catalogue(census: &DenseCensus, stored_bits: f64) -> Vec<Action> {
    let mut out = Vec::new();
    for f in &census.families {
        for codec in DENSE_CODECS {
            if f.units > 1 {
                if let Some(a) = price(
                    &f.name,
                    ActionScope::Layer,
                    f.access,
                    1,
                    f.bytes_per_unit,
                    f.access.activated_bytes(f.bytes_per_unit),
                    codec,
                    stored_bits,
                ) {
                    out.push(a);
                }
            }
            if let Some(a) = price(
                &f.name,
                ActionScope::Family,
                f.access,
                f.units,
                f.resident_bytes(),
                f.activated_bytes(),
                codec,
                stored_bits,
            ) {
                out.push(a);
            }
        }
    }
    out.sort_by_key(|a| std::cmp::Reverse(a.activated_saving()));
    out
}

/// Total activated saving if EVERY dense family moved to one codec.
///
/// A physical ceiling and nothing more: it assumes every action succeeds
/// behaviourally, which is the entire programme stated as a premise.
pub fn whole_side_ceiling(census: &DenseCensus, stored_bits: f64, codec: Codec) -> u64 {
    catalogue(census, stored_bits)
        .iter()
        .filter(|a| a.codec == codec.name && a.scope == ActionScope::Family)
        .map(Action::activated_saving)
        .sum()
}

#[cfg(test)]
#[path = "actions_tests.rs"]
mod actions_tests;
