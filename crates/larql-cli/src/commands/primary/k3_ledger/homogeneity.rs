//! **Is one measured layer representative of its family?**
//!
//! [`super::access::DenseCensus`] multiplies ONE measured KDA layer by 69
//! and ONE measured MLA layer by 24. That is sound only if layers within a
//! family are structurally identical, which the shard sizes suggest and
//! nothing had checked.
//!
//! The assumption is load-bearing: every per-layer opportunity figure in
//! K3-PRECISION-1A divides by it. So it is witnessed from headers BEFORE
//! any weight is read, rather than discovered incidentally later.
//!
//! ```text
//! ASSUMED     layer 49 x 69 == every KDA layer
//! WITNESSED   all 69 KDA layers carry the same roles at the same bytes
//! ```
//!
//! A family is homogeneous when every member has the same set of family
//! roles AND the same total bytes per role. Both halves matter: equal
//! totals with different roles would be a different layer that happens to
//! weigh the same, which is exactly the failure a byte-only check misses.
//!
//! [`super::access::LayerKind`] decides which families a layer SHOULD
//! have — a dense layer legitimately has no router, and flagging that as
//! heterogeneity would be reporting the topology as a defect.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::access::{LayerKind, LayerTopology};

/// One layer's measured family byte totals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerProfile {
    /// Layer index, 0-based against tensor names.
    pub index: usize,
    /// Family name -> total bytes in that family for this layer.
    pub families: BTreeMap<String, u64>,
}

impl LayerProfile {
    /// Total bytes across every family.
    pub fn total_bytes(&self) -> u64 {
        self.families.values().sum()
    }
}

/// Why a family is not homogeneous.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Divergence {
    /// A layer carries a different set of roles than the family's first.
    RoleSetDiffers {
        /// The layer that differs.
        index: usize,
        /// What it has.
        found: Vec<String>,
        /// What the reference layer has.
        expected: Vec<String>,
    },
    /// A role is present everywhere but weighs differently.
    ByteCountDiffers {
        /// The layer that differs.
        index: usize,
        /// Which role.
        role: String,
        /// Its bytes here.
        found: u64,
        /// Its bytes in the reference layer.
        expected: u64,
    },
}

/// The verdict for one family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyWitness {
    /// Which family — `KDA`, `MLA`, `MoE`.
    pub family: String,
    /// How many layers were checked.
    pub members: usize,
    /// The layer every other was compared against.
    pub reference: usize,
    /// Everything that differed. Empty means witnessed homogeneous.
    pub divergences: Vec<Divergence>,
}

impl FamilyWitness {
    /// Whether multiplying the reference layer by `members` is a witnessed
    /// fact rather than an assumption.
    pub fn is_homogeneous(&self) -> bool {
        self.divergences.is_empty()
    }
}

/// Check one family's layers against the first of them.
///
/// `roles` restricts the comparison to the families that layer kind is
/// expected to have — a dense layer has no router, and reporting that as
/// heterogeneity would be reporting the topology.
pub fn witness_family(
    family: &str,
    layers: &[LayerProfile],
    roles: &[&str],
) -> Option<FamilyWitness> {
    let (reference, rest) = layers.split_first()?;
    fn keep<'a>(p: &'a LayerProfile, roles: &[&str]) -> BTreeMap<&'a str, u64> {
        p.families
            .iter()
            .filter(|(k, _)| roles.contains(&k.as_str()))
            .map(|(k, v)| (k.as_str(), *v))
            .collect()
    }
    let want = keep(reference, roles);
    let mut divergences = Vec::new();

    for layer in rest {
        let got = keep(layer, roles);
        if got.keys().ne(want.keys()) {
            divergences.push(Divergence::RoleSetDiffers {
                index: layer.index,
                found: got.keys().map(|s| (*s).to_string()).collect(),
                expected: want.keys().map(|s| (*s).to_string()).collect(),
            });
            continue;
        }
        for (role, bytes) in &want {
            let found = got.get(role).copied().unwrap_or(0);
            if found != *bytes {
                divergences.push(Divergence::ByteCountDiffers {
                    index: layer.index,
                    role: (*role).to_string(),
                    found,
                    expected: *bytes,
                });
            }
        }
    }
    Some(FamilyWitness {
        family: family.to_string(),
        members: layers.len(),
        reference: reference.index,
        divergences,
    })
}

/// The whole-model witness: attention families and the MoE surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomogeneityWitness {
    /// One entry per family checked.
    pub families: Vec<FamilyWitness>,
}

impl HomogeneityWitness {
    /// Build the witness from every layer profile and the model's topology.
    ///
    /// `kda_layers` is the 0-based tensor index set for the KDA family; the
    /// rest of the decoder is MLA. MoE surfaces are checked across every
    /// layer the topology calls `Moe`, regardless of attention family.
    pub fn build(profiles: &[LayerProfile], kda_layers: &[usize], topology: LayerTopology) -> Self {
        let pick = |f: &dyn Fn(&LayerProfile) -> bool| -> Vec<LayerProfile> {
            profiles.iter().filter(|p| f(p)).cloned().collect()
        };
        let kda = pick(&|p| kda_layers.contains(&p.index));
        let mla = pick(&|p| !kda_layers.contains(&p.index));
        let moe = pick(&|p| topology.kind(p.index) == LayerKind::Moe);
        let dense = pick(&|p| topology.kind(p.index) == LayerKind::Dense);

        let mut families = Vec::new();
        families.extend(witness_family("KDA self_attn", &kda, &["self_attn"]));
        families.extend(witness_family("MLA self_attn", &mla, &["self_attn"]));
        families.extend(witness_family(
            "MoE surfaces",
            &moe,
            &["shared_experts", "LatentMoE wrapper", "router"],
        ));
        families.extend(witness_family("dense MLP", &dense, &["dense MLP"]));
        Self { families }
    }

    /// Whether every family checked is homogeneous.
    pub fn all_homogeneous(&self) -> bool {
        self.families.iter().all(FamilyWitness::is_homogeneous)
    }

    /// Every divergence found, across all families.
    pub fn divergence_count(&self) -> usize {
        self.families.iter().map(|f| f.divergences.len()).sum()
    }
}

#[cfg(test)]
#[path = "homogeneity_tests.rs"]
mod homogeneity_tests;
