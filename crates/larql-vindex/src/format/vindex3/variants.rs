//! Representation variants — a region set carries bytes, a profile picks which
//! (spec §9.1).
//!
//! # The one legal representation model
//!
//! A profile saying `"format": "mxfp4"` cannot turn Q6_K bytes into MXFP4 bytes
//! by declaration. Exactly one model is legal: **a region set may carry several
//! physically present variants, and a profile selects a present one.** There is
//! no runtime conversion anywhere in this module, and that is the point — "the
//! bytes executed are the bytes stored" (§10) holds by construction rather than
//! by discipline.
//!
//! # Why the refusal is the interesting part
//!
//! Selecting a variant that was never extracted is the failure this type
//! exists to make loud. It has to fail **closed**, and it has to fail naming
//! three things: the region set, the variant that was asked for, and the
//! variants actually present. A bare "variant not found" sends someone to
//! re-extract a terabyte before discovering that the pack they wanted is
//! spelled `native-mxfp4` and they asked for `mxfp4`.
//!
//! It also has to fail **before any byte is read**. A container whose profile
//! selects an absent variant is unservable, and finding that out during a bind
//! — after the segment files are resident — turns a naming error into a
//! partially-initialised load.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::format::capability::authority::Fidelity;

/// One physically present encoding of a region set.
///
/// `storage` is a segment key relative to the container root, the same
/// vocabulary `index.segments` uses — a key the loader composes a filename
/// from, never a filename it globbed for (§12.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredVariant {
    #[serde(deserialize_with = "super::container_path::serde_rules::relative")]
    pub storage: String,
    pub fidelity: Fidelity,
}

impl StoredVariant {
    pub fn new(storage: impl Into<String>, fidelity: Fidelity) -> Self {
        Self {
            storage: storage.into(),
            fidelity,
        }
    }
}

/// The variants one region set physically carries, and which of them is
/// canonical.
///
/// `baseline` is the authority: additional variants are opt-in, per-component
/// and individually removable, so a pack can be deleted without the region set
/// becoming unservable. That only holds if the baseline is itself present,
/// which [`Self::validate`] is what checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionSetVariants {
    pub baseline: String,
    pub variants: BTreeMap<String, StoredVariant>,
}

impl RegionSetVariants {
    /// A region set carrying exactly one variant, which is therefore also the
    /// baseline — the shape every container written before packs existed has.
    pub fn single(name: impl Into<String>, stored: StoredVariant) -> Self {
        let name = name.into();
        Self {
            baseline: name.clone(),
            variants: BTreeMap::from([(name, stored)]),
        }
    }

    /// Add a variant beside the baseline, which is never rewritten (§9.1).
    pub fn with_variant(mut self, name: impl Into<String>, stored: StoredVariant) -> Self {
        self.variants.insert(name.into(), stored);
        self
    }

    /// Present variant names, sorted.
    ///
    /// Sorted because this list goes straight into a refusal message, and a
    /// diagnostic whose contents reorder between runs cannot be diffed.
    pub fn present(&self) -> Vec<String> {
        self.variants.keys().cloned().collect()
    }

    /// Resolve `name` against what is physically here.
    pub fn select(&self, region_set: &str, name: &str) -> Result<&StoredVariant, VariantDefect> {
        self.variants
            .get(name)
            .ok_or_else(|| VariantDefect::AbsentVariant {
                region_set: region_set.to_string(),
                requested: name.to_string(),
                present: self.present(),
            })
    }

    /// Resolve the baseline — what a profile that names no variant gets.
    pub fn select_baseline(&self, region_set: &str) -> Result<&StoredVariant, VariantDefect> {
        self.select(region_set, &self.baseline)
    }

    /// Structural checks that do not depend on any profile.
    fn validate(&self, region_set: &str, out: &mut Vec<VariantDefect>) {
        if self.variants.is_empty() {
            out.push(VariantDefect::NoVariants {
                region_set: region_set.to_string(),
            });
            return;
        }
        if !self.variants.contains_key(&self.baseline) {
            out.push(VariantDefect::BaselineAbsent {
                region_set: region_set.to_string(),
                baseline: self.baseline.clone(),
                present: self.present(),
            });
        }
    }
}

/// Every region set a container catalogues, and the variants each carries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VariantCatalogue {
    sets: BTreeMap<String, RegionSetVariants>,
}

impl VariantCatalogue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_set(mut self, region_set: impl Into<String>, variants: RegionSetVariants) -> Self {
        self.sets.insert(region_set.into(), variants);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Catalogued region-set names, sorted — the other half of a refusal.
    pub fn region_sets(&self) -> Vec<String> {
        self.sets.keys().cloned().collect()
    }

    pub fn get(&self, region_set: &str) -> Option<&RegionSetVariants> {
        self.sets.get(region_set)
    }

    /// Resolve one `region_set::variant` pair.
    pub fn select(&self, region_set: &str, variant: &str) -> Result<&StoredVariant, VariantDefect> {
        self.set(region_set)?.select(region_set, variant)
    }

    /// Resolve a region set's baseline.
    pub fn select_baseline(&self, region_set: &str) -> Result<&StoredVariant, VariantDefect> {
        self.set(region_set)?.select_baseline(region_set)
    }

    fn set(&self, region_set: &str) -> Result<&RegionSetVariants, VariantDefect> {
        self.sets
            .get(region_set)
            .ok_or_else(|| VariantDefect::UnknownRegionSet {
                region_set: region_set.to_string(),
                catalogued: self.region_sets(),
            })
    }

    /// Every structural defect, rather than the first.
    ///
    /// All of them, because a container with two broken region sets should
    /// take one repair pass and not two: fixing what the first error names
    /// only to be shown the second is how a load-time check earns a reputation
    /// for wasting time.
    pub fn defects(&self) -> Vec<VariantDefect> {
        let mut out = Vec::new();
        for (region_set, variants) in &self.sets {
            variants.validate(region_set, &mut out);
        }
        out
    }
}

/// Why a variant selection cannot be honoured.
///
/// Every variant names what was asked for *and* what is there, because the
/// remedy differs and only the pair distinguishes them: a typo is fixed in the
/// profile, a genuinely missing pack is fixed by extracting it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VariantDefect {
    #[error(
        "region set '{region_set}' has no variant '{requested}'; present: [{}]",
        present.join(", ")
    )]
    AbsentVariant {
        region_set: String,
        requested: String,
        present: Vec<String>,
    },

    #[error(
        "unknown region set '{region_set}'; catalogued: [{}]",
        catalogued.join(", ")
    )]
    UnknownRegionSet {
        region_set: String,
        catalogued: Vec<String>,
    },

    #[error(
        "region set '{region_set}' names baseline '{baseline}', which is not present; present: [{}]",
        present.join(", ")
    )]
    BaselineAbsent {
        region_set: String,
        baseline: String,
        present: Vec<String>,
    },

    #[error("region set '{region_set}' catalogues no variants at all")]
    NoVariants { region_set: String },
}

#[cfg(test)]
#[path = "variants_tests.rs"]
mod tests;
