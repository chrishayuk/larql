//! **L1 of LOWERING-PLUGIN-1: a lowering provider names itself.**
//!
//! The forecast (`docs/represent/forecasts/lowering-plugin-1.json`)
//! predicts that every `PlanBackend` states a versioned identity and that
//! the diagnostic name stays non-authoritative. Three things are held
//! here, and the third is the control that keeps the first two from being
//! a rename: the in-tree providers each state a valid, distinct identity;
//! an identity is the PROVIDER's and not its configuration's; and the two
//! fields are independent in both directions — one name over two
//! identities, and one identity under two names.

use super::super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
    PlanBackend, ProjectCall, RoutedFfnCall, WeightFormat, WeightFormats, WeightSlice,
};
use super::super::device::DevicePlanBackend;
use super::super::lowering::LoweringIdentity;
use super::super::production::ProductionBackend;
use super::super::reference::ReferenceBackend;
use crate::error::VindexError;
use larql_compute::cpu::CpuBackend;

fn in_tree() -> Vec<(&'static str, LoweringIdentity)> {
    vec![
        ("reference", ReferenceBackend::new().identity()),
        ("production", ProductionBackend::new().identity()),
        (
            "device",
            DevicePlanBackend::new(CpuBackend, "cpu-as-device", WeightFormat::F32).identity(),
        ),
    ]
}

#[test]
fn every_in_tree_provider_states_a_valid_distinct_identity() {
    let providers = in_tree();
    for (who, id) in &providers {
        id.validate().unwrap_or_else(|e| panic!("{who}: {e}"));
        assert_eq!(id.revision, 1, "{who}: every provider starts at revision 1");
    }
    let families: Vec<&str> = providers.iter().map(|(_, id)| id.family.as_str()).collect();
    assert_eq!(families, ["reference", "cpu-production", "device-matmul"]);
    let mut distinct = providers
        .iter()
        .map(|(_, id)| id.clone())
        .collect::<Vec<_>>();
    distinct.sort();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        providers.len(),
        "two providers share an identity"
    );
}

/// A provider's identity is stable across instances and across the
/// configuration it is built with. The device backend is the case that
/// matters: its NAME is per instance and names the device and realisation
/// (`metal-r2-f16`), and its format table changes what it asks the loader
/// for — neither is who the provider is.
#[test]
fn identity_is_the_providers_not_its_configurations() {
    assert_eq!(
        ProductionBackend::new().identity(),
        ProductionBackend::new().identity()
    );
    let f32_device = DevicePlanBackend::new(CpuBackend, "device-r1-f32", WeightFormat::F32);
    let f16_device = DevicePlanBackend::with_formats(
        CpuBackend,
        "device-r2-f16",
        WeightFormats::uniform(WeightFormat::F16),
    );
    assert_ne!(f32_device.name(), f16_device.name());
    assert_eq!(f32_device.identity(), f16_device.identity());
    assert_eq!(f32_device.identity().to_string(), "device-matmul/v1");
}

/// A backend that answers for another's arithmetic under a name and an
/// identity of its own choosing — so the two fields can be varied
/// independently, which is the only way to show one is not derived from
/// the other.
struct Relabelled {
    inner: ReferenceBackend,
    name: &'static str,
    identity: LoweringIdentity,
}

impl Relabelled {
    fn new(name: &'static str, family: &str, revision: u32) -> Self {
        Self {
            inner: ReferenceBackend::new(),
            name,
            identity: LoweringIdentity::new(family, revision),
        }
    }
}

impl PlanBackend for Relabelled {
    fn name(&self) -> &str {
        self.name
    }

    fn identity(&self) -> LoweringIdentity {
        self.identity.clone()
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.inner.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.inner.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.inner.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.inner.attention_step(call)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.routed_ffn(call)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.inner
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.inner.residual_add(acc, delta)
    }
}

/// The control. One diagnostic name over two identities are two
/// providers; one identity under two names is one provider. A field that
/// were a rename of the name could satisfy neither half.
#[test]
fn identity_is_authority_and_name_is_presentation() {
    let same_name_a = Relabelled::new("metal", "provider-a", 1);
    let same_name_b = Relabelled::new("metal", "provider-b", 1);
    assert_eq!(same_name_a.name(), same_name_b.name());
    assert_ne!(same_name_a.identity(), same_name_b.identity());

    let same_id_a = Relabelled::new("metal-r2-f16", "provider-a", 1);
    let same_id_b = Relabelled::new("metal-r3-nvfp4", "provider-a", 1);
    assert_ne!(same_id_a.name(), same_id_b.name());
    assert_eq!(same_id_a.identity(), same_id_b.identity());

    // And a revision is part of the identity: the same family at another
    // revision is a different authority, whatever it is called.
    let revised = Relabelled::new("metal-r2-f16", "provider-a", 2);
    assert_eq!(revised.name(), same_id_a.name());
    assert_ne!(revised.identity(), same_id_a.identity());

    // Dispatched through the trait object, the identity is the one the
    // provider stated — nothing between the caller and the provider
    // substitutes a default.
    let erased: &dyn PlanBackend = &same_name_b;
    assert_eq!(erased.identity(), LoweringIdentity::new("provider-b", 1));
}

/// An identity a provider could state and a registry could not key is
/// refused by name — the shape L2 will lean on.
#[test]
fn an_unkeyable_identity_is_refused_before_anything_relies_on_it() {
    for (family, revision, what) in [
        ("", 1, "empty"),
        ("cpu production", 1, "` `"),
        ("cpu/v1", 1, "`/`"),
        ("cpu-production", 0, "revision 0"),
    ] {
        let err = Relabelled::new("x", family, revision)
            .identity()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains(what), "{family:?}/{revision}: {err}");
    }
}
