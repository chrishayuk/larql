//! The census multiplies one layer by 69. This is what turns that into a
//! witnessed fact — and what it must refuse to call homogeneous.

use super::*;

fn profile(index: usize, pairs: &[(&str, u64)]) -> LayerProfile {
    LayerProfile {
        index,
        families: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
    }
}

/// Three KDA layers with K3's measured attention weight.
fn kda(indices: &[usize]) -> Vec<LayerProfile> {
    indices
        .iter()
        .map(|i| profile(*i, &[("self_attn", 887_800_832)]))
        .collect()
}

#[test]
fn identical_layers_are_witnessed_homogeneous() {
    let w = witness_family("KDA", &kda(&[1, 2, 3]), &["self_attn"]).expect("non-empty");
    assert!(w.is_homogeneous());
    assert_eq!(w.members, 3);
    assert_eq!(w.reference, 1, "the first layer is the reference");
    assert!(w.divergences.is_empty());
}

#[test]
fn one_byte_of_difference_refuses_the_witness() {
    let mut layers = kda(&[1, 2, 3]);
    layers[2].families.insert("self_attn".into(), 887_800_831);
    let w = witness_family("KDA", &layers, &["self_attn"]).unwrap();
    assert!(!w.is_homogeneous());
    assert_eq!(
        w.divergences,
        vec![Divergence::ByteCountDiffers {
            index: 3,
            role: "self_attn".into(),
            found: 887_800_831,
            expected: 887_800_832,
        }]
    );
}

#[test]
fn equal_bytes_with_different_roles_is_still_heterogeneous() {
    // The failure a byte-only check misses: same weight, different layer.
    let a = profile(1, &[("self_attn", 100), ("router", 50)]);
    let b = profile(2, &[("self_attn", 100), ("LatentMoE wrapper", 50)]);
    assert_eq!(a.total_bytes(), b.total_bytes(), "identical totals");

    let w = witness_family(
        "MoE",
        &[a, b],
        &["self_attn", "router", "LatentMoE wrapper"],
    )
    .unwrap();
    assert!(
        !w.is_homogeneous(),
        "equal totals must not pass as equal structure"
    );
    assert!(matches!(
        w.divergences[0],
        Divergence::RoleSetDiffers { index: 2, .. }
    ));
}

#[test]
fn a_role_set_difference_does_not_also_report_byte_differences() {
    // One finding per layer: a layer with the wrong roles is reported once,
    // not once per role it happens to weigh differently.
    let a = profile(1, &[("self_attn", 100), ("router", 50)]);
    let b = profile(2, &[("self_attn", 999)]);
    let w = witness_family("X", &[a, b], &["self_attn", "router"]).unwrap();
    assert_eq!(w.divergences.len(), 1);
}

#[test]
fn roles_outside_the_checked_set_are_ignored() {
    // A KDA check must not fail because two layers differ on something it
    // is not asking about.
    let a = profile(1, &[("self_attn", 100), ("norms / res_proj", 7)]);
    let b = profile(2, &[("self_attn", 100), ("norms / res_proj", 9)]);
    let w = witness_family("KDA", &[a, b], &["self_attn"]).unwrap();
    assert!(w.is_homogeneous());
}

#[test]
fn an_empty_family_yields_no_witness() {
    assert!(witness_family("KDA", &[], &["self_attn"]).is_none());
}

#[test]
fn a_single_member_family_is_trivially_homogeneous() {
    let w = witness_family("dense MLP", &kda(&[0]), &["self_attn"]).unwrap();
    assert!(w.is_homogeneous());
    assert_eq!(w.members, 1);
}

// ---- the whole-model shape -------------------------------------------

/// K3 in miniature: layer 0 dense, 1-3 MoE, layer 2 the MLA one.
fn k3_shaped() -> Vec<LayerProfile> {
    vec![
        profile(
            0,
            &[("self_attn", 887_800_832), ("dense MLP", 1_453_326_336)],
        ),
        profile(
            1,
            &[
                ("self_attn", 887_800_832),
                ("shared_experts", 264_241_152),
                ("LatentMoE wrapper", 102_767_616),
                ("router", 12_848_640),
            ],
        ),
        profile(
            2,
            &[
                ("self_attn", 464_392_192),
                ("shared_experts", 264_241_152),
                ("LatentMoE wrapper", 102_767_616),
                ("router", 12_848_640),
            ],
        ),
        profile(
            3,
            &[
                ("self_attn", 887_800_832),
                ("shared_experts", 264_241_152),
                ("LatentMoE wrapper", 102_767_616),
                ("router", 12_848_640),
            ],
        ),
    ]
}

#[test]
fn the_dense_layer_lacking_a_router_is_topology_not_heterogeneity() {
    // Layer 0 has no router, shared experts or wrapper. Reporting that as a
    // divergence would be reporting `first_k_dense_replace` as a defect.
    let w = HomogeneityWitness::build(&k3_shaped(), &[0, 1, 3], LayerTopology::new(4, 1));
    assert!(w.all_homogeneous(), "{:?}", w.families);
    assert_eq!(w.divergence_count(), 0);

    let names: Vec<&str> = w.families.iter().map(|f| f.family.as_str()).collect();
    assert!(names.contains(&"KDA self_attn"));
    assert!(names.contains(&"MLA self_attn"));
    assert!(names.contains(&"MoE surfaces"));
}

#[test]
fn kda_and_mla_are_checked_separately_not_against_each_other() {
    // They legitimately differ — 887.80 MB vs 464.39 MB. A single
    // whole-model check would call that heterogeneity.
    let w = HomogeneityWitness::build(&k3_shaped(), &[0, 1, 3], LayerTopology::new(4, 1));
    let kda = w
        .families
        .iter()
        .find(|f| f.family == "KDA self_attn")
        .unwrap();
    let mla = w
        .families
        .iter()
        .find(|f| f.family == "MLA self_attn")
        .unwrap();
    assert_eq!(kda.members, 3);
    assert_eq!(mla.members, 1);
    assert!(kda.is_homogeneous() && mla.is_homogeneous());
}

#[test]
fn a_divergent_moe_layer_is_caught_across_attention_families() {
    // MoE surfaces are checked over every MoE layer, KDA and MLA alike, so
    // a wrapper that differs only on MLA layers cannot hide.
    let mut layers = k3_shaped();
    layers[2].families.insert("LatentMoE wrapper".into(), 1);
    let w = HomogeneityWitness::build(&layers, &[0, 1, 3], LayerTopology::new(4, 1));
    assert!(!w.all_homogeneous());
    assert_eq!(w.divergence_count(), 1);
}

#[test]
fn a_witness_round_trips_through_serde() {
    let w = HomogeneityWitness::build(&k3_shaped(), &[0, 1, 3], LayerTopology::new(4, 1));
    let back: HomogeneityWitness =
        serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
    assert_eq!(w, back);
    assert_eq!(profile(7, &[("a", 1), ("b", 2)]).total_bytes(), 3);
}
