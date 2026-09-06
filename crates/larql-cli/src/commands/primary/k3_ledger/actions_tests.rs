//! K3-ACTIONS-1. Physical opportunity only — no test here asserts, implies
//! or predicts any behavioural consequence.

use super::super::geometry::k3_reference;
use super::*;

fn census() -> DenseCensus {
    k3_reference().dense_census
}

const BF16: f64 = 16.0;
const GB: f64 = 1e9;

fn q8() -> Codec {
    DENSE_CODECS[0]
}

// ---- representation vs execution role --------------------------------

#[test]
fn the_latent_moe_wrapper_is_dense_by_representation_and_routed_by_function() {
    assert_eq!(
        ExecutionRole::of("LatentMoE wrapper"),
        ExecutionRole::RoutedPathAlwaysOn
    );
    assert_eq!(
        ExecutionRole::of("router"),
        ExecutionRole::RoutedPathAlwaysOn
    );
    assert_eq!(
        ExecutionRole::RoutedPathAlwaysOn.label(),
        "routed-path always-on",
        "filing it as plain dense loses where compressing it acts"
    );
}

#[test]
fn attention_is_dense_path_and_the_head_is_vocabulary() {
    assert_eq!(ExecutionRole::of("KDA self_attn"), ExecutionRole::DensePath);
    assert_eq!(
        ExecutionRole::of("shared_experts"),
        ExecutionRole::DensePath
    );
    assert_eq!(ExecutionRole::of("lm_head"), ExecutionRole::Vocabulary);
    assert_eq!(ExecutionRole::of("embed_tokens"), ExecutionRole::Vocabulary);
    assert_eq!(ExecutionRole::DensePath.label(), "dense-path");
    assert_eq!(ExecutionRole::Vocabulary.label(), "vocabulary");
}

// ---- codec conventions -----------------------------------------------

#[test]
fn codec_widths_are_all_in_not_payload() {
    // R0. Q8_0 is 34 bytes per 32 values, so 8.5 — not 8.
    assert_eq!(q8().all_in_bits, 8.5);
    assert_eq!(DENSE_CODECS[1].all_in_bits, 6.5625, "Q6_K");
    assert_eq!(
        DENSE_CODECS[3].all_in_bits, 4.25,
        "MXFP4 is 4.25 all-in, not 4.0"
    );
}

// ---- pricing ---------------------------------------------------------

#[test]
fn a_gathers_saving_is_on_what_it_touches_not_what_it_stores() {
    // embed_tokens: 2.35 GB resident, 14 KB activated. Compressing it frees
    // resident memory and removes almost no traffic. Pricing it by resident
    // size would make it look like a 1.2 GB traffic win.
    let c = census();
    let f = c.family("embed_tokens").unwrap();
    let a = price(
        "embed_tokens",
        ActionScope::Family,
        f.access,
        f.units,
        f.resident_bytes(),
        f.activated_bytes(),
        q8(),
        BF16,
    )
    .unwrap();
    assert!(
        (a.resident_saving() as f64 / GB - 1.101).abs() < 0.01,
        "a real resident win"
    );
    assert!(a.activated_saving() < 10_000, "and almost no traffic win");
}

#[test]
fn kda_is_the_largest_single_action_at_every_codec() {
    let cat = catalogue(&census(), BF16);
    let top = &cat[0];
    assert_eq!(top.family, "KDA self_attn");
    assert_eq!(
        top.codec, "MXFP4",
        "sorted by saving, so the deepest codec leads"
    );
    assert!(
        (top.activated_saving() as f64 / GB - 44.99).abs() < 0.05,
        "KDA @ MXFP4 got {:.2} GB",
        top.activated_saving() as f64 / GB
    );
    // The conservative end of the same family.
    let q8 = cat
        .iter()
        .find(|a| a.family == "KDA self_attn" && a.codec == "Q8_0")
        .unwrap();
    assert!(
        (q8.activated_saving() as f64 / GB - 28.71).abs() < 0.05,
        "KDA @ Q8 got {:.2} GB",
        q8.activated_saving() as f64 / GB
    );
    // Even the CONSERVATIVE KDA action exceeds all routed traffic.
    assert!(q8.activated_saving() as f64 / GB > 25.83);
}

#[test]
fn one_kda_layer_is_of_the_order_of_the_whole_kimi_exchange_neighbourhood() {
    // Kimi's largest exchange neighbourhood was ~430 MB, and it took five
    // authority runs to close. ONE K3 KDA layer at Q8 is that size.
    let c = census();
    let f = c.family("KDA self_attn").unwrap();
    let per_layer = f.bytes_per_unit;
    let saved = per_layer - (per_layer as f64 * 8.5 / 16.0) as u64;
    assert!(
        (saved as f64 / 1e6 - 415.9).abs() < 1.0,
        "one KDA layer at Q8 saves {:.1} MB",
        saved as f64 / 1e6
    );
}

#[test]
fn savings_scale_monotonically_with_codec_width() {
    let cat = catalogue(&census(), BF16);
    let kda: Vec<&Action> = cat.iter().filter(|a| a.family == "KDA self_attn").collect();
    let q8 = kda.iter().find(|a| a.codec == "Q8_0").unwrap();
    let q6 = kda.iter().find(|a| a.codec == "Q6_K").unwrap();
    let q4 = kda.iter().find(|a| a.codec == "Q4_K").unwrap();
    assert!(q8.activated_saving() < q6.activated_saving());
    assert!(q6.activated_saving() < q4.activated_saving());
}

#[test]
fn a_codec_no_narrower_than_the_stored_width_saves_nothing() {
    let c = census();
    let f = c.family("KDA self_attn").unwrap();
    let wide = Codec {
        name: "BF16",
        all_in_bits: 16.0,
    };
    let a = price(
        "KDA self_attn",
        ActionScope::Family,
        f.access,
        f.units,
        f.resident_bytes(),
        f.activated_bytes(),
        wide,
        BF16,
    )
    .unwrap();
    assert_eq!(a.activated_saving(), 0, "saturating, never negative");
    assert_eq!(a.resident_saving(), 0);
}

#[test]
fn a_zero_or_negative_width_is_refused_not_divided_by() {
    let c = census();
    let f = c.family("KDA self_attn").unwrap();
    let bad = Codec {
        name: "zero",
        all_in_bits: 0.0,
    };
    assert!(price("x", ActionScope::Family, f.access, 1, 1, 1, bad, BF16).is_none());
    assert!(price("x", ActionScope::Family, f.access, 1, 1, 1, q8(), 0.0).is_none());
}

// ---- the catalogue ---------------------------------------------------

#[test]
fn the_catalogue_covers_every_family_at_every_codec_in_both_scopes() {
    let c = census();
    let cat = catalogue(&c, BF16);
    let multi = c.families.iter().filter(|f| f.units > 1).count();
    assert_eq!(cat.len(), (c.families.len() + multi) * DENSE_CODECS.len());
    // Single-unit families have no separate per-layer scope to offer.
    assert!(cat
        .iter()
        .filter(|a| a.family == "lm_head")
        .all(|a| a.scope == ActionScope::Family));
}

#[test]
fn a_family_wide_figure_is_a_ceiling_and_a_layer_is_the_candidate() {
    let cat = catalogue(&census(), BF16);
    let family = cat
        .iter()
        .find(|a| {
            a.family == "KDA self_attn" && a.codec == "Q8_0" && a.scope == ActionScope::Family
        })
        .unwrap();
    let layer = cat
        .iter()
        .find(|a| a.family == "KDA self_attn" && a.codec == "Q8_0" && a.scope == ActionScope::Layer)
        .unwrap();

    assert!(
        !family.scope.is_atomic_candidate(),
        "no ONE run can decide 69 layers"
    );
    assert!(layer.scope.is_atomic_candidate());
    assert!(family.scope.label().contains("CEILING"));
    assert_eq!(layer.scope.label(), "layer");

    assert_eq!(family.units, 69);
    assert_eq!(layer.units, 1);
    // 28.71 GB vs 415.9 MB — the same arithmetic, wildly different claims.
    assert!((family.activated_saving() as f64 / GB - 28.71).abs() < 0.05);
    assert!((layer.activated_saving() as f64 / 1e6 - 415.9).abs() < 1.0);
    assert_eq!(family.activated_saving() / 69, layer.activated_saving());
}

#[test]
fn ceilings_sum_family_scope_only_and_do_not_double_count() {
    // Both scopes are in the catalogue; a ceiling that summed both would
    // report roughly 1 + 1/69 of the truth.
    let c = census();
    let by_hand: u64 = catalogue(&c, BF16)
        .iter()
        .filter(|a| a.codec == "Q8_0" && a.scope == ActionScope::Family)
        .map(Action::activated_saving)
        .sum();
    assert_eq!(whole_side_ceiling(&c, BF16, DENSE_CODECS[0]), by_hand);
}

#[test]
fn whole_side_ceilings_are_physical_not_behavioural() {
    let c = census();
    // If EVERY dense family moved, activated traffic falls by these. Each
    // assumes the entire REPRESENT programme succeeded.
    let q8 = whole_side_ceiling(&c, BF16, DENSE_CODECS[0]) as f64 / GB;
    let q6 = whole_side_ceiling(&c, BF16, DENSE_CODECS[1]) as f64 / GB;
    let q4 = whole_side_ceiling(&c, BF16, DENSE_CODECS[2]) as f64 / GB;
    assert!((q8 - 52.11).abs() < 0.05, "Q8 ceiling {q8:.2}");
    assert!((q6 - 65.57).abs() < 0.05, "Q6 ceiling {q6:.2}");
    assert!((q4 - 79.90).abs() < 0.05, "Q4 ceiling {q4:.2}");
    // Against 111.16 GB of dense traffic and 25.83 GB routed.
    assert!(
        q8 > 25.83,
        "even Q8 across the dense side exceeds ALL routed traffic"
    );
}

#[test]
fn the_routed_path_always_on_surfaces_are_a_named_slice_of_the_catalogue() {
    let cat = catalogue(&census(), BF16);
    let routed_side: u64 = cat
        .iter()
        .filter(|a| {
            a.role == ExecutionRole::RoutedPathAlwaysOn
                && a.codec == "Q8_0"
                // Family scope ONLY: the catalogue also carries per-layer
                // entries, and summing both reports ~1 + 1/92 of the truth.
                && a.scope == ActionScope::Family
        })
        .map(Action::activated_saving)
        .sum();
    // Wrapper + router at Q8: 10.63 GB x (1 - 8.5/16).
    assert!(
        (routed_side as f64 / GB - 4.99).abs() < 0.05,
        "got {:.2} GB",
        routed_side as f64 / GB
    );
}

#[test]
fn an_action_round_trips_through_serde() {
    let a = catalogue(&census(), BF16).remove(0);
    let back: Action = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
    assert_eq!(a, back);
}
