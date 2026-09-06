//! K3-LEDGER-2. The two K3-DENSE-1 errors, made unrepresentable.
//!
//! Every byte figure below was measured from `moonshotai/Kimi-K3`
//! safetensors headers on 2026-09-03 via HTTP range requests — no weights
//! downloaded. A census that stops reproducing them has changed meaning.

use super::*;

// Measured per-unit resident bytes, BF16.
const KDA_ATTN: u64 = 887_800_000; // layer 49: q,k,v,g,o each 176.16 MB
const MLA_ATTN: u64 = 464_390_000; // layer 3: g_proj + o_proj dominate
const SHARED: u64 = 264_240_000; // 3 x [6144,7168]
const LATENT: u64 = 102_770_000; // routed_expert up/down [3584,7168] + norm
const ROUTER: u64 = 12_850_000; // gate [896,7168]
const DENSE0_MLP: u64 = 1_453_320_000; // layer 0: 3 x [33792,7168]
                                       // EXACT, not the rounded 2348.81 MB display figure: 163840 * 7168 * 2.
                                       // A rounded constant divides to 14,335 and quietly loses a byte per row.
const LM_HEAD: u64 = 2_348_810_240; // [163840,7168]
const EMBED: u64 = 2_348_810_240; // [163840,7168] — SAME SIZE, different access
const NORMS: u64 = 60_000;

const N_KDA: usize = 69;
const N_MLA: usize = 24;
const N_MOE: usize = 92;
const N_LAYERS: usize = 93;

fn k3_dense_census() -> DenseCensus {
    DenseCensus::new(vec![
        Family::full_read("KDA self_attn", KDA_ATTN, N_KDA),
        Family::full_read("shared_experts", SHARED, N_MOE),
        Family::full_read("MLA self_attn", MLA_ATTN, N_MLA),
        Family::full_read("LatentMoE wrapper", LATENT, N_MOE),
        Family::full_read("lm_head", LM_HEAD, 1),
        Family::full_read("dense layer-0 MLP", DENSE0_MLP, 1),
        Family::full_read("router", ROUTER, N_MOE),
        Family::full_read("norms / res_proj", NORMS, N_LAYERS),
        Family {
            name: "embed_tokens".into(),
            bytes_per_unit: EMBED,
            units: 1,
            access: AccessMode::Gather {
                rows_per_token: 1,
                total_rows: 163_840,
            },
        },
    ])
}

// ---- the embed/lm_head asymmetry -------------------------------------

#[test]
fn two_tensors_of_identical_size_move_wildly_different_bytes() {
    let c = k3_dense_census();
    let head = c.family("lm_head").expect("lm_head");
    let embed = c.family("embed_tokens").expect("embed_tokens");

    assert_eq!(
        head.resident_bytes(),
        embed.resident_bytes(),
        "untied, and byte-identical in the checkpoint"
    );
    assert_eq!(
        head.activated_bytes(),
        LM_HEAD,
        "a matmul over the vocabulary IS a full read"
    );
    assert_eq!(
        embed.activated_bytes(),
        EMBED / 163_840,
        "a gather touches ONE row — 14.3 KB, not 2.35 GB"
    );
    assert!(
        embed.activated_bytes() < head.activated_bytes() / 100_000,
        "the overcount this removes is five orders of magnitude"
    );
}

#[test]
fn a_gather_cannot_be_priced_as_a_full_read() {
    let g = AccessMode::Gather {
        rows_per_token: 1,
        total_rows: 163_840,
    };
    assert!(!g.is_full_read());
    assert!(AccessMode::FullRead.is_full_read());
    assert_eq!(
        g.activated_bytes(2_348_810_240),
        14_336,
        "7168 cols x 2 bytes"
    );
}

#[test]
fn a_gather_of_every_row_equals_a_full_read() {
    let g = AccessMode::Gather {
        rows_per_token: 128,
        total_rows: 128,
    };
    assert_eq!(g.activated_bytes(4096), 4096);
}

#[test]
fn empty_denominators_do_not_divide_by_zero() {
    assert_eq!(
        AccessMode::Gather {
            rows_per_token: 1,
            total_rows: 0
        }
        .activated_bytes(99),
        0
    );
    assert_eq!(
        AccessMode::Routed {
            active: 16,
            total: 0
        }
        .activated_bytes(99),
        0
    );
}

#[test]
fn routed_access_prices_the_active_fraction() {
    // K3's own 16-of-896.
    let r = AccessMode::Routed {
        active: 16,
        total: 896,
    };
    assert_eq!(r.activated_bytes(896_000), 16_000);
    assert!(!r.is_full_read());
}

#[test]
fn a_gather_on_a_huge_tensor_does_not_overflow() {
    // resident * rows would exceed u64 without the u128 widening.
    let g = AccessMode::Gather {
        rows_per_token: 163_839,
        total_rows: 163_840,
    };
    let big = 2_348_810_240u64;
    assert_eq!(g.activated_bytes(big), big * 163_839 / 163_840);
}

// ---- layer topology --------------------------------------------------

#[test]
fn layer_zero_is_dense_and_the_rest_are_moe() {
    let t = LayerTopology::new(93, 1);
    assert_eq!(t.kind(0), LayerKind::Dense, "first_k_dense_replace = 1");
    assert_eq!(t.kind(1), LayerKind::Moe);
    assert_eq!(t.kind(92), LayerKind::Moe);
    assert_eq!(
        t.n_moe(),
        92,
        "shared experts, wrapper and router multiply by THIS"
    );
    assert_eq!(t.n_dense(), 1);
}

#[test]
fn a_topology_cannot_claim_more_dense_layers_than_it_has() {
    let t = LayerTopology::new(4, 99);
    assert_eq!(t.n_dense(), 4);
    assert_eq!(t.n_moe(), 0);
}

// ---- the K3-DENSE-1 decomposition, pinned ----------------------------

#[test]
fn the_measured_k3_decomposition_holds() {
    let c = k3_dense_census();
    let gb = |b: u64| b as f64 / 1e9;

    assert!(
        (gb(c.activated_bytes()) - 111.16).abs() < 0.05,
        "{}",
        gb(c.activated_bytes())
    );
    assert!(
        (gb(c.resident_bytes()) - 113.51).abs() < 0.05,
        "resident includes the whole embed table: {}",
        gb(c.resident_bytes())
    );
    // Resident exceeds activated by exactly the embed table it never reads.
    let diff = c.resident_bytes() - c.activated_bytes();
    assert!((gb(diff) - 2.35).abs() < 0.01, "{}", gb(diff));
}

#[test]
fn kda_attention_dominates_by_a_factor_of_two_and_a_half() {
    let c = k3_dense_census();
    assert_eq!(
        c.families[0].name, "KDA self_attn",
        "the census leads with the largest"
    );
    assert!((c.activated_share("KDA self_attn") - 0.551).abs() < 0.005);
    assert!((c.activated_share("shared_experts") - 0.219).abs() < 0.005);

    let kda = c.family("KDA self_attn").unwrap().activated_bytes();
    let next = c.family("shared_experts").unwrap().activated_bytes();
    assert!(
        kda as f64 / next as f64 > 2.5,
        "KDA must lead the next family by >2.5x, got {:.2}",
        kda as f64 / next as f64
    );
}

#[test]
fn the_latent_moe_wrapper_is_a_third_of_the_routed_traffic_it_serves() {
    // 9.45 GB of always-on BF16 wrapping a 25.83 GB MXFP4 bank. The routed
    // side is NOT "already at its codec floor with nothing left to do".
    let c = k3_dense_census();
    let wrapper = c.family("LatentMoE wrapper").unwrap().activated_bytes() as f64;
    assert!((wrapper / 1e9 - 9.45).abs() < 0.05);
    assert!((wrapper / 25.83e9 - 0.366).abs() < 0.01);
}

#[test]
fn the_census_orders_by_activated_not_resident() {
    // embed_tokens is the joint-largest RESIDENT family and nearly the
    // smallest ACTIVATED one. Ordering by the wrong quantity would put it
    // near the top of a traffic report.
    let c = k3_dense_census();
    let names: Vec<&str> = c.families.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names[0], "KDA self_attn");
    assert_eq!(*names.last().unwrap(), "embed_tokens");
}

#[test]
fn an_unknown_family_has_no_share_and_no_entry() {
    let c = k3_dense_census();
    assert!(
        c.family("vision_tower").is_none(),
        "vision is excluded, not zero"
    );
    assert_eq!(c.activated_share("vision_tower"), 0.0);
}

#[test]
fn an_empty_census_has_no_shares() {
    let c = DenseCensus::new(vec![]);
    assert_eq!(c.activated_bytes(), 0);
    assert_eq!(c.resident_bytes(), 0);
    assert_eq!(c.activated_share("anything"), 0.0);
}

#[test]
fn a_census_round_trips_through_serde() {
    let c = k3_dense_census();
    let back: DenseCensus = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(c, back);
}
