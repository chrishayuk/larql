//! The stats observer and its basis identity (V3-OBS-1, P5's treatment
//! arm): a seeded basis is reproducible and orthonormal, a basis's hash
//! is a function of its rows alone, projection is linear, the probe is
//! a raw dot product, and one row per write lands on a real plan.

use super::decode::fixture as golden_fixture;
use crate::format::vindex3::fixtures::{G_HIDDEN, G_LAYERS, G_TOKENS};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::observe_stats::{
    FixedBasis, HeadProbe, StatsObserver, NORM_METHOD, PROBE_METHOD, SEEDED_BASIS_PROVIDER,
    SUPPLIED_BASIS_PROVIDER,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

const HIDDEN: usize = 16;
const DIMS: usize = 3;
const SEED: u64 = 42;
/// f32-cast rows of an f64-orthonormal basis: dots land within this.
const ORTHONORMAL_TOLERANCE: f64 = 1e-6;

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

#[test]
fn a_seeded_basis_is_reproducible_and_orthonormal() {
    let a = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    let b = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.rows(), b.rows());
    let identity = a.identity();
    assert_eq!(identity.provider, SEEDED_BASIS_PROVIDER);
    assert_eq!((identity.dims, identity.hidden), (DIMS, HIDDEN));
    assert_eq!(identity.hash_hex.len(), 64, "sha-256 as hex");
    assert_eq!((a.dims(), a.hidden()), (DIMS, HIDDEN));
    for (i, row) in a.rows().iter().enumerate() {
        for (j, other) in a.rows().iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot(row, other) - expected).abs() < ORTHONORMAL_TOLERANCE,
                "rows {i},{j}: {}",
                dot(row, other)
            );
        }
    }
    let other_seed = FixedBasis::seeded(HIDDEN, DIMS, SEED + 1).unwrap();
    assert_ne!(other_seed.identity().hash_hex, identity.hash_hex);
}

#[test]
fn a_seeded_basis_refuses_impossible_dimensions() {
    assert!(FixedBasis::seeded(HIDDEN, 0, SEED).is_err());
    assert!(FixedBasis::seeded(HIDDEN, HIDDEN + 1, SEED).is_err());
    assert!(
        FixedBasis::seeded(HIDDEN, HIDDEN, SEED).is_ok(),
        "a full basis is allowed"
    );
}

#[test]
fn the_hash_is_a_function_of_the_rows_alone() {
    let seeded = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    let supplied = FixedBasis::from_rows(None, "renamed", HIDDEN, seeded.rows().to_vec()).unwrap();
    assert_eq!(supplied.identity().hash_hex, seeded.identity().hash_hex);
    assert_eq!(supplied.identity().provider, SUPPLIED_BASIS_PROVIDER);
    assert_eq!(supplied.identity().id, "renamed");
    let named =
        FixedBasis::from_rows(Some("reader"), "r1", HIDDEN, seeded.rows().to_vec()).unwrap();
    assert_eq!(named.identity().provider, "reader");
    assert_eq!(named.identity().hash_hex, seeded.identity().hash_hex);
    // A different row, a different hash.
    let mut rows = seeded.rows().to_vec();
    rows[0][0] += 1.0;
    let changed = FixedBasis::from_rows(None, "renamed", HIDDEN, rows).unwrap();
    assert_ne!(changed.identity().hash_hex, seeded.identity().hash_hex);
}

#[test]
fn supplied_rows_are_checked_for_shape() {
    assert!(FixedBasis::from_rows(None, "x", 0, vec![vec![]]).is_err());
    assert!(FixedBasis::from_rows(None, "x", HIDDEN, vec![]).is_err());
    assert!(FixedBasis::from_rows(None, "x", HIDDEN, vec![vec![0.0; HIDDEN - 1]]).is_err());
    assert!(FixedBasis::from_rows(None, "x", HIDDEN, vec![vec![0.0; HIDDEN]]).is_ok());
}

#[test]
fn projection_is_linear_and_recovers_the_basis_rows() {
    let basis = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    for (i, row) in basis.rows().iter().enumerate() {
        let coords = basis.project(row);
        for (j, c) in coords.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((f64::from(*c) - expected).abs() < ORTHONORMAL_TOLERANCE);
        }
    }
    let a: Vec<f32> = (0..HIDDEN).map(|i| i as f32 * 0.25).collect();
    let b: Vec<f32> = (0..HIDDEN).map(|i| 1.0 - i as f32 * 0.5).collect();
    let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let pa = basis.project(&a);
    let pb = basis.project(&b);
    let ps = basis.project(&sum);
    for k in 0..DIMS {
        assert!((f64::from(pa[k] + pb[k]) - f64::from(ps[k])).abs() < ORTHONORMAL_TOLERANCE);
    }
}

#[test]
fn the_probe_is_a_raw_dot_product_with_checked_shape() {
    assert!(HeadProbe::new(vec![1, 2], vec![vec![0.0; HIDDEN]], HIDDEN).is_err());
    assert!(HeadProbe::new(vec![1], vec![vec![0.0; HIDDEN + 1]], HIDDEN).is_err());
    let rows = vec![
        (0..HIDDEN).map(|i| i as f32).collect::<Vec<f32>>(),
        vec![1.0; HIDDEN],
    ];
    let probe = HeadProbe::new(vec![7, 9], rows.clone(), HIDDEN).unwrap();
    assert_eq!(probe.tokens(), &[7, 9]);
    let basis = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    let mut observer = StatsObserver::new(basis, Some(probe));
    let after: Vec<f32> = (0..HIDDEN).map(|i| 0.5 - i as f32 * 0.1).collect();
    let delta = vec![0.0; HIDDEN];
    observer.carrier_write(CarrierWriteRecord {
        layer: 3,
        site: SublayerSite::Ffn,
        position: 2,
        delta: &delta,
        after: &after,
        layer_scale: Some(0.5),
    });
    let row = &observer.rows[0];
    assert_eq!(
        (row.layer, row.site, row.position),
        (3, SublayerSite::Ffn, 2)
    );
    assert_eq!(row.layer_scale, Some(0.5));
    assert_eq!(row.delta_norm, 0.0);
    assert!((row.norm - dot(&after, &after).sqrt()).abs() < ORTHONORMAL_TOLERANCE);
    assert_eq!(row.probe.len(), 2);
    // The observer accumulates in f64 in row order and casts once; the
    // same arithmetic here must land on the same f32 bits.
    for (k, expected_row) in rows.iter().enumerate() {
        assert_eq!(
            row.probe[k].to_bits(),
            (dot(expected_row, &after) as f32).to_bits()
        );
    }
    assert_eq!(row.projection.len(), DIMS);
    assert_eq!(observer.probe().unwrap().tokens(), &[7, 9]);
    assert_eq!(observer.basis().dims(), DIMS);
    assert!(!NORM_METHOD.is_empty() && !PROBE_METHOD.is_empty());
}

#[test]
fn one_row_per_write_lands_on_a_real_plan_and_events_are_counted_not_kept() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let basis = FixedBasis::seeded(G_HIDDEN, DIMS, SEED).unwrap();
    let mut observer = StatsObserver::new(basis, None);
    let mut session = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    for &token in G_TOKENS.iter() {
        session.step_observed(token, &mut observer).unwrap();
    }
    let writes_per_step = 2 * G_LAYERS;
    assert_eq!(observer.rows.len(), writes_per_step * G_TOKENS.len());
    // embed + (write, attention, write, ffn) per layer + logits, per step.
    assert_eq!(observer.events, (2 + 4 * G_LAYERS) * G_TOKENS.len());
    for (index, row) in observer.rows.iter().enumerate() {
        assert_eq!(row.position, index / writes_per_step);
        assert_eq!(row.layer, (index % writes_per_step) / 2);
        assert_eq!(
            row.site,
            if index % 2 == 0 {
                SublayerSite::Attention
            } else {
                SublayerSite::Ffn
            }
        );
        assert!(row.norm > 0.0 && row.delta_norm > 0.0);
        assert_eq!(row.projection.len(), DIMS);
        assert!(row.probe.is_empty());
        assert_eq!(row.layer_scale, None);
    }
    let taken = observer.take_rows();
    assert_eq!(taken.len(), writes_per_step * G_TOKENS.len());
    assert!(observer.rows.is_empty());
}

/// The seeded basis is a reproducible artifact: two runs with one
/// (hidden, dims, seed) share a coordinate system by construction, and
/// that only holds across builds if the generator cannot drift. Its
/// content hash for one triple is pinned here, so any change to the
/// generator — even one that keeps the basis orthonormal, such as
/// shifting the draw's offset — is a deliberate contract change with a
/// pin to update, never a silent re-projection of every record.
const PINNED_SEEDED_HASH: &str = "956019731c8ff12d7193d4a35039aa13770d79b0e191a05789f6bc30927fcada";
const PINNED_FIRST_VALUE_BITS: u32 = 1031304468;

#[test]
fn the_seeded_basis_is_pinned_across_builds() {
    let basis = FixedBasis::seeded(HIDDEN, DIMS, SEED).unwrap();
    assert_eq!(basis.identity().hash_hex, PINNED_SEEDED_HASH);
    assert_eq!(basis.rows()[0][0].to_bits(), PINNED_FIRST_VALUE_BITS);
}
