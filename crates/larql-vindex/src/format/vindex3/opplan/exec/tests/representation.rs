//! STATE-1: continuation state has a physical representation, and the
//! materialized form is one of them — the control arm.
//!
//! The gates here are deliberately boring: the round trip is identity,
//! the footprint is the geometry's own arithmetic, the contract is
//! exactness. Their value is what they pin for the first non-trivial
//! representation: it must come back through the SAME `store` /
//! `reconstruct` / `contract` seam these tests freeze, judged against
//! this arm — not as an engine special case beside it.
//!
//! The fixture is heterogeneous and asymmetric on purpose (one KV layer,
//! one two-buffer recurrence with non-square shapes, one stateless
//! layer, a logical position no row count could reproduce) so an
//! implementation that conflates any two of those facts cannot pass by
//! coincidence.

use super::super::continuation::{
    ContinuationState, LayerContinuationGeometry, RecurrentBufferGeometry, RecurrentGeometry,
    StateInitialization,
};
use super::super::kv::LayerKvGeometry;
use super::super::representation::{
    ContinuationPlan, ContinuationRepresentation, CorrectnessContract,
};
use larql_models::inventory::report::RecurrentStateDtype;

/// Awkward on purpose: not a power of two, not equal to any other width
/// in the fixture.
const KV_DIM: usize = 6;
/// Non-square, so a transposed or flattened read cannot round-trip.
const DELTA_SHAPE: [usize; 3] = [3, 5, 2];
const CONV_SHAPE: [usize; 2] = [7, 3];
/// A logical position no row count in the fixture can reproduce: the
/// state below retains 2 KV rows.
const POSITION: usize = 41;

fn fixture_geometry() -> Vec<LayerContinuationGeometry> {
    vec![
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: KV_DIM,
            window: None,
        }),
        LayerContinuationGeometry::Recurrent(RecurrentGeometry {
            buffers: vec![
                RecurrentBufferGeometry {
                    shape: DELTA_SHAPE.to_vec(),
                    dtype: RecurrentStateDtype::Float32,
                    initialization: StateInitialization::Zeros,
                },
                RecurrentBufferGeometry {
                    shape: CONV_SHAPE.to_vec(),
                    dtype: RecurrentStateDtype::Float32,
                    initialization: StateInitialization::Zeros,
                },
            ],
        }),
        LayerContinuationGeometry::Stateless,
    ]
}

/// Deterministic, non-repeating values so a dropped or reordered cell
/// changes the state.
fn awkward_values(count: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..count)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// A canonical state with every kind of content populated.
fn populated_state() -> ContinuationState {
    let geometry = fixture_geometry();
    let mut state = ContinuationState::prepare(&geometry);

    let kv = state.layer_mut(0).kv_mut().expect("layer 0 is KV");
    kv.append(awkward_values(KV_DIM, 1), awkward_values(KV_DIM, 2));
    kv.append(awkward_values(KV_DIM, 3), awkward_values(KV_DIM, 4));

    let recurrent = state
        .layer_mut(1)
        .recurrent_mut()
        .expect("layer 1 is recurrent");
    for (index, seed) in [(0usize, 5u32), (1, 6)] {
        let cells = recurrent.buffer_mut(index).cells_mut();
        cells.copy_from_slice(&awkward_values(cells.len(), seed));
    }

    state.set_position(POSITION);
    state
}

/// The control arm's round trip changes nothing: every KV row, every
/// recurrent cell, and the explicitly-owned position come back
/// bit-for-bit.
#[test]
fn materialized_round_trip_is_bit_identical() {
    let plan = ContinuationPlan::materialized();
    let canonical = populated_state();

    let stored = plan.store(canonical.clone());
    assert_eq!(
        stored.representation(),
        ContinuationRepresentation::Materialized
    );

    let reconstructed = plan.reconstruct(stored).expect("same representation");
    assert_eq!(reconstructed, canonical, "round trip must be identity");
    assert_eq!(
        reconstructed.position(),
        POSITION,
        "position is owned state, and must survive the round trip \
         even though no row count in the fixture could re-derive it"
    );
}

/// An empty state's position survives too: with zero rows retained there
/// is nothing to re-derive a position FROM, which is exactly why it is
/// carried explicitly.
#[test]
fn round_trip_preserves_position_with_nothing_else_retained() {
    let plan = ContinuationPlan::materialized();
    let mut canonical = ContinuationState::prepare(&fixture_geometry());
    canonical.set_position(POSITION);

    let reconstructed = plan
        .reconstruct(plan.store(canonical))
        .expect("same representation");
    assert_eq!(reconstructed.position(), POSITION);
}

/// Holding the canonical state costs exactly what the geometry says must
/// persist: the KV share scales with positions, the recurrent share does
/// not, and the stateless layer contributes nothing.
#[test]
fn materialized_footprint_is_the_geometry_footprint() {
    let plan = ContinuationPlan::materialized();
    let geometry = fixture_geometry();

    let recurrent_elements: usize =
        DELTA_SHAPE.iter().product::<usize>() + CONV_SHAPE.iter().product::<usize>();

    let (short, long) = (128usize, 32_768usize);
    for positions in [short, long] {
        assert_eq!(
            plan.stored_elements_at(&geometry, positions),
            KV_DIM * 2 * positions + recurrent_elements,
            "materialized stores the geometry's answer, nothing more"
        );
    }

    // Only the KV share moved between the two context lengths.
    let grown = plan.stored_elements_at(&geometry, long) - recurrent_elements;
    let base = plan.stored_elements_at(&geometry, short) - recurrent_elements;
    assert_eq!(grown, base * (long / short), "KV share scales exactly");
}

/// The control arm's promise is exactness, and the promise is read from
/// the plan — a caller never inspects a representation's internals to
/// find out what it was guaranteed.
#[test]
fn the_control_arm_promises_exactness() {
    let plan = ContinuationPlan::materialized();
    assert_eq!(
        plan.representation(),
        ContinuationRepresentation::Materialized
    );
    assert_eq!(plan.contract(), CorrectnessContract::Exact);
}
