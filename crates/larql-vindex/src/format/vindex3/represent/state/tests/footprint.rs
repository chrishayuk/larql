//! **4b-e: what the bound surface costs, and what it deliberately does
//! not count.**
//!
//! The sharp one is the container/search-problem boundary:
//!
//! ```text
//! container stores A B C D,  surface is A B C
//!   container gains stored tensor E   → the footprint is UNCHANGED
//!   surface grows to include E        → the old binding no longer
//!                                       applies; rebind; E is counted
//! ```
//!
//! Storage changing and the search problem changing are different
//! events, and only the second may move a price.

use super::super::super::compiler::{read_source_identity, SourceIdentity};
use super::super::super::map::PrecisionMap;
use super::super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::super::policy::Role;
use super::super::accounting::{read_source_storage, PhysicalAccountingFacts, TensorIdentity};
use super::super::candidate::Footprint;
use super::super::footprint::{CompiledBytes, FootprintError, PackCompiledBytes, SurfaceFootprint};
use super::super::identity::RepresentationState;
use super::super::realization::LogicalBytes;
use super::super::resolved::{
    LayoutAdmission, NoLayoutConstraint, PackLayoutAdmission, ResolvedEncoding,
};
use super::super::surface::{SurfaceTensor, TensorSurface};
use super::container;

fn read(path: &std::path::Path) -> (SourceIdentity, PhysicalAccountingFacts) {
    let identity = read_source_identity(path).expect("identity");
    let facts = read_source_storage(path, &identity).expect("facts");
    (identity, facts)
}

/// A surface over exactly what the container stores, with an
/// NVFP4-admissible shape so a compiled price exists.
fn surface_of(facts: &PhysicalAccountingFacts) -> TensorSurface {
    TensorSurface::new(facts.tensors().map(|(id, _)| {
        SurfaceTensor::new(&id.object, &id.tensor, Role::DecoderLinear, vec![64, 64])
    }))
    .expect("one entry per stored tensor")
}

/// **What a `[64, 64]` NVFP4 pack occupies, derived from the FORMAT and
/// not from the code under test.**
///
/// ```text
/// groups per row = k / 16                  = 4
/// E2M1 codes     = rows × groups × 8 bytes = 64 × 4 × 8 = 2048
/// E4M3 scales    = rows × groups           = 64 × 4     =  256
/// tensor scale   = one f32                              =    4
///                                                    total 2308
/// ```
///
/// Written out because a test that asks the oracle what the oracle
/// says would pass over any arithmetic at all: replacing the pack
/// layout with `numel × 2` survived the first version of this file.
/// Cf. `feedback_self_consistency_needs_a_foreign_reference`.
const NVFP4_64X64: u64 = 2048 + 256 + 4;

fn nvfp4_map() -> PrecisionMap {
    PrecisionMap {
        name: "compile-everything".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![],
    }
}

fn source_map() -> PrecisionMap {
    PrecisionMap {
        name: "compile-nothing".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec![],
        exceptions: vec![],
    }
}

fn priced(
    facts: &PhysicalAccountingFacts,
    model: &SourceIdentity,
    surface: &TensorSurface,
    layout: &dyn LayoutAdmission,
) -> SurfaceFootprint {
    let bound = facts.bind(model, surface).expect("READY");
    SurfaceFootprint::new(
        &bound,
        surface,
        layout,
        &PackCompiledBytes,
        &[DTYPE_NVFP4.to_string()],
    )
    .expect("every admitted encoding is priced")
}

fn state(
    model: &SourceIdentity,
    surface: &TensorSurface,
    map: &PrecisionMap,
    layout: &dyn LayoutAdmission,
) -> RepresentationState {
    RepresentationState::resolve(model, surface, map, layout)
}

// ------------------------------------------------- source is the sealed len

#[test]
fn a_source_state_costs_exactly_what_the_container_stores() {
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);

    let expected: u64 = facts.tensors().map(|(_, f)| f.logical_bytes.get()).sum();
    let state = state(&model, &surface, &source_map(), &PackLayoutAdmission);
    assert_eq!(state.decisions().compiled(), 0, "nothing is compiled");
    assert_eq!(
        footprint.logical_bytes(&state),
        LogicalBytes::new(expected),
        "the sum of the sealed lengths, and nothing computed from a dtype"
    );
    assert!(expected > 0);
}

#[test]
fn a_compiled_state_costs_the_pack_layouts_stored_length() {
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);

    let compiled = state(&model, &surface, &nvfp4_map(), &PackLayoutAdmission);
    assert_eq!(
        compiled.decisions().compiled(),
        surface.len(),
        "every tensor compiles at [64, 64]"
    );
    assert_eq!(
        footprint.logical_bytes(&compiled),
        LogicalBytes::new(NVFP4_64X64 * surface.len() as u64)
    );

    let source = state(&model, &surface, &source_map(), &PackLayoutAdmission);
    assert!(
        footprint.logical_bytes(&compiled) < footprint.logical_bytes(&source),
        "NVFP4 is smaller than the stored source, which is the whole point"
    );
}

#[test]
fn a_layout_refusal_is_priced_as_source_and_not_as_the_encoding_it_wanted() {
    // The state a footprint blind to `effective()` would misprice: the
    // map says compile, the layout refuses, the container carries the
    // source bytes — and claiming the compiled price here would book a
    // saving that was never made.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let stored = facts.tensors().next().expect("a tensor");
    let surface = TensorSurface::new([SurfaceTensor::new(
        &stored.0.object,
        &stored.0.tensor,
        Role::DecoderLinear,
        vec![64, 24],
    )])
    .expect("one tensor");

    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);
    let refused = state(&model, &surface, &nvfp4_map(), &PackLayoutAdmission);
    assert!(
        matches!(
            refused.decisions().decisions()[0].encoding,
            ResolvedEncoding::LayoutRefused { .. }
        ),
        "k = 24 is not a whole number of NVFP4 groups"
    );
    assert_eq!(
        footprint.logical_bytes(&refused),
        LogicalBytes::new(stored.1.logical_bytes.get()),
        "the source bytes the container actually carries"
    );

    // And the price table holds no NVFP4 entry for it, because the
    // layout refused it — the two agree by construction.
    let protected = state(&model, &surface, &source_map(), &PackLayoutAdmission);
    assert_eq!(
        footprint.logical_bytes(&refused),
        footprint.logical_bytes(&protected),
        "one physical state, one price"
    );
}

// ------------------------------- container storage vs the search problem

#[test]
fn a_tensor_the_container_gains_does_not_move_the_surfaces_footprint() {
    // **The boundary, permanently.** The container grows a stored
    // tensor the surface does not name. The search problem did not
    // change, so the price must not.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let full = surface_of(&facts);

    // A B C out of A B C D…: a strict subset is the "surface" and the
    // rest is the container's business.
    let mut entries = full.entries().to_vec();
    let extra = entries.pop().expect("more than one");
    let surface = TensorSurface::new(entries).expect("distinct tensors");

    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);
    let resolved = state(&model, &surface, &source_map(), &PackLayoutAdmission);
    let priced_bytes = footprint.logical_bytes(&resolved);

    let whole: u64 = facts.tensors().map(|(_, f)| f.logical_bytes.get()).sum();
    assert!(
        priced_bytes.get() < whole,
        "the container stores more than the surface names, which is the premise"
    );

    // Now the surface grows to include it. The old binding no longer
    // applies — the state is resolved against a different surface — and
    // the new footprint counts it.
    let grown = full;
    assert!(
        footprint
            .try_logical_bytes(&state(&model, &grown, &source_map(), &PackLayoutAdmission))
            .is_err(),
        "a state over the grown surface is not this footprint's to price"
    );
    let regrown = priced(&facts, &model, &grown, &PackLayoutAdmission);
    let grown_bytes =
        regrown.logical_bytes(&state(&model, &grown, &source_map(), &PackLayoutAdmission));
    let extra_bytes = facts
        .get(&TensorIdentity::new(&extra.object, &extra.tensor))
        .expect("the container stores it")
        .logical_bytes;
    assert_eq!(
        grown_bytes.get(),
        priced_bytes.get() + extra_bytes.get(),
        "and it is counted exactly once, at its sealed length"
    );
}

// ----------------------------------------------- one state, one price

#[test]
fn two_realizations_of_one_state_cost_the_same() {
    // A cross-stage invariant. `Source` and `LayoutRefused` are two
    // FACTS and one physical state (1a), so a footprint that read the
    // fact rather than what is presented would give one state two
    // prices and make the graph's `physical_delta` meaningless.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let stored = facts.tensors().next().expect("a tensor");
    let surface = TensorSurface::new([SurfaceTensor::new(
        &stored.0.object,
        &stored.0.tensor,
        Role::DecoderLinear,
        vec![64, 24],
    )])
    .expect("one tensor");
    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);

    let refused = state(&model, &surface, &nvfp4_map(), &PackLayoutAdmission);
    let protected = state(&model, &surface, &source_map(), &PackLayoutAdmission);
    assert_eq!(refused.id(), protected.id(), "one physical state");
    assert_ne!(
        refused.decisions().decisions()[0].encoding,
        protected.decisions().decisions()[0].encoding,
        "two realizations"
    );
    assert_eq!(
        footprint.logical_bytes(&refused),
        footprint.logical_bytes(&protected)
    );
}

// --------------------------------------------- misses made impossible

#[test]
fn an_admitted_encoding_nothing_prices_is_refused_before_the_search_starts() {
    // `PackLayoutAdmission` declares nothing about Q6_K, so it admits
    // it, and `PackCompiledBytes` cannot price it. A state selecting
    // Q6_K would have no footprint at all — reported here, naming the
    // tensor and the encoding, rather than at the first candidate.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let bound = facts.bind(&model, &surface).expect("READY");

    let err = SurfaceFootprint::new(
        &bound,
        &surface,
        &PackLayoutAdmission,
        &PackCompiledBytes,
        &["Q6_K".to_string()],
    )
    .expect_err("nothing prices a Q6_K pack in this build");
    match &err {
        FootprintError::Unpriceable { encoding, .. } => assert_eq!(encoding, "Q6_K"),
        other => panic!("expected an unpriceable encoding, got {other}"),
    }
    assert!(
        err.to_string().contains("would have no footprint at all"),
        "{err}"
    );
}

#[test]
fn an_encoding_the_layout_refuses_needs_no_price() {
    // The congruence that makes the table total: a tensor the layout
    // refuses presents source bytes, so no compiled price is required
    // and its absence is not a gap. Under an oracle that refuses
    // NOTHING, the same pair becomes required — and unpriceable.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let stored = facts.tensors().next().expect("a tensor");
    let surface = TensorSurface::new([SurfaceTensor::new(
        &stored.0.object,
        &stored.0.tensor,
        Role::DecoderLinear,
        vec![64, 24],
    )])
    .expect("one tensor");
    let bound = facts.bind(&model, &surface).expect("READY");
    let encodings = [DTYPE_NVFP4.to_string()];

    SurfaceFootprint::new(
        &bound,
        &surface,
        &PackLayoutAdmission,
        &PackCompiledBytes,
        &encodings,
    )
    .expect("the layout refuses k = 24, so no NVFP4 price is needed");

    assert!(
        SurfaceFootprint::new(
            &bound,
            &surface,
            &NoLayoutConstraint,
            &PackCompiledBytes,
            &encodings,
        )
        .is_err(),
        "an oracle that admits it makes the missing price a real gap"
    );
}

#[test]
fn a_binding_for_another_surface_prices_nothing() {
    // The bind proved completeness for ONE population. Building a price
    // table for a different one would price tensors nothing had checked
    // were stored, so `new` refuses before any table exists.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let bound = facts.bind(&model, &surface).expect("READY");

    let mut entries = surface.entries().to_vec();
    entries.pop();
    let other = TensorSurface::new(entries).expect("distinct tensors");

    match SurfaceFootprint::new(
        &bound,
        &other,
        &PackLayoutAdmission,
        &PackCompiledBytes,
        &[DTYPE_NVFP4.to_string()],
    ) {
        Err(FootprintError::ForeignSurface { priced: p, asked }) => {
            assert_eq!(p, surface.identity());
            assert_eq!(asked, other.identity());
        }
        other => panic!("expected a foreign surface, got {other:?}"),
    }
}

#[test]
fn a_state_from_another_surface_is_reported_and_not_answered() {
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);

    let mut entries = surface.entries().to_vec();
    entries.pop();
    let other = TensorSurface::new(entries).expect("distinct tensors");
    let foreign = state(&model, &other, &source_map(), &PackLayoutAdmission);

    match footprint.try_logical_bytes(&foreign) {
        Err(FootprintError::ForeignSurface { priced: p, asked }) => {
            assert_eq!(p, surface.identity());
            assert_eq!(asked, other.identity());
        }
        other => panic!("expected a foreign surface, got {other:?}"),
    }
    assert_eq!(footprint.surface_identity(), surface.identity());
}

#[test]
#[should_panic(expected = "was asked about a state resolved against")]
fn the_trait_method_panics_on_the_one_thing_it_cannot_report() {
    // `Footprint` returns bytes with no channel for a miss, and giving
    // it one would break stage 2's census. So the contract is checkable
    // in advance and violating it is loud rather than silently priced.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let footprint = priced(&facts, &model, &surface, &PackLayoutAdmission);

    let mut entries = surface.entries().to_vec();
    entries.pop();
    let other = TensorSurface::new(entries).expect("distinct tensors");
    footprint.logical_bytes(&state(&model, &other, &source_map(), &PackLayoutAdmission));
}

#[test]
fn nothing_but_nvfp4_is_priced_by_this_build() {
    assert_eq!(
        PackCompiledBytes.compiled_bytes(DTYPE_NVFP4, &[64, 64]),
        Some(LogicalBytes::new(NVFP4_64X64)),
        "the pack's own stored length, not a width multiplied by a count"
    );
    assert!(PackCompiledBytes
        .compiled_bytes("Q6_K", &[64, 64])
        .is_none());
    assert!(
        PackCompiledBytes
            .compiled_bytes(DTYPE_NVFP4, &[64, 24])
            .is_none(),
        "a shape the pack refuses has no price, which is not the same as zero"
    );
}
