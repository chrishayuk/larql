//! **The identity contract, stated adversarially.**
//!
//! Each test below names one way the digest could be wrong. Two failure
//! directions, and they are not symmetric:
//!
//! ```text
//! SPLIT   one state looks like two   →  the search re-measures what it
//!                                       has already refused, and every
//!                                       alternative route to a map
//!                                       reads as novel
//! MERGE   two states look like one   →  evidence from one is credited
//!                                       to the other
//! ```
//!
//! Splitting is the expensive one and merging is the dangerous one, so
//! both get tests, and the ones that assert *sameness* are as load-
//! bearing as the ones that assert difference.

mod source_seal;

use std::collections::BTreeMap;

use super::super::compiler::SourceIdentity;
use super::super::map::{Exception, PrecisionMap};
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::*;

// ---------------------------------------------------------------- fixtures

fn model(graph: &str) -> SourceIdentity {
    SourceIdentity {
        manifest_hash: "manifest-aaaa".into(),
        graph_hash: graph.into(),
        segments: BTreeMap::from([
            ("target.decoder_stack".to_string(), "seg-dddd".to_string()),
            ("target.embedding".to_string(), "seg-eeee".to_string()),
        ]),
    }
}

/// A small decoder surface: q/k/v at two depths, all NVFP4-admissible.
fn decoder_surface() -> TensorSurface {
    let mut entries = Vec::new();
    for layer in [0u32, 1] {
        for proj in ["q_proj", "k_proj", "v_proj"] {
            entries.push(SurfaceTensor::new(
                "target.decoder_stack",
                format!("{layer}.self_attn.{proj}.weight"),
                Role::DecoderLinear,
                vec![64, 64],
            ));
        }
    }
    TensorSurface::new(entries).expect("distinct tensors")
}

/// A map that compiles decoder-linear to NVFP4, with the given
/// exceptions, under whatever `name` — the name is a label and the tests
/// vary it on purpose.
fn map(name: &str, exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: name.into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions,
    }
}

fn protect(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: None,
    }
}

fn compile(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: Some(DTYPE_NVFP4.into()),
    }
}

fn state(m: &PrecisionMap, surface: &TensorSurface) -> RepresentationState {
    RepresentationState::resolve(&model("graph-1111"), surface, m, &PackLayoutAdmission)
}

// ------------------------------------------------------------- determinism

#[test]
fn the_same_map_on_the_same_model_is_the_same_state() {
    let s = decoder_surface();
    let m = map("r1", vec![protect("v_proj")]);
    assert_eq!(state(&m, &s).id(), state(&m, &s).id());
}

#[test]
fn the_order_a_surface_was_enumerated_in_does_not_reach_identity() {
    // A container is walked object by object and segment by segment;
    // which tensor arrives first is incidental file order. If it reached
    // the digest, re-reading the same container on another machine could
    // manufacture a second state for one model.
    let forward = decoder_surface();
    let reversed = {
        let mut e = forward.entries().to_vec();
        e.reverse();
        TensorSurface::new(e).expect("distinct tensors")
    };
    assert_eq!(forward.identity(), reversed.identity());

    let m = map("r1", vec![]);
    assert_eq!(state(&m, &forward).id(), state(&m, &reversed).id());
}

// ------------------------------------------------- syntax must not split

#[test]
fn different_recipes_with_identical_resolved_decisions_are_one_state() {
    // Written differently, decides identically. `q_proj -> NVFP4` is
    // exactly what the default already does, so the exception is a
    // no-op — and a search that treated this as an unexplored state
    // would spend an authority run to learn nothing.
    let s = decoder_surface();
    let plain = map("plain", vec![]);
    let redundant = map("also-plain", vec![compile("q_proj")]);

    assert_eq!(
        state(&plain, &s).decisions(),
        state(&redundant, &s).decisions(),
        "the two maps must resolve identically for this test to mean anything"
    );
    assert_eq!(state(&plain, &s).id(), state(&redundant, &s).id());
}

#[test]
fn the_maps_name_is_a_label_and_does_not_reach_identity() {
    let s = decoder_surface();
    assert_eq!(
        state(&map("r1-protect-v", vec![protect("v_proj")]), &s).id(),
        state(&map("candidate-47", vec![protect("v_proj")]), &s).id()
    );
}

#[test]
fn a_shadowed_exception_does_not_move_identity() {
    // The first match decides, so a second rule for the same projection
    // can never fire. It changes the text and nothing else.
    let s = decoder_surface();
    let one = map("one", vec![protect("v_proj")]);
    let shadowed = map("shadowed", vec![protect("v_proj"), compile("v_proj")]);

    assert_eq!(state(&one, &s).id(), state(&shadowed, &s).id());
}

// --------------------------------------------- decisions must not merge

#[test]
fn exception_order_that_changes_a_decision_moves_identity() {
    // The same two rules, swapped. Now `v_proj -> NVFP4` fires first and
    // the protection is the dead rule, so this is a different physical
    // representation and must not inherit the other's evidence.
    let s = decoder_surface();
    let protected = map("protected", vec![protect("v_proj"), compile("v_proj")]);
    let compiled = map("compiled", vec![compile("v_proj"), protect("v_proj")]);

    assert_ne!(
        protected.describe(),
        compiled.describe(),
        "ordering is the only difference"
    );
    assert_ne!(state(&protected, &s).id(), state(&compiled, &s).id());
}

#[test]
fn the_same_map_on_a_different_model_is_a_different_state() {
    let s = decoder_surface();
    let m = map("r1", vec![protect("v_proj")]);
    let layout = &PackLayoutAdmission;

    let a = RepresentationState::resolve(&model("graph-1111"), &s, &m, layout);
    let b = RepresentationState::resolve(&model("graph-2222"), &s, &m, layout);
    assert_eq!(a.decisions(), b.decisions(), "identical decisions");
    assert_ne!(
        a.id(),
        b.id(),
        "identical payloads under a different semantic graph are still a different model"
    );

    // And a changed payload segment, with index and graph unchanged.
    let mut moved = model("graph-1111");
    moved
        .segments
        .insert("target.decoder_stack".into(), "seg-ffff".into());
    assert_ne!(
        a.id(),
        RepresentationState::resolve(&moved, &s, &m, layout).id()
    );
}

// ----------------------------------------------------- surface movement

#[test]
fn a_reshaped_tensor_moves_identity_though_every_decision_is_unchanged() {
    // The sharpest form of the surface test: no decision changes at all.
    // `[128, 64]` is as NVFP4-admissible as `[64, 64]`, so the resolved
    // vector is byte-identical — and the model is not the same model.
    let base = decoder_surface();
    let widened = {
        let mut e = base.entries().to_vec();
        e[0].shape = vec![128, 64];
        TensorSurface::new(e).expect("distinct tensors")
    };
    let m = map("r1", vec![]);

    assert_eq!(
        state(&m, &base).decisions().canonical(),
        state(&m, &widened).decisions().canonical(),
        "no decision moved"
    );
    assert_ne!(state(&m, &base).id(), state(&m, &widened).id());
}

#[test]
fn adding_a_tensor_moves_identity_though_surviving_decisions_are_unchanged() {
    let base = decoder_surface();
    let grown = {
        let mut e = base.entries().to_vec();
        e.push(SurfaceTensor::new(
            "target.decoder_stack",
            "2.self_attn.q_proj.weight",
            Role::DecoderLinear,
            vec![64, 64],
        ));
        TensorSurface::new(e).expect("distinct tensors")
    };
    let m = map("r1", vec![]);

    for d in state(&m, &base).decisions().decisions() {
        assert_eq!(
            state(&m, &grown).decisions().get(&d.object, &d.tensor),
            Some(&d.encoding),
            "every surviving tensor decides the same"
        );
    }
    assert_ne!(state(&m, &base).id(), state(&m, &grown).id());
}

#[test]
fn a_role_reclassification_moves_identity() {
    // Roles come from the plan where the plan binds an operand and from
    // name classification otherwise, and that answer has changed before:
    // 4.05 B Gated DeltaNet weights classified `unknown` because
    // `_proj_qkv.` is not `_proj.`. No byte on disk moves, but the set
    // of maps that would compile the tensor does, so it is a different
    // search problem and must be a different state.
    //
    // The map here compiles nothing, so both classifications resolve to
    // source precision and the decision vector cannot be what separates
    // them.
    let base = decoder_surface();
    let reclassified = {
        let mut e = base.entries().to_vec();
        e[0].role = Role::Unknown;
        TensorSurface::new(e).expect("distinct tensors")
    };
    let compiles_nothing = PrecisionMap {
        name: "source-only".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec![],
        exceptions: vec![],
    };

    assert_eq!(
        state(&compiles_nothing, &base).decisions().canonical(),
        state(&compiles_nothing, &reclassified)
            .decisions()
            .canonical(),
        "no decision moved"
    );
    assert_ne!(
        state(&compiles_nothing, &base).id(),
        state(&compiles_nothing, &reclassified).id()
    );
}

// ------------------------------------------------------------- aliasing

#[test]
fn two_objects_sharing_bytes_are_two_surface_entries() {
    // A tied embedding and output head are one payload and two objects.
    // REPRESENT compiles one pack per OBJECT and a map resolves per
    // `(object, tensor)`, so the surface keeps them apart — collapsing
    // them would make identity claim an agreement the compiler does not
    // enforce. That the bytes are shared is already recorded, once, by
    // the model identity's per-segment hashes.
    let surface = TensorSurface::new(vec![
        SurfaceTensor::new("target.embedding", "weight", Role::Embedding, vec![64, 64]),
        SurfaceTensor::new("target.head", "weight", Role::OutputHead, vec![64, 64]),
    ])
    .expect("two objects, one tensor name each");
    assert_eq!(surface.len(), 2, "same tensor name, different objects");

    let embedding_only = PrecisionMap {
        name: "embedding-only".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["embedding".into()],
        exceptions: vec![],
    };
    let decided = state(&embedding_only, &surface);
    assert!(decided
        .decisions()
        .get("target.embedding", "weight")
        .expect("embedding")
        .is_compiled());
    assert!(!decided
        .decisions()
        .get("target.head", "weight")
        .expect("head")
        .is_compiled());
}

#[test]
fn a_surface_cannot_name_one_tensor_twice() {
    // Silently keeping the last would hand back an identity for a model
    // that does not exist.
    let err = TensorSurface::new(vec![
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.q.weight",
            Role::DecoderLinear,
            vec![64, 64],
        ),
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.q.weight",
            Role::DecoderLinear,
            vec![64, 32],
        ),
    ])
    .expect_err("a duplicate pair is a bug in whatever built the surface");
    assert!(format!("{err}").contains("0.q.weight"), "{err}");
}

// -------------------------------------------------- effective, not declared

#[test]
fn a_layout_refusal_presents_source_and_identifies_as_source() {
    // `k = 24` is not a multiple of the NVFP4 16-element group, so the
    // compiler carries the tensor verbatim whatever the map says. A
    // state that believed it had compiled this tensor would price bytes
    // it never saved.
    //
    // So the refused map and the protecting map are ONE state — same
    // bytes presented — while the decision vector still says which fact
    // produced it.
    let surface = TensorSurface::new(vec![SurfaceTensor::new(
        "target.decoder_stack",
        "0.self_attn.q_proj.weight",
        Role::DecoderLinear,
        vec![64, 24],
    )])
    .expect("one tensor");

    let wants_it = map("wants-it", vec![]);
    let protects_it = map("protects-it", vec![protect("q_proj")]);

    let refused = state(&wants_it, &surface);
    let protected = state(&protects_it, &surface);

    assert_eq!(
        refused
            .decisions()
            .get("target.decoder_stack", "0.self_attn.q_proj.weight"),
        Some(&ResolvedEncoding::LayoutRefused {
            encoding: DTYPE_NVFP4.into()
        }),
        "the map admits it; the layout does not"
    );
    assert_eq!(refused.decisions().layout_refused().len(), 1);
    assert_eq!(protected.decisions().layout_refused().len(), 0);
    assert_eq!(refused.decisions().compiled(), 0, "nothing was compiled");

    assert_eq!(
        refused.id(),
        protected.id(),
        "both present source bytes, so both are the same representation state"
    );
}

#[test]
fn a_layout_that_declares_no_constraint_does_not_refuse() {
    // Refusal takes positive evidence from whoever owns the layout. An
    // oracle with no rule for an encoding must not invent one, or a
    // tensor leaves the action space forever on no evidence.
    let surface = TensorSurface::new(vec![SurfaceTensor::new(
        "target.decoder_stack",
        "0.self_attn.q_proj.weight",
        Role::DecoderLinear,
        vec![64, 24],
    )])
    .expect("one tensor");
    let m = map("wants-it", vec![]);

    let decisions = resolve(&m, &surface, &NoLayoutConstraint);
    assert_eq!(decisions.compiled(), 1);
    assert!(decisions.layout_refused().is_empty());

    // And an encoding PackLayoutAdmission holds no rule for is likewise
    // not refused by it — the NVFP4 group rule is about NVFP4.
    let q8 = PrecisionMap {
        name: "q8".into(),
        encoding: "Q8_0".into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![],
    };
    assert_eq!(resolve(&q8, &surface, &PackLayoutAdmission).compiled(), 1);
}

// ------------------------------------------------------------- provenance

#[test]
fn the_canonical_form_is_versioned() {
    // A persisted DAG that outlives a change to the canonical form must
    // be recognisably stale rather than silently colliding.
    assert!(
        STATE_ID_VERSION.starts_with("represent-state-id/"),
        "{STATE_ID_VERSION}"
    );
}

#[test]
fn a_state_carries_the_surface_it_was_resolved_against() {
    // An id names a surface; a reader must be able to check that the
    // surface in hand is that one without re-deriving the digest.
    let s = decoder_surface();
    let m = map("r1", vec![]);
    assert_eq!(state(&m, &s).surface_identity(), s.identity());
}

#[test]
fn an_id_is_short_enough_to_print_and_long_enough_to_be_an_id() {
    let s = decoder_surface();
    let id = state(&map("r1", vec![]), &s).id().clone();
    assert_eq!(id.as_str().len(), 64, "sha256 hex");
    assert_eq!(id.short().len(), 12);
    assert!(id.as_str().starts_with(id.short()));
}

#[test]
fn a_state_answers_what_it_is_without_re_deriving_it() {
    // The accessors a DAG node needs before any policy exists: which
    // model, which tensor, and the printable id an edge or a report
    // names it by.
    let s = decoder_surface();
    let m = map("r1", vec![protect("v_proj")]);
    let resolved = state(&m, &s);

    assert_eq!(resolved.model(), &model("graph-1111"));
    assert_eq!(
        format!("{}", resolved.id()),
        resolved.id().as_str(),
        "an id prints as itself"
    );

    assert!(!s.is_empty());
    assert_eq!(
        s.get("target.decoder_stack", "0.self_attn.v_proj.weight")
            .map(|t| t.role),
        Some(Role::DecoderLinear)
    );
    assert!(
        s.get("target.decoder_stack", "0.self_attn.no_such.weight")
            .is_none(),
        "lookup is by the identity pair, not a prefix match"
    );

    assert_eq!(resolved.decisions().len(), s.len(), "one decision each");
    assert!(!resolved.decisions().is_empty());

    let empty = TensorSurface::new(vec![]).expect("an empty surface is a surface");
    assert!(empty.is_empty());
    assert!(resolve(&m, &empty, &NoLayoutConstraint).is_empty());
}
