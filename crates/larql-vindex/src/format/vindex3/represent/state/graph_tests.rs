//! **1b: what the graph must never lose.**
//!
//! Stage 1a deliberately collapsed a protection and a layout refusal
//! into one physical identity. That is right for evidence and wrong for
//! actions, so the danger this file exists to catch is a *stage-1b
//! information-loss bug*: a node keyed on the physical id quietly
//! overwriting one realization with the other.

use super::super::compiler::{SourceIdentity, SourceSemanticIdentity};
use super::super::map::{Exception, PrecisionMap};
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::*;

// ---------------------------------------------------------------- fixtures

fn model() -> SourceIdentity {
    SourceIdentity::synthetic(
        "manifest-aaaa",
        "graph-1111",
        [("target.decoder_stack".to_string(), "seg-dddd".to_string())],
    )
}

fn tensor(projection: &str, shape: Vec<usize>) -> SurfaceTensor {
    SurfaceTensor::new(
        "target.decoder_stack",
        format!("0.self_attn.{projection}.weight"),
        Role::DecoderLinear,
        shape,
    )
}

/// q/k/v, all NVFP4-admissible.
fn admissible_surface() -> TensorSurface {
    TensorSurface::new(
        ["q_proj", "k_proj", "v_proj"]
            .into_iter()
            .map(|p| tensor(p, vec![64, 64])),
    )
    .expect("distinct tensors")
}

/// q admissible, v refused by the layout — `k = 24` is not a multiple of
/// the NVFP4 16-element group.
fn mixed_surface() -> TensorSurface {
    TensorSurface::new([
        tensor("q_proj", vec![64, 64]),
        tensor("v_proj", vec![64, 24]),
    ])
    .expect("distinct tensors")
}

fn map(roles: &[&str], exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "m".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: roles.iter().map(|r| (*r).to_string()).collect(),
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

/// Resolve a map into a realization priced at `bytes`.
fn st(m: &PrecisionMap, surface: &TensorSurface, bytes: u64) -> ResolvedState {
    ResolvedState::new(
        RepresentationState::resolve(&model(), surface, m, &PackLayoutAdmission),
        LogicalBytes::new(bytes),
    )
}

/// A map compiling nothing — the root of most of these graphs.
fn nothing() -> PrecisionMap {
    map(&[], vec![])
}

fn by(who: &str) -> Provenance {
    Provenance::new(who)
}

fn dag(root: ResolvedState) -> RepresentationStateGraph {
    RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, root)
}

// --------------------------------------------------- 1. two parents, one state

#[test]
fn two_parents_converge_on_one_state_and_both_provenances_survive() {
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    // Two different one-move states, each compiling a different
    // projection, arriving at the same cost.
    let p1 = st(
        &map(
            &["decoder-linear"],
            vec![protect("k_proj"), protect("v_proj")],
        ),
        &s,
        2000,
    );
    let p2 = st(
        &map(
            &["decoder-linear"],
            vec![protect("q_proj"), protect("v_proj")],
        ),
        &s,
        2000,
    );
    assert_ne!(p1.physical_id(), p2.physical_id(), "different maps");

    let p1_id = g
        .apply(root.physical_id(), Action::new("+q"), p1, by("rung4/i1"))
        .expect("root -> p1");
    let p2_id = g
        .apply(root.physical_id(), Action::new("+k"), p2, by("rung4/i1"))
        .expect("root -> p2");

    // One child, reached from both, by different moves.
    let child = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000);
    let from_p1 = g
        .apply(&p1_id, Action::new("+k"), child.clone(), by("rung5/N1"))
        .expect("p1 -> c");
    let from_p2 = g
        .apply(&p2_id, Action::new("+q"), child, by("rung5/N2"))
        .expect("p2 -> c");

    assert_eq!(from_p1, from_p2, "one physical state, two discoveries");
    assert_eq!(g.len(), 4, "root, p1, p2, c");

    let incoming = g.incoming(&from_p1);
    assert_eq!(incoming.len(), 2);
    let discoverers: Vec<&str> = {
        let mut v: Vec<&str> = incoming
            .iter()
            .flat_map(|e| e.provenance())
            .map(|p| p.by.as_str())
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(discoverers, vec!["rung5/N1", "rung5/N2"]);
    assert!(g.is_acyclic());
}

// ------------------------------------- 2. different recipes, one physical state

#[test]
fn different_recipes_with_one_effective_representation_add_no_second_node() {
    // R5-F3: `P - K25 + H` was a map already measured and rejected under
    // another name. A graph keyed on recipes calls that novel.
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    let plain = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000);
    // Same decisions, written with a shadowed rule and a redundant one.
    let ornate = st(
        &map(
            &["decoder-linear"],
            vec![protect("v_proj"), compile("v_proj"), compile("q_proj")],
        ),
        &s,
        1000,
    );
    assert_eq!(plain.physical_id(), ornate.physical_id());
    assert_eq!(plain.realization_id(), ornate.realization_id());

    let a = g
        .apply(root.physical_id(), Action::new("direct"), plain, by("a"))
        .expect("direct");
    let b = g
        .apply(root.physical_id(), Action::new("scenic"), ornate, by("b"))
        .expect("scenic");

    assert_eq!(a, b);
    assert_eq!(g.len(), 2, "root and one child");
    assert_eq!(
        g.node(&a).expect("child").realization_count(),
        1,
        "identical facts are one realization"
    );
    assert_eq!(g.edge_count(), 2, "two moves, one destination");
}

// ------------------- 3. one physical state, two action-relevant realizations

#[test]
fn one_physical_state_keeps_both_realizations() {
    // THE stage-1b hazard. Both maps present `v_proj` at source
    // precision — one because a protection holds it there, one because
    // the NVFP4 layout refuses `k = 24`. Same bytes, same evidence, and
    // NOT the same action space: `unprotect v_proj` is a legal move from
    // the first and no move whatever produces a compiled `v_proj` from
    // the second.
    let s = mixed_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    let protected = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000);
    let refused = st(&map(&["decoder-linear"], vec![]), &s, 1000);

    assert_eq!(
        protected.physical_id(),
        refused.physical_id(),
        "1a collapses them, correctly"
    );
    assert_ne!(
        protected.realization_id(),
        refused.realization_id(),
        "1b must not"
    );

    let id = g
        .apply(
            root.physical_id(),
            Action::new("protect-v"),
            protected.clone(),
            by("a"),
        )
        .expect("protected");
    g.apply(
        root.physical_id(),
        Action::new("compile-all"),
        refused.clone(),
        by("b"),
    )
    .expect("refused");

    assert_eq!(g.len(), 2, "one physical child");
    let node = g.node(&id).expect("child");
    assert_eq!(node.realization_count(), 2, "neither overwrote the other");

    let v = "0.self_attn.v_proj.weight";
    assert_eq!(
        node.realization(protected.realization_id())
            .expect("protected realization")
            .state()
            .decisions()
            .get("target.decoder_stack", v),
        Some(&ResolvedEncoding::Source)
    );
    assert_eq!(
        node.realization(refused.realization_id())
            .expect("refused realization")
            .state()
            .decisions()
            .get("target.decoder_stack", v),
        Some(&ResolvedEncoding::LayoutRefused {
            encoding: DTYPE_NVFP4.into()
        })
    );

    // And the edges say which realization each move actually built.
    let built: Vec<&RealizationId> = g
        .incoming(&id)
        .iter()
        .map(|e| e.child_realization())
        .collect();
    assert_eq!(built.len(), 2);
    assert!(built.contains(&protected.realization_id()));
    assert!(built.contains(&refused.realization_id()));
}

// ------------------------------------------------------ 4. idempotent redisco

#[test]
fn rediscovering_an_edge_does_not_accumulate_duplicate_provenance() {
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());
    let child = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000);

    let apply = |g: &mut RepresentationStateGraph, who: &str| {
        g.apply(
            root.physical_id(),
            Action::new("+qk").adding(["q", "k"]),
            child.clone(),
            by(who),
        )
        .expect("apply")
    };

    let id = apply(&mut g, "rung5/N1");
    apply(&mut g, "rung5/N1");
    apply(&mut g, "rung5/N1");
    assert_eq!(g.edge_count(), 1, "one move is one edge");
    assert_eq!(
        g.incoming(&id)[0].provenance().len(),
        1,
        "a replayed round must not inflate the record"
    );

    // A genuinely different discoverer of the same edge is information.
    apply(&mut g, "agent/session-7");
    assert_eq!(g.edge_count(), 1);
    assert_eq!(g.incoming(&id)[0].provenance().len(), 2);

    // Order within an exchange is not identity: the same move written
    // the other way round is the same edge.
    g.apply(
        root.physical_id(),
        Action::new("+qk").adding(["k", "q"]),
        child.clone(),
        by("rung5/N1"),
    )
    .expect("apply");
    assert_eq!(g.edge_count(), 1, "`+q +k` and `+k +q` are one move");
}

// ------------------------------------------------- 5. the monotone invariant

#[test]
fn a_transition_that_does_not_improve_physically_is_refused() {
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    let worse = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 4000);
    let err = g
        .apply(root.physical_id(), Action::new("worse"), worse, by("a"))
        .expect_err("+1000 bytes");
    assert!(format!("{err}").contains("1000"), "{err}");

    // Zero is refused too, and this is not a corner case: from a
    // layout-refused realization, "unprotect" changes the facts and no
    // bytes at all. Physical dominance already prunes it, and admitting
    // it would put a zero-weight edge in the structure the strict
    // decrease is what keeps acyclic.
    let level = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 3000);
    g.apply(root.physical_id(), Action::new("level"), level, by("a"))
        .expect_err("net zero");

    assert_eq!(g.len(), 1, "neither refusal left a node behind");
    assert_eq!(g.edge_count(), 0);
}

#[test]
fn the_strict_policy_makes_the_graph_acyclic_and_the_loose_one_does_not() {
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let a = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 2000);

    assert!(TransitionPolicy::StrictlyImprovingPhysical.guarantees_acyclic());
    assert!(!TransitionPolicy::Unconstrained.guarantees_acyclic());

    // Unconstrained admits a move back up, and a cycle with it.
    let mut loose = RepresentationStateGraph::new(TransitionPolicy::Unconstrained, root.clone());
    let a_id = loose
        .apply(root.physical_id(), Action::new("down"), a, by("a"))
        .expect("down");
    loose
        .apply(&a_id, Action::new("back up"), root.clone(), by("a"))
        .expect("an unconstrained graph admits it");
    assert!(
        !loose.is_acyclic(),
        "the objective is no longer bytes, so the structure is a graph"
    );

    // The same pair of moves under the strict policy cannot close.
    let mut strict = dag(root.clone());
    let a2 = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 2000);
    let a2_id = strict
        .apply(root.physical_id(), Action::new("down"), a2, by("a"))
        .expect("down");
    strict
        .apply(&a2_id, Action::new("back up"), root, by("a"))
        .expect_err("a cycle needs a non-decreasing edge");
    assert!(strict.is_acyclic());
}

// ------------------------------------------------------------ 6. round-trip

#[test]
fn a_graph_survives_serialization_with_identity_facts_and_edges_intact() {
    let s = mixed_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());
    let protected = st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000);
    let refused = st(&map(&["decoder-linear"], vec![]), &s, 1000);
    let id = g
        .apply(
            root.physical_id(),
            Action::new("protect-v"),
            protected.clone(),
            by("a"),
        )
        .expect("protected");
    g.apply(
        root.physical_id(),
        Action::new("compile-all"),
        refused,
        by("b"),
    )
    .expect("refused");

    let json = serde_json::to_string(&g).expect("serialize");
    let back: RepresentationStateGraph = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back, g, "byte-for-byte the same graph");
    assert_eq!(back.policy(), TransitionPolicy::StrictlyImprovingPhysical);
    assert_eq!(back.root(), root.physical_id());
    assert_eq!(back.model(), &model());
    assert_eq!(back.surface_identity(), s.identity());

    let node = back.node(&id).expect("child survived");
    assert_eq!(node.realization_count(), 2, "both facts survived");
    assert_eq!(node.logical_bytes(), LogicalBytes::new(1000));
    assert_eq!(back.incoming(&id).len(), 2, "both edges survived");
    assert!(node.realization(protected.realization_id()).is_some());
}

// ------------------------------------------------------- refusals that guard

#[test]
fn an_edge_cannot_begin_at_a_state_the_graph_never_held() {
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root);
    let stranger = st(&map(&["decoder-linear"], vec![protect("q_proj")]), &s, 2500);
    let child = st(&map(&["decoder-linear"], vec![]), &s, 1000);

    let err = g
        .apply(stranger.physical_id(), Action::new("x"), child, by("a"))
        .expect_err("unknown parent");
    assert!(format!("{err}").contains("never held"), "{err}");
}

#[test]
fn one_graph_holds_one_model_and_one_surface() {
    // A map's physical prize is a property of the model it resolves
    // against, so a graph that mixed models would hold costs that cannot
    // be compared.
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    let other_model = SourceIdentity {
        semantic: SourceSemanticIdentity {
            graph_hash: "graph-2222".into(),
            ..model().semantic
        },
        ..model()
    };
    let elsewhere = ResolvedState::new(
        RepresentationState::resolve(
            &other_model,
            &s,
            &map(&["decoder-linear"], vec![]),
            &PackLayoutAdmission,
        ),
        LogicalBytes::new(1000),
    );
    let err = g
        .apply(root.physical_id(), Action::new("x"), elsewhere, by("a"))
        .expect_err("different container");
    assert!(format!("{err}").contains("different container"), "{err}");

    // Same model, different enumerated surface.
    let grown = TensorSurface::new(
        s.entries()
            .iter()
            .cloned()
            .chain([tensor("o_proj", vec![64, 64])]),
    )
    .expect("distinct");
    let err = g
        .apply(
            root.physical_id(),
            Action::new("x"),
            st(&map(&["decoder-linear"], vec![]), &grown, 1000),
            by("a"),
        )
        .expect_err("different surface");
    assert!(
        format!("{err}").contains("different search problem"),
        "{err}"
    );
}

#[test]
fn one_physical_state_cannot_be_priced_two_ways() {
    // Two realizations that present the same bytes must agree about how
    // many. A disagreement is a bug in whoever read the ledger, and
    // silently keeping the first would make every downstream delta wrong
    // by the difference.
    let s = mixed_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    g.apply(
        root.physical_id(),
        Action::new("protect-v"),
        st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000),
        by("a"),
    )
    .expect("first");

    let err = g
        .apply(
            root.physical_id(),
            Action::new("compile-all"),
            st(&map(&["decoder-linear"], vec![]), &s, 900),
            by("b"),
        )
        .expect_err("same physical state, different footprint");
    assert!(format!("{err}").contains("already priced"), "{err}");
}

#[test]
fn a_realization_names_both_of_its_identities() {
    let s = mixed_surface();
    let r = st(&map(&["decoder-linear"], vec![]), &s, 1000);
    assert_eq!(r.physical_id(), r.state().id());
    assert_eq!(r.realization_id().as_str().len(), 64);
    assert_eq!(r.realization_id().short().len(), 12);
    assert_eq!(
        format!("{}", r.realization_id()),
        r.realization_id().as_str()
    );
    assert_eq!(r.logical_bytes().get(), 1000);
    assert_eq!(format!("{}", r.logical_bytes()), "1000 B");
    assert_eq!(
        LogicalBytes::new(900).delta_from(LogicalBytes::new(1000)),
        -100
    );
}

#[test]
fn an_edge_reports_the_move_and_what_it_cost() {
    // The record a `MeasurementKey` and an `explain(state)` will read
    // back: what was exchanged, how many bytes it bought, and every note
    // anyone attached — with the note kept as text so that nothing can
    // rank on it.
    let s = admissible_surface();
    let root = st(&nothing(), &s, 3000);
    let mut g = dag(root.clone());

    let exchange = Action::new("−M26 +E24").removing(["M26"]).adding(["E24"]);
    let id = g
        .apply(
            root.physical_id(),
            exchange,
            st(&map(&["decoder-linear"], vec![protect("v_proj")]), &s, 1000),
            by("rung5/N3").noting("candidate U1, ranked first, diagnostic 2.1720e-3"),
        )
        .expect("exchange");

    let edge = g.incoming(&id)[0];
    assert_eq!(edge.parent(), root.physical_id());
    assert_eq!(edge.action().label, "−M26 +E24");
    assert_eq!(edge.action().removed, ["M26"]);
    assert_eq!(edge.action().added, ["E24"]);
    assert_eq!(edge.physical_delta(), -2000);
    assert_eq!(
        edge.provenance()[0].note.as_deref(),
        Some("candidate U1, ranked first, diagnostic 2.1720e-3")
    );
    assert_eq!(g.outgoing(root.physical_id()).len(), 1);

    // And the graph's own inventory answers without a lookup.
    assert!(!g.is_empty());
    assert_eq!(g.nodes().count(), 2);
    assert_eq!(g.edges().count(), 1);
    let node = g.node(&id).expect("child");
    assert_eq!(node.physical_id(), &id);
    assert_eq!(node.realizations().count(), 1);
}
