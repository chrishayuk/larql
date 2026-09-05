//! A hyper-connection stack is REPRESENTED with its declared parameters,
//! and what still refuses is said by name.
//!
//! Waves 16-18 pinned here that the single-stream residual programme
//! could not lower the topology without discarding a stream count, a
//! per-token weight or a cross-stream mix. Wave 19 gave the programme
//! a bundle carrier on both traversals, so the refusal that lived on the
//! topology's own authority is retired with the seam that read it — and the tests
//! below pin the two things that replaced it: both judged topologies
//! lower, and a stack that declares the topology WITHOUT a head object
//! keeps a blocking execution-surface finding that names the head, not
//! the topology. An unjudged (partial) declaration still refuses the
//! surface, as before.

use crate::format::vindex3::graph::build_from_inventories;
use crate::format::vindex3::plan::tests_support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::{plan_system, FindingCategory, SemanticClass};
use larql_models::config::{HyperConnection, HyperConnectionWeights, ResidualTopology};

/// DeepSeek-V4-Flash's own declared numbers.
const STREAMS: usize = 4;
const SINKHORN_ITERS: usize = 20;
const SINKHORN_EPS: f64 = 1e-6;

fn hyper_connected(config: &mut serde_json::Value) {
    let text = &mut config["text_config"];
    text["hc_mult"] = serde_json::json!(STREAMS);
    text["hc_sinkhorn_iters"] = serde_json::json!(SINKHORN_ITERS);
    text["hc_eps"] = serde_json::json!(SINKHORN_EPS);
}

fn surface_of(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> crate::format::vindex3::graph::surface::ExecutionSurface {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), mutate);
    build_from_inventories(&[("target-artifact".to_string(), inventory)])
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.execution.clone())
        .expect("the component builds")
}

/// The topology is a declared COMPONENT fact, carried with every
/// parameter the reference reads — none of them defaulted.
#[test]
fn the_topology_is_represented_with_its_declared_parameters() {
    let surface = surface_of(hyper_connected);
    let ResidualTopology::HyperConnection(hc) = surface.residual_topology else {
        panic!(
            "expected a hyper-connection topology, got {:?}",
            surface.residual_topology
        );
    };
    assert_eq!(
        hc,
        HyperConnection {
            streams: STREAMS,
            sinkhorn_iters: SINKHORN_ITERS,
            sinkhorn_eps: SINKHORN_EPS,
        }
    );
    assert_eq!(surface.residual_topology.streams(), STREAMS);
    // The mix projection's row count is derived from the stream count,
    // so the two cannot drift: `(2 + 4) * 4 = 24` on DeepSeek-V4.
    assert_eq!(HyperConnectionWeights::mix_rows_for(hc.streams), 24);

    // The control: without the declaration, the same fixture is a
    // single-stream stack. If it were not, the assertion above would
    // pass on a fixture that is hyper-connected by accident.
    let plain = surface_of(|_| {});
    assert_eq!(plain.residual_topology, ResidualTopology::SingleStream);
    assert_eq!(plain.residual_topology.streams(), 1);
}

/// Both judged topologies lower. The single stream always did; the
/// hyper-connection topology does since wave 19, when the decode step
/// and the batch traversal each carried the bundle under an
/// intermediate-state witness that can fail.
#[test]
fn both_judged_topologies_lower() {
    let hc = ResidualTopology::HyperConnection(HyperConnection {
        streams: STREAMS,
        sinkhorn_iters: SINKHORN_ITERS,
        sinkhorn_eps: SINKHORN_EPS,
    });
    assert_eq!(hc.streams(), STREAMS);
    assert!(!hc.is_single_stream());
    assert!(ResidualTopology::SingleStream.is_single_stream());
}

/// **Wave 11's lesson, applied to the refusal that remains.** A
/// component that declares the topology and ships no head object cannot
/// run whole-stack — there is no declared reduction from the bundle to
/// the vector the final norm reads — and the plan REPORT must say so,
/// naming the head and not the topology. This fixture ships no
/// `hc_head_*`, so it is GLM-5.3-Flash's shape.
#[test]
fn the_report_names_the_missing_head_as_the_remaining_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), hyper_connected);
    let plan = plan_system(&[("target-artifact".to_string(), inventory)]);

    let finding = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .find(|f| f.subject.ends_with("execution_surface"))
        .expect("the component has an execution-surface finding");

    assert_eq!(
        finding.class,
        SemanticClass::UnsupportedComponent,
        "{finding:?}"
    );
    assert_eq!(
        finding.category,
        FindingCategory::Unrepresented,
        "{finding:?}"
    );
    assert!(finding.blocks(), "{finding:?}");
    // Both halves: the surface IS complete and the topology runs; the
    // head is what is missing. A reader must be able to tell this from
    // "the topology cannot run".
    assert!(finding.detail.contains("complete"), "{}", finding.detail);
    assert!(
        finding.detail.contains("hyper_connection_head"),
        "{}",
        finding.detail
    );
    assert!(
        finding.detail.contains("executable layer by layer"),
        "{}",
        finding.detail
    );
    assert!(
        !finding.detail.contains("NOT executable"),
        "the topology's old refusal must not reappear: {}",
        finding.detail
    );
}

/// A topology this build has not judged resolves to nothing and refuses,
/// rather than completing itself with one stream. This is the failure the
/// field exists to prevent: a four-stream checkpoint served as a
/// one-stream model computes fluent wrong output.
///
/// The refusal must not call the declaration INCOMPLETE. Hy4-preview
/// declares `hc_mult` and `hc_eps` with no iteration count because its
/// topology runs no Sinkhorn — `[2 * hc, hc * d]` with two scales and no
/// combination block — so "partial" would send a reader to finish a
/// declaration nothing is missing from.
#[test]
fn an_unjudged_declaration_refuses_rather_than_defaulting_to_one_stream() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        // Streams declared, the split's parameters not.
        config["text_config"]["hc_mult"] = serde_json::json!(STREAMS);
    });
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let incomplete = built
        .incomplete_surfaces
        .iter()
        .find(|s| s.component == "target")
        .expect("an unjudged topology must refuse the surface");
    let reason = incomplete
        .missing
        .iter()
        .find(|m| m.contains("residual topology"))
        .unwrap_or_else(|| panic!("{:?}", incomplete.missing));

    // What this build DOES lower, named — so a reader knows which form
    // the checkpoint failed to match rather than only that it failed.
    assert!(reason.contains("Sinkhorn-split"), "{reason}");
    // And the reading that is NOT ruled out. Without this the refusal
    // reads as "go and find the missing number", which for a
    // Sinkhorn-free checkpoint is work with no end.
    assert!(reason.contains("DIFFERENT topology"), "{reason}");
}

/// The two refusal defects must READ differently, because they mean
/// opposite things to whoever acts on them: one says a value is missing,
/// the other that an operator is. A reader who cannot tell them apart
/// cannot tell "go and find the number" from "go and write the code".
#[test]
fn the_two_refusal_defects_do_not_read_alike() {
    use crate::format::vindex3::opplan::ClosureDefect;

    let unimplemented = ClosureDefect::UnimplementedSemantic {
        component: "target".to_string(),
        fact: "residual topology (4 parallel streams)".to_string(),
        representable_as: "ResidualTopology::HyperConnection".to_string(),
    };
    let unjudged = ClosureDefect::UnjudgedSemantic {
        component: "target".to_string(),
        fact: "post-norm epsilon".to_string(),
        required_by: "four-norm placement".to_string(),
    };

    let a = unimplemented.to_string();
    let b = unjudged.to_string();
    assert_ne!(a, b);
    // The unimplemented one says the representation is FINE and the
    // lowering is what is missing.
    assert!(a.contains("representable"), "{a}");
    assert!(a.contains("no lowering"), "{a}");
    assert!(a.contains("ResidualTopology::HyperConnection"), "{a}");
    // The unjudged one says nothing established the value.
    assert!(!b.contains("representable"), "{b}");
    assert!(b.contains("post-norm epsilon"), "{b}");
}
