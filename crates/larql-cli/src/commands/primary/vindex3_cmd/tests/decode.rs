//! The greedy loop and the preparation seam: ids come out through the
//! sink, the sink can halt the loop, and a container is prepared and
//! dispatched by the one authority both verbs share.

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;

use super::super::decode::{greedy_decode, Flow};
use larql_inference::vindex3::OpenedComponent;

use super::super::prepare::{
    parse_representation_source, prepare, wanted_representation, with_plan_backend, BackendVisitor,
    DEFAULT_COMPONENT,
};
use super::super::ExecBackend;

const PROMPT: [u32; 3] = [1, 2, 3];
const NEW_TOKENS: usize = 4;

/// The dense fixture, encoded and prepared for the reference arm.
fn prepared_dense(root: &Path) -> OpenedComponent {
    let checkpoint = root.join("checkpoint");
    let container = root.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    prepare(
        &container,
        DEFAULT_COMPONENT,
        ExecBackend::Reference,
        RepresentationSource::Auto,
    )
    .unwrap()
}

#[test]
fn the_sink_sees_every_id_the_loop_returns() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
    let mut seen = Vec::new();
    let decoded = greedy_decode(&mut session, &PROMPT, NEW_TOKENS, &mut |id, _| {
        seen.push(id);
        Ok(Flow::Continue)
    })
    .unwrap();
    assert_eq!(decoded.generated, seen);
    assert_eq!(decoded.generated.len(), NEW_TOKENS);
    // The first generated token comes out of the prompt phase; every
    // later one is a timed decode step.
    assert_eq!(decoded.step_seconds.len(), NEW_TOKENS - 1);
    assert_eq!(session.position(), PROMPT.len() + NEW_TOKENS - 1);
}

#[test]
fn a_halting_sink_ends_the_decode_before_the_next_step() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
    let mut seen = 0usize;
    let decoded = greedy_decode(&mut session, &PROMPT, NEW_TOKENS, &mut |_, _| {
        seen += 1;
        Ok(if seen == 2 {
            Flow::Halt
        } else {
            Flow::Continue
        })
    })
    .unwrap();
    // The halting id is still reported — the sink saw it — and no step
    // ran after it.
    assert_eq!(decoded.generated.len(), 2);
    assert_eq!(decoded.step_seconds.len(), 1);
    assert_eq!(session.position(), PROMPT.len() + 1);
}

#[test]
fn zero_new_tokens_ingests_the_prompt_and_generates_nothing() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
    let decoded = greedy_decode(&mut session, &PROMPT, 0, &mut |_, _| {
        panic!("nothing should reach the sink")
    })
    .unwrap();
    assert!(decoded.generated.is_empty());
    assert!(decoded.step_seconds.is_empty());
    assert!(decoded.priced_step.is_none());
    assert_eq!(session.position(), PROMPT.len());
}

#[test]
fn an_empty_prompt_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
    let err = greedy_decode(&mut session, &[], NEW_TOKENS, &mut |_, _| {
        Ok(Flow::Continue)
    })
    .err()
    .expect("an empty prompt has nothing to condition on");
    assert!(err.to_string().contains("no tokens"), "{err}");
    assert_eq!(session.position(), 0);
}

#[test]
fn a_sink_error_aborts_the_decode() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
    let err = greedy_decode(&mut session, &PROMPT, NEW_TOKENS, &mut |_, _| {
        Err("the stream closed".into())
    })
    .err()
    .expect("a sink failure is the decode's failure");
    assert_eq!(err.to_string(), "the stream closed");
}

#[test]
fn the_loop_is_deterministic_across_fresh_sessions() {
    let root = tempfile::tempdir().unwrap();
    let prepared = prepared_dense(root.path());
    let backend = ReferenceBackend::new();
    let run = || {
        let mut session = DecodeSession::new(&prepared.plan, &prepared.store, &backend).unwrap();
        greedy_decode(&mut session, &PROMPT, NEW_TOKENS, &mut |_, _| {
            Ok(Flow::Continue)
        })
        .unwrap()
        .generated
    };
    assert_eq!(run(), run());
}

/// A visitor that reports which realisation it was handed.
struct NameOf;

impl BackendVisitor for NameOf {
    type Out = String;

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<String, Box<dyn std::error::Error>> {
        Ok(backend.name().to_string())
    }
}

#[test]
fn the_dispatch_hands_the_visitor_the_named_realisation() {
    assert_eq!(
        with_plan_backend(ExecBackend::Reference, NameOf).unwrap(),
        "reference-f32"
    );
    assert_eq!(
        with_plan_backend(ExecBackend::Production, NameOf).unwrap(),
        "production-larql-compute"
    );
    // Same kernels as `production`; the representation is chosen upstream.
    assert_eq!(
        with_plan_backend(ExecBackend::ProductionNvfp4, NameOf).unwrap(),
        "production-larql-compute"
    );
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
#[test]
fn a_lowered_arm_is_refused_by_the_interpreter_dispatch() {
    let err = with_plan_backend(ExecBackend::MetalLowered, NameOf)
        .expect_err("a lowered arm does not run through the interpreter");
    assert!(err.to_string().contains("lowered"), "{err}");
}

#[test]
fn the_canonical_arms_want_no_pack_and_the_nvfp4_arm_wants_its_encoding() {
    assert_eq!(wanted_representation(ExecBackend::Reference), None);
    assert_eq!(wanted_representation(ExecBackend::Production), None);
    assert_eq!(
        wanted_representation(ExecBackend::ProductionNvfp4),
        Some(DTYPE_NVFP4)
    );
}

#[test]
fn the_source_policy_names_three_values_and_refuses_the_rest() {
    assert!(matches!(
        parse_representation_source("auto").unwrap(),
        RepresentationSource::Auto
    ));
    assert!(matches!(
        parse_representation_source("stored").unwrap(),
        RepresentationSource::Stored
    ));
    assert!(matches!(
        parse_representation_source("transient").unwrap(),
        RepresentationSource::Transient
    ));
    let err = parse_representation_source("cached").err().unwrap();
    assert!(err.to_string().contains("cached"), "{err}");
}

#[test]
fn a_directory_that_is_not_a_container_cannot_be_prepared() {
    let root = tempfile::tempdir().unwrap();
    assert!(prepare(
        root.path(),
        DEFAULT_COMPONENT,
        ExecBackend::Reference,
        RepresentationSource::Auto,
    )
    .is_err());
}

/// The one-opening-authority invariant, pinned at the source: the CLI's
/// preparation names no inspection, no plan construction and no operand
/// store opening of its own. Everything a container *is* when it runs
/// comes from `larql_inference::vindex3::open_component`.
#[test]
fn the_cli_preparation_opens_nothing_itself() {
    let source = include_str!("../prepare.rs");
    for forbidden in [
        "inspect_container(",
        "plan_component_ops(",
        "OperandStore::open",
    ] {
        assert!(
            !source.contains(forbidden),
            "prepare.rs re-implements the opener: found `{forbidden}`"
        );
    }
    assert!(source.contains("open_component("));
}

/// The CLI's policy reaches the store: what `--representation-source`
/// asks for is what the opened store was bound under, and the opener's
/// declared identity comes back with it.
#[test]
fn the_prepared_store_carries_the_requested_source_policy() {
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint");
    let container = root.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    let opened = prepare(
        &container,
        DEFAULT_COMPONENT,
        ExecBackend::ProductionNvfp4,
        RepresentationSource::Transient,
    )
    .unwrap();
    assert_eq!(
        opened.store.representation_source(),
        RepresentationSource::Transient
    );
    assert_eq!(opened.want.as_deref(), Some(DTYPE_NVFP4));
    assert_eq!(opened.model_name, "dense");
}
