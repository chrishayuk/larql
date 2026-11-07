//! The plan-gated encoder through the real shared ingest, on a local
//! fixture checkpoint (the same spec form, minus the network).

use super::*;
use crate::commands::primary::hf_auto::encode::*;
use crate::commands::primary::hf_auto::refusal::WHOLE_MODEL_LABEL;
use crate::commands::primary::vindex3_cmd::tests::fixture_dir;
use larql_vindex::format::generation::{detect_generation, ContainerGeneration};

fn encode(
    encoder: PlanGatedEncoder,
    judged: bool,
) -> (tempfile::TempDir, PathBuf, Result<(), BoxErr>, String) {
    let dir = fixture_dir(judged);
    let out = dir.path().join("container");
    let mut status = Vec::new();
    let result = encoder.encode(&dir.path().display().to_string(), &out, &mut status);
    (dir, out, result, String::from_utf8(status).unwrap())
}

#[test]
fn an_admissible_checkpoint_encodes_a_vindex3_container() {
    for encoder in [
        PlanGatedEncoder::for_run(),
        PlanGatedEncoder { scope: None },
    ] {
        let (_dir, out, result, status) = encode(encoder, true);
        result.unwrap();
        assert!(matches!(
            detect_generation(&out),
            Ok(ContainerGeneration::V3)
        ));
        assert!(status.contains("plan admissible"), "{status}");
        assert!(status.contains("encoding to VINDEX3"), "{status}");
        assert!(status.contains("encoded"), "{status}");
    }
}

#[test]
fn an_inadmissible_checkpoint_is_refused_with_its_findings_and_writes_nothing() {
    let (_dir, out, result, status) = encode(PlanGatedEncoder { scope: None }, false);
    assert!(
        !status.contains("encoding"),
        "nothing announced before admission: {status}"
    );
    let err = result.unwrap_err().to_string();
    assert!(err.contains("cannot run"), "{err}");
    assert!(err.contains("some_future_field_nobody_reviewed"), "{err}");
    assert!(err.contains("larql vindex3 plan"), "{err}");
    assert!(!out.exists(), "a refusal writes no container");
}

#[test]
fn the_run_gate_is_text_generation() {
    assert_eq!(PlanGatedEncoder::for_run().scope, Some(RUN_CAPABILITY));
}

#[test]
fn scope_labels_read_as_prose() {
    assert_eq!(PlanGatedEncoder::for_run().scope_label(), "text generation");
    assert_eq!(
        PlanGatedEncoder { scope: None }.scope_label(),
        WHOLE_MODEL_LABEL
    );
    for capability in larql_vindex::format::vindex3::plan::capability::Capability::ALL {
        let label = PlanGatedEncoder {
            scope: Some(capability),
        }
        .scope_label();
        let snake = serde_json::to_value(capability).unwrap();
        assert_eq!(label, snake.as_str().unwrap().replace('_', " "));
    }
}

#[test]
fn narration_lines_name_their_figures() {
    assert!(staging_line(2048).contains("staged"));
    let line = transfer_line(&larql_vindex::format::vindex3::artifact::RemoteTransfer {
        name: "m".into(),
        fetched: 1,
        declared: 2,
        tensors: 3,
    });
    assert!(
        line.contains("fetched") && line.contains("3 tensor(s)"),
        "{line}"
    );
}
