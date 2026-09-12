//! Independent consumer crate: only public APIs, no writer-side object
//! survives fixture construction. Every hostile case starts from real bytes.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::{
    encode::segment::read_segment_header,
    fixtures::{dense_f32_model, encode_fixture_container},
    index::Vindex3Index,
    represent::{
        candidate_authority::{
            read_candidate, verify_candidate, CandidateAuthorityRefusal, CANDIDATE_INDEX_FILE,
        },
        compile_representation,
        compiler::read_source_identity,
        map::PrecisionMap,
        policy::classify_in,
        state::{
            PackLayoutAdmission, RepresentationState, RepresentationStateId, SurfaceTensor,
            TensorSurface,
        },
        RepresentSpec,
    },
};

struct Fixture {
    dir: tempfile::TempDir,
    source: PathBuf,
    expected: RepresentationStateId,
    surface: TensorSurface,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let source = dir.path().join("source");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, "target");
    let index: Vindex3Index =
        serde_json::from_slice(&std::fs::read(source.join("index.json")).unwrap()).unwrap();
    let mut tensors = BTreeMap::new();
    for entry in index.representations.values() {
        let (header, _) = read_segment_header(&source.join(&entry.segment)).unwrap();
        for t in header.tensors {
            let role = classify_in(true, &entry.object, &t.name, &t.shape);
            let tensor = SurfaceTensor::new(&entry.object, &t.name, role, t.shape);
            tensors.insert((entry.object.clone(), t.name), tensor);
        }
    }
    let surface = TensorSurface::new(tensors.into_values()).unwrap();
    let spec = RepresentSpec::nvfp4();
    let map =
        PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);
    let expected = RepresentationState::resolve(
        &read_source_identity(&source).unwrap(),
        &surface,
        &map,
        &PackLayoutAdmission,
    )
    .id()
    .clone();
    Fixture {
        dir,
        source,
        expected,
        surface,
    }
}

fn compile(f: &Fixture, name: &str, spec: &RepresentSpec) -> PathBuf {
    let out = f.dir.path().join(name);
    // The report is dropped here; every compiler/store/arena/file handle
    // created by the public compiler has already been dropped on return.
    drop(compile_representation(&f.source, &out, spec).unwrap());
    out
}

fn edit(out: &Path, f: impl FnOnce(&mut serde_json::Value)) {
    let path = out.join(CANDIDATE_INDEX_FILE);
    let mut doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    f(&mut doc);
    std::fs::write(path, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
}

#[test]
fn fresh_public_reader_reconstructs_the_independently_predicted_state() {
    let f = fixture();
    let out = compile(&f, "candidate", &RepresentSpec::nvfp4());
    let expected = f.expected.clone();
    let moved = tempfile::tempdir().unwrap();
    let destination = moved.path().join("renamed");
    std::fs::rename(out, &destination).unwrap();
    drop(f); // checkpoint, source container and compiler-side fixture gone
    let state = read_candidate(&destination).unwrap();
    assert_eq!(state.id(), &expected);
    assert!(state.decisions().compiled() > 0);
}

#[test]
fn payload_mutation_refuses_even_when_every_metadata_field_is_unchanged() {
    let f = fixture();
    let out = compile(&f, "candidate", &RepresentSpec::nvfp4());
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join(CANDIDATE_INDEX_FILE)).unwrap()).unwrap();
    let payloads = doc["authority"]["payloads"].as_object().unwrap();
    let payload = payloads
        .iter()
        .find(|(object, _)| object.contains("decoder"))
        .unwrap()
        .1;
    let file = out.join(payload["file"].as_str().unwrap());
    let mut bytes = std::fs::read(&file).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(file, bytes).unwrap();
    assert!(
        matches!(read_candidate(&out), Err(CandidateAuthorityRefusal::Binding { what, .. }) if what.contains("payload"))
    );
}

#[test]
fn effective_decisions_surface_and_source_are_bound_individually() {
    let f = fixture();
    for field in ["decisions", "surface", "source"] {
        let out = compile(&f, field, &RepresentSpec::nvfp4());
        edit(&out, |index| match field {
            "decisions" => {
                index["authority"]["decisions"]["decisions"][0]["encoding"] =
                    serde_json::json!({"Compiled":"Q6_K"})
            }
            "surface" => {
                index["authority"]["surface"]["entries"][0]["shape"][0] = serde_json::json!(999)
            }
            "source" => {
                index["source"]["identity"]["semantic"]["graph_hash"] = serde_json::json!("changed")
            }
            _ => unreachable!(),
        });
        assert!(
            matches!(read_candidate(&out), Err(CandidateAuthorityRefusal::Binding { what, .. }) if what == "metadata"),
            "{field}"
        );
    }
}

#[test]
fn a_corrupt_stored_id_is_detected_and_the_recomputed_id_is_reported() {
    let f = fixture();
    let out = compile(&f, "candidate", &RepresentSpec::nvfp4());
    edit(&out, |index| {
        index["authority"]["state_id"] = serde_json::json!("invented")
    });
    match read_candidate(&out).unwrap_err() {
        CandidateAuthorityRefusal::StoredState { stored, recomputed } => {
            assert_eq!(stored.as_str(), "invented");
            assert_eq!(recomputed, f.expected);
        }
        e => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn caller_request_cannot_author_the_compilation_and_valid_y_is_not_requested_x() {
    let f = fixture();
    let source_state = RepresentationState::resolve(
        &read_source_identity(&f.source).unwrap(),
        &f.surface,
        &source_map(),
        &PackLayoutAdmission,
    );
    use larql_vindex::format::vindex3::represent::{
        measurement::EvidenceScale,
        state::{EvidenceBank, InstrumentSemantics, MeasurementKey},
    };
    let bank = EvidenceBank::new("fixture/v1", "manifest", ["one"], 1);
    let instrument = InstrumentSemantics::new("kl", "distribution", "all", "fixture/v1");
    let key = MeasurementKey::new(
        source_state.id(),
        &bank.id(),
        EvidenceScale::Diagnostic,
        &instrument.id(),
    );
    let executor_restatement = key.clone();
    assert_eq!(executor_restatement, key);
    let requested = key.state().clone();
    assert_ne!(requested, f.expected);
    let out = compile(&f, "caller-asks-for-source", &RepresentSpec::nvfp4());
    let established = read_candidate(&out).unwrap();
    assert_eq!(established.id(), &f.expected);
    let refusal = verify_candidate(&out, &requested).unwrap_err();
    assert!(refusal.to_string().contains(requested.as_str()));
    assert!(refusal.to_string().contains(f.expected.as_str()));
    assert!(
        matches!(refusal, CandidateAuthorityRefusal::RequestedState { expected, observed } if expected == requested && observed == f.expected)
    );
    assert_eq!(verify_candidate(&out, &f.expected).unwrap(), established);
}

#[test]
fn requested_map_provenance_and_lowering_choices_do_not_replace_results() {
    let f = fixture();
    let out = compile(&f, "candidate", &RepresentSpec::nvfp4());
    edit(&out, |index| {
        index["map"] = serde_json::to_value(source_map()).unwrap();
        index["source"]["locator_hint"] = serde_json::json!("/elsewhere");
        index["source"]["identity"]["artifact"] = serde_json::json!("re-exported");
        index["selected_representation"] = serde_json::json!("source");
        index["lowering_provider"] = serde_json::json!("outside/v999");
    });
    assert_eq!(read_candidate(&out).unwrap().id(), &f.expected);
}

#[test]
fn missing_and_unknown_authority_refuse() {
    let f = fixture();
    for field in ["absent", "schema", "state_semantics"] {
        let out = compile(&f, field, &RepresentSpec::nvfp4());
        edit(&out, |index| {
            if field == "absent" {
                index.as_object_mut().unwrap().remove("authority");
            } else {
                index["authority"][field] = serde_json::json!("unknown/v9");
            }
        });
        assert!(match read_candidate(&out) {
            Err(CandidateAuthorityRefusal::Missing) => field == "absent",
            Err(CandidateAuthorityRefusal::Version { .. }) => field != "absent",
            result => panic!("unexpected {result:?}"),
        });
    }
}

fn source_map() -> PrecisionMap {
    PrecisionMap {
        name: "source".into(),
        encoding: "NVFP4_PACK_V1".into(),
        roles: vec![],
        exceptions: vec![],
    }
}

#[test]
fn actual_layout_refusal_persists_source_despite_the_requested_encoding() {
    use larql_vindex::format::vindex3::represent::{
        kquant,
        state::{LayoutAdmission, NoLayoutConstraint, ResolvedEncoding},
    };
    struct KQuantAdmission;
    impl LayoutAdmission for KQuantAdmission {
        fn admits(&self, encoding: &str, tensor: &SurfaceTensor) -> bool {
            kquant::lookup(encoding)
                .unwrap()
                .plan(&tensor.shape, &tensor.tensor)
                .is_ok()
        }
    }
    let f = fixture();
    let mut spec = RepresentSpec::nvfp4();
    spec.encoding = "Q6_K".into();
    let map =
        PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);
    let model = read_source_identity(&f.source).unwrap();
    let effective = RepresentationState::resolve(&model, &f.surface, &map, &KQuantAdmission);
    let requested = RepresentationState::resolve(&model, &f.surface, &map, &NoLayoutConstraint);
    assert_ne!(effective.id(), requested.id());
    assert!(effective.decisions().compiled() > 0);
    assert!(!effective.decisions().layout_refused().is_empty());
    let out = compile(&f, "mixed-layout", &spec);
    let read = read_candidate(&out).unwrap();
    assert_eq!(read, effective);
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join(CANDIDATE_INDEX_FILE)).unwrap()).unwrap();
    for refused in read.decisions().layout_refused() {
        assert!(matches!(
            refused.encoding,
            ResolvedEncoding::LayoutRefused { .. }
        ));
        let payload = &doc["authority"]["payloads"][&refused.object];
        let (header, _) =
            read_segment_header(&out.join(payload["file"].as_str().unwrap())).unwrap();
        assert_eq!(
            header
                .tensors
                .iter()
                .find(|t| t.name == refused.tensor)
                .unwrap()
                .dtype,
            "F32"
        );
    }
}
