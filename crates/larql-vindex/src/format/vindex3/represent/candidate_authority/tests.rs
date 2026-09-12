//! Structured hostile controls re-seal metadata deliberately: these must
//! reach relation checks, not merely fail the outer integrity checksum.
use super::*;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::represent::{compile_representation, RepresentSpec};

fn candidate() -> tempfile::TempDir {
    let source = tempfile::tempdir().unwrap();
    let checkpoint = source.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let container = source.path().join("source");
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");
    let out = tempfile::tempdir().unwrap();
    compile_representation(&container, out.path(), &RepresentSpec::nvfp4()).unwrap();
    out
}

fn index(dir: &Path) -> CandidateIndex {
    serde_json::from_slice(&std::fs::read(dir.join(CANDIDATE_INDEX_FILE)).unwrap()).unwrap()
}

fn persist(dir: &Path, mut index: CandidateIndex) {
    let digest = binding(&index, index.authority.as_ref().unwrap()).unwrap();
    index.authority.as_mut().unwrap().binding_sha256 = digest;
    super::super::compiler::write_index_atomically(&index, &dir.join(CANDIDATE_INDEX_FILE))
        .unwrap();
}

fn assert_invalid(dir: &Path, text: &str) {
    let refusal = read_candidate(dir).unwrap_err();
    assert!(
        matches!(refusal, CandidateAuthorityRefusal::Invalid(_)),
        "{refusal}"
    );
    assert!(refusal.to_string().contains(text), "{refusal}");
}

#[test]
fn deserialisation_cannot_bypass_surface_and_decision_invariants() {
    let dir = candidate();
    let original = index(dir.path());
    for arm in [
        "surface-order",
        "duplicate",
        "decision-order",
        "missing",
        "empty-encoding",
    ] {
        let mut changed = serde_json::to_value(&original).unwrap();
        match arm {
            "surface-order" => changed["authority"]["surface"]["entries"]
                .as_array_mut()
                .unwrap()
                .reverse(),
            "duplicate" => {
                let entries = changed["authority"]["surface"]["entries"]
                    .as_array_mut()
                    .unwrap();
                entries.push(entries[0].clone());
            }
            "decision-order" => changed["authority"]["decisions"]["decisions"]
                .as_array_mut()
                .unwrap()
                .reverse(),
            "missing" => {
                changed["authority"]["decisions"]["decisions"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            "empty-encoding" => {
                changed["authority"]["decisions"]["decisions"][0]["encoding"] =
                    serde_json::json!({"Compiled":""})
            }
            _ => unreachable!(),
        }
        persist(dir.path(), serde_json::from_value(changed).unwrap());
        let expected = match arm {
            "surface-order" | "decision-order" => "canonical order",
            "duplicate" => "twice",
            "missing" => "every surface tensor exactly once",
            "empty-encoding" => "non-source encoding",
            _ => unreachable!(),
        };
        assert_invalid(dir.path(), expected);
    }
}

#[test]
fn individually_valid_payloads_do_not_excuse_inconsistent_ledger_relations() {
    let dir = candidate();
    let original = index(dir.path());
    for arm in [
        "missing-payload",
        "outside",
        "encoding",
        "key",
        "unsealed",
        "overlap",
        "unsafe-path",
    ] {
        let mut changed = original.clone();
        let first = changed.ledger.sealed.values().next().unwrap().clone();
        match arm {
            "missing-payload" => {
                changed
                    .authority
                    .as_mut()
                    .unwrap()
                    .payloads
                    .remove(&first.object);
            }
            "outside" => {
                changed
                    .ledger
                    .sealed
                    .values_mut()
                    .next()
                    .unwrap()
                    .target_offset = u64::MAX
            }
            "encoding" => {
                changed.ledger.sealed.values_mut().next().unwrap().encoding = "wrong".into()
            }
            "key" => {
                let (key, seal) = changed.ledger.sealed.pop_first().unwrap();
                changed.ledger.sealed.insert(format!("bad-{key}"), seal);
            }
            "unsealed" => {
                changed.ledger.sealed.pop_first();
            }
            "overlap" => {
                let second = changed.ledger.sealed.values_mut().nth(1).unwrap();
                second.target_offset = first.target_offset;
                second.target_len = first.target_len;
                second.target_hash = first.target_hash;
            }
            "unsafe-path" => {
                changed
                    .authority
                    .as_mut()
                    .unwrap()
                    .payloads
                    .values_mut()
                    .next()
                    .unwrap()
                    .file = "../outside".into()
            }
            _ => unreachable!(),
        }
        persist(dir.path(), changed);
        let expected = match arm {
            "missing-payload" => "no payload for sealed object",
            "outside" => "outside its payload",
            "encoding" => "disagrees with effective decisions",
            "key" => "ledger key",
            "unsealed" => "no completed seal",
            "overlap" => "seals overlap",
            "unsafe-path" => "not a relative file",
            _ => unreachable!(),
        };
        assert_invalid(dir.path(), expected);
    }
}

#[test]
fn operand_hash_still_binds_after_the_outer_metadata_is_resealed() {
    let dir = candidate();
    let mut changed = index(dir.path());
    changed
        .ledger
        .sealed
        .values_mut()
        .next()
        .unwrap()
        .target_hash = "wrong".into();
    persist(dir.path(), changed);
    assert!(
        matches!(read_candidate(dir.path()), Err(CandidateAuthorityRefusal::Binding { what, .. }) if what.starts_with("operand"))
    );
}

#[test]
fn finalising_cannot_claim_a_seal_outside_the_completed_surface() {
    let dir = candidate();
    let mut idx = index(dir.path());
    let a = idx.authority.take().unwrap();
    let empty = TensorSurface::new([]).unwrap();
    let err = finish(
        &mut idx,
        dir.path(),
        empty,
        ResolvedDecisionVector::default(),
        a.payloads.into_iter().map(|(o, p)| (o, p.file)).collect(),
        a.executable_root,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("outside the compilation's effective surface"));
}

#[test]
fn a_sidecar_cannot_downgrade_its_root_binding_to_an_inline_candidate() {
    let dir = candidate();
    let mut idx = index(dir.path());
    idx.authority.as_mut().unwrap().executable_root = ExecutableRootBinding::InlineCandidate;
    persist(dir.path(), idx); // valid outer binding, wrong relationship
    assert_invalid(dir.path(), "does not match the candidate authority carrier");
}
