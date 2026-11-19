//! FACT-ID-1 internals the statement surface cannot reach on its own: the
//! single-token rule on tokenizers other than the fixtures', the
//! fact-id collision refusal (a 64-bit collision cannot be produced on
//! demand), and the labels of every status, realization and durability.

use crate::executor::facts::{collision, single_token_id};
use crate::executor::{AssertedFact, FactDurability, FactRealization, FactStatus};
use larql_vindex::FactKey;

fn word_level(vocab: &str, pre_tokenizer: &str) -> larql_inference::tokenizers::Tokenizer {
    let json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{pre_tokenizer},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"[UNK]"}}}}"#
    );
    larql_inference::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
}

#[test]
fn a_target_is_admitted_only_as_one_token_that_decodes_back_to_it() {
    let tok = word_level(
        r#""[UNK]":0,"Paris":1,"New":2,"York":3"#,
        r#"{"type":"WhitespaceSplit"}"#,
    );
    assert_eq!(single_token_id(&tok, "Paris"), Some(1));
    assert_eq!(single_token_id(&tok, "New York"), None, "two tokens");
    // One token, but not this text: the unknown token decodes to [UNK].
    assert_eq!(single_token_id(&tok, "Lyon"), None);
}

#[test]
fn a_collision_names_both_keys_and_the_shared_id() {
    let key = FactKey::new("France", "capital");
    let other = FactKey::new("Spain", "capital");
    let text = collision(&key, &other);
    assert!(text.contains(&key.fact_id()), "{text}");
    assert!(
        text.contains("(france, capital)") && text.contains("(spain, capital)"),
        "{text}"
    );
}

#[test]
fn every_status_realization_and_durability_has_its_label() {
    let statuses = [
        (FactStatus::Active, "active"),
        (FactStatus::MultiToken, "multi_token"),
        (FactStatus::Unverified, "unverified"),
        (FactStatus::Ambiguous, "ambiguous"),
        (FactStatus::Inconsistent, "inconsistent"),
        (FactStatus::Compose, "compose"),
        (FactStatus::Degraded, "degraded"),
    ];
    for (status, label) in statuses {
        assert_eq!(status.label(), label);
    }
    assert_eq!(FactRealization::Knn.label(), "knn");
    assert_eq!(FactRealization::Compose.label(), "compose");
    let durabilities = [
        (FactDurability::Container, "container"),
        (FactDurability::Patch, "patch"),
        (FactDurability::Session, "session"),
        (FactDurability::Unknown, "unknown"),
    ];
    for (durability, label) in durabilities {
        assert_eq!(durability.label(), label);
    }
}

#[test]
fn a_fact_reports_its_key_in_normal_form() {
    let fact = AssertedFact {
        id: FactKey::new("New_York", "Located-In").fact_id(),
        entity: "New_York".into(),
        relation: "Located-In".into(),
        target: "USA".into(),
        target_id: Some(1),
        confidence: Some(0.9),
        realization: FactRealization::Knn,
        status: FactStatus::Active,
        version: 1,
        durability: FactDurability::Session,
        origin: "session".into(),
        layers: vec![0],
        slots: vec![],
    };
    assert_eq!(fact.key(), FactKey::new("new york", "located in"));
    assert_eq!(fact.key().fact_id(), fact.id);
}

/// The KNN insert path refuses a binding with no tokenizer on its own. The
/// fact check refuses first through the statement surface, so this calls the
/// path directly: each refusal stands without relying on the other.
#[test]
fn the_v3_knn_insert_path_refuses_a_binding_with_no_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let container =
        larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack(dir.path());
    let mut session = crate::executor::Session::new();
    session.set_verifier("");
    let stmt = format!(
        "USE \"{}\";",
        container.display().to_string().replace('\\', "\\\\")
    );
    session.execute(&crate::parse(&stmt).unwrap()).unwrap();
    let err = session
        .exec_insert_knn_v3("a", "b", "[7]", None, None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("INSERT needs a tokenizer") && err.contains("no tokenizer.json"),
        "{err}"
    );
    assert!(session.asserted_facts().unwrap().is_empty());
}
