//! The CLI's `USE` hook: paths pass through, names miss with the CLI's
//! guidance, and the session carries the hook.

use super::*;

#[test]
fn a_path_shaped_literal_passes_through_unchanged() {
    let literal = "./larql-test-no-such-container";
    assert_eq!(
        resolve_use_literal(literal).unwrap(),
        Some(PathBuf::from(literal))
    );
}

#[test]
fn an_uncached_name_is_refused_with_the_fetch_guidance() {
    let err = resolve_use_literal("larql-test-no-such-model").unwrap_err();
    assert!(err.contains("never downloads"), "{err}");
    assert!(
        err.contains("larql run hf://<owner>/larql-test-no-such-model"),
        "{err}"
    );
}

#[test]
fn the_session_reports_the_guidance_from_use() {
    let mut session = session_with(None).unwrap();
    let stmt = larql_lql::parse("USE \"larql-test-no-such-model\";").unwrap();
    let err = session.execute(&stmt).unwrap_err().to_string();
    assert!(err.contains("never downloads"), "{err}");
}

#[test]
fn a_session_binds_under_stored_unless_told_otherwise() {
    // Not `session_with(None)`: that reads the environment, which a test must not depend on.
    assert_eq!(
        larql_lql::Session::new().representation_source(),
        RepresentationSource::Stored
    );
    for (word, source) in [
        ("stored", RepresentationSource::Stored),
        ("auto", RepresentationSource::Auto),
        ("transient", RepresentationSource::Transient),
    ] {
        assert_eq!(
            session_with(Some(word)).unwrap().representation_source(),
            source,
            "{word}"
        );
        assert_eq!(parse_representation_source(word).unwrap(), source);
    }
}

#[test]
fn an_unknown_source_is_an_error_and_never_a_default() {
    for word in ["", "Stored", "exact", "q8"] {
        let err = session_with(Some(word)).err().expect(word);
        assert!(err.contains("expected stored, auto or transient"), "{err}");
    }
}
