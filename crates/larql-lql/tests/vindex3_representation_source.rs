//! The representation source an LQL session binds a VINDEX3 container under: `stored` unless the embedding
//! program chooses otherwise, said in the `USE` reply, and applied to the container the session executes.

use larql_lql::{parse, RepresentationSource, Session, DEFAULT_REPRESENTATION_SOURCE};
use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;

fn use_container(session: &mut Session, container: &std::path::Path) -> Vec<String> {
    let statement = format!(
        "USE \"{}\";",
        container.display().to_string().replace('\\', "\\\\")
    );
    session.execute(&parse(&statement).unwrap()).unwrap()
}

fn execution_line(reply: &[String]) -> &str {
    reply
        .iter()
        .find(|l| l.starts_with("Execution:"))
        .unwrap_or_else(|| panic!("USE names its execution policy: {reply:?}"))
}

#[test]
fn a_session_binds_under_stored_by_default_and_says_so() {
    assert_eq!(DEFAULT_REPRESENTATION_SOURCE, RepresentationSource::Stored);
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let mut session = Session::new();
    assert_eq!(
        session.representation_source(),
        RepresentationSource::Stored
    );
    let reply = use_container(&mut session, &container);
    let line = execution_line(&reply);
    assert!(line.contains("representation source stored"), "{line}");
    assert!(line.contains("nothing is manufactured at load"), "{line}");
}

#[test]
fn an_explicit_source_is_the_one_a_later_use_binds_under() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    for (source, said) in [
        (RepresentationSource::Auto, "representation source auto"),
        (
            RepresentationSource::Transient,
            "representation source transient",
        ),
        (RepresentationSource::Stored, "representation source stored"),
    ] {
        let mut session = Session::new();
        session.set_representation_source(source);
        assert_eq!(session.representation_source(), source);
        let reply = use_container(&mut session, &container);
        assert!(execution_line(&reply).contains(said), "{reply:?}");
    }
}
