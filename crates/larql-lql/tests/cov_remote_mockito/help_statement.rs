//! `HELP` on a remote session: answered locally, never forwarded and never
//! refused as unsupported.

use super::*;

#[test]
fn help_is_answered_locally_on_a_remote_session() {
    // The server mocks `/v1/stats` only: any forwarded request would get
    // mockito's 501 and fail the statement.
    let (_server, mut session) = connect();
    let overview = ok_joined(&mut session, "HELP;");
    assert!(overview.starts_with("LQL statements"), "{overview}");
    let page = ok_joined(&mut session, "HELP EXIT CODES;");
    assert!(page.starts_with("EXIT CODES"), "{page}");
}

#[test]
fn an_unknown_topic_is_the_same_error_on_a_remote_session() {
    let (_server, mut session) = connect();
    let err = run(&mut session, "HELP REMOVE DIRECTON;").unwrap_err();
    assert!(err.contains("no help topic"), "{err}");
    assert!(err.contains("REMOVE DIRECTION"), "{err}");
}
