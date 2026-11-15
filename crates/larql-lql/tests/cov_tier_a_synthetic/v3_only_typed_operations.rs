//! The typed operations only a VINDEX3 binding answers refuse a VINDEX2 one by name.

use larql_lql::operations::CaptureSite;

use super::fresh_session;

#[test]
fn v3_only_typed_operations_refuse_a_v2_binding() {
    let (session, _dir, _) = fresh_session();
    let geometry = session
        .capture_geometry(CaptureSite::Logits, &[])
        .unwrap_err();
    assert!(
        geometry.to_string().contains("not bound to one"),
        "{geometry}"
    );
    let edits = session.edits_in_effect().unwrap_err();
    assert!(edits.to_string().contains("not bound to one"), "{edits}");
    let pending = session.has_pending_edits().unwrap_err();
    assert!(
        pending.to_string().contains("not bound to one"),
        "{pending}"
    );
}
