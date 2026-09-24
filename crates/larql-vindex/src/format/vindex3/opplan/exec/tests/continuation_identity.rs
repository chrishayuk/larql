//! **C1 of CONTINUATION-PLUGIN-1: a continuation provider names itself.**
//!
//! The forecast (`docs/represent/forecasts/continuation-plugin-1.json`)
//! predicts that every production continuation provider states a versioned
//! identity validated by the SAME rule a lowering identity obeys — one
//! shared validator, not a copy. The shared-rule gate below is the control
//! that keeps that from being a claim: the two planes are driven over one
//! table of inputs and must agree on every verdict, differing only in the
//! plane their refusals name. (`canonical/v1` is stated in `larql-kv` and
//! gated there.)

use super::super::continuation_identity::ContinuationIdentity;
use super::super::kv::RowKvState;
use super::super::lowering::LoweringIdentity;

#[test]
fn the_row_provider_states_a_valid_identity() {
    let id = RowKvState::identity();
    id.validate().unwrap();
    assert_eq!(id, ContinuationIdentity::new("row", 1));
    assert_eq!(id.to_string(), "row/v1");
}

#[test]
fn identity_is_family_and_revision_both() {
    let row = ContinuationIdentity::new("row", 1);
    assert_eq!(row, ContinuationIdentity::new("row".to_string(), 1));
    assert_ne!(row, ContinuationIdentity::new("row", 2));
    assert_ne!(row, ContinuationIdentity::new("canonical", 1));
    assert!(ContinuationIdentity::new("row", 1) < ContinuationIdentity::new("row", 2));
}

#[test]
fn identity_round_trips_through_serde() {
    let id = ContinuationIdentity::new("hostile-test-provider", 77);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, r#"{"family":"hostile-test-provider","revision":77}"#);
    assert_eq!(
        serde_json::from_str::<ContinuationIdentity>(&json).unwrap(),
        id
    );
}

/// One table, both planes: every verdict agrees, and each refusal names
/// its own plane and carries the same reason.
#[test]
fn continuation_and_lowering_share_one_rule() {
    let cases: &[(&str, u32, Option<&str>)] = &[
        ("row", 1, None),
        ("hostile-test-provider", 77, None),
        ("snake_case_9", 3, None),
        ("", 1, Some("the family is empty")),
        ("  ", 1, Some("the family is empty")),
        ("two words", 1, Some("contains ` `")),
        ("row/v1", 1, Some("contains `/`")),
        ("row", 0, Some("declares revision 0, which means unstated")),
    ];
    for &(family, revision, reason) in cases {
        let continuation = ContinuationIdentity::new(family, revision).validate();
        let lowering = LoweringIdentity::new(family, revision).validate();
        match reason {
            None => {
                continuation.unwrap_or_else(|e| panic!("{family:?}/{revision}: {e}"));
                lowering.unwrap_or_else(|e| panic!("{family:?}/{revision}: {e}"));
            }
            Some(reason) => {
                let c = continuation.unwrap_err().to_string();
                let l = lowering.unwrap_err().to_string();
                let (_, c_reason) = c
                    .split_once("continuation identity: ")
                    .unwrap_or_else(|| panic!("refusal does not name its plane: {c}"));
                let (_, l_reason) = l
                    .split_once("lowering identity: ")
                    .unwrap_or_else(|| panic!("refusal does not name its plane: {l}"));
                assert!(c_reason.contains(reason), "{c}");
                assert_eq!(
                    c_reason, l_reason,
                    "the planes disagree on why {family:?}/{revision} is refused"
                );
            }
        }
    }
}
