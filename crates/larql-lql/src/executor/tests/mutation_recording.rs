//! `refresh_recorded_patch_ops_for_slots`: after a refine pass rewrites
//! an installed slot's vectors in the overlay, the recorded INSERT for
//! that slot must carry the vectors the overlay now holds.

use super::*;

const LAYER: usize = 0;
const FEATURE: usize = 1;
/// The fixture's hidden size (`make_test_vindex_dir`).
const HIDDEN: usize = 4;

fn recorded_insert(layer: usize, feature: usize) -> larql_vindex::PatchOp {
    larql_vindex::PatchOp::Insert {
        layer,
        feature,
        relation: None,
        entity: "e".into(),
        target: "t".into(),
        confidence: None,
        gate_vector_b64: None,
        up_vector_b64: None,
        down_vector_b64: None,
        down_meta: None,
    }
}

fn recorded_vectors(session: &Session) -> Vec<[bool; 3]> {
    session
        .patch_recording
        .as_ref()
        .expect("recording")
        .operations
        .iter()
        .filter_map(|op| match op {
            larql_vindex::PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => Some([
                gate_vector_b64.is_some(),
                up_vector_b64.is_some(),
                down_vector_b64.is_some(),
            ]),
            _ => None,
        })
        .collect()
}

#[test]
fn the_recorded_insert_takes_every_vector_the_overlay_holds_for_its_slot() {
    let (mut session, dir) = vindex_session("refresh_recorded_vectors");
    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![
            recorded_insert(LAYER, FEATURE),
            // A different slot: nothing was refreshed for it.
            recorded_insert(LAYER + 1, FEATURE),
        ],
    });
    {
        let Backend::Vindex { patched, .. } = &mut session.backend else {
            panic!("the fixture binds a legacy vindex");
        };
        // `insert_feature` is what gives a slot a gate override at all.
        patched.insert_feature(
            LAYER,
            FEATURE,
            vec![1.0; HIDDEN],
            larql_vindex::FeatureMeta {
                top_token: "t".into(),
                top_token_id: 0,
                c_score: 1.0,
                top_k: Vec::new(),
            },
        );
        patched.set_up_vector(LAYER, FEATURE, vec![2.0; HIDDEN]);
        patched.set_down_vector(LAYER, FEATURE, vec![3.0; HIDDEN]);
    }

    session
        .refresh_recorded_patch_ops_for_slots(&[(LAYER, FEATURE)])
        .expect("refresh");

    assert_eq!(
        recorded_vectors(&session),
        vec![[true, true, true], [false, false, false]]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_slot_with_no_overlay_vectors_leaves_its_recorded_insert_alone() {
    let (mut session, dir) = vindex_session("refresh_no_overlay");
    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![recorded_insert(LAYER, FEATURE)],
    });

    session
        .refresh_recorded_patch_ops_for_slots(&[(LAYER, FEATURE)])
        .expect("refresh");

    assert_eq!(recorded_vectors(&session), vec![[false, false, false]]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refreshing_no_slots_or_with_no_recording_is_a_no_op() {
    let (mut session, dir) = vindex_session("refresh_noop");

    session
        .refresh_recorded_patch_ops_for_slots(&[(LAYER, FEATURE)])
        .expect("no recording");
    assert!(session.patch_recording.is_none());

    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![recorded_insert(LAYER, FEATURE)],
    });
    session
        .refresh_recorded_patch_ops_for_slots(&[])
        .expect("no slots");
    assert_eq!(recorded_vectors(&session), vec![[false, false, false]]);
    let _ = std::fs::remove_dir_all(&dir);
}
