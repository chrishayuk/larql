//! PATCH EDGE CASES

use super::*;

#[test]
fn patch_empty_operations() {
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![],
    };
    assert_eq!(patch.len(), 0);
    let (i, u, d) = patch.counts();
    assert_eq!((i, u, d), (0, 0, 0));
}

#[test]
fn patch_multiple_patches_stack() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Patch 1: update F0
    let p1 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    patched.try_apply_patch(p1).expect("patch applies");

    // Patch 2: update F1
    let p2 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 1,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "Munich".into(),
                top_token_id: 301,
                c_score: 0.95,
            }),
        }],
    };
    patched.try_apply_patch(p2).expect("patch applies");

    assert_eq!(patched.num_patches(), 2);
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "London");
    assert_eq!(patched.feature_meta(0, 1).unwrap().top_token, "Munich");
    assert_eq!(patched.feature_meta(0, 2).unwrap().top_token, "Europe"); // unchanged
}

#[test]
fn patched_vindex_later_patch_overrides_earlier() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Both patches modify F0
    let p1 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    let p2 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "Tokyo".into(),
                top_token_id: 400,
                c_score: 0.88,
            }),
        }],
    };
    patched.try_apply_patch(p1).expect("patch applies");
    patched.try_apply_patch(p2).expect("patch applies");

    // P2 wins
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Tokyo");
}
