//! COMPILE INTO VINDEX integration tests

use super::*;

#[test]
fn compile_into_vindex_no_patches_succeeds() {
    let (mut session, dir) = vindex_session("compile_nopatches_v");

    let output = compile_output(&dir, "compiled.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        lql_path(&output)
    ))
    .unwrap();
    let out = session
        .execute(&stmt)
        .expect("COMPILE INTO VINDEX should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Compiled"),
        "expected compile output: {joined}"
    );
    assert!(output.exists(), "compiled vindex directory should exist");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_path_into_vindex_uses_supplied_source_without_active_backend() {
    let dir = make_test_vindex_dir("compile_path_source");
    let output = compile_output(&dir, "compiled_from_path.vindex");
    let mut session = Session::new();

    let stmt = parser::parse(&format!(
        r#"COMPILE "{}" INTO VINDEX "{}";"#,
        lql_path(&dir),
        lql_path(&output)
    ))
    .unwrap();
    let out = session
        .execute(&stmt)
        .expect("path-form COMPILE INTO VINDEX should load its source");
    let joined = out.join("\n");

    assert!(
        joined.contains("Compiled"),
        "expected compile output: {joined}"
    );
    assert!(output.exists(), "compiled vindex directory should exist");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_path_into_model_reports_supplied_source_requirements() {
    let dir = make_test_vindex_dir("compile_path_model_source");
    let output = compile_output(&dir, "model_out");
    let mut session = Session::new();

    let stmt = parser::parse(&format!(
        r#"COMPILE "{}" INTO MODEL "{}";"#,
        lql_path(&dir),
        lql_path(&output)
    ))
    .unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("browse-only source should fail after path source is loaded");

    assert!(
        err.to_string().contains("requires model weights"),
        "expected source-level model-weight error, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_into_vindex_with_down_overrides_bakes_them() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("compile_bake_down");

    // Create a synthetic down_weights.bin — per-layer [hidden, intermediate] f32.
    // hidden=4, intermediate=3, num_layers=2.
    let layer_floats = 4 * 3;
    let total = 2 * layer_floats;
    let bytes: Vec<u8> = (0..total)
        .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
        .collect();
    std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "test".into(),
                top_token_id: 5,
                c_score: 0.9,
                top_k: vec![TopKEntry {
                    token: "test".into(),
                    token_id: 5,
                    logit: 0.9,
                }],
            },
        );
        overlay.set_down_vector(0, 0, vec![0.5, 0.6, 0.7, 0.8]);
    }
    let output = compile_output(&dir, "compiled_baked.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        lql_path(&output)
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("COMPILE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Down overrides baked"),
        "expected baked overrides: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_on_conflict_fail_detects_collision() {
    use larql_vindex::{PatchOp, VindexPatch};

    let (mut session, dir) = vindex_session("compile_conflict_fail");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        let mkp = |e: &str| VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("r".into()),
                entity: e.into(),
                target: "t".into(),
                confidence: Some(0.9),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = compile_output(&dir, "compiled_fail.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT FAIL;"#,
        lql_path(&output)
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(
        result.is_err(),
        "ON CONFLICT FAIL should error on collision"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("FAIL") || msg.contains("colliding"),
        "error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_on_conflict_last_wins_succeeds() {
    use larql_vindex::{PatchOp, VindexPatch};

    let (mut session, dir) = vindex_session("compile_conflict_lw");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        let mkp = |e: &str| VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("r".into()),
                entity: e.into(),
                target: "t".into(),
                confidence: Some(0.9),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = compile_output(&dir, "compiled_lw.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT LAST_WINS;"#,
        lql_path(&output)
    ))
    .unwrap();
    assert!(
        session.execute(&stmt).is_ok(),
        "LAST_WINS should succeed despite collision"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
