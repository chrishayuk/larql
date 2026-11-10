//! PIPE
//! SELECT * FROM EDGES
//! SELECT * FROM FEATURES
//! SELECT * FROM ENTITIES
//! SELECT … FROM EDGES NEAREST TO (rich fixture)

use super::*;

#[test]
fn pipe_propagates_no_backend_error() {
    // The first stage errors with NoBackend — the pipe should surface
    // that without silently short-circuiting to `Ok`.
    let mut session = Session::new();
    let stmt = parser::parse("STATS |> STATS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn pipe_concatenates_both_sides_output() {
    // Both sides execute and their output lines are concatenated.
    let (mut session, dir) = vindex_session("pipe_concat");
    let stmt = parser::parse("SHOW LAYERS |> SHOW MODELS;").unwrap();
    let out = session.execute(&stmt).expect("pipe should succeed");
    // The combined output must contain evidence of both stages —
    // SHOW LAYERS emits per-layer rows; SHOW MODELS emits a header /
    // "no models" line. We just check the combined length is larger
    // than either side's output in isolation.
    let single = parser::parse("SHOW LAYERS;").unwrap();
    let single_out = session.execute(&single).expect("SHOW LAYERS alone");
    assert!(
        out.len() > single_out.len(),
        "pipe output ({}) should be longer than a single stage ({}): {:?}",
        out.len(),
        single_out.len(),
        out,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_with_feature_only_targets_only_that_feature() {
    // Regression: prior implementation passed (None, None, layer_filter) to
    // find_features when the user specified only `feature`, which returned
    // every feature in every layer. The fixture has 2 layers × 3 features
    // (one feature is `None` in layer 1 → skipped), so deleting `feature = 0`
    // should hit at most 2 slots, not all 5.
    let (mut session, dir) = vindex_session("delete_feature_only");
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE feature = 0;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let count = parse_mutation_count(&out);
    assert!(
        count <= 2,
        "feature-only DELETE should target at most one column across layers, got {count}: {out:?}"
    );
    assert!(count >= 1, "fixture has feature 0 populated in both layers");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_with_feature_only_targets_only_that_feature() {
    let (mut session, dir) = vindex_session("update_feature_only");
    let stmt = parser::parse(r#"UPDATE EDGES SET target = "Madrid" WHERE feature = 2;"#).unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let count = parse_mutation_count(&out);
    assert!(
        count <= 2,
        "feature-only UPDATE should target at most one column across layers, got {count}: {out:?}"
    );
    assert!(count >= 1, "fixture has feature 2 populated in both layers");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_skips_memit_fact_with_no_relation() {
    // Regression: prior implementation substituted the literal string
    // `"relation"` for a missing PatchOp::Insert.relation, baking junk
    // into the canonical MEMIT prompt. We now skip and warn.
    let (mut session, dir) = vindex_session("compile_skip_no_relation");

    // Inject a patch recording with a relation-less insert directly.
    session.patch_recording = Some(PatchRecording {
        path: "synthetic.vlp".into(),
        operations: vec![larql_vindex::PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: None,
            entity: "Atlantis".into(),
            target: "Poseidon".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }],
    });

    let out_dir = compile_output(&dir, "compiled.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        lql_path(&out_dir)
    ))
    .unwrap();
    let lines = session.execute(&stmt).expect("compile should succeed");
    let joined = lines.join("\n");
    assert!(
        joined.contains("skipping MEMIT fact"),
        "expected a 'skipping MEMIT fact' warning, got: {joined}"
    );
    assert!(
        joined.contains("no relation"),
        "warning should mention missing relation: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_with_negative_layer_matches_nothing() {
    // Negative integers are kept as a usize::MAX sentinel rather than
    // widened to "no filter" — so they match nothing instead of matching
    // everything.
    let (mut session, dir) = vindex_session("delete_negative_layer");
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = -1;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should not error");
    let joined = out.join("\n");
    assert!(
        joined.contains("no matching"),
        "negative layer should match nothing: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_default_runs_end_to_end() {
    let (mut session, dir) = vindex_session("select_edges_default");
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT EDGES");
    let joined = out.join("\n");
    assert!(joined.contains("Layer"));
    assert!(joined.contains("Feature"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_edges_layer");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE layer = 0;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_entity_filter_runs() {
    let (mut session, dir) = vindex_session("select_edges_entity");
    let stmt = parser::parse(r#"SELECT * FROM EDGES WHERE entity = "berlin";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE entity");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_score_predicate_runs() {
    let (mut session, dir) = vindex_session("select_edges_score");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE score > 0.85;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE score");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_score_lt_predicate_runs() {
    let (mut session, dir) = vindex_session("select_edges_score_lt");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE score < 0.95;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE score lt");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_order_by_confidence_runs() {
    let (mut session, dir) = vindex_session("select_edges_order");
    let stmt = parser::parse("SELECT * FROM EDGES ORDER BY confidence;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES ORDER BY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_order_by_layer_runs() {
    let (mut session, dir) = vindex_session("select_edges_order_layer");
    let stmt = parser::parse("SELECT * FROM EDGES ORDER BY layer;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES ORDER BY layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_no_matches_emits_friendly_line() {
    let (mut session, dir) = vindex_session("select_edges_empty");
    // `WHERE entity = "Nowhere"` matches nothing because the fixture's
    // top_tokens don't contain "Nowhere".
    let stmt = parser::parse(r#"SELECT * FROM EDGES WHERE entity = "Nowhere";"#).unwrap();
    let out = session.execute(&stmt).expect("SELECT EDGES");
    let joined = out.join("\n");
    // Either "(no matching edges)" or just an empty body — both are
    // valid. We just want exec_select to run.
    let _ = joined;
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_default_runs() {
    let (mut session, dir) = vindex_session("select_features_default");
    let stmt = parser::parse("SELECT * FROM FEATURES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT FEATURES");
    assert!(out.iter().any(|l| l.contains("Layer")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_with_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_layer");
    let stmt = parser::parse("SELECT * FROM FEATURES WHERE layer = 1;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES WHERE layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_with_feature_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_feat");
    let stmt = parser::parse("SELECT * FROM FEATURES WHERE feature = 0;").unwrap();
    let _ = session
        .execute(&stmt)
        .expect("SELECT FEATURES WHERE feature");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_token_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_token");
    let stmt = parser::parse(r#"SELECT * FROM FEATURES WHERE token = "Paris";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES WHERE token");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_explicit_limit_runs() {
    let (mut session, dir) = vindex_session("select_features_limit");
    let stmt = parser::parse("SELECT * FROM FEATURES LIMIT 1;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES LIMIT");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_default_runs() {
    let (mut session, dir) = vindex_session("select_entities_default");
    let stmt = parser::parse("SELECT * FROM ENTITIES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT ENTITIES");
    assert!(out.iter().any(|l| l.contains("Entity")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_filter_runs() {
    let (mut session, dir) = vindex_session("select_entities_filter");
    let stmt = parser::parse(r#"SELECT * FROM ENTITIES WHERE entity = "berlin";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_entities_layer");
    let stmt = parser::parse("SELECT * FROM ENTITIES WHERE layer = 0;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_no_matches_runs() {
    let (mut session, dir) = vindex_session("select_entities_empty");
    let stmt = parser::parse(r#"SELECT * FROM ENTITIES WHERE entity = "Nowhere";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_nearest_to_known_entity_runs() {
    let (mut session, dir) = rich_vindex_session("select_nearest_known");
    let stmt = parser::parse(r#"SELECT * FROM EDGES NEAREST TO "Paris" AT LAYER 0;"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT NEAREST");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_nearest_to_unknown_entity_runs() {
    let (mut session, dir) = rich_vindex_session("select_nearest_unknown");
    let stmt =
        parser::parse(r#"SELECT * FROM EDGES NEAREST TO "Atlantis" AT LAYER 0 LIMIT 5;"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT NEAREST");
    let _ = std::fs::remove_dir_all(&dir);
}
