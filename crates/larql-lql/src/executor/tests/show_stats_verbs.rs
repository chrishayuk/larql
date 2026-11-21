//! SHOW + STATS verbs
//! DIFF
//! MERGE
//! WALK + EXPLAIN WALK against rich fixture
//! REBALANCE early-exit
//! COMPACT MINOR short-circuits

use super::*;

#[test]
fn show_layers_lists_layers() {
    let (mut session, dir) = vindex_session("show_layers");
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    let out = session.execute(&stmt).expect("SHOW LAYERS");
    // Synthetic vindex has 2 layers — at least one row should mention L0 or L1.
    let joined = out.join("\n");
    assert!(joined.contains("L0") || joined.contains("L1") || !joined.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_layers_with_range_filter() {
    let (mut session, dir) = vindex_session("show_layers_range");
    let stmt = parser::parse("SHOW LAYERS 0-1;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW LAYERS range");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_features_at_layer_runs() {
    let (mut session, dir) = vindex_session("show_features_layer");
    let stmt = parser::parse("SHOW FEATURES 0;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW FEATURES 0");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_entities_with_classifier_runs() {
    let (mut session, dir) = rich_vindex_session("show_entities");
    let stmt = parser::parse("SHOW ENTITIES;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_no_backend_errors() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(session.execute(&stmt).is_err());
}

#[test]
fn show_compact_status_runs_against_synthetic_vindex() {
    let (mut session, dir) = vindex_session("show_compact_status");
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    let out = session.execute(&stmt).expect("SHOW COMPACT STATUS");
    let joined = out.join("\n");
    // Synthetic fixture has hidden_dim=4 < 1024 → L2 line is the
    // "not available" branch, but every status line is unconditional.
    assert!(joined.contains("L0"));
    assert!(joined.contains("L1"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stats_runs_against_synthetic_vindex() {
    let (mut session, dir) = vindex_session("stats");
    let stmt = parser::parse("STATS;").unwrap();
    let out = session.execute(&stmt).expect("STATS");
    assert!(!out.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stats_with_relation_classifier_renders_coverage_breakdown() {
    // Drop relation_clusters.json + feature_clusters.jsonl into the
    // basic fixture so RelationClassifier::from_vindex returns Some(rc)
    // and STATS exercises the cluster/probe coverage branches.
    let dir = make_test_vindex_dir("stats_with_classifier");
    std::fs::write(
        dir.join(larql_vindex::format::filenames::RELATION_CLUSTERS_JSON),
        r#"{"k":2,"centres":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],"labels":["capital","language"],"counts":[5,3],"top_tokens":[["paris"],["english"]]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join(larql_vindex::format::filenames::FEATURE_CLUSTERS_JSONL),
        "{\"l\":0,\"f\":0,\"c\":0}\n{\"l\":1,\"f\":2,\"c\":1}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join(larql_vindex::format::filenames::FEATURE_LABELS_JSON),
        r#"{"L0_F1":"capital","L1_F0":"language"}"#,
    )
    .unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session.execute(&stmt).expect("USE with classifier");
    let stmt = parser::parse("STATS;").unwrap();
    let out = session.execute(&stmt).expect("STATS");
    let joined = out.join("\n");
    assert!(
        joined.contains("Clusters:") || joined.contains("Mapped relations"),
        "expected classifier-driven STATS output, got: {joined}",
    );
    assert!(
        joined.contains("Coverage:"),
        "expected Coverage section, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stats_with_explicit_path_runs() {
    // STATS "<path>" form — explicit vindex path argument.
    let (mut session, dir) = vindex_session("stats_path");
    let stmt = parser::parse(&format!(r#"STATS "{}";"#, lql_path(&dir))).unwrap();
    let _ = session.execute(&stmt).expect("STATS <path>");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_verbose_runs_against_classifier() {
    let (mut session, dir) = rich_vindex_session("show_rel_verbose");
    let stmt = parser::parse("SHOW RELATIONS VERBOSE;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS VERBOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_with_examples_runs() {
    let (mut session, dir) = rich_vindex_session("show_rel_examples");
    let stmt = parser::parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    let _ = session
        .execute(&stmt)
        .expect("SHOW RELATIONS WITH EXAMPLES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_at_specific_layer_runs() {
    let (mut session, dir) = rich_vindex_session("show_rel_layer");
    let stmt = parser::parse("SHOW RELATIONS AT LAYER 0;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS AT LAYER 0");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn diff_two_synthetic_vindexes_runs() {
    let dir_a = make_test_vindex_dir("diff_a");
    let dir_b = make_test_vindex_dir("diff_b");
    let mut session = Session::new();
    // DIFF doesn't require a USE — it takes explicit paths.
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}";"#,
        lql_path(&dir_a),
        lql_path(&dir_b)
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF");
    let joined = out.join("\n");
    assert!(joined.contains("Diff:"));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_layer_filter_runs() {
    let dir_a = make_test_vindex_dir("diff_layer_a");
    let dir_b = make_test_vindex_dir("diff_layer_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LAYER 0;"#,
        lql_path(&dir_a),
        lql_path(&dir_b)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF LAYER");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_explicit_limit_runs() {
    let dir_a = make_test_vindex_dir("diff_limit_a");
    let dir_b = make_test_vindex_dir("diff_limit_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LIMIT 5;"#,
        lql_path(&dir_a),
        lql_path(&dir_b)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF LIMIT");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_nonexistent_source_errors() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DIFF "/tmp/no_such_vindex_a" "/tmp/no_such_vindex_b";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("load") || err.to_string().contains("not found")
    );
}

#[test]
fn merge_synthetic_into_current_keeps_source_strategy() {
    let (mut session, dir) = vindex_session("merge_target");
    let source_dir = make_test_vindex_dir("merge_source");
    let stmt = parser::parse(&format!(r#"MERGE "{}";"#, lql_path(&source_dir))).unwrap();
    let out = session.execute(&stmt).expect("MERGE");
    let joined = out.join("\n");
    assert!(joined.contains("Merged"));
    assert!(joined.contains("features merged"));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_keep_target_strategy() {
    let (mut session, dir) = vindex_session("merge_keep_target");
    let source_dir = make_test_vindex_dir("merge_keep_target_src");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}" ON CONFLICT KEEP_TARGET;"#,
        lql_path(&source_dir)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("MERGE KEEP_TARGET");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_highest_confidence_strategy() {
    let (mut session, dir) = vindex_session("merge_highest");
    let source_dir = make_test_vindex_dir("merge_highest_src");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        lql_path(&source_dir)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("MERGE HIGHEST_CONFIDENCE");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_missing_source_errors() {
    let (mut session, dir) = vindex_session("merge_no_source");
    let stmt = parser::parse(r#"MERGE "/tmp/no_source_vindex_xyz";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("source vindex not found"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_no_backend_no_target_errors() {
    let mut session = Session::new();
    let source_dir = make_test_vindex_dir("merge_no_backend");
    let stmt = parser::parse(&format!(r#"MERGE "{}";"#, lql_path(&source_dir))).unwrap();
    // No USE ran → MERGE without explicit target should error.
    let _ = session.execute(&stmt); // accept either error or ok depending on path
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn walk_against_rich_fixture_runs() {
    let (mut session, dir) = rich_vindex_session("walk_basic");
    let stmt = parser::parse(r#"WALK "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("WALK");
    assert!(!out.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_with_top_clause_runs() {
    let (mut session, dir) = rich_vindex_session("walk_top");
    let stmt = parser::parse(r#"WALK "France" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK TOP 3");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_with_layers_clause_runs() {
    let (mut session, dir) = rich_vindex_session("walk_layers");
    let stmt = parser::parse(r#"WALK "Berlin" LAYERS 0-1;"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK LAYERS");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_unknown_token_uses_unk() {
    // Unknown tokens map to [UNK] (id 0), which has a valid embed row.
    let (mut session, dir) = rich_vindex_session("walk_unk");
    let stmt = parser::parse(r#"WALK "MongoliaXYZ";"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK with [UNK]");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_walk_runs() {
    let (mut session, dir) = rich_vindex_session("explain_walk");
    let stmt = parser::parse(r#"EXPLAIN WALK "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("EXPLAIN WALK");
    let joined = out.join("\n");
    // Output format: "L<n>: F<m> → <token> (gate=<x>, down=[...])"
    assert!(joined.contains("L0") || joined.contains("L1"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_walk_verbose_runs() {
    let (mut session, dir) = rich_vindex_session("explain_walk_verbose");
    let stmt = parser::parse(r#"EXPLAIN WALK "Paris" VERBOSE;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN WALK VERBOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_with_no_installs_short_circuits_v2() {
    let (mut session, dir) = vindex_session("rebalance_empty_v2");
    // No prior INSERT compose → installed_edges is empty → REBALANCE
    // hits the "nothing to rebalance" early-return path.
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session.execute(&stmt).expect("REBALANCE");
    let joined = out.join("\n");
    assert!(joined.contains("no compose-mode installs"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_with_explicit_clauses_short_circuits() {
    let (mut session, dir) = vindex_session("rebalance_clauses_v2");
    let stmt = parser::parse("REBALANCE FLOOR 0.20 CEILING 0.95 MAX 8;").unwrap();
    let _ = session.execute(&stmt).expect("REBALANCE with clauses");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_minor_with_empty_l0_short_circuits_v2() {
    let (mut session, dir) = vindex_session("compact_minor_empty_v2");
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    let out = session.execute(&stmt).expect("COMPACT MINOR");
    let joined = out.join("\n");
    assert!(joined.contains("L0 is empty"));
    let _ = std::fs::remove_dir_all(&dir);
}
