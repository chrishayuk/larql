//! No-weights vindex → browse-only INSERT branches
//! Alphabetic tokenizer → template-decoy loop in capture.rs
//! Custom layer bands → describe/exec.rs knowledge + output bands

use super::*;

#[test]
fn insert_compose_no_weights_uses_embedding_mode() {
    // has_model_weights=false → plan.rs sets use_constellation=false, so
    // capture.rs takes its early return (42-45) and exec_insert/mod.rs
    // takes the "embedding (no model weights)" summary arm (218-219).
    // balance + cross-fact checks are skipped (the `if use_constellation`
    // at mod.rs:108 is false).
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE COMPOSE;"#,
    )
    .expect("compose insert should still succeed in embedding mode");
    assert!(
        out.iter()
            .any(|l| l.contains("embedding (no model weights")),
        "expected embedding-mode summary, got: {out:?}"
    );
}

#[test]
fn insert_knn_no_weights_uses_embedding_key() {
    // has_model_weights=false → knn.rs takes the no-weights branch
    // (90-110): load_vindex_embeddings, entity_query_vec (the `?` at 108
    // succeeds because "[1]" tokenises in-vocab), and the
    // "embedding key (no model weights)" summary arm (151).
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert should succeed via embedding key");
    assert!(
        out.iter()
            .any(|l| l.contains("embedding key (no model weights")),
        "expected embedding-key summary, got: {out:?}"
    );
}

#[test]
fn insert_knn_no_weights_at_layer_hint() {
    // Same no-weights KNN path but with an explicit AT LAYER so knn.rs's
    // layer-hint branch (41-42) runs alongside the no-weights key build.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "language", "[4]") AT LAYER 1 MODE KNN;"#,
    )
    .expect("knn insert at layer should succeed");
    assert!(!out.is_empty());
}

#[test]
fn insert_compose_alphabetic_tokenizer_pushes_template_decoys() {
    // The on-disk tokenizer now decodes ids 0..31 to alphabetic 3+-char
    // words, so capture.rs's template-decoy loop (153-169) passes the
    // `word.len()>=3 && all alphabetic && !=entity` guard (161-164),
    // pushes decoys (165-167), and hits the
    // `template_decoys_added >= template_decoy_count` break (155) after
    // 10 are added. Entity "alpha" is in the vocab, so line 163's
    // `!word.eq_ignore_ascii_case(entity)` is exercised on both sides.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    write_alphabetic_tokenizer(dir.path(), 32);
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("alpha", "capital", "bravo") MODE COMPOSE AT LAYER 0;"#,
    )
    .expect("compose insert with alphabetic tokenizer should succeed");
    assert!(
        out.iter().any(|l| l.contains("Inserted")),
        "expected Inserted summary, got: {out:?}"
    );
}

#[test]
fn describe_knowledge_and_output_bands_render() {
    // Non-overlapping bands: syntax=(5,5) (no real layer), knowledge=(1,1),
    // output=(0,0). With two KNN entries — one at layer 1, one at layer 0 —
    // describe_format_and_split (format.rs:67-72) buckets the layer-1 edge
    // into knowledge and the layer-0 edge into output (neither is in the
    // syntax range, which is checked first). That makes BOTH
    // `formatted.knowledge` (exec.rs:107-115) and `formatted.output_band`
    // (exec.rs:116-129) non-empty in a single DESCRIBE.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_layer_bands(dir.path(), (5, 5), (1, 1), (0, 0));
    let (mut session, _dir, _) = use_dir(dir);

    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 1;"#,
    )
    .expect("knn insert at layer 1");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "language", "[8]") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert at layer 0");

    let out = try_run(&mut session, r#"DESCRIBE "[1]";"#).expect("describe ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Edges (L"),
        "expected knowledge band header, got:\n{joined}"
    );
    assert!(
        joined.contains("Output (L"),
        "expected output band header, got:\n{joined}"
    );
}

#[test]
fn describe_output_band_brief_cap() {
    // Brief mode uses DESCRIBE_MAX_OUTPUT_BRIEF for the output band cap
    // (exec.rs:121-122) rather than max_edges. Drive the output band in
    // BRIEF mode so the brief cap arm runs.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_layer_bands(dir.path(), (5, 5), (3, 3), (0, 1));
    let (mut session, _dir, _) = use_dir(dir);

    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert at layer 0");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "language", "[8]") MODE KNN AT LAYER 1;"#,
    )
    .expect("knn insert at layer 1");

    let out = try_run(&mut session, r#"DESCRIBE "[1]" BRIEF;"#).expect("describe brief ok");
    assert!(
        out.join("\n").contains("Output (L"),
        "expected output band in brief mode, got:\n{out:?}"
    );
}
