//! Architecture B KNN Store tests — UNIFIED into FFN overlay

use super::*;

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_populates_store() {
    // INSERT on a browse-only vindex (no model weights) uses embedding-key fallback
    let (mut session, dir) = vindex_session("knn_insert");

    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    let out = session.execute(&stmt).expect("INSERT should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Inserted"),
        "expected insert confirmation: {joined}"
    );
    assert!(
        joined.contains("KNN store"),
        "expected KNN store mode: {joined}"
    );
    assert!(joined.contains("1 entries"), "expected 1 entry: {joined}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_multiple_facts() {
    let (mut session, dir) = vindex_session("knn_multi");

    for (entity, target) in &[
        ("Atlantis", "Poseidon"),
        ("Lemuria", "Mu"),
        ("Agartha", "Shambhala"),
    ] {
        let sql = format!(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "capital", "{target}");"#
        );
        let stmt = parser::parse(&sql).unwrap();
        session.execute(&stmt).expect("INSERT should succeed");
    }

    // Check KNN store has 3 entries
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Wakanda", "capital", "Birnin");"#,
    )
    .unwrap();
    let out = session.execute(&stmt).expect("INSERT should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("4 entries"), "expected 4 entries: {joined}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_describe_shows_inserted_edges() {
    let (mut session, dir) = vindex_session("knn_describe");

    // Insert a fact
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    session.execute(&stmt).expect("INSERT");

    // Verify the KNN store is populated by checking via the overlay accessor
    let overlay = session.patched_overlay_mut().expect("vindex backend");
    let knn_entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(knn_entries.len(), 1, "expected 1 KNN entry for Atlantis");
    assert_eq!(knn_entries[0].1.target_token, "Poseidon");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_delete_removes_entries() {
    let (mut session, dir) = vindex_session("knn_delete");

    // Insert two facts for different entities
    for sql in &[
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Lemuria", "capital", "Mu");"#,
    ] {
        let stmt = parser::parse(sql).unwrap();
        session.execute(&stmt).expect("INSERT");
    }

    // Verify both in store
    let overlay = session.patched_overlay_mut().expect("vindex");
    assert_eq!(overlay.knn_store.len(), 2);

    // Delete Atlantis via direct KNN store removal (since base features may not exist)
    overlay.knn_store.remove_by_entity("Atlantis");
    assert_eq!(overlay.knn_store.len(), 1);
    assert_eq!(overlay.knn_store.entries_for_entity("Atlantis").len(), 0);
    assert_eq!(overlay.knn_store.entries_for_entity("Lemuria").len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_compile_saves_and_loads() {
    let (mut session, dir) = vindex_session("knn_compile");

    // Insert a fact
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    session.execute(&stmt).expect("INSERT");

    // Compile
    let output = compile_output(&dir, "compiled_knn.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        lql_path(&output)
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("COMPILE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("KNN store: 1 entries"),
        "expected KNN count: {joined}"
    );

    // Verify knn_store.bin exists
    assert!(
        output.join("knn_store.bin").exists(),
        "knn_store.bin should be in compiled vindex"
    );

    // Load the compiled vindex and verify KNN store survives round-trip
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&output))).unwrap();
    session.execute(&stmt).expect("USE compiled vindex");

    // Check the KNN store is loaded with the fact
    let overlay = session.patched_overlay_mut().expect("vindex");
    let entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(
        entries.len(),
        1,
        "expected 1 KNN entry after compile+reload"
    );
    assert_eq!(entries[0].1.target_token, "Poseidon");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn knn_store_patch_op_serialization() {
    // Verify InsertKnn PatchOp serializes and deserializes correctly
    let op = larql_vindex::PatchOp::InsertKnn {
        layer: 26,
        entity: "Atlantis".into(),
        relation: "capital".into(),
        target: "Poseidon".into(),
        target_id: 42,
        confidence: Some(1.0),
        key_vector_b64: larql_vindex::patch::core::encode_gate_vector(&[1.0, 0.0, 0.0, 0.0]),
    };
    let json = serde_json::to_string(&op).unwrap();
    assert!(
        json.contains("insert_knn"),
        "expected insert_knn tag: {json}"
    );
    assert!(json.contains("Atlantis"), "expected entity: {json}");

    // Round-trip
    let decoded: larql_vindex::PatchOp = serde_json::from_str(&json).unwrap();
    match decoded {
        larql_vindex::PatchOp::InsertKnn {
            entity,
            target,
            layer,
            ..
        } => {
            assert_eq!(entity, "Atlantis");
            assert_eq!(target, "Poseidon");
            assert_eq!(layer, 26);
        }
        _ => panic!("expected InsertKnn variant"),
    }
}

#[test]
fn knn_store_delete_knn_patch_op() {
    let op = larql_vindex::PatchOp::DeleteKnn {
        entity: "Atlantis".into(),
    };
    let json = serde_json::to_string(&op).unwrap();
    assert!(
        json.contains("delete_knn"),
        "expected delete_knn tag: {json}"
    );

    let decoded: larql_vindex::PatchOp = serde_json::from_str(&json).unwrap();
    match decoded {
        larql_vindex::PatchOp::DeleteKnn { entity } => {
            assert_eq!(entity, "Atlantis");
        }
        _ => panic!("expected DeleteKnn variant"),
    }
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_at_layer_hint() {
    let (mut session, dir) = vindex_session("knn_layer_hint");

    // Synthetic vindex has only 2 layers (0, 1), so use AT LAYER 0
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon") AT LAYER 0;"#,
    ).unwrap();
    let out = session.execute(&stmt).expect("INSERT AT LAYER");
    let joined = out.join("\n");
    assert!(joined.contains("L0"), "expected L0 in output: {joined}");

    // Verify it went to layer 0
    let overlay = session.patched_overlay_mut().expect("vindex");
    let entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 0, "expected layer 0");

    let _ = std::fs::remove_dir_all(&dir);
}
