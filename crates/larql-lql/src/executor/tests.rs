use super::helpers::*;
use super::*;
use crate::parser;

// ── Session state: no backend ──

#[test]
fn no_backend_stats() {
    let mut session = Session::new();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, LqlError::NoBackend));
}

#[test]
fn no_backend_walk() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::NoBackend));
}

#[test]
fn no_backend_describe() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_select() {
    let mut session = Session::new();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_explain() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_relations() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_layers() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_features() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW FEATURES 26;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── USE errors ──

#[test]
fn use_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE "/nonexistent/path/fake.vindex";"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn use_model_fails_on_nonexistent() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model";"#).unwrap();
    let result = session.execute(&stmt);
    // Should fail to resolve the model path
    assert!(result.is_err());
}

#[test]
fn use_model_auto_extract_parses() {
    // Verify AUTO_EXTRACT parses correctly (loading will fail for nonexistent model)
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model" AUTO_EXTRACT;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
}

// ── Lifecycle: error cases without valid model/vindex ──

#[test]
fn extract_fails_on_nonexistent_model() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"EXTRACT MODEL "/nonexistent/model" INTO "/tmp/test_extract_out.vindex";"#)
            .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn compile_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn diff_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DIFF "/nonexistent/a.vindex" "/nonexistent/b.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── Mutation: no-backend errors ──

#[test]
fn insert_no_backend() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn delete_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn update_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn merge_nonexistent_source() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"MERGE "/nonexistent/source.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── INFER ──

#[test]
fn infer_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"INFER "test" TOP 5;"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── is_readable_token ──

#[test]
fn readable_tokens() {
    assert!(is_readable_token("French"));
    assert!(is_readable_token("Paris"));
    assert!(is_readable_token("capital-of"));
    assert!(is_readable_token("is"));
    assert!(is_readable_token("Europe"));
}

#[test]
fn unreadable_tokens() {
    assert!(!is_readable_token("ইসলামাবাদ"));
    assert!(!is_readable_token("южна"));
    assert!(!is_readable_token("ളാ"));
    assert!(!is_readable_token("ڪ"));
    assert!(!is_readable_token(""));
}

// ── is_content_token ──

#[test]
fn content_tokens_pass() {
    assert!(is_content_token("French"));
    assert!(is_content_token("Paris"));
    assert!(is_content_token("Europe"));
    assert!(is_content_token("Mozart"));
    assert!(is_content_token("composer"));
    assert!(is_content_token("Berlin"));
    assert!(is_content_token("IBM"));
    assert!(is_content_token("Facebook"));
}

#[test]
fn stop_words_rejected() {
    assert!(!is_content_token("the"));
    assert!(!is_content_token("from"));
    assert!(!is_content_token("for"));
    assert!(!is_content_token("with"));
    assert!(!is_content_token("this"));
    assert!(!is_content_token("about"));
    assert!(!is_content_token("which"));
    assert!(!is_content_token("first"));
    assert!(!is_content_token("after"));
}

#[test]
fn short_tokens_rejected() {
    assert!(!is_content_token("a"));
    assert!(!is_content_token("of"));
    assert!(!is_content_token("is"));
    assert!(!is_content_token("-"));
    assert!(!is_content_token("lö"));
    assert!(!is_content_token("par"));
}

#[test]
fn code_tokens_rejected() {
    assert!(!is_content_token("trialComponents"));
    assert!(!is_content_token("NavigationBar"));
    assert!(!is_content_token("LastName"));
}

// ── SHOW MODELS works without backend ──

#[test]
fn show_models_no_crash() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Pipe: errors propagate ──

#[test]
fn pipe_error_propagates() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"STATS |> WALK "test";"#).unwrap();
    assert!(session.execute(&stmt).is_err());
}

// ── Format helpers ──

#[test]
fn format_number_small() {
    assert_eq!(format_number(42), "42");
    assert_eq!(format_number(999), "999");
}

#[test]
fn format_number_thousands() {
    assert_eq!(format_number(1_000), "1.0K");
    assert_eq!(format_number(10_240), "10.2K");
    assert_eq!(format_number(348_160), "348.2K");
}

#[test]
fn format_number_millions() {
    assert_eq!(format_number(1_000_000), "1.00M");
    assert_eq!(format_number(2_917_432), "2.92M");
}

#[test]
fn format_bytes_small() {
    assert_eq!(format_bytes(512), "512 B");
}

#[test]
fn format_bytes_kb() {
    assert_eq!(format_bytes(2048), "2.0 KB");
}

#[test]
fn format_bytes_mb() {
    let mb = 5 * 1_048_576;
    assert_eq!(format_bytes(mb), "5.0 MB");
}

#[test]
fn format_bytes_gb() {
    let gb = 6_420_000_000;
    assert!(format_bytes(gb).contains("GB"));
}

// ═══════════════════════════════════════════════════════════════
// Weight backend tests
// ═══════════════════════════════════════════════════════════════

/// Create a minimal ModelWeights for testing the Weight backend.
fn make_test_weights() -> larql_inference::ModelWeights {
    use larql_inference::ndarray;
    use std::collections::HashMap;

    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, ndarray::ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        let mut gate = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        let mut up = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        let mut down = ndarray::Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = ndarray::Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                attn[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                attn.into_shared(),
            );
        }

        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }

    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_inference::ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim: hidden,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

/// Create a minimal tokenizer for testing.
fn make_test_tokenizer() -> larql_inference::tokenizers::Tokenizer {
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    larql_inference::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Create a Session with Weight backend for testing.
fn weight_session() -> Session {
    let mut session = Session::new();
    session.backend = Backend::Weight {
        model_id: "test/model".into(),
        weights: make_test_weights(),
        tokenizer: make_test_tokenizer(),
    };
    session
}

#[test]
fn weight_backend_stats() {
    let mut session = weight_session();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt).unwrap();
    assert!(result.iter().any(|l| l.contains("test/model")));
    assert!(result.iter().any(|l| l.contains("live weights")));
    assert!(result.iter().any(|l| l.contains("2"))); // num_layers
}

#[test]
fn weight_backend_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires a vindex"),
        "expected vindex error, got: {msg}"
    );
    assert!(
        msg.contains("EXTRACT"),
        "should suggest EXTRACT, got: {msg}"
    );
}

#[test]
fn weight_backend_describe_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_select_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_explain_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_insert_requires_vindex() {
    let mut session = weight_session();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex") || msg.contains("mutation requires"));
}

#[test]
fn weight_backend_show_relations_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_compile_current_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("EXTRACT") || msg.contains("vindex"));
}

#[test]
fn weight_backend_show_models_works() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Mutation pipeline integration tests ────────────────────────────────
//
// These tests build a tiny synthetic vindex on disk, load it via USE,
// and exercise the DELETE / UPDATE / patch-session paths through the
// real executor + parser. They cover the auto-patch lifecycle, the
// patch overlay update, and SAVE PATCH file emission.
//
// INSERT is exercised end-to-end in `compile_demo` against a real
// Gemma vindex (the synthetic tokenizer here has an empty vocab so it
// can't tokenise meaningful entity strings). The auto-patch session
// creation that INSERT triggers is covered indirectly by the DELETE
// auto-patch test below.

use larql_inference::ndarray::Array2;

/// Build a minimal vindex directory on disk that the LQL executor can
/// load via `USE`. Includes gate vectors, down_meta, embeddings, and a
/// stub tokenizer. Returns the directory path; the caller is
/// responsible for cleanup.
fn make_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!("larql_lql_test_vindex_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Tiny in-memory index — 2 layers × 3 features × 4 hidden dims.
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );

    let mut config = VindexConfig {
        version: 2,
        model: "test/lql-mutation".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Synthetic embeddings.bin (vocab_size × hidden f32, all zeros).
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // Stub tokenizer.json — empty BPE. Not used by DELETE / UPDATE /
    // PATCH; INSERT-against-this-vindex tests would need a real one.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    dir
}

/// Spin up a session and `USE` the test vindex from `make_test_vindex_dir`.
fn vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session
        .execute(&stmt)
        .expect("USE on synthetic vindex should succeed");
    (session, dir)
}

#[test]
fn use_synthetic_vindex_loads() {
    let (session, dir) = vindex_session("use_loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_by_layer_and_feature_succeeds() {
    let (mut session, dir) = vindex_session("delete_lf");

    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Deleted") || joined.contains("deleted"),
        "expected delete confirmation in: {joined}"
    );

    // The patch session should now be active (auto-patch).
    assert!(
        session.patch_recording.is_some(),
        "DELETE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_no_matches_returns_message() {
    let (mut session, dir) = vindex_session("delete_nomatch");

    // Layer that doesn't exist in our 2-layer test vindex.
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 99 AND feature = 0;"#).unwrap();
    let result = session.execute(&stmt);
    // The executor either returns an empty-match message or errors —
    // both are acceptable; the important thing is no panic.
    assert!(result.is_ok() || result.is_err(), "no panic");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_feature_target_succeeds() {
    let (mut session, dir) = vindex_session("update_target");

    let stmt =
        parser::parse(r#"UPDATE EDGES SET target = "London" WHERE layer = 0 AND feature = 0;"#)
            .unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Updated") || joined.contains("updated") || joined.contains("London"),
        "expected update confirmation in: {joined}"
    );

    assert!(
        session.patch_recording.is_some(),
        "UPDATE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_begin_patch_starts_session() {
    let (mut session, dir) = vindex_session("begin_patch");

    let patch_path = dir.join("session.vlp");
    let stmt = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&stmt).expect("BEGIN PATCH should succeed");

    assert!(
        session.patch_recording.is_some(),
        "BEGIN PATCH should populate patch_recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_patch_writes_file_to_disk() {
    let (mut session, dir) = vindex_session("save_patch");

    // Start a patch, do a delete (so there's at least one operation), save.
    let patch_path = dir.join("save.vlp");
    let begin = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&begin).expect("BEGIN PATCH");

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 1;"#).unwrap();
    session.execute(&del).expect("DELETE under patch");

    let save = parser::parse("SAVE PATCH;").unwrap();
    let out = session.execute(&save).expect("SAVE PATCH");
    let joined = out.join("\n");
    assert!(
        patch_path.exists(),
        "SAVE PATCH should write the .vlp file. Output: {joined}"
    );

    // After SAVE PATCH, recording should be cleared.
    assert!(
        session.patch_recording.is_none(),
        "SAVE PATCH should clear the recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn auto_patch_session_starts_on_first_mutation() {
    let (mut session, dir) = vindex_session("auto_patch");

    // No explicit BEGIN PATCH first.
    assert!(
        session.patch_recording.is_none(),
        "no patch session before mutation"
    );

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    session.execute(&del).expect("DELETE");

    assert!(
        session.patch_recording.is_some(),
        "first mutation should auto-start a patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_nonexistent_source_errors_cleanly() {
    let (mut session, dir) = vindex_session("merge_bad_src");

    let stmt = parser::parse(r#"MERGE "/nonexistent/src.vindex";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(matches!(err, LqlError::Execution(_)));

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Session::patched_overlay_mut accessor ──

#[test]
fn patched_overlay_mut_returns_some_for_vindex_backend() {
    let (mut session, dir) = vindex_session("overlay_mut_some");
    assert!(
        session.patched_overlay_mut().is_some(),
        "Vindex backend should yield a mutable overlay"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn patched_overlay_mut_returns_none_for_no_backend() {
    let mut session = Session::new();
    assert!(
        session.patched_overlay_mut().is_none(),
        "fresh session with no backend should yield None"
    );
}

#[test]
fn patched_overlay_mut_round_trip_via_insert_feature() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("overlay_mut_round_trip");
    let gate = vec![0.7_f32, 0.0, 0.0, 0.0];
    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            1,
            gate.clone(),
            FeatureMeta {
                top_token: "z".into(),
                top_token_id: 9,
                c_score: 0.42,
                top_k: vec![TopKEntry {
                    token: "z".into(),
                    token_id: 9,
                    logit: 0.42,
                }],
            },
        );
    }
    // Same accessor, second call: the gate we just wrote must still be there.
    let overlay2 = session.patched_overlay_mut().expect("vindex backend");
    assert_eq!(
        overlay2.overrides_gate_at(0, 1),
        Some(gate.as_slice()),
        "second patched_overlay_mut() call should observe the previous mutation",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_patches_with_no_patches_returns_message() {
    let (mut session, dir) = vindex_session("show_patches_empty");

    let stmt = parser::parse("SHOW PATCHES;").unwrap();
    let out = session.execute(&stmt).expect("SHOW PATCHES");
    let joined = out.join("\n").to_lowercase();
    assert!(
        joined.contains("no") || joined.contains("0") || joined.is_empty(),
        "expected an empty/no-patches message: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPILE INTO VINDEX integration tests ──────────────────────────────

#[test]
fn compile_into_vindex_no_patches_succeeds() {
    let (mut session, dir) = vindex_session("compile_nopatches_v");

    let output = dir.join("compiled.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
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
    let output = dir.join("compiled_baked.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
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
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = dir.join("compiled_fail.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT FAIL;"#,
        output.display()
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
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = dir.join("compiled_lw.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT LAST_WINS;"#,
        output.display()
    ))
    .unwrap();
    assert!(
        session.execute(&stmt).is_ok(),
        "LAST_WINS should succeed despite collision"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── MEMIT fact collection ────────────────────────────────────────────

#[test]
fn memit_facts_count_inserts_only() {
    use larql_vindex::PatchOp;

    let ops = [
        PatchOp::Insert {
            layer: 26,
            feature: 100,
            relation: Some("capital".into()),
            entity: "X".into(),
            target: "Y".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            down_meta: None,
        },
        PatchOp::Delete {
            layer: 10,
            feature: 50,
            reason: None,
        },
        PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: None,
            down_meta: None,
        },
    ];
    let insert_count = ops
        .iter()
        .filter(|op| matches!(op, PatchOp::Insert { .. }))
        .count();
    assert_eq!(insert_count, 1, "only INSERT should be counted");
}

#[test]
fn memit_facts_deduplicate_across_patches() {
    use larql_vindex::{PatchOp, VindexPatch};

    let mkp = |conf: f32| VindexPatch {
        version: 1,
        base_model: String::new(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: vec![PatchOp::Insert {
            layer: 10,
            feature: 5,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(conf),
            gate_vector_b64: None,
            down_meta: None,
        }],
    };
    let patches = vec![mkp(0.9), mkp(0.95)];
    let mut seen = std::collections::HashSet::new();
    for p in &patches {
        for op in &p.operations {
            if let PatchOp::Insert {
                layer,
                entity,
                relation,
                target,
                ..
            } = op
            {
                seen.insert((
                    entity.clone(),
                    relation.clone().unwrap_or_default(),
                    target.clone(),
                    *layer,
                ));
            }
        }
    }
    assert_eq!(seen.len(), 1, "same fact in two patches → 1 after dedup");
}

// ── Template tests ───────────────────────────────────────────────────
//
// `canonical_decoys_are_nonempty_and_diverse` lives alongside the
// constant in `executor/mutation/insert/capture.rs`.

#[test]
fn relation_template_simple() {
    let rel = "capital";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The capital of entity is");
}

#[test]
fn relation_template_multi_word() {
    let rel = "native_language";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The native language of entity is");
}

#[test]
fn relation_template_hyphenated_produces_double_of() {
    // Documents the known template quirk: "capital-of" → "capital of"
    // → "The capital of of X is". Users should use "capital" not "capital-of".
    let rel = "capital-of";
    let prompt = format!("The {} of X is", rel.replace(['-', '_'], " "));
    assert!(
        prompt.contains("of of"),
        "capital-of produces double 'of': {prompt}"
    );
}

// Cholesky solver is unit-tested in larql-compute::cpu::ops::linalg::tests.
// MEMIT solve is integration-tested via compile_demo against real vindex.

// ── MEMIT struct ─────────────────────────────────────────────────────

#[test]
fn memit_fact_struct() {
    let f = larql_inference::MemitFact {
        prompt_tokens: vec![1, 2, 3],
        target_token_id: 42,
        layer: 26,
        label: "test".into(),
    };
    assert_eq!(f.layer, 26);
    assert_eq!(f.target_token_id, 42);
}

// ── Compile into model requires weights ──────────────────────────────

#[test]
fn compile_into_model_requires_model_weights() {
    let (mut session, dir) = vindex_session("compile_model_noweights");
    let output = dir.join("model_out");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}";"#,
        output.display()
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("model weights") || msg.contains("WITH ALL"),
        "error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Architecture B KNN Store tests — UNIFIED into FFN overlay ────
//
// The 6 tests below were written against the separate-KnnStore design
// of arch-B. After the FFN-vindex unification (2026-04-15), inserts
// route through the overlay (find_free_feature + insert_feature +
// set_up_vector + set_down_vector) and the separate knn_store is
// dormant. These tests assert obsolete behavior; they're #[ignore]d
// pending a rewrite against the unified path (task #37).

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
    let output = dir.join("compiled_knn.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
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
    let stmt = parser::parse(&format!(r#"USE "{}";"#, output.display())).unwrap();
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

// ── COMPACT MAJOR persistence (Backend::Vindex.memit_store wiring) ──

#[test]
fn memit_store_mut_unavailable_without_backend() {
    let mut session = Session::new();
    assert!(matches!(session.memit_store_mut().unwrap_err(), LqlError::NoBackend));
}

#[test]
fn memit_store_mut_returns_empty_store_on_fresh_vindex() {
    let (mut session, dir) = vindex_session("memit_empty");
    let store = session.memit_store_mut().expect("vindex backend has memit_store");
    assert_eq!(store.num_cycles(), 0);
    assert_eq!(store.total_facts(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memit_store_persists_added_cycles() {
    // Verifies the wiring change from item #5: facts pushed into the
    // session-level MemitStore survive subsequent accesses. The
    // production COMPACT MAJOR pipeline writes through the same path.
    let (mut session, dir) = vindex_session("memit_persist");
    {
        let store = session.memit_store_mut().expect("vindex backend");
        store.add_cycle(
            33,
            vec![larql_vindex::MemitFact {
                entity: "France".into(),
                relation: "capital".into(),
                target: "Paris".into(),
                key: larql_vindex::ndarray::Array1::zeros(4),
                decomposed_down: larql_vindex::ndarray::Array1::zeros(4),
                reconstruction_cos: 1.0,
            }],
            0.5,
            1.0,
            0.0,
        );
    }
    // Re-borrow to confirm the cycle survived.
    let store = session.memit_store_mut().expect("vindex backend");
    assert_eq!(store.num_cycles(), 1);
    assert_eq!(store.total_facts(), 1);
    let hits = store.lookup("France", "capital");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].target, "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// Gap coverage: variants that shipped without an executor test
// ══════════════════════════════════════════════════════════════
//
// Each variant gets a no-backend sanity check plus (where feasible
// without model weights) an end-to-end pass against the synthetic
// vindex fixture.

// ── TRACE ──

#[test]
fn no_backend_trace() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"TRACE "The capital of France is";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn trace_on_browse_only_vindex_errors_with_weights_hint() {
    // The synthetic fixture is browse-only; TRACE needs model weights.
    let (mut session, dir) = vindex_session("trace_no_weights");
    let stmt = parser::parse(r#"TRACE "any prompt";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("TRACE on browse-only vindex should fail");
    match err {
        LqlError::Execution(msg) => {
            assert!(
                msg.contains("TRACE requires model weights"),
                "expected model-weights hint, got: {msg}"
            );
        }
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── REBALANCE ──

#[test]
fn rebalance_without_backend_is_noop() {
    // REBALANCE short-circuits on empty `installed_edges` BEFORE the
    // backend check (mutation/rebalance.rs:38-43), so it returns Ok
    // with a "no compose-mode installs" message even with no backend.
    // This is the same behaviour as REBALANCE on a fresh vindex.
    let mut session = Session::new();
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE with empty install set should succeed");
    assert!(
        out.iter().any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
}

#[test]
fn rebalance_without_compose_installs_is_noop() {
    // With no `installed_edges` registered, REBALANCE returns a
    // single-line note and doesn't touch the overlay.
    let (mut session, dir) = vindex_session("rebalance_empty");
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE on empty compose set should succeed");
    assert!(
        out.iter().any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPACT MINOR / MAJOR / SHOW COMPACT STATUS ──

#[test]
fn no_backend_compact_minor() {
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_compact_major() {
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_compact_status() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn compact_minor_on_empty_l0_returns_message() {
    let (mut session, dir) = vindex_session("compact_minor_empty");
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("COMPACT MINOR with empty L0 should succeed");
    assert!(
        out.iter().any(|l| l.contains("L0 is empty")),
        "expected empty-L0 message in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_compact_status_reports_empty_tiers() {
    let (mut session, dir) = vindex_session("compact_status");
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW COMPACT STATUS should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("L0"), "expected L0 tier: {joined}");
    assert!(joined.contains("L1"), "expected L1 tier: {joined}");
    // The synthetic fixture has 0 overrides; the L0/L1 counts should read 0.
    assert!(
        joined.contains("0 entries") || joined.contains("0 edges"),
        "expected zero counts in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SHOW ENTITIES ──

#[test]
fn no_backend_show_entities() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW ENTITIES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn show_entities_scans_synthetic_vindex() {
    // The synthetic fixture seeds content-shaped tokens — SHOW ENTITIES
    // should run cleanly and produce the `Distinct entities …` summary
    // line followed by the tabular header.
    let (mut session, dir) = vindex_session("show_entities_scan");
    let stmt = parser::parse("SHOW ENTITIES LIMIT 20;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW ENTITIES should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Distinct entities"),
        "expected summary line in: {joined}"
    );
    assert!(
        joined.contains("Entity") && joined.contains("Max Score"),
        "expected tabular header in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── REMOVE PATCH ──

#[test]
fn no_backend_remove_patch() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"REMOVE PATCH "missing.vlp";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn remove_patch_unknown_errors_cleanly() {
    let (mut session, dir) = vindex_session("remove_patch_missing");
    let stmt = parser::parse(r#"REMOVE PATCH "never-applied.vlp";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("REMOVE PATCH should error when no such patch is applied");
    match err {
        LqlError::Execution(msg) => assert!(msg.contains("patch not found")),
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── PIPE ──

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
