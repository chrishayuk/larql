//! InferenceWeights format dispatch

use super::*;

#[test]
fn knn_insert_q4k_flagged_no_weights_uses_embedding_fallback() {
    // A vindex with quant=Q4K but has_model_weights=false must still use the
    // embedding-key fallback path (not the InferenceWeights path). The quant
    // flag should be irrelevant when there are no weights to load.
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_test_q4k_embed_fallback_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

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
    let gate0 = ndarray::Array2::<f32>::zeros((num_features, hidden));
    let gate1 = ndarray::Array2::<f32>::zeros((num_features, hidden));
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
    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/q4k-no-weights".into(),
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
        quant: larql_vindex::QuantFormat::Q4K, // quantised flag…
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false, // …but no weights on disk
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session.execute(&stmt).expect("USE");

    // INSERT must succeed via the embedding-key fallback — not attempt to load q4k weights.
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    let out = session
        .execute(&stmt)
        .expect("INSERT should use embedding fallback on q4k+no-weights");
    let joined = out.join("\n");
    assert!(
        joined.contains("KNN store"),
        "expected KNN store mode: {joined}"
    );
    assert!(
        joined.contains("embedding key"),
        "expected embedding-key mode (no weights): {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_on_q4k_vindex_returns_clear_error() {
    // TRACE should return a helpful error on q4k vindexes rather than the
    // cryptic "load_model_weights only handles float weights" message.
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_test_q4k_trace_error_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 2;
    let num_layers = 2;
    let vocab_size = 10;

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
    let gate0 = ndarray::Array2::<f32>::zeros((num_features, hidden));
    let meta0 = vec![
        Some(make_meta("test", 1, 0.5)),
        Some(make_meta("foo", 2, 0.3)),
    ];
    let index = VectorIndex::new(
        vec![Some(gate0.clone()), Some(gate0)],
        vec![Some(meta0.clone()), Some(meta0)],
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/q4k-trace".into(),
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
        quant: larql_vindex::QuantFormat::Q4K,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session.execute(&stmt).expect("USE");

    let stmt = parser::parse(r#"TRACE "hello world";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("T2") || msg.contains("q4k") || msg.contains("quantised"),
        "expected clear q4k error, got: {msg}"
    );
    assert!(
        !msg.contains("only handles float"),
        "must not expose internal loader error: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
