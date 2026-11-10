use super::helpers::*;
use super::*;
use crate::parser;

/// Render a filesystem path for safe inclusion inside an LQL quoted string
/// literal. The LQL lexer interprets `\` as an escape introducer, so a raw
/// Windows path like `C:\Users\runner\...` ends up decoded as
/// `C:UsersRUNNER...`. Doubling each backslash leaves the path untouched
/// after the lexer's escape pass on every platform.
fn lql_path(path: impl AsRef<std::path::Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

/// RAII guard that forces `LARQL_MEMIT_ENABLE=1` on the current thread via the
/// `larql_compute::options` thread-local override and clears it on drop.
/// Replaces `std::env::set_var`/`remove_var`, which mutate process-global env
/// and race the concurrent `getenv` other parallel tests drive on the decode
/// path → SIGSEGV in libc. The COMPILE paths read this flag through
/// `larql_compute::options::env_value`, so the override is honoured.
struct MemitEnableGuard;

impl MemitEnableGuard {
    fn on() -> Self {
        larql_compute::options::set_env_override("LARQL_MEMIT_ENABLE", Some("1"));
        MemitEnableGuard
    }
}

impl Drop for MemitEnableGuard {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
    }
}

// Weight backend tests

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
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
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

mod moe_fixtures;
mod vindex_fixtures;
use moe_fixtures::*;
use vindex_fixtures::*;

/// A fresh COMPILE destination *beside* the test vindex at `source`.
///
/// COMPILE refuses an output inside its source and one that already
/// holds anything, so the destination is a sibling, and a leftover from
/// an earlier run of the same test (the names are deterministic) is
/// cleared here — the test owns this path.
fn compile_output(source: &std::path::Path, name: &str) -> std::path::PathBuf {
    let stem = source
        .file_name()
        .expect("test vindex dir has a name")
        .to_string_lossy();
    let output = source.with_file_name(format!("{stem}.{name}"));
    let _ = std::fs::remove_dir_all(&output);
    output
}

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
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
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
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on synthetic vindex should succeed");
    (session, dir)
}

//
// `make_test_vindex_dir` ships a stub BPE that returns no token ids
// for any input, which short-circuits every entity-anchored verb
// (SELECT WHERE entity, NEAREST TO, DESCRIBE, walk-based EDGES).
// The "rich" fixture below upgrades two pieces:
//
//   1. Tokenizer is a `WordLevel` model with a vocab that covers a
//      handful of named entities and template words. Encoding
//      "Paris" returns `[1]`, "France" returns `[2]`, etc.
//
//   2. Embeddings are non-zero, distinguishable rows so the entity
//      query vector built by `entity_query_vec` is non-trivial and
//      walks against it produce non-zero gate scores.
//
// Same shape as the basic fixture (2 layers × 3 features × 4 hidden)
// — only the tokenizer + embedding payload differ.

const RICH_FIXTURE_VOCAB: &[(&str, u32)] = &[
    ("[UNK]", 0),
    ("Paris", 1),
    ("France", 2),
    ("Berlin", 3),
    ("Germany", 4),
    ("Spain", 5),
    ("Madrid", 6),
    ("London", 7),
    ("English", 8),
    ("the", 9),
    ("of", 10),
    ("is", 11),
    ("capital", 12),
    ("language", 13),
    ("currency", 14),
    ("Atlantis", 15),
];

fn rich_fixture_tokenizer_json() -> String {
    let vocab_json: String = RICH_FIXTURE_VOCAB
        .iter()
        .map(|(t, id)| format!(r#""{t}":{id}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab_json}}},"unk_token":"[UNK]"}}}}"#
    )
}

/// Build a test vindex with a real WordLevel tokenizer + non-zero
/// embeddings so entity-anchored verbs produce real outputs.
fn make_rich_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!("larql_lql_rich_test_vindex_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = RICH_FIXTURE_VOCAB.len();

    // Gate vectors aligned with the embedding rows. Magnitudes are
    // chosen so the dot-product walk produces gate scores well above
    // `DESCRIBE_GATE_THRESHOLD = 5.0`, which lets DESCRIBE / EXPLAIN
    // surface real edges from the synthetic vindex.
    const RICH_GATE_MAG: f32 = 50.0;
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = RICH_GATE_MAG;
    gate0[[1, 1]] = RICH_GATE_MAG;
    gate0[[2, 2]] = RICH_GATE_MAG;
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = RICH_GATE_MAG;
    gate1[[1, 0]] = RICH_GATE_MAG * 0.5;
    gate1[[2, 2]] = -RICH_GATE_MAG;

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
        Some(make_meta("Paris", 1, 0.95)),
        Some(make_meta("Berlin", 3, 0.88)),
        Some(make_meta("language", 13, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Madrid", 6, 0.90)),
        None,
        Some(make_meta("English", 8, 0.70)),
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
        model: "test/rich-fixture".into(),
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
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Non-zero embeddings: row[i] = unit vector pointing at axis i % hidden.
    // Distinguishable per-token-id, deterministic, and non-trivial when
    // averaged.
    let mut embed_bytes = Vec::with_capacity(vocab_size * hidden * 4);
    for tid in 0..vocab_size {
        for d in 0..hidden {
            let v = if d == tid % hidden { 1.0f32 } else { 0.1f32 };
            embed_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    std::fs::write(dir.join("tokenizer.json"), rich_fixture_tokenizer_json()).unwrap();

    // Probe-only relation classifier: each (layer, feature) maps to a
    // relation label. No clusters are needed for label resolution; the
    // classifier returns probe-confirmed labels first.
    let feature_labels = serde_json::json!({
        "L0_F0": "capital",
        "L0_F1": "capital",
        "L0_F2": "language",
        "L1_F0": "capital",
        "L1_F2": "language",
    });
    std::fs::write(
        dir.join("feature_labels.json"),
        serde_json::to_string(&feature_labels).unwrap(),
    )
    .unwrap();

    dir
}

mod architecture_b_knn_store_tests_unified_i;
mod compact_major_persistence_backend_vindex;
mod compile_into_model;
mod compile_into_vindex_integration_tests;
mod describe_on_moe_router_fixture_try_moe_d;
mod describe_rich_fixture;
mod diff_into_patch;
mod full_fixture_real_modelweights_safetenso;
mod full_fixture_tests_real_modelweights_on;
mod infer_trace_explain_infer_mode_variants;
mod inferenceweights_format_dispatch;
mod memit_fact_collection;
mod pipe;
mod session_patched_overlay_mut_accessor;
mod session_state_no_backend;
mod show_stats_verbs;
mod weight_backend_tests;
