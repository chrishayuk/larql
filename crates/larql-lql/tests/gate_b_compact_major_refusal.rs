//! Gate B, B6 ruling D2 (`docs/gate-b-b6.md` r3 §3.2): COMPACT MAJOR refuses
//! before any work, on every binding.
//!
//! The fixture is the one on which COMPACT MAJOR used to run its whole MEMIT
//! pipeline: a 1024-dim vindex with model weights and an applied compose
//! patch. Before B6 the statement added a cycle to the session's MEMIT store,
//! wrote `memit_store.json` into the bound vindex, advanced the epoch, and
//! recorded nothing a replay could reproduce. Now it refuses with a fixed
//! text, and all four parts of the observable state are unchanged: the
//! session (its fingerprint and asserted facts), the bound directory's files,
//! and no output anywhere.

use std::collections::BTreeMap;
use std::path::Path;

use larql_lql::executor::Session;
use larql_lql::parser;

/// The full refusal, pinned.
const REFUSAL: &str = "Execution error: COMPACT MAJOR is refused: a MEMIT cycle has no ledger \
     representation, so its result could not be replayed or reconstructed from a patch. Restoring \
     this statement needs a ledger-backed MEMIT operation under its own contract. Nothing was \
     changed.";

/// Every form of the statement.
const FORMS: [&str; 4] = [
    "COMPACT MAJOR;",
    "COMPACT MAJOR FULL;",
    "COMPACT MAJOR WITH LAMBDA = 0.01;",
    "COMPACT MAJOR FULL WITH LAMBDA = 0.01;",
];

const COMPOSE: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#;
const KNN: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "capital", "[4]") AT LAYER 0 MODE KNN;"#;

fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, sql: &str) -> Vec<String> {
    try_run(session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// Build a 1024-hidden vindex+weights dir that clears the COMPACT MAJOR
/// hidden-dim guard. Intermediate size is kept tiny so the gate/up/down
/// slabs stay small. Uses the on-disk null-pre_tokenizer tokenizer so
/// bracketed prompts (`[1]`) encode to a single in-vocab id.
fn write_large_vindex_dir(dir: &std::path::Path) {
    use larql_inference::ndarray::Array2;
    use larql_inference::test_utils::synthetic_tokenizer_json;
    use larql_models::{detect_from_json, ModelWeights, WeightArray};
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype, VindexConfig,
        VindexLayerInfo, VindexModelConfig,
    };
    use std::collections::HashMap;

    const VOCAB: usize = 32;
    const HIDDEN: usize = 1024;
    const INTER: usize = 64;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 64;
    const NUM_LAYERS: usize = 2;

    std::fs::create_dir_all(dir).unwrap();

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
    });
    let arch = detect_from_json(&arch_json);
    let arch_family = arch.family().to_string();

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0x600d_face_u64;
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    let new_vocab = VOCAB + 1;
    let mut embed_arr = Array2::<f32>::zeros((new_vocab, HIDDEN));
    let base_embed = rand_mat(VOCAB, HIDDEN, 0.05);
    for (i, row) in base_embed.rows().into_iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            embed_arr[[i, j]] = *v;
        }
    }
    for j in 0..HIDDEN {
        embed_arr[[VOCAB, j]] = 0.005_f32 * ((j % 13) as f32 + 1.0);
    }
    let embed = embed_arr.into_shared();
    let lm_head = rand_mat(new_vocab, HIDDEN, 0.05);
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;
    for layer in 0..NUM_LAYERS {
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.05));
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.05));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.05));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.05));
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    let weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: embed.clone(),
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: new_vocab,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    };

    let n_features = INTER;
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..NUM_LAYERS)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * HIDDEN)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, HIDDEN), data).unwrap())
        })
        .collect();
    let down_meta = vec![None; NUM_LAYERS];
    let vindex = larql_vindex::VectorIndex::new(gate_vectors, down_meta, NUM_LAYERS, HIDDEN);

    let bpf = 4_usize;
    let row_bytes = HIDDEN * bpf;
    let layer_bytes = INTER * row_bytes;
    let layers: Vec<VindexLayerInfo> = (0..NUM_LAYERS)
        .map(|li| VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: INTER,
            num_experts: None,
            num_features_per_expert: None,
        })
        .collect();

    let model_config = VindexModelConfig {
        model_type: arch_family.clone(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
        ..Default::default()
    };

    let mut config = VindexConfig {
        version: 2,
        model: "test/large-fixture".into(),
        family: arch_family,
        source: None,
        checksums: None,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: new_vocab,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: layers.clone(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    vindex.save_vindex(dir, &mut config).unwrap();

    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, dir, &mut build_cb).unwrap();

    let embed_slice = embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // Null-pre_tokenizer tokenizer (vocab "[0]".."[VOCAB-1]") so bracketed
    // prompts encode to a single in-vocab id during the MEMIT residual
    // capture pass.
    std::fs::write(
        dir.join("tokenizer.json"),
        synthetic_tokenizer_json(new_vocab),
    )
    .unwrap();
}

fn use_large(dir: &std::path::Path) -> Session {
    let mut session = Session::new();
    try_run(&mut session, &format!(r#"USE "{}";"#, sql_path(dir))).expect("USE large fixture");
    session
}

/// Every file under `dir`, by relative path, with its bytes.
fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(dir).unwrap().display().to_string();
                out.insert(rel, std::fs::read(&path).unwrap());
            }
        }
    }
    out
}

/// What a refused statement must leave alone: the session's fingerprint
/// (overlay, journal, recording, counters), its asserted facts, its MEMIT
/// store as SHOW COMPACT STATUS reports it, and every file of the bound directory.
#[derive(PartialEq, Debug)]
struct Observed {
    fingerprint: String,
    facts: String,
    compact_status: Vec<String>,
    files: BTreeMap<String, Vec<u8>>,
}

fn observe(session: &mut Session, dir: &Path) -> Observed {
    Observed {
        fingerprint: format!("{:?}", session.state_fingerprint().components()),
        facts: format!("{:?}", session.asserted_facts().ok()),
        compact_status: run(session, "SHOW COMPACT STATUS;"),
        files: files(dir),
    }
}

fn assert_every_form_refuses_without_change(session: &mut Session, dir: &Path) {
    for form in FORMS {
        let before = observe(session, dir);
        let err = try_run(session, form).expect_err(form);
        assert_eq!(err, REFUSAL, "{form}");
        assert!(
            observe(session, dir) == before,
            "{form}: the observable state changed"
        );
        assert!(
            !dir.join("memit_store.json").exists(),
            "{form}: wrote a MEMIT store"
        );
    }
}

/// The case COMPACT MAJOR used to solve: weights, hidden_dim 1024, an
/// applied compose patch, a session KNN fact and an open recording.
#[test]
fn compact_major_refuses_where_it_used_to_solve() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let patches = tempfile::tempdir().expect("tempdir");
    let vlp = patches.path().join("major.vlp");
    let mut session = use_large(dir.path());
    run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    );
    run(&mut session, COMPOSE);
    run(&mut session, "SAVE PATCH;");

    let mut session = use_large(dir.path());
    run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    );
    run(
        &mut session,
        &format!(
            r#"BEGIN PATCH "{}";"#,
            sql_path(&patches.path().join("open.vlp"))
        ),
    );
    run(&mut session, KNN);
    assert_every_form_refuses_without_change(&mut session, dir.path());
}

/// With nothing to compact it still refuses: the refusal comes before any
/// look at the session.
#[test]
fn compact_major_refuses_on_an_empty_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());
    assert_every_form_refuses_without_change(&mut session, dir.path());
}

/// A vindex without weights, and an unbound session, get the same refusal:
/// none of the old preconditions is checked first.
#[test]
fn compact_major_refuses_before_any_precondition() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let idx = dir.path().join("index.json");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&idx).unwrap()).unwrap();
    v["has_model_weights"] = serde_json::Value::Bool(false);
    std::fs::write(&idx, serde_json::to_string(&v).unwrap()).unwrap();
    let mut session = use_large(dir.path());
    assert_every_form_refuses_without_change(&mut session, dir.path());

    let mut unbound = Session::new();
    for form in FORMS {
        assert_eq!(try_run(&mut unbound, form).unwrap_err(), REFUSAL, "{form}");
    }
}
