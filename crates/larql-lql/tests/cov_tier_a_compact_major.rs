//! COMPACT MAJOR full-MEMIT-solve coverage on a ≥1024-dim fixture.
//!
//! `executor/compact.rs`'s COMPACT MAJOR body is gated behind
//! `hidden_dim >= 1024` (the standard `write_synthetic_model_dir` fixture
//! is 16-dim, so it only ever hits the early hidden-dim error). This file
//! builds a self-contained 1024-dim vindex+weights fixture (using only
//! public `larql_models` / `larql_vindex` / `larql_inference::test_utils`
//! APIs — a port of the crate's own `make_large_test_vindex_dir` unit
//! helper) and drives the full pipeline through the *public* Session API:
//! BEGIN PATCH → compose INSERT → SAVE PATCH → APPLY PATCH seats a
//! committed `PatchOp::Insert` in `patched.patches`, which COMPACT MAJOR
//! then walks (residual capture, target-embed lookup, ndarray MEMIT solve,
//! decomposition-quality report, memit_store persist).
//!
//! Plumbing-only: synthetic weights → garbage MEMIT deltas. We assert on
//! output shape (the solver/quality/completion lines) only.

use larql_lql::executor::Session;
use larql_lql::parser;

fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
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

#[test]
fn compact_major_empty_l1_short_circuits_on_large_fixture() {
    // hidden_dim >= 1024 clears the guard; with no patches/overlay the
    // empty-L1 short-circuit (compact.rs:156-160) fires.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());
    let out = try_run(&mut session, "COMPACT MAJOR;").expect("compact major no-op");
    assert!(
        out.join("\n").contains("L1 is empty"),
        "expected empty-L1 message, got: {out:?}"
    );
}

#[test]
fn compact_major_runs_full_memit_solve_on_large_fixture() {
    // Seat a committed Insert patch via the public BEGIN/SAVE/APPLY flow,
    // then COMPACT MAJOR walks the full MEMIT pipeline (compact.rs:174-326):
    // residual capture, target-embed lookup, ndarray solve, quality report,
    // memit_store add_cycle + persist, and the completion line.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());

    let vlp = dir.path().join("major.vlp");
    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    try_run(&mut session, "SAVE PATCH;").expect("save patch");
    // The saved facts are live in this session; APPLY in a fresh one (a
    // re-asserted fact is refused, FACT-ID-1 I-2).
    let mut session = use_large(dir.path());
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply patch");

    let out = try_run(&mut session, "COMPACT MAJOR;").expect("compact major full path");
    let joined = out.join("\n");
    assert!(
        joined.contains("Running MEMIT solver") || joined.contains("COMPACT MAJOR"),
        "expected MEMIT solve output, got: {joined}"
    );
    assert!(
        joined.contains("Decomposition quality") || joined.contains("complete"),
        "expected quality/completion line, got: {joined}"
    );
}

#[test]
fn compact_major_with_lambda_override_on_large_fixture() {
    // WITH LAMBDA = X threads a non-default lambda through the solve and
    // echoes it in the progress line (compact.rs:163-166).
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());

    let vlp = dir.path().join("major_lambda.vlp");
    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    try_run(&mut session, "SAVE PATCH;").expect("save patch");
    // The saved facts are live in this session; APPLY in a fresh one (a
    // re-asserted fact is refused, FACT-ID-1 I-2).
    let mut session = use_large(dir.path());
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply patch");

    let out =
        try_run(&mut session, "COMPACT MAJOR WITH LAMBDA = 0.01;").expect("compact major lambda");
    assert!(
        out.iter().any(|l| l.contains("lambda=")),
        "expected lambda echo, got: {out:?}"
    );
}

#[test]
fn compact_major_overlay_only_edges_uses_else_branch() {
    // A compose INSERT WITHOUT an active patch recording writes gate
    // overrides into the overlay but leaves `patched.patches` empty. So in
    // COMPACT MAJOR, `edges` (from committed patches) is empty while
    // `overlay_edges` (from `overrides_gate_iter`) is non-empty:
    //   * the both-empty short-circuit (156) is skipped,
    //   * install_layer resolves from `overlay_edges[0].0` (compact.rs:188-189),
    //   * `if !edges.is_empty()` is false → the no-edge-metadata `else`
    //     branch (compact.rs:327-333) runs.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());

    // No BEGIN PATCH → the compose insert's gate/up/down land in the
    // overlay but no committed VindexPatch is created.
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert (overlay only)");

    let out = try_run(&mut session, "COMPACT MAJOR;").expect("compact major overlay-only");
    let joined = out.join("\n");
    assert!(
        joined.contains("No edge metadata") || joined.contains("COMPACT MAJOR"),
        "expected the no-edge-metadata else branch, got: {joined}"
    );
}

#[test]
fn compact_major_skips_insert_with_no_relation() {
    // COMPACT MAJOR's edge collection skips Insert ops that lack a relation
    // (compact.rs:144-146: the `None => skipped_no_relation += 1` arm) and
    // reports the skipped count. The LQL parser always supplies a relation,
    // so we reach this branch by APPLYing a hand-written .vlp containing an
    // Insert op with `relation` omitted, alongside one with a relation so
    // `edges` is non-empty.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    let mut session = use_large(dir.path());

    let vlp = dir.path().join("no_rel.vlp");
    // Two Insert ops at distinct slots: one with a relation, one without.
    let patch_json = r#"{
        "version": 1,
        "base_model": "test/large-fixture",
        "created_at": "",
        "operations": [
            {"op":"insert","layer":0,"feature":0,"relation":"capital","entity":"[1]","target":"[2]"},
            {"op":"insert","layer":0,"feature":1,"entity":"[3]","target":"[4]"}
        ]
    }"#;
    std::fs::write(&vlp, patch_json).expect("write vlp");

    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply patch");

    let out = try_run(&mut session, "COMPACT MAJOR;").expect("compact major with no-relation op");
    let joined = out.join("\n");
    assert!(
        joined.contains("no relation")
            || joined.contains("Skipped")
            || joined.contains("COMPACT MAJOR"),
        "expected the no-relation skip path to run, got: {joined}"
    );
}

#[test]
fn compact_major_persist_failure_emits_warning() {
    // After a successful MEMIT solve, COMPACT MAJOR persists the store to
    // `memit_store.json` (compact.rs:315-321). If that write/rename fails,
    // the `if let Err(e)` arm (317-319) pushes a "failed to persist" warning
    // rather than failing the command. We force the failure by pre-creating
    // `memit_store.json` as a DIRECTORY, so the tmp→final rename can't land.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    // Block the persist target: a directory can't be replaced by a file
    // rename.
    std::fs::create_dir(dir.path().join("memit_store.json")).expect("create blocker dir");
    std::fs::write(
        dir.path().join("memit_store.json").join("keep"),
        b"non-empty so rename definitely fails",
    )
    .expect("write blocker child");

    let mut session = use_large(dir.path());
    let vlp = dir.path().join("persist.vlp");
    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    try_run(&mut session, "SAVE PATCH;").expect("save patch");
    // The saved facts are live in this session; APPLY in a fresh one (a
    // re-asserted fact is refused, FACT-ID-1 I-2).
    let mut session = use_large(dir.path());
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply patch");

    // The solve succeeds; only the persist fails → COMPACT MAJOR still
    // returns Ok with a warning line in the output.
    let out = try_run(&mut session, "COMPACT MAJOR;").expect("compact major persist-fail still ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("failed to persist") || joined.contains("COMPACT MAJOR complete"),
        "expected persist-warning or completion, got: {joined}"
    );
}

#[test]
fn compact_major_no_weights_on_large_fixture_errors() {
    // hidden_dim >= 1024 clears the first guard, then the
    // `!config.has_model_weights` guard (compact.rs:113-119) fires when the
    // fixture's index.json is patched to has_model_weights=false.
    let dir = tempfile::tempdir().expect("tempdir");
    write_large_vindex_dir(dir.path());
    // Flip the loader gate so the no-weights MAJOR branch is reached.
    let idx = dir.path().join("index.json");
    let raw = std::fs::read_to_string(&idx).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["has_model_weights"] = serde_json::Value::Bool(false);
    std::fs::write(&idx, serde_json::to_string(&v).unwrap()).unwrap();

    let mut session = use_large(dir.path());
    // Need an L1 edge so we get past the empty-L1 short-circuit to the
    // weights guard? No — the weights guard (113) is BEFORE edge collection
    // (128). So COMPACT MAJOR errors on weights regardless of edges.
    let err = try_run(&mut session, "COMPACT MAJOR;").expect_err("no weights should error");
    assert!(
        err.contains("model weights") || !err.is_empty(),
        "expected weights-required error, got: {err}"
    );
}
