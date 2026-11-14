//! MoE vindex fixtures, full sessions and output parsers.

use crate::parser;
use larql_inference::ndarray::Array2;

#[allow(unused_imports)]
use super::*;

/// Build a synthetic vindex with an MoE router so the `Backend::Vindex.router`
/// field gets populated by `RouterIndex::load` during USE. Unlocks the
/// `try_moe_describe` path in `describe/moe.rs`.
pub(super) fn make_moe_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype, VindexConfig,
        VindexLayerInfo, VindexModelConfig,
    };

    const NUM_EXPERTS: usize = 4;
    const TOP_K: usize = 2;

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_moe_test_vindex_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Reuse the make_test_weights/make_test_vindex helpers from the
    // full fixture for tensor scaffolding.
    let mut weights = larql_inference::test_utils::make_test_weights();
    {
        use larql_inference::ndarray::Array2;
        let new_vocab = weights.vocab_size + 1;
        let hidden = weights.hidden_size;
        let mut extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.embed.rows().into_iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                extended[[i, j]] = *v;
            }
        }
        for j in 0..hidden {
            extended[[weights.vocab_size, j]] = 0.01_f32 * (j as f32 + 1.0);
        }
        weights.embed = extended.into_shared();
        weights.vocab_size = new_vocab;
        let mut lm_extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.lm_head.rows().into_iter().enumerate() {
            if i >= new_vocab {
                break;
            }
            for (j, v) in row.iter().enumerate() {
                lm_extended[[i, j]] = *v;
            }
        }
        weights.lm_head = lm_extended.into_shared();
        let embed_key = weights.arch.embed_key().to_string();
        weights.tensors.insert(embed_key, weights.embed.clone());
    }
    let vindex = larql_inference::test_utils::make_test_vindex(&weights);

    let bpf = 4_usize;
    let row_bytes = weights.hidden_size * bpf;
    let layer_bytes = weights.intermediate_size * row_bytes;
    let layers: Vec<VindexLayerInfo> = (0..weights.num_layers)
        .map(|li| VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: weights.intermediate_size,
            num_experts: Some(NUM_EXPERTS),
            num_features_per_expert: Some(weights.intermediate_size / NUM_EXPERTS),
        })
        .collect();

    let model_config = VindexModelConfig {
        model_type: weights.arch.family().to_string(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: Some(MoeConfig {
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            shared_expert: false,
            shared_expert_intermediate_size: None,
            router_type: "top_k_softmax".into(),
            moe_intermediate_size: None,
            hybrid: false,
        }),
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
        model: format!("test/moe-fixture-{tag}"),
        family: weights.arch.family().to_string(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
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

    vindex.save_vindex(&dir, &mut config).unwrap();

    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, &dir, &mut build_cb).unwrap();

    // `write_model_weights` rewrites `model_config` from the arch
    // (`VindexModelConfig::from_arch`), which clobbers our manually-set
    // MoE entry — `tinymodel` is dense so the arch-derived config has
    // moe=None. Patch the file back in place so RouterIndex::load picks
    // up the fixture's MoE configuration during USE.
    {
        let index_path = dir.join("index.json");
        let mut on_disk: VindexConfig =
            serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
        if let Some(mc) = on_disk.model_config.as_mut() {
            mc.moe = Some(MoeConfig {
                num_experts: NUM_EXPERTS,
                top_k: TOP_K,
                shared_expert: false,
                shared_expert_intermediate_size: None,
                router_type: "top_k_softmax".into(),
                moe_intermediate_size: None,
                hybrid: false,
            });
        }
        std::fs::write(&index_path, serde_json::to_string_pretty(&on_disk).unwrap()).unwrap();
    }

    let embed_slice = weights.embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    let tok = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size - 1);
    tok.save(dir.join("tokenizer.json").to_str().unwrap(), false)
        .unwrap();

    // Router weights: per_layer = num_experts*hidden + num_experts.
    // Use a deterministic LCG so each layer has different scores and
    // top-k selection isn't always the same expert.
    let per_layer = NUM_EXPERTS * weights.hidden_size + NUM_EXPERTS;
    let total = per_layer * weights.num_layers;
    let mut router_bytes = Vec::with_capacity(total * bpf);
    let mut state: u64 = 0x00c0_ffee_4200_0000;
    for _ in 0..total {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (state as u32) as f32 / u32::MAX as f32 * 0.4 - 0.2;
        router_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("router_weights.bin"), router_bytes).unwrap();

    dir
}

pub(super) fn moe_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_moe_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on MoE synthetic vindex should succeed");
    (session, dir)
}

pub(super) fn full_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_full_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on full synthetic vindex should succeed");
    (session, dir)
}

//
// `canonical_decoys_are_nonempty_and_diverse` lives alongside the
// constant in `executor/mutation/insert/capture.rs`.

// Cholesky solver is unit-tested in larql-compute::cpu::ops::linalg::tests.
// MEMIT solve is integration-tested via compile_demo against real vindex.

//
// The 6 tests below were written against the separate-KnnStore design
// of arch-B. After the FFN-vindex unification (2026-04-15), inserts
// route through the overlay (find_free_feature + insert_feature +
// set_up_vector + set_down_vector) and the separate knn_store is
// dormant. These tests assert obsolete behavior; they're #[ignore]d
// pending a rewrite against the unified path (task #37).

//
// These tests verify that the format-agnostic abstraction routes correctly
// without branching on `config.quant` in callers.

// Gap coverage: variants that shipped without an executor test
//
// Each variant gets a no-backend sanity check plus (where feasible
// without model weights) an end-to-end pass against the synthetic
// vindex fixture.

/// Returns the integer count parsed from a "Deleted N features ..." or
/// "Updated N features ..." confirmation line. Used by feature-only
/// regression tests to assert exact match counts rather than "any match".
pub(super) fn parse_mutation_count(out: &[String]) -> usize {
    let joined = out.join("\n");
    for line in joined.lines() {
        for word in line.split_ascii_whitespace() {
            if let Ok(n) = word.parse::<usize>() {
                return n;
            }
        }
    }
    panic!("no integer count found in mutation output: {joined}");
}

// Fixture-driven SELECT / DESCRIBE / WALK tests
//
// These exercise the full verb pipeline (filter extraction → scan
// → render) against the synthetic / rich fixtures. They primarily
// drive coverage on the verb modules under `executor/query/`.

// SELECT verbs against the synthetic fixture mostly drive the verb
// pipeline (filter extraction → scan → render). The fixture's
// `feature_meta` may surface as None after the disk round-trip, so we
// don't pin specific tokens — the assertions check that the verb
// runs end-to-end and emits the structural elements (header / dashes /
// "no match" line) we know are unconditional.

/// Build a `Session` with `Backend::Weight` populated from the
/// synthetic test fixtures — no on-disk model required. This unlocks
/// the dense INFER / EXPLAIN INFER paths that short-circuit before
/// the vindex branch.
pub(super) fn weight_backend_session(model_id: &str) -> Session {
    use larql_inference::ndarray::Array2;

    let mut weights = larql_inference::test_utils::make_test_weights();
    // make_test_weights produces vocab_size=32 with embed [32, 16]; the
    // companion tokenizer puts [UNK] at id 32 — out of bounds. Extend
    // embed + lm_head by one row so any UNK token resolves to a valid
    // embedding (mirrors the trick in make_full_test_vindex_dir).
    let new_vocab = weights.vocab_size + 1;
    let hidden = weights.hidden_size;
    let mut extended = Array2::<f32>::zeros((new_vocab, hidden));
    for (i, row) in weights.embed.rows().into_iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            extended[[i, j]] = *v;
        }
    }
    for j in 0..hidden {
        extended[[weights.vocab_size, j]] = 0.01_f32 * (j as f32 + 1.0);
    }
    weights.embed = extended.into_shared();
    let mut lm_extended = Array2::<f32>::zeros((new_vocab, hidden));
    for (i, row) in weights.lm_head.rows().into_iter().enumerate() {
        if i >= new_vocab {
            break;
        }
        for (j, v) in row.iter().enumerate() {
            lm_extended[[i, j]] = *v;
        }
    }
    weights.lm_head = lm_extended.into_shared();
    weights.vocab_size = new_vocab;
    let embed_key = weights.arch.embed_key().to_string();
    weights.tensors.insert(embed_key, weights.embed.clone());

    // Match the embed by passing vocab_size-1 to the tokenizer so [UNK]
    // lands at id new_vocab-1 (= old vocab_size), inside the extended
    // embed table.
    let tokenizer = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size - 1);
    let mut session = Session::new();
    session.backend = Backend::Weight {
        model_id: model_id.into(),
        weights,
        tokenizer,
    };
    session
}

/// Build a vindex matching `make_test_vindex_dir` but with three
/// deliberate divergences vs. the base fixture so DIFF surfaces
/// real edits:
///   L0F0: Paris → Madrid           (modified)
///   L0F1: French → None            (removed)
///   L1F1: None   → Rome            (added)
pub(super) fn make_modified_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_modified_test_vindex_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

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
        // L0F0: Paris → Madrid (token modified, c_score also moved)
        Some(make_meta("Madrid", 110, 0.92)),
        // L0F1: French → None (removed)
        None,
        // L0F2: Europe unchanged
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        // L1F0: Berlin unchanged
        Some(make_meta("Berlin", 200, 0.90)),
        // L1F1: None → Rome (added)
        Some(make_meta("Rome", 211, 0.85)),
        // L1F2: Spain unchanged
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

    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();
    dir
}

/// Build a `VindexPatch` containing a single Insert op. Used to
/// pre-seat committed patches before invoking COMPACT MAJOR.
pub(super) fn mk_insert_patch(
    layer: usize,
    feature: usize,
    entity: &str,
    relation: Option<&str>,
    target: &str,
) -> larql_vindex::VindexPatch {
    larql_vindex::VindexPatch {
        version: 1,
        base_model: String::new(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: vec![larql_vindex::PatchOp::Insert {
            layer,
            feature,
            relation: relation.map(str::to_string),
            entity: entity.into(),
            target: target.into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }],
    }
}
