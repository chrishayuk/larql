//! Rich, full and large vindex fixtures and sessions.

use crate::parser;

#[allow(unused_imports)]
use super::*;

/// Spin up a session and `USE` the rich test vindex.
pub(super) fn rich_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_rich_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on rich synthetic vindex should succeed");
    (session, dir)
}

//
// `make_full_test_vindex_dir` produces a vindex with `has_model_weights
// = true`, populated via `larql_inference::test_utils::make_test_weights`
// (TinyModelArch, 2 layers × 16 hidden × 32 intermediate × vocab 32).
// `larql_vindex::write_model_weights` writes the full attention + FFN
// + lm_head + norm weight files into the vindex directory, so
// `load_model_weights` succeeds and downstream INFER / TRACE / EXPLAIN
// INFER / COMPACT MAJOR / REBALANCE-with-installs / INSERT-compose all
// have real (random) weights to forward through.
//
// The same WordLevel tokenizer that `make_test_tokenizer(32)` produces
// is written to the vindex so prompts tokenise to ids 0..31.

pub(super) fn make_full_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype, VindexConfig,
        VindexLayerInfo, VindexModelConfig,
    };

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_full_test_vindex_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // `make_test_weights()` produces vocab_size = 32 with embed shape
    // [32, 16]. The companion `make_test_tokenizer(32)` adds `[UNK]`
    // at id 32 — out of the embed's bounds. Extend the embed by one
    // row so any UNK-tagged token still resolves to a valid embedding.
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
        // UNK row gets a small constant so it embeds to something
        // distinguishable from id 0.
        for j in 0..hidden {
            extended[[weights.vocab_size, j]] = 0.01_f32 * (j as f32 + 1.0);
        }
        weights.embed = extended.into_shared();
        weights.vocab_size = new_vocab;
        // Mirror into the lm_head if it shared the original embed.
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
        // Update embed_key tensor too so manifest stays consistent.
        let embed_key = weights.arch.embed_key().to_string();
        weights.tensors.insert(embed_key, weights.embed.clone());
    }
    let vindex = larql_inference::test_utils::make_test_vindex(&weights);

    // Gate offsets: each layer's gate matrix is `intermediate × hidden`
    // floats. Lay them out contiguously starting at 0.
    let bpf = 4_usize; // f32
    let row_bytes = weights.hidden_size * bpf;
    let layer_bytes = weights.intermediate_size * row_bytes;
    let mut layers: Vec<VindexLayerInfo> = Vec::new();
    for li in 0..weights.num_layers {
        layers.push(VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: weights.intermediate_size,
            num_experts: None,
            num_features_per_expert: None,
        });
    }

    let model_config = VindexModelConfig {
        model_type: weights.arch.family().to_string(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
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
        model: format!("test/full-fixture-{tag}"),
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

    // 1. Save the index (gate vectors + down_meta + config).
    vindex.save_vindex(&dir, &mut config).unwrap();

    // 2. Write the model weight files (attn / up / down / norms / lm_head).
    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, &dir, &mut build_cb).unwrap();

    // 3. Write embeddings.bin from `weights.embed`.
    let embed_slice = weights.embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // 4. Write a tokenizer.json that maps token-id N to "[N]" string.
    //    Pass `vocab_size - 1` so the tokenizer's `[UNK]` lands at id
    //    `vocab_size - 1` (we extended the embed by one above), keeping
    //    every produced id inside the embed table.
    let tok = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size - 1);
    tok.save(dir.join("tokenizer.json").to_str().unwrap(), false)
        .unwrap();

    dir
}
