//! Synthetic on-disk model directories.
//!
//! Split out of `test_utils.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::WeightArray;
use ndarray::Array2;

/// Serialise the synthetic `make_test_weights()` model + matching
/// vindex + tokenizer to an on-disk directory that any code path
/// reaching for `larql_vindex::load_vindex_config` /
/// `load_model_weights` will accept.
///
/// Replaces the previous "set `LARQL_MODEL` to a real Gemma snapshot"
/// pattern: tests can call this with a `tempfile::TempDir` and exercise
/// the full disk-loading pipeline without depending on multi-gigabyte
/// model artifacts in `~/.cache`.
///
/// The fixture is **synthetic**: the weights produce garbage logits.
/// Tests asserting plumbing (correct files written, correct error on
/// missing config, correct dispatch on backend type, etc.) work fine;
/// tests asserting semantic content ("model predicts Paris") still
/// need a real model and don't belong in `tests/`.
///
/// Layout written:
/// ```text
/// dir/
///   index.json              -- VindexConfig with has_model_weights=true
///   tokenizer.json          -- WordLevel "[0]".."[VOCAB-1]" tokenizer
///   embeddings.bin          -- VOCAB × HIDDEN f32 (from weights.embed)
///   weight_manifest.json    -- per-tensor offset/length manifest
///   attn_weights.bin        -- per-layer Q/K/V/O + norms
///   up_weights.bin          -- per-layer gate + up
///   down_weights.bin        -- per-layer down
///   norms.bin               -- final norm
///   lm_head.bin             -- output projection
///   gate_vectors.bin        -- vindex gate matrices (from make_test_vindex)
///   down_meta.bin           -- vindex down metadata (empty per layer)
/// ```
pub fn write_synthetic_model_dir(dir: &std::path::Path) -> Result<(), String> {
    use larql_vindex::{
        write_model_weights, ExtractLevel, MoeConfig, StorageDtype, VindexConfig, VindexModelConfig,
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {e}"))?;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_vindex(&weights);

    // ── tokenizer.json ────────────────────────────────────────────────
    // Write a tokenizer that encodes `[N]` to id N *as a single token*
    // — `make_test_tokenizer`'s Whitespace pre-tokenizer would split
    // `[1]` into `[`, `1`, `]`, all of which UNK, blowing up the
    // embedding lookup with id=vocab_size. The on-disk fixture uses a
    // pre-tokenizer-free variant so test prompts like `EXPLAIN INFER
    // "[1]"` lookup directly. `tokenizer` is kept above for any caller
    // that needs the in-memory shape.
    let _ = &tokenizer; // returned by make_test_tokenizer; not the on-disk shape
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, synthetic_tokenizer_json(weights.vocab_size))
        .map_err(|e| format!("write tokenizer.json: {e}"))?;

    // ── model_config + index.json ─────────────────────────────────────
    // `has_model_weights=true` is the gate the loader checks; without
    // it `load_model_weights` errors with "rebuild with extract --level
    // all". model_config carries the arch fields detect_from_json needs
    // to reconstruct the tinymodel arch on the loader side.
    let model_config = VindexModelConfig {
        model_type: "tinymodel".into(),
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
        layer_rope_theta: None,
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
        model: "synthetic/tinymodel".into(),
        family: "tinymodel".into(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    // Writes index.json + gate_vectors.bin + down_meta.bin.
    // `save_vindex` mutates `config` to record layer manifests.
    index
        .save_vindex(dir, &mut config)
        .map_err(|e| format!("save_vindex: {e}"))?;

    // ── Model weights (attn / up / down / norms / lm_head) ────────────
    let mut cb = larql_vindex::SilentBuildCallbacks;
    write_model_weights(&weights, dir, &mut cb).map_err(|e| format!("write_model_weights: {e}"))?;

    // ── Embeddings (vocab × hidden f32, little-endian) ────────────────
    let embed_slice = weights.embed.as_slice().ok_or("embed not contiguous")?;
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * 4);
    for &v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), &embed_bytes)
        .map_err(|e| format!("write embeddings.bin: {e}"))?;

    Ok(())
}

/// Serialise the synthetic `make_test_q4k_weights()` model + matching
/// Q4_K vindex to an on-disk directory that the strict
/// `open_inference_vindex` loader will accept.
///
/// Companion to [`write_synthetic_model_dir`]. Use this when a test
/// needs to exercise the Q4_K loader resolution order (attn_weights_q4k
/// → interleaved_kquant → lm_head_q4) without a real Gemma snapshot on
/// disk.
///
/// Layout written:
/// ```text
/// dir/
///   index.json                       -- VindexConfig with quant=Q4K
///   tokenizer.json                   -- WordLevel "[0]".."[VOCAB-1]"
///   gate_vectors.bin                 -- empty per-layer (vindex contract)
///   down_meta.bin                    -- empty per-layer
///   attn_weights_q4k.bin             -- Q/K/V/O quantised per layer
///   attn_weights_q4k_manifest.json
///   interleaved_kquant.bin              -- [gate|up|down] per layer
///   interleaved_kquant_manifest.json
///   lm_head_q4.bin                   -- tied embed quantised
///   norms.bin                        -- f32 norms (unchanged from non-Q4 path)
/// ```
pub fn write_synthetic_q4k_model_dir(dir: &std::path::Path) -> Result<(), String> {
    write_synthetic_q4k_model_dir_layers(dir, Q4K_TEST_NUM_LAYERS)
}

/// Layer-parametrised sibling of [`write_synthetic_q4k_model_dir`]. Same
/// on-disk layout, but the serialised model has `num_layers` decoder
/// layers — for tests that need a depth-fraction layer index to land
/// inside the model (e.g. the LQL FR3 relation resolver's probe layer,
/// which clamps to ≥3 and so needs a model deeper than the 2-layer default).
pub fn write_synthetic_q4k_model_dir_layers(
    dir: &std::path::Path,
    num_layers: usize,
) -> Result<(), String> {
    use larql_vindex::{
        write_model_weights_kquant, ExtractLevel, MoeConfig, SilentBuildCallbacks, StorageDtype,
        VindexConfig, VindexModelConfig,
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {e}"))?;

    let weights = make_test_q4k_weights_layers(num_layers);

    // ── tokenizer.json ────────────────────────────────────────────────
    std::fs::write(
        dir.join("tokenizer.json"),
        synthetic_tokenizer_json(weights.vocab_size),
    )
    .map_err(|e| format!("write tokenizer.json: {e}"))?;

    // ── model_config + index.json ─────────────────────────────────────
    let model_config = VindexModelConfig {
        model_type: "gemma3_text".into(),
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
        layer_rope_theta: None,
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
        model: "synthetic/gemma3_q4k".into(),
        family: "gemma3".into(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::Q4K,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    // Use an empty in-memory index for `save_vindex` (writes the
    // mandatory gate_vectors.bin + down_meta.bin + index.json scaffolding).
    let empty_index = larql_vindex::VectorIndex::new(
        vec![None; weights.num_layers],
        vec![None; weights.num_layers],
        weights.num_layers,
        weights.hidden_size,
    );
    empty_index
        .save_vindex(dir, &mut config)
        .map_err(|e| format!("save_vindex: {e}"))?;

    // ── Q4K weights (attn_weights_q4k + interleaved_kquant + lm_head_q4 + norms) ──
    let mut cb = SilentBuildCallbacks;
    write_model_weights_kquant(&weights, dir, &mut cb)
        .map_err(|e| format!("write_model_weights_kquant: {e}"))?;

    // ── Embeddings (required by `load_model_weights_kquant` — the Q4K
    //    writer doesn't emit them on its own). ─────────────────────
    let embed_slice = weights.embed.as_slice().ok_or("embed not contiguous")?;
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * 4);
    for &v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), &embed_bytes)
        .map_err(|e| format!("write embeddings.bin: {e}"))?;

    Ok(())
}

/// Build a tokenizer JSON whose vocab is `[0]`..`[vocab_size-1]` and
/// whose `pre_tokenizer` is **null** — so bracketed forms encode as a
/// single token instead of being split into `[`, `N`, `]` (all UNK)
/// by [`make_test_tokenizer`]'s Whitespace pre-tokenizer.
///
/// Used only by [`write_synthetic_model_dir`] so on-disk-fixture
/// callers can write test prompts like `"[1]"` and have them
/// encode to a single in-vocab id. `make_test_tokenizer` is kept
/// in its prior shape for backward-compatibility with in-memory
/// fixture consumers.
///
/// `[UNK]` is mapped to **id 0** (a real, in-range vocab slot) so any
/// stray UNK from text the loader processes through the model still
/// hits a valid embedding row — saves the embed lookup from panicking
/// with "Index N must be less than axis length N" when something
/// outside the bracket form sneaks into encoding.
/// Build the on-disk tokenizer JSON whose vocab is `[0]`..`[vocab_size-1]`
/// and whose `pre_tokenizer` is **null** — bracketed forms encode as a
/// single token. Public so tests can build a matching in-memory
/// `Tokenizer` without going through `write_synthetic_model_dir`.
///
/// A `Strip` normalizer trims surrounding whitespace, so the spaced form a
/// KNN or compose INSERT encodes (`" [5]"`) is the single token 5, as
/// `"[5]"` is. Without it every spaced target was `[UNK]`, and a KNN entry
/// stored its target text beside id 0 — which LQL now refuses (FACT-ID-1
/// D-5: a KNN target is one token). Only leading and trailing whitespace
/// is trimmed, so a multi-word prompt encodes as before, and no id gains a
/// second spelling.
pub fn synthetic_tokenizer_json(vocab_size: usize) -> String {
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(0.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": { "type": "Strip", "strip_left": true, "strip_right": true },
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });
    serde_json::to_string(&tokenizer_json).expect("synthetic tokenizer json")
}

// ── Alternate-arch fixtures ─────────────────────────────────────────────
//
// `make_test_weights` uses the `tinymodel` arch which leaves many optional
// branches dormant (no bias keys, no QK norm, no post norms, gated FFN
// only). The fixtures below pin those branches by routing through a
// real arch impl that enables them. Each fixture provides exactly the
// tensors + vectors the matching forward path needs to reach finite
// output without panicking.

pub(crate) fn rand_mat_seeded(rows: usize, cols: usize, scale: f32, seed: u64) -> WeightArray {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

// `make_gemma3_test_weights` and `make_starcoder2_test_weights` moved
// to `larql_models::test_fixtures` (ADR-0022 Step 2e) so larql-compute
// can run arch-specific spine tests too. Re-exported here so existing
// `crate::test_utils::make_*_test_weights` paths in inference test
// modules and downstream test crates (larql-kv) keep working.
pub use larql_models::test_fixtures::{make_gemma3_test_weights, make_starcoder2_test_weights};

// ── Q4_K-aware synthetic fixtures moved to `larql_models::test_fixtures` ──
// (ADR-0022 Step 3b) so larql-compute's kquant_forward tests can
// construct realistic Q4K-sized ModelWeights. Re-exported for existing
// `crate::test_utils::*` callers.
pub use larql_models::test_fixtures::{
    arc_mmap_from_bytes, make_test_q4k_weights, make_test_q4k_weights_layers,
    make_test_q4k_weights_rope_scaled, make_test_q4k_weights_silu, make_test_q4k_weights_wide,
    make_test_q4k_weights_with_dims, Q4K_TEST_HIDDEN, Q4K_TEST_INTER, Q4K_TEST_INTER_WIDE,
    Q4K_TEST_NUM_LAYERS, Q4K_TEST_VOCAB,
};
