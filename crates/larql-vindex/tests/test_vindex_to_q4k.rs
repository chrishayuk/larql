//! Smoke + happy-path tests for `quant::convert_q4k::vindex_to_q4k`.
//!
//! Three flavours of test:
//!   1. **Lifecycle / error paths** (no real weights needed) — pin
//!      preconditions and refusal messages.
//!   2. **Config defaults** — assert the Q4K_M mix stays the default.
//!   3. **End-to-end happy path** — synthesise a tiny safetensors
//!      model, stream-extract it to a float vindex, run
//!      `vindex_to_q4k`, then verify the output layout, manifest,
//!      and weight round-trip on a sampled Q4_K block.

use larql_vindex::format::filenames::*;
use std::path::PathBuf;

use larql_vindex::quant::{vindex_to_q4k, Q4kConvertConfig};

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!("q4k_cli_{label}_{}_{}", std::process::id(), ts));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Minimal index.json fixture parameterised by the two fields Q4K
/// converter inspects before it tries to load the real weights.
fn write_stub_index(dir: &std::path::Path, has_model_weights: bool, quant: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let idx = serde_json::json!({
        "version": 2,
        "model": "synthetic/q4k-test",
        "family": "synthetic",
        "num_layers": 2,
        "hidden_size": 256,
        "intermediate_size": 256,
        "vocab_size": 16,
        "embed_scale": 1.0,
        "extract_level": if has_model_weights { "inference" } else { "browse" },
        "dtype": "f32",
        "quant": quant,
        "layers": [
            {"layer": 0, "num_features": 4, "offset": 0,     "length": 1024},
            {"layer": 1, "num_features": 4, "offset": 1024,  "length": 1024},
        ],
        "down_top_k": 1,
        "has_model_weights": has_model_weights,
    });
    std::fs::write(
        dir.join("index.json"),
        serde_json::to_string_pretty(&idx).unwrap(),
    )
    .unwrap();
}

#[test]
fn q4k_refuses_existing_output_without_force() {
    let tmp = TempDir::new("no_force");
    let src = tmp.0.join("src.vindex");
    write_stub_index(&src, true, "none");
    let dst = tmp.0.join("dst.vindex");
    std::fs::create_dir_all(&dst).unwrap();

    let config = Q4kConvertConfig {
        force: false,
        ..Default::default()
    };
    let err = vindex_to_q4k(&src, &dst, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exists"),
        "expected 'exists' in error; got {msg}"
    );
}

#[test]
fn q4k_refuses_source_without_model_weights() {
    let tmp = TempDir::new("no_weights");
    let src = tmp.0.join("src.vindex");
    write_stub_index(&src, /*has_model_weights=*/ false, "none");
    let dst = tmp.0.join("dst.vindex");

    let config = Q4kConvertConfig::default();
    let err = vindex_to_q4k(&src, &dst, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("no model weights") && msg.contains("--level inference"),
        "error should point at the extract-level mismatch; got {msg}"
    );
    assert!(
        !dst.exists(),
        "dst should not be created on precondition failure"
    );
}

#[test]
fn q4k_refuses_already_quantised_source() {
    let tmp = TempDir::new("already_q4k");
    let src = tmp.0.join("src.vindex");
    write_stub_index(&src, true, "q4k");
    let dst = tmp.0.join("dst.vindex");

    let config = Q4kConvertConfig::default();
    let err = vindex_to_q4k(&src, &dst, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("already quantised") || msg.contains("already"),
        "error should say source is already quantised; got {msg}"
    );
    assert!(
        !dst.exists(),
        "dst should not be created on precondition failure"
    );
}

#[test]
fn q4k_config_defaults_match_q4k_m_mix() {
    // Sanity on the library's default — Q4K_M (Q4_K gate/up + Q6_K down).
    let c = Q4kConvertConfig::default();
    assert!(!c.down_q4k);
    assert!(!c.force);
}

// ─── End-to-end happy path ─────────────────────────────────────────
//
// Build a tiny synthetic safetensors model on disk, stream-extract it
// to a float vindex (with full model weights), then run
// `vindex_to_q4k` and verify:
//   - Output directory exists, staging tmp is gone (atomic rename).
//   - `index.json` has `quant=q4k`, `has_model_weights=true`,
//     checksums cleared.
//   - All Q4K weight files + manifests are present.
//   - Source's f32 weight files are NOT hard-linked into the dst
//     (they'd bloat output and never be read).
//   - A sampled Q4_K attention slice round-trips back to source
//     within tolerance — proves the manifest → bytes correspondence
//     is what the loader expects.

/// Llama-shaped synthetic-model fixture used by the end-to-end Q4_K
/// tests. Writes `config.json`, `tokenizer.json`, and a
/// `model.safetensors` packed with deterministic per-tensor ramps
/// (`(i as f32) * 0.01`) into `model_dir`. Returns the tokenizer so
/// callers can drive `build_vindex_streaming` without re-reading the
/// tokenizer file.
fn write_synthetic_llama_model(
    model_dir: &std::path::Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    use std::collections::HashMap;

    std::fs::create_dir_all(model_dir).unwrap();
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

#[test]
fn q4k_end_to_end_from_synthetic_safetensors() {
    use larql_vindex::QuantFormat;

    let tmp = TempDir::new("e2e_happy");
    let model_dir = tmp.0.join("model");
    let src_dir = tmp.0.join("src.vindex");
    let dst_dir = tmp.0.join("dst.vindex");

    // Tiny llama-shaped config — dims chosen so each tensor pads to
    // exactly one 256-element Q4_K super-block (hidden=8, intermediate=4).
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;
    let tokenizer =
        write_synthetic_llama_model(&model_dir, hidden, intermediate, num_layers, vocab);

    // Stream-extract to a *float* vindex (QuantFormat::None) at level=Inference
    // so all weight files land. This is the precondition vindex_to_q4k
    // expects: full model weights + quant=none.
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/q4k-e2e-source",
        &src_dir,
        4,
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // Sanity: source carries the float weights vindex_to_q4k expects.
    assert!(src_dir.join("up_weights.bin").exists());
    assert!(src_dir.join("down_weights.bin").exists());
    assert!(src_dir.join("attn_weights.bin").exists());
    let src_cfg = larql_vindex::load_vindex_config(&src_dir).unwrap();
    assert!(src_cfg.has_model_weights);
    assert_eq!(src_cfg.quant, QuantFormat::None);

    // ── Convert ──
    let report = vindex_to_q4k(&src_dir, &dst_dir, &Q4kConvertConfig::default()).unwrap();

    // ── Atomic rename: staging is gone, output dir is there ──
    assert!(
        !tmp.0.join("dst.vindex.tmp").exists(),
        "staging dir should be cleaned up"
    );
    assert!(dst_dir.exists());

    // ── Output layout ──
    //
    // New writers emit kquant-canonical filenames; legacy q4k names
    // must NOT be present in fresh output.
    for f in [
        "index.json",
        "attn_weights_kquant.bin",
        "attn_weights_kquant_manifest.json",
        "interleaved_kquant.bin",
        "interleaved_kquant_manifest.json",
        "lm_head_kquant.bin",
        "norms.bin",
        "weight_manifest.json",
    ] {
        assert!(dst_dir.join(f).exists(), "expected {f} in output");
    }
    for f in [
        "attn_weights_q4k.bin",
        "attn_weights_q4k_manifest.json",
        "interleaved_q4k.bin",
        "interleaved_q4k_manifest.json",
        "lm_head_q4.bin",
    ] {
        assert!(
            !dst_dir.join(f).exists(),
            "legacy q4k file {f} must not appear in fresh writes"
        );
    }

    // The f32 weight files vindex_to_q4k explicitly skips from hard-linking.
    for f in [
        "attn_weights.bin",
        "up_weights.bin",
        "down_weights.bin",
        "interleaved.bin",
        LM_HEAD_BIN,
    ] {
        assert!(
            !dst_dir.join(f).exists(),
            "{f} should NOT have been hard-linked (the Q4K weight files replace it)"
        );
    }

    // Aux files that ARE hard-linked through.
    assert!(
        dst_dir.join("down_meta.bin").exists(),
        "down_meta.bin should be hard-linked"
    );

    // ── Manifest ──
    let dst_cfg = larql_vindex::load_vindex_config(&dst_dir).unwrap();
    assert_eq!(dst_cfg.quant, QuantFormat::Q4K);
    assert!(dst_cfg.has_model_weights);
    assert!(
        dst_cfg.checksums.is_none(),
        "checksums must be cleared (source's no longer apply)"
    );

    // ── Round-trip: dequantise the layer-0 Q tensor and confirm we get
    // back the source synthetic ramp (within Q4_K block error). Same
    // pattern as `streaming_extract_q4k_from_safetensors`'s round-trip.
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&dst_dir, &mut lcb).unwrap();
    index.load_attn_kquant(&dst_dir).unwrap();
    let slices = index.attn_kquant_layer_data(0).expect("layer 0 attn data");
    assert_eq!(slices[0].1, "Q4_K", "Q slot format");
    assert_eq!(slices[2].1, "Q6_K", "V slot format");

    // Q is hidden×hidden = 64 elements, padded to one 256-elem super-block.
    let padded_cols = 256usize;
    let q_dequant =
        larql_models::quant::ggml::dequantize_q4_k(slices[0].0, hidden * padded_cols).unwrap();
    let expected: Vec<f32> = (0..(hidden * hidden)).map(|i| (i as f32) * 0.01).collect();
    for row in 0..hidden {
        for col in 0..hidden {
            let i = row * hidden + col;
            let v = expected[i];
            let got = q_dequant[row * padded_cols + col];
            assert!(
                (got - v).abs() < 0.03,
                "Q[r{row} c{col}] round-trip diverged: got {got}, expected {v}"
            );
        }
    }

    // ── Report shape ──
    assert!(report.compression > 0.0, "compression must be reported");
    assert!(
        report.aux_linked_count > 0,
        "at least one aux file should land via hard-link"
    );
    assert!(
        !report.walk_backend.is_empty(),
        "walk_backend description must be populated"
    );
}

/// Round-trip the W2 feature-major down emit: convert with
/// `feature_major_down=true`, load, then ask the dispatch path for one
/// feature's down vector. With the new file present, the dispatch
/// should serve the row from `down_features_q4k.bin` and skip the
/// cache (asserted via `kquant_ffn_cache_stats`).
#[test]
fn q4k_feature_major_down_round_trip() {
    use larql_vindex::QuantFormat;

    let tmp = TempDir::new("fm_down");
    let model_dir = tmp.0.join("model");
    let src_dir = tmp.0.join("src.vindex");
    let dst_dir = tmp.0.join("dst.vindex");

    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;
    let tokenizer =
        write_synthetic_llama_model(&model_dir, hidden, intermediate, num_layers, vocab);

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/fm-down",
        &src_dir,
        4,
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    let convert_config = Q4kConvertConfig {
        feature_major_down: true,
        ..Default::default()
    };
    vindex_to_q4k(&src_dir, &dst_dir, &convert_config).unwrap();

    // ── Files emitted ──
    assert!(
        dst_dir.join(DOWN_FEATURES_KQUANT_BIN).exists(),
        "down_features_q4k.bin must be emitted when feature_major_down=true"
    );
    assert!(
        dst_dir.join(DOWN_FEATURES_KQUANT_MANIFEST_JSON).exists(),
        "down_features_q4k_manifest.json must be emitted alongside it"
    );

    // ── Load + dispatch through the feature-major path ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&dst_dir, &mut lcb).unwrap();
    assert!(
        index.has_down_features_kquant(),
        "loader must surface the feature-major down file"
    );

    // Cache-bypass evidence: ask for one feature's down. The W2 path
    // serves it from `down_features_q4k.bin` without populating the
    // legacy cache.
    let mut out = vec![0.0f32; hidden];
    let alpha = 1.0f32;
    let layer = 0;
    let feat = 1usize;
    assert!(
        index.kquant_down_feature_scaled_add(layer, feat, alpha, &mut out),
        "feature-major down decode must succeed when the file is present"
    );
    let (cache_slots, cache_bytes) = index.kquant_ffn_cache_stats();
    assert_eq!(
        (cache_slots, cache_bytes),
        (0, 0),
        "feature-major path must NOT have populated the legacy kquant_ffn_layer cache"
    );

    // ── Round-trip the values: decoded row must approximate
    //    down_proj[:, feat] from the source synthetic ramp ──
    // Each synthetic tensor's ramp restarts from 0, so down_proj's
    // values are `(i * 0.01)` for `i in 0..hidden*intermediate`. With
    // shape [hidden, intermediate] row-major, feature `feat`'s vector
    // is `[down_proj[h, feat] for h in 0..hidden]`, i.e.
    // `[(h * intermediate + feat) * 0.01 for h in 0..hidden]`.
    let expected: Vec<f32> = (0..hidden)
        .map(|h| ((h * intermediate + feat) as f32) * 0.01)
        .collect();
    for (h, &got) in out.iter().enumerate() {
        let want = expected[h];
        assert!(
            (got - want).abs() < 0.05,
            "down[{layer}][feat={feat}][{h}] diverged: got {got}, expected {want}"
        );
    }
}

/// End-to-end legacy-name round-trip: write a kquant vindex with the
/// new canonical filenames, then rename the on-disk files to the
/// pre-rename `*_q4k.bin` / `lm_head_q4.bin` names and confirm the
/// dual-read loaders still pick them up.
///
/// This is the regression that protects existing HF-hosted vindexes
/// (gemma3-4b-v2 etc.) — they were published before the rename and we
/// must keep loading them without re-extraction.
#[test]
fn legacy_q4k_filenames_load_via_dual_read() {
    use larql_vindex::QuantFormat;

    let tmp = TempDir::new("legacy_dual_read");
    let model_dir = tmp.0.join("model");
    let src_dir = tmp.0.join("src.vindex");
    let dst_dir = tmp.0.join("dst.vindex");

    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;
    let tokenizer =
        write_synthetic_llama_model(&model_dir, hidden, intermediate, num_layers, vocab);

    // 1. Write an f32 vindex.
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/legacy-round-trip",
        &src_dir,
        4,
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // 2. Convert to k-quant. New writers emit *_kquant.bin filenames.
    vindex_to_q4k(&src_dir, &dst_dir, &Q4kConvertConfig::default()).unwrap();

    // Sanity: kquant filenames are present, q4k-named files absent.
    assert!(dst_dir.join(ATTN_WEIGHTS_KQUANT_BIN).is_file());
    assert!(dst_dir.join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON).is_file());
    assert!(dst_dir.join(INTERLEAVED_KQUANT_BIN).is_file());
    assert!(dst_dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON).is_file());
    assert!(dst_dir.join(LM_HEAD_KQUANT_BIN).is_file());

    // 3. Simulate a pre-rename on-disk layout: rename the new files
    //    back to the legacy q4k names.
    std::fs::rename(
        dst_dir.join(ATTN_WEIGHTS_KQUANT_BIN),
        dst_dir.join(LEGACY_ATTN_WEIGHTS_Q4K_BIN),
    )
    .unwrap();
    std::fs::rename(
        dst_dir.join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON),
        dst_dir.join(LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON),
    )
    .unwrap();
    std::fs::rename(
        dst_dir.join(INTERLEAVED_KQUANT_BIN),
        dst_dir.join(LEGACY_INTERLEAVED_Q4K_BIN),
    )
    .unwrap();
    std::fs::rename(
        dst_dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON),
        dst_dir.join(LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON),
    )
    .unwrap();
    std::fs::rename(
        dst_dir.join(LM_HEAD_KQUANT_BIN),
        dst_dir.join(LEGACY_LM_HEAD_Q4_BIN),
    )
    .unwrap();

    // 4. Confirm dual-read picks up the legacy names.
    assert!(has_kquant_attn_weights(&dst_dir));
    assert!(has_kquant_interleaved(&dst_dir));
    assert!(has_kquant_lm_head(&dst_dir));

    // 5. Load each through the real loader entry points.
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&dst_dir, &mut lcb).unwrap();
    index
        .load_attn_kquant(&dst_dir)
        .expect("load_attn_kquant must accept legacy attn_weights_q4k.bin");
    index
        .load_interleaved_kquant(&dst_dir)
        .expect("load_interleaved_kquant must accept legacy interleaved_q4k.bin");
    index
        .load_lm_head_kquant(&dst_dir)
        .expect("load_lm_head_kquant must accept legacy lm_head_q4.bin");

    assert!(index.attn_kquant_layer_data(0).is_some());
    assert!(index.has_interleaved_kquant());
    assert!(index.has_lm_head_kquant());
}
