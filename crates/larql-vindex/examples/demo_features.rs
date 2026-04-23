//! Vindex Feature Showcase — demonstrates the complete larql-vindex API.
//!
//! Covers: build, KNN, walk, PatchedVindex (readonly base + overlay), layer bands,
//! MoE layout, binary down_meta, f16 storage, source provenance, checksum verification,
//! patches (create, apply, revert, bake down), extract pipeline, GGUF key normalization,
//! Vindexfile parsing, HuggingFace path handling, and quantization formats.
//!
//! Run: cargo run -p larql-vindex --example vindex_demo

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2, ArcArray2};
use std::collections::HashMap;

fn main() {
    println!("=== Vindex Feature Showcase ===\n");

    // ── 1. Build in-memory ──
    section("1. Build in-memory index");
    let index = build_demo_index();
    println!("  {} layers, {} features, {} with metadata",
        index.num_layers, index.total_gate_vectors(), index.total_down_meta());

    // ── 2. Layer bands ──
    section("2. Layer bands (per-family, exact boundaries)");
    for &(family, layers) in &[
        ("gpt2", 12), ("llama", 32), ("gemma3", 34),
        ("qwen2", 40), ("llama", 80), ("mixtral", 32),
    ] {
        match larql_vindex::LayerBands::for_family(family, layers) {
            Some(b) => println!("  {:<8} {:>2}L  syntax={:>2}-{:<2}  knowledge={:>2}-{:<2}  output={:>2}-{:<2}",
                family, layers, b.syntax.0, b.syntax.1, b.knowledge.0, b.knowledge.1, b.output.0, b.output.1),
            None => println!("  {:<8} {:>2}L  (too few layers)", family, layers),
        }
    }

    // ── 3. Gate KNN ──
    section("3. Gate KNN");
    let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    println!("  Query [1,0,0,0]:");
    for (feat, score) in index.gate_knn(0, &q, 3) {
        let tok = index.feature_meta(0, feat).map(|m| m.top_token.clone()).unwrap_or_else(|| "-".into());
        println!("    F{}: {} ({:.1})", feat, tok, score);
    }

    // ── 4. Walk ──
    section("4. Walk (multi-layer)");
    let trace = index.walk(&q, &[0, 1], 2);
    for (layer, hits) in &trace.layers {
        if hits.is_empty() { println!("  L{}: (none)", layer); continue; }
        for h in hits { println!("  L{}: F{} → {} ({:.1})", layer, h.feature, h.meta.top_token, h.gate_score); }
    }

    // ── 5. MoE ──
    section("5. MoE layout (2 experts × 3 features)");
    let moe_index = build_moe_index();
    println!("  Total features at L0: {}", moe_index.num_features(0));
    for &(label, ref q) in &[
        ("Expert 0 [1,0,0,0]", vec![1.0, 0.0, 0.0, 0.0]),
        ("Expert 1 [0,0,0,1]", vec![0.0, 0.0, 0.0, 1.0]),
    ] {
        println!("  {}:", label);
        for (f, s) in moe_index.gate_knn(0, &Array1::from_vec(q.clone()), 2) {
            let e = if f < 3 { 0 } else { 1 };
            let tok = moe_index.feature_meta(0, f).map(|m| m.top_token.clone()).unwrap_or_else(|| "-".into());
            println!("    E{}:F{} → {} ({:.1})", e, f % 3, tok, s);
        }
    }

    // ── 6. Mutate (via PatchedVindex — base is readonly) ──
    section("6. Mutate via PatchedVindex (readonly base, patch overlay)");
    let base = build_demo_index();
    let mut patched = larql_vindex::PatchedVindex::new(base);
    let slot = patched.find_free_feature(0).unwrap();
    patched.insert_feature(
        0, slot,
        vec![0.0, 0.0, 0.0, 10.0],
        meta("Canberra", 104, 0.85),
    );
    println!("  Inserted F{} → Canberra (patch overlay)", slot);
    patched.delete_feature(0, 2);
    println!("  Deleted F2 (was Tokyo, patch overlay)");
    println!("  Overrides: {}", patched.num_overrides());

    // Bake down to a clean VectorIndex for saving
    let index = patched.bake_down();
    println!("  Baked down: base files untouched, new index has changes");

    // ── 7. f32 save + checksums ──
    section("7. Save (f32, binary down_meta, checksums)");
    let dir = std::env::temp_dir().join("larql_vindex_showcase");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = index.save_gate_vectors(&dir).unwrap();
    let _dm_count = index.save_down_meta(&dir).unwrap();

    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    println!("  down_meta.bin:   {} bytes (binary only — JSONL no longer written)", bin_size);
    assert!(!dir.join("down_meta.jsonl").exists());

    let config = make_config("showcase", 2, 4, 5, layer_infos, larql_vindex::StorageDtype::F32);
    VectorIndex::save_config(&config, &dir).unwrap();

    if let Some(ref checksums) = config.checksums {
        let results = larql_vindex::format::checksums::verify_checksums(&dir, checksums).unwrap();
        print!("  Checksums: ");
        let all_ok = results.iter().all(|(_, ok)| *ok);
        println!("{}", if all_ok { "ALL OK" } else { "FAILED" });
    }

    // ── 8. Reload ──
    section("8. Reload and verify");
    // Write a minimal tokenizer (needed for binary down_meta token resolution)
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();
    let lc = larql_vindex::load_vindex_config(&dir).unwrap();
    println!("  Version: {}, dtype: {}, extract: {}", lc.version, lc.dtype, lc.extract_level);
    println!("  Features: {}, with meta: {}", loaded.total_gate_vectors(), loaded.total_down_meta());
    if let Some(src) = &lc.source {
        println!("  Source: {}", src.huggingface_repo.as_deref().unwrap_or("?"));
    }
    let hits = loaded.gate_knn(0, &Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), 1);
    let meta = loaded.feature_meta(0, hits[0].0).unwrap();
    println!("  KNN [1,0,0,0] → F{}: token_id={} (score={:.2}) ✓",
        hits[0].0, meta.top_token_id, meta.c_score);
    let _ = std::fs::remove_dir_all(&dir);

    // ── 9. f16 storage ──
    section("9. f16 storage (half precision)");
    let dir16 = std::env::temp_dir().join("larql_vindex_showcase_f16");
    let _ = std::fs::remove_dir_all(&dir16);
    std::fs::create_dir_all(&dir16).unwrap();

    let idx16 = build_demo_index();
    // Manually write gate vectors as f16
    let gate_data = idx16.gate_vectors_at(0).unwrap().as_slice().unwrap();
    let f16_bytes = larql_models::quant::half::encode_f16(gate_data);
    let f32_bytes_len = gate_data.len() * 4;
    println!("  Gate L0: {} bytes (f32) → {} bytes (f16) = {:.0}% smaller",
        f32_bytes_len, f16_bytes.len(), (1.0 - f16_bytes.len() as f64 / f32_bytes_len as f64) * 100.0);

    // Round-trip: f32 → f16 → f32
    let decoded = larql_models::quant::half::decode_f16(&f16_bytes);
    let max_err: f32 = gate_data.iter().zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max round-trip error: {:.6}", max_err);
    let _ = std::fs::remove_dir_all(&dir16);

    // ── 10. Extract pipeline ──
    section("10. Extract pipeline (synthetic model)");
    let dir_ext = std::env::temp_dir().join("larql_vindex_showcase_extract");
    let _ = std::fs::remove_dir_all(&dir_ext);
    std::fs::create_dir_all(&dir_ext).unwrap();

    let weights = make_synthetic_model();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir_ext.join("tokenizer.json"), tok_json).unwrap();

    let mut ecb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "demo/synthetic",
        &dir_ext,
        3,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut ecb,
    ).unwrap();

    let ext_config = larql_vindex::load_vindex_config(&dir_ext).unwrap();
    println!("  Model: {}", ext_config.model);
    println!("  Layers: {}, hidden: {}, features: {}",
        ext_config.num_layers, ext_config.hidden_size, ext_config.intermediate_size);
    println!("  Extract level: {}, dtype: {}", ext_config.extract_level, ext_config.dtype);
    println!("  Has weights: {}", ext_config.has_model_weights);

    let files: Vec<_> = std::fs::read_dir(&dir_ext).unwrap()
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            (name, size)
        })
        .collect();
    let mut files = files;
    files.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  Files:");
    for (name, size) in &files {
        if *size > 1024 { println!("    {:<30} {:.1} KB", name, *size as f64 / 1024.0); }
        else { println!("    {:<30} {} B", name, size); }
    }

    // Load and query the extracted model
    let ext_index = VectorIndex::load_vindex(&dir_ext, &mut cb).unwrap();
    let ext_q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let ext_hits = ext_index.gate_knn(0, &ext_q, 1);
    println!("  KNN [1,0,...] → F{} (score={:.1})", ext_hits[0].0, ext_hits[0].1);
    let _ = std::fs::remove_dir_all(&dir_ext);

    // ── 11. Patches ──
    section("11. Patches (create, apply, stack, revert, bake)");
    let dir_p = std::env::temp_dir().join("larql_vindex_showcase_patch");
    let _ = std::fs::remove_dir_all(&dir_p);
    std::fs::create_dir_all(&dir_p).unwrap();

    let base = build_demo_index();
    let mut patched = larql_vindex::PatchedVindex::new(base);

    // Create patch
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "demo".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("Medical facts".into()),
        author: Some("demo".into()),
        tags: vec!["medical".into()],
        operations: vec![
            larql_vindex::PatchOp::Insert {
                layer: 0, feature: 4,
                relation: Some("treats".into()),
                entity: "aspirin".into(), target: "headache".into(),
                confidence: Some(0.85),
                gate_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[0.0, 0.0, 0.0, 10.0])),
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: "headache".into(), top_token_id: 200, c_score: 4.2,
                }),
            },
            larql_vindex::PatchOp::Delete {
                layer: 0, feature: 2,
                reason: Some("incorrect".into()),
            },
        ],
    };

    // Save + load .vlp
    let vlp_path = dir_p.join("medical.vlp");
    patch.save(&vlp_path).unwrap();
    let loaded_patch = larql_vindex::VindexPatch::load(&vlp_path).unwrap();
    let (ins, _upd, del) = loaded_patch.counts();
    println!("  Created: medical.vlp ({} bytes, {} ins, {} del)",
        std::fs::metadata(&vlp_path).unwrap().len(), ins, del);

    // Apply
    patched.apply_patch(loaded_patch);
    println!("  Applied: {} patches, {} overrides", patched.num_patches(), patched.num_overrides());
    println!("    F0 = {}", patched.feature_meta(0, 0).map(|m| m.top_token.clone()).unwrap_or_else(|| "(none)".into()));
    println!("    F2 = {}", patched.feature_meta(0, 2).map(|m| m.top_token.clone()).unwrap_or_else(|| "(none)".into()));
    println!("    F4 = {}", patched.feature_meta(0, 4).map(|m| m.top_token.clone()).unwrap_or_else(|| "(none)".into()));

    // KNN with patch
    let pq = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let phits = patched.gate_knn(0, &pq, 1);
    println!("  KNN [0,0,0,1] → F{}: {}",
        phits[0].0, patched.feature_meta(0, phits[0].0).map(|m| m.top_token.clone()).unwrap_or_else(|| "?".into()));

    // Bake down
    let baked = patched.bake_down();
    println!("  Baked: {} features, {} with meta", baked.total_gate_vectors(), baked.total_down_meta());

    // Revert
    patched.remove_patch(0);
    println!("  Reverted: F2 = {} (restored)",
        patched.feature_meta(0, 2).map(|m| m.top_token.clone()).unwrap_or_else(|| "(none)".into()));

    let _ = std::fs::remove_dir_all(&dir_p);

    // ── 12. Describe types ──
    section("12. Describe types");
    println!("  LabelSource: probe={}, cluster={}, pattern={}, none={}",
        larql_vindex::LabelSource::Probe,
        larql_vindex::LabelSource::Cluster,
        larql_vindex::LabelSource::Pattern,
        larql_vindex::LabelSource::None);

    let edge = larql_vindex::DescribeEdge {
        relation: Some("capital".into()),
        source: larql_vindex::LabelSource::Probe,
        target: "Paris".into(),
        gate_score: 1436.9,
        layer_min: 27, layer_max: 27,
        count: 1, also_tokens: vec![],
    };
    println!("  Edge: {} → {} ({:.1}, {})",
        edge.relation.as_deref().unwrap_or("?"), edge.target, edge.gate_score, edge.source);

    // ── 13. GGUF key normalization ──
    section("13. GGUF key normalization");
    let keys = [
        ("blk.0.attn_q.weight", "layers.0.self_attn.q_proj.weight"),
        ("blk.15.ffn_gate.weight", "layers.15.mlp.gate_proj.weight"),
        ("token_embd.weight", "embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
        ("blk.3.ffn_down.weight", "layers.3.mlp.down_proj.weight"),
    ];
    for (gguf_key, expected) in &keys {
        let normalized = larql_models::loading::gguf::normalize_gguf_key(gguf_key);
        let status = if normalized == *expected { "OK" } else { "MISMATCH" };
        println!("  {} → {} ({})", gguf_key, normalized, status);
    }

    // ── 14. Vindexfile parsing ──
    section("14. Vindexfile parsing");
    let vf_str = r#"
FROM ./base.vindex
PATCH ./medical.vlp
PATCH ./company.vlp
INSERT ("Acme Corp", "headquarters", "London")
DELETE entity = "OldCo" AND relation = "status" AND target = "active"
LABELS ./labels.json
EXPOSE browse inference

STAGE prod
  PATCH ./production-only.vlp
  EXPOSE browse inference

STAGE edge
  EXPOSE browse
"#;
    let vf = larql_vindex::vindexfile::parse_vindexfile_str(vf_str).unwrap();
    println!("  Directives: {}", vf.directives.len());
    println!("  Stages: {}", vf.stages.len());
    for d in &vf.directives {
        match d {
            larql_vindex::VindexfileDirective::From(p) => println!("    FROM {}", p),
            larql_vindex::VindexfileDirective::Patch(p) => println!("    PATCH {}", p),
            larql_vindex::VindexfileDirective::Insert { entity, relation, target } =>
                println!("    INSERT ({}, {}, {})", entity, relation, target),
            larql_vindex::VindexfileDirective::Delete { entity, relation, target } =>
                println!("    DELETE entity={} relation={} target={}", entity, relation, target),
            larql_vindex::VindexfileDirective::Labels(p) => println!("    LABELS {}", p),
            larql_vindex::VindexfileDirective::Expose(levels) => println!("    EXPOSE {:?}", levels),
        }
    }
    for stage in &vf.stages {
        println!("  Stage '{}': {} directives", stage.name, stage.directives.len());
    }

    // ── 15. HuggingFace path handling ──
    section("15. HuggingFace path handling");
    let hf_paths = [
        ("hf://chrishayuk/gemma-3-4b-it-vindex", true),
        ("hf://user/repo@v2.0", true),
        ("./local.vindex", false),
        ("/absolute/path", false),
    ];
    for (path, expected) in &hf_paths {
        let is_hf = larql_vindex::is_hf_path(path);
        let status = if is_hf == *expected { "OK" } else { "MISMATCH" };
        println!("  {} → hf={} ({})", path, is_hf, status);
    }
    println!("  Supported: USE \"hf://user/repo\"; downloads and loads automatically");
    println!("  CLI: larql hf download user/repo [-o local/]");
    println!("  CLI: larql hf publish ./local.vindex --repo user/repo");

    // ── 16. Quantization formats ──
    section("16. Quantization formats (quant module)");

    // f16 round-trip
    let f16_vals = [0.0f32, 1.0, -1.0, 0.5, 100.0, std::f32::consts::PI];
    let f16_encoded = larql_models::quant::half::encode_f16(&f16_vals);
    let f16_decoded = larql_models::quant::half::decode_f16(&f16_encoded);
    print!("  f16 round-trip: ");
    for (orig, dec) in f16_vals.iter().zip(f16_decoded.iter()) {
        let err = (orig - dec).abs();
        print!("{orig}→{dec:.2}(err={err:.4}) ");
    }
    println!("✓");

    // bf16 round-trip
    let bf16_encoded = larql_models::quant::half::encode_bf16(&f16_vals);
    let bf16_decoded = larql_models::quant::half::decode_bf16(&bf16_encoded);
    print!("  bf16 round-trip: ");
    for (orig, dec) in f16_vals.iter().zip(bf16_decoded.iter()) {
        let err = (orig - dec).abs();
        print!("{orig}→{dec:.2}(err={err:.4}) ");
    }
    println!("✓");

    // GGML Q4_0
    let mut q4_block = vec![0x00u8, 0x3C]; // scale=1.0
    q4_block.extend_from_slice(&[0x19; 16]); // lo=9-8=1, hi=1-8=-7
    let q4_result = larql_models::quant::ggml::dequantize(&q4_block, 2, 32).unwrap();
    println!("  GGML Q4_0: scale=1.0, quant=0x19 → [{:.1}, {:.1}, ...] (32 values) ✓",
        q4_result[0], q4_result[1]);

    // GGML Q8_0
    let mut q8_block = vec![0x00u8, 0x3C]; // scale=1.0
    q8_block.push(42); q8_block.push(0xD6u8); // 42, -42 as i8
    q8_block.extend_from_slice(&[0u8; 30]);
    let q8_result = larql_models::quant::ggml::dequantize(&q8_block, 6, 32).unwrap();
    println!("  GGML Q8_0: scale=1.0, quants=[42,-42,...] → [{:.1}, {:.1}, ...] ✓",
        q8_result[0], q8_result[1]);

    // MXFP4
    let mxfp4_blocks = vec![0x37u8; 16]; // lo=7(6.0), hi=3(1.5)
    let mxfp4_scales = vec![127u8]; // e8m0 = 1.0
    let mxfp4_result = larql_models::quant::mxfp4::dequantize_expert(
        &mxfp4_blocks, &mxfp4_scales, 1, 1,
    );
    println!("  MXFP4: scale=1.0(e8m0=127), quant=0x37 → [{:.1}, {:.1}, ...] (32 values) ✓",
        mxfp4_result[0], mxfp4_result[1]);

    // e8m0 scale examples
    print!("  e8m0 scales: ");
    for exp in [0u8, 125, 126, 127, 128, 129, 130] {
        let val = larql_models::quant::mxfp4::e8m0_to_f32(exp);
        print!("{exp}→{val} ");
    }
    println!("✓");

    // Type info
    println!("  GGML types: F32={}, F16={}, Q4_0={}, Q8_0={}",
        larql_models::quant::ggml::type_name(0),
        larql_models::quant::ggml::type_name(1),
        larql_models::quant::ggml::type_name(2),
        larql_models::quant::ggml::type_name(6));
    println!("  Supported: f16, bf16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4");

    println!("\n=== Done ({} features demonstrated) ===", 16);
}

// ── Helpers ──

fn section(name: &str) { println!("\n── {} ──\n", name); }

fn meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(), top_token_id: id, c_score: score,
        top_k: vec![TopKEntry { token: token.into(), token_id: id, logit: score }],
    }
}

fn build_demo_index() -> VectorIndex {
    let h = 4;
    let mut g0 = Array2::<f32>::zeros((5, h));
    g0[[0, 0]] = 10.0; g0[[1, 1]] = 10.0; g0[[2, 2]] = 10.0;
    g0[[3, 0]] = 5.0; g0[[3, 1]] = 5.0;
    let g1 = Array2::<f32>::zeros((5, h));
    let m0 = vec![
        Some(meta("Paris", 100, 0.95)), Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)), Some(meta("European", 103, 0.70)), None,
    ];
    VectorIndex::new(vec![Some(g0), Some(g1)], vec![Some(m0), Some(vec![None; 5])], 2, h)
}

fn build_moe_index() -> VectorIndex {
    let h = 4;
    let mut g = Array2::<f32>::zeros((6, h));
    g[[0, 0]] = 10.0; g[[1, 1]] = 10.0; g[[2, 2]] = 10.0;
    g[[3, 3]] = 10.0; g[[4, 0]] = 5.0; g[[4, 3]] = 5.0; g[[5, 1]] = 3.0;
    let m = vec![
        Some(meta("Paris", 100, 0.95)), Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)), Some(meta("London", 103, 0.90)),
        Some(meta("Rome", 104, 0.85)), Some(meta("Madrid", 105, 0.80)),
    ];
    VectorIndex::new(vec![Some(g)], vec![Some(m)], 1, h)
}

fn make_config(model: &str, layers: usize, hidden: usize, intermediate: usize,
    layer_infos: Vec<larql_vindex::VindexLayerInfo>, dtype: larql_vindex::StorageDtype) -> VindexConfig {
    VindexConfig {
        version: 2, model: model.into(), family: "demo".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some(format!("demo/{model}")),
            huggingface_revision: None, safetensors_sha256: None,
            extracted_at: "2026-04-01T00:00:00Z".into(),
            larql_version: env!("CARGO_PKG_VERSION").into(),
        }),
        checksums: larql_vindex::format::checksums::compute_checksums(
            &std::env::temp_dir().join("larql_vindex_showcase")).ok(),
        num_layers: layers, hidden_size: hidden, intermediate_size: intermediate,
        vocab_size: 200, embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse, dtype,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None, layers: layer_infos, down_top_k: 1,
        has_model_weights: false, model_config: None,
    }
}

fn make_synthetic_model() -> larql_models::ModelWeights {
    let (num_layers, hidden, intermediate, vocab_size) = (2, 8, 4, 16);
    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        let mut gate = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate { gate[[i, i % hidden]] = 1.0 + layer as f32; }
        tensors.insert(format!("layers.{layer}.mlp.gate_proj.weight"), gate.into_shared());

        let mut up = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate { up[[i, (i + 1) % hidden]] = 0.5; }
        tensors.insert(format!("layers.{layer}.mlp.up_proj.weight"), up.into_shared());

        let mut down = Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate { down[[i % hidden, i]] = 0.3; }
        tensors.insert(format!("layers.{layer}.mlp.down_proj.weight"), down.into_shared());

        for s in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut a = Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden { a[[i, i]] = 1.0; }
            tensors.insert(format!("layers.{layer}.self_attn.{s}.weight"), a.into_shared());
        }
        vectors.insert(format!("layers.{layer}.input_layernorm.weight"), vec![1.0; hidden]);
        vectors.insert(format!("layers.{layer}.post_attention_layernorm.weight"), vec![1.0; hidden]);
    }
    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    let mut embed = Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size { embed[[i, i % hidden]] = 1.0; }

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": hidden,
        "num_hidden_layers": num_layers, "intermediate_size": intermediate,
        "head_dim": hidden, "num_attention_heads": 1,
        "num_key_value_heads": 1, "rope_theta": 10000.0, "vocab_size": vocab_size,
    }));

    let embed = embed.into_shared();
    larql_models::ModelWeights {
        tensors, vectors, raw_bytes: std::collections::HashMap::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed: embed.clone(), lm_head: embed.clone(),
        num_layers, hidden_size: hidden, intermediate_size: intermediate, vocab_size,
        head_dim: hidden, num_q_heads: 1, num_kv_heads: 1, rope_base: 10000.0, arch,
    }
}
