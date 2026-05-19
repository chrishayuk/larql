//! Synthetic-safetensors fixtures for the streaming-extract MoE arms.
//!
//! Hand-built, deterministic, in-process — no HuggingFace, no large
//! model downloads. Each fixture writes a tempdir tree (`config.json` +
//! `tokenizer.json` + `model.safetensors`) shaped like a real
//! architecture and drives [`larql_vindex::build_vindex_streaming`]
//! against it. The point is to exercise the per-format arms inside
//! `extract::streaming::stages::*` that the dense Llama fixture in
//! `test_vindex.rs` doesn't reach:
//!
//! - `gate_vectors::write_gate_vectors` — standard MoE arm
//! - `down_meta::write_down_meta` — standard MoE arm
//! - `router_weights::write_router_weights` — whole body (early-returns
//!   on dense; only fires when `is_moe`)
//! - `index_json::write_index_json` — MoE config branch + has-experts
//!   per-layer tracking
//!
//! Single Mixtral-shaped happy path is enough to flip all four files
//! into "MoE arm exercised" territory.

use std::collections::HashMap;
use std::path::Path;

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, KquantWriteOptions, QuantFormat, SilentBuildCallbacks,
    StorageDtype, WriteWeightsOptions,
};

/// Build a tiny Mixtral-shaped model (block-sparse MoE FFN with
/// `num_experts` experts per layer). Deterministic per-tensor ramps
/// so two runs against the same dims produce identical vindexes.
///
/// Returns the in-memory tokenizer so callers can drive
/// `build_vindex_streaming` without re-reading the JSON file.
fn write_synthetic_mixtral_model(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
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

    // Embedding + final norm.
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard Llama-style attention.
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
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Block-sparse MoE: router + per-expert gate (w1) / down (w2) / up (w3).
        push(
            &format!("{lp}.block_sparse_moe.gate.weight"),
            vec![num_experts, hidden],
        );
        for e in 0..num_experts {
            let ep = format!("{lp}.block_sparse_moe.experts.{e}");
            push(&format!("{ep}.w1.weight"), vec![intermediate, hidden]);
            push(&format!("{ep}.w2.weight"), vec![hidden, intermediate]);
            push(&format!("{ep}.w3.weight"), vec![intermediate, hidden]);
        }
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

    // Minimal BPE tokenizer — enough for safetensors-backed extracts
    // that don't need to encode strings.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_exercises_moe_arms() {
    // Tiny dims chosen so each FFN row pads to a clean Q4_K boundary if
    // the test ever extends to quant=Q4K. For now we extract f32 at
    // Browse level — covers gate / down_meta / router / index_json.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-synthetic",
        &output_dir,
        5, // down_top_k
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false, // drop_gate_vectors
        &mut cb,
    )
    .expect("streaming extract on mixtral fixture");

    // ── Outputs the MoE arms must produce ───────────────────────
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(
        output_dir.join("router_weights.bin").exists(),
        "MoE arm must write router_weights.bin (router_weights.rs whole body)"
    );
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());
    assert!(output_dir.join("index.json").exists());

    // ── index.json carries MoE config (index_json.rs MoE branch) ──
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    let model_cfg = config.model_config.expect("model_config present");
    let moe = model_cfg.moe.expect("MoE config recorded");
    assert_eq!(moe.num_experts, num_experts);
    assert_eq!(moe.top_k, num_experts_per_tok);

    // ── layer_infos record per-expert geometry (gate_vectors arm) ──
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(layer_info.num_experts, Some(num_experts));
        assert_eq!(layer_info.num_features_per_expert, Some(intermediate));
        // Total = num_experts × intermediate.
        assert_eq!(layer_info.num_features, num_experts * intermediate);
    }

    // ── router_weights.bin shape: per-layer router (+ optional bias) ──
    // Each router is `num_experts × hidden` f32 = 4 floats × 4 bytes = 16 B.
    // Two layers → ≥ 32 B (more if biases happened to be present).
    let router_bytes = std::fs::metadata(output_dir.join("router_weights.bin"))
        .unwrap()
        .len();
    let min_expected = (num_layers * num_experts * hidden * 4) as u64;
    assert!(
        router_bytes >= min_expected,
        "router_weights.bin {router_bytes} B < expected {min_expected} B"
    );
}

// ─── Gemma 4 hybrid MoE (PackedBF16 expert format) ───────────────────

/// Build a tiny Gemma 4 26B-A4B-shaped hybrid MoE model: dense MLP +
/// per-layer expert block, with experts stored packed as the
/// `experts.gate_up_proj` / `experts.down_proj` BF16 tensor pair.
///
/// `extract::streaming::stages::gate_vectors` and `down_meta` both have
/// dedicated `PackedBF16 + is_moe` arms that route through the **dense**
/// MLP gate/down for KNN routing while leaving the packed tensors
/// untouched (the q4k writer consumes those later). This fixture
/// exercises that route end-to-end.
///
/// Note: the dense FFN keys overlap with Llama's, but the runtime
/// dispatch picks the PackedBF16 arm because Gemma4Arch advertises
/// `expert_format() == ExpertFormat::PackedBF16` whenever
/// `enable_moe_block=true`.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_gemma4_hybrid_moe(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    moe_intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();

    // Gemma 4 detection: model_type that starts with "gemma4". We use
    // a flat config (no `text_config` nesting) — `detect_from_json`
    // falls back to the top level when `text_config` is absent.
    let config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        // Hybrid MoE flag — flips Gemma4Arch.is_moe / is_hybrid_moe / expert_format.
        "enable_moe_block": true,
        "num_experts": num_experts,
        "top_k_experts": num_experts_per_token,
        "moe_intermediate_size": moe_intermediate,
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

    // Globals
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard Llama-style attention (Gemma 4 inherits this).
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
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        // Hybrid MoE renames post_feedforward_layernorm → _1 (dense
        // branch); the streaming pipeline doesn't need it for Browse
        // level, but the loader at the end will look for it.
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Dense MLP — both branches coexist in hybrid MoE; gate_vectors'
        // PackedBF16 arm reads from here.
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
        // Packed expert tensors — only consumed by the q4k writer at
        // QuantFormat::Q4K. With QuantFormat::None they're present but
        // unused by Browse-level extraction.
        push(
            &format!("{lp}.experts.gate_up_proj"),
            vec![num_experts, 2 * moe_intermediate, hidden],
        );
        push(
            &format!("{lp}.experts.down_proj"),
            vec![num_experts, hidden, moe_intermediate],
        );
        // Router (hybrid MoE: `router.proj` not `gate.weight`).
        push(
            &format!("{lp}.router.proj.weight"),
            vec![num_experts, hidden],
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
#[serial_test::serial]
fn streaming_extract_gemma4_hybrid_moe_exercises_packed_bf16_arms() {
    // Tiny dims; enable_moe_block flips Gemma4Arch into the hybrid
    // MoE configuration where expert_format == PackedBF16 and is_moe
    // is true, hitting the `PackedBF16 && is_moe` arms in
    // stages/gate_vectors.rs and stages/down_meta.rs.
    let hidden = 8usize;
    let intermediate = 4usize;
    let moe_intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_gemma4_hybrid_moe(
        &model_dir,
        hidden,
        intermediate,
        moe_intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/gemma4-hybrid-moe-synthetic",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("streaming extract on gemma4 hybrid MoE fixture");

    // Outputs the hybrid MoE arms must produce.
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(
        output_dir.join("router_weights.bin").exists(),
        "MoE arm must write router_weights.bin"
    );
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());

    // index.json carries hybrid-MoE config — exercises
    // `arch.is_hybrid_moe()` branch in write_index_json.
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.family, "gemma4");
    let model_cfg = config.model_config.expect("model_config present");
    let moe = model_cfg.moe.expect("MoE config recorded");
    assert!(moe.hybrid, "Gemma 4 26B A4B is hybrid MoE");
    assert_eq!(moe.num_experts, num_experts);
    assert_eq!(moe.top_k, num_experts_per_tok);
    assert_eq!(moe.moe_intermediate_size, Some(moe_intermediate));

    // The hybrid arm uses the dense FFN gate for routing (NOT per-expert
    // gate), so layer_infos.num_features should match the dense width
    // (`intermediate`), with no per-expert breakdown.
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(
            layer_info.num_features, intermediate,
            "hybrid MoE routes through dense gate (intermediate width)"
        );
        assert_eq!(layer_info.num_experts, None);
        assert_eq!(layer_info.num_features_per_expert, None);
    }
}

// ─── gpt-oss (PackedMxfp4 expert format) ─────────────────────────────

/// Build a tiny gpt-oss-shaped MoE model: experts packed as MXFP4
/// (e8m0 scales + 4-bit nibbles), gate and up projections fused into
/// one tensor pair (`gate_up_proj_blocks` + `gate_up_proj_scales`),
/// down projections in a separate pair.
///
/// `extract::streaming::stages::gate_vectors` and `down_meta` both have
/// dedicated `PackedMxfp4` arms that:
/// 1. Find the packed tensor pair via `arch.packed_gate_up_blocks_key` /
///    `arch.packed_down_blocks_key`.
/// 2. Deserialise safetensors directly to read the U8 byte payload.
/// 3. Call `format::quant::mxfp4::dequantize_all_experts` to recover
///    f32 expert matrices.
/// 4. (gate) Slice the first half of each expert's rows as the gate
///    portion and write to `gate_vectors.bin`.
/// 5. (down_meta) Use the recovered down matrices for embed-projection
///    top-K extraction.
///
/// Block byte payload is all-zero (MXFP4 nibble 0 dequantises to 0.0)
/// and scales are all `127` (e8m0 → scale=1.0). The decoded tensors
/// are therefore zero-filled but the *dispatch path* runs end-to-end,
/// which is what we're after for coverage.
///
/// MXFP4 constraint: `in_features = groups × 32`, so dimensions must
/// be multiples of 32. We use `hidden = intermediate = 32` (groups=1)
/// for the smallest possible payload.
fn write_synthetic_gpt_oss_model(
    model_dir: &Path,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    // Fixed dims chosen to satisfy MXFP4's `in_features = groups × 32`
    // constraint with the smallest-possible groups=1.
    let hidden = 32usize;
    let intermediate = 32usize;
    let groups = 1usize; // hidden / 32
    let groups_down = 1usize; // intermediate / 32
    let out_features_gate_up = 2 * intermediate; // fused gate+up: 64

    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_token,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Track tensors as either F32 or U8. Each entry is
    // (name, dtype, shape, raw_bytes).
    let mut entries: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let push_f32 = |entries: &mut Vec<_>, name: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        entries.push((name, safetensors::Dtype::F32, shape, bytes));
    };

    let push_u8 = |entries: &mut Vec<_>, name: String, shape: Vec<usize>, fill: u8| {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = vec![fill; n];
        entries.push((name, safetensors::Dtype::U8, shape, bytes));
    };

    // Globals
    push_f32(
        &mut entries,
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
    );
    push_f32(&mut entries, "model.norm.weight".into(), vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard attention.
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Router (gpt-oss: `mlp.router.weight`, NOT `block_sparse_moe.gate`).
        push_f32(
            &mut entries,
            format!("{lp}.mlp.router.weight"),
            vec![num_experts, hidden],
        );

        // Packed MXFP4 expert blocks (U8) + e8m0 scales (U8). All-zero
        // blocks → MXFP4_TABLE[0]=0.0 → zero-filled dequant. Scale 127
        // → e8m0 → 1.0. The dequantize path runs end-to-end either way.
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.gate_up_proj_blocks"),
            vec![num_experts, out_features_gate_up, groups, 16],
            0,
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.gate_up_proj_scales"),
            vec![num_experts, out_features_gate_up, groups],
            127, // e8m0 = 1.0
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.down_proj_blocks"),
            vec![num_experts, hidden, groups_down, 16],
            0,
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.down_proj_scales"),
            vec![num_experts, hidden, groups_down],
            127,
        );
    }

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = entries
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
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
#[serial_test::serial]
fn streaming_extract_gpt_oss_exercises_packed_mxfp4_arms() {
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_gpt_oss_model(
        &model_dir,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/gpt-oss-synthetic",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("streaming extract on gpt-oss MXFP4 fixture");

    // Outputs the PackedMxfp4 arm must produce.
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(output_dir.join("router_weights.bin").exists());
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.family, "gpt_oss");

    // num_features per layer = num_experts × (out_features_gate_up / 2)
    // = num_experts × intermediate (since out_features_gate_up = 2*intermediate).
    let intermediate = 32usize;
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(layer_info.num_experts, Some(num_experts));
        assert_eq!(layer_info.num_features_per_expert, Some(intermediate));
        assert_eq!(layer_info.num_features, num_experts * intermediate);
    }
}

// ─── LARQL_SUMMARY_FEATURES_PER_EXPERT path (gate_vectors SVD + down_meta cap) ──

/// RAII guard for the `LARQL_SUMMARY_FEATURES_PER_EXPERT` env var. The
/// summary tier is gated on a process-global env var, so tests reading
/// or writing it must serialise via `#[serial]` (and the var must be
/// cleared on drop so neighbouring tests aren't affected).
struct SummaryEnvGuard {
    prev: Option<String>,
}

impl SummaryEnvGuard {
    fn set(value: &str) -> Self {
        let prev = std::env::var("LARQL_SUMMARY_FEATURES_PER_EXPERT").ok();
        std::env::set_var("LARQL_SUMMARY_FEATURES_PER_EXPERT", value);
        Self { prev }
    }
}

impl Drop for SummaryEnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var("LARQL_SUMMARY_FEATURES_PER_EXPERT", v),
            None => std::env::remove_var("LARQL_SUMMARY_FEATURES_PER_EXPERT"),
        }
    }
}

/// Build a Mixtral fixture with a tokenizer that has a non-empty vocab,
/// so `down_meta` actually decodes some `token_id → string` and exercises
/// the `TopKEntry` construction branch (lines 200-204) instead of having
/// every decode return an empty string.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_mixtral_model_with_real_tokenizer(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    // Reuse the model-side fixture (config + safetensors); only the
    // tokenizer side gets the richer vocab.
    let _ = write_synthetic_mixtral_model(
        model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    // Populate the BPE `vocab` map directly so `decode(&[id], true)`
    // (which skips special tokens) still returns the printable token
    // string for every ID in 0..min(vocab, 8). IDs beyond that decode
    // to empty and exercise the `.filter(|s| !s.is_empty())` skip path.
    let vocab_entries: Vec<String> = (0..(vocab.min(8)))
        .map(|i| format!("\"tok{i}\":{i}"))
        .collect();
    let tok_json = format!(
        r#"{{"version":"1.0","model":{{"type":"BPE","vocab":{{{}}},"merges":[]}},"added_tokens":[]}}"#,
        vocab_entries.join(",")
    );
    std::fs::write(model_dir.join("tokenizer.json"), &tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_resumes_when_run_twice_on_same_output_dir() {
    // First-pass extract writes outputs + clears the checkpoint on
    // success. To exercise the resumed_* branches at the head of each
    // streaming stage, we pre-seed the output dir with a checkpoint
    // marking every phase complete before the second pass — then re-run.
    // Each stage's `resumed_*` early-return branch fires, and the
    // pre-existing output files are left untouched.
    use larql_vindex::extract::{Checkpoint, ExtractPhase};

    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    // ── First pass: build the full vindex (and have all output files
    //    on disk so the resume path doesn't choke).
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-resume",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("first-pass extract");
    let first_down_meta_size = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();

    // ── Seed a checkpoint that marks every phase complete. This is the
    //    on-disk state an extractor would observe after crashing right
    //    before the final cleanup step. Forces the second pass to hit
    //    the resumed_* branches in every stage.
    let mut cp = Checkpoint::default();
    for phase in [ExtractPhase::Gate, ExtractPhase::DownMeta] {
        cp.mark(phase, &output_dir).expect("seed checkpoint");
    }
    assert!(output_dir.join(".extract_checkpoint.json").exists());

    // ── Second pass on the SAME output_dir — hits resumed_* paths.
    let mut cb2 = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-resume",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb2,
    )
    .expect("second-pass extract reuses checkpoint");
    let second_down_meta_size = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();
    // Resumed phases don't re-write the file, so size is identical.
    assert_eq!(first_down_meta_size, second_down_meta_size);
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_with_real_tokenizer_records_top_k_entries() {
    // Same fixture as the baseline, but with a richer tokenizer so
    // down_meta can actually decode token IDs to strings. Exercises the
    // `TopKEntry` construction inside `down_meta::write_down_meta`'s
    // inner loop (lines 192-213) that was previously skipped because
    // every decode returned an empty string.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model_with_real_tokenizer(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-real-tok",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("streaming extract with real tokenizer");

    assert!(output_dir.join("down_meta.bin").exists());
    let bytes = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();
    assert!(
        bytes > 0,
        "down_meta.bin must be non-empty when at least one feature decodes a token"
    );
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_with_drop_gate_vectors_removes_zero_byte_file() {
    // `drop_gate_vectors: true` requires `quant: Q4K` (the gate is
    // rebuilt from Q4K weights at load time). The streaming extractor
    // still walks the gate loop to populate `layer_infos` for
    // index.json, but pipes bytes to /dev/null and removes the
    // zero-byte gate_vectors.bin afterward.
    //
    // This exercises the `GateSink::Discard` path + the cleanup at
    // the end of `write_gate_vectors`.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-drop-gate",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::Q4K, // required by drop_gate_vectors
        WriteWeightsOptions {
            level: ExtractLevel::Browse,
            ffn_compact: false,
        },
        KquantWriteOptions::default(),
        true, // drop_gate_vectors
        &mut cb,
    )
    .expect("streaming extract with drop_gate_vectors=true");

    // gate_vectors.bin should have been removed (was zero bytes).
    assert!(
        !output_dir.join("gate_vectors.bin").exists(),
        "drop_gate_vectors=true must clean up the empty file"
    );

    // index.json still records per-layer geometry (the gate loop walked
    // every layer to populate layer_infos).
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), num_layers);
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_with_summary_k_runs_svd_and_caps_down_meta() {
    // Same Mixtral fixture as the baseline test, but with
    // `LARQL_SUMMARY_FEATURES_PER_EXPERT=2`. Triggers:
    //   - the SVD-summary path in `gate_vectors.rs` Standard-MoE branch
    //     (writes K rows per expert instead of full intermediate=4)
    //   - the down_meta `summary_k`-cap branch (truncates `num_features`
    //     to K=2 per expert)
    //
    // Output assertions confirm both paths fired by checking that
    // num_features_per_expert collapses from intermediate=4 to K=2 in
    // the recorded layer_infos.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;
    let summary_k = 2usize;

    let _env = SummaryEnvGuard::set(&summary_k.to_string());

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-summary-k",
        &output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("streaming extract on mixtral fixture with summary_k");

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        // SVD path: num_features_per_expert collapses to K, regardless
        // of original intermediate.
        assert_eq!(layer_info.num_features_per_expert, Some(summary_k));
        assert_eq!(layer_info.num_features, num_experts * summary_k);
    }

    // gate_vectors.bin now stores K floats per expert per layer instead
    // of `intermediate`. Same hidden width (=8 floats/row).
    let gate_bytes = std::fs::metadata(output_dir.join("gate_vectors.bin"))
        .unwrap()
        .len();
    let expected = (num_layers * num_experts * summary_k * hidden * 4) as u64;
    assert_eq!(
        gate_bytes, expected,
        "gate_vectors.bin sized for K rows × hidden, not intermediate"
    );
}
