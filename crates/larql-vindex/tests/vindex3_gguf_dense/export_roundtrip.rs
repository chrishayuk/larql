//! The miniature Gemma 3 container exported and read back through
//! LARQL's own GGUF reader: names, order, shapes, types, metadata,
//! tokenizer arrays and tensor bytes.

use larql_models::loading::gguf::{GgufFile, GgufValue};
use larql_models::quant::half::bf16_to_f32;
use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::fixtures::*;
use larql_vindex::format::vindex3::gguf::dense::spm_vocab::ScoreSource;
use larql_vindex::format::vindex3::gguf::dense::{export_dense, ContextSource, DenseExportOptions};

use super::common::{meta, strings, tensor_bytes, Fixture};
use super::sentencepiece::encode_model;

const TYPE_F32: u32 = 0;
const TYPE_BF16: u32 = 30;
const PER_LAYER: usize = 13;

/// The segment bytes of `object/tensor` in the container.
fn segment_bytes(fx: &Fixture, object: &str, tensor: &str) -> Vec<u8> {
    let path = fx.container.join(format!("segments/{object}.bin"));
    let (header, start) = read_segment_header(&path).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    let raw = std::fs::read(&path).unwrap();
    raw[(start + t.offset) as usize..(start + t.offset + t.len) as usize].to_vec()
}

#[test]
fn every_tensor_leaves_named_shaped_typed_and_in_reference_order() {
    let fx = Fixture::new();
    let (report, out) = fx.export("model.gguf");
    let f = GgufFile::open(&out).unwrap();
    let names: Vec<&str> = f.tensor_infos.iter().map(|t| t.name()).collect();
    assert_eq!(names.len(), 2 + GEMMA3_FIXTURE_LAYERS * PER_LAYER);
    assert_eq!(report.tensors, names.len());
    assert_eq!(names.first(), Some(&"token_embd.weight"));
    assert_eq!(names.last(), Some(&"output_norm.weight"));
    assert!(
        !names.contains(&"output.weight"),
        "a tied head emits no output.weight"
    );
    // Layers ascend, and within a layer the segment's sorted order holds.
    assert_eq!(names[1], "blk.0.attn_norm.weight");
    assert_eq!(names[1 + PER_LAYER], "blk.1.attn_norm.weight");
    for t in &f.tensor_infos {
        let norm = t.name().ends_with("_norm.weight");
        assert_eq!(
            t.tensor_type(),
            if norm { TYPE_F32 } else { TYPE_BF16 },
            "{}",
            t.name()
        );
    }
    let q = f
        .tensor_infos
        .iter()
        .find(|t| t.name() == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(
        q.dims(),
        &[
            GEMMA3_FIXTURE_HIDDEN as u64,
            (GEMMA3_FIXTURE_Q_HEADS * GEMMA3_FIXTURE_HEAD_DIM) as u64
        ]
    );
    let embd = f
        .tensor_infos
        .iter()
        .find(|t| t.name() == "token_embd.weight")
        .unwrap();
    assert_eq!(
        embd.dims(),
        &[GEMMA3_FIXTURE_HIDDEN as u64, GEMMA3_FIXTURE_VOCAB as u64]
    );
    assert_eq!(report.bf16_tensors, 1 + GEMMA3_FIXTURE_LAYERS * 7);
    assert_eq!(report.verify.tensors, names.len());
    assert!(report.excluded_components.is_empty());
}

#[test]
fn matrices_pass_through_and_norms_carry_their_declared_offset() {
    let fx = Fixture::new();
    let (_, out) = fx.export("model.gguf");
    let f = GgufFile::open(&out).unwrap();
    let embedding = segment_bytes(&fx, "target.embedding", "weight");
    assert_eq!(
        tensor_bytes(&f, &out, "token_embd.weight", embedding.len()),
        embedding
    );
    let q = segment_bytes(&fx, "target.decoder_stack", "3.self_attn.q_proj.weight");
    assert_eq!(tensor_bytes(&f, &out, "blk.3.attn_q.weight", q.len()), q);

    for (object, tensor, name) in [
        (
            "target.decoder_stack",
            "2.input_layernorm.weight",
            "blk.2.attn_norm.weight",
        ),
        (
            "target.decoder_stack",
            "2.post_feedforward_layernorm.weight",
            "blk.2.post_ffw_norm.weight",
        ),
        (
            "target.decoder_stack",
            "4.self_attn.k_norm.weight",
            "blk.4.attn_k_norm.weight",
        ),
        ("target.final_norm", "weight", "output_norm.weight"),
    ] {
        let src = segment_bytes(&fx, object, tensor);
        let want: Vec<u8> = src
            .chunks_exact(2)
            .flat_map(|c| (bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) + 1.0).to_le_bytes())
            .collect();
        assert_eq!(tensor_bytes(&f, &out, name, want.len()), want, "{name}");
    }
}

#[test]
fn metadata_states_the_graph() {
    let fx = Fixture::new();
    let (report, out) = fx.export("model.gguf");
    let f = GgufFile::open(&out).unwrap();
    let u = |k: &str| meta(&f, k).as_u32().unwrap();
    let fl = |k: &str| meta(&f, k).as_f64().unwrap();
    assert_eq!(meta(&f, "general.architecture").as_str(), Some("gemma3"));
    assert_eq!(meta(&f, "general.type").as_str(), Some("model"));
    assert_eq!(meta(&f, "general.name").as_str(), Some("Gemma 3 Tiny It"));
    assert_eq!(
        meta(&f, "general.sampling.top_k"),
        &GgufValue::I32(GEMMA3_FIXTURE_TOP_K as i32)
    );
    assert_eq!(
        meta(&f, "general.sampling.top_p"),
        &GgufValue::F32(GEMMA3_FIXTURE_TOP_P as f32)
    );
    assert_eq!(u("gemma3.block_count") as usize, GEMMA3_FIXTURE_LAYERS);
    assert_eq!(u("gemma3.context_length") as usize, GEMMA3_FIXTURE_CONTEXT);
    assert_eq!(report.context_source, ContextSource::Graph);
    assert_eq!(u("gemma3.embedding_length") as usize, GEMMA3_FIXTURE_HIDDEN);
    assert_eq!(
        u("gemma3.feed_forward_length") as usize,
        GEMMA3_FIXTURE_INTER
    );
    assert_eq!(
        u("gemma3.attention.head_count") as usize,
        GEMMA3_FIXTURE_Q_HEADS
    );
    assert_eq!(
        u("gemma3.attention.head_count_kv") as usize,
        GEMMA3_FIXTURE_KV_HEADS
    );
    assert_eq!(
        u("gemma3.attention.key_length") as usize,
        GEMMA3_FIXTURE_HEAD_DIM
    );
    assert_eq!(
        u("gemma3.attention.sliding_window") as usize,
        GEMMA3_FIXTURE_WINDOW
    );
    assert_eq!(
        meta(&f, "gemma3.rope.scaling.type").as_str(),
        Some("linear")
    );
    assert_eq!(fl("gemma3.rope.scaling.factor"), GEMMA3_FIXTURE_ROPE_FACTOR);
    assert_eq!(fl("gemma3.rope.freq_base"), GEMMA3_FIXTURE_GLOBAL_THETA);
    assert_eq!(fl("gemma3.rope.freq_base_swa"), GEMMA3_FIXTURE_LOCAL_THETA);
    assert_eq!(
        meta(&f, "gemma3.attention.layer_norm_rms_epsilon"),
        &GgufValue::F32(GEMMA3_FIXTURE_EPS as f32)
    );
    assert!(!f
        .metadata
        .contains_key("gemma3.attention.sliding_window_pattern"));
    assert_eq!(u("general.file_type"), 32);
    assert_eq!(u("general.quantization_version"), 2);
}

#[test]
fn the_tokenizer_arrays_reach_the_file() {
    let fx = Fixture::new();
    let (report, out) = fx.export("model.gguf");
    let f = GgufFile::open(&out).unwrap();
    assert_eq!(meta(&f, "tokenizer.ggml.model").as_str(), Some("llama"));
    assert_eq!(meta(&f, "tokenizer.ggml.pre").as_str(), Some("default"));
    let tokens = strings(meta(&f, "tokenizer.ggml.tokens"));
    assert_eq!(tokens.len(), GEMMA3_FIXTURE_VOCAB);
    assert_eq!(
        &tokens[..GEMMA3_FIXTURE_PIECES.len() - 3],
        &GEMMA3_FIXTURE_PIECES[..14]
    );
    assert_eq!(
        tokens[14], "  ",
        "a user-defined token's ▁ is pre-normalised to a space"
    );
    assert_eq!(
        tokens[GEMMA3_FIXTURE_VOCAB - 1],
        format!("[PAD{}]", GEMMA3_FIXTURE_VOCAB - 1)
    );
    let GgufValue::Array(types) = meta(&f, "tokenizer.ggml.token_type") else {
        panic!()
    };
    let types: Vec<i32> = types
        .iter()
        .map(|t| match t {
            GgufValue::I32(v) => *v,
            _ => panic!(),
        })
        .collect();
    // pad/eos/bos/unk control; bytes; normal; control turn markers;
    // user-defined; normal; `<unused0>` looks special; padding unused.
    assert_eq!(
        &types[..17],
        &[3, 3, 3, 3, 6, 6, 6, 1, 1, 1, 1, 1, 3, 3, 4, 1, 3]
    );
    assert!(types[17..].iter().all(|t| *t == 5));
    let GgufValue::Array(scores) = meta(&f, "tokenizer.ggml.scores") else {
        panic!()
    };
    let scores: Vec<f32> = scores
        .iter()
        .map(|s| match s {
            GgufValue::F32(v) => *v,
            _ => panic!(),
        })
        .collect();
    assert_eq!(scores[0], -1000.0);
    assert_eq!(scores[4], 0.0);
    assert_eq!((scores[7], scores[8], scores[15]), (-0.0, -1.0, -5.0));
    assert_eq!(scores[GEMMA3_FIXTURE_VOCAB - 1], -10000.0);
    assert_eq!(report.score_source, ScoreSource::DerivedFromVocabularyOrder);
    assert_eq!(
        report.vocab_padded,
        GEMMA3_FIXTURE_VOCAB - GEMMA3_FIXTURE_PIECES.len()
    );
    for (k, v) in [
        ("tokenizer.ggml.bos_token_id", 2),
        ("tokenizer.ggml.eos_token_id", 1),
        ("tokenizer.ggml.unknown_token_id", 3),
        ("tokenizer.ggml.padding_token_id", 0),
    ] {
        assert_eq!(meta(&f, k).as_u32(), Some(v), "{k}");
    }
    assert_eq!(
        meta(&f, "tokenizer.ggml.add_bos_token"),
        &GgufValue::Bool(true)
    );
    assert_eq!(
        meta(&f, "tokenizer.ggml.add_sep_token"),
        &GgufValue::Bool(false)
    );
    assert_eq!(
        meta(&f, "tokenizer.ggml.add_eos_token"),
        &GgufValue::Bool(false)
    );
    assert_eq!(
        meta(&f, "tokenizer.ggml.add_space_prefix"),
        &GgufValue::Bool(false)
    );
    assert_eq!(
        meta(&f, "tokenizer.chat_template").as_str(),
        Some(GEMMA3_FIXTURE_TEMPLATE)
    );
}

#[test]
fn a_sentencepiece_model_supplies_the_scores() {
    let fx = Fixture::new();
    let pieces: Vec<(&str, f32, u64)> = GEMMA3_FIXTURE_PIECES
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                *p,
                -(i as f32) * 0.5,
                if p.starts_with("<0x") { 6 } else { 1 },
            )
        })
        .collect();
    let model = fx.out("tokenizer.model");
    std::fs::write(&model, encode_model(&pieces)).unwrap();
    let out = fx.out("model.gguf");
    let report = export_dense(
        &fx.container,
        &out,
        &DenseExportOptions {
            sentencepiece_model: Some(model.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.score_source, ScoreSource::SentencePieceModel);
    assert_eq!(report.sentencepiece_model.as_deref(), Some(model.as_path()));
    let f = GgufFile::open(&out).unwrap();
    let GgufValue::Array(scores) = meta(&f, "tokenizer.ggml.scores") else {
        panic!()
    };
    assert_eq!(scores[8], GgufValue::F32(-4.0));
    assert_eq!(scores[15], GgufValue::F32(-7.5));

    // A model the container carries wins over none at all.
    std::fs::copy(&model, fx.container.join("tokenizer.model")).unwrap();
    let carried = export_dense(
        &fx.container,
        &fx.out("carried.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap();
    assert_eq!(carried.score_source, ScoreSource::SentencePieceModel);
}

#[test]
fn a_container_without_a_context_length_states_the_family_default() {
    let fx = Fixture::new();
    fx.edit_json("system_graph.json", |g| {
        for c in g["components"].as_array_mut().unwrap() {
            c["execution"]
                .as_object_mut()
                .unwrap()
                .remove("context_length");
        }
    });
    let (report, out) = fx.export("model.gguf");
    assert_eq!(report.context_source, ContextSource::TargetDefault);
    let f = GgufFile::open(&out).unwrap();
    assert_eq!(meta(&f, "gemma3.context_length").as_u32(), Some(131_072));
}
