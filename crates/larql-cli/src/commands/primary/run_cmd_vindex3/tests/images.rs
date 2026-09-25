//! Exercise real image decoding, checkpoint loaders, vision forward and projector.
//! The language-model input seam has separate numerical parity tests in inference.
use super::super::{inputs, ResidentModel};
use super::*;
use larql_inference::vindex3::{input::InputPosition, Vindex3Runtime};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

fn vision_checkpoint(dir: &Path, hidden: usize) {
    let config = serde_json::json!({"model_type":"gemma3","text_config":{"model_type":"gemma3_text","hidden_size":hidden,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":256001},"vision_config":{"hidden_size":4,"intermediate_size":8,"num_attention_heads":2,"num_hidden_layers":1,"patch_size":1,"image_size":16,"num_channels":3,"layer_norm_eps":1e-6,"hidden_act":"gelu_pytorch_tanh"}});
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, ones: bool| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .flat_map(|i| {
                let v = if ones {
                    1.0f32
                } else {
                    ((i * 17 + 3) % 41) as f32 / 41.0 - 0.5
                };
                v.to_le_bytes()
            })
            .collect();
        tensors.push((name, shape, data));
    };
    let prefix = "vision_tower.vision_model.";
    add(
        format!("{prefix}embeddings.patch_embedding.weight"),
        vec![4, 3, 1, 1],
        false,
    );
    add(
        format!("{prefix}embeddings.patch_embedding.bias"),
        vec![4],
        false,
    );
    add(
        format!("{prefix}embeddings.position_embedding.weight"),
        vec![256, 4],
        false,
    );
    for norm in [
        "post_layernorm",
        "encoder.layers.0.layer_norm1",
        "encoder.layers.0.layer_norm2",
    ] {
        add(format!("{prefix}{norm}.weight"), vec![4], true);
        add(format!("{prefix}{norm}.bias"), vec![4], false);
    }
    for (proj, out, input) in [
        ("self_attn.q_proj", 4, 4),
        ("self_attn.k_proj", 4, 4),
        ("self_attn.v_proj", 4, 4),
        ("self_attn.out_proj", 4, 4),
        ("mlp.fc1", 8, 4),
        ("mlp.fc2", 4, 8),
    ] {
        add(
            format!("{prefix}encoder.layers.0.{proj}.weight"),
            vec![out, input],
            false,
        );
        add(
            format!("{prefix}encoder.layers.0.{proj}.bias"),
            vec![out],
            false,
        );
    }
    add(
        "multi_modal_projector.mm_input_projection_weight".into(),
        vec![4, hidden],
        false,
    );
    add(
        "multi_modal_projector.mm_soft_emb_norm.weight".into(),
        vec![4],
        false,
    );
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, shape, data)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                    .unwrap(),
            )
        })
        .collect();
    safetensors::tensor::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn gemma3_vision_source_produces_distinct_finite_unscaled_prefixes() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let runtime = Vindex3Runtime::open(&container, "target", ProductionBackend::new())
        .unwrap()
        .prepare()
        .unwrap();
    let source = tempfile::tempdir().unwrap();
    vision_checkpoint(source.path(), runtime.operands().hidden());
    let first = root.path().join("red.png");
    let second = root.path().join("blue.png");
    image::RgbImage::from_pixel(16, 16, image::Rgb([255, 0, 0]))
        .save(&first)
        .unwrap();
    image::RgbImage::from_pixel(16, 16, image::Rgb([0, 0, 255]))
        .save(&second)
        .unwrap();
    let mut args = args(&[container.to_str().unwrap(), PROMPT]);
    args.mm_weights = Some(source.path().to_path_buf());
    args.image = vec![first, second];
    let tokenizer = larql_vindex::load_vindex_tokenizer(&container).unwrap();
    let eos = larql_inference::EosConfig::builtin();
    let model = ResidentModel {
        container: &container,
        family: "gemma3",
        plan: runtime.plan(),
        ops: runtime.operands(),
        backend: runtime.backend(),
        tokenizer: &tokenizer,
        eos: &eos,
        engine: "test",
        args: &args,
        continuations: &larql_kv::shipped_continuations(),
    };
    let rows = inputs::image_inputs(&model, &[1, 2]).unwrap();
    assert_eq!(rows.len(), 2 * 258 + 2);
    assert!(matches!(rows[0], InputPosition::Token(255999)));
    assert!(matches!(rows[257], InputPosition::Token(256000)));
    assert!(matches!(rows[516], InputPosition::Token(1)));
    let vectors: Vec<_> = rows
        .iter()
        .filter_map(|r| {
            if let InputPosition::Embedding(v) = r {
                Some(v)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(vectors.len(), 512);
    assert!(vectors
        .iter()
        .all(|r| r.len() == runtime.operands().hidden() && r.iter().all(|v| v.is_finite())));
    assert_ne!(
        vectors[0], vectors[256],
        "different pixels must reach the language-model input"
    );
    let wrong = ResidentModel {
        family: "llama",
        ..model
    };
    assert!(inputs::image_inputs(&wrong, &[1]).is_err());
}
