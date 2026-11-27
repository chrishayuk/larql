//! A GPT-Neo miniature in Hugging Face's own `GPTNeoForCausalLM` layout:
//! `nn.Linear` projections (`[out, in]`), unbiased separate Q/K/V under
//! `attn.attention`, a biased `out_proj`, LayerNorm with bias, a non-gated
//! biased MLP, learned absolute positions, a head tied to `wte`, and an
//! `attention_types` interleave of `global` and `local` layers whose local
//! window is shorter than the fixture's token count — so a local layer
//! executed as a global one cannot pass.
//!
//! [`GptNeoMiniature`] holds the stored tensors for an independent
//! forward.

use std::path::Path;

use super::fixtures::{lcg_values, norm_values, ShardBuilder};

pub const NEO_HIDDEN: usize = 16;
pub const NEO_LAYERS: usize = 2;
pub const NEO_HEADS: usize = 4;
pub const NEO_HEAD_DIM: usize = NEO_HIDDEN / NEO_HEADS;
pub const NEO_INNER: usize = 4 * NEO_HIDDEN;
pub const NEO_VOCAB: usize = 32;
pub const NEO_POSITIONS: usize = 32;
/// Shorter than [`NEO_TOKENS`], so the local layer's window binds.
pub const NEO_WINDOW: usize = 3;
pub const NEO_EPS: f64 = 1e-5;
pub const NEO_TOKENS: [u32; 10] = [3, 17, 28, 0, 11, 5, 9, 2, 30, 7];
/// Layer 0 is `global`, layer 1 `local`.
pub const NEO_LAYER_KINDS: [&str; NEO_LAYERS] = ["global", "local"];
/// Unscaled scores need larger projections to make attention non-uniform
/// enough that a window, or a scale, shows.
const ATTENTION_SCALE: f32 = 8.0;
const POSITION_SCALE: f32 = 10.0;

pub struct GptNeoLayer {
    pub ln_1_w: Vec<f32>,
    pub ln_1_b: Vec<f32>,
    /// `[hidden, hidden]` each, output-major.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub out: Vec<f32>,
    pub out_b: Vec<f32>,
    pub ln_2_w: Vec<f32>,
    pub ln_2_b: Vec<f32>,
    /// `[inner, hidden]`.
    pub c_fc_w: Vec<f32>,
    pub c_fc_b: Vec<f32>,
    /// `[hidden, inner]`.
    pub c_proj_w: Vec<f32>,
    pub c_proj_b: Vec<f32>,
}

pub struct GptNeoMiniature {
    pub wte: Vec<f32>,
    pub wpe: Vec<f32>,
    pub layers: Vec<GptNeoLayer>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
}

/// One miniature's deliberate variations from the plain model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptNeoForm {
    /// Also serialise the expanded `attention_layers`, as `save_pretrained`
    /// does.
    pub attention_layers: bool,
    /// Make `attention_layers` disagree with `attention_types`.
    pub contradictory_layers: bool,
    /// Ship a `q_proj.bias`, which GPT-Neo does not have.
    pub q_bias: bool,
}

impl Default for GptNeoForm {
    fn default() -> Self {
        Self {
            attention_layers: true,
            contradictory_layers: false,
            q_bias: false,
        }
    }
}

impl GptNeoMiniature {
    pub fn seeded() -> Self {
        let h = NEO_HIDDEN;
        let scaled = |n: usize, seed: u64| -> Vec<f32> {
            lcg_values(n, seed)
                .into_iter()
                .map(|v| v * ATTENTION_SCALE)
                .collect()
        };
        let layers = (0..NEO_LAYERS)
            .map(|layer| {
                let s = 100 + layer as u64 * 100;
                GptNeoLayer {
                    ln_1_w: norm_values(h, s),
                    ln_1_b: lcg_values(h, s + 1),
                    q: scaled(h * h, s + 2),
                    k: scaled(h * h, s + 3),
                    v: lcg_values(h * h, s + 4),
                    out: lcg_values(h * h, s + 5),
                    out_b: lcg_values(h, s + 6),
                    ln_2_w: norm_values(h, s + 7),
                    ln_2_b: lcg_values(h, s + 8),
                    c_fc_w: lcg_values(NEO_INNER * h, s + 9),
                    c_fc_b: lcg_values(NEO_INNER, s + 10),
                    c_proj_w: lcg_values(h * NEO_INNER, s + 11),
                    c_proj_b: lcg_values(h, s + 12),
                }
            })
            .collect();
        Self {
            wte: lcg_values(NEO_VOCAB * h, 1),
            wpe: lcg_values(NEO_POSITIONS * h, 2)
                .into_iter()
                .map(|v| v * POSITION_SCALE)
                .collect(),
            layers,
            ln_f_w: norm_values(h, 3),
            ln_f_b: lcg_values(h, 4),
        }
    }
}

pub fn miniature_gpt_neo(dir: &Path) -> GptNeoMiniature {
    miniature_gpt_neo_with(dir, GptNeoForm::default())
}

pub fn miniature_gpt_neo_with(dir: &Path, form: GptNeoForm) -> GptNeoMiniature {
    let mut config = serde_json::json!({
        "architectures": ["GPTNeoForCausalLM"],
        "model_type": "gpt_neo",
        "activation_function": "gelu_new",
        "attention_dropout": 0,
        "attention_types": [[NEO_LAYER_KINDS, 1]],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "embed_dropout": 0,
        "gradient_checkpointing": false,
        "hidden_size": NEO_HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": null,
        "layer_norm_epsilon": NEO_EPS,
        "max_position_embeddings": NEO_POSITIONS,
        "num_heads": NEO_HEADS,
        "num_layers": NEO_LAYERS,
        "resid_dropout": 0,
        "summary_activation": null,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": true,
        "summary_type": "cls_index",
        "summary_use_proj": true,
        "use_cache": true,
        "vocab_size": NEO_VOCAB,
        "window_size": NEO_WINDOW
    });
    if form.attention_layers {
        let mut layers = NEO_LAYER_KINDS.to_vec();
        if form.contradictory_layers {
            layers.reverse();
        }
        config["attention_layers"] = serde_json::json!(layers);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let model = GptNeoMiniature::seeded();
    let h = NEO_HIDDEN;
    let mut shard = ShardBuilder::new();
    shard.push("transformer.wte.weight", &[NEO_VOCAB, h], &model.wte);
    shard.push("transformer.wpe.weight", &[NEO_POSITIONS, h], &model.wpe);
    shard.push("transformer.ln_f.weight", &[h], &model.ln_f_w);
    shard.push("transformer.ln_f.bias", &[h], &model.ln_f_b);
    for (i, l) in model.layers.iter().enumerate() {
        let b = format!("transformer.h.{i}");
        let a = format!("{b}.attn.attention");
        shard.push(&format!("{b}.ln_1.weight"), &[h], &l.ln_1_w);
        shard.push(&format!("{b}.ln_1.bias"), &[h], &l.ln_1_b);
        shard.push(&format!("{a}.q_proj.weight"), &[h, h], &l.q);
        shard.push(&format!("{a}.k_proj.weight"), &[h, h], &l.k);
        shard.push(&format!("{a}.v_proj.weight"), &[h, h], &l.v);
        shard.push(&format!("{a}.out_proj.weight"), &[h, h], &l.out);
        shard.push(&format!("{a}.out_proj.bias"), &[h], &l.out_b);
        if form.q_bias {
            shard.push(&format!("{a}.q_proj.bias"), &[h], &lcg_values(h, 9));
        }
        shard.push(&format!("{b}.ln_2.weight"), &[h], &l.ln_2_w);
        shard.push(&format!("{b}.ln_2.bias"), &[h], &l.ln_2_b);
        shard.push(&format!("{b}.mlp.c_fc.weight"), &[NEO_INNER, h], &l.c_fc_w);
        shard.push(&format!("{b}.mlp.c_fc.bias"), &[NEO_INNER], &l.c_fc_b);
        shard.push(
            &format!("{b}.mlp.c_proj.weight"),
            &[h, NEO_INNER],
            &l.c_proj_w,
        );
        shard.push(&format!("{b}.mlp.c_proj.bias"), &[h], &l.c_proj_b);
    }
    shard.write(dir);
    model
}
