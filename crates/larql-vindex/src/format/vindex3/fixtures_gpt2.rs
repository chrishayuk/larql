//! A GPT-2 miniature in Hugging Face's own `GPT2LMHeadModel` layout:
//! `transformers.Conv1D` projections stored `[in, out]`, Q/K/V fused into
//! one `attn.c_attn` (bias fused too), LayerNorm with bias at every site, a
//! non-gated biased MLP, a learned absolute position table `wpe`, the head
//! tied to `wte`, and the registered causal-mask buffers older conversions
//! serialise (`attn.bias`, `attn.masked_bias`).
//!
//! [`Gpt2Miniature`] holds the raw tensors exactly as stored, so a test can
//! run an independent forward over them and hold the generic plan to it.
//! The variants take away or change exactly one declared fact.

use std::path::Path;

use super::fixtures::{lcg_values, norm_values, ShardBuilder};

pub const GPT2_HIDDEN: usize = 16;
pub const GPT2_LAYERS: usize = 2;
pub const GPT2_HEADS: usize = 4;
pub const GPT2_HEAD_DIM: usize = GPT2_HIDDEN / GPT2_HEADS;
pub const GPT2_INNER: usize = 4 * GPT2_HIDDEN;
pub const GPT2_VOCAB: usize = 32;
pub const GPT2_POSITIONS: usize = 16;
pub const GPT2_EPS: f64 = 1e-5;
pub const GPT2_TOKENS: [u32; 8] = [3, 17, 28, 0, 11, 5, 9, 2];
/// The seeded values sit in ±0.05; the position table is scaled up so a
/// dropped or mis-indexed position row cannot hide under rounding.
const POSITION_SCALE: f32 = 10.0;
/// What `GPT2Attention` registers as `masked_bias`.
pub const GPT2_MASK_FILL: f32 = -1e4;

/// One layer's stored tensors, in the checkpoint's own layout.
pub struct Gpt2Layer {
    pub ln_1_w: Vec<f32>,
    pub ln_1_b: Vec<f32>,
    /// `[hidden, 3·hidden]`, input-major.
    pub c_attn_w: Vec<f32>,
    pub c_attn_b: Vec<f32>,
    /// `[hidden, hidden]`, input-major.
    pub attn_c_proj_w: Vec<f32>,
    pub attn_c_proj_b: Vec<f32>,
    pub ln_2_w: Vec<f32>,
    pub ln_2_b: Vec<f32>,
    /// `[hidden, inner]`, input-major.
    pub c_fc_w: Vec<f32>,
    pub c_fc_b: Vec<f32>,
    /// `[inner, hidden]`, input-major.
    pub mlp_c_proj_w: Vec<f32>,
    pub mlp_c_proj_b: Vec<f32>,
}

/// The whole miniature, as stored.
pub struct Gpt2Miniature {
    pub wte: Vec<f32>,
    pub wpe: Vec<f32>,
    pub layers: Vec<Gpt2Layer>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
    /// What the checkpoint declares for `scale_attn_weights`.
    pub scale_attn_weights: Option<bool>,
}

/// One miniature's deliberate variations from the plain model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpt2Form {
    /// Tensor-name prefix: `"transformer."` (what `save_pretrained`
    /// writes) or `""` (the original OpenAI conversion, `h.0.…`).
    pub prefix: &'static str,
    /// `scale_attn_weights`, when declared.
    pub scale_attn_weights: Option<bool>,
    /// Declare `scale_attn_by_inverse_layer_idx: true`, which this build
    /// represents only as off.
    pub inverse_layer_idx: bool,
    /// Ship the registered mask buffers beside the parameters.
    pub buffers: bool,
    /// Store a mask that is NOT the causal one (key `i + 1` visible to
    /// query `i` on layer 0) — a buffer the declaration does not reproduce.
    pub corrupt_mask: bool,
    /// The stored `masked_bias` fill.
    pub mask_fill: f32,
    /// Declare a `rope_theta` the family never reads — a checkpoint that
    /// then claims rotation.
    pub declares_rope: bool,
}

impl Default for Gpt2Form {
    fn default() -> Self {
        Self {
            prefix: "transformer.",
            scale_attn_weights: Some(true),
            inverse_layer_idx: false,
            buffers: true,
            corrupt_mask: false,
            mask_fill: GPT2_MASK_FILL,
            declares_rope: false,
        }
    }
}

impl Gpt2Miniature {
    pub fn seeded(scale_attn_weights: Option<bool>) -> Self {
        let h = GPT2_HIDDEN;
        let layers = (0..GPT2_LAYERS)
            .map(|layer| {
                let s = 100 + layer as u64 * 100;
                Gpt2Layer {
                    ln_1_w: norm_values(h, s),
                    ln_1_b: lcg_values(h, s + 1),
                    c_attn_w: lcg_values(h * 3 * h, s + 2),
                    c_attn_b: lcg_values(3 * h, s + 3),
                    attn_c_proj_w: lcg_values(h * h, s + 4),
                    attn_c_proj_b: lcg_values(h, s + 5),
                    ln_2_w: norm_values(h, s + 6),
                    ln_2_b: lcg_values(h, s + 7),
                    c_fc_w: lcg_values(h * GPT2_INNER, s + 8),
                    c_fc_b: lcg_values(GPT2_INNER, s + 9),
                    mlp_c_proj_w: lcg_values(GPT2_INNER * h, s + 10),
                    mlp_c_proj_b: lcg_values(h, s + 11),
                }
            })
            .collect();
        Self {
            wte: lcg_values(GPT2_VOCAB * h, 1),
            wpe: lcg_values(GPT2_POSITIONS * h, 2)
                .into_iter()
                .map(|v| v * POSITION_SCALE)
                .collect(),
            layers,
            ln_f_w: norm_values(h, 3),
            ln_f_b: lcg_values(h, 4),
            scale_attn_weights,
        }
    }
}

pub fn miniature_gpt2(dir: &Path) -> Gpt2Miniature {
    miniature_gpt2_with(dir, Gpt2Form::default())
}

pub fn miniature_gpt2_with(dir: &Path, form: Gpt2Form) -> Gpt2Miniature {
    let mut config = serde_json::json!({
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "activation_function": "gelu_new",
        "add_cross_attention": false,
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "initializer_range": 0.02,
        "layer_norm_epsilon": GPT2_EPS,
        "n_embd": GPT2_HIDDEN,
        "n_head": GPT2_HEADS,
        "n_inner": null,
        "n_layer": GPT2_LAYERS,
        "n_positions": GPT2_POSITIONS,
        "reorder_and_upcast_attn": false,
        "scale_attn_by_inverse_layer_idx": form.inverse_layer_idx,
        "summary_activation": null,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": true,
        "summary_type": "cls_index",
        "summary_use_proj": true,
        "use_cache": true,
        "vocab_size": GPT2_VOCAB
    });
    if let Some(scale) = form.scale_attn_weights {
        config["scale_attn_weights"] = serde_json::json!(scale);
    }
    if form.declares_rope {
        config["rope_theta"] = serde_json::json!(10000.0);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let model = Gpt2Miniature::seeded(form.scale_attn_weights);
    let h = GPT2_HIDDEN;
    let p = form.prefix;
    let mut shard = ShardBuilder::new();
    shard.push(&format!("{p}wte.weight"), &[GPT2_VOCAB, h], &model.wte);
    shard.push(&format!("{p}wpe.weight"), &[GPT2_POSITIONS, h], &model.wpe);
    shard.push(&format!("{p}ln_f.weight"), &[h], &model.ln_f_w);
    shard.push(&format!("{p}ln_f.bias"), &[h], &model.ln_f_b);
    for (i, l) in model.layers.iter().enumerate() {
        let b = format!("{p}h.{i}");
        shard.push(&format!("{b}.ln_1.weight"), &[h], &l.ln_1_w);
        shard.push(&format!("{b}.ln_1.bias"), &[h], &l.ln_1_b);
        shard.push(&format!("{b}.attn.c_attn.weight"), &[h, 3 * h], &l.c_attn_w);
        shard.push(&format!("{b}.attn.c_attn.bias"), &[3 * h], &l.c_attn_b);
        shard.push(
            &format!("{b}.attn.c_proj.weight"),
            &[h, h],
            &l.attn_c_proj_w,
        );
        shard.push(&format!("{b}.attn.c_proj.bias"), &[h], &l.attn_c_proj_b);
        shard.push(&format!("{b}.ln_2.weight"), &[h], &l.ln_2_w);
        shard.push(&format!("{b}.ln_2.bias"), &[h], &l.ln_2_b);
        shard.push(&format!("{b}.mlp.c_fc.weight"), &[h, GPT2_INNER], &l.c_fc_w);
        shard.push(&format!("{b}.mlp.c_fc.bias"), &[GPT2_INNER], &l.c_fc_b);
        shard.push(
            &format!("{b}.mlp.c_proj.weight"),
            &[GPT2_INNER, h],
            &l.mlp_c_proj_w,
        );
        shard.push(&format!("{b}.mlp.c_proj.bias"), &[h], &l.mlp_c_proj_b);
        if form.buffers {
            let n = GPT2_POSITIONS;
            let mut mask: Vec<f32> = (0..n * n)
                .map(|i| if i % n <= i / n { 1.0 } else { 0.0 })
                .collect();
            if form.corrupt_mask && i == 0 {
                mask[1] = 1.0;
            }
            shard.push(&format!("{b}.attn.bias"), &[1, 1, n, n], &mask);
            shard.push(&format!("{b}.attn.masked_bias"), &[], &[form.mask_fill]);
        }
    }
    shard.write(dir);
    model
}
