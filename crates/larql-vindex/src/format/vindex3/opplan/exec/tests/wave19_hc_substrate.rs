//! **Wave 19a — the hidden-5 hyper-connection substrate.**
//!
//! A finite synthetic container whose geometry agrees with the wave-17
//! oracle (`hyper_connection_oracle.json`: 4 streams, hidden 5, 20
//! Sinkhorn iterations, `hc_eps` 1e-6, `norm_eps` 1e-6) and whose
//! hyper-connection operands ARE the oracle's weights, at every site of
//! every layer. The oracle is not adapted; the container is built to
//! meet it, so a traversal that hands a site the oracle's own state can
//! be checked against the oracle's own stage outputs.
//!
//! Every ordinary operand is a deterministic finite value from the
//! shared fixture generators — never the pattern-byte fixture, whose
//! BF16 bytes can spell NaN. Everything is stored F32 so the operands
//! the executor reads are exactly the values written.
//!
//! ```text
//! Headless             Llama-shaped, dense FFN, the six site operands per
//!                      layer and NO head object — GLM-5.3-Flash's shape.
//!                      Proves the layer contract needs no `hc_head_*`.
//! HeadBearing          the same plus the head's three operands (the
//!                      oracle's), so the whole-stack exit exists.
//! Hybrid               Gemma-4-shaped: dense + routed FFN whose router
//!                      and expert pre-norm read the RAW residual. The
//!                      only estate where "the FFN received the reduced
//!                      vector, not the bundle" is observable.
//! HybridWithLayerScale the hybrid plus a `layer_scalar`, the unjudged
//!                      combination preparation must refuse.
//! ```
//!
//! Encoded through the doctored `encode_graph` seam, as wave 18's
//! carriage fixture is: every hyper-connected container is inadmissible
//! on purpose while the public refusal stands.

use std::path::Path;

use super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::encode_graph;
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::opplan::exec::hyper_connection::{
    Bundle, HeadWeights, HC_HEAD_SCALE_LEN, HC_SCALE_LEN,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use crate::format::vindex3::plan::plan_system;
use larql_models::config::HyperConnection;
use serde_json::Value;

const ORACLE: &str = include_str!("hyper_connection_oracle.json");

// ── Geometry, the oracle's ──
pub(super) const HIDDEN: usize = 5;
pub(super) const STREAMS: usize = 4;
pub(super) const POSITIONS: usize = 3;
pub(super) const LAYERS: usize = 2;
pub(super) const VOCAB: usize = 7;
/// `(2 + 4) · 4`.
pub(super) const MIX_ROWS: usize = 24;
/// `4 · 5`.
const BUNDLE_WIDTH: usize = 20;
/// The component's `rms_norm_eps`, equal to the oracle's `norm_eps` so
/// stage one runs at the eps the oracle ran at.
pub(super) const NORM_EPS: f64 = 1e-6;

// ── Ordinary operator geometry ──
const HEADS: usize = 1;
const KV_HEADS: usize = 1;
/// Even, for rotary pairing; deliberately not `hidden / heads`.
const HEAD_DIM: usize = 4;
const INTER: usize = 4;
// ── Gemma-shaped hybrid FFN ──
const EXPERTS: usize = 2;
const TOP_K: usize = 1;
const MOE_INTER: usize = 4;
const WINDOW: usize = 8;
const FULL_LAYER: usize = 1;
const FULL_THETA: f64 = 1_000_000.0;
const SLIDING_THETA: f64 = 10_000.0;
const PARTIAL_ROTARY: f64 = 0.5;
const SOFTCAP: f64 = 30.0;

/// Seed of the embedding table, shared with the witness so it can
/// recompute the row a token embeds to.
const EMBED_SEED: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Variant {
    Headless,
    HeadBearing,
    Hybrid,
    HybridWithLayerScale,
}

/// The wave-17 oracle, read for its weights, its input and its stage
/// outputs.
pub(super) struct Oracle {
    doc: Value,
}

/// A site's three operands, owned — written into every site of every
/// layer.
pub(super) struct OwnedSite {
    pub(super) mix_fn: Vec<f32>,
    pub(super) base: Vec<f32>,
    pub(super) scale: Vec<f32>,
}

/// The head's operands, owned.
pub(super) struct OwnedHead {
    pub(super) reduce_fn: Vec<f32>,
    pub(super) base: Vec<f32>,
    pub(super) scale: f32,
}

impl OwnedHead {
    pub(super) fn weights(&self) -> HeadWeights<'_> {
        HeadWeights {
            reduce_fn: &self.reduce_fn,
            base: &self.base,
            scale: self.scale,
        }
    }
}

impl Oracle {
    pub(super) fn load() -> Self {
        let doc: Value = serde_json::from_str(ORACLE).expect("oracle json parses");
        assert_eq!(doc["streams"].as_u64().unwrap() as usize, STREAMS);
        assert_eq!(doc["hidden"].as_u64().unwrap() as usize, HIDDEN);
        assert_eq!(doc["positions"].as_u64().unwrap() as usize, POSITIONS);
        assert_eq!(doc["mix_rows"].as_u64().unwrap() as usize, MIX_ROWS);
        assert_eq!(doc["norm_eps"].as_f64().unwrap(), NORM_EPS);
        Self { doc }
    }

    fn floats(&self, path: &[&str]) -> Vec<f32> {
        let mut node = &self.doc;
        for step in path {
            node = &node[step];
        }
        node.as_array()
            .unwrap_or_else(|| panic!("{path:?} is an array"))
            .iter()
            .map(|v| v.as_f64().expect("float element") as f32)
            .collect()
    }

    pub(super) fn topology(&self) -> HyperConnection {
        HyperConnection {
            streams: STREAMS,
            sinkhorn_iters: self.doc["sinkhorn_iters"].as_u64().unwrap() as usize,
            sinkhorn_eps: self.doc["hc_eps"].as_f64().unwrap(),
        }
    }

    pub(super) fn site(&self) -> OwnedSite {
        OwnedSite {
            mix_fn: self.floats(&["weights", "hc_fn"]),
            base: self.floats(&["weights", "hc_base"]),
            scale: self.floats(&["weights", "hc_scale"]),
        }
    }

    pub(super) fn head(&self) -> OwnedHead {
        OwnedHead {
            reduce_fn: self.floats(&["weights", "head_fn"]),
            base: self.floats(&["weights", "head_base"]),
            scale: self.doc["weights"]["head_scale"].as_f64().unwrap() as f32,
        }
    }

    /// The oracle's state at `position`, as a bundle.
    pub(super) fn input(&self, position: usize) -> Bundle {
        let x = self.floats(&["input", "x"]);
        let at = position * BUNDLE_WIDTH;
        Bundle::from_flat(STREAMS, HIDDEN, x[at..at + BUNDLE_WIDTH].to_vec()).unwrap()
    }

    /// One position's slice of a recorded stage.
    pub(super) fn stage(&self, name: &str, position: usize) -> Vec<f32> {
        let all = self.floats(&["stages", name]);
        let width = all.len() / POSITIONS;
        all[position * width..(position + 1) * width].to_vec()
    }
}

/// The encoded container and its plan, sources kept alive.
pub(super) struct Substrate {
    _source: tempfile::TempDir,
    pub(super) container: tempfile::TempDir,
    pub(super) inspection: SystemInspection,
    pub(super) plan: ComponentOpPlan,
}

/// The row token `token` embeds to on the Llama-shaped variants —
/// recomputed from the generator rather than read back, so the witness
/// has an independent statement of what entered the stack.
pub(super) fn llama_embedding_row(token: u32) -> Vec<f32> {
    let table = lcg_values(VOCAB * HIDDEN, EMBED_SEED);
    let at = token as usize * HIDDEN;
    table[at..at + HIDDEN].to_vec()
}

fn llama_config(hyper_connected: bool) -> Value {
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": NORM_EPS,
        "rope_theta": SLIDING_THETA
    });
    if hyper_connected {
        hc_keys(&mut config);
    }
    config
}

fn hc_keys(config: &mut Value) {
    let hc = Oracle::load().topology();
    config["hc_mult"] = serde_json::json!(hc.streams);
    config["hc_sinkhorn_iters"] = serde_json::json!(hc.sinkhorn_iters);
    config["hc_eps"] = serde_json::json!(hc.sinkhorn_eps);
}

fn push_sites(shard: &mut ShardBuilder, prefix: &str, oracle: &Oracle) {
    let site = oracle.site();
    for name in ["attn", "ffn"] {
        shard.push(
            &format!("{prefix}.hc_{name}_fn"),
            &[MIX_ROWS, BUNDLE_WIDTH],
            &site.mix_fn,
        );
        shard.push(&format!("{prefix}.hc_{name}_base"), &[MIX_ROWS], &site.base);
        shard.push(
            &format!("{prefix}.hc_{name}_scale"),
            &[HC_SCALE_LEN],
            &site.scale,
        );
    }
}

fn write_llama(dir: &Path, hyper_connected: bool, head: bool) {
    std::fs::write(
        dir.join("config.json"),
        llama_config(hyper_connected).to_string(),
    )
    .unwrap();
    let oracle = Oracle::load();
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, EMBED_SEED),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_values(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 3),
    );
    for layer in 0..LAYERS {
        let seed = 100 + layer as u64 * 20;
        let prefix = format!("model.layers.{layer}");
        let q_rows = HEADS * HEAD_DIM;
        let kv_rows = KV_HEADS * HEAD_DIM;
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, HIDDEN],
            &lcg_values(q_rows * HIDDEN, seed),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_rows, HIDDEN],
            &lcg_values(kv_rows * HIDDEN, seed + 1),
        );
        shard.push(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[kv_rows, HIDDEN],
            &lcg_values(kv_rows * HIDDEN, seed + 2),
        );
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, q_rows],
            &lcg_values(HIDDEN * q_rows, seed + 3),
        );
        shard.push(
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 4),
        );
        shard.push(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 5),
        );
        shard.push(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTER, HIDDEN],
            &lcg_values(INTER * HIDDEN, seed + 6),
        );
        shard.push(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTER, HIDDEN],
            &lcg_values(INTER * HIDDEN, seed + 7),
        );
        shard.push(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, INTER],
            &lcg_values(HIDDEN * INTER, seed + 8),
        );
        if hyper_connected {
            push_sites(&mut shard, &prefix, &oracle);
        }
    }
    if head {
        let head = oracle.head();
        shard.push("hc_head_fn", &[STREAMS, BUNDLE_WIDTH], &head.reduce_fn);
        shard.push("hc_head_base", &[STREAMS], &head.base);
        shard.push("hc_head_scale", &[HC_HEAD_SCALE_LEN], &[head.scale]);
    }
    shard.write(dir);
}

/// Round-to-nearest-even bf16 bytes of `values`, little-endian — the
/// expert banks' storage, as the Gemma 4 miniature writes them.
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|v| {
            let bits = v.to_bits();
            let rounding = 0x7FFF + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding)) >> 16) as u16
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// The Gemma-4-shaped hybrid: the plan-gate miniature's estate at the
/// oracle's geometry, plus the six site operands per layer, minus the
/// layer scalar unless asked for.
fn write_gemma_hybrid(dir: &Path, layer_scale: bool) {
    let layer_types: Vec<&str> = (0..LAYERS)
        .map(|i| {
            if i == FULL_LAYER {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let mut text_config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "global_head_dim": HEAD_DIM,
        "num_global_key_value_heads": KV_HEADS,
        "attention_k_eq_v": true,
        "attention_bias": false,
        "enable_moe_block": true,
        "num_experts": EXPERTS,
        "top_k_experts": TOP_K,
        "moe_intermediate_size": MOE_INTER,
        "hidden_activation": "gelu_pytorch_tanh",
        "final_logit_softcapping": SOFTCAP,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": VOCAB,
        "use_double_wide_mlp": false,
        "num_kv_shared_layers": 0,
        "vocab_size": VOCAB,
        "sliding_window": WINDOW,
        "rms_norm_eps": NORM_EPS,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": PARTIAL_ROTARY,
                "rope_theta": FULL_THETA,
                "rope_type": "proportional"
            },
            "sliding_attention": { "rope_theta": SLIDING_THETA, "rope_type": "default" }
        },
        "layer_types": layer_types,
        "tie_word_embeddings": true
    });
    hc_keys(&mut text_config);
    let config = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "float32",
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": text_config
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let oracle = Oracle::load();
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.language_model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, EMBED_SEED),
    );
    shard.push(
        "model.language_model.norm.weight",
        &[HIDDEN],
        &norm_values(HIDDEN, 2),
    );
    for layer in 0..LAYERS {
        let seed = 900 + layer as u64 * 40;
        let prefix = format!("model.language_model.layers.{layer}");
        let full = layer == FULL_LAYER;
        let q_rows = HEADS * HEAD_DIM;
        let kv_rows = KV_HEADS * HEAD_DIM;
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, HIDDEN],
            &lcg_values(q_rows * HIDDEN, seed),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_rows, HIDDEN],
            &lcg_values(kv_rows * HIDDEN, seed + 1),
        );
        if !full {
            shard.push(
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_rows, HIDDEN],
                &lcg_values(kv_rows * HIDDEN, seed + 2),
            );
        }
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, q_rows],
            &lcg_values(HIDDEN * q_rows, seed + 3),
        );
        shard.push(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            &norm_values(HEAD_DIM, seed + 4),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            &norm_values(HEAD_DIM, seed + 5),
        );
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ]
        .iter()
        .enumerate()
        {
            shard.push(
                &format!("{prefix}.{norm}.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, seed + 6 + i as u64),
            );
        }
        if layer_scale {
            shard.push(&format!("{prefix}.layer_scalar"), &[1], &[0.9]);
        }
        shard.push(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTER, HIDDEN],
            &lcg_values(INTER * HIDDEN, seed + 13),
        );
        shard.push(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTER, HIDDEN],
            &lcg_values(INTER * HIDDEN, seed + 14),
        );
        shard.push(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, INTER],
            &lcg_values(HIDDEN * INTER, seed + 15),
        );
        shard.push(
            &format!("{prefix}.router.proj.weight"),
            &[EXPERTS, HIDDEN],
            &lcg_values(EXPERTS * HIDDEN, seed + 16),
        );
        shard.push(
            &format!("{prefix}.router.scale"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 17),
        );
        shard.push(
            &format!("{prefix}.router.per_expert_scale"),
            &[EXPERTS],
            &norm_values(EXPERTS, seed + 18),
        );
        shard.push_bytes(
            &format!("{prefix}.experts.gate_up_proj"),
            "BF16",
            &[EXPERTS, 2 * MOE_INTER, HIDDEN],
            &bf16_bytes(&lcg_values(EXPERTS * 2 * MOE_INTER * HIDDEN, seed + 19)),
        );
        shard.push_bytes(
            &format!("{prefix}.experts.down_proj"),
            "BF16",
            &[EXPERTS, HIDDEN, MOE_INTER],
            &bf16_bytes(&lcg_values(EXPERTS * HIDDEN * MOE_INTER, seed + 20)),
        );
        push_sites(&mut shard, &prefix, &oracle);
    }
    shard.write(dir);
}

/// Encode a written checkpoint through the doctored seam and plan it.
fn encode_and_plan(source: tempfile::TempDir, name: &str) -> Substrate {
    let inventory = larql_models::inventory::build_inventory(source.path()).unwrap();
    let named = vec![(name.to_string(), inventory)];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        outcome.closed(),
        "{name}: the substrate must close: {:?}",
        outcome.defects
    );
    let plan = outcome.plan.expect("a closed outcome carries a plan");
    Substrate {
        _source: source,
        container,
        inspection,
        plan,
    }
}

pub(super) fn build(variant: Variant) -> Substrate {
    let source = tempfile::tempdir().unwrap();
    match variant {
        Variant::Headless => write_llama(source.path(), true, false),
        Variant::HeadBearing => write_llama(source.path(), true, true),
        Variant::Hybrid => write_gemma_hybrid(source.path(), false),
        Variant::HybridWithLayerScale => write_gemma_hybrid(source.path(), true),
    }
    encode_and_plan(source, "hc-substrate")
}

/// The same Llama estate with no topology declared and no site operands:
/// the single-stream control the carrier change must leave untouched.
pub(super) fn single_stream_sibling() -> Substrate {
    let source = tempfile::tempdir().unwrap();
    write_llama(source.path(), false, false);
    encode_and_plan(source, "single-stream-sibling")
}
