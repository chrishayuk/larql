//! A dense Gemma 4 miniature the lowered Metal session accepts, and the
//! helpers its device tests share.
//!
//! Dense (no MoE block) with a per-layer `layer_scalar`, as the E-series
//! checkpoints are; sliding layers with a small window and one full layer,
//! so a short decode crosses the window. Every matrix value is a multiple
//! of [`F16_EXACT_STEP`] inside f16's exact range, so the lowering's f16
//! residency holds the very numbers the f32 reference reads: lowered vs
//! reference then differs by accumulation order only, which is what the
//! lowered-parity gate measures.

use std::path::Path;

use crate::format::vindex3::fixtures::{
    encode_fixture_container, lcg_values, norm_values, ShardBuilder,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_compute_metal::MetalBackend;

pub(super) const HIDDEN: usize = 64;
pub(super) const Q_HEADS: usize = 4;
pub(super) const HEAD_DIM: usize = 16;
pub(super) const KV_HEADS: usize = 2;
pub(super) const GLOBAL_HEAD_DIM: usize = 32;
pub(super) const GLOBAL_KV_HEADS: usize = 1;
pub(super) const INTER: usize = 128;
pub(super) const VOCAB: usize = 64;
/// Two layers, full then sliding. Depth is the binding constraint on the
/// parity gate, not the semantics: with no layer scalar anywhere, the
/// lowering's own f32 drift against the reference measured 4.3e-7,
/// 6.4e-7 and 1.05e-6 at layers 0..3 of a three-layer variant, so a
/// third layer would gate accumulated reassociation rather than the ops.
pub(super) const LAYERS: usize = 2;
pub(super) const FULL_LAYER: usize = 0;
/// Smaller than every test's sequence, so the sliding layers drop rows.
pub(super) const WINDOW: usize = 4;
const FULL_THETA: f64 = 1_000_000.0;
const SLIDING_THETA: f64 = 10_000.0;
const PARTIAL_ROTARY: f64 = 0.25;
const SOFTCAP: f64 = 30.0;
const NORM_EPS: f64 = 1e-6;
/// The checkpoint's per-layer output scalar: near one, distinct per layer.
const LAYER_SCALAR_BASE: f32 = 0.85;
const LAYER_SCALAR_STEP: f32 = 0.05;
/// Grid every matrix value is rounded to. A multiple of `2^-10` below 2
/// in magnitude is exact in f16 (10 mantissa bits).
const F16_EXACT_STEP: f32 = 1.0 / 1024.0;
pub(super) const LAYER_SCALAR: &str = "layer_scalar";

/// The parity gate a lowered arm is held to against the reference
/// interpreter: per-layer plane `rel_rms` (G4.3, carried into
/// `docs/gemma4-e4b-1.md` §5 as "lowered Metal vs reference").
pub(super) const LOWERED_PARITY_REL_RMS: f64 = 1e-6;

/// `lcg_values` rounded onto the f16-exact grid.
fn f16_exact(n: usize, seed: u64) -> Vec<f32> {
    lcg_values(n, seed)
        .into_iter()
        .map(|v| (v / F16_EXACT_STEP).round() * F16_EXACT_STEP)
        .collect()
}

/// The dense Gemma 4 miniature checkpoint.
pub(super) fn dense_gemma4(dir: &Path) {
    let layer_types: Vec<&str> = (0..LAYERS)
        .map(|i| {
            if i == FULL_LAYER {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let text_config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "global_head_dim": GLOBAL_HEAD_DIM,
        "num_global_key_value_heads": GLOBAL_KV_HEADS,
        "attention_k_eq_v": false,
        "attention_bias": false,
        "enable_moe_block": false,
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
    let config = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "float32",
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": text_config
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut shard = ShardBuilder::new();
    shard.push(
        "model.language_model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &f16_exact(VOCAB * HIDDEN, 1),
    );
    shard.push(
        "model.language_model.norm.weight",
        &[HIDDEN],
        &norm_values(HIDDEN, 2),
    );
    for layer in 0..LAYERS {
        let seed = 700 + layer as u64 * 40;
        let prefix = format!("model.language_model.layers.{layer}");
        let (head_dim, kv_heads) = if layer == FULL_LAYER {
            (GLOBAL_HEAD_DIM, GLOBAL_KV_HEADS)
        } else {
            (HEAD_DIM, KV_HEADS)
        };
        let q_rows = Q_HEADS * head_dim;
        let kv_rows = kv_heads * head_dim;
        for (suffix, shape, s) in [
            ("self_attn.q_proj.weight", [q_rows, HIDDEN], seed),
            ("self_attn.k_proj.weight", [kv_rows, HIDDEN], seed + 1),
            ("self_attn.v_proj.weight", [kv_rows, HIDDEN], seed + 2),
            ("self_attn.o_proj.weight", [HIDDEN, q_rows], seed + 3),
            ("mlp.gate_proj.weight", [INTER, HIDDEN], seed + 4),
            ("mlp.up_proj.weight", [INTER, HIDDEN], seed + 5),
            ("mlp.down_proj.weight", [HIDDEN, INTER], seed + 6),
        ] {
            shard.push(
                &format!("{prefix}.{suffix}"),
                &shape,
                &f16_exact(shape[0] * shape[1], s),
            );
        }
        shard.push(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 7),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 8),
        );
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            shard.push(
                &format!("{prefix}.{norm}.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, seed + 9 + i as u64),
            );
        }
        shard.push(
            &format!("{prefix}.{LAYER_SCALAR}"),
            &[1],
            &[LAYER_SCALAR_BASE + LAYER_SCALAR_STEP * layer as f32],
        );
    }
    shard.write(dir);
}

/// An encoded container, its closed plan and its operand store.
pub(super) struct Fixture {
    /// Held so the container outlives the store that maps it.
    _container: tempfile::TempDir,
    pub plan: ComponentOpPlan,
    pub store: OperandStore,
}

/// Encode `write` into a fresh container and plan its target component.
pub(super) fn fixture(write: impl FnOnce(&Path), name: &str) -> Fixture {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _container: container,
        plan: outcome.plan.expect("a closed outcome carries its plan"),
        store,
    }
}

/// The Metal device, which these tests require: FAIL, never skip, when it
/// is absent — the shaders compile at runtime, so a skip reads as a pass.
pub(super) fn metal() -> MetalBackend {
    MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    )
}

/// `‖a − b‖ / ‖a‖` over every element of two equally shaped planes.
pub(super) fn rel_rms(reference: &[Vec<f32>], got: &[Vec<f32>]) -> f64 {
    assert_eq!(reference.len(), got.len(), "position counts differ");
    let (mut err, mut sig) = (0.0f64, 0.0f64);
    for (r, g) in reference.iter().zip(got) {
        assert_eq!(r.len(), g.len(), "row widths differ");
        for (x, y) in r.iter().zip(g) {
            err += (f64::from(*x) - f64::from(*y)).powi(2);
            sig += f64::from(*x).powi(2);
        }
    }
    (err / sig).sqrt()
}
