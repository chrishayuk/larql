//! A Llama-shaped miniature whose residual is a Sinkhorn hyper-connected
//! BUNDLE of streams rather than one `[hidden]` row per position: the six
//! site operands per layer (`hc_{attn,ffn}_{fn,base,scale}`) at the
//! declared geometry, and the head's three operands (`hc_head_*`) that
//! reduce the bundle before the output head — so the estate is admissible
//! and the whole stack executes.
//!
//! Public so a sibling crate's tests can show what a statement does on a
//! component whose per-layer planes are bundles. Every operand is a
//! deterministic finite F32 value from [`lcg_values`] / [`norm_values`],
//! so the executor reads exactly what was written. The executor's own
//! hyper-connection witnesses (`opplan/exec/tests/wave19_hc_*`) build the
//! same estate at the wave-17 oracle's geometry and values; this one only
//! needs to be a real, runnable hyper-connected container.

use std::path::{Path, PathBuf};

use super::fixtures::{encode_fixture_container, lcg_values, norm_values, ShardBuilder};
use super::opplan::exec::hyper_connection::{HC_HEAD_SCALE_LEN, HC_SCALE_LEN};

pub const HC_HIDDEN: usize = 8;
pub const HC_LAYERS: usize = 2;
/// The declared stream count (`hc_mult`).
pub const HC_STREAMS: usize = 4;
pub const HC_VOCAB: usize = 16;
pub const HC_SINKHORN_ITERS: usize = 20;
pub const HC_SINKHORN_EPS: f64 = 1e-6;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const INTER: usize = 8;
const NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;
/// `(2 + streams) · streams`: the pre, post and residual mixes of a site.
const MIX_ROWS: usize = (2 + HC_STREAMS) * HC_STREAMS;
/// `streams · hidden`: one position's flattened bundle.
const BUNDLE_WIDTH: usize = HC_STREAMS * HC_HIDDEN;
/// The component name the container is encoded under.
const COMPONENT: &str = "hc";

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "llama",
        "hidden_size": HC_HIDDEN,
        "num_hidden_layers": HC_LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": HC_VOCAB,
        "rms_norm_eps": NORM_EPS,
        "rope_theta": ROPE_THETA,
        "hc_mult": HC_STREAMS,
        "hc_sinkhorn_iters": HC_SINKHORN_ITERS,
        "hc_eps": HC_SINKHORN_EPS
    })
}

/// Write the checkpoint (`config.json` + one F32 shard) into `dir`.
pub fn hyper_connected_llama(dir: &Path) {
    std::fs::write(dir.join("config.json"), config().to_string()).unwrap();
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[HC_VOCAB, HC_HIDDEN],
        &lcg_values(HC_VOCAB * HC_HIDDEN, 1),
    );
    shard.push(
        "model.norm.weight",
        &[HC_HIDDEN],
        &norm_values(HC_HIDDEN, 2),
    );
    shard.push(
        "lm_head.weight",
        &[HC_VOCAB, HC_HIDDEN],
        &lcg_values(HC_VOCAB * HC_HIDDEN, 3),
    );
    let q_rows = HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
    for layer in 0..HC_LAYERS {
        let seed = 100 + layer as u64 * 20;
        let prefix = format!("model.layers.{layer}");
        for (offset, (leaf, rows, cols)) in [
            ("self_attn.q_proj.weight", q_rows, HC_HIDDEN),
            ("self_attn.k_proj.weight", kv_rows, HC_HIDDEN),
            ("self_attn.v_proj.weight", kv_rows, HC_HIDDEN),
            ("self_attn.o_proj.weight", HC_HIDDEN, q_rows),
            ("mlp.gate_proj.weight", INTER, HC_HIDDEN),
            ("mlp.up_proj.weight", INTER, HC_HIDDEN),
            ("mlp.down_proj.weight", HC_HIDDEN, INTER),
        ]
        .into_iter()
        .enumerate()
        {
            shard.push(
                &format!("{prefix}.{leaf}"),
                &[rows, cols],
                &lcg_values(rows * cols, seed + offset as u64),
            );
        }
        for (offset, norm) in ["input_layernorm", "post_attention_layernorm"]
            .into_iter()
            .enumerate()
        {
            shard.push(
                &format!("{prefix}.{norm}.weight"),
                &[HC_HIDDEN],
                &norm_values(HC_HIDDEN, seed + 10 + offset as u64),
            );
        }
        for (offset, site) in ["attn", "ffn"].into_iter().enumerate() {
            let site_seed = seed + 12 + 3 * offset as u64;
            shard.push(
                &format!("{prefix}.hc_{site}_fn"),
                &[MIX_ROWS, BUNDLE_WIDTH],
                &lcg_values(MIX_ROWS * BUNDLE_WIDTH, site_seed),
            );
            shard.push(
                &format!("{prefix}.hc_{site}_base"),
                &[MIX_ROWS],
                &lcg_values(MIX_ROWS, site_seed + 1),
            );
            shard.push(
                &format!("{prefix}.hc_{site}_scale"),
                &[HC_SCALE_LEN],
                &norm_values(HC_SCALE_LEN, site_seed + 2),
            );
        }
    }
    shard.push(
        "hc_head_fn",
        &[HC_STREAMS, BUNDLE_WIDTH],
        &lcg_values(HC_STREAMS * BUNDLE_WIDTH, 4),
    );
    shard.push("hc_head_base", &[HC_STREAMS], &lcg_values(HC_STREAMS, 5));
    shard.push(
        "hc_head_scale",
        &[HC_HEAD_SCALE_LEN],
        &norm_values(HC_HEAD_SCALE_LEN, 6),
    );
    shard.write(dir);
}

/// Write the checkpoint under `work` (which must exist), encode it through
/// the production writer, and return the container's path.
pub fn hyper_connected_container(work: &Path) -> PathBuf {
    let checkpoint = work.join("hc-checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let container = work.join("hc.vindex3");
    encode_fixture_container(hyper_connected_llama, &checkpoint, &container, COMPONENT);
    container
}
