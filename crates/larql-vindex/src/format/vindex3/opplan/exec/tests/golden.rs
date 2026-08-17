//! Stage B: the executor against an independent golden implementation of
//! the **judged semantics** — the operations production larql never
//! implemented (sigmoid attention gate, parameter-free QK norm, split
//! scales, four-norm placement).
//!
//! The oracle is deliberately literal and shares *nothing* with the
//! execution stack: it reads the safetensors bytes with its own
//! ten-line parser (not `larql-models`), hardcodes the fixture's judged
//! semantics as plain arithmetic (no plan, no surface, no shared enums,
//! no dispatcher), and records a checkpoint at every novel semantic
//! boundary so a failure names the stage, not the layer.
//!
//! The fixture is a miniature Glimmer anatomy with deliberately awkward
//! dimensions (hidden 12, 3q/1kv, head_dim 4, ffn 20, vocab 29, seq 5)
//! so no broadcasting accident can hide:
//!
//! ```text
//! layer 0: Sliding(3) + RoPE(500000) + gated attention
//! layer 1: Full     + NoPE          + gated attention
//! ```

use std::path::Path;

use super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::{execute_text, ExecutionTrace};
use crate::format::vindex3::opplan::plan_component_ops;

// ── The awkward fixture geometry, shared only as *numbers* ──
pub(super) const G_HIDDEN: usize = 12;
pub(super) const G_Q_HEADS: usize = 3;
pub(super) const G_KV_HEADS: usize = 1;
pub(super) const G_HEAD_DIM: usize = 4;
pub(super) const G_FFN: usize = 20;
pub(super) const G_VOCAB: usize = 29;
pub(super) const G_LAYERS: usize = 2;
pub(super) const G_WINDOW: usize = 3;
pub(super) const G_TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

/// Optional attention operands the miniature can carry (A-9.1): Q/K/V/O
/// projection biases (with `attention_bias: true` declared) and
/// per-query-head sink logits. `perturb` names one of those tensors by
/// suffix; the writer scales it by [`PERTURB_GAIN`], so a test can ask
/// whether that single operand is load-bearing.
#[derive(Default, Clone, Copy)]
pub(super) struct MiniatureExtras {
    pub attention_bias: bool,
    pub sinks: bool,
    pub perturb: Option<&'static str>,
}

/// Multiplier applied to a perturbed extra operand.
pub(super) const PERTURB_GAIN: f32 = 3.0;

/// The five extra attention operands, by layer-relative suffix.
pub(super) const BIAS_SUFFIXES: [&str; 4] = [
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
];
pub(super) const SINKS_SUFFIX: &str = "self_attn.sinks";

/// The two-layer miniature Glimmer checkpoint (F32, judged family).
pub(super) fn miniature_glimmer(dir: &Path) {
    miniature_glimmer_with(dir, MiniatureExtras::default());
}

/// The miniature with the A-9.1 extras.
pub(super) fn miniature_glimmer_with(dir: &Path, extras: MiniatureExtras) {
    let mut config = serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "torch_dtype": "float32",
            "model_type": "muse_glimmer_text",
            "hidden_size": G_HIDDEN,
            "num_hidden_layers": G_LAYERS,
            "intermediate_size": G_FFN,
            "num_attention_heads": G_Q_HEADS,
            "num_key_value_heads": G_KV_HEADS,
            "head_dim": G_HEAD_DIM,
            "vocab_size": G_VOCAB,
            "sliding_window": G_WINDOW,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 0.0],
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 20.0
    });
    if extras.attention_bias {
        config["attention_bias"] = serde_json::json!(true);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = G_Q_HEADS * G_HEAD_DIM;
    let kv_rows = G_KV_HEADS * G_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[G_VOCAB, G_HIDDEN],
        &lcg_values(G_VOCAB * G_HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[G_HIDDEN], &norm_values(G_HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[G_VOCAB, G_HIDDEN],
        &lcg_values(G_VOCAB * G_HIDDEN, 3),
    );
    for layer in 0..G_LAYERS {
        let seed = 900 + layer as u64 * 30;
        let prefix = format!("model.layers.{layer}");
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, G_HIDDEN],
                lcg_values(q_rows * G_HIDDEN, seed),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, G_HIDDEN],
                lcg_values(kv_rows * G_HIDDEN, seed + 1),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, G_HIDDEN],
                lcg_values(kv_rows * G_HIDDEN, seed + 2),
            ),
            (
                "self_attn.o_proj.weight",
                vec![G_HIDDEN, q_rows],
                lcg_values(G_HIDDEN * q_rows, seed + 3),
            ),
            (
                "self_attn.gate_proj.weight",
                vec![q_rows, G_HIDDEN],
                lcg_values(q_rows * G_HIDDEN, seed + 4),
            ),
            (
                "input_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 5),
            ),
            (
                "post_attention_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 6),
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 7),
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 8),
            ),
            (
                "mlp.gate_proj.weight",
                vec![G_FFN, G_HIDDEN],
                lcg_values(G_FFN * G_HIDDEN, seed + 9),
            ),
            (
                "mlp.up_proj.weight",
                vec![G_FFN, G_HIDDEN],
                lcg_values(G_FFN * G_HIDDEN, seed + 10),
            ),
            (
                "mlp.down_proj.weight",
                vec![G_HIDDEN, G_FFN],
                lcg_values(G_HIDDEN * G_FFN, seed + 11),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
        // Extras: values well away from zero so an unapplied bias or sink
        // is a visible absence, not a rounding-level one.
        let mut extra = |suffix: &str, len: usize, seed: u64| {
            let mut values: Vec<f32> = lcg_values(len, seed).into_iter().map(|v| v * 4.0).collect();
            if extras.perturb == Some(suffix) {
                for v in &mut values {
                    *v *= PERTURB_GAIN;
                }
            }
            shard.push(&format!("{prefix}.{suffix}"), &[len], &values);
        };
        if extras.attention_bias {
            extra(BIAS_SUFFIXES[0], q_rows, seed + 12);
            extra(BIAS_SUFFIXES[1], kv_rows, seed + 13);
            extra(BIAS_SUFFIXES[2], kv_rows, seed + 14);
            extra(BIAS_SUFFIXES[3], G_HIDDEN, seed + 15);
        }
        if extras.sinks {
            extra(SINKS_SUFFIX, G_Q_HEADS, seed + 16);
        }
    }
    shard.write(dir);
}

// ── The oracle's own safetensors reader: header JSON + F32 bytes ──

struct RawWeights {
    tensors: std::collections::BTreeMap<String, Vec<f32>>,
}

impl RawWeights {
    fn read(dir: &Path) -> Self {
        use std::io::Read;
        let mut file = std::fs::File::open(dir.join("model.safetensors")).unwrap();
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).unwrap();
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        let mut payload = Vec::new();
        file.read_to_end(&mut payload).unwrap();
        let tensors = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, desc)| {
                let offsets = desc["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;
                let values = payload[start..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                (name.clone(), values)
            })
            .collect();
        Self { tensors }
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.tensors[name]
    }

    /// An operand the checkpoint may or may not carry (the A-9.1 extras).
    fn maybe(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(Vec::as_slice)
    }
}

/// `x[i] += b[i]` when the operand exists — the golden statement of "a
/// projection bias is added to the projection's output".
fn add_if_present(x: &mut [f32], b: Option<&[f32]>) {
    if let Some(b) = b {
        assert_eq!(x.len(), b.len());
        for (x, b) in x.iter_mut().zip(b) {
            *x += b;
        }
    }
}

// ── The literal golden forward, checkpoints at every novel boundary ──

/// Named checkpoints around the judged semantics of one layer.
#[derive(Default)]
pub(super) struct GoldenLayer {
    pub q_after_pf_norm: Vec<f32>,
    pub q_after_query_scale: Vec<f32>,
    pub q_after_position: Vec<f32>,
    pub gate_sigmoid: Vec<f32>,
    pub attention_before_gate: Vec<f32>,
    pub attention_after_gate: Vec<f32>,
    pub post_attention: Vec<Vec<f32>>,
    pub post_layer: Vec<Vec<f32>>,
}

pub(super) struct GoldenTrace {
    pub layers: Vec<GoldenLayer>,
    pub logits: Vec<f32>,
}

fn mv(w: &[f32], out: usize, inp: usize, x: &[f32]) -> Vec<f32> {
    (0..out)
        .map(|o| (0..inp).map(|i| w[o * inp + i] * x[i]).sum())
        .collect()
}

/// The *centred* RMS norm — `MuseGlimmerTextCenteredRMSNorm`, whose gain
/// is `1 + w` because the checkpoint stores its weights around zero.
/// Upstream uses this class for all four decoder-layer norms and the
/// plain [`rms`] for the final norm; two helpers here because they are
/// two operations, not one with a flag.
fn rms_centred(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = 1.0 / ((ss / x.len() as f64) + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| ((*v as f64) * inv) as f32 * (1.0 + w[i]))
        .collect()
}

fn rms(x: &[f32], w: Option<&[f32]>, eps: f64) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = 1.0 / ((ss / x.len() as f64) + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| ((*v as f64) * inv) as f32 * w.map(|w| w[i]).unwrap_or(1.0))
        .collect()
}

/// The judged semantics, written out longhand. Literal by design: this
/// function is the independent statement of what Glimmer's operations
/// *mean*, and it must stay free of every execution-stack type.
pub(super) fn golden_forward(dir: &Path) -> GoldenTrace {
    let w = RawWeights::read(dir);
    let q_rows = G_Q_HEADS * G_HEAD_DIM;
    let kv_rows = G_KV_HEADS * G_HEAD_DIM;

    let embed = w.get("model.embed_tokens.weight");
    // Glimmer's `MuseGlimmerTextNormedEmbedding` RMS-normalises every
    // looked-up row *weightlessly* — no tensor records it, which is why
    // the IR needed a judged `EmbeddingNorm` for it. `rms` with `None`
    // weight is the weightless form.
    let mut h: Vec<Vec<f32>> = G_TOKENS
        .iter()
        .map(|&t| {
            let row = &embed[t as usize * G_HIDDEN..(t as usize + 1) * G_HIDDEN];
            rms(row, None, 1e-5)
        })
        .collect();

    let mut layers = Vec::new();
    for layer in 0..G_LAYERS {
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        let mut trace = GoldenLayer::default();

        // Per-position projections with the judged Q/K pipeline.
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut normed_inputs = Vec::new();
        for (position, row) in h.iter().enumerate() {
            let pre = rms_centred(row, w.get(&name("input_layernorm.weight")), 1e-5);
            let mut q = mv(
                w.get(&name("self_attn.q_proj.weight")),
                q_rows,
                G_HIDDEN,
                &pre,
            );
            let mut k = mv(
                w.get(&name("self_attn.k_proj.weight")),
                kv_rows,
                G_HIDDEN,
                &pre,
            );
            let mut v = mv(
                w.get(&name("self_attn.v_proj.weight")),
                kv_rows,
                G_HIDDEN,
                &pre,
            );
            // A-9.1: projection biases, before anything reads Q/K/V.
            add_if_present(&mut q, w.maybe(&name("self_attn.q_proj.bias")));
            add_if_present(&mut k, w.maybe(&name("self_attn.k_proj.bias")));
            add_if_present(&mut v, w.maybe(&name("self_attn.v_proj.bias")));

            // Parameter-free QK norm: RMS per head, no weights.
            for head in q.chunks_exact_mut(G_HEAD_DIM) {
                let normed = rms(head, None, 1e-5);
                head.copy_from_slice(&normed);
            }
            for head in k.chunks_exact_mut(G_HEAD_DIM) {
                let normed = rms(head, None, 1e-5);
                head.copy_from_slice(&normed);
            }
            if position == h.len() - 1 {
                trace.q_after_pf_norm = q.clone();
            }
            // Declared query factor on normalised Q, before position.
            for value in &mut q {
                *value *= 3.87;
            }
            if position == h.len() - 1 {
                trace.q_after_query_scale = q.clone();
            }
            // Layer 0 rotates (theta 500000); layer 1 is NoPE.
            if layer == 0 {
                for head_values in q
                    .chunks_exact_mut(G_HEAD_DIM)
                    .chain(k.chunks_exact_mut(G_HEAD_DIM))
                {
                    let half = G_HEAD_DIM / 2;
                    for i in 0..half {
                        let inv_freq = 500000.0f64.powf(-2.0 * i as f64 / G_HEAD_DIM as f64);
                        let angle = position as f64 * inv_freq;
                        let (sin_t, cos_t) = (angle.sin() as f32, angle.cos() as f32);
                        let x0 = head_values[i];
                        let x1 = head_values[half + i];
                        head_values[i] = x0 * cos_t - x1 * sin_t;
                        head_values[half + i] = x0 * sin_t + x1 * cos_t;
                    }
                }
            }
            if position == h.len() - 1 {
                trace.q_after_position = q.clone();
            }
            queries.push(q);
            keys.push(k);
            values.push(v);
            normed_inputs.push(pre);
        }

        // Attention with span, score scale, softcap; then the sigmoid gate.
        let mut attn_out = Vec::new();
        for position in 0..h.len() {
            // Layer 0 slides with window 3; layer 1 attends fully.
            let start = if layer == 0 {
                (position + 1).saturating_sub(G_WINDOW)
            } else {
                0
            };
            let mut concat = vec![0.0f32; q_rows];
            for q_head in 0..G_Q_HEADS {
                let q_slice = &queries[position][q_head * G_HEAD_DIM..(q_head + 1) * G_HEAD_DIM];
                let mut scores: Vec<f32> = (start..=position)
                    .map(|kp| {
                        // Single KV head: every query head reads it.
                        let k_slice = &keys[kp][..G_HEAD_DIM];
                        let dot: f32 = q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum();
                        let scaled = dot * (1.0 / (G_HEAD_DIM as f32).sqrt());
                        50.0 * (scaled / 50.0).tanh()
                    })
                    .collect();
                // A-9.1: a sink is one more logit per query head that
                // takes softmax mass and has no value row — stated here
                // in the denominator form (the reference executor uses
                // the append-and-drop form; two transcriptions).
                let sink = w.maybe(&name("self_attn.sinks")).map(|s| s[q_head]);
                let mut max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if let Some(sink) = sink {
                    max = max.max(sink);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                if let Some(sink) = sink {
                    sum += (sink - max).exp();
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                let out = &mut concat[q_head * G_HEAD_DIM..(q_head + 1) * G_HEAD_DIM];
                for (offset, kp) in (start..=position).enumerate() {
                    for (acc, v) in out.iter_mut().zip(&values[kp][..G_HEAD_DIM]) {
                        *acc += scores[offset] * v;
                    }
                }
            }
            if position == h.len() - 1 {
                trace.attention_before_gate = concat.clone();
            }
            // sigmoid(gate_proj(normed attention input)) ⊙ heads, pre-o_proj.
            let gate = mv(
                w.get(&name("self_attn.gate_proj.weight")),
                q_rows,
                G_HIDDEN,
                &normed_inputs[position],
            );
            if position == h.len() - 1 {
                trace.gate_sigmoid = gate.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect();
            }
            for (c, g) in concat.iter_mut().zip(&gate) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
            if position == h.len() - 1 {
                trace.attention_after_gate = concat.clone();
            }
            let mut projected = mv(
                w.get(&name("self_attn.o_proj.weight")),
                G_HIDDEN,
                q_rows,
                &concat,
            );
            add_if_present(&mut projected, w.maybe(&name("self_attn.o_proj.bias")));
            attn_out.push(projected);
        }

        // Four-norm placement: post-attention norm (eps 1e-8) on the
        // attention output before its residual add.
        for (row, out) in h.iter_mut().zip(&attn_out) {
            let normed = rms_centred(out, w.get(&name("post_attention_layernorm.weight")), 1e-8);
            for (a, b) in row.iter_mut().zip(&normed) {
                *a += b;
            }
        }
        trace.post_attention = h.clone();

        // Pre-FFN norm (1e-5), gated SiLU FFN, post-FFN norm (1e-8), residual.
        for row in h.iter_mut() {
            let pre = rms_centred(row, w.get(&name("pre_feedforward_layernorm.weight")), 1e-5);
            let gate = mv(w.get(&name("mlp.gate_proj.weight")), G_FFN, G_HIDDEN, &pre);
            let up = mv(w.get(&name("mlp.up_proj.weight")), G_FFN, G_HIDDEN, &pre);
            let inner: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = mv(
                w.get(&name("mlp.down_proj.weight")),
                G_HIDDEN,
                G_FFN,
                &inner,
            );
            let normed = rms_centred(
                &down,
                w.get(&name("post_feedforward_layernorm.weight")),
                1e-8,
            );
            for (a, b) in row.iter_mut().zip(&normed) {
                *a += b;
            }
        }
        trace.post_layer = h.clone();
        layers.push(trace);
    }

    // Final norm, head, output multiplier, output softcap.
    let final_hidden = rms(h.last().unwrap(), Some(w.get("model.norm.weight")), 1e-5);
    let logits: Vec<f32> = mv(w.get("lm_head.weight"), G_VOCAB, G_HIDDEN, &final_hidden)
        .into_iter()
        .map(|l| {
            let scaled = l * 0.196;
            20.0 * (scaled / 20.0).tanh()
        })
        .collect();
    GoldenTrace { layers, logits }
}

// ── The Stage B gate ──

/// Tolerance between two independent naive f32 implementations on
/// 12-wide state: reassociation only.
const GOLDEN_TOLERANCE: f32 = 1e-5;

pub(super) fn executor_trace(dir: &Path) -> ExecutionTrace {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    executor_trace_from(container.path())
}

/// Plan and execute from an existing (possibly mutated) container.
pub(super) fn executor_trace_from(container: &Path) -> ExecutionTrace {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container, &inspection).unwrap();
    execute_text(&outcome.plan.unwrap(), &store, &G_TOKENS).unwrap()
}

pub(super) fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

/// THE Stage B gate: the container-driven executor reproduces the
/// independent literal statement of the judged semantics — the novel
/// operations production larql never implemented — layer by layer.
#[test]
fn executor_matches_the_independent_golden_semantics() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let executed = executor_trace(dir.path());

    for layer in 0..G_LAYERS {
        let attn = max_abs(
            &executed.layers[layer].post_attention,
            &golden.layers[layer].post_attention,
        );
        assert!(
            attn < GOLDEN_TOLERANCE,
            "layer {layer} post_attention diverges from golden: {attn:e}"
        );
        let post = max_abs(
            &executed.layers[layer].post_layer,
            &golden.layers[layer].post_layer,
        );
        assert!(
            post < GOLDEN_TOLERANCE,
            "layer {layer} post_layer diverges from golden: {post:e}"
        );
    }
    let logits = executed.logits.unwrap();
    let worst = logits
        .iter()
        .zip(&golden.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < GOLDEN_TOLERANCE,
        "logits diverge from golden: {worst:e}"
    );
    eprintln!("golden margins: logits max_abs {worst:e}");

    // The golden trace's own boundary checkpoints double as semantic
    // self-checks (recorded at the last position):
    let layer0 = &golden.layers[0];
    let layer1 = &golden.layers[1];
    // Parameter-free norm and the 3.87 query scale each moved Q.
    assert_ne!(layer0.q_after_pf_norm, layer0.q_after_query_scale);
    // RoPE rotated layer 0's Q at a nonzero position…
    assert_ne!(layer0.q_after_query_scale, layer0.q_after_position);
    // …and NoPE means layer 1's positional stage is bit-identically a no-op.
    assert_eq!(layer1.q_after_query_scale, layer1.q_after_position);
    // The gate is a real sigmoid gate: strictly inside (0, 1), and it
    // actually changed the aggregated attention output.
    assert!(layer0.gate_sigmoid.iter().all(|g| *g > 0.0 && *g < 1.0));
    assert_ne!(layer0.attention_before_gate, layer0.attention_after_gate);
}
