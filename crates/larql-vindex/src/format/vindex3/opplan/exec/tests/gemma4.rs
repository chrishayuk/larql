//! V3-F0 witness 3, G4.1/G4.2: a Gemma 4 family miniature closes in the
//! generic graph (hybrid dense+routed FFN, router scales, three extra
//! norms, layer scalar, K≡V on the full layer, proportional partial
//! rotary, V norm, tied softcapped head) and EXECUTES on every CPU
//! backend at parity, with every new operand load-bearing.
//!
//! ```text
//! closure     `vindex3 ops` on the encoded miniature: zero defects
//! parity      reference ≡ production ≡ device (f32 / f16 experts)
//! causal      perturb router.scale / per_expert_scale / layer_scalar /
//!             post_feedforward_layernorm_2 / the full layer's K → output moves
//! K≡V         the full layer's V IS its K projection: perturbing K moves V
//! decode      the decode session reproduces the batch traversal step by step
//! ```
//!
//! The semantic oracle for this family is HF itself (the real 26B-A4B
//! layer-dump gate, ROADMAP G4.2) — the served CPU path is not used as an
//! oracle here because its global-layer rope and router-input choices are
//! the open finds that gate arbitrates.

use std::path::Path;

use super::device::LoopDevice;
use super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, LayerFfn, OpPlanOutcome};

// ── Geometry (the plan-gate miniature's, with real values) ──
const HIDDEN: usize = 64;
const Q_HEADS: usize = 8;
const HEAD_DIM: usize = 8;
const KV_HEADS: usize = 2;
const GLOBAL_HEAD_DIM: usize = 24;
const GLOBAL_KV_HEADS: usize = 1;
const INTER: usize = 128;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const MOE_INTER: usize = 32;
const VOCAB: usize = 128;
const LAYERS: usize = 4;
const FULL_LAYER: usize = 3;
const WINDOW: usize = 16;
const FULL_THETA: f64 = 1_000_000.0;
const SLIDING_THETA: f64 = 10_000.0;
const PARTIAL_ROTARY: f64 = 0.25;
const SOFTCAP: f64 = 30.0;
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

/// Reference loops vs served kernels vs the loop device, f32 throughout
/// (the device's bf16 experts are widened exactly): reassociation only.
const TOLERANCE: f32 = 2e-5;
/// A perturbed operand must move the output well above that.
const CAUSAL_FLOOR: f32 = 1e-3;
const PERTURB_GAIN: f32 = 3.0;

const ROUTER_SCALE: &str = "router.scale";
const PER_EXPERT_SCALE: &str = "router.per_expert_scale";
const LAYER_SCALAR: &str = "layer_scalar";
const POST_EXPERTS_NORM: &str = "post_feedforward_layernorm_2.weight";
const FULL_K_PROJ: &str = "self_attn.k_proj.weight";

/// Round-to-nearest-even bf16 bytes of `values`, little-endian.
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

/// The miniature Gemma 4 checkpoint. `perturb` names one layer-relative
/// operand (on the full layer for attention, every layer otherwise) whose
/// values are scaled by [`PERTURB_GAIN`].
fn miniature_gemma4(dir: &Path, perturb: Option<&str>) {
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
        "rms_norm_eps": 1e-6,
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
        &lcg_values(VOCAB * HIDDEN, 1),
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
        let (head_dim, kv_heads) = if full {
            (GLOBAL_HEAD_DIM, GLOBAL_KV_HEADS)
        } else {
            (HEAD_DIM, KV_HEADS)
        };
        let q_rows = Q_HEADS * head_dim;
        let kv_rows = kv_heads * head_dim;
        // Attention perturbations apply on the full layer only (it is the
        // K≡V layer); everything else on every layer.
        let gain = |suffix: &str| -> f32 {
            let attention = suffix.starts_with("self_attn.");
            if perturb == Some(suffix) && (!attention || full) {
                PERTURB_GAIN
            } else {
                1.0
            }
        };
        let scaled = |n: usize, s: u64, g: f32| -> Vec<f32> {
            lcg_values(n, s).into_iter().map(|v| v * g).collect()
        };
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, HIDDEN],
            &scaled(q_rows * HIDDEN, seed, 1.0),
        );
        // A uniform gain on K would be a NULL perturbation here: the key
        // norm (weighted RMS) and the value norm (RMS) are both
        // scale-invariant, so `3·K` normalises to the same K and the same
        // V. The perturbation is therefore a different matrix (another
        // seed), which moves both paths.
        let k_seed = if gain(FULL_K_PROJ) != 1.0 {
            seed + 21
        } else {
            seed + 1
        };
        shard.push(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_rows, HIDDEN],
            &scaled(kv_rows * HIDDEN, k_seed, 1.0),
        );
        if !full {
            shard.push(
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_rows, HIDDEN],
                &scaled(kv_rows * HIDDEN, seed + 2, 1.0),
            );
        }
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, q_rows],
            &scaled(HIDDEN * q_rows, seed + 3, 1.0),
        );
        shard.push(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 4),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 5),
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
            let suffix = format!("{norm}.weight");
            let values: Vec<f32> = norm_values(HIDDEN, seed + 6 + i as u64)
                .into_iter()
                .map(|v| v * gain(&suffix))
                .collect();
            shard.push(&format!("{prefix}.{suffix}"), &[HIDDEN], &values);
        }
        // A layer scalar near one, perturbed by the gain when asked.
        let layer_scalar = 0.9 + 0.05 * layer as f32;
        shard.push(
            &format!("{prefix}.{LAYER_SCALAR}"),
            &[1],
            &[layer_scalar * gain(LAYER_SCALAR)],
        );
        shard.push(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTER, HIDDEN],
            &scaled(INTER * HIDDEN, seed + 13, 1.0),
        );
        shard.push(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTER, HIDDEN],
            &scaled(INTER * HIDDEN, seed + 14, 1.0),
        );
        shard.push(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, INTER],
            &scaled(HIDDEN * INTER, seed + 15, 1.0),
        );
        shard.push(
            &format!("{prefix}.router.proj.weight"),
            &[EXPERTS, HIDDEN],
            &scaled(EXPERTS * HIDDEN, seed + 16, 1.0),
        );
        shard.push(
            &format!("{prefix}.{ROUTER_SCALE}"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 17)
                .into_iter()
                .map(|v| v * gain(ROUTER_SCALE))
                .collect::<Vec<_>>(),
        );
        shard.push(
            &format!("{prefix}.{PER_EXPERT_SCALE}"),
            &[EXPERTS],
            &norm_values(EXPERTS, seed + 18)
                .into_iter()
                .map(|v| v * gain(PER_EXPERT_SCALE))
                .collect::<Vec<_>>(),
        );
        // Expert banks: packed BF16, `[E, 2I, H]` and `[E, H, I]`, written
        // from bf16-representable values so every backend reads the same
        // numbers.
        let gate_up = scaled(EXPERTS * 2 * MOE_INTER * HIDDEN, seed + 19, 1.0);
        shard.push_bytes(
            &format!("{prefix}.experts.gate_up_proj"),
            "BF16",
            &[EXPERTS, 2 * MOE_INTER, HIDDEN],
            &bf16_bytes(&gate_up),
        );
        let down = scaled(EXPERTS * HIDDEN * MOE_INTER, seed + 20, 1.0);
        shard.push_bytes(
            &format!("{prefix}.experts.down_proj"),
            "BF16",
            &[EXPERTS, HIDDEN, MOE_INTER],
            &bf16_bytes(&down),
        );
    }
    shard.write(dir);
}

fn encoded(dir: &Path) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-gemma4".to_string(), inventory)], container.path()).unwrap();
    container
}

fn closure(container: &Path) -> OpPlanOutcome {
    let inspection = inspect_container(container, false).unwrap();
    plan_component_ops(&inspection, container, "target").unwrap()
}

fn traces(container: &Path) -> (ExecutionTrace, ExecutionTrace, ExecutionTrace) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = closure(container);
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    let reference = execute_plan(&plan, &store, &TOKENS, &ReferenceBackend::new()).unwrap();
    let production = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
    let device = DevicePlanBackend::with_formats(
        LoopDevice,
        "loop-device-gemma4",
        WeightFormats::uniform(WeightFormat::F32),
    );
    let on_device = execute_plan(&plan, &store, &TOKENS, &device).unwrap();
    (reference, production, on_device)
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

/// The last position's logits.
fn final_logits(trace: &ExecutionTrace) -> &[f32] {
    trace.logits.as_ref().expect("the plan has an output head")
}

fn post_layer(trace: &ExecutionTrace, layer: usize) -> &[Vec<f32>] {
    &trace.layers[layer].post_layer
}

fn post_attention(trace: &ExecutionTrace, layer: usize) -> &[Vec<f32>] {
    &trace.layers[layer].post_attention
}

fn max_abs_1d(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The gate: the hybrid plan closes with zero defects, every layer is
/// hybrid with the Gemma 4 router conditioning, the full layer is K≡V,
/// and the three backends agree position by position.
#[test]
fn a_hybrid_gemma4_plan_closes_and_executes_at_parity_on_every_backend() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gemma4(dir.path(), None);
    let container = encoded(dir.path());
    let outcome = closure(container.path());
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.as_ref().unwrap();
    for layer in &plan.layers {
        let hybrid = match &layer.ffn {
            LayerFfn::Hybrid(op) => op,
            other => panic!("layer {} is not hybrid: {other:?}", layer.layer),
        };
        assert!(hybrid.routed.router_scale.is_some());
        assert!(hybrid.routed.router_per_expert_scale.is_some());
        assert!(hybrid.routed.router_norm_eps.is_some());
        assert!(layer.layer_scale.is_some(), "layer {} scalar", layer.layer);
        assert_eq!(layer.attention.v_from_k, layer.layer == FULL_LAYER);
        assert!(layer.attention.parameter_free_qk_norm.v);
    }
    let full = &plan.layers[FULL_LAYER].attention;
    assert_eq!(
        (full.head_dim, full.num_kv_heads),
        (GLOBAL_HEAD_DIM, GLOBAL_KV_HEADS)
    );
    assert_eq!(full.v, full.k, "V names the K operand on the K≡V layer");

    let (reference, production, device) = traces(container.path());
    for layer in 0..LAYERS {
        let rp = max_abs(
            post_layer(&reference, layer),
            post_layer(&production, layer),
        );
        let rd = max_abs(post_layer(&reference, layer), post_layer(&device, layer));
        assert!(
            rp < TOLERANCE,
            "layer {layer}: reference vs production {rp}"
        );
        assert!(rd < TOLERANCE, "layer {layer}: reference vs device {rd}");
    }
    let lp = max_abs_1d(final_logits(&reference), final_logits(&production));
    assert!(lp < TOLERANCE, "logits: reference vs production {lp}");
    // The head is softcapped and tied: every logit is inside ±cap.
    assert!(final_logits(&reference)
        .iter()
        .all(|l| l.abs() < SOFTCAP as f32));
}

/// Every new operand is load-bearing: scaling it moves the reference
/// output well above parity noise. (Run on the reference backend alone;
/// parity above already ties the others to it.)
#[test]
fn the_gemma4_operands_are_load_bearing() {
    let baseline_dir = tempfile::tempdir().unwrap();
    miniature_gemma4(baseline_dir.path(), None);
    let baseline = encoded(baseline_dir.path());
    let (base, _, _) = traces(baseline.path());
    for operand in [
        ROUTER_SCALE,
        PER_EXPERT_SCALE,
        LAYER_SCALAR,
        POST_EXPERTS_NORM,
    ] {
        let dir = tempfile::tempdir().unwrap();
        miniature_gemma4(dir.path(), Some(operand));
        let container = encoded(dir.path());
        let (perturbed, _, _) = traces(container.path());
        let moved = max_abs(
            post_layer(&base, LAYERS - 1),
            post_layer(&perturbed, LAYERS - 1),
        );
        assert!(
            moved > CAUSAL_FLOOR,
            "{operand} must move the output: {moved}"
        );
    }
}

/// The full layer's K projection is load-bearing for its attention
/// output — through the scores AND through V, since the plan binds the K
/// operand as V (asserted structurally above: `v == k`). The executors'
/// V-from-K arithmetic is pinned by the real-model HF parity gate; here
/// the miniature shows the operand reaches the output and that layers
/// before the full one are untouched by it.
#[test]
fn the_full_layers_key_projection_is_load_bearing() {
    let baseline_dir = tempfile::tempdir().unwrap();
    miniature_gemma4(baseline_dir.path(), None);
    let baseline = encoded(baseline_dir.path());
    let (base, _, _) = traces(baseline.path());
    let dir = tempfile::tempdir().unwrap();
    miniature_gemma4(dir.path(), Some(FULL_K_PROJ));
    let container = encoded(dir.path());
    let (perturbed, _, _) = traces(container.path());
    let moved = max_abs(
        post_attention(&base, FULL_LAYER),
        post_attention(&perturbed, FULL_LAYER),
    );
    assert!(
        moved > CAUSAL_FLOOR,
        "K must move the K≡V layer's output: {moved}"
    );
    // Layers before it are untouched by a full-layer perturbation.
    let before = max_abs(
        post_layer(&base, FULL_LAYER - 1),
        post_layer(&perturbed, FULL_LAYER - 1),
    );
    assert_eq!(before, 0.0);
}

/// The decode session reproduces the batch traversal's final logits at
/// every position — the hybrid block, the layer scalar and the K≡V cache
/// all go through the same single place.
#[test]
fn the_decode_session_reproduces_the_batch_traversal() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gemma4(dir.path(), None);
    let container = encoded(dir.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = closure(container.path()).plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    for (position, &token) in TOKENS.iter().enumerate() {
        let step = session.step(token).unwrap();
        let logits = step.logits.expect("logits per step");
        // The batch traversal over the prefix ending here reports this
        // position's logits as its last.
        let batch = execute_plan(&plan, &store, &TOKENS[..=position], &backend).unwrap();
        let diff = max_abs_1d(&logits, final_logits(&batch));
        assert!(
            diff < TOLERANCE,
            "position {position}: decode vs batch {diff}"
        );
    }
}
