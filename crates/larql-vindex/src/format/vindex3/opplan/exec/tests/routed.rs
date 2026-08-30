//! A-9.2 gates: a mixture of experts expressed **entirely inside the
//! generic graph** — expert bank as an object, router and expert operands
//! as roles, one routed FFN op — and executed by every backend against
//! the served production forward.
//!
//! The fixture is a miniature of the GPT-OSS *family* (its judged
//! architecture, not a shape lookalike): packed MXFP4 experts with scales
//! and biases, a biased router with top-k-then-softmax, clamped GLU,
//! attention sinks and Q/K/V/O biases. So one fixture exercises A-9.1 and
//! A-9.2 on the real family and the oracle is the served CPU path
//! (`load_model_dir` + `ExpertWeightFfn`), sharing no code with the plan
//! executor.
//!
//! It also carries GPT-OSS's YaRN block, so A-9.3 (scaled frequencies +
//! attention amplitude in the interpreter) is gated by the same oracle.
//!
//! ```text
//! parity      served forward ≡ reference ≡ production ≡ device (native MXFP4 experts)
//! causal      perturb router weight / one expert's gate_up / down / bias → output moves
//! closure     both ways: MoE judgment ⇒ router + bank operands; stray bank operand ⇒ op;
//!             expert operand in the stack ⇒ misplaced; wrong expert count ⇒ geometry
//! absence     dense plans still serialise byte-identically (the golden gate above)
//! universality the executor reaches the bank only through the op plan's OperandRefs
//! ```

use std::path::Path;

use larql_compute::forward::hooks::RecordHook;

use super::device::LoopDevice;
use super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::{encode_system, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::graph::{ObjectKind, OperandRole};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, LayerFfn, OpPlanOutcome};

// ── Geometry: MXFP4 needs k ≡ 0 (mod 32) on both projections ──
const HIDDEN: usize = 32;
const INTER: usize = 32;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 29;
const LAYERS: usize = 2;
const WINDOW: usize = 4;
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];
/// GPT-OSS's published clamp.
const SWIGLU_LIMIT: f64 = 7.0;

/// Naive-loop vs BLAS/f32 on a 32-wide model: reassociation only.
const TOLERANCE: f32 = 2e-5;
/// A perturbed operand must move the layer that consumes it well above
/// that; measured deltas sit around 1e-2..1e0.
const CAUSAL_FLOOR: f32 = 1e-3;
/// YaRN's control moves less on this fixture (see the test); ten times the
/// parity tolerance still separates "executed" from "noise" cleanly.
const YARN_CAUSAL_FLOOR: f32 = 10.0 * TOLERANCE;

/// Which extra operand to perturb (scale by [`PERTURB_GAIN`]) in the
/// checkpoint, by layer-relative suffix.
const PERTURB_GAIN: f32 = 3.0;
const ROUTER_WEIGHT: &str = "mlp.router.weight";
const GATE_UP_BLOCKS: &str = "mlp.experts.gate_up_proj_blocks";
const DOWN_BLOCKS: &str = "mlp.experts.down_proj_blocks";
const DOWN_BIAS: &str = "mlp.experts.down_proj_bias";

/// The miniature GPT-OSS checkpoint. `perturb` names one operand whose
/// f32 source values are scaled before packing (blocks) or writing.
fn miniature_gpt_oss(dir: &Path, perturb: Option<&str>) {
    let config = serde_json::json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "torch_dtype": "float32",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 10000.0,
        // GPT-OSS's published YaRN block: the ramp AND the 1.3466 amplitude
        // (A-9.3) — every layer rotates at scaled frequencies and every
        // logit is rescaled.
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
            "truncate": false
        },
        "rms_norm_eps": 1e-5,
        "attention_bias": true,
        "swiglu_limit": SWIGLU_LIMIT,
        "tie_word_embeddings": false,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ]
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = Q_HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_values(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 3),
    );
    for layer in 0..LAYERS {
        let seed = 700 + layer as u64 * 40;
        let prefix = format!("model.layers.{layer}");
        let gain = |suffix: &str| {
            if perturb == Some(suffix) {
                PERTURB_GAIN
            } else {
                1.0
            }
        };
        let scaled = |n: usize, s: u64, g: f32| -> Vec<f32> {
            lcg_values(n, s).into_iter().map(|v| v * g).collect()
        };
        // Dense: attention (+ biases, sinks), norms, router.
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, HIDDEN],
                scaled(q_rows * HIDDEN, seed, 1.0),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 1, 1.0),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 2, 1.0),
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN, q_rows],
                scaled(HIDDEN * q_rows, seed + 3, 1.0),
            ),
            (
                "self_attn.q_proj.bias",
                vec![q_rows],
                scaled(q_rows, seed + 4, 2.0),
            ),
            (
                "self_attn.k_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 5, 2.0),
            ),
            (
                "self_attn.v_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 6, 2.0),
            ),
            (
                "self_attn.o_proj.bias",
                vec![HIDDEN],
                scaled(HIDDEN, seed + 7, 2.0),
            ),
            (
                "self_attn.sinks",
                vec![Q_HEADS],
                scaled(Q_HEADS, seed + 8, 4.0),
            ),
            (
                "input_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 9),
            ),
            (
                "post_attention_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 10),
            ),
            (
                ROUTER_WEIGHT,
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 11, 4.0 * gain(ROUTER_WEIGHT)),
            ),
            (
                "mlp.router.bias",
                vec![EXPERTS],
                scaled(EXPERTS, seed + 12, 1.0),
            ),
            (
                "mlp.experts.gate_up_proj_bias",
                vec![EXPERTS, 2 * INTER],
                scaled(EXPERTS * 2 * INTER, seed + 13, 1.0),
            ),
            (
                DOWN_BIAS,
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 14, 1.0 * gain(DOWN_BIAS)),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
        // Packed MXFP4 experts: quantise each expert's f32 matrix with the
        // executor's own quantiser (the checkpoint's packing convention;
        // the models-crate decoder is the independent reader).
        for (suffix, rows, k, s) in [
            (GATE_UP_BLOCKS, 2 * INTER, HIDDEN, seed + 15),
            (DOWN_BLOCKS, HIDDEN, INTER, seed + 16),
        ] {
            let groups = k / 32;
            let mut blocks = Vec::new();
            let mut scales = Vec::new();
            for e in 0..EXPERTS {
                let values = scaled(rows * k, s + e as u64 * 100, 2.0 * gain(suffix));
                let LoadedWeight::Mxfp4 { packed, scales: sc } =
                    quantize_mxfp4(&values, rows, k, suffix).unwrap()
                else {
                    unreachable!()
                };
                blocks.extend_from_slice(&packed.as_slice()[..rows * groups * 16]);
                scales.extend_from_slice(&sc.as_slice()[..rows * groups]);
            }
            shard.push_bytes(
                &format!("{prefix}.{suffix}"),
                "U8",
                &[EXPERTS, rows, groups, 16],
                &blocks,
            );
            let scales_name = suffix.replace("_blocks", "_scales");
            shard.push_bytes(
                &format!("{prefix}.{scales_name}"),
                "U8",
                &[EXPERTS, rows, groups],
                &scales,
            );
        }
    }
    shard.write(dir);
}

struct OracleTrace {
    post_attention: Vec<Vec<Vec<f32>>>,
    post_layer: Vec<Vec<Vec<f32>>>,
    logits: Vec<f32>,
}

/// The served CPU forward: `load_model_dir` dequantises the packed experts
/// at load, `ExpertWeightFfn` routes and runs them, the layer hook taps
/// the same boundaries the executor traces.
fn oracle(dir: &Path) -> OracleTrace {
    let weights = larql_models::load_model_dir(dir).unwrap();
    let view = larql_models::WeightsView::dense(&weights);
    let ffn = larql_compute::ffn::ExpertWeightFfn { weights: &weights };
    let mut hook = RecordHook::for_layers(0..LAYERS);
    let mut h = larql_compute::forward::embed_tokens_pub(&weights, &TOKENS);
    for layer in 0..LAYERS {
        let (h_next, _, _, _) = larql_compute::forward::layer::run_layer_with_capture_hooked(
            view, &h, layer, &ffn, false, false, None, None, &mut hook,
        )
        .unwrap();
        h = h_next;
    }
    let last = h
        .slice(ndarray::s![TOKENS.len() - 1..TOKENS.len(), ..])
        .to_owned();
    let logits = larql_compute::forward::predict::raw::hidden_to_raw_logits(&weights, &last);
    let to_rows = |a: &ndarray::Array2<f32>| -> Vec<Vec<f32>> {
        a.outer_iter().map(|row| row.to_vec()).collect()
    };
    OracleTrace {
        post_attention: (0..LAYERS)
            .map(|l| to_rows(&hook.post_attention[&l]))
            .collect(),
        post_layer: (0..LAYERS).map(|l| to_rows(&hook.post_layer[&l])).collect(),
        logits,
    }
}

fn encoded(dir: &Path) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-gpt-oss".to_string(), inventory)], container.path()).unwrap();
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
    // The device declares native MXFP4 for the FFN class: the expert bank
    // is bound as its stored bytes and read by the device's mxfp4 gemv.
    let device = DevicePlanBackend::with_formats(
        LoopDevice,
        "loop-device-routed",
        WeightFormats {
            attention: WeightFormat::F32,
            ffn: WeightFormat::Mxfp4,
            head: WeightFormat::F32,
        },
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

fn assert_matches_oracle(trace: &ExecutionTrace, oracle: &OracleTrace, label: &str) {
    for layer in 0..LAYERS {
        let attn = max_abs(
            &trace.layers[layer].post_attention,
            &oracle.post_attention[layer],
        );
        assert!(
            attn < TOLERANCE,
            "{label}: layer {layer} post_attention {attn:e}"
        );
        let post = max_abs(&trace.layers[layer].post_layer, &oracle.post_layer[layer]);
        assert!(
            post < TOLERANCE,
            "{label}: layer {layer} post_layer {post:e}"
        );
    }
    let logits = max_abs(
        std::slice::from_ref(trace.logits.as_ref().unwrap()),
        std::slice::from_ref(&oracle.logits),
    );
    assert!(logits < TOLERANCE, "{label}: logits {logits:e}");
}

// ── Parity: served forward ≡ every backend ──

#[test]
fn a_routed_plan_matches_the_served_forward_on_every_backend() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let oracle = oracle(dir.path());
    let container = encoded(dir.path());
    let (reference, production, device) = traces(container.path());
    assert_matches_oracle(&reference, &oracle, "reference vs served");
    assert_matches_oracle(&production, &oracle, "production vs served");
    assert_matches_oracle(&device, &oracle, "device(mxfp4) vs served");
}

// ── The graph: the bank is an object, the plan is routed, nothing else ──

#[test]
fn the_expert_bank_is_a_first_class_object_and_the_ffn_is_routed() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let container = encoded(dir.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let bank = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::ExpertBank)
        .expect("an expert-bank object");
    assert_eq!(
        bank.source_bindings.len(),
        LAYERS,
        "one binding per routed layer"
    );
    assert!(
        bank.representations[0].encoding.contains("MXFP4"),
        "{:?}",
        bank.representations
    );
    let stack = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::DecoderStack)
        .unwrap();
    assert!(
        !stack.representations[0].encoding.contains("MXFP4"),
        "{:?}",
        stack.representations
    );
    // The container writes no MoE side-channel: the graph is the whole story.
    let listing: Vec<String> = std::fs::read_dir(container.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !listing.iter().any(|f| f.contains("moe_manifest")),
        "{listing:?}"
    );
    let outcome = closure(container.path());
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    for layer in &plan.layers {
        let Some(LayerFfn::Routed(op)) = &layer.ffn else {
            panic!("layer {} planned dense", layer.layer)
        };
        assert_eq!(op.experts, EXPERTS);
        assert_eq!(op.top_k, TOP_K);
        let crate::format::vindex3::opplan::ExpertBank::Packed { gate_up, down } = &op.bank else {
            panic!(
                "layer {} planned a per-expert bank on a packed fixture",
                layer.layer
            )
        };
        assert_eq!(gate_up.weights.object, bank.id);
        assert_eq!(down.weights.object, bank.id);
        assert!(gate_up.scales.is_some() && down.scales.is_some());
        assert!(gate_up.bias.is_some() && down.bias.is_some());
        assert_eq!(op.router.object, stack.id);
        assert!(op.router_bias.is_some());
        assert!(matches!(
            op.gate_policy,
            larql_models::ExpertGatePolicy::ClampedGlu { .. }
        ));
    }
}

// ── Causal: router and every expert-bank operand are load-bearing ──

#[test]
fn router_and_expert_operands_are_load_bearing() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let base = traces(encoded(dir.path()).path()).0;
    for suffix in [ROUTER_WEIGHT, GATE_UP_BLOCKS, DOWN_BLOCKS, DOWN_BIAS] {
        let dir = tempfile::tempdir().unwrap();
        miniature_gpt_oss(dir.path(), Some(suffix));
        let moved = traces(encoded(dir.path()).path()).0;
        let layer0 = max_abs(&base.layers[0].post_layer, &moved.layers[0].post_layer);
        let logits = max_abs(
            std::slice::from_ref(base.logits.as_ref().unwrap()),
            std::slice::from_ref(moved.logits.as_ref().unwrap()),
        );
        eprintln!("perturb {suffix}: layer0 post_layer {layer0:e}, logits {logits:e}");
        assert!(
            layer0 > CAUSAL_FLOOR,
            "perturbing `{suffix}` moved layer 0 by only {layer0:e} — carried but not executed"
        );
        assert!(
            logits > TOLERANCE,
            "perturbing `{suffix}`: logits moved only {logits:e}"
        );
    }
}

// ── Closure, fail-closed both ways ──

fn mutate_graph(container: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let path = container.join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    mutate(&mut graph);
    std::fs::write(&path, graph.to_string()).unwrap();
}

#[test]
fn bank_operands_without_a_moe_judgment_refuse() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let container = encoded(dir.path());
    mutate_graph(container.path(), |graph| {
        let target = graph["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == "target")
            .unwrap();
        target["execution"]["ffn"]
            .as_object_mut()
            .unwrap()
            .remove("moe");
    });
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    // Every router and expert operand now implies an op the surface lacks
    // — 8 per layer — and the dense FFN operands are missing.
    let implied = outcome
        .defects
        .iter()
        .filter(|d| matches!(d, ClosureDefect::OperandImpliesAbsentOp { required_primitive, .. } if required_primitive.contains("routed FFN")))
        .count();
    assert_eq!(implied, 8 * LAYERS, "{:?}", outcome.defects);
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::MissingOperand {
            role: OperandRole::FfnUp,
            ..
        }
    )));
}

#[test]
fn a_moe_judgment_with_the_wrong_expert_count_refuses_on_geometry() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let container = encoded(dir.path());
    mutate_graph(container.path(), |graph| {
        let target = graph["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == "target")
            .unwrap();
        target["execution"]["ffn"]["moe"]["experts"] = serde_json::json!(EXPERTS + 1);
    });
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    let geometry = outcome
        .defects
        .iter()
        .filter(|d| matches!(d, ClosureDefect::GeometryMismatch { .. }))
        .count();
    // router weight + router bias + 6 bank operands, per layer.
    assert_eq!(geometry, 8 * LAYERS, "{:?}", outcome.defects);
}

#[test]
fn an_expert_operand_left_in_the_stack_is_misplaced() {
    // Undo the carve-out in the persisted graph: drop the bank object so
    // the stack's binding owns the expert tensors again.
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let mut graph = crate::format::vindex3::graph::build_from_inventories(&[(
        "mini-gpt-oss".to_string(),
        inventory.clone(),
    )])
    .graph;
    graph.objects.retain(|o| o.kind != ObjectKind::ExpertBank);
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::encode::encode_graph(
        &graph,
        &[("mini-gpt-oss".to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    let outcome = closure(container.path());
    assert!(!outcome.closed());
    let misplaced = outcome
        .defects
        .iter()
        .filter(|d| {
            matches!(
                d,
                ClosureDefect::MisplacedOperand {
                    belongs_in: ObjectKind::ExpertBank,
                    ..
                }
            )
        })
        .count();
    assert_eq!(misplaced, 6 * LAYERS, "{:?}", outcome.defects);
}

// ── A-9.3: the carried YaRN block is executed, not merely carried ──

#[test]
fn the_plan_executes_yarn_and_its_factor_is_load_bearing() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), None);
    let container = encoded(dir.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = closure(container.path()).plan.unwrap();
    assert!(plan.layers.iter().all(|l| l
        .attention
        .softmax()
        .unwrap()
        .position
        .yarn()
        .is_some_and(|y| y.factor == 32.0)));
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let base = execute_plan(&plan, &store, &TOKENS, &ReferenceBackend::new()).unwrap();

    // Mutate only the persisted graph: a different factor is a different
    // frequency ramp AND a different amplitude — position 0 moves too.
    mutate_graph(container.path(), |graph| {
        let target = graph["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == "target")
            .unwrap();
        for layer in target["attention"].as_array_mut().unwrap() {
            layer["position"]["scaling"]["factor"] = serde_json::json!(4.0);
        }
    });
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = closure(container.path()).plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let moved = execute_plan(&plan, &store, &TOKENS, &ReferenceBackend::new()).unwrap();
    // Position 0 has no rotation, only the amplitude: if it moves, the
    // amplitude is executed, not just the ramp.
    let position0 = (base.layers[0].post_attention[0].iter())
        .zip(&moved.layers[0].post_attention[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let layer0 = max_abs(
        &base.layers[0].post_attention,
        &moved.layers[0].post_attention,
    );
    eprintln!("yarn factor 32→4: layer0 post_attention {layer0:e}, position 0 {position0:e}");
    // The effect is bounded by the fixture's attention temperature: on
    // 32-wide random weights the logits are near-uniform, so a 1.18×
    // amplitude change moves post-attention by ~4e-4 — an order of
    // magnitude above parity noise, which is the causal claim.
    assert!(
        layer0 > YARN_CAUSAL_FLOOR,
        "YaRN factor carried but not executed: {layer0:e}"
    );
    assert!(
        position0 > YARN_CAUSAL_FLOOR,
        "YaRN amplitude not executed at position 0: {position0:e}"
    );
}
