//! **Wave 18 — hyper-connection addressability, the carriage witness.**
//!
//! The claim, kept narrow on purpose: LARQL can assign semantic operand
//! identity to the Sinkhorn hyper-connection vocabulary and carry those
//! operands through its normal execution-planning machinery. Six site
//! operands per layer classify to roles, are required by closure, are
//! checked against the DECLARED stream count's geometry, and are bound
//! into the plan; the head's three bare operands are placed as their own
//! object under the declaration and bound beside them. Since wave 19 the
//! executor traverses the bundle; what it still refuses — a whole-stack
//! image with no head object — it refuses at its door by the head's
//! name, through the same fact the plan report reads.
//!
//! **What this file does NOT claim.** It does not make DeepSeek-V4
//! plannable: DeepSeek remains blocked by an independently unsupported
//! base tensor dialect (`attn.wq_a`, `attn.wkv`, `attn_norm`,
//! `ffn.experts.N.w1`), and its reference supplied wave 17's arithmetic
//! oracle, not this wave's addressability witness. It does not execute a
//! hyper-connected checkpoint; no payload for one exists on this machine.
//!
//! # Three kinds of evidence from three checkpoints
//!
//! ```text
//! synthetic          a two-layer dense stack that DECLARES the topology
//!                    and ships every site operand at the declared
//!                    geometry — the one place closure can be watched
//!                    holding, and each of its refusals can be made to
//!                    fire on its own
//! DeepSeek-V4-Flash  REAL headers: the head's three bare groups gain an
//!                    owner under the declaration and lose it without
//!                    one; `mtp.0`'s eighteen hyper-connection tensors
//!                    stay external. The dialect-blocked control.
//! Kimi-K3            REAL headers: the four `*_res_*` operands the K3
//!                    programme expected this wave to address are a
//!                    `[hidden]` norm and a `[1, hidden]` projection —
//!                    not a Sinkhorn site under ANY stream count. The
//!                    transfer question, answered by shape.
//! ```
//!
//! The synthetic fixture's geometry is small (hidden 64, four streams)
//! and asymmetric: `(2 + 4) · 4 = 24` mix rows against `4 · 64 = 256`
//! columns, so no dimension equals another and a transposed or
//! misassigned operand cannot pass the shape check by coincidence.

use crate::format::vindex3::encode::{encode_graph, encode_system_unenforced};
use crate::format::vindex3::graph::roles::classify_stack_tensor_on;
use crate::format::vindex3::graph::{
    build_from_inventories, LayerOperator, ObjectKind, OperandRole,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::hyper_connection::{HC_HEAD_SCALE_LEN, HC_SCALE_LEN};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};
use crate::format::vindex3::plan::plan_system;
use crate::format::vindex3::plan::tests_support::{custom_artifact, header_only_shards};
use larql_models::config::{HyperConnection, HyperConnectionWeights, ResidualTopology};

/// The synthetic component's geometry.
const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const STREAMS: usize = 4;
const SINKHORN_ITERS: usize = 20;
const SINKHORN_EPS: f64 = 1e-6;
/// `(2 + 4) · 4`.
const MIX_ROWS: usize = 24;
/// `4 · 64`.
const BUNDLE_WIDTH: usize = 256;

/// Hy4-preview's site shape, `[2·hc, hc·hidden]`, at this fixture's
/// geometry — the Sinkhorn-free form that must not bind to a Sinkhorn
/// role.
const PREPOST_MIX_ROWS: usize = 8;

/// A dense two-layer stack, with or without the topology declared.
fn config(hyper_connected: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    if hyper_connected {
        config["hc_mult"] = serde_json::json!(STREAMS);
        config["hc_sinkhorn_iters"] = serde_json::json!(SINKHORN_ITERS);
        config["hc_eps"] = serde_json::json!(SINKHORN_EPS);
    }
    config
}

type Tensors = Vec<(String, Vec<usize>)>;

/// The ordinary two-norm estate: 9 stack operands per layer, plus
/// embedding, final norm and head.
fn dense_tensors() -> Tensors {
    let mut tensors: Tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![128, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![128, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for (leaf, shape) in [
            ("self_attn.q_proj.weight", vec![64, HIDDEN]),
            ("self_attn.k_proj.weight", vec![16, HIDDEN]),
            ("self_attn.v_proj.weight", vec![16, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, 64]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("mlp.gate_proj.weight", vec![256, HIDDEN]),
            ("mlp.up_proj.weight", vec![256, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, 256]),
        ] {
            tensors.push((format!("{stack}.{leaf}"), shape));
        }
    }
    tensors
}

/// The six site operands per layer, at the declared geometry, spelled as
/// DeepSeek-V4 and GLM-5.3-Flash both spell them.
fn site_tensors() -> Tensors {
    let mut tensors = Tensors::new();
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for site in ["attn", "ffn"] {
            tensors.push((
                format!("{stack}.hc_{site}_fn"),
                vec![MIX_ROWS, BUNDLE_WIDTH],
            ));
            tensors.push((format!("{stack}.hc_{site}_base"), vec![MIX_ROWS]));
            tensors.push((format!("{stack}.hc_{site}_scale"), vec![HC_SCALE_LEN]));
        }
    }
    tensors
}

/// The head's three bare operands, at the head's own geometry.
fn head_tensors() -> Tensors {
    vec![
        ("hc_head_fn".to_string(), vec![STREAMS, BUNDLE_WIDTH]),
        ("hc_head_base".to_string(), vec![STREAMS]),
        ("hc_head_scale".to_string(), vec![HC_HEAD_SCALE_LEN]),
    ]
}

/// The encoded container and its plan outcome, sources kept alive.
struct Planned {
    _source: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: crate::format::vindex3::inspect::SystemInspection,
    outcome: OpPlanOutcome,
}

/// Encode through the doctored-write seam: the plan is built and its
/// graph encoded WITHOUT the admissibility gate, so a headless
/// hyper-connected fixture — inadmissible by the head's own finding — can
/// still be constructed and its closure and refusals proven downstream.
/// (A head-bearing estate is admissible since wave 19; see
/// [`the_production_encode_gate_admits_a_head_bearing_container_and_refuses_a_headless_one`].)
fn plan(config: serde_json::Value, tensors: Tensors) -> Planned {
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config, &borrowed);
    let named = vec![("hc-artifact".to_string(), inventory)];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    Planned {
        _source: source,
        container,
        inspection,
        outcome,
    }
}

fn hyper_connected() -> Planned {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    plan(config(true), tensors)
}

/// **The positive witness.** A component that declares the topology and
/// ships every site operand closes, and the plan carries the bundle: the
/// topology on the component, six bound operands per layer, at the
/// geometry the declaration implies.
///
/// Every binding is checked by NAME as well as by presence. A plan that
/// bound `hc_ffn_fn` into the attention site would satisfy a count and
/// run the wrong site's weights at every layer.
#[test]
fn a_declared_topology_with_every_site_operand_closes_and_is_carried() {
    let planned = hyper_connected();
    assert!(planned.outcome.closed(), "{:?}", planned.outcome.defects);
    let plan = planned.outcome.plan.as_ref().unwrap();

    assert_eq!(
        plan.residual_topology,
        ResidualTopology::HyperConnection(HyperConnection {
            streams: STREAMS,
            sinkhorn_iters: SINKHORN_ITERS,
            sinkhorn_eps: SINKHORN_EPS,
        })
    );
    assert_eq!(plan.layers.len(), LAYERS);
    for layer in &plan.layers {
        let sites = layer
            .hyper_connection
            .as_ref()
            .unwrap_or_else(|| panic!("layer {} carries no sites", layer.layer));
        let l = layer.layer;
        assert_eq!(sites.attention.mix_fn.tensor, format!("{l}.hc_attn_fn"));
        assert_eq!(sites.attention.base.tensor, format!("{l}.hc_attn_base"));
        assert_eq!(sites.attention.scale.tensor, format!("{l}.hc_attn_scale"));
        assert_eq!(sites.ffn.mix_fn.tensor, format!("{l}.hc_ffn_fn"));
        assert_eq!(sites.ffn.base.tensor, format!("{l}.hc_ffn_base"));
        assert_eq!(sites.ffn.scale.tensor, format!("{l}.hc_ffn_scale"));
        assert_eq!(sites.attention.mix_fn.shape, [MIX_ROWS, BUNDLE_WIDTH]);
        assert_eq!(sites.ffn.base.shape, [MIX_ROWS]);
        assert_eq!(sites.ffn.scale.shape, [HC_SCALE_LEN]);
        assert_eq!(sites.attention.mix_fn.object, "target.decoder_stack");
        // 9 ordinary + 6 site operands, all consumed.
        assert_eq!(layer.operands_accounted, 15);
        assert_eq!(layer.operands_present, 15);
    }
    // No head was shipped, and none is invented: GLM-5.3-Flash's shape.
    assert!(plan.hyper_connection_head.is_none());
}

/// **The control that makes the witness mean something.** The same six
/// operands on a component that declares ONE stream are strays, each
/// naming the topology as the primitive it would need. Without this a
/// vocabulary that classified the six and consumed them anywhere would
/// pass the test above.
#[test]
fn site_operands_on_a_single_stream_component_are_refused_as_strays() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let planned = plan(config(false), tensors);
    assert!(planned.outcome.plan.is_none());

    let mut refused: Vec<String> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } => {
                assert!(
                    required_primitive.contains("hyper-connection residual topology"),
                    "{required_primitive}"
                );
                Some(tensor.clone())
            }
            _ => None,
        })
        .collect();
    refused.sort();
    let mut expected: Vec<String> = site_tensors()
        .into_iter()
        .map(|(name, _)| name.trim_start_matches("model.layers.").to_string())
        .collect();
    expected.sort();
    assert_eq!(refused, expected, "{:?}", planned.outcome.defects);
    // And nothing else went wrong: the ordinary operands are fine.
    assert_eq!(planned.outcome.defects.len(), expected.len());
}

/// A declared topology whose layer is missing one site operand refuses
/// with that operand's ROLE named — a bundle the traversal could not
/// expand at the FFN site of layer 1, not a partially hyper-connected
/// stack.
#[test]
fn a_missing_site_operand_is_a_named_closure_defect() {
    let mut tensors = dense_tensors();
    tensors.extend(
        site_tensors()
            .into_iter()
            .filter(|(name, _)| name != "model.layers.1.hc_ffn_scale"),
    );
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::MissingOperand {
            layer: 1,
            role: OperandRole::HcFfnScale,
        }]
    );
}

/// **The Hy4 falsifier, made to fire.** A site whose mix projection has
/// the Sinkhorn-free `[2·hc, hc·hidden]` shape binds to the role by name
/// and fails its geometry: the contract is derived from the declared
/// stream count, and `[8, 256]` is not `[24, 256]`. This is the check
/// that would catch Hy4-preview's operands if a spelling ever let them
/// through, and it is written against the shape, not the name.
#[test]
fn a_sinkhorn_free_site_shape_fails_the_declared_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors().into_iter().map(|(name, shape)| {
        if name == "model.layers.0.hc_attn_fn" {
            (name, vec![PREPOST_MIX_ROWS, BUNDLE_WIDTH])
        } else {
            (name, shape)
        }
    }));
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::GeometryMismatch {
            tensor: "target.decoder_stack/0.hc_attn_fn".to_string(),
            expected: vec![MIX_ROWS, BUNDLE_WIDTH],
            actual: vec![PREPOST_MIX_ROWS, BUNDLE_WIDTH],
        }]
    );
}

/// The site scale is exactly three scalars. A checkpoint offering two
/// (Hy4-preview's count) is describing a different operation, and the
/// plan says so by geometry rather than binding two scalars into three
/// slots.
#[test]
fn a_two_entry_site_scale_fails_the_declared_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors().into_iter().map(|(name, shape)| {
        if name == "model.layers.1.hc_ffn_scale" {
            (name, vec![HC_SCALE_LEN - 1])
        } else {
            (name, shape)
        }
    }));
    let planned = plan(config(true), tensors);
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::GeometryMismatch {
            tensor: "target.decoder_stack/1.hc_ffn_scale".to_string(),
            expected: vec![HC_SCALE_LEN],
            actual: vec![HC_SCALE_LEN - 1],
        }]
    );
}

/// **The head, placed and bound.** Under the declaration the three bare
/// groups become one object with three bindings, and the plan binds them
/// at the head's OWN geometry — one row per stream and a single scalar,
/// not a site's `(2 + hc)·hc` rows and three scalars.
#[test]
fn the_head_is_placed_as_its_own_object_and_bound_at_its_own_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.closed(), "{:?}", planned.outcome.defects);

    let head_object = planned
        .inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::HyperConnectionHead)
        .expect("the head object is placed");
    assert_eq!(head_object.id, "target.hyper_connection_head");
    let mut prefixes: Vec<&str> = head_object
        .source_bindings
        .iter()
        .map(|b| b.tensor_prefix.as_str())
        .collect();
    prefixes.sort_unstable();
    assert_eq!(prefixes, ["hc_head_base", "hc_head_fn", "hc_head_scale"]);

    let plan = planned.outcome.plan.as_ref().unwrap();
    let head = plan.hyper_connection_head.as_ref().expect("bound");
    assert_eq!(head.reduce_fn.object, "target.hyper_connection_head");
    assert_eq!(head.reduce_fn.tensor, "hc_head_fn");
    assert_eq!(head.reduce_fn.shape, [STREAMS, BUNDLE_WIDTH]);
    assert_eq!(head.base.tensor, "hc_head_base");
    assert_eq!(head.base.shape, [STREAMS]);
    assert_eq!(head.scale.tensor, "hc_head_scale");
    assert_eq!(head.scale.shape, [HC_HEAD_SCALE_LEN]);
}

/// A head stored at a SITE's geometry — the mistake wave 17 corrected in
/// wave 16's record, arriving as bytes — fails the head's contract rather
/// than binding a Sinkhorn split into an operation that runs none.
#[test]
fn a_head_at_a_sites_geometry_fails_the_heads_contract() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(
        head_tensors()
            .into_iter()
            .map(|(name, shape)| match name.as_str() {
                "hc_head_fn" => (name, vec![MIX_ROWS, BUNDLE_WIDTH]),
                "hc_head_scale" => (name, vec![HC_SCALE_LEN]),
                _ => (name, shape),
            }),
    );
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    let mut mismatches: Vec<(String, Vec<usize>, Vec<usize>)> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::GeometryMismatch {
                tensor,
                expected,
                actual,
            } => Some((tensor.clone(), expected.clone(), actual.clone())),
            _ => None,
        })
        .collect();
    mismatches.sort();
    assert_eq!(
        mismatches,
        [
            (
                "target.hyper_connection_head/hc_head_fn".to_string(),
                vec![STREAMS, BUNDLE_WIDTH],
                vec![MIX_ROWS, BUNDLE_WIDTH],
            ),
            (
                "target.hyper_connection_head/hc_head_scale".to_string(),
                vec![HC_HEAD_SCALE_LEN],
                vec![HC_SCALE_LEN],
            ),
        ]
    );
}

/// The head's three bare names on a component that declares ONE stream
/// have no owner: the builder refuses them by name, the disagreement
/// between estate and declaration stated, and the plan is built without
/// them — the ordinary single-stream program, exactly as before.
#[test]
fn head_operands_without_the_declaration_stay_unplaced_with_the_disagreement_named() {
    let mut tensors = dense_tensors();
    tensors.extend(head_tensors());
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(false), &borrowed);
    let built = build_from_inventories(&[("hc-artifact".to_string(), inventory)]);

    assert!(
        !built
            .graph
            .objects
            .iter()
            .any(|o| o.kind == ObjectKind::HyperConnectionHead),
        "a single-stream component must not own a hyper-connection head"
    );
    for group in ["hc_head_fn", "hc_head_base", "hc_head_scale"] {
        let unplaced = built
            .unplaced
            .iter()
            .find(|u| u.prefix == group)
            .unwrap_or_else(|| panic!("{group} was placed anyway: {:?}", built.unplaced));
        assert!(
            unplaced.reason.contains("declares no Sinkhorn-split"),
            "{group}: {}",
            unplaced.reason
        );
        assert!(
            unplaced.reason.contains("recognised"),
            "{}",
            unplaced.reason
        );
    }
}

/// **The payload half runs, and the control proves what still refuses.**
/// Addressability satisfied, the executor's preparation step prepares a
/// hyper-connected stack that carries a head object and executes it
/// (wave 19); the same stack with no head object is refused at the
/// executor's door, naming the HEAD — a whole-stack image has no declared
/// reduction from the bundle — and not the topology. Both halves against
/// the same fixture family, so the refusal cannot be a broken store or a
/// broken plan wearing the topology's name.
#[test]
fn a_head_bearing_stack_executes_and_a_headless_whole_stack_is_refused_at_the_executors_door() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let with_head = plan(config(true), tensors);
    let plan_with_head = with_head.outcome.plan.as_ref().unwrap();
    assert!(plan_with_head.hyper_connection_head.is_some());
    let store = OperandStore::open(with_head.container.path(), &with_head.inspection).unwrap();
    let trace = execute_text(plan_with_head, &store, &[1, 2, 3])
        .expect("a hyper-connected stack with a head executes");
    assert_eq!(trace.executed_layers, vec![0, 1]);
    assert_eq!(
        trace.logits.expect("the head prices the vocabulary").len(),
        128
    );

    let headless = hyper_connected();
    let hc_plan = headless.outcome.plan.as_ref().unwrap();
    assert!(hc_plan.hyper_connection_head.is_none());
    let store = OperandStore::open(headless.container.path(), &headless.inspection).unwrap();
    let err = execute_text(hc_plan, &store, &[1, 2, 3])
        .unwrap_err()
        .to_string();
    assert!(err.contains("hyper_connection_head"), "{err}");
    assert!(err.contains("layer-range"), "{err}");
    assert!(
        !err.contains("traversal"),
        "the topology's old refusal must not reappear: {err}"
    );

    // The control: one stream, same estate minus the sites, executes.
    let single = plan(config(false), dense_tensors());
    let single_plan = single.outcome.plan.as_ref().unwrap();
    assert_eq!(
        single_plan.residual_topology,
        ResidualTopology::SingleStream
    );
    let store = OperandStore::open(single.container.path(), &single.inspection).unwrap();
    execute_text(single_plan, &store, &[1, 2, 3]).expect("a single-stream plan executes");
}

/// **The encode boundary follows the same fact.** The production writers
/// admit a hyper-connected estate that carries a head object — its
/// `hc_*` keys are carried and its execution surface is executable — and
/// refuse the same estate without one, itemising the head's finding.
/// Wave 18 closed addressability without opening a path that wrote a
/// container it could not run; wave 19 opens exactly the path it can.
#[test]
fn the_production_encode_gate_admits_a_head_bearing_container_and_refuses_a_headless_one() {
    let source = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let named = vec![("hc-with-head".to_string(), inventory)];
    // Named first, so a refusal reads as the finding it is rather than
    // as a count.
    let system = plan_system(&named);
    let blocking: Vec<String> = system
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(
        system.admissible,
        "a head-bearing hyper-connected estate is admissible: {blocking:#?}"
    );
    let out = tempfile::tempdir().unwrap();
    encode_system_unenforced(&named, out.path())
        .expect("a head-bearing hyper-connected estate encodes");

    let source = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let out = tempfile::tempdir().unwrap();
    let err = encode_system_unenforced(&[("hc-headless".to_string(), inventory)], out.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("inadmissible"), "{err}");
    assert!(err.contains("hyper_connection_head"), "{err}");
}

/// A single-stream plan serialises exactly as it did before the topology
/// travelled on the plan: no `residual_topology`, no `hyper_connection`,
/// no `hyper_connection_head` key. Every stored plan stays byte-comparable.
#[test]
fn a_single_stream_plan_serialises_as_before_wave_18() {
    let single = plan(config(false), dense_tensors());
    let json = serde_json::to_string(single.outcome.plan.as_ref().unwrap()).unwrap();
    assert!(!json.contains("residual_topology"), "{json}");
    assert!(!json.contains("hyper_connection"), "{json}");

    let hc = hyper_connected();
    let json = serde_json::to_string(hc.outcome.plan.as_ref().unwrap()).unwrap();
    assert!(json.contains("\"residual_topology\""), "{json}");
    assert!(json.contains("\"hyper_connection\""), "{json}");
    assert!(json.contains("\"mix_fn\""), "{json}");
}

// ── Real headers: DeepSeek-V4-Flash ──

const HEADERS: &str = include_str!("fixtures/hc_operand_headers.json");
const DEEPSEEK: &str = "deepseek-ai/DeepSeek-V4-Flash";

/// DeepSeek-V4-Flash's config, trimmed to what the builder reads, every
/// value from the checkpoint's own `config.json`.
fn deepseek_config(mutate: impl FnOnce(&mut serde_json::Value)) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20
    });
    mutate(&mut config);
    config
}

/// The fixture's flat `{name: {dtype, shape, bytes, shard}}` census,
/// regrouped into per-shard safetensors headers with sequential offsets
/// — what `header_only_shards` writes. Byte counts are the real ones.
fn deepseek_shards() -> serde_json::Map<String, serde_json::Value> {
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    let mut shards: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut offsets: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for (name, tensor) in fixture[DEEPSEEK].as_object().unwrap() {
        let shard = tensor["shard"].as_str().unwrap().to_string();
        let bytes = tensor["bytes"].as_u64().unwrap();
        let offset = offsets.entry(shard.clone()).or_insert(0);
        let entry = shards
            .entry(shard.clone())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        entry.as_object_mut().unwrap().insert(
            name.clone(),
            serde_json::json!({
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [*offset, *offset + bytes],
            }),
        );
        *offset += bytes;
    }
    shards
}

fn deepseek_graph(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> crate::format::vindex3::graph::BuiltGraph {
    let dir = tempfile::tempdir().unwrap();
    let inventory = header_only_shards(dir.path(), &deepseek_config(mutate), &deepseek_shards());
    build_from_inventories(&[("deepseek".to_string(), inventory)])
}

/// **The head gains an owner, `mtp.0` does not.** On DeepSeek-V4-Flash's
/// real headers the three bare `hc_head_*` groups — findings 68, 69 and
/// 70 of the cached plan — become one placed object, the layer sites
/// stay inside the decoder stack, and the eighteen hyper-connection
/// tensors under `mtp.0` remain in the external namespace with the same
/// honest refusal they had. The leak the baseline could not yet test
/// for is tested here: no object binding starts with `mtp`.
///
/// This is the whole of what wave 18 does to DeepSeek. Its base dialect
/// (`attn.wq_a`, `attn.wkv`, `attn_norm`, `ffn.experts.N.w1`) is
/// untouched, so its surface still does not build and it stays blocked.
#[test]
fn deepseeks_head_is_owned_under_the_declaration_and_mtp_stays_external() {
    let built = deepseek_graph(|_| {});

    let head = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::HyperConnectionHead)
        .expect("DeepSeek's head is placed");
    let mut prefixes: Vec<&str> = head
        .source_bindings
        .iter()
        .map(|b| b.tensor_prefix.as_str())
        .collect();
    prefixes.sort_unstable();
    assert_eq!(prefixes, ["hc_head_base", "hc_head_fn", "hc_head_scale"]);
    let head_bytes: u64 = head.source_bindings.iter().map(|b| b.bytes).sum();
    // (4 · 16384 + 4 + 1) · 4 bytes of F32, from the real headers.
    assert_eq!(head_bytes, (4 * 16384 + 4 + 1) * 4);

    // The layer sites are stack tensors, owned with the rest of the layer.
    let stack = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::DecoderStack)
        .expect("the stack is placed");
    assert!(stack
        .source_bindings
        .iter()
        .any(|b| b.tensor_prefix == "layers"));

    // mtp: unplaced, honestly, and NOT leaked into any object.
    let mtp = built
        .unplaced
        .iter()
        .find(|u| u.prefix == "mtp")
        .expect("mtp stays unplaced");
    assert!(
        mtp.reason.contains("no placement rule owns this group"),
        "{}",
        mtp.reason
    );
    for object in &built.graph.objects {
        for binding in &object.source_bindings {
            assert!(
                !binding.tensor_prefix.starts_with("mtp"),
                "object `{}` absorbed `{}`",
                object.id,
                binding.tensor_prefix
            );
        }
    }
    // Total fate: every recognised hyper-connection tensor on the surface
    // is either in the stack (layer sites), in the head object, or under
    // the external namespace. Nothing is neither.
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    for name in fixture[DEEPSEEK].as_object().unwrap().keys() {
        if !name.contains("hc_") {
            continue;
        }
        let owned = built
            .graph
            .objects
            .iter()
            .any(|o| o.source_bindings.iter().any(|b| b.covers(name)));
        let external = name.starts_with("mtp.");
        assert!(
            owned != external,
            "{name}: owned {owned}, external {external} — every hyper-connection tensor \
             has exactly one fate"
        );
    }
}

/// **The dialect control on the head.** Withdraw the iteration count and
/// the declaration resolves to NO topology — Hy4-preview's shape — so the
/// same three bare groups lose their owner and say why. A placement rule
/// that matched on the name alone would place them anyway.
#[test]
fn deepseeks_head_loses_its_owner_when_the_topology_is_not_declared() {
    for withdraw in [
        // Hy4's shape: streams and epsilon without an iteration count.
        Box::new(|c: &mut serde_json::Value| {
            c.as_object_mut().unwrap().remove("hc_sinkhorn_iters");
        }) as Box<dyn FnOnce(&mut serde_json::Value)>,
        // No declaration at all: a single-stream checkpoint with the
        // head's names in its estate.
        Box::new(|c: &mut serde_json::Value| {
            let object = c.as_object_mut().unwrap();
            object.remove("hc_mult");
            object.remove("hc_eps");
            object.remove("hc_sinkhorn_iters");
        }),
    ] {
        let built = deepseek_graph(withdraw);
        assert!(
            !built
                .graph
                .objects
                .iter()
                .any(|o| o.kind == ObjectKind::HyperConnectionHead),
            "the head must not be owned without the declaration"
        );
        for group in ["hc_head_fn", "hc_head_base", "hc_head_scale"] {
            let unplaced = built
                .unplaced
                .iter()
                .find(|u| u.prefix == group)
                .unwrap_or_else(|| panic!("{group} was placed: {:?}", built.unplaced));
            assert!(
                unplaced.reason.contains("declares no Sinkhorn-split"),
                "{group}: {}",
                unplaced.reason
            );
        }
    }
}

// ── Real headers: Kimi-K3 — the transfer question ──

const K3_HEADERS: &str = include_str!("../../plan/tests/fixtures/k3_two_layer_headers.json");
const K3_HIDDEN: usize = 7168;

/// **Wave 18 did not transfer to K3, and the shapes say why.** The K3
/// programme expected its four `*_res_{norm,proj}` operands to be
/// hyper-connection operands that this wave's generic roles would
/// address. Read from K3's own headers, they are a `[hidden]` norm and a
/// `[1, hidden]` projection per sublayer — and a Sinkhorn site's mix
/// projection is `[(2 + hc)·hc, hc·hidden]`, which equals `[1, hidden]`
/// for NO stream count (the smallest, hc = 1, is `[3, hidden]`). They
/// are a different residual topology (K3 calls it AttnRes: config keys
/// `attn_res_block_size`, `output_attn_res_proj`), not this one's
/// second dialect, and giving them a Sinkhorn role would have been the
/// accommodation the K3 witness warned against.
///
/// Both halves asserted: the names do not classify under any operator
/// this wave touched, and the geometry could not have bound even if a
/// name had matched. The first alone could be a spelling gap; the
/// second says it is not.
#[test]
fn k3s_residual_operands_are_not_sinkhorn_sites_under_any_stream_count() {
    let fixture: serde_json::Value = serde_json::from_str(K3_HEADERS).unwrap();
    let mut shapes: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for header in fixture["shards"].as_object().unwrap().values() {
        for (name, tensor) in header.as_object().unwrap() {
            if let Some(leaf) = name.strip_prefix("language_model.model.layers.0.") {
                if leaf.contains("_res_") {
                    shapes.insert(
                        leaf.to_string(),
                        tensor["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect(),
                    );
                }
            }
        }
    }
    // The four the K3 witness names, at the shapes the checkpoint writes.
    assert_eq!(
        shapes,
        [
            ("mlp_res_norm.weight".to_string(), vec![K3_HIDDEN]),
            ("mlp_res_proj.weight".to_string(), vec![1, K3_HIDDEN]),
            (
                "self_attention_res_norm.weight".to_string(),
                vec![K3_HIDDEN]
            ),
            (
                "self_attention_res_proj.weight".to_string(),
                vec![1, K3_HIDDEN]
            ),
        ]
        .into_iter()
        .collect()
    );

    // Half one: no spelling classifies, on the operator K3's layer 0 runs.
    for leaf in shapes.keys() {
        assert_eq!(
            classify_stack_tensor_on(&format!("0.{leaf}"), LayerOperator::Kda),
            None,
            "{leaf} acquired a role — wave 18 accommodated K3 instead of transferring"
        );
    }
    // Half two: no Sinkhorn stream count makes a `[1, hidden]` projection
    // a site's mix projection, nor a `[hidden]` norm any site operand.
    for streams in 1..=16 {
        let mix = vec![
            HyperConnectionWeights::mix_rows_for(streams),
            streams * K3_HIDDEN,
        ];
        assert_ne!(
            mix, shapes["self_attention_res_proj.weight"],
            "hc = {streams}"
        );
        assert_ne!(
            vec![HyperConnectionWeights::mix_rows_for(streams)],
            shapes["self_attention_res_norm.weight"],
            "hc = {streams}"
        );
    }
}

// ── The mixer-on-hyper-connection arm, reached ──

/// **The unjudged arm is reachable, and its `||` is load-bearing.** PR
/// #417's advisory mutation run replaced `is_mamba2() || is_conv_qkv()`
/// with `&&` in the arm that refuses a hyper-connected component's
/// mixer-only layer, and nothing noticed — no fixture declared the
/// topology on a Mamba2 stack. This one does: the pure-SSM miniature with
/// `hc_mult`, `hc_sinkhorn_iters` and `hc_eps` added to its own config.
///
/// What the arm protects against is stated by the control below: without
/// the declaration the same estate CLOSES, so under the declaration a
/// build that lost this arm would plan a hyper-connected component whose
/// every layer carries no site — a plan that disagrees with itself. The
/// refusal must name every layer, because the traversal has no judged
/// form for a one-sublayer block in a bundle, and a stack that is
/// refused on layer 0 alone would read as "layer 1 is fine".
#[test]
fn a_hyper_connected_mixer_only_stack_is_refused_as_unjudged_on_every_layer() {
    use super::mamba2::miniature_mamba2;
    use larql_models::inventory::build_inventory;

    let plan_mixer = |declare_topology: bool| {
        let dir = tempfile::tempdir().unwrap();
        miniature_mamba2(dir.path(), None);
        if declare_topology {
            // The fixture writes a bare `Infinity`, deliberately, so its
            // config is not JSON to serde. Declare the topology as a text
            // insertion at the opening brace instead.
            let path = dir.path().join("config.json");
            let config = std::fs::read_to_string(&path).unwrap();
            let declared = config.replacen(
                '{',
                &format!(
                    "{{\"hc_mult\":{STREAMS},\"hc_sinkhorn_iters\":{SINKHORN_ITERS},\
                     \"hc_eps\":{SINKHORN_EPS},"
                ),
                1,
            );
            std::fs::write(&path, declared).unwrap();
        }
        let inventory = build_inventory(dir.path()).unwrap();
        // The declaration reached the resolved surface — so a refusal
        // below is the arm's, not a parse that silently dropped the keys.
        let topology = inventory
            .resolved
            .execution
            .as_ref()
            .and_then(|e| e.residual_topology);
        assert_eq!(
            topology.map(|t| t.streams()),
            Some(if declare_topology { STREAMS } else { 1 })
        );
        let named = vec![("mamba2-mini".to_string(), inventory)];
        let system = plan_system(&named);
        let container = tempfile::tempdir().unwrap();
        encode_graph(&system.graph, &named, container.path()).unwrap();
        let inspection = inspect_container(container.path(), false).unwrap();
        let layers = inspection
            .graph
            .components
            .iter()
            .find(|c| c.id == "target")
            .expect("the mixer stack is the target component")
            .num_layers;
        (
            plan_component_ops(&inspection, container.path(), "target").unwrap(),
            layers,
        )
    };

    // The control first: the same estate, one stream, closes.
    let (plain, layers) = plan_mixer(false);
    assert!(plain.closed(), "{:?}", plain.defects);
    assert!(layers >= 2, "the fixture must have more than one layer");

    // Under the declaration: refused on EVERY layer, by name, and for
    // nothing else.
    let (hyper, _) = plan_mixer(true);
    assert!(hyper.plan.is_none());
    let mut refused_layers: Vec<usize> = hyper
        .outcome_layers_refused_for(HC_ON_MIXER_FACT_FRAGMENT)
        .collect();
    refused_layers.sort_unstable();
    assert_eq!(
        refused_layers,
        (0..layers).collect::<Vec<_>>(),
        "every mixer layer must be refused: {:?}",
        hyper.defects
    );
    assert_eq!(
        hyper.defects.len(),
        layers,
        "the arm is the ONLY refusal on this estate: {:?}",
        hyper.defects
    );
}

/// The fact the arm reports, as `ClosureDefect::UnjudgedSemantic` spells
/// it. Matched as a fragment so the layer suffix can vary.
const HC_ON_MIXER_FACT_FRAGMENT: &str = "hyper-connection sites on a mixer-only layer";

trait RefusedLayers {
    /// The layer index named by every `UnjudgedSemantic` defect whose
    /// fact carries `fragment` and the required-by names the traversal.
    fn outcome_layers_refused_for(&self, fragment: &str) -> impl Iterator<Item = usize> + '_;
}

impl RefusedLayers for OpPlanOutcome {
    fn outcome_layers_refused_for(&self, fragment: &str) -> impl Iterator<Item = usize> + '_ {
        let fragment = fragment.to_string();
        self.defects.iter().filter_map(move |d| match d {
            ClosureDefect::UnjudgedSemantic {
                fact, required_by, ..
            } if fact.contains(&fragment) && required_by.contains("traversal") => fact
                .rsplit("(layer ")
                .next()
                .and_then(|tail| tail.trim_end_matches(')').parse().ok()),
            _ => None,
        })
    }
}
