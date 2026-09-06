//! Operand closure for `ExpertFormat::PerExpert` MoE (P3d-e): Kimi
//! Linear's spellings — indexed per-expert `w1`/`w2`/`w3`, a bias-
//! corrected router, and a shared expert — carved into the expert-bank
//! object and bound with zero defects.
//!
//! Attention is deliberately plain softmax (`linear_attn_config` declares
//! every layer full-attention, no KDA): this rung is MoE operand closure,
//! and pulling KDA/MLA geometry into the fixture would test a different
//! rung's machinery under this one's name. See `opplan/exec/tests/kda*`
//! for KDA and `plan/tests/mla_nope.rs` for MLA.

use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::graph::{ObjectKind, OperandRole};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, ExpertBank, FfnIdentity, LayerFfn, OpPlanOutcome,
};
use crate::format::vindex3::plan::tests_support::custom_artifact;

const HIDDEN: usize = 32;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const EXPERTS: usize = 3;
const TOP_K: usize = 2;
const MOE_INTER: usize = 16;
const SHARED_EXPERTS: usize = 1;
const LAYERS: usize = 2;
const VOCAB: usize = 64;

fn kimi_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 4,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        // Every layer is plain full attention — see module docs.
        "linear_attn_config": {
            "kda_layers": [],
            "full_attn_layers": (1..=LAYERS).collect::<Vec<_>>()
        },
        // Routed on every layer: no dense prefix to keep the fixture
        // small (closure routes by operand evidence, not this count).
        "first_k_dense_replace": 0,
        "num_experts": EXPERTS,
        "num_experts_per_token": TOP_K,
        "num_shared_experts": SHARED_EXPERTS,
        "moe_intermediate_size": MOE_INTER,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
        "routed_scaling_factor": 2.446,
    })
}

/// One routed layer's tensors at the checkpoint's own naming — everything
/// [`KimiLinearArch`](larql_models::architectures::kimi::KimiLinearArch)'s
/// key methods would resolve, plus plain attention/norm operands.
fn kimi_layer_tensors(layer: usize) -> Vec<(String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.");
    let mut tensors = vec![
        (
            format!("{prefix}self_attn.q_proj.weight"),
            vec![Q_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.k_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.v_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.o_proj.weight"),
            vec![HIDDEN, Q_HEADS * HEAD_DIM],
        ),
        (format!("{prefix}input_layernorm.weight"), vec![HIDDEN]),
        (
            format!("{prefix}post_attention_layernorm.weight"),
            vec![HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.gate.weight"),
            vec![EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.gate.e_score_correction_bias"),
            vec![EXPERTS],
        ),
        (
            format!("{prefix}block_sparse_moe.shared_experts.gate_proj.weight"),
            vec![MOE_INTER * SHARED_EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.shared_experts.up_proj.weight"),
            vec![MOE_INTER * SHARED_EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.shared_experts.down_proj.weight"),
            vec![HIDDEN, MOE_INTER * SHARED_EXPERTS],
        ),
    ];
    for expert in 0..EXPERTS {
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w1.weight"),
            vec![MOE_INTER, HIDDEN],
        ));
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w3.weight"),
            vec![MOE_INTER, HIDDEN],
        ));
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w2.weight"),
            vec![HIDDEN, MOE_INTER],
        ));
    }
    tensors
}

fn kimi_tensors() -> Vec<(String, Vec<usize>)> {
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![VOCAB, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        tensors.extend(kimi_layer_tensors(layer));
    }
    tensors
}

/// Encode a Kimi-shaped variant and plan its target component.
fn plan_variant(mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> OpPlanOutcome {
    plan_variant_with(kimi_config(), mutate)
}

/// [`plan_variant`] under a different declaration — the dense-prefix
/// tests change what the config SAYS, not only what the estate holds.
fn plan_variant_with(
    config: serde_json::Value,
    mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
) -> OpPlanOutcome {
    plan_variant_editing_graph(config, mutate, |_| {})
}

/// [`plan_variant_with`], with the encoded container's graph edited
/// before it is planned — how a container written by an OLDER encoder is
/// put in front of this planner: the tensors are today's, the graph says
/// what a schema-6 graph said.
fn plan_variant_editing_graph(
    config: serde_json::Value,
    mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    edit_graph: impl FnOnce(&mut serde_json::Value),
) -> OpPlanOutcome {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = kimi_tensors();
    mutate(&mut tensors);
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(dir.path(), &config, &borrowed);
    let named = vec![("kimi-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let graph_path = out.path().join("system_graph.json");
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    edit_graph(&mut graph);
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    plan_component_ops(&inspection, out.path(), "target").unwrap()
}

/// The routed-FFN surface of the graph's one component, as JSON.
fn moe_surface(graph: &mut serde_json::Value) -> &mut serde_json::Value {
    &mut graph["components"][0]["execution"]["ffn"]["moe"]
}

/// Strip the shared-expert width the way a graph written before
/// 2026-09-03 (semantics 9) never carried it, keeping the count.
fn strip_shared_width(graph: &mut serde_json::Value) {
    let moe = moe_surface(graph);
    assert_eq!(
        moe["shared_experts"], SHARED_EXPERTS,
        "the fixture declares one"
    );
    assert!(
        moe.as_object_mut()
            .unwrap()
            .remove("shared_expert_intermediate_size")
            .is_some(),
        "today's encoder writes the width; the edit must actually remove it"
    );
}

/// The whole point of this rung: a `PerExpert` MoE bank — router,
/// bias-corrected selection, shared expert, and every one of `EXPERTS`
/// experts' `w1`/`w2`/`w3` on every layer — closes with zero defects.
#[test]
fn a_per_expert_kimi_shaped_estate_closes() {
    let outcome = plan_variant(|_| {});
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    for layer in &plan.layers {
        let Some(LayerFfn::Routed(op)) = &layer.ffn else {
            panic!("layer {}: planned non-routed", layer.layer);
        };
        assert_eq!(op.experts, EXPERTS);
        assert_eq!(op.top_k, TOP_K);
        let ExpertBank::PerExpert { gate, up, down } = &op.bank else {
            panic!("layer {}: planned a packed bank", layer.layer);
        };
        assert_eq!(gate.len(), EXPERTS);
        assert_eq!(up.len(), EXPERTS);
        assert_eq!(down.len(), EXPERTS);
        for operand in gate.iter().chain(up).chain(down) {
            assert_eq!(operand.object, "target.expert_bank");
        }
        let shared = op.shared.as_ref().expect("kimi declares a shared expert");
        assert_eq!(shared.intermediate_size, MOE_INTER * SHARED_EXPERTS);
        assert_eq!(shared.gate.object, "target.decoder_stack");
    }
}

/// Every expert's bank operand physically carves into the component's
/// `ExpertBank` object — not left sitting in the decoder stack, which was
/// the P3d-d defect: `expert_bank_prefix` answered `None` for every
/// `PerExpert` layer, so no tensor ever moved.
#[test]
fn the_per_expert_bank_carves_out_of_the_decoder_stack() {
    let outcome = plan_variant(|_| {});
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let Some(LayerFfn::Routed(op)) = &plan.layers[0].ffn else {
        panic!("planned non-routed");
    };
    let ExpertBank::PerExpert { gate, .. } = &op.bank else {
        panic!("planned a packed bank");
    };
    assert_eq!(gate[0].object, "target.expert_bank");
    assert_ne!(gate[0].object, "target.decoder_stack");
}

/// The routed op transcribes the surface's whole routing rule: the
/// sigmoid scoring function, renormalisation, AND the branch scale — the
/// one an executor cannot recover from the operands, so a plan that
/// dropped it would execute a routed sum 2.446× too small in silence.
#[test]
fn the_routed_op_carries_the_declared_sigmoid_rule_and_branch_scale() {
    use larql_models::config::{ExpertRoutingPolicy, MoeRouterKind};
    let outcome = plan_variant(|_| {});
    let plan = outcome.plan.unwrap();
    let Some(LayerFfn::Routed(op)) = &plan.layers[0].ffn else {
        panic!("planned non-routed");
    };
    assert_eq!(op.router_kind, MoeRouterKind::Sigmoid);
    assert_eq!(
        op.routing_policy,
        ExpertRoutingPolicy::NormalisedOverSelected
    );
    assert_eq!(op.branch_scale, Some(2.446));
}

/// A surface that declares no branch scale plans none, and the executor
/// reads that as exactly 1: the absent declaration is the identity, not
/// a zero and not a default borrowed from another model.
#[test]
fn an_undeclared_branch_scale_plans_as_none_and_executes_as_one() {
    let outcome = plan_variant_editing_graph(
        kimi_config(),
        |_| {},
        |graph| {
            let moe = moe_surface(graph);
            assert_eq!(moe["branch_scale"], 2.446, "the fixture declares one");
            moe.as_object_mut().unwrap().remove("branch_scale");
        },
    );
    let plan = outcome.plan.unwrap();
    let Some(LayerFfn::Routed(op)) = &plan.layers[0].ffn else {
        panic!("planned non-routed");
    };
    assert_eq!(op.branch_scale, None);
    assert_eq!(op.executed_branch_scale(), 1.0);
}

/// Removing one expert's `w1` — the 256-tensor set closure the carving
/// rung exists to prove — is a `MissingOperand` for that exact expert
/// index, not a silent gap.
#[test]
fn a_missing_expert_tensor_names_its_exact_index() {
    const MISSING_EXPERT: u16 = 1;
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| {
            name != &format!("model.layers.0.block_sparse_moe.experts.{MISSING_EXPERT}.w1.weight")
        });
    });
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand { layer: 0, role: OperandRole::PerExpertGate(e) }
                if *e == MISSING_EXPERT
        )),
        "{:?}",
        outcome.defects
    );
}

/// A stray tensor for an expert index beyond the declared count is a
/// defect naming the routed-FFN judgment as the reason, not a silent
/// extra expert — on all three per-expert roles: the guard binds the
/// expert id once across `w1`/`w3`/`w2`'s three OR-pattern alternatives,
/// and only exercising `w1` would leave the other two unproven.
#[test]
fn an_expert_index_beyond_the_declared_count_is_a_stray() {
    for (leaf, shape) in [
        ("w1", vec![MOE_INTER, HIDDEN]),
        ("w3", vec![MOE_INTER, HIDDEN]),
        ("w2", vec![HIDDEN, MOE_INTER]),
    ] {
        let outcome = plan_variant(|tensors| {
            tensors.push((
                format!("model.layers.0.block_sparse_moe.experts.{EXPERTS}.{leaf}.weight"),
                shape,
            ));
        });
        assert!(
            outcome.defects.iter().any(|d| matches!(
                d,
                ClosureDefect::OperandImpliesAbsentOp { tensor, required_primitive, .. }
                    if tensor.contains(&format!("experts.{EXPERTS}.{leaf}"))
                        && required_primitive.contains("does not declare")
            )),
            "{leaf}: {:?}",
            outcome.defects
        );
    }
}

/// A shared-expert tensor on a layer whose judgment declares no shared
/// expert (`shared_experts: 0`) is a stray, naming the routed-FFN
/// judgment — the same "declares none" guard the expert-index check
/// above uses, on the always-active branch instead of the routed one.
#[test]
fn a_shared_expert_operand_with_none_declared_is_a_stray() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = kimi_config();
    config["num_shared_experts"] = serde_json::json!(0);
    let mut tensors = kimi_tensors();
    tensors.push((
        "model.layers.0.block_sparse_moe.shared_experts.gate_proj.weight".to_string(),
        vec![MOE_INTER, HIDDEN],
    ));
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(dir.path(), &config, &borrowed);
    let named = vec![("kimi-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, out.path(), "target").unwrap();
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp { tensor, required_primitive, .. }
                if tensor.contains("shared_experts.gate_proj")
                    && required_primitive.contains("declares none")
        )),
        "{:?}",
        outcome.defects
    );
}

/// Dropping the shared expert's down projection is `MissingOperand` for
/// `SharedExpertDown` — the always-active branch closes exactly like the
/// routed one, operand for operand.
#[test]
fn a_missing_shared_expert_operand_is_named() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| {
            name != "model.layers.0.block_sparse_moe.shared_experts.down_proj.weight"
        });
    });
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                layer: 0,
                role: OperandRole::SharedExpertDown
            }
        )),
        "{:?}",
        outcome.defects
    );
}

/// The router's bias-correction tensor is a first-class role, distinct
/// from the router weight — dropping only the bias reports only the
/// bias missing.
#[test]
fn the_router_bias_and_weight_are_independently_required() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| {
            name != "model.layers.0.block_sparse_moe.gate.e_score_correction_bias"
        });
    });
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                layer: 0,
                role: OperandRole::MoeRouterBias
            }
        )),
        "{:?}",
        outcome.defects
    );
    assert!(
        !outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                layer: 0,
                role: OperandRole::MoeRouterWeight
            }
        )),
        "{:?}",
        outcome.defects
    );
}

/// Every carved expert-bank tensor lands in `ObjectKind::ExpertBank`, and
/// every dense/router/shared operand stays in the decoder stack — the
/// placement half of closure, checked directly against the encoded
/// container's own objects rather than through the plan.
#[test]
fn carving_places_every_expert_tensor_in_the_bank_object_and_nothing_else() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = custom_artifact(
        dir.path(),
        &kimi_config(),
        &kimi_tensors()
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect::<Vec<_>>(),
    );
    let named = vec![("kimi-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    let bank = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::ExpertBank)
        .expect("a PerExpert layer must carve a bank object");
    let bank_tensors: usize = bank.source_bindings.iter().map(|b| b.tensors).sum();
    // 3 tensors/expert * EXPERTS experts * LAYERS routed layers.
    assert_eq!(bank_tensors, 3 * EXPERTS * LAYERS);
    let stack = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::DecoderStack)
        .expect("the stack object still exists");
    for binding in &stack.source_bindings {
        assert!(
            !binding.tensor_prefix.contains(".experts."),
            "an expert tensor was left in the stack: {}",
            binding.tensor_prefix
        );
    }
}

// ── The dense prefix is declared, not inferred ────────────────────────
//
// `first_k_dense_replace` is the surface's schedule (`dense_prefix_layers`).
// Closure used to route a layer by operand evidence alone, so a routed
// layer whose expert bank was missing quietly planned as dense. The
// declaration is now the authority and the operands are evidence that it
// is honoured; disagreement in either direction is a defect, not a
// different plan.

/// Kimi's own config with a one-layer dense prefix.
fn dense_prefix_config(prefix: usize) -> serde_json::Value {
    let mut config = kimi_config();
    config["first_k_dense_replace"] = serde_json::json!(prefix);
    config
}

/// Replace layer 0's routed block with the dense MLP a Kimi dense-prefix
/// layer carries.
fn make_layer_0_dense(tensors: &mut Vec<(String, Vec<usize>)>) {
    tensors.retain(|(name, _)| {
        !(name.starts_with("model.layers.0.") && name.contains("block_sparse_moe"))
    });
    let inter = HIDDEN * 4;
    tensors.push((
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        vec![inter, HIDDEN],
    ));
    tensors.push((
        "model.layers.0.mlp.up_proj.weight".to_string(),
        vec![inter, HIDDEN],
    ));
    tensors.push((
        "model.layers.0.mlp.down_proj.weight".to_string(),
        vec![HIDDEN, inter],
    ));
}

fn identity_defect(layer: usize, declared: FfnIdentity, evidence: FfnIdentity) -> ClosureDefect {
    ClosureDefect::FfnIdentityMismatch {
        layer,
        declared,
        evidence,
    }
}

/// A declared dense-prefix layer plans dense, and the layers after it
/// keep their routed plan: the declaration is valid as dense when the
/// estate agrees with it.
///
/// Three layers, not the fixture's two: with a single routed layer the
/// encoder names the bank's tensors relative to that one layer's prefix
/// and the layer segment vanishes from the name (`0.w1.weight`), which
/// closure then cannot place — an encoder naming artefact of a one-layer
/// bank, not a closure fact, and outside this test's claim.
#[test]
fn a_declared_dense_prefix_layer_plans_dense_and_the_rest_routed() {
    const THREE: usize = 3;
    let mut config = dense_prefix_config(1);
    config["num_hidden_layers"] = serde_json::json!(THREE);
    config["linear_attn_config"]["full_attn_layers"] =
        serde_json::json!((1..=THREE).collect::<Vec<_>>());
    let outcome = plan_variant_with(config, |tensors| {
        tensors.extend(kimi_layer_tensors(2));
        make_layer_0_dense(tensors);
    });
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert_eq!(plan.layers.len(), THREE);
    assert!(
        matches!(plan.layers[0].ffn, Some(LayerFfn::Dense(_))),
        "layer 0 is the declared dense prefix"
    );
    for layer in &plan.layers[1..] {
        assert!(
            matches!(layer.ffn, Some(LayerFfn::Routed(_))),
            "layer {} is routed by declaration and by its bank",
            layer.layer
        );
    }
}

/// A routed layer whose expert bank and router are missing is refused by
/// name. It used to plan as dense with no defect — the silent identity
/// change a missing bank must never cause.
#[test]
fn a_routed_layer_with_no_expert_bank_is_refused_not_planned_dense() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| {
            !(name.starts_with("model.layers.1.") && name.contains("block_sparse_moe"))
        })
    });
    assert!(!outcome.closed(), "a bank-less routed layer must not close");
    let mismatch = identity_defect(1, FfnIdentity::Routed, FfnIdentity::Dense);
    assert!(outcome.defects.contains(&mismatch), "{:?}", outcome.defects);
    let text = mismatch.to_string();
    assert!(
        text.starts_with(
            "layer 1: the surface declares a routed FFN but the operands are those of a dense"
        ),
        "{text}"
    );
    assert!(
        !outcome
            .defects
            .iter()
            .any(|d| matches!(d, ClosureDefect::FfnIdentityMismatch { layer: 0, .. })),
        "layer 0 still agrees with its declaration: {:?}",
        outcome.defects
    );
}

/// The other direction: a declared-dense prefix layer that still carries
/// routed operands is the same disagreement, refused the same way rather
/// than routed on the strength of the stray operands.
#[test]
fn a_dense_prefix_layer_carrying_routed_operands_is_refused() {
    let outcome = plan_variant_with(dense_prefix_config(1), |_| {});
    assert!(
        !outcome.closed(),
        "routed operands on a declared-dense layer must not close"
    );
    assert!(
        outcome
            .defects
            .contains(&identity_defect(0, FfnIdentity::Dense, FfnIdentity::Routed)),
        "{:?}",
        outcome.defects
    );
}

// ── K3-RESIDENCY-VERTICAL-1 / V1: the shared expert is planned or refused, never dropped ──

/// A graph that declares the shared branch by COUNT alone — every real
/// container encoded before the width joined the surface — plans the
/// branch at the width its own gate tensor stores, identically to a
/// graph that declares the width.
#[test]
fn a_legacy_graph_without_the_shared_width_plans_the_shared_expert_from_its_tensors() {
    let declared = plan_variant(|_| {});
    let legacy = plan_variant_editing_graph(kimi_config(), |_| {}, strip_shared_width);
    assert!(legacy.closed(), "{:?}", legacy.defects);
    let (declared, legacy) = (declared.plan.unwrap(), legacy.plan.unwrap());
    for (d, l) in declared.layers.iter().zip(&legacy.layers) {
        let (Some(LayerFfn::Routed(d)), Some(LayerFfn::Routed(l))) = (&d.ffn, &l.ffn) else {
            panic!("both plans route every layer");
        };
        let shared = l
            .shared
            .as_ref()
            .expect("the legacy graph's shared expert is planned, not dropped");
        assert_eq!(shared.intermediate_size, MOE_INTER * SHARED_EXPERTS);
        assert_eq!(d.shared, l.shared, "same op from either graph");
    }
    assert_eq!(
        declared.planned_operands().len(),
        legacy.planned_operands().len(),
        "the shared projections are planned operands under either graph"
    );
}

/// The width read from the gate holds the other two tensors to it: a
/// legacy graph whose `up` disagrees is refused at closure, naming the
/// tensor and both shapes — never planned at whichever width came first.
#[test]
fn a_legacy_graph_whose_shared_tensors_disagree_on_width_refuses_by_name() {
    let outcome = plan_variant_editing_graph(
        kimi_config(),
        |tensors| {
            let up = tensors
                .iter_mut()
                .find(|(name, _)| {
                    name.ends_with("layers.0.block_sparse_moe.shared_experts.up_proj.weight")
                })
                .expect("the fixture spells the shared up projection");
            up.1 = vec![MOE_INTER * SHARED_EXPERTS + 8, HIDDEN];
        },
        strip_shared_width,
    );
    assert!(!outcome.closed(), "a disagreeing branch must not close");
    let named = outcome.defects.iter().any(|d| {
        matches!(
            d,
            ClosureDefect::GeometryMismatch { tensor, expected, actual }
                if tensor.ends_with("0.block_sparse_moe.shared_experts.up_proj.weight")
                    && *expected == vec![MOE_INTER * SHARED_EXPERTS, HIDDEN]
                    && *actual == vec![MOE_INTER * SHARED_EXPERTS + 8, HIDDEN]
        )
    });
    assert!(named, "{:?}", outcome.defects);
}

/// A graph that DECLARES the width keeps it as the authority: tensors
/// that disagree with the declaration are refused against the
/// declaration, not against each other.
#[test]
fn a_declared_shared_width_is_held_against_the_tensors() {
    let outcome = plan_variant(|tensors| {
        for (name, shape) in tensors.iter_mut() {
            if name.ends_with("layers.1.block_sparse_moe.shared_experts.gate_proj.weight") {
                *shape = vec![MOE_INTER * SHARED_EXPERTS * 2, HIDDEN];
            }
        }
    });
    assert!(!outcome.closed());
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::GeometryMismatch { tensor, expected, .. }
                if tensor.ends_with("1.block_sparse_moe.shared_experts.gate_proj.weight")
                    && *expected == vec![MOE_INTER * SHARED_EXPERTS, HIDDEN]
        )),
        "{:?}",
        outcome.defects
    );
}
