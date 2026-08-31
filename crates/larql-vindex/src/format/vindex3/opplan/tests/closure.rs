//! Operand-closure gates: every shortfall blocks, itemised — including
//! the discovery that motivated this rung, an operand implying a
//! primitive the surface's judged semantics do not carry.

use super::encoded_fixture;
use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect};
use crate::format::vindex3::plan::tests_support::custom_artifact;

/// A dense single-layer artifact with a precise operand estate, plus the
/// mutations each test applies.
fn dense_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    })
}

/// The closed 10-operand estate: 4 attn + 2 norms (two-norm placement) +
/// 3 mlp + nothing else, plus embedding/final-norm/head.
fn dense_tensors() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("model.embed_tokens.weight", vec![128, 64]),
        ("model.norm.weight", vec![64]),
        ("lm_head.weight", vec![128, 64]),
        ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
        ("model.layers.0.self_attn.k_proj.weight", vec![16, 64]),
        ("model.layers.0.self_attn.v_proj.weight", vec![16, 64]),
        ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
        ("model.layers.0.input_layernorm.weight", vec![64]),
        ("model.layers.0.post_attention_layernorm.weight", vec![64]),
        ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
        ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
        ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
    ]
}

/// Encode a dense variant and plan its target component.
fn plan_variant(
    mutate: impl FnOnce(&mut Vec<(&'static str, Vec<usize>)>),
) -> crate::format::vindex3::opplan::OpPlanOutcome {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    mutate(&mut tensors);
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (*name, shape.as_slice()))
        .collect();
    let inventory = custom_artifact(dir.path(), &dense_config(), &borrowed);
    let named = vec![("dense-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    plan_component_ops(&inspection, out.path(), "target").unwrap()
}

/// The untouched estate closes: 9/9 stack operands accounted (4 attn +
/// 2 norms + 3 mlp).
#[test]
fn a_closed_estate_plans() {
    let outcome = plan_variant(|_| {});
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert_eq!(plan.layers[0].operands_accounted, 9);
    assert!(plan.layers[0].post_attention_norm.is_none()); // two-norm
}

/// A checkpoint with no standalone `lm_head.weight` and no declared
/// `tie_word_embeddings` — the near-universal tied-embeddings shape, and
/// exactly Gemma 2's — still gets an output head, synthesised from the
/// embedding object rather than left absent.
#[test]
fn a_missing_lm_head_falls_back_to_the_embedding_object_when_tied() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| *name != "lm_head.weight");
    });
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let output = plan.output.expect("tied embeddings must synthesise a head");
    assert_eq!(output.projection.object, "target.embedding");
}

/// The same missing tensor, but `tie_word_embeddings: false` — the
/// GPT-OSS/OLMoE shape. The fallback must NOT fire: reusing the embedding
/// here would silently serve the wrong projection for a checkpoint that
/// lost its real one, so the plan stays head-less rather than guessing.
#[test]
fn a_missing_lm_head_stays_absent_when_explicitly_untied() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = dense_config();
    config["tie_word_embeddings"] = serde_json::json!(false);
    let mut tensors = dense_tensors();
    tensors.retain(|(name, _)| *name != "lm_head.weight");
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (*name, shape.as_slice()))
        .collect();
    let inventory = custom_artifact(dir.path(), &config, &borrowed);
    let named = vec![("dense-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, out.path(), "target").unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert!(plan.output.is_none());
}

/// THE discovery gate: an attention-gate operand whose judged semantics
/// the surface does not carry refuses with the primitive named — the
/// exact refusal the real Glimmer container must produce.
#[test]
fn an_attention_gate_operand_names_the_required_primitive() {
    let outcome = plan_variant(|tensors| {
        tensors.push(("model.layers.0.self_attn.gate_proj.weight", vec![64, 64]));
    });
    assert!(outcome.plan.is_none());
    let defect = outcome
        .defects
        .iter()
        .find_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } => Some((tensor.clone(), required_primitive.clone())),
            _ => None,
        })
        .expect("the gate operand must be reported");
    assert!(defect.0.contains("self_attn.gate_proj"));
    assert!(defect.1.contains("attention output gate"));
}

/// A tensor no role classifies blocks — never a silently skipped file.
#[test]
fn an_unclassified_operand_blocks() {
    let outcome = plan_variant(|tensors| {
        tensors.push(("model.layers.0.self_attn.mystery.weight", vec![64, 64]));
    });
    assert!(outcome.plan.is_none());
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::UnclassifiedOperand { tensor, .. } if tensor.contains("mystery")
    )));
}

/// A missing operand for an implied op blocks with the role named.
#[test]
fn a_missing_value_projection_blocks() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| !name.contains("v_proj"));
    });
    assert!(outcome.plan.is_none());
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::MissingOperand {
            layer: 0,
            role: OperandRole::AttnV
        }
    )));
}

/// A stored shape contradicting the surface's geometry blocks.
#[test]
fn a_geometry_mismatch_blocks() {
    let outcome = plan_variant(|tensors| {
        for (name, shape) in tensors.iter_mut() {
            if name.contains("k_proj") {
                *shape = vec![64, 64]; // surface says 2 kv-heads * 8 = 16 rows
            }
        }
    });
    assert!(outcome.plan.is_none());
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::GeometryMismatch { tensor, expected, actual }
            if tensor.contains("k_proj") && expected == &vec![16, 64] && actual == &vec![64, 64]
    )));
}

/// Closure judges each layer against ITS OWN head geometry: a layer whose
/// policy declares a different `head_dim` / KV-head count expects the
/// matching projection shapes, and the mismatch names that layer's
/// tensors with the per-layer expectation — the component surface's
/// geometry is not what closure reads.
#[test]
fn closure_checks_shapes_against_the_layers_own_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let tensors = dense_tensors();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (*name, shape.as_slice()))
        .collect();
    let mut inventory = custom_artifact(dir.path(), &dense_config(), &borrowed);
    // The config says head_dim 8 / 2 KV heads; declare layer 0 at four
    // times the head width and one KV head, as a Gemma-4-style global
    // layer would, without touching the stored tensors. (Not 16 / 1: that
    // gives kv_rows 16 = 2 × 8, and a coincident product would hide the
    // K/V check.)
    let varied_head_dim = 32;
    let varied_kv_heads = 1;
    inventory.resolved.layers[0].head_dim = varied_head_dim;
    inventory.resolved.layers[0].num_kv_heads = varied_kv_heads;
    let named = vec![("dense-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, out.path(), "target").unwrap();
    assert!(outcome.plan.is_none());
    let hidden = 64;
    let q_heads = 8;
    // q/o expect q_heads × 32 rows; k/v expect 1 × 32 rows.
    let expect_q = vec![q_heads * varied_head_dim, hidden];
    let expect_kv = vec![varied_kv_heads * varied_head_dim, hidden];
    for (needle, expected) in [
        ("q_proj", &expect_q),
        ("k_proj", &expect_kv),
        ("v_proj", &expect_kv),
    ] {
        assert!(
            outcome.defects.iter().any(|d| matches!(
                d,
                ClosureDefect::GeometryMismatch { tensor, expected: e, actual: _ }
                    if tensor.contains(needle) && e == expected
            )),
            "{needle} must be judged against the layer's geometry: {:?}",
            outcome.defects
        );
    }
}

/// A single-sided QK-norm pair blocks with the missing half named.
#[test]
fn a_single_sided_qk_norm_blocks() {
    let outcome = plan_variant(|tensors| {
        tensors.push(("model.layers.0.self_attn.q_norm.weight", vec![8]));
    });
    assert!(outcome.plan.is_none());
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::MissingOperand {
            layer: 0,
            role: OperandRole::AttnKNorm
        }
    )));
}

/// A multi-tensor single-operand object (two embedding tensors) blocks.
#[test]
fn a_multi_tensor_embedding_object_blocks() {
    let outcome = plan_variant(|tensors| {
        tensors.push(("model.embed_tokens.extra.weight", vec![128, 64]));
    });
    assert!(outcome.plan.is_none());
    assert!(outcome.defects.iter().any(|d| matches!(
        d,
        ClosureDefect::ObjectShape { object, .. } if object == "target.embedding"
    )));
}

/// Tampering with the persisted placement or FFN gating turns consumed
/// operands into unrepresented ones — the plan refuses rather than
/// running the wrong program shape.
#[test]
fn tampered_surface_ops_turn_operands_unrepresented() {
    let fixture = encoded_fixture();
    let graph_path = fixture
        .container
        .path()
        .join(crate::format::vindex3::encode::SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    let target = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    // The four-norm fixture claims two-norm placement and ungated FFN.
    target["execution"]["norm"]["placement"] = serde_json::json!("pre_only");
    target["execution"]["ffn"]["ffn_type"] = serde_json::json!("standard");
    std::fs::write(&graph_path, graph.to_string()).unwrap();

    let inspection = inspect_container(fixture.container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, &fixture.root, "target").unwrap();
    assert!(outcome.plan.is_none());
    let primitives: Vec<&str> = outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                required_primitive, ..
            } => Some(required_primitive.as_str()),
            _ => None,
        })
        .collect();
    assert!(primitives.iter().any(|p| p.contains("four-norm placement")));
    assert!(primitives.iter().any(|p| p.contains("gated FFN")));
}

/// A component without a policy table cannot be planned.
#[test]
fn a_missing_attention_table_refuses() {
    let fixture = encoded_fixture();
    let graph_path = fixture
        .container
        .path()
        .join(crate::format::vindex3::encode::SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("attention");
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let inspection = inspect_container(fixture.container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, &fixture.root, "target").unwrap();
    assert!(outcome.plan.is_none());
    assert!(outcome
        .defects
        .iter()
        .any(|d| matches!(d, ClosureDefect::MissingAttentionTable { .. })));
}

/// A missing directory entry is a hard error (G3's coherence broke), and
/// an unknown component is named.
#[test]
fn hard_errors_name_their_cause() {
    let fixture = encoded_fixture();
    let err = plan_component_ops(&fixture.inspection, &fixture.root, "nonexistent").unwrap_err();
    assert!(err.to_string().contains("nonexistent"));

    let index_path = fixture
        .container
        .path()
        .join(crate::format::filenames::INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let removed = index["representations"]
        .as_object_mut()
        .unwrap()
        .remove("target.decoder_stack@BF16");
    assert!(removed.is_some());
    std::fs::write(&index_path, index.to_string()).unwrap();
    let inspection = inspect_container(fixture.container.path(), false).unwrap();
    let err = plan_component_ops(&inspection, &fixture.root, "target").unwrap_err();
    assert!(err.to_string().contains("target.decoder_stack@BF16"));
}

/// Every defect variant prints its coordinates — a refusal must be a
/// work item, not a mystery.
#[test]
fn defect_display_names_the_coordinates() {
    let cases = [
        (
            ClosureDefect::MissingSurface {
                component: "target".into(),
            },
            "target",
        ),
        (
            ClosureDefect::MissingAttentionTable {
                component: "draft".into(),
            },
            "draft",
        ),
        (
            ClosureDefect::UnclassifiedOperand {
                object: "target.decoder_stack".into(),
                tensor: "0.mystery.weight".into(),
            },
            "mystery",
        ),
        (
            ClosureDefect::OperandImpliesAbsentOp {
                object: "target.decoder_stack".into(),
                tensor: "0.self_attn.gate_proj.weight".into(),
                required_primitive: "attention output gate".into(),
            },
            "required primitive",
        ),
        (
            ClosureDefect::MissingOperand {
                layer: 7,
                role: OperandRole::AttnV,
            },
            "layer 7",
        ),
        (
            ClosureDefect::DuplicateOperand {
                layer: 2,
                role: OperandRole::FfnUp,
            },
            "layer 2",
        ),
        (
            ClosureDefect::GeometryMismatch {
                tensor: "t".into(),
                expected: vec![16, 64],
                actual: vec![64, 64],
            },
            "[16, 64]",
        ),
        (
            ClosureDefect::ObjectShape {
                object: "target.embedding".into(),
                detail: "expected exactly 1 tensor".into(),
            },
            "target.embedding",
        ),
    ];
    for (defect, needle) in cases {
        let printed = defect.to_string();
        assert!(printed.contains(needle), "`{printed}` lacks `{needle}`");
    }
}

/// A container whose persisted graph lost its surfaces refuses before
/// touching a segment.
#[test]
fn a_stripped_surface_refuses() {
    let fixture = encoded_fixture();
    let graph_path = fixture
        .container
        .path()
        .join(crate::format::vindex3::encode::SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    for component in graph["components"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("execution");
    }
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let inspection = inspect_container(fixture.container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, &fixture.root, "target").unwrap();
    assert!(outcome.plan.is_none());
    assert!(outcome
        .defects
        .iter()
        .any(|d| matches!(d, ClosureDefect::MissingSurface { .. })));
    let _ = &fixture.named; // sources unused: the refusal is container-only
}
