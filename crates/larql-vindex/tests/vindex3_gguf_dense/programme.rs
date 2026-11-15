//! The graph's model against the target's built-in one, the role
//! assignment, the tensor planner and the arch keys — each driven from
//! the fixture's real operation plan, mutated one fact at a time.

use larql_models::config::{Activation, PositionPolicy};
use larql_models::loading::gguf::GgufValue;
use larql_vindex::format::vindex3::fixtures::*;
use larql_vindex::format::vindex3::fixtures_routed::miniature_routed;
use larql_vindex::format::vindex3::gguf::dense::gemma3::GEMMA3;
use larql_vindex::format::vindex3::gguf::dense::metadata::{arch_entries, ArchFacts};
use larql_vindex::format::vindex3::gguf::dense::programme::{
    read_programme, Programme, RopeFacts, SlidingFacts,
};
use larql_vindex::format::vindex3::gguf::dense::roles::{assign_roles, RoleAssignment};
use larql_vindex::format::vindex3::gguf::dense::target::{
    supported_families, target_for_family, AttentionScaleRule, DenseRole, DenseTarget,
};
use larql_vindex::format::vindex3::gguf::dense::tensors::{plan_tensors, ShapeFacts, SourceTensor};
use larql_vindex::format::vindex3::gguf::dense::{export_dense, DenseExportOptions};
use larql_vindex::format::vindex3::graph::policy::AttentionSpan;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};

use super::common::Fixture;

fn fixture_plan() -> (Fixture, ComponentOpPlan) {
    let fx = Fixture::new();
    let inspection = inspect_container(&fx.container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &fx.container, "target").unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    (fx, plan)
}

fn refusals(plan: &ComponentOpPlan) -> String {
    read_programme(plan, &GEMMA3, GEMMA3_FIXTURE_HIDDEN)
        .expect_err("the programme must refuse")
        .join("\n")
}

fn softmax(
    plan: &mut ComponentOpPlan,
    layer: usize,
) -> &mut larql_vindex::format::vindex3::opplan::AttentionOp {
    plan.layers[layer].attention.softmax_mut().unwrap()
}

#[test]
fn the_fixture_programme_reads_cleanly() {
    let (_fx, plan) = fixture_plan();
    let p = read_programme(&plan, &GEMMA3, GEMMA3_FIXTURE_HIDDEN).unwrap();
    assert_eq!(
        p,
        Programme {
            num_q_heads: GEMMA3_FIXTURE_Q_HEADS,
            num_kv_heads: GEMMA3_FIXTURE_KV_HEADS,
            head_dim: GEMMA3_FIXTURE_HEAD_DIM,
            intermediate_size: GEMMA3_FIXTURE_INTER,
            full_rope: RopeFacts {
                theta: GEMMA3_FIXTURE_GLOBAL_THETA,
                linear_factor: Some(GEMMA3_FIXTURE_ROPE_FACTOR)
            },
            sliding: Some(SlidingFacts {
                window: GEMMA3_FIXTURE_WINDOW,
                theta: GEMMA3_FIXTURE_LOCAL_THETA,
                period: GEMMA3_FIXTURE_PERIOD
            }),
        }
    );
}

#[test]
fn built_in_behaviour_the_graph_contradicts_refuses() {
    let (_fx, base) = fixture_plan();
    let check = |mutate: &dyn Fn(&mut ComponentOpPlan), needle: &str| {
        let mut plan = base.clone();
        mutate(&mut plan);
        let msg = refusals(&plan);
        assert!(msg.contains(needle), "expected `{needle}` in:\n{msg}");
    };
    check(
        &|p| p.embedding.as_mut().unwrap().scale = Some(1.0),
        "embedding scale",
    );
    check(
        &|p| p.output.as_mut().unwrap().multiplier = Some(2.0),
        "multiplier",
    );
    check(
        &|p| p.layers[1].residual_scale = Some(0.5),
        "scales its residual",
    );
    check(&|p| softmax(p, 0).score_scale = 1.0, "scales scores");
    check(
        &|p| softmax(p, 0).logit_softcapping = Some(50.0),
        "soft-caps",
    );
    check(&|p| softmax(p, 0).v_from_k = true, "K-as-V");
    check(
        &|p| {
            let q = softmax(p, 0).q.clone();
            softmax(p, 0).q_bias = Some(q);
        },
        "biases",
    );
    check(
        &|p| softmax(p, 0).position = PositionPolicy::None,
        "position",
    );
    check(&|p| softmax(p, 0).window = None, "slides without a window");
    check(&|p| softmax(p, 0).span = AttentionSpan::Windowed, "span");
    check(&|p| softmax(p, 2).num_q_heads += 1, "attention geometry");
    check(
        &|p| softmax(p, 3).span = AttentionSpan::Full,
        "no single period",
    );
    check(
        &|p| softmax(p, GEMMA3_FIXTURE_PERIOD - 1).span = AttentionSpan::Sliding,
        "every layer slides",
    );
    check(
        &|p| {
            if let Some(LayerFfn::Dense(f)) = p.layers[0].ffn.as_mut() {
                f.activation = Activation::Silu;
            }
        },
        "FFN",
    );
    check(
        &|p| {
            if let Some(LayerFfn::Dense(f)) = p.layers[3].ffn.as_mut() {
                f.intermediate_size += 1;
            }
        },
        "FFN width",
    );
    check(&|p| p.layers.clear(), "nothing for the target");
}

#[test]
fn roles_follow_the_plan() {
    let (_fx, mut plan) = fixture_plan();
    let roles = assign_roles(&plan).unwrap();
    assert_eq!(roles.len(), 2 + GEMMA3_FIXTURE_LAYERS * 13);
    let norm = roles
        .iter()
        .find(|r| r.role == DenseRole::PostFfnNorm)
        .unwrap();
    assert_eq!(norm.weight_offset, 1.0);
    assert!(
        roles.iter().all(|r| r.role != DenseRole::OutputHead),
        "tied"
    );

    // An untied head binds its own operand.
    let mut untied = plan.clone();
    let mut head = untied.output.as_ref().unwrap().projection.clone();
    head.tensor = "head.weight".into();
    untied.output.as_mut().unwrap().projection = head;
    assert!(assign_roles(&untied)
        .unwrap()
        .iter()
        .any(|r| r.role == DenseRole::OutputHead));

    plan.layers[2].ffn = None;
    assert!(assign_roles(&plan)
        .unwrap_err()
        .to_string()
        .contains("has no FFN"));
}

fn source(object: &str, name: &str, dtype: &str, shape: &[u64]) -> SourceTensor {
    SourceTensor {
        object: object.into(),
        name: name.into(),
        dtype: dtype.into(),
        shape: shape.to_vec(),
        span: (std::path::PathBuf::from("x"), 0, 0),
    }
}

fn role(object: &str, tensor: &str, role: DenseRole, layer: Option<usize>) -> RoleAssignment {
    RoleAssignment {
        object: object.into(),
        tensor: tensor.into(),
        role,
        layer,
        weight_offset: 0.0,
    }
}

#[test]
fn the_tensor_planner_refuses_what_it_cannot_lower() {
    let (_fx, plan) = fixture_plan();
    let programme = read_programme(&plan, &GEMMA3, GEMMA3_FIXTURE_HIDDEN).unwrap();
    let facts = ShapeFacts {
        programme: &programme,
        hidden_size: GEMMA3_FIXTURE_HIDDEN,
        vocab_size: GEMMA3_FIXTURE_VOCAB,
    };
    let h = GEMMA3_FIXTURE_HIDDEN as u64;
    let v = GEMMA3_FIXTURE_VOCAB as u64;
    let sources = vec![
        source("e", "weight", "F16", &[v, h]),
        source("n", "weight", "F32", &[h + 1]),
        source("x", "stray", "F32", &[1]),
        source("o", "weight", "F32", &[v, h]),
    ];
    let roles = vec![
        role("e", "weight", DenseRole::Embedding, None),
        role("n", "weight", DenseRole::FinalNorm, None),
        role("o", "weight", DenseRole::OutputHead, None),
        role("o", "weight", DenseRole::OutputHead, None),
        role("gone", "weight", DenseRole::Query, Some(0)),
    ];
    let msg = plan_tensors(&sources, &roles, &GEMMA3, &facts)
        .unwrap_err()
        .join("\n");
    for needle in [
        "stored F16",
        "has shape",
        "no role in the operation plan",
        "carries two roles",
        "no selected segment holds it",
        "layer 0 binds no PreAttentionNorm",
    ] {
        assert!(msg.contains(needle), "expected `{needle}` in:\n{msg}");
    }

    // A target without a name for a role it is handed refuses by role.
    let nameless = DenseTarget {
        tensor_names: &[(DenseRole::Embedding, "token_embd.weight")],
        required_layer_roles: &[],
        ..GEMMA3
    };
    let only_norm = vec![source("n", "weight", "F32", &[h])];
    let msg = plan_tensors(
        &only_norm,
        &[role("n", "weight", DenseRole::FinalNorm, None)],
        &nameless,
        &facts,
    )
    .unwrap_err()
    .join("\n");
    assert!(
        msg.contains("no tensor name") && msg.contains("binds no Embedding"),
        "{msg}"
    );

    // An F32 matrix keeps its precision.
    let f32_head = vec![
        source("e", "weight", "F32", &[v, h]),
        source("n", "weight", "BF16", &[h]),
    ];
    let planned = plan_tensors(
        &f32_head,
        &[
            role("e", "weight", DenseRole::Embedding, None),
            role("n", "weight", DenseRole::FinalNorm, None),
        ],
        &nameless_with_norm(),
        &facts,
    )
    .unwrap();
    assert_eq!(planned[0].target_type, 0);
}

fn nameless_with_norm() -> DenseTarget {
    DenseTarget {
        tensor_names: &[
            (DenseRole::Embedding, "token_embd.weight"),
            (DenseRole::FinalNorm, "output_norm.weight"),
        ],
        required_layer_roles: &[],
        ..GEMMA3
    }
}

#[test]
fn the_target_table_answers_by_role_and_family() {
    assert_eq!(
        target_for_family("gemma3_text").map(|t| t.architecture),
        Some("gemma3")
    );
    assert!(target_for_family("qwen3").is_none());
    assert!(supported_families().contains(&"gemma3"));
    assert_eq!(
        GEMMA3.tensor_name(DenseRole::Query, Some(7)).as_deref(),
        Some("blk.7.attn_q.weight")
    );
    assert_eq!(GEMMA3.tensor_name(DenseRole::Query, None), None);
    assert_eq!(GEMMA3.tensor_name(DenseRole::Embedding, Some(1)), None);
    assert_eq!(AttentionScaleRule::InverseSqrtHeadDim.value(256), 0.0625);
    assert!(DenseRole::KeyNorm.is_norm() && !DenseRole::FfnUp.is_norm());
}

#[test]
fn arch_keys_state_a_non_default_period_and_a_softcap() {
    let programme = Programme {
        num_q_heads: 8,
        num_kv_heads: 4,
        head_dim: 256,
        intermediate_size: 1024,
        full_rope: RopeFacts {
            theta: 1e6,
            linear_factor: None,
        },
        sliding: Some(SlidingFacts {
            window: 512,
            theta: 1e4,
            period: 4,
        }),
    };
    let entries = arch_entries(&ArchFacts {
        target: &GEMMA3,
        programme: &programme,
        num_layers: 8,
        hidden_size: 64,
        context_length: 8192,
        rms_eps: 1e-6,
        file_type: 0,
        final_logit_softcapping: Some(30.0),
    });
    let get = |k: &str| {
        entries
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(
        get("gemma3.attention.sliding_window_pattern"),
        Some(GgufValue::U32(4))
    );
    assert_eq!(
        get("gemma3.final_logit_softcapping"),
        Some(GgufValue::F32(30.0))
    );
    assert_eq!(get("gemma3.rope.scaling.type"), None);

    let no_swa = Programme {
        sliding: None,
        ..programme
    };
    let entries = arch_entries(&ArchFacts {
        target: &GEMMA3,
        programme: &no_swa,
        num_layers: 8,
        hidden_size: 64,
        context_length: 8192,
        rms_eps: 1e-6,
        file_type: 0,
        final_logit_softcapping: None,
    });
    assert!(entries
        .iter()
        .all(|(k, _)| !k.contains("swa") && !k.contains("sliding")));
}

#[test]
fn a_graph_the_target_cannot_reproduce_refuses_the_export() {
    let fx = Fixture::new();
    fx.edit_json("system_graph.json", |g| {
        g["components"][0]["execution"]["ffn"]["activation"] = "silu".into();
    });
    let err = export_dense(
        &fx.container,
        &fx.out("x.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("gemma3 disagree"), "{err}");
}

#[test]
fn a_container_without_a_primary_text_model_or_surface_refuses() {
    let fx = Fixture::new();
    fx.edit_json("system_graph.json", |g| {
        g["components"][0]["execution"]
            .as_object_mut()
            .unwrap()
            .remove("head");
    });
    let err = export_dense(
        &fx.container,
        &fx.out("x.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("no head"), "{err}");

    fx.edit_json("system_graph.json", |g| {
        g["components"][0]
            .as_object_mut()
            .unwrap()
            .remove("execution");
    });
    let err = export_dense(
        &fx.container,
        &fx.out("x.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("no execution surface"), "{err}");

    fx.edit_json("system_graph.json", |g| {
        g["components"][0]["role"] = "drafter".into()
    });
    let err = export_dense(
        &fx.container,
        &fx.out("x.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("Absent"), "{err}");
}

fn plan_of(write: impl FnOnce(&std::path::Path)) -> ComponentOpPlan {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), "other");
    let inspection = inspect_container(container.path(), false).unwrap();
    let component = inspection
        .graph
        .primary_text_component()
        .unwrap()
        .id
        .clone();
    plan_component_ops(&inspection, container.path(), &component)
        .unwrap()
        .plan
        .expect("the fixture plans")
}

#[test]
fn operators_and_ffns_the_dense_lowering_has_no_roles_for_refuse() {
    let recurrent = assign_roles(&plan_of(hybrid_lllf_f32_model)).unwrap_err();
    assert!(
        recurrent.to_string().contains("softmax attention only"),
        "{recurrent}"
    );
    let routed = assign_roles(&plan_of(miniature_routed)).unwrap_err();
    assert!(routed.to_string().contains("routed FFN"), "{routed}");
}
