//! The hybrid Mamba2Attn witness, in miniature (the OuteAI rehearsal):
//! a checkpoint interleaving Mamba2 mixers with conv-QKV attention
//! blocks admits with every judgment explicit — the mamba_ssm key
//! dialect with its recorded family defaults, the conv-QKV operator
//! (never plain softmax), the tensor-evidence base settlement for an
//! ambiguous `attention_layers_idx`, and the declared MLP absence.
//!
//! Awkward miniature dimensions, every width distinct: hidden 12; mamba
//! mixers d_inner 24, 4 heads × 6, state 5, conv 3 (in_proj rows 62);
//! attention blocks 2 heads × 4, 1 KV head, conv 2, rotary 2 of 4
//! (in_proj rows 16). `attention_layers_idx: [1, 3]` over 4 layers fits
//! BOTH index bases — the live J5 shape — so admission itself exercises
//! the settlement.

use std::path::Path;

use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::graph::roles::classify_stack_tensor_on;
use crate::format::vindex3::graph::{
    build_from_inventories, execution_completeness, CompletenessDefect, LayerOperator, OperandRole,
};
use crate::format::vindex3::plan::plan_system;
use larql_models::inventory::build_inventory;

const H_HIDDEN: usize = 12;
const H_LAYERS: usize = 4;
const H_VOCAB: usize = 23;
// The Mamba2 mixer's widths (mamba_ssm dialect, defaults recorded).
const H_M_HEADS: usize = 4;
const H_M_HEAD_DIM: usize = 6;
const H_M_STATE: usize = 5;
const H_M_CONV: usize = 3;
const H_M_D_INNER: usize = 24;
const H_M_CONV_DIM: usize = H_M_D_INNER + 2 * H_M_STATE; // 34
const H_M_IN_PROJ_ROWS: usize = 2 * H_M_D_INNER + 2 * H_M_STATE + H_M_HEADS; // 62
                                                                             // The conv-QKV attention block's widths.
const H_A_HEADS: usize = 2;
const H_A_KV_HEADS: usize = 1;
const H_A_HEAD_DIM: usize = 4;
const H_A_CONV: usize = 2;
const H_A_QKV_ROWS: usize = (H_A_HEADS + 2 * H_A_KV_HEADS) * H_A_HEAD_DIM; // 16
const H_A_OUT_WIDTH: usize = H_A_HEADS * H_A_HEAD_DIM; // 8
/// Zero-based attention layers — the reading the tensor estate proves.
const H_ATTN_LAYERS: [usize; 2] = [1, 3];

/// Write the miniature hybrid checkpoint, in the OuteAI declaration
/// shape: the mamba_ssm key dialect (no `n_groups`, no `rms_norm`), the
/// `attention_*` block, a base-ambiguous `attention_layers_idx`, and
/// `mlp_intermediate_size: 0`.
fn miniature_hybrid(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "mamba2",
        "hidden_size": H_HIDDEN,
        "num_hidden_layers": H_LAYERS,
        "vocab_size": H_VOCAB,
        "intermediate_size": H_M_D_INNER,
        "state_size": H_M_STATE,
        "mamba2_num_heads": H_M_HEADS,
        "mamba2_head_dim": H_M_HEAD_DIM,
        "expand": 2,
        "mamba2_conv_kernel": H_M_CONV,
        "chunk_size": 8,
        "time_step_limit": [0.0, "Infinity"],
        "use_mamba2_bias": false,
        "use_conv_bias": true,
        "num_attention_heads": H_A_HEADS,
        "num_key_value_heads": H_A_KV_HEADS,
        "attention_head_dim": H_A_HEAD_DIM,
        "attention_conv_kernel": H_A_CONV,
        "rope_emb_dim": 2,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "use_attention_qkv_bias": false,
        "use_attention_out_bias": false,
        "attention_layers_idx": [1, 3],
        "mlp_intermediate_size": 0,
        "mlp_padding_size": 128,
        "use_mlp_bias": false,
        "hidden_act": "silu",
        "layer_norm_epsilon": 1e-5,
        "residual_in_fp32": true,
        "max_position_embeddings": 64,
        "tie_embedding_weights": true
    })
    .to_string()
    .replace("\"Infinity\"", "Infinity");
    std::fs::write(dir.join("config.json"), config).unwrap();

    let mut shard = ShardBuilder::new();
    let mut push = |name: String, shape: &[usize], values: Vec<f32>| {
        shard.push(&name, shape, &values);
    };
    push(
        "backbone.embeddings.weight".into(),
        &[H_VOCAB, H_HIDDEN],
        lcg_values(H_VOCAB * H_HIDDEN, 1),
    );
    push(
        "backbone.norm_f.weight".into(),
        &[H_HIDDEN],
        norm_values(H_HIDDEN, 2),
    );
    for layer in 0..H_LAYERS {
        let seed = 700 + layer as u64 * 20;
        let p = format!("backbone.layers.{layer}");
        push(
            format!("{p}.norm.weight"),
            &[H_HIDDEN],
            norm_values(H_HIDDEN, seed),
        );
        if H_ATTN_LAYERS.contains(&layer) {
            push(
                format!("{p}.mixer.in_proj.weight"),
                &[H_A_QKV_ROWS, H_HIDDEN],
                lcg_values(H_A_QKV_ROWS * H_HIDDEN, seed + 1),
            );
            push(
                format!("{p}.mixer.conv1d.weight"),
                &[H_A_QKV_ROWS, 1, H_A_CONV],
                lcg_values(H_A_QKV_ROWS * H_A_CONV, seed + 2),
            );
            push(
                format!("{p}.mixer.conv1d.bias"),
                &[H_A_QKV_ROWS],
                lcg_values(H_A_QKV_ROWS, seed + 3),
            );
            push(
                format!("{p}.mixer.out_proj.weight"),
                &[H_HIDDEN, H_A_OUT_WIDTH],
                lcg_values(H_HIDDEN * H_A_OUT_WIDTH, seed + 4),
            );
        } else {
            push(
                format!("{p}.mixer.in_proj.weight"),
                &[H_M_IN_PROJ_ROWS, H_HIDDEN],
                lcg_values(H_M_IN_PROJ_ROWS * H_HIDDEN, seed + 1),
            );
            push(
                format!("{p}.mixer.conv1d.weight"),
                &[H_M_CONV_DIM, 1, H_M_CONV],
                lcg_values(H_M_CONV_DIM * H_M_CONV, seed + 2),
            );
            push(
                format!("{p}.mixer.conv1d.bias"),
                &[H_M_CONV_DIM],
                lcg_values(H_M_CONV_DIM, seed + 3),
            );
            push(
                format!("{p}.mixer.A_log"),
                &[H_M_HEADS],
                lcg_values(H_M_HEADS, seed + 4),
            );
            push(
                format!("{p}.mixer.D"),
                &[H_M_HEADS],
                lcg_values(H_M_HEADS, seed + 5),
            );
            push(
                format!("{p}.mixer.dt_bias"),
                &[H_M_HEADS],
                lcg_values(H_M_HEADS, seed + 6),
            );
            push(
                format!("{p}.mixer.norm.weight"),
                &[H_M_D_INNER],
                norm_values(H_M_D_INNER, seed + 7),
            );
            push(
                format!("{p}.mixer.out_proj.weight"),
                &[H_HIDDEN, H_M_D_INNER],
                lcg_values(H_HIDDEN * H_M_D_INNER, seed + 8),
            );
        }
    }
    shard.write(dir);
}

/// **The hybrid admits with every judgment explicit and nothing
/// blocking.** The base-ambiguous index set is settled by the tensor
/// estate; the census names 2 full / 2 Mamba2 recurrent; both operation
/// surfaces are present; and the per-layer table carries the two
/// operators — never plain softmax on the attention layers.
#[test]
fn a_hybrid_checkpoint_admits_with_both_programs_declared() {
    let dir = tempfile::tempdir().unwrap();
    miniature_hybrid(dir.path());
    let inventory = build_inventory(dir.path()).unwrap();
    let plan = plan_system(&[("hybrid-mini".to_string(), inventory.clone())]);
    let blocking: Vec<String> = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking: {blocking:?}");

    let census = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("census finding");
    assert!(census.detail.contains("2 full"), "{}", census.detail);
    assert!(
        census.detail.contains("2 Mamba2 recurrent"),
        "{}",
        census.detail
    );

    let component = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .expect("target component");
    let operators: Vec<&LayerOperator> = component
        .attention
        .as_ref()
        .expect("per-layer table")
        .iter()
        .map(|l| &l.operator)
        .collect();
    assert_eq!(
        operators,
        vec![
            &LayerOperator::Mamba2,
            &LayerOperator::ConvQkvAttention,
            &LayerOperator::Mamba2,
            &LayerOperator::ConvQkvAttention,
        ],
        "the attention layers must carry their own operator, never softmax"
    );
    let surface = component.execution.as_ref().expect("surface");
    let mamba2 = surface.mamba2.expect("mamba2 surface");
    // The dialect's recorded family defaults, live in the graph.
    assert_eq!(mamba2.geometry.n_groups, 1);
    assert!(mamba2.geometry.rms_norm);
    let conv_qkv = surface.conv_qkv.expect("conv-qkv surface");
    assert_eq!(conv_qkv.qkv_rows(), H_A_QKV_ROWS);
    assert_eq!(conv_qkv.rotary_dim, 2);
}

/// **The same tensor name is two different operands under two
/// operators** — the collision the operator-gated tables exist to
/// resolve, one lineage over from KDA/MLA.
#[test]
fn the_mixer_spelling_classifies_by_the_layer_operator() {
    let name = "1.mixer.in_proj.weight";
    assert_eq!(
        classify_stack_tensor_on(name, LayerOperator::Mamba2),
        Some((1, OperandRole::Mamba2InProj))
    );
    assert_eq!(
        classify_stack_tensor_on(name, LayerOperator::ConvQkvAttention),
        Some((1, OperandRole::ConvQkvInProj))
    );
    // The shared pre-mixer norm is the SAME declaration on both kinds.
    assert_eq!(
        classify_stack_tensor_on("3.norm.weight", LayerOperator::ConvQkvAttention),
        Some((3, OperandRole::Mamba2PreMixerNorm))
    );
}

/// **A program that runs conv-QKV attention requires its surface group**
/// — stripping it is a named completeness defect, exactly as the mamba2
/// group behaves.
#[test]
fn stripping_the_conv_qkv_surface_is_a_named_defect() {
    let dir = tempfile::tempdir().unwrap();
    miniature_hybrid(dir.path());
    let inventory = build_inventory(dir.path()).unwrap();
    let mut graph = build_from_inventories(&[("hybrid-mini".to_string(), inventory)]).graph;
    let component = graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    component.execution.as_mut().unwrap().conv_qkv = None;
    let named: Vec<String> = execution_completeness(&graph)
        .into_iter()
        .filter(|d| {
            matches!(
                d,
                CompletenessDefect::MissingOperationSurface {
                    operation: "conv-qkv attention",
                    ..
                }
            )
        })
        .map(|d| d.to_string())
        .collect();
    assert_eq!(named.len(), 1, "the missing group must be named: {named:?}");
}

/// **The hybrid encodes — closure holds over BOTH operand programs —
/// and execution refuses by name.** The plan carries the two operators
/// with every operand accounted (5 on an attention layer, 9 on a mixer
/// layer); preparing it is refused as "represented but not executable",
/// and the continuation planner refuses to flatten the two-region state
/// (KV cache AND conv history) to the cache alone. The executor rung
/// flips both refusals.
#[test]
fn the_hybrid_encodes_and_execution_refuses_by_name() {
    use crate::format::vindex3::inspect::inspect_container;
    use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
    use crate::format::vindex3::opplan::exec::operands::OperandStore;
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::production::ProductionBackend;
    use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};

    let dir = tempfile::tempdir().unwrap();
    miniature_hybrid(dir.path());
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("hybrid-mini.vindex3");
    encode_checkpoint(dir.path(), &container).expect("closure holds over both programs");

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("closure held");
    for (index, layer) in plan.layers.iter().enumerate() {
        if H_ATTN_LAYERS.contains(&index) {
            let LayerAttention::ConvQkv(op) = &layer.attention else {
                panic!("layer {index} must be conv-QKV: {:?}", layer.attention);
            };
            assert!(op.conv1d_bias.is_some(), "use_conv_bias: true");
            assert!(layer.ffn.is_none(), "no MLP exists in this lineage");
            assert_eq!(layer.operands_accounted, 5);
        } else {
            assert!(
                matches!(layer.attention, LayerAttention::Mamba2(_)),
                "layer {index} must be the mixer"
            );
            assert_eq!(layer.operands_accounted, 9);
        }
    }

    // Represented is not executable: preparation refuses by name.
    let store = OperandStore::open(&container, &inspection).unwrap();
    let Err(err) = PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    ) else {
        panic!("no executor exists for this operator yet — load must refuse");
    };
    assert!(
        err.to_string().contains("represented but not executable"),
        "the refusal must say WHY: {err}"
    );

    // And the continuation planner refuses to flatten the two-region
    // state to the KV half.
    let err = plan_continuation_geometry(&plan).expect_err("two regions, one vocabulary");
    assert!(err.contains("conv history"), "{err}");
    assert!(err.contains("refusing to flatten"), "{err}");
}
