//! **K3-REP-GATE-1 — the output gate, declared, addressed and executed.**
//!
//! Two declared facts, each held to the shipped operands from BOTH sides:
//! `linear_attn_config.use_full_rank_gate` (KDA's output gate is one
//! full-rank `g_proj` instead of the low-rank pair) and
//! `mla_use_output_gate` (MLA gates its aggregated value before `o_proj`).
//! The synthetic `KKKM` hybrid ([`hybrid_kda_mla_f32_model_with`]) stages
//! every agreement and every disagreement between declaration and
//! estate, so the closure rule can be watched holding and each refusal
//! made to fire on its own.
//!
//! The two spellings are the same — `self_attn.g_proj.weight` on a KDA
//! layer and on an MLA layer — and identity is DECLARED: a layer never
//! acquires a gate form by shipping a tensor spelled for it. What it ships
//! is judged against what the component says, and a disagreement is named
//! rather than resolved.
//!
//! Nothing here claims Kimi-K3 executes: its MLA layers carry the
//! unaddressed q-LoRA triple, its own cell.
use larql_models::inventory::build_inventory;

use crate::format::vindex3::encode::{encode_graph, encode_system};
use crate::format::vindex3::fixtures_kimi::{
    hybrid_kda_mla_f32_model_with, HybridGateForms, KdaGateShipped,
};
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, KdaOutputGate, LayerAttention, OpPlanOutcome,
};
use crate::format::vindex3::plan::plan_system;

const ARTIFACT: &str = "kkkm-gates";

struct Staged {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: crate::format::vindex3::inspect::SystemInspection,
    outcome: OpPlanOutcome,
}

/// Encode through the graph seam rather than the production writer: the
/// production encoder refuses to KEEP an estate whose operands do not
/// close (see [`the_production_encoder_refuses_a_disagreeing_estate`]),
/// and the point of most arms here is to prove what closure says BELOW
/// that refusal.
fn stage(forms: HybridGateForms) -> Staged {
    let src = tempfile::tempdir().unwrap();
    hybrid_kda_mla_f32_model_with(src.path(), forms);
    let named = vec![(ARTIFACT.to_string(), build_inventory(src.path()).unwrap())];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    Staged {
        _src: src,
        container,
        inspection,
        outcome,
    }
}

fn implied_absent(outcome: &OpPlanOutcome) -> Vec<(String, String)> {
    outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } => Some((tensor.clone(), required_primitive.clone())),
            _ => None,
        })
        .collect()
}

fn missing(outcome: &OpPlanOutcome) -> Vec<(usize, OperandRole)> {
    outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::MissingOperand { layer, role } => Some((*layer, *role)),
            _ => None,
        })
        .collect()
}

fn executes(staged: &Staged) {
    let plan = staged.outcome.plan.as_ref().expect("closed");
    let store = OperandStore::open(staged.container.path(), &staged.inspection).unwrap();
    execute_text(plan, &store, &[1, 2, 3]).expect("prepares and executes end to end");
}

/// The baseline: Kimi Linear's own forms, unchanged by this rung — the
/// pair on every KDA layer, no MLA gate, neither key declared.
#[test]
fn the_low_rank_ungated_baseline_closes_and_executes_unchanged() {
    let staged = stage(HybridGateForms::KIMI_LINEAR);
    assert!(staged.outcome.closed(), "{:?}", staged.outcome.defects);
    let plan = staged.outcome.plan.as_ref().unwrap();
    for layer in &plan.layers {
        match &layer.attention {
            LayerAttention::Kda(k) => {
                assert!(matches!(k.output_gate, KdaOutputGate::LowRank { .. }))
            }
            LayerAttention::Mla(m) => assert!(m.output_gate.is_none()),
            other => panic!("unexpected operator {other:?}"),
        }
    }
    executes(&staged);
}

/// **Kimi-K3's shape.** Both gates declared and shipped: the plan closes
/// with zero defects, the KDA ops carry the full-rank form as a TYPE, the
/// MLA op carries its gate operand, and the whole stack prepares through
/// the public loader and executes.
#[test]
fn both_gates_declared_and_shipped_close_and_execute() {
    let staged = stage(HybridGateForms::KIMI_K3);
    assert!(staged.outcome.closed(), "{:?}", staged.outcome.defects);
    let plan = staged.outcome.plan.as_ref().unwrap();
    let mut kda = 0;
    let mut mla = 0;
    for layer in &plan.layers {
        match &layer.attention {
            LayerAttention::Kda(k) => {
                kda += 1;
                let KdaOutputGate::FullRank { g_proj } = &k.output_gate else {
                    panic!("declared full rank, planned {:?}", k.output_gate.form());
                };
                assert!(g_proj.tensor.ends_with("self_attn.g_proj.weight"));
                assert_eq!(g_proj.shape, vec![k.value_width(), 32], "[Hv·Dv, hidden]");
            }
            LayerAttention::Mla(m) => {
                mla += 1;
                let gate = m.output_gate.as_ref().expect("declared, shipped");
                assert!(gate.tensor.ends_with("self_attn.g_proj.weight"));
                assert_eq!(gate.shape, vec![m.num_heads * m.v_head_dim, 32]);
            }
            other => panic!("unexpected operator {other:?}"),
        }
    }
    assert_eq!((kda, mla), (3, 1), "the KKKM hybrid");
    executes(&staged);

    // What the Metal loader reads, on the container this rung actually
    // wrote: both declarations reach the graph under the names
    // `kimi_source` consults, so its refusal fires from the surface and
    // not from a missing tensor.
    let graph: serde_json::Value = serde_json::from_slice(
        &std::fs::read(staged.container.path().join("system_graph.json")).unwrap(),
    )
    .unwrap();
    let execution = &graph["components"][0]["execution"];
    assert_eq!(execution["kda_use_full_rank_gate"], true);
    assert!(
        execution["mla"]["output_gate"].is_object(),
        "{}",
        execution["mla"]
    );
}

/// The pair under a full-rank declaration: the shipped form implies a
/// gate the component never chose, and the declared form is missing —
/// both named, on every KDA layer, and nothing plans.
#[test]
fn the_low_rank_pair_under_a_full_rank_declaration_is_refused_by_name() {
    let staged = stage(HybridGateForms {
        kda_declared_full_rank: Some(true),
        kda_shipped: KdaGateShipped::LowRankPair,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.plan.is_none());
    let implied = implied_absent(&staged.outcome);
    assert_eq!(
        implied.len(),
        6,
        "g_a_proj and g_b_proj on three layers: {implied:?}"
    );
    for (tensor, primitive) in &implied {
        assert!(tensor.ends_with("g_a_proj.weight") || tensor.ends_with("g_b_proj.weight"));
        assert!(primitive.contains("`use_full_rank_gate`"), "{primitive}");
        assert!(primitive.contains("low-rank"), "{primitive}");
    }
    let missing = missing(&staged.outcome);
    assert_eq!(
        missing,
        vec![
            (0, OperandRole::KdaGProj),
            (1, OperandRole::KdaGProj),
            (2, OperandRole::KdaGProj)
        ]
    );
}

/// `g_proj` without the declaration: the operand implies a form the
/// component never chose — identity is declared, not read off a tensor —
/// and the declared pair is missing.
#[test]
fn g_proj_without_the_declaration_is_refused_by_name() {
    let staged = stage(HybridGateForms {
        kda_declared_full_rank: None,
        kda_shipped: KdaGateShipped::FullRank,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.plan.is_none());
    let implied = implied_absent(&staged.outcome);
    assert_eq!(implied.len(), 3, "{implied:?}");
    for (tensor, primitive) in &implied {
        assert!(tensor.ends_with("self_attn.g_proj.weight"));
        assert!(
            primitive.contains("declares no `use_full_rank_gate`"),
            "{primitive}"
        );
    }
    let mut missing = missing(&staged.outcome);
    missing.sort();
    assert_eq!(
        missing,
        vec![
            (0, OperandRole::KdaGAProj),
            (0, OperandRole::KdaGBProj),
            (1, OperandRole::KdaGAProj),
            (1, OperandRole::KdaGBProj),
            (2, OperandRole::KdaGAProj),
            (2, OperandRole::KdaGBProj),
        ]
    );
}

/// Declared `false` is the pair, exactly as undeclared is: the
/// reference's own default, written out. A layer shipping `g_proj` under
/// an explicit `false` is refused the same way.
#[test]
fn a_declared_low_rank_form_reads_as_the_pair() {
    let staged = stage(HybridGateForms {
        kda_declared_full_rank: Some(false),
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.closed(), "{:?}", staged.outcome.defects);
    let refused = stage(HybridGateForms {
        kda_declared_full_rank: Some(false),
        kda_shipped: KdaGateShipped::FullRank,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(refused.outcome.plan.is_none());
    assert_eq!(implied_absent(&refused.outcome).len(), 3);
}

/// Declared full rank with no gate operand at all: the missing role is
/// named, and no other defect is invented for it.
#[test]
fn a_declared_full_rank_gate_with_no_operand_names_the_role() {
    let staged = stage(HybridGateForms {
        kda_declared_full_rank: Some(true),
        kda_shipped: KdaGateShipped::None,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.plan.is_none());
    assert!(implied_absent(&staged.outcome).is_empty());
    assert_eq!(
        missing(&staged.outcome),
        vec![
            (0, OperandRole::KdaGProj),
            (1, OperandRole::KdaGProj),
            (2, OperandRole::KdaGProj)
        ]
    );
}

/// `g_proj` on the MLA layer without `mla_use_output_gate`: the same
/// spelling the KDA layers carry, refused for what it would imply on
/// THIS operator.
#[test]
fn an_mla_g_proj_without_the_declaration_is_refused_by_name() {
    let staged = stage(HybridGateForms {
        mla_declared_gate: None,
        mla_shipped_gate: true,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.plan.is_none());
    let implied = implied_absent(&staged.outcome);
    assert_eq!(implied.len(), 1, "{implied:?}");
    // Object-relative, as closure names tensors: `{layer}.{suffix}`.
    assert!(
        implied[0].0.ends_with("3.self_attn.g_proj.weight"),
        "{}",
        implied[0].0
    );
    assert!(
        implied[0].1.contains("declares no `mla_use_output_gate`"),
        "{}",
        implied[0].1
    );
    assert!(missing(&staged.outcome).is_empty());
}

/// A declared MLA gate with no operand names the role on the MLA layer
/// and nowhere else.
#[test]
fn a_declared_mla_gate_with_no_operand_names_the_role() {
    let staged = stage(HybridGateForms {
        mla_declared_gate: Some(true),
        mla_shipped_gate: false,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.plan.is_none());
    assert!(implied_absent(&staged.outcome).is_empty());
    assert_eq!(
        missing(&staged.outcome),
        vec![(3, OperandRole::MlaOutputGate)]
    );
}

/// `mla_use_output_gate: false` is no gate, and an MLA layer shipping
/// `g_proj` under it is refused exactly as under no declaration.
#[test]
fn a_declared_false_mla_gate_is_no_gate() {
    let staged = stage(HybridGateForms {
        mla_declared_gate: Some(false),
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(staged.outcome.closed(), "{:?}", staged.outcome.defects);
    let refused = stage(HybridGateForms {
        mla_declared_gate: Some(false),
        mla_shipped_gate: true,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(refused.outcome.plan.is_none());
    assert_eq!(implied_absent(&refused.outcome).len(), 1);
}

/// **The production encoder keeps only what closes.** Kimi-K3's forms
/// encode through the production writer; a disagreeing estate is refused
/// at encode, naming the tensor and what it implies — nothing written.
#[test]
fn the_production_encoder_refuses_a_disagreeing_estate() {
    let encode = |forms: HybridGateForms| -> Result<(), String> {
        let src = tempfile::tempdir().unwrap();
        hybrid_kda_mla_f32_model_with(src.path(), forms);
        let named = vec![(ARTIFACT.to_string(), build_inventory(src.path()).unwrap())];
        let container = tempfile::tempdir().unwrap();
        encode_system(&named, container.path())
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    encode(HybridGateForms::KIMI_K3).expect("both gates declared and shipped encode");
    encode(HybridGateForms::KIMI_LINEAR).expect("Kimi Linear's forms encode");
    let refused = encode(HybridGateForms {
        kda_shipped: KdaGateShipped::FullRank,
        ..HybridGateForms::KIMI_LINEAR
    })
    .expect_err("g_proj under no declaration must not be kept");
    assert!(refused.contains("0.self_attn.g_proj.weight"), "{refused}");
    assert!(
        refused.contains("declares no `use_full_rank_gate`"),
        "{refused}"
    );
    let refused = encode(HybridGateForms {
        mla_shipped_gate: true,
        ..HybridGateForms::KIMI_LINEAR
    })
    .expect_err("an MLA g_proj under no declaration must not be kept");
    assert!(refused.contains("3.self_attn.g_proj.weight"), "{refused}");
    assert!(
        refused.contains("declares no `mla_use_output_gate`"),
        "{refused}"
    );
}

/// The two gates are independent declarations: the MLA gate alone, on a
/// KDA estate that keeps its pair, closes and executes — and so does the
/// full-rank KDA gate alone on an ungated MLA layer.
#[test]
fn each_gate_closes_on_its_own() {
    let mla_only = stage(HybridGateForms {
        mla_declared_gate: Some(true),
        mla_shipped_gate: true,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(mla_only.outcome.closed(), "{:?}", mla_only.outcome.defects);
    executes(&mla_only);
    let kda_only = stage(HybridGateForms {
        kda_declared_full_rank: Some(true),
        kda_shipped: KdaGateShipped::FullRank,
        ..HybridGateForms::KIMI_LINEAR
    });
    assert!(kda_only.outcome.closed(), "{:?}", kda_only.outcome.defects);
    executes(&kda_only);
}
