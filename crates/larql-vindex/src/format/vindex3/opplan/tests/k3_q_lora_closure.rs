//! **K3-MLA-Q-LORA-1 — the factorised query, declared and addressed.**
//!
//! One declared fact, `q_lora_rank`, held to the shipped operands from
//! BOTH sides: a component that declares it requires `q_a_proj`,
//! `q_a_layernorm` and `q_b_proj` and refuses a dense `q_proj`; a
//! component that does not requires `q_proj` and refuses the triple. The
//! synthetic `KKKM` hybrid stages every agreement and every disagreement,
//! so the rule can be watched holding and each refusal made to fire on
//! its own.
//!
//! # Why the declaration has to decide
//!
//! `q_proj` and `q_b_proj` have the SAME row count — `Hq*q_head_dim`,
//! 18432 on the real checkpoint — and differ only in their column count,
//! `hidden` against the rank. A build that picked the form from the
//! estate would be deciding the form from the very thing the form
//! decides, and on a checkpoint shipping both it would decide silently.
//! So identity is declared, and what is shipped is judged against it.
//!
//! The plan stage says nothing about any of this: `q_lora_rank` is a
//! representable config leaf in every arm below, which
//! `identity::the_semantics_version_is_pinned_to_known_verdicts` pins
//! from the other side. Closure is where the operands meet the
//! declaration.
//!
//! Nothing here claims Kimi-K3 executes: its routed FFN still carries an
//! unrepresented latent-MoE wrapper and an unjudged MXFP4 declaration.

use larql_models::inventory::build_inventory;

use crate::format::vindex3::encode::encode_graph;
use crate::format::vindex3::fixtures_kimi::{
    hybrid_kda_mla_f32_model_with_query, HybridQueryForms, MlaQueryShipped, MLA_Q_LORA_RANK,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, LayerAttention, MlaQueryProjection, OpPlanOutcome,
};
use crate::format::vindex3::plan::plan_system;

const ARTIFACT: &str = "kkkm-query";

struct Staged {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    outcome: OpPlanOutcome,
}

/// Encode through the graph seam, as `k3_rep_gate_closure` does and for
/// the same reason: the production writer refuses to KEEP an estate whose
/// operands do not close, and most arms here exist to see what closure
/// says BELOW that refusal.
fn stage(query: HybridQueryForms) -> Staged {
    let src = tempfile::tempdir().unwrap();
    hybrid_kda_mla_f32_model_with_query(src.path(), query);
    let named = vec![(ARTIFACT.to_string(), build_inventory(src.path()).unwrap())];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    Staged {
        _src: src,
        _container: container,
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

fn missing_roles(outcome: &OpPlanOutcome) -> Vec<String> {
    outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::MissingOperand { role, .. } => Some(format!("{role:?}")),
            _ => None,
        })
        .collect()
}

/// **The positive arm.** Declared and shipped: the MLA layer closes, and
/// its op carries the factorisation as a form.
///
/// First, because every refusal below is vacuous if the agreeing estate
/// does not close.
#[test]
fn a_declared_rank_with_the_triple_shipped_closes() {
    let staged = stage(HybridQueryForms::KIMI_K3);
    assert!(
        staged.outcome.defects.is_empty(),
        "the agreeing estate must close: {:?}",
        staged.outcome.defects
    );

    let mla = staged
        .outcome
        .plan
        .as_ref()
        .expect("the op plan")
        .layers
        .iter()
        .find_map(|l| match &l.attention {
            LayerAttention::Mla(op) => Some(op),
            _ => None,
        })
        .expect("the hybrid's one MLA layer");

    match &mla.query {
        MlaQueryProjection::LowRank {
            q_a_proj,
            q_a_norm,
            q_b_proj,
            q_a_norm_eps,
        } => {
            assert!(q_a_proj.tensor.ends_with("q_a_proj.weight"));
            assert!(q_a_norm.tensor.ends_with("q_a_layernorm.weight"));
            assert!(q_b_proj.tensor.ends_with("q_b_proj.weight"));
            // The COLUMN count is the discriminator: `q_b_proj` has the
            // same rows as the `q_proj` it replaces.
            assert_eq!(
                q_b_proj.shape[1], MLA_Q_LORA_RANK,
                "q_b_proj's columns are the rank, not hidden"
            );
            assert_eq!(q_a_proj.shape[0], MLA_Q_LORA_RANK);
            assert_eq!(
                *q_a_norm_eps,
                Some(1e-6),
                "the q-A epsilon is the family's own class default, carried with the form"
            );
        }
        other => panic!("expected the factorised form, got {other:?}"),
    }
}

/// **The other positive arm.** No rank declared, one dense `q_proj`
/// shipped: unchanged, and still the direct form.
///
/// This is what keeps the rule from having been implemented as "always
/// require the triple".
#[test]
fn no_declared_rank_with_a_dense_q_proj_closes_unchanged() {
    let staged = stage(HybridQueryForms::KIMI_LINEAR);
    assert!(
        staged.outcome.defects.is_empty(),
        "Kimi Linear's own shape must still close: {:?}",
        staged.outcome.defects
    );
    let mla = staged
        .outcome
        .plan
        .as_ref()
        .expect("the op plan")
        .layers
        .iter()
        .find_map(|l| match &l.attention {
            LayerAttention::Mla(op) => Some(op),
            _ => None,
        })
        .expect("the hybrid's one MLA layer");
    match &mla.query {
        MlaQueryProjection::Direct { q_proj } => {
            assert!(q_proj.tensor.ends_with("q_proj.weight"));
        }
        other => panic!("expected the direct form, got {other:?}"),
    }
}

/// **The disagreement, declaration-side.** A declared rank with a dense
/// `q_proj` shipped: the reference's `__init__` is an if/else and never
/// builds this.
///
/// Both halves are named — the `q_proj` implies an op the component did
/// not choose, and each absent member of the triple is reported by role —
/// so a reader is told what was shipped AND what was expected.
#[test]
fn a_declared_rank_shipping_a_dense_q_proj_is_refused_from_both_sides() {
    let staged = stage(HybridQueryForms::DECLARED_BUT_DENSE);

    let implied = implied_absent(&staged.outcome);
    assert!(
        implied
            .iter()
            .any(|(tensor, primitive)| tensor.ends_with("q_proj.weight")
                && primitive.contains("q_lora_rank")),
        "the shipped q_proj must be named against the declaration: {implied:?}"
    );

    let missing = missing_roles(&staged.outcome);
    for role in ["MlaQAProj", "MlaQANorm", "MlaQBProj"] {
        assert!(
            missing.iter().any(|m| m == role),
            "the declared form's absent {role} must be named: {missing:?}"
        );
    }
}

/// **The disagreement, estate-side.** The triple shipped with no rank
/// declared: identity is not acquired by shipping tensors spelled for it.
#[test]
fn the_triple_shipped_without_a_declaration_is_refused() {
    let staged = stage(HybridQueryForms::UNDECLARED_BUT_FACTORISED);

    let implied = implied_absent(&staged.outcome);
    for spelling in ["q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight"] {
        assert!(
            implied
                .iter()
                .any(|(tensor, primitive)| tensor.ends_with(spelling)
                    && primitive.contains("q_lora_rank")),
            "{spelling} must be named against the missing declaration: {implied:?}"
        );
    }
    assert!(
        missing_roles(&staged.outcome)
            .iter()
            .any(|m| m == "MlaQProj"),
        "and the dense projection the component DID declare must be reported absent"
    );
}

/// Each member of the triple, absent on its own, is named on its own.
///
/// Three arms rather than one: a closure that reported "the query is
/// incomplete" would pass a single check and tell a reader nothing about
/// which operand to go and find.
#[test]
fn each_absent_member_of_the_triple_is_named_individually() {
    // The fixture ships the triple as a unit, so this arm reaches the
    // same rule through the declaration-side disagreement, which is the
    // only way to make members absent without a fourth fixture shape.
    let staged = stage(HybridQueryForms::DECLARED_BUT_DENSE);
    let missing = missing_roles(&staged.outcome);
    assert_eq!(
        missing
            .iter()
            .filter(|m| m.starts_with("MlaQ") && *m != "MlaQProj")
            .count(),
        3,
        "all three members named separately, not as one defect: {missing:?}"
    );
}

/// The fixture's own discriminator: `q_b_proj` and the `q_proj` it
/// replaces really do share a row count here, so the column check above
/// is load-bearing rather than incidental.
#[test]
fn the_two_query_projections_share_a_row_count_in_this_fixture() {
    let factorised = stage(HybridQueryForms::KIMI_K3);
    let dense = stage(HybridQueryForms::KIMI_LINEAR);
    let rows = |staged: &Staged| -> usize {
        staged
            .outcome
            .plan
            .as_ref()
            .expect("the op plan")
            .layers
            .iter()
            .find_map(|l| match &l.attention {
                LayerAttention::Mla(op) => Some(op.query.operands()),
                _ => None,
            })
            .expect("an MLA layer")
            .iter()
            .find(|(name, _)| name.ends_with("q_b_proj") || name.ends_with("q_proj"))
            .expect("a query projection")
            .1
            .shape[0]
    };
    assert_eq!(
        rows(&factorised),
        rows(&dense),
        "if these differed, a row-count check would separate the forms and the \
         declaration would not be load-bearing"
    );
    let hidden = factorised
        .outcome
        .plan
        .as_ref()
        .expect("the op plan")
        .layers
        .iter()
        .find_map(|l| match &l.attention {
            LayerAttention::Mla(op) => Some(op.query.operands()[0].1.shape[1]),
            _ => None,
        })
        .expect("an MLA layer");
    assert_ne!(
        MLA_Q_LORA_RANK, hidden,
        "the rank must differ from hidden, or the columns would not separate the forms"
    );
    assert_eq!(
        HybridQueryForms::KIMI_K3.shipped,
        MlaQueryShipped::Factorised
    );
}
