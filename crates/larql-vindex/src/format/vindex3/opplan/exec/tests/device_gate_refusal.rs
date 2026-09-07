//! The device paths refuse Kimi-K3's declared output gates BY NAME
//! (K3-REP-GATE-1, freeze D6 / P8), and the decision is a pure function
//! of declared facts — witnessed here without a GPU.
//!
//! What the Metal loader binds today: the low-rank pair by name for KDA
//! (`f32s[5]`/`[6]`) and no gate at all for MLA. A full-rank KDA layer
//! would otherwise fail on a missing tensor name; a gated MLA layer would
//! otherwise run UNGATED with every shape still closing — the silent
//! failure this refusal exists to make impossible.
//! The same module also witnesses the LOWERED routed FFN's refusal of a
//! latent routed branch (K3-LATENTMOE-1, freeze D12) — the one device
//! path that cannot express the wrapper, for the same reason and by the
//! same means: a pure decision from declared facts, named before any
//! tensor is bound.
use crate::format::vindex3::opplan::exec::device_refusal::{
    declared_gate_refusal, fused_query_gate_projection_refusal, lowered_latent_branch_refusal,
};
use crate::format::vindex3::opplan::{LatentBranchOp, LatentNormOp, OperandRef};

#[test]
fn a_full_rank_kda_layer_is_refused_by_the_forms_name() {
    let refusal = declared_gate_refusal(4, false, true, false, None).expect("refused");
    assert!(refusal.starts_with("layer 4:"), "{refusal}");
    assert!(refusal.contains("`use_full_rank_gate`"), "{refusal}");
    assert!(refusal.contains("low-rank g_a_proj/g_b_proj"), "{refusal}");
}

#[test]
fn a_gated_mla_layer_is_refused_by_the_gates_name() {
    let refusal = declared_gate_refusal(7, true, false, true, None).expect("refused");
    assert!(refusal.starts_with("layer 7:"), "{refusal}");
    assert!(refusal.contains("`mla_use_output_gate`"), "{refusal}");
    assert!(refusal.contains("ungated"), "{refusal}");
}

/// Each declaration refuses only the layers of ITS operator: a full-rank
/// KDA declaration says nothing about MLA layers, and an MLA gate says
/// nothing about KDA layers — the refusal is per layer, by operator.
#[test]
fn each_declaration_refuses_only_its_own_operators_layers() {
    assert!(declared_gate_refusal(0, true, true, false, None).is_none());
    assert!(declared_gate_refusal(0, false, false, true, None).is_none());
}

#[test]
fn kimi_linears_own_forms_are_not_refused() {
    for mla in [false, true] {
        assert!(declared_gate_refusal(0, mla, false, false, None).is_none());
    }
}

/// **K3-MLA-Q-LORA-1.** A factorised MLA query is refused by the form's
/// name and by its rank, before any tensor is bound.
///
/// The rank is in the message because the refusal has to be actionable:
/// "this container declares q_lora_rank" tells a reader what to look for
/// in the config, where "the MLA path failed" would send them looking for
/// a `q_proj` the checkpoint never shipped.
#[test]
fn a_factorised_mla_query_is_refused_by_the_forms_name() {
    let refusal = declared_gate_refusal(9, true, false, false, Some(1536)).expect("refused");
    assert!(refusal.starts_with("layer 9:"), "{refusal}");
    assert!(refusal.contains("`q_lora_rank: 1536`"), "{refusal}");
    assert!(
        refusal.contains("q_b_proj into the q_proj slot"),
        "{refusal}"
    );
}

/// The declaration refuses only MLA layers: a KDA layer has no query
/// factorisation to bind wrongly.
#[test]
fn a_declared_rank_does_not_refuse_kda_layers() {
    assert!(declared_gate_refusal(0, false, false, false, Some(1536)).is_none());
}

/// And the direct form is not refused, on either operator — the arm that
/// keeps the rule from having been implemented as "refuse every MLA
/// layer".
#[test]
fn an_undeclared_rank_is_not_refused() {
    for mla in [false, true] {
        assert!(declared_gate_refusal(0, mla, false, false, None).is_none());
    }
}

// ── The lowered routed FFN and the latent branch (K3-LATENTMOE-1) ────

/// An operand reference standing for a wrapper tensor. Only its presence
/// matters to the refusal, which is decided before anything is bound.
fn operand(tensor: &str) -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: tensor.to_string(),
        dtype: "BF16".to_string(),
        shape: vec![1, 1],
    }
}

/// An operand spelled NOTHING like K3's, so an assertion that passes on
/// it is reading the op rather than a name written into the refusal.
const FOREIGN_DOWN: &str = "17.moe.bottleneck_in.weight";

fn latent(width: usize, norm: bool) -> LatentBranchOp {
    LatentBranchOp {
        width,
        down: operand("routed_expert_down_proj.weight"),
        up: operand("routed_expert_up_proj.weight"),
        norm: norm.then(|| LatentNormOp {
            weight: operand("routed_expert_norm.weight"),
            eps: 1e-5,
        }),
    }
}

/// The refusal names the declaration, the width, and the program spelled
/// from the op's OWN operands. K3's own 3584 is used so the message a
/// reader would actually see is the one asserted.
#[test]
fn a_latent_routed_branch_is_refused_by_the_declarations_name() {
    let refusal = lowered_latent_branch_refusal(11, Some(&latent(3584, true))).expect("refused");
    assert!(refusal.starts_with("layer 11:"), "{refusal}");
    assert!(
        refusal.contains("`routed_expert_hidden_size: 3584`"),
        "{refusal}"
    );
    assert!(refusal.contains("routed_expert_down_proj"), "{refusal}");
    assert!(refusal.contains("routed_expert_up_proj"), "{refusal}");
    assert!(refusal.contains("routed_expert_norm"), "{refusal}");
}

/// A branch WITHOUT the norm is still refused, and the message says the
/// wrapper carries none — the refusal is about the bottleneck, not about
/// the norm inside it.
#[test]
fn a_latent_branch_without_a_norm_is_refused_too() {
    let refusal = lowered_latent_branch_refusal(0, Some(&latent(1024, false))).expect("refused");
    assert!(
        refusal.contains("`routed_expert_hidden_size: 1024`"),
        "{refusal}"
    );
    assert!(refusal.contains("no norm"), "{refusal}");
    assert!(!refusal.contains("routed_expert_norm"), "{refusal}");
}

/// The program in the message comes from the OP's operands, not from
/// spellings written into the refusal. A container spelling its
/// bottleneck some other way must be named by ITS name — otherwise the
/// message sends a reader looking for a tensor their checkpoint does not
/// contain.
#[test]
fn the_refusal_spells_the_program_from_the_operands_it_was_given() {
    let mut op = latent(512, false);
    op.down = operand(FOREIGN_DOWN);
    let refusal = lowered_latent_branch_refusal(2, Some(&op)).expect("refused");
    assert!(refusal.contains(FOREIGN_DOWN), "{refusal}");
    assert!(
        !refusal.contains("routed_expert_down_proj"),
        "the refusal restated a spelling instead of reading the operand: {refusal}"
    );
}

/// And an ordinary routed layer is NOT refused — the arm that keeps this
/// from having been implemented as "refuse every routed FFN", which would
/// have taken every MoE model off the lowered path at once.
#[test]
fn an_undeclared_branch_leaves_the_lowered_routed_path_open() {
    assert!(lowered_latent_branch_refusal(0, None).is_none());
    assert!(lowered_latent_branch_refusal(93, None).is_none());
}

/// The fused query/gate refusal is ONE sentence, read by the decode site
/// and the batch site alike. Asserted here rather than at each site
/// because that is the property: two copies of a message drift, and a
/// reader meeting the second one would not know it was the same refusal.
#[test]
fn the_fused_query_gate_refusal_names_the_projection_and_the_absent_kernel() {
    let refusal = fused_query_gate_projection_refusal();
    assert!(refusal.contains("fused query/gate projection"), "{refusal}");
    assert!(refusal.contains("no device kernel"), "{refusal}");
}
