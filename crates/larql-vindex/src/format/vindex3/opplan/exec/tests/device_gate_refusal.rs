//! The device paths refuse Kimi-K3's declared output gates BY NAME
//! (K3-REP-GATE-1, freeze D6 / P8), and the decision is a pure function
//! of declared facts — witnessed here without a GPU.
//!
//! What the Metal loader binds today: the low-rank pair by name for KDA
//! (`f32s[5]`/`[6]`) and no gate at all for MLA. A full-rank KDA layer
//! would otherwise fail on a missing tensor name; a gated MLA layer would
//! otherwise run UNGATED with every shape still closing — the silent
//! failure this refusal exists to make impossible.
use crate::format::vindex3::opplan::exec::device::declared_gate_refusal;

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
