//! The acceptance test for plan-derived roles: every operand of every
//! token-mixer vocabulary is named, and none of them classify `Unknown`.
//!
//! The bar is deliberately not "the model got smaller". A compile can
//! shrink for the wrong reason. What has to hold is that the *report*
//! stops saying `unknown` about tensors the container has always known
//! the role of, while the tensor names and shapes it reports are
//! untouched — classification changed, nothing else did.

use std::collections::BTreeSet;

use super::plan_roles::plan_roles;
use super::policy::{classify_in, Role};
use crate::format::vindex3::fixtures::{encode_fixture_container, hybrid_lllf_f32_model};
use crate::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::tests::mamba2::miniature_mamba2;

/// The pure-SSM fixture, as a plain writer this file can pass around.
fn mamba2_f32_model(dir: &std::path::Path) {
    miniature_mamba2(dir, None);
}

fn roles_for(write: impl FnOnce(&std::path::Path)) -> Vec<(String, Role)> {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), "plan-roles");
    let inspection = inspect_container(container.path(), false).unwrap();
    let mut out: Vec<(String, Role)> = plan_roles(container.path(), &inspection)
        .into_iter()
        .map(|((_, tensor), role)| (tensor, role))
        .collect();
    out.sort();
    out
}

/// Qwen3.8's vocabulary — the one that was silently stranding 16.6% of
/// the hero model's decoder stack. `in_proj_qkv`, `in_proj_a`,
/// `in_proj_b` and `in_proj_z` all classified `Unknown` by name; only
/// `out_proj` matched the substring test.
#[test]
fn gated_delta_operands_are_named_where_the_name_test_saw_nothing() {
    let roles = roles_for(hybrid_lllf_f32_model);
    let by_suffix = |suffix: &str| -> Option<Role> {
        roles
            .iter()
            .find(|(t, _)| t.ends_with(suffix))
            .map(|(_, r)| *r)
    };

    // The bulk matmuls of the recurrence.
    for suffix in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight",
    ] {
        assert_eq!(
            by_suffix(suffix),
            Some(Role::RecurrenceProjection),
            "{suffix}"
        );
    }
    // The control path.
    for suffix in [
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
    ] {
        assert_eq!(by_suffix(suffix), Some(Role::RecurrenceControl), "{suffix}");
    }

    // The regression itself, stated as the comparison it is: the name
    // test still answers `Unknown` for these; the plan does not.
    for suffix in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_z.weight",
    ] {
        let (name, _) = roles.iter().find(|(t, _)| t.ends_with(suffix)).unwrap();
        assert_eq!(
            classify_in(true, "target.decoder_stack", name, &[8, 8]),
            Role::Unknown,
            "{suffix} should still be invisible to the name test — if it is not, this test \
             is no longer measuring the defect it was written for"
        );
    }
}

/// Kimi Linear's vocabulary: KDA against MLA. MLA is not a recurrence —
/// it keeps a per-position cache — so its operands are decoder-linear,
/// and reading them as a recurrence's would be the same class of error
/// one operator over.
#[test]
fn kda_and_mla_operands_are_named_and_not_confused_with_each_other() {
    let roles = roles_for(hybrid_kda_mla_f32_model);
    let of = |suffix: &str| -> Option<Role> {
        roles
            .iter()
            .find(|(t, _)| t.ends_with(suffix))
            .map(|(_, r)| *r)
    };
    // KDA layer 0.
    for suffix in ["0.self_attn.q_proj.weight", "0.self_attn.o_proj.weight"] {
        assert_eq!(of(suffix), Some(Role::RecurrenceProjection), "{suffix}");
    }
    for suffix in ["0.self_attn.f_a_proj.weight", "0.self_attn.b_proj.weight"] {
        assert_eq!(of(suffix), Some(Role::RecurrenceControl), "{suffix}");
    }
    // MLA layer 3 — same `q_proj`/`o_proj` spelling, different operator.
    for suffix in [
        "3.self_attn.q_proj.weight",
        "3.self_attn.kv_a_proj_with_mqa.weight",
        "3.self_attn.kv_b_proj.weight",
        "3.self_attn.o_proj.weight",
    ] {
        assert_eq!(of(suffix), Some(Role::DecoderLinear), "{suffix}");
    }
}

/// Mamba2's vocabulary — a third recurrence family, whose whole block is
/// the mixer and which ships no FFN at all.
#[test]
fn mamba2_operands_are_named() {
    let roles = roles_for(mamba2_f32_model);
    let of = |suffix: &str| -> Option<Role> {
        roles
            .iter()
            .find(|(t, _)| t.ends_with(suffix))
            .map(|(_, r)| *r)
    };
    assert_eq!(of("mixer.in_proj.weight"), Some(Role::RecurrenceProjection));
    assert_eq!(
        of("mixer.out_proj.weight"),
        Some(Role::RecurrenceProjection)
    );
}

/// The invariant that keeps this a classification change: across all
/// three vocabularies, no operand the plan binds is left `Unknown`.
#[test]
fn no_planned_operand_of_any_vocabulary_classifies_unknown() {
    for (write, name) in [
        (
            hybrid_lllf_f32_model as fn(&std::path::Path),
            "gated-deltanet",
        ),
        (hybrid_kda_mla_f32_model as fn(&std::path::Path), "kda/mla"),
        (mamba2_f32_model as fn(&std::path::Path), "mamba2"),
    ] {
        let roles = roles_for(write);
        assert!(!roles.is_empty(), "{name}: the plan bound no operands");
        let unknown: BTreeSet<&str> = roles
            .iter()
            .filter(|(_, r)| *r == Role::Unknown)
            .map(|(t, _)| t.as_str())
            .collect();
        assert!(unknown.is_empty(), "{name}: still unknown: {unknown:?}");
    }
}
