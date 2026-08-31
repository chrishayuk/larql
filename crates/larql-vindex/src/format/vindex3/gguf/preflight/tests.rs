//! The preflight's job is to be right about refusals. A false "ready"
//! ships split quantisation groups; a false refusal is merely annoying.

use super::*;

fn geom(key_heads: usize, value_heads: usize, value_head_dim: usize) -> VHeadGeometry {
    VHeadGeometry {
        key_heads,
        value_heads,
        value_head_dim,
    }
}

/// Qwen3.8's real geometry, from the hero container.
#[test]
fn qwens_geometry_permutes_whole_groups() {
    let g = geom(16, 48, 128);
    assert_eq!(g.v_per_k(), Some(3), "48 value heads over 16 key heads");
    assert!(
        g.reorder_preserves_groups(16),
        "128 is eight whole 16-element groups, so the reorder moves groups intact"
    );
}

/// **The near miss.** A head dimension that is not a whole number of
/// groups splits them, and the damage is invisible: the weights stay
/// finite, and only a fidelity run would notice.
#[test]
fn a_head_dim_that_is_not_whole_groups_is_refused() {
    for head_dim in [120, 24, 100] {
        let g = geom(16, 48, head_dim);
        assert!(
            !g.reorder_preserves_groups(16),
            "head_dim {head_dim} is not a whole number of groups and must not pass"
        );
    }
    // And the boundary either side of Qwen's own value.
    assert!(
        geom(16, 48, 112).reorder_preserves_groups(16),
        "112 = 7 groups"
    );
    assert!(!geom(16, 48, 127).reorder_preserves_groups(16));
}

/// The constraint binds only where there are groups to split. A BF16
/// export permutes freely.
#[test]
fn the_group_constraint_does_not_apply_to_unquantised_weights() {
    let g = geom(16, 48, 120);
    assert!(
        !g.reorder_preserves_groups(16),
        "would split groups under NVFP4"
    );
    // With no block quantisation there is no group boundary to respect,
    // which the caller expresses by not asking.
    assert!(
        g.v_per_k().is_some(),
        "the head grouping itself is still fine"
    );
}

#[test]
fn value_heads_that_do_not_divide_by_key_heads_have_no_grouping() {
    assert_eq!(
        geom(16, 50, 128).v_per_k(),
        None,
        "50 does not group under 16"
    );
    assert_eq!(geom(0, 48, 128).v_per_k(), None, "no key heads at all");
}

/// A refusal must show its working — the tensor, the operation, the
/// axis, and the arithmetic that failed. "Unsupported shape" would send
/// the reader to the wrong place entirely.
#[test]
fn a_geometry_refusal_names_the_operation_axis_and_arithmetic() {
    let r = Refusal::IncompatibleGeometry {
        operation: "qwen35 V-head reorder",
        axis: "columns (the input axis, which NVFP4 groups run along)",
        invariant: "value_head_dim % nvfp4_group == 0",
        detail: "head dimension 120 is not a whole number of 16-element groups".into(),
    };
    let msg = r.to_string();
    for probe in [
        "V-head reorder",
        "columns",
        "value_head_dim % nvfp4_group",
        "120",
        "re-quantisation",
    ] {
        assert!(
            msg.contains(probe),
            "the refusal must mention {probe}: {msg}"
        );
    }
}

/// A missing semantic points at the artifact, not at the target. The
/// distinction matters: one is a graph that has not finished being
/// honest, the other would be a lowering bug.
#[test]
fn a_missing_semantic_says_which_fact_and_who_needed_it() {
    let r = Refusal::MissingSemantic {
        requirement: "execution.context_length",
        required_by: "qwen35.context_length",
    };
    let msg = r.to_string();
    assert!(msg.contains("execution.context_length"));
    assert!(msg.contains("qwen35.context_length"));
    assert!(
        msg.contains("the artifact missing a fact"),
        "the refusal must place the defect: {msg}"
    );
}

/// The `-exp` on `ssm_a` is arithmetic on weights, and it is only
/// legitimate because an operand declares itself a log decay. Without
/// that role the transform would be "`-exp` because the tensor is called
/// `A_log`", which is the source-family assumption the graph exists to
/// replace.
#[test]
fn a_missing_log_decay_role_refuses_rather_than_assuming_the_tensor_name() {
    let r = Refusal::MissingSemantic {
        requirement: "operand role `log decay`",
        required_by: "qwen35 ssm_a stores -exp(log decay), not the log parameter",
    };
    let msg = r.to_string();
    assert!(msg.contains("log decay"));
    assert!(
        msg.contains("-exp"),
        "the refusal must say what the target would have done with it: {msg}"
    );
}

/// **Detection, as distinct from message quality.**
///
/// The refusal-rendering tests above prove a `Refusal` explains itself.
/// They do not prove the preflight NOTICES. These two do: a surface with
/// the transform fact runs, the same surface without it refuses, and
/// nothing else about the input changes between them.
use super::tests_support::qwen_shaped_surface;

#[test]
fn preflight_detects_a_missing_log_decay_role() {
    let surface = qwen_shaped_surface();

    let with_role = qwen35_preflight(
        &surface,
        true,
        TransformFacts {
            log_decay_role_present: true,
        },
    );
    let without = qwen35_preflight(
        &surface,
        true,
        TransformFacts {
            log_decay_role_present: false,
        },
    );

    assert!(
        without.refusals.iter().any(|r| matches!(
            r,
            Refusal::MissingSemantic { requirement, .. } if *requirement == "operand role `log decay`"
        )),
        "removing the role must produce exactly that refusal"
    );
    assert!(
        !with_role.refusals.iter().any(|r| matches!(
            r,
            Refusal::MissingSemantic { requirement, .. } if *requirement == "operand role `log decay`"
        )),
        "the role being present must not produce it"
    );
    assert_eq!(
        without.refusals.len(),
        with_role.refusals.len() + 1,
        "one fact removed, one refusal added — nothing else moved"
    );
}

/// And the geometry gate is reached through the real entry point, not
/// only through `VHeadGeometry` in isolation.
#[test]
fn preflight_detects_a_head_dim_that_would_split_groups() {
    let mut surface = qwen_shaped_surface();
    if let Some(la) = surface.linear_attention.as_mut() {
        la.value_head_dim = 120;
    }
    let pf = qwen35_preflight(
        &surface,
        true,
        TransformFacts {
            log_decay_role_present: true,
        },
    );
    assert!(
        pf.refusals
            .iter()
            .any(|r| matches!(r, Refusal::IncompatibleGeometry { .. })),
        "120 is not a whole number of 16-element groups and must refuse"
    );
    assert!(!pf.ready(), "a refusing preflight is not ready");

    // The same surface at Qwen's real head dim is ready.
    surface.linear_attention.as_mut().unwrap().value_head_dim = 128;
    let ok = qwen35_preflight(
        &surface,
        true,
        TransformFacts {
            log_decay_role_present: true,
        },
    );
    assert!(ok.ready(), "128 passes every gate: {:?}", ok.refusals);
}
