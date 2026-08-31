//! Three near-misses, each perfectly plausible, each caught only here.

use super::*;

/// Qwen3.8's real full-attention Q: 24 heads x 256 x 2 for the fused
/// gate = 12288 rows.
const Q_HEADS: usize = 24;
const HEAD_DIM: usize = 256;

/// **The disagreement that coverage cannot see.** Both modules are
/// self-consistent; only comparing them finds it.
#[test]
fn a_head_dim_disagreement_passes_coverage_and_fails_geometry() {
    // The graph says head_dim 256, so the target expects 12288 rows.
    let expected = vec![2 * Q_HEADS as u64 * HEAD_DIM as u64, 5120];
    // The physical tensor is shaped as though head_dim were 128.
    let planned = vec![2 * Q_HEADS as u64 * 128, 5120];

    let err = reconcile(
        "blk.3.attn_q.weight",
        &planned,
        &expected,
        "q_heads x head_dim x fused_gate_factor",
    )
    .expect_err("the two derivations disagree");

    let msg = err.to_string();
    assert!(
        msg.contains("[6144, 5120]"),
        "shows what was planned: {msg}"
    );
    assert!(
        msg.contains("[12288, 5120]"),
        "shows what was expected: {msg}"
    );
    assert!(
        msg.contains("each is self-consistent"),
        "the refusal must say why this needs its own phase: {msg}"
    );

    // And agreement is silent.
    assert!(reconcile("blk.3.attn_q.weight", &expected, &expected, "…").is_ok());
}

/// **The plausible convolution.** `[10240, 1, 4]` squeezes; `[10240, 2, 4]`
/// must not become something plausible.
#[test]
fn only_a_singleton_conv_axis_may_be_squeezed() {
    assert_eq!(
        squeeze_singleton("blk.0.ssm_conv1d.weight", &[10240, 1, 4], 1).unwrap(),
        vec![10240, 4]
    );

    let err = squeeze_singleton("blk.0.ssm_conv1d.weight", &[10240, 2, 4], 1)
        .expect_err("a real channel axis is not a singleton");
    assert!(
        err.to_string().contains("never collapse real channels"),
        "{err}"
    );

    // The axis matters too — squeezing the wrong one refuses.
    assert!(squeeze_singleton("blk.0.ssm_conv1d.weight", &[10240, 1, 4], 0).is_err());
}

/// **The most dangerous of the three**, because an ordinary-width Q is a
/// perfectly normal tensor. Pinned semantically rather than at Qwen's
/// particular 12288.
#[test]
fn an_unfused_query_width_is_refused_even_though_it_is_a_plausible_tensor() {
    assert!(expect_fused_query_width("blk.3.attn_q.weight", 12288, Q_HEADS, HEAD_DIM).is_ok());

    let err = expect_fused_query_width("blk.3.attn_q.weight", 6144, Q_HEADS, HEAD_DIM)
        .expect_err("ordinary Q width must refuse");
    let msg = err.to_string();
    assert!(msg.contains("fuses the output gate"), "{msg}");
    assert!(
        msg.contains("plausible tensor and the wrong one"),
        "the refusal must say why it is dangerous: {msg}"
    );

    // The rule is semantic, so a different model's geometry still works.
    assert!(expect_fused_query_width("blk.0.attn_q.weight", 2 * 8 * 64, 8, 64).is_ok());
}

/// **The cross-representation invariant.** Encoding and auxiliary
/// tensors may differ between selections; semantic geometry may not.
#[test]
fn both_selections_of_one_model_share_a_semantic_shape_digest() {
    // The same model's target geometry, derived from either selection.
    let bf16: Vec<TargetGeometry> = vec![
        TargetGeometry {
            name: "blk.0.ffn_down.weight".into(),
            dims: vec![5120, 17408],
        },
        TargetGeometry {
            name: "blk.0.attn_qkv.weight".into(),
            dims: vec![10240, 5120],
        },
        TargetGeometry {
            name: "token_embd.weight".into(),
            dims: vec![248320, 5120],
        },
    ];
    let nvfp4 = bf16.clone();

    assert_eq!(
        semantic_digest(bf16.clone()),
        semantic_digest(nvfp4),
        "representation choice must not move the model's geometry"
    );

    // Order must not matter — the two walks may enumerate differently.
    let mut shuffled = bf16.clone();
    shuffled.reverse();
    assert_eq!(semantic_digest(bf16.clone()), semantic_digest(shuffled));

    // And a genuine geometry change does move it, or the digest is
    // useless as a regression signal.
    let mut changed = bf16.clone();
    changed[0].dims = vec![5120, 17407];
    assert_ne!(semantic_digest(bf16), semantic_digest(changed));
}

/// `.scale` siblings are target-ABI auxiliaries. If they entered the
/// digest, the NVFP4 selection could never match the BF16 one and the
/// invariant would be untestable.
#[test]
fn scale_siblings_are_not_model_geometry() {
    let semantic = vec![TargetGeometry {
        name: "blk.0.ffn_down.weight".into(),
        dims: vec![5120, 17408],
    }];
    let mut with_scale = semantic.clone();
    with_scale.push(TargetGeometry {
        name: "blk.0.ffn_down.scale".into(),
        dims: vec![1],
    });
    assert_ne!(
        semantic_digest(semantic),
        semantic_digest(with_scale),
        "the digest is over what it is given — so the caller must give it \
         semantic targets only, which is why scales are excluded upstream"
    );
}
