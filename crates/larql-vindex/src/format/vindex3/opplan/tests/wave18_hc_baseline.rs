//! **Wave 18 baseline — what the role vocabulary says about hyper-connection
//! operands BEFORE wave 18 changes it.**
//!
//! No capability is added here. This is the instrument, committed first so
//! that when wave 18 moves something, the movement is measured against a
//! recorded state rather than a remembered one.
//!
//! # Why the question is asked HERE and not through the pipeline
//!
//! [`crate::format::vindex3::opplan::build`] returns on the
//! residual-topology refusal before it reads a single operand — its own
//! comment says so. So a hyper-connection checkpoint produces NO
//! [`ClosureDefect::UnclassifiedOperand`](super::super::ClosureDefect) for
//! its HC tensors, and that absence is **structural silence, not
//! evidence**: nothing asked. Asking the classifier directly is the only
//! way to observe the vocabulary's actual state, and it is the same move
//! `k3_representable.rs` makes one stage down for the same reason.
//!
//! # The three subjects are three different states
//!
//! ```text
//! GLM-5.3-Flash      ordinary operands classify, HC operands do not
//!                    -> the CONTROLLED subject: one variable
//! Kimi-K3            ordinary operands classify, four generic HC gaps
//!                    -> the TRANSFER witness, pinned in
//!                       `plan::tests::k3_representable`
//! DeepSeek-V4-Flash  NOTHING classifies — `attn.wq_a`, `attn.wkv`,
//!                    `attn_norm`, `ffn.experts.N.w1` are a foreign
//!                    dialect, independent of hyper-connections
//!                    -> the DIALECT-BLOCKED control
//! ```
//!
//! Wave 17's arithmetic oracle came from DeepSeek's reference, so the
//! frozen forecast centred DeepSeek here too. That choice is falsified for
//! an ADDRESSABILITY experiment: adding HC roles to DeepSeek would move it
//! from "all unclassified" to "nearly all unclassified", attributing
//! nothing. GLM isolates the variable; K3 tests whether it transfers.
//! DeepSeek staying blocked is then evidence in its own right — wave 18
//! must not appear to unblock a checkpoint whose base dialect it never
//! taught. See `wave18-execution-notes.json`.
//!
//! # The fixture
//!
//! Real safetensors headers, fetched over HTTP range requests by
//! `scripts/hc_header_fixture_export.py`, payload never touched. Names,
//! dtypes and shapes are what each checkpoint writes. Indexed families are
//! capped at two real members because 896 experts exercise one classifier
//! path 896 times.
use crate::format::vindex3::graph::roles::classify_stack_tensor_on;
use crate::format::vindex3::graph::{LayerOperator, OperandRole};

const HEADERS: &str = include_str!("fixtures/hc_operand_headers.json");

const GLM: &str = "zai-org/GLM-5.3-Flash";
const DEEPSEEK: &str = "deepseek-ai/DeepSeek-V4-Flash";
const HY4: &str = "tencent/Hy4-preview";

/// GLM-5.3-Flash's layer 0 is a Kimi-Delta layer — `A_log`, `dt_bias`,
/// `f_a_proj`, three `*_conv1d` — not softmax, and
/// [`LayerOperator::Kda`]'s own documentation names GLM-5.3-Flash's 34 as
/// observed. Classifying it under `Softmax` would ask a question the graph
/// never asks and would report a vocabulary gap that does not exist.
const GLM_OPERATOR: LayerOperator = LayerOperator::Kda;

/// Everything on GLM's layer 0 that the vocabulary does not name today.
///
/// NINE, not six, and the extra three matter. GLM-5.3-Flash is FP8, so
/// each `mlp.*_proj` carries a `weight_scale_inv` block-scale sidecar, and
/// those are unaddressed for a reason that has nothing to do with residual
/// topology. Pinning all nine is what makes this a controlled experiment
/// rather than a targeted one: wave 18 must move the six and leave the
/// three exactly where they are. A rule loose enough to swallow a scale
/// sidecar would pass a six-element assertion and fail this one.
const GLM_UNCLASSIFIED: [&str; 9] = [
    // wave 18's target
    "0.hc_attn_base",
    "0.hc_attn_fn",
    "0.hc_attn_scale",
    "0.hc_ffn_base",
    "0.hc_ffn_fn",
    "0.hc_ffn_scale",
    // NOT wave 18's — FP8 block scales, their own rung
    "0.mlp.down_proj.weight_scale_inv",
    "0.mlp.gate_proj.weight_scale_inv",
    "0.mlp.up_proj.weight_scale_inv",
];

/// The one operand DeepSeek-V4-Flash shares with the role vocabulary.
///
/// Exactly one, measured: `ffn_norm` reads as the post-attention norm. Its
/// whole attention block (`attn.wq_a`, `attn.wq_b`, `attn.wkv`,
/// `attn.wo_a`, `attn_norm`), its expert spelling (`ffn.experts.N.w1`) and
/// its hyper-connection operands are all foreign. One overlapping name out
/// of a layer is the measure of how far DeepSeek is from plannable, and it
/// is a sharper statement than "nothing classifies" — which is what this
/// test asserted before the fixture was consulted.
const DEEPSEEK_CLASSIFIED: [&str; 1] = ["0.ffn_norm.weight"];

/// Object-relative name: the classifier is asked about `0.<rest>`, never
/// the artifact-global spelling, because the leading layer index is what
/// it parses the layer number out of.
fn layer_relative(name: &str) -> Option<String> {
    let (_, rest) = name.split_once(".layers.").or_else(|| {
        name.starts_with("layers.")
            .then(|| name.split_once("layers.").unwrap())
    })?;
    Some(rest.to_string())
}

fn names_for(repo: &str) -> Vec<String> {
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    fixture[repo]
        .as_object()
        .unwrap_or_else(|| panic!("fixture carries no {repo}"))
        .keys()
        .cloned()
        .collect()
}

fn split(repo: &str, operator: LayerOperator) -> (Vec<(String, OperandRole)>, Vec<String>) {
    let mut classified = Vec::new();
    let mut unclassified = Vec::new();
    for name in names_for(repo) {
        let Some(relative) = layer_relative(&name) else {
            continue;
        };
        match classify_stack_tensor_on(&relative, operator) {
            Some((_, role)) => classified.push((relative, role)),
            None => unclassified.push(relative),
        }
    }
    classified.sort();
    unclassified.sort();
    (classified, unclassified)
}

/// **The controlled subject.** GLM's ordinary operands classify; its six
/// hyper-connection operands do not, alongside three FP8 scale sidecars
/// that are pinned so wave 18 can be seen NOT to move them.
///
/// Both halves are asserted, and the first is what makes the second mean
/// anything: a run reporting "the HC operands are unclassified" would read
/// identically if the classifier were broken, the operator wrong, or the
/// fixture empty. The classified set is the control that rules all three
/// out through the same call, on the same checkpoint, in the same test.
#[test]
fn glm_classifies_its_ordinary_operands_and_not_its_hyper_connection_six() {
    let (classified, unclassified) = split(GLM, GLM_OPERATOR);

    assert_eq!(
        unclassified, GLM_UNCLASSIFIED,
        "GLM's unclassified set changed — six hyper-connection operands \
         and three FP8 scale sidecars, no more and no fewer"
    );
    assert!(
        classified.len() >= 20,
        "the control is thin — {} operands classified",
        classified.len()
    );
    // Named roles, not merely a count: a vocabulary that returned one
    // arbitrary role for everything would satisfy a count.
    for expected in [
        OperandRole::FfnDown,
        OperandRole::PreAttentionNorm,
        OperandRole::PostAttentionNorm,
    ] {
        assert!(
            classified.iter().any(|(_, role)| *role == expected),
            "control operand {expected:?} did not classify: {classified:?}"
        );
    }
}

/// **The dialect-blocked control.** DeepSeek-V4-Flash classifies NOTHING,
/// and its hyper-connection operands are the smaller half of that.
///
/// This is why wave 18 does not take DeepSeek as its subject. `attn.wq_a`,
/// `attn.wkv`, `attn_norm` and `ffn.experts.N.w1` are unaddressed for a
/// reason that has nothing to do with residual topology, so HC roles alone
/// could not make this checkpoint plannable — and if a wave-18 change ever
/// appears to, that is the programme boundary leaking, not progress.
#[test]
fn deepseek_is_blocked_by_its_base_dialect_and_not_only_by_hyper_connections() {
    let (classified, unclassified) = split(DEEPSEEK, LayerOperator::Mla);

    let names: Vec<&str> = classified.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names, DEEPSEEK_CLASSIFIED,
        "DeepSeek's overlap with the role vocabulary changed"
    );
    // The non-HC half, named, so this cannot be read as an HC finding.
    for foreign in [
        "0.attn.wq_a.weight",
        "0.attn.wkv.weight",
        "0.attn_norm.weight",
    ] {
        assert!(
            unclassified.iter().any(|n| n == foreign),
            "{foreign} missing from the fixture — the dialect evidence is gone"
        );
    }
}

/// Hy4-preview's Sinkhorn-free variant is out of wave 18's scope, and its
/// operands are recorded so that scope is a measured fact rather than an
/// assertion in prose. Its role lives in the module path
/// (`hc_attn_layer.hc_pre.hc_fn`), which is a third spelling again.
#[test]
fn hy4s_prepost_variant_is_unaddressed_and_spelled_differently_from_both() {
    let (_, unclassified) = split(HY4, LayerOperator::Mla);
    assert_eq!(
        unclassified,
        [
            "0.hc_attn_layer.hc_pre.hc_base",
            "0.hc_attn_layer.hc_pre.hc_fn",
            "0.hc_attn_layer.hc_pre.hc_scale",
            "0.hc_mlp_layer.hc_pre.hc_base",
            "0.hc_mlp_layer.hc_pre.hc_fn",
            "0.hc_mlp_layer.hc_pre.hc_scale",
        ]
    );
}

/// The fixture carries `mtp.*` deliberately: wave 18 must not cause an
/// external sub-model's tensors to acquire a primary-stack fate.
///
/// The exclusion itself is already pinned one stage up, in
/// `graph::tests::build`, where the whole `mtp.*` family surfaces in
/// `unplaced` — this asserts only that the evidence is present here to
/// test against once HC roles exist. The leak test belongs with the
/// implementation, because before it there is nothing that could leak.
#[test]
fn the_fixture_carries_the_external_mtp_namespace_to_test_against() {
    let mtp: Vec<String> = names_for(DEEPSEEK)
        .into_iter()
        .filter(|n| n.starts_with("mtp."))
        .collect();
    assert!(
        mtp.len() >= 20,
        "too little mtp evidence to test a leak against: {mtp:?}"
    );
    assert!(
        mtp.iter().any(|n| n.contains("hc_")),
        "the mtp namespace must carry its own hyper-connection operands — \
         they are the ones that could be captured by a wave-18 rule"
    );
}
