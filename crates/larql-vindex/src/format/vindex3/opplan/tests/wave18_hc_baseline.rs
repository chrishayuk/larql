//! **Wave 18 — what the role vocabulary says about hyper-connection
//! operands, measured against the recorded baseline.**
//!
//! This file was committed BEFORE the implementation (`feb98a73`) as the
//! instrument, and the diff of this file against that commit is the
//! movement wave 18 made. The baseline it recorded, verbatim:
//!
//! ```text
//! GLM-5.3-Flash      9 unclassified: six hyper-connection operands and
//!                    three FP8 `weight_scale_inv` sidecars
//! DeepSeek-V4-Flash  1 classified: `ffn_norm.weight`, and nothing else
//! Hy4-preview        6 unclassified, spelled a third way
//! ```
//!
//! After wave 18, GLM's six classify to named roles and its three
//! sidecars do not move; DeepSeek's six classify to the SAME roles from a
//! different path prefix and a different dtype, and its base dialect is
//! as foreign as before; Hy4 is untouched. Everything asserted here is
//! the classifier's own answer over each checkpoint's real safetensors
//! headers.
//!
//! # Why the question is asked HERE and not through the pipeline
//!
//! Before wave 18, [`crate::format::vindex3::opplan::build`] returned on
//! the residual-topology refusal before it read a single operand, so a
//! hyper-connection checkpoint produced NO
//! [`ClosureDefect::UnclassifiedOperand`](super::super::ClosureDefect) for
//! its HC tensors — **structural silence, not evidence**. Wave 18 removed
//! that early return (closure now runs), but this probe keeps asking the
//! classifier directly: GLM's surface carries FP8 sidecars and DeepSeek's
//! surface does not build at all, so neither can reach closure on a real
//! header set, and the direct question is the only one both can answer.
//! The pipeline-level carriage is witnessed on a synthetic estate in
//! `wave18_hc_carriage.rs`.
//!
//! # The three subjects are three different states
//!
//! ```text
//! GLM-5.3-Flash      ordinary operands classify, HC operands NOW classify,
//!                    FP8 sidecars still do not
//!                    -> the CONTROLLED subject: one variable moved
//! Kimi-K3            ordinary operands classify; its four `*_res_*`
//!                    operands are NOT hyper-connection operands (a
//!                    `[1, hidden]` projection is no Sinkhorn site under
//!                    any stream count) — see `wave18_hc_carriage.rs`
//!                    -> the TRANSFER witness, answered: nothing to
//!                       transfer to
//! DeepSeek-V4-Flash  HC operands classify; `attn.wq_a`, `attn.wkv`,
//!                    `attn_norm`, `ffn.experts.N.w1` remain a foreign
//!                    dialect, independent of hyper-connections
//!                    -> the DIALECT-BLOCKED control
//! ```
//!
//! Wave 17's arithmetic oracle came from DeepSeek's reference, so the
//! frozen forecast centred DeepSeek here too. That choice was falsified
//! for an ADDRESSABILITY experiment before any code was written: adding
//! HC roles moves DeepSeek from "one operand classifies" to "seven do",
//! attributing nothing about the topology. GLM isolates the variable.
//! DeepSeek staying blocked is evidence in its own right — wave 18 must
//! not appear to unblock a checkpoint whose base dialect it never taught.
//! See `wave18-execution-notes.json`.
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

/// Everything on GLM's layer 0 that the vocabulary does not name after
/// wave 18: the three FP8 block-scale sidecars, and nothing else.
///
/// The baseline pinned NINE — these three plus the six hyper-connection
/// operands — so that a wave-18 rule loose enough to swallow a scale
/// sidecar would fail rather than pass a narrower assertion. The six
/// moved; these did not. They are their own rung, and it has nothing to
/// do with residual topology.
const GLM_UNCLASSIFIED: [&str; 3] = [
    "0.mlp.down_proj.weight_scale_inv",
    "0.mlp.gate_proj.weight_scale_inv",
    "0.mlp.up_proj.weight_scale_inv",
];

/// The six hyper-connection operands and the role each acquired. Named,
/// not counted: a vocabulary that bound all six to one role would satisfy
/// "six classified" and run the wrong site's weights.
const HC_ROLES: [(&str, OperandRole); 6] = [
    ("0.hc_attn_base", OperandRole::HcAttnBase),
    ("0.hc_attn_fn", OperandRole::HcAttnMixFn),
    ("0.hc_attn_scale", OperandRole::HcAttnScale),
    ("0.hc_ffn_base", OperandRole::HcFfnBase),
    ("0.hc_ffn_fn", OperandRole::HcFfnMixFn),
    ("0.hc_ffn_scale", OperandRole::HcFfnScale),
];

/// Every operand DeepSeek-V4-Flash's layer 0 shares with the role
/// vocabulary after wave 18: `ffn_norm` (the one it shared before) and
/// the six hyper-connection operands. Seven of a layer, measured.
///
/// Its whole attention block (`attn.wq_a`, `attn.wq_b`, `attn.wkv`,
/// `attn.wo_a`, `attn_norm`) and its expert spelling
/// (`ffn.experts.N.w1`) remain foreign. That is the measure of how far
/// DeepSeek is from plannable, and it did not change: wave 18 taught
/// this build the topology's operands, not DeepSeek's dialect.
const DEEPSEEK_CLASSIFIED: [&str; 7] = [
    "0.ffn_norm.weight",
    "0.hc_attn_base",
    "0.hc_attn_fn",
    "0.hc_attn_scale",
    "0.hc_ffn_base",
    "0.hc_ffn_fn",
    "0.hc_ffn_scale",
];

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

fn fixture() -> serde_json::Value {
    serde_json::from_str(HEADERS).unwrap()
}

fn names_for(repo: &str) -> Vec<String> {
    fixture()[repo]
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

/// The header-declared dtype of one tensor, by its artifact-global name.
fn dtype_of(repo: &str, name: &str) -> String {
    fixture()[repo][name]["dtype"]
        .as_str()
        .unwrap_or_else(|| panic!("{repo} has no {name}"))
        .to_string()
}

/// **The controlled subject.** GLM's ordinary operands classify, its six
/// hyper-connection operands now classify to six named roles, and its
/// three FP8 scale sidecars are exactly where the baseline left them.
///
/// Every half is asserted, and the first is what makes the rest mean
/// anything: a run reporting "the sidecars are unclassified" would read
/// identically if the classifier were broken, the operator wrong, or the
/// fixture empty. The classified set is the control that rules all three
/// out through the same call, on the same checkpoint, in the same test.
#[test]
fn glm_classifies_its_hyper_connection_six_and_not_its_fp8_sidecars() {
    let (classified, unclassified) = split(GLM, GLM_OPERATOR);

    assert_eq!(
        unclassified, GLM_UNCLASSIFIED,
        "GLM's unclassified set changed — three FP8 scale sidecars, no more and no fewer"
    );
    for (name, role) in HC_ROLES {
        assert!(
            classified.contains(&(name.to_string(), role)),
            "{name} did not classify as {role:?}: {classified:?}"
        );
    }
    assert!(
        classified.len() >= 26,
        "the control is thin — {} operands classified",
        classified.len()
    );
    // Named ordinary roles, not merely a count.
    for expected in [
        OperandRole::FfnDown,
        OperandRole::PreAttentionNorm,
        OperandRole::PostAttentionNorm,
        OperandRole::KdaDtBias,
    ] {
        assert!(
            classified.iter().any(|(_, role)| *role == expected),
            "control operand {expected:?} did not classify: {classified:?}"
        );
    }
}

/// **The dialect-blocked control.** DeepSeek-V4-Flash's hyper-connection
/// operands classify — to the same roles as GLM's — and everything else
/// about its layer stays foreign.
///
/// This is why wave 18 does not take DeepSeek as its subject and why its
/// staying blocked is evidence: `attn.wq_a`, `attn.wkv`, `attn_norm` and
/// `ffn.experts.N.w1` are unaddressed for a reason that has nothing to do
/// with residual topology, so HC roles alone cannot make this checkpoint
/// plannable — and if a wave-18 change ever appears to, that is the
/// programme boundary leaking, not progress.
#[test]
fn deepseek_classifies_the_six_and_stays_blocked_by_its_base_dialect() {
    let (classified, unclassified) = split(DEEPSEEK, LayerOperator::Mla);

    let names: Vec<&str> = classified.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names, DEEPSEEK_CLASSIFIED,
        "DeepSeek's overlap with the role vocabulary changed"
    );
    for (name, role) in HC_ROLES {
        assert!(
            classified.contains(&(name.to_string(), role)),
            "{name} did not classify as {role:?}: {classified:?}"
        );
    }
    // The non-HC half, named, so this cannot be read as an HC finding.
    for foreign in [
        "0.attn.wq_a.weight",
        "0.attn.wkv.weight",
        "0.attn_norm.weight",
        "0.ffn.experts.0.w1.weight",
    ] {
        assert!(
            unclassified.iter().any(|n| n == foreign),
            "{foreign} missing from the fixture — the dialect evidence is gone"
        );
    }
}

/// **The dialect control the forecast asked for.** The same semantic role
/// resolves from a different path prefix and a different stored dtype on
/// the two checkpoints, and the role carries neither: GLM stores the mix
/// projection as BF16 under `model.language_model.layers.N.`, DeepSeek as
/// F32 under bare `layers.N.`. A role that pinned either would have
/// confused the semantic with the physical.
#[test]
fn the_same_role_resolves_from_two_dialects_and_carries_no_dtype() {
    let (glm, _) = split(GLM, GLM_OPERATOR);
    let (deepseek, _) = split(DEEPSEEK, LayerOperator::Mla);
    for (name, role) in HC_ROLES {
        let on = |set: &[(String, OperandRole)]| {
            set.iter()
                .find(|(n, _)| n == name)
                .map(|(_, r)| *r)
                .unwrap_or_else(|| panic!("{name} missing"))
        };
        assert_eq!(on(&glm), role);
        assert_eq!(on(&deepseek), role);
    }
    // The physical facts the role does NOT carry, read from the headers.
    assert_eq!(
        dtype_of(GLM, "model.language_model.layers.0.hc_attn_fn"),
        "BF16"
    );
    assert_eq!(dtype_of(DEEPSEEK, "layers.0.hc_attn_fn"), "F32");
    // And the one they share: the base and scale are F32 on both.
    assert_eq!(
        dtype_of(GLM, "model.language_model.layers.0.hc_attn_base"),
        dtype_of(DEEPSEEK, "layers.0.hc_attn_base")
    );
}

/// Hy4-preview's Sinkhorn-free variant is out of wave 18's scope, and its
/// operands are recorded so that scope is a measured fact rather than an
/// assertion in prose. Its role lives in the module path
/// (`hc_attn_layer.hc_pre.hc_fn`), which is a third spelling again, and
/// wave 18 moved none of it — the frozen forecast's "UNCHANGED" row.
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
/// The baseline recorded this as the one falsifier that could not yet be
/// tested, because nothing could leak before HC roles existed. It is
/// tested now, on these same headers, in
/// `wave18_hc_carriage::deepseeks_head_is_owned_under_the_declaration_and_mtp_stays_external`;
/// this asserts only that the evidence that test relies on is still here.
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
