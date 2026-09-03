//! Hybrid linear-attention interleaves (Qwen3.5/Kimi-Linear-style):
//! `layer_types` declaring a span kind outside VINDEX3's executable
//! vocabulary must block honestly — not fabricate a collapsed "all full"
//! resolution — and must not take unrelated per-layer facts (rope theta)
//! down with it.

use super::support::{
    declare_gated_delta_geometry, declare_hybrid_cadence, glimmer_shaped_target_with,
    FIXTURE_LAYERS,
};
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding, SemanticClass};

/// The Glimmer-shaped fixture with its `layer_types` swapped for a
/// Qwen3.5-style hybrid interleave — three `linear_attention` layers to
/// one `full_attention` layer — plus the declared-but-unexecuted hybrid
/// linear-attention / MTP / mRoPE fields a real Qwen3.5 `config.json`
/// carries alongside it.
fn hybrid_findings() -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        declare_hybrid_cadence(config);
        declare_gated_delta_geometry(config);
        config["text_config"]["attn_output_gate"] = serde_json::json!(true);
        config["text_config"]["output_gate_type"] = serde_json::json!("swish");
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        config["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(false);
        // The fixture head is 8 wide, so a fraction of 0.5 gives a
        // 4-dim rotary block and 2 frequency slots: `sum(section) * 2 ==
        // rotary_dim`. Small, but it is the same identity the real
        // checkpoint closes with 11+11+10 over a 64-dim block, and
        // `a_section_that_does_not_close_the_arithmetic_blocks` proves
        // the gate can still refuse at this size.
        config["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
        config["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.5);
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(true);
        config["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([1, 1, 0]);
    });
    let named = vec![("target-artifact".to_string(), inventory)];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

fn finding_for<'a>(findings: &'a [PlannedFinding], suffix: &str) -> &'a PlannedFinding {
    findings
        .iter()
        .find(|f| f.subject.ends_with(suffix))
        .unwrap_or_else(|| panic!("no finding for `{suffix}`"))
}

/// QW-3.5A: `layer_types` is now authoritative graph truth, so a declared
/// `linear_attention` interleave is **carried**, not refused.
///
/// This test was the honesty gate for the previous rung, where both
/// `text_config.layer_types` findings had to block because the graph's
/// only vocabulary was a sliding/full span and a recurrence had no home
/// in it. The home now exists — [`LayerOperator::GatedDelta`] — so the
/// same two findings must go representable. What is deliberately kept is
/// every assertion that made the old test able to fail: the resolution
/// must still not be a fabricated all-full array, and the full declared
/// interleave must still survive into the report.
///
/// The negative control lives in
/// [`an_unrecognised_spelling_still_blocks_on_both_findings`] — without
/// it, "both findings are representable" would also be satisfied by a
/// build that simply stopped checking.
#[test]
fn a_declared_linear_attention_interleave_is_carried_on_both_findings() {
    let findings = hybrid_findings();
    let layer_types_findings: Vec<&PlannedFinding> = findings
        .iter()
        .filter(|f| f.subject == "text_config.layer_types")
        .collect();
    assert_eq!(
        layer_types_findings.len(),
        2,
        "expected the comparator finding and the carriage finding"
    );

    let fabricated_all_full = serde_json::json!(vec!["full_attention"; FIXTURE_LAYERS]);
    for finding in &layer_types_findings {
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{finding:?}"
        );
        assert!(!finding.blocks(), "{finding:?}");
        assert_ne!(
            finding.resolved,
            Some(fabricated_all_full.clone()),
            "carried, but never as a collapsed all-full resolution: {finding:?}"
        );
    }

    // Unchanged from the previous rung: the full declared interleave
    // survives to the report whatever the graph resolved.
    let declared_array: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    assert!(
        layer_types_findings
            .iter()
            .any(|f| f.declared == Some(serde_json::json!(declared_array))),
        "the full declared interleave must survive to the report"
    );
}

/// **Negative control for the rung.** A spelling this schema still has no
/// operator for must block on BOTH findings, exactly as
/// `linear_attention` used to.
///
/// This is what stops QW-3.5A from having been implemented as "stop
/// grading `layer_types`". The two findings that just went representable
/// for a recurrence must still be able to refuse, and the only difference
/// between the two fixtures is the spelling.
#[test]
fn an_unrecognised_spelling_still_blocks_on_both_findings() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    // Not in any vocabulary this build knows.
                    "hyena_attention"
                }
            })
            .collect();
        config["text_config"]["layer_types"] = serde_json::json!(layer_types);
    });
    let findings: Vec<PlannedFinding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();
    let layer_types_findings: Vec<&PlannedFinding> = findings
        .iter()
        .filter(|f| f.subject == "text_config.layer_types")
        .collect();
    assert_eq!(layer_types_findings.len(), 2);
    for finding in &layer_types_findings {
        assert!(
            finding.blocks(),
            "an unjudged spelling must still block: {finding:?}"
        );
        assert_ne!(
            finding.category,
            FindingCategory::Representable,
            "{finding:?}"
        );
    }
}

/// The graph records the cadence the checkpoint declares, layer by layer,
/// and the census is exact: **48 recurrent / 16 softmax on an LLLF
/// cadence** for the real Qwen3.8 shape.
///
/// Counting alone would pass on any 48/16 arrangement, so position is
/// asserted too — a shuffled table with the same totals fails here.
#[test]
fn the_declared_cadence_is_carried_into_the_graph_layer_by_layer() {
    use crate::format::vindex3::graph::policy::AttentionSpan;
    use crate::format::vindex3::graph::{build_from_inventories, LayerOperator};

    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        declare_hybrid_cadence(config);
        // Required for the layers to grade `GatedDelta` at all — without
        // it they are an *unidentified* recurrence, which is what
        // `an_unidentified_recurrence_is_never_graded_gated_delta` pins.
        declare_gated_delta_geometry(config);
    });
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let table = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("the text component carries a per-layer table");

    assert_eq!(table.len(), FIXTURE_LAYERS);
    for (i, layer) in table.iter().enumerate() {
        if i % 4 == 3 {
            assert_eq!(layer.operator, LayerOperator::Softmax, "layer {i}");
            assert_eq!(
                layer.span,
                Some(AttentionSpan::Full),
                "layer {i} is a softmax layer and must state its span"
            );
        } else {
            assert_eq!(layer.operator, LayerOperator::GatedDelta, "layer {i}");
            // The repair itself: a recurrence has no span, rather than a
            // `Full` a KV planner would read as liveness.
            assert_eq!(layer.span, None, "layer {i} must carry no span");
        }
    }
    let recurrent = table
        .iter()
        .filter(|l| l.operator == LayerOperator::GatedDelta)
        .count();
    assert_eq!(
        (recurrent, table.len() - recurrent),
        (FIXTURE_LAYERS * 3 / 4, FIXTURE_LAYERS / 4),
        "3:1 recurrent-to-softmax, the Qwen3.8 cadence"
    );
}

/// `full_attention_interval` corroborates `layer_types`; it is never the
/// source of truth, and an interval that CONTRADICTS the array stops
/// being a benign alias and blocks.
///
/// Without the second half of this test the alias class would only ever
/// have proven that the canonical key was *present*, which a disagreeing
/// checkpoint also satisfies.
#[test]
fn a_contradicting_full_attention_interval_stops_being_a_benign_alias() {
    let build = |interval: u64| {
        let dir = tempfile::tempdir().unwrap();
        let inventory = glimmer_shaped_target_with(dir.path(), |config| {
            let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                })
                .collect();
            config["text_config"]["layer_types"] = serde_json::json!(layer_types);
            config["text_config"]["full_attention_interval"] = serde_json::json!(interval);
        });
        plan_system(&[("target-artifact".to_string(), inventory)])
            .artifacts
            .into_iter()
            .flat_map(|a| a.findings)
            .find(|f| f.subject.ends_with("full_attention_interval"))
            .expect("a finding for full_attention_interval")
    };

    // 4 is what the LLLF array actually implies: corroboration.
    let agreeing = build(4);
    assert_eq!(agreeing.class, SemanticClass::Alias);
    assert!(!agreeing.blocks(), "{agreeing:?}");

    // 3 describes a different cadence entirely. The array still wins —
    // nothing about the graph changes — but the alias is now a second,
    // disagreeing authority and must not pass as benign.
    let contradicting = build(3);
    assert_eq!(contradicting.class, SemanticClass::Unknown);
    assert!(
        contradicting.blocks(),
        "a contradicting interval must block: {contradicting:?}"
    );
}

/// **Regression guard.** The span fix must not take unrelated per-layer
/// facts down with it: rope theta is carried independently of span, and
/// must stay representable even while `layer_types` blocks.
#[test]
fn unrelated_per_layer_facts_still_carry_with_a_hybrid_interleave() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.rope_parameters.rope_theta",
        "text_config.layer_rope_theta",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
    }
}

/// The hybrid fields that still have no destination stay honestly
/// `unrepresented`.
///
/// QW-1 gave the five linear GEOMETRY fields a real destination, QW-2
/// gave `mamba_ssm_dtype` one, and QW-3.5B gave the partial/M-RoPE facts
/// one — see
/// [`the_linear_geometry_is_carried_into_the_operator`] and
/// [`the_state_precision_moves_with_its_executor`]. What remains here has
/// none: MTP has no head object, and `output_gate_type` has no
/// attributable OWNER — the gate itself is judged and executed since
/// QW-3.5C, but HF reads this key nowhere and there is a second, genuine
/// silu gate in DeltaNet, so which operator it describes is unknown.
/// mRoPE left this list at QW-3.5B and `attn_output_gate` at QW-3.5C. Each stays
/// honestly `unrepresented` rather than claiming a home that does not
/// exist.
#[test]
fn declared_hybrid_fields_without_a_destination_stay_unrepresented() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.output_gate_type",
        "text_config.mtp_num_hidden_layers",
        "text_config.mtp_use_dedicated_embeddings",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(finding.class, SemanticClass::ExecutionSemantic, "{subject}");
        assert_eq!(
            finding.category,
            FindingCategory::Unrepresented,
            "{subject}: {}",
            finding.detail
        );
        assert!(
            finding.detail.contains("no schema field"),
            "{subject}: {} — must not be silently dropped, must name the missing judgement",
            finding.detail
        );
        assert!(finding.blocks(), "{subject}");
    }
}

/// `mamba_ssm_dtype` moved only when something could honour it.
///
/// QW-1 deliberately left it blocking while the field was parsed and
/// nearby: with no executor able to keep a recurrence at a declared
/// precision, claiming carriage would have asserted a runtime surface that
/// could not use the value. QW-2's reference operator allocates and
/// accumulates `GatedDeltaState` at exactly this precision, so the claim is
/// now true — and it is the ONLY blocker that rung moved.
#[test]
fn the_state_precision_moves_with_its_executor() {
    let findings = hybrid_findings();
    let finding = finding_for(&findings, "text_config.mamba_ssm_dtype");
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());
    assert!(
        finding.detail.contains("state_dtype") || finding.detail.contains("linear_attention"),
        "must name where it lands: {}",
        finding.detail
    );
}

/// The five linear geometry fields now terminate in a real operator.
///
/// Each lands on `ExecutionSurface.linear_attention` and is consumed by
/// `GatedDeltaOp`, and together they derive the `qkv_channels` and
/// `value_width` that the nine `LinearAttn*` operand contracts close
/// against stored tensors. That whole path is why these grade `Lowered`
/// while `mamba_ssm_dtype` beside them does not.
#[test]
fn the_linear_geometry_is_carried_into_the_operator() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.linear_conv_kernel_dim",
        "text_config.linear_key_head_dim",
        "text_config.linear_value_head_dim",
        "text_config.linear_num_key_heads",
        "text_config.linear_num_value_heads",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
        assert!(
            finding.detail.contains("linear_attention"),
            "{subject} must name where it lands: {}",
            finding.detail
        );
    }
}

/// `full_attention_interval` is a redundant spelling of the same
/// interleave `layer_types` states explicitly, and the parser reads the
/// array — so this leaf grades `Alias` and does not block, even while
/// the interleave it aliases is itself unrepresented.
#[test]
fn full_attention_interval_is_a_non_blocking_alias() {
    let findings = hybrid_findings();
    let finding = finding_for(&findings, "text_config.full_attention_interval");
    assert_eq!(finding.class, SemanticClass::Alias);
    assert!(!finding.blocks());
}

/// The informational per-component attention-policy summary counts a
/// recurrence as itself, and reserves the "no execution vocabulary"
/// disclosure for spellings that genuinely have none.
///
/// Both halves matter. Before QW-3.5A the hybrid fixture's 3-in-4 layers
/// were disclosed as unexecutable; they are now counted as
/// gated-delta recurrent, and the disclosure clause must have
/// *disappeared* rather than lingering with a zero count — a clause that
/// is always emitted is one this assertion could never fail on.
#[test]
fn attention_policy_summary_counts_a_recurrence_and_reserves_the_disclosure() {
    let summary_for = |findings: &[PlannedFinding]| {
        findings
            .iter()
            .find(|f| f.subject == "attention_policy")
            .expect("attention policy finding")
            .detail
            .clone()
    };

    let hybrid = summary_for(&hybrid_findings());
    let recurrent = FIXTURE_LAYERS * 3 / 4;
    assert!(
        hybrid.contains(&format!("{recurrent} gated-delta recurrent")),
        "{hybrid}"
    );
    assert!(
        !hybrid.contains("no execution vocabulary"),
        "nothing is unexecutable here any more: {hybrid}"
    );

    // The control: a spelling with no operator still produces the
    // disclosure, so the assertion above is about this build's behaviour
    // and not about the sentence having been deleted.
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["layer_types"] =
            serde_json::json!(vec!["hyena_attention"; FIXTURE_LAYERS]);
    });
    let unknown: Vec<PlannedFinding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();
    let unknown = summary_for(&unknown);
    assert!(unknown.contains("no execution vocabulary"), "{unknown}");
    assert!(
        unknown.contains(&format!("{FIXTURE_LAYERS} declared span(s)")),
        "{unknown}"
    );
}

/// QW-3.5B: the partial and multi-axis rotary facts reach a real
/// consumer, so they stop blocking.
///
/// The evidence claim is deliberately narrower than "proven correct":
/// `partial_rotary_factor` is falsified on the text path by
/// `opplan::exec::tests::mrope_parity`, while `mrope_section` and
/// `mrope_interleaved` are carried and lowered on the strength of the
/// declaration — text positions make them unidentifiable, as that
/// module's measured table records. Carriage is what this gate asserts;
/// numerical identifiability is a separate question with a separate
/// answer.
#[test]
fn the_rotary_facts_are_carried_into_the_position_policy() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.partial_rotary_factor",
        "text_config.rope_parameters.mrope_section",
        "text_config.rope_parameters.mrope_interleaved",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
    }
}

/// **The real Qwen3.5 shape: the fraction is declared under
/// `rope_parameters` only, and it must still lower — with its mrope.**
///
/// `hybrid_findings` mirrors Qwen3.8, which writes the fraction at both
/// spots. Every Qwen3.5 checkpoint (0.8B through 397B-A17B) writes only
/// the nested one, and until wave 8 the parser read only the flat one:
/// no layer carried a rotary fraction, `probe_partial_rotary_factor` and
/// `mrope_of` both answered `None`, and three leaves refused a family
/// whose text path this build executes. The reference reads the nested
/// spelling (`rope_parameters_dict.get("partial_rotary_factor", 1.0)`),
/// so this is a carriage gap on the parser side of the boundary, not a
/// new position policy.
#[test]
fn the_nested_only_partial_rotary_spelling_lowers_with_its_mrope() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        declare_hybrid_cadence(config);
        declare_gated_delta_geometry(config);
        config["text_config"]["attn_output_gate"] = serde_json::json!(true);
        config["text_config"]["output_gate_type"] = serde_json::json!("swish");
        // Same closing geometry as `hybrid_findings`, nested spelling only.
        config["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.5);
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(true);
        config["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([1, 1, 0]);
    });
    let findings: Vec<PlannedFinding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();
    assert!(
        !findings
            .iter()
            .any(|f| f.subject == "text_config.partial_rotary_factor"),
        "the fixture must not declare the flat spelling, or it proves nothing"
    );
    for subject in [
        "text_config.rope_parameters.partial_rotary_factor",
        "text_config.rope_parameters.mrope_section",
        "text_config.rope_parameters.mrope_interleaved",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
    }
}

/// **The value-sensitive half.** Two spellings of the partial rotary that
/// DISAGREE must block, even though both parse.
///
/// The fixture declares `partial_rotary_factor` at `text_config` and
/// again under `rope_parameters`. Agreement is the ordinary case and
/// carries; disagreement means the checkpoint states two different
/// execution semantics and a parser silently picks one.
#[test]
fn two_disagreeing_partial_rotary_spellings_block() {
    let build = |nested: f64| {
        let dir = tempfile::tempdir().unwrap();
        let inventory = glimmer_shaped_target_with(dir.path(), |config| {
            config["text_config"]["partial_rotary_factor"] = serde_json::json!(0.25);
            config["text_config"]["rope_parameters"]["partial_rotary_factor"] =
                serde_json::json!(nested);
            config["text_config"]["rope_parameters"]["mrope_section"] =
                serde_json::json!([2, 2, 1]);
            config["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(true);
        });
        plan_system(&[("target-artifact".to_string(), inventory)])
            .artifacts
            .into_iter()
            .flat_map(|a| a.findings)
            .filter(|f| f.subject.contains("two spellings"))
            .collect::<Vec<_>>()
    };
    assert!(
        build(0.25).is_empty(),
        "agreeing spellings are the ordinary case and must not block"
    );
    let disagreeing = build(0.5);
    assert_eq!(disagreeing.len(), 1, "{disagreeing:?}");
    assert!(disagreeing[0].blocks(), "{:?}", disagreeing[0]);
}

/// The arithmetic gate refuses a section that does not close.
///
/// `sum(section) * 2 == rotary_dim == head_dim * partial_rotary_factor`.
/// A section summing to 3 needs a 6-dim rotary block; the fixture's is 4.
/// Without this, "the mrope facts are carried" would also be satisfied by
/// a probe that never checked the arithmetic at all — which is precisely
/// how the 128-head-width misreading would have slipped through.
#[test]
fn a_section_that_does_not_close_the_arithmetic_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
        config["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.5);
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(true);
        // Sums to 3, so it describes a 6-dim rotary block on a head whose
        // fraction gives 4.
        config["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([1, 1, 1]);
    });
    let findings: Vec<PlannedFinding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();
    for subject in [
        "text_config.rope_parameters.mrope_section",
        "text_config.rope_parameters.mrope_interleaved",
    ] {
        let finding = finding_for(&findings, subject);
        assert!(
            finding.blocks(),
            "a section whose arithmetic does not close must refuse: {finding:?}"
        );
    }
}

/// QW-3.5C: the gate's EXISTENCE is carried; its declared TYPE is not.
///
/// These two keys sit side by side in the config and receive opposite
/// verdicts, which is the whole point. `attn_output_gate` is
/// corroborated by the stored projection carrying `2 · heads · head_dim`
/// rows, so it has both a consumer and an independent witness.
/// `output_gate_type: "swish"` has neither: HF reads it nowhere, and
/// swish gating (`x · silu(g)`) is not what the reference implementation
/// computes (`x · sigmoid(g)`). Resolving it on the resemblance to
/// DeltaNet's genuine silu gate would be a semantic-ownership guess.
#[test]
fn the_gate_exists_is_carried_but_its_declared_type_is_not() {
    let findings = hybrid_findings();
    let exists = finding_for(&findings, "text_config.attn_output_gate");
    assert_eq!(
        exists.category,
        FindingCategory::Representable,
        "{}",
        exists.detail
    );
    assert!(!exists.blocks());

    let kind = finding_for(&findings, "text_config.output_gate_type");
    assert_eq!(
        kind.category,
        FindingCategory::Unrepresented,
        "ownership unresolved is `Unrepresented`, never `Mismatched` — the latter would \
         assert the two authorities describe the same subject: {}",
        kind.detail
    );
    assert!(kind.blocks());
}
