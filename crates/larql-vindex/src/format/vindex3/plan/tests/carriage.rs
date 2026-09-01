//! The VINDEX3-boundary authority gate: parser consumption is not
//! representation authority.
//!
//! Every test here pairs a positive and a negative arm on the *same* key,
//! because the instrument's claim is discriminative: it must fire when a
//! declared fact is dropped and stay silent when the same fact is carried.
//! A gate that only ever fires proves nothing about the facts it passes.

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::carriage::{rule_for, Carriage, CARRIAGE_RULES};
use crate::format::vindex3::plan::{plan_system, Finding, FindingCategory, SemanticClass};

/// Plan the Glimmer-shaped fixture with `mutate` applied to its config.
fn plan_with(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<Finding> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target_with(dir.path(), mutate),
    )];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// The finding whose subject ends with `suffix`.
fn finding_for<'a>(findings: &'a [Finding], suffix: &str) -> &'a Finding {
    findings
        .iter()
        .find(|f| f.subject.ends_with(suffix))
        .unwrap_or_else(|| panic!("no finding for `{suffix}`"))
}

/// **The positive control.** A `yarn` block that is *incomplete* — GPT-OSS
/// declares `factor: 32` but this checkpoint omits
/// `original_max_position_embeddings`, which YaRN's correction bounds are
/// defined against — resolves to no scaling at all (`yarn_rope_scaling`
/// refuses a malformed block), so the position policy carries plain rope.
/// The carriage probe answers `default`, the comparison against the
/// declaration fails, and the plan refuses: a `yarn` the container cannot
/// honour is a dropped fact, whatever the reason.
#[test]
fn an_incomplete_yarn_block_resolves_to_plain_rope_and_blocks() {
    let findings = plan_with(|config| {
        config["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        config["text_config"]["rope_parameters"]["factor"] = serde_json::json!(32.0);
    });
    let finding = finding_for(&findings, "rope_parameters.rope_type");

    assert_eq!(finding.category, FindingCategory::Mismatched);
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(finding.declared, Some(serde_json::json!("yarn")));
    assert_eq!(finding.resolved, Some(serde_json::json!("default")));
    assert_eq!(
        finding.carriage,
        Some(Carriage::Parsed),
        "a dropped fact reaches the parser and no further"
    );
    assert!(finding.blocks(), "a dropped execution semantic must block");
}

/// **The positive arm of A-9.0.** A complete `yarn` block — GPT-OSS's, with
/// its `original_max_position_embeddings` — becomes `PositionPolicy::Yarn`
/// on every rotating layer, and the probe reads `yarn` back out of the
/// built table. Every leaf of the block is judged against what the policy
/// carries, not merely credited for having been parsed.
#[test]
fn a_complete_yarn_block_is_carried_as_position_policy_yarn() {
    let findings = plan_with(|config| {
        let rp = &mut config["text_config"]["rope_parameters"];
        rp["rope_type"] = serde_json::json!("yarn");
        rp["factor"] = serde_json::json!(32.0);
        rp["beta_fast"] = serde_json::json!(32.0);
        rp["beta_slow"] = serde_json::json!(1.0);
        rp["truncate"] = serde_json::json!(false);
        rp["original_max_position_embeddings"] = serde_json::json!(4096);
    });
    let rope_type = finding_for(&findings, "rope_parameters.rope_type");
    assert_eq!(
        rope_type.category,
        FindingCategory::Representable,
        "{rope_type:?}"
    );
    assert_eq!(rope_type.resolved, Some(serde_json::json!("yarn")));
    assert_eq!(rope_type.carriage, Some(Carriage::Represented));
    assert!(!rope_type.blocks());

    for (leaf, declared) in [
        ("factor", serde_json::json!(32.0)),
        ("beta_fast", serde_json::json!(32.0)),
        ("beta_slow", serde_json::json!(1.0)),
        ("truncate", serde_json::json!(false)),
        ("original_max_position_embeddings", serde_json::json!(4096)),
    ] {
        let f = finding_for(&findings, &format!("rope_parameters.{leaf}"));
        assert_eq!(f.category, FindingCategory::Representable, "{leaf}: {f:?}");
        assert_eq!(f.declared, Some(declared), "{leaf}");
        assert_eq!(f.carriage, Some(Carriage::Represented), "{leaf}");
        assert!(!f.blocks(), "{leaf}");
    }
}

/// **The negative control, on the same key.** Muse-Glimmer declares
/// `rope_type: "default"` — unscaled rope, which is exactly what
/// `PositionPolicy` expresses. The probe agrees with the declaration and
/// the fact is reported as carried, so the gate is discriminating between
/// values rather than objecting to the key's existence.
#[test]
fn declared_default_rope_type_is_carried_not_blocked() {
    let findings = plan_with(|_| {});
    let finding = finding_for(&findings, "rope_parameters.rope_type");

    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.declared, Some(serde_json::json!("default")));
    assert_eq!(finding.resolved, Some(serde_json::json!("default")));
    assert_eq!(finding.carriage, Some(Carriage::Represented));
    assert!(!finding.blocks());
}

/// Every execution-semantic leaf the registry knows has a carriage rule.
/// The plan blocks an unruled execution-semantic leaf ("parser
/// consumption is not representation authority" — the safety net stays
/// in `carriage_finding`), and this pins that the net catches nothing
/// today: `partial_rotary_factor` was the last live occupant until G4.0
/// judged it against `PositionPolicy::PartialRope`, and
/// `num_kv_shared_layers` until G4.0 gave it a rule; a leaf added to the
/// registry without a rule fails here before it fails on a checkpoint.
#[test]
fn every_execution_semantic_leaf_has_a_carriage_rule() {
    use crate::format::vindex3::plan::semantics::EXECUTION_SEMANTIC_KEYS;
    let unruled: Vec<&str> = EXECUTION_SEMANTIC_KEYS
        .iter()
        .copied()
        .filter(|leaf| rule_for(leaf).is_none())
        .collect();
    assert!(
        unruled.is_empty(),
        "execution-semantic leaves with no carriage rule: {unruled:?}"
    );
}

/// A declared partial rotary is carried into the layer policy, at the
/// value declared.
///
/// This test previously asserted the opposite — that the fact was parsed
/// and dropped, which was the honest verdict while no resolver honoured
/// it. QW-3.5B moved `partial_rotary_factor` into the trait default, so
/// the generic path now resolves `PositionPolicy::PartialRope` and the
/// probe answers. The assertion that keeps this from being a weakening is
/// the value: the probe must report the DECLARED fraction, so a resolver
/// that carried some other fraction still fails here.
#[test]
fn a_partial_rotary_factor_is_carried_at_the_declared_value() {
    let findings = plan_with(|config| {
        config["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
    });
    let finding = finding_for(&findings, "partial_rotary_factor");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());
    assert_eq!(
        finding.resolved,
        Some(serde_json::json!(0.5)),
        "the probe must report the declared fraction, not merely answer: {}",
        finding.detail
    );

    // A full rotary is not a partial one: declaring 1.0 must NOT put the
    // model on the partial path, or every ordinary checkpoint changes
    // execution. This is the arm that stops the change above from being
    // "always resolve PartialRope".
    let full = plan_with(|config| {
        config["text_config"]["partial_rotary_factor"] = serde_json::json!(1.0);
    });
    let full = finding_for(&full, "partial_rotary_factor");
    assert!(
        full.resolved != Some(serde_json::json!(1.0)),
        "a fraction of 1.0 is a full rotary and must not resolve as a partial one: {}",
        full.detail
    );
}

/// `swiglu_limit` on an architecture whose FFN gate is plain gating: the
/// surface carries `ExpertGatePolicy::Gated`, so there is no limit to
/// answer with, and the declared clamp is reported as unrepresented —
/// which is the truth about this fixture, and blocks.
#[test]
fn swiglu_limit_on_a_plain_gated_ffn_is_unrepresented_and_blocks() {
    let findings = plan_with(|config| {
        config["text_config"]["swiglu_limit"] = serde_json::json!(7.0);
    });
    let finding = finding_for(&findings, "swiglu_limit");

    assert_eq!(
        finding.category,
        FindingCategory::Unrepresented,
        "{finding:?}"
    );
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert!(
        finding.detail.contains("no built component answered"),
        "{}",
        finding.detail
    );
    assert!(finding.blocks());
}

/// The rule itself is on the books, reaching `Represented` at the gate
/// policy — so a GPT-OSS-shaped surface, whose architecture resolves
/// `ExpertGatePolicy::ClampedGlu { limit: swiglu_limit, .. }`, answers the
/// probe with the limit and the fact is judged carried. (An end-to-end
/// GPT-OSS fixture is A-9.2's; here the rule and its probe are pinned.)
#[test]
fn swiglu_limit_has_a_carriage_rule_at_the_gate_policy() {
    let rule = rule_for("swiglu_limit").expect("swiglu_limit is judged");
    assert_eq!(rule.reaches, Carriage::Represented);
    assert!(rule.site.contains("gate_policy"), "{}", rule.site);
    assert!(rule.probe.is_some());
}

/// A fact that honestly stops at the parser is *reported*, not hidden.
/// `max_position_embeddings` is a KV-allocation bound that no generic op
/// reads; the rule says so and carries the reason, so the absence is a
/// judgement on the report rather than a silence in it.
#[test]
fn a_fact_that_stops_at_the_parser_is_reported_with_its_reason() {
    let findings = plan_with(|config| {
        config["text_config"]["max_position_embeddings"] = serde_json::json!(131072);
    });
    let finding = finding_for(&findings, "max_position_embeddings");

    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.carriage, Some(Carriage::Parsed));
    assert!(
        finding.detail.contains("stops at the parser by judgement"),
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());
}

/// `mlp_bias` gets the same treatment as `attention_bias`: no schema field,
/// judged inert at the parser because operand closure over the checkpoint's
/// actual FFN bias tensors is the real gate (G5b), not this boolean. Granite
/// 4.1 3B/8B/30B all declare `false`.
#[test]
fn mlp_bias_is_reported_and_does_not_block() {
    let findings = plan_with(|config| {
        config["text_config"]["mlp_bias"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "mlp_bias");

    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.carriage, Some(Carriage::Parsed));
    assert!(
        finding.detail.contains("stops at the parser by judgement"),
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());
}

/// f32 narrowing is not a dropped fact. GPT-OSS declares `rms_norm_eps:
/// 1e-5` and the surface carries `9.999999747378752e-6` — the same value
/// through f32, bit for bit. Reporting that would be the gate misreading
/// its own instrument.
#[test]
fn f32_narrowing_is_not_reported_as_a_dropped_fact() {
    let findings = plan_with(|_| {});
    let finding = finding_for(&findings, "rms_norm_eps");

    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.declared, Some(serde_json::json!(1e-5)));
    assert!(!finding.blocks());
}

/// …and the tolerance is not so loose that a real change slips through:
/// the fixture's post-norm epsilon is three orders of magnitude from its
/// pre-norm one, and they stay distinguishable as f32.
#[test]
fn f32_agreement_still_separates_genuinely_different_epsilons() {
    let findings = plan_with(|_| {});
    let pre = finding_for(&findings, "rms_norm_eps");
    let post = finding_for(&findings, "post_norm_eps");

    assert_eq!(pre.declared, Some(serde_json::json!(1e-5)));
    assert_eq!(post.declared, Some(serde_json::json!(1e-8)));
    assert_ne!(
        pre.resolved, post.resolved,
        "1e-5 and 1e-8 must not collapse into one carried value"
    );
    assert!(!pre.blocks() && !post.blocks());
}

/// `query_pre_attn_scalar` is the derived-value counterpart to
/// `rms_norm_eps`'s f32-narrowing case above: the checkpoint declares the
/// raw scalar (256), VINDEX3's execution surface stores the score scale
/// execution actually reads (`256^-0.5`) — a different JSON value, carried
/// through `canonical_declared` rather than compared as raw JSON. The
/// finding stays representable, and says so explicitly rather than
/// silently matching.
#[test]
fn query_pre_attn_scalar_agrees_via_the_canonical_score_scale_derivation() {
    let findings = plan_with(|config| {
        config["text_config"]["query_pre_attn_scalar"] = serde_json::json!(256.0);
    });
    let finding = finding_for(&findings, "query_pre_attn_scalar");

    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.declared, Some(serde_json::json!(256.0)));
    assert!(!finding.blocks());
    assert!(
        finding.detail.contains("canonical conversion"),
        "{}",
        finding.detail
    );
}

/// An alias is benign only while the canonical spelling it defers to is
/// genuinely declared and consumed. GPT-OSS ships both
/// `experts_per_token` and `num_experts_per_tok`; the parser reads the
/// latter.
#[test]
fn an_alias_backed_by_its_canonical_spelling_does_not_block() {
    let findings = plan_with(|config| {
        config["text_config"]["experts_per_token"] = serde_json::json!(4);
        config["text_config"]["num_experts_per_tok"] = serde_json::json!(4);
    });
    let finding = finding_for(&findings, "experts_per_token");

    assert_eq!(finding.class, SemanticClass::Alias);
    assert!(!finding.blocks());
}

/// …and an alias with no canonical spelling behind it is the *only*
/// carrier of its fact, so it grades `Unknown` and blocks. Without this
/// arm, listing a key in the alias registry would be a way to silence it.
#[test]
fn an_unbacked_alias_blocks_rather_than_silencing_the_fact() {
    let findings = plan_with(|config| {
        config["text_config"]["experts_per_token"] = serde_json::json!(4);
    });
    let finding = finding_for(&findings, "experts_per_token");

    assert_eq!(
        finding.class,
        SemanticClass::Unknown,
        "an alias with nothing behind it is unjudged, not benign"
    );
    assert!(finding.blocks());
}

/// Training-time facts are judged inert, and say which path they belong
/// to rather than being dropped as unclassified noise.
#[test]
fn training_only_facts_are_judged_inert() {
    let findings = plan_with(|config| {
        config["text_config"]["router_aux_loss_coef"] = serde_json::json!(0.9);
        config["text_config"]["output_router_logits"] = serde_json::json!(false);
    });
    for subject in ["router_aux_loss_coef", "output_router_logits"] {
        let finding = finding_for(&findings, subject);
        assert_eq!(finding.class, SemanticClass::TrainingOnly, "{subject}");
        assert!(!finding.blocks(), "{subject}");
    }
}

/// The census reports every declared key, so `unrepresented: N` is a
/// count against a stated denominator rather than a lower bound. Before
/// this, consumed keys produced no finding at all and the report could
/// not be audited for exhaustiveness.
#[test]
fn every_declared_key_receives_a_finding() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |_| {});
    let declared: Vec<String> = inventory
        .config_keys
        .iter()
        .map(|f| f.path.clone())
        .collect();
    let findings = plan_with(|_| {});

    for path in &declared {
        assert!(
            findings.iter().any(|f| &f.subject == path),
            "`{path}` is declared but receives no judgement"
        );
    }
}

/// Carriage stages are ordered, so "deeper than" is a comparison rather
/// than a convention every call site has to remember.
#[test]
fn carriage_stages_are_ordered() {
    assert!(Carriage::Parsed < Carriage::Represented);
    assert!(Carriage::Represented < Carriage::Lowered);
    assert!(Carriage::Lowered < Carriage::Executed);
}

/// The stage name every finding's `detail` prints — all four, not just
/// the ones a rule happens to reach in the fixtures above.
#[test]
fn carriage_stage_names_print_as_the_report_shows_them() {
    assert_eq!(Carriage::Parsed.name(), "parsed");
    assert_eq!(Carriage::Represented.name(), "represented");
    assert_eq!(Carriage::Lowered.name(), "lowered");
    assert_eq!(Carriage::Executed.name(), "executed");
}

/// Every rule claiming carriage past the parser must ship a probe: the
/// claim is checked against the built graph, never trusted. A rule that
/// stops at `parsed` has nothing to read back and must not pretend to.
#[test]
fn rules_claiming_carriage_must_be_checkable() {
    for rule in CARRIAGE_RULES {
        if rule.reaches > Carriage::Parsed {
            assert!(
                rule.probe.is_some(),
                "`{}` claims `{}` with no probe to check it",
                rule.leaf,
                rule.reaches.name()
            );
        } else {
            assert!(
                rule.probe.is_none(),
                "`{}` stops at the parser and has nothing to read back",
                rule.leaf
            );
        }
        assert!(
            !rule.site.is_empty(),
            "`{}` must name where it lands or why it stops",
            rule.leaf
        );
    }
}

/// Rules are keyed by leaf name and must be unique: two entries for one
/// leaf would make the governing judgement depend on table order.
#[test]
fn rules_are_unique_per_leaf() {
    for rule in CARRIAGE_RULES {
        let matches = CARRIAGE_RULES
            .iter()
            .filter(|r| r.leaf == rule.leaf)
            .count();
        assert_eq!(matches, 1, "`{}` has {matches} rules", rule.leaf);
        assert!(rule_for(rule.leaf).is_some());
    }
}

/// **Positive control for nested components.** A perception tower's
/// declared interleave and rope base reach the graph's per-layer policy
/// table, derived from that component's own topology — the same shape the
/// text path uses, one level down, with no component-specific code.
///
/// Before this, nested components carried `attention: None`, so the facts
/// were parsed into `ComponentTopology` and dropped before the graph with
/// nothing reporting the loss.
#[test]
fn a_nested_components_declared_policy_reaches_the_graph() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target_with(dir.path(), |_| {}),
    )];
    let graph = plan_system(&named).graph;
    let vision = graph
        .components
        .iter()
        .find(|c| c.id == "vision")
        .expect("the vision component exists");
    let table = vision
        .attention
        .as_ref()
        .expect("a nested component carries its own per-layer policy");

    assert_eq!(table.len(), 4, "one entry per declared layer");
    assert_eq!(
        table
            .iter()
            .map(|l| l.declared_name().expect("a softmax span renders"))
            .collect::<Vec<_>>(),
        vec![
            "window_attention",
            "window_attention",
            "window_attention",
            "full_attention"
        ],
        "the declared interleave is carried verbatim, not flattened"
    );
    for layer in table {
        assert_eq!(
            layer.position,
            larql_models::config::PositionPolicy::Rope { theta: 10000.0 },
            "the tower's own rope base, not the text model's"
        );
        assert_eq!(
            layer.window, None,
            "a spatial window is not a position count"
        );
    }
}

/// …and the carriage gate reports those facts as carried, so the vision
/// tower no longer blocks the plan.
#[test]
fn nested_policy_facts_are_reported_as_carried() {
    let findings = plan_with(|_| {});
    for subject in [
        "vision_config.layer_types",
        "vision_config.rope_parameters.rope_theta",
        "vision_config.rope_parameters.rope_type",
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

/// **Negative control.** Perturb one vision layer's declared span and the
/// gate must catch the disagreement — otherwise the positive control
/// above only proves the probe reads *something*, not that it reads the
/// right thing.
#[test]
fn a_perturbed_nested_interleave_is_caught() {
    let findings = plan_with(|config| {
        config["vision_config"]["layer_types"] = serde_json::json!([
            "window_attention",
            "window_attention",
            "window_attention",
            "window_attention"
        ]);
        // …while the *tensors* still evidence the original 4-layer tower,
        // so only the declared interleave moved.
    });
    let finding = finding_for(&findings, "vision_config.layer_types");
    assert_eq!(
        finding.declared,
        Some(serde_json::json!([
            "window_attention",
            "window_attention",
            "window_attention",
            "window_attention"
        ]))
    );
    assert_eq!(
        finding.resolved, finding.declared,
        "the graph must follow the declaration, not a remembered shape"
    );
}

/// …and the same for the rope base: change it and the carried value
/// changes with it, so the probe is reading the tower's own fact rather
/// than a constant.
#[test]
fn a_perturbed_nested_rope_base_is_followed() {
    let findings = plan_with(|config| {
        config["vision_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(250000.0);
    });
    let finding = finding_for(&findings, "vision_config.rope_parameters.rope_theta");
    assert_eq!(finding.resolved, Some(serde_json::json!(250000.0)));
    assert_eq!(finding.category, FindingCategory::Representable);
}

/// A span spelling the vocabulary does not contain refuses the table
/// rather than resolving to a default. "Not sliding, therefore full" is
/// exactly how `layer_types` was silently ignored before, and the
/// nested path must not reintroduce it.
#[test]
fn an_unknown_span_spelling_refuses_rather_than_defaulting() {
    use crate::format::vindex3::graph::policy::AttentionSpan;
    assert_eq!(
        AttentionSpan::from_declared("sliding_attention"),
        Some(AttentionSpan::Sliding)
    );
    assert_eq!(
        AttentionSpan::from_declared("window_attention"),
        Some(AttentionSpan::Windowed)
    );
    assert_eq!(
        AttentionSpan::from_declared("full_attention"),
        Some(AttentionSpan::Full)
    );
    assert_eq!(
        AttentionSpan::from_declared("chunked_attention"),
        None,
        "an unjudged spelling must refuse, not resolve to full attention"
    );

    let findings = plan_with(|config| {
        config["vision_config"]["layer_types"] = serde_json::json!([
            "chunked_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ]);
    });
    let finding = finding_for(&findings, "vision_config.layer_types");
    assert!(
        finding.blocks(),
        "an unrepresentable span must block: {}",
        finding.detail
    );
}

/// Every span in the vocabulary round-trips through its declared name, so
/// `from_declared` and `declared_name` cannot drift apart and make the
/// probe compare against a spelling no checkpoint uses.
#[test]
fn span_names_round_trip() {
    use crate::format::vindex3::graph::policy::AttentionSpan;
    for span in [
        AttentionSpan::Sliding,
        AttentionSpan::Full,
        AttentionSpan::Windowed,
    ] {
        assert_eq!(
            AttentionSpan::from_declared(span.declared_name()),
            Some(span)
        );
    }
}

/// A rule whose probe never answers (`probe_unrepresented`) reports the
/// declared fact as dropped at the boundary, naming its site — the honest
/// verdict for a leaf the schema has no field for (`mscale_all_dim`), and
/// it blocks.
///
/// The example used to be `norm_topk_prob`, which has since gained a real
/// destination (`ExecutionSurface.ffn.moe.routing_policy`). The idiom is
/// what this test pins, so it moved to a leaf that still has no field
/// rather than being deleted with the rule it happened to name.
#[test]
fn a_no_schema_field_rule_reports_the_fact_unrepresented_and_blocks() {
    let findings = plan_with(|config| {
        config["text_config"]["rope_scaling"] = serde_json::json!({
            "rope_type": "yarn",
            "mscale_all_dim": 1.0,
        });
    });
    let finding = finding_for(&findings, "mscale_all_dim");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert!(
        finding.detail.contains("no schema field"),
        "{}",
        finding.detail
    );
    assert!(finding.blocks());
}

/// Granite's `residual_multiplier` is read by the generic resolution into
/// the surface's residual scale and judged there: a declaration is carried
/// to the lowered op and the probe answers with the stored value.
#[test]
fn a_declared_residual_multiplier_is_carried_to_the_residual_scale() {
    let declared = 0.22;
    let findings = plan_with(|config| {
        config["text_config"]["residual_multiplier"] = serde_json::json!(declared);
    });
    let finding = finding_for(&findings, "residual_multiplier");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.carriage, Some(Carriage::Lowered));
    let resolved = finding.resolved.as_ref().and_then(|v| v.as_f64()).unwrap();
    assert!((resolved - declared).abs() < 1e-6);
}

/// `query_pre_attn_scalar` is judged in the SCORE-SCALE space: the
/// declared scalar is canonicalised to `scalar^-0.5` before comparing with
/// the surface's score scale, so a family that resolves its score scale
/// from head_dim disagrees (and says so) rather than comparing 256 to 0.35.
#[test]
fn query_pre_attn_scalar_is_compared_in_score_scale_space() {
    use crate::format::vindex3::plan::carriage::canonical_declared;
    let scalar: f64 = 256.0;
    let canonical = canonical_declared("query_pre_attn_scalar", &serde_json::json!(scalar));
    let expected = scalar.powf(-0.5);
    assert!((canonical.as_f64().unwrap() - expected).abs() < 1e-12);
    // A non-numeric declaration passes through untouched.
    assert_eq!(
        canonical_declared("query_pre_attn_scalar", &serde_json::json!("x")),
        serde_json::json!("x")
    );
    // Any other leaf is untouched.
    assert_eq!(
        canonical_declared("rope_theta", &serde_json::json!(scalar)),
        serde_json::json!(scalar)
    );
    let findings = plan_with(|config| {
        config["text_config"]["query_pre_attn_scalar"] = serde_json::json!(scalar);
    });
    let finding = finding_for(&findings, "query_pre_attn_scalar");
    // The resolution reads the scalar into the surface's score scale
    // (256 → 0.0625), and the comparison agrees in that space — not by
    // comparing 256 with 0.0625 as raw JSON.
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(expected as f32)));
}

/// A tensor group the graph cannot place — an unknown prefix with no
/// object kind — surfaces as an unrepresented finding that names the
/// prefix and the builder's reason, and it blocks.
#[test]
fn an_unplaceable_tensor_group_is_reported_and_blocks() {
    use crate::format::vindex3::plan::tests_support::custom_artifact;
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    let mystery: [usize; 2] = [4, 4];
    let inventory = custom_artifact(
        dir.path(),
        &config,
        &[
            ("model.embed_tokens.weight", &[128usize, 64]),
            ("mystery_block.weight", &mystery),
        ],
    );
    let plan = plan_system(&[("odd-artifact".to_string(), inventory)]);
    let finding = plan
        .artifacts
        .iter()
        .flat_map(|a| a.findings.iter())
        .find(|f| f.subject.starts_with("mystery_block"))
        .expect("the unplaced group is reported");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert_eq!(finding.class, SemanticClass::Unknown);
    assert!(finding.blocks());
    assert!(!plan.admissible);
}

/// A nested component whose execution surface cannot be completed
/// (vision tower missing its head count) surfaces as an
/// `execution_surface` finding naming the missing fact, and it blocks.
#[test]
fn an_incomplete_nested_surface_is_reported_and_blocks() {
    let findings = plan_with(|config| {
        config["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_attention_heads");
    });
    let finding = finding_for(&findings, "vision.execution_surface");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert!(
        finding.detail.contains("num_attention_heads"),
        "{}",
        finding.detail
    );
    assert!(finding.blocks());
}

/// **A declaration no reference implementation reads still has to agree.**
///
/// SmolLM2-135M ships `rope_interleaved: false` under `model_type: llama`,
/// which has no such field — transformers reads it nowhere. This build
/// pairs split-half, so `false` agrees and the key is satisfied.
///
/// The second arm is the whole point of reading it at all: `true` names a
/// different operator, and an unread agreement would have let that
/// through as a rotation performed against different partners in silence.
#[test]
fn a_declared_rotary_pairing_must_agree_with_the_one_that_runs() {
    let findings = plan_with(|config| {
        config["text_config"]["rope_interleaved"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "rope_interleaved");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "split-half agrees with `false`: {}",
        finding.detail
    );
    assert!(!finding.blocks());

    let findings = plan_with(|config| {
        config["text_config"]["rope_interleaved"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "rope_interleaved");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "an interleaved pairing is not what this build performs: {}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(false)));
}

/// The same shape for `use_mrope`, checked against the policy the axis
/// geometry resolves rather than against the flag itself.
#[test]
fn a_declared_mrope_flag_must_agree_with_the_resolved_policy() {
    // Qwen2.5-0.5B's real declaration: `false`, on a config carrying no
    // axis geometry, so no multi-axis policy resolves and the two agree.
    let findings = plan_with(|config| {
        config["text_config"]["use_mrope"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "use_mrope");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );

    // Claiming multi-axis without the geometry to build one is a claim
    // nothing can honour, and must not pass merely because the flag was
    // read.
    let findings = plan_with(|config| {
        config["text_config"]["use_mrope"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "use_mrope");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "no mrope_section resolves no MRope policy: {}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(false)));
}

/// The rotary SCHEDULE, answered layer by layer from the graph.
///
/// The fixture already leaves layers 3 and 7 unrotated through
/// `layer_rope_theta`'s zero sentinel, so a `no_rope_layers` mask stating
/// the same schedule must be reported as carried. The second arm is what
/// makes that non-vacuous: a mask that schedules NOTHING changes the
/// answer, so the probe is reading the graph rather than echoing the
/// declaration back.
#[test]
fn a_declared_rope_schedule_is_carried_layer_by_layer() {
    let agreeing = serde_json::json!([1, 1, 1, 0, 1, 1, 1, 0]);
    let findings = plan_with(|config| {
        config["text_config"]["no_rope_layers"] = agreeing.clone();
    });
    let finding = finding_for(&findings, "no_rope_layers");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(
        finding.resolved,
        Some(agreeing),
        "the probe must answer the schedule in the checkpoint's own polarity: {}",
        finding.detail
    );

    // Every layer NoPE: a different declaration must produce a different
    // carried answer, or the first arm proves nothing.
    let all_nope = serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0]);
    let findings = plan_with(|config| {
        config["text_config"]["no_rope_layers"] = all_nope.clone();
    });
    assert_eq!(
        finding_for(&findings, "no_rope_layers").resolved,
        Some(all_nope),
        "a mask that rotates nowhere must be carried as such"
    );
}

/// The positional SCHEME, and the null answer that is an answer.
#[test]
fn a_stack_that_rotates_nowhere_reports_no_scheme() {
    // The fixture rotates on six of its eight layers, so a declared
    // `rope` agrees with what the graph carries.
    let findings = plan_with(|config| {
        config["text_config"]["position_embedding_type"] = serde_json::json!("rope");
    });
    let finding = finding_for(&findings, "position_embedding_type");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!("rope")));

    // Take the rotation away and the same declaration must stop being
    // satisfied. `null` here is the graph saying "this stack encodes no
    // position" — reporting it as unknown would hide exactly the case
    // the rule exists for.
    let findings = plan_with(|config| {
        config["text_config"]["position_embedding_type"] = serde_json::json!("rope");
        config["text_config"]["no_rope_layers"] = serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0]);
    });
    let finding = finding_for(&findings, "position_embedding_type");
    assert_eq!(
        finding.resolved,
        Some(serde_json::Value::Null),
        "a stack that rotates nowhere answers null: {}",
        finding.detail
    );
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "a declared scheme the graph does not carry must not read as satisfied: {}",
        finding.detail
    );
}
