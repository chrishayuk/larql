//! Execution-surface gates (V3-G5a): the builder fills the surface from
//! resolution and evidence, and completeness derives from the operations
//! a component's objects imply — refusals, never defaults.

use larql_models::config::{Activation, FfnType, NormType};

use crate::format::vindex3::graph::object::{LogicalObject, ObjectKind};
use crate::format::vindex3::graph::{
    build_from_inventories, execution_completeness, CompletenessDefect,
};
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense,
};

fn glimmer_pair() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    (target_dir, drafter_dir, named)
}

/// THE 5a gate, build half: every fixture component gets a complete
/// surface, resolved values landing where the ops read them.
#[test]
fn builder_fills_every_component_surface() {
    let (_a, _b, named) = glimmer_pair();
    let built = build_from_inventories(&named);
    assert!(
        built.incomplete_surfaces.is_empty(),
        "{:?}",
        built.incomplete_surfaces
    );
    assert!(execution_completeness(&built.graph).is_empty());

    let target = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    let surface = target.execution.as_ref().unwrap();
    // An attention-class program carries its groups (schema 6: presence
    // means the program runs them).
    let attention = surface.attention.as_ref().expect("the program attends");
    let ffn = surface.ffn.as_ref().expect("the program runs an FFN");
    // Scales stay separate: declared query factor, canonical score scale.
    let query_scale = attention.query_scale.expect("declared qk_scale_factor");
    assert!((query_scale - 3.87).abs() < 1e-12);
    assert!((attention.score_scale - (8f64).powf(-0.5)).abs() < 1e-12);
    // Judged semantics from the registered family.
    assert!(attention.output_gate.is_some());
    assert!(attention.parameter_free_qk_norm.q);
    assert_eq!(attention.num_q_heads, 8);
    assert_eq!(attention.num_kv_heads, 2);
    assert_eq!(ffn.intermediate_size, Some(256));
    // Each norm site carries a complete spec. The declared post_norm_eps
    // reaches the post sites as a distinct value, and Glimmer's centred
    // layer norms reach every site that needs them — while the final norm
    // keeps its own, uncentred, offset.
    let post = surface
        .norm
        .post
        .expect("four-norm stack judges its post sites");
    assert!((post.eps - 1e-8).abs() < 1e-20);
    assert_eq!(surface.norm.pre.weight_offset, 1.0);
    assert_eq!(post.weight_offset, 1.0);
    assert_eq!(surface.norm.final_norm.weight_offset, 0.0);
    assert!((surface.norm.final_norm.eps - surface.norm.pre.eps).abs() < 1e-20);
    let head = surface.head.as_ref().expect("target owns embedding + head");
    assert_eq!(head.vocab_size, 128);
    let output_multiplier = head.output_multiplier.expect("declared output_multiplier");
    assert!((output_multiplier - 0.196).abs() < 1e-12);

    // The drafter declares no vocab and owns no embedding/head objects —
    // its surface is complete *without* a head group.
    let draft = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "draft")
        .unwrap();
    assert!(draft.execution.as_ref().unwrap().head.is_none());
}

/// Perception surface: derived from the nested reading — head_dim from
/// hidden/heads, norm kind from the declared epsilon spelling, FFN gating
/// from tensor evidence (the fixture tower has no gate tensors).
#[test]
fn perception_surface_derives_from_nested_evidence() {
    let (_a, _b, named) = glimmer_pair();
    let built = build_from_inventories(&named);
    let vision = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "vision")
        .unwrap();
    let surface = vision.execution.as_ref().unwrap();
    let attention = surface.attention.as_ref().expect("the tower attends");
    let ffn = surface.ffn.as_ref().expect("the tower runs an FFN");
    assert_eq!(attention.num_q_heads, 4);
    assert_eq!(attention.head_dim, 8); // 32 / 4, derived
    assert_eq!(ffn.activation, Activation::Gelu);
    assert_eq!(ffn.ffn_type, FfnType::Standard);
    assert_eq!(surface.norm.pre.kind, NormType::LayerNorm); // layer_norm_eps spelling
    assert!(surface.head.is_none());
}

/// A missing source fact is a *recorded refusal*: strip the resolution's
/// execution block (a pre-v3 inventory) and the builder reports the
/// component instead of defaulting a surface into existence.
#[test]
fn missing_resolution_is_an_incomplete_surface_not_a_default() {
    let (_a, _b, mut named) = glimmer_pair();
    named[0].1.resolved.execution = None;
    let built = build_from_inventories(&named);
    let incomplete = built
        .incomplete_surfaces
        .iter()
        .find(|s| s.component == "target")
        .expect("target must be reported");
    assert!(incomplete.missing.iter().any(|m| m.contains("execution")));
    assert!(execution_completeness(&built.graph).iter().any(
        |d| matches!(d, CompletenessDefect::MissingSurface { component } if component == "target")
    ));
}

/// Per-layer head-geometry variation (Gemma 4's global layers) is
/// carried on the layer's policy, not averaged into the surface and not
/// refused: the surface builds with the component's declared geometry
/// and the varying layer's policy states its own.
#[test]
fn non_uniform_head_geometry_is_carried_per_layer() {
    let (_a, _b, mut named) = glimmer_pair();
    let varied_head_dim = 16;
    let varied_kv_heads = 1;
    let uniform_head_dim = named[0].1.resolved.layers[1].head_dim;
    let uniform_kv_heads = named[0].1.resolved.layers[1].num_kv_heads;
    named[0].1.resolved.layers[0].head_dim = varied_head_dim;
    named[0].1.resolved.layers[0].num_kv_heads = varied_kv_heads;
    let built = build_from_inventories(&named);
    assert!(
        !built
            .incomplete_surfaces
            .iter()
            .any(|s| s.component == "target"),
        "per-layer geometry must not refuse the surface: {:?}",
        built.incomplete_surfaces
    );
    let target = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    let table = target.attention.as_ref().expect("policy table");
    let g0 = table[0].geometry.expect("layer 0 geometry recorded");
    let g1 = table[1].geometry.expect("layer 1 geometry recorded");
    assert_eq!(
        (g0.head_dim, g0.num_kv_heads),
        (varied_head_dim, varied_kv_heads)
    );
    assert_eq!(
        (g1.head_dim, g1.num_kv_heads),
        (uniform_head_dim, uniform_kv_heads)
    );
    // The surface still states the component's declared geometry.
    let surface = target.execution.as_ref().expect("surface built");
    let attention = surface.attention.as_ref().expect("the program attends");
    assert_eq!(attention.head_dim, uniform_head_dim);
    assert_eq!(attention.num_kv_heads, uniform_kv_heads);
}

/// Every nested-derivation refusal names its missing fact — a bare
/// topology reports all of them at once, and an indivisible head split
/// names the geometry.
#[test]
fn nested_refusals_name_every_missing_fact() {
    use crate::format::vindex3::graph::surface::surface_from_nested;
    let bare = larql_models::inventory::components::ComponentTopology {
        name: "vision".to_string(),
        model_type: None,
        hidden_size: Some(32),
        intermediate_size: None,
        num_layers: Some(2),
        num_attention_heads: None,
        num_key_value_heads: None,
        head_dim: None,
        layer_types: None,
        norm_eps: None,
        norm_kind: None,
        hidden_act: None,
        rope_theta: None,
        rope_type: None,
        max_position_embeddings: None,
        patch: Default::default(),
        tower: Default::default(),
    };
    let missing = surface_from_nested(&bare, false).unwrap_err();
    for fact in [
        "num_attention_heads",
        "head_dim",
        "intermediate_size",
        "hidden_act",
        "norm_eps",
    ] {
        assert!(
            missing.iter().any(|m| m.contains(fact)),
            "`{fact}` not reported in {missing:?}"
        );
    }

    let indivisible = larql_models::inventory::components::ComponentTopology {
        num_attention_heads: Some(5),
        ..bare
    };
    let missing = surface_from_nested(&indivisible, false).unwrap_err();
    assert!(missing
        .iter()
        .any(|m| m.contains("not divisible by 5 heads")));
}

/// Head-surface refusals: a missing vocab and a pre-v3 inventory both
/// come back as facts, not defaults.
#[test]
fn head_surface_refusals_are_facts() {
    use crate::format::vindex3::graph::surface::head_from_resolved;
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory.resolved.vocab_size = None;
    assert!(head_from_resolved(&inventory)
        .unwrap_err()
        .contains(&"vocab_size".to_string()));
    inventory.resolved.execution = None;
    assert!(head_from_resolved(&inventory).unwrap_err()[0].contains("execution"));
}

/// Completeness defects print the component they indict.
#[test]
fn completeness_defects_display_the_component() {
    let surface = CompletenessDefect::MissingSurface {
        component: "target".to_string(),
    };
    assert!(surface.to_string().contains("`target`"));
    let head = CompletenessDefect::MissingHeadSurface {
        component: "draft".to_string(),
    };
    assert!(head.to_string().contains("`draft`"));
    assert!(head.to_string().contains("head group"));
    let norm = CompletenessDefect::MissingNormPlacement {
        component: "target".to_string(),
    };
    assert!(norm.to_string().contains("norm placement"));
    let operation = CompletenessDefect::MissingOperationSurface {
        component: "target".to_string(),
        operation: "attention",
    };
    assert!(operation.to_string().contains("`target`"));
    assert!(operation.to_string().contains("attention group"));
}

/// **Schema 6: the operation surfaces follow the program — each family
/// that runs must find its group present.** Stripping a group whose
/// operators are in the table is a completeness defect naming the
/// operation; a mixer-only program legitimately carries neither
/// attention nor FFN and is complete without them.
#[test]
fn a_program_that_runs_a_family_requires_its_surface_group() {
    let (_a, _b, named) = glimmer_pair();
    let mut built = build_from_inventories(&named);
    assert!(execution_completeness(&built.graph).is_empty());
    let strip = |built: &mut crate::format::vindex3::graph::BuiltGraph,
                 f: &dyn Fn(&mut crate::format::vindex3::graph::ExecutionSurface)| {
        let target = built
            .graph
            .components
            .iter_mut()
            .find(|c| c.id == "target")
            .unwrap();
        f(target.execution.as_mut().unwrap());
    };
    // The target attends and runs an FFN; each group is load-bearing.
    strip(&mut built, &|s| s.attention = None);
    let defects = execution_completeness(&built.graph);
    assert!(
        defects.iter().any(|d| matches!(
            d,
            CompletenessDefect::MissingOperationSurface {
                operation: "attention",
                ..
            }
        )),
        "{defects:?}"
    );
    let (_a, _b, named) = glimmer_pair();
    let mut built = build_from_inventories(&named);
    strip(&mut built, &|s| s.ffn = None);
    let defects = execution_completeness(&built.graph);
    assert!(
        defects.iter().any(|d| matches!(
            d,
            CompletenessDefect::MissingOperationSurface {
                operation: "ffn",
                ..
            }
        )),
        "{defects:?}"
    );
}

/// A component owning a head object whose surface lost its head group is
/// a completeness defect — the op-implication rule, exercised directly.
#[test]
fn head_objects_without_a_head_surface_are_a_defect() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let mut built = build_from_inventories(&named);
    assert!(execution_completeness(&built.graph).is_empty());
    let target = built
        .graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    target.execution.as_mut().unwrap().head = None;
    assert!(execution_completeness(&built.graph)
        .iter()
        .any(|d| matches!(d, CompletenessDefect::MissingHeadSurface { .. })));
}

/// A component owning a decoder stack whose surface lost its norm
/// placement is the sibling defect to the head one, and a distinct
/// variant: where the norms sit is not derivable from the fact that
/// norms exist, so a surface that does not say must be refused rather
/// than defaulted to pre- or post-norm.
#[test]
fn a_decoder_stack_without_a_norm_placement_is_a_defect() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let mut built = build_from_inventories(&named);
    assert!(execution_completeness(&built.graph).is_empty());
    let target = built
        .graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    target.execution.as_mut().unwrap().norm.placement = None;
    // The dense fixture's target owns only an Embedding, which implies the
    // head ops and not the stack ones. Give it a decoder stack so the norm
    // rule is the one under test — otherwise this passes on the head rule
    // and says nothing about norm placement.
    built.graph.objects.push(LogicalObject {
        id: "target.decoder_stack".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::DecoderStack,
        source_bindings: Vec::new(),
        representations: Vec::new(),
    });
    let defects = execution_completeness(&built.graph);
    assert!(
        defects.iter().any(
            |d| matches!(d, CompletenessDefect::MissingNormPlacement { component }
                if component == "target")
        ),
        "{defects:?}"
    );
    // The head surface is untouched, so the two defects must not travel
    // together — otherwise this test would pass on the head rule.
    assert!(!defects
        .iter()
        .any(|d| matches!(d, CompletenessDefect::MissingHeadSurface { .. })));
}

/// The norm-placement defect names its component and its own reason.
#[test]
fn the_norm_placement_defect_prints_its_own_diagnostic() {
    let norm = CompletenessDefect::MissingNormPlacement {
        component: "target".to_string(),
    };
    assert!(norm.to_string().contains("`target`"));
    assert!(norm.to_string().contains("norm placement"));
    // Distinct text from its sibling: a reader must be able to tell which
    // half of the surface is missing.
    let head = CompletenessDefect::MissingHeadSurface {
        component: "target".to_string(),
    };
    assert_ne!(norm.to_string(), head.to_string());
}

/// A component with no executable objects and no surface is complete —
/// the arm that says "nothing is implied here, so nothing is missing".
/// Without this the empty case would be indistinguishable from a defect
/// that simply was not detected.
#[test]
fn a_component_implying_no_ops_needs_no_surface() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let mut built = build_from_inventories(&named);
    let target = built
        .graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    target.execution = None;
    // Drop every object belonging to the component, so nothing implies a
    // stack or a head any more.
    built.graph.objects.retain(|o| o.component != "target");
    assert!(
        execution_completeness(&built.graph).is_empty(),
        "a component that implies no operations cannot be missing a surface"
    );
}

/// **The execution semantic is the authority, not the tokenizer's copy.**
///
/// `max_position_embeddings` says how far the programme is declared to
/// run and changes what a forward pass does. `model_max_length` in
/// `tokenizer_config.json` is a serving bound on a different component.
/// They usually agree, which is exactly why the disagreement is planted
/// here: a lowering that reached for the tokenizer's number would pass
/// every test where the two match, and be wrong on the one model where
/// they do not.
#[test]
fn context_length_comes_from_the_execution_semantic_not_the_tokenizer_bound() {
    let dir = tempfile::tempdir().unwrap();
    let inv = crate::format::vindex3::plan::tests_support::known_dense_with_config(
        dir.path(),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 262144
        }),
    );
    let surface = crate::format::vindex3::graph::surface::surface_from_resolved(&inv)
        .expect("a complete config yields a surface");
    assert_eq!(
        surface.context_length,
        Some(262144),
        "the graph records the judged execution semantic"
    );
}

/// A checkpoint that declares no extent records none. Absence is a fact
/// about the source, and inventing a default here would be the silent
/// substitution the whole surface exists to refuse.
#[test]
fn a_checkpoint_with_no_declared_extent_records_none() {
    let dir = tempfile::tempdir().unwrap();
    let inv = known_dense(dir.path());
    let surface =
        crate::format::vindex3::graph::surface::surface_from_resolved(&inv).expect("surface");
    assert_eq!(
        surface.context_length, None,
        "undeclared is None, never a chosen number"
    );
}

/// The field is additive within GRAPH_SCHEMA 6: a graph written before
/// it existed still parses, and one written with it round-trips.
#[test]
fn context_length_is_additive_within_schema_six() {
    use crate::format::vindex3::graph::surface::ExecutionSurface;
    let dir = tempfile::tempdir().unwrap();
    let inv = known_dense(dir.path());
    let mut surface = crate::format::vindex3::graph::surface::surface_from_resolved(&inv).unwrap();

    // A v6 graph written before the field existed: the key is simply
    // absent from the JSON.
    let json = serde_json::to_value(&surface).unwrap();
    assert!(
        json.get("context_length").is_none(),
        "None is skipped, so old and new writers agree byte-for-byte when it is absent"
    );
    let reparsed: ExecutionSurface = serde_json::from_value(json).unwrap();
    assert_eq!(reparsed, surface, "an old graph still reads");

    // And one that carries it round-trips unchanged.
    surface.context_length = Some(262144);
    let json = serde_json::to_value(&surface).unwrap();
    assert_eq!(json["context_length"], 262144);
    let reparsed: ExecutionSurface = serde_json::from_value(json).unwrap();
    assert_eq!(
        reparsed.context_length,
        Some(262144),
        "a new graph round-trips"
    );
}
