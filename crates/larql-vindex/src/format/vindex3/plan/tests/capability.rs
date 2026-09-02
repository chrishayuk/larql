//! Capability-scoped admissibility and availability (CAP-0/CAP-1/CAP-2)
//! and graph-backed modality (PERCEPTION-2).
//!
//! Two load-bearing controls here:
//!
//! * [`the_binding_edge_is_not_the_tower`] — the one a
//!   `component == "vision"` implementation fails, taken from a real
//!   checkpoint.
//! * [`a_component_id_is_an_identifier_and_nothing_more`] — the one a
//!   `id.contains("vision")` implementation fails. It is deliberately
//!   adversarial: a component *named* `vision` that perceives audio.

use crate::format::vindex3::graph::{
    Component, ComponentRole, EncoderGeometry, LogicalObject, Modality, ObjectKind,
    PerceptionComponent, PerceptionTransform, ProjectionGeometry, SourceBinding, SystemGraph,
    GRAPH_SCHEMA,
};
use crate::format::vindex3::plan::capability::{
    admissible_for, available_for, requires, supported, Capability,
};
use crate::format::vindex3::plan::report::{
    Finding, FindingCategory, PlannedFinding, SemanticClass,
};

/// A blocking finding, as the document would carry it.
///
/// Ids are distinct per fixture because a closure now reports *which*
/// findings block: two fixtures sharing an id would let a wrong-set bug
/// pass as a right-count one.
fn blocking(component: &str, subject: &str) -> PlannedFinding {
    PlannedFinding::assign(
        next_id(),
        raw(FindingCategory::Unrepresented, component, subject),
    )
}

fn representable(component: &str, subject: &str) -> PlannedFinding {
    PlannedFinding::assign(
        next_id(),
        raw(FindingCategory::Representable, component, subject),
    )
}

fn raw(category: FindingCategory, component: &str, subject: &str) -> Finding {
    Finding {
        category,
        class: SemanticClass::ExecutionSemantic,
        component: component.to_string(),
        subject: subject.to_string(),
        declared: None,
        resolved: None,
        carriage: None,
        detail: "test fixture".into(),
    }
}

/// Monotonic fixture ids, so no two findings in one test share one.
fn next_id() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn perception(id: &str, modality: Modality, encoder: bool) -> Component {
    Component {
        id: id.to_string(),
        role: ComponentRole::Perception,
        source_artifact: "art".into(),
        num_layers: 0,
        hidden_size: 0,
        attention: None,
        execution: None,
        perception: Some(PerceptionComponent {
            modality,
            transform: if encoder {
                PerceptionTransform::Encoder(EncoderGeometry {
                    depth: Some(27),
                    width: Some(1152),
                    num_heads: Some(16),
                })
            } else {
                PerceptionTransform::DirectProjection(ProjectionGeometry::default())
            },
        }),
    }
}

fn language() -> Component {
    Component {
        id: "target".into(),
        role: ComponentRole::PrimaryText,
        source_artifact: "art".into(),
        num_layers: 30,
        hidden_size: 2816,
        attention: None,
        execution: None,
        perception: None,
    }
}

/// An object with real tensors behind it.
fn backed(component: &str) -> LogicalObject {
    LogicalObject {
        id: format!("{component}.perception_tower"),
        component: component.to_string(),
        kind: ObjectKind::PerceptionTower,
        source_bindings: vec![SourceBinding {
            artifact: "art".into(),
            tensor_prefix: format!("model.{component}"),
            tensors: 4,
            bytes: 1024,
        }],
        representations: Vec::new(),
    }
}

fn graph(components: Vec<Component>, objects: Vec<LogicalObject>) -> SystemGraph {
    SystemGraph {
        schema: GRAPH_SCHEMA,
        components,
        objects,
        edges: Vec::new(),
    }
}

/// A graph shaped like Gemma 4 26B-A4B: a language model and one image
/// encoder with weights. No audio component at all.
fn gemma_like() -> SystemGraph {
    graph(
        vec![language(), perception("vision", Modality::Image, true)],
        vec![backed("vision")],
    )
}

#[test]
fn language_subjects_are_required_by_every_capability() {
    let g = gemma_like();
    let f = blocking("text", "text_config.layer_types");
    for capability in Capability::ALL {
        assert!(requires(capability, &f, &g));
    }
}

#[test]
fn a_perception_component_is_reached_only_by_its_own_modality() {
    let g = gemma_like();
    let f = blocking("vision", "vision_config.num_heads");
    assert!(!requires(Capability::TextGeneration, &f, &g));
    assert!(requires(Capability::ImageConditioned, &f, &g));
    assert!(!requires(Capability::AudioConditioned, &f, &g));
}

/// CAP-1's control: `vision_start_token_id` and `vision_end_token_id` are
/// real Qwen3.8-27B findings that **block** and live at `component: "root"`.
/// They are the image→language binding edge, not the tower. A component
/// filter counts them as root-level facts everything needs, and so keeps
/// text generation blocked for a reason text can never reach — worth
/// exactly two blocking findings on the real checkpoint (18, not 20).
#[test]
fn the_binding_edge_is_not_the_tower() {
    let g = gemma_like();
    for subject in ["vision_start_token_id", "vision_end_token_id"] {
        let f = blocking("root", subject);
        assert!(f.blocks(), "the premise of the control");
        assert!(
            !requires(Capability::TextGeneration, &f, &g),
            "{subject} lives at root but binds an image"
        );
        assert!(requires(Capability::ImageConditioned, &f, &g));
    }
}

/// PERCEPTION-2's control. The component id carries no meaning: a component
/// *named* `vision` that declares audio is audio, and one named `camera`
/// that declares image is image. Any implementation matching
/// `id.contains("vision")` fails the first half.
#[test]
fn a_component_id_is_an_identifier_and_nothing_more() {
    let misleading = graph(
        vec![
            language(),
            perception("vision", Modality::Audio, false),
            perception("camera", Modality::Image, true),
        ],
        vec![backed("vision"), backed("camera")],
    );

    let f = blocking("vision", "some_component_owned_fact");
    assert!(
        requires(Capability::AudioConditioned, &f, &misleading),
        "a component named `vision` that declares audio is audio"
    );
    assert!(!requires(Capability::ImageConditioned, &f, &misleading));

    let f = blocking("camera", "some_component_owned_fact");
    assert!(
        requires(Capability::ImageConditioned, &f, &misleading),
        "a component named `camera` that declares image is image"
    );
    assert!(!requires(Capability::AudioConditioned, &f, &misleading));
}

/// Renaming perception components must not move any capability result.
#[test]
fn renaming_a_component_changes_nothing() {
    let named = graph(
        vec![language(), perception("vision", Modality::Image, true)],
        vec![backed("vision")],
    );
    let renamed = graph(
        vec![
            language(),
            perception("perceptron_7", Modality::Image, true),
        ],
        vec![backed("perceptron_7")],
    );
    let a = blocking("vision", "owned_fact");
    let b = blocking("perceptron_7", "owned_fact");
    for capability in Capability::ALL {
        assert_eq!(
            requires(capability, &a, &named),
            requires(capability, &b, &renamed),
            "{capability:?} moved when the component was renamed"
        );
        assert_eq!(
            available_for(capability, &named),
            available_for(capability, &renamed)
        );
    }
}

/// Fail closed twice over: an unclassified subject, and a component the
/// graph has never heard of. Neither may be guessed from its spelling.
#[test]
fn an_unknown_component_fails_closed_rather_than_guessing() {
    let g = gemma_like();
    // `mtp.fc` used to stand here as the componentless-unknown example.
    // QW-3.5D gave the MTP namespace a real classification, so it is no
    // longer unclassified and no longer belongs in this test — see
    // `the_mtp_head_is_required_by_drafting_alone`. Replaced with a
    // namespace nothing has judged, which is what this gate is about.
    for (component, subject) in [("", "adapter.fc"), ("vision_ghost", "vision_ghost.thing")] {
        let f = blocking(component, subject);
        for capability in Capability::ALL {
            assert!(
                requires(capability, &f, &g),
                "{component}/{subject} must fail closed for {capability:?}"
            );
        }
    }
}

#[test]
fn a_capability_is_judged_only_on_its_own_closure() {
    let g = gemma_like();
    let findings = vec![
        representable("text", "text_config.hidden_size"),
        blocking("vision", "vision_config.num_heads"),
        blocking("root", "vision_start_token_id"),
    ];

    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert!(text.admissible);
    assert_eq!(text.blocking, 0);

    let image = admissible_for(Capability::ImageConditioned, &findings, &g);
    assert!(!image.admissible);
    assert_eq!(
        image.blocking, 2,
        "the tower AND the binding edge, not just the tower"
    );
}

/// CAP-2's control, from Gemma 4 26B-A4B: it declares `audio_config` and
/// `audio_token_id`, every audio finding is representable, and it ships
/// **zero** audio tensors.
///
/// Understanding a modality and possessing its operands are independent.
/// Folding them together would report the container as not *understanding*
/// audio, when what it lacks is weights — and would send a reader to the
/// parser instead of to the checkpoint.
#[test]
fn understood_is_not_the_same_question_as_present() {
    let g = gemma_like();
    let findings = vec![representable("audio", "audio_config")];

    let audio = admissible_for(Capability::AudioConditioned, &findings, &g);
    assert!(
        audio.admissible,
        "nothing about audio blocks — its semantics are understood"
    );
    assert!(
        !audio.available,
        "and yet the checkpoint ships no audio component at all"
    );

    let image = admissible_for(Capability::ImageConditioned, &findings, &g);
    assert!(image.admissible && image.available);
    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert!(text.admissible && text.available);
}

/// A perception component that exists but is backed by no tensors is not
/// available either — declaring a modality is not shipping one.
#[test]
fn a_component_without_operands_is_not_available() {
    let g = graph(
        vec![language(), perception("audio", Modality::Audio, false)],
        Vec::new(),
    );
    assert!(!available_for(Capability::AudioConditioned, &g));
    assert!(available_for(Capability::TextGeneration, &g));
}

/// Both species answer availability the same way — a direct projection is
/// as present as an encoder when its tensors are there (Gemma 4 12B).
#[test]
fn both_perception_species_can_be_available() {
    let g = graph(
        vec![
            language(),
            perception("vision", Modality::Image, false),
            perception("audio", Modality::Audio, false),
        ],
        vec![backed("vision"), backed("audio")],
    );
    assert!(available_for(Capability::ImageConditioned, &g));
    assert!(available_for(Capability::AudioConditioned, &g));
}

/// The whole-model invariant is not weakened.
#[test]
fn capability_admissibility_does_not_imply_model_completeness() {
    let g = gemma_like();
    let findings = vec![blocking("vision", "vision_config.num_heads")];
    assert!(admissible_for(Capability::TextGeneration, &findings, &g).admissible);
    assert!(findings.iter().any(|f| f.blocks()));
}

/// EXEC-1: three independent verdicts, and the state that only exists
/// because they are independent.
///
/// Gemma 4 26B-A4B carries all three in one real container:
///
/// ```text
/// text   understood + present + runnable
/// image  understood + present + NOT runnable here
/// audio  understood + absent
/// ```
///
/// `image` is the one that matters. Without a separate `executable`, "the
/// vision weights were never touched during a text request" is ambiguous
/// between *the planner correctly did not select them* and *nothing could
/// have run them anyway*. Only the first is a claim about the architecture,
/// and today it is the second that is true: there is no perception executor
/// on this path.
#[test]
fn the_three_verdicts_are_independent() {
    let g = gemma_like();
    let findings = vec![representable("audio", "audio_config")];

    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert!(text.admissible && text.available && text.supported);
    assert!(text.runnable());

    let image = admissible_for(Capability::ImageConditioned, &findings, &g);
    assert!(
        image.admissible && image.available,
        "the tower is understood and its weights are here"
    );
    assert!(
        !image.supported,
        "and this build still has no perception executor"
    );
    assert!(!image.runnable());

    let audio = admissible_for(Capability::AudioConditioned, &findings, &g);
    assert!(audio.admissible);
    assert!(!audio.available && !audio.supported && !audio.runnable());
}

/// Absent operands are never runnable, however well the build supports the
/// capability — and `supported` still reports the build's own answer.
#[test]
fn what_is_not_present_is_not_runnable() {
    let g = graph(vec![language()], Vec::new());
    let findings: Vec<PlannedFinding> = Vec::new();
    assert!(admissible_for(Capability::TextGeneration, &findings, &g).runnable());
    for capability in [Capability::ImageConditioned, Capability::AudioConditioned] {
        assert!(!available_for(capability, &g));
        assert!(!admissible_for(capability, &findings, &g).runnable());
    }
}

/// `supported` asks nothing of any checkpoint: it is the build's answer,
/// identical across every container.
#[test]
fn support_is_a_property_of_the_build_not_the_checkpoint() {
    assert!(supported(Capability::TextGeneration));
    assert!(!supported(Capability::ImageConditioned));
    assert!(!supported(Capability::AudioConditioned));
}

/// The state that motivated the rename: Qwen3.8 text has the weights and a
/// decoder executor, and is still not runnable because its semantics are
/// unresolved. Reporting that as `executable = true` would read as "safe to
/// run", which it is not.
#[test]
fn present_and_supported_is_still_not_runnable_while_semantics_are_open() {
    let g = gemma_like();
    let findings = vec![blocking("text", "text_config.layer_types")];
    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert!(!text.admissible);
    assert!(text.available && text.supported);
    assert!(!text.runnable());
}

// ── QW-3.5D: capability relevance as a separate judgement ────────────

fn declared_bool(component: &str, subject: &str, value: bool) -> PlannedFinding {
    PlannedFinding::assign(
        next_id(),
        Finding {
            declared: Some(serde_json::json!(value)),
            ..raw(FindingCategory::Unrepresented, component, subject)
        },
    )
}

/// **D1.** The draft head is required by `Drafting` and by nothing else.
///
/// Both of its spellings: the `text_config.mtp_*` config keys and the
/// `mtp.*` tensor namespace, which carries no component at all.
#[test]
fn the_mtp_head_is_required_by_drafting_alone() {
    let g = graph(vec![language()], Vec::new());
    for subject in [
        "text_config.mtp_num_hidden_layers",
        "text_config.mtp_use_dedicated_embeddings",
        "mtp.fc",
        "mtp.layers",
        "mtp.pre_fc_norm_embedding",
    ] {
        let component = if subject.starts_with("mtp.") {
            ""
        } else {
            "text"
        };
        let f = blocking(component, subject);
        assert!(requires(Capability::Drafting, &f, &g), "{subject}");
        for other in [
            Capability::TextGeneration,
            Capability::ImageConditioned,
            Capability::AudioConditioned,
        ] {
            assert!(
                !requires(other, &f, &g),
                "{subject} must not gate {other:?} — an optional draft head is not a \
                 prerequisite for base decode"
            );
        }
    }
}

/// **D2.** `language_model_only` is excluded only while the graph
/// corroborates it as a composition statement.
///
/// `false` beside a real perception component agrees. A checkpoint
/// claiming `true` while shipping one is contradicting itself, and keeps
/// blocking.
#[test]
fn language_model_only_is_excluded_only_when_the_graph_agrees() {
    let multimodal = gemma_like();
    let text_only = graph(vec![language()], Vec::new());

    let says_multimodal = declared_bool("root", "language_model_only", false);
    let says_text_only = declared_bool("root", "language_model_only", true);

    assert!(!requires(
        Capability::TextGeneration,
        &says_multimodal,
        &multimodal
    ));
    assert!(!requires(
        Capability::TextGeneration,
        &says_text_only,
        &text_only
    ));
    // Contradictions both ways.
    assert!(
        requires(Capability::TextGeneration, &says_text_only, &multimodal),
        "`true` beside a perception component is a contradiction, not a disposition"
    );
    assert!(
        requires(Capability::TextGeneration, &says_multimodal, &text_only),
        "`false` with nothing to be multimodal about is a contradiction too"
    );
}

#[test]
fn a_closure_names_its_blockers_and_not_merely_how_many() {
    // Why the count was not enough: two checkpoints can report "6 blocking"
    // and share no concept at all. Leverage — "retiring idea X clears N
    // rows" — needs the SET, and computing it from whole-model findings
    // instead made every estimate an upper bound.
    let g = graph(
        vec![language(), perception("vis", Modality::Image, true)],
        Vec::new(),
    );
    let text_blocker = blocking("text", "text_config.layer_types");
    let vision_blocker = blocking("vis", "vision_config.num_heads");
    let fine = representable("text", "text_config.hidden_size");
    let text_id = text_blocker.id;
    let vision_id = vision_blocker.id;
    let findings = vec![text_blocker, vision_blocker, fine];

    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert_eq!(
        text.blocker_ids,
        vec![text_id],
        "text names its own blocker"
    );
    assert_eq!(text.blocking, text.blocker_ids.len(), "count and set agree");

    let image = admissible_for(Capability::ImageConditioned, &findings, &g);
    assert!(
        image.blocker_ids.contains(&vision_id),
        "the vision blocker belongs to the image closure"
    );
    // The discriminating arm: the two closures must not report the same
    // set. A `blocker_ids` that simply echoed every blocking finding would
    // pass every assertion above.
    assert_ne!(text.blocker_ids, image.blocker_ids);
    assert!(!text.blocker_ids.contains(&vision_id));
}

#[test]
fn an_admissible_closure_names_no_blockers() {
    let g = graph(vec![language()], Vec::new());
    let findings = vec![representable("text", "text_config.hidden_size")];
    let text = admissible_for(Capability::TextGeneration, &findings, &g);
    assert!(text.admissible);
    assert!(text.blocker_ids.is_empty());
}
