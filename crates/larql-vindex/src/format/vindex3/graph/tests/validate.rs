//! Structural validation + serde round trip.

use larql_models::config::PositionPolicy;

use crate::format::vindex3::graph::policy::{AttentionSpan, LayerOperator};
use crate::format::vindex3::graph::{
    AttentionLayerPolicy, Component, ComponentRole, GraphDefect, HiddenStateEdge, LogicalObject,
    ObjectKind, SourceBinding, SystemGraph, GRAPH_SCHEMA,
};

fn component(id: &str, role: ComponentRole, layers: usize) -> Component {
    Component {
        id: id.to_string(),
        role,
        source_artifact: "fixture".to_string(),
        num_layers: layers,
        hidden_size: 64,
        attention: Some(vec![
            AttentionLayerPolicy {
                operator: LayerOperator::Softmax,
                span: Some(AttentionSpan::Sliding),
                window: Some(16),
                position: PositionPolicy::Rope { theta: 10000.0 },
                geometry: None,
                v_from_k: false,
                declared_span: None,
            };
            layers
        ]),
        execution: None,
        perception: None,
    }
}

fn object(id: &str, component: &str, kind: ObjectKind) -> LogicalObject {
    LogicalObject {
        id: id.to_string(),
        component: component.to_string(),
        kind,
        source_bindings: vec![SourceBinding {
            artifact: "fixture".to_string(),
            tensor_prefix: "model.layers".to_string(),
            tensors: 4,
            bytes: 128,
        }],
        representations: Vec::new(),
    }
}

fn valid_graph() -> SystemGraph {
    SystemGraph {
        schema: GRAPH_SCHEMA,
        components: vec![
            component("target", ComponentRole::PrimaryText, 8),
            component("draft", ComponentRole::Drafter, 2),
        ],
        objects: vec![
            object("target.decoder_stack", "target", ObjectKind::DecoderStack),
            object(
                "draft.feature_projector",
                "draft",
                ObjectKind::FeatureProjector,
            ),
        ],
        edges: vec![HiddenStateEdge {
            producer_component: "target".to_string(),
            producer_layers: vec![1, 3, 5],
            consumer_component: "draft".to_string(),
            consumer_object: "draft.feature_projector".to_string(),
            block_size: Some(4),
        }],
    }
}

#[test]
fn a_coherent_graph_validates_clean() {
    assert!(valid_graph().validate().is_empty());
}

#[test]
fn duplicate_ids_are_defects() {
    let mut graph = valid_graph();
    graph
        .components
        .push(component("target", ComponentRole::PrimaryText, 8));
    graph.objects.push(object(
        "target.decoder_stack",
        "target",
        ObjectKind::DecoderStack,
    ));
    let defects = graph.validate();
    assert!(defects
        .iter()
        .any(|d| matches!(d, GraphDefect::DuplicateComponentId(id) if id == "target")));
    assert!(defects
        .iter()
        .any(|d| matches!(d, GraphDefect::DuplicateObjectId(id) if id == "target.decoder_stack")));
}

#[test]
fn orphaned_references_are_defects() {
    let mut graph = valid_graph();
    graph.objects.push(object(
        "ghost.embedding",
        "ghost_component",
        ObjectKind::Embedding,
    ));
    graph.edges.push(HiddenStateEdge {
        producer_component: "missing".to_string(),
        producer_layers: vec![1],
        consumer_component: "draft".to_string(),
        consumer_object: "not_an_object".to_string(),
        block_size: None,
    });
    let defects = graph.validate();
    assert!(defects
        .iter()
        .any(|d| matches!(d, GraphDefect::ObjectOrphaned { component, .. } if component == "ghost_component")));
    assert!(defects
        .iter()
        .any(|d| matches!(d, GraphDefect::EdgeOrphaned { detail } if detail.contains("missing"))));
    assert!(defects.iter().any(
        |d| matches!(d, GraphDefect::EdgeOrphaned { detail } if detail.contains("not_an_object"))
    ));
}

#[test]
fn an_unbound_object_is_a_defect() {
    let mut graph = valid_graph();
    graph.objects[0].source_bindings.clear();
    assert!(graph
        .validate()
        .iter()
        .any(|d| matches!(d, GraphDefect::ObjectUnbound(id) if id == "target.decoder_stack")));
}

#[test]
fn graph_round_trips_through_json_with_nope_policies() {
    let mut graph = valid_graph();
    // A NoPE layer must survive serialisation as a policy variant.
    graph.components[0].attention.as_mut().unwrap()[3] = AttentionLayerPolicy {
        operator: LayerOperator::Softmax,
        span: Some(AttentionSpan::Full),
        window: None,
        position: PositionPolicy::None,
        geometry: None,
        v_from_k: false,
        declared_span: None,
    };
    let json = serde_json::to_string_pretty(&graph).unwrap();
    let back: SystemGraph = serde_json::from_str(&json).unwrap();
    assert!(back.validate().is_empty());
    assert_eq!(
        back.components[0].position_policy(3),
        Some(PositionPolicy::None)
    );
    assert_eq!(back.edges, graph.edges);
}

/// Drill finding F10: "the text model" is answered by uniqueness, never
/// by first match — ambiguity names the candidates and refuses.
#[test]
fn primary_text_resolution_is_unique_or_refused() {
    use crate::format::vindex3::graph::PrimaryTextLookup;

    // Exactly one primary: resolved.
    let graph = valid_graph();
    assert_eq!(
        graph.primary_text_component().unwrap().id,
        "target",
        "a unique primary_text component resolves"
    );

    // None: absent, distinctly from ambiguous (encode's drafter-only
    // fallback depends on the distinction).
    let none = SystemGraph {
        schema: GRAPH_SCHEMA,
        components: vec![component("draft", ComponentRole::Drafter, 2)],
        objects: vec![],
        edges: vec![],
    };
    assert_eq!(
        none.primary_text_component().unwrap_err(),
        PrimaryTextLookup::Absent
    );

    // Two: refused, naming both candidates — never first-wins.
    let two = SystemGraph {
        schema: GRAPH_SCHEMA,
        components: vec![
            component("target", ComponentRole::PrimaryText, 8),
            component("second_text", ComponentRole::PrimaryText, 4),
        ],
        objects: vec![],
        edges: vec![],
    };
    let err = two.primary_text_component().unwrap_err();
    assert_eq!(
        err,
        PrimaryTextLookup::Ambiguous(vec!["target".to_string(), "second_text".to_string()])
    );
    let message = err.to_string();
    assert!(
        message.contains("exactly one")
            && message.contains("target")
            && message.contains("second_text"),
        "the refusal names the candidates: {message}"
    );
}
