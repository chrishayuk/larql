//! The pairwise semantic links of the G4 authority chain.
//!
//! Three links, each producing itemised [`EquivalenceCheck`]s:
//!
//! 1. **Declared ≡ Resolved** — the plan's value comparison, lifted into
//!    check rows (plus the admissibility gate itself: a container whose
//!    source no longer plans admissible could not be re-encoded today).
//! 2. **Resolved ≡ Graph** — the graph is *rebuilt from the source now*
//!    and compared against the resolved topology field by field. This is
//!    deliberately an independent re-derivation, not a call into the
//!    builder's own mapping, so a builder bug shows up as a broken link
//!    instead of agreeing with itself.
//! 3. **Graph ≡ Encoded** — the rebuilt graph against the graph the
//!    container persisted, element by element (components, objects,
//!    edges), so drift on either side names the element that moved.

use larql_models::inventory::ArchitectureInventory;
use serde_json::{json, Value};

use super::{Authority, EquivalenceCheck};
use crate::format::vindex3::graph::{Component, ComponentRole, SystemGraph};
use crate::format::vindex3::plan::{FindingCategory, SystemPlan};

/// Span spellings the inventory's resolved table uses.
const ATTENTION_SLIDING: &str = "sliding";
const ATTENTION_FULL: &str = "full";

/// Link 1: the plan's declared-vs-resolved value comparison, as check rows.
pub fn declared_vs_resolved(plan: &SystemPlan) -> Vec<EquivalenceCheck> {
    let mut checks = vec![EquivalenceCheck {
        left: Authority::Declared,
        right: Authority::Resolved,
        subject: "plan.admissible".to_string(),
        left_value: json!(plan.summary.blocking),
        right_value: json!(0),
        pass: plan.admissible,
        detail: if plan.admissible {
            "the encode gate still holds against today's source".to_string()
        } else {
            format!(
                "{} blocking finding(s) — the source no longer plans admissible",
                plan.summary.blocking
            )
        },
    }];
    for artifact in &plan.artifacts {
        for finding in artifact.findings.iter().filter(|f| {
            matches!(
                f.category,
                FindingCategory::Representable | FindingCategory::Mismatched
            ) && f.declared.is_some()
        }) {
            checks.push(EquivalenceCheck {
                left: Authority::Declared,
                right: Authority::Resolved,
                subject: format!("{}:{}", artifact.name, finding.subject),
                left_value: finding.declared.clone().unwrap_or(Value::Null),
                right_value: finding.resolved.clone().unwrap_or(Value::Null),
                pass: finding.category == FindingCategory::Representable,
                detail: finding.detail.clone(),
            });
        }
    }
    checks
}

/// Link 2: the rebuilt graph against the resolved topology it claims to
/// carry, one topology row and one attention row per component.
pub fn resolved_vs_graph(
    named: &[(String, ArchitectureInventory)],
    graph: &SystemGraph,
) -> Vec<EquivalenceCheck> {
    let mut checks = Vec::new();
    for component in &graph.components {
        let Some((_, inventory)) = named.iter().find(|(n, _)| *n == component.source_artifact)
        else {
            checks.push(EquivalenceCheck {
                left: Authority::Resolved,
                right: Authority::Graph,
                subject: format!("{}.source_artifact", component.id),
                left_value: Value::Null,
                right_value: json!(component.source_artifact),
                pass: false,
                detail: "component sourced from an artifact not in the verify set".to_string(),
            });
            continue;
        };
        if component.role == ComponentRole::Perception {
            checks.push(perception_topology(component, inventory));
        } else {
            checks.push(text_topology(component, inventory));
            checks.push(attention_policy(component, inventory));
        }
        checks.push(execution_surface(component, inventory, graph));
    }
    checks
}

/// The execution surface, re-derived from today's inventory and compared
/// against what the graph carries (V3-G5a). Re-derivation goes through the
/// same judged mapping the builder uses — the independence this link buys
/// is over the *inputs* (today's checkpoint resolution vs the persisted
/// component), and a mapping bug is G5b's behavioural parity to catch.
fn execution_surface(
    component: &Component,
    inventory: &ArchitectureInventory,
    graph: &SystemGraph,
) -> EquivalenceCheck {
    use crate::format::vindex3::graph::surface;
    use crate::format::vindex3::graph::ObjectKind;

    let rederived = match component.role {
        ComponentRole::Perception => {
            let nested = inventory.nested_components.iter().find(|n| {
                component.id == n.name || component.id.starts_with(&format!("{}_", n.name))
            });
            nested
                .ok_or_else(|| vec!["nested component reading".to_string()])
                .and_then(|nested| {
                    let has_gate = graph.objects.iter().any(|o| {
                        o.component == component.id
                            && o.kind == ObjectKind::PerceptionTower
                            && surface::gate_evidence(inventory, o)
                    });
                    surface::surface_from_nested(nested, has_gate)
                })
        }
        _ => surface::surface_from_resolved(inventory).and_then(|mut built| {
            let owns_head = graph.objects.iter().any(|o| {
                o.component == component.id
                    && matches!(o.kind, ObjectKind::Embedding | ObjectKind::OutputHead)
            });
            if owns_head {
                built.head = Some(surface::head_from_resolved(inventory)?);
            }
            if let Some(stack) = graph
                .objects
                .iter()
                .find(|o| o.component == component.id && o.kind == ObjectKind::DecoderStack)
            {
                surface::attach_stack_evidence(&mut built, inventory, stack)?;
            }
            Ok(built)
        }),
    };
    let (left, detail_left) = match rederived {
        Ok(surface) => (serde_json::to_value(&surface).unwrap_or(Value::Null), None),
        Err(missing) => (Value::Null, Some(missing.join(", "))),
    };
    let right = component
        .execution
        .as_ref()
        .and_then(|s| serde_json::to_value(s).ok())
        .unwrap_or(Value::Null);
    let pass = left != Value::Null && left == right;
    EquivalenceCheck {
        left: Authority::Resolved,
        right: Authority::Graph,
        subject: format!("{}.execution_surface", component.id),
        detail: match detail_left {
            Some(missing) => {
                format!("surface not derivable from today's source — missing: {missing}")
            }
            None if pass => "execution surface carried into the graph".to_string(),
            None => "graph surface differs from today's re-derivation".to_string(),
        },
        left_value: left,
        right_value: right,
        pass,
    }
}

/// Topology of a text-path component vs the inventory's resolution.
fn text_topology(component: &Component, inventory: &ArchitectureInventory) -> EquivalenceCheck {
    let resolved = json!({
        "num_layers": inventory.resolved.num_layers,
        "hidden_size": inventory.resolved.hidden_size,
    });
    let graph = json!({
        "num_layers": component.num_layers,
        "hidden_size": component.hidden_size,
    });
    EquivalenceCheck {
        left: Authority::Resolved,
        right: Authority::Graph,
        subject: format!("{}.topology", component.id),
        pass: resolved == graph,
        detail: "depth and width carried into the graph".to_string(),
        left_value: resolved,
        right_value: graph,
    }
}

/// Topology of a perception component vs its nested component reading.
fn perception_topology(
    component: &Component,
    inventory: &ArchitectureInventory,
) -> EquivalenceCheck {
    let nested = inventory
        .nested_components
        .iter()
        .find(|n| component.id == n.name || component.id.starts_with(&format!("{}_", n.name)));
    let Some(nested) = nested else {
        return EquivalenceCheck {
            left: Authority::Resolved,
            right: Authority::Graph,
            subject: format!("{}.topology", component.id),
            left_value: Value::Null,
            right_value: json!(component.id),
            pass: false,
            detail: "perception component has no matching nested config reading".to_string(),
        };
    };
    let resolved = json!({
        "num_layers": nested.num_layers.unwrap_or(0),
        "hidden_size": nested.hidden_size.unwrap_or(0),
    });
    let graph = json!({
        "num_layers": component.num_layers,
        "hidden_size": component.hidden_size,
    });
    EquivalenceCheck {
        left: Authority::Resolved,
        right: Authority::Graph,
        subject: format!("{}.topology", component.id),
        pass: resolved == graph,
        detail: "perception depth and width carried into the graph".to_string(),
        left_value: resolved,
        right_value: graph,
    }
}

/// Per-layer attention policy: span, window and position of every layer,
/// re-derived from the inventory's resolved table and compared against the
/// graph's persisted table.
fn attention_policy(component: &Component, inventory: &ArchitectureInventory) -> EquivalenceCheck {
    let resolved: Vec<Value> = inventory
        .resolved
        .layers
        .iter()
        .map(|layer| {
            // The checkpoint's own declared kind is the authority when it
            // states one. `layer.attention` is the parser's sliding/full
            // boolean, and a boolean **cannot express a recurrence** — so
            // comparing the graph against it would demand the graph
            // re-adopt the collapse that reported Kimi Linear's 20 KDA
            // layers, and Inkling-Small's 35 sliding ones, as full
            // attention.
            //
            // This stays a real check rather than becoming a tautology:
            // `LayerPolicy::declared_kind` and the graph's
            // `AttentionLayerPolicy` are two separately stored
            // resolutions, written at different times by different code,
            // and either can drift from the other. What changed is which
            // fact is compared, not whether one is.
            let span = match &layer.declared_kind {
                // A recurrence has no span, and saying so is the point.
                Some(larql_models::config::LayerKind::Recurrent(_)) => Value::Null,
                Some(larql_models::config::LayerKind::Sliding { .. }) => json!(ATTENTION_SLIDING),
                Some(larql_models::config::LayerKind::Full) => json!(ATTENTION_FULL),
                // A spelling with no kind, or no declaration at all: the
                // boolean is the only source of truth there is.
                Some(larql_models::config::LayerKind::Unexpressed { .. }) | None => {
                    json!(layer.attention)
                }
            };
            json!({
                "span": span,
                "window": layer.window,
                "position": layer.position,
            })
        })
        .collect();
    let graph: Vec<Value> = component
        .attention
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|policy| {
            json!({
                "span": policy.span,
                "window": policy.window,
                "position": policy.position,
            })
        })
        .collect();
    let first_mismatch = resolved
        .iter()
        .zip(&graph)
        .position(|(a, b)| a != b)
        .or((resolved.len() != graph.len()).then_some(resolved.len().min(graph.len())));
    EquivalenceCheck {
        left: Authority::Resolved,
        right: Authority::Graph,
        subject: format!("{}.attention_policy", component.id),
        pass: first_mismatch.is_none(),
        detail: match first_mismatch {
            None => format!("{} layers agree (span, window, position)", resolved.len()),
            Some(layer) => format!(
                "first disagreement at layer {layer} ({} resolved vs {} graph layers)",
                resolved.len(),
                graph.len()
            ),
        },
        left_value: Value::Array(resolved),
        right_value: Value::Array(graph),
    }
}

/// Link 3: the rebuilt graph against the container's persisted graph —
/// schema, then every component, object and the edge list by id.
pub fn graph_vs_encoded(rebuilt: &SystemGraph, encoded: &SystemGraph) -> Vec<EquivalenceCheck> {
    let mut checks = vec![element_check(
        "schema".to_string(),
        Some(&rebuilt.schema),
        Some(&encoded.schema),
    )];
    for id in id_union(
        rebuilt.components.iter().map(|c| c.id.as_str()),
        encoded.components.iter().map(|c| c.id.as_str()),
    ) {
        checks.push(element_check(
            format!("component {id}"),
            rebuilt.components.iter().find(|c| c.id == id),
            encoded.components.iter().find(|c| c.id == id),
        ));
    }
    for id in id_union(
        rebuilt.objects.iter().map(|o| o.id.as_str()),
        encoded.objects.iter().map(|o| o.id.as_str()),
    ) {
        checks.push(element_check(
            format!("object {id}"),
            rebuilt.objects.iter().find(|o| o.id == id),
            encoded.objects.iter().find(|o| o.id == id),
        ));
    }
    checks.push(element_check(
        "edges".to_string(),
        Some(&rebuilt.edges),
        Some(&encoded.edges),
    ));
    checks
}

/// Sorted union of the ids on both sides, so an element missing from
/// either authority still gets a row.
fn id_union<'a>(
    left: impl Iterator<Item = &'a str>,
    right: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let mut ids: Vec<String> = left.chain(right).map(str::to_string).collect();
    ids.sort();
    ids.dedup();
    ids
}

/// One Graph ≡ Encoded row: serialised-form equality, with absence on
/// either side reported as `null` rather than skipped.
fn element_check<T: serde::Serialize>(
    subject: String,
    rebuilt: Option<T>,
    encoded: Option<T>,
) -> EquivalenceCheck {
    let left = rebuilt
        .map(|v| serde_json::to_value(v).unwrap_or(Value::Null))
        .unwrap_or(Value::Null);
    let right = encoded
        .map(|v| serde_json::to_value(v).unwrap_or(Value::Null))
        .unwrap_or(Value::Null);
    let pass = left != Value::Null && right != Value::Null && left == right;
    EquivalenceCheck {
        left: Authority::Graph,
        right: Authority::Encoded,
        detail: if pass {
            "rebuilt and persisted forms identical".to_string()
        } else if left == Value::Null {
            format!("`{subject}` exists only in the container")
        } else if right == Value::Null {
            format!("`{subject}` exists only in the rebuilt graph")
        } else {
            format!("`{subject}` differs between the rebuilt and persisted graph")
        },
        subject,
        left_value: left,
        right_value: right,
        pass,
    }
}
