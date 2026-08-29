//! Semantic representability plan over architecture inventories (V3-G1/G2).
//!
//! Consumes the G0 inventory (`larql inspect-hf`) for one or more artifacts
//! treated as a model system, and answers: **can the VINDEX3 schema
//! faithfully describe this system — and if not, exactly why not?**
//!
//! Since G2, "representable" has one definition: **the system-graph builder
//! placed it** ([`super::graph::build_from_inventories`]). Objects the
//! builder placed are representable with their graph ids as proof; groups
//! it could not place, and interfaces it could not resolve, come back as
//! blocking findings. There is no separate capability table to drift out of
//! sync with the schema.
//!
//! The other finding sources:
//!
//! - `mismatched` — declared-vs-resolved value comparison (`consumed` is
//!   never trusted; values are compared);
//! - **every** declared config key, graded by semantic class, where a key
//!   nobody has judged (`unknown`) blocks. The census covers consumed,
//!   metadata and unconsumed keys alike, so `unrepresented: N` is a count
//!   against a stated denominator rather than a lower bound.
//! - **carriage** — for execution-semantic keys, how far VINDEX3 actually
//!   carries the fact past the parser ([`carriage`]). This is the axis
//!   that keeps `consumed` from being misread as `represented`: a key the
//!   parser reads and the schema then drops used to produce no finding at
//!   all, which is how GPT-OSS's YaRN scaling would have executed as
//!   plain rope with the plan reporting nothing.
//!
//! The verdict is fail-closed and the exit gate is mechanical:
//! `blocking == 0` before a single weight byte is converted.

mod attention_policy;
pub mod capability;
pub mod carriage;
pub mod compare;
pub mod report;
pub mod semantics;

#[cfg(test)]
mod tests;
/// Glimmer-shaped inventory fixtures, shared with the graph tests.
#[cfg(test)]
pub mod tests_support;

use larql_models::inventory::{ArchitectureInventory, KeyStatus};

use super::graph::{build_from_inventories, BuiltGraph, Component, ComponentRole};

pub use report::{
    ArtifactPlan, Finding, FindingCategory, InterfacePlan, PlanSummary, SemanticClass, SystemPlan,
    PLAN_SCHEMA,
};

/// Build the system plan over one or more inventories.
///
/// `named` pairs one display name per inventory (the CLI passes directory
/// stems).
pub fn plan_system(named: &[(String, ArchitectureInventory)]) -> SystemPlan {
    let built = build_from_inventories(named);

    let artifacts: Vec<ArtifactPlan> = named
        .iter()
        .map(|(name, inventory)| plan_artifact(name, inventory, &built))
        .collect();

    let interfaces: Vec<InterfacePlan> = built
        .graph
        .edges
        .iter()
        .map(|edge| InterfacePlan {
            producer_component: edge.producer_component.clone(),
            producer_layers: edge.producer_layers.clone(),
            consumer_component: edge.consumer_component.clone(),
            consumer_object: edge.consumer_object.clone(),
            block_size: edge.block_size,
        })
        .collect();

    let mut summary = PlanSummary::default();
    for finding in artifacts.iter().flat_map(|a| &a.findings) {
        match finding.category {
            FindingCategory::Representable => summary.representable += 1,
            FindingCategory::Mismatched => summary.mismatched += 1,
            FindingCategory::Unrepresented => summary.unrepresented += 1,
            FindingCategory::Interface => summary.interfaces += 1,
        }
        if finding.blocks() {
            summary.blocking += 1;
        }
    }
    summary.interfaces += interfaces.len();
    // Whole-model completeness, unchanged: every declared semantic fact of
    // this checkpoint has a faithful home. Deliberately still a single
    // Boolean over everything — see `capability` for why execution needs a
    // different question rather than a weaker version of this one.
    let admissible = summary.blocking == 0;
    let capabilities = capability::Capability::ALL
        .iter()
        .map(|c| {
            capability::admissible_for(*c, artifacts.iter().flat_map(|a| &a.findings), &built.graph)
        })
        .collect();

    SystemPlan {
        schema: PLAN_SCHEMA,
        artifacts,
        interfaces,
        admissible,
        capabilities,
        summary,
        graph: built.graph,
    }
}

/// Plan one artifact: value comparison, unconsumed-key grading, and the
/// graph builder's verdict on its tensors, topology and interfaces.
fn plan_artifact(
    name: &str,
    inventory: &ArchitectureInventory,
    built: &BuiltGraph,
) -> ArtifactPlan {
    let mut findings = compare::compare(inventory);
    findings.extend(config_key_findings(inventory, built));
    findings.extend(placed_object_findings(name, built));
    findings.extend(unplaced_group_findings(name, built));
    findings.extend(attention_policy_findings(name, built));
    findings.extend(execution_surface_findings(name, built));
    findings.extend(unresolved_interface_findings(name, built));
    ArtifactPlan {
        name: name.to_string(),
        model_type: inventory.identity.model_type.clone(),
        findings,
    }
}

/// Every declared config key, graded by the semantics registry and — for
/// the execution-semantic ones — by how far VINDEX3 actually carries it.
///
/// The census is over *all* keys, not just the unconsumed ones. Reporting
/// only the unconsumed keys made `unrepresented: N` a lower bound with no
/// stated denominator, and hid the failure this gate exists to catch: a
/// key the parser reads (`consumed`) that VINDEX3 then drops. See
/// [`carriage`] for why parser consumption is not representation
/// authority.
fn config_key_findings(inventory: &ArchitectureInventory, built: &BuiltGraph) -> Vec<Finding> {
    inventory
        .config_keys
        .iter()
        .map(|fact| {
            let leaf = semantics::leaf_of(&fact.path);
            let component = semantics::component_of(&fact.path);
            match fact.status {
                // Read by nothing: the original G1 finding. Carriage is
                // moot — a fact no parser read cannot be carried anywhere.
                KeyStatus::Unconsumed => Finding {
                    category: FindingCategory::Unrepresented,
                    class: unconsumed_class(leaf, inventory),
                    component,
                    subject: fact.path.clone(),
                    declared: Some(fact.value.clone()),
                    resolved: None,
                    carriage: None,
                    detail: "declared by the checkpoint, read by nothing in any registered \
                             parser"
                        .to_string(),
                },
                KeyStatus::Metadata => Finding {
                    category: FindingCategory::Representable,
                    class: SemanticClass::MetadataOnly,
                    component,
                    subject: fact.path.clone(),
                    declared: Some(fact.value.clone()),
                    resolved: None,
                    carriage: None,
                    detail: "identity or training-time fact, inert for a forward pass".to_string(),
                },
                KeyStatus::Consumed => carriage_finding(fact, leaf, component, built),
            }
        })
        .collect()
}

/// Class for an unconsumed key. A registered alias is only benign while
/// its canonical spelling is genuinely declared *and* consumed in the
/// same config — otherwise the alias is the only carrier of the fact and
/// grades `Unknown`, which blocks.
fn unconsumed_class(leaf: &str, inventory: &ArchitectureInventory) -> SemanticClass {
    let class = semantics::classify_key(leaf);
    if class != SemanticClass::Alias {
        return class;
    }
    let Some(canonical) = semantics::alias_canonical(leaf) else {
        return SemanticClass::Unknown;
    };
    let backed = inventory.config_keys.iter().any(|other| {
        other.status == KeyStatus::Consumed
            && (other.path == canonical || other.path.ends_with(&format!(".{canonical}")))
    });
    if !backed {
        return SemanticClass::Unknown;
    }
    // Presence of the canonical key is not enough. An alias is benign
    // only while it *corroborates* the canonical fact; one that
    // contradicts it is a second, disagreeing authority, and grading that
    // `Alias` would be exactly the "way to silence a key" the class
    // contract forbids. Qwen3.8 declares `full_attention_interval: 4`
    // beside a 64-entry `layer_types`, and the two agree — but nothing
    // checked that until this rung, so a checkpoint whose interval
    // disagreed with its own array would have passed silently.
    if alias_contradicts_canonical(leaf, inventory) {
        return SemanticClass::Unknown;
    }
    SemanticClass::Alias
}

/// Whether a registered alias disagrees with the canonical fact it is
/// supposed to restate.
///
/// Only aliases with a checkable relationship are examined; one with no
/// derivation into the canonical form cannot contradict it and answers
/// `false`. Never the source of truth: this decides only whether the
/// alias is *benign*, and `layer_types` remains the authority the graph
/// is built from either way.
fn alias_contradicts_canonical(leaf: &str, inventory: &ArchitectureInventory) -> bool {
    const FULL_ATTENTION_INTERVAL: &str = "full_attention_interval";
    if leaf != FULL_ATTENTION_INTERVAL {
        return false;
    }
    let value_of = |name: &str| {
        inventory
            .config_keys
            .iter()
            .find(|f| semantics::leaf_of(&f.path) == name)
            .map(|f| &f.value)
    };
    let Some(interval) = value_of(FULL_ATTENTION_INTERVAL)
        .and_then(serde_json::Value::as_u64)
        .filter(|n| *n > 0)
    else {
        // An interval this build cannot read is not a corroboration.
        return true;
    };
    let Some(declared) = value_of("layer_types").and_then(serde_json::Value::as_array) else {
        return true;
    };
    // "Every Nth layer attends fully": layer i is full iff (i+1) % N == 0.
    !declared.iter().enumerate().all(|(i, entry)| {
        entry.as_str().is_some_and(|spelling| {
            spelling.eq_ignore_ascii_case(larql_models::config::LAYER_TYPE_FULL_ATTENTION)
                == (i as u64 + 1).is_multiple_of(interval)
        })
    })
}

/// The carriage verdict for one consumed key: does VINDEX3 carry it past
/// the parser, and does what it carries still equal what was declared?
fn carriage_finding(
    fact: &larql_models::inventory::ConfigKeyFact,
    leaf: &str,
    component_name: String,
    built: &BuiltGraph,
) -> Finding {
    let class = semantics::classify_key(leaf);
    // Tensor semantics are proven carried by the placed-object findings
    // (the graph holds the operands themselves), and interface semantics
    // by the resolved edges — both classes are demonstrated *elsewhere* in
    // the plan, so passing them through here is not a hole. `Unknown` has
    // no such elsewhere: nothing proves it, so — same as an unconsumed key
    // — it must not take this exit. Before this arm named it, a key the
    // parser read but this registry had never classified graded
    // `representable` here regardless, which is exactly the "consumed but
    // unjudged" shape the module exists to refuse (A-11 census, 2026-08-18:
    // Granite's four multipliers and 37 other keys were silently passing
    // this way — `plan/tests/semantics.rs::every_consumed_leaf_key_is_judged`
    // now keeps the registry complete enough that this arm cannot fire).
    if class != SemanticClass::ExecutionSemantic && class != SemanticClass::Unknown {
        return Finding {
            category: FindingCategory::Representable,
            class,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: "read by a registered parser".to_string(),
        };
    }
    if class == SemanticClass::Unknown {
        return Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::Unknown,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: "consumed by a registered parser, but the semantics registry has never \
                     classified this key — parser consumption is not representation \
                     authority, so an unjudged key blocks whether or not a parser reads it"
                .to_string(),
        };
    }
    let Some(rule) = carriage::rule_for(leaf) else {
        return Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::ExecutionSemantic,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: "execution-semantic and parsed, but no carriage rule states whether \
                     VINDEX3 represents it — parser consumption is not representation \
                     authority, so this blocks until judged"
                .to_string(),
        };
    };
    // A rule that honestly stops at the parser carries its justification
    // in `site`; there is nothing to read back.
    if rule.reaches == carriage::Carriage::Parsed {
        return Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: format!("stops at the parser by judgement — {}", rule.site),
        };
    }
    let ctx = carriage::ProbeContext {
        span: carriage::ProbeContext::span_of(&fact.path),
        declared: &fact.value,
    };
    let carried = component_for_key(built, &component_name)
        .and_then(|component| rule.probe.and_then(|probe| probe(component, &ctx)));
    // Compared against a *canonicalised* declared value: for leaves where
    // VINDEX3 legitimately stores a renamed or derived form of the same
    // fact (see [`carriage::canonical_declared`]), this is the raw
    // declaration re-expressed the same way the parser/runtime already
    // does — not a loosened comparison. Findings still report the raw
    // `fact.value` so the checkpoint's own spelling stays on the record.
    let comparable_declared = carriage::canonical_declared(leaf, &fact.value);
    match carried {
        // The schema holds a value: compare it to the declaration. This
        // is where a dropped fact dies — GPT-OSS declares `yarn` and the
        // position policy can only answer `default`.
        Some(carried) if values_agree(&carried, &comparable_declared) => {
            let detail = if comparable_declared == fact.value {
                format!("carried to `{}` at {}", rule.reaches.name(), rule.site)
            } else {
                format!(
                    "carried to `{}` at {} — declared `{}` and stored `{}` are the same fact \
                     under the canonical conversion VINDEX3 already applies at runtime, not \
                     compared as raw JSON",
                    rule.reaches.name(),
                    rule.site,
                    fact.value,
                    carried
                )
            };
            Finding {
                category: FindingCategory::Representable,
                class: SemanticClass::ExecutionSemantic,
                component: component_name,
                subject: fact.path.clone(),
                declared: Some(fact.value.clone()),
                resolved: Some(carried),
                carriage: Some(rule.reaches),
                detail,
            }
        }
        Some(carried) => Finding {
            category: FindingCategory::Mismatched,
            class: SemanticClass::ExecutionSemantic,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: Some(carried),
            carriage: Some(carriage::Carriage::Parsed),
            detail: format!(
                "parsed, but VINDEX3 carries a different value at {} — the declared fact is \
                 dropped at the container boundary",
                rule.site
            ),
        },
        // No component could answer. Reported, never assumed correct:
        // the rule claims carriage that nothing here demonstrates.
        None => Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::ExecutionSemantic,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: format!(
                "rule claims `{}` at {}, but no built component answered the probe",
                rule.reaches.name(),
                rule.site
            ),
        },
    }
}

/// The built component a config path belongs to. `text`/`language` and
/// root-level keys describe the main text component; `<name>_config`
/// keys describe the component of that name.
fn component_for_key<'a>(built: &'a BuiltGraph, component_name: &str) -> Option<&'a Component> {
    const ROOT: &str = "root";
    const TEXT: &str = "text";
    built
        .graph
        .components
        .iter()
        .find(|c| c.id == component_name)
        .or_else(|| {
            (component_name == ROOT || component_name == TEXT)
                .then(|| {
                    built
                        .graph
                        .components
                        .iter()
                        .find(|c| c.role == ComponentRole::PrimaryText)
                })
                .flatten()
        })
}

/// JSON equality up to the precision the schema actually stores.
///
/// Exact first; then equality **after an f32 round-trip**, because parts
/// of the surface narrow these facts to f32 on the way in. GPT-OSS
/// declares `rms_norm_eps: 1e-5` and the graph carries
/// `9.999999747378752e-6` — not a different value but the same one seen
/// through f32, bit for bit. Reporting that as a dropped fact would be
/// the gate misreading its own instrument, so the rule is the precise
/// relationship rather than a chosen tolerance: a genuine change (Muse
/// Glimmer's 1e-5 pre vs 1e-8 post norms) still differs as f32.
fn values_agree(carried: &serde_json::Value, declared: &serde_json::Value) -> bool {
    match (carried.as_array(), declared.as_array()) {
        (Some(a), Some(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_agree(x, y))
        }
        _ => match (carried.as_f64(), declared.as_f64()) {
            (Some(a), Some(b)) => a == b || a as f32 == b as f32,
            _ => carried == declared,
        },
    }
}

/// One representable finding per logical object this artifact's tensors
/// bind into — the graph id is the proof of a home.
fn placed_object_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .graph
        .objects
        .iter()
        .filter(|object| {
            object
                .source_bindings
                .iter()
                .any(|b| b.artifact == artifact)
        })
        .map(|object| {
            let bytes: u64 = object
                .source_bindings
                .iter()
                .filter(|b| b.artifact == artifact)
                .map(|b| b.bytes)
                .sum();
            let encodings: Vec<&str> = object
                .representations
                .iter()
                .map(|r| r.encoding.as_str())
                .collect();
            Finding {
                category: FindingCategory::Representable,
                class: SemanticClass::TensorSemantic,
                component: object.component.clone(),
                subject: object.id.clone(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: format!(
                    "placed as `{}` ({} bytes from this artifact; encodings: {})",
                    object.kind.name(),
                    bytes,
                    encodings.join(", "),
                ),
            }
        })
        .collect()
}

/// Blocking finding per tensor group the builder could not place.
fn unplaced_group_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .unplaced
        .iter()
        .filter(|u| u.artifact == artifact)
        .map(|u| Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::Unknown,
            component: String::new(),
            subject: u.prefix.clone(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: u.reason.clone(),
        })
        .collect()
}

/// The attention policy of each component this artifact sourced: recorded
/// per layer in the graph (span, window, position incl. NoPE), so a hybrid
/// interleave is representable — and directly consumable by KV planning.
fn attention_policy_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .graph
        .components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.role != ComponentRole::Perception)
        .filter_map(|component| {
            let table = component.attention.as_ref()?;
            let census = attention_policy::AttentionCensus::of(table);
            Some(Finding {
                category: if census.blocks() {
                    FindingCategory::Unrepresented
                } else {
                    FindingCategory::Representable
                },
                class: SemanticClass::ExecutionSemantic,
                component: component.id.clone(),
                subject: "attention_policy".to_string(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: census.describe(&component.id),
            })
        })
        .collect()
}

/// Execution-surface verdict per component this artifact sourced: a
/// representable finding when the surface is complete, a blocking one
/// itemising the missing source facts when it is not (V3-G5a). An
/// executor with a partial surface would have to default, which G5
/// forbids — so incompleteness refuses conversion up front.
fn execution_surface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    let mut findings: Vec<Finding> = built
        .graph
        .components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.execution.is_some())
        .map(|component| Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: component.id.clone(),
            subject: format!("{}.execution_surface", component.id),
            declared: None,
            resolved: None,
            carriage: None,
            detail: format!(
                "execution surface complete (attention, ffn, norm{})",
                if component
                    .execution
                    .as_ref()
                    .is_some_and(|s| s.head.is_some())
                {
                    ", head"
                } else {
                    ""
                }
            ),
        })
        .collect();
    findings.extend(
        built
            .incomplete_surfaces
            .iter()
            .filter(|s| s.artifact == artifact)
            .map(|s| Finding {
                category: FindingCategory::Unrepresented,
                class: SemanticClass::ExecutionSemantic,
                component: s.component.clone(),
                subject: format!("{}.execution_surface", s.component),
                declared: None,
                resolved: None,
                carriage: None,
                detail: format!(
                    "execution surface incomplete — missing: {}",
                    s.missing.join(", ")
                ),
            }),
    );
    findings
}

/// Blocking finding per interface the builder could not resolve.
fn unresolved_interface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .unresolved_interfaces
        .iter()
        .filter(|u| u.artifact == artifact)
        .map(|u| Finding {
            category: FindingCategory::Interface,
            class: SemanticClass::InterfaceSemantic,
            component: String::new(),
            subject: "hidden_state_interface".to_string(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: u.reason.clone(),
        })
        .collect()
}
