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

use larql_models::detect::find_architecture;
use larql_models::inventory::{ArchitectureInventory, KeyStatus};

use super::graph::{build_from_inventories, BuiltGraph, Component, ComponentRole};

pub use report::{
    ArtifactPlan, ArtifactSource, Finding, FindingCategory, FindingId, InterfacePlan, PlanSummary,
    PlannedFinding, PlannerIdentity, SemanticClass, SystemPlan, VerdictCacheKey,
    PLANNER_SEMANTICS_VERSION, PLAN_SCHEMA,
};

/// Build the system plan over one or more inventories.
///
/// `named` pairs one display name per inventory (the CLI passes directory
/// stems). Each artifact's source is the path its inventory recorded, with
/// no revision — the local-checkpoint case. A caller that knows more
/// (a repo commit) uses [`plan_system_with_sources`].
pub fn plan_system(named: &[(String, ArchitectureInventory)]) -> SystemPlan {
    let sources: Vec<ArtifactSource> = named
        .iter()
        .map(|(_, inventory)| ArtifactSource::local(inventory.path.clone()))
        .collect();
    plan_sourced(named, &sources)
}

/// [`plan_system`] with each artifact's source stated by the caller —
/// `sources[i]` describes `named[i]`. Refuses mismatched lengths rather
/// than pairing verdicts with the wrong subjects.
pub fn plan_system_with_sources(
    named: &[(String, ArchitectureInventory)],
    sources: &[ArtifactSource],
) -> Result<SystemPlan, crate::error::VindexError> {
    if named.len() != sources.len() {
        return Err(crate::error::VindexError::Parse(format!(
            "{} artifact(s) but {} source(s): every artifact needs exactly one source",
            named.len(),
            sources.len()
        )));
    }
    Ok(plan_sourced(named, sources))
}

/// The planner proper. `sources` is exactly as long as `named`; both
/// public entry points guarantee it.
fn plan_sourced(
    named: &[(String, ArchitectureInventory)],
    sources: &[ArtifactSource],
) -> SystemPlan {
    let built = build_from_inventories(named);

    let mut artifacts: Vec<ArtifactPlan> = named
        .iter()
        .zip(sources)
        .map(|((name, inventory), source)| plan_artifact(name, inventory, source, &built))
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

    // One id space over the whole document: a capability closure points
    // into it, and two artifacts must not both own `0`.
    let mut next_id = 0usize;
    for artifact in &mut artifacts {
        for finding in &mut artifact.findings {
            finding.id = FindingId(next_id);
            next_id += 1;
        }
    }

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
        planner: PlannerIdentity::current(),
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
    source: &ArtifactSource,
    built: &BuiltGraph,
) -> ArtifactPlan {
    let mut findings = compare::compare(inventory);
    findings.extend(architecture_identity_findings(name, inventory, built));
    findings.extend(undeclared_family_findings(inventory));
    findings.extend(config_key_findings(inventory, built));
    findings.extend(placed_object_findings(name, built));
    findings.extend(unplaced_group_findings(name, built));
    findings.extend(attention_policy_findings(name, built));
    findings.extend(execution_surface_findings(name, built));
    findings.extend(unresolved_interface_findings(name, built));
    ArtifactPlan {
        name: name.to_string(),
        source: source.clone(),
        model_type: inventory.identity.model_type.clone(),
        // Ids are stamped per artifact and renumbered across the document
        // below, so an artifact planned alone and the same artifact
        // planned beside others carry the same findings either way.
        findings: findings
            .into_iter()
            .enumerate()
            .map(|(i, f)| PlannedFinding::assign(i, f))
            .collect(),
    }
}

/// Who this checkpoint says it is — and whether this build can answer.
///
/// A separate gate from [`undeclared_family_findings`], which asks what
/// the *layers* run. That one deliberately passes a checkpoint that
/// declares an attention shape, because a declared head geometry is a
/// program statement. It is not an identity statement: `num_attention_heads`
/// says nothing about norm placement, QK norm, embedding scaling or gating,
/// and `GenericArch` supplies Llama-shaped answers for all of them. So a
/// checkpoint could declare an architecture nothing here recognises, be
/// served from those defaults, and raise no finding at all — 15 of the 42
/// `model_type` strings in the conformance corpus, over 30 checkpoints.
///
/// Two distinct failures, kept distinct because they need different fixes:
///
/// ```text
/// Unknown   one declaration, no registered family
///           -> VINDEX3 lacks the semantics. RED.
///
/// Conflict  container and text component declare identities that
///           resolve DIFFERENTLY -> this build would serve one of two
///           different models depending on which level it read. RED.
/// ```
///
/// `Unknown` is deliberately not [`SemanticClass::UnsupportedComponent`].
/// The checkpoint naming a family is not the same as VINDEX3 understanding
/// it: `UnsupportedComponent` claims the semantics are understood and only
/// the implementation is missing, which would promote every unrecognised
/// model to AMBER and destroy the distinction AMBER exists to carry.
/// **Scope: the identity of the model that SERVES TEXT.** The gate asks
/// the artifact carrying the primary text component and no other. A
/// drafter or a perception tower is a separate sub-model reached only
/// through its own capability, and `muse_glimmer_assistant` is the live
/// case: `detect_from_json` leaves it generic *deliberately* (weighted QK
/// norms, no gate, unjudged), a decision recorded in the registry as an
/// absence. Blocking on it here would not have made that decision
/// visible — it would have refused a container over a judgement someone
/// already made.
///
/// KNOWN GAP, stated rather than hidden: an auxiliary component with an
/// unrecognised identity is still served generically. For a drafter the
/// cost is bounded — speculative decoding verifies against the target, so
/// a wrong draft head costs throughput, not output — but the registry has
/// no way today to say "generic ON PURPOSE" as opposed to "not yet
/// judged", and until it does, this gate cannot tell those apart.
fn architecture_identity_findings(
    artifact: &str,
    inventory: &ArchitectureInventory,
    built: &BuiltGraph,
) -> Vec<Finding> {
    let declared = &inventory.identity.model_type;
    // An empty declaration is a different finding (the checkpoint states
    // no identity at all) and belongs to the census, not here.
    if declared.is_empty() {
        return Vec::new();
    }
    // The component this artifact serves text from, if it does. When the
    // graph has no unambiguous primary text component the gate stays
    // silent: that is its own finding, raised elsewhere, and guessing an
    // owner here would attribute the refusal to the wrong sub-model.
    let Ok(primary) = built.graph.primary_text_component() else {
        return Vec::new();
    };
    if primary.source_artifact != artifact {
        return Vec::new();
    }
    let component = primary.id.clone();
    let resolved = find_architecture(declared);
    let mut findings = Vec::new();

    // The two levels are compared by what they RESOLVE to, not by string
    // equality: 27 of the 28 corpus checkpoints declaring at both levels
    // use the `<container>_text` suffix form, which is one identity spelled
    // twice. Only a divergence that changes which implementation answers is
    // a conflict.
    if let Some(container) = &inventory.identity.container_model_type {
        let container_resolved = find_architecture(container);
        // Three answers, not two. The gate's principle is unchanged —
        // which config level happened to be read must never decide which
        // architecture the runtime serves — but container identity and
        // component identity are not competing claims about the same
        // level of abstraction. A container may DECLARE which
        // architecture occupies its text slot, and when it does, the two
        // levels agreeing with that declaration is not a conflict.
        //
        // Lineage only. The declaration confers nothing: it says who
        // occupies the slot, never what may execute. See
        // `ArchitectureEntry::components`.
        let differs = match (container_resolved, resolved) {
            (Some(a), Some(b)) if std::ptr::eq(a, b) => false,
            (Some(container_entry), Some(_)) => {
                // Directional: the CONTAINER declares its component, and
                // the reverse is never consulted.
                container_entry
                    .declares_component(larql_models::detect::registry::ComponentRole::Text)
                    != Some(declared.as_str())
            }
            (None, None) => false,
            _ => true,
        };
        if differs {
            findings.push(Finding {
                category: FindingCategory::Mismatched,
                class: SemanticClass::Unknown,
                component: component.clone(),
                subject: "architecture_identity".to_string(),
                declared: Some(serde_json::Value::String(container.clone())),
                resolved: Some(serde_json::Value::String(declared.clone())),
                carriage: None,
                detail: format!(
                    "the container declares `{container}` and its text component declares                      `{declared}`, and the two resolve to different architectures                      ({} vs {}) — which model this build serves would depend on which                      level it happened to read, so the identity is refused rather than                      picked",
                    container_resolved.map_or("no registered family", |e| e.model_type),
                    resolved.map_or("no registered family", |e| e.model_type),
                ),
            });
        }
    }

    if resolved.is_none() {
        findings.push(Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::Unknown,
            component,
            // Not `model_type`: the config-key census already reports a
            // finding under that subject (the key was read by a parser,
            // which is true and separate). Two findings sharing a subject
            // would read as one contradicting itself.
            subject: "architecture_family".to_string(),
            declared: Some(serde_json::Value::String(declared.clone())),
            resolved: None,
            carriage: None,
            detail: format!(
                "`{declared}` matches no registered family, so detection resolves to the                  generic architecture and serves Llama-shaped defaults for norm placement,                  QK norm, embedding scaling and gating — facts this checkpoint never                  declared. An unrecognised identity is refused, not approximated"
            ),
        });
    }
    findings
}

/// The fail-closed layer census (schema 6, drill F3): a checkpoint whose
/// family no registry recognises AND that declares no per-layer topology
/// has stated NOTHING about what its layers run. Before this finding,
/// every such layer resolved to `(Softmax, Full)` and the census failed
/// open — the mamba2 witness was reported as a 48-layer softmax tower
/// with invented head geometry, saved only by its unconsumed keys. A
/// registered family IS a judgment (the match arm is the declaration);
/// a declared interleave or uniform kind is one; generic-plus-silence is
/// neither, and resolving it to softmax is a fabrication.
fn undeclared_family_findings(inventory: &ArchitectureInventory) -> Vec<Finding> {
    let declares_layers = inventory
        .resolved
        .layers
        .iter()
        .any(|l| l.declared_kind.is_some() || l.declared_span.is_some());
    // A declared attention-head geometry IS a program declaration: the
    // checkpoint states softmax-shaped attention in its own words, and
    // resolving its layers to softmax reads that declaration rather than
    // inventing one. What fails closed is generic-plus-SILENCE — no
    // family, no per-layer topology, no attention shape (the pure-SSM
    // case, where the old path invented 8/4 heads).
    let declares_attention_shape = inventory.config_keys.iter().any(|fact| {
        matches!(
            semantics::leaf_of(&fact.path),
            "num_attention_heads" | "n_head"
        )
    });
    if !inventory.detection.generic_fallback || declares_layers || declares_attention_shape {
        return Vec::new();
    }
    vec![Finding {
        category: FindingCategory::Unrepresented,
        class: SemanticClass::ExecutionSemantic,
        component: String::new(),
        subject: "layer_census".to_string(),
        declared: None,
        resolved: None,
        carriage: None,
        detail: format!(
            "no registered family and no declared per-layer topology: {} layer(s) would \
             resolve to softmax/full by default, which is a fabrication, not a resolution — \
             the census fails closed until the family is judged or the checkpoint declares \
             its layers",
            inventory.resolved.num_layers
        ),
    }]
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
            // A key declared with no value states that the subject does
            // not apply, and there is nothing for the container to carry
            // or to drop. It cannot be a silent-default bug, because
            // there is no declared value to default away from.
            //
            // Gemma 4's dense sizes are the witness: they declare
            // `top_k_experts: null` and `expert_intermediate_size: null`
            // — a dense model saying it has no expert bank — and the
            // carriage rule demanded a home at
            // `ExecutionSurface.ffn.moe.top_k`, which no dense component
            // can answer. `num_experts: null` sat beside them already
            // grading representable, so the two nulls were being judged
            // differently; this is what makes them agree.
            //
            // Narrow on purpose: null only. A key declared with a value
            // this build cannot represent still blocks, which the
            // `a_declared_value_still_blocks` control holds.
            if fact.value.is_null() {
                return Finding {
                    category: FindingCategory::Representable,
                    class: semantics::classify_key(leaf),
                    component,
                    subject: fact.path.clone(),
                    declared: Some(fact.value.clone()),
                    resolved: None,
                    carriage: None,
                    detail: "declared with no value — the checkpoint states the subject does                              not apply, so there is nothing to represent"
                        .to_string(),
                };
            }
            match fact.status {
                // Read by nothing: the original G1 finding. Carriage is
                // moot — a fact no parser read cannot be carried anywhere.
                KeyStatus::Unconsumed => {
                    let class = unconsumed_class(leaf, &fact.value, inventory);
                    Finding {
                        category: FindingCategory::Unrepresented,
                        class,
                        component,
                        subject: fact.path.clone(),
                        declared: Some(fact.value.clone()),
                        resolved: None,
                        carriage: None,
                        // A named component is the whole value of the
                        // class: "read by nothing" counts keys, while
                        // this counts jobs.
                        detail: match semantics::unsupported_component(leaf) {
                            Some(component) if class == SemanticClass::UnsupportedComponent => {
                                format!(
                                    "configures `{component}`, which this build does not \
                                     implement — architecture work, not a normalisation gap"
                                )
                            }
                            _ => "declared by the checkpoint, read by nothing in any registered \
                                  parser"
                                .to_string(),
                        },
                    }
                }
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
                KeyStatus::Consumed => carriage_finding(fact, leaf, component, built, inventory),
            }
        })
        .collect()
}

/// Class for an unconsumed key. A registered alias is only benign while
/// its canonical spelling is genuinely declared *and* consumed in the
/// same config — otherwise the alias is the only carrier of the fact and
/// grades `Unknown`, which blocks.
fn unconsumed_class(
    leaf: &str,
    value: &serde_json::Value,
    inventory: &ArchitectureInventory,
) -> SemanticClass {
    let class = semantics::classify_key_at(leaf, value);
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
    inventory: &ArchitectureInventory,
) -> Finding {
    let class = semantics::classify_key_at(leaf, &fact.value);
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
    // `UnsupportedComponent` cannot take this exit either, and for a
    // sharper reason than `Unknown` can. That class means "we know
    // exactly what component this configures and have no implementation
    // for it" — so a parser reading the key proves recognition and
    // nothing more. Letting it grade `representable` here says the fact
    // has a home when the component it configures does not exist in this
    // build, and it made DeepSeek-V4 lose three blockers the moment
    // wave 16 started READING `hc_mult` — the row looking closer to
    // admissible purely because the topology became recognised, which is
    // the failure that wave's own falsifier named.
    if class != SemanticClass::ExecutionSemantic
        && class != SemanticClass::Unknown
        && class != SemanticClass::UnsupportedComponent
    {
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
    if class == SemanticClass::UnsupportedComponent {
        return Finding {
            category: FindingCategory::Unrepresented,
            class,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: format!(
                "read by a registered parser, and it configures {} — recognised, and \
                 not implemented by this build",
                semantics::unsupported_component(leaf).unwrap_or("an absent component")
            ),
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
    // A declaration a companion switches off has nothing for the graph to
    // carry, and probing for it asks the wrong question: the graph is
    // right to hold no value, so the comparison would report agreement as
    // a dropped fact.
    if let Some(switch) = carriage::disabled_by_companion(
        &fact.path,
        inventory
            .config_keys
            .iter()
            .map(|k| (k.path.as_str(), &k.value)),
    ) {
        return Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: component_name,
            subject: fact.path.clone(),
            declared: Some(fact.value.clone()),
            resolved: None,
            carriage: Some(carriage::Carriage::Parsed),
            detail: format!(
                "declared and switched off by `{switch}` — the value is inert, and the \
                 graph carrying no window agrees with the checkpoint rather than dropping \
                 its declaration"
            ),
        };
    }
    let ctx = carriage::ProbeContext {
        span: carriage::ProbeContext::span_of(&fact.path),
        declared: &fact.value,
        family: find_architecture(&inventory.identity.model_type).map(|entry| entry.model_type),
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
            // The aliases resolve only when the primary is unique —
            // ambiguity yields no component rather than the first one
            // (drill F10).
            (component_name == ROOT || component_name == TEXT)
                .then(|| built.graph.primary_text_component().ok())
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
        .map(|component| {
            // Name the groups that are actually present — presence follows
            // the program (schema 6), so the sentence must too. Listing
            // absent groups as complete was the fabrication, restated.
            let surface = component.execution.as_ref().expect("filtered above");
            let mut groups: Vec<&str> = Vec::new();
            if surface.attention.is_some() {
                groups.push("attention");
            }
            if surface.mamba2.is_some() {
                groups.push("mamba2 mixer");
            }
            if surface.ffn.is_some() {
                groups.push("ffn");
            }
            groups.push("norm");
            if surface.head.is_some() {
                groups.push("head");
            }
            // A complete surface is not an executable one. Where the
            // judged placement is a shape this build refuses to lower,
            // the report must SAY so — otherwise the row reads as "every
            // declaration has a home" while `plan_component_ops` returns
            // no plan at all, which is the looks-supported failure the
            // whole instrument exists to catch. Read from the same
            // authority the op plan refuses on, so the two cannot drift.
            // The topology is asked FIRST, because it decides what the
            // residual even is; the placement is asked only of a stack
            // whose residual this build can represent.
            let refusal = surface
                .residual_topology
                .unimplemented_reason()
                .map(|reason| (format!("{:?}", surface.residual_topology), reason))
                .or_else(|| {
                    surface.norm.placement.and_then(|p| {
                        p.unimplemented_reason()
                            .map(|reason| (format!("its norm placement ({p:?})"), reason))
                    })
                });
            match refusal {
                Some((what, reason)) => Finding {
                    category: FindingCategory::Unrepresented,
                    class: SemanticClass::UnsupportedComponent,
                    component: component.id.clone(),
                    subject: format!("{}.execution_surface", component.id),
                    declared: None,
                    resolved: None,
                    carriage: None,
                    detail: format!(
                        "execution surface complete ({}), and {what} is representable \
                         but NOT executable by this build — {reason}",
                        groups.join(", ")
                    ),
                },
                None => Finding {
                    category: FindingCategory::Representable,
                    class: SemanticClass::ExecutionSemantic,
                    component: component.id.clone(),
                    subject: format!("{}.execution_surface", component.id),
                    declared: None,
                    resolved: None,
                    carriage: None,
                    detail: format!("execution surface complete ({})", groups.join(", ")),
                },
            }
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

// ── The plan-by-source sequence, in one place ────────────────────────

/// Plan artifacts the caller has already resolved.
///
/// `specs[i]` is the argument the caller was given for `resolved[i]` —
/// the string the user typed, which is what the verdict should name, not
/// the local directory an `hf://` reference happened to stage into.
///
/// This exists because three front doors were carrying their own copy of
/// the same fifteen lines (`vindex plan`, `larql vindex3 plan`, and
/// `POST /v1/plan`): resolve, name each artifact's source with the commit
/// its facts were read at, then plan. A verdict that differs by which
/// door asked for it is not a verdict, so the sequence is one function
/// and the doors differ only in how they report it.
///
/// Resolution stays with the caller: it is the step that can be slow and
/// can fail, and each door reports staging differently — the CLIs to
/// stderr as it happens, the server as a field in its response.
pub fn plan_resolved(
    specs: &[std::path::PathBuf],
    resolved: Vec<super::artifact::ResolvedArtifact>,
) -> Result<SystemPlan, crate::error::VindexError> {
    if specs.len() != resolved.len() {
        return Err(crate::error::VindexError::Parse(format!(
            "{} spec(s) but {} resolved artifact(s): every artifact needs exactly one spec",
            specs.len(),
            resolved.len()
        )));
    }
    // The verdict names its subject: the argument as given and, for a
    // repo, the commit the facts were read at.
    let sources: Vec<ArtifactSource> = specs
        .iter()
        .zip(&resolved)
        .map(|(spec, a)| ArtifactSource {
            path: spec.display().to_string(),
            revision: a.commit().map(str::to_string),
            unpinned_revision: a.unpinned_revision().map(str::to_string),
        })
        .collect();
    let named: Vec<(String, ArchitectureInventory)> = resolved
        .into_iter()
        .map(|a| (a.name, a.inventory))
        .collect();
    plan_system_with_sources(&named, &sources)
}
