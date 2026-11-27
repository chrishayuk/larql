//! The planner entry points: from inventories to a system plan.

use super::super::graph::{build_from_inventories, BuiltGraph};
use larql_models::detect::find_architecture;
use larql_models::inventory::{ArchitectureInventory, KeyStatus};

#[allow(unused_imports)]
use super::*;

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
pub(super) fn plan_sourced(
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
pub(super) fn plan_artifact(
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
    findings.extend(excluded_tensor_findings(inventory));
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
pub(super) fn architecture_identity_findings(
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
pub(super) fn undeclared_family_findings(inventory: &ArchitectureInventory) -> Vec<Finding> {
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
pub(super) fn config_key_findings(
    inventory: &ArchitectureInventory,
    built: &BuiltGraph,
) -> Vec<Finding> {
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
pub(super) fn unconsumed_class(
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
