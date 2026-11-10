//! Findings about carried vs declared config facts.

use super::super::graph::{BuiltGraph, Component, ComponentRole};
use larql_models::detect::find_architecture;
use larql_models::inventory::ArchitectureInventory;

#[allow(unused_imports)]
use super::*;

/// Whether a registered alias disagrees with the canonical fact it is
/// supposed to restate.
///
/// Only aliases with a checkable relationship are examined; one with no
/// derivation into the canonical form cannot contradict it and answers
/// `false`. Never the source of truth: this decides only whether the
/// alias is *benign*, and `layer_types` remains the authority the graph
/// is built from either way.
pub(super) fn alias_contradicts_canonical(leaf: &str, inventory: &ArchitectureInventory) -> bool {
    /// Period spellings of the `layer_types` interleave: "every Nth layer
    /// attends fully" — Qwen3.5's `full_attention_interval`, Gemma 3's
    /// `_sliding_window_pattern`.
    const PERIOD_ALIASES: &[&str] = &["full_attention_interval", "_sliding_window_pattern"];
    if !PERIOD_ALIASES.contains(&leaf) {
        return false;
    }
    let value_of = |name: &str| {
        inventory
            .config_keys
            .iter()
            .find(|f| semantics::leaf_of(&f.path) == name)
            .map(|f| &f.value)
    };
    let Some(interval) = value_of(leaf)
        .and_then(serde_json::Value::as_u64)
        .filter(|n| *n > 0)
    else {
        // An interval this build cannot read is not a corroboration.
        return true;
    };
    let Some(declared) = value_of("layer_types").and_then(serde_json::Value::as_array) else {
        return true;
    };
    // "Every Nth layer attends fully", by the one stride derivation.
    !declared.iter().enumerate().all(|(i, entry)| {
        entry.as_str().is_some_and(|spelling| {
            spelling.eq_ignore_ascii_case(larql_models::config::LAYER_TYPE_FULL_ATTENTION)
                == larql_models::config::is_full_at_period(i, interval as usize)
        })
    })
}

/// The carriage verdict for one consumed key: does VINDEX3 carry it past
/// the parser, and does what it carries still equal what was declared?
pub(super) fn carriage_finding(
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
        // A RoPE base's layers come from the one scoping derivation, which
        // also reads the sibling declarations (the flat Gemma 3
        // `rope_theta` speaks for the full span only beside a
        // `rope_local_base_freq`); every other fact names its span by path.
        span: if rope_scope::is_rope_base(&fact.path) {
            rope_scope::rope_base_scope(&fact.path, &inventory.config_keys).probe_span()
        } else {
            carriage::ProbeContext::span_of(&fact.path)
        },
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
pub(super) fn component_for_key<'a>(
    built: &'a BuiltGraph,
    component_name: &str,
) -> Option<&'a Component> {
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
pub(super) fn values_agree(carried: &serde_json::Value, declared: &serde_json::Value) -> bool {
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
pub(super) fn placed_object_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
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
pub(super) fn unplaced_group_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
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
pub(super) fn attention_policy_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
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
