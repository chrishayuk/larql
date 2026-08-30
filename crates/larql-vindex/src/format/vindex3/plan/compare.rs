//! Declared-vs-resolved value comparison for execution-semantic facts.
//!
//! `consumed` in the inventory says the parser *reads* a key; it does not
//! say resolution *honoured* it. The rope-theta finding that motivated this
//! module was exactly that shape: `rope_parameters.rope_theta` classified
//! consumed while resolution fell through to a default 50× smaller. Each
//! comparator here reads the declared value straight out of the flattened
//! config facts and compares it against what resolution actually produced,
//! emitting `Mismatched` (blocking) or `Representable` (checked and equal).

use larql_models::config::PositionPolicy;
use larql_models::inventory::{ArchitectureInventory, ConfigKeyFact};
use serde_json::Value;

use super::super::graph::policy::{
    resolve_layer_kind, AttentionSpan, LayerOperator, RecurrenceKind,
};

/// Mirror of the graph builder's rule, so the comparator grades against
/// the same operator the graph recorded. See `graph::build::recurrence_kind`.
fn recurrence_kind(inventory: &ArchitectureInventory) -> Option<RecurrenceKind> {
    if inventory.resolved.kda.is_some() {
        return Some(RecurrenceKind::Kda);
    }
    inventory
        .resolved
        .linear_attention
        .is_some()
        .then_some(RecurrenceKind::GatedDelta)
}

/// Mirror of the graph builder's rule. See `graph::build::uses_mla`.
fn uses_mla(inventory: &ArchitectureInventory) -> bool {
    inventory
        .resolved
        .execution
        .as_ref()
        .is_some_and(|e| e.mla.is_some())
}
use super::report::{Finding, FindingCategory, SemanticClass};
use super::semantics::component_of;

/// Sliding layer label used in the inventory's per-layer table.
const ATTENTION_SLIDING: &str = "sliding";
const ATTENTION_FULL: &str = "full";

/// Extractor of the resolved counterpart for one declared scalar.
type ResolvedScalar = fn(&ArchitectureInventory) -> Option<u64>;

/// Scalar topology facts checked one-to-one: declared leaf name → resolved
/// value. Every entry is tensor-semantic: a container encoding the wrong
/// width or depth stores the wrong operands.
const SCALAR_CHECKS: &[(&str, ResolvedScalar)] = &[
    ("hidden_size", |inv| Some(inv.resolved.hidden_size as u64)),
    ("num_hidden_layers", |inv| {
        Some(inv.resolved.num_layers as u64)
    }),
    ("intermediate_size", |inv| {
        Some(inv.resolved.intermediate_size as u64)
    }),
    ("num_attention_heads", |inv| {
        Some(inv.resolved.num_q_heads as u64)
    }),
    ("num_key_value_heads", |inv| {
        Some(inv.resolved.num_kv_heads as u64)
    }),
    ("head_dim", |inv| Some(inv.resolved.head_dim as u64)),
    ("vocab_size", |inv| {
        inv.resolved.vocab_size.map(|v| v as u64)
    }),
    ("sliding_window", |inv| {
        inv.resolved.sliding_window.map(|v| v as u64)
    }),
];

/// Declared value for a leaf name: `text_config.<leaf>` wins over a root
/// `<leaf>` (multimodal nesting mirrors the parser's own preference).
fn declared(facts: &[ConfigKeyFact], leaf: &str) -> Option<(String, Value)> {
    const TEXT_PREFIX: &str = "text_config.";
    let nested = format!("{TEXT_PREFIX}{leaf}");
    facts
        .iter()
        .find(|f| f.path == nested)
        .or_else(|| facts.iter().find(|f| f.path == leaf))
        .map(|f| (f.path.clone(), f.value.clone()))
}

/// The declared RoPE bases, each with the layers it speaks for: the
/// checkpoint-wide base (flat `rope_theta`, transformers-5's flat
/// `rope_parameters.rope_theta`) speaks for every layer; the per-layer-type
/// bases (`rope_parameters.full_attention.rope_theta` /
/// `…sliding_attention.rope_theta` — Gemma 3/4) each speak for their span
/// only. Returns `(path, theta, span-or-None)` per declaration so a
/// mismatch names its source and is judged against the right layers —
/// Gemma 4 declares 1e6 on full layers and 1e4 on sliding ones, and
/// comparing either against the whole table is a false mismatch.
fn declared_rope_thetas(facts: &[ConfigKeyFact]) -> Vec<(String, f64, Option<&'static str>)> {
    const PER_SPAN: &[(&str, &str)] = &[
        (
            "text_config.rope_parameters.full_attention.rope_theta",
            ATTENTION_FULL,
        ),
        ("rope_parameters.full_attention.rope_theta", ATTENTION_FULL),
        (
            "text_config.rope_parameters.sliding_attention.rope_theta",
            ATTENTION_SLIDING,
        ),
        (
            "rope_parameters.sliding_attention.rope_theta",
            ATTENTION_SLIDING,
        ),
    ];
    const WHOLE_TABLE: &[&str] = &[
        "text_config.rope_parameters.rope_theta",
        "rope_parameters.rope_theta",
        "text_config.rope_theta",
        "rope_theta",
    ];
    let mut out: Vec<(String, f64, Option<&'static str>)> = PER_SPAN
        .iter()
        .filter_map(|(path, span)| {
            facts
                .iter()
                .find(|f| f.path == *path)
                .and_then(|f| f.value.as_f64().map(|v| (f.path.clone(), v, Some(*span))))
        })
        .collect();
    // A whole-table base only speaks when no per-span base was declared —
    // the per-span form is the more specific spelling of the same fact.
    if out.is_empty() {
        out.extend(WHOLE_TABLE.iter().find_map(|path| {
            facts
                .iter()
                .find(|f| f.path == *path)
                .and_then(|f| f.value.as_f64().map(|v| (f.path.clone(), v, None)))
        }));
    }
    out
}

/// Run every comparator over one inventory.
pub fn compare(inventory: &ArchitectureInventory) -> Vec<Finding> {
    let facts = &inventory.config_keys;
    let mut findings = Vec::new();

    for (leaf, resolve) in SCALAR_CHECKS {
        let Some((path, value)) = declared(facts, leaf) else {
            continue; // not declared — nothing to disagree with
        };
        let Some(declared_value) = value.as_u64() else {
            continue; // non-numeric declaration is a config oddity, not ours
        };
        let resolved_value = resolve(inventory);
        let agrees = resolved_value == Some(declared_value);
        findings.push(value_finding(
            &path,
            agrees,
            SemanticClass::TensorSemantic,
            value.clone(),
            resolved_value.map(Into::into),
        ));
    }

    findings.extend(rope_theta_findings(inventory));
    findings.extend(layer_rope_theta_findings(inventory));
    findings.extend(layer_types_finding(inventory));
    findings.extend(duplicate_spelling_findings(inventory));
    findings
}

fn value_finding(
    path: &str,
    agrees: bool,
    class: SemanticClass,
    declared: Value,
    resolved: Option<Value>,
) -> Finding {
    Finding {
        category: if agrees {
            FindingCategory::Representable
        } else {
            FindingCategory::Mismatched
        },
        class,
        component: component_of(path),
        subject: path.to_string(),
        carriage: None,
        detail: if agrees {
            "declared and resolved agree".to_string()
        } else {
            "resolution does not honour the declared value".to_string()
        },
        declared: Some(declared),
        resolved,
    }
}

/// Each declared θ vs the resolved policy of the layers it speaks for.
fn rope_theta_findings(inventory: &ArchitectureInventory) -> Vec<Finding> {
    let layers = &inventory.resolved.layers;
    if layers.is_empty() {
        return Vec::new();
    }
    // A per-layer declaration overrides the uniform one; that comparator
    // owns the answer then.
    if inventory
        .config_keys
        .iter()
        .any(|f| f.path.ends_with("layer_rope_theta"))
    {
        return Vec::new();
    }
    declared_rope_thetas(&inventory.config_keys)
        .into_iter()
        .filter_map(|(path, declared_theta, span)| {
            let in_scope: Vec<_> = layers
                .iter()
                .filter(|l| span.is_none_or(|s| l.attention == s))
                .collect();
            let first = in_scope.first()?;
            // A rope base declared on a stack that applies **no rotation**
            // is inert, not a disagreement. Kimi Linear ships
            // `rope_theta: 10000.0` beside `mla_use_nope: true`, and its
            // own reference implementation contains no rotary code at all
            // — the field is a leftover its forward never reads.
            //
            // Conditional on EVERY in-scope layer being NoPE, so a rotary
            // stack whose theta resolves wrong still fails here. That is
            // the failure this comparator was written for (rope theta
            // resolving 50× smaller than declared), and it must stay
            // reachable.
            if in_scope
                .iter()
                .all(|l| l.position == PositionPolicy::None)
            {
                return Some(Finding {
                    category: FindingCategory::Representable,
                    class: SemanticClass::ExecutionSemantic,
                    component: component_of(&path),
                    subject: path.clone(),
                    declared: Some(declared_theta.into()),
                    resolved: None,
                    carriage: None,
                    detail: "declared, and provably not applied: no layer in scope encodes                              position by rotation"
                        .to_string(),
                });
            }
            let disagreeing = in_scope
                .iter()
                .filter(|l| l.position.rope_theta() != Some(declared_theta))
                .count();
            Some(value_finding(
                &path,
                disagreeing == 0,
                SemanticClass::ExecutionSemantic,
                declared_theta.into(),
                serde_json::to_value(first.position).ok(),
            ))
        })
        .collect()
}

/// Per-layer declared θ array vs the resolved per-layer position policy.
///
/// The declared side carries the upstream form verbatim, `0.0` NoPE
/// sentinels included, so the comparison interprets each element through
/// [`PositionPolicy::from_declared_theta`] — the same single boundary the
/// resolver uses. A resolver without a NoPE concept mismatches on exactly
/// the sentinel layers; one that honours the policy agrees layer by layer.
fn layer_rope_theta_findings(inventory: &ArchitectureInventory) -> Option<Finding> {
    let fact = inventory
        .config_keys
        .iter()
        .find(|f| f.path.ends_with("layer_rope_theta"))?;
    let declared_array = fact.value.as_array()?;
    let layers = &inventory.resolved.layers;
    let disagreeing: Vec<usize> = layers
        .iter()
        .filter(|l| {
            declared_array
                .get(l.layer)
                .and_then(Value::as_f64)
                // Compare what the array declares — the base, or the NoPE
                // sentinel — not the whole variant: a YaRN block on top of
                // the same theta is a separate fact with its own carriage.
                .is_some_and(|declared| {
                    PositionPolicy::from_declared_theta(declared).rope_theta()
                        != l.position.rope_theta()
                })
        })
        .map(|l| l.layer)
        .collect();
    let agrees = disagreeing.is_empty() && declared_array.len() == layers.len();
    Some(Finding {
        category: if agrees {
            FindingCategory::Representable
        } else {
            FindingCategory::Mismatched
        },
        class: SemanticClass::ExecutionSemantic,
        component: component_of(&fact.path),
        subject: fact.path.clone(),
        carriage: None,
        detail: if agrees {
            "per-layer position policies all honoured (NoPE included)".to_string()
        } else {
            format!(
                "{} of {} layers resolve a different position policy than declared \
                 (layers {:?}…)",
                disagreeing.len(),
                layers.len(),
                disagreeing.iter().take(4).collect::<Vec<_>>(),
            )
        },
        declared: Some(fact.value.clone()),
        resolved: serde_json::to_value(layers.iter().map(|l| l.position).collect::<Vec<_>>()).ok(),
    })
}

/// Declared `layer_types` interleave vs the resolved per-layer table.
fn layer_types_finding(inventory: &ArchitectureInventory) -> Option<Finding> {
    let fact = inventory
        .config_keys
        .iter()
        .find(|f| semantics_is_layer_types(&f.path))?;
    let declared_array = fact.value.as_array()?;
    let layers = &inventory.resolved.layers;
    // A layer disagrees when its own declared spelling names something
    // outside the schema's executable span vocabulary (a hybrid
    // linear-attention layer, e.g.), or when it names something the
    // vocabulary does have and the resolved boolean split answers the
    // opposite. Checking sliding-ness alone — the previous shape — let a
    // spelling like `linear_attention` pass silently as "agrees, full
    // attention" purely because neither side claims sliding: the same
    // collapse the carriage gate exists to catch, one function over.
    let disagreeing = layers
        .iter()
        .filter(|l| {
            let Some(declared) = declared_array.get(l.layer).and_then(Value::as_str) else {
                return true;
            };
            let (operator, span) = resolve_layer_kind(
                Some(declared),
                l.attention == ATTENTION_SLIDING,
                recurrence_kind(inventory),
                uses_mla(inventory),
            );
            match operator {
                // A recurrence round-trips by construction, so this arm
                // proves only that the graph records what was declared —
                // it is NOT evidence the layer really is a recurrence.
                //
                // The independent authority for that is operand evidence:
                // the op plan picks its operator from the presence of
                // `linear_attn.in_proj_qkv` and never from this spelling.
                // Those two tables cannot be compared yet — the inventory
                // carries tensor GROUP prefixes, not per-layer operand
                // names, and no hybrid container exists to plan against
                // while encode still refuses. The comparison is therefore
                // OWED at the first Qwen3.8 encode (census: 48
                // `GatedDeltaOp` + 16 `Softmax`), and this arm is a
                // recorded declaration until then. Said plainly here so
                // the arm is not read as a check it is not.
                LayerOperator::GatedDelta | LayerOperator::Kda => false,
                // A recurrence this build cannot identify does not
                // round-trip: the checkpoint declared an operator and the
                // graph carries no executable one. That is a genuine
                // disagreement, and grading it as agreement is what let
                // GLM-5.3-Flash's 34 KDA layers read as solved.
                LayerOperator::Recurrent => true,
                // The genuine comparison: the declared spelling against a
                // span the *parser's* boolean produced, not against
                // itself. MLA shares this arm — the checkpoint's
                // `layer_types` vocabulary has no separate spelling for
                // it, so the ONLY declared fact to compare is the span
                // (`full_attention`), identical to a plain softmax layer.
                LayerOperator::Softmax | LayerOperator::Mla => span
                    .map(AttentionSpan::declared_name)
                    .is_none_or(|carried| !declared.eq_ignore_ascii_case(carried)),
            }
        })
        .count();
    let agrees = disagreeing == 0 && declared_array.len() == layers.len();
    Some(Finding {
        category: if agrees {
            FindingCategory::Representable
        } else {
            FindingCategory::Mismatched
        },
        class: SemanticClass::ExecutionSemantic,
        component: component_of(&fact.path),
        subject: fact.path.clone(),
        carriage: None,
        detail: if agrees {
            {
                let recurrent = declared_array
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|d| {
                        matches!(
                            resolve_layer_kind(
                                Some(d),
                                false,
                                recurrence_kind(inventory),
                                uses_mla(inventory)
                            )
                            .0,
                            LayerOperator::GatedDelta | LayerOperator::Kda
                        )
                    })
                    .count();
                if recurrent > 0 {
                    format!(
                        "declared interleave honoured ({recurrent} recurrent / {} \
                         softmax, of {} layers)",
                        layers.len() - recurrent,
                        layers.len()
                    )
                } else {
                    format!(
                        "declared interleave honoured ({} sliding / {} full)",
                        inventory.resolved.attention.sliding_layers,
                        inventory.resolved.attention.full_layers
                    )
                }
            }
        } else {
            format!("{disagreeing} layers resolve a different attention kind than declared")
        },
        declared: Some(Value::from(declared_array.len() as u64)),
        resolved: Some(Value::from(layers.len() as u64)),
    })
}

/// Paths that are two spellings of **one** fact: `(plain, nested)`,
/// matched as suffixes within a single component.
///
/// A registered list, not a heuristic over repeated leaf names. The
/// heuristic version flagged Gemma 4, which declares `rope_theta` and
/// `rope_type` at BOTH `rope_parameters.full_attention.*` and
/// `rope_parameters.sliding_attention.*` — two facts about two layer
/// classes, correctly disagreeing, and not a checkpoint contradicting
/// itself. `a_per_layer_class_pair_is_not_a_duplicate_spelling` keeps
/// that distinction pinned.
const DUPLICATE_SPELLINGS: &[(&str, &str)] = &[(
    "partial_rotary_factor",
    "rope_parameters.partial_rotary_factor",
)];

/// A fact the checkpoint spells two ways, disagreeing with itself.
///
/// Qwen3.8 declares `partial_rotary_factor` at `text_config` and again
/// inside `text_config.rope_parameters`, and HF reads only the
/// `rope_parameters` one. Both being *present* proves nothing: the parser
/// picks one, and if the other disagrees the checkpoint states two
/// different execution semantics while the plan reports whichever was
/// read. On this checkpoint they agree (both `0.25`), so this moves no
/// count — it closes a gate that could not previously fail, the same
/// shape as the `full_attention_interval` corroboration.
fn duplicate_spelling_findings(inventory: &ArchitectureInventory) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (plain, nested) in DUPLICATE_SPELLINGS {
        let nested_suffix = format!(".{nested}");
        let plain_suffix = format!(".{plain}");
        for nested_fact in inventory
            .config_keys
            .iter()
            .filter(|f| f.path.ends_with(&nested_suffix))
        {
            let component = component_of(&nested_fact.path);
            let Some(plain_fact) = inventory.config_keys.iter().find(|f| {
                f.path.ends_with(&plain_suffix)
                    && !f.path.ends_with(&nested_suffix)
                    && component_of(&f.path) == component
            }) else {
                continue;
            };
            if plain_fact.value == nested_fact.value {
                continue;
            }
            findings.push(Finding {
                category: FindingCategory::Mismatched,
                class: SemanticClass::ExecutionSemantic,
                component,
                subject: format!("{plain} (two spellings)"),
                carriage: None,
                detail: format!(
                    "the checkpoint spells one fact two ways and they disagree: `{}`={} vs \
                     `{}`={}; a parser reads one and the other is silently dropped",
                    plain_fact.path, plain_fact.value, nested_fact.path, nested_fact.value
                ),
                declared: Some(nested_fact.value.clone()),
                resolved: Some(plain_fact.value.clone()),
            });
        }
    }
    findings
}

/// The text-path `layer_types` fact — vision towers declare their own list
/// under `vision_config`, which describes a different component.
fn semantics_is_layer_types(path: &str) -> bool {
    path == "layer_types" || path == "text_config.layer_types"
}
