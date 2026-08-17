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

use super::report::{Finding, FindingCategory, SemanticClass};
use super::semantics::component_of;

/// Sliding layer label used in the inventory's per-layer table.
const ATTENTION_SLIDING: &str = "sliding";

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

/// Declared RoPE base, in the parser's own specificity order. Returns the
/// path that supplied it so a mismatch names its source.
fn declared_rope_theta(facts: &[ConfigKeyFact]) -> Option<(String, f64)> {
    const CANDIDATES: &[&str] = &[
        "text_config.rope_parameters.full_attention.rope_theta",
        "rope_parameters.full_attention.rope_theta",
        "text_config.rope_parameters.rope_theta",
        "rope_parameters.rope_theta",
        "text_config.rope_theta",
        "rope_theta",
    ];
    CANDIDATES.iter().find_map(|path| {
        facts
            .iter()
            .find(|f| f.path == *path)
            .and_then(|f| f.value.as_f64().map(|v| (f.path.clone(), v)))
    })
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

    findings.extend(rope_theta_finding(inventory));
    findings.extend(layer_rope_theta_findings(inventory));
    findings.extend(layer_types_finding(inventory));
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

/// Uniform declared θ vs the resolved policy of every layer.
fn rope_theta_finding(inventory: &ArchitectureInventory) -> Option<Finding> {
    let (path, declared_theta) = declared_rope_theta(&inventory.config_keys)?;
    let layers = &inventory.resolved.layers;
    if layers.is_empty() {
        return None;
    }
    // A per-layer declaration overrides the uniform one; that comparator
    // owns the answer then.
    if inventory
        .config_keys
        .iter()
        .any(|f| f.path.ends_with("layer_rope_theta"))
    {
        return None;
    }
    let disagreeing = layers
        .iter()
        .filter(|l| l.position.rope_theta() != Some(declared_theta))
        .count();
    Some(value_finding(
        &path,
        disagreeing == 0,
        SemanticClass::ExecutionSemantic,
        declared_theta.into(),
        serde_json::to_value(layers[0].position).ok(),
    ))
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
    const SLIDING_DECLARATION: &str = "sliding_attention";
    let fact = inventory
        .config_keys
        .iter()
        .find(|f| semantics_is_layer_types(&f.path))?;
    let declared_array = fact.value.as_array()?;
    let layers = &inventory.resolved.layers;
    let disagreeing = layers
        .iter()
        .filter(|l| {
            declared_array
                .get(l.layer)
                .and_then(Value::as_str)
                .is_some_and(|declared| {
                    declared.eq_ignore_ascii_case(SLIDING_DECLARATION)
                        != (l.attention == ATTENTION_SLIDING)
                })
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
            format!(
                "declared interleave honoured ({} sliding / {} full)",
                inventory.resolved.attention.sliding_layers,
                inventory.resolved.attention.full_layers
            )
        } else {
            format!("{disagreeing} layers resolve a different attention kind than declared")
        },
        declared: Some(Value::from(declared_array.len() as u64)),
        resolved: Some(Value::from(layers.len() as u64)),
    })
}

/// The text-path `layer_types` fact — vision towers declare their own list
/// under `vision_config`, which describes a different component.
fn semantics_is_layer_types(path: &str) -> bool {
    path == "layer_types" || path == "text_config.layer_types"
}
