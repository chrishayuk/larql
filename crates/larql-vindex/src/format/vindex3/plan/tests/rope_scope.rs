//! The one derivation of which layers a declared RoPE base speaks for.

use larql_models::inventory::{ConfigKeyFact, KeyStatus};

use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::plan::rope_scope::{is_rope_base, rope_base_scope, RopeBaseScope};

const FULL_SECTION: &str = "rope_parameters.full_attention.rope_theta";
const SLIDING_SECTION: &str = "rope_parameters.sliding_attention.rope_theta";
const LOCAL_BASE: &str = "rope_local_base_freq";
const FLAT_BASE: &str = "rope_theta";
const THETA: f64 = 10_000.0;

fn facts(paths: &[&str]) -> Vec<ConfigKeyFact> {
    paths
        .iter()
        .map(|path| ConfigKeyFact {
            path: (*path).to_string(),
            value: serde_json::json!(THETA),
            status: KeyStatus::Consumed,
        })
        .collect()
}

#[test]
fn a_lone_flat_base_speaks_for_the_whole_stack() {
    let facts = facts(&[FLAT_BASE]);
    assert_eq!(rope_base_scope(FLAT_BASE, &facts), RopeBaseScope::Stack);
    assert_eq!(RopeBaseScope::Stack.probe_span(), None);
}

#[test]
fn a_section_names_its_own_span() {
    let facts = facts(&[FULL_SECTION, SLIDING_SECTION]);
    assert_eq!(
        rope_base_scope(SLIDING_SECTION, &facts),
        RopeBaseScope::Span(AttentionSpan::Sliding)
    );
    assert_eq!(
        rope_base_scope(FULL_SECTION, &facts),
        RopeBaseScope::Span(AttentionSpan::Full)
    );
}

/// The flat Gemma 3 pair: the local base is the sliding span's, and the
/// `rope_theta` beside it narrows to the full span.
#[test]
fn the_flat_pair_splits_by_span() {
    let facts = facts(&[FLAT_BASE, LOCAL_BASE]);
    assert_eq!(
        rope_base_scope(LOCAL_BASE, &facts),
        RopeBaseScope::Span(AttentionSpan::Sliding)
    );
    assert_eq!(
        rope_base_scope(FLAT_BASE, &facts),
        RopeBaseScope::Span(AttentionSpan::Full)
    );
    assert_eq!(
        RopeBaseScope::Span(AttentionSpan::Full).probe_span(),
        Some(AttentionSpan::Full)
    );
}

/// A full-span section overrides the flat base, which then speaks for
/// no layer on its own.
#[test]
fn a_full_section_supersedes_the_flat_base() {
    let facts = facts(&[FLAT_BASE, LOCAL_BASE, FULL_SECTION]);
    assert_eq!(
        rope_base_scope(FLAT_BASE, &facts),
        RopeBaseScope::Superseded
    );
    assert_eq!(RopeBaseScope::Superseded.probe_span(), None);
}

/// Siblings only count within one component: a tower's local base does
/// not narrow the text stack's.
#[test]
fn another_components_base_does_not_narrow() {
    let facts = facts(&[FLAT_BASE, "vision_config.rope_local_base_freq"]);
    assert_eq!(rope_base_scope(FLAT_BASE, &facts), RopeBaseScope::Stack);
}

#[test]
fn only_rope_bases_are_recognised() {
    assert!(is_rope_base(FLAT_BASE));
    assert!(is_rope_base(LOCAL_BASE));
    assert!(is_rope_base(SLIDING_SECTION));
    assert!(!is_rope_base("sliding_window"));
}
