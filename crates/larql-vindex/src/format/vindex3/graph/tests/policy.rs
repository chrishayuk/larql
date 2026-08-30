//! `resolve_layer_kind` — the layer-blind twin of `graph::build::
//! operator_and_span`, exercised when a checkpoint declares no per-layer
//! `layer_types`/interleave at all.

use crate::format::vindex3::graph::policy::{resolve_layer_kind, LayerOperator, RecurrenceKind};

/// A full (non-sliding) layer with no declared kind, on a family that
/// declares MLA, resolves to `Mla` — not the plain softmax default a
/// family without MLA would get.
#[test]
fn a_full_layer_with_no_declared_kind_is_mla_when_the_family_declares_it() {
    let (operator, span) = resolve_layer_kind(None, false, None, true);
    assert_eq!(operator, LayerOperator::Mla);
    assert!(span.is_some());
}

/// The same layer, on a family that does NOT declare MLA, stays plain
/// softmax — `mla` must be load-bearing, not a silent upgrade.
#[test]
fn a_full_layer_with_no_declared_kind_stays_softmax_without_mla() {
    let (operator, _) = resolve_layer_kind(None, false, None, false);
    assert_eq!(operator, LayerOperator::Softmax);
}

/// A sliding layer never becomes MLA, even on an MLA family — no evidence
/// in this build associates MLA with a sliding span.
#[test]
fn a_sliding_layer_stays_softmax_even_on_an_mla_family() {
    let (operator, _) = resolve_layer_kind(None, true, None, true);
    assert_eq!(operator, LayerOperator::Softmax);
}

/// A declared recurrence still resolves to its own operator regardless of
/// `mla` — the two facts are independent, and `mla` must not shadow a
/// recurrence declaration.
#[test]
fn a_declared_recurrence_ignores_the_mla_flag() {
    let (operator, span) = resolve_layer_kind(
        Some("linear_attention"),
        false,
        Some(RecurrenceKind::Kda),
        true,
    );
    assert_eq!(operator, LayerOperator::Kda);
    assert!(span.is_none(), "a recurrence has no span");
}
