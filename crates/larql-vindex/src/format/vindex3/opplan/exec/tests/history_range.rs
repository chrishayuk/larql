//! CONTINUATION-VIEW-1 V1: one retention authority.
//!
//! [`HistoryRange`] is the only place the plan's "earliest position a step
//! may still read" is decided. These witnesses hold V1 to its two claims:
//! it changes no behaviour (the table below pins it to the two formulas it
//! replaced), and no backend derives a floor itself (S2, a source scan with
//! a seeded-violation control).

use super::super::super::super::graph::policy::AttentionSpan;
use super::super::continuation::plan_continuation_geometry;
use super::super::kv::HistoryRange;
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer, G_WINDOW};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};

const SPANS: [AttentionSpan; 3] = [
    AttentionSpan::Sliding,
    AttentionSpan::Full,
    AttentionSpan::Windowed,
];
const WINDOWS: [Option<usize>; 5] = [None, Some(1), Some(3), Some(1024), Some(4096)];
const POSITIONS: usize = 2100;

/// The floor both backends computed before V1, transcribed verbatim from
/// `production::source_start` and the reference backend's local match.
fn replaced_formula(span: AttentionSpan, window: Option<usize>, position: usize) -> usize {
    match (span, window) {
        (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
        (AttentionSpan::Sliding, None) | (AttentionSpan::Full, _) => 0,
        (AttentionSpan::Windowed, _) => 0,
    }
}

#[test]
fn the_authority_reproduces_the_replaced_formula_everywhere() {
    for span in SPANS {
        for window in WINDOWS {
            let history = HistoryRange::of_span(span, window);
            for position in 0..POSITIONS {
                assert_eq!(
                    history.required_start(position),
                    replaced_formula(span, window, position),
                    "{span:?} window {window:?} position {position}"
                );
                let range = history.required_range(position);
                assert_eq!(
                    range.end,
                    position + 1,
                    "a step always reads its own position"
                );
                assert!(range.start <= position, "the range is never empty");
            }
        }
    }
}

#[test]
fn only_a_bounded_sliding_span_trails() {
    for span in SPANS {
        for window in WINDOWS {
            let trails = matches!(
                HistoryRange::of_span(span, window),
                HistoryRange::Trailing(_)
            );
            let expected = span == AttentionSpan::Sliding && window.is_some();
            assert_eq!(trails, expected, "{span:?} window {window:?}");
        }
    }
}

#[test]
fn plan_geometry_carries_the_ops_authority() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "v1-history",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let mut trailing = 0;
    for (layer, g) in plan.layers.iter().zip(&geometry) {
        let LayerAttention::Softmax(op) = &layer.attention else {
            continue;
        };
        let kv = g.kv().expect("a softmax layer keeps K/V rows");
        assert_eq!(kv.history, op.history(), "geometry and plan op must agree");
        if kv.history == HistoryRange::Trailing(G_WINDOW) {
            trailing += 1;
        }
    }
    assert_eq!(
        trailing, 1,
        "the fixture has exactly one sliding layer of window {G_WINDOW}"
    );
}

// ---- S2: no backend derives a floor itself --------------------------------

const BACKENDS: [(&str, &str); 3] = [
    ("reference.rs", include_str!("../reference.rs")),
    ("production.rs", include_str!("../production.rs")),
    ("device.rs", include_str!("../device.rs")),
];

/// Code (comments stripped) that decides a floor from a span or a window.
fn floor_derivations(source: &str) -> Vec<String> {
    const PATTERNS: [&str; 4] = [
        "AttentionSpan::",
        "saturating_sub(window",
        "call.window",
        "call.span",
    ];
    source
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let code = line.split("//").next().unwrap_or("");
            PATTERNS
                .iter()
                .find(|p| code.contains(*p))
                .map(|p| format!("line {}: `{p}` in `{}`", i + 1, code.trim()))
        })
        .collect()
}

#[test]
fn s2_no_backend_derives_a_floor_from_span_or_window() {
    for (name, source) in BACKENDS {
        let found = floor_derivations(source);
        assert!(found.is_empty(), "{name} derives a floor itself: {found:?}");
    }
}

#[test]
fn s2_scan_sees_a_seeded_derivation() {
    // The exact shape V1 removed. If the scan cannot see this, its clean
    // verdict above means nothing.
    let seeded = "        let start = match (call.span, call.window) {\n            \
                  (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),\n";
    assert!(!floor_derivations(seeded).is_empty());
    // A comment naming the old shape is not a derivation.
    assert!(floor_derivations("// was: (position + 1).saturating_sub(window)").is_empty());
}

#[test]
fn s2_the_backends_that_compute_a_start_read_the_authority() {
    for (name, source) in &BACKENDS[..2] {
        assert!(
            source.contains(".history()"),
            "{name} computes a source start and must read it from the authority"
        );
    }
}
