//! CONTINUATION-VIEW-1 V2: the logical continuation view.
//!
//! The view's invariants (`base <= end`, reads only inside `[base, end)`,
//! `end` logical), that a non-row backing and a retained range read the
//! same rows as the row-backed path, and the two structural falsifiers V2
//! closes: S1 (no absolute indexing into provider rows outside the view)
//! and S3 (a step whose view lacks a row the plan requires is refused by
//! name before any backend runs).

use super::super::backend::AttentionStepCall;
use super::super::kv_view::{KvRows, KvView, ViewRefusal};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedAttention, PreparedOperands};
use super::super::reference::ReferenceBackend;
use crate::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_HIDDEN, G_WINDOW,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::plan_component_ops;

fn rows(n: usize, width: usize, salt: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|p| {
            (0..width)
                .map(|c| salt + p as f32 * 10.0 + c as f32)
                .collect()
        })
        .collect()
}

/// A contiguous store: row p at `data[(p - base) * width ..]` — the shape
/// canonical/v1's matrix will lend in V3, read here through `KvRows`.
struct Contiguous {
    base: usize,
    width: usize,
    data: Vec<f32>,
}

impl KvRows for Contiguous {
    fn row(&self, position: usize) -> &[f32] {
        let start = (position - self.base) * self.width;
        &self.data[start..start + self.width]
    }
}

fn contiguous(base: usize, rows: &[Vec<f32>]) -> Contiguous {
    Contiguous {
        base,
        width: rows[0].len(),
        data: rows.iter().flatten().copied().collect(),
    }
}

// ---- the view's invariants ------------------------------------------------

#[test]
fn a_view_refuses_a_base_past_its_end() {
    let store = contiguous(0, &rows(2, 4, 0.0));
    let refused = KvView::new(5, 3, &store, &store).unwrap_err();
    assert_eq!((refused.base, refused.end), (5, 3));
}

#[test]
fn every_backing_reads_the_same_rows_by_absolute_position() {
    let (k, v) = (rows(9, 4, 0.0), rows(9, 4, 0.5));
    let full = KvView::over_rows(&k, &v);
    let (ck, cv) = (contiguous(0, &k), contiguous(0, &v));
    let dynamic = KvView::new(0, 9, &ck, &cv).unwrap();
    // A retained range: only positions 5.. are held, by either backing.
    let retained_rows = KvView::rows_from(5, &k[5..], &v[5..]).unwrap();
    let (rk, rv) = (contiguous(5, &k[5..]), contiguous(5, &v[5..]));
    let retained_dyn = KvView::new(5, 9, &rk, &rv).unwrap();
    for p in 0..9 {
        assert_eq!(full.key(p), dynamic.key(p));
        assert_eq!(full.value(p), dynamic.value(p));
    }
    for view in [retained_rows, retained_dyn] {
        assert_eq!((view.base(), view.end()), (5, 9));
        for p in 5..9 {
            assert_eq!(view.key(p), full.key(p), "absolute position {p}");
            assert_eq!(view.value(p), full.value(p));
        }
    }
}

#[test]
fn covers_names_what_is_missing() {
    let (k, v) = (rows(4, 2, 0.0), rows(4, 2, 0.0));
    let view = KvView::rows_from(6, &k, &v).unwrap(); // holds [6, 10)
    assert!(view.covers(6..10).is_ok());
    assert!(
        view.covers(8..8).is_ok(),
        "an empty requirement is always covered"
    );
    let refused = view.covers(5..10).unwrap_err();
    assert_eq!(
        refused,
        ViewRefusal {
            needed: 5..10,
            base: 6,
            end: 10
        }
    );
    assert!(refused.to_string().contains("[6, 10)"), "{refused}");
    assert!(view.covers(6..11).is_err(), "past end is refused too");
}

#[test]
#[should_panic(expected = "executor bug: read position 5 outside the checked view [6, 10)")]
fn a_read_below_base_is_named_not_an_index_panic() {
    let (k, v) = (rows(4, 2, 0.0), rows(4, 2, 0.0));
    let view = KvView::rows_from(6, &k, &v).unwrap();
    let _ = view.key(5);
}

// ---- S3: a step lacking a required row is refused before any backend ------

struct Sliding {
    _container: tempfile::TempDir,
    ops: PreparedOperands,
    plan: crate::format::vindex3::opplan::ComponentOpPlan,
}

/// The glimmer fixture: layer 0 is a sliding window of `G_WINDOW`.
fn sliding() -> Sliding {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "v2-view",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let ops = PreparedOperands::load(
        &plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    Sliding {
        _container: container,
        ops,
        plan,
    }
}

#[test]
fn s3_a_view_missing_a_required_row_cannot_become_a_step() {
    let fixture = sliding();
    let layer = &fixture.plan.layers[0];
    let PreparedAttention::Softmax(attention) = &fixture.ops.layers()[0].attention else {
        panic!("layer 0 of the fixture is softmax");
    };
    let op = layer.attention.softmax().unwrap();
    let input = vec![vec![0.25f32; G_HIDDEN]];
    let call = || attention.call(op, &input, layer.declared_norm_eps, G_HIDDEN);

    let position = 7;
    let required_start = position + 1 - G_WINDOW; // 5
    let (k, v) = (rows(position, 4, 0.0), rows(position, 4, 0.0));

    // Holding exactly the required rows [5, 7) is enough.
    let exact = KvView::rows_from(required_start, &k[required_start..], &v[required_start..]);
    assert!(AttentionStepCall::new(call(), position, exact.unwrap()).is_ok());
    // Holding everything is enough (a provider may retain a superset).
    assert!(AttentionStepCall::new(call(), position, KvView::over_rows(&k, &v)).is_ok());

    // Dropping ONE required row is refused, by name, before any backend.
    let short = KvView::rows_from(
        required_start + 1,
        &k[required_start + 1..],
        &v[required_start + 1..],
    );
    let refused = AttentionStepCall::new(call(), position, short.unwrap())
        .err()
        .unwrap();
    assert_eq!(refused.needed, required_start..position);
    assert_eq!(refused.base, required_start + 1);

    // A view that does not end at the step's position is refused too.
    let behind = KvView::over_rows(&k[..position - 1], &v[..position - 1]);
    assert!(AttentionStepCall::new(call(), position, behind).is_err());
}

#[test]
fn s3_a_full_span_layer_requires_every_earlier_row() {
    let fixture = sliding();
    let layer = &fixture.plan.layers[1]; // the full-span layer
    let PreparedAttention::Softmax(attention) = &fixture.ops.layers()[1].attention else {
        panic!("layer 1 of the fixture is softmax");
    };
    let op = layer.attention.softmax().unwrap();
    let input = vec![vec![0.25f32; G_HIDDEN]];
    let position = 7;
    let (k, v) = (rows(position, 4, 0.0), rows(position, 4, 0.0));
    let from_one = KvView::rows_from(1, &k[1..], &v[1..]).unwrap();
    let call = attention.call(op, &input, layer.declared_norm_eps, G_HIDDEN);
    let refused = AttentionStepCall::new(call, position, from_one)
        .err()
        .unwrap();
    assert_eq!(refused.needed, 0..position);
}

// ---- S1: no absolute indexing into provider rows outside the view ---------

const SOURCES: [(&str, &str); 6] = [
    ("reference.rs", include_str!("../reference.rs")),
    ("production.rs", include_str!("../production.rs")),
    ("device.rs", include_str!("../device.rs")),
    ("conv_qkv.rs", include_str!("../conv_qkv.rs")),
    ("mod.rs", include_str!("../mod.rs")),
    ("decode.rs", include_str!("../decode.rs")),
];

/// Code (comments stripped) that indexes a provider's rows by position
/// instead of reading them through the view.
fn absolute_indexing(source: &str) -> Vec<String> {
    const PATTERNS: [&str; 6] = [
        "step.keys",
        "step.values",
        "past_keys[",
        "past_values[",
        "keys(layer_index)[",
        "keys(index)[",
    ];
    source
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let code = line.split("//").next().unwrap_or("");
            PATTERNS
                .iter()
                .find(|p| code.contains(*p))
                .map(|p| format!("line {}: `{p}`", i + 1))
        })
        .collect()
}

#[test]
fn s1_no_provider_row_is_indexed_outside_the_view() {
    for (name, source) in SOURCES {
        let found = absolute_indexing(source);
        assert!(
            found.is_empty(),
            "{name} indexes provider rows directly: {found:?}"
        );
    }
}

#[test]
fn s1_scan_sees_a_seeded_index() {
    assert!(!absolute_indexing("                    step.keys[p].as_slice()").is_empty());
    assert!(!absolute_indexing("            &past_keys[index]").is_empty());
    assert!(absolute_indexing("// was: step.keys[p].as_slice()").is_empty());
}
