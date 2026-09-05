//! **The anti-cheat: no field without a call behind it.**
//!
//! Stage 1d walks a STORED snapshot and fails on any key that names a
//! conclusion. This walks a RENDERED response and fails on any leaf that
//! names nothing at all — the same discipline pointed the other way.
//!
//! Both directions matter. An undeclared field is a value the facade
//! invented; an unreached declaration is a registry entry that has
//! stopped describing the code, and the next field added under it would
//! inherit its alibi.

use super::super::super::state::tests::container;
use super::super::origin::{walk, Coverage, Origin, Rendered};
use super::{priced_record, priced_record_from, reloaded, view};

/// Assert both directions for one rendered value.
fn declared<T: Rendered>(rendered: &T) -> Coverage {
    let coverage = walk(rendered).expect("a view serializes");
    assert!(
        coverage.undeclared.is_empty(),
        "fields rendered with no substrate call behind them: {:?}",
        coverage.undeclared
    );
    let origins = T::origins();
    assert!(
        coverage.unreached(&origins).is_empty(),
        "origins declared but never rendered: {:?}",
        coverage.unreached(&origins)
    );
    coverage
}

/// The same two directions over several renders of one view type.
///
/// Undeclared fields are caught per render; unreached declarations are
/// caught against the UNION, because a type with alternative shapes can
/// only describe all of them across all of them.
fn declared_across<T: Rendered>(rendered: &[T]) {
    let mut covered = std::collections::BTreeSet::new();
    for one in rendered {
        let coverage = walk(one).expect("a view serializes");
        assert!(
            coverage.undeclared.is_empty(),
            "fields rendered with no substrate call behind them: {:?}",
            coverage.undeclared
        );
        covered.extend(coverage.covered);
    }
    let origins = T::origins();
    let unreached: Vec<&str> = origins
        .iter()
        .map(|o| o.field.as_str())
        .filter(|f| !covered.contains(*f))
        .collect();
    assert!(
        unreached.is_empty(),
        "origins declared but never rendered by any shape: {unreached:?}"
    );
}

#[test]
fn every_rendered_field_names_the_call_that_produced_it() {
    let snap = reloaded();
    let facade = view(&snap);
    let p = &snap.frontier()[0].state.clone();
    let (left, right) = {
        let edge = snap.graph().edges().next().expect("the record has edges");
        (edge.parent().clone(), edge.child().clone())
    };

    declared(&facade.describe());
    declared(&facade.current());
    declared(&facade.frontier());
    declared(&facade.explain(p).expect("the graph holds it"));
    declared(&facade.compare(&left, &right).expect("both are held"));
    declared(&facade.evidence(None));
    declared(&facade.evidence(Some(p)));
    // `next_experiment` answers in THREE shapes and the registry
    // describes all three, so coverage is their union: a variant that
    // never renders leaves its declarations unreached, which is the
    // same stale-registry defect seen from the other side. The refusing
    // record cannot reach `Available`, and only a record with real
    // accounting facts can.
    let dir = container::dense();
    let answering = priced_record(dir.path());
    let closed = priced_record_from(
        dir.path(),
        ["compile-all", "compile-all-by-another-name"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    declared_across(&[
        facade.next_experiment(),
        view(&answering).next_experiment(),
        view(&closed).next_experiment(),
    ]);
}

#[test]
fn a_field_with_no_declaration_is_caught() {
    #[derive(serde::Serialize)]
    struct Smuggled {
        recommendation: &'static str,
    }
    impl Rendered for Smuggled {
        fn origins() -> Vec<Origin> {
            Vec::new()
        }
    }

    let coverage = walk(&Smuggled {
        recommendation: "try E24 next",
    })
    .expect("serializes");
    assert_eq!(
        coverage.undeclared.iter().collect::<Vec<_>>(),
        vec!["recommendation"],
        "an undeclared leaf must be reported, or the check proves nothing"
    );
}

#[test]
fn a_declaration_for_a_field_that_is_gone_is_caught() {
    #[derive(serde::Serialize)]
    struct Shrunk {
        kept: u8,
    }
    impl Rendered for Shrunk {
        fn origins() -> Vec<Origin> {
            vec![
                Origin::new("kept", "still here"),
                Origin::new("dropped", "removed last week"),
            ]
        }
    }

    let coverage = walk(&Shrunk { kept: 1 }).expect("serializes");
    assert!(coverage.undeclared.is_empty());
    assert_eq!(coverage.unreached(&Shrunk::origins()), vec!["dropped"]);
}

#[test]
fn an_embedded_substrate_type_is_covered_whole() {
    #[derive(serde::Serialize)]
    struct Nested {
        margin: Inner,
    }
    #[derive(serde::Serialize)]
    struct Inner {
        limit: f64,
        observed: f64,
    }
    impl Rendered for Nested {
        fn origins() -> Vec<Origin> {
            vec![Origin::new("margin", "ConstraintVector.margins")]
        }
    }

    let coverage = walk(&Nested {
        margin: Inner {
            limit: 3.5e-3,
            observed: 3.6e-3,
        },
    })
    .expect("serializes");
    assert!(
        coverage.undeclared.is_empty(),
        "descent must stop at a declared subtree, not walk into the substrate's own fields"
    );
    assert!(coverage.covered.contains("margin"));
}

#[test]
fn an_absent_field_is_covered_by_what_was_declared_beneath_it() {
    #[derive(serde::Serialize)]
    struct Maybe {
        incumbent: Option<u8>,
        orphan: Option<u8>,
    }
    impl Rendered for Maybe {
        fn origins() -> Vec<Origin> {
            vec![Origin::new("incumbent.state", "FrontierEntry.state")]
        }
    }

    let coverage = walk(&Maybe {
        incumbent: None,
        orphan: None,
    })
    .expect("serializes");
    assert_eq!(
        coverage.undeclared.iter().collect::<Vec<_>>(),
        vec!["orphan"],
        "an absence is excused by a declaration beneath it and by nothing else"
    );
}

#[test]
fn an_empty_collection_is_covered_by_its_declared_shape() {
    #[derive(serde::Serialize)]
    struct Empties {
        failures: Vec<u8>,
        orphan: Vec<u8>,
    }
    impl Rendered for Empties {
        fn origins() -> Vec<Origin> {
            vec![Origin::new("failures[].criterion", "Margin.criterion")]
        }
    }

    let coverage = walk(&Empties {
        failures: Vec::new(),
        orphan: Vec::new(),
    })
    .expect("serializes");
    assert_eq!(
        coverage.undeclared.iter().collect::<Vec<_>>(),
        vec!["orphan"]
    );
}

#[test]
fn an_origin_re_roots_under_a_container() {
    let origin = Origin::new("admitted", "FrontierEntry::admitted()");
    assert_eq!(origin.under("states[]").field, "states[].admitted");
    assert_eq!(
        origin.under("").field,
        "admitted",
        "an empty prefix must not grow a separator"
    );
    assert_eq!(origin.under("states[]").call, origin.call);
}
