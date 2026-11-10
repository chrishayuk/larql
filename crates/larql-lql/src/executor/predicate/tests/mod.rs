//! Gates for the predicate grammar: what parses into which filter, that
//! anything else refuses the whole clause, and that the two shared
//! matchers mean what every verb relies on.

use super::*;

fn cond(field: &str, op: CompareOp, value: Value) -> Condition {
    Condition {
        field: field.into(),
        op,
        value,
    }
}

fn string(s: &str) -> Value {
    Value::String(s.into())
}

fn refused(conditions: &[Condition], scope: PredicateScope) -> String {
    EdgePredicate::parse(conditions, scope)
        .expect_err("the clause must be refused")
        .to_string()
}

// ── Accepted clauses ─────────────────────────────────────────────────────

#[test]
fn an_empty_clause_constrains_nothing() {
    let predicate = EdgePredicate::parse(&[], PredicateScope::EDGES).unwrap();
    assert_eq!(predicate, EdgePredicate::default());
}

#[test]
fn every_edges_field_parses_into_its_filter() {
    let conditions = [
        cond("layer", CompareOp::Eq, Value::Integer(5)),
        cond("feature", CompareOp::Eq, Value::Integer(7)),
        cond("entity", CompareOp::Eq, string("France")),
        cond("relation", CompareOp::Like, string("capital%")),
        cond("score", CompareOp::Gt, Value::Number(0.5)),
    ];

    let predicate = EdgePredicate::parse(&conditions, PredicateScope::EDGES).unwrap();

    assert_eq!(predicate.layer, Some(5));
    assert_eq!(predicate.feature, Some(7));
    assert_eq!(predicate.entity, Some("France"));
    assert_eq!(
        predicate.relation,
        Some(RelationPredicate {
            op: RelationOp::Like,
            wanted: "capital%"
        })
    );
    assert_eq!(
        predicate.score,
        Some(ScorePredicate {
            op: ScoreOp::Gt,
            threshold: 0.5
        })
    );
}

#[test]
fn confidence_is_score_and_an_integer_threshold_is_a_number() {
    let conditions = [cond("confidence", CompareOp::Lte, Value::Integer(1))];
    let predicate = EdgePredicate::parse(&conditions, PredicateScope::EDGES).unwrap();
    assert_eq!(
        predicate.score,
        Some(ScorePredicate {
            op: ScoreOp::Lte,
            threshold: 1.0
        })
    );
}

#[test]
fn token_is_entity_where_the_scope_lists_it() {
    let conditions = [cond("token", CompareOp::Eq, string("Paris"))];
    for scope in [PredicateScope::FEATURES, PredicateScope::ENTITIES] {
        let predicate = EdgePredicate::parse(&conditions, scope).unwrap();
        assert_eq!(predicate.entity, Some("Paris"));
    }
}

#[test]
fn each_relation_and_score_operator_is_accepted() {
    for op in [CompareOp::Eq, CompareOp::Neq, CompareOp::Like] {
        let conditions = [cond("relation", op, string("capital"))];
        assert!(EdgePredicate::parse(&conditions, PredicateScope::EDGES).is_ok());
    }
    for op in [
        CompareOp::Eq,
        CompareOp::Neq,
        CompareOp::Gt,
        CompareOp::Lt,
        CompareOp::Gte,
        CompareOp::Lte,
    ] {
        let conditions = [cond("score", op, Value::Number(0.5))];
        assert!(EdgePredicate::parse(&conditions, PredicateScope::EDGES).is_ok());
    }
}

// ── Refused clauses ──────────────────────────────────────────────────────

#[test]
fn an_unknown_field_is_refused_and_the_usable_fields_are_named() {
    let err = refused(
        &[cond("colour", CompareOp::Eq, string("red"))],
        PredicateScope::EDGES,
    );
    assert!(err.contains("unsupported WHERE clause"), "{err}");
    assert!(err.contains("`colour` is not a field of EDGES"), "{err}");
    assert!(
        err.contains("layer, feature, entity, relation, score"),
        "{err}"
    );
}

#[test]
fn a_field_outside_the_scope_is_refused() {
    let score = [cond("score", CompareOp::Gt, Value::Number(0.5))];
    assert!(refused(&score, PredicateScope::FEATURES).contains("not a field of FEATURES"));
    let feature = [cond("feature", CompareOp::Eq, Value::Integer(1))];
    assert!(refused(&feature, PredicateScope::ENTITIES).contains("not a field of ENTITIES"));
    let token = [cond("token", CompareOp::Eq, string("Paris"))];
    assert!(refused(&token, PredicateScope::EDGES).contains("not a field of EDGES"));
}

#[test]
fn field_names_are_case_sensitive_so_a_miscased_field_cannot_be_dropped() {
    let err = refused(
        &[cond("Entity", CompareOp::Eq, string("France"))],
        PredicateScope::EDGES,
    );
    assert!(err.contains("`Entity` is not a field"), "{err}");
}

#[test]
fn one_invalid_condition_refuses_a_clause_of_otherwise_valid_ones() {
    let conditions = [
        cond("layer", CompareOp::Eq, Value::Integer(0)),
        cond("feature", CompareOp::Eq, Value::Integer(0)),
        cond("colour", CompareOp::Eq, string("red")),
    ];
    assert!(refused(&conditions, PredicateScope::EDGES).contains("colour"));
}

#[test]
fn a_field_constrained_twice_is_refused_as_an_unevaluated_range() {
    let conditions = [
        cond("layer", CompareOp::Eq, Value::Integer(20)),
        cond("layer", CompareOp::Eq, Value::Integer(30)),
    ];
    let err = refused(&conditions, PredicateScope::EDGES);
    assert!(
        err.contains("`layer` is constrained more than once"),
        "{err}"
    );
}

#[test]
fn score_and_confidence_together_are_the_same_field_twice() {
    let conditions = [
        cond("score", CompareOp::Gt, Value::Number(0.1)),
        cond("confidence", CompareOp::Lt, Value::Number(0.9)),
    ];
    let err = refused(&conditions, PredicateScope::EDGES);
    assert!(
        err.contains("`score` is constrained more than once"),
        "{err}"
    );
}

#[test]
fn layer_and_feature_accept_only_equality_on_a_non_negative_integer() {
    for field in ["layer", "feature"] {
        for bad in [
            cond(field, CompareOp::Gt, Value::Integer(20)),
            cond(field, CompareOp::Lte, Value::Integer(20)),
            cond(field, CompareOp::Neq, Value::Integer(20)),
            cond(field, CompareOp::Eq, Value::Integer(-1)),
            cond(field, CompareOp::Eq, Value::Number(1.5)),
            cond(field, CompareOp::Eq, string("oops")),
            cond(field, CompareOp::In, Value::List(vec![Value::Integer(1)])),
        ] {
            let err = refused(std::slice::from_ref(&bad), PredicateScope::EDGES);
            assert!(
                err.contains(&format!("{field} = <non-negative integer>")),
                "{err}"
            );
        }
    }
}

#[test]
fn entity_accepts_only_equality_on_a_string() {
    for bad in [
        cond("entity", CompareOp::Neq, string("France")),
        cond("entity", CompareOp::Like, string("Fr%")),
        cond("entity", CompareOp::Eq, Value::Integer(5)),
        cond(
            "entity",
            CompareOp::In,
            Value::List(vec![string("France"), string("Germany")]),
        ),
    ] {
        let err = refused(std::slice::from_ref(&bad), PredicateScope::EDGES);
        assert!(err.contains("entity = <string>"), "{err}");
    }
}

#[test]
fn relation_refuses_ordering_operators_and_non_strings() {
    for bad in [
        cond("relation", CompareOp::Gt, string("capital")),
        cond(
            "relation",
            CompareOp::In,
            Value::List(vec![string("capital")]),
        ),
        cond("relation", CompareOp::Eq, Value::Integer(1)),
    ] {
        let err = refused(std::slice::from_ref(&bad), PredicateScope::EDGES);
        assert!(err.contains("relation {= | != | LIKE} <string>"), "{err}");
    }
}

#[test]
fn score_refuses_like_in_and_non_numbers() {
    for bad in [
        cond("score", CompareOp::Like, Value::Number(0.5)),
        cond(
            "score",
            CompareOp::In,
            Value::List(vec![Value::Number(0.5)]),
        ),
        cond("score", CompareOp::Eq, string("high")),
    ] {
        let err = refused(std::slice::from_ref(&bad), PredicateScope::EDGES);
        assert!(
            err.contains("score {= | != | < | <= | > | >=} <number>"),
            "{err}"
        );
    }
}

#[test]
fn a_refusal_quotes_the_condition_as_written() {
    let err = refused(
        &[cond("layer", CompareOp::Gte, Value::Integer(20))],
        PredicateScope::EDGES,
    );
    assert!(err.contains("`layer >= 20` is not evaluated"), "{err}");
}

// ── Matchers ─────────────────────────────────────────────────────────────

fn relation(op: RelationOp, wanted: &str) -> RelationPredicate<'_> {
    RelationPredicate { op, wanted }
}

#[test]
fn relation_equality_overlaps_in_either_direction_ignoring_case() {
    let eq = relation(RelationOp::Eq, "Capital");
    assert!(eq.matches("capital_of"));
    assert!(relation(RelationOp::Eq, "capital_of").matches("capital"));
    assert!(!relation(RelationOp::Eq, "actor").matches("director"));
}

#[test]
fn an_unlabelled_feature_satisfies_no_relation_condition() {
    for op in [RelationOp::Eq, RelationOp::Neq, RelationOp::Like] {
        assert!(!relation(op, "capital").matches(""));
        assert!(!relation(op, "%").matches(""));
    }
}

#[test]
fn relation_inequality_selects_features_labelled_something_else() {
    let neq = relation(RelationOp::Neq, "capital");
    assert!(neq.matches("director"));
    assert!(!neq.matches("capital_of"));
}

#[test]
fn relation_like_honours_leading_and_trailing_wildcards() {
    let like = |pattern| relation(RelationOp::Like, pattern);
    assert!(like("%capital%").matches("capital_of"));
    assert!(like("%of").matches("capital_of"));
    assert!(like("capital%").matches("capital_of"));
    assert!(like("capital").matches("capital"));
    assert!(like("%").matches("anything"));
    assert!(!like("%xyz").matches("capital_of"));
    assert!(!like("xyz%").matches("capital_of"));
    assert!(!like("different").matches("capital_of"));
}

#[test]
fn each_score_operator_compares_against_the_threshold() {
    let score = |op| ScorePredicate { op, threshold: 0.5 };
    assert!(score(ScoreOp::Gt).matches(0.6) && !score(ScoreOp::Gt).matches(0.5));
    assert!(score(ScoreOp::Lt).matches(0.4) && !score(ScoreOp::Lt).matches(0.5));
    assert!(score(ScoreOp::Gte).matches(0.5) && !score(ScoreOp::Gte).matches(0.4));
    assert!(score(ScoreOp::Lte).matches(0.5) && !score(ScoreOp::Lte).matches(0.6));
}

#[test]
fn score_equality_uses_the_tolerance_band() {
    let within = SCORE_EQ_TOLERANCE / 2.0;
    let eq = ScorePredicate {
        op: ScoreOp::Eq,
        threshold: 0.5,
    };
    let neq = ScorePredicate {
        op: ScoreOp::Neq,
        threshold: 0.5,
    };
    assert!(eq.matches(0.5 + within) && !eq.matches(0.6));
    assert!(neq.matches(0.6) && !neq.matches(0.5 + within));
}

#[test]
fn a_nan_score_satisfies_no_score_condition() {
    // A NaN compares false with everything; `!=` must not turn that
    // into a match, or a corrupt score would be selected by exclusion.
    for op in [
        ScoreOp::Eq,
        ScoreOp::Neq,
        ScoreOp::Gt,
        ScoreOp::Lt,
        ScoreOp::Gte,
        ScoreOp::Lte,
    ] {
        let predicate = ScorePredicate { op, threshold: 0.5 };
        assert!(!predicate.matches(f32::NAN), "{op:?}");
    }
}
