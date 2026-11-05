//! The statement's shape.

use larql_lql::{parse, Statement};

fn parsed(sql: &str) -> Statement {
    parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

#[test]
fn a_single_column() {
    let Statement::UpdateWeights {
        tensor,
        row,
        cols,
        delta,
    } = parsed(
        r#"UPDATE WEIGHTS SET DELTA = 0.125 WHERE TENSOR = "26.mlp.down_proj.weight" AND ROW = 100 AND COL = 200;"#,
    )
    else {
        panic!("not UpdateWeights")
    };
    assert_eq!(
        (tensor.as_str(), row, cols, delta),
        ("26.mlp.down_proj.weight", 100, vec![200], 0.125)
    );
}

#[test]
fn a_column_list_a_negative_delta_and_any_case() {
    let Statement::UpdateWeights { cols, delta, .. } = parsed(
        r#"update weights set delta = -0.0078125 where tensor = "t" and row = 1 and col in (3, 5, 7)"#,
    ) else {
        panic!("not UpdateWeights")
    };
    assert_eq!((cols, delta), (vec![3, 5, 7], -0.0078125));
}

#[test]
fn the_words_stay_usable_as_names_elsewhere() {
    // DELTA, TENSOR, ROW and COL are read in place, not reserved.
    assert!(parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("row", "col", "delta");"#
    )
    .is_ok());
}

#[test]
fn an_edge_update_still_parses() {
    assert!(matches!(
        parsed(
            r#"UPDATE EDGES SET target = "London" WHERE entity = "John" AND relation = "lives-in";"#
        ),
        Statement::Update { .. }
    ));
}

#[test]
fn malformed_statements_are_refused() {
    for bad in [
        r#"UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = "t" AND ROW = 1;"#,
        r#"UPDATE WEIGHTS SET DELTA = 0.5 WHERE ROW = 1 AND COL = 2;"#,
        r#"UPDATE WEIGHTS SET DELTA = "x" WHERE TENSOR = "t" AND ROW = 1 AND COL = 2;"#,
        r#"UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = "t" AND ROW = -1 AND COL = 2;"#,
        r#"UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = "t" AND ROW = 1 AND COL IN ();"#,
    ] {
        assert!(parse(bad).is_err(), "should not parse: {bad}");
    }
}
