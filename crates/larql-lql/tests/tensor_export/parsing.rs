//! The statements' shapes, and what a malformed one is told.

use larql_lql::{parse, CapabilityProfile, Statement};

fn parsed(sql: &str) -> Statement {
    parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn parse_error(sql: &str) -> String {
    match parse(sql) {
        Ok(stmt) => panic!("{sql} parsed as {stmt:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn show_tensors_with_and_without_a_filter() {
    assert!(matches!(
        parsed("SHOW TENSORS;"),
        Statement::ShowTensors { filter: None }
    ));
    let Statement::ShowTensors { filter } = parsed("show tensors \"target.embedding\"") else {
        panic!("not ShowTensors")
    };
    assert_eq!(filter.as_deref(), Some("target.embedding"));
}

#[test]
fn export_weight_in_its_plainest_form() {
    let Statement::ExportWeight {
        tensor,
        rows,
        columns,
        output,
        bits,
    } = parsed(r#"EXPORT WEIGHT "0.mlp.down_proj.weight" TO "down.npy";"#)
    else {
        panic!("not ExportWeight")
    };
    assert_eq!(
        (tensor.as_str(), rows, columns, output.as_str(), bits),
        ("0.mlp.down_proj.weight", None, None, "down.npy", false)
    );
}

#[test]
fn export_weight_with_every_clause_in_any_case() {
    let Statement::ExportWeight {
        tensor,
        rows,
        columns,
        output,
        bits,
    } = parsed(
        r#"export weight "target.embedding@BF16::weight" rows (7, 2) columns (0, 5, 1) to "e.npy" as bits"#,
    )
    else {
        panic!("not ExportWeight")
    };
    assert_eq!(tensor, "target.embedding@BF16::weight");
    assert_eq!(rows, Some(vec![7, 2]));
    assert_eq!(columns, Some(vec![0, 5, 1]));
    assert_eq!(output, "e.npy");
    assert!(bits);
}

#[test]
fn malformed_exports_are_parse_errors() {
    for (sql, says) in [
        (r#"EXPORT "t" TO "x.npy";"#, "WEIGHT"),
        (r#"EXPORT WEIGHT "t" "x.npy";"#, "TO"),
        (
            r#"EXPORT WEIGHT "t" COLUMNS (1) ROWS (2) TO "x.npy";"#,
            "TO",
        ),
        (r#"EXPORT WEIGHT "t" ROWS () TO "x.npy";"#, ""),
        (r#"EXPORT WEIGHT "t" TO "x.npy" AS VALUES;"#, "BITS"),
        (r#"EXPORT WEIGHT t TO "x.npy";"#, ""),
    ] {
        let err = parse_error(sql);
        assert!(err.contains(says), "{sql}: {err}");
    }
}

#[test]
fn a_mistyped_show_names_tensors_among_its_continuations() {
    let err = parse_error("SHOW TENSOR;");
    assert!(err.contains("TENSORS"), "{err}");
}

#[test]
fn the_words_stay_usable_as_names_elsewhere() {
    // EXPORT, WEIGHT, ROWS, COLUMNS, AS, BITS and TENSORS are read in place, not reserved.
    assert!(parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("export", "rows", "bits");"#
    )
    .is_ok());
}

#[test]
fn only_export_writes_and_the_public_profile_refuses_it() {
    let show = parsed("SHOW TENSORS;");
    let export = parsed(r#"EXPORT WEIGHT "t" TO "x.npy";"#);
    assert!(!show.writes_artifact());
    assert!(export.writes_artifact());
    assert_eq!(show.verb(), "SHOW TENSORS");
    assert_eq!(export.verb(), "EXPORT WEIGHT");
    assert!(CapabilityProfile::PublicExplorer.check(&show).is_ok());
    assert!(CapabilityProfile::PublicExplorer.check(&export).is_err());
}
