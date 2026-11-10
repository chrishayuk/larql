//! `Statement::verb` — the name a capability refusal reports for each
//! statement — checked against one parsed example of every statement kind.

use larql_lql::parse;

/// `(LQL source, the verb its parsed statement must report)`.
const CASES: &[(&str, &str)] = &[
    (r#"EXTRACT MODEL "m" INTO "out.vindex";"#, "EXTRACT"),
    (r#"COMPILE CURRENT INTO MODEL "out";"#, "COMPILE"),
    (r#"DIFF "a.vindex" "b.vindex";"#, "DIFF"),
    (r#"USE "a.vindex";"#, "USE"),
    (r#"WALK "The capital of France is" TOP 5;"#, "WALK"),
    (r#"INFER "[0]" COMPARE;"#, "INFER"),
    ("SELECT * FROM EDGES LIMIT 5;", "SELECT"),
    (r#"DESCRIBE "France";"#, "DESCRIBE"),
    (r#"EXPLAIN INFER "[1] [2]";"#, "EXPLAIN"),
    (
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
        "INSERT",
    ),
    (
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
        "DELETE",
    ),
    (
        r#"UPDATE EDGES SET target = "x" WHERE layer = 0 AND feature = 0;"#,
        "UPDATE",
    ),
    (
        r#"UPDATE WEIGHTS SET DELTA = 0.125 WHERE TENSOR = "26.mlp.down_proj.weight" AND ROW = 100 AND COL = 200;"#,
        "UPDATE WEIGHTS",
    ),
    (r#"MERGE "b.vindex";"#, "MERGE"),
    ("REBALANCE;", "REBALANCE"),
    ("SHOW RELATIONS;", "SHOW RELATIONS"),
    ("SHOW LAYERS;", "SHOW LAYERS"),
    ("SHOW FEATURES 26;", "SHOW FEATURES"),
    ("SHOW ENTITIES;", "SHOW ENTITIES"),
    ("SHOW MODELS;", "SHOW MODELS"),
    ("SHOW COMPONENTS;", "SHOW COMPONENTS"),
    ("SHOW REPRESENTATIONS;", "SHOW REPRESENTATIONS"),
    ("SHOW PROVENANCE;", "SHOW PROVENANCE"),
    ("SHOW AUTHORITY;", "SHOW AUTHORITY"),
    (
        "VERIFY REPRESENTATION 'c' USING CONTRACT 'x' AGAINST 's';",
        "VERIFY REPRESENTATION",
    ),
    (
        "VERIFY OPERATOR 'o' USING CONTRACT 'c' IN WITNESS 'w' AGAINST 'm';",
        "VERIFY OPERATOR",
    ),
    ("STATS;", "STATS"),
    ("SHOW COMPACT STATUS;", "SHOW COMPACT STATUS"),
    (
        r#"COMPACT INTO VINDEX "out.vindex";"#,
        "COMPACT INTO VINDEX",
    ),
    ("COMPACT MINOR;", "COMPACT MINOR"),
    ("COMPACT MAJOR;", "COMPACT MAJOR"),
    (r#"BEGIN PATCH "p.vlp";"#, "BEGIN PATCH"),
    ("SAVE PATCH;", "SAVE PATCH"),
    (r#"APPLY PATCH "p.vlp";"#, "APPLY PATCH"),
    ("SHOW PATCHES;", "SHOW PATCHES"),
    (r#"REMOVE PATCH "p.vlp";"#, "REMOVE PATCH"),
    (r#"TRACE "[1] [2]" FOR "[5]";"#, "TRACE"),
    ("SHOW MODELS |> SHOW LAYERS;", "|> (pipe)"),
];

#[test]
fn every_statement_kind_reports_its_verb() {
    let wrong: Vec<String> = CASES
        .iter()
        .filter_map(|(source, verb)| match parse(source) {
            Ok(statement) if statement.verb() == *verb => None,
            Ok(statement) => Some(format!(
                "{source}: verb {:?}, want {verb:?}",
                statement.verb()
            )),
            Err(e) => Some(format!("{source}: {e}")),
        })
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn verbs_are_distinct() {
    // A refusal names the statement by its verb, so two kinds sharing one
    // would make the refusal ambiguous.
    let mut verbs: Vec<&str> = CASES.iter().map(|(_, verb)| *verb).collect();
    verbs.sort_unstable();
    let before = verbs.len();
    verbs.dedup();
    assert_eq!(verbs.len(), before, "duplicate verb in {verbs:?}");
}
