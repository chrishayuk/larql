//! Parse errors that say what would have been accepted, and where the
//! syntax is. What they list comes from the help catalogue, so each list is
//! checked here against the catalogue and not against a copy of it.

use larql_lql::help::{grammar, lookup};

use crate::support::*;

/// A word no statement and no continuation is close to in spelling.
const NOT_A_WORD: &str = "QQQQQQQQ";

/// The `HELP <topic>;` an error message tells the reader to run.
fn advised_topics(message: &str) -> Vec<&str> {
    message
        .match_indices("HELP ")
        .filter_map(|(at, _)| {
            let after = &message[at + "HELP ".len()..];
            after.find(';').map(|end| &after[..end])
        })
        .collect()
}

#[test]
fn an_unknown_statement_lists_every_statement_keyword_and_points_at_help() {
    let message = parse_error(&format!("{NOT_A_WORD} \"x\";"));
    assert!(
        message.starts_with(&format!(
            "Parse error: expected statement keyword, got {NOT_A_WORD}. Statements start with: "
        )),
        "{message}"
    );
    assert!(
        message.ends_with("Run HELP; for the syntax of each."),
        "{message}"
    );
    for (_, page) in statement_pages() {
        for example in page.examples {
            let keyword = example
                .split_whitespace()
                .next()
                .unwrap()
                .trim_end_matches(';');
            assert!(
                message.contains(&format!(" {keyword}")),
                "{keyword} (from {example}) is not listed: {message}"
            );
        }
    }
}

#[test]
fn the_keyword_list_is_sorted_and_has_no_duplicates() {
    let keywords = grammar::leading_keywords();
    let mut sorted = keywords.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(keywords, sorted);
    for expected in ["ADD", "DISCARD", "HELP", "REMOVE", "USE", "VERIFY"] {
        assert!(
            keywords.contains(&expected),
            "{expected} missing: {keywords:?}"
        );
    }
}

#[test]
fn a_mistyped_statement_keyword_is_guessed() {
    let message = parse_error("INFR \"x\";");
    assert!(
        message.contains("got INFR. Did you mean INFER?"),
        "{message}"
    );
    let message = parse_error("DISCRAD WEIGHT EDITS;");
    assert!(message.contains("Did you mean DISCARD?"), "{message}");
    // The right-hand side of a pipe is a statement too.
    let message = parse_error("STATS |> SHWO LAYERS;");
    assert!(
        message.contains("got SHWO. Did you mean SHOW?"),
        "{message}"
    );
}

#[test]
fn what_stopped_the_parser_is_shown_as_written() {
    for (source, found) in [
        ("\"just a string\";", "got the string \"just a string\"."),
        ("42;", "got the number 42."),
        ("FROM x;", "got FROM."),
        ("(", "got LParen."),
    ] {
        let message = parse_error(source);
        assert!(message.contains(found), "{source}: {message}");
    }
}

#[test]
fn a_keyword_with_several_continuations_names_them_all() {
    // Every leading word of a several-word statement name, from the catalogue.
    let mut leading: Vec<&str> = statement_pages()
        .filter_map(|(_, page)| page.name.split_once(' ').map(|(first, _)| first))
        .collect();
    leading.sort_unstable();
    leading.dedup();
    assert!(
        leading.contains(&"REMOVE") && leading.contains(&"VERIFY"),
        "{leading:?}"
    );

    for word in leading {
        let message = parse_error(&format!("{word} {NOT_A_WORD};"));
        assert!(
            message.starts_with(&format!("Parse error: after {word} expected ")),
            "{message}"
        );
        assert!(message.contains(&format!("got {NOT_A_WORD}.")), "{message}");
        let continuations = grammar::continuations(word);
        assert!(!continuations.is_empty(), "{word}");
        for continuation in &continuations {
            assert!(
                message.contains(continuation),
                "{word}: no {continuation} in {message}"
            );
        }
        // Whatever page the message points at exists.
        for topic in advised_topics(&message) {
            assert!(
                lookup::resolve(topic).is_ok(),
                "{word}: HELP {topic}; in {message}"
            );
        }
    }
}

#[test]
fn the_continuations_of_each_keyword_are_the_ones_asked_for() {
    assert_eq!(grammar::continuations("REMOVE"), ["DIRECTION", "PATCH"]);
    assert_eq!(grammar::continuations("ADD"), ["LOW RANK"]);
    assert_eq!(grammar::continuations("DISCARD"), ["WEIGHT EDITS"]);
    assert_eq!(
        grammar::continuations("VERIFY"),
        ["PATCH", "LEDGER", "REPRESENTATION", "OPERATOR"]
    );
    for word in ["BEGIN", "SAVE", "APPLY"] {
        assert_eq!(grammar::continuations(word), ["PATCH"], "{word}");
    }
    assert_eq!(
        grammar::continuations("COMPACT"),
        ["INTO VINDEX", "MINOR", "MAJOR"]
    );
    assert_eq!(grammar::continuations("SHOW").len(), 12);
    assert!(grammar::continuations("SHOW").contains(&"TENSORS"));
    assert_eq!(grammar::continuations("EXPORT"), ["WEIGHT"]);
    assert!(grammar::continuations("INFER").is_empty());
}

#[test]
fn the_typo_from_the_audit_now_says_what_was_meant() {
    assert_eq!(
        parse_error("REMOVE DIRECTON \"x\" FROM \"t\";"),
        "Parse error: after REMOVE expected DIRECTION or PATCH, got DIRECTON. \
         Did you mean REMOVE DIRECTION? Run HELP REMOVE DIRECTION; for the syntax."
    );
}

#[test]
fn one_continuation_points_straight_at_its_page() {
    for (source, expected) in [
        (
            "ADD LOW RANKK \"x\";",
            "Parse error: after ADD expected LOW RANK, got RANKK. Run HELP ADD LOW RANK; for the syntax.",
        ),
        (
            "ADD \"x\";",
            "Parse error: after ADD expected LOW RANK, got the string \"x\". Run HELP ADD LOW RANK; for the syntax.",
        ),
        (
            "DISCARD WEIGHT;",
            "Parse error: after DISCARD expected WEIGHT EDITS, got the end of the statement. Run HELP DISCARD WEIGHT EDITS; for the syntax.",
        ),
        (
            "DISCARD WEIGHTS EDITS;",
            "Parse error: after DISCARD expected WEIGHT EDITS, got WEIGHTS. Run HELP DISCARD WEIGHT EDITS; for the syntax.",
        ),
        (
            "BEGIN \"x.vlp\";",
            "Parse error: after BEGIN expected PATCH, got the string \"x.vlp\". Run HELP BEGIN PATCH; for the syntax.",
        ),
        (
            "SAVE;",
            "Parse error: after SAVE expected PATCH, got the end of the statement. Run HELP SAVE PATCH; for the syntax.",
        ),
        (
            "APPLY 3;",
            "Parse error: after APPLY expected PATCH, got the number 3. Run HELP APPLY PATCH; for the syntax.",
        ),
    ] {
        assert_eq!(parse_error(source), expected, "{source}");
    }
}

#[test]
fn several_continuations_guess_or_point_at_the_overview() {
    let message = parse_error("VERIFY LEDGERS \"x\";");
    assert_eq!(
        message,
        "Parse error: after VERIFY expected PATCH, LEDGER, REPRESENTATION or OPERATOR, got LEDGERS. \
         Did you mean VERIFY LEDGER? Run HELP VERIFY LEDGER; for the syntax."
    );
    let message = parse_error("SHOW COMPACT;");
    assert!(
        message.starts_with("Parse error: after SHOW expected COMPONENTS, "),
        "{message}"
    );
    assert!(message.contains("COMPACT STATUS"), "{message}");
    let message = parse_error("SHOW COMPAKT STATUS;");
    assert!(
        message.ends_with(
            "Did you mean SHOW COMPACT STATUS? Run HELP SHOW COMPACT STATUS; for the syntax."
        ),
        "{message}"
    );
    let message = parse_error(&format!("SHOW {NOT_A_WORD};"));
    assert!(
        message.ends_with("Run HELP; for the syntax of each."),
        "{message}"
    );
    let message = parse_error("COMPACT;");
    assert!(
        message.starts_with("Parse error: after COMPACT expected INTO VINDEX, MINOR or MAJOR, got the end of the statement."),
        "{message}"
    );
}

#[test]
fn a_statement_with_fixed_sub_forms_names_them() {
    for (source, expected) in [
        (
            "COMPILE INTO VINDEX \"x\";",
            "Parse error: after COMPILE expected CURRENT or a quoted vindex path, got INTO. Run HELP COMPILE; for the syntax.",
        ),
        (
            "COMPILE CURRENT VINDEX \"x\";",
            "Parse error: after the COMPILE source expected INTO, got VINDEX. Run HELP COMPILE; for the syntax.",
        ),
        (
            "COMPILE CURRENT INTO VNDEX \"x\";",
            "Parse error: after COMPILE ... INTO expected VINDEX or MODEL, got VNDEX. Run HELP COMPILE; for the syntax.",
        ),
        (
            "UPDATE WEIGHT SET DELTA = 1;",
            "Parse error: after UPDATE expected EDGES or WEIGHTS, got WEIGHT. Run HELP UPDATE WEIGHTS; for the syntax.",
        ),
        (
            "EXPLAIN \"x\";",
            "Parse error: after EXPLAIN expected WALK or INFER, got the string \"x\". Run HELP EXPLAIN; for the syntax.",
        ),
    ] {
        let message = parse_error(source);
        assert_eq!(message, expected, "{source}");
        for topic in advised_topics(&message) {
            assert!(lookup::resolve(topic).is_ok(), "HELP {topic}; in {message}");
        }
    }
}

#[test]
fn the_statements_these_errors_guard_still_parse() {
    for source in [
        "COMPILE CURRENT INTO VINDEX \"out\";",
        "COMPILE \"a.vindex\" INTO MODEL \"out\";",
        "UPDATE EDGES SET target = \"x\" WHERE layer = 0 AND feature = 0;",
        "EXPLAIN WALK \"x\";",
        "EXPLAIN INFER \"x\";",
        "SHOW COMPACT STATUS;",
        "REMOVE PATCH \"p.vlp\";",
        "VERIFY REPRESENTATION 'c' USING CONTRACT 'x' AGAINST 's';",
    ] {
        parsed(source);
    }
}
