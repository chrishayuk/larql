//! What `USE` says a VINDEX3 container serves is the help catalogue's list,
//! and the catalogue's claims about what a statement needs are what the
//! executor refuses without.
//!
//! The container is the BF16 fixture with a Q8_0 pack and no tokenizer, so
//! every statement that needs a tokenizer refuses for that reason, and one
//! that needs a VINDEX2 vindex refuses as unsupported.

use larql_lql::help::{render, Need};
use larql_lql::{Session, Statement};
use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;

use crate::support::*;

/// What the executor says when a VINDEX3 binding does not serve a statement.
const UNSUPPORTED: &str = "is not supported on a VINDEX3 container";
/// What every refusal for a missing tokenizer mentions.
const TOKENIZER: &str = "tokenizer";

struct Bound {
    _dir: tempfile::TempDir,
    banner: Vec<String>,
    session: Session,
}

fn bound() -> Bound {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let mut session = Session::new();
    // No verifier, whatever this machine has installed.
    session.set_verifier("");
    let banner = execute(
        &mut session,
        &format!(
            "USE \"{}\";",
            container.display().to_string().replace('\\', "\\\\")
        ),
    )
    .unwrap();
    Bound {
        _dir: dir,
        banner,
        session,
    }
}

/// Statements this test must not execute: they would write into the working
/// directory, rebind the session, or load a model.
fn must_not_run(statement: &Statement) -> bool {
    statement.writes_artifact() || matches!(statement, Statement::Use { .. })
}

#[test]
fn the_banner_is_the_catalogues_list_and_points_at_help() {
    let bound = bound();
    let supported = bound
        .banner
        .iter()
        .find(|l| l.starts_with("Supported: "))
        .expect("a Supported line");
    assert_eq!(
        supported,
        &format!(
            "Supported: {}. Tokenizer: absent (token-id capability only).",
            render::vindex3_supported()
        )
    );
    assert!(
        bound.banner.iter().any(|l| l.contains("HELP;")),
        "{:?}",
        bound.banner
    );
    // The statements the audit found missing are there.
    for name in [
        "REMOVE DIRECTION",
        "ADD LOW RANK",
        "DISCARD WEIGHT EDITS",
        "VERIFY PATCH",
        "VERIFY LEDGER",
        "VERIFY REPRESENTATION",
        "VERIFY OPERATOR",
        "HELP",
    ] {
        assert!(supported.contains(name), "{name} missing: {supported}");
    }
}

#[test]
fn the_banner_lists_every_statement_except_the_vindex2_only_ones() {
    let supported = render::vindex3_supported();
    let listed: Vec<&str> = supported.split(", ").collect();
    for (_, page) in statement_pages() {
        assert_eq!(
            listed.contains(&page.name),
            !page.needs(Need::Vindex2),
            "{}",
            page.name
        );
    }
}

#[test]
fn without_a_tokenizer_the_banner_names_what_will_be_refused() {
    let bound = bound();
    let line = bound
        .banner
        .iter()
        .find(|l| l.starts_with("Refused without a tokenizer: "))
        .expect("a line naming the statements that need a tokenizer");
    assert!(line.contains(&render::vindex3_needs_tokenizer()), "{line}");
    for name in ["INFER", "WALK", "DESCRIBE", "SELECT", "TRACE", "INSERT"] {
        assert!(line.contains(name), "{name} missing: {line}");
    }
    assert!(!line.contains("UPDATE WEIGHTS"), "{line}");
}

#[test]
fn stats_and_the_capability_refusal_print_the_same_list() {
    let mut bound = bound();
    let supported = render::vindex3_supported();
    let stats = execute(&mut bound.session, "STATS;").unwrap().join("\n");
    assert!(
        stats.contains(&format!("Capabilities:    {supported}")),
        "{stats}"
    );
    let refusal = execute(&mut bound.session, "COMPACT MINOR;")
        .unwrap_err()
        .to_string();
    assert!(refusal.contains(UNSUPPORTED), "{refusal}");
    assert!(refusal.contains(&supported), "{refusal}");
    assert!(refusal.contains("Run HELP;"), "{refusal}");
}

#[test]
fn a_page_that_says_vindex2_is_refused_on_vindex3_and_no_other_page_is() {
    let mut bound = bound();
    for (_, page) in statement_pages() {
        let statement = parsed(page.examples[0]);
        if must_not_run(&statement) {
            continue;
        }
        let refused_as_unsupported = bound
            .session
            .execute(&statement)
            .err()
            .is_some_and(|e| e.to_string().contains(UNSUPPORTED));
        assert_eq!(
            refused_as_unsupported,
            page.needs(Need::Vindex2),
            "{}: {}",
            page.name,
            page.examples[0]
        );
    }
}

#[test]
fn a_page_that_says_tokenizer_is_refused_without_one_and_no_other_page_is() {
    // Judged on each page's first example, its plainest form: a clause can
    // add a need the statement does not have by itself (EXPLAIN WALK needs a
    // tokenizer; EXPLAIN INFER, the first example, does not).
    let mut bound = bound();
    for (_, page) in statement_pages() {
        let statement = parsed(page.examples[0]);
        if must_not_run(&statement) || page.needs(Need::Vindex2) {
            continue;
        }
        let refused_for_tokenizer = bound
            .session
            .execute(&statement)
            .err()
            .is_some_and(|e| e.to_string().contains(TOKENIZER));
        assert_eq!(
            refused_for_tokenizer,
            page.needs(Need::Tokenizer),
            "{}: {}",
            page.name,
            page.examples[0]
        );
    }
}

#[test]
fn a_page_that_says_verifier_is_refused_without_one() {
    let mut bound = bound();
    for (_, page) in statement_pages() {
        if !page.needs(Need::Verifier) {
            continue;
        }
        let err = execute(&mut bound.session, page.examples[0])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("needs an independent verifier and found none"),
            "{}: {err}",
            page.name
        );
    }
}

#[test]
fn a_page_that_says_vindex3_refuses_an_unbound_session() {
    for (_, page) in statement_pages() {
        let statement = parsed(page.examples[0]);
        if !page.needs(Need::Vindex3) || must_not_run(&statement) {
            continue;
        }
        let err = Session::new().execute(&statement).unwrap_err().to_string();
        assert!(err.contains("No backend loaded"), "{}: {err}", page.name);
    }
}

#[test]
fn a_page_that_says_nothing_bound_does_not_ask_for_a_backend() {
    for (_, page) in statement_pages() {
        let statement = parsed(page.examples[0]);
        if !page.needs(Need::Nothing) || must_not_run(&statement) {
            continue;
        }
        let mut session = Session::new();
        session.set_verifier("");
        if let Err(e) = session.execute(&statement) {
            assert!(
                !e.to_string().contains("No backend loaded"),
                "{}: {e}",
                page.name
            );
        }
    }
}
