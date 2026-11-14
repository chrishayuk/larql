//! `HELP` as a statement: how it parses, what it needs, and who may run it.

use larql_lql::help::StatementTopic;
use larql_lql::{run_batch_outcome, CapabilityProfile, Session, Statement, EXIT_ERROR, EXIT_OK};
use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;

use crate::support::*;

fn topic_of(source: &str) -> Option<String> {
    match parsed(source) {
        Statement::Help { topic } => topic,
        other => panic!("{source} parsed as {other:?}"),
    }
}

#[test]
fn it_parses_with_and_without_a_topic() {
    assert_eq!(topic_of("HELP;"), None);
    assert_eq!(topic_of("help"), None);
    assert_eq!(
        topic_of("HELP REMOVE DIRECTION;").as_deref(),
        Some("REMOVE DIRECTION")
    );
    assert_eq!(
        topic_of("HELP \"exit codes\";").as_deref(),
        Some("exit codes")
    );
    assert_eq!(topic_of("HELP EXIT CODES").as_deref(), Some("EXIT CODES"));
}

#[test]
fn a_topic_that_is_not_words_is_a_parse_error_that_says_what_a_topic_is() {
    for (source, found) in [
        ("HELP 5;", "the number 5"),
        ("HELP 0.5;", "the number 0.5"),
        ("HELP (WEIGHTS);", "LParen"),
        ("HELP = ;", "Eq"),
    ] {
        let message = parse_error(source);
        assert!(message.contains("HELP takes a topic in words"), "{message}");
        assert!(message.contains(&format!("got {found}")), "{message}");
        assert!(message.contains("Run HELP;"), "{message}");
    }
}

#[test]
fn it_reports_its_verb_and_its_topic_and_writes_nothing() {
    let statement = parsed("HELP WEIGHTS;");
    assert_eq!(statement.verb(), "HELP");
    assert_eq!(statement.help_topic(), StatementTopic::Help);
    assert!(!statement.writes_artifact());
    assert!(!parsed("HELP;").writes_artifact());
}

#[test]
fn it_needs_no_backend() {
    let mut session = Session::new();
    let overview = execute(&mut session, "HELP;").unwrap();
    assert!(overview[0].starts_with("LQL statements"), "{overview:?}");
    assert!(execute(&mut session, "HELP COMPILE;").is_ok());
    // The session is as unbound as before.
    assert!(execute(&mut session, "STATS;").is_err());
}

#[test]
fn it_answers_the_same_bound_or_unbound() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let mut bound = Session::new();
    execute(
        &mut bound,
        &format!(
            "USE \"{}\";",
            container.display().to_string().replace('\\', "\\\\")
        ),
    )
    .unwrap();
    for source in [
        "HELP;",
        "HELP ADD LOW RANK;",
        "HELP BATCH;",
        "HELP PATCHES;",
    ] {
        assert_eq!(
            execute(&mut bound, source).unwrap(),
            help(source),
            "{source}"
        );
    }
}

#[test]
fn every_capability_profile_permits_it() {
    for profile in [CapabilityProfile::Full, CapabilityProfile::PublicExplorer] {
        for source in ["HELP;", "HELP REMOVE DIRECTION;", "HELP EXIT CODES;"] {
            assert!(
                profile.check(&parsed(source)).is_ok(),
                "{source} under {profile:?}"
            );
            let mut session = Session::new();
            session.set_profile(profile);
            assert_eq!(execute(&mut session, source).unwrap(), help(source));
        }
    }
    // An unknown topic is the same error under the public profile: not a refusal.
    let mut public = Session::new();
    public.set_profile(CapabilityProfile::PublicExplorer);
    let err = execute(&mut public, "HELP NOPE NOPE;").unwrap_err();
    assert!(err.to_string().starts_with("no help topic"), "{err}");
}

#[test]
fn the_public_profile_names_help_among_what_it_serves() {
    let mut public = Session::new();
    public.set_profile(CapabilityProfile::PublicExplorer);
    let refusal = execute(&mut public, "USE \"x\";").unwrap_err().to_string();
    assert!(refusal.contains("HELP"), "{refusal}");
    // HELP in a pipe with a refused statement: the pipe is still judged whole.
    assert!(execute(&mut public, "HELP |> USE \"x\";").is_err());
    assert!(execute(&mut public, "HELP |> HELP BATCH;").is_ok());
}

#[test]
fn in_a_batch_it_is_output_and_an_unknown_topic_is_a_failure() {
    let ok = run_batch_outcome("HELP; HELP EXIT CODES;");
    assert_eq!(ok.exit_code(), EXIT_OK);
    assert!(ok.lines.iter().any(|l| l.starts_with("EXIT CODES")));

    let failed = run_batch_outcome("HELP REMOVE DIRECTON;");
    assert_eq!(failed.exit_code(), EXIT_ERROR);
    assert_eq!(failed.errors, 1);
    assert!(
        failed.lines[0]
            .starts_with("Error: no help topic \"REMOVE DIRECTON\". Nearest: REMOVE DIRECTION."),
        "{:?}",
        failed.lines
    );
}

#[test]
fn it_still_runs_after_a_failure_in_a_batch() {
    // HELP writes nothing, so the rule that refuses writes after a failure does not touch it.
    let outcome = run_batch_outcome("STATS; HELP BATCH;");
    assert_eq!(outcome.errors, 1);
    assert!(outcome.lines.iter().any(|l| l.starts_with("BATCH")));
}
