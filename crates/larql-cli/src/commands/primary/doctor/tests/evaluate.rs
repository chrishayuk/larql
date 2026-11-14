use super::super::evaluate;
use super::super::network::Probe;
use super::super::report::{Group, Status};
use super::super::server::ServerLocation;
use super::healthy;

#[test]
fn a_healthy_machine_is_ready_and_exits_zero() {
    let report = evaluate(&healthy());
    assert_eq!(report.problems(), 0, "{}", report.render_text());
    assert_eq!(report.count(Status::Warn), 0);
    assert!(report.outcome().is_ok());
}

#[test]
fn checks_come_out_grouped_in_display_order() {
    let report = evaluate(&healthy());
    let mut groups: Vec<Group> = report.checks.iter().map(|c| c.group).collect();
    groups.dedup();
    assert_eq!(
        groups,
        [
            Group::Larql,
            Group::Platform,
            Group::Server,
            Group::Caches,
            Group::HuggingFace,
            Group::Updates
        ]
    );
}

#[test]
fn offline_skips_only_the_network() {
    let mut facts = healthy();
    facts.hub = Probe::Skipped;
    facts.release = Probe::Skipped;
    let report = evaluate(&facts);
    assert_eq!(report.count(Status::Skip), 2);
    assert!(report.outcome().is_ok());
}

#[test]
fn a_missing_server_stops_the_quick_start() {
    let mut facts = healthy();
    facts.server.location = ServerLocation::Missing;
    facts.server.version = None;
    let report = evaluate(&facts);
    assert_eq!(report.problems(), 1);
    assert!(report.outcome().is_err());
}

#[test]
fn warnings_alone_do_not_fail() {
    let mut facts = healthy();
    facts.token_source = None;
    facts.release = Probe::Done("v9.0.0".into());
    let report = evaluate(&facts);
    assert_eq!(report.count(Status::Warn), 2);
    assert!(report.outcome().is_ok());
}
