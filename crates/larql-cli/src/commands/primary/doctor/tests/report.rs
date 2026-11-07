use super::super::report::{Check, Group, Report, Status};

fn sample() -> Report {
    Report {
        checks: vec![
            Check::ok(Group::Larql, "larql", "0.2.1"),
            Check::warn(Group::Platform, "Memory", "low", "add RAM"),
            Check::skip(Group::Updates, "Latest release", "skipped"),
        ],
    }
}

#[test]
fn each_status_has_its_symbol() {
    assert_eq!(Status::Ok.symbol(), "✓");
    assert_eq!(Status::Warn.symbol(), "!");
    assert_eq!(Status::Fail.symbol(), "✗");
    assert_eq!(Status::Skip.symbol(), "-");
}

#[test]
fn every_group_has_a_heading() {
    let labels: Vec<_> = [
        Group::Larql,
        Group::Platform,
        Group::Server,
        Group::Caches,
        Group::HuggingFace,
        Group::Updates,
    ]
    .iter()
    .map(|g| g.label())
    .collect();
    assert_eq!(
        labels,
        [
            "LARQL",
            "Platform",
            "Server",
            "Caches",
            "Hugging Face",
            "Updates"
        ]
    );
}

#[test]
fn text_has_headings_aligned_names_and_hints() {
    let text = sample().render_text();
    let expected = "LARQL\n  ✓ larql           0.2.1\n\nPlatform\n  ! Memory          low\n      → add RAM\n\nUpdates\n  - Latest release  skipped\n\nReady for the quick start, with 1 warning(s) above.\n";
    assert_eq!(text, expected);
}

#[test]
fn checks_in_one_group_share_its_heading() {
    let report = Report {
        checks: vec![
            Check::ok(Group::Caches, "a", "x"),
            Check::ok(Group::Caches, "b", "y"),
        ],
    };
    assert_eq!(report.render_text().matches("Caches").count(), 1);
}

#[test]
fn verdict_names_problems_before_warnings() {
    let mut report = sample();
    assert!(report.outcome().is_ok());
    report.checks.push(Check::fail(
        Group::Caches,
        "Free disk",
        "1 GB",
        "free space",
    ));
    assert_eq!(report.problems(), 1);
    assert!(report
        .verdict()
        .starts_with("1 problem(s) will stop the quick start"));
    assert!(report.verdict().contains("✗"));
    assert_eq!(
        report.outcome().unwrap_err(),
        "larql doctor found 1 problem(s)"
    );
}

#[test]
fn a_clean_report_says_so() {
    let report = Report {
        checks: vec![Check::ok(Group::Larql, "larql", "0.2.1")],
    };
    assert_eq!(
        report.verdict(),
        "Ready for the quick start. No problems found."
    );
}

#[test]
fn an_empty_report_still_renders_a_verdict() {
    let text = Report::default().render_text();
    assert!(text.ends_with("No problems found.\n"), "{text}");
}

#[test]
fn json_carries_counts_and_every_check() {
    let json = sample().to_json("0.2.1");
    assert_eq!(json["larql_version"], "0.2.1");
    assert_eq!(json["ready"], true);
    assert_eq!(json["problems"], 0);
    assert_eq!(json["warnings"], 1);
    let checks = json["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 3);
    assert_eq!(checks[0]["group"], "larql");
    assert_eq!(checks[0]["status"], "ok");
    assert!(checks[0].get("hint").is_none());
    assert_eq!(checks[1]["status"], "warn");
    assert_eq!(checks[1]["hint"], "add RAM");
    assert_eq!(checks[2]["group"], "updates");
    assert_eq!(checks[2]["status"], "skip");
}

#[test]
fn json_names_the_hugging_face_group_in_kebab_case() {
    let report = Report {
        checks: vec![Check::fail(Group::HuggingFace, "hub", "down", "fix")],
    };
    let json = report.to_json("0.2.1");
    assert_eq!(json["checks"][0]["group"], "hugging-face");
    assert_eq!(json["checks"][0]["status"], "fail");
    assert_eq!(json["ready"], false);
}
