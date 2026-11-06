//! The refusal text: short, itemised, pointing at the full plan.

use crate::commands::primary::hf_auto::refusal::*;
use crate::commands::primary::vindex3_cmd::tests::fixture_dir;
use larql_vindex::format::vindex3::plan::capability::Capability;
use larql_vindex::format::vindex3::plan::{plan_system, FindingCategory, SystemPlan};

fn plan(judged: bool) -> SystemPlan {
    let dir = fixture_dir(judged);
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    plan_system(&[("fixture".to_string(), inventory)])
}

fn blocker(n: usize) -> Blocker {
    Blocker {
        component: "text".into(),
        subject: format!("key_{n}"),
        category: "unrepresented",
        detail: "no home in the schema".into(),
    }
}

#[test]
fn whole_model_judgement_lists_every_blocker() {
    let (verdict, blockers) = judge(&plan(false), None);
    assert_eq!(verdict, Verdict::Blocked);
    assert!(blockers
        .iter()
        .any(|b| b.subject.contains("some_future_field_nobody_reviewed")));
    let (verdict, blockers) = judge(&plan(true), None);
    assert_eq!(verdict, Verdict::Admissible);
    assert!(blockers.is_empty());
}

#[test]
fn capability_judgement_reads_the_capability_status() {
    let (verdict, _) = judge(&plan(true), Some(Capability::TextGeneration));
    assert_eq!(verdict, Verdict::Admissible);
    let blocked = plan(false);
    let (verdict, blockers) = judge(&blocked, Some(Capability::TextGeneration));
    let status = blocked
        .capabilities
        .iter()
        .find(|c| c.capability == Capability::TextGeneration)
        .unwrap();
    assert_eq!(verdict == Verdict::Blocked, !status.admissible);
    assert_eq!(blockers.len(), status.blocker_ids.len());
}

#[test]
fn a_capability_the_plan_does_not_report_or_the_checkpoint_lacks_refuses() {
    let mut p = plan(true);
    // The fixture is text-only: image conditioning is not available.
    let (verdict, _) = judge(&p, Some(Capability::ImageConditioned));
    assert_ne!(verdict, Verdict::Admissible);
    p.capabilities.clear();
    let (verdict, blockers) = judge(&p, Some(Capability::TextGeneration));
    assert_eq!(verdict, Verdict::Unsupported);
    assert!(blockers.is_empty());
    let mut p = plan(true);
    for status in &mut p.capabilities {
        status.supported = false;
    }
    assert_eq!(
        judge(&p, Some(Capability::TextGeneration)).0,
        Verdict::Unsupported
    );
    for status in &mut p.capabilities {
        status.available = false;
    }
    assert_eq!(
        judge(&p, Some(Capability::TextGeneration)).0,
        Verdict::Unavailable
    );
}

#[test]
fn render_is_short_itemised_and_points_at_the_plan() {
    let spec = "hf://acme/x@abc";
    let text = render(
        spec,
        "text generation",
        Verdict::Blocked,
        &[blocker(0), blocker(1)],
    );
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 4, "{text}");
    assert!(lines[0].contains("2 blocking finding(s)"));
    assert_eq!(
        lines[1],
        "  - [unrepresented] text: key_0 — no home in the schema"
    );
    assert_eq!(lines[3], plan_hint(spec, "text generation"));
    assert!(lines[3].contains("larql vindex3 plan hf://acme/x@abc --output plan.json"));
    assert!(lines[3].contains("gated on text generation only"));
}

#[test]
fn render_caps_the_list_and_counts_the_rest() {
    let many: Vec<_> = (0..MAX_LISTED_BLOCKERS + 3).map(blocker).collect();
    let text = render("hf://a/b", "text generation", Verdict::Blocked, &many);
    assert_eq!(text.matches("\n  - ").count(), MAX_LISTED_BLOCKERS);
    assert!(text.contains("… and 3 more"), "{text}");
    let exact: Vec<_> = (0..MAX_LISTED_BLOCKERS).map(blocker).collect();
    assert!(!render("hf://a/b", "x", Verdict::Blocked, &exact).contains("more"));
}

#[test]
fn every_verdict_has_a_headline() {
    for (verdict, needle) in [
        (Verdict::Admissible, "is admissible"),
        (Verdict::Unsupported, "no executor"),
        (Verdict::Unavailable, "ships nothing"),
    ] {
        assert!(render("hf://a/b", "text generation", verdict, &[]).contains(needle));
    }
}

#[test]
fn truncate_keeps_the_first_line_within_budget() {
    assert_eq!(truncate("short", 10), "short");
    assert_eq!(truncate("first\nsecond", 10), "first");
    assert_eq!(truncate("abcdefghijk", 5), "abcd…");
    assert_eq!(truncate("ééééééé", 4).chars().count(), 4);
    assert_eq!(truncate("", 4), "");
    assert_eq!(truncate("wrapped   \t  detail", 40), "wrapped detail");
}

#[test]
fn category_labels_match_the_plan_json() {
    for category in [
        FindingCategory::Representable,
        FindingCategory::Mismatched,
        FindingCategory::Unrepresented,
        FindingCategory::Interface,
    ] {
        assert_eq!(
            serde_json::to_value(category).unwrap().as_str().unwrap(),
            category_label(category)
        );
    }
}
