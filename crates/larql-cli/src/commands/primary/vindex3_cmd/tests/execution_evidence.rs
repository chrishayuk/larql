//! MEM-002 I2b: what a record says about the computation, and when a campaign
//! attempt stops on it. End to end, through the binary, in
//! `crates/larql-cli/tests/test_campaign_execution_identity.rs`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::format::vindex3::opplan::exec::operands::SelectedRepresentation;

use super::super::generate::admission::RunMode;
use super::super::generate::execution_evidence::{
    class_with_execution, effective_problems, execution_line, quantisation_problem,
    selection_block, selection_problems, EffectiveFacts, ExecutionRequest, PinFacts,
};
use super::super::generate::instance_evidence::EvidenceClass;
use super::super::prepare::{backend_flag_name, campaign_supports_backend, wanted_encoding};
use super::super::ExecBackend;

const WANTED: &str = "NVFP4";
const CANONICAL: &str = "BF16";
const PROVIDER: &str = "cpu-production/v1";

fn selected(encoding: &str, stored: bool) -> SelectedRepresentation {
    SelectedRepresentation {
        encoding: encoding.to_string(),
        stored,
        codec: None,
    }
}

fn selection(objects: &[(&str, &str, bool)]) -> BTreeMap<String, SelectedRepresentation> {
    objects
        .iter()
        .map(|(object, encoding, stored)| (object.to_string(), selected(encoding, *stored)))
        .collect()
}

fn pin(representation: &str, converts: bool) -> PinFacts {
    PinFacts {
        representation: representation.to_string(),
        converts,
    }
}

/// Facts with nothing wrong under [`WANTED`].
fn sound() -> EffectiveFacts {
    EffectiveFacts {
        provider: PROVIDER.to_string(),
        pinned_lowering: vec![PROVIDER.to_string()],
        pins: vec![pin(WANTED, false), pin(CANONICAL, true)],
        overlaid: false,
        process_error: None,
        kquant_widened: false,
        runtime_quantised: 0,
    }
}

fn one_problem(facts: &EffectiveFacts, wanted: Option<&str>, says: &str) {
    let problems = effective_problems(wanted, facts);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains(says), "{problems:?}");
}

#[test]
fn sound_facts_have_no_problem() {
    assert_eq!(
        effective_problems(Some(WANTED), &sound()),
        Vec::<String>::new()
    );
}

#[test]
fn an_image_with_no_pins_is_incomplete() {
    let facts = EffectiveFacts {
        pins: vec![],
        ..sound()
    };
    one_problem(&facts, None, "pinned no operand");
}

#[test]
fn pins_naming_another_provider_are_incomplete() {
    let mut facts = sound();
    facts.pinned_lowering = vec!["other/v1".to_string()];
    one_problem(&facts, None, "other/v1");
    // A mixed image is not one provider's either.
    facts.pinned_lowering = vec![PROVIDER.to_string(), "other/v1".to_string()];
    one_problem(&facts, None, "other/v1");
}

#[test]
fn an_overlaid_source_is_incomplete() {
    let facts = EffectiveFacts {
        overlaid: true,
        ..sound()
    };
    one_problem(&facts, None, "overlaid");
}

#[test]
fn unreadable_process_arithmetic_is_incomplete() {
    let facts = EffectiveFacts {
        process_error: Some("no pool".to_string()),
        ..sound()
    };
    one_problem(&facts, None, "no pool");
}

#[test]
fn a_wanted_encoding_nothing_was_pinned_from_is_a_fallback() {
    let facts = EffectiveFacts {
        pins: vec![pin(CANONICAL, false)],
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "no operand was pinned");
}

#[test]
fn a_pin_that_converts_the_wanted_encoding_is_a_fallback() {
    let facts = EffectiveFacts {
        pins: vec![pin(WANTED, false), pin(WANTED, true), pin(WANTED, true)],
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "2 operand(s)");
}

#[test]
fn widening_stored_packs_is_a_fallback_only_when_an_encoding_was_wanted() {
    let facts = EffectiveFacts {
        kquant_widened: true,
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "widens");
    assert!(effective_problems(None, &facts).is_empty());
}

#[test]
fn a_manufactured_representation_is_a_fallback_only_when_one_was_wanted() {
    let facts = EffectiveFacts {
        runtime_quantised: 3,
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "quantised 3");
    assert!(effective_problems(None, &facts).is_empty());
    assert_eq!(quantisation_problem(0), None);
}

#[test]
fn a_wanted_encoding_with_no_pack_is_a_fallback() {
    let canonical = selection(&[("a", CANONICAL, false), ("b", CANONICAL, false)]);
    let problems = selection_problems(Some(WANTED), &canonical);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("fall back"), "{problems:?}");
    // Nothing wanted, nothing to fall back from.
    assert!(selection_problems(None, &canonical).is_empty());
}

/// A pack of another encoding isn't the one asked for, and a canonical object
/// whose encoding happens to carry the wanted name isn't from a pack.
#[test]
fn only_a_pack_of_the_wanted_encoding_counts() {
    let other = selection(&[("a", "Q4_K", true)]);
    assert_eq!(selection_problems(Some(WANTED), &other).len(), 1);
    let not_a_pack = selection(&[("a", WANTED, false)]);
    assert_eq!(selection_problems(Some(WANTED), &not_a_pack).len(), 1);
}

/// A limit stated in the doc: real packs cover some objects, so a canonical
/// object beside a pack is recorded and not refused.
#[test]
fn partial_pack_coverage_is_recorded_not_refused() {
    let partial = selection(&[("a", WANTED, true), ("b", CANONICAL, false)]);
    assert!(selection_problems(Some(WANTED), &partial).is_empty());
    let block = selection_block(&partial);
    assert_eq!(block[0]["from_pack"], true);
    assert_eq!(block[0]["encoding"], WANTED);
    assert_eq!(block[1]["from_pack"], false);
    assert_eq!(block[1]["encoding"], CANONICAL);
    assert_eq!(block[1]["object"], "b");
}

#[test]
fn an_empty_selection_is_incomplete() {
    let problems = selection_problems(None, &BTreeMap::new());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("no object"), "{problems:?}");
}

#[test]
fn execution_problems_invalidate_a_campaign_and_leave_a_diagnostic_run_alone() {
    let problems = vec!["x".to_string()];
    for class in [EvidenceClass::IdentityConsistent, EvidenceClass::Unverified] {
        assert_eq!(
            class_with_execution(RunMode::Campaign, class, &problems),
            EvidenceClass::Invalid
        );
        assert_eq!(class_with_execution(RunMode::Campaign, class, &[]), class);
    }
    assert_eq!(
        class_with_execution(RunMode::Diagnostic, EvidenceClass::Diagnostic, &problems),
        EvidenceClass::Diagnostic
    );
}

#[test]
fn the_console_line_says_what_was_found() {
    assert_eq!(
        execution_line("admission", &[]),
        "execution identity (admission): complete"
    );
    let line = execution_line("after run", &["a".to_string(), "b".to_string()]);
    assert_eq!(line, "execution identity (after run): a; b");
}

#[test]
fn a_request_is_recorded_with_each_plugin_digest_or_a_null_for_an_unreadable_one() {
    let dir = tempfile::tempdir().unwrap();
    let readable = dir.path().join("plugin.bin");
    std::fs::write(&readable, b"bytes").unwrap();
    let missing = PathBuf::from("/nonexistent/plugin.bin");
    let block = ExecutionRequest {
        backend: "production".to_string(),
        wanted_encoding: Some(WANTED.to_string()),
        plugins: vec![readable.clone(), missing],
        lowering: Some("other/v1".to_string()),
        representation: Some(WANTED.to_string()),
    }
    .block();
    assert_eq!(block["backend"], "production");
    assert_eq!(block["wanted_encoding"], WANTED);
    assert_eq!(block["lowering"], "other/v1");
    assert_eq!(block["representation"], WANTED);
    let expected = larql_vindex::format::checksums::sha256_file(&readable).unwrap();
    assert_eq!(block["plugins"][0]["sha256"], expected.as_str());
    assert!(block["plugins"][1]["sha256"].is_null());
}

#[test]
fn the_flag_spelling_and_the_wanted_encoding_come_from_one_derivation() {
    assert_eq!(
        backend_flag_name(ExecBackend::ProductionNvfp4),
        "production-nvfp4"
    );
    assert_eq!(
        wanted_encoding(ExecBackend::ProductionNvfp4, None).as_deref(),
        Some(WANTED)
    );
    assert_eq!(wanted_encoding(ExecBackend::Production, None), None);
    assert_eq!(
        wanted_encoding(ExecBackend::Production, Some("Q4_K")).as_deref(),
        Some("Q4_K")
    );
}

#[test]
fn every_cpu_backend_is_supported_in_a_campaign() {
    for backend in [
        ExecBackend::Reference,
        ExecBackend::Production,
        ExecBackend::ProductionNvfp4,
        ExecBackend::ProductionQ8,
        ExecBackend::ProductionQ6k,
        ExecBackend::ProductionQ4k,
        ExecBackend::ProductionQ4kQ8k,
        ExecBackend::ProductionNvfp4Q8,
    ] {
        assert!(campaign_supports_backend(backend), "{backend:?}");
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
#[test]
fn a_device_backend_is_refused_in_a_campaign() {
    assert!(!campaign_supports_backend(ExecBackend::Metal));
}
