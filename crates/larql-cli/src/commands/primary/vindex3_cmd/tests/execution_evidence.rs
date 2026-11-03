//! MEM-002 I2b: what a record says about the computation, and when a campaign
//! attempt stops on it. End to end, through the binary, in
//! `crates/larql-cli/tests/test_campaign_execution_identity.rs`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::format::vindex3::opplan::exec::operands::SelectedRepresentation;

use super::super::generate::admission::RunMode;
use super::super::generate::execution_evidence::{
    class_with_execution, effective_problems, execution_line, ran_problems, selection_block,
    selection_problems, EffectiveFacts, ExecutionRequest, PinFacts, PinUse, PlanRan,
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

fn pin(representation: &str, pin_use: PinUse) -> PinFacts {
    PinFacts {
        object: "decoder".to_string(),
        representation: representation.to_string(),
        pin_use,
        program_compiles: false,
        object_from_pack: true,
    }
}

/// Facts with nothing wrong under [`WANTED`].
fn sound() -> EffectiveFacts {
    EffectiveFacts {
        provider: PROVIDER.to_string(),
        pinned_lowering: vec![PROVIDER.to_string()],
        pins: vec![
            pin(WANTED, PinUse::InPlace),
            pin(CANONICAL, PinUse::Converts),
        ],
        overlaid: false,
        process_error: None,
        program_declared: true,
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
        pins: vec![pin(CANONICAL, PinUse::InPlace)],
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "no operand runs it in place");
}

/// An embedding-only pack: one gathered table from the wanted encoding, and no
/// wanted kernel anywhere.
#[test]
fn a_pin_that_only_gathers_the_wanted_encoding_is_not_the_wanted_kernel() {
    let facts = EffectiveFacts {
        pins: vec![pin(WANTED, PinUse::Other), pin(CANONICAL, PinUse::Converts)],
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "no operand runs it in place");
}

#[test]
fn a_gathered_table_beside_kernels_in_place_is_not_a_problem() {
    let facts = EffectiveFacts {
        pins: vec![pin(WANTED, PinUse::InPlace), pin(WANTED, PinUse::Other)],
        ..sound()
    };
    assert_eq!(
        effective_problems(Some(WANTED), &facts),
        Vec::<String>::new()
    );
}

#[test]
fn a_pin_that_converts_the_wanted_encoding_is_a_fallback() {
    let facts = EffectiveFacts {
        pins: vec![
            pin(WANTED, PinUse::InPlace),
            pin(WANTED, PinUse::Converts),
            pin(WANTED, PinUse::Converts),
        ],
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "2 operand(s)");
}

#[test]
fn a_container_with_no_precision_program_cannot_show_its_coverage() {
    let facts = EffectiveFacts {
        program_declared: false,
        ..sound()
    };
    one_problem(&facts, Some(WANTED), "no precision program");
    assert!(effective_problems(None, &facts).is_empty());
}

fn compiled_elsewhere(representation: &str, object_from_pack: bool) -> PinFacts {
    PinFacts {
        object: "head".to_string(),
        program_compiles: true,
        object_from_pack,
        ..pin(representation, PinUse::Converts)
    }
}

/// The program says this object was compiled, and it has no pack.
#[test]
fn an_object_the_program_compiles_with_no_pack_is_a_fallback() {
    let mut facts = sound();
    facts.pins.push(compiled_elsewhere(CANONICAL, false));
    facts.pins.push(compiled_elsewhere(CANONICAL, false));
    let problems = effective_problems(Some(WANTED), &facts);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("[\"head\"]"), "{problems:?}");
}

/// A tensor carried verbatim inside a pack is the encoder's decision, and a bank
/// already packed isn't compiled again. Neither is a missing pack.
#[test]
fn a_carried_tensor_and_an_already_packed_bank_are_not_fallbacks() {
    let mut facts = sound();
    facts.pins.push(compiled_elsewhere(CANONICAL, true));
    facts.pins.push(compiled_elsewhere("MXFP4", false));
    assert_eq!(
        effective_problems(Some(WANTED), &facts),
        Vec::<String>::new()
    );
}

/// Protected or uncompiled roles: the program doesn't compile them, so a pack
/// isn't owed.
#[test]
fn an_object_the_program_does_not_compile_needs_no_pack() {
    let mut facts = sound();
    facts.pins.push(PinFacts {
        object_from_pack: false,
        ..pin(CANONICAL, PinUse::Converts)
    });
    assert_eq!(
        effective_problems(Some(WANTED), &facts),
        Vec::<String>::new()
    );
}

#[test]
fn a_kernel_the_ledger_never_counted_did_not_run() {
    let plans = vec!["FusedKQuant".to_string()];
    let ran = |plan: &str| {
        vec![PlanRan {
            plan: plan.to_string(),
            calls: 3,
            bytes: 9,
        }]
    };
    assert_eq!(
        ran_problems(&plans, &ran("FusedKQuant")),
        Vec::<String>::new()
    );
    let problems = ran_problems(&plans, &ran("BlasF32"));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("FusedKQuant"), "{problems:?}");
    assert_eq!(ran_problems(&plans, &[]).len(), 1);
    // Nothing named to look for: a mapped bank, or nothing wanted.
    assert!(ran_problems(&[], &ran("BlasF32")).is_empty());
}

#[test]
fn a_wanted_encoding_with_no_pack_is_a_fallback() {
    let canonical = selection(&[("a", CANONICAL, false), ("b", CANONICAL, false)]);
    let problems = selection_problems(Some(WANTED), &canonical, &[]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("fall back"), "{problems:?}");
    // Nothing wanted, nothing to fall back from.
    assert!(selection_problems(None, &canonical, &[]).is_empty());
}

/// A pack of another encoding isn't the one asked for, and a canonical object
/// whose encoding happens to carry the wanted name isn't from a pack.
#[test]
fn only_a_pack_of_the_wanted_encoding_counts() {
    let other = selection(&[("a", "Q4_K", true)]);
    assert_eq!(selection_problems(Some(WANTED), &other, &[]).len(), 1);
    let not_a_pack = selection(&[("a", WANTED, false)]);
    assert_eq!(selection_problems(Some(WANTED), &not_a_pack, &[]).len(), 1);
}

#[test]
fn a_pack_the_index_describes_and_the_disk_lacks_is_a_fallback() {
    let packed = selection(&[("a", WANTED, true), ("b", CANONICAL, false)]);
    let problems = selection_problems(Some(WANTED), &packed, &["a".to_string()]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("not on disk"), "{problems:?}");
    // A canonical object not hydrated yet is the store's business, not a fallback.
    assert!(selection_problems(Some(WANTED), &packed, &["b".to_string()]).is_empty());
}

#[test]
fn the_selection_block_says_which_objects_came_from_a_pack() {
    let partial = selection(&[("a", WANTED, true), ("b", CANONICAL, false)]);
    let block = selection_block(&partial);
    assert_eq!(block[0]["from_pack"], true);
    assert_eq!(block[0]["encoding"], WANTED);
    assert_eq!(block[1]["from_pack"], false);
    assert_eq!(block[1]["encoding"], CANONICAL);
    assert_eq!(block[1]["object"], "b");
}

#[test]
fn an_empty_selection_is_incomplete() {
    let problems = selection_problems(None, &BTreeMap::new(), &[]);
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
        component: "target".to_string(),
        backend: "production".to_string(),
        wanted_encoding: Some(WANTED.to_string()),
        plugins: vec![readable.clone(), missing],
        lowering: Some("other/v1".to_string()),
        representation: Some(WANTED.to_string()),
    }
    .block();
    assert_eq!(block["component"], "target");
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
