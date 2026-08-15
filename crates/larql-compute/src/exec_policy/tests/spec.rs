use super::*;
use crate::exec_policy::{ExecutionStrategy, ExpertGroupSite};

fn site(layer: usize, phase: Option<Phase>, step: Option<u64>) -> ExpertGroupSite {
    ExpertGroupSite {
        layer,
        phase,
        step,
        slots: 4,
    }
}

/// The plain form: named layers, every decode token.
#[test]
fn skip_layers_builds_a_decode_only_mask() {
    let p = parse("skip-layers:20,22").expect("parses");
    assert_eq!(
        p.expert_group(&site(20, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        p.expert_group(&site(21, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Canonical
    );
}

/// Every form is decode-only and cannot be made otherwise from the
/// environment. Skipping expert groups during prefill perturbs the
/// prompt's own representation — a different experiment with a different
/// control, and not one a typo in a bench command should be able to
/// start.
#[test]
fn every_env_form_is_decode_only() {
    for spec in [
        "skip-layers:20",
        "skip-layers:20:every-2",
        "skip-layers:20:token-0",
    ] {
        let p = parse(spec).expect("parses");
        assert_eq!(
            p.expert_group(&site(20, Some(Phase::Prefill), Some(0))),
            ExecutionStrategy::Canonical,
            "{spec} must not fire during prefill"
        );
    }
}

#[test]
fn step_selectors_parse() {
    let every4 = parse("skip-layers:20:every-4").expect("parses");
    assert_eq!(
        every4.expert_group(&site(20, Some(Phase::Decode), Some(8))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        every4.expert_group(&site(20, Some(Phase::Decode), Some(9))),
        ExecutionStrategy::Canonical
    );

    let one = parse("skip-layers:20:token-3").expect("parses");
    assert_eq!(
        one.expert_group(&site(20, Some(Phase::Decode), Some(3))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        one.expert_group(&site(20, Some(Phase::Decode), Some(4))),
        ExecutionStrategy::Canonical
    );
}

/// The name carries the parsed configuration, so the announcement line
/// and the ledger's policy line describe what actually ran rather than
/// echoing the operator's string back at them.
#[test]
fn the_built_policy_names_its_parsed_configuration() {
    let p = parse("skip-layers:22,20:every-4").expect("parses");
    let name = p.name();
    assert!(name.contains("layers=[20,22]"), "{name}");
    assert!(name.contains("phase=decode"), "{name}");
    assert!(name.contains("steps=every-4th"), "{name}");
}

/// Malformed input is an ERROR, never a silent canonical fallback. A
/// silent fallback makes an A/B compare canonical against canonical and
/// report "no change" — an instrument that cannot fail on known-
/// different input.
#[test]
fn malformed_specs_are_refused_not_downgraded() {
    for spec in [
        "",
        "skip",
        "skip-layers:",
        "skip-layers:,",
        "skip-layers:twenty",
        "skip-layers:20:",
        "skip-layers:20:sometimes",
        "skip-layers:20:every-0",
        "skip-layers:20:every-x",
        "skip-layers:20:token-x",
        "canonical",
    ] {
        assert!(parse(spec).is_err(), "{spec:?} must be refused");
    }
}

/// A zero period is refused explicitly rather than parsed into a mask
/// that skips nothing — the latter would install successfully, announce
/// itself, and then do nothing.
#[test]
fn a_zero_period_is_refused_with_a_reason() {
    let err = parse("skip-layers:20:every-0")
        .map(|_| ())
        .expect_err("must refuse");
    assert!(err.contains("period must be >= 1"), "{err}");
}

/// Every error names the grammar, because the operator is at a shell
/// prompt and the next thing they need is what to type instead.
#[test]
fn errors_carry_the_usage_line() {
    let err = parse("nonsense").map(|_| ()).expect_err("must refuse");
    assert!(err.contains("skip-layers:"), "{err}");
    assert!(err.contains("trace:"), "{err}");
}

/// A trace spec loads the file and replays exactly its pairs.
#[test]
fn trace_spec_loads_and_replays_the_file() {
    let dir = std::env::temp_dir().join(format!("larql-spec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("ok.trace");
    let mut t = super::super::trace::Trace::new("unit test");
    t.record(20, 0);
    t.record(20, 4);
    t.write(&path).expect("writes");

    let p = parse(&format!("trace:{}", path.display())).expect("parses");
    assert_eq!(
        p.expert_group(&site(20, Some(Phase::Decode), Some(4))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        p.expert_group(&site(20, Some(Phase::Decode), Some(1))),
        ExecutionStrategy::Canonical
    );
    assert!(p.name().contains("pairs=2"), "{}", p.name());
    let _ = std::fs::remove_dir_all(&dir);
}

/// An EMPTY trace is refused. It would install, announce itself, and
/// then behave exactly like canonical execution — producing an A/B whose
/// two arms are the same program while the log says a policy was
/// active.
#[test]
fn an_empty_trace_is_refused() {
    let dir = std::env::temp_dir().join(format!("larql-spec-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("empty.trace");
    super::super::trace::Trace::new("nothing")
        .write(&path)
        .expect("writes");
    let err = parse(&format!("trace:{}", path.display()))
        .map(|_| ())
        .expect_err("must refuse");
    assert!(err.contains("records no skips"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A missing trace file is an error that names the path.
#[test]
fn a_missing_trace_file_is_refused() {
    let err = parse("trace:/definitely/not/here.trace")
        .map(|_| ())
        .expect_err("must refuse");
    assert!(err.contains("here.trace"), "{err}");
}

/// `from_env` with the variable unset leaves execution canonical — the
/// production default, and the one that must cost nothing.
#[test]
fn from_env_unset_installs_nothing() {
    let _g = crate::movement_ledger::bytes::COUNTER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    crate::exec_policy::uninstall();
    options::set_env_override(options::ENV_EXEC_POLICY, None);
    let guard = from_env().expect("unset is not an error");
    assert!(guard.is_none());
    assert_eq!(crate::exec_policy::installed_name(), None);
    options::clear_fast_path_overrides();
}

/// A valid value installs, is visible to the ledger by name, and
/// uninstalls when the guard drops — the CLI holds that guard for the
/// span of a run.
#[test]
fn from_env_installs_and_the_guard_uninstalls() {
    let _g = crate::movement_ledger::bytes::COUNTER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    crate::exec_policy::uninstall();
    options::set_env_override(options::ENV_EXEC_POLICY, Some("skip-layers:20:every-4"));
    {
        let guard = from_env().expect("parses").expect("installs");
        let name = crate::exec_policy::installed_name().expect("named");
        assert!(name.contains("layers=[20]"), "{name}");
        assert!(name.contains("every-4th"), "{name}");
        drop(guard);
    }
    assert_eq!(
        crate::exec_policy::installed_name(),
        None,
        "the guard must uninstall so a later arm is not silently policed"
    );
    options::clear_fast_path_overrides();
}

/// A malformed value is an ERROR out of `from_env`, and nothing is
/// installed. The CLI turns this into a non-zero exit: a silent
/// canonical fallback would make an A/B compare a program against
/// itself and report "no change".
#[test]
fn from_env_refuses_a_malformed_value_without_installing() {
    let _g = crate::movement_ledger::bytes::COUNTER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    crate::exec_policy::uninstall();
    options::set_env_override(options::ENV_EXEC_POLICY, Some("skip-everything"));
    let err = from_env().map(|_| ()).expect_err("must refuse");
    assert!(err.contains("unrecognised exec policy"), "{err}");
    assert_eq!(crate::exec_policy::installed_name(), None);
    options::clear_fast_path_overrides();
}
