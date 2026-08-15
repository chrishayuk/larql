use super::*;

fn site(layer: usize, phase: Option<Phase>, step: Option<u64>) -> ExpertGroupSite {
    ExpertGroupSite {
        layer,
        phase,
        step,
        slots: 4,
    }
}

/// A trace names `(layer, step)` cells, and only those cells skip.
#[test]
fn replays_exactly_the_recorded_cells() {
    let trace = TraceReplay::new(Phase::Decode, [(20, 0), (20, 1), (22, 5)]);
    for (layer, step) in [(20usize, 0u64), (20, 1), (22, 5)] {
        assert_eq!(
            trace.expert_group(&site(layer, Some(Phase::Decode), Some(step))),
            ExecutionStrategy::Skip,
            "({layer}, {step}) is in the trace"
        );
    }
    // Same layer, unrecorded step.
    assert_eq!(
        trace.expert_group(&site(20, Some(Phase::Decode), Some(2))),
        ExecutionStrategy::Canonical
    );
    // Same step, unrecorded layer — layer and step are a joint key, not
    // two independent filters.
    assert_eq!(
        trace.expert_group(&site(21, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Canonical
    );
}

/// A trace recorded against decode steps must not fire at the same
/// indices during prefill, where step 7 is a prompt position and not the
/// token the trace is about. That is why the phase is a required
/// constructor argument rather than an option.
#[test]
fn a_decode_trace_never_fires_during_prefill() {
    let trace = TraceReplay::new(Phase::Decode, [(20, 7)]);
    assert_eq!(
        trace.expert_group(&site(20, Some(Phase::Prefill), Some(7))),
        ExecutionStrategy::Canonical
    );
    assert_eq!(
        trace.expert_group(&site(20, None, Some(7))),
        ExecutionStrategy::Canonical
    );
}

/// An undeclared step cannot be addressed by a trace. Refuse — never
/// fall through to "the next recorded one".
#[test]
fn an_undeclared_step_is_refused() {
    let trace = TraceReplay::new(Phase::Decode, [(20, 0)]);
    assert_eq!(
        trace.expert_group(&site(20, Some(Phase::Decode), None)),
        ExecutionStrategy::Canonical
    );
}

/// Duplicate cells collapse, so `len` is the number of distinct skips a
/// replay should produce — the count a caller compares its observed skip
/// count against to detect that the two runs diverged.
#[test]
fn duplicate_cells_collapse_into_one_skip() {
    let trace = TraceReplay::new(Phase::Decode, [(20, 0), (20, 0), (20, 0)]);
    assert_eq!(trace.len(), 1);
    assert!(!trace.is_empty());
}

/// An empty trace is a valid null arm: it installs, reports itself, and
/// skips nothing — the control every replay result needs.
#[test]
fn an_empty_trace_skips_nothing() {
    let trace = TraceReplay::new(Phase::Decode, Vec::<(usize, u64)>::new());
    assert!(trace.is_empty());
    assert_eq!(
        trace.expert_group(&site(20, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Canonical
    );
    assert!(trace.name().contains("pairs=0"), "{}", trace.name());
}

/// The name carries the phase and the trace size, so a replay result can
/// never be read without knowing which trace produced it.
#[test]
fn name_describes_the_trace() {
    let trace = TraceReplay::new(Phase::Decode, [(20, 0), (21, 1)]);
    let name = trace.name();
    assert!(name.contains("phase=decode"), "{name}");
    assert!(name.contains("pairs=2"), "{name}");
}
