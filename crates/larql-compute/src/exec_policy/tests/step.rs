use super::*;
use crate::movement_ledger::bytes::COUNTER_LOCK;
use crate::movement_ledger::PhaseScope;

/// Before any boundary has been crossed there is no token to address.
/// `None`, never `0` — `0` is a legitimate index a policy selects on, so
/// returning it here would make "skip on token 0" fire before the first
/// token existed.
#[test]
fn current_is_none_before_the_first_boundary() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let _p = PhaseScope::new(Phase::Decode);
    assert_eq!(current(), None);
}

/// The first advance yields index 0, and it counts from there.
#[test]
fn indices_are_zero_based_and_monotonic() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let _p = PhaseScope::new(Phase::Decode);
    advance();
    assert_eq!(current(), Some(0));
    advance();
    advance();
    assert_eq!(current(), Some(2));
    reset();
}

/// The load-bearing property: prefill positions and decode steps have
/// SEPARATE indices. A single counter would put `gpt-oss-20b`'s ~130
/// chat-template prefill positions in front of decode step 0, so a policy
/// written as "skip on token 7" would fire inside the system prompt.
#[test]
fn prefill_and_decode_indices_are_independent() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    {
        let _p = PhaseScope::new(Phase::Prefill);
        for _ in 0..130 {
            advance();
        }
        assert_eq!(current(), Some(129));
    }
    let _d = PhaseScope::new(Phase::Decode);
    assert_eq!(current(), None, "decode has not started yet");
    advance();
    assert_eq!(current(), Some(0), "decode's first step is index 0");
    reset();
}

/// A boundary crossed with no phase scope advances nothing — an
/// unattributed driver loop must not corrupt either phase's index. Same
/// refusal contract the ledger's own phase attribution uses.
#[test]
fn an_unattributed_boundary_advances_nothing() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    {
        let _p = PhaseScope::new(Phase::Decode);
        advance();
        assert_eq!(current(), Some(0));
    }
    // No scope active: neither the read nor the write is attributable.
    advance();
    advance();
    assert_eq!(current(), None, "no phase declared — refuse, do not guess");

    let _p = PhaseScope::new(Phase::Decode);
    assert_eq!(
        current(),
        Some(0),
        "the unattributed advances must not have moved decode's index"
    );
    reset();
}

/// `reset` restarts both buckets, so a harness running two arms in one
/// process can give arm B its own step 0.
#[test]
fn reset_restarts_both_phases() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    {
        let _p = PhaseScope::new(Phase::Prefill);
        advance();
    }
    {
        let _p = PhaseScope::new(Phase::Decode);
        advance();
    }
    reset();
    let _p = PhaseScope::new(Phase::Prefill);
    assert_eq!(current(), None);
    drop(_p);
    let _d = PhaseScope::new(Phase::Decode);
    assert_eq!(current(), None);
}
