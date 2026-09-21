//! **Every refusal names what it promises to name.**
//!
//! A refusal in this programme is contract surface, not a log line. Each
//! one exists so a reader can act on it — go and build THIS map, this
//! gate has moved, the corpus you have is not the corpus the experiment
//! was identified over — and a variant that holds the facts and then
//! renders a message without them is the same dead end as no refusal at
//! all.
//!
//! So these are constructed directly rather than triggered. WHICH
//! condition produces each one is asserted in the other test modules;
//! what is checked here is that the message carries the facts the
//! variant holds. The `must_say` matches are exhaustive, so a new
//! variant cannot be added without deciding what its message must say.

use super::super::super::measure::outcome::{ExecutionFailure, MeasurementRefusal};
use super::super::super::measure::TEACHER_FORCED_TWO_ARM;
use super::super::super::measurement::EvidenceScale;
use super::super::super::state::key::MeasurementKey;
use super::super::super::state::protocol::ProtocolMismatch;
use super::super::executor::{ExecutionRefusal, LocatorRefusal, Misdirected};
use super::super::RequestRefusal;
use super::fixtures;

fn key() -> MeasurementKey {
    fixtures::key_for(&fixtures::p(), EvidenceScale::Authority)
}

fn other_key() -> MeasurementKey {
    fixtures::key_for(&fixtures::t1(), EvidenceScale::Authority)
}

fn assert_says(what: &str, said: String, needles: Vec<String>) {
    for needle in needles {
        assert!(
            said.contains(&needle),
            "{what} must name `{needle}` for a reader to act on it, and said: {said}"
        );
    }
}

// ------------------------------------------------------ request refusals

fn request_refusals() -> Vec<RequestRefusal> {
    vec![
        RequestRefusal::NoProtocol,
        RequestRefusal::Protocol(ProtocolMismatch::Bank {
            declared: fixtures::selection_bank().id(),
            searched: fixtures::selection_bank().id(),
        }),
        RequestRefusal::NoSuchMap {
            detail: "no edit named `+E24`".into(),
        },
        RequestRefusal::UnresolvableLayout {
            named: "some-layout/v1".into(),
            detail: "this build implements two".into(),
        },
        RequestRefusal::StateMismatch {
            named: key().state().clone(),
            resolved: other_key().state().clone(),
        },
        RequestRefusal::UnresolvableGate {
            named: "some-gate/v1".into(),
            detail: "no gate named that".into(),
        },
        RequestRefusal::GateRedefined {
            named: "kimi-logit-balanced-v1".into(),
        },
    ]
}

fn must_say(refusal: &RequestRefusal) -> Vec<String> {
    match refusal {
        // The absent authority, said in the reader's terms: the record
        // holds digests and not declarations.
        RequestRefusal::NoProtocol => vec!["digest".into()],
        // Delegated whole, so the mismatch's own two digests survive.
        RequestRefusal::Protocol(m) => vec![m.to_string()],
        RequestRefusal::NoSuchMap { detail } => vec![detail.clone()],
        RequestRefusal::UnresolvableLayout { named, detail } => {
            vec![named.clone(), detail.clone()]
        }
        // BOTH states: a reader diffing one digest against nothing
        // cannot tell which end moved.
        RequestRefusal::StateMismatch { named, resolved } => {
            vec![named.short().into(), resolved.short().into()]
        }
        RequestRefusal::UnresolvableGate { named, detail } => {
            vec![named.clone(), detail.clone()]
        }
        RequestRefusal::GateRedefined { named } => vec![named.clone()],
    }
}

#[test]
fn every_request_refusal_names_what_it_promises() {
    for refusal in request_refusals() {
        assert_says("a request refusal", refusal.to_string(), must_say(&refusal));
    }
}

// ------------------------------------------------------ locator refusals

fn locator_refusals() -> Vec<LocatorRefusal> {
    vec![
        LocatorRefusal::NotHeld {
            what: "container".into(),
            identity: "abc123".into(),
        },
        LocatorRefusal::NotWhatItClaims {
            what: "quality bank".into(),
            path: "/tmp/somewhere".into(),
            detail: "its manifest digests to something else".into(),
        },
        LocatorRefusal::NotBuilt {
            state: key().state().clone(),
            map: "protect-everything".into(),
        },
    ]
}

fn locator_must_say(refusal: &LocatorRefusal) -> Vec<String> {
    match refusal {
        LocatorRefusal::NotHeld { what, identity } => vec![what.clone(), identity.clone()],
        LocatorRefusal::NotWhatItClaims { what, path, detail } => {
            vec![what.clone(), path.clone(), detail.clone()]
        }
        // The state AND the map: "nothing is built" is a dead end, and
        // "build this map" is an instruction.
        LocatorRefusal::NotBuilt { state, map } => vec![state.short().into(), map.clone()],
    }
}

#[test]
fn every_locator_refusal_names_what_it_promises() {
    for refusal in locator_refusals() {
        assert_says(
            "a locator refusal",
            refusal.to_string(),
            locator_must_say(&refusal),
        );
    }
}

// ---------------------------------------------------- execution refusals

fn execution_refusals() -> Vec<ExecutionRefusal> {
    vec![
        ExecutionRefusal::NoSuchProcedure {
            named: "some-future-procedure/v1".into(),
            implemented: vec![TEACHER_FORCED_TWO_ARM.into()],
        },
        // The empty case is separate: a build performing nothing must
        // not render an empty list and read as though it performs one
        // procedure whose name is blank.
        ExecutionRefusal::NoSuchProcedure {
            named: TEACHER_FORCED_TWO_ARM.into(),
            implemented: Vec::new(),
        },
        ExecutionRefusal::Locator(LocatorRefusal::NotBuilt {
            state: key().state().clone(),
            map: "protect-everything".into(),
        }),
        ExecutionRefusal::NotInstructable {
            procedure: TEACHER_FORCED_TWO_ARM.into(),
            detail: "the declared evidence bank names no samples".into(),
        },
        ExecutionRefusal::Measurement(MeasurementRefusal::Execution(
            ExecutionFailure::BackendUnavailable {
                detail: "needs a macOS build with the `gpu` feature".into(),
            },
        )),
        ExecutionRefusal::ObservedAnotherExperiment(Box::new(Misdirected {
            requested: key(),
            observed: other_key(),
        })),
    ]
}

fn execution_must_say(refusal: &ExecutionRefusal) -> Vec<String> {
    match refusal {
        ExecutionRefusal::NoSuchProcedure { named, implemented } => {
            let mut needles = vec![named.clone()];
            match implemented.is_empty() {
                true => needles.push("nothing".into()),
                false => needles.extend(implemented.iter().cloned()),
            }
            needles
        }
        // Delegated whole, so a locator's own instruction survives
        // being wrapped.
        ExecutionRefusal::Locator(r) => vec![r.to_string()],
        ExecutionRefusal::NotInstructable { procedure, detail } => {
            vec![procedure.clone(), detail.clone()]
        }
        ExecutionRefusal::Measurement(r) => vec![r.to_string()],
        // BOTH experiments, for the same reason as StateMismatch.
        ExecutionRefusal::ObservedAnotherExperiment(m) => {
            vec![m.requested.short().into(), m.observed.short().into()]
        }
    }
}

#[test]
fn every_execution_refusal_names_what_it_promises() {
    for refusal in execution_refusals() {
        assert_says(
            "an execution refusal",
            refusal.to_string(),
            execution_must_say(&refusal),
        );
    }
}

/// A wrapped refusal must not lose the inner one's facts. Both `Locator`
/// and `Measurement` delegate, and delegation is the kind of thing a
/// later edit quietly replaces with a summary.
#[test]
fn wrapping_a_refusal_keeps_the_inner_ones_message() {
    let inner = LocatorRefusal::NotBuilt {
        state: key().state().clone(),
        map: "protect-everything".into(),
    };
    let wrapped: ExecutionRefusal = inner.clone().into();
    assert_eq!(wrapped.to_string(), inner.to_string());
}
