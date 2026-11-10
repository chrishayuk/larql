use std::sync::{Arc, Mutex};

use larql_lql::balance_trace::{
    BalanceDecision, BalanceEnd, BalanceObserver, BalanceStep, BalanceTermination, StepContext,
    TraceEvent, TraceRecorder,
};

use crate::support::*;

fn steps(events: &[TraceEvent]) -> Vec<&BalanceStep> {
    events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::Step(s) = e {
                Some(s)
            } else {
                None
            }
        })
        .collect()
}

fn ends(events: &[TraceEvent]) -> Vec<&BalanceEnd> {
    events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::End(s) = e {
                Some(s)
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn an_install_with_no_observer_is_deterministic_so_the_comparisons_below_mean_something() {
    let fixture = Fixture::new();
    let a = install_and_save(
        &fixture,
        &mut fixture.session(),
        "a.vlp",
        &[INSERT, INSERT_SECOND],
    );
    let b = install_and_save(
        &fixture,
        &mut fixture.session(),
        "b.vlp",
        &[INSERT, INSERT_SECOND],
    );
    assert!(!a.is_empty());
    assert_eq!(
        a, b,
        "two plain installs of the same facts must save the same bytes"
    );
}

#[test]
fn an_observer_changes_neither_the_saved_install_nor_its_predictions() {
    let fixture = Fixture::new();
    let mut plain = fixture.session();
    let plain_bytes = install_and_save(&fixture, &mut plain, "plain.vlp", &[INSERT, INSERT_SECOND]);

    let recorder = TraceRecorder::new();
    let mut observed = fixture.session();
    observed.set_balance_observer(Some(Box::new(recorder.clone())));
    let observed_bytes = install_and_save(
        &fixture,
        &mut observed,
        "observed.vlp",
        &[INSERT, INSERT_SECOND],
    );

    assert_eq!(
        plain_bytes, observed_bytes,
        "the saved install must be bit-identical"
    );
    for prompt in ["The a of is", "The c of is", "a b c"] {
        assert_eq!(
            infer(&mut plain, prompt),
            infer(&mut observed, prompt),
            "{prompt}"
        );
    }
    assert!(
        !recorder.events().is_empty(),
        "the observer must have been called"
    );
    assert_eq!(observed.balance_observer_failures(), 0);
}

#[test]
fn every_iteration_is_reported_once_and_agrees_with_the_band() {
    let fixture = Fixture::new();
    let recorder = TraceRecorder::new();
    let mut session = fixture.session();
    session.set_balance_observer(Some(Box::new(recorder.clone())));
    run(&mut session, INSERT);

    let events = recorder.events();
    let steps = steps(&events);
    let ends = ends(&events);
    assert_eq!(ends.len(), 1, "one install, one end");
    assert!(!steps.is_empty());
    assert_eq!(
        steps.len(),
        ends[0].iterations,
        "one step per iteration that read a probability"
    );
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(step.iteration, i, "iterations are consecutive from zero");
        assert!((0.0..=1.0).contains(&step.controller_target_prob));
        assert!(step.down_norm.is_finite());
        let p = step.controller_target_prob;
        match step.decision {
            BalanceDecision::InBand => assert!((FLOOR..=CEILING).contains(&p), "iter {i}: {p}"),
            BalanceDecision::Amplify { factor } => {
                assert!(p < FLOOR, "iter {i}: {p}");
                assert!(factor > 1.0);
            }
            BalanceDecision::Shrink { factor } => {
                assert!(p > CEILING, "iter {i}: {p}");
                assert!((0.0..1.0).contains(&factor));
            }
            BalanceDecision::Saturated => assert!(p < FLOOR, "iter {i}: {p}"),
        }
    }
    let last = steps.last().unwrap();
    match ends[0].termination {
        BalanceTermination::InBand => assert_eq!(last.decision, BalanceDecision::InBand),
        BalanceTermination::Saturated => assert_eq!(last.decision, BalanceDecision::Saturated),
        BalanceTermination::IterationsExhausted => {
            assert!(matches!(
                last.decision,
                BalanceDecision::Amplify { .. } | BalanceDecision::Shrink { .. }
            ))
        }
    }
    assert!(ends[0].final_down_norm.is_finite());
}

#[test]
fn the_reported_down_norms_follow_and_anchor_to_the_controllers_own_state() {
    let fixture = Fixture::new();
    let recorder = TraceRecorder::new();
    let mut session = fixture.session();
    session.set_balance_observer(Some(Box::new(recorder.clone())));
    run(&mut session, INSERT);
    run(&mut session, INSERT_SECOND);
    let events = recorder.events();

    // Split into installs at each End.
    let mut installs: Vec<(Vec<&BalanceStep>, &BalanceEnd)> = Vec::new();
    let mut open: Vec<&BalanceStep> = Vec::new();
    for event in &events {
        match event {
            TraceEvent::Step(s) => open.push(s),
            TraceEvent::End(e) => installs.push((std::mem::take(&mut open), e)),
        }
    }
    assert_eq!(installs.len(), 2);
    assert!(open.is_empty(), "every install ends");

    let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b.abs().max(1.0);
    for (steps, end) in &installs {
        // Each reported norm is the previous one scaled by the decision taken there.
        for pair in steps.windows(2) {
            if let BalanceDecision::Amplify { factor } | BalanceDecision::Shrink { factor } =
                pair[0].decision
            {
                assert!(
                    close(pair[1].down_norm, pair[0].down_norm * factor),
                    "{pair:?}"
                );
            }
        }
        // Absolute anchors, which a report made after the controller acted would break: the
        // pass keeps the down vector it snapshotted at the first amplify-mode step with the
        // highest probability, as it was before that step's scaling...
        if end.rolled_back {
            let mut best: Option<&BalanceStep> = None;
            for step in steps.iter().filter(|s| {
                matches!(
                    s.decision,
                    BalanceDecision::Amplify { .. } | BalanceDecision::Saturated
                )
            }) {
                if best.is_none_or(|b| step.controller_target_prob > b.controller_target_prob) {
                    best = Some(step);
                }
            }
            let best = best.expect("a rollback needs an amplify-mode step");
            assert!(
                close(end.final_down_norm, best.down_norm),
                "{best:?} vs {end:?}"
            );
        }
        // ...and an install that stops in band keeps exactly what the last step reported.
        if end.termination == BalanceTermination::InBand {
            assert!(close(end.final_down_norm, steps.last().unwrap().down_norm));
        }
    }
    assert!(
        installs.iter().any(|(_, e)| e.rolled_back),
        "the fixture must exercise a rollback or the anchor above checks nothing"
    );
}

/// Makes forward passes through the context at every step, as a harness measuring neighbours does.
struct Probing {
    seen: Arc<Mutex<Vec<(String, usize)>>>,
}

impl BalanceObserver for Probing {
    fn step(&mut self, _step: &BalanceStep, context: &StepContext<'_>) {
        let top = context
            .top_k(context.canonical_prompt(), 3)
            .expect("a forward pass");
        let decoded = context.decode(top[0].0);
        self.seen.lock().unwrap().push((decoded, top.len()));
        assert!(
            !context.entity().is_empty()
                && !context.relation().is_empty()
                && !context.target().is_empty()
        );
    }
    fn end(&mut self, _end: &BalanceEnd, context: &StepContext<'_>) {
        context
            .top_k("The a of is", 1)
            .expect("a forward pass at the end");
    }
}

#[test]
fn forward_passes_made_by_the_observer_do_not_change_the_install() {
    let fixture = Fixture::new();
    let plain_bytes = install_and_save(
        &fixture,
        &mut fixture.session(),
        "plain.vlp",
        &[INSERT, INSERT_SECOND],
    );

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut probing = fixture.session();
    probing.set_balance_observer(Some(Box::new(Probing { seen: seen.clone() })));
    let probed_bytes = install_and_save(
        &fixture,
        &mut probing,
        "probed.vlp",
        &[INSERT, INSERT_SECOND],
    );

    assert_eq!(
        plain_bytes, probed_bytes,
        "reading the composed program must not alter it"
    );
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(
        seen.iter().all(|(_, n)| *n == 3),
        "each probe returned its top 3"
    );
}

struct Panicking {
    calls: Arc<Mutex<usize>>,
}

impl BalanceObserver for Panicking {
    fn step(&mut self, _: &BalanceStep, _: &StepContext<'_>) {
        *self.calls.lock().unwrap() += 1;
        panic!("observer failure");
    }
    fn end(&mut self, _: &BalanceEnd, _: &StepContext<'_>) {
        *self.calls.lock().unwrap() += 1;
    }
}

#[test]
fn a_panicking_observer_is_detached_and_the_install_is_unchanged() {
    let fixture = Fixture::new();
    let plain_bytes = install_and_save(
        &fixture,
        &mut fixture.session(),
        "plain.vlp",
        &[INSERT, INSERT_SECOND],
    );

    let calls = Arc::new(Mutex::new(0));
    let mut broken = fixture.session();
    broken.set_balance_observer(Some(Box::new(Panicking {
        calls: calls.clone(),
    })));
    let broken_bytes = install_and_save(
        &fixture,
        &mut broken,
        "broken.vlp",
        &[INSERT, INSERT_SECOND],
    );

    assert_eq!(
        plain_bytes, broken_bytes,
        "an observer failure must not touch the edit"
    );
    assert_eq!(
        *calls.lock().unwrap(),
        1,
        "it is detached at its first panic and never called again"
    );
    assert_eq!(broken.balance_observer_failures(), 1);
}

#[test]
fn an_observer_stays_for_later_installs_and_none_removes_it() {
    let fixture = Fixture::new();
    let recorder = TraceRecorder::new();
    let mut session = fixture.session();
    session.set_balance_observer(Some(Box::new(recorder.clone())));
    run(&mut session, INSERT);
    run(&mut session, INSERT_SECOND);
    assert_eq!(
        ends(&recorder.events()).len(),
        2,
        "both installs were observed"
    );

    session.set_balance_observer(None);
    let before = recorder.events().len();
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "c", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );
    assert_eq!(
        recorder.events().len(),
        before,
        "a removed observer sees nothing more"
    );
}

/// Panics in `step` and again when it is dropped.
struct DropBomb;

impl BalanceObserver for DropBomb {
    fn step(&mut self, _: &BalanceStep, _: &StepContext<'_>) {
        panic!("observer failure");
    }
    fn end(&mut self, _: &BalanceEnd, _: &StepContext<'_>) {}
}

impl Drop for DropBomb {
    fn drop(&mut self) {
        panic!("observer destructor failure");
    }
}

#[test]
fn an_observer_whose_destructor_also_panics_still_cannot_reach_the_install() {
    let fixture = Fixture::new();
    let plain_bytes = install_and_save(&fixture, &mut fixture.session(), "plain.vlp", &[INSERT]);

    let mut broken = fixture.session();
    broken.set_balance_observer(Some(Box::new(DropBomb)));
    let broken_bytes = install_and_save(&fixture, &mut broken, "broken.vlp", &[INSERT]);

    assert_eq!(plain_bytes, broken_bytes);
    assert_eq!(broken.balance_observer_failures(), 1);
}
