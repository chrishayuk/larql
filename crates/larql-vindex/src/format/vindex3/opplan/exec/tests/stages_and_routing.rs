//! The stage ledger and the routing trace, on their own terms: a stage
//! records what it contains, a stage inside a stage is counted and voids
//! the sum, a capture returns exactly the selections made while it was
//! open, and a fingerprint tells two selections apart by order.

use super::super::routing_trace;
use super::super::stages::{ledger, stage, Stage, StageLedger, StageTally};

/// An exact tally is a claim about one ledger's own writes, so these
/// tests open their stages on a ledger they construct.
///
/// They used to assert on the process ledger under a module lock, which
/// is not exclusion: `stage()` is opened by real execution in
/// `production.rs`, `decode.rs` and `experts.rs`, so every other test in
/// this binary that executes a layer writes the same counters. The lock
/// serialised three tests against each other and nothing against the
/// other four and a half thousand. It read `calls == 11` on a CI runner
/// where the test had opened one stage.
fn ledger_of_its_own() -> StageLedger {
    StageLedger::new()
}

#[test]
fn a_stage_records_its_extent_and_a_reset_clears_it() {
    let ledger = ledger_of_its_own();
    {
        let _s = ledger.staged(Stage::Router);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let router = ledger.get(Stage::Router);
    assert_eq!(router.calls, 1);
    assert!(router.nanos >= 2_000_000, "{router:?}");
    assert_eq!(ledger.get(Stage::Attention), StageTally::default());
    assert_eq!(ledger.nested(), 0);
    assert_eq!(ledger.total_nanos(), router.nanos);
    ledger.reset();
    assert_eq!(ledger.get(Stage::Router), StageTally::default());
}

#[test]
fn a_stage_inside_a_stage_is_counted_as_nesting() {
    let ledger = ledger_of_its_own();
    {
        let _outer = ledger.staged(Stage::Attention);
        let _inner = ledger.staged(Stage::Router);
    }
    assert_eq!(ledger.nested(), 1);
    assert_eq!(ledger.get(Stage::Attention).calls, 1);
    assert_eq!(ledger.get(Stage::Router).calls, 1);
    // Sequential stages do not nest.
    ledger.reset();
    {
        let _a = ledger.staged(Stage::RoutedExperts);
    }
    {
        let _b = ledger.staged(Stage::SharedExpert);
    }
    assert_eq!(ledger.nested(), 0);
    assert_eq!(ledger.all().iter().filter(|(_, t)| t.calls == 1).count(), 2);
}

#[test]
fn two_ledgers_do_not_see_each_others_stages() {
    let mine = ledger_of_its_own();
    let theirs = ledger_of_its_own();
    {
        let _s = mine.staged(Stage::Router);
    }
    assert_eq!(mine.get(Stage::Router).calls, 1);
    assert_eq!(theirs.get(Stage::Router), StageTally::default());
    assert_eq!(theirs.total_nanos(), 0);
}

/// The witness that the fix is a fix, rather than a rewrite that happens
/// to pass. It reproduces the contention the CI failure was made of —
/// other threads opening the same stage through the ordinary `stage()`
/// entry point — and asserts an exact tally straight through it.
///
/// Before this change the assertion below WAS the failing assertion, and
/// under this much traffic it fails every run rather than one in fifty:
/// the counters it read were `LEDGER`'s, and these threads write them.
#[test]
fn an_owned_tally_is_exact_while_the_process_ledger_is_hammered() {
    const FOREIGN: u64 = 32;
    let mine = ledger_of_its_own();
    let before = ledger().get(Stage::Router).calls;

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..(FOREIGN / 4) {
                    let _foreign = stage(Stage::Router);
                }
            });
        }
        // Opened while the foreign threads are live, not after they join.
        let _s = mine.staged(Stage::Router);
        std::thread::sleep(std::time::Duration::from_millis(2));
    });

    let router = mine.get(Stage::Router);
    assert_eq!(router.calls, 1, "an owned ledger records only its own");
    assert!(router.nanos >= 2_000_000, "{router:?}");
    assert_eq!(mine.total_nanos(), router.nanos);

    // And the traffic was real: the shared ledger took every foreign
    // stage. Without this the test above could pass by nothing happening.
    let after = ledger().get(Stage::Router).calls;
    assert!(
        after >= before + FOREIGN,
        "the foreign stages must land on the process ledger: {before} -> {after}"
    );
}

#[test]
fn the_free_helper_records_into_the_process_ledger() {
    // The wiring witness for `stage()`, and the one claim that survives
    // concurrency: the process ledger is shared, so this asserts that it
    // MOVED, never what it now reads.
    //
    // A delta is only sound while the counters rise monotonically, and
    // nothing in THIS binary resets them — the reads below are the only
    // uses of the process ledger left in the crate. `larql-cli` does
    // reset it between generations, which is right for measuring one
    // generation and is a different process from this test binary.
    let before = ledger().get(Stage::Prefetch);
    {
        let _s = stage(Stage::Prefetch);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let after = ledger().get(Stage::Prefetch);
    assert!(
        after.calls > before.calls,
        "stage() must reach the process ledger: {before:?} -> {after:?}"
    );
    assert!(
        after.nanos >= before.nanos + 2_000_000,
        "and carry the extent: {before:?} -> {after:?}"
    );
}

#[test]
fn every_stage_has_a_distinct_name() {
    let names: std::collections::BTreeSet<&str> = Stage::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names.len(), Stage::ALL.len());
}

/// KNOWN HAZARD, deliberately not fixed here. `routing_trace`'s capture
/// is a process-global `Mutex<Option<Vec<..>>>`, and `production.rs`
/// records into it from the same function that opens the router stage —
/// so a concurrent execution appends foreign selections to an open
/// capture, in production as well as under test.
///
/// The stage ledger above could be fixed without deciding anything,
/// because execution keeps the process ledger and only the tests moved
/// off it. This one cannot: making the capture sound means deciding
/// whether a capture is scoped to an execution or to a thread, and
/// whether the router always runs on the thread that opened it. That is
/// a production-semantics question about execution-scoped accounting,
/// not a test-flake fix, and it is not settled as a drive-by here.
#[test]
fn a_capture_returns_the_selections_made_while_it_was_open() {
    routing_trace::record(&[(1, 0.5)]);
    routing_trace::start_capture();
    routing_trace::record(&[(3, 0.7), (1, 0.3)]);
    routing_trace::record(&[(0, 1.0)]);
    let captured = routing_trace::take_capture();
    assert_eq!(captured, vec![vec![3, 1], vec![0]]);
    routing_trace::record(&[(9, 1.0)]);
    assert!(
        routing_trace::take_capture().is_empty(),
        "closed captures record nothing"
    );
}

#[test]
fn a_fingerprint_is_order_sensitive_and_layer_aware() {
    let a = routing_trace::fingerprint(&[vec![3, 1], vec![0]]);
    let b = routing_trace::fingerprint(&[vec![1, 3], vec![0]]);
    let c = routing_trace::fingerprint(&[vec![3], vec![1, 0]]);
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_eq!(a, routing_trace::fingerprint(&[vec![3, 1], vec![0]]));
}
