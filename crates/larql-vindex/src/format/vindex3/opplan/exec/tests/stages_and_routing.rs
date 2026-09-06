//! The stage ledger and the routing trace, on their own terms: a stage
//! records what it contains, a stage inside a stage is counted and voids
//! the sum, a capture returns exactly the selections made while it was
//! open, and a fingerprint tells two selections apart by order.

use super::super::routing_trace;
use super::super::stages::{ledger, stage, Stage, StageTally};

/// The ledger is process-wide, so these tests hold one lock between them
/// rather than race each other's counters.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn a_stage_records_its_extent_and_a_reset_clears_it() {
    let _serial = SERIAL.lock().unwrap();
    ledger().reset();
    {
        let _s = stage(Stage::Router);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let router = ledger().get(Stage::Router);
    assert_eq!(router.calls, 1);
    assert!(router.nanos >= 2_000_000, "{router:?}");
    assert_eq!(ledger().get(Stage::Attention), StageTally::default());
    assert_eq!(ledger().nested(), 0);
    assert_eq!(ledger().total_nanos(), router.nanos);
    ledger().reset();
    assert_eq!(ledger().get(Stage::Router), StageTally::default());
}

#[test]
fn a_stage_inside_a_stage_is_counted_as_nesting() {
    let _serial = SERIAL.lock().unwrap();
    ledger().reset();
    {
        let _outer = stage(Stage::Attention);
        let _inner = stage(Stage::Router);
    }
    assert_eq!(ledger().nested(), 1);
    assert_eq!(ledger().get(Stage::Attention).calls, 1);
    assert_eq!(ledger().get(Stage::Router).calls, 1);
    // Sequential stages do not nest.
    ledger().reset();
    {
        let _a = stage(Stage::RoutedExperts);
    }
    {
        let _b = stage(Stage::SharedExpert);
    }
    assert_eq!(ledger().nested(), 0);
    assert_eq!(
        ledger().all().iter().filter(|(_, t)| t.calls == 1).count(),
        2
    );
    ledger().reset();
}

#[test]
fn every_stage_has_a_distinct_name() {
    let names: std::collections::BTreeSet<&str> = Stage::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names.len(), Stage::ALL.len());
}

#[test]
fn a_capture_returns_the_selections_made_while_it_was_open() {
    let _serial = SERIAL.lock().unwrap();
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
