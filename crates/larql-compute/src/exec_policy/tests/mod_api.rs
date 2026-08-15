use super::policies::{LayerStepMask, StepSelector};
use super::*;
use crate::movement_ledger::bytes::{self, COUNTER_LOCK};
use crate::movement_ledger::{decisions, Phase, PhaseScope, Tier};

/// gpt-oss-20b's real per-layer expert traffic under Q6_K, rounded: one
/// top-4 group at 24 layers is the banked 2.09 GB/token. Using a real
/// magnitude rather than `1` keeps the avoided-byte assertions honest
/// about the scale they operate at.
const GROUP_PHYSICAL: u64 = 87_000_000;
const GROUP_SEMANTIC: u64 = 81_500_000;
const SLOTS: usize = 4;

fn group_movement() -> OperandMovement {
    OperandMovement::fully_consumed(GROUP_SEMANTIC, GROUP_PHYSICAL, Tier::Dram)
}

/// Every counter this module touches is process-wide, so each test must
/// hold the ledger's own serialiser and start from a known zero.
fn fresh() {
    uninstall();
    step::reset();
    decisions::reset();
    bytes::reset_for_test();
    coverage::reset_for_test();
}

/// The default. No policy installed means canonical execution, the bytes
/// land where they always did, and the surface fires — an engine built
/// with this module behaves exactly like one built without it.
#[test]
fn no_policy_means_canonical_and_bytes_move() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();

    let strategy = resolve_expert_group(20, SLOTS, group_movement());
    assert_eq!(strategy, ExecutionStrategy::Canonical);
    assert_eq!(installed_name(), None, "no policy to name");

    let b = bytes::snapshot();
    assert_eq!(b.physical_touched, GROUP_PHYSICAL);
    assert_eq!(Surface::MoeExperts.fired(), 1, "the surface must fire");

    let d = decisions::snapshot();
    assert_eq!(d.requested, 1);
    assert_eq!(d.executed, 1);
    assert_eq!(d.skipped, 0);
    assert_eq!(d.physical_avoided, 0);
    fresh();
}

/// The denominator runs unconditionally. Without it a "0% skip rate" and
/// "the seam was never reached" would be the same reading.
#[test]
fn requested_counts_even_with_no_policy_installed() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();
    for layer in 0..24 {
        resolve_expert_group(layer, SLOTS, group_movement());
    }
    let d = decisions::snapshot();
    assert_eq!(d.requested, 24);
    assert_eq!(d.skip_rate(), Some(0.0));
    assert!(d.is_measured());
    fresh();
}

/// A skip records avoided bytes and moves NO byte counter. Folding
/// avoided bytes into `physical_touched` would put a byte that never
/// crossed the memory bus into a bandwidth measurement.
#[test]
fn a_skip_records_avoided_bytes_and_moves_none() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();
    let _p = PhaseScope::new(Phase::Decode);
    step::advance();

    let _guard = install(Arc::new(LayerStepMask::new([20]).with_phase(Phase::Decode)));
    let strategy = resolve_expert_group(20, SLOTS, group_movement());
    assert_eq!(strategy, ExecutionStrategy::Skip);

    let b = bytes::snapshot();
    assert_eq!(b.physical_touched, 0, "a skipped group moves no bytes");
    assert_eq!(b.semantic_requested, 0);
    assert_eq!(
        Surface::MoeExperts.fired(),
        0,
        "coverage evidence is about bytes that MOVED"
    );

    let d = decisions::snapshot();
    assert_eq!(d.requested, 1);
    assert_eq!(d.executed, 0);
    assert_eq!(d.skipped, 1);
    assert_eq!(d.physical_avoided, GROUP_PHYSICAL);
    assert_eq!(d.semantic_avoided, GROUP_SEMANTIC);
    drop(_guard);
    fresh();
}

/// `touched + avoided` reconstructs the canonical arm's traffic. This is
/// the property that makes a skipping run comparable to a canonical one
/// without re-running it, and it only holds because both arms price the
/// operation with the same shape arithmetic.
#[test]
fn touched_plus_avoided_reconstructs_the_canonical_arm() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();
    let _p = PhaseScope::new(Phase::Decode);
    step::advance();

    const LAYERS: usize = 24;
    let canonical_total = GROUP_PHYSICAL * LAYERS as u64;

    let _guard = install(Arc::new(
        LayerStepMask::new([20, 21, 22]).with_phase(Phase::Decode),
    ));
    for layer in 0..LAYERS {
        resolve_expert_group(layer, SLOTS, group_movement());
    }
    let b = bytes::snapshot();
    let d = decisions::snapshot();

    assert_eq!(d.requested, LAYERS as u64);
    assert_eq!(d.skipped, 3);
    assert!(d.is_consistent());
    assert_eq!(
        b.physical_touched + d.physical_avoided,
        canonical_total,
        "the two terms must partition what the canonical arm would move"
    );
    assert_eq!(d.avoided_share(b.physical_touched), Some(3.0 / 24.0));
    drop(_guard);
    fresh();
}

/// Dropping the guard restores canonical execution. A policy left
/// installed by one arm would silently change every later arm's numerics
/// — which is precisely the failure this whole module exists to make
/// visible rather than commit.
#[test]
fn dropping_the_guard_restores_canonical_execution() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();
    let _p = PhaseScope::new(Phase::Decode);
    step::advance();

    {
        let _guard = install(Arc::new(LayerStepMask::new([20]).with_phase(Phase::Decode)));
        assert!(installed_name().is_some());
        assert_eq!(decide_expert_group(20, SLOTS), ExecutionStrategy::Skip);
    }
    assert_eq!(installed_name(), None);
    assert_eq!(
        decide_expert_group(20, SLOTS),
        ExecutionStrategy::Canonical,
        "the same site that skipped must run canonically once disarmed"
    );
    fresh();
}

/// The installed name reaches the report. A skip rate printed without the
/// policy that produced it is unreadable — and "0 skips under policy X"
/// and "0 skips with no policy" are different facts.
#[test]
fn installed_name_reports_the_actual_policy() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();
    let _guard = install(Arc::new(
        LayerStepMask::new([20])
            .with_phase(Phase::Decode)
            .with_steps(StepSelector::Exactly(0)),
    ));
    let name = installed_name().expect("a policy is installed");
    assert!(name.contains("layers=[20]"), "{name}");
    assert!(name.contains("steps=exactly(0)"), "{name}");
    drop(_guard);
    fresh();
}

/// The site a policy sees carries the ambient phase and step, not values
/// the dispatch path had to thread through its own signature. A policy
/// that never saw them could not express "skip on decode step 3".
#[test]
fn the_site_carries_the_ambient_phase_and_step() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();

    /// Records the site it was asked about, so the test can inspect what
    /// the seam actually built rather than what it hoped it built.
    struct Recorder(std::sync::Mutex<Vec<ExpertGroupSite>>);
    impl ExecutionPolicy for Recorder {
        fn name(&self) -> &str {
            "recorder"
        }
        fn expert_group(&self, site: &ExpertGroupSite) -> ExecutionStrategy {
            self.0.lock().unwrap().push(*site);
            ExecutionStrategy::Canonical
        }
    }

    let rec = Arc::new(Recorder(std::sync::Mutex::new(Vec::new())));
    let _guard = install(rec.clone());
    let _p = PhaseScope::new(Phase::Decode);
    step::advance();
    step::advance();
    resolve_expert_group(20, SLOTS, group_movement());

    let seen = rec.0.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].layer, 20);
    assert_eq!(seen[0].slots, SLOTS);
    assert_eq!(seen[0].phase, Some(Phase::Decode));
    assert_eq!(seen[0].step, Some(1), "two advances = zero-based index 1");
    drop(_guard);
    fresh();
}

/// A policy that only overrides `name` inherits the safe default — it
/// must not have to opt back in to correctness for an operation class it
/// does not care about.
#[test]
fn the_trait_default_is_canonical() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fresh();

    struct Indifferent;
    impl ExecutionPolicy for Indifferent {
        fn name(&self) -> &str {
            "indifferent"
        }
    }

    let _guard = install(Arc::new(Indifferent));
    assert_eq!(decide_expert_group(20, SLOTS), ExecutionStrategy::Canonical);
    drop(_guard);
    fresh();
}
