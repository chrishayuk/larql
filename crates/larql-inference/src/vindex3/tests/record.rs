//! V3-STREAM-1 witnesses (`docs/v3-stream-1-run-record.md`): S1 lossless,
//! S2 replay identity, S3 sequencing, S4 bounded and counted live loss,
//! S5 observation parity through the recorder, S6 receipt integrity, S7
//! provenance on the record; S8 on a real container behind
//! `LARQL_V3_CONTAINER`.

use std::sync::mpsc::sync_channel;

use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_HIDDEN, G_LAYERS, G_TOKENS};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::observe::{
    NoopObserver, RecordingObserver, StepObserver,
};
use larql_vindex::format::vindex3::opplan::exec::observe_stats::{FixedBasis, StatsObserver};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::provenance::{ExecutionProvenance, RunProvenance};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use crate::vindex3::record::{
    hash_lines, EventKind, RecordError, RecordedEvent, RunIdentity, RunRecord, RunRecorder,
    RECORD_SCHEMA,
};

const COMPONENT: &str = "target";
const DIMS: usize = 3;
const SEED: u64 = 11;
/// F3: a tap this small on the golden run must drop most of it.
const SMALL_TAP: usize = 8;
const LARGE_TAP: usize = 1_000;

struct Fixture {
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn fixture() -> Fixture {
    let container = super::container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _container: container,
        plan: outcome.plan.unwrap(),
        store,
    }
}

fn stats() -> StatsObserver {
    StatsObserver::new(FixedBasis::seeded(G_HIDDEN, DIMS, SEED).unwrap(), None)
}

fn identity() -> RunIdentity {
    RunIdentity::new("run-test", "mini-glimmer", COMPONENT, &G_TOKENS)
}

/// Step every token with `observer`, returning the per-position logits.
fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    observer: &mut dyn StepObserver,
) -> Vec<Vec<f32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    G_TOKENS
        .iter()
        .map(|&t| session.step_observed(t, observer).unwrap().logits.unwrap())
        .collect()
}

/// Part-by-part equality that names the first differing event instead
/// of dumping two records.
fn assert_record_eq(read: &RunRecord, written: &RunRecord) {
    assert_eq!(read.identity, written.identity, "identity");
    assert_eq!(read.provenance, written.provenance, "provenance");
    assert_eq!(
        read.provenance_fingerprint, written.provenance_fingerprint,
        "fingerprint"
    );
    assert_eq!(read.receipt, written.receipt, "receipt");
    assert_eq!(read.events.len(), written.events.len(), "event count");
    for (index, (a, b)) in read.events.iter().zip(&written.events).enumerate() {
        assert_eq!(a, b, "event {index} differs after the round trip");
    }
    assert_eq!(read, written);
}

/// F1: per position 1 embedded + 1 entering + per layer 2 sites × (stats,
/// write, boundary) + 1 logits.
fn forecast_events_per_position() -> usize {
    1 + 1 + G_LAYERS * 2 * 3 + 1
}

#[test]
fn s1_the_recorder_is_lossless_and_matches_the_forecast() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let mut plain = RecordingObserver::default();
    run(&f.plan, &ops, &backend, &mut plain);
    let mut recorder = RunRecorder::for_image(identity(), &ops, Some(stats()));
    run(&f.plan, &ops, &backend, &mut recorder);
    let structural = plain.events.len();
    let writes = 2 * G_LAYERS * G_TOKENS.len();
    let entering = G_TOKENS.len();
    assert_eq!(recorder.events().len(), structural + writes + entering);
    assert_eq!(
        recorder.events().len(),
        forecast_events_per_position() * G_TOKENS.len(),
        "F1"
    );
    // Stats precede their write, as the executor fires them.
    let kinds: Vec<&EventKind> = recorder.events().iter().map(|e| &e.event).collect();
    let first_stats = kinds
        .iter()
        .position(|k| matches!(k, EventKind::CarrierStats { .. }))
        .unwrap();
    assert!(matches!(
        kinds[first_stats + 1],
        EventKind::CarrierWrite { .. }
    ));
    assert!(matches!(kinds[0], EventKind::Embedded));
    assert_eq!(recorder.events()[0].position, 0);
    assert!(matches!(kinds[1], EventKind::EnteringCarrier { hidden } if *hidden == G_HIDDEN));
    assert_eq!(recorder.events()[1].position, 0);
    assert!(!kinds.iter().any(|k| matches!(k, EventKind::Unknown { .. })));
}

#[test]
fn s3_sequences_are_gapless_and_timestamps_and_positions_are_monotonic() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let mut recorder = RunRecorder::for_image(identity(), &ops, Some(stats()));
    run(&f.plan, &ops, &backend, &mut recorder);
    let mut last_ts = 0u64;
    let mut last_pos = 0usize;
    for (index, event) in recorder.events().iter().enumerate() {
        assert_eq!(event.sequence, index as u64);
        assert!(event.timestamp_ns >= last_ts);
        assert!(event.position >= last_pos);
        last_ts = event.timestamp_ns;
        last_pos = event.position;
    }
    assert_eq!(last_pos, G_TOKENS.len() - 1);
    let per_position = forecast_events_per_position();
    for (index, event) in recorder.events().iter().enumerate() {
        assert_eq!(event.position, index / per_position, "event {index}");
    }
}

#[test]
fn s2_s6_s7_a_written_record_reads_back_equal_verified_and_carries_its_provenance() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let mut recorder = RunRecorder::for_image(identity(), &ops, Some(stats()));
    run(&f.plan, &ops, &backend, &mut recorder);
    recorder.complete();
    // S7: the provenance is what the executor states for this image.
    let expected = RunProvenance::new(ExecutionProvenance::of(&ops), Some(&stats()));
    assert_eq!(recorder.provenance(), &expected);
    let record = recorder.finish();
    assert_eq!(record.provenance_fingerprint, expected.fingerprint());
    assert_eq!(
        record.receipt.provenance_fingerprint,
        expected.fingerprint()
    );
    assert_eq!(record.identity.schema, RECORD_SCHEMA);
    assert!(record.receipt.complete);
    assert_eq!(record.receipt.events, record.events.len() as u64);
    assert_eq!(
        record.receipt.last_sequence,
        Some(record.events.len() as u64 - 1)
    );
    assert_eq!(record.receipt.live_dropped, 0);

    // S2: round trip, every float by bits (PartialEq on f32/f64 is bit
    // equality for finite values, and the rows are finite).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.jsonl");
    record.write_jsonl(&path).unwrap();
    let read = RunRecord::read_jsonl(&path).unwrap();
    assert_record_eq(&read, &record);
    assert!(record
        .events
        .iter()
        .flat_map(|e| match &e.event {
            EventKind::CarrierStats { projection, .. } => projection.clone(),
            _ => Vec::new(),
        })
        .all(f32::is_finite));

    // S6: the receipt hashes the event lines and nothing else.
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].contains("\"header\""));
    assert!(lines[lines.len() - 1].contains("\"receipt\""));
    let event_lines: Vec<String> = lines[1..lines.len() - 1]
        .iter()
        .map(|l| l.to_string())
        .collect();
    assert_eq!(hash_lines(&event_lines), record.receipt.log_sha256);

    // Flip one byte of one event line: refused as a hash mismatch.
    let mut tampered = lines.clone();
    let victim = tampered[3].replace("\"position\":0", "\"position\":1");
    assert_ne!(victim, tampered[3], "the tamper must change the line");
    tampered[3] = &victim;
    let tampered_path = dir.path().join("tampered.jsonl");
    std::fs::write(&tampered_path, tampered.join("\n") + "\n").unwrap();
    assert!(matches!(
        RunRecord::read_jsonl(&tampered_path),
        Err(RecordError::HashMismatch { .. })
    ));

    // Drop the receipt: a prefix, refused as such.
    let prefix_path = dir.path().join("prefix.jsonl");
    std::fs::write(&prefix_path, lines[..lines.len() - 1].join("\n") + "\n").unwrap();
    assert!(matches!(
        RunRecord::read_jsonl(&prefix_path),
        Err(RecordError::MissingReceipt)
    ));

    // Drop the header: refused.
    let headless_path = dir.path().join("headless.jsonl");
    std::fs::write(&headless_path, lines[1..].join("\n") + "\n").unwrap();
    assert!(matches!(
        RunRecord::read_jsonl(&headless_path),
        Err(RecordError::MissingHeader)
    ));

    // Remove one event line but keep the receipt: the hash catches it
    // first; a receipt whose hash still matched would be caught by the
    // count.
    let short_path = dir.path().join("short.jsonl");
    let mut short = lines.clone();
    short.remove(2);
    std::fs::write(&short_path, short.join("\n") + "\n").unwrap();
    assert!(RunRecord::read_jsonl(&short_path).is_err());

    // Garbage is a parse error with its line number.
    let garbage_path = dir.path().join("garbage.jsonl");
    std::fs::write(&garbage_path, "{\"header\":1}\nnot json\n").unwrap();
    assert!(matches!(
        RunRecord::read_jsonl(&garbage_path),
        Err(RecordError::Parse { line: 1, .. })
    ));
    assert!(matches!(
        RunRecord::read_jsonl(&dir.path().join("absent.jsonl")),
        Err(RecordError::Io(_))
    ));
}

#[test]
fn s4_the_live_tap_never_blocks_and_its_loss_is_counted_outside_the_channel() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();

    // No consumer, a small tap: the run completes, the tap holds
    // exactly its capacity, the ledger holds exactly the rest (F3).
    let (sender, receiver) = sync_channel(SMALL_TAP);
    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_live_tap(sender);
    run(&f.plan, &ops, &backend, &mut recorder);
    let total = forecast_events_per_position() * G_TOKENS.len();
    assert_eq!(recorder.events().len(), total, "the record is unaffected");
    let ledger = recorder.drop_ledger().unwrap().clone();
    assert_eq!(ledger.dropped as usize, total - SMALL_TAP);
    assert_eq!(ledger.first_dropped_sequence, Some(SMALL_TAP as u64));
    assert_eq!(ledger.last_dropped_sequence, Some(total as u64 - 1));
    let delivered: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
    assert_eq!(delivered.len(), SMALL_TAP);
    assert_eq!(delivered.last().unwrap().sequence, SMALL_TAP as u64 - 1);
    assert_eq!(delivered[0], recorder.events()[0]);
    let record = recorder.finish();
    assert_eq!(record.receipt.live_dropped as usize, total - SMALL_TAP);

    // A tap that is never full: zero drops, every event delivered.
    let (sender, receiver) = sync_channel(LARGE_TAP);
    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_live_tap(sender);
    run(&f.plan, &ops, &backend, &mut recorder);
    assert_eq!(recorder.drop_ledger().unwrap().dropped, 0);
    let delivered: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
    assert_eq!(delivered, recorder.events());

    // A disconnected consumer is loss, not a stall.
    let (sender, receiver) = sync_channel(LARGE_TAP);
    drop(receiver);
    let mut recorder = RunRecorder::for_image(identity(), &ops, None).with_live_tap(sender);
    run(&f.plan, &ops, &backend, &mut recorder);
    let structural_only = (1 + 1 + G_LAYERS * 2 * 2 + 1) * G_TOKENS.len();
    assert_eq!(
        recorder.events().len(),
        structural_only,
        "no stats without an observer"
    );
    assert_eq!(
        recorder.drop_ledger().unwrap().dropped as usize,
        structural_only
    );
}

#[test]
fn s5_observation_parity_holds_through_the_recorder_on_both_backends() {
    fn check<B: PlanBackend>(backend: &B) {
        let f = fixture();
        let ops = PreparedOperands::load(&f.plan, &f.store, backend, ExecutionSlice::Full).unwrap();
        let plain = run(&f.plan, &ops, backend, &mut NoopObserver);
        let (sender, _receiver) = sync_channel(SMALL_TAP);
        let mut recorder =
            RunRecorder::for_image(identity(), &ops, Some(stats())).with_live_tap(sender);
        let recorded = run(&f.plan, &ops, backend, &mut recorder);
        assert_eq!(plain, recorded, "the recorder changed the logits");
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}

#[test]
fn an_incomplete_record_says_so_and_an_explicit_provenance_is_accepted() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let provenance = RunProvenance::new(ExecutionProvenance::of(&ops), None);
    let mut recorder = RunRecorder::arm(identity(), provenance.clone(), None);
    run(&f.plan, &ops, &backend, &mut recorder);
    let record = recorder.finish();
    assert!(!record.receipt.complete);
    assert_eq!(record.provenance_fingerprint, provenance.fingerprint());
    assert_eq!(
        record.provenance,
        serde_json::to_value(&provenance).unwrap()
    );
    assert_eq!(record.identity.tokens, G_TOKENS.to_vec());
    assert_eq!(record.identity.run_id, "run-test");
    assert!(record.identity.started_unix_ms > 0);
}

/// S8: the same properties on a real container, written to disk and
/// read back; report the record size.
///
/// ```sh
/// LARQL_V3_CONTAINER=~/chris-models/granite-4.2-3b.s6.vindex3 \
///   cargo test --release -p larql-inference --lib vindex3::tests::record::s8 -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a real VINDEX3 container in LARQL_V3_CONTAINER"]
fn s8_real_container_record_writes_and_replays() {
    let Ok(path) = std::env::var("LARQL_V3_CONTAINER") else {
        panic!("set LARQL_V3_CONTAINER to a container directory");
    };
    let root = std::path::Path::new(&path);
    let inspection = inspect_container(root, false).unwrap();
    let outcome = plan_component_ops(&inspection, root, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let tokens: Vec<u32> = (1..=8).collect();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    let plain: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| session.step(t).unwrap().logits.unwrap())
        .collect();
    let mut probe = RunRecorder::arm(
        RunIdentity::new("probe", "probe", COMPONENT, &tokens),
        RunProvenance::new(ExecutionProvenance::of(&ops), None),
        None,
    );
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    session.step_observed(tokens[0], &mut probe).unwrap();
    let width = probe
        .events()
        .iter()
        .find_map(|e| match e.event {
            EventKind::EnteringCarrier { hidden, .. } => Some(hidden),
            _ => None,
        })
        .unwrap();
    let stats = StatsObserver::new(FixedBasis::seeded(width, DIMS, SEED).unwrap(), None);
    let (sender, receiver) = sync_channel(SMALL_TAP);
    let mut recorder = RunRecorder::for_image(
        RunIdentity::new(
            "granite-s8",
            inspection.index.model.as_str(),
            COMPONENT,
            &tokens,
        ),
        &ops,
        Some(stats),
    )
    .with_live_tap(sender);
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    let recorded: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            session
                .step_observed(t, &mut recorder)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    assert_eq!(plain, recorded, "S5 on the real subject");
    recorder.complete();
    let expected = (1 + 1 + plan.layers.len() * 2 * 3 + 1) * tokens.len();
    assert_eq!(recorder.events().len(), expected, "F2");
    let dropped = recorder.drop_ledger().unwrap().dropped;
    let record = recorder.finish();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("granite.jsonl");
    record.write_jsonl(&out).unwrap();
    let read = RunRecord::read_jsonl(&out).unwrap();
    assert_eq!(read, record);
    let bytes = std::fs::metadata(&out).unwrap().len();
    let delivered = std::iter::from_fn(|| receiver.try_recv().ok()).count();
    println!(
        "S8 {path}: {} events ({} per token), record {} bytes, provenance {}, live tap delivered {delivered} dropped {dropped}",
        record.events.len(),
        record.events.len() / tokens.len(),
        bytes,
        record.provenance_fingerprint
    );
}

// ── V3-LENS-1 on the record ──────────────────────────────────────────

#[test]
fn lens_readouts_are_recorded_in_sequence_priced_on_the_receipt_and_replay_equal() {
    use crate::vindex3::{LensLayers, LensSites, LogitLens};
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let plain = run(&f.plan, &ops, &backend, &mut NoopObserver);
    let sites = LensSites {
        layers: LensLayers::All,
        attention: false,
        ffn: true,
    };
    let lens = LogitLens::new(&ops, &backend, sites, vec![3, 17], 2);
    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_lens(Box::new(lens));
    let recorded = run(&f.plan, &ops, &backend, &mut recorder);
    assert_eq!(plain, recorded, "the lens on the record changed the logits");
    let readouts: Vec<&RecordedEvent> = recorder
        .events()
        .iter()
        .filter(|e| matches!(e.event, EventKind::Readout { .. }))
        .collect();
    assert_eq!(
        readouts.len(),
        G_LAYERS * G_TOKENS.len(),
        "one readout per armed site"
    );
    // A readout follows its write's stats and precedes the structural write.
    let first = recorder
        .events()
        .iter()
        .position(|e| matches!(e.event, EventKind::Readout { .. }))
        .unwrap();
    assert!(matches!(
        recorder.events()[first - 1].event,
        EventKind::CarrierStats { .. }
    ));
    assert!(matches!(
        recorder.events()[first + 1].event,
        EventKind::CarrierWrite { .. }
    ));
    for r in &readouts {
        let EventKind::Readout {
            site,
            method,
            tokens,
            top,
            ..
        } = &r.event
        else {
            unreachable!()
        };
        assert!(matches!(site, crate::vindex3::Site::Ffn));
        assert_eq!(method, crate::vindex3::LENS_METHOD);
        assert_eq!(tokens.iter().map(|t| t.id).collect::<Vec<_>>(), vec![3, 17]);
        assert!(tokens.iter().all(|t| t.logprob <= 0.0 && t.rank >= 1));
        assert_eq!(top.len(), 2);
    }
    // The last layer's readout of the greedy argmax has rank 1: the lens
    // at the exit is the executor's own distribution.
    let last = readouts.last().unwrap();
    let EventKind::Readout { top, .. } = &last.event else {
        unreachable!()
    };
    let argmax = plain
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;
    assert_eq!(top[0].id, argmax);

    recorder.complete();
    let total = recorder.events().len();
    let record = recorder.finish();
    assert_eq!(
        record.receipt.head_passes as usize,
        G_LAYERS * G_TOKENS.len()
    );
    assert_eq!(record.receipt.lens_failure, None);
    assert_eq!(record.events.len(), total);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lens.jsonl");
    record.write_jsonl(&path).unwrap();
    assert_record_eq(&RunRecord::read_jsonl(&path).unwrap(), &record);
}

#[test]
fn a_failing_lens_is_named_on_the_receipt_and_the_record_is_still_complete() {
    use crate::vindex3::{LensSites, LogitLens};
    let f = fixture();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let lens = LogitLens::new(&ops, &backend, LensSites::every_ffn(), vec![u32::MAX], 0);
    let mut recorder = RunRecorder::for_image(identity(), &ops, None).with_lens(Box::new(lens));
    run(&f.plan, &ops, &backend, &mut recorder);
    recorder.complete();
    let record = recorder.finish();
    assert_eq!(
        record.receipt.head_passes, 1,
        "it stopped after the first refusal"
    );
    assert!(record
        .receipt
        .lens_failure
        .as_deref()
        .unwrap()
        .contains("outside the head's vocabulary"));
    assert!(!record
        .events
        .iter()
        .any(|e| matches!(e.event, EventKind::Readout { .. })));
    assert!(record.receipt.complete);
}

/// V3-INTERVENE-1, IP6: an intervened run's record carries the
/// declaration in its identity and receipt, the firing as an event at its
/// position ahead of the write it changed, counts firings against
/// declarations, reads back equal, and names a declared address the run
/// never reached — only once the run is complete.
#[test]
fn ip6_an_intervened_record_carries_its_declaration_firings_and_refusal() {
    use crate::vindex3::record::{Kind, Site};
    use larql_vindex::format::vindex3::opplan::exec::intervene::{
        Address, Intervention, InterventionPlan,
    };
    use larql_vindex::format::vindex3::opplan::exec::observe::SublayerSite;

    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let layer = G_LAYERS - 1;
    let fired_at = 2usize;
    let never = 40usize;
    let declared = InterventionPlan::none()
        .with(Intervention::zero(
            Address::new(layer, SublayerSite::Ffn, [fired_at, never]).unwrap(),
        ))
        .unwrap();

    let step_all = |plan: &InterventionPlan, recorder: &mut RunRecorder| -> usize {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&f.plan, &ops, &backend, &mut kv).unwrap();
        G_TOKENS
            .iter()
            .map(|&t| {
                session
                    .step_intervened(t, recorder, plan)
                    .unwrap()
                    .firings
                    .len()
            })
            .sum()
    };

    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_interventions(&declared);
    assert_eq!(
        step_all(&declared, &mut recorder),
        1,
        "one position reached, one firing"
    );
    recorder.complete();
    let written = recorder.finish();

    // Identity: the declaration's hash joins it, and a baseline differs.
    assert_eq!(
        written.identity.intervention_sha256,
        declared.declaration_sha256()
    );
    assert!(written.identity.intervention_sha256.is_some());
    let baseline = RunRecorder::for_image(identity(), &ops, None).finish();
    assert_eq!(baseline.identity.intervention_sha256, None);
    assert_ne!(
        baseline.identity.intervention_sha256, written.identity.intervention_sha256,
        "an intervened run is never read as a baseline"
    );

    // The event: at its position, before the write's stats and its
    // structural event, exactly once.
    let idx = written
        .events
        .iter()
        .position(|e| matches!(e.event, EventKind::Intervened { .. }))
        .expect("the firing is on the record");
    assert_eq!(written.events[idx].position, fired_at);
    assert_eq!(
        written.events[idx].event,
        EventKind::Intervened {
            layer,
            site: Site::Ffn,
            intervention: Kind::Zero,
        }
    );
    assert!(matches!(
        &written.events[idx + 1].event,
        EventKind::CarrierStats { layer: l, site: Site::Ffn, .. } if *l == layer
    ));
    assert!(matches!(
        &written.events[idx + 2].event,
        EventKind::CarrierWrite { layer: l, site: Site::Ffn, .. } if *l == layer
    ));
    assert_eq!(
        written
            .events
            .iter()
            .filter(|e| matches!(e.event, EventKind::Intervened { .. }))
            .count(),
        1
    );

    // The receipt: counted, and the unreached position named.
    assert_eq!(written.receipt.interventions_declared, 1);
    assert_eq!(written.receipt.interventions_applied, 1);
    let refusal = written
        .receipt
        .intervention_refusal
        .as_deref()
        .expect("position 40 was never reached");
    assert!(refusal.contains("position 40"), "{refusal}");
    assert!(refusal.contains("executed 5 position(s)"), "{refusal}");

    // Round trip: the record reads back equal, identity and receipt included.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("intervened.jsonl");
    written.write_jsonl(&path).unwrap();
    let read = RunRecord::read_jsonl(&path).unwrap();
    assert_record_eq(&read, &written);
    assert_eq!(read.identity, written.identity);
    assert_eq!(read.receipt, written.receipt);

    // An incomplete record is a prefix: it has not failed to reach anything.
    let mut prefix = RunRecorder::for_image(identity(), &ops, None).with_interventions(&declared);
    step_all(&declared, &mut prefix);
    let prefix = prefix.finish();
    assert!(!prefix.receipt.complete);
    assert_eq!(prefix.receipt.intervention_refusal, None);
    assert_eq!(prefix.receipt.interventions_applied, 1);

    // A declaration the run fully reaches carries no refusal.
    let reached = InterventionPlan::none()
        .with(Intervention::zero(
            Address::new(layer, SublayerSite::Ffn, [fired_at]).unwrap(),
        ))
        .unwrap();
    let mut recorder = RunRecorder::for_image(identity(), &ops, None).with_interventions(&reached);
    assert_eq!(step_all(&reached, &mut recorder), 1);
    recorder.complete();
    let complete = recorder.finish();
    assert_eq!(complete.receipt.intervention_refusal, None);
    assert_eq!(complete.receipt.interventions_declared, 1);
    assert_eq!(complete.receipt.interventions_applied, 1);

    // An unintervened record still says so on its receipt.
    assert_eq!(baseline.receipt.interventions_declared, 0);
    assert_eq!(baseline.receipt.interventions_applied, 0);
    assert_eq!(baseline.receipt.intervention_refusal, None);
}
