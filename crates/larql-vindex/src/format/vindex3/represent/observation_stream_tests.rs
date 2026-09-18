//! **The gate REAL-EVIDENCE-1 must pass before any statistic is read.**
//!
//! A bank rederived from the persisted stream must equal the bank built
//! directly from the same observations, exactly. If it does not, the
//! recording layer has changed the experiment and every later number is
//! a property of the recorder.

use super::super::bank::{BankBuilder, PositionObservation, RouteChange};
use super::{
    read_stream, rederive_bank, take_sequences, write_stream, StreamError, StreamIdentity,
};

/// A deterministic observation with realistic shape: a truncated top-N
/// window, both arms' logits at THE SAME ids, full-vocabulary logsumexp,
/// and routes that sometimes move.
fn observation(seq: u32, pos: u32) -> PositionObservation {
    let n = 32usize;
    let top_ids: Vec<u32> = (0..n as u32).map(|i| i * 7 + seq).collect();
    // Position-dependent divergence, as the real banks show: later
    // positions diverge more. Nothing here is a claim about the model;
    // it is a fixture with the right SHAPE so the projection is exercised.
    let drift = 1.0e-4 * (pos as f32 + 1.0) + 1.0e-5 * seq as f32;
    let baseline_logits: Vec<f32> = (0..n).map(|i| 8.0 - i as f32 * 0.25).collect();
    let candidate_logits: Vec<f32> = baseline_logits
        .iter()
        .enumerate()
        .map(|(i, v)| v + drift * (i as f32 % 3.0 - 1.0))
        .collect();
    let route_moved = (seq + pos).is_multiple_of(5);
    PositionObservation {
        sequence: seq,
        position: pos,
        top_ids: top_ids.clone(),
        baseline_logits,
        // Deliberately NOT a round number. A flat 9.5 survives rounding
        // to 3dp unchanged, so a lossy recorder passed the gate
        // undetected until this fixture carried real precision.
        baseline_logsumexp: 9.5 + seq as f32 * 0.000_123_4 + pos as f32 * 0.000_071_3,
        candidate_logits,
        candidate_logsumexp: 9.5 + drift + seq as f32 * 0.000_091_7,
        baseline_argmax: top_ids[0],
        candidate_argmax: if route_moved { top_ids[1] } else { top_ids[0] },
        baseline_top10: top_ids[..10].to_vec(),
        candidate_top10: top_ids[..10].to_vec(),
        baseline_routes: vec![vec![0, 1], vec![2, 3]],
        candidate_routes: vec![vec![0, 1], vec![if route_moved { 4 } else { 2 }, 3]],
        route_changes: vec![RouteChange {
            layer: 1,
            boundary_margin: if route_moved { 0.05 } else { 0.4 },
            weight_mass_moved: if route_moved { 0.11 } else { 0.0 },
        }],
        top10_change: None,
        top1_change: None,
    }
}

fn stream(sequences: u32, positions: u32) -> Vec<PositionObservation> {
    (0..sequences)
        .flat_map(|s| (0..positions).map(move |p| observation(s, p)))
        .collect()
}

fn identity(sequences: u32, positions: u32) -> StreamIdentity {
    StreamIdentity {
        source_identity: "kimi-linear-48b/s6".into(),
        candidate_identity: "kimi-map-l20-26q80".into(),
        protocol_identity: "teacher-forced-two-arm/v1".into(),
        code_identity: "test".into(),
        bank_identity: "fixture-bank".into(),
        draw_identity: "draw-0".into(),
        sequences,
        positions_per_sequence: positions,
    }
}

/// **THE GATE.** Persisted observations reproduce the authoritative
/// summary exactly.
#[test]
fn a_bank_rederived_from_the_stream_equals_the_bank_built_directly() {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(8, 32);

    // The authoritative summary, as the live run would produce it.
    let mut direct = BankBuilder::new();
    for o in &observations {
        direct.observe(o);
    }
    let direct = direct.finish();

    let manifest = write_stream(dir.path(), &identity(8, 32), &observations).unwrap();
    let (read_back, loaded) = read_stream(dir.path()).unwrap();

    assert_eq!(
        read_back, manifest,
        "the manifest must survive the round trip"
    );
    assert_eq!(
        loaded, observations,
        "the observations must survive verbatim; a lossy stream is a different experiment"
    );
    assert_eq!(
        rederive_bank(&loaded),
        direct,
        "GATE: the rederived bank must equal the directly built one exactly"
    );
    assert_eq!(manifest.observations, 256);
}

/// The stream is authoritative, so its bytes are bound to its manifest.
#[test]
fn an_altered_body_is_refused_rather_than_read() {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(2, 4);
    write_stream(dir.path(), &identity(2, 4), &observations).unwrap();

    let body = dir.path().join("observations.jsonl.gz");
    let mut bytes = std::fs::read(&body).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&body, &bytes).unwrap();

    match read_stream(dir.path()) {
        Err(StreamError::Tampered { .. }) => {}
        other => panic!("an altered stream must refuse, got {other:?}"),
    }
}

/// A manifest that disagrees with its body about how much evidence
/// exists is refused, not reconciled.
#[test]
fn a_miscounted_manifest_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(2, 4);
    write_stream(dir.path(), &identity(2, 4), &observations).unwrap();

    let path = dir.path().join("stream-manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    m["observations"] = serde_json::json!(999);
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();

    match read_stream(dir.path()) {
        Err(StreamError::CountMismatch {
            declared: 999,
            found: 8,
        }) => {}
        other => panic!("expected a count refusal, got {other:?}"),
    }
}

#[test]
fn an_unknown_format_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    write_stream(dir.path(), &identity(1, 2), &stream(1, 2)).unwrap();
    let path = dir.path().join("stream-manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    m["format"] = serde_json::json!("represent-observation-stream/v99");
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    assert!(matches!(
        read_stream(dir.path()),
        Err(StreamError::UnknownFormat(_))
    ));
}

/// **The ladder takes WHOLE sequences.** Positions inside a
/// teacher-forced sequence are not identically distributed — the real
/// banks show a U-shaped mean and a max growing ~23x across a sequence —
/// so a partial sequence is not a smaller sample of the same thing.
#[test]
fn the_ladder_selects_whole_sequences_and_its_banks_are_projections() {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(8, 32);
    write_stream(dir.path(), &identity(8, 32), &observations).unwrap();
    let (_, loaded) = read_stream(dir.path()).unwrap();

    for depth in [1u32, 2, 4, 8] {
        let chosen: Vec<u32> = (0..depth).collect();
        let subset = take_sequences(&loaded, &chosen);
        assert_eq!(
            subset.len() as u32,
            depth * 32,
            "depth {depth} must be whole sequences"
        );
        assert!(
            subset.iter().all(|o| o.sequence < depth),
            "no observation may come from outside the chosen sequences"
        );
        // The projection is what a depth result IS.
        assert_eq!(rederive_bank(&subset).positions, (depth * 32) as u64);
    }

    // Disjoint draws share no observation — the independent-block arm.
    let a = take_sequences(&loaded, &[0, 1, 2, 3]);
    let b = take_sequences(&loaded, &[4, 5, 6, 7]);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|x| !b.contains(x)));
    assert_ne!(
        rederive_bank(&a),
        rederive_bank(&b),
        "disjoint draws of a heterogeneous landscape should not agree exactly"
    );
}

/// Identity is the scientific contract: every field the freeze names is
/// carried, and a stream that loses one is not evidence about a run.
#[test]
fn the_manifest_carries_every_identity_the_freeze_requires() {
    let dir = tempfile::tempdir().unwrap();
    let m = write_stream(dir.path(), &identity(3, 4), &stream(3, 4)).unwrap();
    assert_eq!(m.source_identity, "kimi-linear-48b/s6");
    assert_eq!(m.candidate_identity, "kimi-map-l20-26q80");
    assert_eq!(m.protocol_identity, "teacher-forced-two-arm/v1");
    assert_eq!(m.bank_identity, "fixture-bank");
    assert_eq!(m.draw_identity, "draw-0");
    assert_eq!((m.sequences, m.positions_per_sequence), (3, 4));
    assert_eq!(m.observations, 12);
    assert!(!m.stream_sha256.is_empty());
    assert!(!m.code_identity.is_empty());
}
