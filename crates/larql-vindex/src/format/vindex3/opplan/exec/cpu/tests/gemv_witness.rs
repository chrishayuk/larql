//! The witness observes the production call site and changes nothing.

use std::sync::Mutex;

use super::super::executor::shared;
use super::super::gemv_witness::{self as witness, Binding, MANIFEST, SCHEMA};
use super::super::kernels::FusedKQuant;
use super::super::projector::WeightRows;
use crate::format::vindex3::opplan::exec::backend::KQuantActivation;
use crate::format::vindex3::represent::kquant::{Q6_K, Q8_0};

const ROWS: usize = 3;
const COLS: usize = 64;

/// A capture is process-wide, so these tests can't overlap.
static SERIAL: Mutex<()> = Mutex::new(());

/// Q8_0 blocks with a different scale and codes in each block.
fn blocks() -> Vec<u8> {
    let mut bytes = Vec::new();
    for b in 0..ROWS * COLS / 32 {
        bytes.extend_from_slice(&(0x3c00u16 + b as u16).to_le_bytes());
        bytes.extend((0..32).map(|j| ((b * 7 + j * 3) % 200) as u8));
    }
    bytes
}

fn x() -> Vec<f32> {
    (0..COLS).map(|i| (i as f32 - 20.0) / 7.0).collect()
}

fn q8_f32(bytes: &[u8]) -> WeightRows<'_> {
    WeightRows::KQuant {
        blocks: bytes,
        codec: Q8_0,
        activation: KQuantActivation::F32,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn it_records_exactly_what_the_kernel_read_and_returned() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    witness::start(1);
    witness::note_operand("layers.0.q_proj", "Q8_0", &bytes);
    let y = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();

    assert_eq!(capture.records.len(), 1);
    let r = &capture.records[0];
    assert_eq!(r.tensor.as_deref(), Some("layers.0.q_proj"));
    assert_eq!(r.blocks, bytes);
    assert_eq!(bits(&r.x), bits(&x));
    assert_eq!(
        bits(&r.y),
        bits(&y),
        "the recorded output is the returned output"
    );
    assert_eq!(r.rows, ROWS);
}

#[test]
fn the_output_bits_are_the_same_with_the_capture_on_or_off() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    let off = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    witness::start(1);
    witness::note_operand("t", "Q8_0", &bytes);
    let on = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    witness::finish();
    assert_eq!(bits(&off), bits(&on));
}

#[test]
fn nothing_is_recorded_unless_a_capture_is_open() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    let _ = witness::finish();
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    witness::note_operand("t", "Q8_0", &bytes);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 0);
    assert_eq!(capture.operands_noted, 0);
    assert!(capture.records.is_empty());
}

#[test]
fn only_q8_0_against_an_f32_activation_is_covered() {
    let _g = SERIAL.lock().unwrap();
    let x = x();
    let (q8, q6) = (blocks(), vec![0u8; 8]);
    witness::start(10);
    witness::record(
        WeightRows::KQuant {
            blocks: &q8,
            codec: Q8_0,
            activation: KQuantActivation::Q8k,
        },
        &x,
        &[0.0],
    );
    witness::record(
        WeightRows::KQuant {
            blocks: &q6,
            codec: Q6_K,
            activation: KQuantActivation::F32,
        },
        &x,
        &[0.0],
    );
    witness::note_operand("q6", "Q6_K", &q6);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 0);
    assert_eq!(capture.operands_noted, 0);
}

#[test]
fn the_limit_keeps_the_first_call_on_each_operand_and_counts_them_all() {
    let _g = SERIAL.lock().unwrap();
    let (a, b, c) = (blocks(), blocks(), blocks());
    let (x1, x2) = (x(), vec![1.0f32; COLS]);
    witness::start(2);
    for (name, bytes) in [("a", &a), ("b", &b), ("c", &c)] {
        witness::note_operand(name, "Q8_0", bytes);
    }
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&a), &x1, ROWS);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&a), &x2, ROWS);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&b), &x1, ROWS);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&c), &x1, ROWS);
    let capture = witness::finish();

    assert_eq!(capture.calls_observed, 4);
    let names: Vec<_> = capture
        .records
        .iter()
        .map(|r| r.tensor.as_deref().unwrap())
        .collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(
        bits(&capture.records[0].x),
        bits(&x1),
        "the first call, not a later one"
    );
    assert_eq!(capture.records[1].call, 2);
}

#[test]
fn batched_positions_are_recorded_with_their_own_outputs() {
    let _g = SERIAL.lock().unwrap();
    let bytes = blocks();
    let (x1, x2) = (x(), vec![0.5f32; COLS]);
    witness::start(1);
    witness::note_operand("t", "Q8_0", &bytes);
    let outs = shared()
        .unwrap()
        .project_many(&FusedKQuant, q8_f32(&bytes), &[&x1, &x2], ROWS);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 2);
    assert_eq!(bits(&capture.records[0].y), bits(&outs[0]));
}

fn binding<'a>(container: &'a std::path::Path) -> Binding<'a> {
    Binding {
        build_commit: Some("abc"),
        binary_sha256: "def",
        container,
        representation: "Q8_0",
        execution_arm: "Direct",
    }
}

#[test]
fn a_written_witness_holds_the_raw_bits_and_their_digests() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    witness::start(1);
    witness::note_operand("t", "Q8_0", &bytes);
    let y = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("w");
    witness::write(&capture, &binding(tmp.path()), &dir).unwrap();

    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
    assert_eq!(m["schema"], SCHEMA);
    assert_eq!(m["build_commit"], "abc");
    assert_eq!(m["execution_arm"], "Direct");
    let r = &m["records"][0];
    assert_eq!(r["tensor"], "t");
    assert_eq!(r["rows"], ROWS);
    assert_eq!(r["columns"], COLS);
    assert_eq!(
        std::fs::read(dir.join(r["blocks"]["file"].as_str().unwrap())).unwrap(),
        bytes
    );
    let stored = std::fs::read(dir.join(r["y"]["file"].as_str().unwrap())).unwrap();
    let want: Vec<u8> = y.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    assert_eq!(stored, want);

    // Never overwritten.
    assert!(witness::write(&capture, &binding(tmp.path()), &dir).is_err());
}

#[test]
fn an_empty_or_unnamed_capture_is_refused_not_written() {
    let _g = SERIAL.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("empty");
    let err = witness::write(&witness::Capture::default(), &binding(tmp.path()), &dir).unwrap_err();
    assert!(err.contains("nothing to witness"), "{err}");
    assert!(!dir.exists());

    let (bytes, x) = (blocks(), x());
    witness::start(1);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();
    let dir = tmp.path().join("unnamed");
    let err = witness::write(&capture, &binding(tmp.path()), &dir).unwrap_err();
    assert!(err.contains("no noted tensor"), "{err}");
    assert!(!dir.exists());
}
