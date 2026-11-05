//! The witness observes the production call site and changes nothing.

use std::sync::Mutex;

use super::super::executor::shared;
use super::super::gemv_witness::{self as witness, Binding, MANIFEST, SCHEMA};
use super::super::kernels::FusedKQuant;
use super::super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::backend::KQuantActivation;
use crate::format::vindex3::represent::kquant::Q8_0;

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
    witness::note_operand("target.decoder_stack", "layers.0.q_proj", "Q8_0", &bytes);
    let y = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();

    assert_eq!(capture.records.len(), 1);
    let r = &capture.records[0];
    assert_eq!(r.operand.as_ref().unwrap().object, "target.decoder_stack");
    assert_eq!(r.operand.as_ref().unwrap().tensor, "layers.0.q_proj");
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
    witness::note_operand("o", "t", "Q8_0", &bytes);
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
    witness::note_operand("o", "t", "Q8_0", &bytes);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 0);
    assert_eq!(capture.operands_noted, 0);
    assert!(capture.records.is_empty());
}

#[test]
fn only_q8_0_is_covered() {
    let _g = SERIAL.lock().unwrap();
    let (x, q6) = (x(), vec![0u8; 8]);
    witness::start(10);
    witness::record(&q6, "Q6_K", &x, &[0.0]);
    witness::note_operand("o", "q6", "Q6_K", &q6);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 0);
    assert_eq!(capture.operands_noted, 0);
}

/// Another kernel handed the same weights.
struct NotTheKernel;

impl DenseProjector for NotTheKernel {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::LibraryOwned
    }

    fn project_rows(&self, _: WeightRows<'_>, _: &[f32], out: &mut [f32]) {
        out.fill(7.0);
    }
}

#[test]
fn another_kernels_output_is_never_recorded_under_this_kernels_name() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    witness::start(1);
    witness::note_operand("o", "t", "Q8_0", &bytes);
    let y = shared()
        .unwrap()
        .project(&NotTheKernel, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();
    assert_eq!(y, [7.0; ROWS]);
    assert_eq!(capture.calls_observed, 0);
    assert!(capture.records.is_empty());
}

#[test]
fn other_bytes_at_a_noted_address_get_no_name_and_are_not_skipped() {
    let _g = SERIAL.lock().unwrap();
    let mut bytes = blocks();
    let x = x();
    witness::start(10);
    witness::note_operand("o", "a", "Q8_0", &bytes);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    // The same allocation, now holding something else, as a reused address would.
    bytes[2] = bytes[2].wrapping_add(1);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();

    assert_eq!(
        capture.records.len(),
        2,
        "the second operand wasn't skipped"
    );
    assert_eq!(capture.records[0].operand.as_ref().unwrap().tensor, "a");
    assert!(
        capture.records[1].operand.is_none(),
        "it wasn't given a's name"
    );

    let tmp = tempfile::tempdir().unwrap();
    let err = witness::write(&capture, &binding(tmp.path()), &tmp.path().join("w")).unwrap_err();
    assert!(err.contains("no noted tensor"), "{err}");
}

#[test]
fn the_batched_output_bits_are_the_same_with_the_capture_on_or_off() {
    let _g = SERIAL.lock().unwrap();
    let bytes = blocks();
    let (x1, x2) = (x(), vec![0.5f32; COLS]);
    let exec = shared().unwrap();
    let off = exec.project_many(&FusedKQuant, q8_f32(&bytes), &[&x1, &x2], ROWS);
    witness::start(1);
    witness::note_operand("o", "t", "Q8_0", &bytes);
    let on = exec.project_many(&FusedKQuant, q8_f32(&bytes), &[&x1, &x2], ROWS);
    witness::finish();
    for (a, b) in off.iter().zip(&on) {
        assert_eq!(bits(a), bits(b));
    }
}

#[test]
fn a_target_that_exists_or_has_no_parent_is_refused_up_front() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(witness::check_target(&tmp.path().join("new")).is_ok());
    assert!(witness::check_target(tmp.path())
        .unwrap_err()
        .contains("exists"));
    assert!(witness::check_target(&tmp.path().join("a/b"))
        .unwrap_err()
        .contains("no parent"));
}

#[test]
fn the_limit_keeps_the_first_call_on_each_operand_and_counts_them_all() {
    let _g = SERIAL.lock().unwrap();
    let (a, b, c) = (blocks(), blocks(), blocks());
    let (x1, x2) = (x(), vec![1.0f32; COLS]);
    witness::start(2);
    for (name, bytes) in [("a", &a), ("b", &b), ("c", &c)] {
        witness::note_operand("o", name, "Q8_0", bytes);
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
        .map(|r| r.operand.as_ref().unwrap().tensor.as_str())
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
    witness::note_operand("o", "t", "Q8_0", &bytes);
    let outs = shared()
        .unwrap()
        .project_many(&FusedKQuant, q8_f32(&bytes), &[&x1, &x2], ROWS);
    let capture = witness::finish();
    assert_eq!(capture.calls_observed, 2);
    assert_eq!(bits(&capture.records[0].y), bits(&outs[0]));
}

fn binding(container: &std::path::Path) -> Binding<'_> {
    static ENVIRONMENT: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    static PLANS: std::sync::OnceLock<Vec<(String, u64)>> = std::sync::OnceLock::new();
    Binding {
        build_commit: Some("abc"),
        binary_sha256: "def",
        container,
        container_index_sha256: "idx",
        component: "target",
        backend: "ProductionQ8",
        representation: "Q8_0",
        representation_source: "stored",
        execution_arm: "Direct",
        tokens: &[1, 2],
        draft_depth: Some(3),
        logit_dump: std::path::Path::new("logits.bin"),
        environment: ENVIRONMENT
            .get_or_init(|| vec![("LARQL_CPU_ARITHMETIC".into(), "q8xq8".into())]),
        platform: "os-arch",
        projection_plans: PLANS.get_or_init(|| vec![("FusedKQuant".into(), 4)]),
    }
}

#[test]
fn a_written_witness_holds_the_raw_bits_and_their_digests() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    witness::start(1);
    witness::note_operand("o", "t", "Q8_0", &bytes);
    let y = shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    witness::note_logit_dump("aaa".into(), 12);
    let capture = witness::finish();

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("w");
    witness::write(&capture, &binding(tmp.path()), &dir).unwrap();

    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
    assert_eq!(m["schema"], SCHEMA);
    assert_eq!(m["build_commit"], "abc");
    assert_eq!(m["execution_arm"], "Direct");
    assert_eq!(m["component"], "target");
    assert_eq!(m["tokens"], serde_json::json!([1, 2]));
    assert_eq!(m["draft_depth"], 3);
    assert_eq!(m["logit_dump"]["sha256"], "aaa");
    assert_eq!(m["logit_dump"]["bytes"], 12);
    assert_eq!(m["container_index_sha256"], "idx");
    assert_eq!(m["environment"]["LARQL_CPU_ARITHMETIC"], "q8xq8");
    assert_eq!(m["projection_plans"]["FusedKQuant"], 4);
    assert_eq!(m["max_operands"], 1);
    assert!(!dir.join(format!("{MANIFEST}.partial")).exists());
    let r = &m["records"][0];
    assert_eq!(r["object"], "o");
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
fn a_capture_with_no_logit_dump_digest_is_refused() {
    let _g = SERIAL.lock().unwrap();
    let (bytes, x) = (blocks(), x());
    witness::start(1);
    witness::note_operand("o", "t", "Q8_0", &bytes);
    shared()
        .unwrap()
        .project(&FusedKQuant, q8_f32(&bytes), &x, ROWS);
    let capture = witness::finish();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("w");
    let err = witness::write(&capture, &binding(tmp.path()), &dir).unwrap_err();
    assert!(err.contains("no logit dump"), "{err}");
    assert!(!dir.exists());
}

#[cfg(unix)]
#[test]
fn a_dangling_symlink_counts_as_existing() {
    let tmp = tempfile::tempdir().unwrap();
    let link = tmp.path().join("l");
    std::os::unix::fs::symlink("/no/such/place", &link).unwrap();
    assert!(witness::check_target(&link).is_err());
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
