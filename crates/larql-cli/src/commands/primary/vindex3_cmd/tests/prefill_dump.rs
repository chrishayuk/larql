//! The fresh-prefill logit dump: its length list, and on the encoded
//! fixture, that each prefill row agrees with the cached decode's row at
//! the same position — the two independent paths an equivalence harness
//! compares against an outside reference.

use super::super::prefill_dump::{parse_lengths, same_vocabulary, PrefillMethod};
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const TOKENS: &str = "1,2,3";
const SEQUENCE: usize = 3;
const PREFIX_LENGTHS: &str = "1,3";
/// Reference-backend f32 agreement between a cached step and a fresh
/// prefill of the same prefix (summation order differs).
const PATH_TOLERANCE: f32 = 1e-4;

fn exec_args(
    container: std::path::PathBuf,
    logit_dump: Option<std::path::PathBuf>,
    prefill: Option<std::path::PathBuf>,
    lengths: Option<&str>,
    method: PrefillMethod,
) -> ExecArgs {
    ExecArgs {
        container,
        component: "target".to_string(),
        tokens: TOKENS.to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Reference,
        representation_source: Some("auto".to_string()),
        generate: None,
        speculate: None,
        stop_at_eos: false,
        residency_curve: false,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: None,
        sealed_latency: None,
        residency_campaign: false,
        verification_manifest: None,
        admission_threshold_bytes: None,
        admission_basis: None,
        expert_access: "demand".to_string(),
        logit_dump,
        prefill_logit_dump: prefill,
        prefill_lengths: lengths.map(str::to_string),
        prefill_method: method,
        gemv_witness: None,
        gemv_witness_max_operands: 1,
        capture: Default::default(),
        rope_angles: Default::default(),
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn omitted_lengths_mean_the_whole_sequence() {
    assert_eq!(parse_lengths(None, SEQUENCE).unwrap(), vec![SEQUENCE]);
    assert_eq!(parse_lengths(Some(" 1, 3"), SEQUENCE).unwrap(), vec![1, 3]);
}

#[test]
fn lengths_outside_the_sequence_are_refused_not_clamped() {
    for bad in ["0", "4", "1,x", ""] {
        assert!(parse_lengths(Some(bad), SEQUENCE).is_err(), "{bad:?}");
    }
}

#[test]
fn a_dump_refuses_rows_of_different_widths() {
    let mut vocab = None;
    assert!(same_vocabulary(&mut vocab, 4, 1).is_ok());
    assert!(same_vocabulary(&mut vocab, 4, 2).is_ok());
    assert!(same_vocabulary(&mut vocab, 5, 3).is_err());
}

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let container = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: container.clone(),
    }))
    .unwrap();
    (dir, container)
}

fn prefill_rows(
    dir: &std::path::Path,
    container: &std::path::Path,
    method: PrefillMethod,
) -> Vec<f32> {
    let out = dir.join(format!("prefill-{method:?}.f32"));
    run(Vindex3Command::Exec(exec_args(
        container.to_path_buf(),
        None,
        Some(out.clone()),
        Some(PREFIX_LENGTHS),
        method,
    )))
    .unwrap();
    read_f32(&out)
}

#[test]
fn each_prefill_row_matches_the_cached_row_at_its_position() {
    let (dir, container) = encoded_fixture();
    let cached = dir.path().join("cached.f32");
    run(Vindex3Command::Exec(exec_args(
        container.clone(),
        Some(cached.clone()),
        None,
        None,
        PrefillMethod::Batched,
    )))
    .unwrap();
    let cached = read_f32(&cached);
    let prefill = prefill_rows(dir.path(), &container, PrefillMethod::Batched);
    let vocab = cached.len() / SEQUENCE;
    let lengths = parse_lengths(Some(PREFIX_LENGTHS), SEQUENCE).unwrap();
    assert_eq!(prefill.len(), lengths.len() * vocab);
    for (row, length) in lengths.iter().enumerate() {
        let want = &cached[(length - 1) * vocab..length * vocab];
        let got = &prefill[row * vocab..(row + 1) * vocab];
        let worst = want
            .iter()
            .zip(got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= PATH_TOLERANCE, "prefix {length}: |Δ| {worst}");
    }
}

/// The batched method against the per-prefix control on the reference
/// backend: BIT-EXACT. The reference interpreter's causal attention reads
/// keys `0..=p` only, so position `p` of one long prefill and the last
/// position of the `p+1` prefix run the same arithmetic in the same order.
#[test]
fn batched_rows_equal_the_per_prefix_control_bit_for_bit() {
    let (dir, container) = encoded_fixture();
    let batched = prefill_rows(dir.path(), &container, PrefillMethod::Batched);
    let control = prefill_rows(dir.path(), &container, PrefillMethod::PerPrefix);
    assert_eq!(batched.len(), control.len());
    assert!(
        batched
            .iter()
            .zip(&control)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "batched and per-prefix rows differ"
    );
}
