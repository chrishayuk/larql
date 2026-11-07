//! The fresh-prefill logit dump: its length list, and on the encoded
//! fixture, that each prefill row agrees with the cached decode's row at
//! the same position — the two independent paths an equivalence harness
//! compares against an outside reference.

use super::super::prefill_dump::parse_lengths;
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
) -> ExecArgs {
    ExecArgs {
        container,
        component: "target".to_string(),
        tokens: TOKENS.to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Reference,
        representation_source: "auto".to_string(),
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
        gemv_witness: None,
        gemv_witness_max_operands: 1,
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
fn each_prefill_row_matches_the_cached_row_at_its_position() {
    let dir = fixture_dir(true);
    let container = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: container.clone(),
    }))
    .unwrap();
    let cached = dir.path().join("cached.f32");
    let prefill = dir.path().join("prefill.f32");
    run(Vindex3Command::Exec(exec_args(
        container.clone(),
        Some(cached.clone()),
        None,
        None,
    )))
    .unwrap();
    run(Vindex3Command::Exec(exec_args(
        container,
        None,
        Some(prefill.clone()),
        Some(PREFIX_LENGTHS),
    )))
    .unwrap();
    let cached = read_f32(&cached);
    let prefill = read_f32(&prefill);
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
