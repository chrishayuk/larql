//! Round-trip parity tests: quantise then dequantise, expect cosine
//! similarity ≥ 0.95 across all four formats and a range of head
//! dims. The threshold is below the upstream paper's 0.99 because
//! our reference uses a small static rotation table; lifting that
//! threshold is a tuning task tracked in
//! `rotorquant-attention-integration`.

use larql_rotorquant::{
    dequantize_k, dequantize_v_with_inverse_rotation, quantize_k, quantize_k_with_scratch,
    quantize_v, KvFormat, KvScratch, RotorQuantError,
};
use std::io::Write;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn run_upstream_triton_reference(
    format: &str,
    input: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Option<Vec<f32>> {
    let Ok(py) = std::env::var("LARQL_ROTORQUANT_REF_PY") else {
        eprintln!("skipping upstream Triton reference: set LARQL_ROTORQUANT_REF_PY");
        return None;
    };
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ref")
        .join("triton_reference.py");
    let mut child = std::process::Command::new(py)
        .arg(script)
        .arg(format)
        .arg(n_rows.to_string())
        .arg(head_dim.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn triton reference");
    {
        let stdin = child.stdin.as_mut().expect("reference stdin");
        for v in input {
            write!(stdin, "{v:.9} ").expect("write reference input");
        }
    }
    let output = child.wait_with_output().expect("wait triton reference");
    if output.status.code() == Some(77) {
        eprintln!(
            "skipping upstream Triton reference: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    assert!(
        output.status.success(),
        "upstream Triton reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let values = String::from_utf8(output.stdout)
        .expect("utf8 reference output")
        .split_whitespace()
        .map(|v| v.parse::<f32>().expect("parse reference float"))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), input.len());
    Some(values)
}

fn round_trip_k(format: KvFormat, n_rows: usize, head_dim: usize) -> f32 {
    let k = synth(n_rows * head_dim, 0x7);
    let qk = quantize_k(format, &k, n_rows, head_dim).expect("quantize_k");
    let recovered = dequantize_k(&qk).expect("dequantize_k");
    cosine(&k, &recovered)
}

fn round_trip_v(format: KvFormat, n_rows: usize, head_dim: usize) -> f32 {
    let v = synth(n_rows * head_dim, 0x9);
    let qv = quantize_v(format, &v, n_rows, head_dim).expect("quantize_v");
    let recovered = dequantize_v_with_inverse_rotation(&qv).expect("dequantize_v");
    cosine(&v, &recovered)
}

const TOL: f32 = 0.95;

#[test]
fn planar3_round_trip_k() {
    let cos = round_trip_k(KvFormat::Planar3, 64, 32);
    assert!(cos >= TOL, "Planar3 K cosine {cos}");
}

#[test]
fn planar4_round_trip_k() {
    let cos = round_trip_k(KvFormat::Planar4, 64, 32);
    assert!(cos >= TOL, "Planar4 K cosine {cos}");
}

#[test]
fn iso3_round_trip_k() {
    let cos = round_trip_k(KvFormat::Iso3, 64, 32);
    assert!(cos >= TOL, "Iso3 K cosine {cos}");
}

#[test]
fn iso4_round_trip_k() {
    let cos = round_trip_k(KvFormat::Iso4, 64, 32);
    assert!(cos >= TOL, "Iso4 K cosine {cos}");
}

#[test]
fn planar3_round_trip_v() {
    let cos = round_trip_v(KvFormat::Planar3, 64, 32);
    assert!(cos >= TOL, "Planar3 V cosine {cos}");
}

#[test]
fn iso3_round_trip_v() {
    let cos = round_trip_v(KvFormat::Iso3, 64, 32);
    assert!(cos >= TOL, "Iso3 V cosine {cos}");
}

#[test]
fn iso3_gemma4b_head_round_trip() {
    // Gemma 4B uses head_dim 320 (not divisible by 4? 320 / 4 = 80, OK).
    let cos = round_trip_k(KvFormat::Iso3, 32, 320);
    assert!(
        cos >= TOL,
        "Iso3 Gemma4B head_dim=320 round-trip cosine {cos}"
    );
}

#[test]
fn iso3_v_round_trip_recovers_original_not_rotated() {
    // The upstream bug (commit 6e5a4aa) was V dequant applying the
    // FORWARD rotation, leaving values in rotated-space rather than
    // original-space. Sanity-check we recover the original (cosine
    // ≥ 0.95 with the input), not some other rotated tensor.
    let v = synth(32 * 16, 0xA);
    let qv = quantize_v(KvFormat::Iso3, &v, 32, 16).unwrap();
    let recovered = dequantize_v_with_inverse_rotation(&qv).unwrap();
    let cos = cosine(&v, &recovered);
    assert!(
        cos >= TOL,
        "V dequant must recover the original input, cosine {cos}"
    );
}

#[test]
fn head_dim_divisibility_is_enforced() {
    // Iso3 needs head_dim % 4 == 0; 33 should fail.
    let v = vec![0.0_f32; 4 * 33];
    let err = quantize_k(KvFormat::Iso3, &v, 4, 33);
    assert!(err.is_err());
}

#[test]
fn kv_scratch_rejects_incompatible_shape() {
    let mut scratch = KvScratch::new(KvFormat::Iso3, 8, 32, 2).expect("scratch");
    assert_eq!(scratch.max_rows(), 16);

    let k = synth(8 * 32, 0xB);
    let qk = quantize_k_with_scratch(KvFormat::Iso3, &k, 8, 32, &mut scratch).expect("quantize");
    assert_eq!(qk.n_rows, 8);

    let err = quantize_k_with_scratch(KvFormat::Planar3, &k, 8, 32, &mut scratch).unwrap_err();
    assert!(matches!(err, RotorQuantError::ScratchFormatMismatch { .. }));

    let err = quantize_k_with_scratch(KvFormat::Iso3, &k, 8, 16, &mut scratch).unwrap_err();
    assert!(matches!(
        err,
        RotorQuantError::ScratchHeadDimMismatch { .. }
    ));

    let too_many_rows = synth(17 * 32, 0xC);
    let err =
        quantize_k_with_scratch(KvFormat::Iso3, &too_many_rows, 17, 32, &mut scratch).unwrap_err();
    assert!(matches!(
        err,
        RotorQuantError::ScratchCapacityExceeded { .. }
    ));
}

#[test]
fn upstream_triton_reference_planar3_round_trip() {
    let input = synth(8 * 32, 0xD);
    let Some(reference) = run_upstream_triton_reference("planar3", &input, 8, 32) else {
        return;
    };
    let qv = quantize_v(KvFormat::Planar3, &input, 8, 32).expect("quantize_v");
    let rust = dequantize_v_with_inverse_rotation(&qv).expect("dequantize_v");
    assert!(cosine(&input, &reference) >= 0.90);
    assert!(cosine(&input, &rust) >= TOL);
    assert!(cosine(&reference, &rust) >= 0.80);
}

#[test]
fn upstream_triton_reference_iso3_round_trip() {
    let input = synth(8 * 32, 0xE);
    let Some(reference) = run_upstream_triton_reference("iso3", &input, 8, 32) else {
        return;
    };
    let qk = quantize_k(KvFormat::Iso3, &input, 8, 32).expect("quantize_k");
    let rust = dequantize_k(&qk).expect("dequantize_k");
    assert!(cosine(&input, &reference) >= 0.90);
    assert!(cosine(&input, &rust) >= TOL);
    assert!(cosine(&reference, &rust) >= 0.80);
}

#[test]
fn upstream_provenance_records_commit_and_vendored_files() {
    let upstream = include_str!("../UPSTREAM.md");
    assert!(upstream.contains("https://github.com/johndpope/llama-cpp-turboquant.git"));
    assert!(upstream.contains("08e025c06ab521e4fa9e5c08b80af57614543e53"));
    assert!(upstream.contains("cpy-planar-iso.cu"));

    let vendor_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/upstream");
    for name in [
        "cpy-planar-iso.cu",
        "cpy-planar-iso.cuh",
        "planar-iso-constants.cuh",
        "set-rows-planar-iso.cuh",
    ] {
        assert!(
            vendor_dir.join(name).is_file(),
            "missing vendored RotorQuant source {name}"
        );
    }

    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("build.rs")
        .is_file());
    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("cuda/ARCHS.txt")
        .is_file());
    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("cuda/larql_rotorquant_kernels.cu")
        .is_file());
    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ref/upstream/triton_planarquant.py")
        .is_file());
    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ref/upstream/triton_isoquant.py")
        .is_file());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ffi_block_layout_matches_upstream() {
    assert_eq!(larql_rotorquant::ffi::BLOCK_ELEMENTS, 128);
    assert_eq!(larql_rotorquant::ffi::block_bytes(KvFormat::Planar3), 50);
    assert_eq!(larql_rotorquant::ffi::block_bytes(KvFormat::Iso3), 50);
    assert_eq!(larql_rotorquant::ffi::block_bytes(KvFormat::Planar4), 68);
    assert_eq!(larql_rotorquant::ffi::block_bytes(KvFormat::Iso4), 68);
}
