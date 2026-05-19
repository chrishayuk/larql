//! Walker accuracy fixture — pins walker behaviour against regressions.
//!
//! Walker output is a function of `(model weights, config, host
//! floating-point order)`. The first two are fully deterministic — the
//! mock fixture in `walker::test_fixture` is byte-identical across runs
//! and platforms. The third is not: the matmul that feeds the top-k
//! selection runs through SIMD code paths that differ between x86 and
//! ARM, which is enough to flip tie-breaking on edges with sub-ULP
//! score differences. macOS-aarch64 and linux-x86_64 therefore produce
//! consistent-but-different edge lists.
//!
//! Two-tier strategy:
//!   * Within a single platform / run, walker output must be exactly
//!     reproducible — we walk the fixture twice and demand byte-equal
//!     canonical output.
//!   * Across platforms, we pin **structural invariants** (edge counts,
//!     normalisation sanity, metadata presence) instead of a SHA. Any
//!     real regression (count drift, broken normalisation, missing
//!     fields, empty tokens) flips a structural assertion; pure
//!     SIMD-driven reordering doesn't.
//!
//! The vector-extractor file output is still hashed byte-for-byte — it
//! doesn't go through the top-k path and is genuinely cross-platform.

use std::path::Path;

#[cfg(not(windows))]
use sha2::{Digest, Sha256};

use larql_core::Graph;
use larql_vindex::walker::{
    attention_walker::AttentionWalker,
    test_fixture::create_mock_model,
    weight_walker::{SilentWalkCallbacks, WalkConfig, WeightWalker},
};

// Vector-extractor imports are only used by the byte-identical golden
// test, which is gated off on Windows (BLAS f32 round-tripping drifts
// the JSONL hash). Keep the imports under the same gate so a Windows
// build doesn't flag them as unused.
#[cfg(not(windows))]
use larql_vindex::walker::vector_extractor::{
    ExtractConfig, SilentExtractCallbacks, VectorExtractor,
};

fn fixture(slug: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_accuracy_{slug}"));
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);
    dir
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// Canonical edge form: tab-separated fields, sorted, one edge per line.
#[cfg(not(windows))]
fn canonicalise_edges(graph: &Graph, layer_field: &str, feature_field: &str) -> Vec<u8> {
    let mut lines: Vec<String> = graph
        .edges()
        .iter()
        .map(|e| {
            let m = e.metadata.as_ref().unwrap();
            let layer = m.get(layer_field).and_then(|v| v.as_u64()).unwrap_or(0);
            let feat = m.get(feature_field).and_then(|v| v.as_u64()).unwrap_or(0);
            format!(
                "{}\t{}\t{}\t{}\t{}\t{:.6}",
                e.subject, e.relation, e.object, layer, feat, e.confidence
            )
        })
        .collect();
    lines.sort();
    let mut out = lines.join("\n").into_bytes();
    out.push(b'\n');
    out
}

// ── Goldens ──────────────────────────────────────────────────────────────
//
// Only the vector-extractor stays byte-identical: it doesn't run a
// top-k selection on matmul output, so it's not exposed to SIMD-level
// reordering. Regenerate after an intentional change with:
//   LARQL_PRINT_GOLDEN=1 cargo test -p larql-vindex --test \
//     test_walker_accuracy -- --nocapture
// then paste the printed hex string below.
//
// Regenerated 2026-05-10: the canonicalisation strips the `_header`
// record so the wall-clock `extraction_date` field doesn't make the
// golden drift every day.
#[cfg(not(windows))]
const GOLDEN_VECTOR_EXTRACTOR_FFN_DOWN_LAYER0: &str =
    "8b5e221b150147ed40b0cfa67fdfc264e0628ab6cd6c59c2f9419e9350589b83";

#[cfg(not(windows))]
fn check_or_print(label: &str, actual: &str, golden: &str) {
    if std::env::var("LARQL_PRINT_GOLDEN").is_ok() {
        eprintln!("{label} = {actual:?}");
        return;
    }
    assert_eq!(
        actual, golden,
        "{label}: walker output drifted — review the change and update the golden if intentional"
    );
}

/// Run the weight walker on layer 0 of a freshly-built fixture, twice,
/// and pin structural invariants on the result.
///
/// We deliberately don't hash edge bytes (see module doc) — the
/// matmul → top-k pipeline gives different tie-breaking on x86 vs ARM.
/// What we *do* require: every walk on the same platform yields the
/// same canonical bytes, the edge count is stable, normalisation
/// produces a max-confidence of 1.0, every confidence sits in [0, 1],
/// every (subject, relation, object) is non-empty, and every edge
/// carries the documented metadata fields.
#[test]
fn weight_walker_layer0_invariants() {
    let cfg = WalkConfig {
        top_k: 3,
        min_score: 0.0,
    };

    let dir_a = fixture("ww_a");
    let walker_a = WeightWalker::load(dir_a.to_str().unwrap()).unwrap();
    let mut g_a = Graph::new();
    walker_a
        .walk_layer(0, &cfg, &mut g_a, &mut SilentWalkCallbacks)
        .unwrap();

    let dir_b = fixture("ww_b");
    let walker_b = WeightWalker::load(dir_b.to_str().unwrap()).unwrap();
    let mut g_b = Graph::new();
    walker_b
        .walk_layer(0, &cfg, &mut g_b, &mut SilentWalkCallbacks)
        .unwrap();

    // BLAS on Windows runners has non-deterministic reduction order
    // between successive matmul calls on the same input (parallel
    // accumulation in OpenBLAS), which trips f32-precision tie-breaking
    // in the top-k path. Linux/macOS BLAS implementations don't show
    // this drift, so we keep the byte-equality check there.
    #[cfg(not(windows))]
    {
        let bytes_a = canonicalise_edges(&g_a, "layer", "feature");
        let bytes_b = canonicalise_edges(&g_b, "layer", "feature");
        assert_eq!(
            sha256_hex(&bytes_a),
            sha256_hex(&bytes_b),
            "weight_walker_layer0: not deterministic within a single run"
        );
    }

    assert_structural_invariants(&g_a, "feature", 0);
    assert_structural_invariants(&g_b, "feature", 0);

    cleanup(&dir_a);
    cleanup(&dir_b);
}

#[test]
fn attention_walker_layer0_invariants() {
    let cfg = WalkConfig {
        top_k: 2,
        min_score: 0.0,
    };

    let dir_a = fixture("aw_a");
    let walker_a = AttentionWalker::load(dir_a.to_str().unwrap()).unwrap();
    let mut g_a = Graph::new();
    walker_a
        .walk_layer(0, &cfg, &mut g_a, &mut SilentWalkCallbacks)
        .unwrap();

    let dir_b = fixture("aw_b");
    let walker_b = AttentionWalker::load(dir_b.to_str().unwrap()).unwrap();
    let mut g_b = Graph::new();
    walker_b
        .walk_layer(0, &cfg, &mut g_b, &mut SilentWalkCallbacks)
        .unwrap();

    // See `weight_walker_layer0_invariants` — BLAS on Windows has
    // non-deterministic reduction order across matmul calls; keep the
    // byte-equality check on the platforms where it holds.
    #[cfg(not(windows))]
    {
        let bytes_a = canonicalise_edges(&g_a, "layer", "head");
        let bytes_b = canonicalise_edges(&g_b, "layer", "head");
        assert_eq!(
            sha256_hex(&bytes_a),
            sha256_hex(&bytes_b),
            "attention_walker_layer0: not deterministic within a single run"
        );
    }

    assert_structural_invariants(&g_a, "head", 0);
    assert_structural_invariants(&g_b, "head", 0);

    cleanup(&dir_a);
    cleanup(&dir_b);
}

fn assert_structural_invariants(graph: &Graph, second_field: &str, expected_layer: u64) {
    let edges = graph.edges();
    assert!(!edges.is_empty(), "expected at least one edge");

    let mut max_conf = 0.0f64;
    for e in edges {
        assert!(!e.subject.is_empty(), "empty subject");
        assert!(!e.relation.is_empty(), "empty relation");
        assert!(!e.object.is_empty(), "empty object");
        assert!(
            (0.0..=1.0).contains(&e.confidence),
            "confidence out of range: {}",
            e.confidence
        );
        if e.confidence > max_conf {
            max_conf = e.confidence;
        }
        let m = e.metadata.as_ref().expect("edge metadata missing");
        let layer = m
            .get("layer")
            .and_then(|v| v.as_u64())
            .expect("layer metadata missing");
        assert_eq!(layer, expected_layer, "wrong layer in metadata");
        assert!(
            m.get(second_field).and_then(|v| v.as_u64()).is_some(),
            "missing `{second_field}` metadata",
        );
        assert!(
            m.get("c_in").and_then(|v| v.as_f64()).is_some(),
            "missing c_in metadata"
        );
        assert!(
            m.get("c_out").and_then(|v| v.as_f64()).is_some(),
            "missing c_out metadata"
        );
    }

    // Per-layer normalisation must hit 1.0 exactly on the top edge.
    assert!(
        (max_conf - 1.0).abs() < 1e-6,
        "max confidence is {max_conf}, expected ~1.0 (per-layer normalisation broken)"
    );
}

// The golden is byte-keyed on the BLAS implementation's f32 output:
// canonicalised JSONL → sha256. Linux (OpenBLAS) and macOS (Accelerate)
// happen to produce matching textual output; Windows OpenBLAS rounds the
// last digit of some entries differently, so the hash drifts. The test
// stays useful as a same-platform regression on Linux/macOS — skipping
// it on Windows rather than weakening it to a "shape only" check.
#[cfg(not(windows))]
#[test]
fn vector_extractor_ffn_down_byte_identical() {
    let dir = fixture("vex");
    let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
    let out = dir.join("output");
    std::fs::create_dir_all(&out).unwrap();

    let cfg = ExtractConfig {
        components: vec!["ffn_down".into()],
        layers: Some(vec![0]),
        top_k: 3,
    };
    let mut cb = SilentExtractCallbacks;
    extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();

    let path = out.join("ffn_down.vectors.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    // Sort lines so re-ordering of feature loops doesn't break the
    // golden — content is what we care about. Skip the `_header`
    // record: it carries `extraction_date` (today's date), which would
    // otherwise drift the hash every wall-clock day.
    let mut lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.contains("\"_header\":true"))
        .collect();
    lines.sort();
    let canonical = lines.join("\n");
    let hex = sha256_hex(canonical.as_bytes());
    check_or_print(
        "vector_extractor_ffn_down_layer0",
        &hex,
        GOLDEN_VECTOR_EXTRACTOR_FFN_DOWN_LAYER0,
    );

    cleanup(&dir);
}

#[test]
fn fixture_is_deterministic_across_runs() {
    // Build the fixture twice and verify the safetensors bytes match.
    // If this test fails the goldens above are unreliable — every
    // walker test depends on `create_mock_model` being a pure function.
    let dir_a = fixture("det_a");
    let dir_b = fixture("det_b");
    let bytes_a = std::fs::read(dir_a.join("model.safetensors")).unwrap();
    let bytes_b = std::fs::read(dir_b.join("model.safetensors")).unwrap();
    assert_eq!(bytes_a, bytes_b, "fixture is not deterministic");
    cleanup(&dir_a);
    cleanup(&dir_b);
}
