//! End-to-end smoke test for the `quant::convert::vindex_to_fp4`
//! library entry. Builds a tiny synthetic source vindex (3 layers,
//! hidden=256), runs the conversion, asserts:
//!
//!  - Expected files land in the output directory.
//!  - `index.json` carries the fp4 manifest with the right precision tags.
//!  - `fp4_compliance.json` sidecar is emitted.
//!  - The reported compression ratio and walk-backend description are
//!    consistent with Option B.
//!  - Atomic-rename: `<out>.tmp/` is cleaned up.
//!  - `force` flag behaves (refuses by default, overwrites when set).

#![cfg(not(target_arch = "wasm32"))]
use larql_vindex::format::filenames::*;
use std::path::{Path, PathBuf};

use larql_vindex::quant::{vindex_to_fp4, Fp4ConvertConfig, Policy, ProjectionOutcome};

/// Minimal tempdir with drop-cleanup.
struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!("fp4_cli_{label}_{}_{}", std::process::id(), ts));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn synth_layer(num_features: usize, hidden: usize, seed: f32) -> Vec<f32> {
    (0..num_features * hidden)
        .map(|i| ((i as f32 + seed * 100.0) * 0.017).sin() * 0.1)
        .collect()
}

/// Build a minimal on-disk f32 vindex at `dir`. Carries 3 layers × 4
/// features × 256 hidden. Matches the shape `vindex_to_fp4` expects:
/// `gate_vectors.bin`, `up_features.bin`, `down_features.bin` all
/// present, plus a valid `index.json`, plus a few auxiliary files to
/// exercise the hard-link branch (tokenizer, norms, embeddings, down_meta).
fn build_minimal_f32_vindex(dir: &Path) -> (usize, usize, Vec<usize>) {
    let hidden = 256;
    let per_layer_features = vec![4usize, 4, 4];
    let num_layers = per_layer_features.len();

    // Write each projection as flat f32.
    for (idx, proj) in ["gate_vectors", "up_features", "down_features"]
        .iter()
        .enumerate()
    {
        let mut bytes = Vec::new();
        for (layer, &n) in per_layer_features.iter().enumerate() {
            let data = synth_layer(n, hidden, (idx + layer) as f32);
            for &v in &data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(dir.join(format!("{proj}.bin")), bytes).unwrap();
    }

    // index.json — matches what a real vindex would carry.
    let total_layer_bytes = per_layer_features[0] * hidden * 4;
    let layers_json: Vec<_> = per_layer_features
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            serde_json::json!({
                "layer": i,
                "num_features": n,
                "offset": i * total_layer_bytes,
                "length": total_layer_bytes as u64,
            })
        })
        .collect();
    let index = serde_json::json!({
        "version": 2,
        "model": "synthetic/fp4-test",
        "family": "synthetic",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": *per_layer_features.iter().max().unwrap(),
        "vocab_size": 16,
        "embed_scale": 1.0,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none",
        "layers": layers_json,
        "down_top_k": 1,
        "has_model_weights": false,
    });
    std::fs::write(
        dir.join("index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();

    // Minimal tokenizer.
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
    )
    .unwrap();

    let down_meta: Vec<Option<Vec<Option<larql_vindex::FeatureMeta>>>> = per_layer_features
        .iter()
        .map(|&n| Some(vec![None; n]))
        .collect();
    larql_vindex::down_meta::write_binary(dir, &down_meta, 1).unwrap();

    // Zero-filled embeddings (so the loader's opportunistic-embed
    // reader has something to look at — not strictly required).
    std::fs::write(dir.join("embeddings.bin"), vec![0u8; 16 * hidden * 4]).unwrap();

    (num_layers, hidden, per_layer_features)
}

#[test]
fn vindex_to_fp4_option_b_smoke() {
    let tmp = TempDir::new("option_b_smoke");
    let src = tmp.0.join("src.vindex");
    std::fs::create_dir_all(&src).unwrap();
    let _ = build_minimal_f32_vindex(&src);
    let dst = tmp.0.join("dst.vindex");

    let config = Fp4ConvertConfig {
        policy: Policy::B,
        ..Default::default()
    };
    let (report, _scan) = vindex_to_fp4(&src, &dst, &config).unwrap();

    // Output layout matches Option B: gate as linked source + up_fp4 + down_fp8.
    assert!(dst.join("index.json").exists(), "index.json missing");
    assert!(
        dst.join("gate_vectors.bin").exists(),
        "gate_vectors.bin (source) not linked"
    );
    assert!(
        dst.join(UP_FEATURES_FP4_BIN).exists(),
        "up FP4 file missing"
    );
    assert!(
        dst.join(DOWN_FEATURES_FP8_BIN).exists(),
        "down FP8 file missing"
    );
    assert!(dst.join("fp4_compliance.json").exists(), "sidecar missing");

    // Staging directory cleaned up.
    let staging = tmp.0.join("dst.vindex.tmp");
    assert!(
        !staging.exists(),
        "staging dir {} should not persist",
        staging.display()
    );

    // index.json carries the fp4 manifest with the right tags.
    let idx_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dst.join("index.json")).unwrap()).unwrap();
    let fp4 = idx_json["fp4"]
        .as_object()
        .expect("fp4 missing from index.json");
    let projs = &fp4["projections"];
    assert_eq!(projs["gate"]["precision"], "f32");
    assert_eq!(projs["up"]["precision"], "fp4");
    assert_eq!(projs["down"]["precision"], "fp8");
    assert_eq!(projs["gate"]["file"], "gate_vectors.bin");
    assert_eq!(projs["up"]["file"], UP_FEATURES_FP4_BIN);
    assert_eq!(projs["down"]["file"], DOWN_FEATURES_FP8_BIN);

    // Report fields consistent with Option B.
    assert_eq!(report.policy, Policy::B);
    assert_eq!(report.per_projection.len(), 3);
    let gate = report
        .per_projection
        .iter()
        .find(|p| p.name == "gate")
        .unwrap();
    let up = report
        .per_projection
        .iter()
        .find(|p| p.name == "up")
        .unwrap();
    let down = report
        .per_projection
        .iter()
        .find(|p| p.name == "down")
        .unwrap();
    assert!(matches!(gate.outcome, ProjectionOutcome::LinkedAsSource));
    assert!(matches!(up.outcome, ProjectionOutcome::WroteFp4));
    assert!(matches!(down.outcome, ProjectionOutcome::WroteFp8));
    assert!(
        report.compression > 1.0,
        "compression should exceed 1× (got {})",
        report.compression
    );
    assert!(
        report.walk_backend.contains("FP4 sparse"),
        "walk backend description should mention FP4 sparse; got {:?}",
        report.walk_backend
    );
}

#[test]
fn vindex_to_fp4_refuses_existing_output() {
    let tmp = TempDir::new("no_force");
    let src = tmp.0.join("src.vindex");
    std::fs::create_dir_all(&src).unwrap();
    let _ = build_minimal_f32_vindex(&src);
    let dst = tmp.0.join("dst.vindex");
    std::fs::create_dir_all(&dst).unwrap();

    let config = Fp4ConvertConfig {
        policy: Policy::B,
        force: false,
        ..Default::default()
    };
    let err = vindex_to_fp4(&src, &dst, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exists"),
        "expected 'exists' in error; got {msg}"
    );
}

#[test]
fn vindex_to_fp4_force_overwrites_existing() {
    let tmp = TempDir::new("force");
    let src = tmp.0.join("src.vindex");
    std::fs::create_dir_all(&src).unwrap();
    let _ = build_minimal_f32_vindex(&src);
    let dst = tmp.0.join("dst.vindex");
    std::fs::create_dir_all(&dst).unwrap();
    std::fs::write(dst.join("stale.bin"), b"stale").unwrap();

    let config = Fp4ConvertConfig {
        policy: Policy::B,
        force: true,
        ..Default::default()
    };
    let _ = vindex_to_fp4(&src, &dst, &config).unwrap();
    assert!(
        !dst.join("stale.bin").exists(),
        "force should have cleared stale contents"
    );
    assert!(dst.join(UP_FEATURES_FP4_BIN).exists());
}

#[test]
fn vindex_to_fp4_no_sidecar_skips_emission() {
    let tmp = TempDir::new("no_sidecar");
    let src = tmp.0.join("src.vindex");
    std::fs::create_dir_all(&src).unwrap();
    let _ = build_minimal_f32_vindex(&src);
    let dst = tmp.0.join("dst.vindex");

    let config = Fp4ConvertConfig {
        emit_sidecar: false,
        ..Default::default()
    };
    let _ = vindex_to_fp4(&src, &dst, &config).unwrap();
    assert!(
        !dst.join("fp4_compliance.json").exists(),
        "sidecar should be absent when emit_sidecar=false"
    );
    // Main manifest still there.
    assert!(dst.join("index.json").exists());
}
