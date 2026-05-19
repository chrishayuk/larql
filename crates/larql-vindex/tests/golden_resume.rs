//! Golden test — `build_vindex_streaming` auto-resume preserves output.
//!
//! Round-3 added phase-level checkpoints (`.extract_checkpoint.json`)
//! and auto-resume: a streaming extract that completes the `Gate` phase
//! marks itself in the checkpoint; a subsequent run reuses the existing
//! `gate_vectors.bin` and regenerates the remaining phases.
//!
//! This test proves the resume path produces a vindex that's bit-equal
//! to the no-resume reference. If a future change to the gate-phase
//! writer (offset math, layer info shape, etc.) drifts away from the
//! resume path, this test fires.
//!
//! Plan:
//!   1. Build a small synthetic safetensors model on disk.
//!   2. Run streaming extract once → reference output. Snapshot every
//!      output file's SHA-256.
//!   3. Build a fresh output dir, copy only `gate_vectors.bin` from the
//!      reference into it, then plant a checkpoint marking the gate
//!      phase complete with the layer_infos that the reference would
//!      have written.
//!   4. Re-run streaming extract on the fresh dir.
//!   5. Assert every reference SHA matches the resumed dir's SHA, and
//!      that the checkpoint file is gone (extract clears it on success).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, KquantWriteOptions, QuantFormat, SilentBuildCallbacks,
    StorageDtype, WriteWeightsOptions,
};

/// Atomic counter for unique tmp dirs in parallel test runs.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let pid = std::process::id();
        let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("larql_resume_{label}_{pid}_{n}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_synth_model(model_dir: &Path) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_theta": 10000.0,
        "vocab_size": 16,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let embed: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    tensors.insert("model.embed_tokens.weight".into(), embed);
    metadata.push(("model.embed_tokens.weight".into(), vec![16, 8]));

    for layer in 0..2 {
        let gate: Vec<f32> = (0..32).map(|i| (i as f32 + layer as f32) * 0.1).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.gate_proj.weight"), gate);
        metadata.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![4, 8],
        ));

        let down: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.down_proj.weight"), down);
        metadata.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![8, 4],
        ));
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
}

fn run_extract(model_dir: &Path, output_dir: &Path) {
    let tok_bytes = std::fs::read(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(&tok_bytes).unwrap();
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        model_dir,
        &tokenizer,
        "test/resume",
        output_dir,
        5,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();
}

fn sha_file(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

/// Hash every regular file under `dir`, keyed by the relative path.
fn snapshot_dir(dir: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for entry in walkdir(dir) {
        if !entry.is_file() {
            continue;
        }
        let rel = entry
            .strip_prefix(dir)
            .unwrap()
            .to_string_lossy()
            .to_string();
        out.insert(rel, sha_file(&entry));
    }
    out
}

fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&p) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    out.push(path);
                }
            }
        }
    }
    out
}

#[test]
fn resume_after_gate_complete_matches_full_run() {
    let model = TempDir::new("model");
    write_synth_model(&model.0);

    // ── Reference: one clean run end-to-end ──
    let ref_dir = TempDir::new("ref");
    run_extract(&model.0, &ref_dir.0);
    let ref_shas = snapshot_dir(&ref_dir.0);
    // Sanity: must have produced the core artifacts.
    assert!(ref_shas.contains_key("gate_vectors.bin"));
    assert!(ref_shas.contains_key("down_meta.bin"));
    assert!(ref_shas.contains_key("index.json"));
    // Successful extract clears the checkpoint.
    assert!(!ref_dir.0.join(".extract_checkpoint.json").exists());

    // ── Resume: pre-populate Gate-complete checkpoint + gate file ──
    let resume_dir = TempDir::new("resume");
    std::fs::copy(
        ref_dir.0.join("gate_vectors.bin"),
        resume_dir.0.join("gate_vectors.bin"),
    )
    .unwrap();

    // Reconstruct the gate_layer_infos the prior run would have saved.
    // We read them from the reference index.json — same values, same
    // shape. (Simpler than re-running the gate phase on a sink.)
    let ref_idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_dir.0.join("index.json")).unwrap()).unwrap();
    let layers = ref_idx["layers"].clone();

    let checkpoint = serde_json::json!({
        "version": 1,
        "model_dir": model.0.display().to_string(),
        "model_name": "test/resume",
        "num_layers": 2,
        "completed": ["gate"],
        "last_update": "2026-04-25T00:00:00Z",
        "gate_layer_infos": layers,
    });
    std::fs::write(
        resume_dir.0.join(".extract_checkpoint.json"),
        serde_json::to_string_pretty(&checkpoint).unwrap(),
    )
    .unwrap();

    // ── Re-run with checkpoint present ──
    run_extract(&model.0, &resume_dir.0);

    let resume_shas = snapshot_dir(&resume_dir.0);
    // Same artifacts, same bytes — except `index.json` carries a fresh
    // `extracted_at` timestamp every run. Compare that one structurally
    // with the timestamp masked.
    for (name, ref_sha) in &ref_shas {
        let got = resume_shas
            .get(name)
            .unwrap_or_else(|| panic!("resume run missing {name}"));
        if name == "index.json" {
            assert_eq!(
                index_without_timestamp(&ref_dir.0),
                index_without_timestamp(&resume_dir.0),
                "index.json (less timestamp) differs between fresh run and resume run",
            );
            continue;
        }
        assert_eq!(
            got, ref_sha,
            "{name} differs between fresh run and resume run",
        );
    }
    // Resume run also clears the checkpoint at the end.
    assert!(!resume_dir.0.join(".extract_checkpoint.json").exists());
}

fn index_without_timestamp(dir: &Path) -> serde_json::Value {
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    if let Some(map) = v.as_object_mut() {
        map.remove("extracted_at");
        if let Some(source) = map
            .get_mut("source")
            .and_then(|value| value.as_object_mut())
        {
            source.remove("extracted_at");
        }
    }
    v
}

#[test]
fn incompatible_checkpoint_is_discarded() {
    // Plant a checkpoint whose `model_dir` doesn't match the run's
    // model_dir — extract must throw it away and run a fresh end-to-end
    // pass, producing the same bytes as a clean run.
    let model = TempDir::new("model_inc");
    write_synth_model(&model.0);

    let ref_dir = TempDir::new("ref_inc");
    run_extract(&model.0, &ref_dir.0);
    let ref_shas = snapshot_dir(&ref_dir.0);

    let stale = TempDir::new("stale");
    let bad_checkpoint = serde_json::json!({
        "version": 1,
        "model_dir": "/some/other/model",
        "model_name": "different/model",
        "num_layers": 99,
        "completed": ["gate", "down_meta", "weights"],
        "last_update": "2020-01-01T00:00:00Z",
        "gate_layer_infos": null,
    });
    std::fs::write(
        stale.0.join(".extract_checkpoint.json"),
        serde_json::to_string_pretty(&bad_checkpoint).unwrap(),
    )
    .unwrap();

    run_extract(&model.0, &stale.0);
    let stale_shas = snapshot_dir(&stale.0);
    for (name, ref_sha) in &ref_shas {
        let got = stale_shas
            .get(name)
            .unwrap_or_else(|| panic!("stale-checkpoint run missing {name}"));
        if name == "index.json" {
            assert_eq!(
                index_without_timestamp(&ref_dir.0),
                index_without_timestamp(&stale.0),
                "index.json (less timestamp) differs from clean run \
                 despite stale checkpoint being discarded",
            );
            continue;
        }
        assert_eq!(
            got, ref_sha,
            "{name} differs from clean run despite stale checkpoint being discarded",
        );
    }
}
