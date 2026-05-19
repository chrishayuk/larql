//! Golden test — save + reload a synthetic vindex, assert byte-for-byte
//! reproducibility and behavioural identity.
//!
//! This is the regression net for "I broke serialisation". One assertion
//! catches:
//! - Filename constants drift (`format::filenames`)
//! - Layer offset / stride math errors in the save path
//! - Endianness / alignment regressions in `decode_floats`
//! - mmap zero-copy path silently falling back to heap copy
//! - KNN result order changing across save/load
//!
//! The "golden" SHA is **not** hard-coded — it's recomputed per run
//! and asserted to be stable across a save/save cycle on identical
//! inputs. That's what we actually care about (determinism), without
//! the headache of a tolerance for floating-point bit shuffling on
//! different hardware.
//!
//! What's checked:
//! 1. Save yields a file whose SHA matches the SHA of a second save
//!    of the same data (determinism — no time / memory-address leakage).
//! 2. Reload + KNN matches the original heap-mode KNN bit-exactly.
//! 3. After reload, `gate_heap_bytes() == 0` (zero-copy invariant).
//! 4. Enable HNSW after reload — top-K still overlaps with brute by
//!    ≥ 4/10 (the codec hasn't degraded recall further).

#![cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};
use sha2::{Digest, Sha256};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let pid = std::process::id();
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("larql_golden_{label}_{pid}_{n}"));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sha256(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

fn synth_query(hidden: usize, seed: u64) -> Array1<f32> {
    let mut state = seed;
    Array1::from_shape_fn(hidden, |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn build_synthetic_vindex(num_layers: usize, features: usize, hidden: usize) -> VectorIndex {
    let mut state = 42u64;
    let mut gate_vectors = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let gate = Array2::from_shape_fn((features, hidden), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        });
        gate_vectors.push(Some(gate));

        let metas: Vec<Option<FeatureMeta>> = (0..features)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{i}"),
                    top_token_id: i as u32,
                    c_score: 0.5,
                    top_k: vec![TopKEntry {
                        token: format!("tok{i}"),
                        token_id: i as u32,
                        logit: 0.5,
                    }],
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }
    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

fn save_full_vindex(
    index: &VectorIndex,
    dir: &std::path::Path,
    num_layers: usize,
    hidden: usize,
    features: usize,
) {
    let layer_infos = index.save_gate_vectors(dir).unwrap();
    index.save_down_meta(dir).unwrap();

    // Minimal tokenizer JSON so load_vindex doesn't choke on the
    // tokenizer.json read in load_vindex_tokenizer.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "golden-test".into(),
        family: "synthetic".into(),
        num_layers,
        hidden_size: hidden,
        intermediate_size: features,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 1,
        ..Default::default()
    };
    VectorIndex::save_config(&config, dir).unwrap();
}

#[test]
fn save_is_deterministic() {
    // Two saves of the same in-memory vindex must produce identical
    // bytes. Catches time-leakage, address-randomisation, or
    // hash-map iteration order in the save path.
    let num_layers = 3;
    let features = 64;
    let hidden = 32;
    let index = build_synthetic_vindex(num_layers, features, hidden);

    let a = TempDir::new("det_a");
    let b = TempDir::new("det_b");
    save_full_vindex(&index, &a.0, num_layers, hidden, features);
    save_full_vindex(&index, &b.0, num_layers, hidden, features);

    let sha_a = sha256(&a.0.join("gate_vectors.bin"));
    let sha_b = sha256(&b.0.join("gate_vectors.bin"));
    assert_eq!(
        sha_a, sha_b,
        "gate_vectors.bin not deterministic across saves"
    );

    let sha_a_meta = sha256(&a.0.join("down_meta.bin"));
    let sha_b_meta = sha256(&b.0.join("down_meta.bin"));
    assert_eq!(sha_a_meta, sha_b_meta, "down_meta.bin not deterministic");
}

#[test]
fn knn_round_trip_preserves_results() {
    // Heap-mode KNN result must match mmap-mode KNN result after
    // save + reload. Bit-for-bit on f32, since neither path does any
    // approximation.
    let num_layers = 3;
    let features = 256;
    let hidden = 64;
    let original = build_synthetic_vindex(num_layers, features, hidden);
    let query = synth_query(hidden, 0xdeadbeef);

    // Heap-mode reference.
    let heap_results = original.gate_knn(1, &query, 10);
    assert_eq!(heap_results.len(), 10);

    // Save, reload via mmap, requery.
    let tmp = TempDir::new("rt");
    save_full_vindex(&original, &tmp.0, num_layers, hidden, features);
    let mut cb = SilentLoadCallbacks;
    let reloaded = VectorIndex::load_vindex(&tmp.0, &mut cb).unwrap();
    let mmap_results = reloaded.gate_knn(1, &query, 10);

    assert_eq!(
        heap_results, mmap_results,
        "KNN results diverged across save/load — mmap path is not bit-exact",
    );
}

#[test]
fn mmap_load_is_zero_copy() {
    // After mmap-load on f32 storage, the gate heap should be empty.
    // Catches accidental clones / fallbacks that bloat RSS.
    let num_layers = 2;
    let features = 128;
    let hidden = 32;
    let original = build_synthetic_vindex(num_layers, features, hidden);

    let tmp = TempDir::new("zc");
    save_full_vindex(&original, &tmp.0, num_layers, hidden, features);
    let mut cb = SilentLoadCallbacks;
    let reloaded = VectorIndex::load_vindex(&tmp.0, &mut cb).unwrap();

    assert!(reloaded.is_mmap(), "expected mmap-mode after load_vindex");
    assert_eq!(
        reloaded.gate_heap_bytes(),
        0,
        "gate heap should be zero on mmap load — got {} bytes",
        reloaded.gate_heap_bytes()
    );
}

#[test]
fn hnsw_after_reload_overlaps_brute() {
    // Wire-up smoke: turning HNSW on against an mmap-reloaded index
    // returns sensible top-K (overlaps brute by at least 4/10 — same
    // bound as `gate_knn_hnsw_smoke` in test_hnsw.rs).
    let num_layers = 1;
    let features = 1024;
    let hidden = 64;
    let original = build_synthetic_vindex(num_layers, features, hidden);

    let tmp = TempDir::new("hnsw");
    save_full_vindex(&original, &tmp.0, num_layers, hidden, features);
    let mut cb = SilentLoadCallbacks;
    let reloaded = VectorIndex::load_vindex(&tmp.0, &mut cb).unwrap();

    let query = synth_query(hidden, 0x31337);
    let brute = reloaded.gate_knn(0, &query, 10);
    let brute_ids: std::collections::HashSet<usize> = brute.iter().map(|(id, _)| *id).collect();

    reloaded.enable_hnsw(200);
    let hnsw = reloaded.gate_knn(0, &query, 10);
    assert_eq!(
        hnsw.len(),
        10,
        "HNSW must return requested top-K post-reload"
    );

    let hnsw_ids: std::collections::HashSet<usize> = hnsw.iter().map(|(id, _)| *id).collect();
    let overlap = hnsw_ids.intersection(&brute_ids).count();
    assert!(
        overlap >= 4,
        "post-reload HNSW recall too low: {overlap}/10",
    );
}
