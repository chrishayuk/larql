//! Norms-writer tests: the MoE bias tensors reach `norms.bin`.
//!
//! §4.7.1 finding 6 was three per-layer MLP tensors dropped at extraction;
//! restoring them on the *architecture* was not enough — the kquant writer
//! never asked for them, so a served model still ran bias-less. These pin
//! that the writer stores the router bias, the fused gate/up bias and the
//! down bias whenever the architecture declares them, and that a
//! no-bias architecture's output is unchanged.

use std::collections::HashMap;

use crate::format::weights::write_f32::WeightSource;

use super::super::norms::write_norms_and_router;

const HIDDEN: usize = 8;
const NUM_LAYERS: usize = 1;
const NUM_EXPERTS: usize = 2;

/// GPT-OSS-shaped arch: declares router + expert biases.
fn biased_arch() -> Box<dyn larql_models::ModelArchitecture> {
    larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 1,
    }))
}

struct BiasSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    tensors: HashMap<String, (Vec<f32>, usize, usize)>,
    vectors: HashMap<String, Vec<f32>>,
}

impl BiasSource {
    fn new() -> Self {
        let arch = biased_arch();
        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        // Router projection (2-D) + bias (1-D).
        if let Some(k) = arch.moe_router_key(0) {
            tensors.insert(k, (vec![0.5f32; NUM_EXPERTS * HIDDEN], NUM_EXPERTS, HIDDEN));
        }
        if let Some(k) = arch.moe_router_bias_key(0) {
            vectors.insert(k, vec![1.5f32; NUM_EXPERTS]);
        }
        // Packed expert biases, 2-D [experts, …].
        if let Some(k) = arch.packed_gate_up_bias_key(0) {
            tensors.insert(
                k,
                (
                    vec![2.5f32; NUM_EXPERTS * 2 * HIDDEN],
                    NUM_EXPERTS,
                    2 * HIDDEN,
                ),
            );
        }
        if let Some(k) = arch.packed_down_bias_key(0) {
            tensors.insert(k, (vec![3.5f32; NUM_EXPERTS * HIDDEN], NUM_EXPERTS, HIDDEN));
        }
        Self {
            arch,
            tensors,
            vectors,
        }
    }
}

impl WeightSource for BiasSource {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        self.tensors.get(key).cloned()
    }
    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.vectors.get(key).cloned()
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

/// A directory of its own for every call, however many callers share a
/// `name` — within this process (a counter) and across processes (the
/// pid). A fixed name per label let two test processes running at once
/// remove each other's files mid-read.
fn temp_dir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir()
        .join("write-kquant-norms-tests")
        .join(format!(
            "{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Two calls under one label are two directories, both present.
#[test]
fn temp_dirs_are_unique_per_invocation() {
    let first = temp_dir("shared-label");
    let second = temp_dir("shared-label");
    assert_ne!(first, second);
    assert!(first.is_dir() && second.is_dir());
}

#[test]
fn moe_bias_tensors_are_written_when_the_arch_declares_them() {
    let source = BiasSource::new();
    let dir = temp_dir("biased");
    let entries = write_norms_and_router(&source, &dir, NUM_LAYERS).unwrap();

    let arch = biased_arch();
    let expect = [
        (arch.moe_router_bias_key(0).unwrap(), NUM_EXPERTS),
        (
            arch.packed_gate_up_bias_key(0).unwrap(),
            NUM_EXPERTS * 2 * HIDDEN,
        ),
        (arch.packed_down_bias_key(0).unwrap(), NUM_EXPERTS * HIDDEN),
    ];
    for (key, len) in expect {
        let entry = entries
            .iter()
            .find(|e| e.key == key)
            .unwrap_or_else(|| panic!("{key} not written — the bias is dropped at extraction"));
        assert_eq!(entry.shape, vec![len], "{key} has the wrong extent");
    }
}

/// A Gemma-4-shaped hybrid MoE declares none of the bias keys; its entry
/// list must not grow phantom entries from this change.
#[test]
fn a_no_bias_arch_writes_no_bias_entries() {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gemma4_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 1,
    }));
    assert!(arch.moe_router_bias_key(0).is_none(), "fixture drifted");
    let source = BiasSource {
        arch,
        tensors: HashMap::new(),
        vectors: HashMap::new(),
    };
    let dir = temp_dir("unbiased");
    let entries = write_norms_and_router(&source, &dir, NUM_LAYERS).unwrap();
    assert!(
        entries.iter().all(|e| !e.key.contains("bias")),
        "no-bias arch grew bias entries: {:?}",
        entries.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
}
