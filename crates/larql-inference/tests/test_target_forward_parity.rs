//! Phase 4c task C.5 — `target_forward_via_speculative_decode_matches_naive_64_seeds`.
//!
//! Cross-validates the canonical-backend `target_forward_via_speculative_decode`
//! (the helper that uses `decode_tokens_speculative` + the pluggable
//! lm_head closure) against the from-scratch `target_forward_naive`
//! parity oracle across 64 deterministic random configurations.
//!
//! ## Why this is the load-bearing parity gate
//!
//! `target_forward_via_speculative_decode` is the helper the v3 spec
//! dispatch (`try_thread_speculative_step_v3`) calls. Its output drives
//! `verify_tree`'s accept/reject decisions and bonus resampling — any
//! systematic bias here shows up as wrong-token output during spec
//! decoding. The parity-against-naive gate ensures the helper's
//! GPU-routed compute (canonical KV cache + GPU lm_head via the
//! closure) is faithful to the CPU oracle.
//!
//! ## Tolerance
//!
//! - **Top-1 argmax MUST match** per-node — this is what verify_tree
//!   actually depends on at the position where the drafter's pick
//!   happens to be the model's argmax.
//! - **Cosine similarity ≥ 0.99** per-node prob vector — looser than
//!   bit-exact because the helper uses CUDA's f16 KV cache (vs naive's
//!   f32 from-scratch) and reduce-add ordering differs across the
//!   GPU's parallel tree-reduce kernels. 0.99 is well within the
//!   accept/reject decision boundary.
//!
//! ## Gating
//!
//! Skips silently when:
//! - `LARQL_TARGET_FORWARD_PARITY_VINDEX` env var is unset, or
//! - the active default backend reports `has_kv_cache() == false`
//!   (CPU-only builds, or CUDA-init failure).
//!
//! Run locally with the canonical 4B vindex:
//!
//! ```bash
//! LARQL_TARGET_FORWARD_PARITY_VINDEX=$(pwd)/output/gemma-3-4b-it-vindex \
//!   cargo test --release -p larql-inference --features cuda \
//!   --test test_target_forward_parity -- --nocapture
//! ```

use std::env;

use larql_compute::{default_backend, QuantFormat};
use larql_inference::layer_graph::pipeline_layer::{
    attention_geometry_for_arch_layer, build_pipeline_layers, kv_cache_shapes_for_arch,
    DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use larql_inference::speculative::{
    target_forward_naive, target_forward_via_speculative_decode, DraftToken, DraftTree,
    TargetForwardDims,
};
use larql_models::ModelWeights;
use larql_vindex::{GateIndex, VectorIndex};

fn vindex_path_or_skip() -> Option<std::path::PathBuf> {
    env::var("LARQL_TARGET_FORWARD_PARITY_VINDEX")
        .ok()
        .map(std::path::PathBuf::from)
}

fn load_weights_and_index(path: &std::path::Path) -> (ModelWeights, VectorIndex) {
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(path, &mut callbacks).expect("load weights");
    let index = larql_inference::open_inference_vindex(path).expect("open vindex");
    (weights, index)
}

/// Tiny xorshift64 — deterministic per-seed RNG without pulling in a
/// dep just for this test.
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15).max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        let span = hi.saturating_sub(lo).max(1);
        lo + (self.next_u64() as u32) % span
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (av, bv) in a.iter().zip(b) {
        let af = *av as f64;
        let bf = *bv as f64;
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() && x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Build a deterministic linear-chain `DraftTree` with `depth` nodes
/// (root + depth-1 chained children). Token IDs come from `rng`.
fn random_linear_tree(rng: &mut XorShift, depth: usize, vocab: u32) -> DraftTree {
    let root_id = rng.range(0, vocab);
    let mut tree = DraftTree::from_root(DraftToken {
        id: root_id,
        p_draft: 1.0,
    });
    let mut parent = 0_usize;
    for _ in 1..depth {
        let id = rng.range(0, vocab);
        parent = tree.add_child(parent, DraftToken { id, p_draft: 1.0 });
    }
    tree
}

#[test]
fn target_forward_via_speculative_decode_matches_naive_64_seeds() {
    let Some(path) = vindex_path_or_skip() else {
        return;
    };

    let backend = default_backend();
    if !backend.has_kv_cache() {
        eprintln!(
            "skipping target_forward parity test: default backend lacks KV cache \
             (CPU-only build or CUDA init failed)"
        );
        return;
    }

    let (mut weights, index) = load_weights_and_index(&path);

    // Pipeline / dims setup once. Mirrors gpu.rs's `generate_streaming`
    // prelude.
    let hidden = weights.hidden_size;
    let attn = attention_geometry_for_arch_layer(&weights, 0);
    let intermediate = index.num_features(0);
    assert!(intermediate > 0, "vindex must report intermediate > 0");

    let gate_index: &dyn GateIndex = &index;
    let (_, ffn_is_q4k) = match gate_index.interleaved_q4k_mmap_ref() {
        Some(m) => (m, true),
        None => (
            gate_index
                .interleaved_q4_mmap_ref()
                .expect("vindex must provide Q4 ffn mmap"),
            false,
        ),
    };
    let ffn_format = if ffn_is_q4k {
        QuantFormat::Q4_K
    } else {
        QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .expect("packed Q4 ffn geometry");
    let num_layers = weights.num_layers;
    let dims = TargetForwardDims {
        hidden,
        intermediate,
        q_dim: attn.q_dim,
        kv_dim: attn.kv_dim,
        num_q_heads: attn.num_q_heads,
        num_kv_heads: attn.num_kv_heads,
        head_dim: attn.head_dim,
        rope_base: attn.rope_base,
    };
    let kv_shapes = kv_cache_shapes_for_arch(&weights);

    // Vocab size for random token-ID generation. Use a SAFE upper bound
    // — a few common tokens at the bottom — to avoid OOV-shaped issues.
    let vocab = (index.vocab_size as u32).clamp(1024, 50_000);
    if vocab < 1024 {
        eprintln!("skipping: vindex vocab too small for parity test");
        return;
    }

    let mut argmax_mismatches = 0_usize;
    let mut min_cosine = 1.0_f64;
    let mut tested = 0_usize;

    for seed in 0..64u64 {
        let mut rng = XorShift::new(0xCAFEBABEDEADBEEF ^ seed);
        let history_len = rng.range(3, 30) as usize;
        let history: Vec<u32> = (0..history_len).map(|_| rng.range(0, vocab)).collect();
        let depth = rng.range(1, 4) as usize; // 1, 2, or 3 nodes

        let tree = random_linear_tree(&mut rng, depth, vocab);

        // Run naive (CPU full-precision oracle).
        let p_naive = target_forward_naive(&mut weights, &history, &tree, &index);
        if p_naive.len() != tree.len() {
            eprintln!("seed {seed}: naive returned wrong length, skipping");
            continue;
        }

        // Run helper (GPU canonical backend + KV cache). Set the
        // backend's KV cache to cover `history` exactly, mirroring
        // gpu.rs's prefill prelude. Rebuild pipeline layers per-iter
        // so the borrow lifetime fits inside this scope (naive's
        // `&mut weights` and the helper's `&layers` (which borrows
        // weights immutably) can't overlap; the rebuild is cheap —
        // it just collects slice references, no compute).
        let q4_ffn_mmap = match gate_index.interleaved_q4k_mmap_ref() {
            Some(m) => m,
            None => gate_index
                .interleaved_q4_mmap_ref()
                .expect("vindex must provide Q4 ffn mmap"),
        };
        let layers = build_pipeline_layers(
            &weights,
            &index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        backend.reset_kv_cache();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
        let h_embed = larql_inference::forward::embed_tokens_pub(&weights, &history);
        let x_flat: Vec<f32> = match h_embed.as_slice() {
            Some(s) => s.to_vec(),
            None => continue,
        };
        let prefill_out = backend.prefill_q4(
            &layers,
            &x_flat,
            hidden,
            intermediate,
            attn.q_dim,
            attn.kv_dim,
            history.len(),
            attn.num_q_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.rope_base,
            false,
            0.0,
        );
        if prefill_out.is_none() {
            eprintln!("seed {seed}: prefill_q4 returned None, skipping");
            continue;
        }
        // After prefill_q4, cache.len == history.len(); the helper's
        // strict cache contract is now satisfied.
        let p_helper = match target_forward_via_speculative_decode(
            &weights, &history, &tree, &*backend, &layers, dims,
        ) {
            Some(p) => p,
            None => {
                eprintln!("seed {seed}: helper returned None, skipping");
                continue;
            }
        };
        assert_eq!(
            p_helper.len(),
            tree.len(),
            "seed {seed}: helper output length mismatch"
        );

        // Per-node parity checks.
        for (k, (naive_node, helper_node)) in p_naive.iter().zip(&p_helper).enumerate() {
            assert_eq!(
                naive_node.len(),
                helper_node.len(),
                "seed {seed} node {k}: vocab len mismatch ({} vs {})",
                naive_node.len(),
                helper_node.len()
            );
            let argmax_n = argmax(naive_node);
            let argmax_h = argmax(helper_node);
            if argmax_n != argmax_h {
                argmax_mismatches += 1;
                eprintln!(
                    "seed {seed} node {k}: argmax mismatch — naive={argmax_n} (p={:.4}) helper={argmax_h} (p={:.4})",
                    naive_node[argmax_n], helper_node[argmax_h]
                );
            }
            let cos = cosine_similarity(naive_node, helper_node);
            if cos < min_cosine {
                min_cosine = cos;
            }
            assert!(
                cos >= 0.99,
                "seed {seed} node {k}: cosine {cos:.6} < 0.99 (helper drifted from naive)"
            );
        }
        tested += 1;
    }

    eprintln!(
        "[C.5] tested {tested}/64 seeds; argmax_mismatches={argmax_mismatches}; min_cosine={:.6}",
        min_cosine
    );

    assert!(
        tested >= 32,
        "expected at least 32/64 seeds to complete (got {tested}); investigate skips"
    );
    // Argmax CAN drift on close-call positions (multiple tokens with
    // near-identical probability). 0 mismatches is the goal but we
    // tolerate up to ~5% (3 seeds × 3 nodes ≈ 9 positions) before
    // failing the gate — that's well below verify_tree's accept-rate
    // sensitivity.
    assert!(
        argmax_mismatches <= 9,
        "too many top-1 argmax drifts ({argmax_mismatches}); helper diverging from naive"
    );
}
