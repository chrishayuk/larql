//! Q4_K-flavoured synthetic vindexes and the mock GPU backend.
//!
//! Split out of `test_utils.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

/// Build a fully-populated synthetic `VectorIndex` that satisfies the
/// cached + direct-matvec decode contract on the Q4_K weights from
/// [`make_test_q4k_weights`]. Quantises Q/K/V/O and gate/up/down to
/// Q4_K bytes via `quantize_q4_k`, installs them as the attn +
/// interleaved Q4_K storage, and synthesises a Q4_K lm_head view from
/// the (tied) embeddings.
pub fn make_test_q4k_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let gate_vectors = vec![None; weights.num_layers];
    let down_meta = vec![None; weights.num_layers];
    let mut index = larql_vindex::VectorIndex::new(
        gate_vectors,
        down_meta,
        weights.num_layers,
        weights.hidden_size,
    );
    index.vocab_size = weights.vocab_size;
    attach_q4k_model_storage_to_vindex(weights, &mut index);
    index
}

/// [`make_test_q4k_vindex`] variant whose **heap gate vectors are the
/// model's `ffn_gate` tensors** (f32) alongside the Q4_K FFN storage.
/// The plain Q4K fixture carries no gate vectors at all, so
/// `gate_scores_batch` returns `None` there (`resolve_gate` finds
/// nothing) — this variant is for tests of the batched-gate-score
/// consumers: the joint selectors and the full-K gemv fast path.
/// (In production the loader synthesises f32 gates from the Q4K bytes —
/// `synthesize_gate_from_q4k`; a heap copy of the source tensor plays
/// that role in-memory.)
pub fn make_test_q4k_vindex_with_model_gate(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let mut index = make_model_gate_vindex(weights);
    index.vocab_size = weights.vocab_size;
    attach_q4k_model_storage_to_vindex(weights, &mut index);
    index
}

/// [`make_test_q4k_vindex`] variant whose heap gate vectors are the
/// **dequantised Q4_K gate bytes** — the exact production shape: the
/// loader synthesises f32 gates from the interleaved Q4K gate slab
/// (`synthesize_gate_from_q4k`), so gate-KNN scores and the Q4K dense
/// base compute from the *same dequantised values*. The model-gate
/// variant above scores gates from the pre-quantisation f32 tensor
/// instead, which diverges from the Q4K bytes by the gate's
/// quantisation error — fine for selection tests, fatal for the
/// base+delta exactness tests (2026-07-30 review, item 16) that
/// compare walk gate scores against the dense Q4K base.
pub fn make_test_q4k_vindex_with_synth_gate(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let staging = make_test_q4k_vindex(weights);
    let hidden = weights.hidden_size;
    let inter = weights.intermediate_size;
    let mut gate_vectors = Vec::with_capacity(weights.num_layers);
    for layer in 0..weights.num_layers {
        let cache = staging
            .kquant_ffn_layer(layer, 0)
            .expect("Q4K fixture must expose a dequantised gate cache");
        assert_eq!(cache.len(), inter * hidden, "gate cache is [inter, hidden]");
        let m = Array2::from_shape_vec((inter, hidden), cache.to_vec()).unwrap();
        gate_vectors.push(Some(m));
    }
    let down_meta = vec![None; weights.num_layers];
    let mut index = larql_vindex::VectorIndex::new(
        gate_vectors,
        down_meta,
        weights.num_layers,
        weights.hidden_size,
    );
    index.vocab_size = weights.vocab_size;
    attach_q4k_model_storage_to_vindex(weights, &mut index);
    index
}

/// Quantise the model's attention + FFN + lm_head tensors to Q4_K and
/// install them as the vindex's `attn_kquant` / `interleaved_kquant` /
/// `lm_head` storage. Shared by [`make_test_q4k_vindex`] and
/// [`make_test_q4k_vindex_with_model_gate`].
pub fn attach_q4k_model_storage_to_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let num_layers = weights.num_layers;
    let arch = &*weights.arch;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_payload: Vec<u8> = Vec::new();
    let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = attn_payload.len();
            let length = bytes.len();
            attn_payload.extend_from_slice(&bytes);
            attn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let mut ffn_payload: Vec<u8> = Vec::new();
    let mut ffn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = ffn_payload.len();
            let length = bytes.len();
            ffn_payload.extend_from_slice(&bytes);
            ffn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let attn_mmap = arc_mmap_from_bytes(&attn_payload);
    let ffn_mmap = arc_mmap_from_bytes(&ffn_payload);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_attn_kquant(attn_mmap, Some(attn_manifest));
        storage.set_interleaved_kquant(ffn_mmap, Some(ffn_manifest));
    }

    // Synth Q4_K lm_head from tied embedding (same lifecycle as
    // `synthesize_lm_head_kquant` on a real tied-embedding vindex).
    let lm_head_slice = weights
        .lm_head
        .as_slice()
        .expect("lm_head contiguous row-major");
    let lm_head_q4 = quantize_q4_k(lm_head_slice);
    let lm_head_mmap = arc_mmap_from_bytes(&lm_head_q4);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_lm_head_kquant_mmap(lm_head_mmap);
    }

    // Also populate the f32 lm_head view so callers reaching
    // `lm_head_knn_backend_skip_q4k` get a non-empty fallback when the
    // backend's Q4_K stride-32 / f16 GEMV paths aren't implemented
    // (e.g. `MockGpuBackend` delegating to `CpuBackend`'s default
    // `q4k_matvec_stride32 → None`). Without this, `forced_logits` and
    // anything else that routes through that helper short-circuits on
    // "vindex lm_head returned no scores".
    let lm_head_f32_bytes: Vec<u8> = lm_head_slice.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lm_head_f32_mmap = arc_mmap_from_bytes(&lm_head_f32_bytes);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_lm_head_f32(lm_head_f32_mmap);
    }
}

/// Like [`make_test_q4k_vindex`] but with no FFN mmap — simulates a
/// pure-MoE model where dense FFN weights don't exist.
pub fn make_test_q4k_vindex_attn_only(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_payload: Vec<u8> = Vec::new();
    let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = attn_payload.len();
            let length = bytes.len();
            attn_payload.extend_from_slice(&bytes);
            attn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let gate_vectors = vec![None; num_layers];
    let down_meta = vec![None; num_layers];
    let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    index.vocab_size = weights.vocab_size;

    let attn_mmap = arc_mmap_from_bytes(&attn_payload);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_attn_kquant(attn_mmap, Some(attn_manifest));
    }

    index
}

/// Minimum Q4_K-aligned hidden / intermediate / expert-intermediate
/// for the Gemma 4 hybrid-MoE fixture. Q4_K requires multiples of 256.
pub const GEMMA4_MOE_HIDDEN: usize = 256;
pub const GEMMA4_MOE_INTER: usize = 256;
pub const GEMMA4_MOE_NUM_EXPERTS: usize = 4;
pub const GEMMA4_MOE_TOP_K: usize = 2;

/// Build a synthetic Gemma 4 hybrid-MoE `ModelWeights`.
///
/// `enable_moe_block=true` plus all the per-layer dense attention + dense
/// FFN tensors a Gemma 4 26B-A4B variant carries, plus the per-layer MoE
/// pieces:
///
/// - Router projection (`vectors[layers.L.router.proj.weight]`).
/// - Packed BF16 expert `gate_up` (`raw_bytes[layers.L.experts.gate_up_proj]`).
/// - Packed BF16 expert `down`    (`raw_bytes[layers.L.experts.down_proj]`).
///
/// All weights are deterministic LCG ramps. Values are math-meaningless;
/// the fixture's job is to satisfy the runtime checks
/// (`arch.is_hybrid_moe()=true`, `weights.get_packed_bytes(...)` non-None,
/// `weights.vectors[router_key]` non-None) so the MoE forward branches
/// in `pipeline_layer::build_moe_weights`,
/// `vindex/kquant_forward/hidden.rs::run_moe_layer_cpu`, and
/// `vindex/kquant_forward/remote_ffn.rs` execute end-to-end.
pub fn make_test_gemma4_moe_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = GEMMA4_MOE_HIDDEN / num_q;
    let num_layers = 2usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": GEMMA4_MOE_HIDDEN,
            "intermediate_size": GEMMA4_MOE_INTER,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_q,
            "num_key_value_heads": num_kv,
            "head_dim": head_dim,
            "vocab_size": GEMMA4_MOE_HIDDEN,
            "enable_moe_block": true,
            "num_experts": GEMMA4_MOE_NUM_EXPERTS,
            "top_k_experts": GEMMA4_MOE_TOP_K,
            "moe_intermediate_size": GEMMA4_MOE_INTER,
            "rope_theta": 10000.0,
        }
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

    let mut seed = 0xb000_1eef_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let hidden = GEMMA4_MOE_HIDDEN;
    let inter = GEMMA4_MOE_INTER;
    let moe_inter = GEMMA4_MOE_INTER;
    let vocab = GEMMA4_MOE_HIDDEN;

    let embed = rand_mat_seeded(vocab, hidden, 0.05, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    for layer in 0..num_layers {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(hidden, q_dim, 0.05, next_seed()),
        );

        // Hybrid: every layer also carries a dense MLP alongside MoE.
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, inter, 0.05, next_seed()),
        );

        // Gemma 4 four-norm layout.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.5; hidden]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.layer_scalar_key(layer) {
            vectors.insert(k, vec![1.0]);
        }

        // ── MoE pieces ───────────────────────────────────────────────
        let router_key = arch
            .moe_router_key(layer)
            .expect("Gemma 4 MoE arch must produce a router key");
        let router_proj: Vec<f32> = (0..GEMMA4_MOE_NUM_EXPERTS * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        vectors.insert(router_key, router_proj);

        // Packed BF16 expert gate_up: num_experts × [2*moe_inter, hidden].
        // BF16 = top 16 bits of the f32 little-endian representation; the
        // per-byte ramp keeps every block non-degenerate without
        // saturating the activation.
        let gate_up_floats_per_expert = 2 * moe_inter * hidden;
        let total_gate_up_bytes = GEMMA4_MOE_NUM_EXPERTS * gate_up_floats_per_expert * 2;
        let mut gate_up_blob = vec![0u8; total_gate_up_bytes];
        for (i, chunk) in gate_up_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.001 - 0.128) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let gate_up_key = arch
            .packed_experts_gate_up_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed gate_up key");
        raw_bytes.insert(gate_up_key, gate_up_blob);

        let down_floats_per_expert = hidden * moe_inter;
        let total_down_bytes = GEMMA4_MOE_NUM_EXPERTS * down_floats_per_expert * 2;
        let mut down_blob = vec![0u8; total_down_bytes];
        for (i, chunk) in down_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.0007 - 0.09) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let down_key = arch
            .packed_experts_down_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed down key");
        raw_bytes.insert(down_key, down_blob);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes,
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: vocab,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

// `synthetic_e2b_like_arch_json` + `make_synthetic_e2b_like_weights`
// moved to `larql_models::test_fixtures` (ADR-0022 Step 2e2). Re-exported
// so existing `crate::test_utils::*` callers (forward/ple.rs tests) and
// downstream test crates keep working.
pub use larql_models::test_fixtures::{
    make_synthetic_e2b_like_weights, synthetic_e2b_like_arch_json,
};
/// Bundled fixture for Q4_K decode-path tests. Mirrors `TestFixtures`.
pub struct Q4KTestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl Q4KTestFixtures {
    pub fn build() -> Self {
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

// ── MockGpuBackend — Q4-capable mock for the GPU decode/prefill paths ────────
//
// Production Metal-only paths (`gpu/decode_loop.rs`, `gpu/prefill.rs`,
// `gpu/forced_logits.rs`, `gpu/mod.rs`, `vindex/kquant_forward/metal.rs`)
// short-circuit when `backend.supports(Capability::DecodeToken | PrefillQ4)`
// returns false — which is the case for `CpuBackend`. To exercise the
// actual function bodies under test we need a backend that advertises
// those capabilities and returns shape-correct (but content-garbage) data
// from `decode_token` / `prefill_kquant`.
//
// Math methods delegate to a wrapped `CpuBackend` so test code that
// happens to read intermediate tensors gets non-garbage values where it
// can; the canned-shape returns from `decode_token` / `prefill_kquant` are
// fine for coverage because the calling code's contract is just
// `Some(Vec<f32>)` of the right length.

/// Minimal Q4-capable compute backend for tests. Delegates math to
/// `CpuBackend` and overrides `supports` + `decode_token` + `prefill_kquant`
/// so the GPU paths in `larql-inference` execute end-to-end. Output
/// values are zeros — tests assert *shape* and *that the call returned
/// Some*, not numerical correctness.
pub struct MockGpuBackend {
    inner: larql_compute::CpuBackend,
    kv_len: std::sync::atomic::AtomicUsize,
    /// `decode_token` answers `None` once this many calls have succeeded.
    /// `usize::MAX` never fails.
    decode_steps_before_failure: usize,
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            inner: larql_compute::CpuBackend,
            kv_len: std::sync::atomic::AtomicUsize::new(0),
            decode_steps_before_failure: usize::MAX,
        }
    }

    /// A backend whose `decode_token` succeeds `steps` times and then
    /// answers `None`: a GPU that fails part-way through a generation.
    pub fn failing_after(steps: usize) -> Self {
        Self {
            decode_steps_before_failure: steps,
            ..Self::new()
        }
    }
}

impl larql_compute::MatMul for MockGpuBackend {
    fn matmul(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        self.inner.matmul(a, b)
    }
    fn matmul_transb(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        self.inner.matmul_transb(a, b)
    }
}

impl larql_compute::QuantMatVec for MockGpuBackend {
    fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
        self.inner.supports_quant(format)
    }
}

impl larql_compute::DecodeBackend for MockGpuBackend {
    fn has_kv_cache(&self) -> bool {
        true
    }

    fn reset_kv_cache(&self) {
        self.kv_len.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_len.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn truncate_kv_cache(&self, len: usize) {
        self.kv_len.store(len, std::sync::atomic::Ordering::Relaxed);
    }

    fn preallocate_kv_cache_per_layer(&self, _shapes: &[(usize, usize)], _max_seq: usize) {
        // No-op — we don't actually hold a cache, just a length counter.
    }

    fn decode_token(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        let succeeded = self
            .kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if succeeded >= self.decode_steps_before_failure {
            return None;
        }
        Some(vec![0.0f32; hidden])
    }

    fn decode_token_with_moe(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Invoke the MoE callback once with a zero residual so the
        // expert dispatch path runs end-to-end.
        let _ = moe_fn(0, &vec![0.0f32; hidden]);
        self.kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; hidden])
    }

    fn decode_token_q4k_moe<'w>(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        _norm_eps: f32,
        get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<f32>> {
        let _ = get_expert(0, 0);
        self.kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; hidden])
    }

    fn prefill_kquant(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        self.kv_len
            .store(seq_len, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; seq_len * hidden])
    }
}

impl larql_compute::ComputeBackend for MockGpuBackend {
    fn name(&self) -> &str {
        "mock-gpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: larql_compute::backend::Capability) -> bool {
        use larql_compute::backend::Capability::*;
        matches!(
            cap,
            DecodeToken
                | DecodeMoe
                | DecodeQ4KMoe
                | PrefillQ4
                | FullPipelineQ4
                | QuantMatVec
                | Q4VecMat
                | Q4PairBatch
        )
    }
}
