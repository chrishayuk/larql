#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::single_range_in_vec_init)]

pub mod accuracy;
pub mod accuracy_suite;
pub mod benchmark;
pub mod graph_walk;
pub mod markov_residual;
pub mod metrics;
pub mod model_config;
pub mod rotorquant;
pub mod shader_bench;
pub mod standard_kv;
pub mod turboquant;

#[cfg(feature = "real-model")]
pub mod real_model;

// unlimited_context re-exports from larql_inference::engines — always available.
pub mod unlimited_context;

#[cfg(feature = "real-model")]
pub mod apollo;

#[cfg(feature = "real-model")]
pub mod vindex_compare;

use metrics::Metrics;
use model_config::ModelConfig;

/// Result of running a strategy on a set of KV vectors.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategyResult {
    pub strategy_name: String,
    pub model_name: String,
    pub seq_len: usize,
    pub metrics: Metrics,
}

/// Trait implemented by each KV cache strategy.
///
/// The benchmark generates random FP16-scale vectors matching real model
/// dimensions, then calls encode/decode on each strategy and measures
/// memory, time, and reconstruction accuracy.
pub trait KvStrategy {
    fn name(&self) -> &str;

    /// Encode a batch of KV vectors. Returns opaque encoded bytes.
    fn encode(&self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<u8>;

    /// Decode encoded bytes back to KV vectors.
    fn decode(
        &self,
        encoded: &[u8],
        num_vectors: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>);

    /// Analytical memory for `seq_len` tokens (config-level, no data needed).
    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize;
}

/// Run encode → decode → measure on synthetic data for any strategy.
pub fn run_strategy_benchmark(
    strategy: &dyn KvStrategy,
    config: &ModelConfig,
    seq_len: usize,
    rng: &mut impl rand::Rng,
) -> StrategyResult {
    let dim = config.kv_dim();
    let num_vectors = seq_len * config.layers * config.kv_heads;

    let keys: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let values: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    let t0 = std::time::Instant::now();
    let encoded = strategy.encode(&keys, &values);
    let encode_us = t0.elapsed().as_secs_f64() * 1e6;

    let t0 = std::time::Instant::now();
    let (dec_keys, dec_values) = strategy.decode(&encoded, num_vectors, dim);
    let decode_us = t0.elapsed().as_secs_f64() * 1e6;

    let flat_orig_k: Vec<f32> = keys.iter().flatten().copied().collect();
    let flat_dec_k: Vec<f32> = dec_keys.iter().flatten().copied().collect();
    let flat_orig_v: Vec<f32> = values.iter().flatten().copied().collect();
    let flat_dec_v: Vec<f32> = dec_values.iter().flatten().copied().collect();

    let mse_k = Metrics::compute_mse(&flat_orig_k, &flat_dec_k);
    let mse_v = Metrics::compute_mse(&flat_orig_v, &flat_dec_v);
    let cos_k = Metrics::compute_cosine(&flat_orig_k, &flat_dec_k);
    let cos_v = Metrics::compute_cosine(&flat_orig_v, &flat_dec_v);

    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    let ip_err = if !keys.is_empty() && !dec_keys.is_empty() {
        Metrics::compute_inner_product_error(&keys[0], &dec_keys[0], &queries)
    } else {
        0.0
    };

    let original_bytes = num_vectors * 2 * dim * 4;
    let compression_ratio = if encoded.is_empty() {
        0.0
    } else {
        original_bytes as f64 / encoded.len() as f64
    };

    StrategyResult {
        strategy_name: strategy.name().to_string(),
        model_name: config.name.to_string(),
        seq_len,
        metrics: Metrics {
            mse: (mse_k + mse_v) / 2.0,
            cosine_sim: (cos_k + cos_v) / 2.0,
            inner_product_error: ip_err,
            compression_ratio,
            encoded_bytes: encoded.len(),
            original_bytes,
            encode_us,
            decode_us,
        },
    }
}
