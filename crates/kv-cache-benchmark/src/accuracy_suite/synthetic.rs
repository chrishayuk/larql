//! Synthetic KV strategy accuracy/throughput report.
//!
//! This default-compiled module lets the accuracy suite report
//! reconstruction quality, compression, and decode throughput without
//! requiring model weights. Real PPL remains in the `real-model` runner.

use crate::model_config::ModelConfig;
use crate::rotorquant::RotorQuantStrategy;
use crate::standard_kv::StandardKv;
use crate::turboquant::TurboQuant;
use crate::{run_strategy_benchmark, KvStrategy};
use rand::SeedableRng;

#[derive(Debug, Clone, serde::Serialize)]
pub struct SyntheticStrategyReport {
    pub strategy: String,
    pub model_name: String,
    pub seq_len: usize,
    pub cosine_sim: f64,
    pub mse: f64,
    pub compression_ratio: f64,
    pub decode_tok_s: f64,
    pub ppl: Option<f64>,
}

pub fn synthetic_strategy_report(
    config: &ModelConfig,
    seq_len: usize,
    seed: u64,
) -> Vec<SyntheticStrategyReport> {
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let tq3 = TurboQuant::new(3);
    let rq_iso3 = RotorQuantStrategy::iso3();
    let rq_planar3 = RotorQuantStrategy::planar3();
    let rq_iso4 = RotorQuantStrategy::iso4();
    let rq_planar4 = RotorQuantStrategy::planar4();
    let strategies: Vec<&dyn KvStrategy> = vec![
        &standard,
        &tq4,
        &tq3,
        &rq_iso3,
        &rq_planar3,
        &rq_iso4,
        &rq_planar4,
    ];

    strategies
        .into_iter()
        .enumerate()
        .map(|(idx, strategy)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + idx as u64);
            let result = run_strategy_benchmark(strategy, config, seq_len, &mut rng);
            let decoded_vectors = seq_len * config.layers * config.kv_heads * 2;
            let decode_secs = result.metrics.decode_us / 1_000_000.0;
            SyntheticStrategyReport {
                strategy: result.strategy_name,
                model_name: result.model_name,
                seq_len: result.seq_len,
                cosine_sim: result.metrics.cosine_sim,
                mse: result.metrics.mse,
                compression_ratio: result.metrics.compression_ratio,
                decode_tok_s: if decode_secs > 0.0 {
                    decoded_vectors as f64 / decode_secs
                } else {
                    f64::INFINITY
                },
                ppl: None,
            }
        })
        .collect()
}

pub fn format_synthetic_strategy_report(rows: &[SyntheticStrategyReport]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Synthetic KV Strategy Accuracy/Throughput ===\n\n");
    out.push_str(&format!(
        "{:<30} {:>9} {:>9} {:>9} {:>12} {:>9}\n",
        "Strategy", "cosine", "MSE", "ratio", "decode tok/s", "PPL",
    ));
    out.push_str(&"-".repeat(86));
    out.push('\n');

    for row in rows {
        let ppl = row
            .ppl
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "real-only".to_string());
        out.push_str(&format!(
            "{:<30} {:>9.4} {:>9.4} {:>8.2}x {:>12.0} {:>9}\n",
            row.strategy, row.cosine_sim, row.mse, row.compression_ratio, row.decode_tok_s, ppl,
        ));
    }

    out
}
