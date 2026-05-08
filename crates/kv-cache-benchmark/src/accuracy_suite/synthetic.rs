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

pub const UPSTREAM_PPL_TOLERANCE_PCT: f64 = 2.0;
pub const ROTORQUANT_ISO3_LLAMA31_8B_WIKITEXT2_PPL: f64 = 6.91;

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
    pub upstream_ppl: Option<f64>,
    pub ppl_delta_pct: Option<f64>,
    pub ppl_within_upstream_tolerance: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PplMeasurement {
    pub strategy: String,
    pub ppl: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UpstreamPplCheck {
    pub strategy: String,
    pub model_name: String,
    pub measured_ppl: f64,
    pub upstream_ppl: f64,
    pub delta_pct: f64,
    pub tolerance_pct: f64,
    pub passed: bool,
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
            let upstream_ppl = upstream_ppl_for(&result.strategy_name, config.name);
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
                upstream_ppl,
                ppl_delta_pct: None,
                ppl_within_upstream_tolerance: None,
            }
        })
        .collect()
}

pub fn apply_ppl_measurements(
    rows: &mut [SyntheticStrategyReport],
    measurements: &[PplMeasurement],
) {
    for row in rows {
        if let Some(measured) = measurements
            .iter()
            .find(|m| strategy_matches(&row.strategy, &m.strategy))
        {
            row.ppl = Some(measured.ppl);
            if let Some(upstream) = row.upstream_ppl {
                let delta = ppl_delta_pct(measured.ppl, upstream);
                row.ppl_delta_pct = Some(delta);
                row.ppl_within_upstream_tolerance = Some(delta <= UPSTREAM_PPL_TOLERANCE_PCT);
            }
        }
    }
}

pub fn upstream_ppl_checks(rows: &[SyntheticStrategyReport]) -> Vec<UpstreamPplCheck> {
    rows.iter()
        .filter_map(|row| {
            let measured = row.ppl?;
            let upstream = row.upstream_ppl?;
            let delta = row
                .ppl_delta_pct
                .unwrap_or_else(|| ppl_delta_pct(measured, upstream));
            Some(UpstreamPplCheck {
                strategy: row.strategy.clone(),
                model_name: row.model_name.clone(),
                measured_ppl: measured,
                upstream_ppl: upstream,
                delta_pct: delta,
                tolerance_pct: UPSTREAM_PPL_TOLERANCE_PCT,
                passed: delta <= UPSTREAM_PPL_TOLERANCE_PCT,
            })
        })
        .collect()
}

pub fn format_synthetic_strategy_report(rows: &[SyntheticStrategyReport]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Synthetic KV Strategy Accuracy/Throughput ===\n\n");
    out.push_str(&format!(
        "{:<30} {:>9} {:>9} {:>9} {:>12} {:>17}\n",
        "Strategy", "cosine", "MSE", "ratio", "decode tok/s", "PPL",
    ));
    out.push_str(&"-".repeat(94));
    out.push('\n');

    for row in rows {
        let ppl = format_ppl_cell(row);
        out.push_str(&format!(
            "{:<30} {:>9.4} {:>9.4} {:>8.2}x {:>12.0} {:>17}\n",
            row.strategy, row.cosine_sim, row.mse, row.compression_ratio, row.decode_tok_s, ppl,
        ));
    }

    out
}

fn upstream_ppl_for(strategy: &str, model_name: &str) -> Option<f64> {
    if strategy.contains("RotorQuant Iso3") && model_name == "Llama 3.1 8B" {
        Some(ROTORQUANT_ISO3_LLAMA31_8B_WIKITEXT2_PPL)
    } else {
        None
    }
}

fn strategy_matches(row_strategy: &str, measurement_strategy: &str) -> bool {
    row_strategy == measurement_strategy
        || row_strategy.contains(measurement_strategy)
        || measurement_strategy.contains(row_strategy)
}

fn ppl_delta_pct(measured: f64, upstream: f64) -> f64 {
    ((measured - upstream).abs() / upstream) * 100.0
}

fn format_ppl_cell(row: &SyntheticStrategyReport) -> String {
    match (row.ppl, row.upstream_ppl, row.ppl_delta_pct) {
        (Some(ppl), Some(_), Some(delta)) => {
            let status = if delta <= UPSTREAM_PPL_TOLERANCE_PCT {
                "ok"
            } else {
                "out"
            };
            format!("{ppl:.2} ({delta:.1}% {status})")
        }
        (Some(ppl), _, _) => format!("{ppl:.2}"),
        (None, Some(upstream), _) => format!("real-only/{upstream:.2}"),
        (None, None, _) => "real-only".to_string(),
    }
}
