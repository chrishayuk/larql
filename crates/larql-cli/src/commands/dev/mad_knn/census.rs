//! Whole-model physical-address opportunity census.
//!
//! This intentionally runs before kNN: popularity-to-oracle headroom answers
//! whether a layer has selective logical objects that any predictor could
//! exploit. It reads an existing capture and performs no model execution.

use std::collections::HashSet;

use serde::Serialize;

use super::evaluate::{
    coverage_at_budget, parse_fraction_list, popularity_rankings, rank_by_density,
};
use super::format::{Capture, SCHEMA_V1};
use super::CensusArgs;

const CENSUS_SCHEMA_V1: &str = "larql.mad-v3-knn.census.v1";

#[derive(Debug, Serialize)]
struct CensusReport {
    schema: &'static str,
    capture_schema: &'static str,
    model: String,
    contribution_engine: Option<String>,
    history_samples: usize,
    query_samples: usize,
    query_regime: Option<String>,
    layers: Vec<LayerCensus>,
}

#[derive(Debug, Serialize)]
struct LayerCensus {
    layer: usize,
    objects: usize,
    queries: usize,
    mean_entropy_nats: f64,
    mean_effective_blocks: f64,
    mean_effective_block_fraction: f64,
    budgets: Vec<BudgetCensus>,
}

#[derive(Debug, Serialize)]
struct BudgetCensus {
    requested_byte_fraction: f64,
    popularity_candidate_byte_fraction: f64,
    popularity_contribution_coverage: f64,
    oracle_candidate_byte_fraction: f64,
    oracle_contribution_coverage: f64,
    oracle_advantage: f64,
}

#[derive(Default)]
struct BudgetAccumulator {
    popularity_bytes: f64,
    popularity_coverage: f64,
    oracle_bytes: f64,
    oracle_coverage: f64,
}

pub fn run_census(args: CensusArgs) -> Result<(), Box<dyn std::error::Error>> {
    let budgets = parse_fraction_list(&args.byte_budgets, "byte-budgets")?;
    let capture = Capture::open(&args.capture)?;
    let history: Vec<usize> = capture
        .manifest
        .samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| (sample.split == "history").then_some(index))
        .collect();
    let queries: Vec<usize> = capture
        .manifest
        .samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            (sample.split == "query"
                && args
                    .query_regime
                    .as_deref()
                    .is_none_or(|regime| sample.regime.as_deref() == Some(regime)))
            .then_some(index)
        })
        .collect();
    if queries.is_empty() {
        return Err(match &args.query_regime {
            Some(regime) => format!("capture has no query rows for regime `{regime}`"),
            None => "capture has no query rows".to_string(),
        }
        .into());
    }

    let popularity = popularity_rankings(&capture, &history);
    let mut layer_ids: Vec<usize> = capture
        .manifest
        .objects
        .iter()
        .map(|object| object.layer)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    layer_ids.sort_unstable();

    let mut layers = Vec::with_capacity(layer_ids.len());
    for layer in layer_ids {
        let objects = capture
            .objects_at(layer)
            .expect("layer came from capture object table");
        let popularity_rank = popularity
            .get(&layer)
            .expect("every contribution layer has a popularity ranking");
        let mut entropy = 0.0;
        let mut effective_blocks = 0.0;
        let mut valid_queries = 0usize;
        let mut accumulators: Vec<BudgetAccumulator> = budgets
            .iter()
            .map(|_| BudgetAccumulator::default())
            .collect();

        for &query in &queries {
            let truth_total: f64 = objects
                .iter()
                .map(|&object| f64::from(capture.contribution(query, object)))
                .sum();
            if truth_total <= f64::EPSILON {
                continue;
            }
            let h: f64 = objects
                .iter()
                .map(|&object| f64::from(capture.contribution(query, object)) / truth_total)
                .filter(|&probability| probability > 0.0)
                .map(|probability| -probability * probability.ln())
                .sum();
            let oracle_rank = rank_by_density(&capture, objects, |object| {
                f64::from(capture.contribution(query, object))
            });
            entropy += h;
            effective_blocks += h.exp();
            valid_queries += 1;

            for (&budget, accumulator) in budgets.iter().zip(&mut accumulators) {
                let (popularity_bytes, popularity_coverage) = coverage_at_budget(
                    &capture,
                    query,
                    objects,
                    popularity_rank,
                    budget,
                    truth_total,
                );
                let (oracle_bytes, oracle_coverage) =
                    coverage_at_budget(&capture, query, objects, &oracle_rank, budget, truth_total);
                accumulator.popularity_bytes += popularity_bytes;
                accumulator.popularity_coverage += popularity_coverage;
                accumulator.oracle_bytes += oracle_bytes;
                accumulator.oracle_coverage += oracle_coverage;
            }
        }

        if valid_queries == 0 {
            return Err(
                format!("layer {layer} has no query with positive contribution mass").into(),
            );
        }
        let count = valid_queries as f64;
        let budget_reports = budgets
            .iter()
            .zip(accumulators)
            .map(|(&budget, accumulator)| {
                let popularity_coverage = accumulator.popularity_coverage / count;
                let oracle_coverage = accumulator.oracle_coverage / count;
                BudgetCensus {
                    requested_byte_fraction: budget,
                    popularity_candidate_byte_fraction: accumulator.popularity_bytes / count,
                    popularity_contribution_coverage: popularity_coverage,
                    oracle_candidate_byte_fraction: accumulator.oracle_bytes / count,
                    oracle_contribution_coverage: oracle_coverage,
                    oracle_advantage: oracle_coverage - popularity_coverage,
                }
            })
            .collect();
        layers.push(LayerCensus {
            layer,
            objects: objects.len(),
            queries: valid_queries,
            mean_entropy_nats: entropy / count,
            mean_effective_blocks: effective_blocks / count,
            mean_effective_block_fraction: effective_blocks / count / objects.len() as f64,
            budgets: budget_reports,
        });
    }

    let report = CensusReport {
        schema: CENSUS_SCHEMA_V1,
        capture_schema: SCHEMA_V1,
        model: capture.manifest.model.clone(),
        contribution_engine: capture.manifest.contribution_engine.clone(),
        history_samples: history.len(),
        query_samples: queries.len(),
        query_regime: args.query_regime,
        layers,
    };
    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(path) => {
            std::fs::write(&path, format!("{json}\n"))?;
            println!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
