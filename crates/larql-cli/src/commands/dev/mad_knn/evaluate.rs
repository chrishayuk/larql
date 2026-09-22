use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use larql_vindex::index::hnsw::HnswLayer;
use ndarray::{Array1, Array2, ArrayView1};
use serde::Serialize;

use super::format::{Capture, SCHEMA_V1};
use super::{EvaluateArgs, SearchKind};

const REPORT_SCHEMA_V1: &str = "larql.mad-v3-knn.evaluation.v1";

#[derive(Debug, Serialize)]
pub struct EvaluationReport {
    schema: &'static str,
    capture_schema: &'static str,
    model: String,
    hidden_size: usize,
    history_samples: usize,
    query_samples: usize,
    query_regime: Option<String>,
    exclude_same_group: bool,
    search: SearchReport,
    layers: Vec<LayerReport>,
}

#[derive(Debug, Serialize)]
struct SearchReport {
    guarantee: &'static str,
    hnsw_m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
}

#[derive(Debug, Serialize)]
struct LayerReport {
    source_layer: usize,
    search_audit: Option<SearchAudit>,
    neighbour_purity: Vec<NeighbourReport>,
    future_contribution_entropy: Vec<EntropyReport>,
    future_addressability: Vec<FutureReport>,
}

#[derive(Debug, Serialize)]
struct EntropyReport {
    horizon: usize,
    target_layer: usize,
    objects: usize,
    queries: usize,
    mean_entropy_nats: f64,
    mean_effective_blocks: f64,
    mean_effective_block_fraction: f64,
}

#[derive(Debug, Serialize)]
struct SearchAudit {
    queries: usize,
    k: usize,
    mean_recall: f64,
    min_recall: f64,
}

#[derive(Debug, Serialize)]
struct NeighbourReport {
    k: usize,
    queries: usize,
    semantic_purity: f64,
    semantic_base_rate: f64,
    semantic_lift: Option<f64>,
    operation_purity: Option<f64>,
    operation_base_rate: Option<f64>,
    operation_lift: Option<f64>,
    wording_purity: f64,
    wording_base_rate: f64,
    wording_lift: Option<f64>,
}

#[derive(Debug, Serialize)]
struct FutureReport {
    horizon: usize,
    target_layer: usize,
    neighbours: usize,
    requested_byte_fraction: f64,
    queries: usize,
    knn_candidate_byte_fraction: f64,
    knn_contribution_coverage: f64,
    shuffle_candidate_byte_fraction: f64,
    shuffle_contribution_coverage: f64,
    popularity_candidate_byte_fraction: f64,
    popularity_contribution_coverage: f64,
    oracle_candidate_byte_fraction: f64,
    oracle_contribution_coverage: f64,
}

#[derive(Default)]
struct PurityAccumulator {
    queries: usize,
    semantic: f64,
    semantic_base: f64,
    operation: f64,
    operation_base: f64,
    operation_queries: usize,
    wording: f64,
    wording_base: f64,
}

struct FutureAccumulator {
    horizon: usize,
    target_layer: usize,
    neighbours: usize,
    budget: f64,
    queries: usize,
    knn_bytes: f64,
    knn_coverage: f64,
    shuffle_bytes: f64,
    shuffle_coverage: f64,
    popularity_bytes: f64,
    popularity_coverage: f64,
    oracle_bytes: f64,
    oracle_coverage: f64,
}

struct EntropyAccumulator {
    horizon: usize,
    target_layer: usize,
    objects: usize,
    queries: usize,
    entropy: f64,
    effective_blocks: f64,
}

pub fn run_evaluate(args: EvaluateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let neighbors = parse_usize_list(&args.neighbors, "neighbors", false)?;
    let budgets = parse_fraction_list(&args.byte_budgets, "byte-budgets")?;
    let horizons = parse_usize_list(&args.horizons, "horizons", false)?;
    let report = evaluate_capture(
        &args.capture,
        &neighbors,
        &budgets,
        &horizons,
        args.search,
        args.hnsw_m,
        args.ef_construction,
        args.ef_search,
        args.audit_queries,
        args.exclude_same_group,
        args.query_regime.as_deref(),
    )?;
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

#[allow(clippy::too_many_arguments)]
fn evaluate_capture(
    capture_dir: &Path,
    neighbors: &[usize],
    budgets: &[f64],
    horizons: &[usize],
    search_kind: SearchKind,
    hnsw_m: usize,
    ef_construction: usize,
    ef_search: usize,
    audit_queries: usize,
    exclude_same_group: bool,
    query_regime: Option<&str>,
) -> Result<EvaluationReport, String> {
    if hnsw_m < 2 {
        return Err("hnsw-m must be at least 2".into());
    }
    if ef_construction == 0 || ef_search == 0 {
        return Err("ef-construction and ef-search must be positive".into());
    }
    let capture = Capture::open(capture_dir)?;
    let history_samples: Vec<usize> = capture
        .manifest
        .samples
        .iter()
        .enumerate()
        .filter_map(|(i, sample)| (sample.split == "history").then_some(i))
        .collect();
    let query_samples: Vec<usize> = capture
        .manifest
        .samples
        .iter()
        .enumerate()
        .filter_map(|(i, sample)| {
            (sample.split == "query"
                && query_regime.is_none_or(|regime| sample.regime.as_deref() == Some(regime)))
            .then_some(i)
        })
        .collect();
    if query_samples.is_empty() {
        return Err(match query_regime {
            Some(regime) => format!("capture has no query rows for regime `{regime}`"),
            None => "capture has no query rows".to_string(),
        });
    }
    let max_k = *neighbors
        .iter()
        .max()
        .ok_or_else(|| "neighbors must not be empty".to_string())?;
    if history_samples.len() < max_k {
        return Err(format!(
            "largest neighbor count is {max_k}, but capture has only {} history rows",
            history_samples.len()
        ));
    }

    let popularity = popularity_rankings(&capture, &history_samples);
    let shuffle_maps: HashMap<usize, HashMap<usize, usize>> = capture
        .manifest
        .objects
        .iter()
        .map(|object| object.layer)
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|layer| (layer, shuffled_signature_samples(&history_samples, layer)))
        .collect();
    let mut layers = Vec::with_capacity(capture.manifest.residual_layers.len());
    for &source_layer in &capture.manifest.residual_layers {
        let history_matrix = normalized_history(&capture, &history_samples, source_layer)?;
        let index = match search_kind {
            SearchKind::Exact => SearchIndex::Exact,
            SearchKind::Hnsw => SearchIndex::Hnsw(HnswLayer::build(
                &history_matrix.view(),
                hnsw_m,
                ef_construction,
            )),
        };
        let search_audit = match &index {
            SearchIndex::Exact => None,
            SearchIndex::Hnsw(_) => Some(audit_search(
                &capture,
                &history_matrix,
                &index,
                &query_samples,
                source_layer,
                max_k,
                ef_search,
                audit_queries,
            )?),
        };

        let mut purity: Vec<PurityAccumulator> = neighbors
            .iter()
            .map(|_| PurityAccumulator::default())
            .collect();
        let mut future = Vec::new();
        let mut entropy = Vec::new();
        for &horizon in horizons {
            let Some(target_layer) = source_layer.checked_add(horizon) else {
                continue;
            };
            if capture.objects_at(target_layer).is_none() {
                continue;
            }
            entropy.push(EntropyAccumulator {
                horizon,
                target_layer,
                objects: capture
                    .objects_at(target_layer)
                    .expect("target existence checked")
                    .len(),
                queries: 0,
                entropy: 0.0,
                effective_blocks: 0.0,
            });
            for &k in neighbors {
                for &budget in budgets {
                    future.push(FutureAccumulator {
                        horizon,
                        target_layer,
                        neighbours: k,
                        budget,
                        queries: 0,
                        knn_bytes: 0.0,
                        knn_coverage: 0.0,
                        shuffle_bytes: 0.0,
                        shuffle_coverage: 0.0,
                        popularity_bytes: 0.0,
                        popularity_coverage: 0.0,
                        oracle_bytes: 0.0,
                        oracle_coverage: 0.0,
                    });
                }
            }
        }

        for &query_sample in &query_samples {
            let query = normalized(capture.residual(query_sample, source_layer)?)?;
            let query_meta = &capture.manifest.samples[query_sample];
            // Group exclusion happens after ANN traversal, so progressively
            // widen the candidate beam until K eligible rows exist. A fixed
            // oversample could silently under-fill queries from a large graph.
            let mut probe_k = max_k;
            let eligible: Vec<usize> = loop {
                let rows: Vec<usize> = index
                    .search(&history_matrix, &query, probe_k, ef_search)
                    .into_iter()
                    .map(|(row, _)| history_samples[row])
                    .filter(|&sample| {
                        !same_excluded_group(
                            query_meta,
                            &capture.manifest.samples[sample],
                            exclude_same_group,
                        )
                    })
                    .collect();
                if rows.len() >= max_k || probe_k == history_samples.len() {
                    break rows;
                }
                probe_k = probe_k.saturating_mul(2).min(history_samples.len());
            };
            let eligible_history: Vec<usize> = history_samples
                .iter()
                .copied()
                .filter(|&sample| {
                    !same_excluded_group(
                        query_meta,
                        &capture.manifest.samples[sample],
                        exclude_same_group,
                    )
                })
                .collect();
            if eligible_history.is_empty() {
                continue;
            }
            let semantic_base = eligible_history
                .iter()
                .filter(|&&sample| {
                    capture.manifest.samples[sample].semantic_id == query_meta.semantic_id
                })
                .count() as f64
                / eligible_history.len() as f64;
            let wording_base = eligible_history
                .iter()
                .filter(|&&sample| {
                    capture.manifest.samples[sample].wording_id == query_meta.wording_id
                })
                .count() as f64
                / eligible_history.len() as f64;
            let operation_base = query_meta.operation_id.as_ref().map(|operation| {
                eligible_history
                    .iter()
                    .filter(|&&sample| {
                        capture.manifest.samples[sample].operation_id.as_ref() == Some(operation)
                    })
                    .count() as f64
                    / eligible_history.len() as f64
            });

            for (slot, &k) in neighbors.iter().enumerate() {
                if eligible.len() < k {
                    continue;
                }
                let top = &eligible[..k];
                let semantic = top
                    .iter()
                    .filter(|&&sample| {
                        capture.manifest.samples[sample].semantic_id == query_meta.semantic_id
                    })
                    .count() as f64
                    / k as f64;
                let wording = top
                    .iter()
                    .filter(|&&sample| {
                        capture.manifest.samples[sample].wording_id == query_meta.wording_id
                    })
                    .count() as f64
                    / k as f64;
                let operation = query_meta.operation_id.as_ref().map(|operation| {
                    top.iter()
                        .filter(|&&sample| {
                            capture.manifest.samples[sample].operation_id.as_ref()
                                == Some(operation)
                        })
                        .count() as f64
                        / k as f64
                });
                purity[slot].queries += 1;
                purity[slot].semantic += semantic;
                purity[slot].semantic_base += semantic_base;
                purity[slot].wording += wording;
                purity[slot].wording_base += wording_base;
                if let (Some(operation), Some(base)) = (operation, operation_base) {
                    purity[slot].operation += operation;
                    purity[slot].operation_base += base;
                    purity[slot].operation_queries += 1;
                }
            }

            for acc in &mut entropy {
                let objects = capture
                    .objects_at(acc.target_layer)
                    .expect("entropy target layer exists");
                let total: f64 = objects
                    .iter()
                    .map(|&object| f64::from(capture.contribution(query_sample, object)))
                    .sum();
                if total <= f64::EPSILON {
                    continue;
                }
                let h = objects
                    .iter()
                    .map(|&object| f64::from(capture.contribution(query_sample, object)) / total)
                    .filter(|&probability| probability > 0.0)
                    .map(|probability| -probability * probability.ln())
                    .sum::<f64>();
                acc.queries += 1;
                acc.entropy += h;
                acc.effective_blocks += h.exp();
            }

            for acc in &mut future {
                if eligible.len() < acc.neighbours {
                    continue;
                }
                let objects = capture
                    .objects_at(acc.target_layer)
                    .expect("accumulator only created for present target layer");
                let truth_total: f64 = objects
                    .iter()
                    .map(|&object| f64::from(capture.contribution(query_sample, object)))
                    .sum();
                if truth_total <= f64::EPSILON {
                    continue;
                }
                let knn_rank = rank_by_density(&capture, objects, |object| {
                    eligible[..acc.neighbours]
                        .iter()
                        .map(|&sample| f64::from(capture.contribution(sample, object)))
                        .sum::<f64>()
                        / acc.neighbours as f64
                });
                let oracle_rank = rank_by_density(&capture, objects, |object| {
                    f64::from(capture.contribution(query_sample, object))
                });
                let shuffle_map = shuffle_maps
                    .get(&acc.target_layer)
                    .expect("every target layer has a shuffle map");
                let shuffle_rank = rank_by_density(&capture, objects, |object| {
                    eligible[..acc.neighbours]
                        .iter()
                        .map(|sample| {
                            let shuffled = shuffle_map
                                .get(sample)
                                .expect("every history row is shuffled");
                            f64::from(capture.contribution(*shuffled, object))
                        })
                        .sum::<f64>()
                        / acc.neighbours as f64
                });
                let popularity_rank = popularity
                    .get(&acc.target_layer)
                    .expect("every target layer has a popularity ranking");
                let (knn_bytes, knn_coverage) = coverage_at_budget(
                    &capture,
                    query_sample,
                    objects,
                    &knn_rank,
                    acc.budget,
                    truth_total,
                );
                let (pop_bytes, pop_coverage) = coverage_at_budget(
                    &capture,
                    query_sample,
                    objects,
                    popularity_rank,
                    acc.budget,
                    truth_total,
                );
                let (shuffle_bytes, shuffle_coverage) = coverage_at_budget(
                    &capture,
                    query_sample,
                    objects,
                    &shuffle_rank,
                    acc.budget,
                    truth_total,
                );
                let (oracle_bytes, oracle_coverage) = coverage_at_budget(
                    &capture,
                    query_sample,
                    objects,
                    &oracle_rank,
                    acc.budget,
                    truth_total,
                );
                acc.queries += 1;
                acc.knn_bytes += knn_bytes;
                acc.knn_coverage += knn_coverage;
                acc.shuffle_bytes += shuffle_bytes;
                acc.shuffle_coverage += shuffle_coverage;
                acc.popularity_bytes += pop_bytes;
                acc.popularity_coverage += pop_coverage;
                acc.oracle_bytes += oracle_bytes;
                acc.oracle_coverage += oracle_coverage;
            }
        }

        let neighbour_purity = neighbors
            .iter()
            .zip(purity)
            .map(|(&k, acc)| {
                let n = acc.queries.max(1) as f64;
                let semantic_purity = acc.semantic / n;
                let semantic_base_rate = acc.semantic_base / n;
                let wording_purity = acc.wording / n;
                let wording_base_rate = acc.wording_base / n;
                let (operation_purity, operation_base_rate, operation_lift) =
                    if acc.operation_queries == 0 {
                        (None, None, None)
                    } else {
                        let operation_n = acc.operation_queries as f64;
                        let value = acc.operation / operation_n;
                        let base = acc.operation_base / operation_n;
                        (Some(value), Some(base), safe_lift(value, base))
                    };
                NeighbourReport {
                    k,
                    queries: acc.queries,
                    semantic_purity,
                    semantic_base_rate,
                    semantic_lift: safe_lift(semantic_purity, semantic_base_rate),
                    operation_purity,
                    operation_base_rate,
                    operation_lift,
                    wording_purity,
                    wording_base_rate,
                    wording_lift: safe_lift(wording_purity, wording_base_rate),
                }
            })
            .collect();
        let future_addressability = future
            .into_iter()
            .map(|acc| {
                let n = acc.queries.max(1) as f64;
                FutureReport {
                    horizon: acc.horizon,
                    target_layer: acc.target_layer,
                    neighbours: acc.neighbours,
                    requested_byte_fraction: acc.budget,
                    queries: acc.queries,
                    knn_candidate_byte_fraction: acc.knn_bytes / n,
                    knn_contribution_coverage: acc.knn_coverage / n,
                    shuffle_candidate_byte_fraction: acc.shuffle_bytes / n,
                    shuffle_contribution_coverage: acc.shuffle_coverage / n,
                    popularity_candidate_byte_fraction: acc.popularity_bytes / n,
                    popularity_contribution_coverage: acc.popularity_coverage / n,
                    oracle_candidate_byte_fraction: acc.oracle_bytes / n,
                    oracle_contribution_coverage: acc.oracle_coverage / n,
                }
            })
            .collect();
        let future_contribution_entropy = entropy
            .into_iter()
            .map(|acc| {
                let n = acc.queries.max(1) as f64;
                let effective = acc.effective_blocks / n;
                EntropyReport {
                    horizon: acc.horizon,
                    target_layer: acc.target_layer,
                    objects: acc.objects,
                    queries: acc.queries,
                    mean_entropy_nats: acc.entropy / n,
                    mean_effective_blocks: effective,
                    mean_effective_block_fraction: effective / acc.objects.max(1) as f64,
                }
            })
            .collect();
        layers.push(LayerReport {
            source_layer,
            search_audit,
            neighbour_purity,
            future_contribution_entropy,
            future_addressability,
        });
    }

    Ok(EvaluationReport {
        schema: REPORT_SCHEMA_V1,
        capture_schema: SCHEMA_V1,
        model: capture.manifest.model.clone(),
        hidden_size: capture.manifest.hidden_size,
        history_samples: history_samples.len(),
        query_samples: query_samples.len(),
        query_regime: query_regime.map(str::to_string),
        exclude_same_group,
        search: match search_kind {
            SearchKind::Exact => SearchReport {
                guarantee: "exact-cosine",
                hnsw_m: None,
                ef_construction: None,
                ef_search: None,
            },
            SearchKind::Hnsw => SearchReport {
                guarantee: "approximate-hnsw-with-exact-rerank-and-sampled-recall-audit",
                hnsw_m: Some(hnsw_m),
                ef_construction: Some(ef_construction),
                ef_search: Some(ef_search),
            },
        },
        layers,
    })
}

enum SearchIndex {
    Exact,
    Hnsw(HnswLayer),
}

impl SearchIndex {
    fn search(
        &self,
        history: &Array2<f32>,
        query: &Array1<f32>,
        k: usize,
        ef_search: usize,
    ) -> Vec<(usize, f32)> {
        match self {
            Self::Exact => exact_search(history, query.view(), k),
            Self::Hnsw(index) => index.search(&history.view(), query, k, ef_search),
        }
    }
}

fn normalized_history(
    capture: &Capture,
    samples: &[usize],
    layer: usize,
) -> Result<Array2<f32>, String> {
    let mut values = Vec::with_capacity(samples.len() * capture.manifest.hidden_size);
    for &sample in samples {
        values.extend(normalized(capture.residual(sample, layer)?)?);
    }
    Array2::from_shape_vec((samples.len(), capture.manifest.hidden_size), values)
        .map_err(|e| format!("construct history matrix: {e}"))
}

fn normalized(mut row: Vec<f32>) -> Result<Array1<f32>, String> {
    let norm = row
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err("cannot normalize a zero or non-finite residual".into());
    }
    row.iter_mut()
        .for_each(|v| *v = (f64::from(*v) / norm) as f32);
    Ok(Array1::from_vec(row))
}

fn exact_search(history: &Array2<f32>, query: ArrayView1<'_, f32>, k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = history
        .outer_iter()
        .enumerate()
        .map(|(row, values)| (row, values.dot(&query)))
        .collect();
    scored.sort_unstable_by(|a, b| descending_f32(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k.min(scored.len()));
    scored
}

#[allow(clippy::too_many_arguments)]
fn audit_search(
    capture: &Capture,
    history: &Array2<f32>,
    index: &SearchIndex,
    query_samples: &[usize],
    layer: usize,
    k: usize,
    ef_search: usize,
    audit_queries: usize,
) -> Result<SearchAudit, String> {
    let mut recalls = Vec::new();
    for &sample in query_samples.iter().take(audit_queries) {
        let query = normalized(capture.residual(sample, layer)?)?;
        let approximate: HashSet<usize> = index
            .search(history, &query, k, ef_search)
            .into_iter()
            .map(|(row, _)| row)
            .collect();
        let exact = exact_search(history, query.view(), k);
        if exact.is_empty() {
            continue;
        }
        let overlap = exact
            .iter()
            .filter(|(row, _)| approximate.contains(row))
            .count();
        recalls.push(overlap as f64 / exact.len() as f64);
    }
    let mean = if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    };
    let min = recalls.iter().copied().reduce(f64::min).unwrap_or(0.0);
    Ok(SearchAudit {
        queries: recalls.len(),
        k,
        mean_recall: mean,
        min_recall: min,
    })
}

pub(super) fn popularity_rankings(
    capture: &Capture,
    history: &[usize],
) -> HashMap<usize, Vec<usize>> {
    let mut by_layer: HashMap<usize, Vec<usize>> = HashMap::new();
    for object in &capture.manifest.objects {
        by_layer.entry(object.layer).or_default();
    }
    for (&layer, ranking) in &mut by_layer {
        let objects = capture
            .objects_at(layer)
            .expect("layer came from object table");
        *ranking = rank_by_density(capture, objects, |object| {
            history
                .iter()
                .map(|&sample| f64::from(capture.contribution(sample, object)))
                .sum::<f64>()
                / history.len() as f64
        });
    }
    by_layer
}

/// Deterministic random single-cycle pairing. Every history residual keeps a
/// real signature from the same target layer, every signature is used exactly
/// once, and no row maps to itself when at least two rows exist.
fn shuffled_signature_samples(history: &[usize], layer: usize) -> HashMap<usize, usize> {
    let mut ordered = history.to_vec();
    ordered.sort_unstable_by_key(|&sample| {
        splitmix64((sample as u64) ^ (layer as u64).rotate_left(32) ^ 0x4d41_4456_334b_4e4e)
    });
    if ordered.len() < 2 {
        return ordered.iter().map(|&sample| (sample, sample)).collect();
    }
    ordered
        .iter()
        .copied()
        .zip(ordered.iter().copied().cycle().skip(1))
        .take(ordered.len())
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(super) fn rank_by_density(
    capture: &Capture,
    objects: &[usize],
    score: impl Fn(usize) -> f64,
) -> Vec<usize> {
    let mut ranked: Vec<(usize, f64)> = objects
        .iter()
        .map(|&object| {
            let density = score(object) / capture.manifest.objects[object].byte_count as f64;
            (object, density)
        })
        .collect();
    ranked.sort_unstable_by(|a, b| descending_f64(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(object, _)| object).collect()
}

pub(super) fn coverage_at_budget(
    capture: &Capture,
    query_sample: usize,
    objects: &[usize],
    ranking: &[usize],
    budget: f64,
    truth_total: f64,
) -> (f64, f64) {
    let total_bytes: u64 = objects
        .iter()
        .map(|&object| capture.manifest.objects[object].byte_count)
        .sum();
    let byte_limit = budget * total_bytes as f64;
    let mut used = 0u64;
    let mut covered = 0.0f64;
    for &object in ranking {
        let bytes = capture.manifest.objects[object].byte_count;
        if used as f64 + bytes as f64 > byte_limit {
            continue;
        }
        used += bytes;
        covered += f64::from(capture.contribution(query_sample, object));
    }
    (
        used as f64 / total_bytes.max(1) as f64,
        covered / truth_total,
    )
}

fn same_excluded_group(
    query: &super::format::SampleDescriptor,
    history: &super::format::SampleDescriptor,
    enabled: bool,
) -> bool {
    enabled && query.group_id.is_some() && query.group_id.as_deref() == history.group_id.as_deref()
}

fn safe_lift(value: f64, baseline: f64) -> Option<f64> {
    if baseline <= f64::EPSILON {
        None
    } else {
        Some(value / baseline)
    }
}

fn descending_f32(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn descending_f64(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn parse_usize_list(spec: &str, name: &str, allow_zero: bool) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    for raw in spec.split(',') {
        let value: usize = raw
            .trim()
            .parse()
            .map_err(|_| format!("--{name} contains invalid integer `{raw}`"))?;
        if !allow_zero && value == 0 {
            return Err(format!("--{name} values must be positive"));
        }
        values.push(value);
    }
    values.sort_unstable();
    values.dedup();
    if values.is_empty() {
        return Err(format!("--{name} must not be empty"));
    }
    Ok(values)
}

pub(super) fn parse_fraction_list(spec: &str, name: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for raw in spec.split(',') {
        let value: f64 = raw
            .trim()
            .parse()
            .map_err(|_| format!("--{name} contains invalid fraction `{raw}`"))?;
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(format!("--{name} values must be in (0, 1]"));
        }
        values.push(value);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    if values.is_empty() {
        return Err(format!("--{name} must not be empty"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::commands::dev::mad_knn::format::{
        CaptureManifest, ObjectDescriptor, SampleDescriptor, CONTRIBUTION_SEMANTICS_V1,
    };

    fn write_f32(path: &Path, values: &[f32]) {
        let mut file = std::fs::File::create(path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let manifest = CaptureManifest {
            schema: SCHEMA_V1.into(),
            model: "mini-glimmer".into(),
            hidden_size: 2,
            residual_layers: vec![0],
            contribution_semantics: CONTRIBUTION_SEMANTICS_V1.into(),
            contribution_engine: None,
            residuals_file: "residuals.f32".into(),
            contributions_file: "contributions.f32".into(),
            seams_file: None,
            seam_layers: Vec::new(),
            seam_kinds: Vec::new(),
            samples: vec![
                SampleDescriptor {
                    id: "h-a".into(),
                    semantic_id: "query-a".into(),
                    operation_id: Some("lookup".into()),
                    wording_id: "template-1".into(),
                    split: "history".into(),
                    group_id: Some("graph-a".into()),
                    regime: Some("a".into()),
                    alias_id: None,
                    layout_id: None,
                    token_count: Some(3),
                    expected_token_ids: None,
                    generated_token_ids: None,
                    answer_exact: None,
                    answer_token_accuracy: None,
                },
                SampleDescriptor {
                    id: "h-b".into(),
                    semantic_id: "query-b".into(),
                    operation_id: Some("lookup".into()),
                    wording_id: "template-1".into(),
                    split: "history".into(),
                    group_id: Some("graph-b".into()),
                    regime: Some("a".into()),
                    alias_id: None,
                    layout_id: None,
                    token_count: Some(3),
                    expected_token_ids: None,
                    generated_token_ids: None,
                    answer_exact: None,
                    answer_token_accuracy: None,
                },
                SampleDescriptor {
                    id: "q-a".into(),
                    semantic_id: "query-a".into(),
                    operation_id: Some("lookup".into()),
                    wording_id: "template-2".into(),
                    split: "query".into(),
                    group_id: Some("graph-a".into()),
                    regime: Some("a".into()),
                    alias_id: None,
                    layout_id: None,
                    token_count: Some(4),
                    expected_token_ids: None,
                    generated_token_ids: None,
                    answer_exact: None,
                    answer_token_accuracy: None,
                },
                SampleDescriptor {
                    id: "q-b".into(),
                    semantic_id: "query-b".into(),
                    operation_id: Some("lookup".into()),
                    wording_id: "template-3".into(),
                    split: "query".into(),
                    group_id: Some("graph-b".into()),
                    regime: Some("a".into()),
                    alias_id: None,
                    layout_id: None,
                    token_count: Some(4),
                    expected_token_ids: None,
                    generated_token_ids: None,
                    answer_exact: None,
                    answer_token_accuracy: None,
                },
            ],
            objects: vec![
                ObjectDescriptor {
                    id: "layer.1.block.0".into(),
                    layer: 1,
                    kind: "ffn_down_block".into(),
                    byte_count: 100,
                    operand: Some("down".into()),
                    channel_start: Some(0),
                    channel_end: Some(1),
                    block_channels: None,
                },
                ObjectDescriptor {
                    id: "layer.1.block.1".into(),
                    layer: 1,
                    kind: "ffn_down_block".into(),
                    byte_count: 100,
                    operand: Some("down".into()),
                    channel_start: Some(1),
                    channel_end: Some(2),
                    block_channels: None,
                },
            ],
        };
        std::fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        // [sample][layer][hidden]
        write_f32(
            &dir.path().join("residuals.f32"),
            &[1.0, 0.0, 0.0, 1.0, 0.9, 0.1, 0.1, 0.9],
        );
        // [sample][object]
        write_f32(
            &dir.path().join("contributions.f32"),
            &[10.0, 0.0, 0.0, 10.0, 8.0, 2.0, 1.0, 9.0],
        );
        dir
    }

    #[test]
    fn exact_oracle_finds_canonical_keys_and_future_objects() {
        let dir = fixture();
        let report = evaluate_capture(
            dir.path(),
            &[1],
            &[0.5],
            &[1],
            SearchKind::Exact,
            16,
            100,
            128,
            0,
            false,
            None,
        )
        .unwrap();
        let layer = &report.layers[0];
        let purity = &layer.neighbour_purity[0];
        assert_eq!(purity.queries, 2);
        assert_eq!(purity.semantic_purity, 1.0);
        assert_eq!(purity.wording_purity, 0.0);
        let future = &layer.future_addressability[0];
        assert_eq!(future.queries, 2);
        assert!((future.knn_candidate_byte_fraction - 0.5).abs() < 1e-12);
        assert!((future.knn_contribution_coverage - 0.85).abs() < 1e-6);
        assert!((future.shuffle_contribution_coverage - 0.15).abs() < 1e-6);
        assert!((future.oracle_contribution_coverage - 0.85).abs() < 1e-6);
        let entropy = &layer.future_contribution_entropy[0];
        assert_eq!(entropy.queries, 2);
        assert!(entropy.mean_effective_blocks > 1.0);
        assert!(entropy.mean_effective_blocks < 2.0);
    }

    #[test]
    fn truncated_plane_refuses_before_scoring() {
        let dir = fixture();
        std::fs::write(dir.path().join("contributions.f32"), [0u8; 4]).unwrap();
        let error = Capture::open(dir.path()).err().unwrap();
        assert!(error.contains("manifest requires exactly"), "{error}");
    }

    #[test]
    fn list_parsers_reject_ambiguous_zero_and_out_of_range_budget() {
        assert!(parse_usize_list("0,4", "neighbors", false).is_err());
        assert!(parse_fraction_list("0.1,1.1", "byte-budgets").is_err());
    }

    #[test]
    fn shuffle_is_a_derangement_and_preserves_every_signature_once() {
        let history = vec![2, 5, 7, 11, 13];
        let shuffled = shuffled_signature_samples(&history, 32);
        assert!(history
            .iter()
            .all(|sample| shuffled.get(sample) != Some(sample)));
        let mut values: Vec<usize> = shuffled.values().copied().collect();
        values.sort_unstable();
        assert_eq!(values, history);
    }
}
