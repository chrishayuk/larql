//! Pure helpers for the HTTP fan-out path. The actual reqwest calls live
//! in `main.rs`; this module owns:
//!   * `resolve_static_only` — static shard route lookup for every layer
//!   * `group_layers_by_url` — partition `(layer, url)` resolutions by shard
//!   * `build_subrequest_body` — set `layer`/`layers` for a per-shard body
//!   * `merge_shard_responses` — fold a list of shard JSON responses into
//!     the unified `{results, latency_ms}` envelope
//!   * `unique_candidate_urls` — dedup grid + static URLs for `/v1/stats`
//!
//! Each is exercised by unit tests in this file.

use std::collections::HashMap;

use serde_json::Value;

use crate::shards::{find_shard_for_layer, Shard};

/// Resolve every entry in `layers` against the static shard list. Returns
/// `Ok(layer → url)` or `Err(first uncovered layer)`. Used as the static-
/// only fallback in `AppState::resolve_all`.
pub fn resolve_static_only(
    shards: &[Shard],
    layers: &[usize],
) -> Result<HashMap<usize, String>, usize> {
    let mut out = HashMap::with_capacity(layers.len());
    for &layer in layers {
        match find_shard_for_layer(shards, layer) {
            Some(s) => {
                out.insert(layer, s.url.clone());
            }
            None => return Err(layer),
        }
    }
    Ok(out)
}

/// Partition a `(layer → url)` resolution map into a `(url → [layer])`
/// inverse so each shard receives one combined request instead of N
/// independent ones. Order within each group is not stable across calls
/// (HashMap iteration); the caller sorts the final results by layer.
pub fn group_layers_by_url(layer_urls: &HashMap<usize, String>) -> HashMap<String, Vec<usize>> {
    let mut by_url: HashMap<String, Vec<usize>> = HashMap::new();
    for (&layer, url) in layer_urls {
        by_url.entry(url.clone()).or_default().push(layer);
    }
    by_url
}

/// Build the JSON body for a single shard's portion of a fan-out request.
/// Mutates `template` in place by setting `layer` (single-layer case) or
/// `layers` (multi-layer case) and removing the other field, so the shard
/// sees exactly one of the two.
pub fn build_subrequest_body(template: &Value, shard_layers: &[usize]) -> Value {
    let mut body = template.clone();
    let obj = body
        .as_object_mut()
        .expect("template must be a JSON object");
    if shard_layers.len() == 1 {
        obj.insert("layer".into(), Value::from(shard_layers[0]));
        obj.remove("layers");
    } else {
        obj.insert(
            "layers".into(),
            Value::Array(shard_layers.iter().map(|&l| Value::from(l)).collect()),
        );
        obj.remove("layer");
    }
    body
}

/// Merge a list of per-shard JSON responses into the unified
/// `{"results": [...], "latency_ms": max}` envelope. Result rows are
/// sorted by `layer` so the caller observes deterministic order.
pub fn merge_shard_responses(responses: &[Value]) -> Value {
    let mut all_results: Vec<Value> = Vec::new();
    let mut max_latency: f64 = 0.0;
    for resp in responses {
        if let Some(arr) = resp.get("results").and_then(|v| v.as_array()) {
            all_results.extend(arr.iter().cloned());
        } else if resp.get("layer").is_some() {
            all_results.push(resp.clone());
        }
        if let Some(ms) = resp.get("latency_ms").and_then(|v| v.as_f64()) {
            if ms > max_latency {
                max_latency = ms;
            }
        }
    }
    all_results.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(0));
    serde_json::json!({
        "results": all_results,
        "latency_ms": round_to_tenth(max_latency),
    })
}

fn round_to_tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// ADR-0021 — outcome of a (possibly hedged) sub-request. The caller
/// uses these flags to bump the two ADR-0021 counters without having
/// to pass metrics handles down into the helper.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HedgeOutcome {
    /// True when the secondary replica was actually dispatched (the
    /// primary's reply didn't arrive within `hedge_after`).
    pub fired: bool,
    /// True when the secondary's reply beat the primary's. Implies
    /// `fired`. Used to compute the wins/fires ratio in metrics.
    pub won: bool,
}

/// ADR-0021 — race a primary HTTP POST against a delayed secondary.
///
/// Sends `body` to `primary_url`. If the primary hasn't responded
/// within `hedge_after`, dispatches the same request to
/// `secondary_url` and returns whichever response arrives first.
/// When `secondary_url` is `None` or `hedge_after` is `None`, behaves
/// exactly like a single POST to `primary_url` (no hedging).
///
/// `target_path` is appended to both URLs (e.g. `/v1/walk-ffn`). The
/// returned `Value` is the parsed JSON body of the winning response.
/// `HedgeOutcome` lets the caller bump the right metrics.
pub async fn hedged_post_json(
    client: &reqwest::Client,
    primary_url: &str,
    secondary_url: Option<&str>,
    hedge_after: Option<std::time::Duration>,
    target_path: &str,
    body: &serde_json::Value,
) -> (Result<serde_json::Value, String>, HedgeOutcome) {
    let primary_target = format!("{primary_url}{target_path}");
    let primary_fut = async {
        client
            .post(&primary_target)
            .json(body)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| e.to_string())
    };

    let (Some(secondary_url), Some(hedge_after)) = (secondary_url, hedge_after) else {
        return (primary_fut.await, HedgeOutcome::default());
    };

    // Primary first; if it lands before the deadline we never hedge.
    tokio::pin!(primary_fut);
    let sleeper = tokio::time::sleep(hedge_after);
    tokio::pin!(sleeper);

    tokio::select! {
        biased;
        result = &mut primary_fut => {
            return (result, HedgeOutcome::default());
        }
        _ = &mut sleeper => {
            // Hedge: race primary vs newly-dispatched secondary.
        }
    }

    let secondary_target = format!("{secondary_url}{target_path}");
    let secondary_fut = async {
        client
            .post(&secondary_target)
            .json(body)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| e.to_string())
    };
    tokio::pin!(secondary_fut);

    tokio::select! {
        biased;
        result = &mut primary_fut => {
            (result, HedgeOutcome { fired: true, won: false })
        }
        result = &mut secondary_fut => {
            (result, HedgeOutcome { fired: true, won: true })
        }
    }
}

/// Build a deduped list of candidate shard URLs from the grid + static
/// pool. Grid URLs come first so a healthy grid is queried before any
/// stale static config.
pub fn unique_candidate_urls(grid_urls: Vec<String>, static_shards: &[Shard]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(grid_urls.len() + static_shards.len());
    for url in grid_urls {
        if !out.contains(&url) {
            out.push(url);
        }
    }
    for s in static_shards {
        if !out.contains(&s.url) {
            out.push(s.url.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shards::parse_shards;
    use serde_json::json;

    // ── resolve_static_only ──────────────────────────────────────────────

    #[test]
    fn resolve_static_only_returns_url_per_layer() {
        let shards = parse_shards("0-4=http://a,5-9=http://b").unwrap();
        let map = resolve_static_only(&shards, &[0, 5, 9]).unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map[&0], "http://a");
        assert_eq!(map[&5], "http://b");
        assert_eq!(map[&9], "http://b");
    }

    #[test]
    fn resolve_static_only_returns_err_on_first_uncovered_layer() {
        let shards = parse_shards("0-4=http://a").unwrap();
        let err = resolve_static_only(&shards, &[0, 1, 7]).unwrap_err();
        assert_eq!(err, 7);
    }

    #[test]
    fn resolve_static_only_empty_request_returns_empty_map() {
        let shards = parse_shards("0-4=http://a").unwrap();
        let map = resolve_static_only(&shards, &[]).unwrap();
        assert!(map.is_empty());
    }

    // ── group_layers_by_url ──────────────────────────────────────────────

    #[test]
    fn group_layers_by_url_collapses_per_shard() {
        let mut input = HashMap::new();
        input.insert(0, "http://a".to_string());
        input.insert(1, "http://a".to_string());
        input.insert(2, "http://b".to_string());
        let grouped = group_layers_by_url(&input);
        assert_eq!(grouped.len(), 2);
        let mut a = grouped["http://a"].clone();
        a.sort();
        assert_eq!(a, vec![0, 1]);
        assert_eq!(grouped["http://b"], vec![2]);
    }

    #[test]
    fn group_layers_by_url_empty_input_yields_empty_output() {
        let grouped = group_layers_by_url(&HashMap::new());
        assert!(grouped.is_empty());
    }

    // ── build_subrequest_body ────────────────────────────────────────────

    #[test]
    fn build_body_single_layer_sets_layer_field() {
        let tmpl = json!({"prompt": "hi", "layer": 99, "layers": [99]});
        let body = build_subrequest_body(&tmpl, &[3]);
        assert_eq!(body["layer"], 3);
        assert!(body.get("layers").is_none(), "layers must be removed");
        assert_eq!(body["prompt"], "hi");
    }

    #[test]
    fn build_body_multi_layer_sets_layers_array() {
        let tmpl = json!({"prompt": "hi", "layer": 99, "layers": [99]});
        let body = build_subrequest_body(&tmpl, &[1, 3, 5]);
        assert_eq!(body["layers"], json!([1, 3, 5]));
        assert!(body.get("layer").is_none(), "layer must be removed");
    }

    #[test]
    #[should_panic(expected = "template must be a JSON object")]
    fn build_body_panics_on_non_object_template() {
        let tmpl = json!([1, 2, 3]);
        let _ = build_subrequest_body(&tmpl, &[0]);
    }

    // ── merge_shard_responses ────────────────────────────────────────────

    #[test]
    fn merge_combines_results_arrays_and_takes_max_latency() {
        let a = json!({
            "results": [{"layer": 0, "v": 1}, {"layer": 1, "v": 2}],
            "latency_ms": 10.0,
        });
        let b = json!({
            "results": [{"layer": 2, "v": 3}],
            "latency_ms": 15.2,
        });
        let merged = merge_shard_responses(&[a, b]);
        let results = merged["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        // Sorted by layer ascending.
        assert_eq!(results[0]["layer"], 0);
        assert_eq!(results[1]["layer"], 1);
        assert_eq!(results[2]["layer"], 2);
        assert!((merged["latency_ms"].as_f64().unwrap() - 15.2).abs() < 1e-9);
    }

    #[test]
    fn merge_handles_responses_with_no_results_array() {
        // Some shard responses are a single layer's row, not a results array.
        let a = json!({"layer": 5, "value": "x"});
        let b = json!({"results": [{"layer": 1, "v": 1}]});
        let merged = merge_shard_responses(&[a, b]);
        let results = merged["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["layer"], 1);
        assert_eq!(results[1]["layer"], 5);
    }

    #[test]
    fn merge_drops_responses_without_layer_or_results() {
        let a = json!({"unrelated": "junk"});
        let b = json!({"results": [{"layer": 0, "v": 1}]});
        let merged = merge_shard_responses(&[a, b]);
        let results = merged["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn merge_empty_returns_empty_results_and_zero_latency() {
        let merged = merge_shard_responses(&[]);
        assert_eq!(merged["results"].as_array().unwrap().len(), 0);
        assert_eq!(merged["latency_ms"], 0.0);
    }

    #[test]
    fn merge_rounds_latency_to_one_decimal() {
        let a = json!({"latency_ms": 12.3456});
        let merged = merge_shard_responses(&[a]);
        assert!((merged["latency_ms"].as_f64().unwrap() - 12.3).abs() < 1e-9);
    }

    // ── unique_candidate_urls ────────────────────────────────────────────

    #[test]
    fn unique_candidates_grid_urls_come_first() {
        let shards = parse_shards("0-1=http://b,2-3=http://c").unwrap();
        let grid = vec!["http://a".to_string(), "http://b".to_string()];
        let out = unique_candidate_urls(grid, &shards);
        assert_eq!(out, vec!["http://a", "http://b", "http://c"]);
    }

    #[test]
    fn unique_candidates_dedup_within_grid_and_against_static() {
        let shards = parse_shards("0-1=http://a").unwrap();
        let grid = vec![
            "http://a".to_string(),
            "http://a".to_string(),
            "http://b".to_string(),
        ];
        let out = unique_candidate_urls(grid, &shards);
        assert_eq!(out, vec!["http://a", "http://b"]);
    }

    #[test]
    fn unique_candidates_empty_grid_returns_static_only() {
        let shards = parse_shards("0-1=http://a,2-3=http://b").unwrap();
        let out = unique_candidate_urls(Vec::new(), &shards);
        assert_eq!(out, vec!["http://a", "http://b"]);
    }

    #[test]
    fn unique_candidates_empty_inputs_yields_empty_output() {
        let out = unique_candidate_urls(Vec::new(), &[]);
        assert!(out.is_empty());
    }

    // ── round_to_tenth ───────────────────────────────────────────────────

    #[test]
    fn round_to_tenth_examples() {
        assert!((round_to_tenth(1.249) - 1.2).abs() < 1e-9);
        assert!((round_to_tenth(1.25) - 1.3).abs() < 1e-9);
        assert_eq!(round_to_tenth(0.0), 0.0);
    }
}
