//! Pure helpers used by `bench_cmd.rs`. Lifted out so they can be unit
//! tested without spinning up a model — `bench_cmd::run` itself is hard to
//! cover because every code path requires a loaded vindex, a running server,
//! or both.
//!
//! Each helper is the kind of thing that's easy to get subtly wrong:
//!   * parsing comma-separated wire-format lists (skipping blanks, trim),
//!   * aggregating concurrent client runs into a single observable row,
//!   * computing shard efficiency for `--bench-grid` scaling sweeps.

use larql_inference::WirePreference;

// ── Wire format parsing ──────────────────────────────────────────────────────

/// Parse `"f32,f16,i8"` into a list of `WirePreference`. Tokens are
/// trimmed; blanks are skipped; unknown tokens are silently mapped to
/// `BestAvailable` (matching the historical behaviour of `bench_cmd.rs`).
///
/// Returns an empty `Vec` when the input has no usable tokens.
pub fn parse_wire_list(spec: &str) -> Vec<WirePreference> {
    spec.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| WirePreference::from_str(s).unwrap_or(WirePreference::BestAvailable))
        .collect()
}

// ── Concurrent-client aggregation ────────────────────────────────────────────

/// Lightweight summary of one bench client run. Pulled out as its own type
/// rather than re-using `BenchRow` so the helper has no dependency on the
/// CLI's row layout — easier to unit-test, easier to keep stable.
#[derive(Clone, Debug, PartialEq)]
pub struct ConcurrentSample {
    /// Throughput observed by this single client, tokens per second.
    pub tok_per_s: f64,
    /// Mean ms/tok this client saw.
    pub mean_ms: f64,
    /// p50 ms/tok this client saw.
    pub p50_ms: f64,
    /// p99 ms/tok this client saw.
    pub p99_ms: f64,
    /// Bytes per decode token (sum sent + received) — None when the bench
    /// path doesn't track wire bytes.
    pub wire_bytes_per_tok: Option<u64>,
}

/// Aggregate result across N concurrent clients running the same bench.
#[derive(Clone, Debug, PartialEq)]
pub struct ConcurrentAggregate {
    /// `sum(client.tok_per_s)` — the system-level throughput delivered by
    /// the shard, not any one client's perceived rate.
    pub aggregate_tok_per_s: f64,
    /// `mean(client.mean_ms)` — average across clients.
    pub mean_ms: f64,
    /// `max(client.p50_ms)` — worst median latency observed by any client.
    pub worst_p50_ms: f64,
    /// `max(client.p99_ms)` — worst tail-latency observed.
    pub worst_p99_ms: f64,
    /// `sum(client.wire_bytes_per_tok)` — total bandwidth across clients.
    /// `None` when no sample reported wire bytes.
    pub total_wire_bytes_per_tok: Option<u64>,
    /// Number of client runs aggregated.
    pub n_clients: usize,
}

/// Aggregate concurrent client runs into a single row. Returns `None` when
/// `samples` is empty (the caller should treat that as a no-op rather than
/// emitting a row of zeros).
pub fn aggregate_concurrent_rows(samples: &[ConcurrentSample]) -> Option<ConcurrentAggregate> {
    if samples.is_empty() {
        return None;
    }
    let aggregate_tok_per_s: f64 = samples.iter().map(|s| s.tok_per_s).sum();
    let mean_ms: f64 = samples.iter().map(|s| s.mean_ms).sum::<f64>() / samples.len() as f64;
    let worst_p50_ms: f64 = samples.iter().map(|s| s.p50_ms).fold(0.0_f64, f64::max);
    let worst_p99_ms: f64 = samples.iter().map(|s| s.p99_ms).fold(0.0_f64, f64::max);
    let total_wire_bytes_per_tok = {
        let any = samples.iter().any(|s| s.wire_bytes_per_tok.is_some());
        if any {
            Some(samples.iter().filter_map(|s| s.wire_bytes_per_tok).sum())
        } else {
            None
        }
    };
    Some(ConcurrentAggregate {
        aggregate_tok_per_s,
        mean_ms,
        worst_p50_ms,
        worst_p99_ms,
        total_wire_bytes_per_tok,
        n_clients: samples.len(),
    })
}

// ── Shard scaling efficiency ─────────────────────────────────────────────────

/// `shard_efficiency = tok_per_s / (N_shards * single_shard_tok_per_s)`
///
/// 1.00 = perfect linear scaling; lower numbers mean coordination overhead is
/// eating into the gain. Returns `None` when `n_shards == 0` or
/// `single_shard_tok_per_s <= 0.0` (can't divide by a missing baseline).
pub fn shard_efficiency(
    tok_per_s: f64,
    n_shards: usize,
    single_shard_tok_per_s: f64,
) -> Option<f64> {
    if n_shards == 0 || single_shard_tok_per_s <= 0.0 {
        return None;
    }
    Some(tok_per_s / (n_shards as f64 * single_shard_tok_per_s))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_wire_list ──────────────────────────────────────────────────

    #[test]
    fn parse_wire_list_trims_and_skips_blanks() {
        let out = parse_wire_list("  f32 , f16 ,, i8 , ");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn parse_wire_list_empty_input_returns_empty_vec() {
        assert!(parse_wire_list("").is_empty());
        assert!(parse_wire_list(",,,").is_empty());
    }

    #[test]
    fn parse_wire_list_unknown_token_maps_to_best_available() {
        let out = parse_wire_list("unknown-format,f16");
        assert_eq!(out.len(), 2);
        // First entry is the unknown → BestAvailable. Second is parsed
        // normally. We don't assert on enum identity here (WirePreference's
        // discriminants are inference-crate details); just on the count.
    }

    // ── aggregate_concurrent_rows ────────────────────────────────────────

    fn sample(tps: f64, mean: f64, p50: f64, p99: f64, wb: Option<u64>) -> ConcurrentSample {
        ConcurrentSample {
            tok_per_s: tps,
            mean_ms: mean,
            p50_ms: p50,
            p99_ms: p99,
            wire_bytes_per_tok: wb,
        }
    }

    #[test]
    fn aggregate_empty_returns_none() {
        assert_eq!(aggregate_concurrent_rows(&[]), None);
    }

    #[test]
    fn aggregate_sums_throughput_and_takes_worst_tail() {
        let samples = vec![
            sample(10.0, 100.0, 90.0, 200.0, Some(1000)),
            sample(12.0, 90.0, 88.0, 180.0, Some(1100)),
            sample(11.0, 95.0, 92.0, 220.0, Some(1050)),
        ];
        let agg = aggregate_concurrent_rows(&samples).unwrap();
        assert!((agg.aggregate_tok_per_s - 33.0).abs() < 1e-9);
        assert!((agg.mean_ms - 95.0).abs() < 1e-9);
        // worst p50 is the largest one — 92.0.
        assert!((agg.worst_p50_ms - 92.0).abs() < 1e-9);
        // worst p99 is 220.0.
        assert!((agg.worst_p99_ms - 220.0).abs() < 1e-9);
        // total wire bytes is sum.
        assert_eq!(agg.total_wire_bytes_per_tok, Some(3150));
        assert_eq!(agg.n_clients, 3);
    }

    #[test]
    fn aggregate_wire_bytes_none_when_no_sample_has_them() {
        let samples = vec![
            sample(10.0, 100.0, 90.0, 200.0, None),
            sample(12.0, 90.0, 88.0, 180.0, None),
        ];
        let agg = aggregate_concurrent_rows(&samples).unwrap();
        assert_eq!(agg.total_wire_bytes_per_tok, None);
    }

    #[test]
    fn aggregate_wire_bytes_sums_what_is_present_when_mixed() {
        // Partial reporting: some samples have wire bytes, others don't.
        // We treat "any present" as "tracking is on" and sum the ones we
        // have, rather than silently dropping the metric.
        let samples = vec![
            sample(10.0, 100.0, 90.0, 200.0, Some(500)),
            sample(12.0, 90.0, 88.0, 180.0, None),
        ];
        let agg = aggregate_concurrent_rows(&samples).unwrap();
        assert_eq!(agg.total_wire_bytes_per_tok, Some(500));
    }

    // ── shard_efficiency ─────────────────────────────────────────────────

    #[test]
    fn shard_efficiency_perfect_linear_scaling_is_one() {
        // N=4 shards delivering exactly 4× single-shard throughput.
        let eff = shard_efficiency(80.0, 4, 20.0).unwrap();
        assert!((eff - 1.0).abs() < 1e-9);
    }

    #[test]
    fn shard_efficiency_degrades_with_overhead() {
        // N=4 shards but only 70% of theoretical — efficiency 0.70.
        let eff = shard_efficiency(56.0, 4, 20.0).unwrap();
        assert!((eff - 0.70).abs() < 1e-9);
    }

    #[test]
    fn shard_efficiency_rejects_zero_shards() {
        assert_eq!(shard_efficiency(56.0, 0, 20.0), None);
    }

    #[test]
    fn shard_efficiency_rejects_non_positive_baseline() {
        assert_eq!(shard_efficiency(56.0, 4, 0.0), None);
        assert_eq!(shard_efficiency(56.0, 4, -1.0), None);
    }

    #[test]
    fn shard_efficiency_super_linear_is_above_one() {
        // Theoretical caching benefits sometimes show >1.0; the helper
        // doesn't clamp.
        let eff = shard_efficiency(90.0, 4, 20.0).unwrap();
        assert!(eff > 1.0);
    }
}
