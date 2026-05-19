//! Pure helpers for the remote-FFN bench path. The I/O-heavy bench
//! execution (`run_remote_ffn_bench`, `run_concurrent_ffn`) lives in
//! `remote_ffn_runtime.rs`; this file holds:
//!   * `combine_concurrent_rows` — fold per-client rows into one aggregate row
//!   * `format_ffn_backend_label` — table-label composer
//!   * `FfnSummary` + `summarize_ffn_result` — decode-result post-processing
//!   * `compute_wire_bytes_per_tok` — wire counter averaging
//!
//! All are exercised by tests in this file.

use super::helpers::{aggregate_concurrent_rows, ConcurrentSample};
use super::row::{compute_percentiles, BenchRow};

/// Fold per-client `BenchRow`s into a single aggregate row. Latency fields
/// use the worst observed value across clients (tail latency is the right
/// thing to report under load); throughput is summed; wire bytes summed
/// when reported. The result row preserves the first client's `backend`
/// and `prefill_ms` (prefill is per-process initialization, not
/// per-client load).
pub(super) fn combine_concurrent_rows(rows: Vec<BenchRow>, n_clients: usize) -> BenchRow {
    debug_assert!(!rows.is_empty());
    let samples: Vec<ConcurrentSample> = rows
        .iter()
        .map(|r| ConcurrentSample {
            tok_per_s: r.tok_per_s,
            mean_ms: r.avg_decode_ms,
            p50_ms: r.p50_ms,
            p99_ms: r.p99_ms,
            wire_bytes_per_tok: r.wire_bytes_per_tok,
        })
        .collect();
    let agg = aggregate_concurrent_rows(&samples).expect("rows is non-empty by debug_assert");
    let first = &rows[0];
    let ffn_rtt_ms = rows
        .iter()
        .filter_map(|r| r.ffn_rtt_ms)
        .fold(0.0_f64, f64::max);
    let attn_ms = rows
        .iter()
        .filter_map(|r| r.attn_ms)
        .fold(0.0_f64, f64::max);
    let ffn_rtt_ms = if ffn_rtt_ms > 0.0 {
        Some(ffn_rtt_ms)
    } else {
        None
    };
    let attn_ms = if attn_ms > 0.0 { Some(attn_ms) } else { None };
    let total_steps: usize = rows.iter().map(|r| r.n_steps).sum();
    BenchRow {
        backend: format!("{} (×{n_clients} concurrent)", first.backend),
        prefill_ms: first.prefill_ms,
        avg_decode_ms: agg.mean_ms,
        p50_ms: agg.worst_p50_ms,
        p99_ms: agg.worst_p99_ms,
        tok_per_s: agg.aggregate_tok_per_s,
        stages: None,
        ffn_rtt_ms,
        attn_ms,
        wire_bytes_per_tok: agg.total_wire_bytes_per_tok,
        shard_efficiency: None,
        n_steps: total_steps,
        note: format!("concurrent={n_clients} | {}", first.note),
    }
}

/// Compose the `"remote-ffn-<mode>[ wire] (<url>)"` backend label.
pub(super) fn format_ffn_backend_label(
    is_batch: bool,
    wire_pref: larql_inference::WirePreference,
    ffn_url: &str,
) -> String {
    let wire_label = match wire_pref {
        larql_inference::WirePreference::BestAvailable => String::new(),
        _ => format!(" [{}]", wire_pref.label()),
    };
    format!(
        "remote-ffn-{}{} ({})",
        if is_batch { "batch" } else { "stream" },
        wire_label,
        ffn_url
    )
}

/// Bench result summary. Folded out of `run_remote_ffn_bench` so the
/// post-result computation (warmup trim, percentile, attn fallback,
/// early-stop note) can be unit-tested without a running shard.
pub(super) struct FfnSummary {
    pub avg_decode_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tok_per_s: f64,
    pub ffn_rtt_ms: Option<f64>,
    pub attn_ms: Option<f64>,
    pub n_steps: usize,
    pub note: String,
}

/// Trim warmup, percentile, derive `attn_ms = avg_decode - avg_ffn_rtt`,
/// and format the early-stop note when `n_measured < target_tokens`.
pub(super) fn summarize_ffn_result(
    decode_ms: &[f64],
    ffn_rtt_ms: &[f64],
    warmup: usize,
    target_tokens: usize,
) -> FfnSummary {
    let n_warm = warmup.min(decode_ms.len());
    let measured = &decode_ms[n_warm..];
    let measured_ffn = &ffn_rtt_ms[n_warm.min(ffn_rtt_ms.len())..];
    let n = measured.len();

    let (avg, p50, p99, tps, ffn, attn) = if n == 0 {
        (0.0, 0.0, 0.0, 0.0, None, None)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        let avg_ffn = if measured_ffn.len() == n {
            Some(measured_ffn.iter().sum::<f64>() / n as f64)
        } else {
            None
        };
        let avg_attn = avg_ffn.map(|f| (avg - f).max(0.0));
        (avg, p50, p99, 1000.0 / avg, avg_ffn, avg_attn)
    };

    let note = if n < target_tokens {
        format!("early stop @{}/{}", n, target_tokens)
    } else {
        String::new()
    };

    FfnSummary {
        avg_decode_ms: avg,
        p50_ms: p50,
        p99_ms: p99,
        tok_per_s: tps,
        ffn_rtt_ms: ffn,
        attn_ms: attn,
        n_steps: n,
        note,
    }
}

/// Average wire bytes per measured decode step. Returns `None` when
/// `n == 0` (no measured tokens → division would be undefined).
pub(super) fn compute_wire_bytes_per_tok(total_bytes: u64, n: usize) -> Option<u64> {
    if n == 0 {
        None
    } else {
        Some(total_bytes / n as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        backend: &str,
        tok: f64,
        mean: f64,
        p50: f64,
        p99: f64,
        ffn: Option<f64>,
        attn: Option<f64>,
        wb: Option<u64>,
        n_steps: usize,
    ) -> BenchRow {
        BenchRow {
            backend: backend.to_string(),
            prefill_ms: 1.5,
            avg_decode_ms: mean,
            p50_ms: p50,
            p99_ms: p99,
            tok_per_s: tok,
            stages: None,
            ffn_rtt_ms: ffn,
            attn_ms: attn,
            wire_bytes_per_tok: wb,
            shard_efficiency: None,
            n_steps,
            note: "ok".into(),
        }
    }

    // ── combine_concurrent_rows ──────────────────────────────────────────

    #[test]
    fn combine_concurrent_aggregates_throughput_and_takes_worst_tail() {
        let rows = vec![
            row(
                "remote-ffn (x)",
                10.0,
                100.0,
                90.0,
                200.0,
                Some(50.0),
                Some(50.0),
                Some(1000),
                5,
            ),
            row(
                "remote-ffn (x)",
                12.0,
                90.0,
                88.0,
                180.0,
                Some(45.0),
                Some(45.0),
                Some(1100),
                5,
            ),
            row(
                "remote-ffn (x)",
                11.0,
                95.0,
                92.0,
                220.0,
                Some(48.0),
                Some(47.0),
                Some(1050),
                5,
            ),
        ];
        let agg = combine_concurrent_rows(rows, 3);
        assert!(agg.backend.contains("×3 concurrent"));
        assert!(agg.note.starts_with("concurrent=3"));
        assert!((agg.tok_per_s - 33.0).abs() < 1e-9);
        assert!((agg.p99_ms - 220.0).abs() < 1e-9);
        assert_eq!(agg.wire_bytes_per_tok, Some(3150));
        assert_eq!(agg.n_steps, 15);
        assert!((agg.prefill_ms - 1.5).abs() < 1e-9);
        assert_eq!(agg.ffn_rtt_ms, Some(50.0));
        assert_eq!(agg.attn_ms, Some(50.0));
    }

    #[test]
    fn combine_concurrent_handles_missing_breakdowns() {
        let rows = vec![
            row("x", 5.0, 100.0, 90.0, 200.0, None, None, None, 1),
            row("x", 6.0, 100.0, 90.0, 200.0, None, None, None, 1),
        ];
        let agg = combine_concurrent_rows(rows, 2);
        assert!(agg.ffn_rtt_ms.is_none());
        assert!(agg.attn_ms.is_none());
        assert!(agg.wire_bytes_per_tok.is_none());
    }

    // ── format_ffn_backend_label ─────────────────────────────────────────

    #[test]
    fn label_uses_stream_or_batch_and_url() {
        let lbl = format_ffn_backend_label(
            false,
            larql_inference::WirePreference::BestAvailable,
            "http://a:8080",
        );
        assert!(lbl.starts_with("remote-ffn-stream"));
        assert!(lbl.contains("http://a:8080"));
        // BestAvailable hides the wire tag.
        assert!(!lbl.contains("["));

        let lbl_batch =
            format_ffn_backend_label(true, larql_inference::WirePreference::F16, "http://b:8080");
        assert!(lbl_batch.starts_with("remote-ffn-batch"));
        // Non-default wire shows in brackets.
        assert!(lbl_batch.contains("["));
    }

    // ── summarize_ffn_result ─────────────────────────────────────────────

    #[test]
    fn summarize_ffn_empty_returns_zeros() {
        let s = summarize_ffn_result(&[], &[], 3, 10);
        assert_eq!(s.n_steps, 0);
        assert_eq!(s.tok_per_s, 0.0);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
        assert!(s.note.starts_with("early stop @0/"));
    }

    #[test]
    fn summarize_ffn_with_full_rtt_data() {
        let decode = vec![10.0; 10];
        let ffn = vec![6.0; 10];
        let s = summarize_ffn_result(&decode, &ffn, 0, 10);
        assert_eq!(s.n_steps, 10);
        assert!((s.tok_per_s - 100.0).abs() < 1e-9);
        assert_eq!(s.ffn_rtt_ms, Some(6.0));
        assert!((s.attn_ms.unwrap() - 4.0).abs() < 1e-9);
        assert!(s.note.is_empty());
    }

    #[test]
    fn summarize_ffn_missing_rtt_leaves_none() {
        let decode = vec![10.0; 5];
        // ffn array shorter than decode after warmup trim.
        let ffn = vec![6.0];
        let s = summarize_ffn_result(&decode, &ffn, 0, 5);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
    }

    #[test]
    fn summarize_ffn_clamps_negative_attn_to_zero() {
        let decode = vec![5.0];
        let ffn = vec![8.0];
        let s = summarize_ffn_result(&decode, &ffn, 0, 1);
        assert_eq!(s.attn_ms, Some(0.0));
    }

    #[test]
    fn summarize_ffn_emits_early_stop_when_below_target() {
        let decode = vec![10.0; 3];
        let s = summarize_ffn_result(&decode, &[], 0, 5);
        assert_eq!(s.n_steps, 3);
        assert!(s.note.contains("early stop @3/5"));
    }

    // ── compute_wire_bytes_per_tok ───────────────────────────────────────

    #[test]
    fn wire_bytes_per_tok_zero_n_returns_none() {
        assert_eq!(compute_wire_bytes_per_tok(1000, 0), None);
    }

    #[test]
    fn wire_bytes_per_tok_divides_evenly() {
        assert_eq!(compute_wire_bytes_per_tok(1000, 5), Some(200));
    }
}
