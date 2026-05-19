//! Pure helpers for the remote-MoE bench path. The I/O-bound bench
//! execution lives in `remote_moe_runtime.rs`; this file holds:
//!   * `parse_shard_segments` — shard-map flag parser
//!   * `format_moe_backend_label` — table-label composer
//!   * `summarize_moe_result` — decode-result post-processing
//!
//! All three are exercised by unit tests in this file. The runtime wrapper
//! depends on them but doesn't pull in any test-only state.

use larql_inference::ffn::moe_remote::ShardConfig;

use super::row::compute_percentiles;

/// Parse the `--moe-shards` flag value into a `Vec<ShardConfig>`. Accepts
/// `"START-END=URL,START-END=URL,..."`. Returns an error message with the
/// offending segment when input is malformed.
pub(super) fn parse_shard_segments(spec: &str) -> Result<Vec<ShardConfig>, String> {
    let mut configs: Vec<ShardConfig> = Vec::new();
    for segment in spec.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.splitn(2, '=');
        let range_str = parts
            .next()
            .ok_or_else(|| format!("malformed shard segment: {segment:?}"))?;
        let url = parts
            .next()
            .ok_or_else(|| format!("missing URL in shard segment: {segment:?}"))?;
        let (start, end_incl) = ShardConfig::parse_range(range_str)
            .ok_or_else(|| format!("bad expert range {range_str:?} in --moe-shards"))?;
        configs.push(ShardConfig::new(start, end_incl, url));
    }
    if configs.is_empty() {
        return Err("--moe-shards: no valid shard segments".into());
    }
    Ok(configs)
}

/// Compose the `"remote-moe-<mode> (<N> shards)"` backend label that goes
/// into the table.
pub(super) fn format_moe_backend_label(is_batch: bool, num_shards: usize) -> String {
    format!(
        "remote-moe-{} ({} shards)",
        if is_batch { "batch" } else { "stream" },
        num_shards
    )
}

/// Bench result summary. Folded out of `run_remote_moe_bench` so the
/// post-result computation (percentile, avg-ffn-rtt, attn-fallback) can be
/// covered by unit tests without booting a real RemoteMoeBackend.
pub(super) struct MoeSummary {
    pub avg_decode_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tok_per_s: f64,
    pub ffn_rtt_ms: Option<f64>,
    pub attn_ms: Option<f64>,
    pub n_steps: usize,
    pub note: String,
}

/// Trim warmup, percentile, derive `attn_ms = avg_decode - avg_ffn`.
pub(super) fn summarize_moe_result(
    decode_ms: &[f64],
    ffn_rtt_ms: &[f64],
    warmup: usize,
    target_tokens: usize,
) -> MoeSummary {
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

    MoeSummary {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_shard_segments ─────────────────────────────────────────────

    #[test]
    fn parse_shard_segments_accepts_well_formed_map() {
        let cfgs = parse_shard_segments("0-31=http://a:8080, 32-63=http://b:8080").unwrap();
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].start, 0);
        assert_eq!(cfgs[0].end, 31);
        assert_eq!(cfgs[0].url, "http://a:8080");
        assert_eq!(cfgs[1].start, 32);
        assert_eq!(cfgs[1].end, 63);
    }

    #[test]
    fn parse_shard_segments_skips_blank_segments() {
        let cfgs = parse_shard_segments("0-31=http://a, , 32-63=http://b").unwrap();
        assert_eq!(cfgs.len(), 2);
    }

    #[test]
    fn parse_shard_segments_rejects_missing_url() {
        let err = parse_shard_segments("0-31").unwrap_err();
        assert!(err.contains("missing URL"), "got: {err}");
    }

    #[test]
    fn parse_shard_segments_rejects_bad_range() {
        let err = parse_shard_segments("notarange=http://a").unwrap_err();
        assert!(err.contains("bad expert range"), "got: {err}");
    }

    #[test]
    fn parse_shard_segments_rejects_empty_spec() {
        let err = parse_shard_segments("").unwrap_err();
        assert!(err.contains("no valid shard segments"), "got: {err}");
        let err = parse_shard_segments(", ,").unwrap_err();
        assert!(err.contains("no valid shard segments"), "got: {err}");
    }

    // ── format_moe_backend_label ─────────────────────────────────────────

    #[test]
    fn label_picks_mode_and_shows_shard_count() {
        assert_eq!(
            format_moe_backend_label(true, 4),
            "remote-moe-batch (4 shards)"
        );
        assert_eq!(
            format_moe_backend_label(false, 1),
            "remote-moe-stream (1 shards)"
        );
    }

    // ── summarize_moe_result ─────────────────────────────────────────────

    #[test]
    fn summarize_no_post_warmup_returns_zeros() {
        let s = summarize_moe_result(&[10.0, 10.0], &[], 5, 10);
        assert_eq!(s.n_steps, 0);
        assert_eq!(s.avg_decode_ms, 0.0);
        assert_eq!(s.tok_per_s, 0.0);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
        assert!(s.note.starts_with("early stop @0/"));
    }

    #[test]
    fn summarize_with_ffn_rtt_derives_attn() {
        let decode = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let ffn = vec![80.0, 80.0, 80.0, 80.0, 80.0];
        let s = summarize_moe_result(&decode, &ffn, 0, 5);
        assert_eq!(s.n_steps, 5);
        assert!((s.avg_decode_ms - 100.0).abs() < 1e-9);
        assert!((s.tok_per_s - 10.0).abs() < 1e-9);
        assert_eq!(s.ffn_rtt_ms, Some(80.0));
        assert!((s.attn_ms.unwrap() - 20.0).abs() < 1e-9);
        assert!(s.note.is_empty(), "n == target so no early-stop note");
    }

    #[test]
    fn summarize_missing_ffn_rtt_leaves_none() {
        let decode = vec![100.0, 100.0, 100.0];
        let ffn = vec![];
        let s = summarize_moe_result(&decode, &ffn, 0, 3);
        assert_eq!(s.n_steps, 3);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
    }

    #[test]
    fn summarize_clamps_negative_attn_to_zero() {
        let decode = vec![50.0];
        let ffn = vec![80.0];
        let s = summarize_moe_result(&decode, &ffn, 0, 1);
        assert_eq!(s.attn_ms, Some(0.0));
    }
}
