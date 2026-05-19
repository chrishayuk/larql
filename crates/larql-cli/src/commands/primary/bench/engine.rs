//! Pure helpers for the KV-engine bench path (markov-rs, unlimited-context).
//! The I/O-bound bench loops live in `engine_runtime.rs`; this file owns:
//!   * `argmax_token` — greedy next-token pick
//!   * `format_engine_label` — engine info → label string (with / without Q4K)
//!   * `EngineSummary` + `summarize_engine_result` — decode-result trim + percentile
//!   * `format_kv_memory_note` — "hot=X cold=Y N× vs std-kv" string
//!
//! All exercised in this file's tests.

use super::row::compute_percentiles;

/// Greedy argmax over a logits slice. Returns 0 on empty input, which is
/// safe because the decode loop bails on empty hidden states before
/// reaching this.
pub(super) fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Compose the table label for a KV engine row. `q4k = true` appends the
/// "Q4K" tag (matches the production naming).
pub(super) fn format_engine_label(name: &str, backend: &str, config: &str, q4k: bool) -> String {
    let q4k_tag = if q4k { " Q4K" } else { "" };
    if config.is_empty() {
        format!("{name} [{backend}]{q4k_tag}")
    } else {
        format!("{name} [{backend}] ({config}){q4k_tag}")
    }
}

/// Decode-loop summary used by both `run_engine` and `run_engine_q4k`.
pub(super) struct EngineSummary {
    pub avg_decode_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tok_per_s: f64,
    pub n_steps: usize,
}

/// Trim warmup, compute percentiles. Pure; safe to call from anywhere.
pub(super) fn summarize_engine_result(decode_ms_all: &[f64], warmup: usize) -> EngineSummary {
    let n_warm = warmup.min(decode_ms_all.len());
    let measured = &decode_ms_all[n_warm..];
    let n = measured.len();
    if n == 0 {
        return EngineSummary {
            avg_decode_ms: 0.0,
            p50_ms: 0.0,
            p99_ms: 0.0,
            tok_per_s: 0.0,
            n_steps: 0,
        };
    }
    let (avg, p50, p99) = compute_percentiles(measured);
    EngineSummary {
        avg_decode_ms: avg,
        p50_ms: p50,
        p99_ms: p99,
        tok_per_s: 1000.0 / avg,
        n_steps: n,
    }
}

/// Render the memory-footprint note for an engine row: hot, cold, and the
/// compression ratio relative to a Standard KV (FP16) baseline.
/// `total = 0` means we couldn't query the engine; emit a 0× ratio.
pub(super) fn format_kv_memory_note(total: usize, cold: usize, kv_ref: usize) -> String {
    let hot = total.saturating_sub(cold);
    let ratio = if total > 0 {
        kv_ref as f64 / total as f64
    } else {
        0.0
    };
    format!(
        "hot={:.1}MB cold={:.1}MB  {:.0}× vs std-kv",
        hot as f64 / 1_048_576.0,
        cold as f64 / 1_048_576.0,
        ratio,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── argmax_token ─────────────────────────────────────────────────────

    #[test]
    fn argmax_token_returns_index_of_max() {
        assert_eq!(argmax_token(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_token(&[9.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_token(&[0.0, 0.0, 5.5]), 2);
    }

    #[test]
    fn argmax_token_empty_returns_zero() {
        assert_eq!(argmax_token(&[]), 0);
    }

    #[test]
    fn argmax_token_handles_nan_gracefully() {
        let v = [f32::NAN, 2.0, 1.0];
        let idx = argmax_token(&v);
        assert!((idx as usize) < v.len());
    }

    // ── format_engine_label ──────────────────────────────────────────────

    #[test]
    fn label_without_config_omits_parens() {
        assert_eq!(
            format_engine_label("markov-rs", "cpu", "", false),
            "markov-rs [cpu]"
        );
    }

    #[test]
    fn label_with_config_shows_parens() {
        assert_eq!(
            format_engine_label("markov-rs", "metal", "lambda=2", false),
            "markov-rs [metal] (lambda=2)"
        );
    }

    #[test]
    fn label_q4k_tag_appears_when_set() {
        assert_eq!(
            format_engine_label("uc", "metal", "", true),
            "uc [metal] Q4K"
        );
        assert_eq!(
            format_engine_label("uc", "metal", "x", true),
            "uc [metal] (x) Q4K"
        );
    }

    // ── summarize_engine_result ──────────────────────────────────────────

    #[test]
    fn summarize_zero_post_warmup_returns_zeros() {
        let s = summarize_engine_result(&[10.0, 10.0], 10);
        assert_eq!(s.n_steps, 0);
        assert_eq!(s.tok_per_s, 0.0);
    }

    #[test]
    fn summarize_computes_tok_per_s_from_avg() {
        let decode = vec![10.0; 5];
        let s = summarize_engine_result(&decode, 0);
        assert_eq!(s.n_steps, 5);
        assert!((s.tok_per_s - 100.0).abs() < 1e-9);
        assert!((s.avg_decode_ms - 10.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_trims_warmup_prefix() {
        // 3 warmup + 5 measured; warmup values are very slow (100ms)
        // while measured are fast (10ms). Final avg should reflect only
        // the measured tail.
        let mut decode = vec![100.0; 3];
        decode.extend(vec![10.0; 5]);
        let s = summarize_engine_result(&decode, 3);
        assert_eq!(s.n_steps, 5);
        assert!((s.avg_decode_ms - 10.0).abs() < 1e-9);
    }

    // ── format_kv_memory_note ────────────────────────────────────────────

    #[test]
    fn kv_memory_note_normal_case() {
        // total = 16 MB, cold = 4 MB → hot = 12 MB. Ratio = 64/16 = 4.
        let s = format_kv_memory_note(16 * 1024 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024);
        assert!(s.contains("hot=12.0MB"));
        assert!(s.contains("cold=4.0MB"));
        assert!(s.contains("4× vs std-kv"));
    }

    #[test]
    fn kv_memory_note_zero_total_emits_zero_ratio() {
        let s = format_kv_memory_note(0, 0, 1024);
        assert!(s.contains("0× vs std-kv"));
    }

    #[test]
    fn kv_memory_note_clamps_hot_when_cold_exceeds_total() {
        // Engine bug guard: cold > total shouldn't underflow.
        let s = format_kv_memory_note(1024, 4096, 0);
        assert!(s.contains("hot=0.0MB"));
    }
}
