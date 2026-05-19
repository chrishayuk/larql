//! Pure helpers for the local Metal/CPU bench path. The I/O-heavy
//! `run_larql` body lives in `local_runtime.rs`; this file owns:
//!   * `backend_name_for` — `"larql-metal"` / `"larql-cpu"`
//!   * `format_early_stop_note` — note string for partial / no-decode runs
//!   * `append_cpu_fallback_note` — makes CPU Q4K fallback rows explicit
//!   * `format_q4k_cache_log` — verbose `-v` cache-stats line
//!
//! All exercised by tests in this file.

/// Returns the table-row backend label for the local bench.
pub(super) fn backend_name_for(metal: bool) -> &'static str {
    if metal {
        "larql-metal"
    } else {
        "larql-cpu"
    }
}

/// Note string for the local bench row: either empty (full target reached),
/// "early stop @n/target …" (partial), or "no decode steps completed …"
/// when `measured_n == 0`.
pub(super) fn format_early_stop_note(
    measured_n: usize,
    target_tokens: usize,
    wall_ms: f64,
) -> String {
    if measured_n == 0 {
        format!("no decode steps completed (wall {:.0}ms)", wall_ms)
    } else if measured_n < target_tokens {
        format!(
            "early stop @{}/{} (EOS or GPU fallback)",
            measured_n, target_tokens
        )
    } else {
        String::new()
    }
}

/// Annotates CPU rows with which Q4K sub-path ran. The cached path
/// uses prefill + KV-cached single-row decode; the legacy path
/// reprocesses the full sequence at every step.
pub(super) fn append_cpu_fallback_note(note: String, cached: bool) -> String {
    let tag = if cached {
        "cpu q4k (KV-cached decode)"
    } else {
        "cpu q4k legacy (O(N²) per-step)"
    };
    if note.is_empty() {
        tag.to_string()
    } else {
        format!("{note}; {tag}")
    }
}

/// Verbose log line for the Q4K dequant-cache stats after a run.
pub(super) fn format_q4k_cache_log(backend_label: &str, slots: usize, bytes: usize) -> String {
    format!(
        "[bench] kquant_ffn_cache after {}: {} populated slots, {:.1} MB",
        backend_label,
        slots,
        bytes as f64 / 1_048_576.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_for_picks_label() {
        assert_eq!(backend_name_for(true), "larql-metal");
        assert_eq!(backend_name_for(false), "larql-cpu");
    }

    #[test]
    fn early_stop_note_empty_when_target_reached() {
        assert!(format_early_stop_note(50, 50, 1234.0).is_empty());
    }

    #[test]
    fn early_stop_note_reports_partial_when_below_target() {
        let s = format_early_stop_note(20, 50, 5000.0);
        assert!(s.starts_with("early stop @20/50"));
        assert!(s.contains("EOS or GPU fallback"));
    }

    #[test]
    fn early_stop_note_reports_wall_when_zero_steps() {
        let s = format_early_stop_note(0, 50, 1234.0);
        assert!(s.starts_with("no decode steps completed"));
        assert!(s.contains("1234ms"));
    }

    #[test]
    fn cpu_fallback_note_labels_cached_vs_legacy() {
        assert_eq!(
            append_cpu_fallback_note(String::new(), true),
            "cpu q4k (KV-cached decode)"
        );
        assert_eq!(
            append_cpu_fallback_note(String::new(), false),
            "cpu q4k legacy (O(N²) per-step)"
        );
    }

    #[test]
    fn cpu_fallback_note_appends_to_existing_note() {
        let s = append_cpu_fallback_note("early stop @4/5".to_string(), true);
        assert_eq!(s, "early stop @4/5; cpu q4k (KV-cached decode)");
    }

    #[test]
    fn q4k_cache_log_reports_slots_and_mb() {
        let s = format_q4k_cache_log("larql-metal", 12, 16 * 1024 * 1024);
        assert!(s.contains("after larql-metal"));
        assert!(s.contains("12 populated slots"));
        assert!(s.contains("16.0 MB"));
    }

    #[test]
    fn q4k_cache_log_zero_bytes_shows_zero() {
        let s = format_q4k_cache_log("larql-cpu", 0, 0);
        assert!(s.contains("0 populated slots"));
        assert!(s.contains("0.0 MB"));
    }
}
