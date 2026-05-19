//! Per-stage timing for KV-cache engines.
//!
//! Enable by constructing engines with `with_profiling(true)`. Each decode
//! step accumulates per-stage wall-clock times; call `stage_summary()` after
//! decoding to retrieve averaged results.
//!
//! Overhead when disabled: one branch per stage (zero-cost in release builds
//! when the compiler inlines `if self.profiling { ... }`).

use larql_inference::DecodeStageSummary;
use std::time::Instant;

/// Accumulator for a single timing stage. Add new samples with `record`.
#[derive(Debug, Clone, Default)]
pub struct StageAccumulator {
    pub total_us: f64,
    pub count: usize,
}

impl StageAccumulator {
    pub fn record(&mut self, t: Instant) {
        self.total_us += t.elapsed().as_secs_f64() * 1e6;
        self.count += 1;
    }

    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_us / self.count as f64
        }
    }
}

// `DecodeStageSummary` lives in `larql-inference::kv_engine` (re-exported as
// `larql_inference::DecodeStageSummary`). See
// `crates/larql-inference/docs/specs/kv-engine-unification.md` §10.4.

/// Per-engine profiling state.
/// Field layout matches `MarkovResidualEngine` — add more engines as needed.
#[derive(Debug, Default)]
pub struct EngineProfiler {
    pub embed: StageAccumulator,
    pub recompute_cold: StageAccumulator,
    pub recompute_hot: StageAccumulator,
    pub attention: StageAccumulator,
    pub ffn: StageAccumulator,
    pub decode_total: StageAccumulator,
}

impl EngineProfiler {
    pub fn summary(&self, engine: &str, backend: &str) -> DecodeStageSummary {
        DecodeStageSummary {
            engine: engine.to_string(),
            backend: backend.to_string(),
            steps: self.decode_total.count,
            avg_embed_us: self.embed.avg_us(),
            avg_recompute_cold_us: self.recompute_cold.avg_us(),
            avg_recompute_hot_us: self.recompute_hot.avg_us(),
            avg_attention_us: self.attention.avg_us(),
            avg_ffn_us: self.ffn.avg_us(),
            avg_total_decode_us: self.decode_total.avg_us(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn stage_accumulator_avg_us_zero_when_empty() {
        let acc = StageAccumulator::default();
        assert_eq!(acc.avg_us(), 0.0);
        assert_eq!(acc.count, 0);
    }

    #[test]
    fn stage_accumulator_records_and_averages() {
        let mut acc = StageAccumulator::default();
        for _ in 0..3 {
            let t = Instant::now();
            sleep(Duration::from_micros(50));
            acc.record(t);
        }
        assert_eq!(acc.count, 3);
        assert!(
            acc.total_us >= 150.0,
            "expected ≥150 µs, got {}",
            acc.total_us
        );
        let avg = acc.avg_us();
        assert!(avg >= 50.0, "expected avg ≥50 µs, got {avg}");
    }

    #[test]
    fn stage_accumulator_clone_preserves_state() {
        let mut acc = StageAccumulator::default();
        let t = Instant::now();
        sleep(Duration::from_micros(10));
        acc.record(t);
        let copy = acc.clone();
        assert_eq!(copy.count, acc.count);
        assert_eq!(copy.total_us, acc.total_us);
    }

    #[test]
    fn engine_profiler_summary_reflects_decode_total_steps() {
        let mut prof = EngineProfiler::default();
        for _ in 0..5 {
            prof.decode_total.record(Instant::now());
        }
        let summary = prof.summary("test-engine", "cpu");
        assert_eq!(summary.engine, "test-engine");
        assert_eq!(summary.backend, "cpu");
        assert_eq!(summary.steps, 5);
        assert_eq!(summary.avg_embed_us, 0.0);
        assert_eq!(summary.avg_attention_us, 0.0);
    }

    #[test]
    fn engine_profiler_summary_carries_per_stage_averages() {
        let mut prof = EngineProfiler::default();
        let t = Instant::now();
        sleep(Duration::from_micros(20));
        prof.embed.record(t);
        prof.attention.record(t);
        prof.ffn.record(t);
        prof.decode_total.record(t);

        let s = prof.summary("e", "metal");
        assert!(s.avg_embed_us > 0.0);
        assert!(s.avg_attention_us > 0.0);
        assert!(s.avg_ffn_us > 0.0);
        assert!(s.avg_total_decode_us > 0.0);
        assert_eq!(s.avg_recompute_total_us(), 0.0);
    }

    #[test]
    fn decode_stage_summary_recompute_total_sums_tiers() {
        let s = DecodeStageSummary {
            engine: "x".into(),
            backend: "cpu".into(),
            steps: 1,
            avg_embed_us: 0.0,
            avg_recompute_cold_us: 12.0,
            avg_recompute_hot_us: 8.0,
            avg_attention_us: 0.0,
            avg_ffn_us: 0.0,
            avg_total_decode_us: 0.0,
        };
        assert_eq!(s.avg_recompute_total_us(), 20.0);
    }

    #[test]
    fn decode_stage_summary_print_with_recompute() {
        // Smoke test that print() handles both branches (recompute_total > 0
        // and == 0). Output goes to stdout — captured by the test harness.
        let with_recompute = DecodeStageSummary {
            engine: "markov-rs".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 100.0,
            avg_recompute_cold_us: 500.0,
            avg_recompute_hot_us: 300.0,
            avg_attention_us: 1500.0,
            avg_ffn_us: 800.0,
            avg_total_decode_us: 3200.0,
        };
        with_recompute.print();

        let no_recompute = DecodeStageSummary {
            engine: "turbo-quant".into(),
            backend: "metal".into(),
            steps: 0,
            avg_embed_us: 0.0,
            avg_recompute_cold_us: 0.0,
            avg_recompute_hot_us: 0.0,
            avg_attention_us: 0.0,
            avg_ffn_us: 0.0,
            avg_total_decode_us: 0.0,
        };
        // total == 0 hits the fallback branch in `pct`.
        no_recompute.print();
    }

    #[test]
    fn decode_stage_summary_clone_is_independent() {
        let s = DecodeStageSummary {
            engine: "a".into(),
            backend: "b".into(),
            steps: 1,
            avg_embed_us: 1.0,
            avg_recompute_cold_us: 2.0,
            avg_recompute_hot_us: 3.0,
            avg_attention_us: 4.0,
            avg_ffn_us: 5.0,
            avg_total_decode_us: 15.0,
        };
        let copy = s.clone();
        assert_eq!(copy.steps, s.steps);
        assert_eq!(copy.avg_total_decode_us, s.avg_total_decode_us);
    }
}
