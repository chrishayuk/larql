//! Numerical comparison utilities for residual captures.
//!
//! All metrics are computed in `f64` to avoid catastrophic cancellation
//! on long vectors with mixed signs (a 5376-wide hidden state has plenty
//! of room for f32 accumulation error to dominate the signal we're
//! actually checking). Outputs are converted back to `f32` at the API
//! boundary — both for memory parity with the captures and because
//! `0.99995_f32` reads more naturally than `0.99995_f64` in test code.
//!
//! Two thresholds, both must pass:
//!   - `cos`: cosine similarity, catches direction drift.
//!   - `rel_max_abs`: max absolute element-wise diff divided by the
//!     reference's L2 norm. Catches position-local regressions that cos
//!     hides (a single dim flipping sign on a wide vector barely moves
//!     cos but spikes max_abs).
//!
//! Both default presets ([`ParityThreshold::tight`] /
//! [`ParityThreshold::loose`]) are calibrated against the worst float
//! noise observed across our four test vindexes — Gemma 3 4B, Gemma 4
//! 31B dense, Llama 2 7B, Mistral 7B v0.1.

use super::capture::ResidualCapture;

/// Per-layer comparison output. `cos` close to 1.0 means matching
/// direction; `max_abs` close to 0.0 means matching pointwise. Both
/// matter — see module docs.
#[derive(Debug, Clone, Copy)]
pub struct LayerStat {
    pub layer: usize,
    pub cos: f32,
    pub max_abs: f32,
    /// L2 norm of the reference (`a`) capture. Useful for callers that
    /// want to compute their own relative metrics.
    pub a_norm: f32,
    /// L2 norm of the comparison (`b`) capture.
    pub b_norm: f32,
}

impl LayerStat {
    /// Max abs diff as a fraction of the reference norm. The relative
    /// scale travels across architectures (Gemma 3 hidden=2560 has
    /// norms ~400, Gemma 4 31B has ~1500) where an absolute threshold
    /// would either be too loose for one or too tight for another.
    pub fn rel_max_abs(&self) -> f32 {
        if self.a_norm > 0.0 {
            self.max_abs / self.a_norm
        } else {
            0.0
        }
    }
}

/// Pair of thresholds — both must pass for a layer to be "clean".
#[derive(Debug, Clone, Copy)]
pub struct ParityThreshold {
    pub cos: f32,
    pub rel_max_abs: f32,
}

impl ParityThreshold {
    /// What we expect when two paths run the same compute. Float noise
    /// across BF16→f32 dequant + BLAS-vs-scalar accumulation order sits
    /// well below these on Gemma 3 / Gemma 4 / Llama 2 / Mistral —
    /// empirically all 158 layers in `test_cpu_metal_parity` fit.
    pub const fn tight() -> Self {
        Self {
            cos: 0.99995,
            rel_max_abs: 0.01,
        }
    }

    /// For paths that go through different kernel families (e.g.
    /// fused mixed-quant vs per-projection) where small absolute
    /// drift accumulates but cos stays high. Used by the looser
    /// regression guards.
    pub const fn loose() -> Self {
        Self {
            cos: 0.999,
            rel_max_abs: 0.05,
        }
    }
}

/// Whole-run report: every layer's stats plus the index of the first
/// layer that breached the threshold.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub layers: Vec<LayerStat>,
    pub first_bad: Option<usize>,
    pub threshold: ParityThreshold,
}

impl ParityReport {
    pub fn is_clean(&self) -> bool {
        self.first_bad.is_none()
    }

    /// Panic-friendly assertion with a useful diagnostic. Tests use
    /// this so a parity break surfaces with first-bad-layer + cos +
    /// max_abs at the failure site, no extra `eprintln!` plumbing.
    pub fn assert_clean(&self) -> Result<(), String> {
        match self.first_bad {
            None => Ok(()),
            Some(l) => {
                let s = &self.layers[l];
                Err(format!(
                    "parity broken at L{l}: cos={:.6} max_abs={:.3e} \
                     ({:.3}% of ref ||{:.2}||); thresholds: cos≥{}, rel≤{}",
                    s.cos,
                    s.max_abs,
                    100.0 * s.rel_max_abs(),
                    s.a_norm,
                    self.threshold.cos,
                    self.threshold.rel_max_abs,
                ))
            }
        }
    }
}

/// Compare two captures layer-by-layer. Each `a.layers[l]` and
/// `b.layers[l]` must have the same length — the comparison surfaces
/// any shape mismatch in the report's first-bad slot.
pub fn compare_captures(
    a: &ResidualCapture,
    b: &ResidualCapture,
    thr: ParityThreshold,
) -> ParityReport {
    let n = a.layers.len().min(b.layers.len());
    let mut stats = Vec::with_capacity(n);
    let mut first_bad: Option<usize> = None;
    for l in 0..n {
        let av = &a.layers[l];
        let bv = &b.layers[l];
        if av.len() != bv.len() {
            // Surface as cos=0, max_abs=inf so callers see it as a hard
            // miss without us inventing a side-channel error type.
            stats.push(LayerStat {
                layer: l,
                cos: 0.0,
                max_abs: f32::INFINITY,
                a_norm: 0.0,
                b_norm: 0.0,
            });
            if first_bad.is_none() {
                first_bad = Some(l);
            }
            continue;
        }
        let s = layer_stat(l, av, bv);
        if (s.cos < thr.cos || s.rel_max_abs() > thr.rel_max_abs) && first_bad.is_none() {
            first_bad = Some(l);
        }
        stats.push(s);
    }
    ParityReport {
        layers: stats,
        first_bad,
        threshold: thr,
    }
}

fn layer_stat(layer: usize, a: &[f32], b: &[f32]) -> LayerStat {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut a_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        a_sq += x * x;
        b_sq += y * y;
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if a_sq > 0.0 && b_sq > 0.0 {
        (dot / (a_sq.sqrt() * b_sq.sqrt())) as f32
    } else {
        0.0
    };
    LayerStat {
        layer,
        cos,
        max_abs,
        a_norm: a_sq.sqrt() as f32,
        b_norm: b_sq.sqrt() as f32,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::capture::ResidualCapture;
    use super::*;

    fn cap(layers: Vec<Vec<f32>>, hidden: usize, seq_len: usize) -> ResidualCapture {
        ResidualCapture {
            layers,
            hidden_size: hidden,
            seq_len,
        }
    }

    #[test]
    fn identical_captures_have_cos_one_and_zero_max_abs() {
        let a = cap(vec![vec![1.0, 2.0, 3.0, 4.0]], 4, 1);
        let b = cap(vec![vec![1.0, 2.0, 3.0, 4.0]], 4, 1);
        let r = compare_captures(&a, &b, ParityThreshold::tight());
        assert!(r.is_clean());
        assert!((r.layers[0].cos - 1.0).abs() < 1e-6);
        assert_eq!(r.layers[0].max_abs, 0.0);
    }

    #[test]
    fn drift_above_threshold_flagged_as_first_bad() {
        // Layer 0 matches, layer 1 has a single huge spike that breaks
        // rel_max_abs even though cos stays high.
        let mut b1 = vec![1.0; 64];
        b1[5] = 100.0; // spike
        let a = cap(vec![vec![1.0; 64], vec![1.0; 64]], 64, 1);
        let b = cap(vec![vec![1.0; 64], b1], 64, 1);
        let r = compare_captures(&a, &b, ParityThreshold::tight());
        assert_eq!(r.first_bad, Some(1));
        assert!(!r.is_clean());
    }

    #[test]
    fn shape_mismatch_surfaces_as_hard_miss() {
        let a = cap(vec![vec![1.0; 64]], 64, 1);
        let b = cap(vec![vec![1.0; 32]], 32, 1);
        let r = compare_captures(&a, &b, ParityThreshold::tight());
        assert_eq!(r.first_bad, Some(0));
        assert_eq!(r.layers[0].max_abs, f32::INFINITY);
    }

    #[test]
    fn assert_clean_returns_err_with_first_bad_detail() {
        let a = cap(vec![vec![1.0; 4]], 4, 1);
        let b = cap(vec![vec![1.0, 1.0, 1.0, 50.0]], 4, 1);
        let r = compare_captures(&a, &b, ParityThreshold::tight());
        let err = r.assert_clean().unwrap_err();
        assert!(err.contains("L0"), "err must name first-bad layer: {err}");
        assert!(err.contains("max_abs"), "err must surface max_abs: {err}");
    }

    #[test]
    fn loose_threshold_accepts_what_tight_rejects() {
        // 5% relative drift — passes loose (≤5%) but fails tight (≤1%).
        let mut b0 = vec![1.0; 100];
        b0[0] = 1.05; // delta 0.05; ||a|| = sqrt(100)=10; rel = 0.05/10 = 0.5% — actually small
                      // Need a bigger delta to land between loose and tight.
        b0[0] = 2.0; // delta 1.0; rel = 1/10 = 10%? still too big for loose.
                     // Just construct directly: rel = 0.03 (between 0.01 and 0.05).
        let mut a0 = vec![0.0; 100];
        a0[0] = 10.0;
        let mut b0 = vec![0.0; 100];
        b0[0] = 10.3; // delta 0.3, ||a||=10, rel=3%
        let a = cap(vec![a0], 100, 1);
        let b = cap(vec![b0], 100, 1);
        let r_tight = compare_captures(&a, &b, ParityThreshold::tight());
        let r_loose = compare_captures(&a, &b, ParityThreshold::loose());
        assert!(!r_tight.is_clean(), "3% rel drift must fail tight");
        assert!(r_loose.is_clean(), "3% rel drift should pass loose");
    }
}
