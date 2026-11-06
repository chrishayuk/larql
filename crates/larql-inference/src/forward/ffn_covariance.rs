//! Streaming accumulator for the uncentered FFN activation covariance
//! `C = (KᵀK) / N` that the MEMIT solve inverts.
//!
//! Accumulates in f64. With `ffn_dim` in the thousands and activation
//! magnitudes ~1e3 (observed on Gemma 4 26B-A4B, `ffn_dim = 2112`), each
//! entry grows to ~1e6 · N; an f32 accumulator (24-bit mantissa) drops the
//! low-order contributions, the Gram matrix stops being numerically PSD, and
//! the downstream Cholesky in `memit.rs` hits a negative pivot. Widening to
//! f64 *after* accumulation cannot recover precision already lost, so the
//! accumulator itself is f64 and the result is narrowed to f32 once, in
//! [`FfnGramAccumulator::finish`].

use ndarray::{Array2, ArrayView2};

/// Running `KᵀK` over captured `(seq, ffn_dim)` activation matrices.
///
/// Memory is fixed at one `(ffn_dim, ffn_dim)` f64 matrix regardless of how
/// many rows are fed in.
pub struct FfnGramAccumulator {
    ktk: Array2<f64>,
    samples: usize,
}

impl FfnGramAccumulator {
    pub fn new(ffn_dim: usize) -> Self {
        Self {
            ktk: Array2::zeros((ffn_dim, ffn_dim)),
            samples: 0,
        }
    }

    pub fn ffn_dim(&self) -> usize {
        self.ktk.nrows()
    }

    /// Number of token positions (rows) accumulated so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Add every row of `k` (shape `(seq, ffn_dim)`): `KᵀK += Σ rowᵀ·row`.
    ///
    /// Panics if `k`'s column count is not this accumulator's `ffn_dim`.
    pub fn add(&mut self, k: ArrayView2<'_, f32>) {
        let ffn_dim = self.ffn_dim();
        assert_eq!(
            k.ncols(),
            ffn_dim,
            "FFN activation width {} does not match accumulator ffn_dim {ffn_dim}",
            k.ncols()
        );
        for row in k.rows() {
            for i in 0..ffn_dim {
                let vi = row[i] as f64;
                if vi == 0.0 {
                    continue;
                }
                for j in 0..ffn_dim {
                    self.ktk[[i, j]] += vi * row[j] as f64;
                }
            }
            self.samples += 1;
        }
    }

    /// `C = (KᵀK) / N`, narrowed to f32, with the sample count `N`.
    /// `None` when no rows were accumulated.
    pub fn finish(self) -> Option<(Array2<f32>, usize)> {
        if self.samples == 0 {
            return None;
        }
        let scale = 1.0_f64 / self.samples as f64;
        Some((self.ktk.mapv(|v| (v * scale) as f32), self.samples))
    }
}
