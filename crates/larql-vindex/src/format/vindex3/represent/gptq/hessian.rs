//! The dead/alive column partition GPTQ derives from a raw calibration
//! Hessian, and the reduced sub-matrix it actually factorises.
//!
//! `ENCODER-R4.md`, "Dead coordinates — an exact rule, not a threshold":
//! a column `j` with raw `H[j,j] == 0` (checked before damping, no
//! epsilon) carried zero energy across the whole calibration pass — its
//! entire row and column in the raw Hessian are exactly zero, because
//! `H[j,j] = sum_x x_j(x)^2` is a sum of non-negative terms, so `== 0`
//! forces `x_j(x) == 0` for every calibration sample, which in turn
//! forces `H[j,m] = sum_x x_j(x) x_m(x) == 0` for every other coordinate
//! `m` — not approximately, exactly, in floating point (every term in
//! that sum is `0.0 * finite`, which IEEE-754 computes as exactly `0.0`).
//!
//! Two ways to honour "no error propagated from or into j" follow from
//! that: rely on the resulting block-diagonal structure to decouple `j`
//! automatically inside a full dense solve, or exclude `j` from the
//! linear algebra outright. This module takes the second, more literal
//! reading — `j` never enters the reduced Hessian GPTQ factorises, so
//! "no error propagated from or into j" is a structural guarantee of the
//! gather, not a numerical property a reader has to trust.

use ndarray::Array2;

/// A raw (undamped) per-input-column second-moment matrix for one
/// calibration site — `H = XᵀX` over calibration activations — together
/// with the dead/alive partition GPTQ derives from it.
///
/// Owns only the matrix and the partition, not how `raw` was produced;
/// candidate-path capture and artifact validation live in the separate
/// REPRESENT calibration/recipe modules.
pub struct SiteHessian {
    raw: Array2<f64>,
    /// Column indices with `raw[j,j] != 0`, ascending — original K order
    /// is exactly ascending index order, so this list already satisfies
    /// the frozen "column order: original K order" for the columns GPTQ
    /// actually touches.
    alive: Vec<usize>,
    /// Column indices with `raw[j,j] == 0`, ascending.
    dead: Vec<usize>,
}

impl SiteHessian {
    /// Partition `raw`'s columns by the exact dead-coordinate rule.
    ///
    /// # Panics
    /// If `raw` is not square.
    pub fn from_raw(raw: Array2<f64>) -> Self {
        let d = raw.shape()[0];
        assert_eq!(
            raw.shape()[1],
            d,
            "SiteHessian::from_raw: matrix must be square, got {}x{}",
            d,
            raw.shape()[1]
        );
        let mut alive = Vec::with_capacity(d);
        let mut dead = Vec::new();
        for j in 0..d {
            if raw[[j, j]] == 0.0 {
                dead.push(j);
            } else {
                alive.push(j);
            }
        }
        Self { raw, alive, dead }
    }

    /// `k` for the site this Hessian was accumulated over.
    pub fn dim(&self) -> usize {
        self.raw.shape()[0]
    }

    /// Original-K-order column indices GPTQ compensates.
    pub fn alive(&self) -> &[usize] {
        &self.alive
    }

    /// Original-K-order column indices that fall back to ordinary
    /// nearest rounding, untouched by any propagated error.
    pub fn dead(&self) -> &[usize] {
        &self.dead
    }

    /// `0.01 * mean(diag(raw))`, over every column of the *original*
    /// `d`-wide Hessian — the literal reading of the frozen "damping
    /// 0.01 x mean(diag(H))" parameter, which does not qualify which
    /// columns "H" ranges over. Dead columns contribute exact zeros to
    /// the sum, so their presence only dilutes the ridge applied to the
    /// alive sub-matrix; they never receive it themselves, since they
    /// never enter [`Self::reduced`].
    pub fn damping_ridge(&self) -> f64 {
        let d = self.dim();
        if d == 0 {
            return 0.0;
        }
        let sum: f64 = (0..d).map(|j| self.raw[[j, j]]).sum();
        0.01 * sum / d as f64
    }

    /// The dense sub-matrix over [`Self::alive`] columns only, in their
    /// original relative K order — what GPTQ's Cholesky factorisation
    /// actually runs against. A 0x0 matrix (every column dead) is a
    /// valid, deliberately trivial result: the zero-compensation oracle
    /// is exactly this case.
    pub fn reduced(&self) -> Array2<f64> {
        let n = self.alive.len();
        let mut out = Array2::<f64>::zeros((n, n));
        for (oi, &i) in self.alive.iter().enumerate() {
            for (oj, &j) in self.alive.iter().enumerate() {
                out[[oi, oj]] = self.raw[[i, j]];
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "tests/hessian.rs"]
mod tests;
