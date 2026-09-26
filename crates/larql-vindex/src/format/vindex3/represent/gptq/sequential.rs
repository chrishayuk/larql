//! The GPTQ sequential column-elimination core.
//!
//! `ENCODER-R4.md`, "Hessian strategy": the canonical update reads a full
//! row of the *upper*-triangular Cholesky factor of `H⁻¹` per column to
//! propagate that column's quantisation error onto every remaining
//! column at once — `Cholesky(H) → H⁻¹ → Cholesky(H⁻¹)`, not merely
//! triangular solves against the first factor.
//!
//! [`larql_compute::cholesky`] returns the *lower* factor `L` (its own
//! convention throughout this workspace), so what this module needs from
//! `Cholesky(H⁻¹)` is `L_of_hinv`, and it reads it as its transpose: row
//! `q` of the upper factor `U = L_of_hinvᵀ`, from column `q` onward,
//! equals column `q` of `L_of_hinv`, from row `q` onward —
//! `U[q, k] = L_of_hinv[k, q]` for `k >= q`. That is the quantity
//! [`EliminationPlan::eliminate_row`] actually reads.
//!
//! This uses the existing hand-rolled `larql_compute::cpu::ops::linalg`
//! Cholesky, not an accelerated LAPACK backend. `ENCODER-R4.md`'s R4.2
//! rejected that backend for the real `d = 8192` site (6+ minutes,
//! incomplete) — but every site this module currently runs against is a
//! small synthetic fixture, where the scalar path is fast, already
//! proven (MEMIT depends on it), and introduces no new dependency. R4.2's
//! LAPACK acceleration is owed when this is wired against a real,
//! full-width calibration Hessian, not before.

use larql_models::quant::fp4::{e2m1_to_f32, f32_to_e2m1};
use larql_models::quant::nvfp4::E2M1_MAX;
use ndarray::Array2;

use crate::error::VindexError;

/// `Cholesky(H⁻¹)`'s lower factor, for a fixed reduced (alive-columns-
/// only) Hessian — everything GPTQ's sequential update needs to
/// propagate error, computed once per site and reused across every row.
pub struct EliminationPlan {
    l_of_hinv: Array2<f64>,
}

impl EliminationPlan {
    /// `Cholesky(reduced_h + ridge*I) → inverse → Cholesky(inverse)`.
    ///
    /// A 0x0 `reduced_h` (every column dead) is a valid input: every step
    /// below degenerates to a 0x0 result with no work done, which is
    /// exactly the zero-compensation oracle's shape — no special case
    /// needed here for it.
    pub fn build(reduced_h: &Array2<f64>, ridge: f64) -> Result<Self, VindexError> {
        let l = larql_compute::cholesky(reduced_h, ridge)
            .map_err(|e| VindexError::Parse(format!("nvfp4-gptq-v1: Cholesky(H_λ) failed: {e}")))?;
        let hinv = larql_compute::cholesky_inverse(&l);
        drop(l);
        let l_of_hinv = larql_compute::cholesky(&hinv, 0.0)
            .map_err(|e| VindexError::Parse(format!("nvfp4-gptq-v1: Cholesky(H⁻¹) failed: {e}")))?;
        Ok(Self { l_of_hinv })
    }

    /// Number of alive columns this plan was built for.
    pub fn n(&self) -> usize {
        self.l_of_hinv.shape()[0]
    }

    /// Run the sequential elimination for one output row, in the frozen
    /// original-K order (index `q` in `w0_alive`/`steps` is the `q`-th
    /// alive column, ascending).
    ///
    /// `w0_alive[q]` is that row's original weight at the `q`-th alive
    /// column; `steps[q]` is that row's already-frozen dequantisation
    /// step (`tensor_scale * e4m3(group_scale)`) for the group the `q`-th
    /// alive column's *original* K index falls in — computed once from
    /// `W0` by the caller, identically to `nvfp4-nearest-v1`, and never
    /// recomputed here. A non-finite or non-positive step (a
    /// pathological all-zero group) quantises to code `0` rather than
    /// dividing by zero, matching `quantize_row_into`'s own guard.
    pub fn eliminate_row(&self, w0_alive: &[f64], steps: &[f32]) -> RowResult {
        let n = self.n();
        debug_assert_eq!(w0_alive.len(), n);
        debug_assert_eq!(steps.len(), n);

        let mut wwork: Vec<f64> = w0_alive.to_vec();
        let mut codes = vec![0u8; n];
        let mut saturated = vec![false; n];

        for q in 0..n {
            let step = steps[q];
            let inv = if step > 0.0 && step.is_finite() {
                (step as f64).recip()
            } else {
                0.0
            };
            let scaled = (wwork[q] * inv) as f32;
            // A saturation event: the pre-round magnitude exceeds the
            // grid's top code before `f32_to_e2m1` clamps it there.
            // `nvfp4-nearest-v1` cannot hit this — its group scale is
            // chosen so the group's own amax lands exactly on the grid
            // top — so this is a real, disclosed cost of freezing scales
            // computed from `W0` against values GPTQ has since moved.
            saturated[q] = scaled.is_finite() && scaled.abs() > E2M1_MAX;
            let code = f32_to_e2m1(scaled) & 0x0F;
            codes[q] = code;

            let quant_value = f64::from(e2m1_to_f32(code)) * f64::from(step);
            let err = wwork[q] - quant_value;
            if err != 0.0 {
                // `l_of_hinv[[q, q]]` is a Cholesky diagonal entry of a
                // matrix `Self::build` only returns `Ok` for after
                // requiring strictly positive pivots — never zero here.
                let diag = self.l_of_hinv[[q, q]];
                for (offset, w) in wwork[(q + 1)..].iter_mut().enumerate() {
                    let k = q + 1 + offset;
                    *w -= (err / diag) * self.l_of_hinv[[k, q]];
                }
            }
        }

        RowResult { codes, saturated }
    }
}

/// One row's GPTQ elimination result, in reduced (alive-only) column
/// order — the caller re-inserts these at their original K positions.
pub struct RowResult {
    /// E2M1 codes (0..=15), one per alive column.
    pub codes: Vec<u8>,
    /// Whether that column's rounding saturated the frozen grid.
    pub saturated: Vec<bool>,
}

#[cfg(test)]
#[path = "tests/sequential.rs"]
mod tests;
