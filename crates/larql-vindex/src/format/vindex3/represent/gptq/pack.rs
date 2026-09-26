//! Compile one `[rows, k]` weight matrix under fixed-grid GPTQ.
//!
//! Orchestrates the pieces the sibling modules own: [`super::hessian`]'s
//! dead/alive partition, [`super::sequential`]'s column elimination, and
//! `nvfp4-nearest-v1`'s own scale derivation (reused verbatim, not
//! reimplemented, so the frozen scale bytes cannot drift from it).

use larql_models::quant::fp4::{pack_nibbles, unpack_nibbles};
use larql_models::quant::fp8::e4m3_to_f32;
use larql_models::quant::nvfp4::{
    quantize_row_into, tensor_scale_for, Nvfp4Matrix, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
};
use ndarray::Array2;

use crate::error::VindexError;

use super::hessian::SiteHessian;
use super::sequential::EliminationPlan;

/// Outcome of compiling one tensor under `nvfp4-gptq-v1`: the pack
/// itself, plus the disclosure numbers `ENCODER-R4.md` requires reported
/// alongside any quality result (dead-coordinate incidence, saturation).
#[derive(Debug)]
pub struct GptqPackOutcome {
    pub matrix: Nvfp4Matrix,
    /// Columns that fell back to ordinary nearest rounding.
    pub dead_columns: usize,
    /// Columns GPTQ actually compensated.
    pub alive_columns: usize,
    /// Element-level saturation events — a scaled magnitude that exceeded
    /// the frozen grid's top code before rounding clamped it there. Zero
    /// for `nvfp4-nearest-v1`, always possible here since compensation
    /// moves values against a scale chosen for the originals.
    pub saturated_elements: usize,
    pub total_elements: usize,
}

/// Compile `[rows, k]` under fixed-grid GPTQ against `h_raw`, the site's
/// raw (undamped) calibration Hessian.
///
/// `h_raw` must be `[k, k]`. Every scale byte is frozen from `w0` alone,
/// before `h_raw` is even consulted — an all-zero `h_raw` is a legal
/// input, and it exercises the zero-compensation oracle: every column is
/// dead by the exact rule ([`SiteHessian::from_raw`]), so the whole
/// matrix falls back to ordinary nearest rounding and the result is
/// byte-identical to `nvfp4-nearest-v1`, payload included.
pub fn quantize_nvfp4_gptq(
    w0: &[f32],
    rows: usize,
    k: usize,
    h_raw: &Array2<f64>,
    name: &str,
) -> Result<GptqPackOutcome, VindexError> {
    quantize_nvfp4_gptq_owned(w0, rows, k, h_raw.clone(), name)
}

/// Consuming path for calibrated REPRESENT: no second raw Hessian copy.
pub(crate) fn quantize_nvfp4_gptq_owned(
    w0: &[f32],
    rows: usize,
    k: usize,
    h_raw: Array2<f64>,
    name: &str,
) -> Result<GptqPackOutcome, VindexError> {
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the NVFP4 \
             {NVFP4_GROUP_ELEMS}-element group"
        )));
    }
    if w0.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            w0.len()
        )));
    }
    if h_raw.nrows() != k || h_raw.ncols() != k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: calibration Hessian is {}x{}, expected {k}x{k}",
            h_raw.nrows(),
            h_raw.ncols()
        )));
    }

    let groups = k / NVFP4_GROUP_ELEMS;
    // Frozen from W0, before any compensation. This is nearest-v1's own
    // function — not a reimplementation — so the tensor scale byte-
    // identity claim below holds by construction, not by a separately
    // maintained formula that could drift from it.
    let tensor_scale = tensor_scale_for(w0);

    let site = SiteHessian::from_raw(h_raw);
    let reduced_h = site.reduced();
    let ridge = site.damping_ridge();
    let alive = site.alive().to_vec();
    let dead_columns = site.dead().len();
    drop(site); // raw H is released before factorization workspace is allocated
    let plan = EliminationPlan::build(&reduced_h, ridge)?;
    drop(reduced_h);

    let mut packed = vec![0u8; rows * groups * NVFP4_GROUP_BYTES];
    let mut scales = vec![0u8; rows * groups];
    let mut saturated_elements = 0usize;
    let mut nearest_scratch = vec![0u8; groups * NVFP4_GROUP_BYTES];

    for (row, row_values) in w0.chunks_exact(k).enumerate() {
        let row_scales = &mut scales[row * groups..(row + 1) * groups];
        // Nearest-v1's own per-row pass gives two things at once: the
        // frozen scale bytes this function must reproduce exactly, and
        // nearest's own codes — which every dead column below simply
        // keeps, satisfying "encode using ordinary fixed-grid nearest
        // E2M1" without a separate code path for it.
        quantize_row_into(row_values, tensor_scale, &mut nearest_scratch, row_scales);
        let mut row_codes = unpack_nibbles(&nearest_scratch);

        if !alive.is_empty() {
            let w0_alive: Vec<f64> = alive.iter().map(|&j| f64::from(row_values[j])).collect();
            let steps: Vec<f32> = alive
                .iter()
                .map(|&j| {
                    let g = j / NVFP4_GROUP_ELEMS;
                    tensor_scale * e4m3_to_f32(row_scales[g])
                })
                .collect();
            let result = plan.eliminate_row(&w0_alive, &steps);
            for (idx, &j) in alive.iter().enumerate() {
                row_codes[j] = result.codes[idx];
                if result.saturated[idx] {
                    saturated_elements += 1;
                }
            }
        }

        let row_packed = pack_nibbles(&row_codes);
        packed[row * groups * NVFP4_GROUP_BYTES..(row + 1) * groups * NVFP4_GROUP_BYTES]
            .copy_from_slice(&row_packed);
    }

    Ok(GptqPackOutcome {
        matrix: Nvfp4Matrix {
            packed,
            scales,
            tensor_scale,
        },
        dead_columns,
        alive_columns: alive.len(),
        saturated_elements,
        total_elements: rows * k,
    })
}

#[cfg(test)]
#[path = "tests/pack.rs"]
mod tests;
