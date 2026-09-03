//! The ecosystem's own K-quant encoder, for comparative work only.
//!
//! ## Why this exists
//!
//! LARQL's native K-quant encoders emit legal, correctly-laid-out bytes
//! — proven bit-for-bit against a foreign fixture — but they do not
//! choose the same values ggml does, and at Q6_K they reconstruct ~11%
//! worse (measured: Q8_0 0.9967, Q4_K 1.0326, Q6_K 1.1146 RMS ratio).
//!
//! REPRESENT's thesis is that it wins the **selection** race, not the
//! codec race. That is only isolable if the codec is held constant
//! against the ecosystem's. With a bit-width-dependent encoder deficit,
//! a matched-byte comparison against a llama.cpp-derived artifact would
//! measure *allocation minus a codec handicap*, and a loss would be
//! unattributable between the two.
//!
//! So comparative artifacts are produced with **ggml's encoder**, and
//! the only independent variable left is where precision is spent.
//!
//! ## Production, not consumption
//!
//! This is a dependency of **artifact production**, never of artifact
//! consumption. The VINDEX3 reader and runtime keep their own native
//! decoders and acquire no llama.cpp dependency — the foreign fixture
//! already established that LARQL interprets ggml bytes exactly.
//!
//! ```text
//! compile a comparative artifact   ggml encoder   (this module)
//! read / execute any artifact      LARQL decoder  (always)
//! ```
//!
//! ## The control is architectural, not statistical
//!
//! There is no runtime switch. If this module is compiled in, K-quant
//! compilation uses it; if it is not, compilation uses the native
//! encoders and says so in the recorded [`EncoderRecipe`]. A future
//! regression in `quantize_q6_k` therefore cannot perturb a comparative
//! campaign, because that campaign never called it.
//!
//! ## Building
//!
//! Off by default. Enable `--features reference-encoder` and point
//! `LARQL_GGML_LIB_DIR` at a built llama.cpp's library directory. The
//! upstream revision is recorded in the artifact's encoder provenance,
//! because for a reference encoder the upstream identity genuinely
//! determines the bytes.

use std::ffi::c_void;

use super::kquant::KQuant;
use crate::error::VindexError;

/// The pinned upstream this build links against.
///
/// Unlike a LARQL recipe name — which the module's own docs say must
/// never be a build id, because a bug fix that changes no chosen value
/// should not churn the identity — an *external* reference encoder's
/// upstream revision does determine the chosen values. So it is part of
/// the provenance, and a campaign that pins it can be reproduced.
///
/// Read from the environment at build time so it cannot drift from the
/// library actually linked.
pub const PINNED_UPSTREAM: &str = match option_env!("LARQL_GGML_REVISION") {
    Some(rev) => rev,
    // Not a silent default: a comparative artifact compiled without a
    // stated upstream says so, and its provenance reads `unpinned`.
    None => "unpinned",
};

unsafe extern "C" {
    /// ggml's own quantiser. `nrows` × `n_per_row` because K-quant
    /// blocks are framed per row — the same fact `KQuant::plan`
    /// enforces, and the reason the row geometry is passed through here
    /// rather than flattened.
    fn ggml_quantize_chunk(
        type_: i32,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
}

/// Encode with ggml, preserving the tensor's row framing.
///
/// `values` is the whole tensor in row-major order; `row_len` is its
/// innermost dimension.
///
/// The geometry is passed through rather than flattened because it
/// decides **legality**: a row that is not a whole number of blocks has
/// no encoding, and ggml would otherwise frame a block across two rows.
/// For a tensor that IS block-aligned the framing does not change the
/// bytes — `[1, 512]` and `[2, 256]` at Q6_K cover the same 256-value
/// windows in the same order — so this is a validity boundary, not a
/// byte-layout one. An earlier version of this doc claimed otherwise and
/// a test written to that claim failed against correct code.
pub fn encode(
    k: KQuant,
    values: &[f32],
    row_len: usize,
    tensor: &str,
) -> Result<Vec<u8>, VindexError> {
    if row_len == 0 || !values.len().is_multiple_of(row_len) {
        return Err(VindexError::Parse(format!(
            "tensor `{tensor}`: {} values do not divide into rows of {row_len}",
            values.len()
        )));
    }
    // The same rule `KQuant::plan` enforces: ggml frames blocks along the
    // row, so a row that is not a whole number of blocks has no legal
    // encoding — ggml would read past the row into its neighbour.
    if !row_len.is_multiple_of(k.elements_per_block) {
        return Err(VindexError::Parse(format!(
            "tensor `{tensor}`: row length {row_len} is not a whole number of {}-element \
             blocks, so {} rows would share a scale",
            k.elements_per_block, k.name
        )));
    }
    let rows = values.len() / row_len;
    // The same geometry the planner reserved, so the segment table and
    // the payload cannot disagree.
    let expect = k.encoded_len(values.len(), tensor)?;
    let mut out = vec![0u8; expect];

    // SAFETY: `src` has `rows * row_len` readable f32s and `dst` has
    // `expect` writable bytes, which is exactly what ggml writes for
    // this type at this geometry — asserted against the returned count
    // below. `imatrix` is null, meaning unweighted, which is what an
    // importance-matrix-free reference encode is.
    let wrote = unsafe {
        ggml_quantize_chunk(
            k.ggml_type as i32,
            values.as_ptr(),
            out.as_mut_ptr() as *mut c_void,
            0,
            rows as i64,
            row_len as i64,
            std::ptr::null(),
        )
    };
    if wrote != expect {
        return Err(VindexError::Parse(format!(
            "tensor `{tensor}`: ggml wrote {wrote} bytes for {}, geometry implies {expect} \
             — the linked ggml disagrees with this build's block table",
            k.name
        )));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "reference_encoder_tests.rs"]
mod tests;
