//! **NVFP4 × Q8** (NVFP4-Q8-1): the stored NVFP4 pack against a Q8
//! activation, multiplied in integers.
//!
//! The weight side is exact. E2M1's magnitudes doubled are the integers
//! `{0, 1, 2, 3, 4, 6, 8, 12}`, so a 16-entry table turns every 4-bit code
//! into an int8 with no rounding at all, and the halving is folded back
//! into the group's scale. What this kernel changes against
//! [`super::kernels::FusedNvfp4`] is ONLY the activation — quantised to
//! int8 once per slab, one scale per NVFP4 group — and the accumulation,
//! which becomes an exact int32 sum per group.

use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

use super::super::backend::Nvfp4Activation;
use super::integer::quantise_activation_blocked;
use super::kernels::{e4m3_steps, E2M1_DOUBLED};
use super::projector::{CpuParallelism, DenseProjector, WeightRows};

/// How many activation elements share one Q8 scale: exactly one NVFP4
/// group. A group already pays one f32 multiply for its E4M3 scale, so
/// folding the activation's scale into the same multiply costs nothing,
/// and no coarser block could be cheaper while being less exact.
pub const NVFP4_Q8_ACTIVATION_BLOCK: usize = NVFP4_GROUP_ELEMS;

/// The fused NVFP4 × Q8 kernel.
///
/// [`CpuParallelism::ExternalPool`] for [`super::kernels::FusedNvfp4`]'s
/// reason: it computes the rows it is handed and spawns nothing.
/// Lossy in the ACTIVATION by declaration; its fidelity contract is
/// NVFP4-Q8-1's F verdict, judged against `FusedNvfp4` on the same bytes.
pub struct FusedNvfp4Q8;

impl DenseProjector for FusedNvfp4Q8 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Nvfp4 {
            packed,
            scales,
            tensor_scale,
            activation: Nvfp4Activation::Q8,
        } = weight_rows
        else {
            panic!("the NVFP4 x Q8 kernel consumes NVFP4 packs bound for a Q8 activation only");
        };
        let k = x.len();
        let groups = k / NVFP4_GROUP_ELEMS;
        // Settled before dispatch, as for `FusedNvfp4`: a mismatch is a
        // bug in the slab cut, not a runtime condition to absorb.
        assert!(
            k.is_multiple_of(NVFP4_GROUP_ELEMS)
                && packed.len() == out.len() * groups * NVFP4_GROUP_BYTES
                && scales.len() == out.len() * groups,
            "nvfp4 slab geometry does not describe [{}, {k}]",
            out.len()
        );
        let (qx, xs) = quantise_activation_blocked(x, NVFP4_Q8_ACTIVATION_BLOCK);
        let steps = e4m3_steps();
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = nvfp4_q8_row_dot(
                &packed[row * groups * NVFP4_GROUP_BYTES..][..groups * NVFP4_GROUP_BYTES],
                &scales[row * groups..][..groups],
                tensor_scale,
                steps,
                &qx,
                &xs,
            );
        }
    }
}

/// One group's 16 codes as the int8 the integer dot consumes: DOUBLED
/// E2M1 values, in element order (byte `b` carries element `2b` in its
/// low nibble and `2b + 1` in its high one).
///
/// `code as f32 * (0.5 * tensor_scale * step)` is exactly the reference
/// decoder's value: doubling and halving are exact in binary floating
/// point, so the only rounding either side performs is the same one.
pub fn nvfp4_group_codes(group: &[u8]) -> [i8; NVFP4_GROUP_ELEMS] {
    let _ = (group, E2M1_DOUBLED);
    todo!("NVFP4-Q8-1: the decode lands after its witness is RED")
}

/// One row against the quantised activation.
#[inline]
fn nvfp4_q8_row_dot(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    steps: &[f32; 256],
    qx: &[i8],
    xs: &[f32],
) -> f32 {
    let _ = (packed, scales, tensor_scale, steps, qx, xs);
    todo!("NVFP4-Q8-1: the row dot lands after its witnesses are RED")
}

/// **The DEFINITION** the vector path must agree with: per group, the
/// EXACT int32 `Σ code_i · qx_i`, scaled once by `e4m3(scale) · xs_g`;
/// the tensor scale, halved to undo the doubling, once per row.
pub fn nvfp4_q8_row_portable(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    steps: &[f32; 256],
    qx: &[i8],
    xs: &[f32],
) -> f32 {
    let _ = (packed, scales, tensor_scale, steps, qx, xs);
    todo!("NVFP4-Q8-1: the definition lands after its witnesses are RED")
}
