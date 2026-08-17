//! Format-aware matrix-operand loading for the plan executor.
//!
//! The interpreter asks the backend which [`WeightFormat`] it computes
//! in and loads every matrix operand through [`load_weight`]; backends
//! receive slices, never operand references, exactly as before. The f16
//! path exists for device residency: a device buffer cache keyed by
//! `(pointer, length)` sees the same allocation on every call and keeps
//! the weight resident instead of re-uploading it per forward.
//!
//! **The bf16 → f16 conversion is exact for every normal-range value.**
//! bf16 carries 7 mantissa bits and f16 carries 10, so any bf16 value
//! whose magnitude lies in f16's normal range converts without rounding.
//! Overflow (|x| ≥ 65520, unrepresentable in f16) fails closed naming
//! the tensor — it would silently become infinity. Values below f16's
//! normal range land on subnormals and may round in the last bits; that
//! tail is a bounded realisation choice, and the parity gates against
//! the f32 backends and the upstream trace are its judge.

use super::backend::{WeightFormat, WeightSlice};
use super::operands::OperandStore;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;
use larql_models::quant::mxfp4::{e8m0_to_f32, MXFP4_TABLE};

/// Alignment (and length granularity) of f16 weight allocations:
/// the Apple-GPU page size. A page-aligned, page-multiple allocation
/// lets a Metal device wrap the memory zero-copy instead of copying it
/// into a private buffer; any other device simply sees ordinary bytes.
pub const DEVICE_PAGE_ALIGN: usize = 16384;

/// Safetensors dtypes this loader can narrow to f16. bf16 converts
/// exactly (normal range); f32 rounds to nearest-even.
const DTYPE_BF16: &str = "BF16";
const DTYPE_F32: &str = "F32";

/// f16 exponent field width and bias.
const F16_EXP_BITS: u32 = 5;
const F16_EXP_BIAS: i32 = 15;
/// f32 exponent bias.
const F32_EXP_BIAS: i32 = 127;
/// f32 mantissa width minus f16 mantissa width: the truncation shift.
const MANTISSA_SHIFT: u32 = 13;

/// A page-aligned, page-multiple, zero-padded byte buffer.
///
/// [`AlignedBytes::as_slice`] returns the *padded* slice on purpose:
/// callers hand the whole allocation to a device so the buffer length
/// stays page-multiple; matrix geometry always travels separately.
#[derive(Debug)]
pub struct AlignedBytes {
    ptr: std::ptr::NonNull<u8>,
    /// Allocation length — `logical` rounded up to the page.
    padded: usize,
    /// Meaningful bytes at the front of the allocation.
    logical: usize,
}

// The buffer is plain owned bytes; nothing about the raw pointer ties
// it to a thread.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate a zeroed, page-aligned buffer holding `logical` bytes.
    pub fn zeroed(logical: usize) -> Self {
        let padded = logical.div_ceil(DEVICE_PAGE_ALIGN).max(1) * DEVICE_PAGE_ALIGN;
        let layout = std::alloc::Layout::from_size_align(padded, DEVICE_PAGE_ALIGN)
            .expect("page-aligned layout is always valid");
        // SAFETY: layout has non-zero size (padded >= one page).
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw).unwrap_or_else(|| {
            std::alloc::handle_alloc_error(layout);
        });
        Self {
            ptr,
            padded,
            logical,
        }
    }

    /// A page-aligned copy of `bytes` — how a natively stored quantised
    /// operand (an MXFP4 expert's blocks or scales) is bound without a
    /// numeric transform: the bytes are the checkpoint's, only the
    /// alignment is ours.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut aligned = Self::zeroed(bytes.len());
        aligned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        aligned
    }

    /// The full padded allocation — page-aligned pointer, page-multiple
    /// length, zero beyond `logical_len`.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is `padded` bytes, initialised (zeroed
        // at alloc, fronts overwritten by the converter).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.padded) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.padded) }
    }

    /// Meaningful bytes at the front of the allocation.
    pub fn logical_len(&self) -> usize {
        self.logical
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.padded, DEVICE_PAGE_ALIGN)
            .expect("layout validated at allocation");
        // SAFETY: allocated with exactly this layout in `zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// One loaded matrix operand, owning its bytes in the format the
/// backend declared.
#[derive(Debug)]
pub enum LoadedWeight {
    F32(Vec<f32>),
    F16(AlignedBytes),
    Mxfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
    },
    Nvfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
        tensor_scale: f32,
    },
}

impl LoadedWeight {
    /// The borrowed view a call struct carries.
    pub fn slice(&self) -> WeightSlice<'_> {
        match self {
            LoadedWeight::F32(w) => WeightSlice::F32(w),
            LoadedWeight::F16(b) => WeightSlice::F16(b.as_slice()),
            LoadedWeight::Mxfp4 { packed, scales } => WeightSlice::Mxfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
            },
            LoadedWeight::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => WeightSlice::Nvfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
                tensor_scale: *tensor_scale,
            },
        }
    }
}

/// Load one matrix operand in `format`, through the closure-verified
/// path only.
pub fn load_weight(
    store: &OperandStore,
    operand: &OperandRef,
    format: WeightFormat,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(store.load(operand)?)),
        WeightFormat::Mxfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            let values = store.load(operand)?;
            quantize_mxfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::Nvfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            let values = store.load(operand)?;
            quantize_nvfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::F16 => {
            let raw = store.load_raw(operand)?;
            match raw.dtype.as_str() {
                DTYPE_BF16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                DTYPE_F32 => Ok(LoadedWeight::F16(f32_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                other => Err(VindexError::Parse(format!(
                    "tensor `{}`: no judged f16 narrowing for dtype `{other}`",
                    operand.tensor
                ))),
            }
        }
    }
}

/// Values converted per parallel work item — large enough that the
/// per-chunk overhead vanishes against a 30B-parameter conversion.
const NARROW_CHUNK_VALUES: usize = 1 << 18;

/// Convert little-endian bf16 bytes to little-endian f16 bytes in a
/// page-aligned buffer. Fails closed on overflow — a weight f16 cannot
/// hold would silently become infinity and poison every dot product it
/// touches.
pub fn bf16_bytes_to_f16(bytes: &[u8], name: &str) -> Result<AlignedBytes, VindexError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: bf16 payload has odd length {}",
            bytes.len()
        )));
    }
    narrow_parallel(bytes, 2, name, |pair| {
        let bf16 = u16::from_le_bytes([pair[0], pair[1]]);
        bf16_to_f16(bf16).ok_or(f32::from_bits(u32::from(bf16) << 16))
    })
}

/// Convert little-endian f32 bytes to little-endian f16 bytes in a
/// page-aligned buffer, rounding to nearest-even. Fails closed on
/// finite overflow, like the bf16 path.
pub fn f32_bytes_to_f16(bytes: &[u8], name: &str) -> Result<AlignedBytes, VindexError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: f32 payload length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    narrow_parallel(bytes, 4, name, |quad| {
        let value = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
        f32_to_f16_rne(value).ok_or(value)
    })
}

/// The shared conversion drive: chunked and parallel — a 30B-parameter
/// model narrows in seconds instead of minutes — with each value
/// converted independently, so parallelism reorders nothing.
/// `convert`'s error is the offending value, reported with the element
/// index of the first failing chunk.
fn narrow_parallel(
    src: &[u8],
    in_width: usize,
    name: &str,
    convert: impl (Fn(&[u8]) -> Result<u16, f32>) + Sync,
) -> Result<AlignedBytes, VindexError> {
    use rayon::prelude::*;

    let values = src.len() / in_width;
    let mut out = AlignedBytes::zeroed(values * 2);
    let dst = out.as_mut_slice();
    dst[..values * 2]
        .par_chunks_mut(NARROW_CHUNK_VALUES * 2)
        .zip(src.par_chunks(NARROW_CHUNK_VALUES * in_width))
        .enumerate()
        .try_for_each(|(chunk_index, (d, s))| {
            for (offset, value) in s.chunks_exact(in_width).enumerate() {
                let f16 = convert(value).map_err(|overflowing| {
                    VindexError::Parse(format!(
                        "tensor `{name}`: value {overflowing} at element {} overflows f16 — \
                         refusing to saturate a weight to infinity",
                        chunk_index * NARROW_CHUNK_VALUES + offset,
                    ))
                })?;
                d[offset * 2..offset * 2 + 2].copy_from_slice(&f16.to_le_bytes());
            }
            Ok::<(), VindexError>(())
        })?;
    Ok(out)
}

/// One f32 value to f16 bits, round-to-nearest-even. `None` on finite
/// overflow; infinities and NaNs pass through as themselves.
fn f32_to_f16_rne(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        let f16_mant: u16 = if mant == 0 { 0 } else { 0x200 };
        return Some(sign | 0x7C00 | f16_mant);
    }
    let new_exp = exp - F32_EXP_BIAS + F16_EXP_BIAS;
    if new_exp <= 0 {
        // Subnormal or underflow. Below 2^-25 even rounding cannot
        // reach the smallest subnormal.
        if new_exp < -10 {
            return Some(sign);
        }
        let full = mant | 0x80_0000;
        let shift = (MANTISSA_SHIFT as i32 + 1 - new_exp) as u32;
        let kept = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round_up = rem > half || (rem == half && kept & 1 == 1);
        // A carry out of the subnormal mantissa lands exactly on the
        // smallest normal encoding, which is correct.
        return Some(sign | (kept + u32::from(round_up)) as u16);
    }
    let kept = mant >> MANTISSA_SHIFT;
    let rem = mant & ((1 << MANTISSA_SHIFT) - 1);
    let half = 1u32 << (MANTISSA_SHIFT - 1);
    let round_up = rem > half || (rem == half && kept & 1 == 1);
    let encoded = ((new_exp as u32) << 10) + kept + u32::from(round_up);
    if encoded >= 0x7C00 {
        return None; // rounded past the largest finite f16
    }
    Some(sign | encoded as u16)
}

/// One bf16 value to f16 bits. `None` on finite overflow; infinities
/// and NaNs pass through as themselves (they are already exceptional
/// in the source and convert exactly).
fn bf16_to_f16(bf16: u16) -> Option<u16> {
    let sign = bf16 & 0x8000; // f16's sign occupies the same bit
    let exp = ((bf16 >> 7) & 0xFF) as i32;
    let mant = u32::from(bf16 & 0x7F); // 7 explicit mantissa bits

    if exp == 0 {
        // bf16 zero or subnormal (< 2^-126): far below f16's subnormal
        // floor, so it is exactly ±0 in f16.
        return Some(sign);
    }
    if exp == 0xFF {
        // Infinity / NaN: map onto f16's exceptional encodings,
        // preserving a set mantissa bit so NaN stays NaN.
        let f16_mant = if mant == 0 { 0 } else { 0x200 };
        return Some(sign | 0x7C00 | f16_mant as u16);
    }
    let new_exp = exp - F32_EXP_BIAS + F16_EXP_BIAS;
    let max_exp = (1 << F16_EXP_BITS) - 1;
    if new_exp >= max_exp {
        return None; // finite value too large for f16
    }
    // bf16's 7 explicit mantissa bits sit in the top of f32's 23; f16
    // keeps the top 10, so normal-range conversion is exact.
    let wide_mant = mant << 16; // position as f32 mantissa
    if new_exp <= 0 {
        // f16 subnormal: shift the implicit one back in. Bits shifted
        // out truncate — the documented inexact tail.
        let shift = MANTISSA_SHIFT + (1 - new_exp) as u32;
        if shift >= 24 {
            return Some(sign); // underflows all the way to zero
        }
        let sub_mant = ((wide_mant | 0x80_0000) >> shift) as u16;
        return Some(sign | sub_mant);
    }
    Some(sign | ((new_exp as u16) << 10) | (wide_mant >> MANTISSA_SHIFT) as u16)
}

/// MXFP4 group geometry, matching the kernel's layout contract exactly:
/// per row, `k/32` groups of 16 packed bytes (lo nibble first) plus one
/// e8m0 scale byte each.
const MXFP4_GROUP_ELEMS: usize = 32;
const MXFP4_GROUP_BYTES: usize = 16;
/// e2m1's largest magnitude; the shared exponent is chosen so the
/// group's max maps at or below it, saturating the rare overshoot.
const MXFP4_MAX_MAG: f32 = 6.0;
/// Exponent of [`MXFP4_MAX_MAG`]'s leading bit: `floor(log2(6)) = 2`.
const MXFP4_EMAX: i32 = 2;

/// Quantise one `[rows, k]` f32 matrix to MXFP4 — the OCP microscaling
/// rule: per 32-element group, shared scale `2^(floor(log2(max|x|)) -
/// 2)` as e8m0, elements rounded to the nearest e2m1 grid value
/// (ties to the even code index), saturating at ±6.
///
/// A lossy realisation by construction; the parity gates against the
/// f16/f32 anchors and the upstream trace are its judge. Layout is the
/// kernel's, and the nibble-order control in the tests pins it against
/// the independent `larql-models` decoder.
pub fn quantize_mxfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    if !k.is_multiple_of(MXFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the MXFP4 32-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / MXFP4_GROUP_ELEMS;
    let mut packed = AlignedBytes::zeroed(rows * groups * MXFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        packed_dst[..rows * groups * MXFP4_GROUP_BYTES]
            .par_chunks_mut(groups * MXFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                for g in 0..groups {
                    let group = &row_values[g * MXFP4_GROUP_ELEMS..(g + 1) * MXFP4_GROUP_ELEMS];
                    let max_abs = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale_byte = if max_abs == 0.0 {
                        0u8 // decodes to 0.0; all codes zero
                    } else {
                        let exponent = max_abs.log2().floor() as i32 - MXFP4_EMAX;
                        (exponent + 127).clamp(1, 254) as u8
                    };
                    row_scales[g] = scale_byte;
                    let scale = e8m0_to_f32(scale_byte);
                    let inv = if scale == 0.0 { 0.0 } else { scale.recip() };
                    let bytes = &mut row_packed[g * MXFP4_GROUP_BYTES..(g + 1) * MXFP4_GROUP_BYTES];
                    for (b, pair) in group.chunks_exact(2).enumerate() {
                        let lo = nearest_mxfp4_code(pair[0] * inv);
                        let hi = nearest_mxfp4_code(pair[1] * inv);
                        bytes[b] = lo | (hi << 4);
                    }
                }
            });
    }
    Ok(LoadedWeight::Mxfp4 { packed, scales })
}

/// Quantise one `[rows, k]` f32 matrix to NVFP4 into page-aligned
/// buffers, delegating the numerics to `larql_models::quant::nvfp4` so
/// the format has exactly one definition — the CPU reference, this
/// loader, and the Metal kernel all read that module's contract.
///
/// The only thing added here is residency: the same page-aligned
/// allocation MXFP4 uses, so a device can wrap the buffers zero-copy.
pub fn quantize_nvfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    use larql_models::quant::nvfp4::{
        quantize_row_into, tensor_scale_for, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
    };
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the NVFP4 \
             {NVFP4_GROUP_ELEMS}-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    // The tensor scale is a property of the whole matrix, so it is chosen
    // once before any row is encoded — rows cannot each pick their own
    // and still decode under one shared scale.
    let tensor_scale = tensor_scale_for(values);
    let mut packed = AlignedBytes::zeroed(rows * groups * NVFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        // Rows are independent given the tensor scale, so the parallelism
        // lives here while the numerics stay in one place
        // (`quant::nvfp4::quantize_row_into`), shared with the CPU
        // reference the kernel is judged against.
        packed_dst[..rows * groups * NVFP4_GROUP_BYTES]
            .par_chunks_mut(groups * NVFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                quantize_row_into(row_values, tensor_scale, row_packed, row_scales);
            });
    }
    Ok(LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    })
}

/// The e2m1 code nearest to `v` (ties to the even code index),
/// saturating at ±6.
fn nearest_mxfp4_code(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0 };
    let mag = v.abs().min(MXFP4_MAX_MAG);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (code, value) in MXFP4_TABLE.iter().enumerate().take(8) {
        let err = (mag - value).abs();
        if err < best_err || (err == best_err && code.is_multiple_of(2)) {
            best = code as u8;
            best_err = err;
        }
    }
    if best == 0 {
        0 // ±0 collapse to +0: the table's -0.0 encodes nothing extra
    } else {
        sign | best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::cpu::ops::q4_common::f16_to_f32;

    fn bf16_of(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn f32_of_bf16(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// Every normal-range bf16 value must convert to f16 exactly.
    #[test]
    fn normal_range_conversion_is_exact() {
        for value in [1.0f32, -2.5, 0.007812, 1023.0, -65504.0, 3.87, 1e-4] {
            let bf16 = bf16_of(value);
            let f16 = bf16_to_f16(bf16).expect("in range");
            assert_eq!(
                f16_to_f32(f16),
                f32_of_bf16(bf16),
                "value {value} must round-trip exactly"
            );
        }
    }

    /// Finite overflow fails closed rather than saturating to infinity.
    #[test]
    fn finite_overflow_is_refused() {
        assert_eq!(bf16_to_f16(bf16_of(65536.0)), None);
        assert_eq!(bf16_to_f16(bf16_of(-1e6)), None);
        let err = bf16_bytes_to_f16(&bf16_of(1e5).to_le_bytes(), "w").unwrap_err();
        assert!(err.to_string().contains("overflows f16"), "{err}");
    }

    /// Exceptional values stay exceptional; zeros stay signed zeros.
    #[test]
    fn zeros_infinities_and_nans_convert_to_themselves() {
        assert_eq!(bf16_to_f16(bf16_of(0.0)), Some(0x0000));
        assert_eq!(bf16_to_f16(bf16_of(-0.0)), Some(0x8000));
        let inf = bf16_to_f16(bf16_of(f32::INFINITY)).unwrap();
        assert_eq!(f16_to_f32(inf), f32::INFINITY);
        let nan = bf16_to_f16(bf16_of(f32::NAN)).unwrap();
        assert!(f16_to_f32(nan).is_nan());
    }

    /// The subnormal tail truncates but stays within one f16 subnormal
    /// step of the true value, and deep underflow lands on zero.
    #[test]
    fn subnormal_tail_is_bounded_and_underflow_is_zero() {
        let tiny = 3.0e-5f32; // below f16's normal floor of ~6.1e-5
        let f16 = bf16_to_f16(bf16_of(tiny)).unwrap();
        let back = f16_to_f32(f16);
        let step = 5.96e-8; // one f16 subnormal quantum
        assert!((back - f32_of_bf16(bf16_of(tiny))).abs() <= step);
        assert_eq!(bf16_to_f16(bf16_of(1e-30)), Some(0));
    }

    /// f32 → f16 rounds to nearest, ties to even, and refuses overflow.
    #[test]
    fn f32_narrowing_rounds_to_nearest_even_and_refuses_overflow() {
        // Exactly representable values pass through unchanged.
        for value in [1.0f32, -0.75, 1536.0, 6.1035156e-5] {
            let f16 = f32_to_f16_rne(value).unwrap();
            assert_eq!(f16_to_f32(f16), value, "{value} is f16-exact");
        }
        // A non-dyadic value rounds to the nearer f16 neighbour: 0.05
        // sits between 0.04998779 (1.22e-5 away) and 0.05001831
        // (1.83e-5 away).
        assert_eq!(f16_to_f32(f32_to_f16_rne(0.05).unwrap()), 0.049987793);
        // 1 + 2^-11 sits exactly between 1.0 and the next f16 up
        // (1 + 2^-10); ties go to the even mantissa, which is 1.0.
        let tie = 1.0 + 2f32.powi(-11);
        assert_eq!(f16_to_f32(f32_to_f16_rne(tie).unwrap()), 1.0);
        // Just above the tie rounds up.
        let above = 1.0 + 2f32.powi(-11) + 2f32.powi(-13);
        assert_eq!(
            f16_to_f32(f32_to_f16_rne(above).unwrap()),
            1.0 + 2f32.powi(-10)
        );
        // Values that round past the largest finite f16 are refused.
        assert_eq!(f32_to_f16_rne(65520.0), None);
        assert!(f32_to_f16_rne(65503.0).is_some());
        let err = f32_bytes_to_f16(&1e6f32.to_le_bytes(), "w").unwrap_err();
        assert!(err.to_string().contains("overflows f16"), "{err}");
    }

    /// Grid-exact values survive MXFP4 quantisation unchanged, and the
    /// packed bytes decode identically through the **independent**
    /// `larql-models` decoder — the layout (lo nibble first, per-row
    /// group order, e8m0 scales) is pinned against the code that has
    /// already read real GPT-OSS checkpoints, not against this
    /// quantiser's own assumptions.
    #[test]
    fn mxfp4_grid_values_round_trip_through_the_independent_decoder() {
        // One row, 32 elements: max 6.0 → shared exponent 0 → scale 1.0,
        // every value on the e2m1 grid.
        let mut row = vec![0.0f32; 32];
        let grid = [0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.5, -6.0];
        row[..grid.len()].copy_from_slice(&grid);
        let LoadedWeight::Mxfp4 { packed, scales } = quantize_mxfp4(&row, 1, 32, "w").unwrap()
        else {
            panic!("quantiser must produce the mxfp4 variant");
        };
        assert_eq!(scales.as_slice()[0], 127, "max 6.0 → 2^0 scale");
        let decoded = larql_models::quant::mxfp4::dequantize_expert(
            &packed.as_slice()[..16],
            &scales.as_slice()[..1],
            1,
            1,
        )
        .unwrap();
        assert_eq!(&decoded[..], &row[..], "grid values must survive exactly");
    }

    /// Off-grid values land within one half-step of the grid, and a
    /// group's error is bounded by its scale (2·scale at saturation).
    #[test]
    fn mxfp4_error_is_bounded_by_the_group_scale() {
        let row: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
        let LoadedWeight::Mxfp4 { packed, scales } = quantize_mxfp4(&row, 1, 64, "w").unwrap()
        else {
            panic!("quantiser must produce the mxfp4 variant");
        };
        let decoded = larql_models::quant::mxfp4::dequantize_expert(
            &packed.as_slice()[..32],
            &scales.as_slice()[..2],
            1,
            2,
        )
        .unwrap();
        for (group, (xs, ds)) in row.chunks(32).zip(decoded.chunks(32)).enumerate() {
            let scale = e8m0_to_f32(scales.as_slice()[group]);
            for (x, d) in xs.iter().zip(ds) {
                assert!(
                    (x - d).abs() <= scale * 2.0 + f32::EPSILON,
                    "group {group}: |{x} - {d}| exceeds 2·scale ({scale})"
                );
            }
        }
    }

    /// Group misalignment and shape mismatches are refused, not padded.
    #[test]
    fn mxfp4_quantiser_fails_closed_on_bad_geometry() {
        let err = quantize_mxfp4(&[0.0; 40], 1, 40, "w").unwrap_err();
        assert!(err.to_string().contains("32-element group"), "{err}");
        let err = quantize_mxfp4(&[0.0; 32], 2, 32, "w").unwrap_err();
        assert!(err.to_string().contains("do not fill"), "{err}");
    }

    /// An all-zero group takes the zero-scale sentinel and decodes to
    /// exact zeros.
    #[test]
    fn mxfp4_zero_group_uses_the_zero_scale_sentinel() {
        let LoadedWeight::Mxfp4 { packed, scales } =
            quantize_mxfp4(&[0.0f32; 32], 1, 32, "w").unwrap()
        else {
            panic!("quantiser must produce the mxfp4 variant");
        };
        assert_eq!(scales.as_slice()[0], 0);
        assert!(packed.as_slice()[..16].iter().all(|&b| b == 0));
    }

    /// The parallel loader must produce **byte-identical** output to the
    /// single-definition reference in `quant::nvfp4`. The loader exists
    /// only for residency and thread-pool reasons; if it drifted, the
    /// Metal kernel would be judged against a CPU reference that no
    /// longer describes the bytes it is handed.
    #[test]
    fn the_parallel_nvfp4_loader_matches_the_reference_exactly() {
        // Awkward geometry on purpose: rows that do not divide evenly
        // across a pool, and a k spanning several groups.
        let (rows, k) = (37, 16 * 11);
        let values: Vec<f32> = (0..rows * k)
            .map(|i| ((i as f32) * 0.0137).sin() * (1.0 + (i % 7) as f32))
            .collect();

        let reference = larql_models::quant::nvfp4::quantize(&values, rows, k).unwrap();
        let LoadedWeight::Nvfp4 {
            packed,
            scales,
            tensor_scale,
        } = quantize_nvfp4(&values, rows, k, "w").unwrap()
        else {
            panic!("loader must produce the nvfp4 variant");
        };

        assert_eq!(tensor_scale, reference.tensor_scale);
        assert_eq!(
            &packed.as_slice()[..reference.packed.len()],
            &reference.packed[..],
            "packed codes must match the reference byte for byte"
        );
        assert_eq!(
            &scales.as_slice()[..reference.scales.len()],
            &reference.scales[..],
            "E4M3 scales must match the reference byte for byte"
        );
    }

    /// Geometry is refused by the loader too, not only by the codec.
    #[test]
    fn the_nvfp4_loader_fails_closed_on_bad_geometry() {
        let err = quantize_nvfp4(&[0.0; 40], 1, 40, "w").unwrap_err();
        assert!(err.to_string().contains("16-element group"), "{err}");
        let err = quantize_nvfp4(&[0.0; 32], 3, 16, "w").unwrap_err();
        assert!(err.to_string().contains("do not fill"), "{err}");
    }

    /// The aligned buffer really is page-aligned, page-multiple, and
    /// zero beyond its logical length.
    #[test]
    fn aligned_bytes_meet_the_device_contract() {
        let converted = bf16_bytes_to_f16(&[0u8; 6], "w").unwrap();
        let slice = converted.as_slice();
        assert_eq!(slice.as_ptr() as usize % DEVICE_PAGE_ALIGN, 0);
        assert_eq!(slice.len() % DEVICE_PAGE_ALIGN, 0);
        assert_eq!(converted.logical_len(), 6);
        assert!(slice.iter().all(|&b| b == 0));
    }
}
