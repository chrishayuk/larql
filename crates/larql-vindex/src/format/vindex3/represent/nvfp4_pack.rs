//! The on-disk layout of a compiled NVFP4 tensor.
//!
//! A VINDEX3 segment describes a tensor by `dtype` + `shape`, and every
//! reader derives the byte layout from those two facts alone. This module is
//! that derivation for [`DTYPE_NVFP4`], so a compiled pack needs no runtime
//! inference — and no architecture registry — to be understood.
//!
//! ## Layout
//!
//! For `shape = [rows, k]` with `groups = k / 16`, one tensor's payload is
//! three contiguous regions:
//!
//! ```text
//! offset                              bytes                    meaning
//! 0                                   rows * groups * 8        E2M1 codes, lo nibble first
//! rows * groups * 8                   rows * groups            E4M3 group scales
//! rows * groups * 9                   4                        f32 tensor scale, LE
//! ```
//!
//! which is exactly [`larql_models::quant::nvfp4::stored_bytes`]. The two
//! scale levels and the element grid are NVFP4's, unchanged — see that
//! module for why both levels exist. Nothing here reinterprets them; this is
//! only where each region sits in a file.
//!
//! ## Why one table row, not three
//!
//! Codes, group scales and the tensor scale could each have been a segment
//! tensor. They are one row because they are one operand: the table would
//! otherwise carry three entries that must be kept consistent by a naming
//! convention, and a reader that resolved two of the three would produce
//! plausible garbage rather than an error. One row, one `len`, one offset —
//! and the split is arithmetic from `shape`.

use larql_models::quant::nvfp4::{stored_bytes, Nvfp4Matrix, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

use crate::error::VindexError;

/// Segment `dtype` label for a compiled NVFP4 tensor.
///
/// Distinct from the runtime's `WeightFormat::Nvfp4`: this names *stored
/// bytes*, that names an execution representation. A container declaring
/// this dtype is asserting the bytes are already in the grid below — no
/// load-time quantisation is owed.
pub const DTYPE_NVFP4: &str = "NVFP4";

/// Where each region of a packed tensor sits, derived from `shape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackLayout {
    pub rows: usize,
    pub k: usize,
    pub groups: usize,
    /// Byte length of the E2M1 code region, which starts at offset 0.
    pub packed_len: usize,
    /// Byte length of the E4M3 group-scale region.
    pub scales_len: usize,
    /// Total payload length — codes + scales + the f32 tensor scale.
    pub total_len: usize,
}

impl PackLayout {
    /// Derive the layout of `[rows, k]`, or say why the shape cannot hold
    /// one.
    ///
    /// Refuses rather than pads: NVFP4's own encoder refuses a `k` that is
    /// not a whole number of groups, and a padded row would decode to
    /// plausible garbage instead of failing.
    pub fn derive(shape: &[usize], name: &str) -> Result<Self, VindexError> {
        let [rows, k] = shape else {
            return Err(VindexError::Parse(format!(
                "tensor `{name}`: NVFP4 stores 2-D matrices; shape is {shape:?}"
            )));
        };
        let (rows, k) = (*rows, *k);
        if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
            return Err(VindexError::Parse(format!(
                "tensor `{name}`: k={k} is not a multiple of the NVFP4 \
                 {NVFP4_GROUP_ELEMS}-element group"
            )));
        }
        let groups = k / NVFP4_GROUP_ELEMS;
        let packed_len = rows * groups * NVFP4_GROUP_BYTES;
        let scales_len = rows * groups;
        Ok(Self {
            rows,
            k,
            groups,
            packed_len,
            scales_len,
            total_len: stored_bytes(rows, k),
        })
    }

    /// Offset of the E4M3 group-scale region.
    pub fn scales_offset(&self) -> usize {
        self.packed_len
    }

    /// Offset of the f32 tensor scale.
    pub fn tensor_scale_offset(&self) -> usize {
        self.packed_len + self.scales_len
    }

    /// Bits per weight this layout achieves — 4.5 for NVFP4's 16-element
    /// group (8 code bytes + 1 scale byte), plus the tensor scale amortised
    /// over the matrix.
    pub fn bits_per_weight(&self) -> f64 {
        if self.rows == 0 || self.k == 0 {
            return 0.0;
        }
        self.total_len as f64 * 8.0 / (self.rows as f64 * self.k as f64)
    }
}

/// Serialise a quantised matrix into its stored form.
///
/// The inverse of [`split`]; the two are pinned against each other so the
/// writer and every reader cannot drift.
pub fn encode(
    matrix: &Nvfp4Matrix,
    layout: &PackLayout,
    name: &str,
) -> Result<Vec<u8>, VindexError> {
    if matrix.packed.len() != layout.packed_len || matrix.scales.len() != layout.scales_len {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: quantised matrix is {}+{} bytes, layout expects {}+{}",
            matrix.packed.len(),
            matrix.scales.len(),
            layout.packed_len,
            layout.scales_len
        )));
    }
    let mut out = Vec::with_capacity(layout.total_len);
    out.extend_from_slice(&matrix.packed);
    out.extend_from_slice(&matrix.scales);
    out.extend_from_slice(&matrix.tensor_scale.to_le_bytes());
    debug_assert_eq!(out.len(), layout.total_len);
    Ok(out)
}

/// Borrow the three regions of a stored payload.
///
/// Returns borrows rather than a copy: the resident loader hands these
/// straight to the device, and a compiled representation exists precisely so
/// that nothing has to be rebuilt on the way.
pub fn split<'a>(
    payload: &'a [u8],
    layout: &PackLayout,
    name: &str,
) -> Result<(&'a [u8], &'a [u8], f32), VindexError> {
    if payload.len() != layout.total_len {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: NVFP4 payload is {} bytes, shape [{}, {}] implies {}",
            payload.len(),
            layout.rows,
            layout.k,
            layout.total_len
        )));
    }
    let packed = &payload[..layout.packed_len];
    let scales = &payload[layout.scales_offset()..layout.tensor_scale_offset()];
    let tail = &payload[layout.tensor_scale_offset()..];
    let tensor_scale = f32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    Ok((packed, scales, tensor_scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::nvfp4::quantize;

    fn matrix(rows: usize, k: usize) -> Nvfp4Matrix {
        let values: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.013)
            .collect();
        quantize(&values, rows, k).expect("aligned fixture")
    }

    #[test]
    fn layout_is_four_and_a_half_bits_per_weight() {
        // 16 elements -> 8 code bytes + 1 scale byte = 4.5 bits, the same
        // figure Q4_K reaches by a different route.
        let l = PackLayout::derive(&[1024, 2560], "w").unwrap();
        assert_eq!(l.groups, 160);
        assert_eq!(l.packed_len, 1024 * 160 * 8);
        assert_eq!(l.scales_len, 1024 * 160);
        assert_eq!(l.total_len, l.packed_len + l.scales_len + 4);
        // The +4 tensor scale is amortised over 2.6M weights, so it rounds
        // away — but it is counted, not ignored.
        assert!(
            (l.bits_per_weight() - 4.5).abs() < 1e-4,
            "{}",
            l.bits_per_weight()
        );
    }

    #[test]
    fn unaligned_k_is_refused_not_padded() {
        let err = PackLayout::derive(&[8, 24], "w").unwrap_err().to_string();
        assert!(err.contains("not a multiple"), "{err}");
    }

    #[test]
    fn non_2d_is_refused() {
        let err = PackLayout::derive(&[16], "w").unwrap_err().to_string();
        assert!(err.contains("2-D"), "{err}");
    }

    #[test]
    fn encode_split_round_trips() {
        let m = matrix(6, 64);
        let l = PackLayout::derive(&[6, 64], "w").unwrap();
        let bytes = encode(&m, &l, "w").unwrap();
        assert_eq!(bytes.len(), l.total_len);

        let (packed, scales, tensor_scale) = split(&bytes, &l, "w").unwrap();
        assert_eq!(packed, &m.packed[..]);
        assert_eq!(scales, &m.scales[..]);
        assert_eq!(tensor_scale.to_bits(), m.tensor_scale.to_bits());
    }

    #[test]
    fn a_truncated_payload_is_an_error_not_a_short_read() {
        let m = matrix(4, 32);
        let l = PackLayout::derive(&[4, 32], "w").unwrap();
        let bytes = encode(&m, &l, "w").unwrap();
        let err = split(&bytes[..bytes.len() - 1], &l, "w")
            .unwrap_err()
            .to_string();
        assert!(err.contains("implies"), "{err}");
    }

    #[test]
    fn regions_do_not_overlap_or_leave_gaps() {
        let l = PackLayout::derive(&[3, 48], "w").unwrap();
        assert_eq!(l.scales_offset(), l.packed_len);
        assert_eq!(l.tensor_scale_offset(), l.packed_len + l.scales_len);
        assert_eq!(l.total_len, l.tensor_scale_offset() + 4);
    }
}

/// The representation ABI a compiled pack was produced against.
///
/// `encoding: "NVFP4"` names a family, not a contract. Today's bytes are
/// "whatever `quantize_nvfp4` currently emits", and that function will be
/// improved — a better rounding rule, a GPTQ-style Hessian-aware encoder,
/// a different scale search. None of those may silently change what an
/// existing container *means*.
///
/// So a pack records the ABI it was compiled against, and a reader refuses
/// a revision it does not implement rather than decoding old bytes under
/// new rules. That is the difference between REPRESENT meaning "compile
/// using a specified representation ABI" and "whatever the current
/// function happens to produce".
///
/// The encoder may then improve freely: a new encoder that emits the same
/// ABI bumps nothing, because the *bytes' meaning* is unchanged and only
/// the choice of encoded values improved. A change to the grid, the group
/// size, the scale types or the region order bumps [`Self::REVISION`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodecIdentity {
    /// Representation family. Stable across revisions.
    pub family: String,
    /// ABI revision. A reader refuses what it does not implement.
    pub revision: u32,
    /// Elements sharing one group scale.
    pub group_elems: usize,
    /// Element grid.
    pub element: String,
    /// Per-group scale type.
    pub group_scale: String,
    /// Per-tensor scale type.
    pub tensor_scale: String,
    /// Region order within a tensor's payload.
    pub layout: String,
}

impl CodecIdentity {
    /// The ABI this build compiles and reads.
    ///
    /// Bump on any change to what the bytes *mean*: grid, group size,
    /// scale types, region order. Not on a better encoder that emits the
    /// same shape.
    pub const REVISION: u32 = 1;

    pub fn nvfp4_v1() -> Self {
        Self {
            family: "nvfp4".into(),
            revision: Self::REVISION,
            group_elems: NVFP4_GROUP_ELEMS,
            element: "e2m1".into(),
            group_scale: "e4m3".into(),
            tensor_scale: "f32-le".into(),
            layout: "codes|group_scales|tensor_scale".into(),
        }
    }

    /// Refuse a pack this build cannot decode under the rules it was
    /// written under.
    ///
    /// The decision is the codec registry's
    /// ([`super::codec::CodecRegistry::admit`]): every registered family
    /// is its OWN contract — `Q4_K` and `Q6_K` are different block
    /// layouts, and filing them under one family would let a reader that
    /// implements one accept the other's bytes on a revision match.
    pub fn admit(&self) -> Result<(), VindexError> {
        super::codec::CodecRegistry::builtin()
            .admit(self)
            .map(|_| ())
            .map_err(Into::into)
    }
}

/// Which compilation algorithm chose the encoded values.
///
/// [`CodecIdentity`] says how to *decode* a pack. This says how its numbers
/// were *chosen*, and the two are independent: nearest-rounding, a better
/// scale search and a GPTQ-style Hessian-aware encoder all emit the same
/// NVFP4 ABI, decode through the same kernel, and pick different values.
///
/// The distinction is load-bearing for one specific claim. "Persisted bytes
/// equal what the runtime would produce transiently" is only true against
/// the encoder that produced the artifact. Improve the encoder and an older
/// pack stays perfectly valid and perfectly decodable while no longer being
/// byte-reproducible by this build — which is a fact the parity gate has to
/// know rather than discover as a failure.
///
/// So, unlike a codec mismatch, an encoder mismatch is **not** a refusal.
/// Nothing is wrong with those bytes. Only the reproducibility claim
/// weakens.
///
/// Identities are stable recipe names, never build ids or git hashes: a
/// materially different algorithm earns a new name, and a bug fix that
/// changes no chosen value does not.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EncoderRecipe {
    /// Algorithm family — `nvfp4-nearest`, later `nvfp4-gptq`.
    pub algorithm: String,
    /// Revision within that family.
    pub revision: u32,
    /// Upstream identity, for an encoder this workspace does not own.
    ///
    /// Absent for LARQL's own encoders, whose identity is the recipe
    /// name. Present for a linked reference encoder, where the upstream
    /// revision is part of what determines the bytes — so it takes part
    /// in equality and therefore in the reproducibility claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl EncoderRecipe {
    /// Round each element to the nearest E2M1 grid point under the amax
    /// scale rule, element-independently. The encoder LARQL has today.
    pub fn nearest_v1() -> Self {
        Self {
            algorithm: "nvfp4-nearest".into(),
            revision: 1,
            source: None,
        }
    }

    /// Fixed-grid GPTQ: only E2M1 code nibbles may differ from
    /// `nearest_v1`; every scale byte is byte-identical by construction.
    /// Frozen in `bench/prompts/quality-bank-1/ENCODER-R4.md` (R4.1); the
    /// algorithm lives in [`super::gptq`].
    pub fn gptq_v1() -> Self {
        Self {
            algorithm: "nvfp4-gptq".into(),
            revision: 1,
            source: None,
        }
    }

    /// The recipe this build compiles with.
    pub fn current() -> Self {
        Self::nearest_v1()
    }

    /// LARQL's own K-quant encoders. Legal, correctly-laid-out bytes
    /// with LARQL's choice of values — measured against ggml at
    /// Q8_0 0.9967, Q4_K 1.0326, Q6_K 1.1146 reconstruction RMS.
    ///
    /// Fine for a self-contained artifact; NOT the encoder a comparison
    /// against a llama.cpp-derived artifact may use, because a
    /// bit-width-dependent codec deficit would be indistinguishable from
    /// an allocation difference.
    pub fn kquant_native_v1() -> Self {
        Self {
            algorithm: "kquant-larql-native".into(),
            revision: 1,
            source: None,
        }
    }

    /// The ecosystem's own encoder, linked and pinned.
    ///
    /// `source` carries the upstream revision because — unlike a LARQL
    /// recipe name, which this type's docs rightly keep free of build
    /// ids — an external encoder's upstream identity genuinely
    /// determines the chosen values. It participates in equality, so
    /// `is_reproducible_by_this_build` is honest across an upstream bump.
    pub fn kquant_ggml_reference(upstream: &str) -> Self {
        Self {
            algorithm: "kquant-ggml-reference".into(),
            revision: 1,
            source: Some(upstream.to_string()),
        }
    }

    /// `nvfp4-nearest-v1`, for reports and CLI output.
    pub fn name(&self) -> String {
        format!("{}-v{}", self.algorithm, self.revision)
    }

    /// Whether this build would reproduce a pack compiled by `self`
    /// byte-for-byte.
    ///
    /// The parity gate's precondition. `false` is not a defect — it means
    /// the artifact predates an encoder change and must be compared by
    /// behaviour rather than by bytes.
    pub fn is_reproducible_by_this_build(&self) -> bool {
        *self == Self::current()
    }
}

#[cfg(test)]
mod encoder_recipe_tests {
    use super::EncoderRecipe;

    #[test]
    fn gptq_v1_names_and_algorithm_are_distinct_from_nearest() {
        let gptq = EncoderRecipe::gptq_v1();
        let nearest = EncoderRecipe::nearest_v1();
        assert_eq!(gptq.name(), "nvfp4-gptq-v1");
        assert_eq!(nearest.name(), "nvfp4-nearest-v1");
        assert_ne!(gptq, nearest);
    }

    #[test]
    fn current_is_still_nearest_v1() {
        // This build's REPRESENT dispatch does not yet select gptq_v1 —
        // see `ENCODER-R4.md` Sequence step 8's remaining scope. When
        // that changes this assertion is exactly what should fail first.
        assert_eq!(EncoderRecipe::current(), EncoderRecipe::nearest_v1());
    }

    #[test]
    fn gptq_v1_is_not_reproducible_by_this_build() {
        // Accurate today: nothing routes REPRESENT through gptq_v1 yet,
        // so a pack claiming that recipe cannot be reproduced by this
        // build's `current()` dispatch.
        assert!(!EncoderRecipe::gptq_v1().is_reproducible_by_this_build());
    }
}
