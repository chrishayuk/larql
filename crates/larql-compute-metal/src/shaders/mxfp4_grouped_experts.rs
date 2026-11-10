//! MXFP4 grouped-expert matvec — a four-arm layout/decode tournament.
//!
//! **K2 of the fused-MXFP4 ladder.** `mxfp4_matvec` (K1) proved MXFP4 can be a
//! compute format. `q6k_grouped_experts` (K3a) proved the expert shape's 0.64
//! was occupancy. This asks the remaining exact question: **at 4-bit, which
//! physical layout and which decode strategy actually runs fastest?**
//!
//! The four arms are deliberately crossed so that no comparison confounds two
//! changes at once:
//!
//! | arm | scale layout | weight decode | isolates |
//! |---|---|---|---|
//! | A | separate stream, 4.25 bpw | 16-entry LUT | checkpoint-style control |
//! | B | interleaved superblock, 4.0625 bpw | 16-entry LUT | **B-A: layout + scale** |
//! | C | interleaved superblock | 256-entry byte-pair LUT | **C-B: pair lookup** |
//! | D | interleaved superblock | 8-entry magnitude + sign | **D-B: table pressure** |
//!
//! B, C and D share a byte-identical artifact, so C-B and D-B are pure decode
//! effects. A alone changes the bytes, so B-A is the layout effect — which is
//! why a three-arm tournament (A/C/D) would have been uninterpretable.
//!
//! ## Layout A — as the checkpoint stores it
//!
//! Two streams. Packed nibbles at `Wp[row * groups * 16 ..]`, e8m0 scales at
//! `Ws[row * groups ..]`. `groups = K / 32`. All-in 4.25 bpw. Two buffer reads
//! per group, from addresses 16x apart.
//!
//! ## Layout B/C/D — one stream, adaptive-delta scales
//!
//! Per 256-weight superblock, 130 contiguous bytes:
//!
//! ```text
//!   [0]      base exponent (e8m0 byte)
//!   [1]      8 delta bits, bit g = group g's exponent offset above base
//!   [2..130] 8 groups x 16 packed bytes
//! ```
//!
//! 130 * 8 / 256 = **4.0625 bpw**, and the scale for group `g` is
//! `e8m0(base + ((deltas >> g) & 1))`. This is the 1-bit-delta arm of the
//! adaptive encoding — the **97.12% common path** measured over 30,720 real K3
//! superblocks. The 2.88% two-bit fallback is NOT implemented here: this bench
//! measures the common path, and the mixed and adversarial fixtures are a
//! separate guard (see the module docs in `k3_ledger::serving_format`).
//!
//! ## Dispatch, copied from K3a
//!
//! Grid `(row_tiles, n_selected)`; `tg_id.y` is the expert slot, reading its
//! payload base from `offsets[slot]`. One simdgroup per output row, lane `l`
//! walking groups `l, l+32, ...` so adjacent lanes cover contiguous bytes.
//! `XSTRIDE` is explicit for the same reason as in `q6k_grouped_experts`: 0
//! shares one input across slots, K gives each slot its own, and getting it
//! wrong yields the wrong expert's product rather than an error.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;

mod kernels;
use kernels::*;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

/// Weights per e8m0 scale group.
pub const GROUP_ELEMS: usize = 32;
/// Packed bytes per group (32 nibbles).
pub const GROUP_BYTES: usize = 16;
/// Groups per interleaved superblock.
pub const GROUPS_PER_SB: usize = 8;
/// Header bytes: base exponent + delta bitmap.
pub const SB_HEADER_BYTES: usize = 2;
/// Total interleaved superblock size.
pub const SB_BYTES: usize = SB_HEADER_BYTES + GROUPS_PER_SB * GROUP_BYTES;

/// Row walk for a dispatch whose output rows *are* the stored rows — no
/// fused halves to choose between, so `frow == row`. A fused gate/up
/// dispatch takes its pair from
/// [`larql_compute::MoeFusedRowLayout::row_walk`] instead.
pub const ROW_BASE_IDENTITY: u32 = 0;
/// Companion of [`ROW_BASE_IDENTITY`].
pub const ROW_STRIDE_IDENTITY: u32 = 1;

/// fp4 (e2m1) values, sign in bit 3.
pub const LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const PRELUDE: &str = r#"
constant uint MXG_ROWS_PER_TG = 4;
constant uint MXG_GROUP_ELEMS = 32;
constant uint MXG_GROUP_BYTES = 16;
constant uint MXG_GROUPS_PER_SB = 8;
constant uint MXG_SB_BYTES = 130;

constant float MXG_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Magnitudes only — arm D pairs this with an explicit sign from bit 3.
constant float MXG_MAG[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

// e8m0 -> f32, reproducing both CPU-reference sentinels. A raw bitcast gives
// +inf for 255, which would diverge on exactly the adversarial input a parity
// test should catch.
inline float mxg_e8m0(uchar b) {
    if (b == 0)   { return 0.0f; }
    if (b == 255) { return NAN;  }
    return as_type<float>(uint(b) << 23);
}
"#;

/// Body shared by the interleaved arms; only the inner decode differs.
fn interleaved(name: &str, decode: &str) -> String {
    format!(
        r#"
kernel void {name}(
    device const uchar*  W       [[buffer(0)]],
    device const uint*   offsets [[buffer(1)]],
    device const float*  X       [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) {{ return; }}

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint sbs    = groups / MXG_GROUPS_PER_SB;
    device const uchar* row_w =
        W + (ulong)offsets[slot] + (ulong)row * sbs * MXG_SB_BYTES;
    device const float* Xs = X + (ulong)slot * XSTRIDE;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {{
        const uint sb   = g / MXG_GROUPS_PER_SB;
        const uint idx  = g % MXG_GROUPS_PER_SB;
        device const uchar* hdr = row_w + (ulong)sb * MXG_SB_BYTES;
        // One stream: the scale arrives with the weights, not from a second
        // buffer 16x away.
        const uchar delta = (hdr[1] >> idx) & 1u;
        const float scale = mxg_e8m0(uchar(hdr[0] + delta));
        device const uchar* blk = hdr + 2u + (ulong)idx * MXG_GROUP_BYTES;
        const uint base = g * MXG_GROUP_ELEMS;
        float part = 0.0f;
{decode}
        acc += scale * part;
    }}
    acc = simd_sum(acc);
    if (lane == 0u) {{ out[slot * N + row] = acc; }}
}}
"#
    )
}

const DECODE_LUT16: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += MXG_LUT[byte & 0x0Fu]         * Xs[base + 2u * b];
            part += MXG_LUT[(byte >> 4u) & 0x0Fu] * Xs[base + 2u * b + 1u];
        }
"#;

/// One indexed load yields both values of the byte.
const DECODE_PAIR: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const float2 pair = MXG_PAIR[blk[b]];
            part += pair.x * Xs[base + 2u * b];
            part += pair.y * Xs[base + 2u * b + 1u];
        }
"#;

/// 8-entry table plus an explicit sign — a third of arm C's table pressure.
/// Table-free: build the f32 directly from the e2m1 bit fields.
///
/// fp4 is `sign(1) | exp(2) | mantissa(1)`. For `exp >= 1` the value is
/// `2^(exp-1) * (1 + m/2)`, which is exactly an f32 with exponent field
/// `126 + exp` and mantissa bit 22 set to `m`. Only `exp == 0` is irregular
/// (0 or 0.5), and that is a select, not a branch. No constant-address-space
/// traffic at all — the arm C result says table pressure is what hurts.
const DECODE_BITS: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            for (uint half_i = 0u; half_i < 2u; ++half_i) {
                const uint c = (half_i == 0u) ? (byte & 0x0Fu) : ((byte >> 4u) & 0x0Fu);
                const uint e = (c >> 1u) & 3u;
                const uint m = c & 1u;
                const uint mag = (e == 0u) ? (m == 0u ? 0u : 0x3F000000u)
                                           : (((126u + e) << 23u) | (m << 22u));
                const float v = as_type<float>(((c & 8u) << 28u) | mag);
                part += v * Xs[base + 2u * b + half_i];
            }
        }
"#;

/// **Ceiling probe, not a candidate.** Same skeleton, same bytes, same X reads,
/// but the nibble goes through a trivial affine map instead of the non-uniform
/// fp4 grid. The gap `E - D` is the exact price of fp4's irregular value set.
const DECODE_AFFINE: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += (float(byte & 0x0Fu) - 8.0f)        * Xs[base + 2u * b];
            part += (float((byte >> 4u) & 0x0Fu) - 8.0f) * Xs[base + 2u * b + 1u];
        }
"#;

/// **Ceiling probe, not a candidate.** Reads every weight byte but never touches
/// X. `F - E` isolates the cost of the input gather and the FMA chain from the
/// cost of streaming the weights, so a slow F means the skeleton itself binds.
const DECODE_NO_X: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += float(byte & 0x0Fu) + float((byte >> 4u) & 0x0Fu);
        }
"#;

const DECODE_MAG_SIGN: &str = r#"
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            const uint lo = byte & 0x0Fu;
            const uint hi = (byte >> 4u) & 0x0Fu;
            const float ml = MXG_MAG[lo & 7u];
            const float mh = MXG_MAG[hi & 7u];
            part += ((lo & 8u) != 0u ? -ml : ml) * Xs[base + 2u * b];
            part += ((hi & 8u) != 0u ? -mh : mh) * Xs[base + 2u * b + 1u];
        }
"#;

/// The 256-entry byte-pair table, emitted as Metal source.
///
/// Generated rather than hand-written so it cannot drift from [`LUT`]: entry
/// `b` is `(LUT[b & 15], LUT[b >> 4])`, the two values that byte decodes to.
fn pair_table() -> String {
    let mut s = String::from("constant float2 MXG_PAIR[256] = {\n");
    for b in 0..256usize {
        if b % 4 == 0 {
            s.push_str("    ");
        }
        s.push_str(&format!(
            "float2({:.1}f, {:.1}f),",
            LUT[b & 0x0F],
            LUT[b >> 4]
        ));
        s.push(if b % 4 == 3 { '\n' } else { ' ' });
    }
    s.push_str("};\n");
    s
}

/// Full Metal source for all four arms.
pub fn shader() -> String {
    let mut s = String::from(PRELUDE);
    s.push_str(&pair_table());
    s.push_str(KERNEL_A);
    s.push_str(KERNEL_A2);
    s.push_str(KERNEL_A2X2);
    s.push_str(KERNEL_A2X2M);
    s.push_str(KERNEL_A2X2S);
    s.push_str(KERNEL_A2X2P);
    s.push_str(KERNEL_A2X4);
    s.push_str(KERNEL_A2X2GU);
    s.push_str(KERNEL_A2DC);
    s.push_str(&interleaved("mxfp4g_inter_lut16", DECODE_LUT16));
    s.push_str(&interleaved("mxfp4g_inter_pair", DECODE_PAIR));
    s.push_str(&interleaved("mxfp4g_inter_magsign", DECODE_MAG_SIGN));
    s.push_str(&interleaved("mxfp4g_inter_bits", DECODE_BITS));
    s.push_str(&interleaved("mxfp4g_inter_affine", DECODE_AFFINE));
    s.push_str(&interleaved("mxfp4g_inter_nox", DECODE_NO_X));
    s
}

macro_rules! arm {
    ($ty:ident, $name:literal) => {
        pub struct $ty;
        impl crate::kernels::TiledKernel for $ty {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = ROWS_PER_TG;
            const THREADS_PER_TG: u64 = THREADS_PER_TG;
        }
    };
}

arm!(KernelSplitLut16, "mxfp4g_split_lut16");
arm!(KernelSplitLut16Vec, "mxfp4g_split_lut16_vec");
/// A2dc — down + weighted combine for top-4, 2 rows per threadgroup.
pub struct KernelDownCombine4;
impl crate::kernels::TiledKernel for KernelDownCombine4 {
    const KERNEL_NAME: &'static str = "mxfp4g_down_combine4";
    const ROWS_PER_TG: u64 = 2;
    const THREADS_PER_TG: u64 = 256;
}
/// A2x2gu — gate+up in one dispatch, 8 logical rows per threadgroup.
pub struct KernelSplitLut16VecX2Gu;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2Gu {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2_gu";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x2p — x2 with the byte-pair LUT, 8 rows per threadgroup.
pub struct KernelSplitLut16VecX2P;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2P {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2p";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x4 — 16 rows per threadgroup (4 simdgroups × 4 rows), 128 threads.
pub struct KernelSplitLut16VecX4;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX4 {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x4";
    const ROWS_PER_TG: u64 = 16;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x2m — x2 over a block's distinct experts (VERIFY-N rung 5b): grid
/// `(row tiles, slots)`, one distinct expert per `y`; at most
/// [`MULTI_MAX_SLOTS`] slots per expert.
pub struct KernelSplitLut16VecX2Multi;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2Multi {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2_multi";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A2x2s — the x2 arm over a block's slots in expert-sorted order, grid
/// `(slots, row tiles)` (VERIFY-N rung 5b).
pub struct KernelSplitLut16VecX2Sorted;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2Sorted {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2_sorted";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// Most slots one distinct expert may carry in the multi arm (its
/// `MXG_MULTI_MAX`): a block's rows, since an expert appears once per row.
pub const MULTI_MAX_SLOTS: usize = 8;
/// A2x2 — 8 rows per threadgroup (4 simdgroups × 2 rows), 128 threads.
pub struct KernelSplitLut16VecX2;
impl crate::kernels::TiledKernel for KernelSplitLut16VecX2 {
    const KERNEL_NAME: &'static str = "mxfp4g_split_lut16_vec_x2";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
arm!(KernelInterLut16, "mxfp4g_inter_lut16");
arm!(KernelInterPair, "mxfp4g_inter_pair");
arm!(KernelInterMagSign, "mxfp4g_inter_magsign");
arm!(KernelInterBits, "mxfp4g_inter_bits");
arm!(KernelInterAffine, "mxfp4g_inter_affine");
arm!(KernelInterNoX, "mxfp4g_inter_nox");

/// Which tournament arm serves the production MXFP4 expert path.
///
/// Names the four *candidate* arms only — the three ceiling probes
/// (`InterBits`, `InterAffine`, `InterNoX`) are diagnostics that do not
/// compute a correct product and are deliberately unselectable here.
///
/// **Fidelity, not throughput, sets the default.** The interleaved layout
/// carries a 1-bit exponent delta per group, so a superblock's eight
/// exponents must span at most one step; that holds for 97.12% of real
/// expert superblocks and the remaining 2.88% can only be encoded by
/// clamping, which alters weights. [`Self::SplitLut16`] stores the
/// checkpoint's own two streams and is exact — which is what lets the
/// native path be parity-gated against the lossless MXFP4→Q6_K transcode
/// it replaces.
///
/// The interleaved arms stay selectable because which is *fastest* is an
/// end-to-end question, not an isolated-kernel one, and they buy 4.0625
/// bpw against arm A's 4.25 — a 4.6% byte difference to weigh against
/// needing a wide-superblock escape hatch to stay exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Mxfp4Arm {
    /// Arm A — separate packed/scale streams, 4.25 bpw, **exact**.
    SplitLut16,
    /// Arm A2 — arm A's layout and math with a vectorised skeleton
    /// (`uint4` weight loads, `float4` X loads). **Exact**, and the
    /// tournament winner at every measured expert shape (+47% on the
    /// gpt-oss down shape, +10-13% elsewhere). Requires every payload
    /// offset to be 16-byte aligned; the encode path checks the built
    /// descriptor table and falls back to arm A when it is not.
    #[default]
    SplitLut16Vec,
    /// Arm B — interleaved superblock, 16-entry LUT decode.
    InterLut16,
    /// Arm C — interleaved, 256-entry byte-pair LUT decode.
    InterPair,
    /// Arm D — interleaved, 8-entry magnitude + sign decode.
    InterMagSign,
}

impl Mxfp4Arm {
    /// Parse an arm name or its tournament letter, case-insensitively.
    ///
    /// `None` for anything unrecognised, so a typo falls back to the
    /// exact default rather than silently selecting a lossy arm.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "split_lut16" | "a" => Self::SplitLut16,
            "split_lut16_vec" | "a2" => Self::SplitLut16Vec,
            "inter_lut16" | "b" => Self::InterLut16,
            "inter_pair" | "c" => Self::InterPair,
            "inter_magsign" | "d" => Self::InterMagSign,
            _ => return None,
        })
    }

    /// Whether this arm reconstructs every MXFP4 codepoint exactly.
    ///
    /// An inexact arm may not be parity-gated against the Q6_K transcode,
    /// and may not serve a model claiming lossless expert weights.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::SplitLut16 | Self::SplitLut16Vec)
    }

    /// Whether this arm's kernel takes the e8m0 exponents as a **separate**
    /// binding rather than interleaved into the weight stream.
    ///
    /// Deliberately a bool rather than a binding type: `shaders` must not
    /// depend on `kernels`, so the mapping to a binding shape is made one
    /// level up, where the pipelines live.
    pub fn is_split_scale(self) -> bool {
        matches!(self, Self::SplitLut16 | Self::SplitLut16Vec)
    }
}

#[cfg(test)]
mod tests;
