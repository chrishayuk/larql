//! MSL kernel sources for the grouped MXFP4 expert arms.

#[allow(unused_imports)]
use super::*;

/// Arm A: separate scale stream, 16-entry LUT decode.
///
/// The only arm that takes `s_offsets` and a row walk, and it takes both for
/// the same reason: it is the arm that serves a **stored** bank rather than a
/// bench fixture, so it cannot assume anything about where the container put
/// the streams or how the fused rows are arranged.
///
/// `s_offsets` replaces a derived `offsets[slot] / 16`. That derivation was a
/// physical-placement invariant — "the exponent for a payload byte at `o`
/// lives at `o/16`" — which holds for two parallel contiguous banks and not
/// for a VINDEX3 container, whose paired regions are placed by the writer and
/// bound by `pair_id`. Nothing established the invariant, so nothing would
/// have caught it being false; the failure is silent wrong numbers.
///
/// `ROWBASE`/`ROWSTRIDE` express which fused rows this dispatch's half owns:
/// `(half * inter, 1)` for contiguous halves, `(half, 2)` for the
/// checkpoint's interleaving. Expressing the half as a byte offset — the way
/// every inline-scale call site does — can only say the former.
pub(super) const KERNEL_A: &str = r#"
kernel void mxfp4g_split_lut16(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    // Output row `row` of this half is fused row `frow` of the stored region.
    // `out` stays keyed on `row`: the destination is dense per half however
    // the source rows are spaced.
    const uint frow = ROWBASE + row * ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot]   + (ulong)frow * groups * MXG_GROUP_BYTES;
    const ulong sbase = (ulong)s_offsets[slot] + (ulong)frow * groups;
    device const uchar* row_p = Wp + pbase;
    device const uchar* row_s = Ws + sbase;
    device const float* Xs = X + (ulong)slot * XSTRIDE;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const float scale = mxg_e8m0(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * MXG_GROUP_BYTES;
        const uint base = g * MXG_GROUP_ELEMS;
        float part = 0.0f;
        for (uint b = 0u; b < MXG_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += MXG_LUT[byte & 0x0Fu]         * Xs[base + 2u * b];
            part += MXG_LUT[(byte >> 4u) & 0x0Fu] * Xs[base + 2u * b + 1u];
        }
        acc += scale * part;
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row] = acc; }
}
"#;

/// Arm A2: arm A's layout and math with a vectorised skeleton.
///
/// The tournament's ceiling probes said the split kernel's deficit on the
/// gpt-oss down shape is the **skeleton**, not the decode: arm A streams
/// each 16-byte group with sixteen single-`uchar` loads, so consecutive
/// lanes read addresses 16 bytes apart and every load moves one byte. Here
/// each group is one `uint4` load — consecutive lanes read consecutive
/// 16-byte chunks (the coalescing shape `q6k_grouped_experts` already has)
/// and the issue rate drops 16×. X moves through `float4`s the same way.
///
/// Alignment contract, stated because `uint4`/`float4` device loads
/// require it: every payload region base must be 16-byte aligned and
/// `XSTRIDE` a multiple of 4 floats. Both hold for the bench fixture and
/// for VINDEX3 payload regions (per-expert payloads are whole groups of
/// 16 bytes); a caller that cannot guarantee them keeps arm A.
pub(super) const KERNEL_A2: &str = r#"
inline float mxg_dot8(uint v, float4 xa, float4 xb) {
    return MXG_LUT[v         & 0x0Fu] * xa.x
         + MXG_LUT[(v >>  4u) & 0x0Fu] * xa.y
         + MXG_LUT[(v >>  8u) & 0x0Fu] * xa.z
         + MXG_LUT[(v >> 12u) & 0x0Fu] * xa.w
         + MXG_LUT[(v >> 16u) & 0x0Fu] * xb.x
         + MXG_LUT[(v >> 20u) & 0x0Fu] * xb.y
         + MXG_LUT[(v >> 24u) & 0x0Fu] * xb.z
         + MXG_LUT[(v >> 28u) & 0x0Fu] * xb.w;
}

kernel void mxfp4g_split_lut16_vec(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * MXG_ROWS_PER_TG + sg_id;
    if (row >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow = ROWBASE + row * ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot]   + (ulong)frow * groups * MXG_GROUP_BYTES;
    const ulong sbase = (ulong)s_offsets[slot] + (ulong)frow * groups;
    device const uint4* row_p = (device const uint4*)(Wp + pbase);
    device const uchar* row_s = Ws + sbase;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const float scale = mxg_e8m0(row_s[g]);
        const uint4 w = row_p[g];
        const uint xb = g * 8u; // group g's X span, in float4s
        float part = mxg_dot8(w.x, Xs4[xb],      Xs4[xb + 1u])
                   + mxg_dot8(w.y, Xs4[xb + 2u], Xs4[xb + 3u])
                   + mxg_dot8(w.z, Xs4[xb + 4u], Xs4[xb + 5u])
                   + mxg_dot8(w.w, Xs4[xb + 6u], Xs4[xb + 7u]);
        acc += scale * part;
    }
    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row] = acc; }
}
"#;

/// Arm A2x2 — A2's layout and math with **two rows per simdgroup sharing
/// one set of X loads** (the A-5a lesson transplanted: the NVFP4 GEMV
/// moved 332 → 373 GB/s from exactly this change, and the expert
/// decomposition priced the deficit in the kernel body, not the routing
/// machinery — indirection measured free, 212 vs 214 GB/s). Per-row group
/// walk and summation order are A2's exactly, so each row's output is
/// bit-identical to A2's.
pub(super) const KERNEL_A2X2: &str = r#"
kernel void mxfp4g_split_lut16_vec_x2(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc0 += scale * part;
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc1 += scale * part;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        out[slot * N + row0] = acc0;
        if (has1) { out[slot * N + row0 + 1u] = acc1; }
    }
}
"#;

/// A2x2p — A2x2 with the 256-entry byte-pair LUT: one `float2` lookup per
/// byte instead of two nibble lookups. **FALSIFIED 2026-08-20** (292 vs
/// x2's 322 GB/s at the gpt-oss expert shape) — unlike the NVFP4 kernel,
/// where the byte LUT carried the remaining slope, here the wider table
/// costs more than the halved lookup count saves. Retained as an arm;
/// fp32-rounding parity (fast-math contracts differently).
pub(super) const KERNEL_A2X2P: &str = r#"
inline float mxg_dot8_pair(uint v, float4 xa, float4 xb) {
    const float2 p0 = MXG_PAIR[v & 0xFFu];
    const float2 p1 = MXG_PAIR[(v >> 8u) & 0xFFu];
    const float2 p2 = MXG_PAIR[(v >> 16u) & 0xFFu];
    const float2 p3 = MXG_PAIR[(v >> 24u) & 0xFFu];
    return p0.x * xa.x + p0.y * xa.y
         + p1.x * xa.z + p1.y * xa.w
         + p2.x * xb.x + p2.y * xb.y
         + p3.x * xb.z + p3.y * xb.w;
}

kernel void mxfp4g_split_lut16_vec_x2p(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            acc0 += scale * (mxg_dot8_pair(w.x, xa, xbv) + mxg_dot8_pair(w.y, xc, xd)
                           + mxg_dot8_pair(w.z, xe, xf) + mxg_dot8_pair(w.w, xg, xh));
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            acc1 += scale * (mxg_dot8_pair(w.x, xa, xbv) + mxg_dot8_pair(w.y, xc, xd)
                           + mxg_dot8_pair(w.z, xe, xf) + mxg_dot8_pair(w.w, xg, xh));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        out[slot * N + row0] = acc0;
        if (has1) { out[slot * N + row0 + 1u] = acc1; }
    }
}
"#;

/// A2x4 — four rows per simdgroup sharing X. **FALSIFIED 2026-08-20**
/// (311 vs x2's 322 GB/s): the extra reuse does not pay for the lost
/// row-tile parallelism at this shape. Retained as an arm; bit-identical
/// to x2 (same per-row walk).
pub(super) const KERNEL_A2X4: &str = r#"
kernel void mxfp4g_split_lut16_vec_x4(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 4u;
    if (row0 >= N) { return; }

    const uint groups = K / MXG_GROUP_ELEMS;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        for (uint r = 0u; r < 4u; ++r) {
            const uint row = row0 + r;
            if (row >= N) { break; }
            const uint frow = ROWBASE + row * ROWSTRIDE;
            const float scale =
                mxg_e8m0((Ws + sbase + (ulong)frow * groups)[g]);
            const uint4 w = ((device const uint4*)(Wp + pbase
                + (ulong)frow * groups * MXG_GROUP_BYTES))[g];
            float part = mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh);
            acc[r] += scale * part;
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < N) { out[slot * N + row0 + r] = total; }
    }
}
"#;

/// A2x2gu — BOTH fused halves in one dispatch: logical rows `0..2N` where
/// row `l < N` walks the gate half into `out`, `l >= N` walks the up half
/// into `out2`. Doubles the threadgroup count the x2 arm halved (the
/// decomposition showed slot-grid parallelism worth +50 GB/s at this
/// shape) and pays one GEMV α per layer instead of two. Per-row body is
/// A2x2's exactly → bit-identical to two x2 dispatches.
pub(super) const KERNEL_A2X2GU: &str = r#"
kernel void mxfp4g_split_lut16_vec_x2_gu(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    device float*        out2      [[buffer(11)]],
    constant uint&       ROWBASE2  [[buffer(12)]],
    constant uint&       ROWSTRIDE2 [[buffer(13)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tg_id.y;
    const uint l0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    const uint total = 2u * N;
    if (l0 >= total) { return; }
    const bool has1 = l0 + 1u < total;

    const uint groups = K / MXG_GROUP_ELEMS;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];

    // Per-row half resolution into scalars (the seg3 lesson: a
    // dynamically indexed pointer array spills).
    const uint l1 = l0 + 1u;
    const bool up0 = l0 >= N;
    const bool up1 = l1 >= N;
    const uint r0 = up0 ? (l0 - N) : l0;
    const uint r1 = up1 ? (l1 - N) : l1;
    const uint frow0 = (up0 ? ROWBASE2 : ROWBASE) + r0 * (up0 ? ROWSTRIDE2 : ROWSTRIDE);
    const uint frow1 = (up1 ? ROWBASE2 : ROWBASE) + r1 * (up1 ? ROWSTRIDE2 : ROWSTRIDE);
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            acc0 += scale * (mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh));
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            acc1 += scale * (mxg_dot8(w.x, xa, xbv) + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf) + mxg_dot8(w.w, xg, xh));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        if (up0) { out2[slot * N + r0] = acc0; } else { out[slot * N + r0] = acc0; }
        if (has1) {
            if (up1) { out2[slot * N + r1] = acc1; } else { out[slot * N + r1] = acc1; }
        }
    }
}
"#;

/// A2dc — the down projection and the weighted combine in ONE dispatch,
/// for top-4 routes: each threadgroup owns 2 output rows × 4 slots (8
/// simdgroups, 256 threads); simdgroup (r,s) computes `down_s[row_r] ·
/// act_s` with A2's exact walk (bit-identical per (row, slot) to the
/// grouped down GEMV), then lane 0s stage the four per-slot dots and one
/// thread folds `h + Σ_s w_s·(dot_s + bias_s)` — the combine kernel's
/// exact order, so the result is bit-identical to the GPU down→combine
/// pair (a CPU emulation of the combine differs at the last ulp: Metal
/// contracts the multiply-add). Removes the down→combine serialization
/// and puts 11520 simdgroups in flight where the split form's down
/// dispatch carries 5760. A/B on gpt-oss was AMBIGUOUS under battery
/// drift (−0.21/+0.12 ms) — opt-in via `LARQL_MXFP4_EXPERT_DC=1` until a
/// rested AC re-run decides.
pub(super) const KERNEL_A2DC: &str = r#"
kernel void mxfp4g_down_combine4(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],   // act, [4, XSTRIDE]
    device float*        new_h     [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    device const float*  Hin       [[buffer(9)]],   // [N] post-attn residual
    constant float*      Wroute    [[buffer(10)]],  // [4] routing weights
    device const float*  Bias      [[buffer(11)]],  // [4, N] staged down bias
    constant uint&       has_bias  [[buffer(12)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint r = sg_id >> 2u;          // 0..2: row within the pair
    const uint slot = sg_id & 3u;        // 0..4
    const uint row = tg_id.x * 2u + r;
    const uint groups = K / MXG_GROUP_ELEMS;

    float dot = 0.0f;
    if (row < N) {
        const ulong pbase = (ulong)offsets[slot] + (ulong)row * groups * MXG_GROUP_BYTES;
        device const uint4* row_p = (device const uint4*)(Wp + pbase);
        device const uchar* row_s = Ws + (ulong)s_offsets[slot] + (ulong)row * groups;
        device const float4* Xs4 = (device const float4*)(X + (ulong)slot * XSTRIDE);
        for (uint g = lane; g < groups; g += 32u) {
            const float scale = mxg_e8m0(row_s[g]);
            const uint4 w = row_p[g];
            const uint xb = g * 8u;
            float part = mxg_dot8(w.x, Xs4[xb],      Xs4[xb + 1u])
                       + mxg_dot8(w.y, Xs4[xb + 2u], Xs4[xb + 3u])
                       + mxg_dot8(w.z, Xs4[xb + 4u], Xs4[xb + 5u])
                       + mxg_dot8(w.w, Xs4[xb + 6u], Xs4[xb + 7u]);
            dot += scale * part;
        }
        dot = simd_sum(dot);
    }

    threadgroup float parts[2][4];
    if (lane == 0u) { parts[r][slot] = dot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per row folds the combine, in the combine kernel's
    // exact order: acc = h[row]; for j: acc += w_j * (dot_j [+ bias_j]).
    if (lane == 0u && slot == 0u && row < N) {
        float acc = Hin[row];
        for (uint j = 0u; j < 4u; ++j) {
            float v = parts[r][j];
            if (has_bias != 0u) { v += Bias[j * N + row]; }
            acc += Wroute[j] * v;
        }
        new_h[row] = acc;
    }
}
"#;

/// A2x2m — VERIFY-N rung 5b: the x2 arm with each threadgroup owning one
/// DISTINCT expert of a block, read once for every slot that selected it.
///
/// `counts[u]` / `lists[u * MAXM ..]` (built on the device by
/// `moe_slots_by_expert`) name the slots of distinct expert `u`; a
/// threadgroup with `counts[u] == 0` exits. Weights and scales come from
/// the FIRST slot's offsets — every slot of one expert carries the same.
/// Each slot reads its own input at `X + slot * XSTRIDE` and writes
/// `out[slot * N + row]`, exactly where the x2 arm writes it, and every
/// output is the x2 arm's expression in the x2 arm's accumulation order
/// (same lane stride over groups, same `scale * part` sum, same
/// `simd_sum`), so it is bit-identical per (row, slot).
pub(super) const KERNEL_A2X2M: &str = r#"
constant uint MXG_MULTI_MAX = 8;

kernel void mxfp4g_split_lut16_vec_x2_multi(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    device const uint*   counts    [[buffer(11)]],
    device const uint*   lists     [[buffer(12)]],
    constant uint&       MAXM      [[buffer(13)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint u = tg_id.y;
    const uint cnt = min(counts[u], MXG_MULTI_MAX);
    if (cnt == 0u) { return; }
    const uint row0 = (tg_id.x * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    device const uint* slots = lists + (ulong)u * MAXM;
    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slots[0]];
    const ulong sbase = (ulong)s_offsets[slots[0]];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;

    // Slot-major: each slot runs the x2 body verbatim. The first slot's
    // pass pulls this simdgroup's two weight rows from DRAM; the rest find
    // them in cache — one DRAM read per distinct expert, no register arrays.
    for (uint m = 0u; m < cnt; ++m) {
        const uint slot = slots[m];
        device const float4* Xs4 =
            (device const float4*)(X + (ulong)slot * XSTRIDE);
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint g = lane; g < groups; g += 32u) {
            const uint xb = g * 8u;
            const float4 xa = Xs4[xb];
            const float4 xbv = Xs4[xb + 1u];
            const float4 xc = Xs4[xb + 2u];
            const float4 xd = Xs4[xb + 3u];
            const float4 xe = Xs4[xb + 4u];
            const float4 xf = Xs4[xb + 5u];
            const float4 xg = Xs4[xb + 6u];
            const float4 xh = Xs4[xb + 7u];
            {
                const float scale = mxg_e8m0(row_s0[g]);
                const uint4 w = row_p0[g];
                float part = mxg_dot8(w.x, xa, xbv)
                           + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf)
                           + mxg_dot8(w.w, xg, xh);
                acc0 += scale * part;
            }
            if (has1) {
                const float scale = mxg_e8m0(row_s1[g]);
                const uint4 w = row_p1[g];
                float part = mxg_dot8(w.x, xa, xbv)
                           + mxg_dot8(w.y, xc, xd)
                           + mxg_dot8(w.z, xe, xf)
                           + mxg_dot8(w.w, xg, xh);
                acc1 += scale * part;
            }
        }
        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        if (lane == 0u) {
            out[slot * N + row0] = acc0;
            if (has1) { out[slot * N + row0 + 1u] = acc1; }
        }
    }
}
"#;

/// A2x2s — VERIFY-N rung 5b: the x2 arm verbatim, with the grid
/// transposed to `(slot, row tile)` and slots taken in `order` — the
/// block's slots sorted by expert. Threadgroups sharing an expert's tile
/// are then neighbours in dispatch order and read it from cache; every
/// slot still gets its own threadgroup (full occupancy) and the x2 arm's
/// exact arithmetic, so it is bit-identical per (row, slot).
pub(super) const KERNEL_A2X2S: &str = r#"
kernel void mxfp4g_split_lut16_vec_x2_sorted(
    device const uchar*  Wp        [[buffer(0)]],
    device const uint*   offsets   [[buffer(1)]],
    device const uchar*  Ws        [[buffer(2)]],
    device const uint*   s_offsets [[buffer(3)]],
    device const float*  X         [[buffer(4)]],
    device float*        out       [[buffer(5)]],
    constant uint&       N         [[buffer(6)]],
    constant uint&       K         [[buffer(7)]],
    constant uint&       XSTRIDE   [[buffer(8)]],
    constant uint&       ROWBASE   [[buffer(9)]],
    constant uint&       ROWSTRIDE [[buffer(10)]],
    device const uint*   order     [[buffer(11)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint slot = order[tg_id.x];
    const uint row0 = (tg_id.y * MXG_ROWS_PER_TG + sg_id) * 2u;
    if (row0 >= N) { return; }
    const bool has1 = row0 + 1u < N;

    const uint groups = K / MXG_GROUP_ELEMS;
    const uint frow0 = ROWBASE + row0 * ROWSTRIDE;
    const uint frow1 = frow0 + ROWSTRIDE;
    const ulong pbase = (ulong)offsets[slot];
    const ulong sbase = (ulong)s_offsets[slot];
    device const uint4* row_p0 =
        (device const uint4*)(Wp + pbase + (ulong)frow0 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s0 = Ws + sbase + (ulong)frow0 * groups;
    device const uint4* row_p1 =
        (device const uint4*)(Wp + pbase + (ulong)frow1 * groups * MXG_GROUP_BYTES);
    device const uchar* row_s1 = Ws + sbase + (ulong)frow1 * groups;
    device const float4* Xs4 =
        (device const float4*)(X + (ulong)slot * XSTRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint xb = g * 8u;
        const float4 xa = Xs4[xb];
        const float4 xbv = Xs4[xb + 1u];
        const float4 xc = Xs4[xb + 2u];
        const float4 xd = Xs4[xb + 3u];
        const float4 xe = Xs4[xb + 4u];
        const float4 xf = Xs4[xb + 5u];
        const float4 xg = Xs4[xb + 6u];
        const float4 xh = Xs4[xb + 7u];
        {
            const float scale = mxg_e8m0(row_s0[g]);
            const uint4 w = row_p0[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc0 += scale * part;
        }
        if (has1) {
            const float scale = mxg_e8m0(row_s1[g]);
            const uint4 w = row_p1[g];
            float part = mxg_dot8(w.x, xa, xbv)
                       + mxg_dot8(w.y, xc, xd)
                       + mxg_dot8(w.z, xe, xf)
                       + mxg_dot8(w.w, xg, xh);
            acc1 += scale * part;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        out[slot * N + row0] = acc0;
        if (has1) { out[slot * N + row0 + 1u] = acc1; }
    }
}
"#;
