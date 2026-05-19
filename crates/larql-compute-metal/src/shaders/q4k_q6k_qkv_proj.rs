//! Fused mixed-quant QKV projection — Q4_K for Q/K rows, Q6_K for V rows.
//!
//! **Q/K branch: 2-way inter-superblock interleaving (same as q4k_matvec).**
//!
//! The previous Q/K branch used `for (sb = lane; sb < superblocks; sb += 32)` —
//! for K=2560 (10 superblocks) only lanes 0..9 were active (31% utilisation).
//! New: `ix = lane & 1` ensures all 32 lanes are busy and adjacent lanes read
//! from different 144-byte superblock regions simultaneously.
//!
//! Lane decomposition for Q/K branch:
//!   ix  = lane & 1      — 0/1: even/odd superblock group
//!   tid = lane >> 1     — 0..15
//!   j   = tid >> 1      — 0..7: sub-block index
//!   sh  = tid & 1       — 0/1: first/last 16 elements
//!   X preloaded into xl[16] before weight reads.
//!
//! **V branch: same inter-superblock Q6_K inner loop as `q6k_matvec`.**
//! Keep this branch mechanically aligned with `q6k_matvec`; it is easy for
//! fused-QKV parity to drift because Q/K and V use different quant formats.

pub const SHADER: &str = r#"
constant uint Q4K_Q6K_ROWS_PER_TG  = 4;
constant uint Q4K_BLOCK_SIZE_MIXED  = 144;
constant uint Q6K_BLOCK_SIZE_MIXED  = 210;

kernel void q4k_q6k_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const float*  X      [[buffer(3)]],
    device float*        Q_out  [[buffer(4)]],
    device float*        K_out  [[buffer(5)]],
    device float*        V_out  [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    const uint superblocks = K / 256u;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        // ── Q/K rows: Q4_K — 2-way inter-superblock interleaving ──
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }

        const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        const uint ix  = lane & 1u;
        const uint tid = lane >> 1u;
        const uint j   = tid >> 1u;
        const uint sh  = tid & 1u;
        const bool hi    = (j & 1u) != 0u;
        const uint group = j >> 1u;

        for (uint sb = ix; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;
            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (j < 4u) {
                sc = uint(sb_bytes[j])      & 0x3Fu;
                mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }
            float scale = d * float(sc);
            float mmin  = dmin * float(mn);

            const uint x_base = sb * 256u + j * 32u + sh * 16u;
            float xl[16];
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

            device const uchar* qs = block + 16u + group * 32u + sh * 16u;
            float dot_acc = 0.0f, sum_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
                sum_acc += xl[l];
            }
            acc += scale * dot_acc - mmin * sum_acc;
        }

        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;

    } else {
        // ── V rows: Q6_K — same inner loop as standalone q6k_matvec ──
        uint local_row = global_row - q_rows - k_rows;
        const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        const uint ix6  = lane & 1u;
        const uint tid6 = lane >> 1u;
        const uint base = tid6 << 2u;
        const uint sc_base = tid6 >> 2u;

        for (uint sb = ix6; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql = block;
            device const uchar* qh = block + 128u;
            device const char* sc = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            const uint xb = sb * 256u + base;
            float xl[16];
            xl[ 0] = X[xb      ]; xl[ 1] = X[xb +  1u];
            xl[ 2] = X[xb +  2u]; xl[ 3] = X[xb +  3u];
            xl[ 4] = X[xb + 64u]; xl[ 5] = X[xb + 65u];
            xl[ 6] = X[xb + 66u]; xl[ 7] = X[xb + 67u];
            xl[ 8] = X[xb +128u]; xl[ 9] = X[xb +129u];
            xl[10] = X[xb +130u]; xl[11] = X[xb +131u];
            xl[12] = X[xb +192u]; xl[13] = X[xb +193u];
            xl[14] = X[xb +194u]; xl[15] = X[xb +195u];

            {
                const uint b = base;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 0u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
            }
            {
                const uint b = base + 64u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 4u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
            }
            {
                const uint b = base + 128u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 8u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
            }
            {
                const uint b = base + 192u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 12u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
            }
        }

        acc = simd_sum(acc);
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// MSL source for the fused RMS-norm + QKV projection variant.
/// Takes raw `H` (un-normalised hidden state) + `norm_weight` instead of
/// pre-normalised `X`, computing the norm cooperatively within each TG.
/// Eliminates the separate `rms_norm` dispatch (saves 34 dispatches/token).
///
/// ## Retention rationale (post 2026-05-09 model-agnosticity audit)
///
/// - **Defused 2026-05-09** as production default (see [ADR-016](../../docs/adr/016-defused-rms-norm-qkv.md)).
///   End-to-end A/B showed −0.30 ms/tok GPU fwd vs the defused path:
///   the fused kernel rereads H+norm_w 3× per TG (4 simdgroups, different
///   stride patterns for Q4_K Q/K vs Q6_K V) which dropped the per-kernel
///   batched throughput from 287 → 199 GB/s. The 1.46 ms/tok kernel cost
///   exceeded the 0.24 ms/tok dispatch saving.
/// - **Status**: Opt-in via `LARQL_QKV_FUSED=1`. Not deleted despite the
///   Gemma A/B loss because:
///   - **Different ALU/bandwidth balance** on M5+/A19 silicon could shift
///     the trade. The dispatch-overhead cost is fixed (~7 µs/dispatch);
///     the kernel-side bandwidth penalty depends on cache hierarchy.
///   - **Architectures with smaller H size could win**: at smaller hidden,
///     the H+norm_w reread cost shrinks while the dispatch saving stays
///     constant. The trade reverses in our favour.
///   - **Models with single-arch attention** (no Q4_K Q/K + Q6_K V mix —
///     all Q4_K or all Q6_K) wouldn't have the stride-pattern conflict
///     that drives the operand reread, so the kernel could be retuned
///     for those archs and re-validated.
/// - **Re-validation gate**: A/B with `LARQL_QKV_FUSED=1` on a vindex
///   with smaller hidden (Gemma 4 E2B at hidden=1536) or non-mixed-quant
///   layout. Promote if batched-diag improves AND end-to-end shows
///   ≥ 1% tok/s gain.
/// - **Deletion criterion**: ADR-016 explicitly retains this kernel as
///   opt-in fallback. Don't delete without superseding ADR-016.
///
/// See `docs/shader-inventory.md` for the retention framework and
/// `docs/adr/016-defused-rms-norm-qkv.md` for the defuse decision.
pub const NORMED_SHADER: &str = r#"

kernel void q4k_q6k_qkv_proj_normed(
    device const uchar*  Wq      [[buffer(0)]],
    device const uchar*  Wk      [[buffer(1)]],
    device const uchar*  Wv      [[buffer(2)]],
    device const float*  H       [[buffer(3)]],   // raw hidden (un-normed)
    device const float*  norm_w  [[buffer(4)]],   // RMS norm weight
    device float*        Q_out   [[buffer(5)]],
    device float*        K_out   [[buffer(6)]],
    device float*        V_out   [[buffer(7)]],
    constant uint&       q_rows  [[buffer(8)]],
    constant uint&       k_rows  [[buffer(9)]],
    constant uint&       v_rows  [[buffer(10)]],
    constant uint&       K       [[buffer(11)]],
    constant float&      eps     [[buffer(12)]],
    constant float&      offset  [[buffer(13)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]],
    uint tid    [[thread_index_in_threadgroup]])
{
    // ── Phase 1: cooperative RMS norm (all 128 threads in TG) ──
    // All threads participate regardless of row validity so barriers are uniform.
    const uint tg_sz = Q4K_Q6K_ROWS_PER_TG * 32u;  // = 128
    float partial = 0.0f;
    for (uint i = tid; i < K; i += tg_sz) {
        float h = H[i];
        partial += h * h;
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[4];
    if (lane == 0u) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0] + tg_p[1] + tg_p[2] + tg_p[3];
    float rms = 1.0f / sqrt(sum_sq / float(K) + eps);

    // ── Phase 2: same Q4_K / Q6_K matvec as q4k_q6k_qkv_proj ──
    // X[i] replaced by H[i] * rms * (offset + norm_w[i]).
    // H and norm_w are 10 KB each — L1-cached after first few TG reads.
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    const uint superblocks = K / 256u;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }
        const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        const uint ix  = lane & 1u;
        const uint ptid = lane >> 1u;
        const uint j   = ptid >> 1u;
        const uint sh  = ptid & 1u;
        const bool hi    = (j & 1u) != 0u;
        const uint group = j >> 1u;

        for (uint sb = ix; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;
            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (j < 4u) {
                sc = uint(sb_bytes[j])      & 0x3Fu;
                mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }
            float scale = d * float(sc);
            float mmin  = dmin * float(mn);

            const uint x_base = sb * 256u + j * 32u + sh * 16u;
            float xl[16];
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                float h = H[x_base + l];
                xl[l] = h * rms * (offset + norm_w[x_base + l]);
            }

            device const uchar* qs = block + 16u + group * 32u + sh * 16u;
            float dot_acc = 0.0f, sum_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
                sum_acc += xl[l];
            }
            acc += scale * dot_acc - mmin * sum_acc;
        }

        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;

    } else {
        uint local_row = global_row - q_rows - k_rows;
        const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        const uint ix6  = lane & 1u;
        const uint tid6 = lane >> 1u;
        const uint base = tid6 << 2u;
        const uint sc_base = tid6 >> 2u;

        for (uint sb = ix6; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            const uint xb = sb * 256u + base;
            float xl[16];
            xl[ 0] = H[xb      ] * rms * (offset + norm_w[xb      ]);
            xl[ 1] = H[xb +  1u] * rms * (offset + norm_w[xb +  1u]);
            xl[ 2] = H[xb +  2u] * rms * (offset + norm_w[xb +  2u]);
            xl[ 3] = H[xb +  3u] * rms * (offset + norm_w[xb +  3u]);
            xl[ 4] = H[xb + 64u] * rms * (offset + norm_w[xb + 64u]);
            xl[ 5] = H[xb + 65u] * rms * (offset + norm_w[xb + 65u]);
            xl[ 6] = H[xb + 66u] * rms * (offset + norm_w[xb + 66u]);
            xl[ 7] = H[xb + 67u] * rms * (offset + norm_w[xb + 67u]);
            xl[ 8] = H[xb +128u] * rms * (offset + norm_w[xb +128u]);
            xl[ 9] = H[xb +129u] * rms * (offset + norm_w[xb +129u]);
            xl[10] = H[xb +130u] * rms * (offset + norm_w[xb +130u]);
            xl[11] = H[xb +131u] * rms * (offset + norm_w[xb +131u]);
            xl[12] = H[xb +192u] * rms * (offset + norm_w[xb +192u]);
            xl[13] = H[xb +193u] * rms * (offset + norm_w[xb +193u]);
            xl[14] = H[xb +194u] * rms * (offset + norm_w[xb +194u]);
            xl[15] = H[xb +195u] * rms * (offset + norm_w[xb +195u]);

            {
                const uint b = base;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 0u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
            }
            {
                const uint b = base + 64u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 4u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
            }
            {
                const uint b = base + 128u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 8u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
            }
            {
                const uint b = base + 192u;
                uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
                float _sc = d * float(sc[sc_base + 12u]);
                acc += _sc * (
                    float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                    float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                    float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                    float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
            }
        }

        acc = simd_sum(acc);
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_q6k_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

/// Marker for the fused-norm variant (takes raw H + norm_weight).
pub struct NormedKernel;
impl crate::kernels::TiledKernel for NormedKernel {
    const KERNEL_NAME: &'static str = "q4k_q6k_qkv_proj_normed";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
