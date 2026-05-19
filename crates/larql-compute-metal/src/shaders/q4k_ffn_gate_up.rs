//! Fused Q4_K gate+up projection — two matvecs sharing the same input vector.
//!
//! Dispatched as `2 × ceil(N/ROWS_PER_TG)` TGs: first half → gate, second → up.
//!
//! **Parallelism — 2-way inter-superblock interleaving:**
//!
//! `ix = lane & 1` splits 32 lanes into two groups:
//!   ix=0 → even superblocks  ix=1 → odd superblocks
//! Adjacent lanes read from different 144-byte superblock regions simultaneously.
//!
//! **Why float4 / dual-sub-block approaches were tried and reverted:**
//! Q4_K gate+up is COMPUTE-BOUND at K=2560 (measured: 272 GB/s, profiler confirms).
//! K=2560 = 10 superblocks × 144 bytes/row fits in GPU L1 cache — the bottleneck
//! is ALU throughput for nibble dequant, not DRAM bandwidth.
//! - 4-way SB interleaving (ix=lane>>3): creates 3 vs 2 SB load imbalance for 10 SBs
//!   → simd_sum waits for slowest ix-group → regression.
//! - float4 with uint16 correction factors: adds ALU complexity (inv16/inv256/inv4096
//!   corrections) to an already ALU-limited kernel → regression.
//!
//! Current approach (simple, 128 threads/TG) is close to optimal for K=2560.

pub const SHADER: &str = r#"
constant uint Q4K_GU_ROWS_PER_TG = 4;
constant uint Q4K_GU_BLOCK_SIZE  = 144;

kernel void q4k_ffn_gate_up(
    device const uchar*  Wg    [[buffer(0)]],
    device const uchar*  Wu    [[buffer(1)]],
    device const float*  X     [[buffer(2)]],
    device float*        G_out [[buffer(3)]],
    device float*        U_out [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    constant uint&       K     [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint tgs_per_mat = (N + Q4K_GU_ROWS_PER_TG - 1u) / Q4K_GU_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    uint row_idx = mat_tg * Q4K_GU_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const uchar* W      = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_GU_BLOCK_SIZE;
    device const uchar* row_w = W + row_idx * bytes_per_row;

    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_GU_BLOCK_SIZE;
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

        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        float dot_acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            dot_acc = fma(nib, xl[l], dot_acc);
        }
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out_buf[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
