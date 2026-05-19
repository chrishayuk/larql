//! Fused Q4_KF (GGUF) gate+up — llama.cpp-exact inner loop, shared input.
//!
//! Two matvecs in one dispatch: gate and up projections read the input ONCE
//! (register-cached yl/yh arrays) and compute both outputs.
//!
//! Layout: threadgroups 0..ceil(N/ROWS_PER_TG)-1 → gate rows,
//!         ceil(N/ROWS_PER_TG)..2*ceil(N/ROWS_PER_TG)-1 → up rows.
//!
//! Uses GGUF 144-byte Q4_K blocks with packed scale+min unpacking.

pub const SHADER: &str = r#"
constant uint Q4KGU_ROWS_PER_SG = 2;
constant uint Q4KGU_SG_PER_TG = 2;
constant uint Q4KGU_ROWS_PER_TG = Q4KGU_ROWS_PER_SG * Q4KGU_SG_PER_TG;  // 4

constant uint16_t gu_kmask1 = 0x3f3f;
constant uint16_t gu_kmask2 = 0x0f0f;
constant uint16_t gu_kmask3 = 0xc0c0;

kernel void q4kf_ffn_gate_up(
    device const uchar*  Wg     [[buffer(0)]],   // gate weights [N, K] GGUF Q4_K
    device const uchar*  Wu     [[buffer(1)]],   // up weights [N, K] GGUF Q4_K
    device const float*  X      [[buffer(2)]],   // f32 input [K]
    device float*        G_out  [[buffer(3)]],   // gate output [N]
    device float*        U_out  [[buffer(4)]],   // up output [N]
    constant uint&       N      [[buffer(5)]],   // inter (output rows per matrix)
    constant uint&       K      [[buffer(6)]],   // hidden (input dim)
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]])
{
    uint tgs_per_mat = (N + Q4KGU_ROWS_PER_TG - 1) / Q4KGU_ROWS_PER_TG;
    bool is_up = (tgpig.x >= tgs_per_mat);
    uint mat_tg = is_up ? (tgpig.x - tgs_per_mat) : tgpig.x;

    uint first_row = (mat_tg * Q4KGU_SG_PER_TG + sgitg) * Q4KGU_ROWS_PER_SG;
    if (first_row >= N) return;

    device const uchar* W = is_up ? Wu : Wg;
    device float* out_buf = is_up ? U_out : G_out;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const uint nb = K / 256;
    const uint gguf_block_size = 144;
    const uint nb01 = nb * gguf_block_size;

    device const uchar* x0 = W + first_row * nb01;
    device const uchar* x1 = W + min(first_row + 1, N - 1) * nb01;
    bool v1 = (first_row + 1 < N);

    device const float* y4 = X + ix * 256 + 64 * iq + 8 * ir;
    float yl[16], yh[16];
    float sumf[2] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    for (int ib = ix; ib < (int)nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < 2; row++) {
            if (row == 1 && !v1) continue;
            device const uchar* blk = (row == 0 ? x0 : x1) + ib * gguf_block_size;
            device const half*     dh = (device const half*)blk;
            device const uint16_t* sc = (device const uint16_t*)(blk + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(blk + 16) + 16 * iq + 4 * ir;

            sc16[0] = sc[0] & gu_kmask1;
            sc16[1] = sc[2] & gu_kmask1;
            sc16[2] = ((sc[4] >> 0) & gu_kmask2) | ((sc[0] & gu_kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & gu_kmask2) | ((sc[2] & gu_kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        y4 += 4 * 256;
    }

    for (short row = 0; row < 2 && first_row + row < N; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0) out_buf[first_row + row] = s;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4; // 2 SG × 2 rows/SG
pub const THREADS_PER_TG: u64 = 64; // 2 SG × 32 lanes

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4kf_ffn_gate_up";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
