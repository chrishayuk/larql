//! Fused QKV — llama.cpp's exact kernel_mul_mv_q4_K_f32, adapted for fused QKV.
//!
//! Uses GGUF `block_q4_K` (144 bytes) with packed 12-byte scales+mins.
//! Inner loop matches llama.cpp byte-for-byte: no float() casts on nibbles,
//! uint16_t mask extraction, FOR_UNROLL, register-based input.
//!
//! N_R0 = 2 (2 rows per simdgroup), N_SG = 2 (2 simdgroups per TG) = 4 rows/TG.

pub const SHADER: &str = r#"
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant uint Q4KG_ROWS_PER_SG = 2;
constant uint Q4KG_SG_PER_TG = 2;
constant uint Q4KG_ROWS_PER_TG = Q4KG_ROWS_PER_SG * Q4KG_SG_PER_TG;  // 4

constant uint16_t kmask1 = 0x3f3f;
constant uint16_t kmask2 = 0x0f0f;
constant uint16_t kmask3 = 0xc0c0;

kernel void q4kf_qkv_proj(
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
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint first_row = (tgpig.x * Q4KG_SG_PER_TG + sgitg) * Q4KG_ROWS_PER_SG;
    if (first_row >= total_rows) return;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const uint nb = K / 256;
    const uint gguf_block_size = 144;  // GGUF Q4_K: 2+2+12+128
    const uint nb01 = nb * gguf_block_size;  // bytes per row

    // Resolve 2 rows: pointers to weight data + output destinations +
    // local row index (within the selected Q/K/V output buffer).
    device const uchar* wp[2];
    device float* op[2];
    uint lri[2];
    bool valid[2];
    for (uint r = 0; r < 2; r++) {
        uint row = first_row + r;
        valid[r] = (row < total_rows);
        uint lr = 0;
        device const uchar* base;
        if (!valid[r]) { wp[r] = Wq; op[r] = Q_out; lri[r] = 0; continue; }
        if (row < q_rows) { base = Wq; op[r] = Q_out; lr = row; }
        else if (row < q_rows + k_rows) { base = Wk; op[r] = K_out; lr = row - q_rows; }
        else { base = Wv; op[r] = V_out; lr = row - q_rows - k_rows; }
        wp[r] = base + lr * nb01;
        lri[r] = lr;
    }

    // Input: register-based (llama.cpp pattern)
    device const float* y4 = X + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
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
            if (!valid[row]) continue;

            // GGUF block: d(2) + dmin(2) + scales(12) + qs(128)
            device const uchar* blk = wp[row] + ib * gguf_block_size;
            device const half*     dh = (device const half*)blk;
            device const uint16_t* sc = (device const uint16_t*)(blk + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(blk + 16) + 16 * iq + 4 * ir;

            // Unpack scales+mins from packed 12-byte format (llama.cpp exact)
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
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

            // Advance to next row's block
            wp[row] += 0;  // pointer already points to correct row
        }

        y4 += 4 * 256;
    }

    for (short row = 0; row < 2; row++) {
        if (!valid[row]) continue;
        float s = simd_sum(sumf[row]);
        // Write to the correct output slot. Every simdgroup previously wrote
        // to `op[row][0]` — multiple SGs racing for index 0 meant only the
        // first 4 Q rows / 4 K rows / 4 V rows ever held real values (the
        // others were clobbered). Using `lri[row]` routes each simdgroup to
        // its own output index.
        if (tiisg == 0) op[row][lri[row]] = s;
    }
}

// Single projection using same GGUF Q4_K kernel (for O projection).
kernel void q4kf_proj(
    device const uchar*  W     [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]])
{
    uint first_row = (tgpig.x * Q4KG_SG_PER_TG + sgitg) * Q4KG_ROWS_PER_SG;
    if (first_row >= N) return;

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

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
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
        if (tiisg == 0) out[first_row + row] = s;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4; // 2 SG × 2 rows/SG
pub const THREADS_PER_TG: u64 = 64; // 2 SG × 32 lanes

/// Two kernels share this file's geometry — fused QKV projection
/// (`q4kf_qkv_proj`) and the per-projection variant (`q4kf_proj`).
pub struct QkvKernel;
impl crate::kernels::TiledKernel for QkvKernel {
    const KERNEL_NAME: &'static str = "q4kf_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

pub struct ProjKernel;
impl crate::kernels::TiledKernel for ProjKernel {
    const KERNEL_NAME: &'static str = "q4kf_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
