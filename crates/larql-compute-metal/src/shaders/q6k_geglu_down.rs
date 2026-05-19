//! Fused GEGLU activation + Q6_K down projection.
//!
//! Twin of `q4k_geglu_down.rs` for the Q6_K format used in production
//! Gemma 3 / Gemma 4 / Llama 2 / Mistral extracts (Ollama's standard
//! convention: Q4_K for gate/up where bandwidth wins, Q6_K for down
//! where precision wins). Without this fusion the production decode
//! path runs:
//!
//!   gate (q4k_ffn_gate_up) → up (same dispatch)
//!   → geglu_silu (separate dispatch + inter-sized buffer write/read)
//!   → q6k_matvec (down projection)
//!
//! Fused, those three become two: gate+up still fused into
//! `q4k_ffn_gate_up`, then this kernel skips the GEGLU dispatch and
//! the `inter`-sized activation buffer round-trip entirely:
//!
//!   `down_out[row] = Σᵢ W_down[row, i] · act(gate[i]) · up[i]`
//!
//! Matches the dispatch shape of the Q4_K version (`q4k_geglu_down`)
//! so callers can route by `down.format`.
//!
//! Dequantisation mirrors `q6k_matvec.rs` exactly — same Q6_K
//! super-block layout (256 values = 210 bytes: 128 lo4 + 64 hi2 +
//! 16 int8 scales + 2-byte f16 d).

pub const SHADER: &str = r#"
constant uint Q6K_GD_ROWS_PER_TG = 4;
constant uint Q6K_GD_BLOCK_SIZE  = 210;

// SiLU + down (Llama, Mistral, Qwen).
kernel void q6k_geglu_silu_down(
    device const uchar*  W_down [[buffer(0)]],   // down weights [N, inter] Q6_K
    device const float*  gate   [[buffer(1)]],   // gate output [inter]
    device const float*  up     [[buffer(2)]],   // up output [inter]
    device float*        out    [[buffer(3)]],   // output [N] (hidden)
    constant uint&       N      [[buffer(4)]],   // hidden (output rows)
    constant uint&       K      [[buffer(5)]],   // inter (input dim, multiple of 256)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]],
    uint tid       [[thread_index_in_threadgroup]])
{
    // 4 simdgroups × 32 lanes = 128 threads per TG.
    // All 4 rows iterate the same K/256 super-blocks. Gate and up windows
    // (256 f32 each) are loaded into TG memory once per super-block by all
    // 128 threads, eliminating 4× redundant device-memory reads per block.
    threadgroup float tg_gate[256];
    threadgroup float tg_up[256];

    uint row_idx       = tg_id * Q6K_GD_ROWS_PER_TG + sg_id;
    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_GD_BLOCK_SIZE;
    device const uchar* row = W_down + row_idx * bytes_per_row;

    float acc = 0.0f;

    for (uint sb = 0u; sb < superblocks; sb++) {
        uint x_base = sb * 256u;

        // Cooperative load: 128 threads each load 2 gate + 2 up values.
        tg_gate[tid]        = gate[x_base + tid];
        tg_gate[tid + 128u] = gate[x_base + tid + 128u];
        tg_up[tid]          = up[x_base + tid];
        tg_up[tid + 128u]   = up[x_base + tid + 128u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_idx < N) {
            device const uchar* block = row + sb * Q6K_GD_BLOCK_SIZE;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            for (uint pass = 0u; pass < 8u; pass++) {
                uint i = pass * 32u + lane;

                uchar lo_byte = ql[i >> 1u];
                uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);

                uchar hi_byte = qh[i >> 2u];
                uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;

                int raw = int(lo4 | (hi2 << 4u)) - 32;
                float w = d * float(sc[i >> 4u]) * float(raw);

                float gi = tg_gate[i];
                float silu_g = gi / (1.0f + exp(-gi));
                float ai = silu_g * tg_up[i];

                acc = fma(w, ai, acc);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (row_idx < N && lane == 0u) out[row_idx] = acc;
}

// GELU-tanh + down (Gemma, GPT-2, Phi).
kernel void q6k_geglu_gelu_tanh_down(
    device const uchar*  W_down [[buffer(0)]],
    device const float*  gate   [[buffer(1)]],
    device const float*  up     [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]],
    uint tid       [[thread_index_in_threadgroup]])
{
    threadgroup float tg_gate[256];
    threadgroup float tg_up[256];

    uint row_idx       = tg_id * Q6K_GD_ROWS_PER_TG + sg_id;
    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_GD_BLOCK_SIZE;
    device const uchar* row = W_down + row_idx * bytes_per_row;

    float acc = 0.0f;
    float c = 0.7978845608f; // sqrt(2/pi)

    for (uint sb = 0u; sb < superblocks; sb++) {
        uint x_base = sb * 256u;

        tg_gate[tid]        = gate[x_base + tid];
        tg_gate[tid + 128u] = gate[x_base + tid + 128u];
        tg_up[tid]          = up[x_base + tid];
        tg_up[tid + 128u]   = up[x_base + tid + 128u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_idx < N) {
            device const uchar* block = row + sb * Q6K_GD_BLOCK_SIZE;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            for (uint pass = 0u; pass < 8u; pass++) {
                uint i = pass * 32u + lane;

                uchar lo_byte = ql[i >> 1u];
                uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);

                uchar hi_byte = qh[i >> 2u];
                uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;

                int raw = int(lo4 | (hi2 << 4u)) - 32;
                float w = d * float(sc[i >> 4u]) * float(raw);

                // GELU-tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
                float gi = tg_gate[i];
                float t = tanh(c * (gi + 0.044715f * gi * gi * gi));
                float gelu_g = 0.5f * gi * (1.0f + t);
                float ai = gelu_g * tg_up[i];

                acc = fma(w, ai, acc);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (row_idx < N && lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 lanes

/// Two activation variants of fused Q6_K GEGLU+down — SiLU (Llama,
/// Mistral) and GELU-tanh (Gemma). Same geometry, distinct kernels.
pub struct SiluKernel;
impl crate::kernels::TiledKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "q6k_geglu_silu_down";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

pub struct GeluTanhKernel;
impl crate::kernels::TiledKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "q6k_geglu_gelu_tanh_down";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
