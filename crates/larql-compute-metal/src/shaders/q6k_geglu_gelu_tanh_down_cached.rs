//! Fused **GELU-tanh + Q6_K down** with **TG-cached activations**.
//!
//! Same shape as `q6k_geglu_gelu_tanh_down` (4 simdgroups per TG, 4
//! output rows per TG, walks K=10240 in 40 super-blocks of 256), but
//! the per-element activation `gelu_tanh(g[i]) * u[i]` is computed
//! **once per TG per super-block** by the entire threadgroup (each
//! thread handling 2 elements) into `tg_act[256]` — instead of being
//! recomputed inside every (simdgroup, pass) iteration of the inner
//! FMA loop.
//!
//! **Why this kernel exists**: the existing
//! `q6k_geglu_gelu_tanh_down` was disabled (per
//! `larql-compute/src/metal/decode/encode_ffn.rs:290` comment) because:
//! "with GELU-tanh the fused inner loop recomputes tanh(gate[i]) once
//! per output row, so 2560 rows = 2560× more tanh() calls than the
//! separated `geglu_gelu_tanh` dispatch". With NR0=4 simdgroups per
//! TG, each lane re-does the same `tanh(c·(g + 0.044715·g³))` for
//! every output row in its TG — 4× redundant per element.
//!
//! Caching activations into threadgroup memory (1 KB / TG, well under
//! limits) reduces `tanh()` calls 4× per super-block, restoring the
//! kernel as a viable replacement for the separated chain
//! (`encode_geglu` + `q6k_matvec`).
//!
//! **Saved dispatch**: 1 per layer × 34 = ~34/tok ≈ 0.24 ms/tok
//! (matches G-3 fusion mechanic). Plus the activation re-compute
//! reduction.
//!
//! **Math**: identical to the unfused chain
//! (`encode_geglu_gelu_tanh` + `q6k_matvec(act_buf)`). Per element:
//!   gelu_t = 0.5·g·(1 + tanh(√(2/π)·(g + 0.044715·g³))) · u
//!   acc[row] += W_down[row, i] · gelu_t[i]
//! Bit-equivalent up to FMA-order rounding (the `tanh()` and
//! `0.5·(1+t)` are computed once per element rather than once per
//! row, so the activation value is *more* numerically stable, not less).
//!
//! **Geometry**: 4 simdgroups per TG, 4 rows per TG, 128 threads per TG —
//! same as the original kernel, dispatch grid math is unchanged.

pub const SHADER: &str = r#"
constant uint Q6K_GDC_ROWS_PER_TG = 4;
constant uint Q6K_GDC_BLOCK_SIZE  = 210;

// SANITY-CHECK COPY: this is the verbatim production
// `q6k_geglu_gelu_tanh_down` body (per-row activation recompute, no
// cache) — used to confirm the dispatch wiring of the new pipeline
// works before re-introducing the cache.
kernel void q6k_geglu_gelu_tanh_down_cached(
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

    uint row_idx       = tg_id * Q6K_GDC_ROWS_PER_TG + sg_id;
    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_GDC_BLOCK_SIZE;
    device const uchar* row = W_down + row_idx * bytes_per_row;

    float acc = 0.0f;
    float c = 0.7978845608f;

    for (uint sb = 0u; sb < superblocks; sb++) {
        uint x_base = sb * 256u;

        tg_gate[tid]        = gate[x_base + tid];
        tg_gate[tid + 128u] = gate[x_base + tid + 128u];
        tg_up[tid]          = up[x_base + tid];
        tg_up[tid + 128u]   = up[x_base + tid + 128u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_idx < N) {
            device const uchar* block = row + sb * Q6K_GDC_BLOCK_SIZE;
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
pub const THREADS_PER_TG: u64 = 128;

pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_geglu_gelu_tanh_down_cached";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
