//! Q6_K × Q8_1 mmvq matvec — INT8 SIMD path via `__dp4a`.
//!
//! **NOT lifted from upstream.** LARQL's `quantize_q6_k`
//! produces an adjacent-pair packed layout (ql byte i holds
//! q6[2i] in low nibble + q6[2i+1] in high nibble; qh byte
//! i/4 holds 4 sequential q6 high-2-bits) which is
//! incompatible with GGUF's interleaved Q6_K layout. The dot
//! impl below is LARQL-native: each iqs ∈ {0..31} covers 8
//! contiguous q6 values from one sub-block, multiplied by 8
//! q8 quants from one Q8_1 block, scaled by one i8 scale.
//! Same `dp4a.s32.s32` + `vsub4.s32.s32.s32.sat` PTX
//! intrinsics established by `cuda-q4k-mmvq-int8`.
//!
//! Q6_K layout per 256-element super-block (210 bytes):
//!   bytes [0..128]   ql[128]      ← lower 4 bits per weight
//!   bytes [128..192] qh[64]       ← upper 2 bits per weight
//!   bytes [192..208] scales[16]   ← i8 sub-block scales (16 sub-blocks of 16)
//!   bytes [208..210] d (fp16)     ← super-block scale
//!
//! `cuda-q6k-mmvq` Phase 1.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::elem::Q8_1Buf;
use super::error::CudaInitError;

const Q6K_BLOCK_ELEMS: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;
const WARP_SIZE: u32 = 32;
/// 4 warps per CUDA block — 4 output rows per block. Same heuristic as q4k_mmvq.
const ROWS_PER_BLOCK: u32 = 4;

const Q6K_MMVQ_SRC: &str = r#"
__device__ float larql_f16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int mant = (unsigned int)h & 0x03ffu;
    int exp = ((int)h >> 10) & 0x1f;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0u) {
            bits = sign;
        } else {
            float v = (float)mant * 5.960464477539063e-8f;
            return (sign != 0u) ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 112) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

__device__ int dp4a_s32(int a, int b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Read 4 bytes as int32 from a 2-byte-aligned base. Q6_K super-blocks
// are 210 bytes; consecutive super-blocks land on 2-aligned (but not
// 4-aligned) addresses, so direct (uint32_t*)cast reads are illegal.
// Mirrors upstream's `get_int_b2`.
__device__ int get_int_b2(const unsigned char* p, int i32) {
    const unsigned short* p16 = (const unsigned short*)p;
    int v  = (int)p16[2 * i32 + 0];
    v     |= (int)p16[2 * i32 + 1] << 16;
    return v;
}

// Saturated subtract of 4 packed s8 values: r = a -[saturated]- b.
__device__ int vsub4_s32_sat(int a, int b) {
    int r;
    asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;"
        : "=r"(r) : "r"(a), "r"(b), "r"(0));
    return r;
}

// Pack 4 q6 6-bit values (in {0..63}) from a vl int + 1 qh
// byte into a single int with one q6 per byte. The vl int
// holds 8 consecutive q6 low nibbles (LARQL's adjacent-pair
// layout: ql byte i holds q6[2i] in low nibble + q6[2i+1] in
// high nibble). The qh byte holds 4 consecutive q6 high
// 2-bits (bits 0..1: q6[base+0], 2..3: +1, 4..5: +2, 6..7: +3).
// The four returned bytes hold q6[base..base+3] each as a
// 6-bit value in 0..63.
__device__ int build_q6_quad_unsigned(int vl, unsigned char qh_byte) {
    int b0 = ( vl        & 0x0F) | (((int)qh_byte >> 0 & 0x03) << 4);
    int b1 = ((vl >>  4) & 0x0F) | (((int)qh_byte >> 2 & 0x03) << 4);
    int b2 = ((vl >>  8) & 0x0F) | (((int)qh_byte >> 4 & 0x03) << 4);
    int b3 = ((vl >> 12) & 0x0F) | (((int)qh_byte >> 6 & 0x03) << 4);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// LARQL's Q6_K layout is adjacent-pair packed (q6[2i], q6[2i+1]
// share ql[i]; qh[i/4] holds 4 sequential q6 high-2-bits).
// Each iqs ∈ {0..31} covers 8 contiguous q6 values at
// positions [8*iqs..8*iqs+7], all within sub-block iqs/2.
//
// The upstream GGUF Q6_K layout is interleaved differently
// (q1/q2/q3/q4 spread across the 128 ql bytes); upstream's
// vec_dot_q6_K_q8_1_impl_mmvq does NOT apply here. This impl
// is LARQL-native.
__device__ float vec_dot_q6_K_q8_1_larql(
    const unsigned char* bq6_K,        // 210-byte Q6_K super-block
    const unsigned char* y_q8_1_base,  // start of Q8_1 input blocks
    int kby,                           // Q8_1 block index aligned with this super-block
    int iqs                            // {0..31}
) {
    const unsigned char* ql     = bq6_K;
    const unsigned char* qh     = bq6_K + 128;
    const signed char*   scales = (const signed char*)(bq6_K + 192);
    float d_f = larql_f16_to_f32(
        (unsigned short)bq6_K[208] | ((unsigned short)bq6_K[209] << 8)
    );

    int sub          = iqs / 2;          // sub-block index (0..15)
    int q8_block_off = iqs / 4;          // Q8_1 block within super-block (0..7)
    int q8_chunk     = 2 * (iqs % 4);    // first int chunk in that block (0,2,4,6)

    int vl = get_int_b2(ql, iqs);        // 4 ql bytes = 8 q6 low nibbles
    unsigned char qh_lo = qh[2 * iqs + 0];  // first 4 q6 high-2-bits
    unsigned char qh_hi = qh[2 * iqs + 1];  // next  4 q6 high-2-bits

    // Build two INT32s, each holding 4 q6 (unsigned 0..63) in bytes.
    int vi_unsigned_lo = build_q6_quad_unsigned( vl,        qh_lo);
    int vi_unsigned_hi = build_q6_quad_unsigned( vl >> 16,  qh_hi);
    // Centre to signed -32..31 with per-byte saturated subtract.
    int vi_lo = vsub4_s32_sat(vi_unsigned_lo, 0x20202020);
    int vi_hi = vsub4_s32_sat(vi_unsigned_hi, 0x20202020);

    // Read the matching Q8_1 chunks. Both chunks live in the
    // same Q8_1 block (q8_chunk and q8_chunk+1).
    const unsigned char* bq8 = y_q8_1_base + (size_t)(kby + q8_block_off) * 36;
    float scale_q8 = larql_f16_to_f32(
        (unsigned short)bq8[0] | ((unsigned short)bq8[1] << 8)
    );
    const unsigned int* q8 = (const unsigned int*)(bq8 + 4);
    int u_lo = q8[q8_chunk + 0];
    int u_hi = q8[q8_chunk + 1];

    int sc = (int)scales[sub];

    int sumi = dp4a_s32(vi_lo, u_lo, 0) + dp4a_s32(vi_hi, u_hi, 0);
    return d_f * scale_q8 * (float)sc * (float)sumi;
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_f32(
    const unsigned char* __restrict__ vbq,   // packed Q6_K weights
    const unsigned char* __restrict__ vy,    // packed block_q8_1[]
    float*               __restrict__ dst,   // output [rows]
    int rows,
    int n_super_blocks                       // hidden / 256
) {
    int tid_x = threadIdx.x;                 // 0..31 (lane within warp)
    int tid_y = threadIdx.y;                 // 0..ROWS_PER_BLOCK-1
    int row   = blockIdx.x * blockDim.y + tid_y;
    if (row >= rows) return;

    int iqs = tid_x;                         // each lane handles one iqs ∈ {0..31}

    const unsigned char* row_base =
        vbq + (size_t)row * (size_t)n_super_blocks * 210ull;

    float partial = 0.0f;
    // One super-block per warp iter; 32 lanes cover all 32 iqs values.
    for (int kbx = 0; kbx < n_super_blocks; ++kbx) {
        int kby = kbx * 8;  // 8 Q8_1 blocks per Q6_K super-block
        partial += vec_dot_q6_K_q8_1_larql(
            row_base + (size_t)kbx * 210,
            vy,
            kby,
            iqs
        );
    }

    // Warp-reduce.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, o);
    }

    if (tid_x == 0) {
        dst[row] = partial;
    }
}
"#;

static Q6K_MMVQ_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn q6k_mmvq_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q6K_MMVQ_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        arch: Some("compute_61"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(Q6K_MMVQ_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile q6k_mmvq: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q6k_mmvq module: {e:?}")))?;
    let func = module
        .load_function("mul_mat_vec_q6_K_q8_1_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q6k_mmvq function: {e:?}")))?;
    let _ = Q6K_MMVQ_FUNC.set((module, func));
    let (_, f) = Q6K_MMVQ_FUNC.get().unwrap();
    Ok(f)
}

/// Q6_K × Q8_1 device matvec. The packed Q6_K weight bytes are
/// cached on the device on first use via
/// `with_q6k_packed_device_buf` (parallel to the Q4_K cache).
pub(crate) fn matvec_device(
    backend: &CudaBackend,
    q6k_data: &[u8],
    x_q8_1: &Q8_1Buf,
    rows: usize,
    hidden: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let drv = backend.driver();
    let mut y_dev = drv.device_alloc_uninit(rows)?;
    matvec_device_into(backend, q6k_data, x_q8_1, &mut y_dev, rows, hidden)?;
    Ok(y_dev)
}

/// `cuda-decode-cuda-graph` companion: write q6k_mmvq output into a
/// caller-provided buffer.
pub(crate) fn matvec_device_into(
    backend: &CudaBackend,
    q6k_data: &[u8],
    x_q8_1: &Q8_1Buf,
    y_dev: &mut CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<(), CudaInitError> {
    if rows == 0 || hidden == 0 || !hidden.is_multiple_of(Q6K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q6k_mmvq shape rows={rows} hidden={hidden}",
        )));
    }
    let n_super_blocks = hidden / Q6K_BLOCK_ELEMS;
    let expected = rows
        .checked_mul(n_super_blocks)
        .and_then(|v| v.checked_mul(Q6K_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q6k byte size overflow".to_string()))?;
    if q6k_data.len() != expected {
        return Err(CudaInitError::DriverMissing(format!(
            "q6k byte length mismatch: got {}, expected {expected}",
            q6k_data.len()
        )));
    }
    if x_q8_1.n_blocks * 32 != hidden {
        return Err(CudaInitError::DriverMissing(format!(
            "q8_1 input block count mismatch: {} blocks for hidden={hidden}",
            x_q8_1.n_blocks,
        )));
    }
    if y_dev.len() != rows {
        return Err(CudaInitError::DriverMissing(format!(
            "q6k_mmvq y length mismatch: y={} != rows={rows}",
            y_dev.len(),
        )));
    }

    let drv = backend.driver();
    let func = q6k_mmvq_function(drv)?;
    let rows_i = rows as i32;
    let n_super_blocks_i = n_super_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
        block_dim: (WARP_SIZE, ROWS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    backend.with_q6k_packed_device_buf(q6k_data, |q6k_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(q6k_dev)
                .arg(&x_q8_1.bytes)
                .arg(y_dev)
                .arg(&rows_i)
                .arg(&n_super_blocks_i)
                .launch(cfg)
                .map_err(|e| CudaInitError::DriverMissing(format!("launch q6k_mmvq: {e:?}")))?;
        }
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::elem::quantize_q8_1_device;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            eprintln!("skipping q6k_mmvq parity test: set LARQL_CUDA_AVAILABLE=1");
            return None;
        }
        CudaBackend::new().ok()
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 0.5
            })
            .collect()
    }

    /// Q6_K mmvq SHALL match the existing f32-cached path on the
    /// SAME Q8_1-dequantised input within ≤ 1e-3 max-element. Same
    /// "isolate kernel arithmetic from Q8_1 noise" trick as in
    /// `q4k_mmvq::tests`.
    #[test]
    fn q6k_mmvq_matches_q6k_f32_on_dequantized_input() {
        use crate::cpu::ops::q4_common::quantize_q6_k;
        use larql_models::quant::half::f16_to_f32;

        let Some(backend) = gpu_or_skip() else { return };

        for &(rows, hidden, seed) in &[(64usize, 256usize, 0x6100u64), (2560, 10240, 0x6200)] {
            let w_q6k = quantize_q6_k(&synth(rows * hidden, seed ^ 0xA1));
            let x = synth(hidden, seed ^ 0xB1);

            // Q8_1 quantise + dequantise on host for the f32 reference.
            let x_dev = backend.htod_f32(&x).expect("htod x");
            let q8_1 = quantize_q8_1_device(&backend, &x_dev, hidden).expect("quantize_q8_1");
            backend.driver().sync().expect("sync");
            let q8_1_bytes: Vec<u8> = backend
                .driver()
                .stream
                .clone_dtoh(&q8_1.bytes)
                .expect("dtoh q8_1");
            let mut x_dq = Vec::with_capacity(hidden);
            for b in 0..(hidden / 32) {
                let base = b * 36;
                let scale =
                    f16_to_f32(u16::from_le_bytes([q8_1_bytes[base], q8_1_bytes[base + 1]]));
                for i in 0..32 {
                    let q = q8_1_bytes[base + 4 + i] as i8;
                    x_dq.push(scale * (q as f32));
                }
            }
            let x_dq_dev = backend.htod_f32(&x_dq).expect("htod x_dq");
            let f32_ref = backend
                .q6k_matvec_device(&w_q6k, &x_dq_dev, rows, hidden)
                .expect("q6k_f32");
            backend.driver().sync().expect("sync");
            let f32_host = backend.driver().to_host(&f32_ref).expect("dtoh f32_ref");

            let mmvq = matvec_device(&backend, &w_q6k, &q8_1, rows, hidden).expect("q6k_mmvq");
            backend.driver().sync().expect("sync");
            let mmvq_host = backend.driver().to_host(&mmvq).expect("dtoh mmvq");

            let max_diff = f32_host
                .iter()
                .zip(&mmvq_host)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // 1e-3 holds at small shapes; longer reductions accumulate
            // fp32 rounding noise (1 ULP at ~1.0 is ~1.2e-7; over 10K
            // accumulations, the worst-case grows to ~1e-3 territory).
            // Use a hidden-aware tolerance.
            let tol = if hidden <= 1024 { 1e-3 } else { 5e-3 };
            assert!(
                max_diff <= tol,
                "rows={rows} hidden={hidden}: max diff {max_diff} > {tol} between \
                 q6k_f32 and q6k_mmvq",
            );
        }
    }
}
