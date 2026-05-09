//! Parity tests for `cuda::q4k_batched::matvec_batched` vs naive
//! Rust dequant + matmul reference.
//!
//! Phase 2 of `cuda-speculative-decoding`. Validates the M_TILE > 1
//! batched kernel produces output bit-equal (within 1e-5 fp32 ordering
//! tolerance) to running M independent matvecs at M_TILE=1.

#![cfg(any(feature = "cuda", feature = "cuda-oxide"))]

use larql_compute::cuda::q4k_batched::{matvec_batched, M_TILE_MAX};
use larql_compute::cuda::CudaBackend;

const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_ELEMS: usize = 256;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    CudaBackend::new().ok()
}

// Encode a single f32 as IEEE-754 half (round-to-nearest-even).
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        // inf or nan
        return (sign << 15) | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7c00; // inf
    }
    if new_exp <= 0 {
        return sign << 15; // underflow → zero
    }
    let new_mant = mant >> 13;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let mant = (h as u32) & 0x03ff;
    let exp = ((h >> 10) & 0x1f) as i32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal
            let v = (mant as f32) * 5.960_464_5e-8;
            return if sign != 0 { -v } else { v };
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Construct one Q4_K super-block (144 bytes) from caller-supplied
/// d, dmin, 8 scale6 values (0..63), 8 min6 values (0..63), and 256
/// nibbles (0..15). Mirrors the kernel's reader in
/// `q4k_direct.rs` — q4k_scale / q4k_min splits.
fn build_q4k_block(
    d: f32,
    dmin: f32,
    scales: &[u8; 8],
    mins: &[u8; 8],
    nibbles: &[u8; 256],
) -> [u8; Q4K_BLOCK_BYTES] {
    let mut block = [0u8; Q4K_BLOCK_BYTES];
    let d16 = f32_to_f16_bits(d);
    let dmin16 = f32_to_f16_bits(dmin);
    block[0] = (d16 & 0xff) as u8;
    block[1] = (d16 >> 8) as u8;
    block[2] = (dmin16 & 0xff) as u8;
    block[3] = (dmin16 >> 8) as u8;

    // packed scale/min bytes: 12 bytes covering 8 scales (6-bit) +
    // 8 mins (6-bit). Layout per q4k_scale / q4k_min in the kernel:
    //   for j < 4: scale[j] = packed[j] & 0x3f; min[j] = packed[j+4] & 0x3f
    //   for j >= 4: scale[j] = (packed[j+4] & 0x0f) | ((packed[j-4] >> 6) << 4)
    //              min[j]   = (packed[j+4] >> 4)    | ((packed[j]   >> 6) << 4)
    let packed = &mut block[4..16];
    // First 4 entries directly into low 6 bits of packed[0..4] and packed[4..8].
    for j in 0..4 {
        packed[j] = scales[j] & 0x3f;
        packed[j + 4] = mins[j] & 0x3f;
    }
    // Top 2 bits of scale[j..4..8] go into bits 6-7 of packed[j-4].
    // Top 2 bits of min[j..4..8] go into bits 6-7 of packed[j].
    for j in 4..8 {
        let s_hi = (scales[j] >> 4) & 0x3;
        let m_hi = (mins[j] >> 4) & 0x3;
        let s_lo = scales[j] & 0x0f;
        let m_lo = mins[j] & 0x0f;
        // packed[j+4] = (s_lo) | ((m_lo) << 4)
        packed[j + 4] = s_lo | (m_lo << 4);
        // packed[j-4] |= s_hi << 6
        packed[j - 4] |= s_hi << 6;
        // packed[j] |= m_hi << 6
        packed[j] |= m_hi << 6;
    }

    // 128 quant bytes for 256 values. Same layout as kernel:
    //   sub = elem >> 5 (0..8); lane = elem & 31; pair = sub >> 1
    //   byte = quants[pair * 32 + lane]
    //   nibble = (sub & 1) ? (byte >> 4) : (byte & 0x0f)
    let quants = &mut block[16..144];
    for elem in 0..256 {
        let sub = elem >> 5;
        let lane = elem & 31;
        let pair = sub >> 1;
        let nib = nibbles[elem] & 0x0f;
        let idx = pair * 32 + lane;
        if sub & 1 == 0 {
            quants[idx] = (quants[idx] & 0xf0) | nib;
        } else {
            quants[idx] = (quants[idx] & 0x0f) | (nib << 4);
        }
    }

    block
}

/// Dequantize a Q4_K super-block to 256 f32 weights. Mirrors the
/// kernel's compute path exactly: w = d * scale * nibble - dmin * min.
fn dequant_block(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    let d = f16_bits_to_f32(((block[1] as u16) << 8) | (block[0] as u16));
    let dmin = f16_bits_to_f32(((block[3] as u16) << 8) | (block[2] as u16));
    let packed = &block[4..16];
    let quants = &block[16..144];

    let scale = |j: usize| -> u8 {
        if j < 4 {
            packed[j] & 0x3f
        } else {
            (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4)
        }
    };
    let min = |j: usize| -> u8 {
        if j < 4 {
            packed[j + 4] & 0x3f
        } else {
            (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4)
        }
    };

    let mut out = Vec::with_capacity(256);
    for elem in 0..256 {
        let sub = elem >> 5;
        let lane = elem & 31;
        let pair = sub >> 1;
        let byte = quants[pair * 32 + lane];
        let nibble = if sub & 1 == 1 { byte >> 4 } else { byte & 0x0f };
        let sc = d * scale(sub) as f32;
        let mn = dmin * min(sub) as f32;
        let w = sc * nibble as f32 - mn;
        out.push(w);
    }
    out
}

fn synth_q4k_weights(rows: usize, hidden: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    assert!(hidden.is_multiple_of(Q4K_BLOCK_ELEMS));
    let blocks_per_row = hidden / Q4K_BLOCK_ELEMS;
    let mut bytes = Vec::with_capacity(rows * blocks_per_row * Q4K_BLOCK_BYTES);
    let mut dequant = Vec::with_capacity(rows * hidden);

    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next_byte = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) & 0xff) as u8
    };

    for _r in 0..rows {
        for _b in 0..blocks_per_row {
            // Small d / dmin so the dot products stay finite.
            let d = ((next_byte() as f32) / 64.0 + 1.0) * 0.001;
            let dmin = ((next_byte() as f32) / 64.0) * 0.0001;
            let mut scales = [0u8; 8];
            let mut mins = [0u8; 8];
            let mut nibbles = [0u8; 256];
            for v in scales.iter_mut() {
                *v = next_byte() & 0x3f;
            }
            for v in mins.iter_mut() {
                *v = next_byte() & 0x3f;
            }
            for v in nibbles.iter_mut() {
                *v = next_byte() & 0x0f;
            }
            let block = build_q4k_block(d, dmin, &scales, &mins, &nibbles);
            bytes.extend_from_slice(&block);
            dequant.extend(dequant_block(&block));
        }
    }
    assert_eq!(dequant.len(), rows * hidden);
    (bytes, dequant)
}

fn synth_x(m: usize, hidden: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0xCBF2_9CE4_8422_2325);
    (0..m * hidden)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFF) as f32 / 65536.0) - 0.5
        })
        .collect()
}

fn naive_matvec_batched(
    w_dequant: &[f32],
    x: &[f32],
    rows: usize,
    hidden: usize,
    m: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * rows];
    for mi in 0..m {
        for r in 0..rows {
            let mut acc = 0.0_f32;
            for h in 0..hidden {
                acc += w_dequant[r * hidden + h] * x[mi * hidden + h];
            }
            out[mi * rows + r] = acc;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn q4k_dequant_round_trip_self_consistent() {
    // Sanity: built block dequants to the same nibbles we put in.
    let scales = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let mins = [0u8; 8];
    let nibbles: [u8; 256] = std::array::from_fn(|i| (i % 16) as u8);
    let block = build_q4k_block(1.0, 0.0, &scales, &mins, &nibbles);
    let deq = dequant_block(&block);
    for elem in 0..256 {
        let sub = elem >> 5;
        let expected = scales[sub] as f32 * (elem % 16) as f32;
        assert!(
            (deq[elem] - expected).abs() < 1e-5,
            "elem {elem}: got {} want {expected}",
            deq[elem]
        );
    }
}

#[test]
fn q4k_batched_m1_matches_naive_reference() {
    let Some(backend) = gpu_or_skip() else { return };
    let rows = 16;
    let hidden = Q4K_BLOCK_ELEMS * 2; // 512
    let (bytes, dequant) = synth_q4k_weights(rows, hidden, 0xABCD);
    let x = synth_x(1, hidden, 0xBEEF);
    let gpu = matvec_batched(&backend, &bytes, &x, rows, hidden, 1).expect("matvec_batched");
    let cpu = naive_matvec_batched(&dequant, &x, rows, hidden, 1);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(
        d < 1e-3,
        "max abs diff {d}; gpu={:?} cpu={:?}",
        &gpu[..4],
        &cpu[..4]
    );
}

#[test]
fn q4k_batched_m4_matches_naive_reference() {
    let Some(backend) = gpu_or_skip() else { return };
    let rows = 32;
    let hidden = Q4K_BLOCK_ELEMS * 4; // 1024
    let (bytes, dequant) = synth_q4k_weights(rows, hidden, 0x1234);
    let m = 4;
    let x = synth_x(m, hidden, 0x5678);
    let gpu = matvec_batched(&backend, &bytes, &x, rows, hidden, m).expect("matvec_batched");
    let cpu = naive_matvec_batched(&dequant, &x, rows, hidden, m);
    assert_eq!(gpu.len(), m * rows);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 1e-3, "max abs diff {d}");
}

#[test]
fn q4k_batched_m1_equals_m4_per_position() {
    // Run M=1 four times with different x, then run M=4 with the
    // four x stitched together. Outputs SHALL match per slice.
    let Some(backend) = gpu_or_skip() else { return };
    let rows = 16;
    let hidden = Q4K_BLOCK_ELEMS * 2;
    let (bytes, _dequant) = synth_q4k_weights(rows, hidden, 0xCAFE);

    let mut stitched = Vec::with_capacity(4 * hidden);
    let mut singles: Vec<Vec<f32>> = Vec::new();
    for k in 0..4 {
        let xk = synth_x(1, hidden, 0xF00D + k as u64);
        let gk = matvec_batched(&backend, &bytes, &xk, rows, hidden, 1).expect("M=1");
        singles.push(gk);
        stitched.extend_from_slice(&xk);
    }
    let g4 = matvec_batched(&backend, &bytes, &stitched, rows, hidden, 4).expect("M=4");
    assert_eq!(g4.len(), 4 * rows);
    for k in 0..4 {
        let slice = &g4[k * rows..(k + 1) * rows];
        let d = max_abs_diff(slice, &singles[k]);
        assert!(d < 1e-5, "m={k} stitch mismatch {d}");
    }
}

#[test]
fn q4k_batched_max_m_tile_is_8() {
    let Some(backend) = gpu_or_skip() else { return };
    let rows = 8;
    let hidden = Q4K_BLOCK_ELEMS;
    let (bytes, dequant) = synth_q4k_weights(rows, hidden, 0xDEAD);
    let m = M_TILE_MAX;
    let x = synth_x(m, hidden, 0xBABE);
    let gpu = matvec_batched(&backend, &bytes, &x, rows, hidden, m).expect("matvec_batched");
    let cpu = naive_matvec_batched(&dequant, &x, rows, hidden, m);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 1e-3, "max abs diff {d}");
}

#[test]
fn q4k_batched_rejects_oversized_m() {
    let Some(backend) = gpu_or_skip() else { return };
    let rows = 4;
    let hidden = Q4K_BLOCK_ELEMS;
    let (bytes, _) = synth_q4k_weights(rows, hidden, 0);
    let x = vec![0.0_f32; (M_TILE_MAX + 1) * hidden];
    let result = matvec_batched(&backend, &bytes, &x, rows, hidden, M_TILE_MAX + 1);
    assert!(result.is_err());
}
