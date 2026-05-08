use cuda_device::{kernel, thread, DisjointSlice};

use super::device_tables::{iso_a, iso_b, iso_c, lm3_value, lm4_value, planar_cos, planar_sin};

fn unpack_code(codes: &[u8], bit_pos: usize, bits: usize) -> usize {
    let byte = bit_pos / 8;
    let shift = bit_pos % 8;
    let mask = (1u32 << bits) - 1;
    let lo = codes[byte] as u32;
    let hi = if shift + bits > 8 && byte + 1 < codes.len() {
        codes[byte + 1] as u32
    } else {
        0
    };
    ((lo | (hi << 8)) >> shift & mask) as usize
}

fn dequantize_planar_elem(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    elem: usize,
    head_dim: usize,
    bits: usize,
    is_4bit: bool,
) -> f32 {
    let row = elem / head_dim;
    let col = elem - row * head_dim;
    let block = col / 2;
    let lane = col - block * 2;
    let blocks_per_row = head_dim / 2;
    let rot = rotation_indices[row * blocks_per_row + block] as usize;

    let base_code = row * head_dim + block * 2;
    let code0 = unpack_code(codes, base_code * bits, bits);
    let code1 = unpack_code(codes, (base_code + 1) * bits, bits);
    let rotated0 = if is_4bit {
        lm4_value(code0)
    } else {
        lm3_value(code0)
    };
    let rotated1 = if is_4bit {
        lm4_value(code1)
    } else {
        lm3_value(code1)
    };

    let c = planar_cos(rot);
    let s = planar_sin(rot);
    let recovered = if lane == 0 {
        c * rotated0 + s * rotated1
    } else {
        -s * rotated0 + c * rotated1
    };
    recovered * norms[row]
}

fn dequantize_iso_elem(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    elem: usize,
    head_dim: usize,
    bits: usize,
    is_4bit: bool,
) -> f32 {
    let row = elem / head_dim;
    let col = elem - row * head_dim;
    let block = col / 4;
    let lane = col - block * 4;
    let blocks_per_row = head_dim / 4;
    let rot = rotation_indices[row * blocks_per_row + block] as usize;

    let base_code = row * head_dim + block * 4;
    let code0 = unpack_code(codes, base_code * bits, bits);
    let code1 = unpack_code(codes, (base_code + 1) * bits, bits);
    let code2 = unpack_code(codes, (base_code + 2) * bits, bits);
    let code3 = unpack_code(codes, (base_code + 3) * bits, bits);
    let rotated0 = if is_4bit {
        lm4_value(code0)
    } else {
        lm3_value(code0)
    };
    let rotated1 = if is_4bit {
        lm4_value(code1)
    } else {
        lm3_value(code1)
    };
    let rotated2 = if is_4bit {
        lm4_value(code2)
    } else {
        lm3_value(code2)
    };
    let rotated3 = if is_4bit {
        lm4_value(code3)
    } else {
        lm3_value(code3)
    };

    let a = iso_a(rot);
    let b = iso_b(rot);
    let c = iso_c(rot);
    let recovered = match lane {
        0 => a * rotated0 + c * rotated1 + b * rotated2,
        1 => b * rotated0 + a * rotated1 + c * rotated2,
        2 => c * rotated0 + b * rotated1 + a * rotated2,
        _ => rotated3,
    };
    recovered * norms[row]
}

#[kernel]
pub fn planar3_dequantize_block(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    mut out: DisjointSlice<f32>,
    head_dim: usize,
) {
    let idx = thread::index_1d();
    let elem = idx.get();
    let Some(slot) = out.get_mut(idx) else {
        return;
    };
    if elem / head_dim >= norms.len() {
        return;
    }
    *slot = dequantize_planar_elem(codes, norms, rotation_indices, elem, head_dim, 3, false);
}

#[kernel]
pub fn planar4_dequantize_block(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    mut out: DisjointSlice<f32>,
    head_dim: usize,
) {
    let idx = thread::index_1d();
    let elem = idx.get();
    let Some(slot) = out.get_mut(idx) else {
        return;
    };
    if elem / head_dim >= norms.len() {
        return;
    }
    *slot = dequantize_planar_elem(codes, norms, rotation_indices, elem, head_dim, 4, true);
}

#[kernel]
pub fn iso3_dequantize_block(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    mut out: DisjointSlice<f32>,
    head_dim: usize,
) {
    let idx = thread::index_1d();
    let elem = idx.get();
    let Some(slot) = out.get_mut(idx) else {
        return;
    };

    let row = elem / head_dim;
    if row >= norms.len() {
        return;
    }
    *slot = dequantize_iso_elem(codes, norms, rotation_indices, elem, head_dim, 3, false);
}

#[kernel]
pub fn iso4_dequantize_block(
    codes: &[u8],
    norms: &[f32],
    rotation_indices: &[u16],
    mut out: DisjointSlice<f32>,
    head_dim: usize,
) {
    let idx = thread::index_1d();
    let elem = idx.get();
    let Some(slot) = out.get_mut(idx) else {
        return;
    };

    if elem / head_dim >= norms.len() {
        return;
    }
    *slot = dequantize_iso_elem(codes, norms, rotation_indices, elem, head_dim, 4, true);
}
