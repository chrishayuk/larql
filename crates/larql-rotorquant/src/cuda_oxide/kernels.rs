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

unsafe fn pack_code(codes: &mut DisjointSlice<u8>, bit_pos: usize, bits: usize, code: usize) {
    let byte = bit_pos / 8;
    let shift = bit_pos % 8;
    let mask = (1u32 << bits) - 1;
    let value = (code as u32) & mask;
    let lo = (unsafe { *codes.get_unchecked_mut(byte) }) as u32;
    let hi = if shift + bits > 8 && byte + 1 < codes.len() {
        (unsafe { *codes.get_unchecked_mut(byte + 1) }) as u32
    } else {
        0
    };
    let combined = lo | (hi << 8);
    let cleared = combined & !(mask << shift);
    let inserted = cleared | (value << shift);
    unsafe {
        *codes.get_unchecked_mut(byte) = (inserted & 0xff) as u8;
    }
    if shift + bits > 8 && byte + 1 < codes.len() {
        unsafe {
            *codes.get_unchecked_mut(byte + 1) = ((inserted >> 8) & 0xff) as u8;
        }
    }
}

fn abs_f32(value: f32) -> f32 {
    if value < 0.0 {
        -value
    } else {
        value
    }
}

fn quantized_value(code: usize, is_4bit: bool) -> f32 {
    if is_4bit {
        lm4_value(code)
    } else {
        lm3_value(code)
    }
}

fn quantize_code(value: f32, is_4bit: bool) -> usize {
    let limit = if is_4bit { 16 } else { 8 };
    let mut best_code = 0usize;
    let mut best_err = abs_f32(value - quantized_value(0, is_4bit));
    let mut code = 1usize;
    while code < limit {
        let candidate = quantized_value(code, is_4bit);
        let err = abs_f32(value - candidate);
        if err < best_err {
            best_err = err;
            best_code = code;
        }
        code += 1;
    }
    best_code
}

fn quantize_row(
    input: &[f32],
    codes: &mut DisjointSlice<u8>,
    norms: &mut DisjointSlice<f32>,
    rotation_indices: &mut DisjointSlice<u16>,
    row: usize,
    head_dim: usize,
    bits: usize,
    block_size: usize,
    is_4bit: bool,
) {
    let row_base = row * head_dim;
    let mut norm = 0.0_f32;
    let mut col = 0usize;
    while col < head_dim {
        let value = abs_f32(input[row_base + col]);
        if value > norm {
            norm = value;
        }
        col += 1;
    }
    let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    unsafe {
        *norms.get_unchecked_mut(row) = norm;
    }

    let blocks_per_row = head_dim / block_size;
    let mut block = 0usize;
    while block < blocks_per_row {
        let offset = block * block_size;
        let rot_index = row * blocks_per_row + block;
        let code_base = row * head_dim + offset;
        if block_size == 2 {
            let x = input[row_base + offset] * inv_norm;
            let y = input[row_base + offset + 1] * inv_norm;
            let mut best_rot = 0u16;
            let mut best_code0 = 0usize;
            let mut best_code1 = 0usize;
            let mut best_err = 3.4028235e38_f32;
            let mut rot = 0usize;
            while rot < 8 {
                let c = planar_cos(rot);
                let s = planar_sin(rot);
                let r0 = c * x - s * y;
                let r1 = s * x + c * y;
                let code0 = quantize_code(r0, is_4bit);
                let code1 = quantize_code(r1, is_4bit);
                let q0 = quantized_value(code0, is_4bit);
                let q1 = quantized_value(code1, is_4bit);
                let d0 = q0 - r0;
                let d1 = q1 - r1;
                let err = d0 * d0 + d1 * d1;
                if err < best_err {
                    best_err = err;
                    best_rot = rot as u16;
                    best_code0 = code0;
                    best_code1 = code1;
                }
                rot += 1;
            }
            unsafe {
                *rotation_indices.get_unchecked_mut(rot_index) = best_rot;
                pack_code(codes, code_base * bits, bits, best_code0);
                pack_code(codes, (code_base + 1) * bits, bits, best_code1);
            }
        } else {
            let x = input[row_base + offset] * inv_norm;
            let y = input[row_base + offset + 1] * inv_norm;
            let z = input[row_base + offset + 2] * inv_norm;
            let w = input[row_base + offset + 3] * inv_norm;
            let mut best_rot = 0u16;
            let mut best_code0 = 0usize;
            let mut best_code1 = 0usize;
            let mut best_code2 = 0usize;
            let mut best_code3 = 0usize;
            let mut best_err = 3.4028235e38_f32;
            let mut rot = 0usize;
            while rot < 16 {
                let a = iso_a(rot);
                let b = iso_b(rot);
                let c = iso_c(rot);
                let r0 = a * x + b * y + c * z;
                let r1 = c * x + a * y + b * z;
                let r2 = b * x + c * y + a * z;
                let r3 = w;
                let code0 = quantize_code(r0, is_4bit);
                let code1 = quantize_code(r1, is_4bit);
                let code2 = quantize_code(r2, is_4bit);
                let code3 = quantize_code(r3, is_4bit);
                let q0 = quantized_value(code0, is_4bit);
                let q1 = quantized_value(code1, is_4bit);
                let q2 = quantized_value(code2, is_4bit);
                let q3 = quantized_value(code3, is_4bit);
                let d0 = q0 - r0;
                let d1 = q1 - r1;
                let d2 = q2 - r2;
                let d3 = q3 - r3;
                let err = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                if err < best_err {
                    best_err = err;
                    best_rot = rot as u16;
                    best_code0 = code0;
                    best_code1 = code1;
                    best_code2 = code2;
                    best_code3 = code3;
                }
                rot += 1;
            }
            unsafe {
                *rotation_indices.get_unchecked_mut(rot_index) = best_rot;
                pack_code(codes, code_base * bits, bits, best_code0);
                pack_code(codes, (code_base + 1) * bits, bits, best_code1);
                pack_code(codes, (code_base + 2) * bits, bits, best_code2);
                pack_code(codes, (code_base + 3) * bits, bits, best_code3);
            }
        }
        block += 1;
    }
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
pub fn planar3_quantize_rows(
    input: &[f32],
    codes: DisjointSlice<u8>,
    norms: DisjointSlice<f32>,
    rotation_indices: DisjointSlice<u16>,
    head_dim: usize,
) {
    let row_idx = thread::index_1d();
    let row = row_idx.get();
    if row >= norms.len() {
        return;
    }
    let mut codes = codes;
    let mut norms = norms;
    let mut rotation_indices = rotation_indices;
    quantize_row(
        input,
        &mut codes,
        &mut norms,
        &mut rotation_indices,
        row,
        head_dim,
        3,
        2,
        false,
    );
}

#[kernel]
pub fn planar4_quantize_rows(
    input: &[f32],
    codes: DisjointSlice<u8>,
    norms: DisjointSlice<f32>,
    rotation_indices: DisjointSlice<u16>,
    head_dim: usize,
) {
    let row_idx = thread::index_1d();
    let row = row_idx.get();
    if row >= norms.len() {
        return;
    }
    let mut codes = codes;
    let mut norms = norms;
    let mut rotation_indices = rotation_indices;
    quantize_row(
        input,
        &mut codes,
        &mut norms,
        &mut rotation_indices,
        row,
        head_dim,
        4,
        2,
        true,
    );
}

#[kernel]
pub fn iso4_quantize_rows(
    input: &[f32],
    codes: DisjointSlice<u8>,
    norms: DisjointSlice<f32>,
    rotation_indices: DisjointSlice<u16>,
    head_dim: usize,
) {
    let row_idx = thread::index_1d();
    let row = row_idx.get();
    if row >= norms.len() {
        return;
    }
    let mut codes = codes;
    let mut norms = norms;
    let mut rotation_indices = rotation_indices;
    quantize_row(
        input,
        &mut codes,
        &mut norms,
        &mut rotation_indices,
        row,
        head_dim,
        4,
        4,
        true,
    );
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
