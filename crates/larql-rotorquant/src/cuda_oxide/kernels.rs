use cuda_device::{kernel, thread, DisjointSlice};

use super::device_tables::{iso_a, iso_b, iso_c, lm3_value};

fn unpack_code(codes: &[u8], bit_pos: usize) -> usize {
    let byte = bit_pos / 8;
    let shift = bit_pos % 8;
    let lo = codes[byte] as u32;
    let hi = if byte + 1 < codes.len() {
        codes[byte + 1] as u32
    } else {
        0
    };
    ((lo | (hi << 8)) >> shift & 0x7) as usize
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

    let col = elem - row * head_dim;
    let block = col / 4;
    let lane = col - block * 4;
    let blocks_per_row = head_dim / 4;
    let rot = rotation_indices[row * blocks_per_row + block] as usize;

    let base_code = row * head_dim + block * 4;
    let rotated0 = lm3_value(unpack_code(codes, base_code * 3));
    let rotated1 = lm3_value(unpack_code(codes, (base_code + 1) * 3));
    let rotated2 = lm3_value(unpack_code(codes, (base_code + 2) * 3));
    let rotated3 = lm3_value(unpack_code(codes, (base_code + 3) * 3));

    let a = iso_a(rot);
    let b = iso_b(rot);
    let c = iso_c(rot);
    let recovered = match lane {
        0 => a * rotated0 + c * rotated1 + b * rotated2,
        1 => b * rotated0 + a * rotated1 + c * rotated2,
        2 => c * rotated0 + b * rotated1 + a * rotated2,
        _ => rotated3,
    };

    *slot = recovered * norms[row];
}
