use super::*;

/// **K5 is BIT-IDENTICAL to K3**, for both arms.
///
/// K5 builds the folded scale in a register (`ws * ascale[b]`) where K3
/// built it in a per-row buffer. Multiplying two f32 gives the same f32
/// whether the result goes through memory first, and every operation
/// after that is unchanged — so no new numerical evidence is needed
/// beyond the Bank-1 run K3 already requires.
///
/// Asserted for the ASYMMETRIC arm (the candidate) and the SYMMETRIC one
/// (the control), because K5 changes both.
// K3/K5 are the register-folded SDOT rows; off aarch64 the portable
// definitions run instead and `q8_row_k3_register` is a `todo!()`
// stub ("K5 has no portable arm; the gate runs on aarch64"). The
// arithmetic these pin does not exist on this target.
#[cfg(target_arch = "aarch64")]
#[test]
fn k5_is_bit_identical_to_k3_on_both_arms() {
    use super::super::super::integer::{
        q8_row_asym_k3, q8_row_k3_register, quantise_activation_asymmetric,
        quantise_activation_blocked,
    };
    const OUT: usize = 6;
    // 88 leaves one block past the group of four and an 8-element tail,
    // the two paths an aligned width never reaches.
    for in_dim in [64usize, 88, 256, 5120] {
        let w = lcg_values(OUT * in_dim, 101);
        let x = outlier_activation(in_dim, 102);
        let (codes, wscales) = q8_parts(&w, in_dim);
        let ablock = 16usize;
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

        // --- asymmetric: the candidate ---
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        for o in 0..OUT {
            let ws = &wscales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = ascales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let fm: Vec<f32> = amids
                .iter()
                .enumerate()
                .map(|(b, m)| ws[b / per_weight] * *m)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let k3 = q8_row_asym_k3(row, &fs, &fm, &qx, in_dim);
            let k5 = q8_row_k3_register(row, ws, &ascales, Some(&amids), &qx, in_dim);
            assert_eq!(
                k5.to_bits(),
                k3.to_bits(),
                "asym in_dim {in_dim} row {o}: K5 {k5} vs K3 {k3}"
            );
        }

        // --- symmetric: the control ---
        let (qxs, sscales) = quantise_activation_blocked(&x, ablock);
        for o in 0..OUT {
            let ws = &wscales[o * per_row..(o + 1) * per_row];
            let fs: Vec<f32> = sscales
                .iter()
                .enumerate()
                .map(|(b, a)| ws[b / per_weight] * *a)
                .collect();
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let k3 = super::super::super::integer::q8_row_k3_sym(row, &fs, &qxs, in_dim);
            let k5 = q8_row_k3_register(row, ws, &sscales, None, &qxs, in_dim);
            assert_eq!(
                k5.to_bits(),
                k3.to_bits(),
                "sym in_dim {in_dim} row {o}: K5 {k5} vs K3 {k3}"
            );
        }
    }
}
