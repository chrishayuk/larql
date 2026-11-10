use super::*;

/// **With equal sub-block scales, sub-blocking reduces to whole-block.**
///
/// The reduction that has to hold: if every activation sub-block inside a
/// weight block carries the SAME scale, then scaling per sub-block and
/// scaling per block describe the same arithmetic, and a generalisation
/// that disagreed there would be a second implementation rather than a
/// generalisation.
///
/// `ablock == block` is deliberately NOT tested here — a sub-block would
/// span both nibble runs, which is out of this kernel's contract and is
/// why the caller routes that case to `q4_row`. The contract is asserted
/// in the kernel, and the routing is checked below.
#[test]
fn subblocking_reduces_to_whole_block_when_the_scales_agree() {
    use super::super::super::integer::{q4_row_portable, q4_row_subblocked};
    const IN: usize = 256;
    let w = lcg_values(IN, 53);
    let (packed, wscales) = q4_parts(&w, IN);
    let x = outlier_activation(IN, 54);
    let (qx, _) = super::super::super::integer::quantise_activation_blocked(&x, Q4_BLOCK);

    let ablock = Q4_BLOCK / 2;
    let per_weight = Q4_BLOCK / ablock;
    // One scale per weight block, repeated across its sub-blocks.
    let folded_sub: Vec<f32> = (0..IN / ablock).map(|s| wscales[s / per_weight]).collect();
    let folded_whole: Vec<f32> = wscales.to_vec();

    let sub = q4_row_subblocked(&packed, &folded_sub, &qx, IN, Q4_BLOCK, ablock);
    let whole = q4_row_portable(&packed, &folded_whole, &qx, IN, Q4_BLOCK);
    let rel = ((sub - whole) as f64 / (whole as f64).abs().max(1e-12)).abs();
    assert!(
        rel < 1e-6,
        "sub-blocked {sub} vs whole-block {whole} (rel {rel:.2e})"
    );
}

/// The whole-block case is routed away from the sub-blocked kernel, and
/// the kernel refuses it rather than reading past a block's bytes.
#[test]
#[should_panic(expected = "q4 sub-blocking needs ablock")]
fn the_subblocked_kernel_refuses_a_straddling_block() {
    use super::super::super::integer::q4_row_subblocked;
    let packed = vec![0u8; 32];
    let folded = vec![1.0f32; 1];
    let qx = vec![1i8; 64];
    q4_row_subblocked(&packed, &folded, &qx, 64, Q4_BLOCK, Q4_BLOCK);
}

/// **The mechanism claim, made falsifiable: a finer activation block must
/// REDUCE error on an outlier-laden activation.**
///
/// This is the whole justification for spending a rung on the activation
/// rather than on the weight format. An outlier channel in a block of 64
/// crushes 63 neighbours; in a block of 16 it crushes 15. If the error
/// did NOT fall with the block, the diagnosis would be wrong and the
/// activation programme would be chasing the wrong variable.
///
/// Stated as a strict ordering rather than a threshold, because the
/// magnitude is what the bank measures and a number pinned here would be
/// a second, weaker claim about the same thing.
#[test]
fn a_finer_activation_block_reduces_error_on_an_outlier_activation() {
    const IN: usize = 1024;
    let x = outlier_activation(IN, 55);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / IN as f64).sqrt();
    assert!(
        peak as f64 / rms > 10.0,
        "the fixture must actually be heavy-tailed or it tests nothing"
    );

    let err = |ablock: usize| {
        let (qx, ascales) = super::super::super::integer::quantise_activation_blocked(&x, ablock);
        let se: f64 = x
            .iter()
            .zip(&qx)
            .enumerate()
            .map(|(i, (v, q))| {
                let back = *q as f64 * ascales[i / ablock] as f64;
                (back - *v as f64).powi(2)
            })
            .sum();
        (se / IN as f64).sqrt()
    };

    let (e64, e32, e16) = (err(64), err(32), err(16));
    assert!(
        e16 < e32 && e32 < e64,
        "finer activation blocks did not reduce reconstruction error: \
         64 -> {e64:.3e}, 32 -> {e32:.3e}, 16 -> {e16:.3e}"
    );
    // And the per-tensor scale must be the worst of all — the arm CPU-5
    // measured at KL 0.00061 with exact weights.
    let act = quantise_activation(&x);
    let tensor: f64 = (x
        .iter()
        .zip(&act.codes)
        .map(|(v, c)| (*c as f64 * act.scale as f64 - *v as f64).powi(2))
        .sum::<f64>()
        / IN as f64)
        .sqrt();
    assert!(
        tensor > e64,
        "per-tensor {tensor:.3e} vs block-64 {e64:.3e}"
    );
}

/// **The asymmetric kernel computes what the format denotes.**
///
/// Bit-identity again: both block sums are integer and exact, and the
/// two float operations per block happen in the same order, so the
/// vectorised path and the definition cannot legitimately differ.
#[test]
fn the_asymmetric_q8_kernel_computes_what_the_format_denotes() {
    use super::super::super::integer::{q8_row_asym_portable, quantise_activation_asymmetric};
    const OUT: usize = 4;
    // 88 = five 16-wide blocks plus an 8-element tail: one group of four,
    // one block past it, then the scalar remainder.
    for in_dim in [64usize, 88, 192, 1024] {
        let w = lcg_values(OUT * in_dim, 61);
        let x = outlier_activation(in_dim, 62);
        let (codes, wscales) = q8_parts(&w, in_dim);
        let ablock = 16usize;
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

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
            // The EXACT path explicitly: K3 is the process default and
            // reassociates by design, so the denotation gate has to name
            // the implementation it is pinning.
            let got =
                super::super::super::integer::q8_row_asym_exact(row, &fs, &fm, &qx, in_dim, ablock);
            let want = q8_row_asym_portable(row, &fs, &fm, &qx, in_dim, ablock);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "asym row {o} at in_dim {in_dim}: {got} vs definition {want}"
            );
        }
    }
}

/// **A constant block reconstructs EXACTLY**, which a symmetric code
/// cannot do.
///
/// The clearest statement of what the offset buys: a block whose values
/// are all the same has zero span, so every code is zero and the offset
/// alone carries it. Symmetric coding has to represent that value as a
/// multiple of `peak/127` and rounds it.
#[test]
fn an_offset_code_reconstructs_a_constant_block_exactly() {
    use super::super::super::integer::quantise_activation_asymmetric;
    let x = vec![0.7314f32; 32];
    let (codes, scales, mids) = quantise_activation_asymmetric(&x, 16);
    assert!(codes.iter().all(|c| *c == 0));
    for (b, m) in mids.iter().enumerate() {
        assert_eq!(*m, x[0], "block {b} offset {m} is not the constant itself");
        assert_eq!(scales[b], 1.0, "a zero-span block takes the sentinel scale");
    }
}

/// **The offset must EARN its place: it beats the symmetric code on a
/// one-sided block, and does not lose on a balanced one.**
///
/// The mechanism claim behind the whole rung, made falsifiable. If a
/// per-block offset did not reduce reconstruction error where blocks are
/// off-centre, there would be nothing to build.
#[test]
fn an_offset_code_beats_the_symmetric_one_where_blocks_are_off_centre() {
    use super::super::super::integer::{
        quantise_activation_asymmetric, quantise_activation_blocked,
    };
    const N: usize = 1024;
    const BLK: usize = 16;

    let err = |x: &[f32], asym: bool| -> f64 {
        let se: f64 = if asym {
            let (c, s, m) = quantise_activation_asymmetric(x, BLK);
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    let r = c[i] as f64 * s[i / BLK] as f64 + m[i / BLK] as f64;
                    (r - *v as f64).powi(2)
                })
                .sum()
        } else {
            let (c, s) = quantise_activation_blocked(x, BLK);
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    let r = c[i] as f64 * s[i / BLK] as f64;
                    (r - *v as f64).powi(2)
                })
                .sum()
        };
        (se / x.len() as f64).sqrt()
    };

    // One-sided: every block strictly positive, so symmetric wastes half
    // its range on a sign that never occurs.
    let one_sided: Vec<f32> = lcg_values(N, 63).iter().map(|v| v.abs() + 1.0).collect();
    let (a, sym) = (err(&one_sided, true), err(&one_sided, false));
    assert!(
        a < sym * 0.75,
        "on a one-sided activation the offset code should be clearly better: \
         asym {a:.3e} vs sym {sym:.3e}"
    );

    // Balanced and heavy-tailed — the real regime. It must not LOSE.
    let real = outlier_activation(N, 64);
    let (a2, sym2) = (err(&real, true), err(&real, false));
    assert!(
        a2 <= sym2 * 1.01,
        "on a balanced activation the offset code must not lose: \
         asym {a2:.3e} vs sym {sym2:.3e}"
    );
}

/// **CPU5-K1 is BIT-IDENTICAL to the kernel it replaces.**
///
/// The whole licence for skipping another 69-prompt bank run. K1 reads
/// `SUM(q)` from a precomputed index instead of recomputing it with a
/// second `SDOT`; an i32 sum of i16 sub-sums taken in order is the same
/// integer the reduction produced, and no float operation changes. So
/// the arithmetic that passed the quality gates is the arithmetic that
/// runs — asserted, not argued.
#[test]
fn the_indexed_asymmetric_row_is_bit_identical_to_the_recomputing_one() {
    use super::super::super::integer::{q8_row_asym_indexed, quantise_activation_asymmetric};
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};

    const OUT: usize = 4;
    for in_dim in [64usize, 192, 1024, 5120] {
        let w = lcg_values(OUT * in_dim, 71);
        let x = outlier_activation(in_dim, 72);
        let LoadedWeight::Q8 {
            codes,
            scales,
            sums,
        } = quantise_q8_indexed_for_test(&w, in_dim)
        else {
            panic!("the indexed quantiser must produce the q8 variant");
        };
        assert!(!sums.is_empty(), "the index must actually be built");
        assert_eq!(
            sums.len(),
            OUT * in_dim.div_ceil(SUM_BLOCK),
            "one sum per {SUM_BLOCK} codes per row"
        );

        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_sum = in_dim.div_ceil(SUM_BLOCK);
        for ablock in [16usize, 32, 64] {
            let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
            let per_weight = Q8_BLOCK / ablock;
            for o in 0..OUT {
                let ws = &scales[o * per_row..(o + 1) * per_row];
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
                let recomputed = super::super::super::integer::q8_row_asym_exact(
                    row, &fs, &fm, &qx, in_dim, ablock,
                );
                let indexed = q8_row_asym_indexed(
                    row,
                    &fs,
                    &fm,
                    &qx,
                    &sums[o * per_sum..(o + 1) * per_sum],
                    in_dim,
                    ablock,
                );
                assert_eq!(
                    indexed.to_bits(),
                    recomputed.to_bits(),
                    "in_dim {in_dim}, ablock {ablock}, row {o}: indexed {indexed} vs \
                     recomputed {recomputed}"
                );
            }
        }
    }
}

/// **A ragged final activation block reads only its own sums.**
///
/// When `in_dim % ablock != 0` the row's last block is short, and so is
/// its run of `SUM_BLOCK` sums. The index used to be strided by
/// `ablock / SUM_BLOCK` regardless, so the last block read past the row:
/// the next row's sums, or — on the terminal row through the unchecked
/// SDOT path — past the slice.
///
/// Each row's sums are handed over as an exact-length slice of a buffer
/// whose tail is a CANARY, so an over-read lands on a value that moves
/// the answer instead of on plausible neighbouring sums. Both the
/// portable and the dispatched (SDOT on aarch64) rows are judged, bit
/// for bit, against the recomputing row.
#[test]
fn a_ragged_final_block_reads_only_its_own_sums() {
    use super::super::super::integer::{
        q8_row_asym_exact, q8_row_asym_indexed, q8_row_asym_indexed_portable,
        quantise_activation_asymmetric,
    };
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};

    /// Far outside any real `SUM_BLOCK` sum (`|sum| <= 16 * 127`).
    const CANARY: i16 = i16::MAX;
    const CANARY_LEN: usize = 64 / SUM_BLOCK;
    const OUT: usize = 3;
    // 80: one full 64-block plus a 16-wide tail. 88: a tail that is not
    // itself a multiple of SUM_BLOCK. 40: a single short block.
    for in_dim in [80usize, 88, 40] {
        let w = lcg_values(OUT * in_dim, 81);
        let x = outlier_activation(in_dim, 82);
        let LoadedWeight::Q8 {
            codes,
            scales,
            sums,
        } = quantise_q8_indexed_for_test(&w, in_dim)
        else {
            panic!("the indexed quantiser must produce the q8 variant");
        };
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_sum = in_dim.div_ceil(SUM_BLOCK);
        for ablock in [32usize, 64] {
            assert_ne!(in_dim % ablock, 0, "the shape must leave a ragged block");
            let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
            let per_weight = Q8_BLOCK / ablock;
            for o in 0..OUT {
                let ws = &scales[o * per_row..(o + 1) * per_row];
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
                let mut guarded = sums[o * per_sum..(o + 1) * per_sum].to_vec();
                guarded.extend([CANARY; CANARY_LEN]);
                let own = &guarded[..per_sum];

                let want = q8_row_asym_exact(row, &fs, &fm, &qx, in_dim, ablock);
                let portable =
                    q8_row_asym_indexed_portable(row, &fs, &fm, &qx, own, in_dim, ablock);
                let dispatched = q8_row_asym_indexed(row, &fs, &fm, &qx, own, in_dim, ablock);
                for (arm, got) in [("portable", portable), ("dispatched", dispatched)] {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{arm} indexed row {o}, in_dim {in_dim}, ablock {ablock}: {got} vs \
                         recomputed {want}"
                    );
                }
            }
        }
    }
}

/// The indexed row refuses a sums slice shorter than its geometry
/// BEFORE the unchecked kernel dereferences it.
#[test]
#[should_panic(expected = "indexed Q8 row needs")]
fn the_indexed_row_refuses_a_short_sums_slice() {
    use super::super::super::integer::q8_row_asym_indexed;
    use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;
    const IN: usize = 80;
    let codes = vec![1i8; IN];
    let qx = vec![1i8; IN];
    let fold = vec![1.0f32; IN.div_ceil(SUM_BLOCK)];
    // One sum short of the row's geometry.
    let sums = vec![0i16; IN.div_ceil(SUM_BLOCK) - 1];
    q8_row_asym_indexed(&codes, &fold, &fold, &qx, &sums, IN, SUM_BLOCK);
}

/// The index is EXACT and fits i16, which is what makes it one bit per
/// weight rather than two.
#[test]
fn the_code_sum_index_is_exact_and_fits_i16() {
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};
    const IN: usize = 320;
    let w = lcg_values(IN, 73);
    let LoadedWeight::Q8 { codes, sums, .. } = quantise_q8_indexed_for_test(&w, IN) else {
        panic!("expected q8");
    };
    for (b, s) in sums.iter().enumerate() {
        let lo = b * SUM_BLOCK;
        let hi = (lo + SUM_BLOCK).min(IN);
        let want: i32 = codes[lo..hi].iter().map(|c| *c as i32).sum();
        assert_eq!(*s as i32, want, "sum block {b}");
        assert!(want.abs() <= SUM_BLOCK as i32 * 127);
    }
}

/// A symmetric arm builds NO index — it has no use for one, and paying
/// ~1 bit/weight of residency and traffic for it would slow the arm that
/// is currently fastest.
#[test]
fn the_symmetric_path_carries_no_index() {
    let LoadedWeight::Q8 { sums, .. } = quantise_q8_for_test(&lcg_values(256, 74), 64) else {
        panic!("expected q8");
    };
    assert!(
        sums.is_empty(),
        "the symmetric quantiser built an index nothing reads"
    );
}

/// **The K2-vs-K3 control: a bug detector, NOT an acceptance substitute.**
///
/// K3 reassociates the sum of already-computed block contributions, so a
/// difference is expected — at the ROUNDING level. This pins the size of
/// it. A reading near 1e-7 says the reassociation is behaving as a
/// reassociation; a reading near 1e-3 says the kernel is computing
/// something else, and catches that before a 50-minute bank run rather
/// than after.
///
/// The frozen quality gates are re-established on the full bank
/// regardless of what this says. A tolerance passed here licenses
/// nothing about the model.
// K3/K5 are the register-folded SDOT rows; off aarch64 the portable
// definitions run instead and `q8_row_k3_register` is a `todo!()`
// stub ("K5 has no portable arm; the gate runs on aarch64"). The
// arithmetic these pin does not exist on this target.
#[cfg(target_arch = "aarch64")]
#[test]
fn k3_moves_the_row_only_at_the_rounding_level() {
    use super::super::super::integer::{
        q8_row_asym_exact, q8_row_asym_k3, quantise_activation_asymmetric,
    };
    const OUT: usize = 8;
    const IN: usize = 5120;
    let w = lcg_values(OUT * IN, 81);
    let x = outlier_activation(IN, 82);
    let (codes, wscales) = q8_parts(&w, IN);
    let ablock = 16usize;
    let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
    let per_row = IN.div_ceil(Q8_BLOCK);
    let per_weight = Q8_BLOCK / ablock;

    let mut worst = 0.0f64;
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
        let row = &codes[o * IN..(o + 1) * IN];
        let k2 = q8_row_asym_exact(row, &fs, &fm, &qx, IN, ablock);
        let k3 = q8_row_asym_k3(row, &fs, &fm, &qx, IN);
        let rel = ((k3 - k2) as f64 / (k2 as f64).abs().max(1e-9)).abs();
        worst = worst.max(rel);
    }
    assert!(
        worst < 1e-4,
        "K3 moves the row by {worst:.3e} relative — that is not reassociation, it is a \
         different computation"
    );
    // And it must not be a no-op dressed as an optimisation: over 5120
    // f32 accumulations SOME difference is expected, and exact agreement
    // would mean the K3 path never ran.
    assert!(
        worst > 0.0,
        "K3 agreed with K2 exactly, so the K3 kernel did not run"
    );
}

/// **K4 is BIT-IDENTICAL to K3**, which is what lets one Bank-1 run
/// cover both.
///
/// K4 replaces four correction `SDOT`s with one 64-bit load of the
/// precomputed sums. Those hold exactly the integers the `SDOT`s
/// produce, so `vcvtq_f32_s32` sees the same lanes and no float
/// operation changes. Unlike K3-vs-K2 this is an equality, not a
/// tolerance — and it is asserted rather than argued, because K1 already
/// demonstrated that an index can be plumbed wrongly while still
/// returning finite, plausible numbers.
#[test]
fn k4_is_bit_identical_to_k3() {
    use super::super::super::integer::{
        q8_row_asym_k3, q8_row_asym_k4, quantise_activation_asymmetric,
    };
    use crate::format::vindex3::opplan::exec::quantise::{quantise_q8_indexed_for_test, SUM_BLOCK};

    const OUT: usize = 6;
    // 5120 is the real width; 80 exercises a group-of-four tail; 88 adds
    // a remainder shorter than one 16-wide block.
    for in_dim in [80usize, 88, 256, 5120] {
        let w = lcg_values(OUT * in_dim, 91);
        let x = outlier_activation(in_dim, 92);
        let LoadedWeight::Q8 {
            codes,
            scales,
            sums,
        } = quantise_q8_indexed_for_test(&w, in_dim)
        else {
            panic!("expected q8");
        };
        let ablock = SUM_BLOCK;
        let (qx, ascales, amids) = quantise_activation_asymmetric(&x, ablock);
        let per_row = in_dim.div_ceil(Q8_BLOCK);
        let per_sum = in_dim.div_ceil(SUM_BLOCK);
        let per_weight = Q8_BLOCK / ablock;

        for o in 0..OUT {
            let ws = &scales[o * per_row..(o + 1) * per_row];
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
            let k4 = q8_row_asym_k4(
                row,
                &fs,
                &fm,
                &qx,
                &sums[o * per_sum..(o + 1) * per_sum],
                in_dim,
            );
            assert_eq!(
                k4.to_bits(),
                k3.to_bits(),
                "in_dim {in_dim}, row {o}: K4 {k4} vs K3 {k3}"
            );
        }
    }
}
