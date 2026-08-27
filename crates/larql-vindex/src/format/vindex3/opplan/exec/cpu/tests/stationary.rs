//! CPU-7C parity: the stationary sweep is a SCHEDULE, not arithmetic.
//!
//! Everything here compares against `q8_row_k3_register` — the frozen K5
//! row itself — rather than against another call into the new module. Two
//! new paths agreeing proves they agree; agreeing with the row the
//! candidate is meant to reschedule is what proves the reschedule.
//!
//! The geometry is passed explicitly, never through the environment: the
//! selectors are `OnceLock`-cached, so a test that set one would fix it
//! for every test that ran after it in the same process.

use super::super::integer::{q8_row_k3_register, quantise_activation_blocked};
use super::super::ledger::Site;
use super::super::projector::WeightRows;
use super::super::stationary::{
    class_enabled_for, enable_all_classes, enabled, geometry, project_rows_many_with, set_enabled,
    set_enabled_for, supports, supports_with, Geometry,
};

/// Five weight blocks of 64. Not a round power of two on purpose: an
/// awkward shape has already exposed three bugs a tidy one hid.
const IN_DIM: usize = 320;

/// Rows, deliberately coprime with every worker count and block size.
const ROWS: usize = 7;

const WEIGHT_BLOCK: usize = 64;
const ACT_BLOCK: usize = 16;

/// Asymmetric coding at block 16 — the CPU-5 candidate's own geometry,
/// which is the one the stationary sweep exists to reschedule.
const GEO: Geometry = Geometry {
    ablock: ACT_BLOCK,
    asym: true,
};

fn weights() -> (Vec<i8>, Vec<f32>) {
    let codes = (0..ROWS * IN_DIM)
        .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
        .collect();
    let scales = (0..ROWS * (IN_DIM / WEIGHT_BLOCK))
        .map(|i| 0.005 + (i % 13) as f32 * 0.001)
        .collect();
    (codes, scales)
}

/// Heavy-tailed, because a residual stream at depth is: one large element
/// per block is what makes the asymmetric midpoint carry real signal.
fn activation(seed: usize) -> Vec<f32> {
    (0..IN_DIM)
        .map(|i| {
            let base = ((i * 7 + seed * 31) % 97) as f32 / 97.0 - 0.5;
            if i % 64 == seed % 64 {
                base * 25.0
            } else {
                base
            }
        })
        .collect()
}

fn rows_of<'a>(codes: &'a [i8], scales: &'a [f32]) -> WeightRows<'a> {
    WeightRows::Q8 {
        codes,
        scales,
        sums: &[],
        block: WEIGHT_BLOCK,
    }
}

/// One position at a time, through the frozen row.
fn reference(codes: &[i8], scales: &[f32], x: &[f32]) -> Vec<f32> {
    let (qx, ascales, amids) = super::super::integer::quantise_activation_asymmetric(x, ACT_BLOCK);
    let per_row = IN_DIM / WEIGHT_BLOCK;
    (0..ROWS)
        .map(|r| {
            q8_row_k3_register(
                &codes[r * IN_DIM..(r + 1) * IN_DIM],
                &scales[r * per_row..(r + 1) * per_row],
                &ascales,
                Some(&amids),
                &qx,
                IN_DIM,
            )
        })
        .collect()
}

fn sweep(codes: &[i8], scales: &[f32], xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let n = xs.len();
    let mut flat = vec![0.0f32; ROWS * n];
    project_rows_many_with(rows_of(codes, scales), xs, &mut flat, n, GEO);
    (0..n)
        .map(|p| (0..ROWS).map(|r| flat[r * n + p]).collect())
        .collect()
}

/// Without this every other test in the file could pass on the looping
/// default, which is correct arithmetic and the wrong experiment.
#[test]
fn the_geometry_admits_a_stationary_sweep() {
    let (codes, scales) = weights();
    for n in [2, 4] {
        assert!(
            supports_with(rows_of(&codes, &scales), IN_DIM, n, GEO),
            "n={n} fell back to the loop, so the parity below would be vacuous"
        );
    }
    // And a geometry it must decline: one activation scale per 32
    // elements is not four-per-weight-block, so the register path's
    // constant-scale-within-a-group assumption does not hold.
    assert!(!supports_with(
        rows_of(&codes, &scales),
        IN_DIM,
        2,
        Geometry {
            ablock: 32,
            asym: true
        }
    ));
}

#[test]
fn every_position_is_bit_identical_to_the_frozen_row() {
    let (codes, scales) = weights();
    let xs: Vec<Vec<f32>> = (0..4).map(activation).collect();
    for n in [2usize, 4] {
        let borrowed: Vec<&[f32]> = xs.iter().take(n).map(Vec::as_slice).collect();
        let got = sweep(&codes, &scales, &borrowed);
        for (p, x) in xs.iter().enumerate().take(n) {
            let want = reference(&codes, &scales, x);
            for r in 0..ROWS {
                assert_eq!(
                    got[p][r].to_bits(),
                    want[r].to_bits(),
                    "n={n} position={p} row={r}: the sweep changed the arithmetic, \
                     not merely the schedule ({} vs {})",
                    got[p][r],
                    want[r]
                );
            }
        }
    }
}

/// The planted violation. Without it, the test above only shows that a
/// comparison CAN pass, not that it can fail.
#[test]
fn perturbing_one_position_moves_that_position_and_no_other() {
    let (codes, scales) = weights();
    let xs: Vec<Vec<f32>> = (0..4).map(activation).collect();
    let clean: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
    let before = sweep(&codes, &scales, &clean);

    let mut tampered = xs[1].clone();
    tampered[IN_DIM - 1] += 3.5;
    let mut dirty_in = clean.clone();
    dirty_in[1] = &tampered;
    let after = sweep(&codes, &scales, &dirty_in);

    assert!(
        (0..ROWS).any(|r| before[1][r].to_bits() != after[1][r].to_bits()),
        "the perturbed position did not move, so this comparison cannot fail \
         and proves nothing about the ones that passed"
    );
    for p in [0usize, 2, 3] {
        for r in 0..ROWS {
            assert_eq!(
                before[p][r].to_bits(),
                after[p][r].to_bits(),
                "position {p} moved when only position 1 was perturbed — the \
                 positions are not independent"
            );
        }
    }
}

/// The activation quantisation must stay PER POSITION. A joint scale over
/// the `n` vectors would be a different representation wearing a
/// schedule's name, so it is checked directly: appending a position with
/// a wildly different dynamic range must not disturb the others.
#[test]
fn a_loud_neighbour_does_not_change_its_neighbours_quantisation() {
    let (codes, scales) = weights();
    let quiet: Vec<Vec<f32>> = (0..2).map(activation).collect();
    let pair: Vec<&[f32]> = quiet.iter().map(Vec::as_slice).collect();
    let alone = sweep(&codes, &scales, &pair);

    let loud: Vec<f32> = activation(9).iter().map(|v| v * 400.0).collect();
    let with_loud: Vec<&[f32]> = vec![&quiet[0], &loud, &quiet[1], &quiet[1]];
    let together = sweep(&codes, &scales, &with_loud);

    for r in 0..ROWS {
        assert_eq!(
            alone[0][r].to_bits(),
            together[0][r].to_bits(),
            "row {r}: a neighbour's dynamic range reached this position's \
             scale, so the sweep shares a quantisation it must not"
        );
    }
}

/// The symmetric arm, which every test above leaves untouched.
///
/// Checked against the frozen K5 row with `amids = None` — the same row,
/// told there is no midpoint term — so this shows the sweep consumes the
/// symmetric representation faithfully rather than merely consistently.
#[test]
fn symmetric_coding_is_bit_identical_to_the_frozen_row() {
    const SYM: Geometry = Geometry {
        ablock: ACT_BLOCK,
        asym: false,
    };
    let (codes, scales) = weights();
    let xs: Vec<Vec<f32>> = (0..2).map(activation).collect();
    let borrowed: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
    let mut flat = vec![0.0f32; ROWS * 2];
    project_rows_many_with(rows_of(&codes, &scales), &borrowed, &mut flat, 2, SYM);

    let per_row = IN_DIM / WEIGHT_BLOCK;
    for (p, x) in xs.iter().enumerate() {
        let (qx, ascales) = quantise_activation_blocked(x, ACT_BLOCK);
        for r in 0..ROWS {
            let want = q8_row_k3_register(
                &codes[r * IN_DIM..(r + 1) * IN_DIM],
                &scales[r * per_row..(r + 1) * per_row],
                &ascales,
                None,
                &qx,
                IN_DIM,
            );
            assert_eq!(
                flat[r * 2 + p].to_bits(),
                want.to_bits(),
                "symmetric position={p} row={r}"
            );
        }
    }
}

/// A representation the sweep does not serve must be DECLINED, not
/// mis-consumed: `Q8` is the only arm with a stationary row.
#[test]
fn a_non_q8_representation_is_declined() {
    let widened = vec![0.0f32; ROWS * IN_DIM];
    assert!(!supports_with(WeightRows::F32(&widened), IN_DIM, 2, GEO));
}

/// An `N` outside the unrolled counts falls to the caller's loop rather
/// than silently running some other geometry.
#[test]
fn an_unlisted_position_count_is_declined() {
    let (codes, scales) = weights();
    for n in [1usize, 3, 8] {
        assert!(
            !supports_with(rows_of(&codes, &scales), IN_DIM, n, GEO),
            "N={n} is not an unrolled count and must not claim support"
        );
    }
}

/// The process-wide arm switches. Serialised: they are global, and a
/// parallel test reading `supports` while this one has flipped them would
/// see whichever write won.
#[test]
#[serial_test::serial]
fn the_arm_switches_round_trip() {
    let restore_global = enabled();
    let restore: Vec<bool> = Site::ALL.iter().map(|s| class_enabled_for(*s)).collect();

    set_enabled(false);
    assert!(!enabled());
    set_enabled(true);
    assert!(enabled());

    for site in Site::ALL {
        set_enabled_for(site, false);
        assert!(!class_enabled_for(site));
        set_enabled_for(site, true);
        assert!(class_enabled_for(site));
    }

    // One class off and the rest on — the shape CPU-7C2's arm E runs.
    set_enabled_for(Site::Ffn, false);
    assert!(!class_enabled_for(Site::Ffn));
    assert!(class_enabled_for(Site::Recurrent));

    enable_all_classes();
    for site in Site::ALL {
        assert!(class_enabled_for(site));
    }

    set_enabled(restore_global);
    for (site, on) in Site::ALL.iter().zip(restore) {
        set_enabled_for(*site, on);
    }
}

/// Under the suite's own environment the arithmetic arm is the default
/// float-activation one, which carries no block geometry — so `geometry`
/// answers `None` and `supports` declines whatever the weights look like.
///
/// This is the ENVIRONMENT-reading pair that `supports_with` was split
/// away from, exercised here rather than by setting a process-wide
/// variable the rest of the suite shares.
#[test]
#[serial_test::serial]
fn the_default_arithmetic_arm_has_no_stationary_geometry() {
    let (codes, scales) = weights();
    assert!(geometry().is_none());
    assert!(!supports(rows_of(&codes, &scales), IN_DIM, 2));
}
