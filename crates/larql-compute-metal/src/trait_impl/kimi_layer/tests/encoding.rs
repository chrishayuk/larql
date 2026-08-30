//! The Q8_0 grouped kernel's decode, against the codec's own reference.
//!
//! The claim under test is narrow: the kernel reads canonical ggml
//! Q8_0 blocks (f16 scale + 32 int8) and computes the same matvec the
//! CPU computes from `dequantize_q8_0` of the same bytes. Quantisation
//! error is deliberately OUT of scope — both arms read the same
//! quantised values, so a disagreement is a decode or addressing fault,
//! never "Q8_0 is lossy".
//!
//! Shapes are chosen to exercise the kernel's actual strides: K = 1088
//! gives 34 blocks per row, one more than the 32 simd lanes, so the
//! block loop's second iteration runs on lanes 0-1; N = 8 spans two row
//! tiles of `ROWS_PER_TG = 4`.

use larql_compute::cpu::ops::q4_common::{dequantize_q8_0, quantize_q8_0};

use crate::trait_impl::bf16_grouped::GroupedShape;
use crate::trait_impl::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use crate::trait_impl::kimi_layer::ExpertEncoding;
use crate::MetalBackend;

const SLOTS: usize = 3;
const N: usize = 8;
const K: usize = 1088;
/// Decode parity, not quantisation quality: both arms read identical
/// quantised values, so only f32 summation order separates them.
const TOLERANCE: f32 = 1e-3;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.4)
        .collect()
}

/// `SLOTS` distinct `[N, K]` expert matrices quantised to Q8_0, banked
/// back to back, plus each slot's byte offset and the dequantised f32
/// form of each expert for the CPU arm.
fn q8_bank() -> (Vec<u8>, Vec<ExpertOffset>, Vec<Vec<f32>>) {
    let per_expert = ExpertEncoding::Q80
        .matrix_bytes(N, K)
        .expect("K is a whole number of blocks");
    let mut bank = Vec::with_capacity(SLOTS * per_expert);
    let mut dequantised = Vec::with_capacity(SLOTS);
    let mut offsets = Vec::with_capacity(SLOTS);
    for slot in 0..SLOTS {
        let q = quantize_q8_0(&synth(N * K, 1.0 + slot as f32));
        assert_eq!(q.len(), per_expert, "codec and layout must agree on bytes");
        offsets.push(ExpertOffset((slot * per_expert) as u32));
        dequantised.push(dequantize_q8_0(&q, N * K).expect("wellformed blocks"));
        bank.extend_from_slice(&q);
    }
    (bank, offsets, dequantised)
}

fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|r| {
            w[r * K..(r + 1) * K]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The kernel decodes exactly what the codec's own dequantiser reads —
/// with a control proving the comparison could fail: the UNQUANTISED
/// weights give a visibly different answer, so agreement with the
/// dequantised arm is not agreement-with-anything.
#[test]
fn q8_0_grouped_matches_the_cpu_dequantised_reference() {
    let m = backend();
    let (bank, offsets, deq) = q8_bank();
    let x = synth(K, 0.31);

    let got = m
        .grouped_experts_encoded(
            ExpertEncoding::Q80,
            &bank,
            &offsets,
            &x,
            GroupedShape {
                n: N,
                k: K,
                layout: InputLayout::Shared,
            },
        )
        .expect("q8_0 grouped dispatch");
    assert_eq!(got.len(), SLOTS * N);

    let mut quantisation_moved_something = false;
    for slot in 0..SLOTS {
        let want = matvec(&deq[slot], &x);
        let err = max_abs(&got[slot * N..(slot + 1) * N], &want);
        assert!(
            err < TOLERANCE,
            "slot {slot}: decode parity broke — max |Δ| {err:e} against the \
             dequantised reference"
        );
        let unquantised = matvec(&synth(N * K, 1.0 + slot as f32), &x);
        if max_abs(&want, &unquantised) > TOLERANCE {
            quantisation_moved_something = true;
        }
    }
    assert!(
        quantisation_moved_something,
        "quantisation changed nothing measurable, so decode parity was proven \
         against a reference indistinguishable from the raw weights"
    );
}

/// Identity travels in the offset table, not in a slot's position.
#[test]
fn the_offset_table_decides_which_expert_a_slot_computes() {
    let m = backend();
    let (bank, offsets, _) = q8_bank();
    let x = synth(K, 0.77);
    let shape = GroupedShape {
        n: N,
        k: K,
        layout: InputLayout::Shared,
    };

    let forward = m
        .grouped_experts_encoded(ExpertEncoding::Q80, &bank, &offsets, &x, shape)
        .expect("forward order");
    let reversed_table: Vec<ExpertOffset> = offsets.iter().rev().copied().collect();
    let reversed = m
        .grouped_experts_encoded(ExpertEncoding::Q80, &bank, &reversed_table, &x, shape)
        .expect("reversed order");

    for slot in 0..SLOTS {
        assert_eq!(
            forward[slot * N..(slot + 1) * N],
            reversed[(SLOTS - 1 - slot) * N..(SLOTS - slot) * N],
            "slot {slot}: reversing the table must reverse the output blocks exactly"
        );
    }
}

/// `XSTRIDE = K`: each slot consumes its OWN input vector — the down
/// projection's regime, where getting the stride wrong computes a real
/// number from the wrong expert's activation.
#[test]
fn per_slot_inputs_reach_their_own_slots() {
    let m = backend();
    let (bank, offsets, deq) = q8_bank();
    let xs: Vec<f32> = (0..SLOTS).flat_map(|s| synth(K, 5.0 + s as f32)).collect();

    let got = m
        .grouped_experts_encoded(
            ExpertEncoding::Q80,
            &bank,
            &offsets,
            &xs,
            GroupedShape {
                n: N,
                k: K,
                layout: InputLayout::PerSlot,
            },
        )
        .expect("per-slot dispatch");

    for slot in 0..SLOTS {
        let want = matvec(&deq[slot], &xs[slot * K..(slot + 1) * K]);
        let err = max_abs(&got[slot * N..(slot + 1) * N], &want);
        assert!(
            err < TOLERANCE,
            "slot {slot}: per-slot input regime broke — max |Δ| {err:e}"
        );
    }
}

/// Bounds come from Q8_0's OWN stride: a bank sized for Q8_0 is smaller
/// than the BF16 arithmetic it stands in for and must still bind; a
/// truncated bank and an unencodable K are refused by name.
#[test]
fn q8_0_bounds_are_checked_at_the_q8_0_stride() {
    let m = backend();
    let (bank, offsets, _) = q8_bank();
    let x = synth(K, 0.11);
    let shape = GroupedShape {
        n: N,
        k: K,
        layout: InputLayout::Shared,
    };

    let short = &bank[..bank.len() - 1];
    match m.grouped_experts_encoded(ExpertEncoding::Q80, short, &offsets, &x, shape) {
        Err(GroupedError::OffsetOutOfRange { slot, .. }) => {
            assert_eq!(slot, SLOTS - 1, "the last slot is the one that ran out")
        }
        other => panic!("a truncated bank must refuse by range, got {other:?}"),
    }

    let bad_k = GroupedShape {
        n: N,
        k: K + 1,
        layout: InputLayout::Shared,
    };
    match m.grouped_experts_encoded(ExpertEncoding::Q80, &bank, &offsets, &x, bad_k) {
        Err(GroupedError::KNotSuperblockAligned { k }) => assert_eq!(k, K + 1),
        other => panic!("an unencodable K must refuse by alignment, got {other:?}"),
    }
}
