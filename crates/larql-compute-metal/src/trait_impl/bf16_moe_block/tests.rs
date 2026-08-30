//! Unit gates for the one-command-buffer MoE block.

use super::*;
use larql_compute::backend::MatMul;

const SLOTS: usize = 3;
const HIDDEN: usize = 64;
const INTER: usize = 32;

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// `SLOTS` distinct `[n, k]` matrices back to back, plus each one's
/// byte offset.
fn bank(n: usize, k: usize, seed: f32) -> (Vec<u8>, Vec<ExpertOffset>) {
    let per = n * k;
    let mut bytes = Vec::with_capacity(SLOTS * per * 2);
    let mut offsets = Vec::with_capacity(SLOTS);
    for slot in 0..SLOTS {
        offsets.push(ExpertOffset((slot * per * 2) as u32));
        for i in 0..per {
            let v = ((i as f32) * 0.011 + seed + slot as f32 * 2.3).sin() * 0.4;
            bytes.extend_from_slice(&narrow(v).to_le_bytes());
        }
    }
    (bytes, offsets)
}

struct Fixture {
    gate: (Vec<u8>, Vec<ExpertOffset>),
    up: (Vec<u8>, Vec<ExpertOffset>),
    down: (Vec<u8>, Vec<ExpertOffset>),
    x: Vec<f32>,
}

fn fixture() -> Fixture {
    fixture_seeded(0.0)
}

/// A fixture whose weights and input differ from every other seed, so a
/// test that batches several blocks cannot pass by accident.
fn fixture_seeded(seed: f32) -> Fixture {
    Fixture {
        gate: bank(INTER, HIDDEN, 0.1 + seed),
        up: bank(INTER, HIDDEN, 1.7 + seed),
        down: bank(HIDDEN, INTER, 3.2 + seed),
        x: (0..HIDDEN)
            .map(|i| ((i as f32) * 0.037 + seed).cos() * 0.6)
            .collect(),
    }
}

impl Fixture {
    fn banks(&self) -> MoeFfnBanks<'_> {
        MoeFfnBanks {
            gate: ExpertBankRef {
                weights: &self.gate.0,
                offsets: &self.gate.1,
            },
            up: ExpertBankRef {
                weights: &self.up.0,
                offsets: &self.up.1,
            },
            down: ExpertBankRef {
                weights: &self.down.0,
                offsets: &self.down.1,
            },
            hidden: HIDDEN,
            inter: INTER,
        }
    }
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

/// The block against the same three grouped dispatches run
/// separately with the activation on the host.
///
/// The GEMVs are the same kernel with the same arguments, so the
/// only thing that can differ is the activation: `exp` from Metal's
/// library against `exp` from the host's. Whether those agree to the
/// last bit is a fact about two libms, not something this crate gets
/// to assume — so the test states the tolerance and the whole-block
/// gate on real weights reports the measured figure.
#[test]
fn the_block_matches_the_same_stages_run_separately() {
    let m = backend();
    let f = fixture();
    let b = f.banks();

    let (block, _gpu) = m.bf16_moe_ffn_block(b, &f.x).expect("block");

    let shared = GroupedShape {
        n: INTER,
        k: HIDDEN,
        layout: InputLayout::Shared,
    };
    let gate = m
        .bf16_grouped_experts(&f.gate.0, &f.gate.1, &f.x, INTER, HIDDEN, shared.layout)
        .expect("gate");
    let up = m
        .bf16_grouped_experts(&f.up.0, &f.up.1, &f.x, INTER, HIDDEN, shared.layout)
        .expect("up");
    let h: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let want = m
        .bf16_grouped_experts(
            &f.down.0,
            &f.down.1,
            &h,
            HIDDEN,
            INTER,
            InputLayout::PerSlot,
        )
        .expect("down");

    assert_eq!(block.len(), SLOTS * HIDDEN);
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let rel = block
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale;
    assert!(rel < 1e-6, "block vs staged stages, rel {rel:e}");
}

/// **The claim rung 4 rests on:** all three lowerings compute the
/// same numbers, bit for bit.
///
/// Each accumulator walks its row identically in every arm — same
/// lane stride, same four-way unroll, same `simd_sum` — and the
/// activation is the same expression whether evaluated in register
/// or through a buffer. If that were false, the fusion measurement
/// would be comparing kernels rather than lowerings, and any
/// bandwidth it reported would be uninterpretable.
#[test]
fn every_lowering_computes_the_same_values() {
    let m = backend();
    let f = fixture();

    let mut reference: Option<Vec<f32>> = None;
    for lowering in [
        BlockLowering::Separate,
        BlockLowering::FusedGateUp(FusedTiling::Rows8),
        BlockLowering::FusedGateUp(FusedTiling::Rows4),
        BlockLowering::FusedGateUpAct(FusedTiling::Rows8),
        BlockLowering::FusedGateUpAct(FusedTiling::Rows4),
    ] {
        let (got, _gpu) = m
            .bf16_moe_ffn_block_lowered(f.banks(), &f.x, lowering)
            .expect("lowered block");
        match &reference {
            None => reference = Some(got),
            Some(want) => assert_eq!(&got, want, "{lowering:?} disagreed with Separate"),
        }
    }
}

/// The fused kernels must read the UP bank, not the gate bank twice
/// — two bank pointers and two offset tables is exactly the shape
/// where a copy-paste binds argument 0 where argument 2 belongs, and
/// gate and up are similar enough that the result would still look
/// like a plausible FFN.
#[test]
fn the_fused_kernels_read_both_banks() {
    let m = backend();
    let f = fixture();
    let other = bank(INTER, HIDDEN, 12.5);

    for lowering in [
        BlockLowering::FusedGateUp(FusedTiling::Rows8),
        BlockLowering::FusedGateUpAct(FusedTiling::Rows4),
    ] {
        let (baseline, _) = m
            .bf16_moe_ffn_block_lowered(f.banks(), &f.x, lowering)
            .expect("baseline");
        let mut swapped = f.banks();
        swapped.up = ExpertBankRef {
            weights: &other.0,
            offsets: &other.1,
        };
        let (got, _) = m
            .bf16_moe_ffn_block_lowered(swapped, &f.x, lowering)
            .expect("swapped up");
        assert_ne!(got, baseline, "{lowering:?} is not reading the up bank");
    }
}

/// Control: the block must actually read all three banks. Swapping
/// the down bank for the gate bank has to change the answer —
/// otherwise a block that silently reused one buffer would pass the
/// parity test above whenever the banks happened to be similar.
#[test]
fn every_bank_reaches_the_output() {
    let m = backend();
    let f = fixture();
    let (baseline, _) = m.bf16_moe_ffn_block(f.banks(), &f.x).expect("baseline");

    for swapped in [0usize, 1, 2] {
        let mut b = f.banks();
        let other = bank(
            if swapped == 2 { HIDDEN } else { INTER },
            if swapped == 2 { INTER } else { HIDDEN },
            9.5,
        );
        let r = ExpertBankRef {
            weights: &other.0,
            offsets: &other.1,
        };
        match swapped {
            0 => b.gate = r,
            1 => b.up = r,
            _ => b.down = r,
        }
        let (got, _) = m.bf16_moe_ffn_block(b, &f.x).expect("swapped");
        assert_ne!(
            got, baseline,
            "replacing bank {swapped} changed nothing — it is not being read"
        );
    }
}

/// A slot count that disagrees between projections is refused, not
/// silently truncated to the shortest table.
#[test]
fn disagreeing_slot_counts_are_refused() {
    let m = backend();
    let f = fixture();
    let mut b = f.banks();
    let short = &f.up.1[..SLOTS - 1];
    b.up = ExpertBankRef {
        weights: &f.up.0,
        offsets: short,
    };
    assert_eq!(
        m.bf16_moe_ffn_block(b, &f.x),
        Err(GroupedError::SlotCountMismatch {
            expected: SLOTS,
            found: SLOTS - 1,
        })
    );

    let mut empty = f.banks();
    empty.gate = ExpertBankRef {
        weights: &f.gate.0,
        offsets: &[],
    };
    assert_eq!(
        m.bf16_moe_ffn_block(empty, &f.x),
        Err(GroupedError::NoExpertsSelected)
    );
}

/// A bad bank is refused BEFORE any encoding starts. Metal aborts
/// the process if a compute encoder is dropped without
/// `end_encoding`, so a refusal discovered halfway through the block
/// would not be an error the caller could handle.
#[test]
fn a_short_bank_is_refused_before_the_encoder_opens() {
    let m = backend();
    let f = fixture();
    let mut b = f.banks();
    let truncated = &f.down.0[..f.down.0.len() / 2];
    b.down = ExpertBankRef {
        weights: truncated,
        offsets: &f.down.1,
    };
    assert!(matches!(
        m.bf16_moe_ffn_block(b, &f.x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));
    // The backend is still usable, which is the real assertion: a
    // mid-encode refusal would have taken the process with it.
    let x = vec![1.0f32; HIDDEN];
    assert!(m.bf16_gemv_force(&f.gate.0, &x, INTER, HIDDEN).is_some());
}

/// **The safety claim the multi-block path rests on:** blocks sharing
/// one encoder also share one scratch set, and that is safe because
/// dispatches in a compute encoder run in order with barriers between
/// them.
///
/// If it were not, block `i+1`'s gate write would race block `i`'s down
/// read and the batch would return a mix of two blocks' arithmetic. So
/// the batch must equal the same blocks run one command buffer each —
/// bit for bit, since it is the same kernels with the same arguments.
#[test]
fn a_batch_equals_the_same_blocks_run_one_per_command_buffer() {
    let m = backend();
    // Distinct weights per block: repeating one block would let a
    // scratch race pass unnoticed, because the racing values would be
    // the values it should have had.
    let fixtures: Vec<Fixture> = (0..4).map(|i| fixture_seeded(i as f32 * 7.0)).collect();
    let calls: Vec<MoeBlockCall<'_>> = fixtures
        .iter()
        .map(|f| MoeBlockCall {
            banks: f.banks(),
            x: &f.x,
        })
        .collect();

    let (batched, _gpu) = m
        .bf16_moe_ffn_blocks(&calls, BlockLowering::Separate)
        .expect("batched");
    assert_eq!(batched.len(), fixtures.len());
    for (i, f) in fixtures.iter().enumerate() {
        let (alone, _) = m.bf16_moe_ffn_block(f.banks(), &f.x).expect("single");
        assert_eq!(
            batched[i], alone,
            "block {i} differs when batched — the shared scratch is racing"
        );
    }
}

/// An empty batch is refused rather than committing an empty encoder.
#[test]
fn an_empty_batch_is_refused() {
    let m = backend();
    assert_eq!(
        m.bf16_moe_ffn_blocks(&[], BlockLowering::Separate)
            .map(|(o, _)| o),
        Err(GroupedError::NoExpertsSelected)
    );
}

/// A bad block anywhere in the batch refuses the whole batch, before
/// the encoder opens.
#[test]
fn one_bad_block_refuses_the_whole_batch() {
    let m = backend();
    let good = fixture();
    let bad = fixture_seeded(3.0);
    let mut broken = bad.banks();
    let truncated = &bad.down.0[..bad.down.0.len() / 2];
    broken.down = ExpertBankRef {
        weights: truncated,
        offsets: &bad.down.1,
    };
    let calls = [
        MoeBlockCall {
            banks: good.banks(),
            x: &good.x,
        },
        MoeBlockCall {
            banks: broken,
            x: &bad.x,
        },
    ];
    assert!(matches!(
        m.bf16_moe_ffn_blocks(&calls, BlockLowering::Separate),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));
    // Still usable: a mid-encode refusal would have aborted the process.
    let x = vec![1.0f32; HIDDEN];
    assert!(m.bf16_gemv_force(&good.gate.0, &x, INTER, HIDDEN).is_some());
}
