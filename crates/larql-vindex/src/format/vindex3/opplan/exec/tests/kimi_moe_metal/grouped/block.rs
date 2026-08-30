//! Rung 3: the whole expert FFN in ONE command buffer.
//!
//! Rung 2 took the kernel to 288-372 GB/s — 80-100% of this machine's
//! roofline — and showed a row-tiling sweep from 1152 to 9216
//! threadgroups moves nothing, so the kernel is not the bottleneck any
//! more. What it left was submission: ~0.30 ms of wall per stage against
//! ~0.14 ms of GPU-busy, paid three times by a three-stage FFN.
//!
//! **The hypothesis:** collapsing three submissions into one is a large
//! block-level win with the arithmetic unchanged. GPU-busy should stay
//! put; wall should fall by roughly two submissions' worth.
//!
//! **The number that matters now is effective end-to-end bandwidth** —
//! stored expert bytes over WALL seconds, not over GPU-busy. The kernel's
//! internal rate is already near roofline; what a decode loop actually
//! gets is the rate including dispatch, and that is the figure worth
//! extrapolating from.
//!
//! Gate and up stay unfused on purpose. Fusing them here would mix
//! "fewer submissions" with "one input read and less intermediate
//! traffic", and neither would be attributable afterwards.

use larql_compute_metal::trait_impl::bf16_moe_block::{
    BlockLowering, ExpertBankRef, FusedTiling, MoeFfnBanks,
};
use larql_compute_metal::trait_impl::grouped_experts::InputLayout;

use super::*;

/// Warmup pairs before the block measurement starts. Sized against the
/// ramp factor the test itself reports, not by feel.
const BLOCK_WARMUP: usize = 25;

/// The three banks a block needs, built once and held for the whole
/// test — see the aliasing note on [`Bank`].
struct BlockBanks {
    gate: Bank,
    up: Bank,
    down: Bank,
}

impl BlockBanks {
    fn build(fx: &Fixture) -> Self {
        Self {
            gate: Bank::build(fx, Stage::Gate),
            up: Bank::build(fx, Stage::Up),
            down: Bank::build(fx, Stage::Down),
        }
    }

    fn refs<'a>(&'a self, fx: &Fixture) -> MoeFfnBanks<'a> {
        let r = |b: &'a Bank| ExpertBankRef {
            weights: &b.bytes,
            offsets: &b.offsets,
        };
        MoeFfnBanks {
            gate: r(&self.gate),
            up: r(&self.up),
            down: r(&self.down),
            hidden: fx.hidden,
            inter: fx.inter,
        }
    }

    /// Stored bytes the block reads: all three projections for every
    /// selected expert.
    fn bytes(&self) -> f64 {
        (self.gate.bytes.len() + self.up.bytes.len() + self.down.bytes.len()) as f64
    }
}

/// The per-stage path this rung replaces: three grouped dispatches, each
/// its own command buffer, with the activation on the host.
fn staged_block(metal: &MetalBackend, fx: &Fixture, banks: &BlockBanks) -> Vec<f32> {
    let gate = metal
        .bf16_grouped_experts(
            &banks.gate.bytes,
            &banks.gate.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("staged gate");
    let up = metal
        .bf16_grouped_experts(
            &banks.up.bytes,
            &banks.up.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("staged up");
    let h = down_inputs(fx, &gate, &up);
    metal
        .bf16_grouped_experts(
            &banks.down.bytes,
            &banks.down.offsets,
            &h,
            fx.hidden,
            fx.inter,
            InputLayout::PerSlot,
        )
        .expect("staged down")
}

/// The gate, unchanged in every respect that matters: every selected
/// expert plus the shared branch, scored against `modeling_kimi.py`'s
/// own per-expert output, at the same tolerance rungs 1 and 2 passed.
///
/// The block moves the activation onto the GPU — it is the one piece
/// that was not already a Metal kernel, and leaving it on the host would
/// force the commit-and-wait this rung exists to remove. `geglu_silu`
/// evaluates the same `(g / (1 + exp(-g))) * up` expression, so the only
/// possible difference is `exp` from Metal's library against `exp` from
/// the host's. This reports what that costs rather than assuming it is
/// nothing.
#[test]
fn the_one_command_buffer_block_matches_the_checkpoints_own_output() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let banks = BlockBanks::build(&fx);
    let (block, _gpu) = metal
        .bf16_moe_ffn_block(banks.refs(&fx), &fx.x)
        .expect("one-CB block");
    let staged = staged_block(&metal, &fx, &banks);

    assert_eq!(block.len(), fx.experts.len() * fx.hidden);
    for (slot, e) in fx.experts.iter().enumerate() {
        let got = &block[slot * fx.hidden..(slot + 1) * fx.hidden];
        let vs_staged = &staged[slot * fx.hidden..(slot + 1) * fx.hidden];
        let vs_oracle = rel_err(got, &e.oracle);
        let vs_stages = rel_err(got, vs_staged);
        let exact = got == vs_staged;
        eprintln!(
            "[block] [{:>9}] one-CB-vs-hf {vs_oracle:.3e}  one-CB-vs-staged {vs_stages:.3e}{}",
            e.id,
            if exact { "  (bit-identical)" } else { "" }
        );
        assert!(
            vs_oracle < REL_TOLERANCE,
            "slot {slot} ({}): one-CB block vs the checkpoint, rel {vs_oracle:e}",
            e.id
        );
        assert!(
            vs_stages < REL_TOLERANCE,
            "slot {slot} ({}): one-CB block vs the staged stages, rel {vs_stages:e}",
            e.id
        );
    }
}

/// **Control.** Rotating the offset tables must rotate which expert each
/// slot answers for, through the block as well as through a single
/// dispatch — the block builds three tables and binds three banks, so it
/// has three more chances to lose the correspondence.
#[test]
fn rotating_the_tables_rotates_the_blocks_slots() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let banks = BlockBanks::build(&fx);
    let (forward, _) = metal
        .bf16_moe_ffn_block(banks.refs(&fx), &fx.x)
        .expect("forward");

    let rotate = |v: &[ExpertOffset]| {
        let mut r = v.to_vec();
        r.rotate_left(1);
        r
    };
    let (g, u, d) = (
        rotate(&banks.gate.offsets),
        rotate(&banks.up.offsets),
        rotate(&banks.down.offsets),
    );
    let mut rotated = banks.refs(&fx);
    rotated.gate.offsets = &g;
    rotated.up.offsets = &u;
    rotated.down.offsets = &d;
    let (after, _) = metal.bf16_moe_ffn_block(rotated, &fx.x).expect("rotated");

    let slots = fx.experts.len();
    for slot in 0..slots {
        let source = (slot + 1) % slots;
        assert_eq!(
            &after[slot * fx.hidden..(slot + 1) * fx.hidden],
            &forward[source * fx.hidden..(source + 1) * fx.hidden],
            "slot {slot} should now hold what slot {source} held"
        );
    }
    assert_ne!(forward, after, "control: the experts must differ at all");
}

/// The measurement this rung exists for: what a whole expert FFN costs
/// as one submission against three, in the three units that matter.
///
/// **GPU-busy** should be unchanged — same kernels, same bytes, plus one
/// cheap element-wise dispatch. **Wall** should fall by roughly two
/// submissions. **Effective bandwidth over wall** is the figure a decode
/// loop actually gets, and the one to extrapolate from.
///
/// Arms interleaved and the ramp factor reported, because this machine's
/// GPU clock ramps within and across runs — a blocked comparison here
/// would credit the second arm with the clock.
#[test]
fn report_one_command_buffer_against_three() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let banks = BlockBanks::build(&fx);
    let bytes = banks.bytes();
    let gbs = |ms: f64| bytes / 1e6 / ms;

    // Wall for each arm, measured in the same interleaved loop. The
    // staged arm's three command buffers each report their own GPU
    // window, so its GPU-busy is their sum.
    let mut staged_wall = Vec::with_capacity(BENCH_ITERS);
    let mut block_wall = Vec::with_capacity(BENCH_ITERS);
    let mut staged_gpu = Vec::with_capacity(BENCH_ITERS);
    let mut block_gpu = Vec::with_capacity(BENCH_ITERS);

    let staged_arm = || {
        let t = Instant::now();
        let gate = metal
            .bf16_grouped_experts_profiled(
                &banks.gate.bytes,
                &banks.gate.offsets,
                &fx.x,
                fx.inter,
                fx.hidden,
                InputLayout::Shared,
            )
            .expect("gate");
        let up = metal
            .bf16_grouped_experts_profiled(
                &banks.up.bytes,
                &banks.up.offsets,
                &fx.x,
                fx.inter,
                fx.hidden,
                InputLayout::Shared,
            )
            .expect("up");
        let h = down_inputs(&fx, &gate.0, &up.0);
        let down = metal
            .bf16_grouped_experts_profiled(
                &banks.down.bytes,
                &banks.down.offsets,
                &h,
                fx.hidden,
                fx.inter,
                InputLayout::PerSlot,
            )
            .expect("down");
        std::hint::black_box(down.0);
        (t.elapsed().as_secs_f64() * 1000.0, gate.1 + up.1 + down.1)
    };
    let block_arm = || {
        let t = Instant::now();
        let (out, gpu) = metal
            .bf16_moe_ffn_block(banks.refs(&fx), &fx.x)
            .expect("block");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    // A far longer warmup than the lighter tests use. The block streams
    // ~121 MiB a call, and at this weight the ramp factor stayed above
    // 1.4 with the shared warmup — the arms were still accelerating well
    // into the measurement window.
    for _ in 0..BLOCK_WARMUP {
        staged_arm();
        block_arm();
    }
    for _ in 0..BENCH_ITERS {
        let (w, g) = staged_arm();
        staged_wall.push(w);
        staged_gpu.push(g);
        let (w, g) = block_arm();
        block_wall.push(w);
        block_gpu.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (staged_wall.len() / 3).max(1);
    let ramp = mean(&staged_wall[..third]) / mean(&staged_wall[staged_wall.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (sw, bw) = (median(&mut staged_wall), median(&mut block_wall));
    let (sg, bg) = (median(&mut staged_gpu), median(&mut block_gpu));

    eprintln!(
        "[block] {} slots, hidden={} inter={}, {:.1} MiB stored across gate+up+down",
        fx.experts.len(),
        fx.hidden,
        fx.inter,
        bytes / (1024.0 * 1024.0),
    );
    eprintln!(
        "[block]   3 command buffers  wall {sw:.3} ms  gpu {sg:.3} ms  \
         effective {:>6.1} GB/s (kernel {:.1})",
        gbs(sw),
        gbs(sg),
    );
    eprintln!(
        "[block]   1 command buffer   wall {bw:.3} ms  gpu {bg:.3} ms  \
         effective {:>6.1} GB/s (kernel {:.1})",
        gbs(bw),
        gbs(bg),
    );
    eprintln!(
        "[block]   one CB vs three: {:.2}x wall, {:.2}x GPU-busy, \
         {:.3} ms of wall removed   [ramp {ramp:.2}x]",
        sw / bw,
        sg / bg,
        sw - bw,
    );
    eprintln!(
        "[block]   host time outside the GPU: staged {:.3} ms, block {:.3} ms",
        sw - sg,
        bw - bg,
    );
    assert!(sw.is_finite() && bw.is_finite() && bw > 0.0);
}

/// Every lowering, at real geometry, against the checkpoint's own output.
///
/// The bit-identity claim is pinned synthetically in
/// `every_lowering_computes_the_same_values`; this checks it survives
/// Kimi's shapes, where `K` is 2304 rather than 64 and a reassociation
/// would have 36x more terms to show up in.
#[test]
fn every_lowering_matches_the_checkpoint_at_real_geometry() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let banks = BlockBanks::build(&fx);

    let mut reference: Option<Vec<f32>> = None;
    for lowering in LOWERINGS {
        let (out, _gpu) = metal
            .bf16_moe_ffn_block_lowered(banks.refs(&fx), &fx.x, lowering)
            .expect("lowered block");
        for (slot, e) in fx.experts.iter().enumerate() {
            let got = &out[slot * fx.hidden..(slot + 1) * fx.hidden];
            let rel = rel_err(got, &e.oracle);
            assert!(
                rel < REL_TOLERANCE,
                "{lowering:?} slot {slot} ({}): rel {rel:e} vs the checkpoint",
                e.id
            );
        }
        match &reference {
            None => {
                eprintln!(
                    "[fuse] {lowering:?} vs the checkpoint: all {} branches pass",
                    fx.experts.len()
                );
                reference = Some(out);
            }
            Some(want) => {
                let exact = &out == want;
                eprintln!(
                    "[fuse] {lowering:?} vs Separate: {}",
                    if exact {
                        "bit-identical".to_string()
                    } else {
                        format!("rel {:e}", rel_err(&out, want))
                    }
                );
                assert_eq!(&out, want, "{lowering:?} changed a value");
            }
        }
    }
}

/// The lowerings under test, control first.
///
/// Both fused kernels appear at both tilings. `Rows8` matches the
/// unfused kernel's tiling and therefore HALVES the threadgroup count
/// for the gate+up work; `Rows4` restores it. Without the second, a
/// fused result could not be told apart from a launch-size result.
const LOWERINGS: [BlockLowering; 5] = [
    BlockLowering::Separate,
    BlockLowering::FusedGateUp(FusedTiling::Rows8),
    BlockLowering::FusedGateUp(FusedTiling::Rows4),
    BlockLowering::FusedGateUpAct(FusedTiling::Rows8),
    BlockLowering::FusedGateUpAct(FusedTiling::Rows4),
];

/// Dispatches each lowering encodes into the one command buffer.
fn dispatch_count(l: BlockLowering) -> usize {
    match l {
        BlockLowering::Separate => 4,          // gate, up, activation, down
        BlockLowering::FusedGateUp(_) => 3,    // gate+up, activation, down
        BlockLowering::FusedGateUpAct(_) => 2, // gate+up+activation, down
    }
}

/// Threadgroups the gate+up work launches — the variable `Rows4` exists
/// to hold constant against `Separate`.
fn gate_up_threadgroups(l: BlockLowering, slots: usize, inter: usize) -> usize {
    let tiles = |rows: usize| inter.div_ceil(rows) * slots;
    match l {
        BlockLowering::Separate => 2 * tiles(8),
        BlockLowering::FusedGateUp(FusedTiling::Rows8)
        | BlockLowering::FusedGateUpAct(FusedTiling::Rows8) => tiles(8),
        BlockLowering::FusedGateUp(FusedTiling::Rows4)
        | BlockLowering::FusedGateUpAct(FusedTiling::Rows4) => tiles(4),
    }
}

/// Intermediate bytes a lowering moves between its own dispatches —
/// **instrumented from the shapes, not claimed.**
///
/// Every arm reads the same weights, so this is the only traffic fusion
/// can remove, and stating it beside the weight total is what keeps the
/// result honest: if intermediates are a fraction of a percent of the
/// bytes, then fusion cannot win on traffic however it is arranged, and
/// any win it does show came from somewhere else.
fn intermediate_bytes(l: BlockLowering, slots: usize, inter: usize) -> f64 {
    let stream = (slots * inter * 4) as f64;
    match l {
        // write gate, write up, read both, write h, read h
        BlockLowering::Separate | BlockLowering::FusedGateUp(_) => 6.0 * stream,
        // write h, read h
        BlockLowering::FusedGateUpAct(_) => 2.0 * stream,
    }
}

/// **Rung 4.** What fusion is worth, with command-buffer structure held
/// constant.
///
/// All three arms are ONE command buffer with the same banks and the
/// same down dispatch, so nothing here is about submission — rung 3
/// already took that out, which is precisely what makes a few percent
/// measurable. The control arm keeps rung 3's exact shape.
///
/// Arms interleaved and the ramp reported: at this weight the machine
/// needed 25 warmup passes before it held still, and the effect being
/// measured is smaller than the drift it would otherwise absorb.
#[test]
fn report_fused_gate_up_against_separate() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let banks = BlockBanks::build(&fx);
    let weight_bytes = banks.bytes();
    let slots = fx.experts.len();

    let arm = |l: BlockLowering| {
        let t = Instant::now();
        let (out, gpu) = metal
            .bf16_moe_ffn_block_lowered(banks.refs(&fx), &fx.x, l)
            .expect("lowered block");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    let mut wall: Vec<Vec<f64>> = LOWERINGS.iter().map(|_| Vec::new()).collect();
    let mut gpu: Vec<Vec<f64>> = LOWERINGS.iter().map(|_| Vec::new()).collect();
    for _ in 0..BLOCK_WARMUP {
        for l in LOWERINGS {
            arm(l);
        }
    }
    for _ in 0..BENCH_ITERS {
        for (i, l) in LOWERINGS.into_iter().enumerate() {
            let (w, g) = arm(l);
            wall[i].push(w);
            gpu[i].push(g);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (wall[0].len() / 3).max(1);
    let ramp = mean(&wall[0][..third]) / mean(&wall[0][wall[0].len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    eprintln!(
        "[fuse] {slots} slots, hidden={} inter={}, weights {:.1} MiB (identical in every arm)",
        fx.hidden,
        fx.inter,
        weight_bytes / (1024.0 * 1024.0),
    );
    let mut baseline: Option<(f64, f64)> = None;
    for (i, l) in LOWERINGS.into_iter().enumerate() {
        let (w, g) = (median(&mut wall[i]), median(&mut gpu[i]));
        let inter_b = intermediate_bytes(l, slots, fx.inter);
        eprintln!(
            "[fuse]   {:<32} {} disp  {:>5} gate/up TGs  wall {w:.3} ms  gpu {g:.3} ms  \
             kernel {:>6.1} GB/s  interm {:>3.0} KiB ({:.3}%){}",
            format!("{l:?}"),
            dispatch_count(l),
            gate_up_threadgroups(l, slots, fx.inter),
            weight_bytes / 1e6 / g,
            inter_b / 1024.0,
            100.0 * inter_b / (weight_bytes + inter_b),
            match baseline {
                None => String::new(),
                Some((bw, bg)) => format!("  [{:.3}x wall, {:.3}x gpu]", bw / w, bg / g),
            }
        );
        if baseline.is_none() {
            baseline = Some((w, g));
        }
    }
    eprintln!("[fuse]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}
