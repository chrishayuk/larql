//! Rung 2: every selected expert in ONE dispatch, and whether that
//! closes the gap rung 1 left.
//!
//! Rung 1 established the arm was submission-bound, not bandwidth-bound:
//! a gemv reading 512 bytes cost 0.197 ms, and folding nine real expert
//! GEMVs into one command buffer won 4.68x with no other change, lifting
//! achieved bandwidth 20.0 → 93.5 GB/s. That left a second question,
//! because 93.5 GB/s is still ~4x off this machine's ~370 GB/s roofline
//! and one Kimi projection at `[1024, 2304]` launches only `1024/8 =
//! 128` threadgroups.
//!
//! **The hypothesis under test:** raising one launch from ~128
//! threadgroups to ~1,150 materially raises achieved bandwidth, because
//! what limits the kernel now is occupancy rather than submission.
//!
//! Nothing else changes. No quantisation, no fused activation, no
//! router, no full-model wiring — the gate/up/down stages are still
//! three stages with the same CPU activation between them, so a gain
//! here is attributable to the dispatch shape and nothing else. Whether
//! gate/up fusion is worth it is the NEXT question, deliberately not
//! confounded with this one.
//!
//! **Each stage is measured separately.** gate and up are `[1024, 2304]`
//! reading a shared hidden state; down is `[2304, 1024]` reading each
//! expert's own activation. Different row geometry, so one aggregate
//! number could hide exactly the occupancy effect being looked for.

use larql_compute_metal::trait_impl::grouped_experts::{ExpertOffset, InputLayout};

use super::*;

mod bench;
mod block;
mod epoch;
mod parity;
mod represent;

/// Which projection of the expert FFN a stage measures. Named rather
/// than indexed because the three differ in row geometry AND in input
/// layout, and a mix-up computes a plausible number from the wrong one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// `w1`, `[inter, hidden]`, one hidden state shared by every slot.
    Gate,
    /// `w3`, `[inter, hidden]`, likewise shared.
    Up,
    /// `w2`, `[hidden, inter]`, each slot reading its OWN activation.
    Down,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Self::Gate => "gate (w1)",
            Self::Up => "up   (w3)",
            Self::Down => "down (w2)",
        }
    }

    /// `(n, k)` for this stage at the fixture's geometry.
    fn shape(self, fx: &Fixture) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (fx.inter, fx.hidden),
            Self::Down => (fx.hidden, fx.inter),
        }
    }

    fn layout(self) -> InputLayout {
        match self {
            Self::Gate | Self::Up => InputLayout::Shared,
            Self::Down => InputLayout::PerSlot,
        }
    }

    fn matrix(self, e: &Expert) -> &[u8] {
        match self {
            Self::Gate => &e.gate,
            Self::Up => &e.up,
            Self::Down => &e.down,
        }
    }
}

/// One stage's weights for every slot, concatenated, plus each slot's
/// byte offset.
///
/// **The staging is the honest cost of Kimi's real layout.** The
/// checkpoint is `ExpertBank::PerExpert` — separate tensors — and a
/// grouped dispatch binds one buffer, so the selected experts have to
/// share one allocation. Here that is a repack done once, outside every
/// timed loop and reported on its own; in a container that stored the
/// bank contiguously it would not exist at all. Which is itself part of
/// what this rung is measuring: whether packing experts together is
/// worth doing in the format.
///
/// **Every bank must outlive every measurement that names it.** The
/// device buffer cache keys on `(ptr, len)`, which is correct for the
/// mmap'd weights it was built for and a trap for a temporary `Vec`: drop
/// one 40.5 MiB bank, allocate the next at the same address with the same
/// length, and the cache hands back the *previous* bank's buffer. The
/// `debug_assert` guarding that is compiled out in `--release`, which is
/// the only profile a bench runs in. This first showed up as an `up`
/// stage reporting 448 GB/s — above this chip's DRAM peak, so impossible
/// — because it was re-reading the gate bank the cache still held.
struct Bank {
    bytes: Vec<u8>,
    offsets: Vec<ExpertOffset>,
    stage_ms: f64,
}

impl Bank {
    fn build(fx: &Fixture, stage: Stage) -> Self {
        let t = Instant::now();
        let per_expert = stage.matrix(&fx.experts[0]).len();
        let mut bytes = Vec::with_capacity(per_expert * fx.experts.len());
        let mut offsets = Vec::with_capacity(fx.experts.len());
        for e in &fx.experts {
            let m = stage.matrix(e);
            assert_eq!(m.len(), per_expert, "{}: ragged expert payload", e.id);
            offsets.push(ExpertOffset(bytes.len() as u32));
            bytes.extend_from_slice(m);
        }
        Self {
            bytes,
            offsets,
            stage_ms: t.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn slots(&self) -> usize {
        self.offsets.len()
    }
}

/// The per-slot inputs the down projection consumes: `silu(gate) * up`
/// for each expert, concatenated — the same values `expert_ffn` forms
/// between its own second and third matvec.
fn down_inputs(fx: &Fixture, gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| silu(g) * u)
        .collect::<Vec<f32>>()
        .chunks_exact(fx.inter)
        .flat_map(|c| c.to_vec())
        .collect()
}

/// Every slot's output for one stage, as N separate `bf16_gemv_force`
/// dispatches — the rung-1 shape, kept as the reference the grouped
/// kernel must reproduce bit for bit.
fn per_expert_stage(metal: &MetalBackend, fx: &Fixture, stage: Stage, x: &[f32]) -> Vec<f32> {
    let (n, k) = stage.shape(fx);
    fx.experts
        .iter()
        .enumerate()
        .flat_map(|(slot, e)| {
            let xs = match stage.layout() {
                InputLayout::Shared => &x[..k],
                InputLayout::PerSlot => &x[slot * k..(slot + 1) * k],
            };
            metal
                .bf16_gemv_force(stage.matrix(e), xs, n, k)
                .expect("per-expert dispatch")
        })
        .collect()
}
