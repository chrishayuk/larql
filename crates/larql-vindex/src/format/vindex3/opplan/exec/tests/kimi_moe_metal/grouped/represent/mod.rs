//! **Q1a — a representation screen on ONE real expert bank.**
//!
//! The BF16 execution programme is closed at 37.33 tok/s with the whole
//! token in one command buffer and no host compute left, so every
//! further gain is fewer BYTES rather than better orchestration. The
//! routed plus shared experts are the largest clean block of that
//! traffic, and the numerically delicate recurrent machinery is not
//! touched here at all.
//!
//! This rung answers one question and stops: **at Kimi's real expert
//! geometry, what does each representation cost in bytes, and what does
//! it cost in answer?** It does NOT choose a policy, wire anything into
//! the trajectory, or quantise a second tensor family. The choice
//! belongs to a quality bank over many positions; what this establishes
//! is the order of magnitude, and whether any candidate is disqualified.
//!
//! Three properties make the screen honest:
//!
//! * The comparison is against `modeling_kimi.py`'s OWN expert outputs,
//!   not against this crate's BF16 device result. BF16 has its own
//!   distance from that oracle, reported alongside, so a format's error
//!   reads as a multiple of the floor rather than in a vacuum.
//! * The error is measured END TO END through the FFN — all three
//!   projections re-encoded, with the gate/up product formed from the
//!   re-encoded values — because that is what a deployed bank does. A
//!   per-tensor reconstruction error would flatter every format.
//! * Every format with a grouped kernel is run through THAT kernel, not
//!   through a decoded stand-in. Only MXFP4, which has no kernel on this
//!   offset-table convention, is simulated, and the cost of simulating
//!   is measured rather than asserted.

use larql_compute::cpu::ops::q4_common::{
    dequantize_q4_k, q4k_matmul_into, q6k_matmul_into, quantize_q4_k, quantize_q6_k,
};
use larql_compute_metal::trait_impl::grouped_experts::InputLayout;

use super::*;
use crate::format::vindex3::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};
use crate::format::vindex3::represent::experiment::{Provenance, RepresentationExperiment};

/// The frozen BF16 reference this whole programme measures against:
/// 27 layers plus the lm_head on device, one command buffer a token,
/// zero host compute, 16/16 tokens byte-identical.
const BF16_REFERENCE_TOKENS_PER_SECOND: f64 = 37.33;

const SUPERBLOCK_ELEMS: usize = 256;
const Q4K_SUPERBLOCK_BYTES: usize = 144;
const Q6K_SUPERBLOCK_BYTES: usize = 210;
const MXFP4_GROUP_ELEMS: usize = 32;
const MXFP4_GROUP_BYTES: usize = 16;

/// The candidate representations for an expert bank.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Bf16,
    Q6K,
    Q4K,
    /// 4.25 bpw, one e8m0 scale per 32 elements. Its grouped kernel
    /// lives on the older `moe_gpu_route` descriptor path, not on this
    /// byte-offset-table convention, so it is screened through a decoded
    /// stand-in and its THROUGHPUT stays an open question.
    Mxfp4,
}

impl Format {
    const ALL: [Format; 4] = [Format::Bf16, Format::Q6K, Format::Q4K, Format::Mxfp4];

    fn label(self) -> &'static str {
        match self {
            Format::Bf16 => "BF16",
            Format::Q6K => "Q6_K",
            Format::Q4K => "Q4_K",
            Format::Mxfp4 => "MXFP4",
        }
    }

    /// Bits per weight, from the format's own block geometry.
    fn bpw(self) -> f64 {
        match self {
            Format::Bf16 => 16.0,
            Format::Q6K => (Q6K_SUPERBLOCK_BYTES * 8) as f64 / SUPERBLOCK_ELEMS as f64,
            Format::Q4K => (Q4K_SUPERBLOCK_BYTES * 8) as f64 / SUPERBLOCK_ELEMS as f64,
            Format::Mxfp4 => (MXFP4_GROUP_BYTES * 8 + 8) as f64 / MXFP4_GROUP_ELEMS as f64,
        }
    }

    /// Whether a grouped Metal kernel reads this format's bytes directly.
    fn native(self) -> bool {
        !matches!(self, Format::Mxfp4)
    }

    /// The bytes a grouped kernel of this format would read.
    ///
    /// `k` is a whole number of superblocks at Kimi's real geometry
    /// (2304 and 1024 are both multiples of 256), so a flat quantisation
    /// never straddles a row boundary. Asserted rather than assumed — a
    /// shape where it did would silently mix two rows into one scale.
    fn encode(self, bf16: &[u8], k: usize) -> Vec<u8> {
        assert!(
            k.is_multiple_of(SUPERBLOCK_ELEMS),
            "k={k} is not a whole number of superblocks, so rows would share a block"
        );
        match self {
            Format::Bf16 => bf16.to_vec(),
            Format::Q6K => quantize_q6_k(&widen(bf16)),
            Format::Q4K => quantize_q4_k(&widen(bf16)),
            Format::Mxfp4 => panic!("MXFP4 has no grouped kernel on this convention"),
        }
    }

    /// The representation's error carried by bf16 bytes: encode, decode,
    /// re-round, and run the proven bf16 kernel.
    ///
    /// The substitution costs one extra bf16 rounding of already-encoded
    /// values. That is NOT free — `report_what_the_simulation_costs`
    /// measures it at ~3e-3 relative, which is a sixth of Q6_K's own
    /// error and a fortieth of a 4-bit format's. So it is a usable
    /// instrument for the 4-bit class and a poor one for 6-bit, which is
    /// why every native format is dispatched natively instead.
    fn simulate(self, bf16: &[u8], n: usize, k: usize) -> Vec<u8> {
        let values = match self {
            Format::Bf16 => return bf16.to_vec(),
            Format::Q6K => {
                larql_models::quant::ggml::q6_k::dequantize_q6_k(&self.encode(bf16, k), n * k)
                    .expect("q6_k decode")
            }
            Format::Q4K => dequantize_q4_k(&self.encode(bf16, k), n * k),
            Format::Mxfp4 => {
                let w = quantize_mxfp4(&widen(bf16), n, k, "screen").expect("mxfp4 encode");
                let LoadedWeight::Mxfp4 { packed, scales } = w else {
                    panic!("quantize_mxfp4 must return Mxfp4");
                };
                larql_models::quant::mxfp4::dequantize_expert(
                    packed.as_slice(),
                    scales.as_slice(),
                    n,
                    k / MXFP4_GROUP_ELEMS,
                )
                .expect("mxfp4 decode")
            }
        };
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// The dispatch with its GPU window, which is the unit the
    /// bandwidth question is asked in: wall time at these durations is
    /// dominated by a fixed ~0.2 ms of submission, so a GB/s taken from
    /// it prices the stack rather than the kernel.
    /// The argument list is the backend's grouped-dispatch signature,
    /// not this shim's: every parameter is forwarded verbatim to
    /// `*_grouped_experts*` below. A params struct would be unpacked
    /// again at that single call site.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_profiled(
        self,
        metal: &MetalBackend,
        bank: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> (Vec<f32>, f64) {
        let r = match self {
            Format::Bf16 | Format::Mxfp4 => {
                metal.bf16_grouped_experts_profiled(bank, offsets, x, n, k, layout)
            }
            Format::Q6K => metal.q6k_grouped_experts_profiled(bank, offsets, x, n, k, layout),
            Format::Q4K => metal.q4k_grouped_experts_profiled(bank, offsets, x, n, k, layout),
        };
        r.unwrap_or_else(|e| panic!("{} grouped dispatch refused: {e}", self.label()))
    }

    /// The argument list is the backend's grouped-dispatch signature,
    /// not this shim's: every parameter is forwarded verbatim to
    /// `*_grouped_experts*` below. A params struct would be unpacked
    /// again at that single call site.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        self,
        metal: &MetalBackend,
        bank: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Vec<f32> {
        let r = match self {
            Format::Bf16 | Format::Mxfp4 => {
                metal.bf16_grouped_experts(bank, offsets, x, n, k, layout)
            }
            Format::Q6K => metal.q6k_grouped_experts(bank, offsets, x, n, k, layout),
            Format::Q4K => metal.q4k_grouped_experts(bank, offsets, x, n, k, layout),
        };
        r.unwrap_or_else(|e| panic!("{} grouped dispatch refused: {e}", self.label()))
    }
}

fn widen(bf16: &[u8]) -> Vec<f32> {
    bf16.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Root-mean-square error relative to the reference's own RMS.
///
/// Reported beside the max, because a single saturated weight moves the
/// max while leaving the answer usable, and a broadly-degraded bank
/// moves the RMS while leaving the max unremarkable.
fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let se: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let ss: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    assert!(ss > 0.0, "degenerate reference");
    (se / ss).sqrt() as f32
}

/// One format's three banks, with the offset tables the kernel reads.
///
/// **Held for the whole test, deliberately.** `bf16_grouped_experts`
/// binds through `BufferCache`, which caches on `(ptr, len)` — sound
/// only for allocations that outlive every dispatch. These banks are all
/// the same length, so dropping one arm's and building the next one's
/// lets the allocator hand back an address the cache still holds, and
/// the gate bank silently dispatches against the previous arm's UP
/// matrix. That is not hypothetical: it turned an identical repeat of
/// the BF16 arm from 2.5e-7 into 1.4e0 the first time this test was
/// written. See `feedback_residency_covers_the_bound_buffer_object`.
struct Arm {
    format: Format,
    gate: Reencoded,
    up: Reencoded,
    down: Reencoded,
}

struct Reencoded {
    bytes: Vec<u8>,
    offsets: Vec<ExpertOffset>,
}

impl Reencoded {
    fn build(fx: &Fixture, stage: Stage, f: Format) -> Self {
        let (n, k) = stage.shape(fx);
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(fx.experts.len());
        for e in &fx.experts {
            // Expert identity travels as `identity -> row range -> output
            // slice`, exactly as the BF16 bank does; only the payload
            // encoding changes.
            let coded = if f.native() {
                f.encode(stage.matrix(e), k)
            } else {
                f.simulate(stage.matrix(e), n, k)
            };
            offsets.push(ExpertOffset(bytes.len() as u32));
            bytes.extend_from_slice(&coded);
        }
        Self { bytes, offsets }
    }

    /// The same bank repeated `times`, as DISTINCT bytes at distinct
    /// addresses.
    ///
    /// Repeating the offset table instead would re-read one copy and
    /// measure the cache; the question here is what the memory system
    /// does when the bytes are actually new, which is the situation a
    /// 26-layer token is in.
    fn replicated(&self, times: usize) -> Self {
        let per = self.bytes.len();
        let mut bytes = Vec::with_capacity(per * times);
        let mut offsets = Vec::with_capacity(self.offsets.len() * times);
        for r in 0..times {
            bytes.extend_from_slice(&self.bytes);
            offsets.extend(
                self.offsets
                    .iter()
                    .map(|o| ExpertOffset(o.0 + (r * per) as u32)),
            );
        }
        Self { bytes, offsets }
    }
}

impl Arm {
    fn build(fx: &Fixture, f: Format) -> Self {
        Self {
            format: f,
            gate: Reencoded::build(fx, Stage::Gate, f),
            up: Reencoded::build(fx, Stage::Up, f),
            down: Reencoded::build(fx, Stage::Down, f),
        }
    }

    /// What a grouped kernel of this format would actually read. For the
    /// simulated arm the carrier is bf16-sized, so the size that matters
    /// is derived from the block geometry instead.
    fn bank_bytes(&self) -> usize {
        let carried = self.gate.bytes.len() + self.up.bytes.len() + self.down.bytes.len();
        if self.format.native() {
            carried
        } else {
            (carried as f64 * self.format.bpw() / 16.0).round() as usize
        }
    }

    /// The whole FFN for every slot, in this representation.
    fn ffn(&self, metal: &MetalBackend, fx: &Fixture) -> Vec<f32> {
        let f = self.format;
        let (n_gu, k_gu) = Stage::Gate.shape(fx);
        let gate = f.dispatch(
            metal,
            &self.gate.bytes,
            &self.gate.offsets,
            &fx.x,
            n_gu,
            k_gu,
            InputLayout::Shared,
        );
        let up = f.dispatch(
            metal,
            &self.up.bytes,
            &self.up.offsets,
            &fx.x,
            n_gu,
            k_gu,
            InputLayout::Shared,
        );
        // The gate/up product is formed from the RE-ENCODED values, so
        // the error the down projection sees is the error a deployed
        // bank would hand it.
        let h = down_inputs(fx, &gate, &up);
        let (n_d, k_d) = Stage::Down.shape(fx);
        f.dispatch(
            metal,
            &self.down.bytes,
            &self.down.offsets,
            &h,
            n_d,
            k_d,
            InputLayout::PerSlot,
        )
    }
}

/// How many times the nine-expert bank is repeated for the throughput
/// arm.
///
/// One layer's real bank is 40.5 MiB in BF16 and runs in ~0.25 ms —
/// inside this chip's cache and close enough to the submission floor
/// that the GPU timer's median came back 4x its own minimum, which the
/// spread check correctly refused. Twelve copies put BF16 at ~486 MiB
/// and the smallest candidate at ~137 MiB, both past any cache, with
/// windows long enough to time. It also matches what a token actually
/// does: 26 layers back to back stream ~10 GB, not 40 MiB.
const THROUGHPUT_REPLICAS: usize = 12;

mod record;
mod screen;
mod throughput;
