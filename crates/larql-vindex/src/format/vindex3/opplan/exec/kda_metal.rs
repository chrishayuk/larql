//! KDA's four wide projections, executed on Metal.
//!
//! Rung 5b of the Kimi Metal ladder, and deliberately narrow: this
//! changes WHERE four matvecs run and nothing else. The convolution,
//! q/k norms, low-rank gates, decay and the recurrence itself stay on
//! the proven CPU path, and the layer's own oracle gate is unchanged.
//!
//! **Two jobs, not one.** The obvious one is compute: q/k/v/o are
//! `[4096, 2304]` and `[2304, 4096]` bf16, 18 MiB each, 72 MiB a layer.
//! The one rung 5a exposed is architectural — each CPU↔GPU
//! command-buffer boundary costs ~0.23 ms, so how many crossings a layer
//! makes matters as much as how fast the kernels are. Four separate
//! crossings would spend ~0.9 ms a layer on orchestration alone, more
//! than the arithmetic.
//!
//! So the shapes here are chosen by the dependency structure:
//!
//! ```text
//! CPU normalised hidden
//!         |
//!   ONE command buffer:  q_proj, k_proj, v_proj   (they share the input)
//!         |
//!   CPU conv + q/k norm + gates + recurrence + gated o_norm
//!         |
//!   ONE command buffer:  o_proj
//!         |
//!   CPU continuation
//! ```
//!
//! Two crossings a layer, not four. That is the floor for this shape:
//! `o_proj`'s input does not exist until the recurrence has run, so no
//! arrangement of these four matrices alone can do better. Getting below
//! two needs the recurrence itself on-device, which is rung 5c's problem
//! and not this one's.
//!
//! **Two q/k/v arms, because rung 2 measured the difference.** The three
//! matrices have identical shape and share an input, which is exactly
//! what `bf16_grouped_experts` was built for — it is a
//! same-shape-projections-sharing-one-input kernel that happens to have
//! been motivated by experts. [`Batched`] issues three dispatches in one
//! command buffer on the checkpoint's own separate tensors; [`Grouped`]
//! issues ONE dispatch of three slots over a contiguous `q|k|v` bank.
//! Rung 2 found the second worth ~1.85x on GPU-busy at expert shapes,
//! and it costs a one-time 56 MiB repack that a container storing the
//! three adjacently would not need.

use std::sync::atomic::{AtomicU64, Ordering};

use larql_compute_metal::trait_impl::bf16_grouped::GroupedShape;
use larql_compute_metal::trait_impl::grouped_experts::{ExpertOffset, InputLayout};
use larql_compute_metal::MetalBackend;

use super::kda::{KdaProjections, KdaWeights};
use super::timing::{timed, OpClass};

/// Bytes per bf16 code.
const BF16_BYTES: usize = 2;

/// Reinterpret bf16 codes as the bytes a Metal buffer binds.
///
/// The shader reads `ushort`, and this crate's `u16` codes are already
/// exactly those bits in host order. Apple Silicon is little-endian and
/// this module is macOS-only, which is the same assumption
/// `MatMul::f16_gemv` has always made about its `&[u8]` weights.
fn codes_as_bytes(w: &[u16]) -> &[u8] {
    // SAFETY: `u16` has no padding and no invalid bit patterns, and `u8`
    // has weaker alignment, so any `&[u16]` is a valid `&[u8]` of twice
    // the length for the same lifetime.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * BF16_BYTES) }
}

/// How q, k and v are submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvSubmission {
    /// Three dispatches in one command buffer, reading the checkpoint's
    /// own separate tensors. No staging.
    Batched,
    /// One grouped dispatch of three slots over a contiguous `q|k|v`
    /// bank. Needs the repack [`MetalKdaProjections::new`] performs once.
    Grouped,
}

/// The four KDA projections on Metal, with the CPU recurrence unchanged.
pub struct MetalKdaProjections<'a> {
    metal: &'a MetalBackend,
    submission: QkvSubmission,
    /// `q|k|v` concatenated, built once — `Grouped` binds one buffer, so
    /// the three have to share an allocation. Empty for `Batched`.
    ///
    /// Held for the projector's whole life rather than rebuilt per call:
    /// the device buffer cache keys on `(ptr, len)`, so a bank that was
    /// dropped and reallocated at the same size would silently alias the
    /// previous one.
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    /// GPU-busy nanoseconds accumulated across this projector's
    /// dispatches.
    ///
    /// The load-bearing diagnostic for this rung, not a nicety: rung 5a
    /// priced a command-buffer crossing at ~0.23 ms, and KDA's dependency
    /// chain forces two of them a layer. Against ~72 MiB of projections
    /// that is a large fraction of the whole cost, and only the GPU
    /// window separates "the kernel is slow" from "the boundary is
    /// expensive". Wall alone cannot tell those apart.
    gpu_nanos: AtomicU64,
}

impl<'a> MetalKdaProjections<'a> {
    /// `w` supplies the layer's weights; only q/k/v are read here, and
    /// only when `submission` is [`QkvSubmission::Grouped`].
    pub fn new(metal: &'a MetalBackend, w: KdaWeights<'_>, submission: QkvSubmission) -> Self {
        let (qkv_bank, qkv_offsets) = match submission {
            QkvSubmission::Batched => (Vec::new(), [ExpertOffset(0); 3]),
            QkvSubmission::Grouped => {
                let per = w.q_proj.len() * BF16_BYTES;
                debug_assert_eq!(w.k_proj.len() * BF16_BYTES, per);
                debug_assert_eq!(w.v_proj.len() * BF16_BYTES, per);
                let mut bank = Vec::with_capacity(3 * per);
                for m in [w.q_proj, w.k_proj, w.v_proj] {
                    bank.extend_from_slice(codes_as_bytes(m));
                }
                let offsets = [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ];
                (bank, offsets)
            }
        };
        Self {
            metal,
            submission,
            qkv_bank,
            qkv_offsets,
            gpu_nanos: AtomicU64::new(0),
        }
    }

    /// GPU-busy milliseconds since the last call, and reset.
    pub fn take_gpu_ms(&self) -> f64 {
        self.gpu_nanos.swap(0, Ordering::Relaxed) as f64 / 1e6
    }

    fn add_gpu(&self, ms: f64) {
        self.gpu_nanos
            .fetch_add((ms * 1e6) as u64, Ordering::Relaxed);
    }

    /// One projection through the grouped kernel, which reports its own
    /// GPU window. A single slot is a degenerate "bank of one" — the
    /// same kernel and the same arithmetic as any other slot count.
    fn grouped(&self, bank: &[u8], offsets: &[ExpertOffset], x: &[f32], n: usize) -> Vec<f32> {
        let (out, gpu) = self
            .metal
            .bf16_grouped_experts_tiled(
                self.metal.default_grouped_handle_pub(),
                bank,
                offsets,
                x,
                GroupedShape {
                    n,
                    k: x.len(),
                    layout: InputLayout::Shared,
                },
            )
            .expect("grouped dispatch");
        self.add_gpu(gpu);
        out
    }
}

impl KdaProjections for MetalKdaProjections<'_> {
    /// **One command buffer for all three.**
    ///
    /// Timed under [`OpClass::KdaQProj`] alone: for a batching backend
    /// q, k and v are one indivisible submission, and splitting one
    /// measured interval three ways would invent a number. The ledger
    /// therefore reads "q" for the whole batch on this arm — stated here
    /// so a q/k/v breakdown is not read off it.
    fn qkv(&self, w: KdaWeights<'_>, x: &[f32], width: usize) -> [Vec<f32>; 3] {
        let _t = timed(OpClass::KdaQProj);
        let k = x.len();
        let mut out = match self.submission {
            QkvSubmission::Batched => {
                let batch = [
                    (codes_as_bytes(w.q_proj), width, k),
                    (codes_as_bytes(w.k_proj), width, k),
                    (codes_as_bytes(w.v_proj), width, k),
                ];
                let (out, gpu) = self
                    .metal
                    .encode_bf16_gemv_multi_profiled(&batch, x)
                    .expect("bf16 qkv batch dispatches");
                self.add_gpu(gpu);
                out
            }
            QkvSubmission::Grouped => self
                .grouped(&self.qkv_bank, &self.qkv_offsets, x, width)
                .chunks_exact(width)
                .map(|c| c.to_vec())
                .collect(),
        };
        let v = out.pop().expect("three results");
        let kk = out.pop().expect("three results");
        let q = out.pop().expect("three results");
        [q, kk, v]
    }

    /// `o_proj` alone — its input does not exist until the recurrence
    /// has run, so this crossing cannot be merged with the one above.
    fn o(&self, w: &[u16], x: &[f32], out: usize) -> Vec<f32> {
        let _t = timed(OpClass::KdaOProj);
        self.grouped(codes_as_bytes(w), &[ExpertOffset(0)], x, out)
    }
}
