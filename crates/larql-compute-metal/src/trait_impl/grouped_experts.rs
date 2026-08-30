//! Grouped-expert dispatch (K3a) — every selected expert in one Metal launch.
//!
//! The measured problem: one K3 expert branch `[3584, 3072]` reaches only 0.64
//! of roofline through `q6k_matvec`, against 0.87-0.90 at K3's large shapes. At
//! `ROWS_PER_TG = 4` that matrix launches 896 threadgroups — too little
//! independent work to hide memory latency. A token selects 16 experts per
//! layer, so dispatching them together supplies 14,336.
//!
//! The API takes an offset table rather than a concatenated buffer so the
//! artifact is untouched, and so K3b can extend it to ragged per-expert position
//! lists without changing the signature.

use crate::MetalBackend;

/// How the input vectors are laid out across expert slots.
///
/// Not a convenience flag: the engine's two expert projections genuinely differ.
/// `gate/up` shares one hidden state across every selected expert, while `down`
/// consumes each expert's own intermediate activation. Choosing wrong produces a
/// plausible number from the wrong expert's input rather than an error, so the
/// caller must state which it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputLayout {
    /// One `[k]` vector, read by every slot — the gate/up regime.
    Shared,
    /// `[n_selected, k]`, each slot reading its own row — the down regime.
    PerSlot,
}

/// Where one selected expert's payload begins, in bytes.
///
/// `repr(transparent)` so a `&[ExpertOffset]` can be handed to the
/// device as bytes without a copy — a table that is CONSTANT for a layer
/// (the q|k|v bases, a single-slot o_proj) should be uploaded once and
/// cached, not rebuilt every step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ExpertOffset(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupedError {
    /// `k` is not a whole number of Q6_K super-blocks.
    KNotSuperblockAligned {
        k: usize,
    },
    /// An offset plus one expert's payload runs past the buffer.
    OffsetOutOfRange {
        slot: usize,
        offset: u32,
        need: usize,
        have: usize,
    },
    NoExpertsSelected,
    /// A bf16 slot offset is odd. The table is in bytes for every codec
    /// in this family, but bf16 payloads bind as `ushort`, so an odd
    /// offset would read misaligned codes — silent garbage rather than a
    /// fault. Refused instead.
    OffsetNotCodeAligned {
        slot: usize,
        offset: u32,
    },
    /// A multi-projection block was handed offset tables of different
    /// lengths. Not a fault at dispatch time — the shorter projections
    /// would compute fewer slots and the block would return a mix of
    /// this token's experts and whatever the pooled output last held.
    SlotCountMismatch {
        expected: usize,
        found: usize,
    },
    /// A layer in a chain routed to an expert its bank does not hold.
    ///
    /// Its OWN variant, not an `OffsetOutOfRange` with `slot` reused as a
    /// layer index: `slot` already means the projection a host-side
    /// bounds check failed on, and a caller that read one as the other
    /// would blame the wrong thing. A control did exactly that.
    LayerRouteNotResident {
        layer: usize,
        refusals: u32,
    },
    /// The model head's shapes disagree with the stack it is attached
    /// to. Its own variant for the same reason as the one above: the
    /// head is not a slot, and reporting it as one would send a reader
    /// looking at the expert tables.
    HeadShapeMismatch {
        vocab: usize,
        hidden: usize,
        have_bytes: usize,
    },
    /// The three projections of one MoE layer disagree about whether a
    /// shared expert exists. `Some` for gate but `None` for down would
    /// run a three-projection MLP with one projection's contribution
    /// missing — plausible numbers from half an expert, which is worse
    /// than any refusal.
    SharedBranchInconsistent,
}

impl std::fmt::Display for GroupedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KNotSuperblockAligned { k } => {
                write!(f, "grouped experts: K={k} is not a multiple of 256")
            }
            Self::OffsetOutOfRange { slot, offset, need, have } => write!(
                f,
                "grouped experts: slot {slot} at offset {offset} needs {need} bytes, buffer has {have}"
            ),
            Self::NoExpertsSelected => write!(f, "grouped experts: empty selection"),
            Self::OffsetNotCodeAligned { slot, offset } => write!(
                f,
                "grouped experts: slot {slot} byte offset {offset} is odd; bf16 payloads \
                 must start on a 2-byte boundary"
            ),
            Self::HeadShapeMismatch { vocab, hidden, have_bytes } => write!(
                f,
                "model head: a {vocab}x{hidden} bf16 projection needs {} bytes, got {have_bytes}",
                vocab * hidden * 2
            ),
            Self::LayerRouteNotResident { layer, refusals } => write!(
                f,
                "grouped experts: layer {layer} routed to {refusals} expert(s) its bank \
                 does not hold"
            ),
            Self::SlotCountMismatch { expected, found } => write!(
                f,
                "grouped experts: every projection of a block must select the same \
                 slots; expected {expected}, found {found}"
            ),
            Self::SharedBranchInconsistent => write!(
                f,
                "grouped experts: gate/up/down disagree about whether a shared expert \
                 exists; the branch is declared per projection but is one semantic fact"
            ),
        }
    }
}

impl std::error::Error for GroupedError {}

/// Q6_K super-block, in weights and in bytes.
pub const Q6K_SUPERBLOCK_ELEMS: usize = 256;
pub const Q6K_SUPERBLOCK_BYTES: usize = 210;
pub const Q4K_SUPERBLOCK_BYTES: usize = 144;

impl MetalBackend {
    /// Run every selected expert's `[n, k]` matvec against a shared input, in
    /// one dispatch.
    ///
    /// Returns `[n_selected, n]` row-major: one output vector per expert, for
    /// the caller to combine with its routing weights. That reduction is
    /// `16 x n` floats against `16 x n x k` weight bytes read, so leaving it
    /// outside costs nothing measurable and keeps this kernel verifiable.
    ///
    /// Bit-exactness: the reduction body is copied verbatim from `q6k_matvec`,
    /// so results must equal stacking individual `q6k_matvec` calls **exactly**.
    /// A tolerance here would let a numerics change hide inside an occupancy
    /// result.
    pub fn q6k_grouped_experts(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<Vec<f32>, GroupedError> {
        self.q6k_grouped_experts_profiled(weights, offsets, x, n, k, layout)
            .map(|(out, _)| out)
    }

    /// Q6_K grouped experts, with the GPU window.
    pub fn q6k_grouped_experts_profiled(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        self.kquant_grouped_experts(
            &self.quant.q6k_grouped_experts_pipeline,
            Q6K_SUPERBLOCK_BYTES,
            weights,
            offsets,
            x,
            n,
            k,
            layout,
        )
    }

    /// Q4_K sibling of [`Self::q6k_grouped_experts`] — what the engine's MoE
    /// down projection needs, since that path stores experts in Q4_K.
    ///
    /// Same contract: bit-exact against stacking per-expert `q4k_matvec` calls,
    /// offset table rather than a concatenated artifact, explicit input layout.
    pub fn q4k_grouped_experts(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<Vec<f32>, GroupedError> {
        self.q4k_grouped_experts_profiled(weights, offsets, x, n, k, layout)
            .map(|(out, _)| out)
    }

    /// Q4_K grouped experts, with the GPU window.
    pub fn q4k_grouped_experts_profiled(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        self.kquant_grouped_experts(
            &self.quant.q4k_grouped_experts_pipeline,
            Q4K_SUPERBLOCK_BYTES,
            weights,
            offsets,
            x,
            n,
            k,
            layout,
        )
    }

    /// The body both k-quant grouped entries share.
    ///
    /// One function rather than two near-copies: the formats differ only
    /// in their pipeline and their superblock size, and the pair had
    /// already drifted apart once — a `get_bytes` aliasing fix applied to
    /// one would have left the other silently wrong.
    #[allow(clippy::too_many_arguments)]
    fn kquant_grouped_experts(
        &self,
        handle: &crate::kernels::KernelHandle,
        superblock_bytes: usize,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        if offsets.is_empty() {
            return Err(GroupedError::NoExpertsSelected);
        }
        if !k.is_multiple_of(Q6K_SUPERBLOCK_ELEMS) {
            return Err(GroupedError::KNotSuperblockAligned { k });
        }
        let per_expert = n * (k / Q6K_SUPERBLOCK_ELEMS) * superblock_bytes;
        for (slot, off) in offsets.iter().enumerate() {
            let need = off.0 as usize + per_expert;
            if need > weights.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot,
                    offset: off.0,
                    need,
                    have: weights.len(),
                });
            }
        }
        let x_needed = match layout {
            InputLayout::Shared => k,
            InputLayout::PerSlot => k * offsets.len(),
        };
        if x.len() < x_needed {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: x_needed,
                have: x.len(),
            });
        }
        let x_stride: u32 = match layout {
            InputLayout::Shared => 0,
            InputLayout::PerSlot => k as u32,
        };

        let raw: Vec<u32> = offsets.iter().map(|o| o.0).collect();
        let buf_w = self.bufs.get_bytes(weights);
        // `uncached_bytes`, NOT `get_bytes`: `raw` is a temporary rebuilt
        // on every call, and the cache keys on `(ptr, len)`. Two calls
        // with different tables of the same length — a Q6_K bank's
        // offsets and a Q4_K bank's, say — reliably land at the same
        // address, and the second silently dispatches against the
        // first's table. The `debug_assert` that catches it is compiled
        // out in release, so this failed intermittently in 2 runs of 5
        // with no error, just a wrong answer.
        let buf_o = self.bufs.uncached_bytes(bytemuck_cast(&raw));
        let buf_x = self.bufs.transient_from_f32(&x[..x_needed]);
        let buf_out = self.bufs.output((offsets.len() * n * 4) as u64);
        let (n_u32, k_u32) = (n as u32, k as u32);
        let row_tiles = (n as u64).div_ceil(handle.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&handle.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_o), 0);
        enc.set_buffer(2, Some(&buf_x), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &x_stride as *const u32 as *const std::ffi::c_void);
        // 2-D grid: row tiles x expert slots. This is the whole point — the
        // y-dimension is the parallelism the model was already supplying.
        enc.dispatch_thread_groups(
            metal::MTLSize::new(row_tiles, offsets.len() as u64, 1),
            metal::MTLSize::new(handle.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/grouped_experts.rs:kquant_grouped",
        );
        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        Ok((
            crate::buffers::read_buffer_f32(&buf_out, offsets.len() * n),
            gpu_ms,
        ))
    }

    /// Threadgroups this dispatch launches, against what one expert would.
    ///
    /// Exposed so a bench can state the occupancy claim numerically instead of
    /// asserting it in prose.
    pub fn grouped_threadgroups(&self, n: usize, n_selected: usize) -> (u64, u64) {
        let rows = self.quant.q6k_grouped_experts_pipeline.rows_per_tg;
        let tiles = (n as u64).div_ceil(rows);
        (tiles, tiles * n_selected as u64)
    }
}

fn bytemuck_cast(v: &[u32]) -> &[u8] {
    // Safety: u32 has no padding and any bit pattern is a valid u8 sequence;
    // length is scaled accordingly. Metal reads this back as `device const uint*`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use larql_compute::QuantMatVec;

    /// K3's expert branch, shrunk in the row dimension to keep the test quick.
    const N: usize = 260; // deliberately not a multiple of ROWS_PER_TG
    const K: usize = 512;
    const SELECTED: usize = 16;

    fn synth(n: usize, k: usize, seed: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| ((((i * 31 + seed * 17) % 251) as f32) - 125.0) * 0.001)
            .collect()
    }

    fn bank() -> (Vec<u8>, Vec<ExpertOffset>, usize) {
        let mut buf = Vec::new();
        let mut offs = Vec::new();
        let mut per = 0usize;
        for e in 0..SELECTED {
            let q = quantize_q6_k(&synth(N, K, e + 1));
            per = q.len();
            offs.push(ExpertOffset(buf.len() as u32));
            buf.extend_from_slice(&q);
        }
        (buf, offs, per)
    }

    #[test]
    fn matches_sixteen_separate_dispatches_exactly() {
        // Bit-exact, not within tolerance: the reduction body is copied
        // verbatim, so any difference means the addressing is wrong.
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, per) = bank();
        let x: Vec<f32> = (0..K).map(|i| ((i % 251) as f32 - 125.0) * 0.01).collect();

        let grouped = be
            .q6k_grouped_experts(&buf, &offs, &x, N, K, InputLayout::Shared)
            .expect("dispatch");

        for (slot, off) in offs.iter().enumerate() {
            let lo = off.0 as usize;
            let solo = be
                .q6k_matvec(&buf[lo..lo + per], &x, N, K)
                .expect("single-expert reference");
            let got = &grouped[slot * N..(slot + 1) * N];
            assert_eq!(got, &solo[..], "slot {slot} diverged from its own dispatch");
        }
    }

    #[test]
    fn the_grid_carries_the_expert_dimension() {
        // The occupancy claim, made numeric: 16 experts must multiply the
        // threadgroup count, not merely reorder the same work.
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (single, grouped) = be.grouped_threadgroups(3584, 16);
        assert_eq!(single, 896, "one expert at ROWS_PER_TG=4");
        assert_eq!(grouped, 14_336, "16 experts dispatched together");
    }

    #[test]
    fn a_single_selected_expert_still_works() {
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, per) = bank();
        let x: Vec<f32> = (0..K).map(|i| ((i % 97) as f32 - 48.0) * 0.02).collect();
        let one = be
            .q6k_grouped_experts(&buf, &offs[..1], &x, N, K, InputLayout::Shared)
            .unwrap();
        let solo = be.q6k_matvec(&buf[..per], &x, N, K).unwrap();
        assert_eq!(one, solo);
    }

    #[test]
    fn slots_may_repeat_the_same_expert() {
        // Real routing can send several positions to one expert; the offset
        // table must tolerate duplicates rather than assume distinctness.
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, _) = bank();
        let x: Vec<f32> = (0..K).map(|i| ((i % 31) as f32 - 15.0) * 0.05).collect();
        let dup = vec![offs[3], offs[3], offs[7]];
        let got = be
            .q6k_grouped_experts(&buf, &dup, &x, N, K, InputLayout::Shared)
            .unwrap();
        assert_eq!(
            &got[..N],
            &got[N..2 * N],
            "same expert must give same output"
        );
    }

    #[test]
    fn per_slot_inputs_match_per_expert_dispatches() {
        // The regime the engine's DOWN projection actually needs: each expert
        // consumes its own intermediate activation, not a shared hidden state.
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, per) = bank();
        let xs: Vec<f32> = (0..K * SELECTED)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.01)
            .collect();

        let grouped = be
            .q6k_grouped_experts(&buf, &offs, &xs, N, K, InputLayout::PerSlot)
            .expect("dispatch");

        for (slot, off) in offs.iter().enumerate() {
            let lo = off.0 as usize;
            let solo = be
                .q6k_matvec(&buf[lo..lo + per], &xs[slot * K..(slot + 1) * K], N, K)
                .expect("reference");
            assert_eq!(
                &grouped[slot * N..(slot + 1) * N],
                &solo[..],
                "slot {slot} read the wrong input row"
            );
        }
    }

    #[test]
    fn shared_and_per_slot_disagree_when_inputs_differ() {
        // Guards the silent-wrong-answer failure: if the stride were ignored,
        // these two would agree and the bug would be invisible.
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, _) = bank();
        let xs: Vec<f32> = (0..K * SELECTED)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.02)
            .collect();
        let shared = be
            .q6k_grouped_experts(&buf, &offs, &xs, N, K, InputLayout::Shared)
            .unwrap();
        let per_slot = be
            .q6k_grouped_experts(&buf, &offs, &xs, N, K, InputLayout::PerSlot)
            .unwrap();
        assert_eq!(
            &shared[..N],
            &per_slot[..N],
            "slot 0 reads x[0..K] either way"
        );
        assert_ne!(
            &shared[N..2 * N],
            &per_slot[N..2 * N],
            "slot 1 must read a different row under PerSlot"
        );
    }

    #[test]
    fn q4k_grouped_matches_per_expert_dispatches_with_per_slot_inputs() {
        // The exact contract the Gemma MoE integration depends on: Q4_K
        // weights, one dispatch, each slot consuming its own activation.
        use larql_compute::cpu::ops::q4_common::quantize_q4_k;
        let Some(be) = MetalBackend::new() else {
            return;
        };

        let mut bank = Vec::new();
        let mut offs = Vec::new();
        let mut per = 0usize;
        for e in 0..SELECTED {
            let q = quantize_q4_k(&synth(N, K, e + 1));
            per = q.len();
            offs.push(ExpertOffset(bank.len() as u32));
            bank.extend_from_slice(&q);
        }
        let xs: Vec<f32> = (0..K * SELECTED)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.01)
            .collect();

        let grouped = be
            .q4k_grouped_experts(&bank, &offs, &xs, N, K, InputLayout::PerSlot)
            .expect("dispatch");

        for (slot, off) in offs.iter().enumerate() {
            let lo = off.0 as usize;
            let solo = be
                .q4k_matvec(&bank[lo..lo + per], &xs[slot * K..(slot + 1) * K], N, K)
                .expect("reference");
            assert_eq!(
                &grouped[slot * N..(slot + 1) * N],
                &solo[..],
                "Q4_K slot {slot} diverged"
            );
        }
    }

    #[test]
    fn rejects_an_offset_that_would_read_past_the_buffer() {
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let (buf, offs, _) = bank();
        let x = vec![0.1f32; K];
        let bad = vec![ExpertOffset(buf.len() as u32 - 16)];
        assert!(matches!(
            be.q6k_grouped_experts(&buf, &bad, &x, N, K, InputLayout::Shared),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
        assert!(matches!(
            be.q6k_grouped_experts(&buf, &offs, &x, N, K + 1, InputLayout::Shared),
            Err(GroupedError::KNotSuperblockAligned { .. })
        ));
        assert!(matches!(
            be.q6k_grouped_experts(&buf, &[], &x, N, K, InputLayout::Shared),
            Err(GroupedError::NoExpertsSelected)
        ));
    }

    #[test]
    fn a_per_slot_input_shorter_than_the_slot_count_is_refused() {
        // PerSlot indexes X at `slot * K`, so an x sized for Shared would read
        // another slot's activation off the end rather than fail — the exact
        // silent-wrong-answer this guard exists to prevent.
        let (buf, offs, _) = bank();
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let shared_sized = vec![0.1f32; K];
        assert!(matches!(
            be.q6k_grouped_experts(&buf, &offs, &shared_sized, N, K, InputLayout::PerSlot),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
        // The same buffer is perfectly valid under Shared.
        assert!(be
            .q6k_grouped_experts(&buf, &offs, &shared_sized, N, K, InputLayout::Shared)
            .is_ok());
    }

    #[test]
    fn the_q4k_path_rejects_the_same_shape_faults_as_q6k() {
        // q4k_grouped_experts carries its own guard prologue; a fault caught on
        // the q6k path proves nothing about this one.
        let x = vec![0.1f32; K];
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let q4k_per_expert = N * (K / 256) * 144;
        let buf = vec![0u8; q4k_per_expert * SELECTED];
        let offs: Vec<ExpertOffset> = (0..SELECTED)
            .map(|e| ExpertOffset((e * q4k_per_expert) as u32))
            .collect();

        assert!(matches!(
            be.q4k_grouped_experts(&buf, &[], &x, N, K, InputLayout::Shared),
            Err(GroupedError::NoExpertsSelected)
        ));
        assert!(matches!(
            be.q4k_grouped_experts(&buf, &offs, &x, N, K + 1, InputLayout::Shared),
            Err(GroupedError::KNotSuperblockAligned { .. })
        ));
        let past_end = vec![ExpertOffset(buf.len() as u32)];
        assert!(matches!(
            be.q4k_grouped_experts(&buf, &past_end, &x, N, K, InputLayout::Shared),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
        assert!(matches!(
            be.q4k_grouped_experts(&buf, &offs, &x, N, K, InputLayout::PerSlot),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn every_error_variant_renders_its_own_diagnostic() {
        let k_err = GroupedError::KNotSuperblockAligned { k: 513 }.to_string();
        assert!(k_err.contains("513"), "{k_err}");

        let off = GroupedError::OffsetOutOfRange {
            slot: 3,
            offset: 4096,
            need: 8192,
            have: 5000,
        }
        .to_string();
        for expected in ["3", "4096", "8192", "5000"] {
            assert!(off.contains(expected), "{off} missing {expected}");
        }

        assert!(GroupedError::NoExpertsSelected
            .to_string()
            .contains("empty selection"));

        let route = GroupedError::LayerRouteNotResident {
            layer: 9,
            refusals: 2,
        }
        .to_string();
        for expected in ["9", "2", "does not hold"] {
            assert!(route.contains(expected), "{route} missing {expected}");
        }

        let odd = GroupedError::OffsetNotCodeAligned {
            slot: 5,
            offset: 4097,
        }
        .to_string();
        for expected in ["5", "4097", "2-byte"] {
            assert!(odd.contains(expected), "{odd} missing {expected}");
        }

        let boxed: Box<dyn std::error::Error> =
            Box::new(GroupedError::KNotSuperblockAligned { k: 7 });
        assert!(boxed.to_string().contains('7'));
    }
}
