//! BF16 grouped-expert dispatch — every selected expert in one launch.
//!
//! The bf16 sibling of [`MetalBackend::q6k_grouped_experts`], sharing its
//! [`InputLayout`] / [`ExpertOffset`] / [`GroupedError`] vocabulary so
//! the family has one contract rather than one per codec. Its own file
//! because `grouped_experts.rs` is already 564 lines and this crate caps
//! at 800.
//!
//! Rung 1 measured that the GPU arm was submission-bound, not
//! bandwidth-bound, and that batching nine expert GEMVs into one command
//! buffer won 4.68x on its own. This is the next question: whether the
//! residue is occupancy. See [`crate::shaders::bf16_grouped_experts`]
//! for the argument and the numbers behind it.
//!
//! [`MetalBackend::q6k_grouped_experts`]: super::grouped_experts

use metal::{Buffer, ComputeCommandEncoderRef};

use super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use crate::kernels::KernelHandle;
use crate::MetalBackend;

/// Bytes per bf16 code, in the offset table and in the device buffer.
const BF16_BYTES: usize = 2;

/// The shape one grouped dispatch computes: `[n, k]` per slot, and where
/// each slot's input comes from.
///
/// Bundled rather than passed loose because the tiled entry point also
/// carries a pipeline handle, and six positional numbers at a call site
/// is how `n` and `k` get swapped — a transposed matvec that returns
/// plausible garbage rather than failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedShape {
    /// Output rows per expert.
    pub n: usize,
    /// Reduction length — the shared dimension.
    pub k: usize,
    pub layout: InputLayout,
}

impl MetalBackend {
    /// Run every selected expert's `[n, k]` bf16 matvec in one dispatch.
    ///
    /// `weights` is one bank holding every selected expert's payload;
    /// `offsets[slot]` is where that slot's `[n, k]` matrix starts,
    /// **in bytes**, matching the rest of the grouped family. Slots may
    /// point anywhere in the bank and may repeat — identity lives in the
    /// table, not in a row's position, which is what lets a caller feed
    /// only the resident/selected experts later.
    ///
    /// Returns `[n_selected, n]` row-major: one output vector per slot,
    /// for the caller to combine with its routing weights. That
    /// reduction is `n_selected x n` floats against `n_selected x n x k`
    /// weight bytes read, so leaving it outside costs nothing measurable
    /// and keeps the kernel verifiable.
    ///
    /// Bit-exactness: the reduction body is copied verbatim from
    /// `bf16_gemv`, so results must equal stacking individual
    /// `bf16_gemv_force` calls **exactly**. A tolerance here would let a
    /// numerics change hide inside an occupancy result.
    pub fn bf16_grouped_experts(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<Vec<f32>, GroupedError> {
        self.bf16_grouped_experts_profiled(weights, offsets, x, n, k, layout)
            .map(|(out, _gpu_ms)| out)
    }

    /// The route-dependent offset table, uploaded fresh.
    ///
    /// `uncached_bytes` rather than `get_bytes`: the buffer cache keys on
    /// `(ptr, len)`, which is right for mmap'd weights that live for the
    /// process and wrong for a small table rebuilt every step — a
    /// recycled allocation at the same address would alias a previous
    /// route's table, and the `debug_assert` that would catch it is
    /// compiled out in release.
    pub(crate) fn offset_table(&self, offsets: &[ExpertOffset]) -> Buffer {
        let raw: Vec<u8> = offsets.iter().flat_map(|o| o.0.to_le_bytes()).collect();
        self.bufs.uncached_bytes(&raw)
    }

    /// The same table when the caller's slice is STABLE for the process
    /// — a layer's q|k|v bases, a single-slot o_proj, a residency map.
    ///
    /// Cached on `(ptr, len)` like any weight, because creating a Metal
    /// buffer is not free: three constant tables per layer, rebuilt every
    /// step across nineteen layers, cost ~25 ms a token in the real
    /// trajectory — more than the GPU work they were describing. The
    /// caller promises the slice outlives the process, exactly as
    /// `get_bytes` already requires of mmap'd weights.
    pub(crate) fn stable_offset_table(&self, offsets: &[ExpertOffset]) -> Buffer {
        // SAFETY: `ExpertOffset` is `repr(transparent)` over `u32`, which
        // has no padding and no invalid bit patterns, so any
        // `&[ExpertOffset]` is a valid `&[u8]` of four times the length.
        let raw = unsafe {
            std::slice::from_raw_parts(
                offsets.as_ptr().cast::<u8>(),
                std::mem::size_of_val(offsets),
            )
        };
        self.bufs.get_bytes(raw)
    }

    /// The default tiling's handle. Callers that want to choose one pick
    /// from [`Self::bf16_grouped_variants`] and go through
    /// [`Self::bf16_grouped_experts_tiled`].
    /// The default tiling's handle, for callers outside this crate that
    /// drive [`Self::bf16_grouped_experts_tiled`] directly.
    pub fn default_grouped_handle_pub(&self) -> &KernelHandle {
        self.default_grouped_handle()
    }

    pub(crate) fn default_grouped_handle(&self) -> &KernelHandle {
        &self.bf16_grouped_experts_pipeline
    }

    /// The same dispatch, also returning the **GPU-side** window for the
    /// command buffer (`GPUEndTime - GPUStartTime`, ms).
    ///
    /// Wall time around this call is CPU encode + submission + GPU
    /// execution + readback, and rung 1 measured the fixed part of that
    /// at ~0.2 ms — comparable to the whole dispatch at Kimi's shapes.
    /// A bandwidth number computed from wall time is therefore a claim
    /// about the *stack*, not about the kernel. This exists so an
    /// occupancy question can be answered in kernel-level units, which
    /// is the only unit in which it means anything.
    pub fn bf16_grouped_experts_profiled(
        &self,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        self.bf16_grouped_experts_tiled(
            self.default_grouped_handle(),
            weights,
            offsets,
            x,
            GroupedShape { n, k, layout },
        )
    }

    /// The same dispatch, binding an ALREADY-RESOLVED buffer at a raw
    /// byte offset.
    ///
    /// Exists to calibrate the binding-alignment requirement: it is the
    /// only way to ask the GPU what it does with an offset the
    /// alignment filter would otherwise refuse to produce. Not a
    /// production path — every real caller goes through
    /// `BufferCache::weights`, which never yields a misaligned offset.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_grouped_experts_at(
        &self,
        w: &Buffer,
        w_offset: u64,
        offsets: &[ExpertOffset],
        x: &[f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    ) -> Result<Vec<f32>, GroupedError> {
        let shape = GroupedShape { n, k, layout };
        let buf_o = self.offset_table(offsets);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((offsets.len() * n * 4) as u64);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w,
                w_offset,
                offsets: &buf_o,
                x: &buf_x,
                out: &buf_out,
            },
            offsets.len(),
            shape,
        );
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/bf16_grouped.rs:at",
        );
        Ok(crate::buffers::read_buffer_f32(&buf_out, offsets.len() * n))
    }

    /// The same dispatch at a caller-chosen row tiling.
    ///
    /// `handle` comes from [`Self::bf16_grouped_variants`], so an
    /// unsupported tiling is unrepresentable rather than an error to
    /// handle. Every variant shares one reduction body — pinned in the
    /// shader module — so switching tilings changes the schedule and no
    /// value, which is the only reason a sweep over them measures
    /// anything.
    pub fn bf16_grouped_experts_tiled(
        &self,
        handle: &KernelHandle,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        shape: GroupedShape,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let x_needed = validate_grouped(weights, offsets, x, shape)?;
        let n = shape.n;

        let (buf_w, w_offset) = self.bufs.weights(weights);
        let buf_o = self.offset_table(offsets);
        let buf_x = self.bufs.transient_from_f32(&x[..x_needed]);
        let buf_out = self.bufs.output((offsets.len() * n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_grouped(
            enc,
            handle,
            GroupedBinding {
                w: &buf_w,
                w_offset,
                offsets: &buf_o,
                x: &buf_x,
                out: &buf_out,
            },
            offsets.len(),
            shape,
        );
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/bf16_grouped.rs:dispatch",
        );

        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        Ok((
            crate::buffers::read_buffer_f32(&buf_out, offsets.len() * n),
            gpu_ms,
        ))
    }
}

/// The four device buffers one grouped dispatch binds.
///
/// Bundled so [`encode_grouped`] stays readable, and so a caller that
/// encodes several dispatches into one encoder — the whole-block path —
/// cannot swap two of them at a call site.
pub(crate) struct GroupedBinding<'a> {
    pub w: &'a Buffer,
    /// Byte offset of the bank within `w` — non-zero when the bank was
    /// resolved into a larger registered region. See
    /// `BufferCache::weights`.
    pub w_offset: u64,
    pub offsets: &'a Buffer,
    pub x: &'a Buffer,
    pub out: &'a Buffer,
}

/// Shape and bounds checks shared by every entry point, returning how
/// many input floats the dispatch will read.
///
/// One function rather than a copy per entry point: a check that passed
/// on one path and not another would be an out-of-bounds device read
/// reachable by one caller and not the other.
pub(crate) fn validate_grouped(
    weights: &[u8],
    offsets: &[ExpertOffset],
    x: &[f32],
    shape: GroupedShape,
) -> Result<usize, GroupedError> {
    if offsets.is_empty() {
        return Err(GroupedError::NoExpertsSelected);
    }
    let per_expert = shape.n * shape.k * BF16_BYTES;
    for (slot, off) in offsets.iter().enumerate() {
        if !(off.0 as usize).is_multiple_of(BF16_BYTES) {
            return Err(GroupedError::OffsetNotCodeAligned {
                slot,
                offset: off.0,
            });
        }
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
    let x_needed = match shape.layout {
        InputLayout::Shared => shape.k,
        InputLayout::PerSlot => shape.k * offsets.len(),
    };
    if x.len() < x_needed {
        return Err(GroupedError::OffsetOutOfRange {
            slot: 0,
            offset: 0,
            need: x_needed,
            have: x.len(),
        });
    }
    Ok(x_needed)
}

/// Encode one grouped dispatch into an existing encoder.
///
/// Encoding only — no command buffer, no commit, no wait. That
/// separation is the whole point: dispatches sharing an encoder run in
/// order with implicit barriers between them, so a caller can chain
/// gate → up → activation → down and pay ONE submission instead of
/// three. Rung 2 measured the submission floor at ~0.15-0.19 ms against
/// a ~0.14 ms kernel, so that is the larger of the two costs.
pub(crate) fn encode_grouped(
    enc: &ComputeCommandEncoderRef,
    handle: &KernelHandle,
    b: GroupedBinding<'_>,
    slots: usize,
    shape: GroupedShape,
) {
    encode_grouped_windowed(enc, handle, b, slots, shape, SlotWindow::default());
}

/// Byte offsets into a dispatch's `x` and `out` bindings.
///
/// The kernel indexes both by its grid-y slot, so a dispatch that
/// computes ONE slot of a larger `[slots, n]` output — the shared
/// expert's own dispatch, whose weights live in their own region rather
/// than the routed bank — lands its row by binding `out` (and, for the
/// per-slot input regime, `x`) at that slot's byte base. Zero for every
/// dispatch that computes its whole output.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SlotWindow {
    pub x_bytes: u64,
    pub out_bytes: u64,
}

/// [`encode_grouped`], with the `x`/`out` bindings windowed to a slot
/// base. Kept as one body so a windowed dispatch cannot drift from the
/// plain one in anything but the two offsets.
pub(crate) fn encode_grouped_windowed(
    enc: &ComputeCommandEncoderRef,
    handle: &KernelHandle,
    b: GroupedBinding<'_>,
    slots: usize,
    shape: GroupedShape,
    window: SlotWindow,
) {
    let n_u32 = shape.n as u32;
    let k_u32 = shape.k as u32;
    let x_stride: u32 = match shape.layout {
        InputLayout::Shared => 0,
        InputLayout::PerSlot => shape.k as u32,
    };
    let row_tiles = (shape.n as u64).div_ceil(handle.rows_per_tg);

    enc.set_compute_pipeline_state(&handle.state);
    enc.set_buffer(0, Some(b.w), b.w_offset);
    enc.set_buffer(1, Some(b.offsets), 0);
    enc.set_buffer(2, Some(b.x), window.x_bytes);
    enc.set_buffer(3, Some(b.out), window.out_bytes);
    enc.set_bytes(4, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &x_stride as *const u32 as *const std::ffi::c_void);
    // 2-D grid: row tiles x expert slots. The y-dimension is the
    // parallelism the model was already supplying and the per-expert
    // dispatch was throwing away.
    enc.dispatch_thread_groups(
        metal::MTLSize::new(row_tiles, slots as u64, 1),
        metal::MTLSize::new(handle.threads_per_tg, 1, 1),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::backend::MatMul;

    const SLOTS: usize = 4;
    const N: usize = 24;
    const K: usize = 128;

    fn narrow(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// A bank of `SLOTS` distinct `[N, K]` expert matrices, laid out
    /// back to back, plus the byte offset of each.
    fn bank() -> (Vec<u8>, Vec<ExpertOffset>) {
        let per_expert = N * K;
        let mut bytes = Vec::with_capacity(SLOTS * per_expert * BF16_BYTES);
        for slot in 0..SLOTS {
            for i in 0..per_expert {
                let v = ((i as f32) * 0.013 + (slot as f32) * 1.7).sin() * 0.4;
                bytes.extend_from_slice(&narrow(v).to_le_bytes());
            }
        }
        let offsets = (0..SLOTS)
            .map(|s| ExpertOffset((s * per_expert * BF16_BYTES) as u32))
            .collect();
        (bytes, offsets)
    }

    fn synth_x(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.021 + seed).cos() * 0.5)
            .collect()
    }

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// One slot's payload as a standalone slice, for the per-expert arm.
    fn slice_for(bank: &[u8], off: ExpertOffset) -> &[u8] {
        let start = off.0 as usize;
        &bank[start..start + N * K * BF16_BYTES]
    }

    /// The load-bearing contract: grouping changes WHERE the work is
    /// scheduled, never what it computes. Bit-identical, not close — a
    /// tolerance here would let a numerics change hide inside an
    /// occupancy result.
    #[test]
    fn grouped_matches_separate_dispatches_exactly() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let x = synth_x(K, 0.3);

        let grouped = m
            .bf16_grouped_experts(&bank_bytes, &offsets, &x, N, K, InputLayout::Shared)
            .expect("grouped dispatch");
        let separate: Vec<f32> = offsets
            .iter()
            .flat_map(|&off| {
                m.bf16_gemv_force(slice_for(&bank_bytes, off), &x, N, K)
                    .expect("per-expert dispatch")
            })
            .collect();

        assert_eq!(grouped.len(), SLOTS * N);
        assert_eq!(grouped, separate, "grouping must not change any value");
    }

    /// Identity travels in the offset table, not in a row's position.
    ///
    /// Reversing the table must reverse the output blocks exactly. A
    /// kernel that derived its expert from `row / rows_per_expert` — the
    /// coincidental-contiguity shortcut — would pass the test above and
    /// fail this one, and would then silently mis-serve any caller that
    /// hands it only the resident or selected experts.
    #[test]
    fn the_offset_table_decides_which_expert_a_slot_computes() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let x = synth_x(K, 0.7);

        let forward = m
            .bf16_grouped_experts(&bank_bytes, &offsets, &x, N, K, InputLayout::Shared)
            .expect("forward order");
        let mut reversed_table = offsets.clone();
        reversed_table.reverse();
        let reversed = m
            .bf16_grouped_experts(&bank_bytes, &reversed_table, &x, N, K, InputLayout::Shared)
            .expect("reversed order");

        for slot in 0..SLOTS {
            let mirror = SLOTS - 1 - slot;
            assert_eq!(
                &reversed[slot * N..(slot + 1) * N],
                &forward[mirror * N..(mirror + 1) * N],
                "slot {slot} should hold what slot {mirror} held"
            );
        }
        assert_ne!(forward, reversed, "control: the experts must differ at all");
    }

    /// A slot may name the same expert twice — the routing table is not
    /// guaranteed distinct once a caller starts feeding resident-only
    /// banks.
    #[test]
    fn slots_may_repeat_the_same_expert() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let x = synth_x(K, 1.1);
        let repeated = vec![offsets[2]; 3];

        let got = m
            .bf16_grouped_experts(&bank_bytes, &repeated, &x, N, K, InputLayout::Shared)
            .expect("repeated slots");
        let once = m
            .bf16_gemv_force(slice_for(&bank_bytes, offsets[2]), &x, N, K)
            .expect("single dispatch");
        for slot in 0..3 {
            assert_eq!(&got[slot * N..(slot + 1) * N], &once[..]);
        }
    }

    /// `PerSlot` gives each expert its own input, and must equal the
    /// per-expert dispatches over those same inputs — exactly.
    #[test]
    fn per_slot_inputs_match_per_expert_dispatches() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let xs: Vec<f32> = (0..SLOTS)
            .flat_map(|s| synth_x(K, s as f32 * 0.9))
            .collect();

        let grouped = m
            .bf16_grouped_experts(&bank_bytes, &offsets, &xs, N, K, InputLayout::PerSlot)
            .expect("per-slot dispatch");
        let separate: Vec<f32> = offsets
            .iter()
            .enumerate()
            .flat_map(|(slot, &off)| {
                let x = &xs[slot * K..(slot + 1) * K];
                m.bf16_gemv_force(slice_for(&bank_bytes, off), x, N, K)
                    .expect("per-expert dispatch")
            })
            .collect();
        assert_eq!(grouped, separate);
    }

    /// Control for the test above: `Shared` and `PerSlot` must disagree
    /// when the inputs actually differ. Without it, a kernel that
    /// ignored `XSTRIDE` would pass every parity test that happened to
    /// use one input.
    #[test]
    fn shared_and_per_slot_disagree_when_the_inputs_differ() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let xs: Vec<f32> = (0..SLOTS)
            .flat_map(|s| synth_x(K, s as f32 * 0.9))
            .collect();

        let per_slot = m
            .bf16_grouped_experts(&bank_bytes, &offsets, &xs, N, K, InputLayout::PerSlot)
            .expect("per-slot");
        let shared = m
            .bf16_grouped_experts(&bank_bytes, &offsets, &xs, N, K, InputLayout::Shared)
            .expect("shared reads only the first K");
        assert_ne!(
            per_slot, shared,
            "XSTRIDE must actually select the input vector"
        );
    }

    /// Shape and alignment faults refuse rather than reading out of
    /// bounds or fetching misaligned codes.
    #[test]
    fn shape_and_alignment_faults_are_refused() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let x = synth_x(K, 0.0);

        assert_eq!(
            m.bf16_grouped_experts(&bank_bytes, &[], &x, N, K, InputLayout::Shared),
            Err(GroupedError::NoExpertsSelected)
        );
        let past_end = [ExpertOffset(bank_bytes.len() as u32)];
        assert!(matches!(
            m.bf16_grouped_experts(&bank_bytes, &past_end, &x, N, K, InputLayout::Shared),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
        let odd = [ExpertOffset(1)];
        assert_eq!(
            m.bf16_grouped_experts(&bank_bytes, &odd, &x, N, K, InputLayout::Shared),
            Err(GroupedError::OffsetNotCodeAligned { slot: 0, offset: 1 })
        );
        let short_x = synth_x(K - 1, 0.0);
        assert!(matches!(
            m.bf16_grouped_experts(&bank_bytes, &offsets, &short_x, N, K, InputLayout::Shared),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
        // PerSlot needs SLOTS times as much input as Shared does.
        assert!(matches!(
            m.bf16_grouped_experts(&bank_bytes, &offsets, &x, N, K, InputLayout::PerSlot),
            Err(GroupedError::OffsetOutOfRange { .. })
        ));
    }

    /// **The claim the whole geometry sweep rests on:** every row tiling
    /// computes the same numbers, bit for bit.
    ///
    /// The variants share one reduction body and differ only in how many
    /// rows a threadgroup covers, so a difference between them is
    /// scheduling. If that were false, a sweep would be comparing
    /// kernels rather than schedules and any bandwidth it reported would
    /// be uninterpretable.
    #[test]
    fn every_row_tiling_computes_the_same_values() {
        let m = backend();
        let (bank_bytes, offsets) = bank();
        let x = synth_x(K, 0.5);

        let mut reference: Option<Vec<f32>> = None;
        for handle in &m.bf16_grouped_variants {
            let (got, _gpu) = m
                .bf16_grouped_experts_tiled(
                    handle,
                    &bank_bytes,
                    &offsets,
                    &x,
                    GroupedShape {
                        n: N,
                        k: K,
                        layout: InputLayout::Shared,
                    },
                )
                .expect("tiled dispatch");
            match &reference {
                None => reference = Some(got),
                Some(want) => assert_eq!(
                    &got, want,
                    "tiling r{} disagreed with r{}",
                    handle.rows_per_tg, m.bf16_grouped_variants[0].rows_per_tg
                ),
            }
        }
        assert_eq!(
            m.bf16_grouped_variants.len(),
            crate::shaders::bf16_grouped_experts::ROWS_PER_TG_VARIANTS.len(),
            "every emitted variant must be bound"
        );
    }

    /// The bound variants carry the tilings the shader module declares,
    /// in the same order, and the default is one of them. Order is
    /// load-bearing: a sweep indexes this list.
    #[test]
    fn the_bound_variants_match_the_declared_tilings() {
        let m = backend();
        for (handle, &rows) in m
            .bf16_grouped_variants
            .iter()
            .zip(&crate::shaders::bf16_grouped_experts::ROWS_PER_TG_VARIANTS)
        {
            assert_eq!(handle.rows_per_tg, rows);
            assert_eq!(
                handle.threads_per_tg,
                rows * crate::shaders::bf16_grouped_experts::THREADS_PER_SIMDGROUP
            );
            assert_eq!(
                handle.kernel_name,
                crate::shaders::bf16_grouped_experts::kernel_name(rows)
            );
        }
        assert_eq!(
            m.bf16_grouped_experts_pipeline.kernel_name, m.bf16_grouped_variants[0].kernel_name,
            "the default must be the first variant"
        );
    }

    /// Geometry is read from the bound pipeline, never hardcoded at the
    /// dispatch site, and the grouped kernel tiles rows exactly like the
    /// single-expert one so the occupancy comparison is like-for-like.
    #[test]
    fn the_grouped_pipeline_tiles_rows_like_the_single_expert_one() {
        let m = backend();
        let g = &m.bf16_grouped_experts_pipeline;
        assert_eq!(
            g.kernel_name,
            crate::shaders::bf16_grouped_experts::kernel_name(
                crate::shaders::bf16_grouped_experts::ROWS_PER_TG
            )
        );
        assert_eq!(
            g.rows_per_tg,
            crate::shaders::bf16_grouped_experts::ROWS_PER_TG
        );
        assert_eq!(
            g.threads_per_tg,
            crate::shaders::bf16_grouped_experts::THREADS_PER_TG
        );
        assert_eq!(g.rows_per_tg, m.bf16_gemv_pipeline.rows_per_tg);
        assert_eq!(g.threads_per_tg, m.bf16_gemv_pipeline.threads_per_tg);
    }
}
