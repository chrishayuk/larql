//! NVFP4 GEMV dispatch for the lowering: kernel selection (the A-5
//! arms and the `x2` default), the fused/segmented forms (A-5b), and the
//! rung-2 glue helpers (residual fold, multi-output norm, pre-norm
//! arms). Split from `mod.rs` so the encoder-level primitives file stays
//! under the size cap; every item is re-exported from `lowering`.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::{set_f32, set_u32, LoweredMatrix, MatvecOperands};
use crate::MetalBackend;

/// Segments the fused NVFP4 GEMV takes at most.
pub const NVFP4_MAX_SEGMENTS: usize = 3;

/// One matrix of a fused (segmented) NVFP4 GEMV: its packed codes and
/// scales, its tensor scale, its row count and where its rows land.
///
/// `packed_offset`/`scales_offset` are byte offsets into their buffers,
/// so segments may be slices of ONE shared allocation (the QKV packing
/// rung): the kernel then streams one contiguous address range, as the
/// flat single-matrix dispatch does. A packed offset must fall on a row
/// boundary, which for NVFP4 is a multiple of 16 bytes — Metal's bind
/// alignment for the `uint2` loads the x2 body performs.
#[derive(Clone, Copy)]
pub struct Nvfp4Segment<'a> {
    pub packed: &'a Buffer,
    pub packed_offset: u64,
    pub scales: &'a Buffer,
    pub scales_offset: u64,
    pub tensor_scale: f32,
    pub out: &'a Buffer,
    pub out_offset: u64,
    pub n: usize,
}

/// Operator control for A-5b fusion: `LARQL_NVFP4_FUSE=0` encodes Q, K,
/// V, gate and up as separate dispatches and keeps residual adds as
/// their own kernel (the α control arm); `=seg` fuses the projections
/// only (isolates rung 2a); unset/other = everything. Read once.
pub const NVFP4_FUSE_ENV: &str = "LARQL_NVFP4_FUSE";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FusionLevel {
    None,
    Segments,
    All,
}

fn fusion_level() -> FusionLevel {
    static LEVEL: std::sync::OnceLock<FusionLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var(NVFP4_FUSE_ENV).as_deref() {
        Ok("0") => FusionLevel::None,
        Ok("seg") => FusionLevel::Segments,
        _ => FusionLevel::All,
    })
}

/// Whether NVFP4 projections sharing an input are fused into one dispatch.
pub fn nvfp4_fusion_enabled() -> bool {
    fusion_level() != FusionLevel::None
}

/// Whether the rung-2 glue fusions are on: residual adds folded into
/// NVFP4 GEMV writes (2a) and same-input norms sharing one reduction
/// (2c). `LARQL_NVFP4_FUSE=seg` turns these off while keeping 2b's
/// projection fusion — the control that isolates them.
pub fn nvfp4_residual_fusion_enabled() -> bool {
    fusion_level() == FusionLevel::All
}

/// The segment a matrix contributes to a fused NVFP4 GEMV, if it is
/// NVFP4-resident and fusion is on; `None` otherwise.
pub fn nvfp4_segment<'a>(
    m: &LoweredMatrix<'a>,
    out: &'a Buffer,
    out_offset: u64,
    n: usize,
) -> Option<Nvfp4Segment<'a>> {
    if !nvfp4_fusion_enabled() {
        return None;
    }
    match m {
        LoweredMatrix::Nvfp4 {
            packed,
            packed_offset,
            scales,
            scales_offset,
            tensor_scale,
        } => Some(Nvfp4Segment {
            packed,
            packed_offset: *packed_offset,
            scales,
            scales_offset: *scales_offset,
            tensor_scale: *tensor_scale,
            out,
            out_offset,
            n,
        }),
        _ => None,
    }
}

/// Which NVFP4 GEMV kernel the lowering dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Kernel {
    /// The original: one row per lane, scalar loads + nibble LUT decode,
    /// 4 rows/TG. Retained as the A-5 control.
    V1,
    /// Vector loads + arithmetic decode; same values to fp32 rounding,
    /// measured no faster (see `shaders/nvfp4_matvec.rs`).
    V2,
    /// A-5 sweep arms: v1's inner loop at `G` groups per lane per step
    /// and `R` rows per threadgroup.
    G2R4,
    G4R4,
    G1R2,
    G1R8,
    G2R2,
    G2R8,
    /// A-5a arms: `RL` rows per lane sharing one X load (4 simdgroups
    /// per TG), optionally with the byte→`float2` LUT (`…B`).
    X2,
    X4,
    X1B,
    X2B,
    X4B,
}

impl Nvfp4Kernel {
    /// Every arm, for sweeps.
    pub const ALL: [Nvfp4Kernel; 13] = [
        Nvfp4Kernel::V1,
        Nvfp4Kernel::V2,
        Nvfp4Kernel::G2R4,
        Nvfp4Kernel::G4R4,
        Nvfp4Kernel::G1R2,
        Nvfp4Kernel::G1R8,
        Nvfp4Kernel::G2R2,
        Nvfp4Kernel::G2R8,
        Nvfp4Kernel::X2,
        Nvfp4Kernel::X4,
        Nvfp4Kernel::X1B,
        Nvfp4Kernel::X2B,
        Nvfp4Kernel::X4B,
    ];

    /// The `LARQL_NVFP4_KERNEL` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Nvfp4Kernel::V1 => "v1",
            Nvfp4Kernel::V2 => "v2",
            Nvfp4Kernel::G2R4 => "g2r4",
            Nvfp4Kernel::G4R4 => "g4r4",
            Nvfp4Kernel::G1R2 => "g1r2",
            Nvfp4Kernel::G1R8 => "g1r8",
            Nvfp4Kernel::G2R2 => "g2r2",
            Nvfp4Kernel::G2R8 => "g2r8",
            Nvfp4Kernel::X2 => "x2",
            Nvfp4Kernel::X4 => "x4",
            Nvfp4Kernel::X1B => "x1b",
            Nvfp4Kernel::X2B => "x2b",
            Nvfp4Kernel::X4B => "x4b",
        }
    }

    /// Parse the `LARQL_NVFP4_KERNEL` spelling.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// Operator override for the NVFP4 GEMV kernel (`Nvfp4Kernel::name`
/// spellings). Read once.
pub const NVFP4_KERNEL_ENV: &str = "LARQL_NVFP4_KERNEL";

/// The kernel in effect: `LARQL_NVFP4_KERNEL` if set, else the default.
pub fn nvfp4_kernel_choice() -> Nvfp4Kernel {
    static CHOICE: std::sync::OnceLock<Nvfp4Kernel> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        std::env::var(NVFP4_KERNEL_ENV)
            .ok()
            .and_then(|s| Nvfp4Kernel::parse(&s))
            .unwrap_or(NVFP4_KERNEL_DEFAULT)
    })
}

/// Default NVFP4 GEMV kernel: `x2` — two rows per lane sharing one X
/// load. A-5a, 2026-08-19: the α/B fit moved from v1's 9.9 µs + bytes/332
/// GB/s to 5.7 µs + bytes/373 (f16 control 387), bit-identical to v1; on
/// the ledger, same ids, Granite 3B 10.9 → 9.6 ms/token, Glimmer 58.5 →
/// 50.1, Gemma 4 15.1 → 14.0, gpt-oss 10.2 → 9.8. v1 stays the control.
const NVFP4_KERNEL_DEFAULT: Nvfp4Kernel = Nvfp4Kernel::X2;

/// One output of a multi-output RMS norm: its weight, the weight offset
/// (1.0 centred / 0.0 uncentred) and where it lands.
#[derive(Clone, Copy)]
pub struct NormOutput<'a> {
    pub weight: &'a Buffer,
    pub offset: f32,
    pub out: &'a Buffer,
}

/// An RMS norm folded into a consumer GEMV's prologue (rung 2d).
#[derive(Clone, Copy)]
pub struct PreNorm<'a> {
    pub weight: &'a Buffer,
    pub eps: f32,
    pub offset: f32,
}

/// Outputs one multi-norm dispatch carries at most.
pub const RMS_NORM_MAX_OUTPUTS: usize = 3;

impl MetalBackend {
    /// Encode `out = W · x` for an NVFP4 matrix into `enc`.
    ///
    /// Every operand is already a device buffer, so nothing crosses to the
    /// host and the caller may chain this with other encodes in one
    /// command buffer. Dispatches are independent unless they share a
    /// buffer; Metal's default `MTLDispatchTypeSerial` encoder orders
    /// them, so a chain that feeds `out` into the next call's `x` is
    /// correctly sequenced without explicit barriers.
    ///
    /// Geometry is the caller's responsibility — this is a lowering
    /// primitive, and validating `k % 16` on every encode would put a
    /// branch in the hot path for an invariant the plan already fixed at
    /// load time.
    pub fn encode_nvfp4_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
    ) {
        self.encode_nvfp4_kernel(nvfp4_kernel_choice(), enc, op, tensor_scale);
    }

    /// Encode with a named NVFP4 kernel arm.
    pub fn encode_nvfp4_kernel(
        &self,
        which: Nvfp4Kernel,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
    ) {
        let q = &self.quant;
        let kernel = match which {
            Nvfp4Kernel::V1 => &q.nvfp4_matvec_pipeline,
            Nvfp4Kernel::V2 => &q.nvfp4_matvec_v2_pipeline,
            Nvfp4Kernel::G2R4 => &q.nvfp4_sweep_pipelines[0],
            Nvfp4Kernel::G4R4 => &q.nvfp4_sweep_pipelines[1],
            Nvfp4Kernel::G1R2 => &q.nvfp4_sweep_pipelines[2],
            Nvfp4Kernel::G1R8 => &q.nvfp4_sweep_pipelines[3],
            Nvfp4Kernel::G2R2 => &q.nvfp4_sweep_pipelines[4],
            Nvfp4Kernel::G2R8 => &q.nvfp4_sweep_pipelines[5],
            Nvfp4Kernel::X2 => &q.nvfp4_sweep_pipelines[6],
            Nvfp4Kernel::X4 => &q.nvfp4_sweep_pipelines[7],
            Nvfp4Kernel::X1B => &q.nvfp4_sweep_pipelines[8],
            Nvfp4Kernel::X2B => &q.nvfp4_sweep_pipelines[9],
            Nvfp4Kernel::X4B => &q.nvfp4_sweep_pipelines[10],
        };
        self.encode_nvfp4_with(kernel, enc, op, tensor_scale);
    }

    /// A-5b: up to three NVFP4 matrices against one `x` in ONE dispatch —
    /// Q+K+V or gate+up — paying the per-dispatch α once. Every segment
    /// must share `k`; a segment's rows land in its own `out` (+offset).
    /// Bit-identical to `x2` per row. Panics on a `k` mismatch: the plan
    /// fixed the geometry, and a silent fallback would hide a defect.
    pub fn encode_nvfp4_matvec_segments(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        k: usize,
        segments: &[Nvfp4Segment<'_>],
    ) {
        self.encode_nvfp4_matvec_segments_residual(enc, x, k, segments, None);
    }

    /// As [`Self::encode_nvfp4_matvec_segments`], with segment 0's rows
    /// written as `residual[row] + acc` — the residual add folded into
    /// the GEMV (A-5b rung 2a). `residual` is read at `segments[0].out`'s
    /// row indexing, offset 0.
    pub fn encode_nvfp4_matvec_segments_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        k: usize,
        segments: &[Nvfp4Segment<'_>],
        residual: Option<&Buffer>,
    ) {
        assert!(
            !segments.is_empty() && segments.len() <= NVFP4_MAX_SEGMENTS,
            "1..=3 segments"
        );
        // seg3t: the grid is tiled so no threadgroup straddles a segment
        // (prefix sums of each segment's tile count) and the segment is
        // resolved once per threadgroup. NOTE the probe's verdict
        // (`examples/qkv_seg3_probe.rs`): this did NOT close the fused
        // form's ~5 µs deficit against a flat single-matrix dispatch —
        // the per-row resolve hypothesis is falsified; what remains is
        // attributed to streaming from three base addresses instead of
        // one. The follow-up rung is loader-level: pack Q/K/V into ONE
        // allocation so the fused dispatch is literally the flat kernel.
        let kernel = &self.quant.nvfp4_matvec_x2_seg3t_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(2, Some(x), 0);
        set_u32(enc, 5, k as u32);
        enc.set_buffer(17, Some(residual.unwrap_or(x)), 0);
        set_u32(enc, 18, residual.is_some() as u32);
        // MSL `uint3` is 16 bytes (non-packed) — bind four words.
        let mut tile_end = [0u32; 4];
        let mut acc = 0u32;
        for (i, end) in tile_end.iter_mut().enumerate().take(NVFP4_MAX_SEGMENTS) {
            acc += segments
                .get(i)
                .map_or(0, |s| (s.n as u32).div_ceil(kernel.rows_per_tg as u32));
            *end = acc;
        }
        enc.set_bytes(19, 16, tile_end.as_ptr() as *const std::ffi::c_void);
        // (packed, scales, out, M, tensor_scale) slots per segment.
        const SLOTS: [(u64, u64, u64, u64, u64); NVFP4_MAX_SEGMENTS] =
            [(0, 1, 3, 4, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15, 16)];
        let mut total_rows = 0u64;
        for (i, slots) in SLOTS.iter().enumerate() {
            let (wp, ws, out, m, ts) = *slots;
            match segments.get(i) {
                Some(seg) => {
                    enc.set_buffer(wp, Some(seg.packed), seg.packed_offset);
                    enc.set_buffer(ws, Some(seg.scales), seg.scales_offset);
                    enc.set_buffer(out, Some(seg.out), seg.out_offset);
                    set_u32(enc, m, seg.n as u32);
                    set_f32(enc, ts, seg.tensor_scale);
                    total_rows += seg.n as u64;
                }
                None => {
                    // Absent segment: M = 0, buffers aliased to the first
                    // so every slot is bound.
                    let first = &segments[0];
                    enc.set_buffer(wp, Some(first.packed), first.packed_offset);
                    enc.set_buffer(ws, Some(first.scales), first.scales_offset);
                    enc.set_buffer(out, Some(first.out), first.out_offset);
                    set_u32(enc, m, 0);
                    set_f32(enc, ts, 0.0);
                }
            }
        }
        let _ = total_rows;
        enc.dispatch_thread_groups(
            metal::MTLSize::new(tile_end[2] as u64, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// One NVFP4 matrix against `op.x`, written as `residual[row] + acc`
    /// (A-5b rung 2a): the residual add folded into the GEMV, the same
    /// fp32 add as the residual kernel — bit-identical to x2-then-add,
    /// one dispatch fewer. `residual` is indexed like `op.out`, offset 0.
    pub fn encode_nvfp4_matvec_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        residual: &Buffer,
    ) {
        let kernel = &self.quant.nvfp4_matvec_x2r_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        set_f32(enc, 6, tensor_scale);
        enc.set_buffer(7, Some(residual), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// VERIFY-N: `out[r] = W · x[r]` for the activation rows of
    /// multi-RHS arm `arm` (an index into
    /// [`MATMUL_ARMS`](crate::shaders::nvfp4_matvec::MATMUL_ARMS)) in one
    /// weight pass. `op.x` holds `rows × op.k` floats row-major and
    /// `op.out` receives `rows × op.n` floats row-major from `out_offset`,
    /// `rows` being the arm's width. Each column matches the x2 GEMV on
    /// that row to fp32 rounding.
    pub fn encode_nvfp4_matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        arm: usize,
    ) {
        self.encode_nvfp4_with(
            &self.quant.nvfp4_matmul_pipelines[arm],
            enc,
            op,
            tensor_scale,
        );
    }

    /// One NVFP4 matrix against `rms_norm(op.x; norm)` with the norm
    /// computed inside the GEMV (A-5b rung 2d): no separate norm
    /// dispatch, no normed intermediate. Parity to fp32 rounding (the
    /// sum of squares reduces in a different order than `rms_norm`).
    pub fn encode_nvfp4_matvec_prenorm(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        norm: &PreNorm<'_>,
    ) {
        let kernel = &self.quant.nvfp4_matvec_x2n_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        set_f32(enc, 6, tensor_scale);
        enc.set_buffer(7, Some(op.x), 0);
        enc.set_buffer(8, Some(norm.weight), 0);
        set_f32(enc, 9, norm.eps);
        set_f32(enc, 10, norm.offset);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// Largest `k` the threadgroup-staged pre-norm form accepts: K floats
    /// plus the reduction scratch must fit Apple's 32 KB threadgroup memory.
    pub const PRENORM_STAGED_MAX_K: usize = 8160;

    /// Rung 2d form B: the normalised input staged in threadgroup memory
    /// once per threadgroup, then an x2 body over it. `op.k` must be
    /// ≤ [`Self::PRENORM_STAGED_MAX_K`].
    pub fn encode_nvfp4_matvec_prenorm_staged(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        norm: &PreNorm<'_>,
    ) {
        assert!(
            op.k <= Self::PRENORM_STAGED_MAX_K,
            "k exceeds threadgroup memory"
        );
        let kernel = &self.quant.nvfp4_matvec_x2m_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        set_f32(enc, 6, tensor_scale);
        enc.set_buffer(8, Some(norm.weight), 0);
        set_f32(enc, 9, norm.eps);
        set_f32(enc, 10, norm.offset);
        enc.set_threadgroup_memory_length(0, (op.k * std::mem::size_of::<f32>()) as u64);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// The v1 NVFP4 GEMV, explicitly (the A/B control arm).
    pub fn encode_nvfp4_matvec_v1(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
    ) {
        self.encode_nvfp4_with(&self.quant.nvfp4_matvec_pipeline, enc, op, tensor_scale);
    }

    /// The v2 NVFP4 GEMV, explicitly.
    pub fn encode_nvfp4_matvec_v2(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
    ) {
        self.encode_nvfp4_with(&self.quant.nvfp4_matvec_v2_pipeline, enc, op, tensor_scale);
    }

    /// As [`Self::encode_nvfp4_matvec`], with the weight streams bound at
    /// byte offsets — the matrix is a row slice of a shared allocation.
    ///
    /// This binds the SAME kernel the unsliced call uses: the body reads
    /// `Wp + row * groups * NVFP4_GROUP_BYTES`, so moving the base is
    /// exactly equivalent to handing it a smaller buffer. Routing a
    /// sliced matrix through the segmented kernel instead would change
    /// which code shape runs, and Metal's fast-math means two code shapes
    /// of one kernel are not bit-identical — a layout change must not
    /// smuggle in an arithmetic one.
    pub fn encode_nvfp4_matvec_sliced(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        packed_offset: u64,
        scales_offset: u64,
    ) {
        let q = &self.quant;
        let kernel = match nvfp4_kernel_choice() {
            Nvfp4Kernel::V1 => &q.nvfp4_matvec_pipeline,
            Nvfp4Kernel::V2 => &q.nvfp4_matvec_v2_pipeline,
            Nvfp4Kernel::G2R4 => &q.nvfp4_sweep_pipelines[0],
            Nvfp4Kernel::G4R4 => &q.nvfp4_sweep_pipelines[1],
            Nvfp4Kernel::G1R2 => &q.nvfp4_sweep_pipelines[2],
            Nvfp4Kernel::G1R8 => &q.nvfp4_sweep_pipelines[3],
            Nvfp4Kernel::G2R2 => &q.nvfp4_sweep_pipelines[4],
            Nvfp4Kernel::G2R8 => &q.nvfp4_sweep_pipelines[5],
            Nvfp4Kernel::X2 => &q.nvfp4_sweep_pipelines[6],
            Nvfp4Kernel::X4 => &q.nvfp4_sweep_pipelines[7],
            Nvfp4Kernel::X1B => &q.nvfp4_sweep_pipelines[8],
            Nvfp4Kernel::X2B => &q.nvfp4_sweep_pipelines[9],
            Nvfp4Kernel::X4B => &q.nvfp4_sweep_pipelines[10],
        };
        self.encode_nvfp4_at(kernel, enc, op, tensor_scale, packed_offset, scales_offset);
    }

    /// As [`Self::encode_nvfp4_matvec_residual`], weights bound at byte
    /// offsets (a sliced matrix under rung 2a's folded residual).
    pub fn encode_nvfp4_matvec_residual_sliced(
        &self,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        residual: &Buffer,
        packed_offset: u64,
        scales_offset: u64,
    ) {
        let kernel = &self.quant.nvfp4_matvec_x2r_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), packed_offset);
        enc.set_buffer(1, Some(op.scales), scales_offset);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        set_f32(enc, 6, tensor_scale);
        enc.set_buffer(7, Some(residual), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    fn encode_nvfp4_with(
        &self,
        kernel: &crate::kernels::KernelHandle,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
    ) {
        self.encode_nvfp4_at(kernel, enc, op, tensor_scale, 0, 0);
    }

    fn encode_nvfp4_at(
        &self,
        kernel: &crate::kernels::KernelHandle,
        enc: &ComputeCommandEncoderRef,
        op: &MatvecOperands<'_>,
        tensor_scale: f32,
        packed_offset: u64,
        scales_offset: u64,
    ) {
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), packed_offset);
        enc.set_buffer(1, Some(op.scales), scales_offset);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        set_f32(enc, 6, tensor_scale);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// Encode `out = table[idx[0]] * scale` — the embedding lookup for
    /// the id a prior `encode_argmax` wrote, on the device (1c). `scale`
    /// 0.0 encodes an absent multiplier, as the head kernel does.
    pub fn encode_embed_gather(
        &self,
        enc: &ComputeCommandEncoderRef,
        table: &Buffer,
        idx: &Buffer,
        out: &Buffer,
        hidden: usize,
        scale: f32,
    ) {
        let pipeline = &self.norms.embed_gather_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(table), 0);
        enc.set_buffer(1, Some(idx), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, hidden as u32);
        set_f32(enc, 4, scale);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }

    /// Encode up to three RMS norms of one `x` (shared `eps`) in ONE
    /// dispatch — bit-identical to three `rms_norm` dispatches.
    pub fn encode_rms_norm_multi(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        len: usize,
        eps: f32,
        outputs: &[NormOutput<'_>],
    ) {
        assert!(
            !outputs.is_empty() && outputs.len() <= RMS_NORM_MAX_OUTPUTS,
            "1..=3 norm outputs"
        );
        let pipeline = &self.norms.rms_norm_multi3_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        for i in 0..RMS_NORM_MAX_OUTPUTS {
            // Absent outputs alias the first so every slot is bound.
            let o = outputs.get(i).unwrap_or(&outputs[0]);
            enc.set_buffer(1 + i as u64, Some(o.weight), 0);
            enc.set_buffer(4 + i as u64, Some(o.out), 0);
            set_f32(enc, 9 + i as u64, o.offset);
        }
        set_u32(enc, 7, len as u32);
        set_f32(enc, 8, eps);
        set_u32(enc, 12, outputs.len() as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(len as u64),
                1,
                1,
            ),
        );
    }
}
