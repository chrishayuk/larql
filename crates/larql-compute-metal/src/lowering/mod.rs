//! Encoder-level primitives for lowering a VINDEX3 `ComponentOpPlan`.
//!
//! **VINDEX3-G6.** The interpreter path in `larql-vindex` returns to the
//! host between every matrix operation, so it commits and waits 209 times
//! per Glimmer token. Measured consequence: the command queue is empty
//! for 215-271 us before each dispatch begins, flat across a 50x range of
//! weight bytes, and a queue-depth A/B collapses per-dispatch cost from
//! 408 us at depth 1 to 57 us at depth 32. That is queue starvation, the
//! same defect `tests/test_cb_queue_starvation.rs` convicted in the
//! serving decoder — which answered it by encoding a whole token into one
//! command buffer with the elementwise glue on the GPU.
//!
//! The functions here are the pieces that let VINDEX3 adopt that shape
//! **without** calling the serving decoder as a black box. Each one
//! *encodes* into a caller-supplied encoder and touches device buffers
//! only: no command buffer, no commit, no wait, no readback. Scheduling
//! becomes the caller's decision, which is the whole point — VINDEX3's
//! plan stays the authority on *what* happens, and lowering owns *how* it
//! is scheduled.
//!
//! The serving path's own encoders (`decode::encode_qkv`, `encode_attn`,
//! `encode_ffn`) are deliberately not reused as-is: they are keyed to
//! `FullPipelineLayer` and to a `QuantFormat` enum that has no NVFP4
//! variant, so adapting a plan into them would be the bypass this rung
//! exists to avoid. The intended end state is that both frontends share
//! primitives at *this* level.

pub mod attention;
pub mod ffn;
pub mod head;
pub mod stack;

use metal::{Buffer, ComputeCommandEncoderRef};

/// A device buffer, re-exported so callers can hold lowering state
/// without linking `metal` themselves. The CLI is the only place a plan
/// and a device meet, and it should not need the graphics API in its
/// dependency list to say "this is resident".
pub use metal::Buffer as DeviceBuffer;

use crate::MetalBackend;

/// The norm applied to a *branch output* before it joins the residual
/// stream, under four-norm (`NormPlacement::PrePost`) placement.
///
/// Its own weight and epsilon because they are not the pre-norm's:
/// Muse-Glimmer uses eps 1e-5 before its blocks and **1e-8** after them,
/// three orders of magnitude apart, and reusing the pre-norm epsilon
/// produces superficially plausible output while lowering a different
/// program.
///
/// `None` means the op is absent (two-norm placement) — a different
/// claim from a norm with a neutral weight.
pub struct PostNorm<'a> {
    pub weight: &'a Buffer,
    pub eps: f32,
    pub weight_offset: f32,
    /// `hidden` floats of scratch. Separate from the branch output
    /// because the norm reduces over the whole vector before writing,
    /// so writing in place would race its own reduction.
    pub scratch: &'a Buffer,
}

/// One matrix operand resident on the device, tagged with the
/// representation it is stored in.
///
/// The lowering dispatches on this rather than taking a single format,
/// so a plan may keep some matrix classes wide and quantise others — the
/// per-class policy VINDEX3 already expresses, executed under one
/// schedule instead of one command buffer per format family.
#[derive(Clone, Copy)]
pub enum LoweredMatrix<'a> {
    /// Little-endian IEEE f16, `[n, k]` row-major.
    F16 { bytes: &'a Buffer },
    /// e2m1 codes + E4M3 group scales + one f32 tensor scale.
    Nvfp4 {
        packed: &'a Buffer,
        scales: &'a Buffer,
        tensor_scale: f32,
    },
    /// The same e2m1 codes under E8M0 group scales, 32 to a group. Kept
    /// as a first-class representation, not a deprecated one: gpt-oss
    /// ships its expert matrices in MXFP4 natively, so there it is the
    /// checkpoint's own storage rather than a choice.
    Mxfp4 {
        packed: &'a Buffer,
        scales: &'a Buffer,
    },
}

/// Where a matvec reads from and writes to, and the geometry of the
/// matrix between them. Grouped because these five always travel
/// together and a transposed `n`/`k` at a call site is invisible.
#[derive(Clone, Copy)]
pub struct MatvecTarget<'a> {
    pub x: &'a Buffer,
    pub out: &'a Buffer,
    /// Byte offset into `out` — lets a K/V projection write straight
    /// into its KV-cache slot.
    pub out_offset: u64,
    /// Output rows.
    pub n: usize,
    /// Input width.
    pub k: usize,
}

/// One quantised matrix and the vectors it maps between, as device
/// buffers. Grouped because a lowered matvec genuinely needs weights,
/// scales, input, output and geometry, and an eight-argument call at
/// every encode site is where transposed buffers hide.
pub struct MatvecOperands<'a> {
    pub packed: &'a Buffer,
    pub scales: &'a Buffer,
    pub x: &'a Buffer,
    pub out: &'a Buffer,
    /// Byte offset into `out`. Lets a K/V projection write directly into
    /// its KV-cache slot instead of writing scratch and copying.
    pub out_offset: u64,
    /// Output rows.
    pub n: usize,
    /// Input width; must be a whole number of the format's groups.
    pub k: usize,
}

/// Bind a `u32` at `index` as inline constant bytes.
pub(crate) fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, value: u32) {
    enc.set_bytes(index, 4, &value as *const u32 as *const std::ffi::c_void);
}

/// Bind an `f32` at `index` as inline constant bytes.
pub(crate) fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, value: f32) {
    enc.set_bytes(index, 4, &value as *const f32 as *const std::ffi::c_void);
}

impl MetalBackend {
    /// Encode `out = W · x` for a matrix in whichever representation it
    /// is resident in.
    pub fn encode_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &LoweredMatrix<'_>,
        at: &MatvecTarget<'_>,
    ) {
        match w {
            LoweredMatrix::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => self.encode_nvfp4_matvec(
                enc,
                &MatvecOperands {
                    packed,
                    scales,
                    x: at.x,
                    out: at.out,
                    out_offset: at.out_offset,
                    n: at.n,
                    k: at.k,
                },
                *tensor_scale,
            ),
            LoweredMatrix::Mxfp4 { packed, scales } => self.encode_mxfp4_matvec(
                enc,
                &MatvecOperands {
                    packed,
                    scales,
                    x: at.x,
                    out: at.out,
                    out_offset: at.out_offset,
                    n: at.n,
                    k: at.k,
                },
            ),
            LoweredMatrix::F16 { bytes } => self.encode_f16_matvec(enc, bytes, at),
        }
    }

    /// Encode `out = W · x` for an f16 matrix into `enc`.
    pub fn encode_f16_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Buffer,
        at: &MatvecTarget<'_>,
    ) {
        let kernel = &self.f16_gemv_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(at.x), 0);
        enc.set_buffer(2, Some(at.out), at.out_offset);
        set_u32(enc, 3, at.n as u32);
        set_u32(enc, 4, at.k as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

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
        let kernel = &self.quant.nvfp4_matvec_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
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

    /// Encode the MXFP4 sibling, same contract.
    pub fn encode_mxfp4_matvec(&self, enc: &ComputeCommandEncoderRef, op: &MatvecOperands<'_>) {
        let kernel = &self.quant.mxfp4_matvec_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// A pooled device buffer of `floats` f32s, for lowering intermediates
    /// that must never reach the host.
    pub fn lowering_scratch(&self, floats: usize) -> Buffer {
        self.bufs.output((floats * 4) as u64)
    }

    /// Return a lowering scratch buffer to the pool. Only valid after the
    /// command buffer that used it has completed.
    pub fn recycle_lowering_scratch(&self, buf: Buffer) {
        self.bufs.recycle(buf);
    }

    /// Upload `x` into a fresh pooled device buffer — the one host→device
    /// crossing a lowered token needs, at its start.
    pub fn lowering_upload(&self, x: &[f32]) -> Option<Buffer> {
        let buf = self.bufs.output((x.len() * 4) as u64);
        let ptr = buf.contents() as *mut f32;
        if ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes and is not
        // bound to any encoder yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), ptr, x.len()) };
        Some(buf)
    }

    /// Read a device buffer back — the one device→host crossing, at the end.
    pub fn lowering_readback(&self, buf: &Buffer, len: usize) -> Option<Vec<f32>> {
        crate::buffers::try_read_buffer_f32(buf, len)
    }

    /// The cached device buffer for a weight stream, keyed on address
    /// identity (see `BufferCache::get_bytes`).
    pub fn lowering_weight(&self, bytes: &[u8]) -> Buffer {
        self.bufs.get_bytes(bytes)
    }

    /// Register a page-aligned, session-lived byte region so a routed
    /// FFN's expert operands can be bound zero-copy (the same
    /// `register_region` the served `--routed-from` path uses). Returns
    /// `false` if `bytes` is not page-aligned — a lowering that copied
    /// 10 GB of experts into owned buffers would defeat the point.
    pub fn lowering_register_region(&self, bytes: &[u8]) -> bool {
        self.bufs.register_region(bytes)
    }

    /// Build (or fetch) a routed layer's expert descriptor table from a
    /// `MoeLayerWeights` whose expert slices lie in registered regions.
    /// `None` = an operand missed its region or the geometry disagrees —
    /// the caller must refuse, never fall back.
    pub fn lowering_moe_descriptor(
        &self,
        layer_idx: usize,
        moe: &larql_compute::MoeLayerWeights<'_>,
        inter: usize,
        hidden: usize,
    ) -> Option<std::sync::Arc<crate::moe_descriptor::MoeExpertDescriptorTable>> {
        self.descriptor_table_for_layer(layer_idx, moe, inter, hidden)
    }

    /// Whether the descriptor MoE path can serve this layer — checked
    /// before encode so a refusal is typed, not a mid-command-buffer
    /// failure.
    pub fn lowering_moe_supported(
        &self,
        moe: &larql_compute::MoeLayerWeights<'_>,
        scratch: &crate::MoeScratch,
    ) -> bool {
        self.gpu_route_supported(moe, scratch)
    }

    /// A command buffer for a lowered unit of work. Owned by the caller,
    /// which decides how much to encode into it before committing —
    /// the decision this whole rung exists to hand over.
    pub fn new_lowering_command_buffer(&self) -> metal::CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }
}

impl MetalBackend {
    /// Encode weightless per-head RMS over `x` **in place**, one
    /// threadgroup per head.
    ///
    /// In place because the interpreter's `qk_norm_in_place` is, and a
    /// lowering that quietly introduced a copy would diverge the moment
    /// a caller relied on aliasing.
    pub fn encode_parameter_free_qk_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let pipeline = &self.norms.qk_norm_parameter_free_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        set_u32(enc, 1, head_dim as u32);
        set_f32(enc, 2, eps);
        // One threadgroup per head; threads cooperate over `head_dim`.
        // Capped at the pipeline's own limit, and at 1024 so the
        // shader's 32-slot simdgroup-partial array cannot overflow.
        let threads = (head_dim as u64)
            .next_power_of_two()
            .clamp(32, pipeline.max_total_threads_per_threadgroup().min(1024));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// Encode a WEIGHTED per-head RMS norm in place — Gemma's `q_norm` /
    /// `k_norm` (`[head_dim]` weight, `1 + w` when `weight_offset` is 1),
    /// through the served `qk_norm` kernel, one threadgroup per head.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_weighted_qk_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        weight: &Buffer,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
        weight_offset: f32,
    ) {
        let pipeline = &self.norms.qk_norm_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_buffer(1, Some(x), x_offset);
        enc.set_buffer(2, Some(weight), 0);
        set_u32(enc, 3, head_dim as u32);
        set_u32(enc, 4, num_heads as u32);
        set_f32(enc, 5, eps);
        set_f32(enc, 6, weight_offset);
        // The served stage's geometry: one threadgroup per head, threads a
        // power of two up to 512 covering `head_dim`.
        let threads = (head_dim as u64)
            .next_power_of_two()
            .clamp(1, crate::stages::qk_norm::MAX_TG_WIDTH);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// Encode `out = a * sigmoid(g)` — the judged attention output gate.
    pub fn encode_sigmoid_gate(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        g: &Buffer,
        out: &Buffer,
        len: usize,
    ) {
        let pipeline = &self.norms.sigmoid_gate_multiply_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(g), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, len as u32);
        let tg = pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, crate::kernels::DISPATCH_TG_MAX_THREADS);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((len as u64).div_ceil(tg), 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }
}

impl MetalBackend {
    /// Encode `x *= scalar` over `len` floats, in place.
    pub fn encode_scale_vector(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        len: usize,
        scalar: f32,
    ) {
        let pipeline = &self.norms.scale_vector_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(x), 0);
        set_u32(enc, 2, len as u32);
        set_f32(enc, 3, scalar);
        dispatch_linear(enc, pipeline, len);
    }

    /// Encode RoPE over `num_heads` heads at `position`, in place.
    ///
    /// `inv_freq` is host-computed as `theta^(-2i/head_dim)` to match the
    /// interpreter's `rope_rotate`; both use the half-split convention
    /// (`x[i]`, `x[i + head_dim/2]` are the real/imaginary pair), which
    /// is the detail an interleaved-convention kernel would get silently
    /// wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        num_heads: usize,
        head_dim: usize,
        inv_freq: &Buffer,
        position: usize,
        amplitude: f32,
    ) {
        let pipeline = &self.attention.rope_at_pos_batched_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), x_offset);
        set_u32(enc, 1, head_dim as u32);
        enc.set_buffer(2, Some(inv_freq), 0);
        set_u32(enc, 3, position as u32);
        // rotary_dim 0 = rotate the whole head, matching `rope_rotate`.
        set_u32(enc, 4, 0);
        set_u32(enc, 5, num_heads as u32);
        // The cos/sin amplitude — 1.0 for plain rope, YaRN's
        // `attention_amplitude` for a scaled layer — comes from the plan's
        // position policy, never invented here (A-9.4).
        set_f32(enc, 6, amplitude);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((head_dim / 2) as u64, num_heads as u64, 1),
            metal::MTLSize::new(1, 1, 1),
        );
    }

    /// Encode `x[off..][i] += bias[i]` over `len` elements — a projection
    /// bias joining its output in place (the same `bias_add` kernel the
    /// decode path uses).
    pub fn encode_bias_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        bias: &Buffer,
        len: usize,
    ) {
        crate::stages::bias_add::encode(
            enc,
            &self.attention.bias_add_pipeline,
            x,
            x_offset,
            bias,
            len,
        );
    }

    /// Encode `out = a + b_scale * b`.
    pub fn encode_residual_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        len: usize,
        b_scale: f32,
    ) {
        let pipeline = &self.norms.residual_add_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, len as u32);
        set_f32(enc, 4, b_scale);
        dispatch_linear(enc, pipeline, len);
    }
}

/// One thread per element, threadgroups sized from the pipeline's own
/// limit rather than a shader constant.
pub(crate) fn dispatch_linear(
    enc: &ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    len: usize,
) {
    let tg = pipeline
        .max_total_threads_per_threadgroup()
        .clamp(1, crate::kernels::DISPATCH_TG_MAX_THREADS);
    enc.dispatch_thread_groups(
        metal::MTLSize::new((len as u64).div_ceil(tg), 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
}

impl MetalBackend {
    /// Encode `out = branch, normalised if the plan carries a post-norm`,
    /// then `h_out = h_in + out`.
    ///
    /// The order is load-bearing and the reason this is one function
    /// rather than two calls at each site: the interpreter normalises the
    /// **branch output** and then adds it to the residual stream. Adding
    /// first and normalising the sum is a different model, and
    /// "post-attention norm" is an ambiguous enough name that a lowering
    /// could plausibly do either.
    pub fn encode_branch_norm_then_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        branch: &Buffer,
        h_out: &Buffer,
        post: Option<&PostNorm<'_>>,
        hidden: usize,
    ) {
        let addend = match post {
            Some(p) => {
                crate::stages::input_norm::encode_f32(
                    enc,
                    &self.norms.rms_norm_pipeline,
                    branch,
                    0,
                    p.weight,
                    p.scratch,
                    0,
                    hidden,
                    p.eps,
                    p.weight_offset,
                );
                p.scratch
            }
            None => branch,
        };
        self.encode_residual_add(enc, h_in, addend, h_out, hidden, 1.0);
    }
}
