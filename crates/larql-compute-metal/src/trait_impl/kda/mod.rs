//! One complete KDA attention operation in ONE GPU ownership interval.
//!
//! Rung 5c. Rung 5b measured the shape this replaces: KDA's projections
//! run 2.8x faster on device (0.25 ms against 0.70), and the two
//! CPU↔GPU crossings the host-side recurrence forces cost 0.40 ms and
//! give 89% of that back. The crossings exist only because the stages
//! between the projections live on the host. This encodes all of them —
//! convolution, q/k norms, low-rank gates, decay, beta, the delta-rule
//! recurrence, the gated norm — into one command buffer with the
//! projections, so a layer's attention costs **one** crossing.
//!
//! ```text
//! upload normalised hidden          <- the only host->device transfer
//!   grouped q|k|v                       (one dispatch, three slots)
//!   conv+silu x3, q/k L2 norm
//!   f_a -> f_b -> decay, g_a -> g_b, b_proj -> beta
//!   recurrence            (reads and writes device-resident state)
//!   gated RMS norm
//!   o_proj
//! read the attention output         <- the only device->host transfer
//! ```
//!
//! **The recurrent state and the three convolution windows stay on
//! device between calls.** Reading them back to keep the host's
//! representation authoritative would reintroduce the crossing this rung
//! exists to remove; [`KdaDeviceState`] owns them for the life of a
//! sequence and [`KdaDeviceState::read_back`] exists only so a gate can
//! check them against the CPU path.
//!
//! Deliberately NOT done: nothing is fused beyond what
//! `exec::kda::step` already fuses, and no stage is reordered. Rung 4
//! is the standing warning — a fusion whose traffic saving is a fraction
//! of a percent can lose by perturbing an access pattern, and none of
//! these stages is where the bytes are.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::bf16_grouped::{encode_grouped, GroupedBinding, GroupedShape};
use super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use crate::shaders::kda as kda_shader;
use crate::MetalBackend;

/// Bytes per bf16 code.
const BF16_BYTES: usize = 2;
/// `o_proj` is one slot at offset zero. A `static` so its address is
/// stable and the device table can be cached rather than rebuilt.
static O_PROJ_SINGLE_SLOT: [ExpertOffset; 1] = [ExpertOffset(0)];
/// The three convolved streams: q, k, v.
const CONV_STREAMS: usize = 3;

/// The geometry one KDA layer runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaShape {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
}

impl KdaShape {
    /// `num_heads * head_dim` — the width every per-channel stage runs at.
    pub fn width(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements of convolution history each stream carries between calls.
    fn conv_tail(self) -> usize {
        self.conv_kernel.saturating_sub(1)
    }
}

/// The layer's weights, as the device sees them.
///
/// bf16 for the four wide projections — the checkpoint's own bytes,
/// bound without a widening pass — and f32 for the small gate matrices
/// and per-channel vectors, which is what they are on disk.
#[derive(Clone, Copy)]
pub struct KdaDeviceWeights<'a> {
    /// `q|k|v` concatenated, `[3][width, hidden]`, with each slot's byte
    /// offset. One buffer because the grouped kernel binds one.
    pub qkv_bank: &'a [u8],
    pub qkv_offsets: &'a [ExpertOffset; CONV_STREAMS],
    /// `[hidden, width]` bf16.
    pub o_proj: &'a [u8],
    /// `[width, conv_kernel]` each.
    pub q_conv1d: &'a [f32],
    pub k_conv1d: &'a [f32],
    pub v_conv1d: &'a [f32],
    /// `[head_dim, hidden]` then `[width, head_dim]`.
    pub f_a_proj: &'a [f32],
    pub f_b_proj: &'a [f32],
    pub g_a_proj: &'a [f32],
    pub g_b_proj: &'a [f32],
    /// `[num_heads, hidden]`.
    pub b_proj: &'a [f32],
    /// `[num_heads]`, `[width]`, `[head_dim]`.
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub o_norm: &'a [f32],
    pub norm_eps: f32,
}

/// The recurrent and convolution state a KDA layer carries between
/// calls, resident on device.
///
/// Nothing here is indexed by position: the recurrent part is one
/// `D x D` matrix per head whatever the sequence length, and the
/// convolution part is the last `kernel - 1` inputs of each stream.
pub struct KdaDeviceState {
    shape: KdaShape,
    /// `[heads, dim, dim]` f32.
    recurrent: Buffer,
    /// Three `[width, kernel-1]` f32 windows, for q, k and v.
    conv: [Buffer; CONV_STREAMS],
}

impl KdaDeviceState {
    /// The zero state a sequence starts from, allocated on device.
    pub fn zeros(metal: &MetalBackend, shape: KdaShape) -> Self {
        let width = shape.width();
        let recurrent = metal
            .bufs()
            .zeroed((shape.num_heads * shape.head_dim * shape.head_dim * 4) as u64);
        let window = (width * shape.conv_tail() * 4) as u64;
        Self {
            shape,
            recurrent,
            conv: [
                metal.bufs().zeroed(window),
                metal.bufs().zeroed(window),
                metal.bufs().zeroed(window),
            ],
        }
    }

    /// Zero the state again, in place, without reallocating.
    ///
    /// A sequence boundary — and, for a measurement, the way to hold the
    /// input constant: the recurrent state advances every step, so a
    /// timed loop over one token would otherwise be scoring a different
    /// hidden state each iteration, and in a MoE layer a different
    /// route.
    ///
    /// Host-side, so it is only legal between steps, after the wait.
    pub fn reset(&self) {
        let g = self.shape;
        let zero = |b: &Buffer| {
            let ptr = b.contents();
            if !ptr.is_null() {
                // SAFETY: shared-storage buffer of exactly this length,
                // and no GPU work is in flight against it — `reset` is
                // only legal between steps.
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, b.length() as usize) };
            }
        };
        zero(&self.recurrent);
        for c in &self.conv {
            zero(c);
        }
        let _ = g;
    }

    /// `(recurrent, [q, k, v] windows)` copied to the host.
    ///
    /// For gates only. Production never needs this — the point of the
    /// rung is that the state does not come back — but a state that
    /// silently diverged from the CPU path would show up only many
    /// tokens later, so it has to be checkable.
    pub fn read_back(&self) -> (Vec<f32>, [Vec<f32>; CONV_STREAMS]) {
        let g = self.shape;
        let width = g.width();
        let tail = width * g.conv_tail();
        let read = |b: &Buffer, n: usize| crate::buffers::read_buffer_f32(b, n);
        (
            read(&self.recurrent, g.num_heads * g.head_dim * g.head_dim),
            [
                read(&self.conv[0], tail),
                read(&self.conv[1], tail),
                read(&self.conv[2], tail),
            ],
        )
    }
}

/// Every boundary the device path produces, named exactly as the CPU
/// path's `KdaPlanes` names them, so a disagreement reports the stage it
/// happened in rather than "the layer".
#[derive(Debug, Clone)]
pub struct KdaDevicePlanes {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub f_lowrank: Vec<f32>,
    pub g_decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub recurrent_out: Vec<f32>,
    pub o_gate: Vec<f32>,
    pub o_norm: Vec<f32>,
    pub output: Vec<f32>,
    /// GPU-busy milliseconds for the one command buffer this step used.
    pub gpu_ms: f64,
}

/// One convolved stream's bindings: where its projection sits inside the
/// grouped `q|k|v` output, its depthwise weights, the window it carries
/// between calls, and where the convolved result goes.
///
/// Bundled because the four travel together and a call site that took
/// them positionally is where the q window gets paired with the k
/// weights — a silent wrong answer, since all three streams have the
/// same shape.
struct ConvStream<'a> {
    src: &'a Buffer,
    src_offset: u64,
    weight: &'a Buffer,
    window: &'a Buffer,
    out: &'a Buffer,
}

/// Per-call scratch, one pop from the pool each so no two alias.
pub struct Scratch {
    pub qkv: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub f_a: Buffer,
    pub f_low: Buffer,
    pub decay: Buffer,
    pub g_a: Buffer,
    pub gate: Buffer,
    pub b_pre: Buffer,
    pub beta: Buffer,
    pub recurrent_out: Buffer,
    pub normed: Buffer,
    pub out: Buffer,
}

impl MetalBackend {
    /// One KDA attention step, one command buffer.
    ///
    /// `x` is the normalised hidden state — `input_layernorm` is the
    /// caller's, exactly as it is for the CPU path. Returns the
    /// attention output and the command buffer's GPU-busy ms.
    ///
    /// `state` is advanced in place and stays on device.
    pub fn kda_attention_step(
        &self,
        w: KdaDeviceWeights<'_>,
        shape: KdaShape,
        state: &KdaDeviceState,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let (s, gpu_ms) = self.kda_attention_encode(w, shape, state, x)?;
        let out = crate::buffers::read_buffer_f32(&s.out, shape.hidden);
        self.recycle_scratch(s);
        Ok((out, gpu_ms))
    }

    /// The same step, additionally reading back every boundary the CPU
    /// path's `KdaPlanes` exposes.
    ///
    /// For gates only, and it costs what it looks like it costs — a
    /// dozen extra device→host reads that production must never do. It
    /// exists because a device recurrence that drifted would otherwise
    /// surface many tokens later as a wrong answer with no stage
    /// attached to it.
    pub fn kda_attention_step_traced(
        &self,
        w: KdaDeviceWeights<'_>,
        shape: KdaShape,
        state: &KdaDeviceState,
        x: &[f32],
    ) -> Result<KdaDevicePlanes, GroupedError> {
        let (s, gpu_ms) = self.kda_attention_encode(w, shape, state, x)?;
        let (width, heads, hidden) = (shape.width(), shape.num_heads, shape.hidden);
        let read = |b: &Buffer, n: usize| crate::buffers::read_buffer_f32(b, n);
        let qkv = read(&s.qkv, CONV_STREAMS * width);
        let planes = KdaDevicePlanes {
            q_proj: qkv[..width].to_vec(),
            k_proj: qkv[width..2 * width].to_vec(),
            v_proj: qkv[2 * width..].to_vec(),
            q_conv: read(&s.q, width),
            k_conv: read(&s.k, width),
            v_conv: read(&s.v, width),
            q_norm: read(&s.q_norm, width),
            k_norm: read(&s.k_norm, width),
            f_lowrank: read(&s.f_low, width),
            g_decay: read(&s.decay, width),
            beta: read(&s.beta, heads),
            recurrent_out: read(&s.recurrent_out, width),
            o_gate: read(&s.gate, width),
            o_norm: read(&s.normed, width),
            output: read(&s.out, hidden),
            gpu_ms,
        };
        self.recycle_scratch(s);
        Ok(planes)
    }

    pub fn recycle_scratch(&self, s: Scratch) {
        for b in [
            s.qkv,
            s.q,
            s.k,
            s.v,
            s.q_norm,
            s.k_norm,
            s.f_a,
            s.f_low,
            s.decay,
            s.g_a,
            s.gate,
            s.b_pre,
            s.beta,
            s.recurrent_out,
            s.normed,
            s.out,
        ] {
            self.bufs().recycle(b);
        }
    }

    fn kda_attention_encode(
        &self,
        w: KdaDeviceWeights<'_>,
        shape: KdaShape,
        state: &KdaDeviceState,
        x: &[f32],
    ) -> Result<(Scratch, f64), GroupedError> {
        Self::validate_kda(w, shape, state, x.len())?;
        let s = self.kda_scratch(shape);
        let buf_x = self.bufs().transient_from_f32(x);
        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_kda_attention(enc, w, shape, state, &buf_x, &s);
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/kda/mod.rs:step",
        );
        Ok((s, crate::decode::gpu_timing::gpu_elapsed_ms(cmd)))
    }

    /// Shape and residency checks, shared by every entry point so a
    /// check that passed on one path and not another cannot exist.
    pub fn validate_kda(
        w: KdaDeviceWeights<'_>,
        shape: KdaShape,
        state: &KdaDeviceState,
        x_len: usize,
    ) -> Result<(), GroupedError> {
        let (hidden, width) = (shape.hidden, shape.width());
        if x_len != hidden {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: hidden,
                have: x_len,
            });
        }
        if state.shape != shape {
            return Err(GroupedError::SlotCountMismatch {
                expected: shape.width(),
                found: state.shape.width(),
            });
        }
        let per_slot = width * hidden * BF16_BYTES;
        for (slot, off) in w.qkv_offsets.iter().enumerate() {
            if off.0 as usize + per_slot > w.qkv_bank.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot,
                    offset: off.0,
                    need: off.0 as usize + per_slot,
                    have: w.qkv_bank.len(),
                });
            }
        }

        Ok(())
    }

    /// Per-call scratch for one attention step.
    pub fn kda_scratch(&self, shape: KdaShape) -> Scratch {
        let (width, dim, heads, hidden) =
            (shape.width(), shape.head_dim, shape.num_heads, shape.hidden);
        Scratch {
            qkv: self.bufs().output((CONV_STREAMS * width * 4) as u64),
            q: self.bufs().output((width * 4) as u64),
            k: self.bufs().output((width * 4) as u64),
            v: self.bufs().output((width * 4) as u64),
            q_norm: self.bufs().output((width * 4) as u64),
            k_norm: self.bufs().output((width * 4) as u64),
            f_a: self.bufs().output((dim * 4) as u64),
            f_low: self.bufs().output((width * 4) as u64),
            decay: self.bufs().output((width * 4) as u64),
            g_a: self.bufs().output((dim * 4) as u64),
            gate: self.bufs().output((width * 4) as u64),
            b_pre: self.bufs().output((heads * 4) as u64),
            beta: self.bufs().output((heads * 4) as u64),
            recurrent_out: self.bufs().output((width * 4) as u64),
            normed: self.bufs().output((width * 4) as u64),
            out: self.bufs().output((hidden * 4) as u64),
        }
    }

    /// Encode the whole attention into an existing encoder, writing the
    /// output to `s.out`. Encoding only — no command buffer, no commit,
    /// no wait, which is what lets a whole decoder layer share one.
    pub fn encode_kda_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: KdaDeviceWeights<'_>,
        shape: KdaShape,
        state: &KdaDeviceState,
        buf_x: &Buffer,
        s: &Scratch,
    ) {
        let (hidden, width, dim, heads) =
            (shape.hidden, shape.width(), shape.head_dim, shape.num_heads);
        let f32b = |v: &[f32]| self.bufs().get_f32(v);
        // Both tables are constant for the layer: the q|k|v bases never
        // move, and o_proj is always one slot at zero. Cached, not
        // rebuilt — see `stable_offset_table`.
        let qkv_offsets = self.stable_offset_table(w.qkv_offsets);
        let o_offsets = self.stable_offset_table(&O_PROJ_SINGLE_SLOT);

        // q|k|v — one grouped dispatch of three slots, all reading `x`.
        let (qkv_w, qkv_w_off) = self.bufs().weights(w.qkv_bank);
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &qkv_w,
                w_offset: qkv_w_off,
                offsets: &qkv_offsets,
                x: buf_x,
                out: &s.qkv,
            },
            CONV_STREAMS,
            GroupedShape {
                n: width,
                k: hidden,
                layout: InputLayout::Shared,
            },
        );

        // Convolution + SiLU per stream, then the q/k L2 norms. Each
        // stream's window is its own, so the three are independent.
        let stream_bytes = (width * 4) as u64;
        let (qw, kw, vw) = (f32b(w.q_conv1d), f32b(w.k_conv1d), f32b(w.v_conv1d));
        for stream in [
            ConvStream {
                src: &s.qkv,
                src_offset: 0,
                weight: &qw,
                window: &state.conv[0],
                out: &s.q,
            },
            ConvStream {
                src: &s.qkv,
                src_offset: stream_bytes,
                weight: &kw,
                window: &state.conv[1],
                out: &s.k,
            },
            ConvStream {
                src: &s.qkv,
                src_offset: 2 * stream_bytes,
                weight: &vw,
                window: &state.conv[2],
                out: &s.v,
            },
        ] {
            self.encode_short_conv(enc, stream, shape);
        }
        // Out of place, so the convolution output stays observable —
        // `q_conv` and `q_norm` are separate boundaries in the CPU
        // trace and a gate that could not see both would let a
        // convolution error hide behind the normalisation that follows.
        self.encode_l2_norm_heads(enc, &s.q, &s.q_norm, shape);
        self.encode_l2_norm_heads(enc, &s.k, &s.k_norm, shape);

        // The low-rank gates. All three read `x`, so they could share a
        // submission — they already do, being in this encoder.
        self.encode_f32_gemv_into(enc, &f32b(w.f_a_proj), buf_x, &s.f_a, dim, hidden);
        self.encode_f32_gemv_into(enc, &f32b(w.f_b_proj), &s.f_a, &s.f_low, width, dim);
        self.encode_decay_gate(
            enc,
            &s.f_low,
            &f32b(w.dt_bias),
            &f32b(w.a_log),
            &s.decay,
            shape,
        );
        self.encode_f32_gemv_into(enc, &f32b(w.g_a_proj), buf_x, &s.g_a, dim, hidden);
        self.encode_f32_gemv_into(enc, &f32b(w.g_b_proj), &s.g_a, &s.gate, width, dim);
        self.encode_f32_gemv_into(enc, &f32b(w.b_proj), buf_x, &s.b_pre, heads, hidden);
        self.encode_beta(enc, &s.b_pre, &s.beta, heads);

        // The delta rule, against device-resident state.
        self.encode_recurrence(enc, state, s, shape);
        self.encode_gated_rms_norm(
            enc,
            &s.recurrent_out,
            &f32b(w.o_norm),
            &s.gate,
            &s.normed,
            shape,
            w.norm_eps,
        );

        // o_proj — a grouped dispatch of one slot, which is the same
        // kernel and the same arithmetic as any other slot count.
        let (o_w, o_w_off) = self.bufs().weights(w.o_proj);
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &o_w,
                w_offset: o_w_off,
                offsets: &o_offsets,
                x: &s.normed,
                out: &s.out,
            },
            1,
            GroupedShape {
                n: hidden,
                k: width,
                layout: InputLayout::Shared,
            },
        );
    }

    /// `f32_gemv` encoded into an existing encoder — the shader is the
    /// crate's own lm-head gemv, reused unchanged for KDA's small f32
    /// gate matrices.
    pub fn encode_f32_gemv_into(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n: usize,
        k: usize,
    ) {
        let kh = &self.f32_gemv_pipeline;
        let (n32, k32) = (n as u32, k as u32);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(kh.rows_per_tg), 1, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );
    }

    fn encode_short_conv(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: ConvStream<'_>,
        shape: KdaShape,
    ) {
        let (width, kernel) = (shape.width() as u32, shape.conv_kernel as u32);
        enc.set_compute_pipeline_state(&self.kda.short_conv_silu);
        enc.set_buffer(0, Some(s.src), s.src_offset);
        enc.set_buffer(1, Some(s.weight), 0);
        enc.set_buffer(2, Some(s.window), 0);
        enc.set_buffer(3, Some(s.out), 0);
        enc.set_bytes(4, 4, &width as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &kernel as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize::new(width as u64, 1, 1),
            MTLSize::new(kda_shader::ELEMENTWISE_THREADS_PER_TG, 1, 1),
        );
    }

    fn encode_l2_norm_heads(
        &self,
        enc: &ComputeCommandEncoderRef,
        v: &Buffer,
        out: &Buffer,
        shape: KdaShape,
    ) {
        let dim = shape.head_dim as u32;
        enc.set_compute_pipeline_state(&self.kda.l2_normalise_heads);
        enc.set_buffer(0, Some(v), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_bytes(2, 4, &dim as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(shape.num_heads as u64, 1, 1),
            MTLSize::new(kda_shader::HEAD_REDUCE_THREADS_PER_TG, 1, 1),
        );
    }

    fn encode_decay_gate(
        &self,
        enc: &ComputeCommandEncoderRef,
        f_low: &Buffer,
        dt_bias: &Buffer,
        a_log: &Buffer,
        decay: &Buffer,
        shape: KdaShape,
    ) {
        let (width, dim) = (shape.width() as u32, shape.head_dim as u32);
        enc.set_compute_pipeline_state(&self.kda.decay_gate);
        enc.set_buffer(0, Some(f_low), 0);
        enc.set_buffer(1, Some(dt_bias), 0);
        enc.set_buffer(2, Some(a_log), 0);
        enc.set_buffer(3, Some(decay), 0);
        enc.set_bytes(4, 4, &width as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &dim as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize::new(width as u64, 1, 1),
            MTLSize::new(kda_shader::ELEMENTWISE_THREADS_PER_TG, 1, 1),
        );
    }

    fn encode_beta(
        &self,
        enc: &ComputeCommandEncoderRef,
        pre: &Buffer,
        beta: &Buffer,
        heads: usize,
    ) {
        let h = heads as u32;
        enc.set_compute_pipeline_state(&self.kda.beta_sigmoid);
        enc.set_buffer(0, Some(pre), 0);
        enc.set_buffer(1, Some(beta), 0);
        enc.set_bytes(2, 4, &h as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize::new(heads as u64, 1, 1),
            MTLSize::new(
                kda_shader::ELEMENTWISE_THREADS_PER_TG.min(heads as u64),
                1,
                1,
            ),
        );
    }

    fn encode_recurrence(
        &self,
        enc: &ComputeCommandEncoderRef,
        state: &KdaDeviceState,
        s: &Scratch,
        shape: KdaShape,
    ) {
        let dim = shape.head_dim as u32;
        let scale = (shape.head_dim as f32).powf(-0.5);
        enc.set_compute_pipeline_state(&self.kda.recurrence);
        enc.set_buffer(0, Some(&state.recurrent), 0);
        enc.set_buffer(1, Some(&s.q_norm), 0);
        enc.set_buffer(2, Some(&s.k_norm), 0);
        enc.set_buffer(3, Some(&s.v), 0);
        enc.set_buffer(4, Some(&s.decay), 0);
        enc.set_buffer(5, Some(&s.beta), 0);
        enc.set_buffer(6, Some(&s.recurrent_out), 0);
        enc.set_bytes(7, 4, &dim as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(shape.num_heads as u64, 1, 1),
            MTLSize::new(kda_shader::RECURRENCE_THREADS_PER_TG, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gated_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        gate: &Buffer,
        out: &Buffer,
        shape: KdaShape,
        eps: f32,
    ) {
        let dim = shape.head_dim as u32;
        enc.set_compute_pipeline_state(&self.kda.gated_rms_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(gate), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, &dim as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(shape.num_heads as u64, 1, 1),
            MTLSize::new(kda_shader::HEAD_REDUCE_THREADS_PER_TG, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests;
