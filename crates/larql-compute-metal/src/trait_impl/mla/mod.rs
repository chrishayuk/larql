//! One complete MLA attention step on device, with a resident cache.
//!
//! Rung 6a. Its point is not MLA's own speed — the real trajectory
//! measured the GPU idling ~24 ms a token across seven CPU MLA layers,
//! more than the 19.2 ms of GPU work it was waiting between. MLA is the
//! operator that lets a token stay GPU-owned across a KDA→MLA→KDA
//! boundary; the kernels are secondary to that.
//!
//! ```text
//! device hidden
//!   q_proj            grouped bf16, one slot
//!   kv_a_proj         grouped bf16, one slot, written INTO the cache
//!   kv_a RMSNorm      every cached position
//!   kv_b_proj         grouped bf16, one slot repeated per position
//!   attention         one threadgroup per head, over every position
//!   o_proj            grouped bf16, one slot
//! ```
//!
//! **The cache stays on device and is written in place.** A position's
//! `kv_a_proj` output goes straight to `cache + pos * stride` by binding
//! the cache at an offset, so there is no append kernel and no host
//! round trip. Only the RAW latents are cached — nothing decompressed
//! ever is, which is the operator's real cost profile: every step
//! re-derives every prior position's `k_nope`/`v` through `kv_b_proj`.
//! That is also why `kv_b_proj` is the grouped kernel's `PerSlot` shape
//! with one bank and one offset repeated: same matrix, one input row per
//! cached position.
//!
//! **The wide matrices are bf16 codes**, like KDA's q/k/v/o (P4c-4). The
//! checkpoint stores them bf16 and the f32 the host reference uses is a
//! lossless upcast of exactly those bits, so widening back on device
//! changes representation and no value.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::bf16_grouped::{encode_grouped, GroupedBinding, GroupedShape};
use super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use crate::shaders::mla as mla_shader;
use crate::MetalBackend;

/// Every slot of a repeated-matrix grouped dispatch reads the same bank
/// base. One `static` so the device table can be cached rather than
/// rebuilt per step — see `stable_offset_table`.
static SINGLE_SLOT: [ExpertOffset; 1] = [ExpertOffset(0)];

/// The geometry one MLA layer runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaShape {
    pub hidden: usize,
    pub num_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl MlaShape {
    /// `nope + rope` — one head's query width.
    pub fn q_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// `kv_lora_rank + rope` — one cache entry.
    pub fn cache_stride(self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
    /// `heads * (nope + v_head_dim)` — one decompressed position.
    pub fn kv_row(self) -> usize {
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }
    /// `heads * v_head_dim` — the attention value, and `o_proj`'s input.
    pub fn value_width(self) -> usize {
        self.num_heads * self.v_head_dim
    }
}

/// The layer's weights as the device sees them.
#[derive(Clone, Copy)]
pub struct MlaDeviceWeights<'a> {
    /// `[heads * q_head_dim, hidden]` bf16.
    pub q_proj: &'a [u8],
    /// `[kv_lora_rank + rope, hidden]` bf16.
    pub kv_a_proj: &'a [u8],
    /// `[kv_lora_rank]` f32 — over the LATENT only, never the shared
    /// rope-K half of the same cache entry.
    pub kv_a_norm: &'a [f32],
    /// `[heads * (nope + v_head_dim), kv_lora_rank]` bf16.
    pub kv_b_proj: &'a [u8],
    /// `[hidden, heads * v_head_dim]` bf16.
    pub o_proj: &'a [u8],
    pub kv_a_norm_eps: f32,
}

/// The growing compressed-latent cache, resident for the sequence.
///
/// One entry per position, `kv_lora_rank + rope` floats, written in
/// place. Capacity is fixed at construction because a Metal buffer
/// cannot grow: a sequence longer than it was built for is refused, not
/// silently truncated.
pub struct MlaDeviceState {
    shape: MlaShape,
    capacity: usize,
    positions: std::cell::Cell<usize>,
    cache: Buffer,
}

impl MlaDeviceState {
    pub fn with_capacity(metal: &MetalBackend, shape: MlaShape, capacity: usize) -> Self {
        assert!(
            capacity <= mla_shader::MAX_POSITIONS,
            "one attention threadgroup scores at most {} positions",
            mla_shader::MAX_POSITIONS
        );
        Self {
            shape,
            capacity,
            positions: std::cell::Cell::new(0),
            cache: metal
                .bufs()
                .zeroed((capacity * shape.cache_stride() * 4) as u64),
        }
    }

    /// Positions cached so far.
    pub fn len(&self) -> usize {
        self.positions.get()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advance the cache to `visible` positions — called by a chain
    /// after its command buffer completes, since a position is only
    /// really cached once the dispatch that wrote it has run.
    pub(crate) fn advance_to(&self, visible: usize) {
        self.positions.set(visible);
    }

    /// Start a new sequence without reallocating.
    pub fn reset(&self) {
        self.positions.set(0);
    }

    /// The cached latents, copied to the host — for gates only. The
    /// point of the rung is that they do not come back.
    pub fn read_back(&self) -> Vec<Vec<f32>> {
        let stride = self.shape.cache_stride();
        let flat = crate::buffers::read_buffer_f32(&self.cache, self.len() * stride);
        flat.chunks_exact(stride).map(|c| c.to_vec()).collect()
    }
}

/// Every boundary the operator crosses for the CURRENT position, named
/// as `exec::mla::MlaTrace` names them.
#[derive(Debug, Clone)]
pub struct MlaDevicePlanes {
    pub q_proj: Vec<f32>,
    pub compressed_kv: Vec<f32>,
    pub kv_a_normed: Vec<f32>,
    pub kv_b: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub attn_value: Vec<f32>,
    pub output: Vec<f32>,
    pub gpu_ms: f64,
}

pub(crate) struct MlaScratch {
    pub(crate) q: Buffer,
    pub(crate) normed: Buffer,
    pub(crate) kv_b: Buffer,
    pub(crate) weights: Buffer,
    pub(crate) value: Buffer,
    pub(crate) out: Buffer,
}

impl MetalBackend {
    /// One MLA step, one command buffer. Returns the attention output
    /// and the GPU-busy ms; the cache advances by one position.
    pub fn mla_attention_step(
        &self,
        w: MlaDeviceWeights<'_>,
        shape: MlaShape,
        state: &MlaDeviceState,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let (s, visible, gpu_ms) = self.encode_mla(w, shape, state, x)?;
        let out = crate::buffers::read_buffer_f32(&s.out, shape.hidden);
        let _ = visible;
        self.recycle_mla(s);
        Ok((out, gpu_ms))
    }

    /// The same step with every boundary read back. Gates only.
    pub fn mla_attention_step_traced(
        &self,
        w: MlaDeviceWeights<'_>,
        shape: MlaShape,
        state: &MlaDeviceState,
        x: &[f32],
    ) -> Result<MlaDevicePlanes, GroupedError> {
        let (s, visible, gpu_ms) = self.encode_mla(w, shape, state, x)?;
        let cur = visible - 1;
        let f = |b: &Buffer, n: usize| crate::buffers::read_buffer_f32(b, n);
        let latent = shape.kv_lora_rank;
        let planes = MlaDevicePlanes {
            q_proj: f(&s.q, shape.num_heads * shape.q_head_dim()),
            compressed_kv: state.read_back().remove(cur),
            kv_a_normed: f(&s.normed, visible * latent)[cur * latent..].to_vec(),
            kv_b: f(&s.kv_b, visible * shape.kv_row())[cur * shape.kv_row()..].to_vec(),
            attn_weights: f(&s.weights, shape.num_heads * visible),
            attn_value: f(&s.value, shape.value_width()),
            output: f(&s.out, shape.hidden),
            gpu_ms,
        };
        self.recycle_mla(s);
        Ok(planes)
    }

    pub(crate) fn recycle_mla(&self, s: MlaScratch) {
        for b in [s.q, s.normed, s.kv_b, s.weights, s.value, s.out] {
            self.bufs().recycle(b);
        }
    }

    fn encode_mla(
        &self,
        w: MlaDeviceWeights<'_>,
        shape: MlaShape,
        state: &MlaDeviceState,
        x: &[f32],
    ) -> Result<(MlaScratch, usize, f64), GroupedError> {
        let visible = Self::validate_mla(shape, state, x.len())?;
        let s = self.mla_scratch(shape, visible);
        let buf_x = self.bufs().transient_from_f32(x);
        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_mla_attention_into(enc, w, shape, state, &buf_x, &s, visible);
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/mla/mod.rs:step",
        );
        state.positions.set(visible);
        Ok((s, visible, crate::decode::gpu_timing::gpu_elapsed_ms(cmd)))
    }

    /// Shape and capacity checks, returning the visible position count.
    /// Shared by every entry point so a check cannot exist on one path
    /// and not another.
    pub(crate) fn validate_mla(
        shape: MlaShape,
        state: &MlaDeviceState,
        x_len: usize,
    ) -> Result<usize, GroupedError> {
        if x_len != shape.hidden || state.shape != shape {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: shape.hidden,
                have: x_len,
            });
        }
        let pos = state.len();
        if pos >= state.capacity {
            return Err(GroupedError::SlotCountMismatch {
                expected: state.capacity,
                found: pos + 1,
            });
        }
        Ok(pos + 1)
    }

    /// Per-call scratch for one MLA step at `visible` cached positions.
    pub(crate) fn mla_scratch(&self, shape: MlaShape, visible: usize) -> MlaScratch {
        let (heads, latent) = (shape.num_heads, shape.kv_lora_rank);
        MlaScratch {
            q: self.bufs().output((heads * shape.q_head_dim() * 4) as u64),
            normed: self.bufs().output((visible * latent * 4) as u64),
            kv_b: self.bufs().output((visible * shape.kv_row() * 4) as u64),
            weights: self.bufs().output((heads * visible * 4) as u64),
            value: self.bufs().output((shape.value_width() * 4) as u64),
            out: self.bufs().output((shape.hidden * 4) as u64),
        }
    }

    /// Encode the whole attention into an existing encoder, writing the
    /// output to `s.out` and the new latent into the resident cache.
    /// Encoding only — which is what lets a decoder layer, or a chain of
    /// them, share one command buffer.
    ///
    /// The caller advances `state.positions` after the wait.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mla_attention_into(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: MlaDeviceWeights<'_>,
        shape: MlaShape,
        state: &MlaDeviceState,
        buf_x: &Buffer,
        s: &MlaScratch,
        visible: usize,
    ) {
        let pos = visible - 1;
        let (heads, latent) = (shape.num_heads, shape.kv_lora_rank);
        let stride = shape.cache_stride();
        let offsets = self.stable_offset_table(&SINGLE_SLOT);
        // `kv_b_proj` runs one slot per cached position, all reading the
        // same matrix — the table is that one offset repeated.
        let repeated: Vec<ExpertOffset> = vec![ExpertOffset(0); visible];
        let kv_b_offsets = self.offset_table(&repeated);
        // Resolved into their registered regions — see
        // `BufferCache::weights`.
        let wts = |v: &[u8]| self.bufs().weights(v);
        let (wq, wa, wb, wo) = (
            wts(w.q_proj),
            wts(w.kv_a_proj),
            wts(w.kv_b_proj),
            wts(w.o_proj),
        );
        let norm_w = self.bufs().get_f32(w.kv_a_norm);

        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &wq.0,
                w_offset: wq.1,
                offsets: &offsets,
                x: buf_x,
                out: &s.q,
            },
            1,
            GroupedShape {
                n: heads * shape.q_head_dim(),
                k: shape.hidden,
                layout: InputLayout::Shared,
            },
        );
        // The new position's compressed latent goes STRAIGHT into the
        // cache: the output binding carries the offset, so there is no
        // append kernel and nothing crosses to the host.
        self.encode_grouped_at_offset(
            enc,
            (&wa.0, wa.1),
            &offsets,
            buf_x,
            &state.cache,
            (pos * stride * 4) as u64,
            GroupedShape {
                n: stride,
                k: shape.hidden,
                layout: InputLayout::Shared,
            },
        );
        self.encode_kv_a_norm(
            enc,
            &state.cache,
            &norm_w,
            &s.normed,
            shape,
            visible,
            w.kv_a_norm_eps,
        );
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &wb.0,
                w_offset: wb.1,
                offsets: &kv_b_offsets,
                x: &s.normed,
                out: &s.kv_b,
            },
            visible,
            GroupedShape {
                n: shape.kv_row(),
                k: latent,
                layout: InputLayout::PerSlot,
            },
        );
        self.encode_mla_attention(enc, s, &state.cache, shape, visible);
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &wo.0,
                w_offset: wo.1,
                offsets: &offsets,
                x: &s.value,
                out: &s.out,
            },
            1,
            GroupedShape {
                n: shape.hidden,
                k: shape.value_width(),
                layout: InputLayout::Shared,
            },
        );
    }

    /// A grouped dispatch whose OUTPUT is bound at a byte offset — how a
    /// new cache entry is written in place.
    #[allow(clippy::too_many_arguments)]
    fn encode_grouped_at_offset(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: (&Buffer, u64),
        offsets: &Buffer,
        x: &Buffer,
        out: &Buffer,
        out_offset: u64,
        shape: GroupedShape,
    ) {
        let handle = self.default_grouped_handle();
        let (n32, k32, stride) = (shape.n as u32, shape.k as u32, 0u32);
        enc.set_compute_pipeline_state(&handle.state);
        enc.set_buffer(0, Some(w.0), w.1);
        enc.set_buffer(1, Some(offsets), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(out), out_offset);
        enc.set_bytes(4, 4, &n32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &stride as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((shape.n as u64).div_ceil(handle.rows_per_tg), 1, 1),
            MTLSize::new(handle.threads_per_tg, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_kv_a_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        cache: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        shape: MlaShape,
        visible: usize,
        eps: f32,
    ) {
        let (latent, stride) = (shape.kv_lora_rank as u32, shape.cache_stride() as u32);
        enc.set_compute_pipeline_state(&self.mla.kv_a_norm);
        enc.set_buffer(0, Some(cache), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &latent as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &stride as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(visible as u64, 1, 1),
            MTLSize::new(mla_shader::THREADS_PER_TG, 1, 1),
        );
    }

    fn encode_mla_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &MlaScratch,
        cache: &Buffer,
        shape: MlaShape,
        visible: usize,
    ) {
        let v = visible as u32;
        let nope = shape.qk_nope_head_dim as u32;
        let rope = shape.qk_rope_head_dim as u32;
        let v_dim = shape.v_head_dim as u32;
        let latent = shape.kv_lora_rank as u32;
        let kv_row = shape.kv_row() as u32;
        let scaling = (shape.q_head_dim() as f64).powf(-0.5) as f32;
        enc.set_compute_pipeline_state(&self.mla.attention);
        enc.set_buffer(0, Some(&s.q), 0);
        enc.set_buffer(1, Some(&s.kv_b), 0);
        enc.set_buffer(2, Some(cache), 0);
        enc.set_buffer(3, Some(&s.weights), 0);
        enc.set_buffer(4, Some(&s.value), 0);
        for (i, val) in [v, nope, rope, v_dim, latent].into_iter().enumerate() {
            enc.set_bytes(
                (5 + i) as u64,
                4,
                &val as *const u32 as *const std::ffi::c_void,
            );
        }
        enc.set_bytes(10, 4, &scaling as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(11, 4, &kv_row as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(shape.num_heads as u64, 1, 1),
            MTLSize::new(mla_shader::THREADS_PER_TG, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests;
