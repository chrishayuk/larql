//! SPLITK-1 dispatch: decode attention with the span split across
//! threadgroups, then merged (`shaders/kv_attention_splitk.rs`).
//!
//! One helper encodes both passes, so the bench, the numerical gate and
//! the lowering dispatch the same thing.

use metal::{Buffer, ComputeCommandEncoderRef};

use crate::kernels::attention::AttentionKernels;

/// `tg_scores` capacity of the pass-1 kernels: the longest chunk.
pub const SPLITK_MAX_CHUNK: u32 = 1024;

/// Most chunks the planner will ask for; sizes [`SplitKScratch`]. The
/// measured optimum is 16 and 64 lost at every span
/// (`ops::attention_geometry`).
pub const SPLITK_MAX_CHUNKS: usize = 16;

/// Device partial buffers for split-K, sized for `rows` positions of an
/// op with at most `max_q_heads` query heads and `max_q_rows =
/// num_q * head_dim` floats per position.
pub struct SplitKScratch<'a> {
    pub o_part: &'a Buffer,
    pub ml_part: &'a Buffer,
    pub rows: usize,
    pub max_q_heads: usize,
    pub max_q_rows: usize,
}

impl SplitKScratch<'_> {
    /// f32 elements of `(o_part, ml_part)` to allocate.
    pub fn lens(rows: usize, max_q_heads: usize, max_q_rows: usize) -> (usize, usize) {
        (
            rows * max_q_rows * SPLITK_MAX_CHUNKS,
            ml_part_len(rows, max_q_heads, SPLITK_MAX_CHUNKS),
        )
    }

    /// Whether this scratch can hold an op of `rows` positions.
    pub fn fits(&self, rows: usize, num_q_heads: usize, q_rows: usize) -> bool {
        rows <= self.rows && num_q_heads <= self.max_q_heads && q_rows <= self.max_q_rows
    }
}

/// Fewest chunks that keep every chunk of `span` within `tg_scores`.
pub fn min_chunks(span: u32) -> usize {
    span.div_ceil(SPLITK_MAX_CHUNK).max(1) as usize
}

/// f32 elements of the `o` partial buffer.
pub fn o_part_len(rows: usize, num_q: usize, n_chunks: usize, head_dim: usize) -> usize {
    rows * num_q * n_chunks * head_dim
}

/// f32 elements of the `(m, l)` partial buffer.
pub fn ml_part_len(rows: usize, num_q: usize, n_chunks: usize) -> usize {
    rows * num_q * n_chunks * 2
}

/// One split-K attention op over `rows` consecutive positions; row `r`
/// attends a cache of length `kv_len + r`.
pub struct SplitKDispatch<'a> {
    pub q: &'a Buffer,
    pub q_offset: u64,
    pub k_cache: &'a Buffer,
    pub v_cache: &'a Buffer,
    pub out: &'a Buffer,
    pub out_offset: u64,
    pub o_part: &'a Buffer,
    pub ml_part: &'a Buffer,
    /// Per-head sink logits, or `None`. Metal needs slot 6 bound either
    /// way; `placeholder` fills it when there are none (never read).
    pub sinks: Option<&'a Buffer>,
    pub placeholder: &'a Buffer,
    pub kv_len: u32,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub score_scale: f32,
    /// 0 = no window.
    pub window: u32,
    /// 0.0 = no softcap.
    pub softcap: f32,
    pub n_chunks: usize,
    /// Sequence slices inside each chunk's threadgroup (`slices x head_dim`
    /// threads), as `kv_attention_seqpar`.
    pub slices: usize,
    pub rows: usize,
}

impl SplitKDispatch<'_> {
    /// Encode pass 1 and the merge into `enc` (a serial encoder, so the
    /// merge sees pass 1's writes).
    pub fn encode(&self, kernels: &AttentionKernels, enc: &ComputeCommandEncoderRef) {
        self.encode_partial(kernels, enc);
        self.encode_merge(kernels, enc);
    }

    /// Pass 1 alone: every chunk's `(o, m, l)` into the partial buffers.
    pub fn encode_partial(&self, kernels: &AttentionKernels, enc: &ComputeCommandEncoderRef) {
        let widest =
            crate::ops::kv_cache::attention_span(self.kv_len + self.rows as u32 - 1, self.window);
        assert!(
            widest.div_ceil(self.n_chunks as u32) <= SPLITK_MAX_CHUNK,
            "split-K chunk of span {widest} / {} chunks exceeds {SPLITK_MAX_CHUNK}",
            self.n_chunks
        );
        let threads = (self.slices.max(1) * self.head_dim) as u64;
        let set_u32 = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const _);
        let set_f32 = |i: u64, v: f32| enc.set_bytes(i, 4, &v as *const f32 as *const _);

        let (pipeline, grid) = if self.rows == 1 {
            (
                &kernels.kv_attend_splitk_pipeline,
                metal::MTLSize::new(self.num_q_heads as u64, self.n_chunks as u64, 1),
            )
        } else {
            (
                &kernels.kv_attend_splitk_rows_pipeline,
                metal::MTLSize::new(
                    self.num_q_heads as u64,
                    self.n_chunks as u64,
                    self.rows as u64,
                ),
            )
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(self.q), self.q_offset);
        enc.set_buffer(1, Some(self.k_cache), 0);
        enc.set_buffer(2, Some(self.v_cache), 0);
        enc.set_buffer(3, Some(self.o_part), 0);
        set_u32(4, self.kv_len);
        set_u32(5, self.head_dim as u32);
        set_u32(6, self.num_q_heads as u32);
        set_u32(7, self.num_kv_heads as u32);
        set_f32(8, self.score_scale);
        set_u32(9, self.window);
        enc.set_buffer(10, Some(self.ml_part), 0);
        set_u32(11, self.n_chunks as u32);
        set_f32(12, self.softcap);
        enc.dispatch_thread_groups(grid, metal::MTLSize::new(threads, 1, 1));
    }

    /// Pass 2 alone: merge the partial buffers into `out`.
    pub fn encode_merge(&self, kernels: &AttentionKernels, enc: &ComputeCommandEncoderRef) {
        let set_u32 = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const _);
        enc.set_compute_pipeline_state(&kernels.kv_attend_splitk_merge_pipeline);
        enc.set_buffer(0, Some(self.o_part), 0);
        enc.set_buffer(1, Some(self.ml_part), 0);
        enc.set_buffer(2, Some(self.out), self.out_offset);
        set_u32(3, self.head_dim as u32);
        set_u32(4, self.num_q_heads as u32);
        set_u32(5, self.n_chunks as u32);
        enc.set_buffer(6, Some(self.sinks.unwrap_or(self.placeholder)), 0);
        set_u32(7, self.sinks.is_some() as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(self.num_q_heads as u64, self.rows as u64, 1),
            metal::MTLSize::new(self.head_dim.min(1024) as u64, 1, 1),
        );
    }
}
