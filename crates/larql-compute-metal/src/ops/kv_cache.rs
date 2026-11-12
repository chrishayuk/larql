//! KV cache management and cached attention dispatch.
//!
//! Per-layer Metal buffers for cached K/V vectors. Grows with generation.
//! At decode time: append new K/V, then attend Q against full cache.

use metal::*;
use std::ffi::c_void;

use crate::buffers::BufferCache;
use crate::ops::kv_seqpar::SEQPAR_MAX_THREADS;

pub const SHORT_ATTENTION_SPAN: u32 = 1024;

/// `kv_attention_long`'s threadgroup scratch bound — mirrors the MSL
/// `tg_scores[4096]` in `shaders/kv_attention.rs`. Spans past this
/// overflow threadgroup memory; the dispatch asserts against it.
pub const LONG_ATTENTION_SPAN: usize = 4096;

/// Maximum head_dim supported by kernels that dispatch exactly one simdgroup
/// per head (32 lanes × 8 elements = 256). Layers with head_dim above this
/// must use the two-simdgroup path or the unfused fallback.
pub const MAX_HEAD_DIM_SINGLE_SG: usize = 256;

/// Maximum head_dim supported by the two-simdgroup kernel path (32 lanes × 16 = 512).
/// Used as the tg_w ceiling when rounding up to the next power of two for
/// kernels that can span two simdgroups.
pub const MAX_HEAD_DIM_DOUBLE_SG: usize = 512;

fn shape_pairs_have_mismatch(existing: &[(usize, usize)], expected: &[(usize, usize)]) -> bool {
    existing.iter().zip(expected.iter()).any(
        |(&(actual_num_kv, actual_head_dim), &(expected_num_kv, expected_head_dim))| {
            actual_num_kv != expected_num_kv || actual_head_dim != expected_head_dim
        },
    )
}

pub fn attention_span(t: u32, window_size: u32) -> u32 {
    if window_size > 0 && t > window_size {
        window_size
    } else {
        t
    }
}

/// KV cache for one layer — pre-allocated Metal buffers.
pub struct LayerKVCache {
    pub k_cache: Buffer, // [max_seq, num_kv_heads, head_dim] f32
    pub v_cache: Buffer, // same
    /// How many rows are currently stored — the span attention reads.
    ///
    /// This is **occupancy, not position**. The two were one field until
    /// sliding windows needed them apart: a window drops the oldest rows,
    /// so occupancy falls while the stream position keeps climbing. Using
    /// this for RoPE would rewind every token after a window slid.
    pub current_len: usize,
    /// Absolute stream position of the NEXT row to be written — what RoPE
    /// must be computed at. Monotonic for the life of the sequence; never
    /// reduced by eviction.
    pub abs_position: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    /// Create empty KV cache for one layer.
    pub fn new(bufs: &BufferCache, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = (max_seq * num_kv_heads * head_dim * 4) as u64;
        Self {
            k_cache: bufs.output(size),
            v_cache: bufs.output(size),
            current_len: 0,
            abs_position: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    /// Reset cache (for new prompt) — both occupancy and position, since
    /// a new prompt restarts the stream.
    pub fn clear(&mut self) {
        self.current_len = 0;
        self.abs_position = 0;
    }

    /// Record one appended row: occupancy grows and the stream advances.
    /// Kept together so a future append site cannot bump one and forget
    /// the other — the failure that would show up as shifted RoPE many
    /// tokens later.
    pub fn advance_one(&mut self) {
        self.current_len += 1;
        self.abs_position += 1;
    }

    /// Record that the cache now holds exactly the first `rows` rows of a
    /// stream: occupancy and position are both `rows`, so the next row is
    /// rotated at position `rows`.
    ///
    /// For a bulk seed (a prefill computed elsewhere and copied in). Setting
    /// only the occupancy left the position where it was, zero on a fresh
    /// cache, and the first decoded token was rotated as if it opened the
    /// sequence.
    pub fn seed(&mut self, rows: usize) {
        self.current_len = rows;
        self.abs_position = rows;
    }

    /// Roll the stream back so that `len` rows remain, dropping the newest.
    ///
    /// The position falls by the number of rows dropped: a rolled-back row
    /// was never part of the stream, so the next one belongs where it was.
    /// Lowering only the occupancy left the position ahead by every
    /// speculative row. A `len` at or past the current occupancy changes
    /// nothing; a rollback cannot bring rows into being.
    ///
    /// Returns the number of rows dropped.
    pub fn roll_back_to(&mut self, len: usize) -> usize {
        let dropped = self.current_len.saturating_sub(len);
        self.current_len -= dropped;
        self.abs_position = self.abs_position.saturating_sub(dropped);
        dropped
    }

    /// Drop all but the newest `window` rows, sliding the window forward.
    ///
    /// Occupancy falls to `window`; `abs_position` is deliberately
    /// untouched, because the surviving rows keep the RoPE they were
    /// written with and the next row still belongs at the position the
    /// stream has actually reached. Softmax over keys is
    /// order-independent, so nothing needs repairing after the move.
    ///
    /// Returns the number of rows dropped.
    pub fn evict_to_window(&mut self, window: usize) -> usize {
        if window == 0 || self.current_len <= window {
            return 0;
        }
        let drop = self.current_len - window;
        let row = self.num_kv_heads * self.head_dim;
        for buf in [&self.k_cache, &self.v_cache] {
            let ptr = buf.contents() as *mut f32;
            if ptr.is_null() {
                return 0;
            }
            // SAFETY: buffers are host-visible and sized `max_seq * row`;
            // `current_len <= max_seq`, so both ranges are in bounds and
            // `copy_within` semantics handle the overlap.
            unsafe {
                let slice = std::slice::from_raw_parts_mut(ptr, self.max_seq * row);
                slice.copy_within(drop * row..self.current_len * row, 0);
            }
        }
        self.current_len = window;
        drop
    }
}

/// Full KV cache for all layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    /// Allocate a KV cache with uniform per-layer dims — the Llama / Mistral
    /// / Gemma 3 case where every layer shares num_kv_heads and head_dim.
    pub fn new(
        bufs: &BufferCache,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(bufs, max_seq, num_kv_heads, head_dim))
            .collect();
        Self { layers }
    }

    /// Allocate with per-layer shapes — Gemma 4 31B alternates sliding
    /// (num_kv=16, head_dim=256) with global (num_kv=4, head_dim=512) layers,
    /// so a single uniform allocation would either over-size globals or
    /// under-size slidings and produce wrong attention reads.
    ///
    /// `shapes[i]` is `(num_kv_heads_i, head_dim_i)` for layer i.
    pub fn new_per_layer(bufs: &BufferCache, shapes: &[(usize, usize)], max_seq: usize) -> Self {
        let layers = shapes
            .iter()
            .map(|&(num_kv, hd)| LayerKVCache::new(bufs, max_seq, num_kv, hd))
            .collect();
        Self { layers }
    }

    /// Allocate with a per-layer capacity as well as a per-layer shape.
    ///
    /// A sliding layer can never hold more than `SLACK * W` rows, so
    /// sizing it for the global default is waste — on Gemma 3 4B, 29 of 34
    /// layers at 4096 rows instead of 2048. `capacities[i]` pairs with
    /// `shapes[i]`; a short `capacities` falls back to `default_capacity`
    /// for the remainder rather than under-allocating, since an
    /// under-sized buffer is an overrun and an over-sized one is only
    /// waste.
    pub fn new_per_layer_with_capacities(
        bufs: &BufferCache,
        shapes: &[(usize, usize)],
        capacities: &[usize],
        default_capacity: usize,
    ) -> Self {
        let layers = shapes
            .iter()
            .enumerate()
            .map(|(i, &(num_kv, hd))| {
                let cap = capacities.get(i).copied().unwrap_or(default_capacity);
                LayerKVCache::new(bufs, cap, num_kv, hd)
            })
            .collect();
        Self { layers }
    }

    /// Total bytes held by every layer's K and V buffers.
    ///
    /// Exists so a test can assert the allocation actually fell, rather
    /// than asserting a row count and assuming the bytes followed.
    pub fn allocated_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.k_cache.length() + l.v_cache.length())
            .sum()
    }

    /// Return true if any already-allocated layer disagrees with the
    /// corresponding expected `(num_kv_heads, head_dim)` shape.
    pub fn has_shape_mismatch(&self, shapes: &[(usize, usize)]) -> bool {
        let existing: Vec<(usize, usize)> = self
            .layers
            .iter()
            .map(|layer| (layer.num_kv_heads, layer.head_dim))
            .collect();
        shape_pairs_have_mismatch(&existing, shapes)
    }

    /// Grow the cache to cover `shapes` **and** `max_seq`, preserving
    /// existing layers that are already big enough.
    ///
    /// The `max_seq` half of that used to be missing, and it was a silent
    /// overrun rather than a slow path. `ensure_kv_cache_for_shapes`
    /// rebuilds only on a *shape* mismatch, so a second, longer prompt with
    /// the same attention geometry kept the buffers sized for the first one
    /// — while the caller had just asked for more room and had no way to
    /// tell it did not get it. `encode_kv_append` then writes at
    /// `current_len` and bumps it with no bound check against `max_seq`, so
    /// the appends run off the end of a buffer allocated as
    /// `max_seq * num_kv_heads * head_dim * 4`.
    ///
    /// Reachable on a real path: `vindex::kquant_forward::metal` sizes the
    /// cache as `token_ids.len().max(MIN_KV_CACHE_SEQ)`, so it varies with
    /// prompt length across calls on one backend. The uniform call sites
    /// (`kv_cache_mut*`) pass the constant `DEFAULT_KV_CACHE_MAX_SEQ` and
    /// never take this branch.
    ///
    /// Regrowing reallocates, which drops that layer's cached K/V. That
    /// matches what already happens on a shape mismatch, and the one caller
    /// that varies `max_seq` calls `reset_kv_cache()` immediately after, so
    /// nothing that was still live is lost.
    pub fn grow_to_shapes(
        &mut self,
        bufs: &BufferCache,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) {
        for (layer, &(num_kv_heads, head_dim)) in self.layers.iter_mut().zip(shapes.iter()) {
            if layer.max_seq < max_seq {
                *layer = LayerKVCache::new(bufs, max_seq, num_kv_heads, head_dim);
            }
        }
        while self.layers.len() < shapes.len() {
            let (num_kv_heads, head_dim) = shapes[self.layers.len()];
            self.layers
                .push(LayerKVCache::new(bufs, max_seq, num_kv_heads, head_dim));
        }
    }

    /// Grow each layer to its *own* capacity, preserving layers already
    /// large enough.
    ///
    /// The capacity-aware twin of [`Self::grow_to_shapes`]. Growing every
    /// layer to one `max_seq` would re-inflate a sliding layer that was
    /// deliberately allocated at `SLACK * W`, so a per-layer bound is not
    /// an optimisation here — it is what makes the smaller allocation
    /// survive the first decode step.
    pub fn grow_to_capacities(
        &mut self,
        bufs: &BufferCache,
        shapes: &[(usize, usize)],
        capacities: &[usize],
        default_capacity: usize,
    ) {
        let cap_at = |i: usize| capacities.get(i).copied().unwrap_or(default_capacity);
        for (i, (layer, &(num_kv_heads, head_dim))) in
            self.layers.iter_mut().zip(shapes.iter()).enumerate()
        {
            let want = cap_at(i);
            if layer.max_seq < want {
                *layer = LayerKVCache::new(bufs, want, num_kv_heads, head_dim);
            }
        }
        while self.layers.len() < shapes.len() {
            let i = self.layers.len();
            let (num_kv_heads, head_dim) = shapes[i];
            self.layers
                .push(LayerKVCache::new(bufs, cap_at(i), num_kv_heads, head_dim));
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}

/// Encode KV append dispatch into an existing encoder.
/// The encoder is NOT ended — caller continues adding dispatches.
///
/// # Panics
///
/// If the cache is full (`current_len >= max_seq`). The append kernel
/// writes row `current_len` with no bound of its own, so a full cache
/// would mean a write past the buffer — memory corruption that surfaces
/// only much later and somewhere else (#229: a sliding layer sized to its
/// window on a path that never compacts). Refusing here turns that into a
/// named failure at the site that owns the fact.
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_append(
    enc: &ComputeCommandEncoderRef,
    cache: &LayerKVCache,
    append_pipeline: &ComputePipelineState,
    new_k: &Buffer,
    new_v: &Buffer,
) {
    assert!(
        cache.current_len < cache.max_seq,
        "KV append past capacity: current_len {} >= max_seq {} — the cache must be \
         sized for every position this path will reach, or compacted before this \
         append; either way the write must not happen (#229)",
        cache.current_len,
        cache.max_seq
    );
    let pos = cache.current_len as u32;
    let num_kv = cache.num_kv_heads as u32;
    let hd = cache.head_dim as u32;
    let total = cache.num_kv_heads * cache.head_dim;

    enc.set_compute_pipeline_state(append_pipeline);
    enc.set_buffer(0, Some(new_k), 0);
    enc.set_buffer(1, Some(new_v), 0);
    enc.set_buffer(2, Some(&cache.k_cache), 0);
    enc.set_buffer(3, Some(&cache.v_cache), 0);
    enc.set_bytes(4, 4, &pos as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &num_kv as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &hd as *const u32 as *const c_void);
    enc.dispatch_threads(
        MTLSize::new(total as u64, 1, 1),
        MTLSize::new(
            crate::kernels::DISPATCH_TG_MAX_THREADS.min(total as u64),
            1,
            1,
        ),
    );
}

/// Encode KV attend dispatch into an existing encoder.
/// The encoder is NOT ended — caller continues adding dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_attend(
    enc: &ComputeCommandEncoderRef,
    cache: &LayerKVCache,
    attend_pipeline: &ComputePipelineState,
    attend_long_pipeline: Option<&ComputePipelineState>,
    q: &Buffer,
    out: &Buffer,
    num_q_heads: usize,
    scale: f32,
    window_size: u32,
    sinks: Option<&[f32]>,
    softcap: f32,
) {
    let t_val = (cache.current_len + 1) as u32;
    let hd = cache.head_dim as u32;
    let num_q_val = num_q_heads as u32;
    let num_kv = cache.num_kv_heads as u32;
    let span = attention_span(t_val, window_size);
    let pipeline = if span > SHORT_ATTENTION_SPAN {
        // No silent fallback to the short kernel: `kv_attention`'s
        // tg_scores holds SHORT_ATTENTION_SPAN entries, and a span past
        // it overflows threadgroup memory. Previously
        // `unwrap_or(attend_pipeline)` let a caller that supplied no
        // long pipeline do exactly that (capability audit F20).
        attend_long_pipeline.expect(
            "attention span exceeds SHORT_ATTENTION_SPAN and no long-attention \
             pipeline was supplied; the short kernel's threadgroup scratch \
             cannot hold this span",
        )
    } else {
        attend_pipeline
    };
    // `kv_attention_long`'s own scratch bound. Until now this held only
    // because the KV cache's default allocation happens to match —
    // an allocation coincidence, not a checked invariant (F14).
    assert!(
        span as usize <= LONG_ATTENTION_SPAN,
        "attention span {span} exceeds kv_attention_long's threadgroup scratch \
         bound ({LONG_ATTENTION_SPAN})"
    );

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(&cache.k_cache), 0);
    enc.set_buffer(2, Some(&cache.v_cache), 0);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(4, 4, &t_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &hd as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &num_q_val as *const u32 as *const c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.set_bytes(9, 4, &window_size as *const u32 as *const c_void);
    // Feature buffers the fallback must carry (audit F7/F8): a fallback
    // kernel that drops sinks or softcap changes the softmax semantics
    // relative to the fused path it replaced.
    crate::stages::sinks::bind(enc, 10, sinks, num_q_heads);
    enc.set_bytes(12, 4, &softcap as *const f32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, 1, 1),
        MTLSize::new(
            crate::kernels::DISPATCH_TG_MAX_THREADS.min(cache.head_dim as u64),
            1,
            1,
        ),
    );
}

/// KV-B1: [`encode_kv_attend`] with the sequence-parallel phase-3 kernel.
///
/// Identical bindings and identical semantics; the only differences are the
/// pipeline and a threadgroup of `slices * head_dim` threads instead of
/// `head_dim`. Phase 3 in the shipped kernel walks the whole span serially
/// with `head_dim` threads and is ~85% of long-span cost; this splits it.
///
/// The result is not bitwise equal to [`encode_kv_attend`] — summing slice
/// partials reassociates the accumulation. Gated with a calibrated
/// tolerance in `tests/test_kernel_kv_attention_seqpar.rs`.
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_attend_seqpar(
    enc: &ComputeCommandEncoderRef,
    cache: &LayerKVCache,
    seqpar_pipeline: &ComputePipelineState,
    seqpar_long_pipeline: &ComputePipelineState,
    q: &Buffer,
    out: &Buffer,
    num_q_heads: usize,
    scale: f32,
    window_size: u32,
    sinks: Option<&[f32]>,
    softcap: f32,
    slices: usize,
) {
    let t_val = (cache.current_len + 1) as u32;
    let hd = cache.head_dim as u32;
    let num_q_val = num_q_heads as u32;
    let num_kv = cache.num_kv_heads as u32;
    let span = attention_span(t_val, window_size);
    let pipeline = if span > SHORT_ATTENTION_SPAN {
        seqpar_long_pipeline
    } else {
        seqpar_pipeline
    };
    assert!(
        span as usize <= LONG_ATTENTION_SPAN,
        "attention span {span} exceeds the seqpar kernel's threadgroup scratch \
         bound ({LONG_ATTENTION_SPAN})"
    );
    assert!(
        slices > 1 && slices * cache.head_dim <= SEQPAR_MAX_THREADS,
        "seqpar slices {slices} x head_dim {} exceed {SEQPAR_MAX_THREADS}, the \
         kernel's tg_partial bound; callers must go through `seqpar_slices_for`",
        cache.head_dim
    );

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(&cache.k_cache), 0);
    enc.set_buffer(2, Some(&cache.v_cache), 0);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(4, 4, &t_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &hd as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &num_q_val as *const u32 as *const c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.set_bytes(9, 4, &window_size as *const u32 as *const c_void);
    crate::stages::sinks::bind(enc, 10, sinks, num_q_heads);
    enc.set_bytes(12, 4, &softcap as *const f32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, 1, 1),
        MTLSize::new((slices * cache.head_dim) as u64, 1, 1),
    );
}

/// Append new K/V to cache and run attention in one command buffer.
/// Returns attention output [num_q_heads, head_dim].
/// Legacy API — creates its own encoders. For merged pipelines, use
/// encode_kv_append + encode_kv_attend directly.
#[allow(clippy::too_many_arguments)]
pub fn append_and_attend(
    cmd: &CommandBufferRef,
    cache: &mut LayerKVCache,
    append_pipeline: &ComputePipelineState,
    attend_pipeline: &ComputePipelineState,
    new_k: &Buffer,
    new_v: &Buffer,
    q: &Buffer,
    out: &Buffer,
    num_q_heads: usize,
    scale: f32,
) {
    // Append in its own encoder
    {
        let enc = cmd.new_compute_command_encoder();
        encode_kv_append(enc, cache, append_pipeline, new_k, new_v);
        enc.end_encoding();
    }

    // Attend in its own encoder (reads from cache written by append)
    {
        let enc = cmd.new_compute_command_encoder();
        encode_kv_attend(
            enc,
            cache,
            attend_pipeline,
            None,
            q,
            out,
            num_q_heads,
            scale,
            0,
            // Legacy bench API: no layer in scope, no sinks/softcap.
            None,
            0.0,
        );
        enc.end_encoding();
    }

    cache.current_len += 1;
}

#[cfg(test)]
mod tests;
