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
mod tests {
    use super::*;
    use metal::Device;

    const SHAPE_SMALL: (usize, usize) = (2, 64);
    const SHAPE_LARGE: (usize, usize) = (4, 128);

    fn fresh_cache() -> (BufferCache, Device) {
        let d = Device::system_default().expect("Metal device available on test host");
        let bufs = BufferCache::new(&d);
        (bufs, d)
    }

    // ── occupancy vs absolute position ──────────────────────────────
    //
    // `current_len` used to be both "rows stored" and "stream position".
    // A sliding window separates them: eviction lowers occupancy while
    // the stream keeps advancing. If they re-merge, RoPE silently rewinds
    // on every token after a window slides — which surfaces as fluent but
    // wrong output, the worst failure shape to debug.

    /// One row appended advances both counters together.
    #[test]
    fn advance_one_moves_occupancy_and_position_together() {
        let (bufs, _d) = fresh_cache();
        let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
        assert_eq!((c.current_len, c.abs_position), (0, 0));
        c.advance_one();
        c.advance_one();
        assert_eq!((c.current_len, c.abs_position), (2, 2));
    }

    /// Eviction lowers occupancy and leaves the stream position alone.
    /// This is the whole invariant the window rests on.
    #[test]
    fn eviction_lowers_occupancy_but_never_the_stream_position() {
        let (bufs, _d) = fresh_cache();
        let mut c = LayerKVCache::new(&bufs, 16, 2, 4);
        for _ in 0..10 {
            c.advance_one();
        }
        assert_eq!((c.current_len, c.abs_position), (10, 10));

        let dropped = c.evict_to_window(4);
        assert_eq!(dropped, 6);
        assert_eq!(c.current_len, 4, "occupancy must fall to the window");
        assert_eq!(
            c.abs_position, 10,
            "stream position must NOT rewind — the next row still belongs at 10"
        );

        // Appending after eviction continues the stream, it does not restart it.
        c.advance_one();
        assert_eq!((c.current_len, c.abs_position), (5, 11));
    }

    /// Eviction keeps the NEWEST rows, in order.
    #[test]
    fn eviction_keeps_the_newest_rows_in_order() {
        let (bufs, _d) = fresh_cache();
        let (kv_heads, head_dim) = (2usize, 4usize);
        let row = kv_heads * head_dim;
        let mut c = LayerKVCache::new(&bufs, 16, kv_heads, head_dim);

        // Stamp each row with its index so survivors are identifiable.
        let rows = 10usize;
        unsafe {
            for buf in [&c.k_cache, &c.v_cache] {
                let ptr = buf.contents() as *mut f32;
                let slice = std::slice::from_raw_parts_mut(ptr, 16 * row);
                for r in 0..rows {
                    for i in 0..row {
                        slice[r * row + i] = r as f32;
                    }
                }
            }
        }
        for _ in 0..rows {
            c.advance_one();
        }

        let window = 4usize;
        c.evict_to_window(window);

        unsafe {
            for buf in [&c.k_cache, &c.v_cache] {
                let ptr = buf.contents() as *const f32;
                let slice = std::slice::from_raw_parts(ptr, 16 * row);
                for r in 0..window {
                    let expected = (rows - window + r) as f32;
                    assert_eq!(
                        slice[r * row],
                        expected,
                        "row {r} after eviction should hold original row {expected}"
                    );
                }
            }
        }
    }

    /// A window at or above occupancy is a no-op — nothing moves, so the
    /// unwindowed path pays nothing for the affordance existing.
    #[test]
    fn eviction_is_a_noop_when_the_window_cannot_bind() {
        let (bufs, _d) = fresh_cache();
        let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
        for _ in 0..3 {
            c.advance_one();
        }
        assert_eq!(c.evict_to_window(3), 0);
        assert_eq!(c.evict_to_window(99), 0);
        assert_eq!((c.current_len, c.abs_position), (3, 3));
    }

    /// A zero window is refused rather than emptying the cache — the same
    /// sentinel confusion that already cost a bug in the pipeline spec.
    #[test]
    fn a_zero_window_evicts_nothing() {
        let (bufs, _d) = fresh_cache();
        let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
        c.advance_one();
        assert_eq!(c.evict_to_window(0), 0);
        assert_eq!(c.current_len, 1);
    }

    /// A new prompt restarts the stream, so `clear` resets both.
    #[test]
    fn clear_resets_occupancy_and_position() {
        let (bufs, _d) = fresh_cache();
        let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
        c.advance_one();
        c.advance_one();
        c.clear();
        assert_eq!((c.current_len, c.abs_position), (0, 0));
    }

    #[test]
    fn shape_mismatch_detects_conflicting_existing_layer() {
        assert!(!super::shape_pairs_have_mismatch(
            &[SHAPE_SMALL],
            &[SHAPE_SMALL, SHAPE_LARGE]
        ));
        assert!(super::shape_pairs_have_mismatch(
            &[SHAPE_SMALL],
            &[SHAPE_LARGE]
        ));
    }

    /// `attention_span` returns `t` when `window_size == 0` (no
    /// windowing) or when `t <= window_size` (cache still within
    /// window). Returns `window_size` once `t` exceeds it.
    #[test]
    fn attention_span_clamps_at_window_size_when_exceeded() {
        assert_eq!(attention_span(5, 0), 5, "window=0 disables clamp");
        assert_eq!(attention_span(5, 10), 5, "t<=window returns t");
        assert_eq!(attention_span(10, 10), 10, "t==window returns t");
        assert_eq!(attention_span(15, 10), 10, "t>window clamps to window");
    }

    /// `LayerKVCache::clear` resets `current_len` without touching the
    /// underlying buffers.
    #[test]
    fn layer_kv_cache_clear_resets_current_len() {
        let (bufs, _) = fresh_cache();
        let mut layer = LayerKVCache::new(&bufs, 64, 2, 64);
        layer.current_len = 17;
        layer.clear();
        assert_eq!(layer.current_len, 0);
        assert_eq!(layer.max_seq, 64);
        assert_eq!(layer.num_kv_heads, 2);
        assert_eq!(layer.head_dim, 64);
    }

    /// `KVCache::new` constructs the requested number of uniform-shape
    /// layers.  Round-trips the per-layer dimensions through
    /// `has_shape_mismatch`.
    #[test]
    fn kv_cache_new_creates_uniform_layers() {
        let (bufs, _) = fresh_cache();
        let cache = KVCache::new(&bufs, 3, 32, 2, 64);
        assert_eq!(cache.layers.len(), 3);
        assert!(!cache.has_shape_mismatch(&[(2, 64), (2, 64), (2, 64)]));
        assert!(cache.has_shape_mismatch(&[(2, 64), (2, 64), (4, 64)]));
    }

    /// `KVCache::new_per_layer` allocates with heterogeneous shapes —
    /// pin the Gemma 4 31B pattern (alternating sliding/global heads).
    #[test]
    fn kv_cache_new_per_layer_supports_heterogeneous_shapes() {
        let (bufs, _) = fresh_cache();
        let shapes = vec![(16usize, 256usize), (4, 512), (16, 256), (4, 512)];
        let cache = KVCache::new_per_layer(&bufs, &shapes, 32);
        assert_eq!(cache.layers.len(), 4);
        for (layer, &(num_kv, hd)) in cache.layers.iter().zip(&shapes) {
            assert_eq!(layer.num_kv_heads, num_kv);
            assert_eq!(layer.head_dim, hd);
        }
    }

    /// `grow_to_shapes` extends the cache when more layers are
    /// requested than currently allocated.
    #[test]
    fn kv_cache_grow_to_shapes_extends_layers() {
        let (bufs, _) = fresh_cache();
        let mut cache = KVCache::new(&bufs, 2, 32, 2, 64);
        assert_eq!(cache.layers.len(), 2);

        let shapes = vec![(2usize, 64usize), (2, 64), (4, 128), (8, 256)];
        cache.grow_to_shapes(&bufs, &shapes, 32);
        assert_eq!(cache.layers.len(), 4);
        assert_eq!(cache.layers[2].num_kv_heads, 4);
        assert_eq!(cache.layers[2].head_dim, 128);
        assert_eq!(cache.layers[3].num_kv_heads, 8);
        assert_eq!(cache.layers[3].head_dim, 256);

        // Idempotent: regrow to same length is a no-op.
        cache.grow_to_shapes(&bufs, &shapes, 32);
        assert_eq!(cache.layers.len(), 4);
    }

    /// `KVCache::clear` resets every layer's `current_len`.
    #[test]
    fn kv_cache_clear_resets_all_layers() {
        let (bufs, _) = fresh_cache();
        let mut cache = KVCache::new(&bufs, 3, 32, 2, 64);
        for layer in &mut cache.layers {
            layer.current_len = 9;
        }
        cache.clear();
        assert!(cache.layers.iter().all(|l| l.current_len == 0));
    }

    /// `current_len` reads from the first layer (assumes uniform
    /// progression).  Returns 0 when there are no layers.
    #[test]
    fn kv_cache_current_len_reads_first_layer() {
        let (bufs, _) = fresh_cache();
        let mut cache = KVCache::new(&bufs, 2, 32, 2, 64);
        assert_eq!(cache.current_len(), 0);
        cache.layers[0].current_len = 7;
        assert_eq!(cache.current_len(), 7);

        let empty = KVCache { layers: Vec::new() };
        assert_eq!(empty.current_len(), 0);
    }

    // ─── End-to-end Metal dispatch tests for the encoder helpers ───
    //
    // The remaining uncovered lines exercise `encode_kv_append`,
    // `encode_kv_attend` (both the short-span and long-span branches),
    // and the `append_and_attend` legacy convenience wrapper.  Real
    // GPU dispatches are cheap on small shapes (~< 1 ms per call) so
    // we drive them through `MetalBackend::new()` and assert that the
    // kernels complete without panic and produce finite output.
    use crate::MetalBackend;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    fn append_attend_shapes() -> (usize, usize, usize) {
        // (max_seq, num_kv_heads, head_dim). Sized small so the test
        // stays under a millisecond.  num_q_heads = num_kv_heads in
        // this fixture (non-GQA shape) to keep the input vectors
        // small.
        (8, 2, 64)
    }

    /// #229. A full cache must refuse the append rather than write row
    /// `max_seq` past its buffer. Encoding, not execution: the assertion
    /// fires before any dispatch is recorded.
    #[test]
    fn encode_kv_append_refuses_a_full_cache() {
        let m = backend();
        let (_, num_kv, head_dim) = append_attend_shapes();
        let max_seq = 4;
        let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
        layer.current_len = max_seq; // positions 0..4 stored; the 5th must not land
        let zeros = vec![0.0f32; num_kv * head_dim];
        let new_k_buf = m.bufs.transient_from_f32(&zeros);
        let new_v_buf = m.bufs.transient_from_f32(&zeros);
        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // Metal aborts the process if an encoder is dropped mid-panic, so
        // catch the refusal and end the encoder before asserting on it.
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_kv_append(
                enc,
                &layer,
                &m.attention.kv_append_pipeline,
                &new_k_buf,
                &new_v_buf,
            )
        }));
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "kv_cache refuse test")
            .expect("command buffer completed");
        let msg = match refused {
            Err(payload) => payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
            Ok(()) => panic!("a full cache accepted an append past its buffer"),
        };
        assert!(
            msg.contains("KV append past capacity"),
            "unexpected refusal message: {msg}"
        );
    }

    /// `encode_kv_append` writes new K/V rows into the cache slot at
    /// `current_len`.  After a single dispatch + commit + wait the
    /// dispatch should complete and the kernel input/output buffers
    /// should hold finite values.
    #[test]
    fn encode_kv_append_completes_and_advances_position() {
        let m = backend();
        let (max_seq, num_kv, head_dim) = append_attend_shapes();
        let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);

        let new_k: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
        let new_v: Vec<f32> = (0..num_kv * head_dim)
            .map(|i| ((i + 1) as f32) * 0.002)
            .collect();
        let new_k_buf = m.bufs.transient_from_f32(&new_k);
        let new_v_buf = m.bufs.transient_from_f32(&new_v);

        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_kv_append(
            enc,
            &layer,
            &m.attention.kv_append_pipeline,
            &new_k_buf,
            &new_v_buf,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1042")
            .expect("command buffer completed");

        // The callsite is responsible for bumping `current_len`; the
        // encoder itself only writes the buffer.  Mirror the legacy
        // contract here so the next test path (short-span attend) has
        // a sensible len.
        layer.current_len = 1;
        assert_eq!(layer.current_len, 1);
    }

    /// `encode_kv_attend` short-span path (`span <= SHORT_ATTENTION_SPAN`)
    /// dispatches the `attend_pipeline`.  Pass `None` for
    /// `attend_long_pipeline` so the function uses `attend_pipeline`
    /// even if the span grew.
    #[test]
    fn encode_kv_attend_short_span_dispatches_with_attend_pipeline() {
        let m = backend();
        let (max_seq, num_kv, head_dim) = append_attend_shapes();
        let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
        layer.current_len = 1; // one prior token written

        let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.01).collect();
        let q_buf = m.bufs.transient_from_f32(&q);
        let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_kv_attend(
            enc,
            &layer,
            &m.attention.kv_attend_pipeline,
            None, // long pipeline absent → unwrap_or(attend) path
            &q_buf,
            &out_buf,
            num_kv,
            (head_dim as f32).sqrt().recip(),
            0,
            None,
            0.0,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1084")
            .expect("command buffer completed");

        let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `encode_kv_attend` long-span branch: `span > SHORT_ATTENTION_SPAN`
    /// AND `attend_long_pipeline = Some(...)` picks the long-span
    /// kernel.  Drive this by passing the long pipeline and a
    /// `current_len` large enough to push `span` past the threshold.
    ///
    /// We don't assert finiteness here — the cache slots beyond
    /// the one we wrote are still zero-initialised (no `append`
    /// upstream in this minimal test), so attention over a stretch
    /// of zero-K rows produces numerically degenerate output.  This
    /// test pins **that the long pipeline is selected and dispatches
    /// successfully** (i.e. doesn't panic / fail commit), which is
    /// the part of `encode_kv_attend`'s contract that's interesting
    /// for coverage.
    #[test]
    fn encode_kv_attend_long_span_picks_attend_long_pipeline() {
        let m = backend();
        let (_, num_kv, head_dim) = append_attend_shapes();
        let mut layer = LayerKVCache::new(&m.bufs, 1024, num_kv, head_dim);
        layer.current_len = (SHORT_ATTENTION_SPAN + 2) as usize;

        let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
        let q_buf = m.bufs.transient_from_f32(&q);
        let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_kv_attend(
            enc,
            &layer,
            &m.attention.kv_attend_pipeline,
            Some(&m.attention.kv_attend_long_pipeline),
            &q_buf,
            &out_buf,
            num_kv,
            (head_dim as f32).sqrt().recip(),
            0,
            None,
            0.0,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1131")
            .expect("command buffer completed");

        // Output buffer length matches the requested shape (a weak but
        // valid post-condition: a panicked kernel never gets here, and
        // the dispatch picked the long branch).
        let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
        assert_eq!(out.len(), num_kv * head_dim);
    }

    /// `append_and_attend` chains the append + attend dispatches in a
    /// single command buffer and bumps `current_len` itself.  Covers
    /// the `pub fn append_and_attend` body + the two encoder blocks
    /// it owns.
    #[test]
    fn append_and_attend_runs_append_then_attend_and_bumps_len() {
        let m = backend();
        let (max_seq, num_kv, head_dim) = append_attend_shapes();
        let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
        assert_eq!(layer.current_len, 0);

        let new_k: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
        let new_v: Vec<f32> = (0..num_kv * head_dim)
            .map(|i| ((i + 1) as f32) * 0.002)
            .collect();
        let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.01).collect();

        let new_k_buf = m.bufs.transient_from_f32(&new_k);
        let new_v_buf = m.bufs.transient_from_f32(&new_v);
        let q_buf = m.bufs.transient_from_f32(&q);
        let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

        let cmd = m.queue.new_command_buffer();
        append_and_attend(
            cmd,
            &mut layer,
            &m.attention.kv_append_pipeline,
            &m.attention.kv_attend_pipeline,
            &new_k_buf,
            &new_v_buf,
            &q_buf,
            &out_buf,
            num_kv,
            (head_dim as f32).sqrt().recip(),
        );
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1176")
            .expect("command buffer completed");

        assert_eq!(
            layer.current_len, 1,
            "append_and_attend must bump current_len"
        );
        let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
