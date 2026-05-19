//! `VindexStorage` — sealed mmap-agnostic byte-handle trait.
//!
//! Goal: every backend (today's mmap; eventual Redis-cached, S3-buffered,
//! GPU-resident) satisfies the same surface. Walk kernels, Metal
//! dispatch, and KNN consume this trait — they don't reach into
//! `Arc<Mmap>` fields directly.
//!
//! ## Status
//!
//! Step 1 of the migration plan in `ROADMAP.md` (P0 active, promoted
//! from P2 on 2026-05-10): trait skeleton only, no impls, no callsite
//! changes. The `MmapStorage` parity wrapper lands in step 2; the
//! Criterion bench gate lands in step 3 before any substore migration.
//!
//! ## Returns: `BytesView<'a>` (per-layer) and `Bytes` (whole-file)
//!
//! Per-layer accessors return `BytesView<'a>` — a borrowed
//! `(&'a Bytes, offset, length)` triple. `as_slice() -> &'a [u8]` is
//! zero atomics (just pointer arithmetic into the parent `Bytes`);
//! `to_owned() -> Bytes` is opt-in for callers that need the
//! refcounted handle (Redis emit, cross-thread shipment). The hot
//! path — walk kernels and Metal dispatch — never pays a refcount
//! bump per layer fetch.
//!
//! Whole-file accessors (`*_whole_buffer`, `lm_head_*`) return
//! `bytes::Bytes` directly — they're fetched once at load time, not
//! per-layer-per-token, so the one-time `Bytes::clone` is invisible.
//!
//! Why not `&[u8]` everywhere? Locks remote backends out: a Redis
//! impl has owned bytes that can't anchor on the substore. By keeping
//! the *whole-file* view as `Bytes` (which can be `Bytes::from_owner`
//! over an mmap or `Bytes::copy_from_slice` over a Redis buffer) and
//! exposing per-layer cuts via `BytesView` borrowing from that
//! `Bytes`, both backends fit. The cost on the mmap side is one
//! `Bytes` per substore field at construction, which is paid once.
//!
//! Why not `Cow<'_, [u8]>`? Same lifetime story but no `to_owned`
//! escape hatch — callers that need to keep bytes alive past the
//! borrow have no clean upgrade path.
//!
//! ## Sealing
//!
//! Sealed via the standard private supertrait pattern: out-of-crate
//! types cannot implement `VindexStorage`. We need this so we can add
//! methods (with defaults) without a major bump as Redis / S3 backends
//! arrive. The cost is that downstream tests can't write a stub impl —
//! acceptable here because the capability traits
//! (`QuantizedFfnAccess`, `NativeFfnAccess`, `Fp4FfnAccess`) already
//! cover the dispatch logic above this layer with their own stubs.
//!
//! ## Method shape
//!
//! Every byte-yielding method returns `Option<...>`. `None` means
//! "this backend doesn't carry that file kind" (e.g., FP4 vindexes
//! don't have Q4_K FFN data; client-only slices have no gate
//! vectors). Callers must treat absence as a fall-through, not an
//! error.
//!
//! ## Out of scope (for now)
//!
//! `Fp4Storage` and `DownMetaMmap` are deliberately not behind this
//! trait. Both carry richer per-feature decoders (FP4/FP8 dequant
//! tables, per-layer offsets + tokenizer for down_meta) that are not
//! a clean fit for the "give me bytes" surface. They keep their own
//! mmap fields on substores and stay reachable as
//! `Arc<Fp4Storage>` / `Arc<DownMetaMmap>` directly. If a Redis-backed
//! FP4 vindex ever lands, the path is to either provide a parallel
//! `Fp4Storage` impl or have `Fp4Storage` consume `VindexStorage`
//! internally — but that's a separate decision from this trait.

use bytes::Bytes;

use crate::config::dtype::StorageDtype;
use crate::index::storage::attn::ATTN_TENSORS_PER_LAYER;
use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
use crate::index::types::GateLayerSlice;

mod mmap_storage;
pub use mmap_storage::MmapStorage;

/// Borrowed view into a substore's whole-file `Bytes`. Carries the
/// (offset, length) cut without paying the refcount bump that
/// `Bytes::slice` would. Callers in the hot path use
/// [`BytesView::as_slice`] (zero atomics — just pointer arithmetic);
/// callers that need an owned handle (Redis shipment, cross-thread,
/// holding bytes past the borrow) opt into a refcounted clone via
/// [`BytesView::to_owned_bytes`].
///
/// The lifetime parameter ties the view to whatever `&self` produced
/// it, so dropping the storage object invalidates outstanding views —
/// the same guarantee `Arc<Mmap>` already enforces transitively.
#[derive(Clone, Copy)]
pub struct BytesView<'a> {
    /// The whole-file backing buffer.
    pub(crate) bytes: &'a Bytes,
    /// Byte offset of this view inside `bytes`.
    pub(crate) offset: usize,
    /// Byte length of this view.
    pub(crate) length: usize,
}

impl<'a> BytesView<'a> {
    /// Construct a view. Used by storage impls; bounds checking is
    /// the impl's responsibility (the trait guarantees every returned
    /// view is in-range).
    #[inline]
    pub(crate) fn new(bytes: &'a Bytes, offset: usize, length: usize) -> Self {
        Self {
            bytes,
            offset,
            length,
        }
    }

    /// Borrow the bytes as a slice. Zero atomics — pure pointer math.
    #[inline]
    pub fn as_slice(&self) -> &'a [u8] {
        &self.bytes[self.offset..self.offset + self.length]
    }

    /// Refcounted, owned handle. One atomic increment. Use when the
    /// caller needs the bytes to outlive the borrow (cross-thread,
    /// stored in a struct, shipped to Redis).
    #[inline]
    pub fn to_owned_bytes(&self) -> Bytes {
        self.bytes.slice(self.offset..self.offset + self.length)
    }

    /// Length of the view in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Bundled view of one layer's gate vectors. Replaces three
/// independent substore reaches (`gate.gate_mmap_bytes` +
/// `gate_mmap_slices[layer]` + `gate_mmap_dtype`) that always travel
/// together. Borrowed from `&self` for the same reason `BytesView` is
/// — gate KNN consumes the whole buffer + slice meta inline and never
/// needs to keep them past the call.
#[derive(Clone, Copy)]
pub struct GateLayerView<'a> {
    /// Whole gate-vectors buffer borrowed from the substore. Callers
    /// slice into this using `slice.float_offset` × dtype byte width.
    pub bytes: &'a Bytes,
    /// Storage dtype of the gate matrix (`F16` or `F32` in production
    /// vindexes; legacy paths may report `F32` for synthesized gates).
    pub dtype: StorageDtype,
    /// Per-layer offset + feature count inside the gate buffer.
    pub slice: GateLayerSlice,
}

/// Sealed mmap-agnostic byte-handle for vindex storage backends.
///
/// See module docs for the design rationale and the migration plan
/// this fits into.
pub trait VindexStorage: sealed::Sealed + Send + Sync {
    // ── FFN ─────────────────────────────────────────────────────────────

    /// Q4_K / Q6_K interleaved FFN slices for one layer:
    /// `[(gate_view, gate_fmt), (up_view, up_fmt), (down_view, down_fmt)]`.
    ///
    /// `None` when this backend has no Q4_K interleaved FFN file or the
    /// layer is out of range. `fmt` is the quant tag (`"Q4_K"` /
    /// `"Q6_K"`) routed through `quant::registry`. Each view borrows
    /// from the substore's whole-file buffer; call `as_slice()` for
    /// the hot-path zero-atomic read.
    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); FFN_COMPONENTS_PER_LAYER]>;

    /// Whole-file Q4_K interleaved FFN buffer. Used by Metal
    /// `kquant_matmul_transb` for full-K decode without per-layer
    /// gathering. Returns `Bytes` (refcounted handle) because this is
    /// fetched once at load time, not per-layer-per-token.
    fn interleaved_kquant_whole_buffer(&self) -> Option<Bytes>;

    /// Whole-file Q4_0 interleaved FFN buffer. The Q4_0 path doesn't
    /// have a per-layer manifest; consumers compute layer offsets
    /// from `num_features`.
    fn interleaved_q4_whole_buffer(&self) -> Option<Bytes>;

    /// W2 feature-major Q4_K down for one layer:
    /// `(view, fmt, padded_width)`. `padded_width` is the row stride
    /// after `pad_rows_to_block` — usually equal to `hidden_size`.
    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(BytesView<'_>, &str, usize)>;

    /// Q4_0 gate vectors for one layer (KNN side-channel — feature
    /// retrieval without dequantising the full layer).
    fn gate_q4_layer_data(&self, layer: usize) -> Option<BytesView<'_>>;

    // ── Attention ───────────────────────────────────────────────────────

    /// Q4_K / Q6_K attention projections for one layer:
    /// `[(Q, fmt), (K, fmt), (V, fmt), (O, fmt)]`. `None` when no Q4_K
    /// attention manifest is loaded or the layer is out of range.
    fn attn_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); ATTN_TENSORS_PER_LAYER]>;

    /// Whole-file Q4_0 attention buffer.
    fn attn_q4_whole_buffer(&self) -> Option<Bytes>;

    /// Q4_0 attention projections for one layer: `[Q, K, V, O]` byte
    /// views.
    fn attn_q4_layer_slices(&self, layer: usize)
        -> Option<[BytesView<'_>; ATTN_TENSORS_PER_LAYER]>;

    /// Q8 attention projections for one layer:
    /// `[(vals, scales), (vals, scales), (vals, scales), (vals, scales)]`
    /// for Q, K, V, O. Scales are returned as a `BytesView` — the
    /// caller reinterprets `as_slice()` as `&[f32]` (today's accessor
    /// does the same via `slice::from_raw_parts`).
    fn attn_q8_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, BytesView<'_>); ATTN_TENSORS_PER_LAYER]>;

    // ── lm_head ─────────────────────────────────────────────────────────

    /// Q4_0 lm_head buffer (`lm_head_q4.bin`).
    fn lm_head_q4_bytes(&self) -> Option<Bytes>;
    /// f16 lm_head buffer (`lm_head.bin` when the source dtype is f16).
    fn lm_head_f16_bytes(&self) -> Option<Bytes>;
    /// f32 lm_head buffer (`lm_head.bin` when the source dtype is f32).
    fn lm_head_f32_bytes(&self) -> Option<Bytes>;

    // ── Gate vectors (KNN) ──────────────────────────────────────────────

    /// Bundled view of one layer's gate vectors: bytes + dtype + slice.
    /// Replaces the three-field reach
    /// (`gate_mmap_bytes` + `gate_mmap_slices[layer]` + `gate_mmap_dtype`)
    /// that always travels together.
    fn gate_layer_view(&self, layer: usize) -> Option<GateLayerView<'_>>;
}

mod sealed {
    /// Crate-private supertrait that prevents out-of-crate impls of
    /// `VindexStorage`. Every type that implements `VindexStorage`
    /// inside this crate must also implement `Sealed`. Lives in a
    /// private module so it can't be named by downstream code.
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VindexStorage` must be object-safe — the migration plan holds
    /// it as `Arc<dyn VindexStorage>` on `VectorIndex`.
    #[test]
    fn trait_is_object_safe() {
        // Compile-time check: if the trait gains a non-object-safe
        // method (generics, `Self` by value, associated consts), this
        // line fails to compile.
        fn _assert_object_safe(_: &dyn VindexStorage) {}
    }

    /// `Arc<dyn VindexStorage>` must be `Send + Sync` so it can be
    /// shared across the rayon worker pool the way today's
    /// `Arc<VectorIndex>` is.
    #[test]
    fn trait_object_is_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        _assert_send_sync::<std::sync::Arc<dyn VindexStorage>>();
    }

    /// `GateLayerView<'_>` is `Copy` — borrowed reference to the
    /// underlying `Bytes`, `Copy` dtype, `Clone` slice. Callers can
    /// pass it around without thinking about refcount semantics.
    #[test]
    fn gate_layer_view_is_borrowed_and_copy() {
        let bytes = Bytes::from_static(b"abc");
        let view = GateLayerView {
            bytes: &bytes,
            dtype: StorageDtype::F16,
            slice: GateLayerSlice {
                float_offset: 0,
                num_features: 1,
            },
        };
        let copied = view; // `Copy`, not move.
        assert_eq!(view.bytes.as_ptr(), copied.bytes.as_ptr());
    }

    /// `BytesView::as_slice` returns the in-range bytes; `to_owned`
    /// returns a refcounted `Bytes` slice. The first costs nothing
    /// extra; the second pays one atomic increment.
    #[test]
    fn bytes_view_as_slice_and_owned() {
        let payload: Vec<u8> = (0u8..32).collect();
        let bytes = Bytes::from(payload);
        let view = BytesView::new(&bytes, 8, 16);
        assert_eq!(view.len(), 16);
        assert!(!view.is_empty());
        assert_eq!(view.as_slice(), &(8u8..24).collect::<Vec<u8>>()[..]);
        let owned = view.to_owned_bytes();
        assert_eq!(owned.as_ref(), view.as_slice());
        // Pointer-equal to a slice into the parent buffer: zero copy.
        assert_eq!(owned.as_ptr(), view.as_slice().as_ptr());
    }
}
