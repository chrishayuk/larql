//! VectorIndex struct and core operations.
//!
//! The 35+ flat fields that used to sit on `VectorIndex` are now split
//! across four typed substores under `crate::index::storage`:
//!
//! - `gate`        — `GateStore`        — gate matrix mmap, decode caches, HNSW
//! - `ffn`         — `FfnStore`         — FFN mmap handles + Q4_K dequant cache + FP4
//! - `projections` — `ProjectionStore`  — lm_head + attention weight mmaps
//! - `metadata`    — `MetadataStore`    — down_meta + per-feature overrides
//!
//! Field names within each store match the legacy flat names so the
//! migration is mechanical: `self.gate_mmap_bytes` →
//! `self.gate.gate_mmap_bytes`. A future PR can drop the redundant
//! `gate_` / `q4k_ffn_` prefixes once all call sites move.
//!
//! ## File layout
//!
//! `mod.rs` keeps the struct definition, constructors (`empty`, `new`,
//! `new_mmap`), small inherent helpers (`is_mmap`, `gate_heap_bytes`,
//! `is_layer_owned`, `owned_layer_range`, `set_layer_range`), the
//! `Clone` impl, and the cross-store regression tests. The five
//! capability-trait impls — `GateLookup`, `PatchOverrides`,
//! `NativeFfnAccess`, `QuantizedFfnAccess`, `Fp4FfnAccess` — each live
//! in a sibling module. Adding a new capability is one new sibling
//! plus one `mod` line here.

use std::sync::Arc;

use ndarray::Array2;

use super::storage::{FfnStore, GateStore, MetadataStore, MmapStorage};
// Re-export all shared types from types.rs so external callers can
// keep using `crate::index::core::{VectorIndex, FeatureMeta, …}` paths.
pub use super::types::*;

mod fp4_ffn;
mod gate_lookup;
mod native_ffn;
mod patch_overrides;
mod quantized_ffn;

/// The full model as a local vector index.
///
/// Composes four substores plus the small set of "shape" fields that
/// every store needs to look at. Storage modes (heap vs mmap) are
/// distinguished by which fields inside `gate` are populated, not by
/// a top-level discriminator.
pub struct VectorIndex {
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Vocab size — set by callers that load lm_head; 0 otherwise.
    pub vocab_size: usize,
    /// Layer range owned by this shard, `None` = all layers.
    pub(crate) layer_range: Option<(usize, usize)>,

    /// Gate matrix storage + decode caches + HNSW index.
    pub gate: GateStore,
    /// FFN heap caches + FP4 storage. Mmap-backed FFN tensors moved
    /// to `storage` (step 6 of the migration).
    pub ffn: FfnStore,
    /// down_meta + per-feature overrides.
    pub metadata: MetadataStore,

    /// Mmap-agnostic byte-handle façade. Loaders mutate via
    /// `Arc::make_mut(&mut self.storage).set_*(...)` — the `Arc`
    /// clones-on-write if other `VectorIndex` clones share it, so
    /// `Clone` stays a single refcount bump.
    ///
    /// `MmapStorage` is the production impl; it implements
    /// [`VindexStorage`] for callers that want the trait abstraction
    /// (walk kernels, Metal dispatch, future Redis / S3 backends —
    /// those would land as separate impls of the trait, but this
    /// concrete type is what `VectorIndex` itself holds).
    pub storage: Arc<MmapStorage>,
}

impl Clone for VectorIndex {
    /// Each substore owns its own Clone semantics — Arc'd mmaps share,
    /// mutex/rwlock caches reset, atomics carry their values across.
    /// `storage` is `Arc<dyn VindexStorage>` so cloning is a single
    /// refcount bump.
    fn clone(&self) -> Self {
        Self {
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            layer_range: self.layer_range,
            gate: self.gate.clone(),
            ffn: self.ffn.clone(),
            metadata: self.metadata.clone(),
            storage: Arc::clone(&self.storage),
        }
    }
}

impl VectorIndex {
    /// Inert "nothing loaded" constructor. Every substore is freshly
    /// allocated at the right shape — adding a new field on a substore
    /// is a single edit there, not in `core.rs`.
    pub(crate) fn empty(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            vocab_size: 0,
            layer_range: None,
            gate: GateStore::empty(num_layers),
            ffn: FfnStore::empty(num_layers),
            metadata: MetadataStore::empty(num_layers),
            storage: Arc::new(MmapStorage::empty(hidden_size)),
        }
    }

    /// Build from heap-allocated components (in-memory builds).
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        let mut v = Self::empty(num_layers, hidden_size);
        v.gate.gate_vectors = gate_vectors;
        v.metadata.down_meta = down_meta;
        // Heap mode — no mmap fields are populated. `storage` stays
        // at `MmapStorage::empty(hidden_size)` from `Self::empty()`.
        v
    }

    /// Build a zero-copy mmap-mode index — gate vectors come from the
    /// supplied mmap; down_meta is optionally mmap'd too.
    pub fn new_mmap(
        gate_mmap: memmap2::Mmap,
        gate_slices: Vec<GateLayerSlice>,
        dtype: crate::config::dtype::StorageDtype,
        down_meta_mmap: Option<DownMetaMmap>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        let mut v = Self::empty(num_layers, hidden_size);
        Arc::make_mut(&mut v.storage).set_gate_vectors(Arc::new(gate_mmap), dtype, gate_slices);
        v.metadata.down_meta_mmap = down_meta_mmap.map(Arc::new);
        v
    }

    /// Returns true if this index uses mmap'd gate vectors (zero heap copy).
    pub fn is_mmap(&self) -> bool {
        self.storage.has_gate_vectors()
    }

    /// Estimated heap bytes used by gate vectors (0 if mmap'd).
    pub fn gate_heap_bytes(&self) -> usize {
        if self.is_mmap() {
            return 0;
        }
        self.gate
            .gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Returns true if `layer` is owned by this shard.
    pub fn is_layer_owned(&self, layer: usize) -> bool {
        match self.layer_range {
            None => true,
            Some((start, end)) => layer >= start && layer < end,
        }
    }

    /// Returns the owned layer range, or `None` if all layers are served.
    pub fn owned_layer_range(&self) -> Option<(usize, usize)> {
        self.layer_range
    }

    /// Set the owned layer range (used by `load_vindex_with_range`).
    pub(crate) fn set_layer_range(&mut self, range: (usize, usize)) {
        self.layer_range = Some(range);
    }
}

#[cfg(test)]
mod refactor_tests {
    //! Coverage for the `empty()` / `new()` / `new_mmap()` / `Clone`
    //! refactor. Each substore handles its own Clone semantics; these
    //! tests pin the cross-store invariants (caches reset, Arc shared,
    //! atomics carry).
    use super::*;
    use crate::index::storage::vindex_storage::VindexStorage;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    #[test]
    fn empty_defaults_for_new_fields() {
        let v = VectorIndex::empty(3, 64);
        assert_eq!(v.num_layers, 3);
        assert_eq!(v.hidden_size, 64);
        assert_eq!(v.vocab_size, 0);
        assert_eq!(v.layer_range, None);

        // GateStore defaults — heap fields + caches. Mmap fields
        // moved to `MmapStorage` in step 6 of the migration.
        assert_eq!(v.gate.gate_vectors.len(), 3);
        assert!(v.gate.gate_vectors.iter().all(|s| s.is_none()));
        assert!(!v.gate.hnsw_enabled.load(Ordering::Relaxed));
        assert_eq!(v.gate.hnsw_ef_search.load(Ordering::Relaxed), 200);
        assert_eq!(v.gate.gate_cache_max_layers.load(Ordering::Relaxed), 0);
        assert_eq!(v.gate.f16_decode_cache.lock().unwrap().len(), 3);
        assert_eq!(v.gate.warmed_gates.read().unwrap().len(), 3);
        assert_eq!(v.gate.hnsw_cache.lock().unwrap().len(), 3);

        // FfnStore defaults — only Q4_K dequant cache + FP4 storage
        // remain. All file-backed mmaps moved to `MmapStorage`.
        assert!(v.ffn.fp4_storage.is_none());
        assert_eq!(v.ffn.kquant_ffn_cache.lock().unwrap().len(), 3);
        assert!(!v.storage.has_down_features());
        assert!(!v.storage.has_up_features());
        assert!(!v.storage.has_interleaved_f32());

        // ProjectionStore is now empty — every former field moved to
        // `MmapStorage`. Remaining type kept as a placeholder for
        // existing `index.projections` callsites.

        // MetadataStore defaults
        assert!(v.metadata.down_meta_mmap.is_none());
        assert!(v.metadata.down_overrides.is_empty());
        assert!(v.metadata.up_overrides.is_empty());

        // Storage façade — empty MmapStorage on a freshly-empty index.
        assert!(!v.storage.has_interleaved_kquant());
        assert!(!v.storage.has_interleaved_q4());
        assert!(!v.storage.has_attn_kquant());
        assert!(!v.storage.has_attn_q4());
        assert!(!v.storage.has_attn_q8());
        assert!(!v.storage.has_lm_head_kquant());
        assert!(!v.storage.has_lm_head_f16());
        assert!(!v.storage.has_lm_head_f32());
        assert!(!v.storage.has_gate_vectors());
        assert!(!v.storage.has_gate_q4());
        assert!(v.storage.interleaved_kquant_whole_buffer().is_none());
        assert!(v.storage.attn_kquant_layer_data(0).is_none());
        assert!(v.storage.lm_head_q4_bytes().is_none());
    }

    #[test]
    fn new_preserves_gate_and_down_meta_overrides_empty() {
        let gate = vec![Some(Array2::<f32>::zeros((2, 4))), None];
        let down = vec![None, Some(vec![None; 5])];
        let v = VectorIndex::new(gate.clone(), down.clone(), 2, 4);
        assert_eq!(v.num_layers, 2);
        assert_eq!(v.hidden_size, 4);
        assert!(v.gate.gate_vectors[0].is_some());
        assert_eq!(v.gate.gate_vectors[0].as_ref().unwrap().shape(), &[2, 4]);
        assert!(v.metadata.down_meta[1].is_some());
        assert_eq!(v.metadata.down_meta[1].as_ref().unwrap().len(), 5);
        assert!(!v.storage.has_gate_vectors());
        assert!(v.ffn.fp4_storage.is_none());
    }

    #[test]
    fn new_mmap_sets_mmap_fields_and_defaults_rest() {
        let bytes = vec![0u8; 1024];
        let tmp = std::env::temp_dir().join(format!("core_mmap_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("fake_gate.bin");
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        let v = VectorIndex::new_mmap(mmap, Vec::new(), crate::StorageDtype::F16, None, 4, 16);
        assert_eq!(v.num_layers, 4);
        assert_eq!(v.hidden_size, 16);
        assert!(v.storage.has_gate_vectors());
        assert!(matches!(v.storage.gate_dtype(), crate::StorageDtype::F16));
        assert!(!v.storage.has_down_features());
        assert!(v.ffn.fp4_storage.is_none());
        assert_eq!(v.vocab_size, 0);
        assert_eq!(v.gate.f16_decode_cache.lock().unwrap().len(), 4);

        // No layer slices supplied → `gate_layer_view` is None
        // (matches the empty-gate-mmap case in MmapStorage).
        let view = v.storage.gate_layer_view(0);
        assert!(view.is_none());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clone_shares_arc_mmap_handles() {
        let tmp = std::env::temp_dir().join(format!("core_clone_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("fake_gate.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let original =
            VectorIndex::new_mmap(mmap, Vec::new(), crate::StorageDtype::F32, None, 2, 8);

        // After step 6 the gate `Arc<Mmap>` lives only inside
        // `MmapStorage::mmap_handles`. `Clone for VectorIndex` is a
        // single `Arc::clone(&self.storage)`, so the storage refcount
        // bumps by 1; the inner mmap_handles share the same `Arc`.
        let storage_strong_before = Arc::strong_count(&original.storage);
        let cloned = original.clone();
        let storage_strong_after = Arc::strong_count(&original.storage);
        assert_eq!(storage_strong_after, storage_strong_before + 1);
        // Both clones see the same backing buffer.
        let src_view = original.storage.gate_bytes_view().expect("src");
        let cloned_view = cloned.storage.gate_bytes_view().expect("cloned");
        assert_eq!(src_view.as_ptr(), cloned_view.as_ptr());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clone_preserves_atomic_values() {
        let v = VectorIndex::empty(2, 8);
        v.gate.hnsw_enabled.store(true, Ordering::Relaxed);
        v.gate.hnsw_ef_search.store(42, Ordering::Relaxed);
        v.gate.gate_cache_max_layers.store(7, Ordering::Relaxed);
        v.ffn
            .kquant_ffn_cache_max_layers
            .store(3, Ordering::Relaxed);

        let cloned = v.clone();
        assert!(cloned.gate.hnsw_enabled.load(Ordering::Relaxed));
        assert_eq!(cloned.gate.hnsw_ef_search.load(Ordering::Relaxed), 42);
        assert_eq!(cloned.gate.gate_cache_max_layers.load(Ordering::Relaxed), 7);
        assert_eq!(
            cloned
                .ffn
                .kquant_ffn_cache_max_layers
                .load(Ordering::Relaxed),
            3
        );

        cloned.gate.hnsw_enabled.store(false, Ordering::Relaxed);
        assert!(v.gate.hnsw_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn q4k_ffn_cache_lru_evicts_when_capped() {
        let v = VectorIndex::empty(5, 8);
        {
            let mut cache = v.ffn.kquant_ffn_cache.lock().unwrap();
            let mut lru = v.ffn.kquant_ffn_cache_lru.lock().unwrap();
            for layer in 0..5 {
                cache[layer][0] = Some(Arc::new(vec![0.0f32; 8]));
                lru.push_front(layer);
            }
        }
        v.set_kquant_ffn_cache_max_layers(2);
        let (slots, _) = v.kquant_ffn_cache_stats();
        assert_eq!(slots, 2);
        let cache = v.ffn.kquant_ffn_cache.lock().unwrap();
        assert!(cache[0][0].is_none());
        assert!(cache[1][0].is_none());
        assert!(cache[3][0].is_some() || cache[4][0].is_some());
    }

    #[test]
    fn clone_resets_mutex_caches_to_fresh() {
        let v = VectorIndex::empty(3, 16);

        {
            let mut cache = v.gate.f16_decode_cache.lock().unwrap();
            cache[1] = Some(vec![1.0, 2.0, 3.0]);
        }
        {
            let mut warm = v.gate.warmed_gates.write().unwrap();
            warm[0] = Some(vec![7.0]);
        }

        let cloned = v.clone();

        let src_cache = v.gate.f16_decode_cache.lock().unwrap();
        assert!(src_cache[1].is_some());
        drop(src_cache);

        let cloned_cache = cloned.gate.f16_decode_cache.lock().unwrap();
        assert_eq!(cloned_cache.len(), 3);
        assert!(cloned_cache.iter().all(|s| s.is_none()));
        drop(cloned_cache);

        let cloned_warm = cloned.gate.warmed_gates.read().unwrap();
        assert!(cloned_warm.iter().all(|s| s.is_none()));
    }

    #[test]
    fn clone_preserves_vec_and_hashmap_fields() {
        let mut v = VectorIndex::empty(2, 4);
        v.metadata
            .down_overrides
            .insert((0, 3), vec![1.0, 2.0, 3.0, 4.0]);
        v.metadata.up_overrides.insert((1, 1), vec![5.0; 4]);

        let cloned = v.clone();
        assert_eq!(
            cloned.metadata.down_overrides.get(&(0, 3)),
            Some(&vec![1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(
            cloned.metadata.up_overrides.get(&(1, 1)),
            Some(&vec![5.0; 4])
        );

        let mut cloned = cloned;
        cloned.metadata.down_overrides.insert((1, 0), vec![9.0; 4]);
        assert!(!v.metadata.down_overrides.contains_key(&(1, 0)));
    }

    #[test]
    fn clone_preserves_layer_range() {
        let mut v = VectorIndex::empty(4, 8);
        v.set_layer_range((1, 3));
        let cloned = v.clone();
        assert_eq!(cloned.layer_range, Some((1, 3)));
        assert_eq!(cloned.owned_layer_range(), Some((1, 3)));
    }

    #[test]
    fn clone_carries_fp4_storage_handle() {
        use super::super::fp4_storage::Fp4Storage;
        use crate::config::types::Fp4Config;

        let manifest = Fp4Config::option_b_default();
        let storage = Fp4Storage {
            manifest,
            gate_mmap: None,
            up_mmap: None,
            down_mmap: None,
            layer_features: vec![4, 4],
            hidden: 256,
        };
        let mut v = VectorIndex::empty(2, 256);
        v.ffn.fp4_storage = Some(Arc::new(storage));

        let src_arc = v.ffn.fp4_storage.as_ref().unwrap().clone();
        let strong_before = Arc::strong_count(&src_arc);
        let cloned = v.clone();
        let strong_after = Arc::strong_count(&src_arc);

        assert!(cloned.ffn.fp4_storage.is_some());
        assert_eq!(strong_after, strong_before + 1);
        assert!(Arc::ptr_eq(
            &src_arc,
            cloned.ffn.fp4_storage.as_ref().unwrap()
        ));
    }

    #[test]
    fn clone_independent_hnsw_cache_allocation() {
        let v = VectorIndex::empty(3, 16);
        let cloned = v.clone();

        {
            let mut c = cloned.gate.hnsw_cache.lock().unwrap();
            c[0] = None;
            assert_eq!(c.len(), 3);
        }
        let src = v.gate.hnsw_cache.lock().unwrap();
        assert_eq!(src.len(), 3);
    }

    /// Exp 26 Q2 regression guard — `num_features` falls back to FP4
    /// manifest when no legacy gate vectors are present.
    #[test]
    fn num_features_falls_back_to_fp4_storage() {
        use super::super::fp4_storage::Fp4Storage;
        use crate::config::types::Fp4Config;

        let storage = Fp4Storage {
            manifest: Fp4Config::option_b_default(),
            gate_mmap: None,
            up_mmap: None,
            down_mmap: None,
            layer_features: vec![10240, 10240, 10240],
            hidden: 2560,
        };
        let mut v = VectorIndex::empty(3, 2560);
        v.ffn.fp4_storage = Some(Arc::new(storage));

        assert_eq!(v.num_features(0), 10240);
        assert_eq!(v.num_features(1), 10240);
        assert_eq!(v.num_features(2), 10240);
        assert_eq!(v.num_features(99), 0);
    }

    #[test]
    fn num_features_fp4_fallback_non_uniform_widths() {
        use super::super::fp4_storage::Fp4Storage;
        use crate::config::types::Fp4Config;

        let storage = Fp4Storage {
            manifest: Fp4Config::option_b_default(),
            gate_mmap: None,
            up_mmap: None,
            down_mmap: None,
            layer_features: vec![6144, 12288, 6144, 12288],
            hidden: 1536,
        };
        let mut v = VectorIndex::empty(4, 1536);
        v.ffn.fp4_storage = Some(Arc::new(storage));

        assert_eq!(v.num_features(0), 6144);
        assert_eq!(v.num_features(1), 12288);
        assert_eq!(v.num_features(2), 6144);
        assert_eq!(v.num_features(3), 12288);
    }

    #[test]
    fn num_features_legacy_wins_when_gate_present() {
        use super::super::fp4_storage::Fp4Storage;
        use crate::config::types::Fp4Config;

        let mut v = VectorIndex::empty(2, 256);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((8, 256)));
        let storage = Fp4Storage {
            manifest: Fp4Config::option_b_default(),
            gate_mmap: None,
            up_mmap: None,
            down_mmap: None,
            layer_features: vec![16, 16],
            hidden: 256,
        };
        v.ffn.fp4_storage = Some(Arc::new(storage));
        assert_eq!(v.num_features(0), 8);
        assert_eq!(v.num_features(1), 16);
    }
}
