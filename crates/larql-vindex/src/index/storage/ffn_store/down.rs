//! Feature-major down projections (`down_features.bin`, f32 mmap).
//!
//! Zero-copy slicing — the per-feature down vector is a `&[f32]` view
//! straight into the mmap, no decode, no clone. Per-layer offsets go
//! through `ffn_layer_byte_offset` so variable per-layer feature counts
//! (MoE shards) address correctly.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::DOWN_FEATURES_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load feature-major down vectors from down_features.bin.
    pub fn load_down_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(DOWN_FEATURES_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "down_features.bin not found. Run: cargo run --release -p larql-vindex --example build_down_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only the activated feature vectors are read per token.
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });
        Arc::make_mut(&mut self.storage).set_down_features(mmap);
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.storage.has_down_features()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns `[hidden_size]` f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let bytes = self.storage.down_features_view()?;
        let mmap: &[u8] = bytes.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate {
            return None;
        }

        let layer_offset = self.ffn_layer_byte_offset(layer, 1);
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() {
            return None;
        }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let bytes = self.storage.down_features_view()?;
        let mmap: &[u8] = bytes.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }

        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = self.ffn_layer_byte_offset(layer, 1);
        let end = start + bytes_per_layer;
        if end > mmap.len() {
            return None;
        }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! Round-trip coverage of the feature-major down loader. Each test
    //! writes a small `down_features.bin` to a tempdir, mmaps it through
    //! `load_down_features`, and asserts the zero-copy slice / layer-
    //! matrix views agree with the bytes we wrote.
    use ndarray::Array2;

    use super::*;

    /// Build a `VectorIndex` with `num_features(layer) = intermediate` so
    /// the feature-major decode arithmetic has a non-zero intermediate
    /// to slice against.
    fn vindex_with_layer_features(
        num_layers: usize,
        intermediate: usize,
        hidden: usize,
    ) -> VectorIndex {
        let mut v = VectorIndex::empty(num_layers, hidden);
        for layer in 0..num_layers {
            v.gate.gate_vectors[layer] = Some(Array2::<f32>::zeros((intermediate, hidden)));
        }
        v
    }

    /// Write `floats` as raw f32 bytes into `dir/down_features.bin`.
    fn write_down_features(dir: &std::path::Path, floats: &[f32]) {
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dir.join(DOWN_FEATURES_BIN), &bytes).unwrap();
    }

    #[test]
    fn load_down_features_errors_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut v = vindex_with_layer_features(1, 2, 4);
        let err = v
            .load_down_features(tmp.path())
            .expect_err("missing file errors");
        assert!(err.to_string().contains("down_features.bin"));
    }

    #[test]
    fn load_down_features_populates_mmap_handle() {
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mut v = vindex_with_layer_features(1, 2, 4);
        assert!(!v.has_down_features());
        v.load_down_features(tmp.path()).unwrap();
        assert!(v.has_down_features());
    }

    #[test]
    fn down_feature_vector_returns_zero_copy_slice() {
        // 1 layer, intermediate=2, hidden=4 → 2 features × 4 floats each.
        let tmp = tempfile::tempdir().unwrap();
        let bytes: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_down_features(tmp.path(), &bytes);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_down_features(tmp.path()).unwrap();

        let f0 = v.down_feature_vector(0, 0).expect("feature 0 present");
        assert_eq!(f0, &[0.0, 1.0, 2.0, 3.0]);
        let f1 = v.down_feature_vector(0, 1).expect("feature 1 present");
        assert_eq!(f1, &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn down_feature_vector_none_when_mmap_not_loaded() {
        let v = vindex_with_layer_features(1, 2, 4);
        // No load_down_features call — mmap is None.
        assert!(v.down_feature_vector(0, 0).is_none());
    }

    #[test]
    fn down_feature_vector_none_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0; 8]);

        // VectorIndex::empty has gate_vectors[*] = None → num_features returns 0.
        let mut v = VectorIndex::empty(1, 4);
        v.load_down_features(tmp.path()).unwrap();
        assert!(v.down_feature_vector(0, 0).is_none());
    }

    #[test]
    fn down_feature_vector_none_for_oob_feature() {
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0; 8]);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_down_features(tmp.path()).unwrap();
        // intermediate=2, so feature=99 is out of range.
        assert!(v.down_feature_vector(0, 99).is_none());
    }

    #[test]
    fn down_feature_vector_none_when_mmap_too_short() {
        // Build a vindex that thinks it has 4 features × 4 floats per
        // layer (= 64 bytes) but write only 16 bytes.
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0, 2.0, 3.0, 4.0]); // 16 bytes

        let mut v = vindex_with_layer_features(1, 4, 4);
        v.load_down_features(tmp.path()).unwrap();
        // feature 0 fits; feature 3 does not.
        assert!(v.down_feature_vector(0, 0).is_some());
        assert!(v.down_feature_vector(0, 3).is_none());
    }

    #[test]
    fn down_layer_matrix_returns_intermediate_by_hidden_view() {
        let tmp = tempfile::tempdir().unwrap();
        let bytes: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_down_features(tmp.path(), &bytes);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_down_features(tmp.path()).unwrap();

        let view = v.down_layer_matrix(0).expect("layer 0 view");
        assert_eq!(view.shape(), &[2, 4]);
        assert_eq!(view[[0, 0]], 0.0);
        assert_eq!(view[[1, 3]], 7.0);
    }

    #[test]
    fn down_layer_matrix_none_when_mmap_not_loaded() {
        let v = vindex_with_layer_features(1, 2, 4);
        assert!(v.down_layer_matrix(0).is_none());
    }

    #[test]
    fn down_layer_matrix_none_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0; 8]);
        let mut v = VectorIndex::empty(1, 4);
        v.load_down_features(tmp.path()).unwrap();
        assert!(v.down_layer_matrix(0).is_none());
    }

    #[test]
    fn down_layer_matrix_none_when_mmap_too_short_for_full_layer() {
        // Tell the index to expect 4 features × 4 floats but only write 8.
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[1.0, 2.0]); // 8 bytes
        let mut v = vindex_with_layer_features(1, 4, 4);
        v.load_down_features(tmp.path()).unwrap();
        assert!(v.down_layer_matrix(0).is_none());
    }

    #[test]
    fn has_down_features_tracks_mmap_state() {
        let tmp = tempfile::tempdir().unwrap();
        write_down_features(tmp.path(), &[0.0; 8]);
        let mut v = vindex_with_layer_features(1, 2, 4);
        assert!(!v.has_down_features());
        v.load_down_features(tmp.path()).unwrap();
        assert!(v.has_down_features());
    }
}
