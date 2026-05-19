//! Feature-major up projections (`up_features.bin`, f32 mmap).
//!
//! Mirror of `down.rs` for the up matrix. `has_full_mmap_ffn` lives
//! here because it's the one cross-cutting predicate (up + down both
//! loaded) — kept on the up side since the up loader is the second
//! to fire by convention.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::UP_FEATURES_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load feature-major up vectors from up_features.bin.
    pub fn load_up_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(UP_FEATURES_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "up_features.bin not found. Run: cargo run --release -p larql-vindex --example build_up_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only activated feature vectors are read per token.
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });
        Arc::make_mut(&mut self.storage).set_up_features(mmap);
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let bytes = self.storage.up_features_view()?;
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

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.storage.has_down_features() && self.storage.has_up_features()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! Round-trip coverage of the feature-major up loader. Mirrors the
    //! test pattern in `down.rs`.
    use ndarray::Array2;

    use super::*;
    use crate::format::filenames::DOWN_FEATURES_BIN;

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

    fn write_features(dir: &std::path::Path, name: &str, floats: &[f32]) {
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(dir.join(name), &bytes).unwrap();
    }

    #[test]
    fn load_up_features_errors_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut v = vindex_with_layer_features(1, 2, 4);
        let err = v
            .load_up_features(tmp.path())
            .expect_err("missing file errors");
        assert!(err.to_string().contains("up_features.bin"));
    }

    #[test]
    fn load_up_features_populates_mmap_handle() {
        let tmp = tempfile::tempdir().unwrap();
        write_features(tmp.path(), UP_FEATURES_BIN, &[1.0; 8]);

        let mut v = vindex_with_layer_features(1, 2, 4);
        assert!(v.up_layer_matrix(0).is_none());
        v.load_up_features(tmp.path()).unwrap();
        assert!(v.up_layer_matrix(0).is_some());
    }

    #[test]
    fn up_layer_matrix_returns_intermediate_by_hidden_view() {
        let tmp = tempfile::tempdir().unwrap();
        let bytes: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_features(tmp.path(), UP_FEATURES_BIN, &bytes);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_up_features(tmp.path()).unwrap();

        let view = v.up_layer_matrix(0).expect("layer 0 view");
        assert_eq!(view.shape(), &[2, 4]);
        assert_eq!(view[[0, 0]], 0.0);
        assert_eq!(view[[1, 3]], 7.0);
    }

    #[test]
    fn up_layer_matrix_none_when_mmap_not_loaded() {
        let v = vindex_with_layer_features(1, 2, 4);
        assert!(v.up_layer_matrix(0).is_none());
    }

    #[test]
    fn up_layer_matrix_none_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_features(tmp.path(), UP_FEATURES_BIN, &[1.0; 8]);
        // gate_vectors all None → num_features returns 0.
        let mut v = VectorIndex::empty(1, 4);
        v.load_up_features(tmp.path()).unwrap();
        assert!(v.up_layer_matrix(0).is_none());
    }

    #[test]
    fn up_layer_matrix_none_when_mmap_too_short() {
        let tmp = tempfile::tempdir().unwrap();
        write_features(tmp.path(), UP_FEATURES_BIN, &[1.0, 2.0]); // 8 bytes
        let mut v = vindex_with_layer_features(1, 4, 4);
        v.load_up_features(tmp.path()).unwrap();
        assert!(v.up_layer_matrix(0).is_none());
    }

    #[test]
    fn has_full_mmap_ffn_requires_both_up_and_down() {
        let tmp = tempfile::tempdir().unwrap();
        write_features(tmp.path(), UP_FEATURES_BIN, &[1.0; 8]);
        write_features(tmp.path(), DOWN_FEATURES_BIN, &[1.0; 8]);

        let mut v = vindex_with_layer_features(1, 2, 4);
        assert!(!v.has_full_mmap_ffn(), "neither loaded yet");

        v.load_up_features(tmp.path()).unwrap();
        assert!(!v.has_full_mmap_ffn(), "up only");

        v.load_down_features(tmp.path()).unwrap();
        assert!(v.has_full_mmap_ffn(), "both loaded");
    }
}
