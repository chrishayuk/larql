//! Interleaved FFN data — `[gate|up|down]` packed per layer in one
//! contiguous f32 file (`interleaved.bin`).
//!
//! Eliminates TLB thrash from three separate mmap files. Per-layer
//! prefetch lets a forward pass tell the kernel which layer's bytes
//! are about to be read.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::INTERLEAVED_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    /// Eliminates TLB thrash from 3 separate mmap files.
    pub fn load_interleaved(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "interleaved.bin not found. Run: cargo run --release -p larql-vindex --example build_interleaved -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: per-layer prefetch issued at query time via prefetch_interleaved_layer.
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });
        Arc::make_mut(&mut self.storage).set_interleaved_f32(mmap);
        Ok(())
    }

    /// Whether interleaved FFN data is loaded.
    pub fn has_interleaved(&self) -> bool {
        self.storage.has_interleaved_f32()
    }

    /// Get gate matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let bytes = self.storage.interleaved_f32_view()?;
        let mmap: &[u8] = bytes.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3); // gate is first
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get up matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let bytes = self.storage.interleaved_f32_view()?;
        let mmap: &[u8] = bytes.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3) + matrix_bytes; // up is second
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get down matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let bytes = self.storage.interleaved_f32_view()?;
        let mmap: &[u8] = bytes.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3) + matrix_bytes * 2; // down is third
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Prefetch next layer's interleaved data into page cache. Unix
    /// only; on Windows the function is a no-op because `madvise`
    /// isn't available and the OS handles readahead itself.
    #[cfg_attr(not(unix), allow(unused_variables))]
    pub fn prefetch_interleaved_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(bytes) = self.storage.interleaved_f32_view() {
            let mmap: &[u8] = bytes.as_ref();
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return;
            }
            let matrix_bytes = intermediate * self.hidden_size * 4;
            let layer_bytes = matrix_bytes * 3;
            let start = self.ffn_layer_byte_offset(layer, 3);
            let end = (start + layer_bytes).min(mmap.len());
            if start >= mmap.len() {
                return;
            }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! Round-trip coverage of the interleaved [gate|up|down] layer
    //! reader. Mirrors the test pattern in `down.rs`/`up.rs`.
    use ndarray::Array2;

    use super::*;

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

    /// Write `[gate | up | down]` for one layer as raw f32 bytes.
    fn write_interleaved(dir: &std::path::Path, gate: &[f32], up: &[f32], down: &[f32]) {
        let mut bytes: Vec<u8> = Vec::new();
        for f in gate.iter().chain(up).chain(down) {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(dir.join(INTERLEAVED_BIN), &bytes).unwrap();
    }

    #[test]
    fn load_interleaved_errors_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut v = vindex_with_layer_features(1, 2, 4);
        let err = v
            .load_interleaved(tmp.path())
            .expect_err("missing file errors");
        assert!(err.to_string().contains("interleaved.bin"));
    }

    #[test]
    fn load_interleaved_populates_mmap_handle() {
        let tmp = tempfile::tempdir().unwrap();
        write_interleaved(tmp.path(), &[1.0; 8], &[2.0; 8], &[3.0; 8]);

        let mut v = vindex_with_layer_features(1, 2, 4);
        assert!(!v.has_interleaved());
        v.load_interleaved(tmp.path()).unwrap();
        assert!(v.has_interleaved());
    }

    #[test]
    fn interleaved_gate_returns_first_third_of_layer() {
        // intermediate=2, hidden=4 → each matrix is 2 × 4 = 8 floats.
        let tmp = tempfile::tempdir().unwrap();
        let gate: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let up: Vec<f32> = (8..16).map(|i| i as f32).collect();
        let down: Vec<f32> = (16..24).map(|i| i as f32).collect();
        write_interleaved(tmp.path(), &gate, &up, &down);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_interleaved(tmp.path()).unwrap();

        let m = v.interleaved_gate(0).expect("gate present");
        assert_eq!(m.shape(), &[2, 4]);
        assert_eq!(m[[0, 0]], 0.0);
        assert_eq!(m[[1, 3]], 7.0);
    }

    #[test]
    fn interleaved_up_returns_second_third_of_layer() {
        let tmp = tempfile::tempdir().unwrap();
        let gate: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let up: Vec<f32> = (8..16).map(|i| i as f32).collect();
        let down: Vec<f32> = (16..24).map(|i| i as f32).collect();
        write_interleaved(tmp.path(), &gate, &up, &down);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_interleaved(tmp.path()).unwrap();

        let m = v.interleaved_up(0).expect("up present");
        assert_eq!(m[[0, 0]], 8.0);
        assert_eq!(m[[1, 3]], 15.0);
    }

    #[test]
    fn interleaved_down_returns_third_third_of_layer() {
        let tmp = tempfile::tempdir().unwrap();
        let gate: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let up: Vec<f32> = (8..16).map(|i| i as f32).collect();
        let down: Vec<f32> = (16..24).map(|i| i as f32).collect();
        write_interleaved(tmp.path(), &gate, &up, &down);

        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_interleaved(tmp.path()).unwrap();

        let m = v.interleaved_down(0).expect("down present");
        assert_eq!(m[[0, 0]], 16.0);
        assert_eq!(m[[1, 3]], 23.0);
    }

    #[test]
    fn interleaved_accessors_none_when_mmap_not_loaded() {
        let v = vindex_with_layer_features(1, 2, 4);
        assert!(v.interleaved_gate(0).is_none());
        assert!(v.interleaved_up(0).is_none());
        assert!(v.interleaved_down(0).is_none());
    }

    #[test]
    fn interleaved_accessors_none_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_interleaved(tmp.path(), &[1.0; 8], &[2.0; 8], &[3.0; 8]);
        // gate_vectors[*] = None → num_features returns 0 for all layers.
        let mut v = VectorIndex::empty(1, 4);
        v.load_interleaved(tmp.path()).unwrap();
        assert!(v.interleaved_gate(0).is_none());
        assert!(v.interleaved_up(0).is_none());
        assert!(v.interleaved_down(0).is_none());
    }

    #[test]
    fn interleaved_accessors_none_when_mmap_too_short() {
        // Vindex expects 4 features × 4 hidden = 16 floats × 3 matrices
        // = 192 bytes per layer, but we write only 8 floats = 32 bytes.
        let tmp = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..8_u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        std::fs::write(tmp.path().join(INTERLEAVED_BIN), &bytes).unwrap();

        let mut v = vindex_with_layer_features(1, 4, 4);
        v.load_interleaved(tmp.path()).unwrap();
        assert!(v.interleaved_gate(0).is_none());
        assert!(v.interleaved_up(0).is_none());
        assert!(v.interleaved_down(0).is_none());
    }

    #[test]
    fn prefetch_interleaved_layer_safe_with_no_mmap() {
        let v = vindex_with_layer_features(1, 2, 4);
        // No mmap loaded → must not panic.
        v.prefetch_interleaved_layer(0);
    }

    #[test]
    fn prefetch_interleaved_layer_safe_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_interleaved(tmp.path(), &[1.0; 8], &[2.0; 8], &[3.0; 8]);
        let mut v = VectorIndex::empty(1, 4);
        v.load_interleaved(tmp.path()).unwrap();
        // num_features returns 0 → early return; must not crash.
        v.prefetch_interleaved_layer(0);
    }

    #[test]
    fn prefetch_interleaved_layer_safe_when_offset_past_end() {
        // Layer 99 is past the file → start >= mmap.len() guard.
        let tmp = tempfile::tempdir().unwrap();
        write_interleaved(tmp.path(), &[1.0; 8], &[2.0; 8], &[3.0; 8]);
        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_interleaved(tmp.path()).unwrap();
        v.prefetch_interleaved_layer(99);
    }

    #[test]
    fn prefetch_interleaved_layer_runs_on_in_range_layer() {
        // Smoke test the happy path — must not panic.
        let tmp = tempfile::tempdir().unwrap();
        write_interleaved(tmp.path(), &[1.0; 8], &[2.0; 8], &[3.0; 8]);
        let mut v = vindex_with_layer_features(1, 2, 4);
        v.load_interleaved(tmp.path()).unwrap();
        v.prefetch_interleaved_layer(0);
    }
}
