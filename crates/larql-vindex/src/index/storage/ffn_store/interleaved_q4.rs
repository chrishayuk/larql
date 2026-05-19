//! Q4_0 interleaved FFN data (`interleaved_q4.bin`).
//!
//! Loaders + per-component dequant. Q4_K/Q6_K (the Ollama-compatible
//! variant) lives in the sibling `interleaved_kquant.rs`; this file is
//! the predecessor format used before the K-quant rollout.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::INTERLEAVED_Q4_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load Q4_0 interleaved FFN data.
    pub fn load_interleaved_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });
        Arc::make_mut(&mut self.storage).set_interleaved_q4(mmap);
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.storage.has_interleaved_q4()
    }

    /// Dequantize one matrix from Q4 interleaved file → f32 Array2.
    /// component: 0=gate, 1=up, 2=down
    fn dequant_q4_matrix(&self, layer: usize, component: usize) -> Option<ndarray::Array2<f32>> {
        let mmap_view = self.storage.interleaved_q4_whole_buffer_view()?;
        let mmap: &[u8] = mmap_view.as_ref();
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }

        let floats_per_matrix = intermediate * self.hidden_size;
        let q4_bytes_per_matrix = floats_per_matrix / larql_models::quant::ggml::Q4_0_BLOCK_ELEMS
            * larql_models::quant::ggml::Q4_0_BLOCK_BYTES;
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

        let start = layer * q4_bytes_per_layer + component * q4_bytes_per_matrix;
        let end = start + q4_bytes_per_matrix;
        if end > mmap.len() {
            return None;
        }

        let q4_data = &mmap[start..end];
        let floats = larql_models::quant::ggml::dequantize_q4_0(q4_data, floats_per_matrix).ok()?;
        ndarray::Array2::from_shape_vec((intermediate, self.hidden_size), floats).ok()
    }

    /// Get gate matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 0)
    }

    /// Get up matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 1)
    }

    /// Get down matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 2)
    }

    /// Prefetch next layer's Q4 data. Unix only; no-op on Windows
    /// where `madvise` isn't available.
    #[cfg_attr(not(unix), allow(unused_variables))]
    pub fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(mmap_view) = self.storage.interleaved_q4_whole_buffer_view() {
            let mmap: &[u8] = mmap_view.as_ref();
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return;
            }
            let q4_bytes_per_matrix = intermediate * self.hidden_size
                / larql_models::quant::ggml::Q4_0_BLOCK_ELEMS
                * larql_models::quant::ggml::Q4_0_BLOCK_BYTES;
            let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
            let start = layer * q4_bytes_per_layer;
            let end = (start + q4_bytes_per_layer).min(mmap.len());
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

#[cfg(test)]
mod tests {
    //! Round-trip coverage of the Q4_0 interleaved [gate|up|down]
    //! reader. Uses real `quantize_q4_0` so the dequant chain runs
    //! end-to-end and we can pin the output is within Q4 noise of
    //! the source floats.
    use ndarray::Array2;

    use super::*;

    const Q4_0_BLOCK_ELEMS: usize = larql_models::quant::ggml::Q4_0_BLOCK_ELEMS;

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

    /// Build per-layer Q4_0 interleaved bytes for `[gate | up | down]`.
    /// Each matrix is `intermediate × hidden` floats (must be a multiple
    /// of `Q4_0_BLOCK_ELEMS`).
    fn write_q4_interleaved(dir: &std::path::Path, layers: &[(Vec<f32>, Vec<f32>, Vec<f32>)]) {
        let mut bytes = Vec::new();
        for (gate, up, down) in layers {
            for matrix in [gate, up, down] {
                let q = larql_models::quant::ggml::quantize_q4_0(matrix);
                bytes.extend_from_slice(&q);
            }
        }
        std::fs::write(dir.join(INTERLEAVED_Q4_BIN), &bytes).unwrap();
    }

    #[test]
    fn load_interleaved_q4_errors_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        let err = v
            .load_interleaved_q4(tmp.path())
            .expect_err("missing file errors");
        assert!(err.to_string().contains("interleaved_q4.bin"));
    }

    #[test]
    fn load_interleaved_q4_populates_mmap_handle() {
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.1_f32; Q4_0_BLOCK_ELEMS];
        write_q4_interleaved(tmp.path(), &[(mat.clone(), mat.clone(), mat)]);

        let mut v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        assert!(!v.has_interleaved_q4());
        v.load_interleaved_q4(tmp.path()).unwrap();
        assert!(v.has_interleaved_q4());
    }

    #[test]
    fn interleaved_q4_gate_round_trips_within_quant_noise() {
        // intermediate=1, hidden=Q4_0_BLOCK_ELEMS=32 → exactly one
        // Q4_0 block per matrix. Q4_0 carries roughly 4-bit precision
        // per scaled value; tolerance ~5% of max abs is generous.
        let hidden = Q4_0_BLOCK_ELEMS;
        let gate: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.01).collect();
        let up: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.02).collect();
        let down: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.03).collect();

        let tmp = tempfile::tempdir().unwrap();
        write_q4_interleaved(tmp.path(), &[(gate.clone(), up.clone(), down.clone())]);

        let mut v = vindex_with_layer_features(1, 1, hidden);
        v.load_interleaved_q4(tmp.path()).unwrap();

        let g = v.interleaved_q4_gate(0).expect("gate dequant");
        assert_eq!(g.shape(), &[1, hidden]);
        // Pin first and last positions roundtrip within Q4 noise.
        for i in [0, 1, hidden - 1] {
            let want = gate[i];
            let got = g[[0, i]];
            assert!(
                (got - want).abs() <= 0.05,
                "gate[{i}] {got} vs {want} > 0.05 Q4 noise tolerance"
            );
        }
    }

    #[test]
    fn interleaved_q4_up_addresses_second_matrix() {
        let hidden = Q4_0_BLOCK_ELEMS;
        let gate = vec![0.0_f32; hidden];
        let up: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.04).collect();
        let down = vec![0.0_f32; hidden];

        let tmp = tempfile::tempdir().unwrap();
        write_q4_interleaved(tmp.path(), &[(gate, up.clone(), down)]);

        let mut v = vindex_with_layer_features(1, 1, hidden);
        v.load_interleaved_q4(tmp.path()).unwrap();
        let m = v.interleaved_q4_up(0).expect("up dequant");
        // Inner value ≠ 0 → the dispatch addressed the second matrix slot.
        assert!(m[[0, 10]].abs() > 0.05);
    }

    #[test]
    fn interleaved_q4_down_addresses_third_matrix() {
        let hidden = Q4_0_BLOCK_ELEMS;
        let gate = vec![0.0_f32; hidden];
        let up = vec![0.0_f32; hidden];
        let down: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.04).collect();

        let tmp = tempfile::tempdir().unwrap();
        write_q4_interleaved(tmp.path(), &[(gate, up, down.clone())]);

        let mut v = vindex_with_layer_features(1, 1, hidden);
        v.load_interleaved_q4(tmp.path()).unwrap();
        let m = v.interleaved_q4_down(0).expect("down dequant");
        assert!(m[[0, 10]].abs() > 0.05);
    }

    #[test]
    fn interleaved_q4_accessors_none_when_mmap_not_loaded() {
        let v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        assert!(v.interleaved_q4_gate(0).is_none());
        assert!(v.interleaved_q4_up(0).is_none());
        assert!(v.interleaved_q4_down(0).is_none());
    }

    #[test]
    fn interleaved_q4_accessors_none_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.0_f32; Q4_0_BLOCK_ELEMS];
        write_q4_interleaved(tmp.path(), &[(mat.clone(), mat.clone(), mat)]);
        // gate_vectors all None → num_features returns 0.
        let mut v = VectorIndex::empty(1, Q4_0_BLOCK_ELEMS);
        v.load_interleaved_q4(tmp.path()).unwrap();
        assert!(v.interleaved_q4_gate(0).is_none());
    }

    #[test]
    fn interleaved_q4_accessors_none_when_mmap_too_short() {
        // Vindex thinks each layer is 1 × 32 floats × 3 matrices = 54 Q4
        // bytes per layer. Write only 18 bytes (one matrix).
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.1_f32; Q4_0_BLOCK_ELEMS];
        let q = larql_models::quant::ggml::quantize_q4_0(&mat);
        std::fs::write(tmp.path().join(INTERLEAVED_Q4_BIN), &q).unwrap();

        let mut v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        v.load_interleaved_q4(tmp.path()).unwrap();
        // gate fits (one matrix), up/down do not.
        assert!(v.interleaved_q4_up(0).is_none());
        assert!(v.interleaved_q4_down(0).is_none());
    }

    #[test]
    fn prefetch_interleaved_q4_layer_safe_with_no_mmap() {
        let v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        v.prefetch_interleaved_q4_layer(0);
    }

    #[test]
    fn prefetch_interleaved_q4_layer_safe_when_intermediate_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.0_f32; Q4_0_BLOCK_ELEMS];
        write_q4_interleaved(tmp.path(), &[(mat.clone(), mat.clone(), mat)]);
        let mut v = VectorIndex::empty(1, Q4_0_BLOCK_ELEMS);
        v.load_interleaved_q4(tmp.path()).unwrap();
        v.prefetch_interleaved_q4_layer(0);
    }

    #[test]
    fn prefetch_interleaved_q4_layer_safe_past_end() {
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.0_f32; Q4_0_BLOCK_ELEMS];
        write_q4_interleaved(tmp.path(), &[(mat.clone(), mat.clone(), mat)]);
        let mut v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        v.load_interleaved_q4(tmp.path()).unwrap();
        v.prefetch_interleaved_q4_layer(99);
    }

    #[test]
    fn prefetch_interleaved_q4_layer_runs_on_in_range_layer() {
        let tmp = tempfile::tempdir().unwrap();
        let mat = vec![0.0_f32; Q4_0_BLOCK_ELEMS];
        write_q4_interleaved(tmp.path(), &[(mat.clone(), mat.clone(), mat)]);
        let mut v = vindex_with_layer_features(1, 1, Q4_0_BLOCK_ELEMS);
        v.load_interleaved_q4(tmp.path()).unwrap();
        v.prefetch_interleaved_q4_layer(0);
    }
}
