//! Shared embedding-table loader.
//!
//! Both the f32 and Q4_K loaders mmap `embeddings.bin`, auto-detect
//! f32 vs f16 storage from the byte count, decode to f32, and reshape
//! to `[vocab_size, hidden_size]`. The f32 loader additionally exposes
//! a "skip embed" path for `LoadWeightsOptions::skip_embed` — used by
//! FFN-service workers that never see token IDs and don't need the
//! embedding table in heap.

use std::path::Path;

use ndarray::Array2;

use crate::config::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::core::IndexLoadCallbacks;

/// Mmap + decode `embeddings.bin` into a `[vocab_size, hidden_size]`
/// array of f32. Auto-detects f32 vs f16 storage from the byte count.
///
/// Emits `on_file_start` / `on_file_done` callbacks around the work.
pub(super) fn load_embeddings(
    dir: &Path,
    config: &VindexConfig,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<Array2<f32>, VindexError> {
    callbacks.on_file_start(
        "embeddings",
        &dir.join(EMBEDDINGS_BIN).display().to_string(),
    );
    let embed_file = std::fs::File::open(dir.join(EMBEDDINGS_BIN))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    let expected_f32_bytes = config.vocab_size * config.hidden_size * 4;
    let embed_dtype = if embed_mmap.len() == expected_f32_bytes {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let arr = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);
    Ok(arr)
}

/// Empty-shape placeholder used by f32 loader's `skip_embed` path.
/// Pinned out as a function so the callsite is self-documenting and the
/// callback ordering matches the non-skipped path.
pub(super) fn empty_embeddings(callbacks: &mut dyn IndexLoadCallbacks) -> Array2<f32> {
    callbacks.on_file_start("embeddings (skipped)", "opts.skip_embed=true");
    let arr = Array2::<f32>::zeros((0, 0));
    callbacks.on_file_done("embeddings", 0, 0.0);
    arr
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::config::dtype::StorageDtype;
    use crate::config::types::QuantFormat;
    use std::cell::RefCell;

    /// Records callback order for assertions.
    #[derive(Default)]
    struct Recording {
        events: RefCell<Vec<String>>,
    }
    impl IndexLoadCallbacks for &Recording {
        fn on_file_start(&mut self, component: &str, path: &str) {
            self.events
                .borrow_mut()
                .push(format!("start:{component}:{path}"));
        }
        fn on_progress(&mut self, _: usize) {}
        fn on_file_done(&mut self, component: &str, records: usize, _: f64) {
            self.events
                .borrow_mut()
                .push(format!("done:{component}:{records}"));
        }
    }

    fn config_for(vocab: usize, hidden: usize) -> VindexConfig {
        VindexConfig {
            version: 2,
            model: "test/model".into(),
            family: "test".into(),
            num_layers: 2,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            vocab_size: vocab,
            embed_scale: 1.0,
            layers: Vec::new(),
            down_top_k: 1,
            has_model_weights: true,
            source: None,
            checksums: None,
            extract_level: crate::ExtractLevel::All,
            dtype: StorageDtype::F32,
            quant: QuantFormat::None,
            layer_bands: crate::LayerBands::for_family("test", 2),
            model_config: None,
            fp4: None,
            ffn_layout: None,
        }
    }

    fn write_embeddings_bin(dir: &std::path::Path, floats: &[f32], dtype: StorageDtype) {
        let bytes = crate::config::dtype::encode_floats(floats, dtype);
        std::fs::write(dir.join(EMBEDDINGS_BIN), &bytes).unwrap();
    }

    #[test]
    fn load_embeddings_decodes_f32_storage() {
        let tmp = tempfile::tempdir().unwrap();
        let vocab = 4;
        let hidden = 3;
        let floats: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F32);

        let config = config_for(vocab, hidden);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        assert_eq!(arr.shape(), &[vocab, hidden]);
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(arr[[3, 2]], 11.0);
        // Both callbacks fired, in order.
        let events = recording.events.borrow();
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("start:embeddings:"));
        assert_eq!(events[1], format!("done:embeddings:{vocab}"));
    }

    #[test]
    fn load_embeddings_falls_back_to_f16_on_byte_count_mismatch() {
        // f16 file has half the byte count of f32; the dtype detector
        // routes through StorageDtype::F16.
        let tmp = tempfile::tempdir().unwrap();
        let vocab = 2;
        let hidden = 4;
        let floats = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F16);

        let config = config_for(vocab, hidden);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        assert_eq!(arr.shape(), &[vocab, hidden]);
        // f16 loses some precision; check within tolerance.
        for (got, want) in arr.iter().zip(floats.iter()) {
            assert!((got - want).abs() < 1e-2, "f16 round-trip: {got} vs {want}");
        }
    }

    #[test]
    fn load_embeddings_errors_on_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = config_for(2, 2);
        let recording = Recording::default();
        let mut cb = &recording;
        let err = load_embeddings(tmp.path(), &config, &mut cb).expect_err("missing file errors");
        // I/O error wrapping; just confirm we got an error rather than
        // producing zero-filled garbage.
        assert!(
            err.to_string().to_lowercase().contains("no such")
                || err.to_string().to_lowercase().contains("not found")
                || err.to_string().to_lowercase().contains("os error")
        );
    }

    #[test]
    fn empty_embeddings_returns_zero_shape_and_fires_callbacks() {
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = empty_embeddings(&mut cb);
        assert_eq!(arr.shape(), &[0, 0]);

        let events = recording.events.borrow();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "start:embeddings (skipped):opts.skip_embed=true");
        assert_eq!(events[1], "done:embeddings:0");
    }
}
