//! Column-replace into `down_weights.bin`. Per-(layer, feature)
//! overrides splice a `[hidden]` vector across all hidden rows of
//! the target feature column.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::DOWN_WEIGHTS_BIN;

use super::{copy_for_patch, detect_down_dtype_bytes, BYTES_PER_F32};

/// Bake down overrides into `down_weights.bin` (per-layer
/// `[hidden, intermediate]` row-major, may be f16 or f32).
pub(in crate::executor::lifecycle::compile) fn patch_down_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    let src = source_dir.join(DOWN_WEIGHTS_BIN);
    let dst = dest_dir.join(DOWN_WEIGHTS_BIN);
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no down_weights.bin — cannot bake overrides".into(),
        ));
    }

    copy_for_patch(&src, &dst)?;

    let total = std::fs::metadata(&dst)
        .map_err(|e| LqlError::exec("stat down_weights.bin", e))?
        .len() as usize;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_layers;
    let elements_per_layer = hidden * intermediate;
    let total_elements = num_layers * elements_per_layer;

    let dtype_bytes = detect_down_dtype_bytes(total, total_elements)?;
    let layer_bytes = elements_per_layer * dtype_bytes;

    // Group overrides by layer so we only touch each layer's slab once.
    let mut by_layer: HashMap<usize, Vec<(usize, &Vec<f32>)>> = HashMap::new();
    for ((l, f), v) in overrides {
        by_layer.entry(*l).or_default().push((*f, v));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open down_weights.bin", e))?;

    let mut buf = vec![0u8; layer_bytes];

    for (layer, layer_overrides) in by_layer {
        let layer_offset = (layer * layer_bytes) as u64;
        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.read_exact(&mut buf)
            .map_err(|e| LqlError::exec("read down_weights slab", e))?;

        for (feature, down_vec) in layer_overrides {
            if down_vec.len() != hidden {
                return Err(LqlError::Execution(format!(
                    "down override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                    down_vec.len()
                )));
            }
            if feature >= intermediate {
                return Err(LqlError::Execution(format!(
                    "down override at L{layer} F{feature} out of range (intermediate = {intermediate})"
                )));
            }
            // Splice the column for `feature` across all `hidden` rows.
            for (row, val) in down_vec.iter().enumerate() {
                let cell = (row * intermediate + feature) * dtype_bytes;
                if dtype_bytes == BYTES_PER_F32 {
                    buf[cell..cell + BYTES_PER_F32].copy_from_slice(&val.to_le_bytes());
                } else {
                    let half_bits: u16 = larql_models::quant::half::f32_to_f16(*val);
                    buf[cell..cell + super::BYTES_PER_F16]
                        .copy_from_slice(&half_bits.to_le_bytes());
                }
            }
        }

        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.write_all(&buf)
            .map_err(|e| LqlError::exec("write down_weights slab", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests for the byte-level weight baker. These build a tiny
    //! synthetic `down_weights.bin` file with known contents, run
    //! `patch_down_weights` against it, then verify the override columns
    //! were spliced into the correct cells (and *only* those cells)
    //! without disturbing any other bytes.
    use super::*;

    fn mini_config(
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) -> larql_vindex::VindexConfig {
        larql_vindex::VindexConfig {
            version: 1,
            model: "test".into(),
            family: "test".into(),
            source: None,
            checksums: None,
            num_layers,
            hidden_size: hidden,
            intermediate_size: intermediate,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::All,
            dtype: larql_vindex::config::dtype::StorageDtype::F32,
            quant: larql_vindex::QuantFormat::None,
            layer_bands: None,
            layers: Vec::new(),
            down_top_k: 10,
            has_model_weights: true,
            model_config: None,
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        }
    }

    fn write_synthetic_f32(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * BYTES_PER_F32);
        for i in 0..total {
            let v = (i as f32) * 0.001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join(DOWN_WEIGHTS_BIN), &bytes).unwrap();
    }

    fn write_synthetic_f16(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * super::super::BYTES_PER_F16);
        for i in 0..total {
            let v = (i as f32) * 0.001;
            let half_bits = larql_models::quant::half::f32_to_f16(v);
            bytes.extend_from_slice(&half_bits.to_le_bytes());
        }
        std::fs::write(dir.join(DOWN_WEIGHTS_BIN), &bytes).unwrap();
    }

    fn read_column_f32(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join(DOWN_WEIGHTS_BIN)).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell = (layer * layer_elems + row * intermediate + feature) * BYTES_PER_F32;
            out.push(f32::from_le_bytes(
                bytes[cell..cell + BYTES_PER_F32].try_into().unwrap(),
            ));
        }
        out
    }

    fn read_column_f16(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join(DOWN_WEIGHTS_BIN)).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell =
                (layer * layer_elems + row * intermediate + feature) * super::super::BYTES_PER_F16;
            let bits = u16::from_le_bytes(
                bytes[cell..cell + super::super::BYTES_PER_F16]
                    .try_into()
                    .unwrap(),
            );
            out.push(larql_models::quant::half::f16_to_f32(bits));
        }
        out
    }

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_pdw_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn patch_down_weights_f32_writes_correct_columns() {
        let tmp = unique_dir("f32");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 4;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let layer = 2;
        let feature = 5;
        let down: Vec<f32> = (0..hidden).map(|r| 100.0 + r as f32).collect();
        overrides.insert((layer, feature), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_column_f32(&dst, layer, feature, hidden, intermediate);
        assert_eq!(read_back, down, "patched column doesn't match override");

        let untouched = read_column_f32(&dst, 0, feature, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((row * intermediate + feature) as f32) * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L0 F5 row {row}: got {val}, expected {expected}"
            );
        }

        let neighbour = read_column_f32(&dst, layer, feature - 1, hidden, intermediate);
        for (row, val) in neighbour.iter().enumerate() {
            let expected = ((layer * hidden * intermediate + row * intermediate + (feature - 1))
                as f32)
                * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L2 F4 row {row}: got {val}, expected {expected}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_f16_writes_correct_columns() {
        let tmp = unique_dir("f16");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 3;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f16(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let down: Vec<f32> = (0..hidden).map(|r| (r as f32) * 0.5 - 1.0).collect();
        overrides.insert((1, 7), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_column_f16(&dst, 1, 7, hidden, intermediate);
        for (i, (got, want)) in read_back.iter().zip(down.iter()).enumerate() {
            assert!(
                (got - want).abs() < 0.01,
                "row {i}: got {got}, expected {want}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_multiple_layers_and_features() {
        let tmp = unique_dir("multi");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 8;
        let hidden = 4;
        let intermediate = 8;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let cases = [(0, 0), (3, 5), (5, 2), (7, 7)];
        for (layer, feature) in cases {
            let v: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            overrides.insert((layer, feature), v);
        }

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        for (layer, feature) in cases {
            let read_back = read_column_f32(&dst, layer, feature, hidden, intermediate);
            let expected: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            assert_eq!(
                read_back, expected,
                "L{layer} F{feature} doesn't match override"
            );
        }

        let untouched = read_column_f32(&dst, 3, 0, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((3 * hidden * intermediate + row * intermediate) as f32) * 0.001;
            assert!((val - expected).abs() < 1e-6, "L3 F0 row {row} disturbed");
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_wrong_shape() {
        let tmp = unique_dir("bad");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 8, 8);
        write_synthetic_f32(&src, 2, 8, 8);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![0.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected wrong-shape override to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("wrong shape"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_unrecognised_dtype_size() {
        let tmp = unique_dir("dtype");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 4, 4);
        std::fs::write(src.join(DOWN_WEIGHTS_BIN), vec![0u8; 100]).unwrap();

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected mismatched dtype to error");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_missing_source_errors() {
        let tmp = unique_dir("missing");
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected missing source to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no down_weights.bin"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
