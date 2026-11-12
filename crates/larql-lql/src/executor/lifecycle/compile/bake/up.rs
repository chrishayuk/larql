//! Bake up overlay entries into `up_weights.bin` (or whichever file
//! the manifest points the up_proj tensors at).
//!
//! Dense FFN at inference reads via `load_model_weights`, which
//! consults `weight_manifest.json` to find each tensor's `(file,
//! offset, length, shape)` entry. Layout per layer:
//!
//! - the file the manifest points to (normally `up_weights.bin`)
//! - per-layer tensor at `entry.offset` with `entry.length` bytes
//! - dtype inferred from `byte_count / expected_floats` (4 = f32,
//!   2 = f16)
//! - shape `[num_features, hidden_size]` row-major; feature `f`'s
//!   row starts at `entry.offset + f × hidden × bpf`
//!
//! `up_features.bin` (a separate feature-major f32 file used only by
//! `walk_ffn_sparse`) is NOT touched here. Writing to the wrong file
//! was the root cause of `refine_demo22`'s regression from 8/10 to
//! 0/10 compiled retrieval.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::WEIGHT_MANIFEST_JSON;

use super::{copy_for_patch, BYTES_PER_F16, BYTES_PER_F32};

/// Manifest key fragment that identifies an up_proj tensor entry.
/// Used to filter `weight_manifest.json` while searching for the
/// per-layer up tensors.
const UP_PROJ_KEY_FRAGMENT: &str = "up_proj";

/// Manifest key fragment that prefixes the layer index in tensor
/// keys (`layers.{N}.…`).
const LAYERS_KEY_FRAGMENT: &str = "layers.";

pub(in crate::executor::lifecycle::compile) fn patch_up_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    up_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if up_overrides.is_empty() {
        return Ok(());
    }

    // Read the weight manifest from the SOURCE vindex — the dest
    // copy was hard-linked from source and we haven't modified the
    // manifest.
    let manifest_path = source_dir.join(WEIGHT_MANIFEST_JSON);
    if !manifest_path.exists() {
        // No manifest means no model weights: a source that declares
        // weights without a manifest is refused before the bake starts.
        // There is no up tensor to write, and nothing that would read one.
        return Ok(());
    }
    let layer_up_lookup = parse_manifest_for_up(&manifest_path)?;
    // The architecture names its own up tensor (`…mlp.up_proj.weight`,
    // `…ffn.w3.weight`, `…ffn.up.weight`). A source recorded before
    // `model_config` has no architecture to ask, and falls back to the
    // one tensor of the layer whose name carries `up_proj`.
    let arch = larql_vindex::arch_from_vindex_config(config);
    let tensors = match &arch {
        Some(_) => parse_manifest_tensors(&manifest_path)?,
        None => HashMap::new(),
    };

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let expected_floats = intermediate * hidden;

    // File handles cached per file so we don't re-open per write.
    let mut file_cache: HashMap<String, std::fs::File> = HashMap::new();

    for ((layer, feature), up_vec) in up_overrides {
        if up_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                up_vec.len()
            )));
        }
        if *feature >= intermediate {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} out of range (intermediate = {intermediate})"
            )));
        }

        let located = match &arch {
            Some(arch) => tensors.get(&arch.ffn_up_key(*layer)),
            None => layer_up_lookup.get(layer),
        };
        let Some((file_name, offset, length)) = located else {
            // The install's up vector has nowhere to go, and an artifact
            // without it would run the slot on the source's up row.
            let wanted = match &arch {
                Some(arch) => format!("the tensor {:?}", arch.ffn_up_key(*layer)),
                None => format!("a single {UP_PROJ_KEY_FRAGMENT} tensor for L{layer}"),
            };
            return Err(LqlError::Execution(format!(
                "{WEIGHT_MANIFEST_JSON} does not locate {wanted}, so the up vector at \
                 L{layer} F{feature} cannot be baked"
            )));
        };

        let bpf = if *length as usize == expected_floats * BYTES_PER_F32 {
            BYTES_PER_F32
        } else if *length as usize == expected_floats * BYTES_PER_F16 {
            BYTES_PER_F16
        } else {
            return Err(LqlError::Execution(format!(
                "up weight for L{layer} has length {length} ≠ \
                 expected {} (f32) or {} (f16)",
                expected_floats * BYTES_PER_F32,
                expected_floats * BYTES_PER_F16,
            )));
        };

        // Lazily open + copy the file if we haven't touched it yet.
        if !file_cache.contains_key(file_name) {
            let src = source_dir.join(file_name);
            let dst = dest_dir.join(file_name);
            if !src.exists() {
                return Err(LqlError::Execution(format!(
                    "weight file {file_name} referenced by manifest but missing from source"
                )));
            }
            copy_for_patch(&src, &dst)?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&dst)
                .map_err(|e| LqlError::exec(format!("open {file_name}"), e))?;
            file_cache.insert(file_name.clone(), f);
        }
        let file = file_cache.get_mut(file_name).unwrap();

        let row_bytes = hidden * bpf;
        let mut row_buf = vec![0u8; row_bytes];
        encode_row(up_vec, bpf, &mut row_buf);

        let feature_offset = offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec(format!("seek {file_name}"), e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec(format!("write {file_name} row"), e))?;
    }
    Ok(())
}

/// Walk the manifest JSON and collect `layer → (file, offset, length)`
/// for every up_proj tensor. Entries without all four fields, or with
/// a key that doesn't match the `layers.{N}.…up_proj` pattern, are
/// silently skipped (works for Llama/Gemma/Mistral; MoE experts and
/// non-standard naming conventions skip and the layer's overlay
/// becomes a silent no-op).
fn parse_manifest_for_up(
    manifest_path: &std::path::Path,
) -> Result<HashMap<usize, (String, u64, u64)>, LqlError> {
    let text = std::fs::read_to_string(manifest_path)
        .map_err(|e| LqlError::exec("read weight_manifest.json", e))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&text).map_err(|e| LqlError::exec("parse weight_manifest.json", e))?;

    let mut out: HashMap<usize, (String, u64, u64)> = HashMap::new();
    let mut ambiguous: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for entry in &entries {
        let Some(key) = entry.get("key").and_then(|v| v.as_str()) else {
            continue;
        };
        if !key.contains(UP_PROJ_KEY_FRAGMENT) {
            continue;
        }
        let Some(file) = entry.get("file").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(offset) = entry.get("offset").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(length) = entry.get("length").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(layer) = parse_layer_from_key(key) else {
            continue;
        };
        // Two tensors of one layer both named `…up_proj…` (per-expert
        // weights) leave no single row to write. The layer is dropped from
        // the lookup, and an override that needs it is refused by name.
        if ambiguous.contains(&layer) {
            continue;
        }
        if out.remove(&layer).is_some() {
            ambiguous.insert(layer);
            continue;
        }
        out.insert(layer, (file.to_string(), offset, length));
    }
    Ok(out)
}

/// Every manifest tensor that says where its bytes are, by key.
fn parse_manifest_tensors(
    manifest_path: &std::path::Path,
) -> Result<HashMap<String, (String, u64, u64)>, LqlError> {
    let text = std::fs::read_to_string(manifest_path)
        .map_err(|e| LqlError::exec("read weight_manifest.json", e))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&text).map_err(|e| LqlError::exec("parse weight_manifest.json", e))?;
    let mut out = HashMap::new();
    for entry in &entries {
        let field = |name: &str| entry.get(name);
        if let (Some(key), Some(file), Some(offset), Some(length)) = (
            field("key").and_then(|v| v.as_str()),
            field("file").and_then(|v| v.as_str()),
            field("offset").and_then(|v| v.as_u64()),
            field("length").and_then(|v| v.as_u64()),
        ) {
            out.insert(key.to_string(), (file.to_string(), offset, length));
        }
    }
    Ok(out)
}

/// Extract the layer index from a manifest key like
/// `model.layers.{N}.mlp.up_proj.weight` → `N`. Returns `None` if
/// the key doesn't follow the `layers.{int}.…` shape.
fn parse_layer_from_key(key: &str) -> Option<usize> {
    let rest = key.split(LAYERS_KEY_FRAGMENT).nth(1)?;
    let layer_str = rest.split('.').next()?;
    layer_str.parse::<usize>().ok()
}

fn encode_row(up_vec: &[f32], bpf: usize, row_buf: &mut [u8]) {
    if bpf == BYTES_PER_F32 {
        for (i, v) in up_vec.iter().enumerate() {
            row_buf[i * BYTES_PER_F32..(i + 1) * BYTES_PER_F32].copy_from_slice(&v.to_le_bytes());
        }
    } else {
        for (i, v) in up_vec.iter().enumerate() {
            let half_bits = larql_models::quant::half::f32_to_f16(*v);
            row_buf[i * BYTES_PER_F16..(i + 1) * BYTES_PER_F16]
                .copy_from_slice(&half_bits.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_from_key_extracts_int() {
        assert_eq!(parse_layer_from_key("model.layers.7.mlp.up_proj"), Some(7));
        assert_eq!(parse_layer_from_key("model.layers.0.mlp.up_proj"), Some(0));
        assert_eq!(
            parse_layer_from_key("model.layers.31.mlp.up_proj.weight"),
            Some(31)
        );
    }

    #[test]
    fn parse_layer_from_key_rejects_non_layer_keys() {
        assert_eq!(parse_layer_from_key("model.embed_tokens"), None);
        assert_eq!(parse_layer_from_key("layers.x.mlp"), None);
        assert_eq!(parse_layer_from_key("nothing here"), None);
    }

    #[test]
    fn encode_row_f32_writes_each_cell() {
        let mut buf = vec![0u8; 3 * BYTES_PER_F32];
        encode_row(&[1.5, -2.0, 0.25], BYTES_PER_F32, &mut buf);
        for (i, want) in [1.5f32, -2.0, 0.25].iter().enumerate() {
            let cell = i * BYTES_PER_F32;
            let got = f32::from_le_bytes(buf[cell..cell + BYTES_PER_F32].try_into().unwrap());
            assert_eq!(got, *want);
        }
    }

    #[test]
    fn encode_row_f16_writes_each_cell() {
        let mut buf = vec![0u8; 3 * BYTES_PER_F16];
        encode_row(&[1.0, 2.0, 3.0], BYTES_PER_F16, &mut buf);
        for (i, want) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            let cell = i * BYTES_PER_F16;
            let bits = u16::from_le_bytes(buf[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!((got - want).abs() < 0.01);
        }
    }

    #[test]
    fn key_fragments_match_actual_manifest_shape() {
        // Pinned: parse_manifest_for_up depends on these exact shapes.
        assert_eq!(UP_PROJ_KEY_FRAGMENT, "up_proj");
        assert_eq!(LAYERS_KEY_FRAGMENT, "layers.");
    }

    // ── End-to-end fixture tests ────────────────────────────

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_bake_up_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

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

    /// Lay out a synthetic up_weights.bin with `num_layers` tensors of
    /// shape `[intermediate, hidden]` packed back-to-back. Returns the
    /// manifest entries so callers can write `weight_manifest.json`
    /// matching the layout.
    fn write_synthetic_up_weights_bin(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
        bpf: usize,
    ) -> Vec<(String, u64, u64)> {
        let row_bytes = hidden * bpf;
        let layer_bytes = intermediate * row_bytes;
        let total = num_layers * layer_bytes;
        let mut bytes: Vec<u8> = Vec::with_capacity(total);

        for layer in 0..num_layers {
            for feat in 0..intermediate {
                for d in 0..hidden {
                    let v = (layer * 100 + feat * 10 + d) as f32 * 0.001;
                    if bpf == BYTES_PER_F32 {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    } else {
                        let h = larql_models::quant::half::f32_to_f16(v);
                        bytes.extend_from_slice(&h.to_le_bytes());
                    }
                }
            }
        }
        std::fs::write(dir.join("up_weights.bin"), &bytes).unwrap();

        let mut entries = Vec::new();
        for layer in 0..num_layers {
            entries.push((
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                (layer * layer_bytes) as u64,
                layer_bytes as u64,
            ));
        }
        entries
    }

    fn write_manifest(dir: &std::path::Path, file: &str, entries: &[(String, u64, u64)]) {
        let json: Vec<serde_json::Value> = entries
            .iter()
            .map(|(key, off, len)| {
                serde_json::json!({
                    "key": key,
                    "file": file,
                    "offset": off,
                    "length": len,
                })
            })
            .collect();
        std::fs::write(
            dir.join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&json).unwrap(),
        )
        .unwrap();
    }

    fn read_up_row_f32(
        dir: &std::path::Path,
        layer: usize,
        feat: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join("up_weights.bin")).unwrap();
        let layer_bytes = intermediate * hidden * BYTES_PER_F32;
        let row_bytes = hidden * BYTES_PER_F32;
        let row_start = layer * layer_bytes + feat * row_bytes;
        (0..hidden)
            .map(|d| {
                let cell = row_start + d * BYTES_PER_F32;
                f32::from_le_bytes(bytes[cell..cell + BYTES_PER_F32].try_into().unwrap())
            })
            .collect()
    }

    #[test]
    fn patch_up_weights_no_overrides_is_noop() {
        let dir = unique_dir("noop");
        std::fs::create_dir_all(&dir).unwrap();
        // Empty src and dst dirs — no manifest, no up_weights.bin needed
        // since the function returns early on empty overrides.
        let cfg = mini_config(2, 4, 4);
        let result = patch_up_weights(&dir, &dir, &cfg, &HashMap::new());
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_skips_silently_when_manifest_absent() {
        let dir = unique_dir("no_manifest");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        // No weight_manifest.json in src — function should silently
        // succeed (skipping the bake) rather than error.
        let result = patch_up_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_ok(), "expected silent skip, got {result:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_f32_writes_correct_row() {
        let dir = unique_dir("f32");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 3;
        let hidden = 4;
        let intermediate = 4;
        let entries =
            write_synthetic_up_weights_bin(&src, num_layers, hidden, intermediate, BYTES_PER_F32);
        write_manifest(&src, "up_weights.bin", &entries);

        let cfg = mini_config(num_layers, hidden, intermediate);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let layer = 1;
        let feat = 2;
        let new_row: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0];
        overrides.insert((layer, feat), new_row.clone());

        patch_up_weights(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_up_row_f32(&dst, layer, feat, hidden, intermediate);
        assert_eq!(read_back, new_row, "patched row mismatched override");

        // Adjacent feature row (feat-1) untouched.
        let neighbour = read_up_row_f32(&dst, layer, feat - 1, hidden, intermediate);
        for (d, val) in neighbour.iter().enumerate() {
            let expected = (layer * 100 + (feat - 1) * 10 + d) as f32 * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "neighbour row d={d} mismatched"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_f16_round_trips_within_tolerance() {
        let dir = unique_dir("f16");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 2;
        let hidden = 4;
        let intermediate = 4;
        let entries =
            write_synthetic_up_weights_bin(&src, num_layers, hidden, intermediate, BYTES_PER_F16);
        write_manifest(&src, "up_weights.bin", &entries);

        let cfg = mini_config(num_layers, hidden, intermediate);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let new_row: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5];
        overrides.insert((0, 1), new_row.clone());

        patch_up_weights(&src, &dst, &cfg, &overrides).unwrap();

        // Read back as f16.
        let bytes = std::fs::read(dst.join("up_weights.bin")).unwrap();
        let row_bytes = hidden * BYTES_PER_F16;
        // Layer 0, feature 1 → byte offset = (0 * intermediate + 1) * row_bytes.
        let row_start = row_bytes;
        for (d, want) in new_row.iter().enumerate() {
            let cell = row_start + d * BYTES_PER_F16;
            let bits = u16::from_le_bytes(bytes[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!(
                (got - want).abs() < 0.01,
                "f16 d={d}: got {got}, want {want}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_rejects_wrong_shape() {
        let dir = unique_dir("wrong_shape");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let entries = write_synthetic_up_weights_bin(&src, 2, 4, 4, BYTES_PER_F32);
        write_manifest(&src, "up_weights.bin", &entries);

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        // Wrong length: 2 instead of 4.
        overrides.insert((0, 0), vec![1.0, 2.0]);

        let err = patch_up_weights(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("wrong shape"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_rejects_out_of_range_feature() {
        let dir = unique_dir("oor_feat");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let entries = write_synthetic_up_weights_bin(&src, 2, 4, 4, BYTES_PER_F32);
        write_manifest(&src, "up_weights.bin", &entries);

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        // Feature index 99 — intermediate is 4.
        overrides.insert((0, 99), vec![0.0; 4]);

        let err = patch_up_weights(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("out of range"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_refuses_a_layer_the_manifest_does_not_locate() {
        let dir = unique_dir("skip_layer");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        // Manifest only describes layer 0 — layer 1 has no entry.
        let mut entries = write_synthetic_up_weights_bin(&src, 2, 4, 4, BYTES_PER_F32);
        entries.truncate(1);
        write_manifest(&src, "up_weights.bin", &entries);

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((1, 0), vec![1.0; 4]); // layer 1 — no manifest entry

        // An up vector with nowhere to go is refused, not dropped.
        let err = patch_up_weights(&src, &dst, &cfg, &overrides)
            .unwrap_err()
            .to_string();
        assert!(err.contains("L1 F0"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_up_weights_rejects_unrecognised_dtype() {
        let dir = unique_dir("dtype");
        let src = dir.join("src");
        let dst = dir.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(1, 4, 4);
        // Length matches neither f32 (64) nor f16 (32) for one 16-element tensor.
        let entries = vec![("model.layers.0.mlp.up_proj.weight".to_string(), 0u64, 50u64)];
        std::fs::write(src.join("up_weights.bin"), vec![0u8; 50]).unwrap();
        write_manifest(&src, "up_weights.bin", &entries);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![0.0; 4]);
        let err = patch_up_weights(&src, &dst, &cfg, &overrides).unwrap_err();
        assert!(err.to_string().contains("expected"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_manifest_for_up_filters_to_up_proj_entries() {
        let dir = unique_dir("parse");
        std::fs::create_dir_all(&dir).unwrap();
        let entries = vec![
            serde_json::json!({"key": "model.layers.0.mlp.up_proj.weight", "file": "up.bin", "offset": 0, "length": 64}),
            serde_json::json!({"key": "model.layers.0.mlp.gate_proj.weight", "file": "gate.bin", "offset": 0, "length": 64}),
            serde_json::json!({"key": "model.layers.5.mlp.up_proj.weight", "file": "up.bin", "offset": 64, "length": 128}),
            serde_json::json!({"key": "model.embed_tokens.weight", "file": "embed.bin", "offset": 0, "length": 32}),
        ];
        let path = dir.join(WEIGHT_MANIFEST_JSON);
        std::fs::write(&path, serde_json::to_string(&entries).unwrap()).unwrap();

        let lookup = parse_manifest_for_up(&path).unwrap();
        assert_eq!(lookup.len(), 2, "only up_proj entries kept");
        assert!(lookup.contains_key(&0));
        assert!(lookup.contains_key(&5));
        assert!(!lookup.contains_key(&999));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_manifest_for_up_skips_malformed_entries() {
        let dir = unique_dir("malformed");
        std::fs::create_dir_all(&dir).unwrap();
        let entries = vec![
            // missing "file"
            serde_json::json!({"key": "model.layers.0.mlp.up_proj.weight", "offset": 0, "length": 64}),
            // missing "offset"
            serde_json::json!({"key": "model.layers.1.mlp.up_proj.weight", "file": "up.bin", "length": 64}),
            // missing "length"
            serde_json::json!({"key": "model.layers.2.mlp.up_proj.weight", "file": "up.bin", "offset": 0}),
            // unparsable layer index
            serde_json::json!({"key": "model.layers.bogus.mlp.up_proj.weight", "file": "up.bin", "offset": 0, "length": 64}),
            // valid
            serde_json::json!({"key": "model.layers.7.mlp.up_proj.weight", "file": "up.bin", "offset": 100, "length": 64}),
        ];
        let path = dir.join(WEIGHT_MANIFEST_JSON);
        std::fs::write(&path, serde_json::to_string(&entries).unwrap()).unwrap();

        let lookup = parse_manifest_for_up(&path).unwrap();
        assert_eq!(lookup.len(), 1);
        assert!(lookup.contains_key(&7));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
