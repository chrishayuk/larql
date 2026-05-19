//! Surgical patcher: row-pad and re-quantize `layers.N.mlp.down_proj.weight`
//! entries in an existing vindex whose extraction predated the row-padding
//! fix (commit introducing `pad_rows_to_256` in `larql-vindex/format/weights/write.rs`).
//!
//! Why: Q4_K/Q6_K super-blocks hold 256 values. The matvec shader assumes
//! each stored row is `cols / 256` full super-blocks; when `cols % 256 != 0`
//! (Gemma 4 26B A4B's dense `intermediate_size=2112` — 8.25 super-blocks per
//! row) the old extractor wrote rows contiguously with no alignment and the
//! shader read wrong bytes for every row past row 0.
//!
//! Re-extracting from scratch works (`larql extract ...`) but is slow and
//! forces the user to re-apply any manual patches (e.g. the outer-norm
//! tensors we back-filled earlier). This tool rebuilds only the `down_proj`
//! bytes in `interleaved_kquant.bin` and updates the manifest.
//!
//! Run with:
//! ```bash
//! cargo run --release -p larql-cli --example patch_down_proj -- \
//!     /path/to/gemma-4-26B-A4B-it.vindex \
//!     /path/to/hf/snapshot-root
//! ```
//!
//! `hf-snapshot-root` is the directory that contains `config.json` and
//! `model.safetensors.index.json` (e.g. `~/.cache/huggingface/hub/
//! models--google--gemma-4-26B-A4B-it/snapshots/<hash>/`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vindex_path: PathBuf = args
        .next()
        .ok_or("usage: patch_down_proj <vindex> <hf-snapshot-root>")?
        .into();
    let hf_root: PathBuf = args
        .next()
        .ok_or("usage: patch_down_proj <vindex> <hf-snapshot-root>")?
        .into();

    println!("vindex   = {}", vindex_path.display());
    println!("hf-root  = {}", hf_root.display());

    // ── Read vindex manifest ──────────────────────────────────────────────
    let ff_path = vindex_path.join("interleaved_q4k_manifest.json");
    let bin_path = vindex_path.join("interleaved_q4k.bin");
    let manifest_json = fs::read_to_string(&ff_path)?;
    let mut manifest: Value = serde_json::from_str(&manifest_json)?;
    let manifest_entries = manifest.as_array_mut().ok_or("manifest not array")?;
    println!("manifest entries: {}", manifest_entries.len());

    // Group entries by layer — assume standard (gate, up, down) ordering.
    let num_layers = manifest_entries.len() / 3;
    println!("inferred layers: {num_layers}");

    // ── Open the existing .bin for read and backup ───────────────────────
    let old_mmap = unsafe { Mmap::map(&fs::File::open(&bin_path)?)? };
    let backup = bin_path.with_extension("bin.bak");
    if !backup.exists() {
        fs::copy(&bin_path, &backup)?;
        println!("backup -> {}", backup.display());
    }

    // ── Read HF safetensors index ─────────────────────────────────────────
    let idx_text = fs::read_to_string(hf_root.join("model.safetensors.index.json"))?;
    let idx: Value = serde_json::from_str(&idx_text)?;
    let weight_map = idx["weight_map"].as_object().ok_or("weight_map missing")?;

    // Cache safetensors shards so we don't re-mmap per layer.
    let mut shards: BTreeMap<String, Mmap> = BTreeMap::new();
    let shard_mmap = |name: &str,
                      shards: &mut BTreeMap<String, Mmap>,
                      hf_root: &Path|
     -> Result<(), Box<dyn std::error::Error>> {
        if !shards.contains_key(name) {
            let p = hf_root.join(name);
            let mm = unsafe { Mmap::map(&fs::File::open(&p)?)? };
            shards.insert(name.to_string(), mm);
        }
        Ok(())
    };

    // ── Build the new .bin in memory (it's only a few hundred MB) ────────
    let mut new_bytes: Vec<u8> = Vec::with_capacity(old_mmap.len() + num_layers * 400_000);
    let mut new_manifest = Vec::with_capacity(manifest_entries.len());

    for layer in 0..num_layers {
        let gate_e = &manifest_entries[layer * 3];
        let up_e = &manifest_entries[layer * 3 + 1];
        let down_e = &manifest_entries[layer * 3 + 2];

        let gate_key = gate_e["key"].as_str().unwrap();
        let up_key = up_e["key"].as_str().unwrap();
        let down_key = down_e["key"].as_str().unwrap();
        assert!(
            gate_key.ends_with(".mlp.gate_proj.weight"),
            "unexpected entry[0]: {gate_key}"
        );
        assert!(
            up_key.ends_with(".mlp.up_proj.weight"),
            "unexpected entry[1]: {up_key}"
        );
        assert!(
            down_key.ends_with(".mlp.down_proj.weight"),
            "unexpected entry[2]: {down_key}"
        );

        // Copy gate and up bytes unchanged.
        let copy_entry = |e: &Value, sink: &mut Vec<u8>| -> (u64, u64) {
            let off = e["offset"].as_u64().unwrap() as usize;
            let len = e["length"].as_u64().unwrap() as usize;
            let dst_off = sink.len() as u64;
            sink.extend_from_slice(&old_mmap[off..off + len]);
            (dst_off, len as u64)
        };
        let (g_off, g_len) = copy_entry(gate_e, &mut new_bytes);
        new_manifest.push(serde_json::json!({
            "key": gate_key, "shape": gate_e["shape"].clone(),
            "format": gate_e["format"].clone(), "offset": g_off, "length": g_len,
        }));
        let (u_off, u_len) = copy_entry(up_e, &mut new_bytes);
        new_manifest.push(serde_json::json!({
            "key": up_key, "shape": up_e["shape"].clone(),
            "format": up_e["format"].clone(), "offset": u_off, "length": u_len,
        }));

        // Read down_proj from HF.
        let hf_key = format!("model.language_model.layers.{layer}.mlp.down_proj.weight");
        let shard_name = weight_map
            .get(&hf_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("HF weight_map missing {hf_key}"))?;
        shard_mmap(shard_name, &mut shards, &hf_root)?;
        let st = SafeTensors::deserialize(&shards[shard_name])?;
        let tensor = st.tensor(&hf_key)?;
        let shape = tensor.shape();
        let rows = shape[0];
        let cols = shape[1];
        assert_eq!(tensor.dtype(), safetensors::Dtype::BF16, "expected bf16");
        // BF16 → f32 row-by-row, with per-row zero padding to next 256-multiple.
        let bytes = tensor.data();
        assert_eq!(bytes.len(), rows * cols * 2);
        let padded_cols = cols.div_ceil(256) * 256;
        let mut padded = vec![0.0f32; rows * padded_cols];
        for r in 0..rows {
            let src_row = &bytes[r * cols * 2..(r + 1) * cols * 2];
            let dst_row = &mut padded[r * padded_cols..r * padded_cols + cols];
            for (i, b) in src_row.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                let f = f32::from_bits((bits as u32) << 16); // bf16 -> f32
                dst_row[i] = f;
            }
        }
        let q_bytes = quantize_q6_k(&padded);
        let expected = rows * (padded_cols / 256) * 210;
        assert_eq!(q_bytes.len(), expected);

        let d_off = new_bytes.len() as u64;
        new_bytes.extend_from_slice(&q_bytes);
        new_manifest.push(serde_json::json!({
            "key": down_key,
            // Stored shape reflects padded cols. Runtime reads shape[1] as K.
            "shape": vec![rows, padded_cols],
            "format": "Q6_K",
            "offset": d_off,
            "length": q_bytes.len(),
        }));
        if layer % 5 == 0 {
            println!(
                "  L{layer:02}  down {} → {} bytes (padded {}→{})",
                down_e["length"],
                q_bytes.len(),
                cols,
                padded_cols
            );
        }
    }

    // ── Write new .bin + manifest ─────────────────────────────────────────
    drop(old_mmap); // release mmap so we can rewrite
    {
        let mut f = fs::File::create(&bin_path)?;
        f.write_all(&new_bytes)?;
    }
    fs::write(&ff_path, serde_json::to_string_pretty(&new_manifest)?)?;
    println!("wrote {} bytes to {}", new_bytes.len(), bin_path.display());
    println!("manifest entries: {}", new_manifest.len());
    Ok(())
}
