//! Patch a Q4K vindex with a missing `lm_head_q4.bin`.
//!
//! For tied-embedding models (Gemma 2/3/4) the output projection is identical
//! to `embed_tokens.weight`, which the vindex stores in `embeddings.bin`.
//! This tool reads that matrix, quantises it to Q4_K (matching the format
//! expected by `load_model_weights_kquant`), and writes `lm_head_q4.bin` next
//! to it.  It also appends a `weight_manifest.json` entry so subsequent
//! loads recognise the file.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example patch_lm_head_q4k -- \
//!     --vindex <q4k_vindex_dir>  [--vocab <N>]  [--hidden <N>]

use std::io::Write as _;
use std::path::PathBuf;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_dir = PathBuf::new();
    let mut vocab_override: Option<usize> = None;
    let mut hidden_override: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_dir = PathBuf::from(&args[i]);
            }
            "--vocab" => {
                i += 1;
                vocab_override = Some(args[i].parse()?);
            }
            "--hidden" => {
                i += 1;
                hidden_override = Some(args[i].parse()?);
            }
            _ => {}
        }
        i += 1;
    }
    if vindex_dir.as_os_str().is_empty() {
        eprintln!("Usage: patch_lm_head_q4k --vindex <dir> [--vocab N] [--hidden N]");
        std::process::exit(1);
    }

    let out_path = vindex_dir.join("lm_head_q4.bin");
    if out_path.exists() {
        eprintln!("lm_head_q4.bin already exists — nothing to do.");
        return Ok(());
    }

    // Infer vocab / hidden from index.json when not overridden.
    let index_path = vindex_dir.join("index.json");
    let (vocab, hidden) = if let (Some(v), Some(h)) = (vocab_override, hidden_override) {
        (v, h)
    } else {
        let cfg_text = std::fs::read_to_string(&index_path)?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;
        let model_cfg = cfg
            .get("model_config")
            .ok_or("no model_config in index.json")?;
        let h = model_cfg["head_dim"]
            .as_u64()
            .and_then(|hd| model_cfg["num_q_heads"].as_u64().map(|q| hd * q))
            .unwrap_or(0) as usize;

        // hidden_size isn't stored directly; read it from embeddings.bin shape.
        let embed_path = vindex_dir.join("embeddings.bin");
        let embed_meta = std::fs::metadata(&embed_path)?;
        // embeddings.bin: vocab_size × hidden_size f32 values.
        // We know vocab_size from weight_manifest.json or from index.
        let manifest_path = vindex_dir.join("weight_manifest.json");
        let manifest_text = std::fs::read_to_string(&manifest_path)?;
        let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)?;
        let embed_entry = manifest.iter().find(|e| {
            e["key"]
                .as_str()
                .map(|k| k.contains("embed_tokens"))
                .unwrap_or(false)
        });
        let (v, hd) = if let Some(e) = embed_entry {
            let shape = e["shape"].as_array().ok_or("bad shape")?;
            (
                shape[0].as_u64().unwrap_or(0) as usize,
                shape[1].as_u64().unwrap_or(0) as usize,
            )
        } else {
            // Fallback: derive from file size and a known hidden dimension.
            let hidden_guess = if h > 0 { h } else { 2560 };
            let v = embed_meta.len() as usize / (hidden_guess * 4);
            (v, hidden_guess)
        };
        (v, hd)
    };

    if vocab == 0 || hidden == 0 {
        return Err(format!(
            "Could not determine vocab ({vocab}) / hidden ({hidden}). \
             Pass --vocab and --hidden explicitly."
        )
        .into());
    }

    println!("=== patch_lm_head_q4k ===");
    println!("  vindex : {}", vindex_dir.display());
    println!("  vocab  : {vocab}");
    println!("  hidden : {hidden}");

    // Read embeddings.bin as f32.
    let embed_path = vindex_dir.join("embeddings.bin");
    let embed_bytes = std::fs::read(&embed_path)?;
    let num_floats = embed_bytes.len() / 4;
    let expected = vocab * hidden;
    if num_floats < expected {
        return Err(format!(
            "embeddings.bin has {num_floats} f32 values, expected {expected} ({vocab}×{hidden})"
        )
        .into());
    }
    let f32_data =
        unsafe { std::slice::from_raw_parts(embed_bytes.as_ptr() as *const f32, expected) };

    // Pad to multiple of 256 (Q4_K superblock size).
    let padded_len = expected.div_ceil(256) * 256;
    let padded: Vec<f32> = if padded_len != expected {
        let mut v = f32_data.to_vec();
        v.resize(padded_len, 0.0);
        v
    } else {
        f32_data.to_vec()
    };

    println!("  Quantising {} f32 → Q4_K …", expected);
    let t0 = std::time::Instant::now();
    let q4k_bytes = quantize_q4_k(&padded);
    println!(
        "  Done in {:.2}s  ({:.1} MB)",
        t0.elapsed().as_secs_f64(),
        q4k_bytes.len() as f64 / 1e6
    );

    // Write lm_head_q4.bin.
    std::fs::write(&out_path, &q4k_bytes)?;
    println!("  Written: {}", out_path.display());

    // Append entry to weight_manifest.json.
    let manifest_path = vindex_dir.join("weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let mut manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)?;
    // Remove any stale entry first.
    manifest.retain(|e| e["key"].as_str() != Some("lm_head.weight"));
    manifest.push(serde_json::json!({
        "key":    "lm_head.weight",
        "kind":   larql_vindex::format::weights::write_f32::kind::TENSOR_Q4K,
        "shape":  [vocab, hidden],
        "offset": 0,
        "length": q4k_bytes.len(),
        "file":   "lm_head_q4.bin"
    }));
    let updated = serde_json::to_string_pretty(&manifest)?;
    let mut f = std::fs::File::create(&manifest_path)?;
    f.write_all(updated.as_bytes())?;
    println!("  Manifest updated.");

    println!("=== Done ===");
    Ok(())
}
