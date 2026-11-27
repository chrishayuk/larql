//! Tensor inventory from safetensors shard headers.
//!
//! Reads each shard's JSON header only — the 8-byte length prefix and the
//! header bytes it announces. No tensor data is touched, so inventorying a
//! 60 GB checkpoint costs a few hundred kilobytes of I/O.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::detect::ModelError;

use super::report::{TensorFact, TensorGroup, TensorInventory};

/// Upper bound on a plausible safetensors header. The largest real-world
/// headers (thousands of tensors) are a few megabytes; a length beyond this
/// means a corrupt or non-safetensors file, and honouring it would allocate
/// the claimed size unchecked.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Filename of the HF shard index, when the checkpoint is sharded.
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

/// Safetensors extension for the unsharded / directory-scan fallback.
const SAFETENSORS_EXT: &str = "safetensors";

/// Key safetensors reserves for free-form metadata inside the header.
const HEADER_METADATA_KEY: &str = "__metadata__";

/// Scan a model directory's safetensors shards into a [`TensorInventory`].
///
/// Shard discovery prefers the HF index file (its `weight_map` names every
/// shard); a checkpoint without one is scanned for `*.safetensors` directly.
/// A directory with no shards at all yields an empty inventory rather than
/// an error — a config-only directory is still inspectable.
pub fn scan_tensors(model_dir: &Path) -> Result<TensorInventory, ModelError> {
    let files = discover_shards(model_dir)?;
    let mut tensors: Vec<TensorFact> = Vec::new();
    for file in &files {
        read_shard_header(model_dir, file, &mut tensors)?;
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let mut groups: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut total_bytes = 0u64;
    for t in &tensors {
        let entry = groups.entry(group_prefix(&t.name)).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.bytes;
        total_bytes += t.bytes;
    }
    Ok(TensorInventory {
        total_tensors: tensors.len(),
        total_bytes,
        groups: groups
            .into_iter()
            .map(|(prefix, (count, bytes))| TensorGroup {
                prefix,
                tensors: count,
                bytes,
            })
            .collect(),
        tensors,
        files,
        excluded: Vec::new(),
    })
}

/// Shard filenames relative to the model dir, sorted and deduplicated.
fn discover_shards(model_dir: &Path) -> Result<Vec<String>, ModelError> {
    let index_path = model_dir.join(SAFETENSORS_INDEX_FILE);
    if index_path.exists() {
        let text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&text)?;
        let mut files: Vec<String> = index["weight_map"]
            .as_object()
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files.dedup();
        return Ok(files);
    }
    let mut files: Vec<String> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    files.sort();
    Ok(files)
}

/// Parse one shard's header into per-tensor facts.
fn read_shard_header(
    model_dir: &Path,
    file: &str,
    out: &mut Vec<TensorFact>,
) -> Result<(), ModelError> {
    let path = model_dir.join(file);
    let mut reader = std::fs::File::open(&path)?;
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(ModelError::Parse(format!(
            "{file}: safetensors header claims {header_len} bytes \
             (limit {MAX_HEADER_BYTES}) — corrupt or not a safetensors file"
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    reader.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let entries = header.as_object().ok_or_else(|| {
        ModelError::Parse(format!("{file}: safetensors header is not a JSON object"))
    })?;
    for (name, desc) in entries {
        if name == HEADER_METADATA_KEY {
            continue;
        }
        let dtype = desc["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> = desc["shape"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        let bytes = desc["data_offsets"]
            .as_array()
            .and_then(
                |offs| match (offs.first()?.as_u64(), offs.get(1)?.as_u64()) {
                    (Some(start), Some(end)) => end.checked_sub(start),
                    _ => None,
                },
            )
            .unwrap_or(0);
        out.push(TensorFact {
            name: name.clone(),
            dtype,
            shape,
            bytes,
            file: file.to_string(),
            derivation: None,
        });
    }
    Ok(())
}

/// Grouping prefix for a tensor name: the path up to the first numeric
/// segment (`model.language_model.layers.0.…` → `model.language_model.layers`),
/// or with the final segment dropped when there is none
/// (`lm_head.weight` → `lm_head`).
pub(super) fn group_prefix(name: &str) -> String {
    let segments: Vec<&str> = name.split('.').collect();
    let cut = segments
        .iter()
        .position(|s| s.parse::<u64>().is_ok())
        .unwrap_or(segments.len().saturating_sub(1));
    if cut == 0 {
        return name.to_string();
    }
    segments[..cut].join(".")
}
