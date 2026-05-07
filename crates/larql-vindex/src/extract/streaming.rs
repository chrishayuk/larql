//! Streaming vindex extraction — build from safetensors without loading the full model.
//!
//! Instead of loading all weights into ModelWeights (which requires the entire model
//! in RAM), this module mmaps safetensors files and processes one layer at a time.
//! Peak memory = 1 layer's tensors + embeddings, not the full model.
//!
//! For a 120B MoE model: ~120 GB as ModelWeights vs ~2 GB streaming.

use crate::extract::stage_labels::*;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::config::dtype::StorageDtype;
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::format::filenames::*;

const MAX_GGUF_GATE_VECTOR_BYTES: u128 = 128 * 1024 * 1024;

/// Mmap'd safetensors file — kept alive for the duration of extraction.
struct MmapShard {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

#[derive(Debug)]
enum WeightSource {
    Safetensors(Vec<PathBuf>),
    Gguf(GgufCatalog),
}

#[derive(Debug)]
struct GgufCatalog {
    files: Vec<PathBuf>,
    architecture: String,
    split_count: usize,
    tensors: HashMap<String, GgufTensorEntry>,
    three_d_tensors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GgufTensorEntry {
    name: String,
    shard_idx: usize,
    dims: Vec<u64>,
    tensor_type: u32,
    tensor_offset: u64,
    data_offset: u64,
}

impl GgufCatalog {
    #[cfg_attr(not(test), allow(dead_code))]
    fn tensor(&self, name: &str) -> Option<&GgufTensorEntry> {
        self.tensors.get(name)
    }
}

impl GgufTensorEntry {
    #[cfg_attr(not(test), allow(dead_code))]
    fn is_3d(&self) -> bool {
        self.dims.len() == 3
    }
}

fn build_gguf_catalog(files: &[PathBuf]) -> Result<GgufCatalog, VindexError> {
    let mut architecture = "unknown".to_string();
    let mut split_count = files.len();
    let mut tensors = HashMap::new();
    let mut three_d_tensors = 0usize;

    for (shard_idx, file) in files.iter().enumerate() {
        let gguf = larql_models::loading::gguf::GgufFile::open(file)?;
        if architecture == "unknown" {
            if let Some(arch) = gguf
                .metadata
                .get("general.architecture")
                .and_then(|v| v.as_str())
            {
                architecture = arch.to_string();
            }
        }
        if let Some(count) = gguf.metadata.get("split.count").and_then(|v| v.as_u32()) {
            split_count = count as usize;
        }
        for info in &gguf.tensor_infos {
            let dims = info.dims().to_vec();
            if dims.len() == 3 {
                three_d_tensors += 1;
            }
            tensors.insert(
                info.name().to_string(),
                GgufTensorEntry {
                    name: info.name().to_string(),
                    shard_idx,
                    dims,
                    tensor_type: info.tensor_type(),
                    tensor_offset: info.offset(),
                    data_offset: gguf.data_offset,
                },
            );
        }
    }

    Ok(GgufCatalog {
        files: files.to_vec(),
        architecture,
        split_count,
        tensors,
        three_d_tensors,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(not(test), allow(dead_code))]
enum GgufExpertComponent {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Default)]
#[cfg_attr(not(test), allow(dead_code))]
struct GgufDeepseek2Layout {
    packed_experts: HashMap<(usize, GgufExpertComponent), String>,
    routers: HashMap<usize, String>,
    shared_experts: HashMap<(usize, GgufExpertComponent), String>,
    attention: Vec<String>,
}

#[cfg_attr(not(test), allow(dead_code))]
fn classify_deepseek2_layout(catalog: &GgufCatalog) -> Result<GgufDeepseek2Layout, VindexError> {
    if catalog.architecture != "deepseek2" {
        return Err(VindexError::Parse(format!(
            "DeepSeek2 GGUF classifier only supports architecture=deepseek2, got {}",
            catalog.architecture
        )));
    }
    let mut layout = GgufDeepseek2Layout::default();
    for name in catalog.tensors.keys() {
        let Some((layer, suffix)) = parse_deepseek2_layer_suffix(name) else {
            continue;
        };
        match suffix {
            "ffn_gate_exps.weight" => {
                layout
                    .packed_experts
                    .insert((layer, GgufExpertComponent::Gate), name.clone());
            }
            "ffn_up_exps.weight" => {
                layout
                    .packed_experts
                    .insert((layer, GgufExpertComponent::Up), name.clone());
            }
            "ffn_down_exps.weight" => {
                layout
                    .packed_experts
                    .insert((layer, GgufExpertComponent::Down), name.clone());
            }
            "ffn_gate_inp.weight" => {
                layout.routers.insert(layer, name.clone());
            }
            "ffn_gate_shexp.weight" => {
                layout
                    .shared_experts
                    .insert((layer, GgufExpertComponent::Gate), name.clone());
            }
            "ffn_up_shexp.weight" => {
                layout
                    .shared_experts
                    .insert((layer, GgufExpertComponent::Up), name.clone());
            }
            "ffn_down_shexp.weight" => {
                layout
                    .shared_experts
                    .insert((layer, GgufExpertComponent::Down), name.clone());
            }
            suffix if suffix.starts_with("attn_") => {
                // Attention tensors are intentionally not part of vindex FFN extraction,
                // but keeping a bucket here documents that they were recognized and skipped.
            }
            _ => {}
        }
    }
    Ok(layout)
}

#[cfg_attr(not(test), allow(dead_code))]
fn parse_deepseek2_layer_suffix(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;
    Some((layer, suffix))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
struct PackedExpertSlice {
    expert_idx: usize,
    expert_count: usize,
    rows: usize,
    cols: usize,
    element_offset: usize,
    element_len: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
fn packed_expert_slice(
    entry: &GgufTensorEntry,
    expert_idx: usize,
) -> Result<PackedExpertSlice, VindexError> {
    if entry.dims.len() != 3 {
        return Err(VindexError::Parse(format!(
            "packed expert tensor {} must be 3D, got dims {:?}",
            entry.name, entry.dims
        )));
    }

    // GGUF/GGML dimension order stores the fastest-moving matrix dimension first.
    // Kimi packed experts use [cols, rows, experts], so each expert slice is a
    // conventional row-major [rows, cols] matrix after dequantization.
    let cols = usize::try_from(entry.dims[0]).map_err(|_| {
        VindexError::Parse(format!(
            "packed expert tensor {} has too-large cols dim {}",
            entry.name, entry.dims[0]
        ))
    })?;
    let rows = usize::try_from(entry.dims[1]).map_err(|_| {
        VindexError::Parse(format!(
            "packed expert tensor {} has too-large rows dim {}",
            entry.name, entry.dims[1]
        ))
    })?;
    let expert_count = usize::try_from(entry.dims[2]).map_err(|_| {
        VindexError::Parse(format!(
            "packed expert tensor {} has too-large expert dim {}",
            entry.name, entry.dims[2]
        ))
    })?;
    if expert_idx >= expert_count {
        return Err(VindexError::Parse(format!(
            "packed expert tensor {} expert index {} out of range 0..{}",
            entry.name, expert_idx, expert_count
        )));
    }
    let element_len = rows.checked_mul(cols).ok_or_else(|| {
        VindexError::Parse(format!(
            "packed expert tensor {} slice element count overflows: rows={} cols={}",
            entry.name, rows, cols
        ))
    })?;
    let element_offset = expert_idx.checked_mul(element_len).ok_or_else(|| {
        VindexError::Parse(format!(
            "packed expert tensor {} expert offset overflows: expert={} len={}",
            entry.name, expert_idx, element_len
        ))
    })?;

    Ok(PackedExpertSlice {
        expert_idx,
        expert_count,
        rows,
        cols,
        element_offset,
        element_len,
    })
}

#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
struct GgufPackedExpertLayer {
    gate: Array2<f32>,
    up: Array2<f32>,
    down: Array2<f32>,
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_deepseek2_packed_expert_layer(
    catalog: &GgufCatalog,
    layout: &GgufDeepseek2Layout,
    layer: usize,
    expert_idx: usize,
) -> Result<GgufPackedExpertLayer, VindexError> {
    let gate_name = layout
        .packed_experts
        .get(&(layer, GgufExpertComponent::Gate))
        .ok_or_else(|| VindexError::MissingTensor(format!("blk.{layer}.ffn_gate_exps.weight")))?;
    let up_name = layout
        .packed_experts
        .get(&(layer, GgufExpertComponent::Up))
        .ok_or_else(|| VindexError::MissingTensor(format!("blk.{layer}.ffn_up_exps.weight")))?;
    let down_name = layout
        .packed_experts
        .get(&(layer, GgufExpertComponent::Down))
        .ok_or_else(|| VindexError::MissingTensor(format!("blk.{layer}.ffn_down_exps.weight")))?;

    Ok(GgufPackedExpertLayer {
        gate: read_packed_expert_slice_by_type(catalog, gate_name, expert_idx)?,
        up: read_packed_expert_slice_by_type(catalog, up_name, expert_idx)?,
        down: read_packed_expert_slice_by_type(catalog, down_name, expert_idx)?,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_packed_expert_slice_by_type(
    catalog: &GgufCatalog,
    tensor_name: &str,
    expert_idx: usize,
) -> Result<Array2<f32>, VindexError> {
    let entry = catalog
        .tensor(tensor_name)
        .ok_or_else(|| VindexError::MissingTensor(tensor_name.to_string()))?;
    match entry.tensor_type {
        larql_models::quant::ggml::TYPE_F32 => {
            read_packed_expert_slice_f32(catalog, tensor_name, expert_idx)
        }
        larql_models::quant::ggml::TYPE_Q4_K => {
            read_packed_expert_slice_q4_k(catalog, tensor_name, expert_idx)
        }
        larql_models::quant::ggml::TYPE_Q6_K => {
            read_packed_expert_slice_q6_k(catalog, tensor_name, expert_idx)
        }
        other => Err(VindexError::UnsupportedDtype(format!(
            "GGUF tensor {} type {} is not supported by packed expert layer reader",
            entry.name, other
        ))),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn write_deepseek2_packed_gate_vectors<W: Write>(
    writer: &mut W,
    catalog: &GgufCatalog,
    layout: &GgufDeepseek2Layout,
    layer: usize,
    offset: u64,
    dtype: StorageDtype,
) -> Result<VindexLayerInfo, VindexError> {
    let gate_name = layout
        .packed_experts
        .get(&(layer, GgufExpertComponent::Gate))
        .ok_or_else(|| VindexError::MissingTensor(format!("blk.{layer}.ffn_gate_exps.weight")))?;
    let entry = catalog
        .tensor(gate_name)
        .ok_or_else(|| VindexError::MissingTensor(gate_name.to_string()))?;
    let [_cols, rows, experts] = packed_expert_dims(entry)?;

    let mut length = 0u64;
    for expert_idx in 0..experts {
        let gate = read_packed_expert_slice_by_type(catalog, gate_name, expert_idx)?;
        length += write_floats(writer, gate.as_slice().unwrap(), dtype)?;
    }

    Ok(VindexLayerInfo {
        layer,
        num_features: rows * experts,
        offset,
        length,
        num_experts: Some(experts),
        num_features_per_expert: Some(rows),
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn packed_expert_dims(entry: &GgufTensorEntry) -> Result<[usize; 3], VindexError> {
    if entry.dims.len() != 3 {
        return Err(VindexError::Parse(format!(
            "packed expert tensor {} must have 3 dims, got {:?}",
            entry.name, entry.dims
        )));
    }
    Ok([
        entry.dims[0] as usize,
        entry.dims[1] as usize,
        entry.dims[2] as usize,
    ])
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_packed_expert_slice_f32(
    catalog: &GgufCatalog,
    tensor_name: &str,
    expert_idx: usize,
) -> Result<Array2<f32>, VindexError> {
    let entry = catalog
        .tensor(tensor_name)
        .ok_or_else(|| VindexError::MissingTensor(tensor_name.to_string()))?;
    if entry.tensor_type != larql_models::quant::ggml::TYPE_F32 {
        return Err(VindexError::UnsupportedDtype(format!(
            "GGUF tensor {} type {} cannot be read by F32 packed-slice reader",
            entry.name, entry.tensor_type
        )));
    }
    let slice = packed_expert_slice(entry, expert_idx)?;
    let raw = read_gguf_tensor_byte_range(
        catalog,
        entry,
        (slice.element_offset as u64)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "GGUF tensor {} slice byte offset overflow: element_offset={}",
                    entry.name, slice.element_offset
                ))
            })?,
        slice
            .element_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "GGUF tensor {} slice byte length overflow: element_len={}",
                    entry.name, slice.element_len
                ))
            })?,
    )?;

    let mut values = Vec::with_capacity(slice.element_len);
    for bytes in raw.chunks_exact(4) {
        values.push(f32::from_le_bytes(
            bytes.try_into().expect("chunks_exact(4)"),
        ));
    }
    Array2::from_shape_vec((slice.rows, slice.cols), values).map_err(|e| {
        VindexError::Parse(format!(
            "GGUF tensor {} expert {} shape error: {}",
            entry.name, expert_idx, e
        ))
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_packed_expert_slice_q4_k(
    catalog: &GgufCatalog,
    tensor_name: &str,
    expert_idx: usize,
) -> Result<Array2<f32>, VindexError> {
    let entry = catalog
        .tensor(tensor_name)
        .ok_or_else(|| VindexError::MissingTensor(tensor_name.to_string()))?;
    if entry.tensor_type != larql_models::quant::ggml::TYPE_Q4_K {
        return Err(VindexError::UnsupportedDtype(format!(
            "GGUF tensor {} type {} cannot be read by Q4_K packed-slice reader",
            entry.name, entry.tensor_type
        )));
    }
    read_block_aligned_packed_expert_slice(
        catalog,
        entry,
        expert_idx,
        "Q4_K",
        larql_models::quant::ggml::Q4_K_BLOCK_ELEMS,
        larql_models::quant::ggml::Q4_K_BLOCK_BYTES,
        larql_models::quant::ggml::dequantize_q4_k,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_packed_expert_slice_q6_k(
    catalog: &GgufCatalog,
    tensor_name: &str,
    expert_idx: usize,
) -> Result<Array2<f32>, VindexError> {
    let entry = catalog
        .tensor(tensor_name)
        .ok_or_else(|| VindexError::MissingTensor(tensor_name.to_string()))?;
    if entry.tensor_type != larql_models::quant::ggml::TYPE_Q6_K {
        return Err(VindexError::UnsupportedDtype(format!(
            "GGUF tensor {} type {} cannot be read by Q6_K packed-slice reader",
            entry.name, entry.tensor_type
        )));
    }
    read_block_aligned_packed_expert_slice(
        catalog,
        entry,
        expert_idx,
        "Q6_K",
        larql_models::quant::ggml::Q6_K_BLOCK_ELEMS,
        larql_models::quant::ggml::Q6_K_BLOCK_BYTES,
        larql_models::quant::ggml::dequantize_q6_k,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_block_aligned_packed_expert_slice(
    catalog: &GgufCatalog,
    entry: &GgufTensorEntry,
    expert_idx: usize,
    dtype_name: &str,
    block_elems: usize,
    block_bytes: usize,
    dequantize: fn(&[u8], usize) -> Result<Vec<f32>, larql_models::ModelError>,
) -> Result<Array2<f32>, VindexError> {
    let slice = packed_expert_slice(entry, expert_idx)?;
    if !slice.element_offset.is_multiple_of(block_elems)
        || !slice.element_len.is_multiple_of(block_elems)
    {
        return Err(VindexError::Parse(format!(
            "GGUF tensor {} {} expert slice must be block-aligned to {} elements (offset={} len={})",
            entry.name, dtype_name, block_elems, slice.element_offset, slice.element_len
        )));
    }
    let block_offset = slice.element_offset / block_elems;
    let block_len = slice.element_len / block_elems;
    let raw = read_gguf_tensor_byte_range(
        catalog,
        entry,
        (block_offset as u64)
            .checked_mul(block_bytes as u64)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "GGUF tensor {} {} slice byte offset overflow: block_offset={}",
                    entry.name, dtype_name, block_offset
                ))
            })?,
        block_len.checked_mul(block_bytes).ok_or_else(|| {
            VindexError::Parse(format!(
                "GGUF tensor {} {} slice byte length overflow: block_len={}",
                entry.name, dtype_name, block_len
            ))
        })?,
    )?;
    let values = dequantize(&raw, slice.element_len)?;
    Array2::from_shape_vec((slice.rows, slice.cols), values).map_err(|e| {
        VindexError::Parse(format!(
            "GGUF tensor {} expert {} {} shape error: {}",
            entry.name, expert_idx, dtype_name, e
        ))
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_gguf_tensor_byte_range(
    catalog: &GgufCatalog,
    entry: &GgufTensorEntry,
    tensor_relative_byte_offset: u64,
    byte_len: usize,
) -> Result<Vec<u8>, VindexError> {
    let shard_path = catalog.files.get(entry.shard_idx).ok_or_else(|| {
        VindexError::Parse(format!(
            "GGUF tensor {} points at missing shard index {}",
            entry.name, entry.shard_idx
        ))
    })?;
    let file = std::fs::File::open(shard_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let tensor_abs_offset = entry
        .data_offset
        .checked_add(entry.tensor_offset)
        .ok_or_else(|| {
            VindexError::Parse(format!(
                "GGUF tensor {} data offset overflow: data_offset={} tensor_offset={}",
                entry.name, entry.data_offset, entry.tensor_offset
            ))
        })?;
    let abs_offset = tensor_abs_offset
        .checked_add(tensor_relative_byte_offset)
        .ok_or_else(|| {
            VindexError::Parse(format!(
                "GGUF tensor {} absolute slice offset overflow: tensor_offset={} slice_byte_offset={}",
                entry.name, tensor_abs_offset, tensor_relative_byte_offset
            ))
        })?;
    let start = usize::try_from(abs_offset).map_err(|_| {
        VindexError::Parse(format!(
            "GGUF tensor {} absolute slice offset {} exceeds usize",
            entry.name, abs_offset
        ))
    })?;
    let end = start.checked_add(byte_len).ok_or_else(|| {
        VindexError::Parse(format!(
            "GGUF tensor {} slice range overflows: start={} len={}",
            entry.name, start, byte_len
        ))
    })?;
    if end > mmap.len() {
        return Err(VindexError::Parse(format!(
            "GGUF tensor {} slice out of bounds (offset {} + size {} > file {})",
            entry.name,
            abs_offset,
            byte_len,
            mmap.len()
        )));
    }
    Ok(mmap[start..end].to_vec())
}

fn build_gguf_streaming(
    catalog: &GgufCatalog,
    output_dir: &Path,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    if catalog.architecture != "deepseek2" {
        return unsupported_gguf_streaming(catalog, catalog.architecture.clone());
    }

    let layout = classify_deepseek2_layout(catalog)?;
    let estimated_gate_bytes = estimate_deepseek2_gate_vector_bytes(catalog, &layout, dtype)?;
    if estimated_gate_bytes > MAX_GGUF_GATE_VECTOR_BYTES {
        return unsupported_gguf_streaming(
            catalog,
            format!(
                "deepseek2 gate_vectors estimate {} exceeds safe streaming budget {}; embeddings/down_meta wiring remains pending",
                estimated_gate_bytes, MAX_GGUF_GATE_VECTOR_BYTES
            ),
        );
    }

    std::fs::create_dir_all(output_dir)?;
    callbacks.on_stage(STAGE_LOADING);
    callbacks.on_stage_done(STAGE_LOADING, 0.0);
    callbacks.on_stage(STAGE_GATE_VECTORS);

    let gate_path = output_dir.join(GATE_VECTORS_BIN);
    let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
    let mut offset = 0u64;
    let mut layers: Vec<usize> = layout
        .packed_experts
        .keys()
        .filter_map(|(layer, component)| {
            (*component == GgufExpertComponent::Gate).then_some(*layer)
        })
        .collect();
    layers.sort_unstable();
    layers.dedup();

    for layer in layers {
        callbacks.on_layer_start(COMP_GATE, layer, layout.packed_experts.len());
        let start = std::time::Instant::now();
        let info = write_deepseek2_packed_gate_vectors(
            &mut gate_file,
            catalog,
            &layout,
            layer,
            offset,
            dtype,
        )?;
        offset += info.length;
        callbacks.on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
    }
    gate_file.flush()?;
    callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);

    unsupported_gguf_streaming(
        catalog,
        "deepseek2 after writing gate_vectors.bin; embeddings/down_meta artifact wiring remains pending".into(),
    )
}

fn estimate_deepseek2_gate_vector_bytes(
    catalog: &GgufCatalog,
    layout: &GgufDeepseek2Layout,
    dtype: StorageDtype,
) -> Result<u128, VindexError> {
    let bytes_per_float = match dtype {
        StorageDtype::F32 => 4u128,
        StorageDtype::F16 => 2u128,
    };
    let mut total = 0u128;
    for ((_, component), tensor_name) in &layout.packed_experts {
        if *component != GgufExpertComponent::Gate {
            continue;
        }
        let entry = catalog
            .tensor(tensor_name)
            .ok_or_else(|| VindexError::MissingTensor(tensor_name.clone()))?;
        let [cols, rows, experts] = packed_expert_dims(entry)?;
        total = total
            .checked_add(cols as u128 * rows as u128 * experts as u128 * bytes_per_float)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "GGUF gate vector byte estimate overflow at tensor {}",
                    entry.name
                ))
            })?;
    }
    Ok(total)
}

fn unsupported_gguf_streaming<T>(
    catalog: &GgufCatalog,
    architecture: String,
) -> Result<T, VindexError> {
    Err(VindexError::UnsupportedGgufStreaming {
        architecture,
        files: catalog.files.len(),
        split_count: catalog.split_count,
        tensor_count: catalog.tensors.len(),
        three_d_tensors: catalog.three_d_tensors,
    })
}

fn discover_weight_source(model_dir: &Path) -> Result<WeightSource, VindexError> {
    let mut st_files = collect_ext(model_dir, "safetensors")?;
    if st_files.is_empty() {
        let weights_dir = model_dir.join("weights");
        if weights_dir.is_dir() {
            st_files = collect_ext(&weights_dir, "safetensors")?;
        }
    }
    st_files.sort();
    if !st_files.is_empty() {
        return Ok(WeightSource::Safetensors(st_files));
    }

    let mut gguf_files = collect_ext(model_dir, "gguf")?;
    if gguf_files.is_empty() {
        for entry in std::fs::read_dir(model_dir)? {
            let path = entry?.path();
            if path.is_dir() {
                gguf_files.extend(collect_ext(&path, "gguf")?);
            }
        }
    }
    gguf_files.sort();
    if gguf_files.is_empty() {
        return Err(VindexError::NoSafetensors(model_dir.to_path_buf()));
    }

    Ok(WeightSource::Gguf(build_gguf_catalog(&gguf_files)?))
}

fn collect_ext(dir: &Path, ext: &str) -> Result<Vec<PathBuf>, VindexError> {
    Ok(std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|actual| actual == ext))
        .collect())
}

#[cfg_attr(not(test), allow(dead_code))]
fn preflight_gguf_streaming(source: &WeightSource) -> Result<(), VindexError> {
    if let WeightSource::Gguf(catalog) = source {
        return Err(VindexError::UnsupportedGgufStreaming {
            architecture: catalog.architecture.clone(),
            files: catalog.files.len(),
            split_count: catalog.split_count,
            tensor_count: catalog.tensors.len(),
            three_d_tensors: catalog.three_d_tensors,
        });
    }
    Ok(())
}

/// Build a vindex by streaming from safetensors files (no full model load).
///
/// Peak memory: embeddings + 1 layer of gate/down weights at a time.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    quant: QuantFormat,
    weight_opts: crate::format::weights::WriteWeightsOptions,
    q4k_opts: crate::format::weights::Q4kWriteOptions,
    // Skip writing `gate_vectors.bin` entirely. Only valid when
    // `quant == Q4K` — the loader synthesizes gate from Q4K at load
    // time. Refused otherwise because without a Q4K interleaved file
    // the gate would be unrecoverable.
    drop_gate_vectors: bool,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    if drop_gate_vectors && quant != QuantFormat::Q4K {
        return Err(VindexError::Parse(
            "--drop-gate-vectors requires --quant q4k (the loader rebuilds gate from Q4K)".into(),
        ));
    }
    std::fs::create_dir_all(output_dir)?;

    // Detect architecture
    let arch = larql_models::detect_architecture_validated(model_dir)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();
    let cfg = arch.config();

    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size;
    let embed_scale = arch.embed_scale();
    let is_moe = arch.is_moe();
    let n_experts = arch.num_experts();

    // Mmap all safetensors files. If the model directory contains GGUF
    // shards instead, fail with a GGUF-specific preflight diagnostic rather
    // than the misleading `NoSafetensors` error. Header-only preflight keeps
    // Kimi-sized probes cheap and avoids eager dequantization.
    let source = discover_weight_source(model_dir)?;
    let st_files = match source {
        WeightSource::Safetensors(st_files) => st_files,
        WeightSource::Gguf(catalog) => {
            return build_gguf_streaming(&catalog, output_dir, dtype, callbacks);
        }
    };

    callbacks.on_stage(STAGE_LOADING);
    eprintln!(
        "  Streaming mode: {} safetensors shards (mmap'd, not loaded)",
        st_files.len()
    );

    // Checkpoint setup with auto-resume. A compatible checkpoint
    // from a previous interrupted run is reused; phases it marked
    // complete are skipped (their output files on disk are reused
    // unchanged). An incompatible checkpoint (different model_dir /
    // num_layers) is discarded.
    let mut checkpoint = match super::checkpoint::Checkpoint::load(output_dir)? {
        Some(prior) if prior.is_compatible_with(model_dir, model_name, num_layers) => {
            eprintln!(
                "  Resuming from checkpoint at {}/{} — phases already complete: {:?}",
                output_dir.display(),
                super::checkpoint::CHECKPOINT_FILE,
                prior.completed,
            );
            prior
        }
        Some(_) => {
            eprintln!(
                "  Checkpoint at {}/{} is incompatible with this run \
                 (different model / layer count) — discarding",
                output_dir.display(),
                super::checkpoint::CHECKPOINT_FILE,
            );
            super::checkpoint::Checkpoint::fresh(model_dir, model_name, num_layers)
        }
        None => super::checkpoint::Checkpoint::fresh(model_dir, model_name, num_layers),
    };

    // (shards vec was for an earlier design — tensor_index + shard_mmaps is the actual approach)

    // SAFETY: We need to hold both the mmap and the SafeTensors that borrows from it.
    // We use a two-phase approach: first mmap all files, then deserialize.
    // The mmaps are kept alive in `shard_mmaps` for the lifetime of the function.
    let shard_mmaps: Vec<MmapShard> = st_files
        .iter()
        .map(|path| {
            let file = std::fs::File::open(path).unwrap();
            let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
            MmapShard { _file: file, mmap }
        })
        .collect();

    // Build a tensor index: key → (shard_idx, tensor_name)
    // We need to find which shard contains each tensor.
    let mut tensor_index: HashMap<String, (usize, String)> = HashMap::new();
    for (shard_idx, shard) in shard_mmaps.iter().enumerate() {
        let st = safetensors::SafeTensors::deserialize(&shard.mmap)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        for name in st.names() {
            let key = normalize_key(name, prefixes);
            tensor_index.insert(key.clone(), (shard_idx, name.to_string()));
        }
    }

    callbacks.on_stage_done(STAGE_LOADING, 0.0);

    // ── 1. Gate vectors (streaming, one layer at a time) ──
    //
    // If `drop_gate_vectors` is set we still walk every layer to build
    // `layer_infos` (num_features per layer is part of `index.json`)
    // but redirect writes to `/dev/null` (`io::sink`). The gate bytes
    // are recoverable from `interleaved_q4k.bin` at load time.
    callbacks.on_stage(STAGE_GATE_VECTORS);
    let gate_path = output_dir.join(GATE_VECTORS_BIN);
    enum GateSink {
        File(BufWriter<std::fs::File>),
        Discard(std::io::Sink),
    }
    impl std::io::Write for GateSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            match self {
                GateSink::File(f) => f.write(buf),
                GateSink::Discard(s) => s.write(buf),
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            match self {
                GateSink::File(f) => f.flush(),
                GateSink::Discard(s) => s.flush(),
            }
        }
    }

    // Auto-resume: if a prior run finished the gate phase and saved
    // `gate_layer_infos`, reuse it and skip the gate loop entirely.
    let resumed_gate = checkpoint.is_complete(super::checkpoint::ExtractPhase::Gate)
        && checkpoint.gate_layer_infos.is_some();
    let mut layer_infos: Vec<VindexLayerInfo> = if resumed_gate {
        eprintln!(
            "  Skipping gate phase ({} layer infos restored from checkpoint; \
             reusing existing {})",
            checkpoint
                .gate_layer_infos
                .as_ref()
                .map(|v| v.len())
                .unwrap_or(0),
            GATE_VECTORS_BIN,
        );
        callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
        checkpoint.gate_layer_infos.clone().unwrap_or_default()
    } else {
        Vec::new()
    };

    // Only allocate the writer + run the loop when the phase isn't
    // already done.
    let mut gate_file: GateSink = if resumed_gate || drop_gate_vectors {
        GateSink::Discard(std::io::sink())
    } else {
        GateSink::File(BufWriter::new(std::fs::File::create(&gate_path)?))
    };
    let mut offset: u64 = 0;

    // Check expert format from the architecture
    let expert_format = arch.expert_format();

    // Skip the per-layer gate loop entirely on resume.
    let layer_count_for_loop = if resumed_gate { 0 } else { num_layers };
    for layer in 0..layer_count_for_loop {
        callbacks.on_layer_start(COMP_GATE, layer, num_layers);
        let start = std::time::Instant::now();

        if expert_format == larql_models::ExpertFormat::PackedMxfp4 {
            // MXFP4 packed experts: dequantize gate_up_proj_blocks per layer
            // The fused tensor is [num_experts, 2*intermediate, groups, 16]
            // First half of output features = gate, second half = up
            let blocks_key = arch.packed_gate_up_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_gate_up_scales_key(layer).unwrap_or_default();

            if let (Some(blocks_info), Some(scales_info)) =
                (tensor_index.get(&blocks_key), tensor_index.get(&scales_key))
            {
                let blocks_st =
                    safetensors::SafeTensors::deserialize(&shard_mmaps[blocks_info.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_st =
                    safetensors::SafeTensors::deserialize(&shard_mmaps[scales_info.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;

                let blocks_view = blocks_st
                    .tensor(&blocks_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_view = scales_st
                    .tensor(&scales_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                let shape = blocks_view.shape();
                let n_exp = shape[0];
                let out_features = shape[1]; // 2 * intermediate (fused gate+up)
                let groups = shape[2];
                let in_features = groups * 32;
                let half = out_features / 2; // gate portion

                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    blocks_view.data(),
                    scales_view.data(),
                    n_exp,
                    out_features,
                    groups,
                )?;

                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;

                for expert_data in &experts {
                    // Extract gate portion (first half rows)
                    let gate_data = &expert_data[..half * in_features];
                    layer_bytes += write_floats(&mut gate_file, gate_data, dtype)?;
                    total_features += half;
                }

                if total_features > 0 {
                    layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(n_exp),
                        num_features_per_expert: Some(half),
                    });
                    offset += layer_bytes;
                }
            }
        } else if expert_format == larql_models::ExpertFormat::PackedBF16 && is_moe {
            // Hybrid MoE (Gemma 4 26B A4B): packed experts stored separately.
            // gate_vectors.bin uses the dense FFN gate for KNN walk routing.
            let gate_key = normalize_key(&arch.ffn_gate_key(layer), prefixes);
            if let Some(tensor) = get_tensor_f32(&shard_mmaps, &tensor_index, &gate_key)? {
                let num_features = tensor.shape()[0];
                let data = tensor.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, dtype)?;
                layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }
        } else if is_moe && n_experts > 0 {
            // Standard MoE (Mixtral): per-expert gate tensors
            let mut total_features = 0usize;
            let mut layer_bytes = 0u64;
            let mut features_per_expert = 0usize;

            for expert in 0..n_experts {
                let gate_key = match arch.expert_ffn_gate_key(layer, expert) {
                    Some(k) => normalize_key(&k, prefixes),
                    None => continue,
                };

                if let Some(tensor) = get_tensor_f32(&shard_mmaps, &tensor_index, &gate_key)? {
                    features_per_expert = tensor.shape()[0];
                    total_features += features_per_expert;
                    let data = tensor.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, dtype)?;
                }
            }

            if total_features > 0 {
                layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features: total_features,
                    offset,
                    length: layer_bytes,
                    num_experts: Some(n_experts),
                    num_features_per_expert: Some(features_per_expert),
                });
                offset += layer_bytes;
            }
        } else {
            // Dense: single gate matrix per layer
            let gate_key = normalize_key(&arch.ffn_gate_key(layer), prefixes);
            if let Some(tensor) = get_tensor_f32(&shard_mmaps, &tensor_index, &gate_key)? {
                let num_features = tensor.shape()[0];
                let data = tensor.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, dtype)?;
                layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }
        }

        callbacks.on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
    }
    gate_file.flush()?;
    // If we were only sinking bytes, don't leave a zero-byte
    // gate_vectors.bin behind for the loader to trip over.
    drop(gate_file);
    if drop_gate_vectors && gate_path.exists() && !resumed_gate {
        let _ = std::fs::remove_file(&gate_path);
    }
    if !resumed_gate {
        callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
        checkpoint.mark_gate_complete(layer_infos.clone(), output_dir)?;
    }

    // ── 1b. Router weights (MoE models only) ──
    if is_moe {
        callbacks.on_stage(STAGE_ROUTER_WEIGHTS);
        let router_path = output_dir.join("router_weights.bin");
        let mut router_file = BufWriter::new(std::fs::File::create(&router_path)?);

        for layer in 0..num_layers {
            let router_key = arch
                .moe_router_key(layer)
                .map(|k| normalize_key(&k, prefixes))
                .unwrap_or_default();

            if let Some(tensor) = get_tensor_f32(&shard_mmaps, &tensor_index, &router_key)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                router_file.write_all(&bytes)?;
            }

            // Also try router bias
            let bias_key = router_key.replace(".weight", ".bias");
            if let Some(tensor) = get_tensor_f32(&shard_mmaps, &tensor_index, &bias_key)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                // Write bias after weight for each layer
                router_file.write_all(&bytes)?;
            }
        }
        router_file.flush()?;
        callbacks.on_stage_done(STAGE_ROUTER_WEIGHTS, 0.0);
    }

    // ── 2. Embeddings ──
    callbacks.on_stage(STAGE_EMBEDDINGS);
    let embed_key = normalize_key(arch.embed_key(), prefixes);
    let embed = get_tensor_f32(&shard_mmaps, &tensor_index, &embed_key)?
        .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
    let vocab_size = embed.shape()[0];
    let embed_data = embed.as_slice().unwrap();
    let embed_bytes = crate::config::dtype::encode_floats(embed_data, dtype);
    std::fs::write(output_dir.join(EMBEDDINGS_BIN), &embed_bytes)?;
    callbacks.on_stage_done(STAGE_EMBEDDINGS, 0.0);

    // ── 3. Down meta (streaming) ──
    //
    // Auto-resume: skip the entire down-meta phase if the prior run
    // already wrote `down_meta.bin`. The file is opaque to us here
    // (we don't reload it), but the loader at the end uses it
    // directly off disk via `mmap`, and the config-write doesn't
    // need any per-layer state from this phase — so a clean skip is
    // safe.
    let resumed_down = checkpoint.is_complete(super::checkpoint::ExtractPhase::DownMeta);
    callbacks.on_stage(STAGE_DOWN_META);
    if resumed_down {
        eprintln!(
            "  Skipping down_meta phase (reusing existing {})",
            DOWN_META_BIN,
        );
    }
    let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> = vec![None; num_layers];

    // Build whole-word vocab once
    let (_ww_ids, _ww_embed) =
        super::build_helpers::build_whole_word_vocab(tokenizer, &embed, vocab_size, hidden_size);

    let down_layer_count = if resumed_down { 0 } else { num_layers };
    for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(down_layer_count) {
        callbacks.on_layer_start(COMP_DOWN, layer, num_layers);
        let start = std::time::Instant::now();

        // Get down matrices for this layer
        let down_matrices: Vec<Array2<f32>> = if expert_format
            == larql_models::ExpertFormat::PackedMxfp4
        {
            // MXFP4: dequantize down_proj_blocks
            let blocks_key = arch.packed_down_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_down_scales_key(layer).unwrap_or_default();
            if let (Some(bi), Some(si)) =
                (tensor_index.get(&blocks_key), tensor_index.get(&scales_key))
            {
                let bst = safetensors::SafeTensors::deserialize(&shard_mmaps[bi.0].mmap)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let sst = safetensors::SafeTensors::deserialize(&shard_mmaps[si.0].mmap)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let bv = bst
                    .tensor(&bi.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let sv = sst
                    .tensor(&si.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let shape = bv.shape();
                let n_exp = shape[0];
                let out_features = shape[1];
                let groups = shape[2];
                let in_features = groups * 32;
                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    bv.data(),
                    sv.data(),
                    n_exp,
                    out_features,
                    groups,
                )?;
                experts
                    .into_iter()
                    .map(|data| Array2::from_shape_vec((out_features, in_features), data).unwrap())
                    .collect()
            } else {
                callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                continue;
            }
        } else if expert_format == larql_models::ExpertFormat::PackedBF16 && is_moe {
            // Hybrid MoE (Gemma 4 26B A4B): use dense FFN down for down_meta.
            // Expert down matrices live per-layer at `layers/layer_{L:02}.weights`
            // (Q4_K), written by the q4k weight writer.
            let down_key = normalize_key(&arch.ffn_down_key(layer), prefixes);
            match get_tensor_f32(&shard_mmaps, &tensor_index, &down_key)? {
                Some(t) => vec![t],
                None => {
                    callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                    continue;
                }
            }
        } else if is_moe && n_experts > 0 {
            let mut mats = Vec::new();
            for expert in 0..n_experts {
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    let nk = normalize_key(&key, prefixes);
                    if let Some(t) = get_tensor_f32(&shard_mmaps, &tensor_index, &nk)? {
                        mats.push(t);
                    }
                }
            }
            mats
        } else {
            let down_key = normalize_key(&arch.ffn_down_key(layer), prefixes);
            match get_tensor_f32(&shard_mmaps, &tensor_index, &down_key)? {
                Some(t) => vec![t],
                None => {
                    callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                    continue;
                }
            }
        };

        if down_matrices.is_empty() {
            callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
            continue;
        }

        let mut feature_offset = 0usize;
        for w_down in &down_matrices {
            let num_features = w_down.shape()[1];
            let batch_size = 1024;

            for batch_start in (0..num_features).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_features);
                callbacks.on_feature_progress(
                    "down",
                    layer,
                    feature_offset + batch_start,
                    down_matrices.iter().map(|m| m.shape()[1]).sum(),
                );

                let w_chunk = w_down
                    .slice(ndarray::s![.., batch_start..batch_end])
                    .to_owned();
                let cpu = larql_compute::CpuBackend;
                use larql_compute::MatMul;
                let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                for feat in batch_start..batch_end {
                    let col = chunk_logits.column(feat - batch_start);
                    let mut scores: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
                    let k = down_top_k.min(scores.len());
                    if k > 0 && k < scores.len() {
                        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                    scores.truncate(k);
                    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let top_k_entries: Vec<larql_models::TopKEntry> = scores
                        .into_iter()
                        .filter_map(|(idx, logit)| {
                            tokenizer
                                .decode(&[idx as u32], true)
                                .ok()
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .map(|token| larql_models::TopKEntry {
                                    token,
                                    token_id: idx as u32,
                                    logit,
                                })
                        })
                        .collect();

                    let (top_token, top_token_id, c_score) =
                        if let Some(first) = top_k_entries.first() {
                            (first.token.clone(), first.token_id, first.logit)
                        } else {
                            (String::new(), 0, 0.0)
                        };

                    let feat_idx = feature_offset + feat;
                    if layer_down_meta.is_none() {
                        *layer_down_meta = Some(Vec::new());
                    }
                    if let Some(ref mut metas) = layer_down_meta {
                        while metas.len() <= feat_idx {
                            metas.push(None);
                        }
                        metas[feat_idx] = Some(crate::FeatureMeta {
                            top_token,
                            top_token_id,
                            c_score,
                            top_k: top_k_entries,
                        });
                    }
                }
            }
            feature_offset += num_features;
        }

        callbacks.on_layer_done(COMP_DOWN, layer, start.elapsed().as_secs_f64() * 1000.0);
    }

    if !resumed_down {
        crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
        callbacks.on_stage_done(STAGE_DOWN_META, 0.0);
        checkpoint.mark(super::checkpoint::ExtractPhase::DownMeta, output_dir)?;
    }

    // ── 4. Tokenizer ──
    callbacks.on_stage(STAGE_TOKENIZER);
    let tokenizer_json = tokenizer
        .to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
    callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);

    // ── 5. Config ──
    let family = arch.family().to_string();
    let config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: false,
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: super::build_helpers::chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level,
        dtype,
        quant,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: Some(VindexModelConfig {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if is_moe {
                Some(crate::MoeConfig {
                    num_experts: n_experts,
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: arch.moe_router_type().to_string(),
                    moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                        Some(arch.moe_intermediate_size())
                    } else {
                        None
                    },
                    hybrid: arch.is_hybrid_moe(),
                })
            } else {
                None
            },
            // Per-layer geometry (Gemma 4)
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            final_logit_softcapping: cfg.final_logit_softcapping,
        }),
        fp4: None,
        ffn_layout: None,
    };

    // Write preliminary index.json (needed by write_model_weights which reads dtype from it)
    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join(INDEX_JSON), config_json)?;

    // ── 6. Model weights (if extract level requires them) ──
    // With quant=q4k we always materialise weights regardless of the
    // declared level — the Q4_K writer emits all of attn, FFN, norms, lm_head
    // in one pass and makes `--level browse --quant q4k` incoherent, so
    // q4k implicitly promotes to "all".
    let needs_weights = extract_level.writes_attn() || quant != QuantFormat::None;
    if needs_weights {
        let shard_refs: Vec<&[u8]> = shard_mmaps.iter().map(|s| s.mmap.as_ref()).collect();
        let streaming_source = crate::format::weights::StreamingWeights {
            shard_mmaps: &shard_refs,
            tensor_index: &tensor_index,
            arch: &*arch,
            num_layers,
        };
        // Thread the extract level into the write options so the
        // writer can skip attn/FFN/lm_head sections per tier.
        let mut level_opts = weight_opts;
        level_opts.level = extract_level;
        match quant {
            QuantFormat::None => {
                crate::format::weights::write_model_weights_with_opts(
                    &streaming_source,
                    output_dir,
                    callbacks,
                    level_opts,
                )?;
            }
            QuantFormat::Q4K => {
                // Q4K doesn't write `up_weights.bin` / `down_weights.bin`
                // at all — the FFN weights live in `interleaved_q4k.bin`.
                // `ffn_compact` is a no-op here by construction. Level
                // gating for Q4K is a future refinement (today Q4K
                // always writes the full set).
                crate::format::weights::write_model_weights_q4k_with_opts(
                    &streaming_source,
                    output_dir,
                    callbacks,
                    q4k_opts,
                )?;
            }
        }
    }

    // Final checksums
    let config_text = std::fs::read_to_string(output_dir.join(INDEX_JSON))?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;
    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();
    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join(INDEX_JSON), config_json)?;

    // Whole extract succeeded — drop the checkpoint so the next
    // visitor sees a clean output dir, not a half-finished one.
    super::checkpoint::Checkpoint::clear(output_dir)?;

    Ok(())
}

/// Get a 2D tensor from mmap'd safetensors, dequantizing to f32.
fn get_tensor_f32(
    shards: &[MmapShard],
    index: &HashMap<String, (usize, String)>,
    key: &str,
) -> Result<Option<Array2<f32>>, VindexError> {
    let (shard_idx, tensor_name) = match index.get(key) {
        Some(v) => v,
        None => return Ok(None),
    };

    let st = safetensors::SafeTensors::deserialize(&shards[*shard_idx].mmap)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let view = st
        .tensor(tensor_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Ok(None);
    }

    let data = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
        safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
        _ => return Ok(None), // skip non-float
    };

    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(Some(arr))
}

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

use crate::config::dtype::write_floats;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

    #[test]
    fn discovers_nested_gguf_directory_instead_of_reporting_no_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        let qdir = model_dir.join("Q4_K_M");
        std::fs::create_dir_all(&qdir).unwrap();
        write_minimal_gguf(
            &qdir.join("Kimi-00001-of-00013.gguf"),
            "deepseek2",
            13,
            true,
        );

        let source = discover_weight_source(&model_dir).unwrap();

        match source {
            WeightSource::Gguf(catalog) => {
                assert_eq!(catalog.files.len(), 1);
                assert!(catalog.files[0].ends_with("Kimi-00001-of-00013.gguf"));
            }
            other => panic!("expected GGUF source, got {other:?}"),
        }
    }

    #[test]
    fn kimi_like_gguf_preflight_reports_actionable_unsupported_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_minimal_gguf(
            &model_dir.join("Kimi-00001-of-00013.gguf"),
            "deepseek2",
            13,
            true,
        );

        let source = discover_weight_source(&model_dir).unwrap();
        let err = preflight_gguf_streaming(&source).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("GGUF"), "{msg}");
        assert!(msg.contains("deepseek2"), "{msg}");
        assert!(msg.contains("split=13"), "{msg}");
        assert!(msg.contains("3D"), "{msg}");
        assert!(!msg.contains("no safetensors"), "{msg}");
    }

    #[test]
    fn build_gguf_catalog_indexes_tensors_across_shards_without_reading_data() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let shard0 = model_dir.join("Kimi-00001-of-00002.gguf");
        let shard1 = model_dir.join("Kimi-00002-of-00002.gguf");
        write_minimal_gguf(&shard0, "deepseek2", 2, true);
        write_minimal_gguf_named_tensor(
            &shard1,
            "deepseek2",
            2,
            "blk.0.attn_q_a.weight",
            &[7168, 1536],
            1,
        );

        let catalog = build_gguf_catalog(&[shard0.clone(), shard1.clone()]).unwrap();

        assert_eq!(catalog.architecture, "deepseek2");
        assert_eq!(catalog.split_count, 2);
        assert_eq!(catalog.files, vec![shard0, shard1]);
        assert_eq!(catalog.tensors.len(), 2);
        assert_eq!(catalog.three_d_tensors, 1);

        let expert = catalog.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(expert.shard_idx, 0);
        assert_eq!(expert.dims, vec![7168, 2048, 384]);
        assert_eq!(expert.tensor_type, 0);
        assert!(expert.is_3d());

        let q_a = catalog.tensor("blk.0.attn_q_a.weight").unwrap();
        assert_eq!(q_a.shard_idx, 1);
        assert_eq!(q_a.dims, vec![7168, 1536]);
        assert!(!q_a.is_3d());
    }

    #[test]
    fn classify_deepseek2_layout_identifies_kimi_packed_expert_roles() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let shard = model_dir.join("Kimi-00001-of-00001.gguf");
        write_minimal_gguf_header(
            &shard,
            "deepseek2",
            1,
            &[
                ("blk.0.ffn_gate_exps.weight", &[7168, 2048, 384], 15),
                ("blk.0.ffn_up_exps.weight", &[7168, 2048, 384], 15),
                ("blk.0.ffn_down_exps.weight", &[2048, 7168, 384], 15),
                ("blk.0.ffn_gate_inp.weight", &[384, 7168], 1),
                ("blk.0.ffn_gate_shexp.weight", &[7168, 2048], 1),
                ("blk.0.attn_q_a.weight", &[1536, 7168], 1),
            ],
        );
        let catalog = build_gguf_catalog(&[shard]).unwrap();

        let layout = classify_deepseek2_layout(&catalog).unwrap();

        assert_eq!(
            layout.packed_experts.get(&(0, GgufExpertComponent::Gate)),
            Some(&"blk.0.ffn_gate_exps.weight".to_string())
        );
        assert_eq!(
            layout.packed_experts.get(&(0, GgufExpertComponent::Up)),
            Some(&"blk.0.ffn_up_exps.weight".to_string())
        );
        assert_eq!(
            layout.packed_experts.get(&(0, GgufExpertComponent::Down)),
            Some(&"blk.0.ffn_down_exps.weight".to_string())
        );
        assert_eq!(
            layout.routers.get(&0),
            Some(&"blk.0.ffn_gate_inp.weight".to_string())
        );
        assert_eq!(
            layout.shared_experts.get(&(0, GgufExpertComponent::Gate)),
            Some(&"blk.0.ffn_gate_shexp.weight".to_string())
        );
        assert!(!layout
            .attention
            .contains(&"blk.0.attn_q_a.weight".to_string()));
    }

    #[test]
    fn packed_expert_slice_maps_kimi_3d_dims_to_2d_expert_geometry() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let shard = model_dir.join("Kimi-00001-of-00001.gguf");
        write_minimal_gguf_header(
            &shard,
            "deepseek2",
            1,
            &[
                ("blk.0.ffn_gate_exps.weight", &[7168, 2048, 384], 15),
                ("blk.0.ffn_down_exps.weight", &[2048, 7168, 384], 15),
            ],
        );
        let catalog = build_gguf_catalog(&[shard]).unwrap();

        let gate =
            packed_expert_slice(catalog.tensor("blk.0.ffn_gate_exps.weight").unwrap(), 7).unwrap();
        assert_eq!(gate.rows, 2048);
        assert_eq!(gate.cols, 7168);
        assert_eq!(gate.expert_count, 384);
        assert_eq!(gate.expert_idx, 7);
        assert_eq!(gate.element_len, 7168 * 2048);
        assert_eq!(gate.element_offset, 7 * 7168 * 2048);

        let down = packed_expert_slice(catalog.tensor("blk.0.ffn_down_exps.weight").unwrap(), 383)
            .unwrap();
        assert_eq!(down.rows, 7168);
        assert_eq!(down.cols, 2048);
        assert_eq!(down.expert_count, 384);
        assert_eq!(down.element_offset, 383 * 2048 * 7168);

        let err = packed_expert_slice(catalog.tensor("blk.0.ffn_gate_exps.weight").unwrap(), 384)
            .unwrap_err();
        assert!(err.to_string().contains("expert index 384 out of range"));
    }

    #[test]
    fn build_gguf_streaming_routes_deepseek2_through_gate_phase_before_next_blocker() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let output_dir = tmp.path().join("out.vindex");
        let gate_name = "blk.0.ffn_gate_exps.weight";
        let gate_shard = model_dir.join("Kimi-00001-of-00001.gguf");
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        write_minimal_gguf_header_with_f32_payload(
            &gate_shard,
            "deepseek2",
            1,
            gate_name,
            &[2, 2, 2],
            &values,
        );

        let source = discover_weight_source(&model_dir).unwrap();
        let WeightSource::Gguf(catalog) = source else {
            panic!("expected GGUF catalog");
        };
        let mut callbacks = crate::extract::callbacks::SilentBuildCallbacks;

        let err = build_gguf_streaming(&catalog, &output_dir, StorageDtype::F32, &mut callbacks)
            .unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("after writing gate_vectors.bin"), "{msg}");
        let gate_bytes = std::fs::read(output_dir.join(GATE_VECTORS_BIN)).unwrap();
        assert_eq!(gate_bytes.len(), values.len() * std::mem::size_of::<f32>());
        let decoded: Vec<f32> = gate_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn write_deepseek2_packed_gate_vectors_streams_one_layer_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let gate_name = "blk.0.ffn_gate_exps.weight";
        let gate_shard = tmp.path().join("Kimi-00001-of-00001.gguf");
        let values = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        write_minimal_gguf_header_with_f32_payload(
            &gate_shard,
            "deepseek2",
            1,
            gate_name,
            &[4, 2, 2],
            &values,
        );

        let catalog = build_gguf_catalog(&[gate_shard]).unwrap();
        let layout = classify_deepseek2_layout(&catalog).unwrap();
        let mut out = Vec::new();

        let info = write_deepseek2_packed_gate_vectors(
            &mut out,
            &catalog,
            &layout,
            0,
            17,
            StorageDtype::F32,
        )
        .unwrap();

        assert_eq!(info.layer, 0);
        assert_eq!(info.num_features, 4);
        assert_eq!(info.num_experts, Some(2));
        assert_eq!(info.num_features_per_expert, Some(2));
        assert_eq!(info.offset, 17);
        assert_eq!(
            info.length,
            (values.len() * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(out.len(), values.len() * std::mem::size_of::<f32>());
        let decoded: Vec<f32> = out
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn read_deepseek2_packed_expert_layer_dispatches_quantized_roles() {
        let tmp = tempfile::tempdir().unwrap();
        let gate_name = "blk.0.ffn_gate_exps.weight";
        let up_name = "blk.0.ffn_up_exps.weight";
        let down_name = "blk.0.ffn_down_exps.weight";

        let gate_shard = tmp.path().join("Kimi-00001-of-00003.gguf");
        let mut gate_values = vec![1.0f32; 256];
        gate_values.extend(vec![2.0f32; 256]);
        write_minimal_gguf_header_with_f32_payload(
            &gate_shard,
            "deepseek2",
            3,
            gate_name,
            &[256, 1, 2],
            &gate_values,
        );

        let up_shard = tmp.path().join("Kimi-00002-of-00003.gguf");
        let mut up_payload = q4_k_constant_block(3);
        up_payload.extend(q4_k_constant_block(4));
        write_minimal_gguf_header_with_raw_payload(
            &up_shard,
            "deepseek2",
            3,
            up_name,
            &[256, 1, 2],
            larql_models::quant::ggml::TYPE_Q4_K,
            &up_payload,
        );

        let down_shard = tmp.path().join("Kimi-00003-of-00003.gguf");
        let mut down_payload = q6_k_constant_block(35);
        down_payload.extend(q6_k_constant_block(36));
        write_minimal_gguf_header_with_raw_payload(
            &down_shard,
            "deepseek2",
            3,
            down_name,
            &[256, 1, 2],
            larql_models::quant::ggml::TYPE_Q6_K,
            &down_payload,
        );

        let catalog = build_gguf_catalog(&[gate_shard, up_shard, down_shard]).unwrap();
        let layout = classify_deepseek2_layout(&catalog).unwrap();

        let expert = read_deepseek2_packed_expert_layer(&catalog, &layout, 0, 1).unwrap();
        assert_eq!(expert.gate.shape(), &[1, 256]);
        assert!(expert.gate.as_slice().unwrap().iter().all(|v| *v == 2.0));
        assert_eq!(expert.up.shape(), &[1, 256]);
        assert!(expert.up.as_slice().unwrap().iter().all(|v| *v == 4.0));
        assert_eq!(expert.down.shape(), &[1, 256]);
        assert!(expert.down.as_slice().unwrap().iter().all(|v| *v == 4.0));
    }

    #[test]
    fn read_packed_expert_slice_f32_mmaps_one_expert_without_full_tensor_load() {
        let tmp = tempfile::tempdir().unwrap();
        let shard = tmp.path().join("Kimi-00001-of-00001.gguf");
        let name = "blk.0.ffn_gate_exps.weight";
        write_minimal_gguf_header_with_f32_payload(
            &shard,
            "deepseek2",
            1,
            name,
            &[4, 3, 2],
            &[
                // Expert 0: 3 rows x 4 cols, row-major after GGUF [cols, rows].
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
                // Expert 1.
                100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 123.0,
            ],
        );
        let catalog = build_gguf_catalog(&[shard]).unwrap();

        let expert0 = read_packed_expert_slice_f32(&catalog, name, 0).unwrap();
        assert_eq!(expert0.shape(), &[3, 4]);
        assert_eq!(
            expert0.as_slice().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,]
        );

        let expert1 = read_packed_expert_slice_f32(&catalog, name, 1).unwrap();
        assert_eq!(expert1.shape(), &[3, 4]);
        assert_eq!(
            expert1.as_slice().unwrap(),
            &[100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 123.0,]
        );

        let err = read_packed_expert_slice_f32(&catalog, name, 2).unwrap_err();
        assert!(err.to_string().contains("expert index 2 out of range"));
    }

    #[test]
    fn read_packed_expert_slice_q4_k_mmaps_block_aligned_expert_only() {
        let tmp = tempfile::tempdir().unwrap();
        let shard = tmp.path().join("Kimi-00001-of-00001.gguf");
        let name = "blk.0.ffn_gate_exps.weight";
        let mut payload = q4_k_constant_block(1);
        payload.extend(q4_k_constant_block(2));
        write_minimal_gguf_header_with_raw_payload(
            &shard,
            "deepseek2",
            1,
            name,
            &[256, 1, 2],
            larql_models::quant::ggml::TYPE_Q4_K,
            &payload,
        );
        let catalog = build_gguf_catalog(&[shard]).unwrap();

        let expert0 = read_packed_expert_slice_q4_k(&catalog, name, 0).unwrap();
        assert_eq!(expert0.shape(), &[1, 256]);
        assert!(expert0.as_slice().unwrap().iter().all(|v| *v == 1.0));

        let expert1 = read_packed_expert_slice_q4_k(&catalog, name, 1).unwrap();
        assert_eq!(expert1.shape(), &[1, 256]);
        assert!(expert1.as_slice().unwrap().iter().all(|v| *v == 2.0));

        let unaligned_shard = tmp.path().join("Kimi-unaligned.gguf");
        write_minimal_gguf_header_with_raw_payload(
            &unaligned_shard,
            "deepseek2",
            1,
            name,
            &[128, 1, 1],
            larql_models::quant::ggml::TYPE_Q4_K,
            &[],
        );
        let unaligned_catalog = build_gguf_catalog(&[unaligned_shard]).unwrap();
        let err = read_packed_expert_slice_q4_k(&unaligned_catalog, name, 0).unwrap_err();
        assert!(err.to_string().contains("block-aligned"), "{err}");
    }

    #[test]
    fn read_packed_expert_slice_q6_k_mmaps_block_aligned_expert_only() {
        let tmp = tempfile::tempdir().unwrap();
        let shard = tmp.path().join("Kimi-00001-of-00001.gguf");
        let name = "blk.0.ffn_down_exps.weight";
        let mut payload = q6_k_constant_block(33);
        payload.extend(q6_k_constant_block(34));
        write_minimal_gguf_header_with_raw_payload(
            &shard,
            "deepseek2",
            1,
            name,
            &[256, 1, 2],
            larql_models::quant::ggml::TYPE_Q6_K,
            &payload,
        );
        let catalog = build_gguf_catalog(&[shard]).unwrap();

        let expert0 = read_packed_expert_slice_q6_k(&catalog, name, 0).unwrap();
        assert_eq!(expert0.shape(), &[1, 256]);
        assert!(expert0.as_slice().unwrap().iter().all(|v| *v == 1.0));

        let expert1 = read_packed_expert_slice_q6_k(&catalog, name, 1).unwrap();
        assert_eq!(expert1.shape(), &[1, 256]);
        assert!(expert1.as_slice().unwrap().iter().all(|v| *v == 2.0));

        let unaligned_shard = tmp.path().join("Kimi-unaligned-q6k.gguf");
        write_minimal_gguf_header_with_raw_payload(
            &unaligned_shard,
            "deepseek2",
            1,
            name,
            &[128, 1, 1],
            larql_models::quant::ggml::TYPE_Q6_K,
            &[],
        );
        let unaligned_catalog = build_gguf_catalog(&[unaligned_shard]).unwrap();
        let err = read_packed_expert_slice_q6_k(&unaligned_catalog, name, 0).unwrap_err();
        assert!(err.to_string().contains("block-aligned"), "{err}");
    }

    fn write_minimal_gguf(path: &std::path::Path, arch: &str, split_count: u16, include_3d: bool) {
        if include_3d {
            write_minimal_gguf_named_tensor(
                path,
                arch,
                split_count,
                "blk.0.ffn_gate_exps.weight",
                &[7168, 2048, 384],
                0,
            );
        } else {
            write_minimal_gguf_header(path, arch, split_count, &[]);
        }
    }

    fn write_minimal_gguf_named_tensor(
        path: &std::path::Path,
        arch: &str,
        split_count: u16,
        name: &str,
        dims: &[u64],
        tensor_type: u32,
    ) {
        write_minimal_gguf_header(path, arch, split_count, &[(name, dims, tensor_type)]);
    }

    fn write_minimal_gguf_header_with_f32_payload(
        path: &std::path::Path,
        arch: &str,
        split_count: u16,
        name: &str,
        dims: &[u64],
        values: &[f32],
    ) {
        let mut payload = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            payload.extend(value.to_le_bytes());
        }
        write_minimal_gguf_header_with_raw_payload(
            path,
            arch,
            split_count,
            name,
            dims,
            0,
            &payload,
        );
    }

    fn write_minimal_gguf_header_with_raw_payload(
        path: &std::path::Path,
        arch: &str,
        split_count: u16,
        name: &str,
        dims: &[u64],
        tensor_type: u32,
        payload: &[u8],
    ) {
        write_minimal_gguf_header(path, arch, split_count, &[(name, dims, tensor_type)]);
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let pos = f.seek(std::io::SeekFrom::End(0)).unwrap();
        let data_offset = pos.div_ceil(32) * 32;
        let padding = (data_offset - pos) as usize;
        f.write_all(&vec![0u8; padding]).unwrap();
        f.write_all(payload).unwrap();
    }

    fn q4_k_constant_block(nibble: u8) -> Vec<u8> {
        assert!(nibble <= 0x0f);
        let mut block = vec![0u8; larql_models::quant::ggml::Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0 scale d
        block[2..4].copy_from_slice(&0u16.to_le_bytes()); // f16 0.0 dmin
        block[4..8].fill(1); // scales 0..3
        block[8..12].fill(0); // mins 0..3; ignored because dmin=0
        block[12..16].fill(1); // scales 4..7 in low nibble; mins ignored because dmin=0
        block[16..].fill(nibble | (nibble << 4));
        block
    }

    fn q6_k_constant_block(raw_value: u8) -> Vec<u8> {
        assert!(raw_value < 64);
        let mut block = vec![0u8; larql_models::quant::ggml::Q6_K_BLOCK_BYTES];
        let lo4 = raw_value & 0x0f;
        let hi2 = (raw_value >> 4) & 0x03;
        block[0..128].fill(lo4 | (lo4 << 4));
        let hi_byte = hi2 | (hi2 << 2) | (hi2 << 4) | (hi2 << 6);
        block[128..192].fill(hi_byte);
        block[192..208].fill(1); // int8 scales
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0 d
        block
    }

    fn write_minimal_gguf_header(
        path: &std::path::Path,
        arch: &str,
        split_count: u16,
        tensors: &[(&str, &[u64], u32)],
    ) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&0x4655_4747u32.to_le_bytes()).unwrap(); // GGUF
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version
        f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap(); // metadata count

        write_string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap(); // string
        write_string(&mut f, arch);

        write_string(&mut f, "split.count");
        f.write_all(&2u32.to_le_bytes()).unwrap(); // uint16
        f.write_all(&split_count.to_le_bytes()).unwrap();

        for (name, dims, tensor_type) in tensors {
            write_string(&mut f, name);
            f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for dim in *dims {
                f.write_all(&dim.to_le_bytes()).unwrap();
            }
            f.write_all(&tensor_type.to_le_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap(); // offset
        }
    }

    fn write_string(mut w: impl Write, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
}
