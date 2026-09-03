//! Every qwen35 GGUF key, and the graph fact that produces it.
//!
//! The rule this module exists to enforce: **no literal unless it is a
//! target constant.** `general.architecture = "qwen35"` is a fact about
//! llama.cpp. `qwen35.ssm.state_size = 128` is a fact about Qwen3.8, and
//! writing it here rather than deriving it would put model knowledge on
//! the target side — the exact leak the independent-backend test exists
//! to catch.
//!
//! Two conventions genuinely belong here, because they are the target's
//! and not the model's:
//!
//! - **MRoPE sections are padded to four.** The graph declares three
//!   (`[11, 11, 10]`); llama.cpp's loader wants a four-element array
//!   with a trailing zero. The zero is llama.cpp's spelling, not a
//!   fourth section the model has.
//! - **`full_attention_interval` is derived, not assumed.** The
//!   converter's default is 4 and Qwen3.8 happens to be 4, but taking
//!   the default would mean a model with a different cadence silently
//!   exports the wrong layer programme. It comes from the declared
//!   per-layer operators.
//!
//! One namespace trap worth stating: `general.file_type = 39`
//! (`MOSTLY_NVFP4`) and `GGML_TYPE_NVFP4 = 40` are unrelated
//! enumerations that happen to sit beside each other. Using one where
//! the other belongs produces a file that loads and misreads.

use crate::format::vindex3::graph::surface::ExecutionSurface;

/// Facts about llama.cpp, not about any model.
pub const ARCHITECTURE: &str = "qwen35";
pub const GENERAL_TYPE: &str = "model";
/// GGUF quantization-version 2.
pub const QUANTIZATION_VERSION: u32 = 2;
/// `LLAMA_FTYPE_MOSTLY_NVFP4`. Not `GGML_TYPE_NVFP4`, which is 40.
pub const FILE_TYPE_MOSTLY_NVFP4: u32 = 39;
/// `LLAMA_FTYPE_MOSTLY_BF16`, for a canonical-selection export.
pub const FILE_TYPE_MOSTLY_BF16: u32 = 32;
/// llama.cpp's MRoPE array width.
pub const ROPE_SECTION_SLOTS: usize = 4;

/// A key/value pair destined for the GGUF header.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Str(String),
    U32(u32),
    F32(f32),
    ArrU32(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetaKey {
    pub key: String,
    pub value: MetaValue,
    /// The graph fact this came from, or `"target constant"`. Carried so
    /// a reviewer can audit the table without reading the code.
    pub derived_from: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataError {
    /// A fact the preflight should already have refused on.
    Missing(&'static str),
    /// The layer cadence does not repeat, so no single interval
    /// describes it — llama.cpp would need the explicit
    /// `attention.recurrent_layers` array instead.
    IrregularAttentionCadence { positions: Vec<usize> },
}

/// Which layers attend, in order. The caller reads these from the
/// graph's declared per-layer operator, never from layer arithmetic.
pub fn full_attention_interval(
    attending_layers: &[usize],
    num_layers: usize,
) -> Result<u32, MetadataError> {
    if attending_layers.len() < 2 {
        return Err(MetadataError::IrregularAttentionCadence {
            positions: attending_layers.to_vec(),
        });
    }
    let first = attending_layers[1] - attending_layers[0];
    let regular = attending_layers
        .windows(2)
        .all(|w| w[1] - w[0] == first)
        // The cadence must also carry to the end of the stack, or the
        // interval describes a prefix and lies about the rest.
        && attending_layers.last().is_some_and(|&l| l + first >= num_layers);
    if !regular {
        return Err(MetadataError::IrregularAttentionCadence {
            positions: attending_layers.to_vec(),
        });
    }
    Ok(first as u32)
}

/// Pad the graph's declared MRoPE sections into llama.cpp's fixed array.
pub fn rope_sections(declared: &[u32]) -> Result<Vec<u32>, MetadataError> {
    if declared.is_empty() || declared.len() > ROPE_SECTION_SLOTS {
        return Err(MetadataError::Missing("position.section"));
    }
    let mut out = declared.to_vec();
    out.resize(ROPE_SECTION_SLOTS, 0);
    Ok(out)
}

/// Build the complete qwen35 metadata table from graph facts.
///
/// Eight parameters because eight separate authorities feed the table;
/// bundling them into a struct would rename the count, not reduce it.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_metadata(
    surface: &ExecutionSurface,
    num_layers: usize,
    hidden_size: usize,
    rope_theta: f64,
    declared_sections: &[u32],
    rotary_fraction: f64,
    attending_layers: &[usize],
    nvfp4_in_use: bool,
) -> Result<Vec<MetaKey>, MetadataError> {
    let attn = surface
        .attention
        .as_ref()
        .ok_or(MetadataError::Missing("execution.attention"))?;
    let ffn = surface
        .ffn
        .as_ref()
        .ok_or(MetadataError::Missing("execution.ffn"))?;
    let la = surface
        .linear_attention
        .as_ref()
        .ok_or(MetadataError::Missing("execution.linear_attention"))?;
    let context = surface
        .context_length
        .ok_or(MetadataError::Missing("execution.context_length"))?;
    // `feed_forward_length` is the DENSE width. A wholly-routed component
    // has none, and stamping a zero would make the file misdescribe
    // itself the way a BF16 export stamped MOSTLY_NVFP4 would.
    let feed_forward_length = ffn
        .intermediate_size
        .ok_or(MetadataError::Missing("execution.ffn.intermediate_size"))?;

    let k = |key: &str, value: MetaValue, derived_from: &'static str| MetaKey {
        key: key.to_string(),
        value,
        derived_from,
    };

    Ok(vec![
        k(
            "general.type",
            MetaValue::Str(GENERAL_TYPE.into()),
            "target constant",
        ),
        k(
            "general.architecture",
            MetaValue::Str(ARCHITECTURE.into()),
            "target constant",
        ),
        k(
            "general.quantization_version",
            MetaValue::U32(QUANTIZATION_VERSION),
            "target constant",
        ),
        // file_type is display metadata, but a BF16 export stamped
        // MOSTLY_NVFP4 would still be the file lying about itself.
        k(
            "general.file_type",
            MetaValue::U32(if nvfp4_in_use {
                FILE_TYPE_MOSTLY_NVFP4
            } else {
                FILE_TYPE_MOSTLY_BF16
            }),
            "the selected representation programme",
        ),
        k(
            "qwen35.block_count",
            MetaValue::U32(num_layers as u32),
            "component.num_layers",
        ),
        k(
            "qwen35.context_length",
            MetaValue::U32(context as u32),
            "execution.context_length",
        ),
        k(
            "qwen35.embedding_length",
            MetaValue::U32(hidden_size as u32),
            "component.hidden_size",
        ),
        k(
            "qwen35.feed_forward_length",
            MetaValue::U32(feed_forward_length as u32),
            "ffn.intermediate_size",
        ),
        k(
            "qwen35.attention.head_count",
            MetaValue::U32(attn.num_q_heads as u32),
            "attention.num_q_heads",
        ),
        k(
            "qwen35.attention.head_count_kv",
            MetaValue::U32(attn.num_kv_heads as u32),
            "attention.num_kv_heads",
        ),
        k(
            "qwen35.attention.key_length",
            MetaValue::U32(attn.head_dim as u32),
            "attention.head_dim",
        ),
        k(
            "qwen35.attention.value_length",
            MetaValue::U32(attn.head_dim as u32),
            "attention.head_dim",
        ),
        k(
            "qwen35.attention.layer_norm_rms_epsilon",
            MetaValue::F32(surface.norm.pre.eps as f32),
            "norm.pre.eps",
        ),
        k(
            "qwen35.rope.freq_base",
            MetaValue::F32(rope_theta as f32),
            "layer position.theta",
        ),
        k(
            "qwen35.rope.dimension_sections",
            MetaValue::ArrU32(rope_sections(declared_sections)?),
            "layer position.section, padded to the target's four slots",
        ),
        k(
            "qwen35.rope.dimension_count",
            MetaValue::U32((attn.head_dim as f64 * rotary_fraction) as u32),
            "attention.head_dim x position.rotary_fraction",
        ),
        k(
            "qwen35.ssm.conv_kernel",
            MetaValue::U32(la.conv_kernel as u32),
            "linear_attention.conv_kernel",
        ),
        k(
            "qwen35.ssm.inner_size",
            MetaValue::U32((la.value_heads * la.value_head_dim) as u32),
            "value_heads x value_head_dim",
        ),
        k(
            "qwen35.ssm.state_size",
            MetaValue::U32(la.key_head_dim as u32),
            "linear_attention.key_head_dim",
        ),
        k(
            "qwen35.ssm.time_step_rank",
            MetaValue::U32(la.value_heads as u32),
            "linear_attention.value_heads",
        ),
        k(
            "qwen35.ssm.group_count",
            MetaValue::U32(la.key_heads as u32),
            "linear_attention.key_heads",
        ),
        k(
            "qwen35.full_attention_interval",
            MetaValue::U32(full_attention_interval(attending_layers, num_layers)?),
            "declared per-layer operators",
        ),
    ])
}

#[cfg(test)]
mod tests;
