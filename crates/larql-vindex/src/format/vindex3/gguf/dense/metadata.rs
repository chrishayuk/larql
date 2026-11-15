//! The architecture's keys, each from a graph fact, in the order the
//! reference converter writes them (`TextModel.set_gguf_parameters`, then
//! the Gemma 3 override, whose re-adds keep their first position).

use larql_models::loading::gguf::GgufValue;

use super::programme::Programme;
use super::target::DenseTarget;

/// `gguf.GGML_QUANT_VERSION`.
pub const QUANTIZATION_VERSION: u32 = 2;
/// `LlamaFileType.ALL_F32`.
pub const FILE_TYPE_ALL_F32: u32 = 0;
/// `LlamaFileType.MOSTLY_BF16`.
pub const FILE_TYPE_MOSTLY_BF16: u32 = 32;
/// `RopeScalingType.LINEAR`'s spelling.
const ROPE_SCALING_LINEAR: &str = "linear";

/// The facts the arch keys state, gathered by the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchFacts<'a> {
    pub target: &'a DenseTarget,
    pub programme: &'a Programme,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub context_length: u64,
    pub rms_eps: f64,
    pub file_type: u32,
    pub final_logit_softcapping: Option<f32>,
}

pub fn arch_entries(f: &ArchFacts<'_>) -> Vec<(String, GgufValue)> {
    let t = f.target;
    let p = f.programme;
    let u = |v: usize| GgufValue::U32(v as u32);
    let mut out = vec![
        (t.key("block_count"), u(f.num_layers)),
        (
            t.key("context_length"),
            GgufValue::U32(f.context_length as u32),
        ),
        (t.key("embedding_length"), u(f.hidden_size)),
        (t.key("feed_forward_length"), u(p.intermediate_size)),
        (t.key("attention.head_count"), u(p.num_q_heads)),
        (t.key("attention.head_count_kv"), u(p.num_kv_heads)),
    ];
    if let Some(factor) = p.full_rope.linear_factor {
        out.push((
            t.key("rope.scaling.type"),
            GgufValue::String(ROPE_SCALING_LINEAR.into()),
        ));
        out.push((t.key("rope.scaling.factor"), GgufValue::F32(factor as f32)));
    }
    out.push((
        t.key("rope.freq_base"),
        GgufValue::F32(p.full_rope.theta as f32),
    ));
    if let Some(s) = &p.sliding {
        out.push((t.key("rope.freq_base_swa"), GgufValue::F32(s.theta as f32)));
    }
    out.push((
        t.key("attention.layer_norm_rms_epsilon"),
        GgufValue::F32(f.rms_eps as f32),
    ));
    out.push((t.key("attention.key_length"), u(p.head_dim)));
    out.push((t.key("attention.value_length"), u(p.head_dim)));
    out.push(("general.file_type".into(), GgufValue::U32(f.file_type)));
    if let Some(cap) = f.final_logit_softcapping {
        out.push((t.key("final_logit_softcapping"), GgufValue::F32(cap)));
    }
    if let Some(s) = &p.sliding {
        out.push((t.key("attention.sliding_window"), u(s.window)));
        // The runtime assumes its default period; any other is stated.
        if s.period != t.default_swa_period {
            out.push((t.key("attention.sliding_window_pattern"), u(s.period)));
        }
    }
    out.push((
        "general.quantization_version".into(),
        GgufValue::U32(QUANTIZATION_VERSION),
    ));
    out
}
