//! The `general.*` keys the reference converter writes ahead of the
//! architecture's own: sampling defaults from `generation_config.json`,
//! then the naming heuristic's components and the size label, in
//! `Metadata.set_gguf_meta_model`'s order.

use larql_models::loading::gguf::GgufValue;

use super::naming::{naming_from_id, size_label};

/// How a `generation_config.json` value is typed in the file.
#[derive(Clone, Copy)]
enum SamplingType {
    Str,
    I32,
    F32,
}

/// `(generation_config key, general.sampling.* suffix, type)`, in the
/// reference's order.
const SAMPLING_KEYS: &[(&str, &str, SamplingType)] = &[
    ("sequence", "sequence", SamplingType::Str),
    ("top_k", "top_k", SamplingType::I32),
    ("top_p", "top_p", SamplingType::F32),
    ("min_p", "min_p", SamplingType::F32),
    ("xtc_probability", "xtc_probability", SamplingType::F32),
    ("xtc_threshold", "xtc_threshold", SamplingType::F32),
    ("temperature", "temp", SamplingType::F32),
    ("penalty_last_n", "penalty_last_n", SamplingType::I32),
    ("penalty_repeat", "penalty_repeat", SamplingType::F32),
    ("mirostat", "mirostat", SamplingType::I32),
    ("mirostat_tau", "mirostat_tau", SamplingType::F32),
    ("mirostat_eta", "mirostat_eta", SamplingType::F32),
];

fn sampling_value(v: &serde_json::Value, ty: SamplingType) -> Option<GgufValue> {
    match ty {
        SamplingType::Str => v.as_str().map(|s| GgufValue::String(s.into())),
        SamplingType::I32 => v.as_i64().map(|i| GgufValue::I32(i as i32)),
        SamplingType::F32 => v.as_f64().map(|f| GgufValue::F32(f as f32)),
    }
}

/// `general.sampling.*` from the container's generation config.
pub fn sampling_entries(generation_config: Option<&serde_json::Value>) -> Vec<(String, GgufValue)> {
    let Some(config) = generation_config else {
        return Vec::new();
    };
    SAMPLING_KEYS
        .iter()
        .filter_map(|(from, suffix, ty)| {
            let v = config.get(*from).filter(|v| !v.is_null())?;
            Some((
                format!("general.sampling.{suffix}"),
                sampling_value(v, *ty)?,
            ))
        })
        .collect()
}

/// `general.name` … `general.size_label` for a model whose source the
/// container records as `model_id`, carrying `total_params` elements.
pub fn naming_entries(model_id: &str, total_params: u64) -> Vec<(String, GgufValue)> {
    let naming = naming_from_id(model_id, total_params as i64);
    let label = naming
        .size_label
        .clone()
        .or_else(|| (total_params > 0).then(|| size_label(total_params)));
    [
        ("general.name", naming.name),
        ("general.version", naming.version),
        ("general.organization", naming.organization),
        ("general.finetune", naming.finetune),
        ("general.basename", naming.basename),
        ("general.size_label", label),
    ]
    .into_iter()
    .filter_map(|(k, v)| Some((k.to_string(), GgufValue::String(v?))))
    .collect()
}
