//! The checkpoint's declared *stored representation* — `quantization_config`.
//!
//! A quantised checkpoint states how its weights are encoded on disk:
//! `quant_method` names the scheme, `modules_to_not_convert` the modules
//! left in the base dtype. That is a tensor-representation fact, not an
//! execution semantic — two checkpoints differing only here compute the same
//! function — but it decides what the bytes *mean*: `openai/gpt-oss-20b`
//! stores its experts as `U8` `*_blocks` / `*_scales` pairs that are MXFP4
//! only by this declaration. A reader that dropped it would place those
//! tensors as raw bytes.
//!
//! Read once, here, and recorded as consumed paths so `config_keys` credits
//! the read (parser consumption is a recorded fact, not a name match — the
//! same discipline as [`super::components`]). The VINDEX3 placement names
//! the affected objects' encoding from it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `config.json` container this reader owns.
pub const QUANTIZATION_CONFIG_KEY: &str = "quantization_config";
/// The scheme name, in the checkpoint's own spelling.
pub const QUANT_METHOD_KEY: &str = "quant_method";
/// Module patterns (glob `*` over dotted paths) kept in the base dtype.
pub const MODULES_TO_NOT_CONVERT_KEY: &str = "modules_to_not_convert";

/// HF's spelling of the MXFP4 scheme (`quantization_config.quant_method`).
pub const QUANT_METHOD_MXFP4: &str = "mxfp4";

/// What the checkpoint declares about its stored representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredRepresentation {
    /// `quant_method`, verbatim (e.g. `mxfp4`).
    pub method: String,
    /// `modules_to_not_convert`, verbatim glob patterns over module paths.
    #[serde(default)]
    pub excluded_modules: Vec<String>,
}

impl StoredRepresentation {
    /// Whether a tensor path is excluded from the scheme by
    /// `modules_to_not_convert`. Patterns are dotted module paths with `*`
    /// wildcards (`model.layers.*.self_attn`); a tensor is excluded when
    /// its name starts with a module the pattern matches.
    pub fn excludes(&self, tensor_name: &str) -> bool {
        self.excluded_modules
            .iter()
            .any(|pattern| glob_prefix_matches(pattern, tensor_name))
    }
}

/// `pattern` (with `*` wildcards) matches a *prefix* of `name` on module
/// boundaries: `model.layers.*.self_attn` matches
/// `model.layers.3.self_attn.q_proj.weight`.
fn glob_prefix_matches(pattern: &str, name: &str) -> bool {
    fn go(p: &[&str], n: &[&str]) -> bool {
        match (p.first(), n.first()) {
            (None, _) => true,
            (Some(&"*"), None) => false,
            (Some(&"*"), Some(_)) => go(&p[1..], &n[1..]) || go(p, &n[1..]),
            (Some(seg), Some(head)) => seg == head && go(&p[1..], &n[1..]),
            (Some(_), None) => false,
        }
    }
    let p: Vec<&str> = pattern.split('.').collect();
    let n: Vec<&str> = name.split('.').collect();
    go(&p, &n)
}

/// One reader's result: the fact and the exact paths it read.
#[derive(Debug, Clone)]
pub struct RepresentationReading {
    pub representation: StoredRepresentation,
    pub consumed_paths: BTreeSet<String>,
}

/// Read `quantization_config`, when the checkpoint declares one with a
/// `quant_method`. Anything else under the container is left unread and
/// therefore unconsumed — surfaced by the planner, not swallowed here.
pub fn read_stored_representation(config: &Value) -> Option<RepresentationReading> {
    let block = config.get(QUANTIZATION_CONFIG_KEY)?.as_object()?;
    let method = block.get(QUANT_METHOD_KEY)?.as_str()?.to_string();
    let mut consumed_paths = BTreeSet::new();
    consumed_paths.insert(format!("{QUANTIZATION_CONFIG_KEY}.{QUANT_METHOD_KEY}"));
    let excluded_modules = match block.get(MODULES_TO_NOT_CONVERT_KEY) {
        Some(Value::Array(items)) => {
            consumed_paths.insert(format!(
                "{QUANTIZATION_CONFIG_KEY}.{MODULES_TO_NOT_CONVERT_KEY}"
            ));
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        }
        _ => Vec::new(),
    };
    Some(RepresentationReading {
        representation: StoredRepresentation {
            method,
            excluded_modules,
        },
        consumed_paths,
    })
}

#[cfg(test)]
mod tests;
