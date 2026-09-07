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
/// HF's spelling of the fine-grained (block-wise) FP8 scheme — the
/// DeepSeek-V3 lineage's, and GLM-5.3-Flash's.
pub const QUANT_METHOD_FP8: &str = "fp8";
/// The element format within an FP8 scheme (`e4m3` / `e5m2`).
pub const FMT_KEY: &str = "fmt";
/// The declared weight tile, `[block_rows, block_cols]`.
pub const WEIGHT_BLOCK_SIZE_KEY: &str = "weight_block_size";
/// How the scheme quantises ACTIVATIONS at run time — a compute-path
/// fact, not a storage one.
pub const ACTIVATION_SCHEME_KEY: &str = "activation_scheme";
/// The only element format this build decodes. `e5m2` is a different
/// codec, not a variant of this one.
pub const FMT_E4M3: &str = "e4m3";

/// What the checkpoint declares about its stored representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredRepresentation {
    /// `quant_method`, verbatim (e.g. `mxfp4`).
    pub method: String,
    /// `modules_to_not_convert`, verbatim glob patterns over module paths.
    #[serde(default)]
    pub excluded_modules: Vec<String>,
    /// `fmt` — which element codec the scheme's values use. **Load
    /// bearing**: `e5m2` and `e4m3` are different formats with the same
    /// byte width, so decoding one as the other produces plausible
    /// numbers from every byte. A scheme declaring an unrecognised `fmt`
    /// must reach a refusal, which is why this is carried rather than
    /// assumed from `method`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fmt: Option<String>,
    /// `weight_block_size` — the declared `[block_rows, block_cols]`
    /// tile.
    ///
    /// **Provenance with a cross-check, NOT the authority.** The tile a
    /// dequantiser applies is derived per tensor from the scale grid
    /// (`weight.shape / weight_scale_inv.shape`), because one checkpoint
    /// may ship several grids — transformers' own dequantiser does this
    /// and cites MoE experts at `[1, 32]` beside dense linears at
    /// `[128, 128]`. Carried so the declaration can be *checked* against
    /// what the tensors say, which is worth more than using it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_block_size: Option<Vec<usize>>,
    /// `activation_scheme` — how the scheme quantises activations at run
    /// time (`dynamic` on GLM-5.3-Flash).
    ///
    /// Carried and deliberately **not** claimed as represented: it
    /// describes an FP8 *compute* path, and a build that dequantises
    /// weights to f32 and runs an f32 GEMM is computing something else —
    /// numerically close, but not the same route. Recording it lets the
    /// planner say so instead of staying silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_scheme: Option<String>,
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

    /// Whether this is the fine-grained FP8 scheme in the one element
    /// format this build decodes.
    ///
    /// Both halves are required. `quant_method: "fp8"` alone does not say
    /// which codec, and an `e5m2` checkpoint would decode to plausible
    /// wrong numbers rather than fail.
    pub fn is_finegrained_fp8_e4m3(&self) -> bool {
        self.method == QUANT_METHOD_FP8
            && self
                .fmt
                .as_deref()
                .is_some_and(|f| f.eq_ignore_ascii_case(FMT_E4M3))
    }

    /// The declared tile as a pair, when it is well formed.
    pub fn declared_tile(&self) -> Option<(usize, usize)> {
        match self.weight_block_size.as_deref() {
            Some([r, c]) => Some((*r, *c)),
            _ => None,
        }
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
    let mut string_key = |key: &str| -> Option<String> {
        let v = block.get(key)?.as_str()?.to_string();
        consumed_paths.insert(format!("{QUANTIZATION_CONFIG_KEY}.{key}"));
        Some(v)
    };
    let fmt = string_key(FMT_KEY);
    let activation_scheme = string_key(ACTIVATION_SCHEME_KEY);
    let weight_block_size = match block.get(WEIGHT_BLOCK_SIZE_KEY) {
        Some(Value::Array(items)) => {
            consumed_paths.insert(format!("{QUANTIZATION_CONFIG_KEY}.{WEIGHT_BLOCK_SIZE_KEY}"));
            Some(
                items
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect(),
            )
        }
        _ => None,
    };
    Some(RepresentationReading {
        representation: StoredRepresentation {
            method,
            excluded_modules,
            fmt,
            weight_block_size,
            activation_scheme,
        },
        consumed_paths,
    })
}

#[cfg(test)]
mod tests;
