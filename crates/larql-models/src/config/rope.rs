//! The `rope_scaling` config block and the per-family parameter sets it
//! resolves into.
//!
//! Selector strings live in [`super::rope_types`]; the math lives in
//! `larql-compute::attention::rope`.

/// RoPE scaling configuration (YaRN, linear, dynamic, llama3, default).
///
/// `scaling_type` matches HF's `rope_type` field:
/// - `"default"` / `""`: no scaling, standard RoPE
/// - `"linear"`: divide position by `factor` (Gemma 3 uses this on global
///   layers only via the structured per-layer-type form)
/// - `"llama3"`: wavelength-dependent per-channel scaling — needs all four
///   `llama3_*` fields populated
/// - `"yarn"`: ramped interpolation/extrapolation blend of `inv_freq` **plus**
///   an amplitude factor on cos/sin — see [`YarnRopeScaling`]
/// - `"dynamic"`: not yet honoured in the forward path
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f64,
    /// `llama3` rope scaling: `low_freq_factor`. Unused for other types.
    pub llama3_low_freq_factor: Option<f64>,
    /// `llama3` rope scaling: `high_freq_factor`. Unused for other types.
    pub llama3_high_freq_factor: Option<f64>,
    /// `llama3` rope scaling: `original_max_position_embeddings`. Unused for other types.
    pub llama3_original_max_position_embeddings: Option<f64>,
    /// `yarn` rope scaling: `beta_fast`. `None` means the paper's default
    /// ([`crate::defaults::YARN_DEFAULT_BETA_FAST`]), which is what HF uses.
    pub yarn_beta_fast: Option<f64>,
    /// `yarn` rope scaling: `beta_slow`. `None` means the paper's default
    /// ([`crate::defaults::YARN_DEFAULT_BETA_SLOW`]).
    pub yarn_beta_slow: Option<f64>,
    /// `yarn` rope scaling: `truncate`. `None` means HF's default of `true`.
    /// GPT-OSS ships `false`, which keeps the correction range fractional and
    /// so changes where the ramp starts and ends.
    pub yarn_truncate: Option<bool>,
    /// `yarn` rope scaling: `mscale` (DeepSeek). Only meaningful together
    /// with [`Self::yarn_mscale_all_dim`].
    pub yarn_mscale: Option<f64>,
    /// `yarn` rope scaling: `mscale_all_dim` (DeepSeek). When this and
    /// `mscale` are both present the attention amplitude is their *ratio*,
    /// which is a different number from the single-argument form.
    pub yarn_mscale_all_dim: Option<f64>,
    /// True when the parsed `rope_scaling` block was the structured Gemma 3
    /// per-layer-type form (`{full_attention: {...}, sliding_attention: {...}}`)
    /// and `factor`/`scaling_type` reflect the *global / full-attention* slot
    /// only. The sliding-attention slot for that form is always
    /// `{rope_type: default}` with the model's `rope_local_base` — no extra
    /// scaling to remember.
    pub gemma3_global_only: bool,
}

impl RopeScaling {
    /// Re-emit this block in the `config.json` shape the detector parses
    /// (`crate::detect::parser`), so a consumer that can only persist JSON
    /// can round-trip a scaling family without re-implementing the parse.
    ///
    /// ## Why this exists
    ///
    /// The vindex format had no `rope_scaling` field at all, so
    /// `VindexModelConfig::from_arch` silently dropped it and every
    /// vindex-served model came back with `rope_scaling: None`. On
    /// `gemma-3-4b-it` — whose HF config is
    /// `{"factor": 8.0, "rope_type": "linear"}` — that made
    /// `rope_position_divisor_for_layer` return 1.0 on the five global
    /// layers instead of 8.0, so the served model rotated global-layer
    /// positions eight times faster than the checkpoint specifies. CPU
    /// and Metal lost it identically, which is exactly why the
    /// CPU-vs-Metal parity suite could not see it: both arms shared the
    /// defective config.
    ///
    /// `parse(to_config_json(x)) == x` is pinned per family by
    /// `rope_scaling_round_trips_*` in this module's tests. Keep the two
    /// in step — this is an inverse, and an inverse that drifts from its
    /// forward is worse than no inverse.
    pub fn to_config_json(&self) -> serde_json::Value {
        use serde_json::{Map, Value};

        // Gemma 3's structured per-layer-type form. The parser lifts the
        // `full_attention` slot and marks `gemma3_global_only`; emit the
        // same shape back so the flag survives, since it is what makes
        // the factor apply to global layers only.
        if self.gemma3_global_only {
            return serde_json::json!({
                "full_attention": {
                    "rope_type": self.scaling_type,
                    "factor": self.factor,
                },
                "sliding_attention": { "rope_type": crate::ROPE_TYPE_DEFAULT },
            });
        }

        let mut obj = Map::new();
        obj.insert("rope_type".into(), Value::String(self.scaling_type.clone()));
        obj.insert("factor".into(), self.factor.into());
        let mut put_f64 = |k: &str, v: Option<f64>| {
            if let Some(v) = v {
                obj.insert(k.into(), v.into());
            }
        };
        put_f64("low_freq_factor", self.llama3_low_freq_factor);
        put_f64("high_freq_factor", self.llama3_high_freq_factor);
        put_f64(
            "original_max_position_embeddings",
            self.llama3_original_max_position_embeddings,
        );
        put_f64("beta_fast", self.yarn_beta_fast);
        put_f64("beta_slow", self.yarn_beta_slow);
        put_f64("mscale", self.yarn_mscale);
        put_f64("mscale_all_dim", self.yarn_mscale_all_dim);
        if let Some(t) = self.yarn_truncate {
            obj.insert("truncate".into(), Value::Bool(t));
        }
        Value::Object(obj)
    }
}

/// `llama3` rope scaling parameters. Lives in larql-models so both the
/// config parser and the forward-pass crate share the same type without
/// crossing dependencies.
///
/// Applied as a wavelength-dependent per-channel adjustment to `inv_freq`
/// — see HF's `_compute_llama3_parameters` in `modeling_rope_utils.py`.
/// The actual math lives in `larql-inference::attention::rope`.
#[derive(Debug, Clone, Copy)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: f64,
}

/// `yarn` rope scaling parameters, mirroring HF's `_compute_yarn_parameters`.
///
/// YaRN differs from every other scaling this crate models in that it changes
/// **two** things, not one:
///
/// 1. `inv_freq` becomes a per-dimension blend of the unscaled
///    (*extrapolation*) and `÷ factor` (*interpolation*) frequencies, ramped
///    linearly across the dimensions between the `beta_fast` and `beta_slow`
///    correction bounds.
/// 2. `cos` and `sin` are both multiplied by an **amplitude** —
///    `0.1·ln(factor) + 1` — which is not a rotation at all. It rescales the
///    Q and K vectors, so it changes attention logits at *every* position,
///    including position 0.
///
/// Point 2 is the one that makes YaRN impossible to approximate by ignoring
/// it: a model served without the amplitude is not "slightly off at long
/// range", it is running attention at the wrong temperature everywhere. For
/// `openai/gpt-oss-20b` (`factor: 32`) the amplitude is **1.3466**, so
/// unscaled cos/sin are 34.7 % too small and `q·k` is off by 1.81×.
///
/// Serialisable and comparable because [`PositionPolicy::Yarn`] carries it
/// into the VINDEX3 container: the block is a judged execution fact, not a
/// parse-time convenience.
///
/// [`PositionPolicy::Yarn`]: super::position::PositionPolicy::Yarn
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YarnRopeScaling {
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: f64,
    /// Round the correction range outward to whole dimensions. HF defaults to
    /// `true`; GPT-OSS ships `false`.
    pub truncate: bool,
    /// DeepSeek's `mscale`, paired with [`Self::mscale_all_dim`]. When both
    /// are present the amplitude is `get_mscale(factor, mscale) /
    /// get_mscale(factor, mscale_all_dim)` rather than `get_mscale(factor)`.
    pub mscale: Option<f64>,
    /// DeepSeek's `mscale_all_dim`. See [`Self::mscale`].
    pub mscale_all_dim: Option<f64>,
}

impl YarnRopeScaling {
    /// HF's `get_mscale`: the amplitude a scale factor implies.
    ///
    /// Returns 1.0 for `scale <= 1`, matching HF exactly — a factor of 1 is
    /// "no extension", and `0.1·ln(1) + 1` would coincidentally also be 1.0
    /// but the guard is what makes `factor < 1` safe.
    fn get_mscale(scale: f64, mscale: f64) -> f64 {
        if scale <= 1.0 {
            return 1.0;
        }
        0.1 * mscale * scale.ln() + 1.0
    }

    /// The attention amplitude this block applies to `cos`/`sin` — the
    /// part of YaRN that moves every logit, at every position.
    ///
    /// Two forms, and picking the wrong one is a silent 35 % error:
    /// * both `mscale` and `mscale_all_dim` present (DeepSeek) → their
    ///   **ratio**, which for the usual `mscale == mscale_all_dim`
    ///   collapses to exactly 1.0;
    /// * otherwise (GPT-OSS) → the single-argument `get_mscale(factor)`.
    ///
    /// This is the single authority; `larql-compute`'s rope module
    /// delegates here, and the VINDEX3 carriage of `rope_scaling` is
    /// judged against it.
    pub fn attention_amplitude(&self) -> f64 {
        match (self.mscale, self.mscale_all_dim) {
            (Some(m), Some(m_all)) => {
                Self::get_mscale(self.factor, m) / Self::get_mscale(self.factor, m_all)
            }
            _ => Self::get_mscale(self.factor, 1.0),
        }
    }
}

#[cfg(test)]
mod to_config_json_tests {
    //! `to_config_json` is an inverse of `detect::parser`'s `rope_scaling`
    //! block. Inverses rot: these pin `parse(emit(x)) == x` for every
    //! family so the pair cannot drift apart silently.

    use super::*;

    /// Round-trip through the real detector, which is the only consumer
    /// that matters — `build_arch_json` feeds `detect_from_json`.
    fn reparse(rs: &RopeScaling) -> RopeScaling {
        let cfg = crate::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "rope_scaling": rs.to_config_json(),
        }));
        cfg.config()
            .rope_scaling
            .clone()
            .expect("emitted block must re-parse")
    }

    fn assert_same(a: &RopeScaling, b: &RopeScaling) {
        assert_eq!(a.scaling_type, b.scaling_type, "scaling_type");
        assert_eq!(a.factor, b.factor, "factor");
        assert_eq!(a.llama3_low_freq_factor, b.llama3_low_freq_factor);
        assert_eq!(a.llama3_high_freq_factor, b.llama3_high_freq_factor);
        assert_eq!(
            a.llama3_original_max_position_embeddings,
            b.llama3_original_max_position_embeddings
        );
        assert_eq!(a.yarn_beta_fast, b.yarn_beta_fast);
        assert_eq!(a.yarn_beta_slow, b.yarn_beta_slow);
        assert_eq!(a.yarn_truncate, b.yarn_truncate);
        assert_eq!(a.yarn_mscale, b.yarn_mscale);
        assert_eq!(a.yarn_mscale_all_dim, b.yarn_mscale_all_dim);
        assert_eq!(a.gemma3_global_only, b.gemma3_global_only);
    }

    fn bare(scaling_type: &str, factor: f64) -> RopeScaling {
        RopeScaling {
            scaling_type: scaling_type.to_string(),
            factor,
            llama3_low_freq_factor: None,
            llama3_high_freq_factor: None,
            llama3_original_max_position_embeddings: None,
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            yarn_truncate: None,
            yarn_mscale: None,
            yarn_mscale_all_dim: None,
            gemma3_global_only: false,
        }
    }

    #[test]
    fn rope_scaling_round_trips_linear() {
        let rs = bare(crate::ROPE_TYPE_LINEAR, 4.0);
        assert_same(&rs, &reparse(&rs));
    }

    #[test]
    fn rope_scaling_round_trips_llama3() {
        let mut rs = bare(crate::ROPE_TYPE_LLAMA3, 8.0);
        rs.llama3_low_freq_factor = Some(1.0);
        rs.llama3_high_freq_factor = Some(4.0);
        rs.llama3_original_max_position_embeddings = Some(8192.0);
        assert_same(&rs, &reparse(&rs));
    }

    /// GPT-OSS's shape: YaRN with an explicit `truncate: false`, which is
    /// not HF's default and changes where the correction ramp starts.
    #[test]
    fn rope_scaling_round_trips_yarn_with_truncate_false() {
        let mut rs = bare(crate::ROPE_TYPE_YARN, 32.0);
        rs.yarn_beta_fast = Some(32.0);
        rs.yarn_beta_slow = Some(1.0);
        rs.yarn_truncate = Some(false);
        assert_same(&rs, &reparse(&rs));
    }

    /// DeepSeek's two amplitude knobs must survive together — when both
    /// are present HF computes a ratio, not the single-argument value.
    #[test]
    fn rope_scaling_round_trips_yarn_with_both_mscales() {
        let mut rs = bare(crate::ROPE_TYPE_YARN, 40.0);
        rs.yarn_mscale = Some(1.0);
        rs.yarn_mscale_all_dim = Some(1.0);
        assert_same(&rs, &reparse(&rs));
    }

    /// Gemma 3's structured per-layer-type form. `gemma3_global_only` is
    /// the field that makes the factor apply to global layers only, so
    /// losing it in the round-trip would silently move the scaling onto
    /// every layer.
    #[test]
    fn rope_scaling_round_trips_gemma3_structured_form() {
        let mut rs = bare(crate::ROPE_TYPE_LINEAR, 8.0);
        rs.gemma3_global_only = true;
        let back = reparse(&rs);
        assert_same(&rs, &back);
        assert!(back.gemma3_global_only, "global-only flag must survive");
    }

    /// The emitted JSON must be the shape HF writes, not an internal
    /// spelling — a human reading `index.json` should see the same block
    /// as the checkpoint's `config.json`.
    #[test]
    fn emitted_json_uses_hf_key_names() {
        let mut rs = bare(crate::ROPE_TYPE_LLAMA3, 8.0);
        rs.llama3_low_freq_factor = Some(1.0);
        let v = rs.to_config_json();
        assert_eq!(v["rope_type"], serde_json::json!("llama3"));
        assert_eq!(v["factor"], serde_json::json!(8.0));
        assert_eq!(v["low_freq_factor"], serde_json::json!(1.0));
        // Absent optionals must not be emitted as nulls — the parser
        // reads them with `as_f64()`, which would then see `None`
        // anyway, but a null in the persisted config misrepresents the
        // checkpoint as having declared the field.
        assert!(v.get("beta_fast").is_none());
        assert!(v.get("truncate").is_none());
    }
}
