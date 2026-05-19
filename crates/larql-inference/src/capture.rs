//! Capture residual stream vectors and sparse activations for entities.
//!
//! High-level API: load a model, tokenize entities, run forward passes,
//! write NDJSON output files compatible with vector-load and vindex builds.

use std::borrow::Cow;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::InferenceError;
use crate::forward::trace_forward;
use crate::model::{
    load_model_dir_validated, load_model_dir_walk_only_validated, resolve_model_path, ModelWeights,
};
use crate::tokenizer::load_tokenizer;

/// Configuration for residual/activation capture.
pub struct CaptureConfig {
    pub layers: Vec<usize>,
    pub prompt_template: Option<String>,
    pub capture_activations: bool,
    pub activation_top_k: usize,
}

pub const DEFAULT_ACTIVATION_TOP_K: usize = 50;
pub const DEFAULT_RESIDUAL_TOP_K: usize = 10;

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            prompt_template: None,
            capture_activations: false,
            activation_top_k: DEFAULT_ACTIVATION_TOP_K,
        }
    }
}

/// Callbacks for capture progress.
pub trait CaptureCallbacks {
    fn on_entity_start(&mut self, _entity: &str, _index: usize, _total: usize) {}
    fn on_entity_done(&mut self, _entity: &str, _layers_captured: usize, _elapsed_ms: f64) {}
}

pub struct SilentCallbacks;
impl CaptureCallbacks for SilentCallbacks {}

/// Loaded model ready for inference and capture.
pub struct InferenceModel {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

// Re-export shared vector types from larql-models.
pub use larql_models::{TopKEntry, VectorFileHeader, VectorRecord};

/// Sparse activation record for NDJSON output.
#[derive(serde::Serialize)]
struct ActivationRecord {
    id: String,
    entity: String,
    layer: usize,
    activations: Vec<FeatureActivation>,
}

#[derive(serde::Serialize)]
struct FeatureActivation {
    feature: usize,
    magnitude: f32,
}

impl InferenceModel {
    /// Load a model from a path or HuggingFace model ID.
    pub fn load(model: &str) -> Result<Self, InferenceError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir_validated(&model_path)?;
        let tokenizer = load_tokenizer(&model_path)?;

        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    /// Load in walk-only mode — never reads FFN tensors from safetensors.
    /// Requires a vindex to serve the FFN path. Peak RSS during load tracks
    /// only the retained (attention / embed / lm_head / norms) weights,
    /// which makes large-model loading (~30B+) feasible on machines that
    /// couldn't hold the full f32-decoded model in memory.
    pub fn load_walk_only(model: &str) -> Result<Self, InferenceError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir_walk_only_validated(&model_path)?;
        let tokenizer = load_tokenizer(&model_path)?;
        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// Mutable accessor — needed by the generate() entry point so the CPU
    /// fallback can dequantise per-layer Q4K tensors into `weights.tensors`.
    /// Metal-only callers can continue to use the shared `weights()`.
    pub fn weights_mut(&mut self) -> &mut ModelWeights {
        &mut self.weights
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Construct an `InferenceModel` directly from in-memory parts.
    /// Bypasses the disk-loading path so tests and embedders that
    /// already hold a `ModelWeights` + `Tokenizer` pair don't need to
    /// materialise a synthetic model directory.
    pub fn from_parts(
        weights: ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            weights,
            tokenizer,
            model_name: model_name.into(),
        }
    }

    /// Capture residuals and optionally activations for a list of entities.
    pub fn capture(
        &self,
        entities: &[String],
        config: &CaptureConfig,
        output_dir: &Path,
        callbacks: &mut dyn CaptureCallbacks,
    ) -> Result<(usize, usize), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        // Residuals writer
        let residual_path = output_dir.join("residuals.vectors.jsonl");
        let res_file = std::fs::File::create(&residual_path)?;
        let mut res_writer = std::io::BufWriter::new(res_file);

        let header = VectorFileHeader {
            _header: true,
            component: "residuals".to_string(),
            model: self.model_name.clone(),
            dimension: self.weights.hidden_size,
            extraction_date: current_date(),
        };
        serde_json::to_writer(&mut res_writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        res_writer.write_all(b"\n")?;

        // Activations writer (optional)
        let act_path = output_dir.join("activations.jsonl");
        let mut act_writer = if config.capture_activations {
            let file = std::fs::File::create(&act_path)?;
            Some(BufWriter::new(file))
        } else {
            None
        };

        let total = entities.len();
        let mut res_count = 0;
        let mut act_count = 0;
        let capture_layers: Cow<'_, [usize]> = if config.layers.is_empty() {
            Cow::Owned(vec![self.weights.num_layers.saturating_sub(1)])
        } else {
            Cow::Borrowed(&config.layers)
        };

        for (i, entity) in entities.iter().enumerate() {
            let start = std::time::Instant::now();
            callbacks.on_entity_start(entity, i, total);

            let prompt = match &config.prompt_template {
                Some(tmpl) => tmpl.replace("{entity}", entity),
                None => entity.clone(),
            };

            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if token_ids.is_empty() {
                continue;
            }

            let trace = trace_forward(
                &self.weights,
                &token_ids,
                &capture_layers,
                config.capture_activations,
                config.activation_top_k,
            );

            // Write residuals
            for (layer, vector) in &trace.residuals {
                let top_k = project_to_vocab(
                    &self.weights.embed,
                    vector,
                    DEFAULT_RESIDUAL_TOP_K,
                    &self.tokenizer,
                );

                let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                    (first.token.clone(), first.token_id, first.logit)
                } else {
                    (String::new(), 0, 0.0)
                };

                let record = VectorRecord {
                    id: format!("{entity}_L{layer}"),
                    layer: *layer,
                    feature: 0,
                    dim: vector.len(),
                    vector: vector.clone(),
                    top_token,
                    top_token_id,
                    c_score,
                    top_k,
                };
                serde_json::to_writer(&mut res_writer, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                res_writer.write_all(b"\n")?;
                res_count += 1;
            }

            // Write activations
            if let Some(ref mut writer) = act_writer {
                for (layer, features) in &trace.activations {
                    let record = ActivationRecord {
                        id: format!("{entity}_L{layer}"),
                        entity: entity.clone(),
                        layer: *layer,
                        activations: features
                            .iter()
                            .map(|&(feat, mag)| FeatureActivation {
                                feature: feat,
                                magnitude: mag,
                            })
                            .collect(),
                    };
                    serde_json::to_writer(&mut *writer, &record)
                        .map_err(|e| InferenceError::Parse(e.to_string()))?;
                    writer.write_all(b"\n")?;
                    act_count += 1;
                }
            }

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            callbacks.on_entity_done(entity, trace.residuals.len(), elapsed_ms);
        }

        res_writer.flush()?;
        if let Some(ref mut writer) = act_writer {
            writer.flush()?;
        }

        Ok((res_count, act_count))
    }
}

/// Project a residual vector onto the embedding matrix to find top-k tokens.
fn project_to_vocab(
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    residual: &[f32],
    k: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TopKEntry> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = k.min(scores.len());
    if k > 0 && k < scores.len() {
        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, logit)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|token| TopKEntry {
                    token,
                    token_id: idx as u32,
                    logit,
                })
        })
        .collect()
}

fn current_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut days = (secs / 86400) as i64;
    let mut year: i64 = 1970;
    loop {
        let n = if is_leap_year(year) { 366 } else { 365 };
        if days < n {
            break;
        }
        days -= n;
        year += 1;
    }
    let month_lens: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for n in month_lens {
        if days < n {
            break;
        }
        days -= n;
        month += 1;
    }
    let day = days + 1;
    format!("{year}-{month:02}-{day:02}")
}

#[inline]
fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert epoch seconds to a YYYY-MM-DD string. Mirrors `current_date` but
    /// takes the timestamp as a parameter so it's deterministic.
    fn date_from_epoch_secs(secs: u64) -> String {
        let mut days = (secs / 86400) as i64;
        let mut year: i64 = 1970;
        loop {
            let n = if is_leap_year(year) { 366 } else { 365 };
            if days < n {
                break;
            }
            days -= n;
            year += 1;
        }
        let month_lens: [i64; 12] = if is_leap_year(year) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 1;
        for n in month_lens {
            if days < n {
                break;
            }
            days -= n;
            month += 1;
        }
        let day = days + 1;
        format!("{year}-{month:02}-{day:02}")
    }

    #[test]
    fn epoch_zero_is_1970_01_01() {
        assert_eq!(date_from_epoch_secs(0), "1970-01-01");
    }

    #[test]
    fn end_of_2024_is_leap_year_aware() {
        // 2024-12-31 23:59:59 UTC = 1735689599 secs since epoch.
        assert_eq!(date_from_epoch_secs(1_735_689_599), "2024-12-31");
        // 2025-01-01 00:00:00 UTC = 1735689600 secs.
        assert_eq!(date_from_epoch_secs(1_735_689_600), "2025-01-01");
    }

    #[test]
    fn march_first_after_feb_29_in_leap_year() {
        // 2024 is leap. 2024-03-01 00:00:00 UTC = 1709251200.
        assert_eq!(date_from_epoch_secs(1_709_251_200), "2024-03-01");
        // 2024-02-29 = 1709164800.
        assert_eq!(date_from_epoch_secs(1_709_164_800), "2024-02-29");
    }

    #[test]
    fn march_first_in_non_leap_year() {
        // 2023-03-01 00:00:00 UTC = 1677628800.
        assert_eq!(date_from_epoch_secs(1_677_628_800), "2023-03-01");
        // 2023 has no Feb 29: 2023-02-28 = 1677542400.
        assert_eq!(date_from_epoch_secs(1_677_542_400), "2023-02-28");
    }

    #[test]
    fn century_boundary_2000_is_leap() {
        // 2000-02-29 = 951782400 (divisible by 400 → leap).
        assert_eq!(date_from_epoch_secs(951_782_400), "2000-02-29");
    }

    #[test]
    fn current_date_returns_well_formed_string() {
        let s = current_date();
        // Format YYYY-MM-DD with zero-padded month/day.
        let parts: Vec<&str> = s.split('-').collect();
        assert_eq!(parts.len(), 3, "expected YYYY-MM-DD, got {s:?}");
        let year: i32 = parts[0].parse().unwrap();
        let month: u32 = parts[1].parse().unwrap();
        let day: u32 = parts[2].parse().unwrap();
        assert!(year >= 2025, "year too small: {year}");
        assert!((1..=12).contains(&month), "bad month: {month}");
        assert!((1..=31).contains(&day), "bad day: {day}");
    }

    #[test]
    fn leap_year_rules() {
        assert!(is_leap_year(2000)); // /400
        assert!(!is_leap_year(1900)); // /100 not /400
        assert!(is_leap_year(2024)); // /4
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(2100)); // /100 not /400
    }

    // ── InferenceModel::capture (full pipeline against the synthetic
    //    in-memory weights, via `InferenceModel::from_parts`) ───────────
    use crate::test_utils::{make_test_weights, synthetic_tokenizer_json};

    fn fresh_inference_model() -> InferenceModel {
        let weights = make_test_weights();
        // On-disk-shape tokenizer so `[N]` encodes to a single in-vocab id
        // (the default `make_test_tokenizer` has a Whitespace pre-tokenizer
        // that splits `[N]` into `[`, `N`, `]` — all UNK).
        let json = synthetic_tokenizer_json(weights.vocab_size);
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("synthetic tokenizer");
        InferenceModel::from_parts(weights, tokenizer, "synthetic-tinymodel")
    }

    #[test]
    fn inference_model_from_parts_and_accessors() {
        let model = fresh_inference_model();
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.hidden_size(), 16);
        assert!(model.weights().vocab_size > 0);
        let _ = model.tokenizer();
    }

    #[test]
    fn inference_model_weights_mut_returns_mutable_borrow() {
        let mut model = fresh_inference_model();
        let original = model.weights().rope_base;
        let w = model.weights_mut();
        w.rope_base = 99999.0;
        assert_eq!(model.weights().rope_base, 99999.0);
        model.weights_mut().rope_base = original;
    }

    /// The in-memory `make_test_tokenizer` path produces a Whitespace
    /// pretokenizer that splits `[N]` into UNK tokens. The on-disk
    /// fixture uses a different (null-pretokenizer) tokenizer JSON —
    /// confirms we're picking up the right one for these tests.
    #[test]
    fn fresh_inference_model_tokenizes_bracket_form_to_single_id() {
        let model = fresh_inference_model();
        let enc = model.tokenizer().encode("[5]", false).expect("encode");
        assert_eq!(enc.get_ids(), &[5u32]);
    }

    /// `capture` happy path — residuals only (no activations). Drives
    /// the entity loop, residual JSONL writing, project_to_vocab, and
    /// the SilentCallbacks no-op trait.
    #[test]
    fn capture_writes_residuals_jsonl_with_silent_callbacks() {
        let model = fresh_inference_model();
        let out_dir = tempfile::tempdir().expect("out tempdir");
        let config = CaptureConfig::default();
        let mut cb = SilentCallbacks;
        // Entities use vocab tokens that the synthetic tokenizer
        // (pre_tokenizer=null in the on-disk variant) decodes to a
        // single id each.
        let entities = vec!["[1]".to_string(), "[2]".to_string()];
        let (res_count, act_count) = model
            .capture(&entities, &config, out_dir.path(), &mut cb)
            .expect("capture must succeed");
        assert!(res_count > 0, "should produce at least one residual record");
        assert_eq!(act_count, 0, "capture_activations=false → no activations");

        let residual_path = out_dir.path().join("residuals.vectors.jsonl");
        let bytes = std::fs::read(&residual_path).expect("residuals.vectors.jsonl");
        let text = String::from_utf8_lossy(&bytes);
        // Header line + at least one record line.
        assert!(
            text.lines().count() > 1,
            "expected header + ≥1 record, got {} line(s)",
            text.lines().count()
        );
        // The activations file is not created when capture_activations
        // is false (the writer is None throughout).
        assert!(!out_dir.path().join("activations.jsonl").exists());
    }

    /// `capture` with activations on — drives the
    /// `if let Some(ref mut writer)` activations branch and the
    /// `FeatureActivation` record path.
    #[test]
    fn capture_writes_activations_jsonl_when_enabled() {
        let model = fresh_inference_model();
        let out_dir = tempfile::tempdir().expect("out tempdir");
        let config = CaptureConfig {
            layers: vec![0, 1],
            prompt_template: Some("entity is {entity}".into()),
            capture_activations: true,
            activation_top_k: 4,
        };
        let mut cb = SilentCallbacks;
        let entities = vec!["[3]".to_string()];
        let (_res, act_count) = model
            .capture(&entities, &config, out_dir.path(), &mut cb)
            .expect("capture must succeed");
        assert!(
            act_count > 0,
            "should produce at least one activation record"
        );

        let act_path = out_dir.path().join("activations.jsonl");
        assert!(act_path.exists(), "activations.jsonl must be written");
    }

    /// `capture` with an empty entity list still creates the residuals
    /// file (just the header line) and returns zero counts.
    #[test]
    fn capture_with_no_entities_writes_header_only() {
        let model = fresh_inference_model();
        let out_dir = tempfile::tempdir().expect("out tempdir");
        let config = CaptureConfig::default();
        let mut cb = SilentCallbacks;
        let (res_count, act_count) = model
            .capture(&[], &config, out_dir.path(), &mut cb)
            .expect("capture must succeed with empty entities");
        assert_eq!(res_count, 0);
        assert_eq!(act_count, 0);
        let residual_path = out_dir.path().join("residuals.vectors.jsonl");
        assert!(residual_path.exists());
    }

    /// `capture` skips entities whose tokenization is empty
    /// (line 185-187 `continue`).
    #[test]
    fn capture_skips_empty_tokenization_entities() {
        let model = fresh_inference_model();
        let out_dir = tempfile::tempdir().expect("out tempdir");
        let config = CaptureConfig::default();
        let mut cb = SilentCallbacks;
        // Empty prompt → tokenizer should produce empty token list,
        // triggering the `continue` branch.
        let entities = vec![String::new()];
        let (res_count, _act_count) = model
            .capture(&entities, &config, out_dir.path(), &mut cb)
            .expect("capture must succeed even when entity skipped");
        // No residuals because every entity was skipped.
        assert_eq!(res_count, 0);
    }

    #[test]
    fn capture_default_config_uses_last_layer_when_empty() {
        // CaptureConfig::default() yields layers=Vec::new(); the capture
        // loop falls through to `[num_layers-1]`. Drives the
        // `if config.layers.is_empty()` branch on line 164.
        let model = fresh_inference_model();
        let out_dir = tempfile::tempdir().expect("out tempdir");
        let cfg = CaptureConfig::default();
        assert!(cfg.layers.is_empty());
        let mut cb = SilentCallbacks;
        let entities = vec!["[1]".to_string()];
        let (res_count, _) = model
            .capture(&entities, &cfg, out_dir.path(), &mut cb)
            .expect("capture");
        // Default config captures last layer only → exactly 1 residual
        // per entity.
        assert_eq!(res_count, 1);
    }
}
