//! Capture artifact contract for MAD-V3-KNN-1/3.
//!
//! Two dense little-endian f32 planes keep a 50k-query corpus mmapable:
//!
//! * residuals: `[sample][residual_layer][hidden]`
//! * contributions: `[sample][object]`
//!
//! The object table supplies the layer and byte cost for every contribution
//! column. A contribution is the squared L2 norm of one raw down-projection
//! block before any post-FFN norm. That boundary is additive and does not
//! change execution; post-norm attribution would require a separate causal
//! definition because RMS normalisation is nonlinear.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

pub const SCHEMA_V1: &str = "larql.mad-v3-knn.capture.v1";
pub const CONTRIBUTION_SEMANTICS_V1: &str = "l2_norm_squared_raw_down_block";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaptureManifest {
    pub schema: String,
    pub model: String,
    pub hidden_size: usize,
    pub residual_layers: Vec<usize>,
    pub contribution_semantics: String,
    /// Numerical realisation of the observer. Older captures predate this
    /// provenance field and deserialize as `None`.
    #[serde(default)]
    pub contribution_engine: Option<String>,
    pub residuals_file: PathBuf,
    pub contributions_file: PathBuf,
    /// Optional ADDR-1 hidden-width seam plane. Values are ordered
    /// `[sample][seam_layer][seam_kind][hidden]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seams_file: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seam_layers: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seam_kinds: Vec<String>,
    pub samples: Vec<SampleDescriptor>,
    pub objects: Vec<ObjectDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SampleDescriptor {
    pub id: String,
    pub semantic_id: String,
    /// Query-operation class (for example `lookup:facility`) independent of
    /// graph/entity identity. Optional for backward-compatible hand fixtures.
    #[serde(default)]
    pub operation_id: Option<String>,
    pub wording_id: String,
    /// `history` rows populate the index; `query` rows are scored.
    pub split: String,
    /// Optional graph/context identity for a held-out-graph exclusion.
    #[serde(default)]
    pub group_id: Option<String>,
    /// Named evaluation arm, e.g. same-graph/unseen-wording.
    #[serde(default)]
    pub regime: Option<String>,
    /// Lexical relation form used by this fixture (for nuisance audits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_id: Option<String>,
    /// Record-order/layout family used by this fixture (for nuisance audits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_id: Option<String>,
    /// Prompt-token position of the captured key. This lets neighbour audits
    /// detect length/position shortcuts without retaining raw prompt text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,
    /// Tokenizer-authoritative continuation expected after the captured
    /// prompt-final residual. Optional for pre-ROUTE-1 captures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_token_ids: Option<Vec<u32>>,
    /// Greedy continuation generated for exactly `expected_token_ids.len()`
    /// positions. This measures behaviour without retaining raw answer text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_token_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_exact: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_token_accuracy: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectDescriptor {
    /// Stable logical address, e.g. `layer.32.ffn.down.channels.1024-1152`.
    pub id: String,
    pub layer: usize,
    pub kind: String,
    /// Bytes whose residency this object represents. The evaluator never
    /// substitutes object count for byte cost.
    pub byte_count: u64,
    #[serde(default)]
    pub operand: Option<String>,
    #[serde(default)]
    pub channel_start: Option<usize>,
    #[serde(default)]
    pub channel_end: Option<usize>,
    /// Nominal contiguous-channel page schema. The final page may contain
    /// fewer channels; this field still identifies the schema it belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_channels: Option<usize>,
}

pub struct Capture {
    pub manifest: CaptureManifest,
    residuals: Mmap,
    contributions: Mmap,
    residual_layer_index: HashMap<usize, usize>,
    objects_by_layer: HashMap<usize, Vec<usize>>,
}

impl Capture {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join("manifest.json");
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
        let manifest: CaptureManifest = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;
        validate_manifest(&manifest)?;

        let residuals_path = resolve_member(dir, &manifest.residuals_file)?;
        let contributions_path = resolve_member(dir, &manifest.contributions_file)?;
        let residuals_file = File::open(&residuals_path)
            .map_err(|e| format!("open {}: {e}", residuals_path.display()))?;
        let contributions_file = File::open(&contributions_path)
            .map_err(|e| format!("open {}: {e}", contributions_path.display()))?;

        // SAFETY: both mappings are read-only and the Capture owns them for
        // its whole lifetime. Writers must publish a completed capture before
        // evaluation; mutating a mapped artifact concurrently is unsupported.
        let residuals = unsafe { Mmap::map(&residuals_file) }
            .map_err(|e| format!("mmap {}: {e}", residuals_path.display()))?;
        // SAFETY: same read-only completed-artifact contract as above.
        let contributions = unsafe { Mmap::map(&contributions_file) }
            .map_err(|e| format!("mmap {}: {e}", contributions_path.display()))?;

        let residual_values = checked_product(&[
            manifest.samples.len(),
            manifest.residual_layers.len(),
            manifest.hidden_size,
        ])?;
        let contribution_values =
            checked_product(&[manifest.samples.len(), manifest.objects.len()])?;
        validate_plane_len(&residuals_path, residuals.len(), residual_values)?;
        validate_plane_len(
            &contributions_path,
            contributions.len(),
            contribution_values,
        )?;
        if let Some(member) = &manifest.seams_file {
            let seams_path = resolve_member(dir, member)?;
            let seams_len = std::fs::metadata(&seams_path)
                .map_err(|e| format!("stat {}: {e}", seams_path.display()))?
                .len() as usize;
            let seam_values = checked_product(&[
                manifest.samples.len(),
                manifest.seam_layers.len(),
                manifest.seam_kinds.len(),
                manifest.hidden_size,
            ])?;
            validate_plane_len(&seams_path, seams_len, seam_values)?;
        }

        let residual_layer_index = manifest
            .residual_layers
            .iter()
            .enumerate()
            .map(|(i, &layer)| (layer, i))
            .collect();
        let mut objects_by_layer: HashMap<usize, Vec<usize>> = HashMap::new();
        for (index, object) in manifest.objects.iter().enumerate() {
            objects_by_layer
                .entry(object.layer)
                .or_default()
                .push(index);
        }

        let capture = Self {
            manifest,
            residuals,
            contributions,
            residual_layer_index,
            objects_by_layer,
        };
        capture.validate_values()?;
        Ok(capture)
    }

    pub fn residual(&self, sample: usize, layer: usize) -> Result<Vec<f32>, String> {
        let layer_index = *self
            .residual_layer_index
            .get(&layer)
            .ok_or_else(|| format!("capture has no residual plane for layer {layer}"))?;
        let base = (sample * self.manifest.residual_layers.len() + layer_index)
            * self.manifest.hidden_size;
        Ok((0..self.manifest.hidden_size)
            .map(|i| read_f32(&self.residuals, base + i))
            .collect())
    }

    pub fn contribution(&self, sample: usize, object: usize) -> f32 {
        read_f32(
            &self.contributions,
            sample * self.manifest.objects.len() + object,
        )
    }

    pub fn objects_at(&self, layer: usize) -> Option<&[usize]> {
        self.objects_by_layer.get(&layer).map(Vec::as_slice)
    }

    fn validate_values(&self) -> Result<(), String> {
        for sample in 0..self.manifest.samples.len() {
            for &layer in &self.manifest.residual_layers {
                let row = self.residual(sample, layer)?;
                if row.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "sample `{}` layer {layer}: residual contains a non-finite value",
                        self.manifest.samples[sample].id
                    ));
                }
                let norm2: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                if norm2 <= f64::EPSILON {
                    return Err(format!(
                        "sample `{}` layer {layer}: residual has zero norm",
                        self.manifest.samples[sample].id
                    ));
                }
            }
        }
        for sample in 0..self.manifest.samples.len() {
            for object in 0..self.manifest.objects.len() {
                let value = self.contribution(sample, object);
                if !value.is_finite() || value < 0.0 {
                    return Err(format!(
                        "sample `{}` object `{}`: contribution must be finite and non-negative, got {value}",
                        self.manifest.samples[sample].id, self.manifest.objects[object].id
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_manifest(manifest: &CaptureManifest) -> Result<(), String> {
    if manifest.schema != SCHEMA_V1 {
        return Err(format!(
            "unsupported schema `{}`; expected `{SCHEMA_V1}`",
            manifest.schema
        ));
    }
    if manifest.contribution_semantics != CONTRIBUTION_SEMANTICS_V1 {
        return Err(format!(
            "unsupported contribution semantics `{}`; expected `{CONTRIBUTION_SEMANTICS_V1}`",
            manifest.contribution_semantics
        ));
    }
    if manifest.model.trim().is_empty() {
        return Err("model must not be empty".into());
    }
    if manifest.hidden_size == 0 {
        return Err("hidden_size must be positive".into());
    }
    if manifest.residual_layers.is_empty() {
        return Err("residual_layers must not be empty".into());
    }
    if manifest.samples.is_empty() {
        return Err("samples must not be empty".into());
    }
    if manifest.objects.is_empty() {
        return Err("objects must not be empty".into());
    }
    if manifest.seams_file.is_some()
        != (!manifest.seam_layers.is_empty() && !manifest.seam_kinds.is_empty())
    {
        return Err("seams_file, seam_layers and seam_kinds must be present together".into());
    }

    let mut layers = HashSet::new();
    for &layer in &manifest.residual_layers {
        if !layers.insert(layer) {
            return Err(format!("duplicate residual layer {layer}"));
        }
    }
    let mut seam_layers = HashSet::new();
    for &layer in &manifest.seam_layers {
        if !seam_layers.insert(layer) {
            return Err(format!("duplicate seam layer {layer}"));
        }
    }
    let mut seam_kinds = HashSet::new();
    for kind in &manifest.seam_kinds {
        if kind.trim().is_empty() || !seam_kinds.insert(kind) {
            return Err(format!("invalid or duplicate seam kind `{kind}`"));
        }
    }

    let mut sample_ids = HashSet::new();
    let mut history = 0usize;
    let mut query = 0usize;
    for sample in &manifest.samples {
        if sample.id.trim().is_empty()
            || sample.semantic_id.trim().is_empty()
            || sample.wording_id.trim().is_empty()
        {
            return Err("sample id, semantic_id and wording_id must not be empty".into());
        }
        if sample
            .operation_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(format!("sample `{}` has an empty operation_id", sample.id));
        }
        if sample
            .regime
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(format!("sample `{}` has an empty regime", sample.id));
        }
        if sample
            .alias_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(format!("sample `{}` has an empty alias_id", sample.id));
        }
        if sample
            .layout_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(format!("sample `{}` has an empty layout_id", sample.id));
        }
        match (
            sample.expected_token_ids.as_ref(),
            sample.generated_token_ids.as_ref(),
            sample.answer_exact,
            sample.answer_token_accuracy,
        ) {
            (None, None, None, None) => {}
            (Some(expected), Some(generated), Some(exact), Some(accuracy)) => {
                if expected.is_empty() || expected.len() != generated.len() {
                    return Err(format!(
                        "sample `{}` has inconsistent answer token lengths",
                        sample.id
                    ));
                }
                if !(0.0..=1.0).contains(&accuracy) || !accuracy.is_finite() {
                    return Err(format!(
                        "sample `{}` has invalid answer_token_accuracy {accuracy}",
                        sample.id
                    ));
                }
                if exact != (expected == generated) {
                    return Err(format!(
                        "sample `{}` has answer_exact inconsistent with its tokens",
                        sample.id
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "sample `{}` must provide all or none of the answer-scoring fields",
                    sample.id
                ))
            }
        }
        if !sample_ids.insert(&sample.id) {
            return Err(format!("duplicate sample id `{}`", sample.id));
        }
        match sample.split.as_str() {
            "history" => history += 1,
            "query" => query += 1,
            other => {
                return Err(format!(
                    "sample `{}` has split `{other}`; expected `history` or `query`",
                    sample.id
                ))
            }
        }
    }
    if history == 0 || query == 0 {
        return Err(format!(
            "capture needs both history and query rows; found history={history}, query={query}"
        ));
    }

    let mut object_ids = HashSet::new();
    for object in &manifest.objects {
        if object.id.trim().is_empty() || object.kind.trim().is_empty() {
            return Err("object id and kind must not be empty".into());
        }
        if !object_ids.insert(&object.id) {
            return Err(format!("duplicate object id `{}`", object.id));
        }
        if object.byte_count == 0 {
            return Err(format!("object `{}` has zero byte_count", object.id));
        }
        match (object.channel_start, object.channel_end) {
            (Some(start), Some(end)) if start >= end => {
                return Err(format!(
                    "object `{}` has invalid channel range {start}..{end}",
                    object.id
                ))
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(format!(
                    "object `{}` must provide both channel_start and channel_end",
                    object.id
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

fn resolve_member(dir: &Path, member: &Path) -> Result<PathBuf, String> {
    if member.is_absolute()
        || member.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "capture member `{}` must be a relative path inside the capture directory",
            member.display()
        ));
    }
    Ok(dir.join(member))
}

fn checked_product(factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |acc, &factor| {
        acc.checked_mul(factor)
            .ok_or_else(|| "capture dimensions overflow address space".to_string())
    })
}

fn validate_plane_len(path: &Path, actual: usize, values: usize) -> Result<(), String> {
    let expected = values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "capture byte length overflows address space".to_string())?;
    if actual != expected {
        return Err(format!(
            "{} has {actual} bytes; manifest requires exactly {expected}",
            path.display()
        ));
    }
    Ok(())
}

fn read_f32(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated plane"),
    )
}
