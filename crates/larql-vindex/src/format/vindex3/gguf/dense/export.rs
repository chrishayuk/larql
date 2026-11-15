//! The dense export pipeline, as one callable boundary.
//!
//! ```text
//! family      the container's declared family → a registered target
//! components  the primary text component; any other needs text-only
//! op plan     roles per operand, and the declared norm offsets
//! programme   the graph's model against the target's built-in one
//! selection   each object's canonical representation, BF16 or F32
//! tensors     roles → names, geometry, encodings, reference order
//! metadata    general.*, the arch keys, tokenizer.*
//! emit        streamed through a partial file, then renamed
//! verify      the finished file re-read through the independent reader
//! ```

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use larql_models::loading::gguf::GgufValue;

use super::general::{naming_entries, sampling_entries};
use super::metadata::{arch_entries, ArchFacts, FILE_TYPE_ALL_F32, FILE_TYPE_MOSTLY_BF16};
use super::programme::read_programme;
use super::roles::assign_roles;
use super::spm_vocab::ScoreSource;
use super::target::{supported_families, target_for_family};
use super::tensors::{plan_tensors, ShapeFacts, SourceTensor};
use super::tokenizer::{read_json, tokenizer_table};
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::gguf::emit::{emit_gguf, verify_emitted, VerifyReport, TYPE_BF16};
use crate::format::vindex3::graph::object::Fidelity;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::VindexError;

const GENERAL_ARCHITECTURE: &str = "general.architecture";
const GENERAL_TYPE: &str = "general.type";
const GENERAL_TYPE_MODEL: &str = "model";
const GENERATION_CONFIG_JSON: &str = "generation_config.json";
const PARTIAL_SUFFIX: &str = "partial";
/// Canonical encodings the exporter writes without re-encoding.
const EXPORTABLE_ENCODINGS: &[&str] = &["BF16", "F32"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DenseExportOptions {
    /// Export the primary text component of a container that also holds
    /// other components (a vision tower), leaving them out.
    pub text_only: bool,
    /// A `tokenizer.model` to read piece scores from when the container
    /// does not carry one.
    pub sentencepiece_model: Option<PathBuf>,
    /// Accept token scores derived from vocabulary order when no
    /// `tokenizer.model` is available. The file is then not exact and is
    /// marked so in its metadata; without this the export refuses.
    pub derived_tokenizer_scores: bool,
}

/// Where the context length came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    Graph,
    /// The container predates `execution.context_length`; the target's
    /// family default was stated instead.
    TargetDefault,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseExport {
    pub out: PathBuf,
    pub bytes: u64,
    pub family: String,
    pub architecture: &'static str,
    pub tensors: usize,
    pub bf16_tensors: usize,
    pub metadata_keys: usize,
    pub excluded_components: Vec<String>,
    pub context_length: u64,
    pub context_source: ContextSource,
    pub score_source: ScoreSource,
    pub sentencepiece_model: Option<PathBuf>,
    pub vocab_tokens: usize,
    pub vocab_padded: usize,
    pub verify: VerifyReport,
}

impl DenseExport {
    /// Whether every byte follows from the container and its declared
    /// authorities — false when token scores were derived.
    pub fn exact(&self) -> bool {
        self.score_source == ScoreSource::SentencePieceModel
    }
}

fn refuse(msg: impl Into<String>) -> VindexError {
    VindexError::Parse(format!("vindex-to-gguf: {}", msg.into()))
}

fn refuse_all(what: &str, refusals: Vec<String>) -> VindexError {
    refuse(format!("{what}:\n  - {}", refusals.join("\n  - ")))
}

/// Read each included object's canonical representation's tensors.
fn select_sources(
    root: &Path,
    inspection: &crate::format::vindex3::inspect::SystemInspection,
    component: &str,
) -> Result<Vec<SourceTensor>, VindexError> {
    let mut sources = Vec::new();
    for object in inspection
        .graph
        .objects
        .iter()
        .filter(|o| o.component == component)
    {
        let canonical: Vec<&str> = object
            .representations
            .iter()
            .filter(|r| r.fidelity == Fidelity::Canonical)
            .map(|r| r.encoding.as_str())
            .collect();
        let [encoding] = canonical[..] else {
            return Err(refuse(format!(
                "`{}` declares {} canonical representations; exactly one is exported",
                object.id,
                canonical.len()
            )));
        };
        if !EXPORTABLE_ENCODINGS.contains(&encoding) {
            return Err(refuse(format!(
                "`{}`'s canonical representation is {encoding}; the exporter writes BF16 and \
                 F32 canonicals only (quantised GGUF output is not implemented)",
                object.id
            )));
        }
        let id = format!("{}@{encoding}", object.id);
        let entry = inspection
            .index
            .representations
            .get(&id)
            .ok_or_else(|| refuse(format!("the index holds no `{id}`")))?;
        let path = root.join(&entry.segment);
        let (header, data_start) = read_segment_header(&path)?;
        for t in header.tensors {
            sources.push(SourceTensor {
                object: object.id.clone(),
                span: (path.clone(), data_start + t.offset, t.len),
                name: t.name,
                dtype: t.dtype,
                shape: t.shape.iter().map(|d| *d as u64).collect(),
            });
        }
    }
    Ok(sources)
}

/// Export `root`'s primary text component to `out` as a GGUF the target
/// runtime loads, verified through the independent reader. `out` must not
/// exist.
pub fn export_dense(
    root: &Path,
    out: &Path,
    options: &DenseExportOptions,
) -> Result<DenseExport, VindexError> {
    if out.exists() {
        return Err(refuse(format!(
            "{} already exists — refusing to overwrite it",
            out.display()
        )));
    }
    let inspection = inspect_container(root, false)?;
    let family = inspection.index.family.clone();
    let target = target_for_family(&family).ok_or_else(|| {
        refuse(format!(
            "the container declares family `{family}`, which no GGUF target lowers \
             (supported: {})",
            supported_families().join(", ")
        ))
    })?;
    let component = inspection
        .graph
        .primary_text_component()
        .map_err(|e| refuse(format!("{e:?}")))?;
    let excluded: Vec<String> = inspection
        .graph
        .components
        .iter()
        .filter(|c| c.id != component.id)
        .map(|c| c.id.clone())
        .collect();
    if !excluded.is_empty() && !options.text_only {
        return Err(refuse(format!(
            "the container also holds {excluded:?}, which a {} GGUF cannot carry — pass \
             --text-only to export the text model alone",
            target.architecture
        )));
    }
    let surface = component.execution.as_ref().ok_or_else(|| {
        refuse(format!(
            "component `{}` carries no execution surface",
            component.id
        ))
    })?;
    let head = surface
        .head
        .as_ref()
        .ok_or_else(|| refuse("the surface carries no head — vocabulary unknown"))?;

    let outcome = plan_component_ops(&inspection, root, &component.id)?;
    let plan = match (&outcome.plan, outcome.closed()) {
        (Some(p), true) => p,
        _ => {
            return Err(refuse(format!(
                "the operation plan does not close: {:?}",
                outcome.defects
            )))
        }
    };
    let roles = assign_roles(plan).map_err(|e| refuse(e.to_string()))?;
    let programme = read_programme(plan, target, component.hidden_size).map_err(|r| {
        refuse_all(
            &format!("the graph and {} disagree", target.architecture),
            r,
        )
    })?;

    let sources = select_sources(root, &inspection, &component.id)?;
    let facts = ShapeFacts {
        programme: &programme,
        hidden_size: component.hidden_size,
        vocab_size: head.vocab_size,
    };
    let plans = plan_tensors(&sources, &roles, target, &facts)
        .map_err(|r| refuse_all("the tensors cannot be lowered", r))?;

    let (context_length, context_source) = match surface.context_length {
        Some(c) => (c, ContextSource::Graph),
        None => (target.default_context_length, ContextSource::TargetDefault),
    };
    let bf16_tensors = plans.iter().filter(|p| p.target_type == TYPE_BF16).count();
    let total_params: u64 = plans
        .iter()
        .map(|p| p.target_shape.iter().product::<u64>())
        .sum();
    let tokenizer = tokenizer_table(
        root,
        target,
        head.vocab_size,
        options.sentencepiece_model.as_deref(),
        options.derived_tokenizer_scores,
    )?;
    let generation_config = read_json(&root.join(GENERATION_CONFIG_JSON))?;

    let mut metadata: Vec<(String, GgufValue)> = vec![
        (
            GENERAL_ARCHITECTURE.into(),
            GgufValue::String(target.architecture.into()),
        ),
        (
            GENERAL_TYPE.into(),
            GgufValue::String(GENERAL_TYPE_MODEL.into()),
        ),
    ];
    metadata.extend(sampling_entries(generation_config.as_ref()));
    metadata.extend(naming_entries(&component.source_artifact, total_params));
    metadata.extend(arch_entries(&ArchFacts {
        target,
        programme: &programme,
        num_layers: component.num_layers,
        hidden_size: component.hidden_size,
        context_length,
        rms_eps: surface.norm.pre.eps,
        file_type: if bf16_tensors > 0 {
            FILE_TYPE_MOSTLY_BF16
        } else {
            FILE_TYPE_ALL_F32
        },
        final_logit_softcapping: head.final_logit_softcapping,
    }));
    metadata.extend(tokenizer.entries.iter().cloned());

    let spans: BTreeMap<String, (PathBuf, u64, u64)> = sources
        .iter()
        .map(|s| (s.address(), s.span.clone()))
        .collect();
    let mut open = |address: &str| -> std::io::Result<Box<dyn Read>> {
        let (path, at, len) = spans
            .get(address)
            .ok_or_else(|| std::io::Error::other(format!("no payload span for `{address}`")))?;
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(*at))?;
        Ok(Box::new(f.take(*len)))
    };

    let partial = {
        let mut name = out.as_os_str().to_owned();
        name.push(".");
        name.push(PARTIAL_SUFFIX);
        PathBuf::from(name)
    };
    let written = emit_gguf(&metadata, &plans, &mut open, &partial).and_then(|_| {
        verify_emitted(&partial, &metadata, &plans, &[]).map_err(|wrong| {
            refuse(format!(
                "the emitted file does not match its plan — {} mismatches, first: {}",
                wrong.len(),
                wrong.first().cloned().unwrap_or_default()
            ))
        })
    });
    let verify = match written {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            return Err(e);
        }
    };
    std::fs::rename(&partial, out)?;
    Ok(DenseExport {
        out: out.to_path_buf(),
        bytes: std::fs::metadata(out)?.len(),
        family,
        architecture: target.architecture,
        tensors: plans.len(),
        bf16_tensors,
        metadata_keys: metadata.len(),
        excluded_components: excluded,
        context_length,
        context_source,
        score_source: tokenizer.score_source,
        sentencepiece_model: tokenizer.sentencepiece_model,
        vocab_tokens: tokenizer.tokens,
        vocab_padded: tokenizer.padded,
        verify,
    })
}
