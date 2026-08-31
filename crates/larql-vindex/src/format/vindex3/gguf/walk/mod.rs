//! Walking a whole model's primary-text surface into a plan, and
//! reporting what happened.
//!
//! The question this answers, and the only one worth asking of a real
//! container:
//!
//! > **Can every physical tensor participating in the primary-text
//! > model be traced from artifact semantics to exactly one correct
//! > target representation?**
//!
//! Three coverage invariants, each catching a different silent failure:
//!
//! ```text
//! SOURCE    every primary-text physical tensor consumed exactly once,
//!           or excluded with a stated reason
//! TARGET    every tensor the programme requires produced exactly once
//! IDENTITY  no two plans claiming one GGUF name
//! ```
//!
//! **Physical, not semantic.** A full-attention `q_proj` carries two
//! roles — query and output gate — in one tensor. Counting roles would
//! see two things to emit; counting tensors sees one, which is what the
//! file contains. Conversely NVFP4 *adds* target tensors, because GGML
//! has no per-tensor scale and the f32 leaves as a sibling. So the two
//! counts differ in both directions and neither is wrong:
//!
//! ```text
//! 851 source tensors  ≠  N target tensors
//! ```
//!
//! Roles arrive as input. The walk never infers one from a tensor name —
//! that is the operation plan's job, and re-deriving it here would
//! reintroduce the exact mistake §5:00 of the film exists to explain.

use std::collections::BTreeMap;

use super::plan::{qwen35_global_name, qwen35_tensor_name, LoweredTensorPlan, RepresentationKind};

/// One physical tensor in the source, with the role the plan assigned.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTensor {
    pub object: String,
    pub name: String,
    /// From the operation plan, never from the name.
    pub role: String,
    /// `None` for model-scope surfaces.
    pub layer: Option<usize>,
    pub representation: RepresentationKind,
}

/// Why a source tensor was not lowered.
#[derive(Debug, Clone, PartialEq)]
pub struct Exclusion {
    pub object: String,
    pub count: usize,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WalkError {
    /// A source tensor no role maps to a target. The 160 layer norms
    /// were exactly this before the real inventory found them.
    Unplanned { name: String, role: String },
    /// Two plans claiming one GGUF name — the file would keep whichever
    /// was written last.
    DuplicateTarget {
        target: String,
        sources: Vec<String>,
    },
    /// A tensor the target programme needs that nothing produced.
    /// `output.weight` is the dangerous member: llama.cpp will tie the
    /// embedding instead and run, producing a different model.
    MissingRequired {
        target: String,
        because: &'static str,
    },
}

/// What the walk did, in categories that can fail independently.
#[derive(Debug, Clone, PartialEq)]
pub struct Ledger {
    pub source_by_object: BTreeMap<String, usize>,
    pub source_total: usize,
    pub accounted: usize,
    pub excluded: Vec<Exclusion>,
    pub target_total: usize,
    pub generated_scale_tensors: usize,
    pub errors: Vec<WalkError>,
}

impl Ledger {
    pub fn ready(&self) -> bool {
        self.errors.is_empty() && self.accounted == self.source_total
    }
}

/// Lower a primary-text surface into plans, and account for all of it.
///
/// `required_targets` are the names the programme must produce —
/// derived from the graph, not from what happened to be planned. That
/// direction matters: deriving them from the plans would make the
/// coverage check tautological.
pub fn walk_primary_text(
    sources: &[SourceTensor],
    excluded: Vec<Exclusion>,
    required_targets: &[(&str, &'static str)],
) -> (Vec<LoweredTensorPlan>, Ledger) {
    let mut plans = Vec::new();
    let mut errors = Vec::new();
    let mut by_object: BTreeMap<String, usize> = BTreeMap::new();
    let mut claimed: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scales = 0usize;

    for t in sources {
        *by_object.entry(t.object.clone()).or_insert(0) += 1;

        let target = match t.layer {
            Some(l) => qwen35_tensor_name(&t.role, l),
            None => qwen35_global_name(&t.role).map(str::to_string),
        };
        let Some(target) = target else {
            errors.push(WalkError::Unplanned {
                name: t.name.clone(),
                role: t.role.clone(),
            });
            continue;
        };
        claimed
            .entry(target.clone())
            .or_default()
            .push(t.name.clone());

        let scale_tensor = (t.representation == RepresentationKind::Nvfp4).then(|| {
            scales += 1;
            target.replace(".weight", ".scale")
        });
        let target_type = match t.representation {
            RepresentationKind::F32 => 0,
            RepresentationKind::Bf16 => 30,
            RepresentationKind::Nvfp4 => larql_models::quant::nvfp4_ggml::TYPE_NVFP4,
        };
        // Transforms are attached by the caller from graph facts; the
        // walk's job is coverage, not arithmetic.
        match LoweredTensorPlan::new(
            t.name.clone(),
            target,
            t.representation,
            target_type,
            vec![],
            vec![],
            scale_tensor,
        ) {
            Ok(p) => plans.push(p),
            Err(_) => unreachable!("no transforms are attached here"),
        }
    }

    for (target, sources) in &claimed {
        if sources.len() > 1 {
            errors.push(WalkError::DuplicateTarget {
                target: target.clone(),
                sources: sources.clone(),
            });
        }
    }
    for (target, because) in required_targets {
        if !claimed.contains_key(*target) {
            errors.push(WalkError::MissingRequired {
                target: (*target).to_string(),
                because,
            });
        }
    }

    let source_total = sources.len();
    let accounted = plans.len();
    let ledger = Ledger {
        source_by_object: by_object,
        source_total,
        accounted,
        excluded,
        target_total: plans.len() + scales,
        generated_scale_tensors: scales,
        errors,
    };
    (plans, ledger)
}

#[cfg(test)]
mod tests;
