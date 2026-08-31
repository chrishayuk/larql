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
//!
//! A fourth invariant, checked per tensor as each plan is made:
//!
//! ```text
//! GEOMETRY  every plan's target shape equals what the graph's facts and
//!           the target ABI say that role must be
//! ```
//!
//! The expectation comes from [`ModelGeometry`] — graph facts — and the
//! plan from the physical tensor through its layout transforms. The two
//! never consult each other, which is what makes their agreement mean
//! something.

use std::collections::BTreeMap;

use super::geometry::{check_target, GeometryError};
use super::plan::{
    qwen35_global_name, qwen35_tensor_name, qwen35_transforms, LoweredTensorPlan, PlanError,
    Qwen35Lowering, RepresentationKind,
};

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
    /// The physical shape, from the segment header. Carried so geometry
    /// can be derived from the tensor rather than assumed from the role.
    pub shape: Vec<u64>,
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
    /// The constructor refused the lowering — a squeeze of a real
    /// channel axis, or a value transform on a quantised source.
    Plan { name: String, error: PlanError },
    /// The metadata's model and the planner's model disagree about this
    /// tensor. Each is self-consistent; only comparing them finds it.
    Geometry(GeometryError),
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
    /// Plans whose target shape was compared with the graph-derived
    /// expectation and agreed. Equal to `accounted` when ready.
    pub geometry_reconciled: usize,
    pub errors: Vec<WalkError>,
}

impl Ledger {
    pub fn ready(&self) -> bool {
        self.errors.is_empty()
            && self.accounted == self.source_total
            && self.geometry_reconciled == self.accounted
    }
}

/// Lower a primary-text surface into plans, and account for all of it.
///
/// `required_targets` are the names the programme must produce —
/// derived from the graph, not from what happened to be planned. That
/// direction matters: deriving them from the plans would make the
/// coverage check tautological.
///
/// `lowering` is the graph's account of the geometry and the declared
/// norm offsets, and the same direction applies: it is read from the
/// surface, never from the tensors being walked.
pub fn walk_primary_text(
    sources: &[SourceTensor],
    excluded: Vec<Exclusion>,
    required_targets: &[(&str, &'static str)],
    lowering: &Qwen35Lowering,
) -> (Vec<LoweredTensorPlan>, Ledger) {
    let mut plans = Vec::new();
    let mut errors = Vec::new();
    let mut by_object: BTreeMap<String, usize> = BTreeMap::new();
    let mut claimed: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scales = 0usize;
    let mut reconciled = 0usize;

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
        // The role's complete transform programme, from the one table.
        let (layout, value) = match qwen35_transforms(&t.role, lowering) {
            Ok(t) => t,
            Err(error) => {
                errors.push(WalkError::Plan {
                    name: t.name.clone(),
                    error,
                });
                continue;
            }
        };
        // Target encoding. A quantised source keeps its encoding — that
        // is the whole point of the export. An unquantised source stays
        // put only when it is a plain 2-D projection: anything that is
        // not a matrix after lowering (norms, 1-D parameters, the
        // convolution kernel) or that had arithmetic applied is stored
        // F32, which is both llama.cpp's convention for these tensors
        // and the exact representation of the f32 arithmetic's result.
        let target_type = match t.representation {
            RepresentationKind::F32 => 0,
            RepresentationKind::Nvfp4 => larql_models::quant::nvfp4_ggml::TYPE_NVFP4,
            RepresentationKind::Bf16 => {
                let stays_matrix = t.shape.len() == 2 && t.role != "causal conv over q|k|v";
                if stays_matrix && value.is_empty() {
                    30
                } else {
                    0
                }
            }
        };
        let plan = match LoweredTensorPlan::new(
            t.name.clone(),
            target.clone(),
            t.representation,
            target_type,
            t.shape.clone(),
            layout,
            value,
            scale_tensor,
        ) {
            Ok(p) => p,
            Err(error) => {
                errors.push(WalkError::Plan {
                    name: t.name.clone(),
                    error,
                });
                continue;
            }
        };
        // GEOMETRY: the plan's shape came from the tensor; the
        // expectation comes from the graph. Compare them here, on every
        // tensor, rather than trusting that both are right.
        match check_target(&target, &t.role, &plan.target_shape, &lowering.model) {
            Ok(_) => reconciled += 1,
            Err(e) => errors.push(WalkError::Geometry(e)),
        }
        plans.push(plan);
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
        geometry_reconciled: reconciled,
        errors,
    };
    (plans, ledger)
}

#[cfg(test)]
mod tests;

/// Read a container's primary-text inventory into the walk's input.
///
/// **The entry point, and the reason it exists as one.** A walk tested
/// only against a hand-built inventory proves the planner reasons
/// correctly about a shape someone typed. It does not prove the
/// exporter can reach a real artifact — the same gap as a refusal test
/// that only checks the message renders.
///
/// Roles are supplied by the caller from the operation plan. This
/// function reads bytes and shapes; it does not decide what anything
/// means.
/// Which representation of an object the export should carry.
///
/// **Not optional, and not "whatever is in the index".** `represent` is
/// archival: the compiled pack lands *beside* the canonical bytes, so a
/// represented container holds `target.decoder_stack@BF16` and
/// `@NVFP4` at once. Walking both produced 1,696 sources for an
/// 848-tensor object and 848 duplicate-target errors — the walk faithfully
/// reporting that it had been asked to emit each tensor twice.
///
/// Selecting is the precision programme's job. The walk takes the
/// answer; it does not pick.
pub type SelectRepresentation<'a> = &'a dyn Fn(&str, &[&str]) -> Option<String>;

pub fn inventory_from_container(
    root: &std::path::Path,
    index: &crate::format::vindex3::index::Vindex3Index,
    roles: &dyn Fn(&str, &str) -> Option<(String, Option<usize>)>,
    surface_is_included: &dyn Fn(&str) -> bool,
    select: SelectRepresentation<'_>,
) -> Result<(Vec<SourceTensor>, Vec<Exclusion>), crate::VindexError> {
    // Group the index by object, so the selector sees an object's whole
    // catalogue rather than one entry at a time.
    let mut by_object: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in index.representations.keys() {
        let object = id.split('@').next().unwrap_or(id).to_string();
        by_object.entry(object).or_default().push(id.clone());
    }
    // Three guardrails, because a selector that answers badly must not
    // be able to quietly shrink the model.
    let mut chosen = std::collections::BTreeSet::new();
    for (object, ids) in &by_object {
        if !surface_is_included(object) {
            continue;
        }
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let Some(pick) = select(object, &refs) else {
            // Declining to choose is not the same as having nothing to
            // choose from. Omitting the object here would drop tensors
            // and still report success.
            return Err(crate::VindexError::Parse(format!(
                "export: the precision programme selected no representation for `{object}`,                  which has {}: {refs:?}",
                refs.len()
            )));
        };
        if !refs.contains(&pick.as_str()) {
            return Err(crate::VindexError::Parse(format!(
                "export: the precision programme selected `{pick}` for `{object}`, which is                  not one of its representations: {refs:?}"
            )));
        }
        chosen.insert(pick);
    }
    let included_objects = by_object.keys().filter(|o| surface_is_included(o)).count();
    if chosen.len() != included_objects {
        return Err(crate::VindexError::Parse(format!(
            "export: {} representations selected for {included_objects} included objects —              exactly one each",
            chosen.len()
        )));
    }

    use crate::format::vindex3::encode::segment::read_segment_header;

    let mut sources = Vec::new();
    let mut excluded = Vec::new();

    for (id, entry) in &index.representations {
        if !chosen.contains(id) {
            continue;
        }
        let object = id.split('@').next().unwrap_or(id).to_string();
        let path = root.join(&entry.segment);
        if !path.exists() {
            continue;
        }
        let (header, _) = read_segment_header(&path)?;

        if !surface_is_included(&object) {
            excluded.push(Exclusion {
                object,
                count: header.tensors.len(),
                reason: "surface not requested",
            });
            continue;
        }

        for t in &header.tensors {
            // Representation is a fact about THIS tensor, not about the
            // object it lives in. An NVFP4 pack quantises only the 2-D
            // projections; its norms, convolution and 1-D parameters
            // stay at source precision, and the segment header says so
            // per tensor. Reading the object's encoding here inflated
            // the represented hero's scale-sibling count to 848 — one
            // per decoder tensor — when the pack actually quantises 496.
            let representation = match t.dtype.as_str() {
                "NVFP4" => RepresentationKind::Nvfp4,
                "F32" => RepresentationKind::F32,
                "BF16" => RepresentationKind::Bf16,
                other => {
                    return Err(crate::VindexError::Parse(format!(
                        "export: tensor `{}` in `{}` has dtype `{other}`, which no lowering                          understands — refusing rather than guessing a representation",
                        t.name, entry.segment
                    )))
                }
            };
            let Some((role, layer)) = roles(&object, &t.name) else {
                // An unroled tensor is still counted, so the ledger's
                // accounted total falls short and the walk refuses.
                sources.push(SourceTensor {
                    object: object.clone(),
                    name: t.name.clone(),
                    role: String::new(),
                    layer: None,
                    representation,
                    shape: t.shape.iter().map(|d| *d as u64).collect(),
                });
                continue;
            };
            sources.push(SourceTensor {
                object: object.clone(),
                name: t.name.clone(),
                role,
                layer,
                representation,
                shape: t.shape.iter().map(|d| *d as u64).collect(),
            });
        }
    }
    Ok((sources, excluded))
}
