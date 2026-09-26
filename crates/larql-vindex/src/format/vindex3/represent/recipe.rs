//! Explicit NVFP4 dispatch with one statistic in flight. Completed packs are
//! mmap-backed; capture reads the candidate prefix in execution order.
use super::calibration::{
    CalibrationArtifact, CalibrationBank, Population, PreparedCalibration, Projection,
    StatisticKind,
};
use super::derivation::{DerivationRecord, GptqDiagnostics, GptqParameters, TensorDerivation};
use super::nvfp4_pack::{self, EncoderRecipe, PackLayout};
use super::physical::{PhysicalStore, WeightRegion};
use super::{RepresentReport, RepresentSpec};
use crate::error::VindexError;
use crate::format::vindex3::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::{
    OperandOverrides, OperandSource, OperandStore,
};
use crate::format::vindex3::opplan::plan_component_ops;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type TensorId = (String, String);
/// Nearest preserves legacy identity and has no calibration requirement.
pub enum Nvfp4Recipe<'a> {
    Nearest,
    Gptq(&'a GptqRequest),
}
/// Reuse never recaptures on mismatch.
#[derive(Debug, Clone)]
pub enum CalibrationInput {
    Existing(PathBuf),
    CaptureTo(PathBuf),
}
/// GPTQ at the named sites, explicitly nearest at every other eligible matrix.
/// This mixed-recipe candidate does not close the uniform R4 experiment.
pub struct GptqRequest {
    pub component: String,
    pub bank: CalibrationBank,
    pub sites: BTreeMap<TensorId, CalibrationInput>,
}

pub fn compile_representation_recipe(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
    recipe: Nvfp4Recipe<'_>,
) -> Result<RepresentReport, VindexError> {
    if spec.encoding != nvfp4_pack::DTYPE_NVFP4 {
        return Err(refused("NVFP4 recipes require the NVFP4 codec"));
    }
    match recipe {
        Nvfp4Recipe::Nearest => super::compile_representation(src, out, spec),
        Nvfp4Recipe::Gptq(request) => {
            if out.exists() {
                return Err(refused("calibrated output must be a fresh directory"));
            }
            let completed = prepare(src, out, spec, request)?;
            let result = super::compile_inner(
                src,
                out,
                spec,
                &super::EncoderRegistry::new(),
                None,
                Some(&completed),
            );
            if result.is_ok() {
                drop(completed);
                std::fs::remove_dir_all(out.join(".represent-recipe-work"))?;
            }
            result
        }
    }
}
fn refused(message: impl std::fmt::Display) -> VindexError {
    VindexError::Parse(format!("CAL-1.2: {message}"))
}

pub(crate) struct Completed {
    pub tensors: BTreeMap<TensorId, (WeightRegion, TensorDerivation)>,
    pub derivation: DerivationRecord,
}
impl Completed {
    pub fn encoder(&self, object: &str) -> EncoderRecipe {
        let recipes: Vec<_> = self
            .derivation
            .tensors
            .iter()
            .filter(|t| t.object == object)
            .map(|t| &t.recipe)
            .collect();
        if recipes.iter().all(|r| **r == EncoderRecipe::gptq_v1()) {
            EncoderRecipe::gptq_v1()
        } else if recipes.iter().all(|r| **r == EncoderRecipe::nearest_v1()) {
            EncoderRecipe::nearest_v1()
        } else {
            EncoderRecipe {
                algorithm: "nvfp4-site-recipes".into(),
                revision: 1,
                source: None,
            }
        }
    }
}
/// Common legacy nearest implementation, including logical-length slicing.
pub(crate) fn nearest(
    values: &[f32],
    layout: PackLayout,
    name: &str,
) -> Result<Vec<u8>, VindexError> {
    use crate::format::vindex3::opplan::exec::weights::{quantize_nvfp4, LoadedWeight};
    let quantised = quantize_nvfp4(values, layout.rows, layout.k, name)?;
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
        ..
    } = &quantised
    else {
        return Err(refused("NVFP4 quantizer returned another format"));
    };
    let mut bytes = Vec::with_capacity(layout.total_len);
    bytes.extend_from_slice(&packed.as_slice()[..packed.logical_len()]);
    bytes.extend_from_slice(&scales.as_slice()[..scales.logical_len()]);
    bytes.extend_from_slice(&tensor_scale.to_le_bytes());
    if bytes.len() != layout.total_len {
        return Err(refused("NVFP4 payload length mismatch"));
    }
    Ok(bytes)
}

fn prepare(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
    request: &GptqRequest,
) -> Result<Completed, VindexError> {
    if request.sites.is_empty() {
        return Err(refused("GPTQ request has no sites"));
    }
    let inspection = inspect_container(src, false)?;
    let plan = plan_component_ops(&inspection, src, &request.component)?
        .plan
        .ok_or_else(|| refused("no calibration plan"))?;
    let store = OperandStore::open(src, &inspection)?;
    let mut order = Vec::new();
    for l in &plan.layers {
        let a = l
            .attention
            .softmax()
            .ok_or_else(|| refused("requires softmax layers"))?;
        let f = l
            .ffn
            .as_ref()
            .and_then(|f| f.dense())
            .ok_or_else(|| refused("requires dense FFNs"))?;
        // O has no capture boundary; its explicitly nearest bytes must still
        // enter the prefix before the FFN input is observed.
        for (op, projection) in [
            (&a.q, Some(Projection::Query)),
            (&a.k, Some(Projection::Key)),
            (&a.v, Some(Projection::Value)),
            (&a.o, None),
        ] {
            order.push((l.layer, projection, op.clone()));
        }
        if let Some(g) = &f.gate {
            order.push((l.layer, Some(Projection::Gate), g.clone()));
        }
        order.push((l.layer, Some(Projection::Up), f.up.clone()));
        order.push((l.layer, Some(Projection::Down), f.down.clone()));
    }
    let declared = super::plan_roles::plan_roles(src, &inspection);
    let text: BTreeSet<_> = inspection
        .graph
        .components
        .iter()
        .filter(|c| c.role == crate::format::vindex3::graph::component::ComponentRole::PrimaryText)
        .map(|c| &c.id)
        .collect();
    let text_objects: BTreeSet<_> = inspection
        .graph
        .objects
        .iter()
        .filter(|o| text.contains(&o.component))
        .map(|o| &o.id)
        .collect();
    let index: crate::format::vindex3::index::Vindex3Index =
        serde_json::from_slice(&std::fs::read(src.join("index.json"))?).map_err(refused)?;
    let mut eligible = BTreeSet::new();
    let mut objects = BTreeSet::new();
    for entry in index
        .representations
        .values()
        .filter(|e| spec.wants(&e.object))
    {
        if !objects.insert(&entry.object) {
            return Err(refused(
                "calibrated compile requires one source representation per selected object",
            ));
        }
        let (header, _) = read_segment_header(&src.join(&entry.segment))?;
        for t in header.tensors {
            let role = declared
                .get(&(entry.object.clone(), t.name.clone()))
                .copied()
                .filter(|_| text_objects.contains(&entry.object))
                .unwrap_or_else(|| {
                    super::policy::classify_in(
                        text_objects.contains(&entry.object),
                        &entry.object,
                        &t.name,
                        &t.shape,
                    )
                });
            if spec.roles.compiles(role)
                && !spec.protect.protects(&t.name)
                && PackLayout::derive(&t.shape, &t.name).is_ok()
            {
                eligible.insert((entry.object.clone(), t.name));
            }
        }
    }
    let ordered: BTreeSet<_> = order
        .iter()
        .map(|(_, _, op)| (op.object.clone(), op.tensor.clone()))
        .collect();
    if !eligible.is_subset(&ordered) {
        return Err(refused(
            "eligible tensor outside supported dense projection schedule; protect it explicitly",
        ));
    }
    for id in request.sites.keys() {
        if !eligible.contains(id)
            || !order
                .iter()
                .any(|(_, p, op)| p.is_some() && (&op.object, &op.tensor) == (&id.0, &id.1))
        {
            return Err(refused(format!(
                "GPTQ site {id:?} is unsupported, protected or not compiled"
            )));
        }
    }
    std::fs::create_dir(out)?;
    let scratch = out.join(".represent-recipe-work");
    std::fs::create_dir(&scratch)?;
    let mut overrides = OperandOverrides::new();
    let mut tensors = BTreeMap::new();
    let mut records = Vec::new();
    for (layer, projection, op) in order {
        let id = (op.object.clone(), op.tensor.clone());
        if !eligible.contains(&id) {
            continue;
        }
        let layout = PackLayout::derive(&op.shape, &op.tensor)?;
        let values = store.load(&op)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(refused("non-finite source weights"));
        }
        let mut hash = Sha256::new();
        for v in &values {
            hash.update(v.to_le_bytes());
        }
        let source_values_sha256 = format!("{:x}", hash.finalize());
        let (bytes, recipe, calibration, parameters, diagnostics) = match request.sites.get(&id) {
            None => (
                nearest(&values, layout, &op.tensor)?,
                EncoderRecipe::nearest_v1(),
                None,
                None,
                None,
            ),
            Some(input) => {
                let prepared = PreparedCalibration::prepare(
                    &plan,
                    OperandSource::overlaid(&store, &overrides),
                    layer,
                    projection.expect("validated site"),
                )?;
                let expected = prepared.key(&request.bank, StatisticKind::DenseGram)?;
                if expected.population != Population::Calibration {
                    return Err(refused("GPTQ requires the calibration population"));
                }
                let artifact = match input {
                    CalibrationInput::Existing(path) => CalibrationArtifact::read(path, &expected)?,
                    CalibrationInput::CaptureTo(path) => {
                        let artifact = prepared.capture(&request.bank, StatisticKind::DenseGram)?;
                        artifact.write(path)?;
                        drop(artifact);
                        CalibrationArtifact::read(path, &expected)?
                    }
                };
                drop(prepared);
                let (manifest, raw) = artifact.into_parts();
                let h =
                    ndarray::Array2::from_shape_vec((layout.k, layout.k), raw).map_err(refused)?;
                let outcome = super::gptq::pack::quantize_nvfp4_gptq_owned(
                    &values,
                    layout.rows,
                    layout.k,
                    h,
                    &op.tensor,
                )?;
                let diagnostics = GptqDiagnostics {
                    dead_columns: outcome.dead_columns,
                    alive_columns: outcome.alive_columns,
                    saturated_elements: outcome.saturated_elements,
                    total_elements: outcome.total_elements,
                };
                let bytes = nvfp4_pack::encode(&outcome.matrix, &layout, &op.tensor)?;
                (
                    bytes,
                    EncoderRecipe::gptq_v1(),
                    Some(manifest),
                    Some(GptqParameters::frozen_v1()),
                    Some(diagnostics),
                )
            }
        };
        let record = TensorDerivation {
            object: op.object.clone(),
            tensor: op.tensor.clone(),
            recipe,
            source_values_sha256,
            payload_sha256: super::compile::hash_bytes(&bytes),
            payload_bytes: bytes.len() as u64,
            calibration,
            parameters,
            diagnostics,
        };
        let path = scratch.join(format!("{}.bin", records.len()));
        write_segment(
            &path,
            "candidate-prefix",
            vec![PlannedTensor {
                relative_name: op.tensor.clone(),
                source_name: op.tensor.clone(),
                dtype: "NVFP4".into(),
                shape: op.shape.clone(),
                len: bytes.len() as u64,
            }],
            |_, w, tap| {
                w.write_all(&bytes)?;
                tap(&bytes);
                Ok(bytes.len() as u64)
            },
        )?;
        let mapped = Arc::new(PhysicalStore::map_segment(
            format!("candidate-prefix-{}", records.len()),
            &path,
        )?);
        let region = mapped
            .whole(&op.tensor)
            .ok_or_else(|| refused("missing completed pack"))?;
        overrides.replace_nvfp4(&op, region.clone());
        records.push(record.clone());
        tensors.insert(id, (region, record));
    }
    let map = super::PrecisionMap::from_policy(
        spec.map_name(),
        &spec.encoding,
        &spec.roles,
        &spec.protect,
    );
    Ok(Completed {
        tensors,
        derivation: DerivationRecord {
            schema: "represent-derivation/v1".into(),
            source_semantic_sha256: super::compiler::read_source_identity(src)?.semantic_digest(),
            allocation_sha256: super::compile::hash_bytes(
                &serde_json::to_vec(&map).map_err(refused)?,
            ),
            tensors: records,
        },
    })
}
#[cfg(test)]
mod tests;
