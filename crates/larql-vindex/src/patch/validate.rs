//! Whether a patch fits what it is about to be applied onto.
//!
//! A patch is checked whole, before any overlay state changes: the model it
//! was written against, every slot it names, and the width of every vector it
//! carries. One check serves both overlays (the VINDEX2 [`PatchedVindex`] and
//! the VINDEX3 knowledge overlay), so the two cannot disagree about which
//! patches are admissible.
//!
//! [`PatchedVindex`]: super::overlay::PatchedVindex

use crate::error::VindexError;

use super::format::{decode_gate_vector, PatchOp, VindexPatch};

/// The shape of the index a patch is applied onto.
pub struct PatchGeometry<'a> {
    /// Width of the residual stream. Every gate, up, down and KNN key vector
    /// is one row of this width. `None` when the index declares no width (a
    /// VINDEX3 component with no embedding op): a patch that carries a vector
    /// is then refused, because its width cannot be checked.
    pub hidden_size: Option<usize>,
    /// Number of layers. A layer index at or past it names nothing.
    pub num_layers: usize,
    /// Number of feature slots at a layer. A feature index at or past it
    /// names nothing; a layer with no feature space reports zero.
    pub features_at: &'a dyn Fn(usize) -> usize,
}

impl VindexPatch {
    /// Refuse a patch written against a different model.
    ///
    /// A slot index or a residual direction means something only in the model
    /// it was read from. Both names must be present: a patch that declares no
    /// model cannot be shown to belong to this one, and that is refused the
    /// same as one that names another.
    pub fn check_base_model(&self, model: &str) -> Result<(), VindexError> {
        if self.base_model.is_empty() {
            return Err(VindexError::Parse(format!(
                "the patch declares no base_model, so it cannot be shown to belong to `{model}`"
            )));
        }
        if self.base_model != model {
            return Err(VindexError::Parse(format!(
                "the patch was written against `{}`, and this is `{model}`",
                self.base_model
            )));
        }
        Ok(())
    }

    /// Refuse a patch with a slot outside the index or a vector of the wrong
    /// width. Nothing is applied by this; it only reads.
    pub fn check_geometry(&self, geometry: &PatchGeometry<'_>) -> Result<(), VindexError> {
        for (i, op) in self.operations.iter().enumerate() {
            match op {
                PatchOp::Insert {
                    layer,
                    feature,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    ..
                }
                | PatchOp::Update {
                    layer,
                    feature,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    ..
                } => {
                    check_slot(i, *layer, *feature, geometry)?;
                    for (field, b64) in [
                        ("gate_vector_b64", gate_vector_b64),
                        ("up_vector_b64", up_vector_b64),
                        ("down_vector_b64", down_vector_b64),
                    ] {
                        if let Some(b64) = b64 {
                            check_vector(i, field, b64, geometry)?;
                        }
                    }
                }
                PatchOp::Delete { layer, feature, .. } => {
                    check_slot(i, *layer, *feature, geometry)?;
                }
                PatchOp::InsertKnn {
                    layer,
                    key_vector_b64,
                    ..
                } => {
                    check_layer(i, *layer, geometry)?;
                    check_vector(i, "key_vector_b64", key_vector_b64, geometry)?;
                }
                // Names an entity, not a slot, and carries no vector.
                PatchOp::DeleteKnn { .. } => {}
                // Names no slot and carries no residual vector. It is bound
                // to a container by its own digests: the VINDEX3 overlay
                // checks those, and the VINDEX2 overlay refuses the op.
                PatchOp::WeightDelta(_)
                | PatchOp::StructuredWeight(_)
                | PatchOp::LowRankColumnsWeight(_) => {}
            }
        }
        Ok(())
    }
}

fn check_layer(op: usize, layer: usize, geometry: &PatchGeometry<'_>) -> Result<(), VindexError> {
    if layer >= geometry.num_layers {
        return Err(VindexError::Parse(format!(
            "patch op {op}: layer {layer} is out of bounds (the index has {} layers)",
            geometry.num_layers
        )));
    }
    Ok(())
}

fn check_slot(
    op: usize,
    layer: usize,
    feature: usize,
    geometry: &PatchGeometry<'_>,
) -> Result<(), VindexError> {
    check_layer(op, layer, geometry)?;
    let features = (geometry.features_at)(layer);
    if feature >= features {
        return Err(VindexError::Parse(format!(
            "patch op {op}: feature {feature} is out of bounds (layer {layer} has {features} features)"
        )));
    }
    Ok(())
}

fn check_vector(
    op: usize,
    field: &str,
    b64: &str,
    geometry: &PatchGeometry<'_>,
) -> Result<(), VindexError> {
    let vector = decode_gate_vector(b64)
        .map_err(|e| VindexError::Parse(format!("patch op {op}: corrupt {field}: {e}")))?;
    let Some(hidden_size) = geometry.hidden_size else {
        return Err(VindexError::Parse(format!(
            "patch op {op}: {field} cannot be checked, because the index declares no hidden size"
        )));
    };
    if vector.len() != hidden_size {
        return Err(VindexError::Parse(format!(
            "patch op {op}: {field} has {} elements, and the index's hidden size is {hidden_size}",
            vector.len()
        )));
    }
    Ok(())
}
